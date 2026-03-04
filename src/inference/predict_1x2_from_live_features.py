from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd

from src.inference.providers.football_data_provider import FootballDataProvider
from src.inference.Team_mapper import map_team

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIVE_FEATURES = PROJECT_ROOT / "data" / "features" / "live_features.csv"
TRAIN_FEATURES = PROJECT_ROOT / "data" / "features" / "train_features.csv"
ART_DIR = PROJECT_ROOT / "artifacts" / "v1_model_freeze" / "1x2"


def _load_artifacts():
    model = joblib.load(ART_DIR / "model.joblib")
    scaler = joblib.load(ART_DIR / "scaler.joblib")
    with open(ART_DIR / "features.json", "r", encoding="utf-8") as f:
        feats = json.load(f)
    return model, scaler, feats


def _to_utc(ts: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        return dt.dt.tz_convert("UTC")
    return dt.dt.tz_localize("UTC")


def _parse_kickoff_utc(s: str) -> pd.Timestamp:
    """
    Robustní parser kickoffu z API / JSON.

    Podporuje např.:
      - ISO (2026-03-01T15:00:00Z)
      - "2026-03-01 15:00"
      - datum s tečkami (1.3.2026) -> DD.MM.YYYY (dayfirst)
      - datum se slash (01/03/2026) -> DD/MM/YYYY (dayfirst)
    """
    if s is None:
        return pd.NaT

    s = str(s).strip()

    # "1.3.2026" / "01.03.2026"
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", s):
        return pd.to_datetime(s, dayfirst=True, utc=True, errors="coerce")

    # "01/03/2026" (ber jako dayfirst)
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
        return pd.to_datetime(s, dayfirst=True, utc=True, errors="coerce")

    # ISO / "YYYY-MM-DD" / "YYYY-MM-DD HH:MM" / "YYYY-MM-DDTHH:MM:SSZ"
    return pd.to_datetime(s, utc=True, errors="coerce")


def _pick_row(
    df: pd.DataFrame,
    home_std: str,
    away_std: str,
    kickoff_utc: str,
) -> tuple[pd.Series, bool]:
    """
    Najde **správný** řádek pro konkrétní fixture.

    NOVĚ (záměrně přísné):
    - Nepoužívá "nejbližší historický zápas".
    - Buď najde záznam ve stejný den (po mapování týmů), nebo vyhodí LookupError.

    Důvod:
    - Při predikci budoucích zápasů je lepší failnout s jasnou chybou,
      než ti potichu vracet úplně jiný (historický) zápas.
    """
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise ValueError("V live_features.csv chybí HomeTeam/AwayTeam.")
    if "kickoff_dt" not in df.columns and "Date" not in df.columns and "kickoff" not in df.columns:
        raise ValueError("V live_features.csv chybí kickoff_dt/Date/kickoff (nemám podle čeho párovat čas).")

    kickoff = _parse_kickoff_utc(kickoff_utc)
    if pd.isna(kickoff):
        raise ValueError(f"Neplatný utc_date/kickoff: {kickoff_utc}")

    d = df.copy()
    d["_home_mapped"] = d["HomeTeam"].astype(str).str.strip().apply(map_team)
    d["_away_mapped"] = d["AwayTeam"].astype(str).str.strip().apply(map_team)

    # Z čeho bereme čas v live_features?
    if "kickoff" in d.columns:
        d["_kdt"] = _to_utc(d["kickoff"])
    elif "kickoff_dt" in d.columns:
        d["_kdt"] = _to_utc(d["kickoff_dt"])
    else:
        d["_kdt"] = _to_utc(d["Date"])

    # 1) Filtr týmů (v obou orientacích)
    cand = d[(d["_home_mapped"] == home_std) & (d["_away_mapped"] == away_std)].copy()
    swapped = False
    if cand.empty:
        cand = d[(d["_home_mapped"] == away_std) & (d["_away_mapped"] == home_std)].copy()
        swapped = True

    if cand.empty:
        raise LookupError(
            f"Nenašel jsem týmový pár {home_std} vs {away_std} v live_features.csv (ani prohozeně). "
            "Pokud jde o budoucí zápas, live_features ho zřejmě vůbec neobsahuje."
        )

    # 2) Přísné párování na datum
    cand["_date"] = cand["_kdt"].dt.date
    target_date = kickoff.date()

    same_day = cand[cand["_date"] == target_date].copy()
    if same_day.empty:
        # vypiš nejbližší dostupná data pro daný pár (debug-friendly)
        avail = (
            cand["_kdt"]
            .dropna()
            .dt.date
            .astype(str)
            .value_counts()
            .sort_index()
            .index.tolist()
        )
        avail_preview = ", ".join(avail[:8]) if avail else "(žádné)"
        raise LookupError(
            f"Nenašel jsem fixture {home_std} vs {away_std} pro datum {target_date} v live_features.csv. "
            f"Dostupná data pro tento pár v live_features: {avail_preview}. "
            "To znamená, že live_features neobsahuje tento budoucí zápas (a nesmí se fallbackovat do historie)."
        )

    # Když je víc záznamů stejný den, vyber ten nejbližší časově.
    same_day["_abs_delta"] = (same_day["_kdt"] - kickoff).abs()
    best = same_day.sort_values("_abs_delta").iloc[0]
    return best, swapped


def _feature_baseline_means(feature_order: list[str]) -> pd.Series:
    """
    Spočítá průměry featur z train_features.csv (jen pro featury, co model potřebuje).
    Použijeme je pro doplnění NaN v live řádku.
    """
    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(f"Chybí {TRAIN_FEATURES}. Nemám z čeho dopočítat baseline pro NaN.")

    # bezpečné načtení jen existujících sloupců
    header = pd.read_csv(TRAIN_FEATURES, nrows=0).columns
    usecols = [c for c in feature_order if c in header]
    df = pd.read_csv(TRAIN_FEATURES, usecols=usecols)

    means = df.mean(numeric_only=True)

    # pro jistotu doplň 0 u featur, co se nepodařilo spočítat
    for c in feature_order:
        if c not in means.index or pd.isna(means.get(c)):
            means[c] = 0.0
    return means[feature_order]


def main():
    provider = FootballDataProvider()
    fixtures = provider.get_upcoming_matches(days_ahead=14)

    if not fixtures:
        print("Žádné nadcházející zápasy z API.")
        return

    fx = fixtures[0]
    home_std = map_team(fx["home_team"])
    away_std = map_team(fx["away_team"])
    kickoff_utc = fx["utc_date"]

    df_live = pd.read_csv(LIVE_FEATURES)
    row, swapped = _pick_row(df_live, home_std, away_std, kickoff_utc)

    model, scaler, feat_order = _load_artifacts()

    X = pd.DataFrame([row.to_dict()])
    X = X[feat_order]

    # doplnění NaN z baseline průměrů
    means = _feature_baseline_means(feat_order)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(means)

    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[0]
    p_home, p_draw, p_away = map(float, probs)

    if swapped:
        p_home, p_away = p_away, p_home

    print(f"Zápas (API): {fx['home_team']} vs {fx['away_team']} | kickoff UTC: {kickoff_utc}")
    print(f"Mapped (API): {home_std} vs {away_std}")
    print(f"Matched row: {row.get('HomeTeam')} vs {row.get('AwayTeam')} | swapped={swapped}")

    print("\n--- 1X2 ---")
    print(f"P(Home)={p_home:.4f} | fair_odds={1/p_home:.2f}")
    print(f"P(Draw)={p_draw:.4f} | fair_odds={1/p_draw:.2f}")
    print(f"P(Away)={p_away:.4f} | fair_odds={1/p_away:.2f}")


if __name__ == "__main__":
    main()