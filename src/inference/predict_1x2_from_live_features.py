from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import joblib

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


def _pick_row(df: pd.DataFrame, home_std: str, away_std: str, kickoff_utc: str):
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        raise ValueError("V live_features.csv chybí HomeTeam/AwayTeam.")
    if "kickoff_dt" not in df.columns and "Date" not in df.columns:
        raise ValueError("V live_features.csv chybí kickoff_dt i Date (nemám podle čeho párovat čas).")

    d = df.copy()
    d["_home_mapped"] = d["HomeTeam"].astype(str).str.strip().apply(map_team)
    d["_away_mapped"] = d["AwayTeam"].astype(str).str.strip().apply(map_team)

    kickoff = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")

    def best_match(cand: pd.DataFrame) -> pd.Series:
        if "kickoff_dt" in cand.columns:
            cand["_kdt"] = _to_utc(cand["kickoff_dt"])
        else:
            cand["_kdt"] = _to_utc(cand["Date"])
        cand["_abs_delta"] = (cand["_kdt"] - kickoff).abs()
        return cand.sort_values("_abs_delta").iloc[0]

    cand1 = d[(d["_home_mapped"] == home_std) & (d["_away_mapped"] == away_std)].copy()
    if len(cand1) > 0:
        return best_match(cand1), False

    cand2 = d[(d["_home_mapped"] == away_std) & (d["_away_mapped"] == home_std)].copy()
    if len(cand2) > 0:
        return best_match(cand2), True

    raise ValueError(
        f"Nenašel jsem zápas {home_std} vs {away_std} v live_features.csv ani v prohozené orientaci."
    )


def _feature_baseline_means(feature_order: list[str]) -> pd.Series:
    """
    Spočítá průměry featur z train_features.csv (jen pro featury, co model potřebuje).
    Použijeme je pro doplnění NaN v live řádku.
    """
    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(f"Chybí {TRAIN_FEATURES}. Nemám z čeho dopočítat baseline pro NaN.")

    df = pd.read_csv(TRAIN_FEATURES, usecols=[c for c in feature_order if c in pd.read_csv(TRAIN_FEATURES, nrows=0).columns])
    # Pozn.: výše je bezpečné, ale dvakrát čte header; je to ok pro malý skript.

    means = df.mean(numeric_only=True)
    # pro jistotu doplň 0 u featur, co se nepodařilo spočítat
    for c in feature_order:
        if c not in means.index or pd.isna(means[c]):
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
    X = X.apply(pd.to_numeric, errors="coerce")  # kdyby něco bylo string
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