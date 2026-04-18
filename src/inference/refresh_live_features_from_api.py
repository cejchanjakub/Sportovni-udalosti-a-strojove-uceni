# src/inference/refresh_live_features_from_api.py
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_features_all import (
    build_features,
    compute_elo_for_all,
    compute_referee_features_for_all,
    compute_h2h_features,
)
from src.inference.providers.football_data_provider import FootballDataProvider
from src.inference.Team_mapper import map_team

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALL_PROCESSED = PROJECT_ROOT / "data" / "processed" / "EPL_all_seasons_processed.csv"
OUT_FEATURES  = PROJECT_ROOT / "data" / "features" / "live_features.csv"
OUT_FIXTURES  = PROJECT_ROOT / "data" / "processed" / "live_fixtures.csv"
OUT_META      = PROJECT_ROOT / "data" / "features" / "live_features_meta.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _season_from_date(d: date) -> str:
    if d.month >= 8:
        return f"{d.year}-{str(d.year + 1)[-2:]}"
    return f"{d.year - 1}-{str(d.year)[-2:]}"


def _fixture_rows(days_ahead: int) -> pd.DataFrame:
    provider = FootballDataProvider()
    fixtures = provider.get_upcoming_matches(days_ahead=days_ahead)

    rows: list[dict] = []
    for fx in fixtures:
        kickoff = pd.to_datetime(fx.get("utc_date"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue

        d = kickoff.date()
        rows.append({
            "Div":         "E0",
            "Date":        d.isoformat(),
            "Time":        kickoff.strftime("%H:%M"),
            "kickoff":     kickoff.isoformat().replace("+00:00", "Z"),
            "HomeTeam":    map_team(fx.get("home_team", "")),  # mapuj na standardní název
            "AwayTeam":    map_team(fx.get("away_team", "")),  # mapuj na standardní název
            "Referee":     fx.get("referee", np.nan),
            "FTHG": np.nan, "FTAG": np.nan, "FTR": np.nan,
            "HTHG": np.nan, "HTAG": np.nan,
            "HS":   np.nan, "AS":   np.nan,
            "HST":  np.nan, "AST":  np.nan,
            "HF":   np.nan, "AF":   np.nan,
            "HC":   np.nan, "AC":   np.nan,
            "HY":   np.nan, "AY":   np.nan,
            "HR":   np.nan, "AR":   np.nan,
            "AvgH": np.nan, "AvgD": np.nan, "AvgA": np.nan,
            "MaxH": np.nan, "MaxD": np.nan, "MaxA": np.nan,
            "season":      _season_from_date(d),
            "source_file": "football-data-api",
            "match_id":    f"api_{fx.get('match_id')}",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    return df


def _dedupe_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures
    fx = fixtures.copy()
    if "match_id" in fx.columns:
        fx = fx.drop_duplicates(subset=["match_id"], keep="last")
    fx = fx.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    return fx


def main(days_ahead: int = 60) -> None:
    started_at = _utc_now_iso()

    if not ALL_PROCESSED.exists():
        raise FileNotFoundError(
            f"Chybí {ALL_PROCESSED}. Nejdřív musí existovat historická processed data."
        )

    base = pd.read_csv(ALL_PROCESSED)

    fixtures = _fixture_rows(days_ahead=days_ahead)
    fixtures = _dedupe_fixtures(fixtures)

    if fixtures.empty:
        meta = {
            "status": "empty",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now_iso(),
            "days_ahead": days_ahead,
            "message": "Žádné fixtures z API (nebo se nepodařilo naparsovat utc_date).",
        }
        atomic_write_json(meta, OUT_META)
        print(meta["message"])
        return

    atomic_write_csv(fixtures, OUT_FIXTURES)
    print(f"Saved fixtures: {OUT_FIXTURES} | rows: {len(fixtures)}")

    # Spoj historická data + fixtures
    combined = pd.concat([base, fixtures], ignore_index=True, sort=False)

    # --- ELO a referee featury musí být počítány na celém datasetu ---
    print("Počítám ELO přes celý dataset (historická + fixtures)...")
    elo_lookup = compute_elo_for_all(combined)

    print("Počítám Referee features přes celý dataset...")
    referee_lookup = compute_referee_features_for_all(combined)

    print("Počítám H2H featury přes celý dataset...")
    h2h_lookup = compute_h2h_features(combined)

    # --- Build features pro celý combined dataset ---
    feat = build_features(combined, elo_lookup, referee_lookup, h2h_lookup=h2h_lookup)

    if "match_id" not in feat.columns:
        raise ValueError(
            "Po build_features chybí sloupec match_id – zkontroluj processed schéma."
        )

    # Vyfiltruj jen nové fixtures (match_id prefix api_)
    out = feat[feat["match_id"].astype(str).str.startswith("api_")].copy()

    atomic_write_csv(out, OUT_FEATURES)
    print(f"Saved live features: {OUT_FEATURES} | rows: {len(out)} | cols: {len(out.columns)}")

    meta = {
        "status": "ok",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now_iso(),
        "days_ahead": days_ahead,
        "fixtures_rows": int(len(fixtures)),
        "live_features_rows": int(len(out)),
        "live_features_cols": int(len(out.columns)),
        "out_fixtures": str(OUT_FIXTURES),
        "out_live_features": str(OUT_FEATURES),
        "out_meta": str(OUT_META),
    }
    atomic_write_json(meta, OUT_META)
    print(f"Saved meta: {OUT_META}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-ahead", type=int, default=60)
    args = parser.parse_args()
    main(days_ahead=args.days_ahead)