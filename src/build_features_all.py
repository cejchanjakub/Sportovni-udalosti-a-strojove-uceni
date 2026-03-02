# src/build_features_all.py

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

ROLL_WINDOWS = [3, 5, 10]


# ============================================================
# UTIL
# ============================================================

def _sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kickoff_dt"] = pd.to_datetime(df["Date"])
    return df.sort_values("kickoff_dt").reset_index(drop=True)


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def _rolling_team_features(df: pd.DataFrame, home_stat: str, away_stat: str, prefix: str) -> pd.DataFrame:
    df = df.copy()

    for w in ROLL_WINDOWS:
        df[f"home_{prefix}_for_roll{w}"] = (
            df.groupby("HomeTeam")[home_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )

        df[f"home_{prefix}_against_roll{w}"] = (
            df.groupby("HomeTeam")[away_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )

        df[f"away_{prefix}_for_roll{w}"] = (
            df.groupby("AwayTeam")[away_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )

        df[f"away_{prefix}_against_roll{w}"] = (
            df.groupby("AwayTeam")[home_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )

        df[f"diff_{prefix}_for_roll{w}"] = df[f"home_{prefix}_for_roll{w}"] - df[f"away_{prefix}_for_roll{w}"]
        df[f"diff_{prefix}_against_roll{w}"] = df[f"home_{prefix}_against_roll{w}"] - df[f"away_{prefix}_against_roll{w}"]

    return df


# ============================================================
# ELO (PRE-MATCH, NO LEAKAGE) - within split
# ============================================================

def _add_elo_features(
    df: pd.DataFrame,
    base_elo: float = 1500.0,
    k: float = 20.0,
    home_adv: float = 50.0,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("kickoff_dt").reset_index(drop=True)

    def expected(a, b):
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    elo: dict[str, float] = {}

    elo_home_list = []
    elo_away_list = []
    elo_diff_list = []

    for _, row in df.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])

        eh = elo.get(home, base_elo)
        ea = elo.get(away, base_elo)

        # pre-match values
        elo_home_list.append(eh)
        elo_away_list.append(ea)
        elo_diff_list.append((eh + home_adv) - ea)

        hg = row.get("FTHG")
        ag = row.get("FTAG")

        # když chybí výsledky (typicky live bez targetů), neupdatuj
        if pd.isna(hg) or pd.isna(ag):
            continue

        if hg > ag:
            s_home = 1.0
        elif hg == ag:
            s_home = 0.5
        else:
            s_home = 0.0

        e_home = expected(eh + home_adv, ea)

        elo[home] = eh + k * (s_home - e_home)
        elo[away] = ea + k * ((1.0 - s_home) - (1.0 - e_home))

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = elo_diff_list
    return df


# ============================================================
# BUILD FEATURES FOR ONE SPLIT
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _sort_matches(df)

    numeric_cols = ["FTHG", "FTAG", "HF", "AF", "HC", "AC", "HY", "AY", "HST", "AST"]
    df = _ensure_numeric(df, numeric_cols)

    # totals (targets)
    if "FTHG" in df.columns and "FTAG" in df.columns:
        df["total_goals"] = df["FTHG"] + df["FTAG"]
    if "HF" in df.columns and "AF" in df.columns:
        df["total_fouls"] = df["HF"] + df["AF"]
    if "HC" in df.columns and "AC" in df.columns:
        df["total_corners"] = df["HC"] + df["AC"]
    if "HY" in df.columns and "AY" in df.columns:
        df["total_cards"] = df["HY"] + df["AY"]
    if "HST" in df.columns and "AST" in df.columns:
        df["total_shots_on_target"] = df["HST"] + df["AST"]

    # ELO (pre-match)
    df = _add_elo_features(df)

    # rolling
    if "FTHG" in df.columns and "FTAG" in df.columns:
        df = _rolling_team_features(df, "FTHG", "FTAG", "goals")
    if "HC" in df.columns and "AC" in df.columns:
        df = _rolling_team_features(df, "HC", "AC", "corners")
    if "HF" in df.columns and "AF" in df.columns:
        df = _rolling_team_features(df, "HF", "AF", "fouls")
    if "HY" in df.columns and "AY" in df.columns:
        df = _rolling_team_features(df, "HY", "AY", "yellow")
    if "HST" in df.columns and "AST" in df.columns:
        df = _rolling_team_features(df, "HST", "AST", "shotsot")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test", "live"]:
        in_path = PROCESSED_DIR / f"{split}.csv"
        out_path = FEATURES_DIR / f"{split}_features.csv"

        df = pd.read_csv(in_path)
        df = build_features(df)

        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} | rows: {len(df)} | cols: {len(df.columns)}")


if __name__ == "__main__":
    main()
