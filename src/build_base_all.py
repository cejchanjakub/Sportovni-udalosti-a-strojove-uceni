import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_all_seasons_processed.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_all_seasons_base.csv"

df = pd.read_csv(IN_PATH, parse_dates=["kickoff"])

# --- columns we want to "freeze" as BASE ---
base_cols = [
    # identifiers & context
    "season", "Div", "Date", "Time", "kickoff", "match_id",
    "HomeTeam", "AwayTeam", "Referee",

    # results / targets
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
    "home_win", "draw", "away_win",
    "total_goals", "over_2_5",

    # match stats (raw facts)
    "HS", "AS", "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR",
    "total_corners", "total_cards", "total_shots", "total_fouls",

    # aggregated odds
    "AvgH", "AvgD", "AvgA",
    "MaxH", "MaxD", "MaxA",
]

missing = [c for c in base_cols if c not in df.columns]
present = [c for c in base_cols if c in df.columns]

print("Present base columns:", len(present))
print("Missing base columns:", missing)

df_base = df[present].copy()

# --- sanity checks ---
print("\nRows:", df_base.shape[0])
print("Unique match_id:", df_base["match_id"].nunique())
print("Missing kickoff:", df_base["kickoff"].isna().sum())
print("Missing Referee:", df_base["Referee"].isna().sum() if "Referee" in df_base.columns else "Referee column not present")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_base.to_csv(OUT_PATH, index=False)
print("\nSaved BASE dataset to:", OUT_PATH)
