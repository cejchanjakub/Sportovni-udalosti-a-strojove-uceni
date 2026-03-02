import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "EPL_2024-25.csv"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "EPL_2024-25_interim.csv"

df = pd.read_csv(RAW_PATH)

core_cols = [
    # match identifiers
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",

    # full-time result
    "FTHG", "FTAG", "FTR",

    # half-time (optional but useful)
    "HTHG", "HTAG",

    # side markets / match stats
    "HS", "AS", "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR",

    # aggregated odds (often present)
    "AvgH", "AvgD", "AvgA",
    "MaxH", "MaxD", "MaxA",
]

# Keep only columns that actually exist (robust across seasons)
existing = [c for c in core_cols if c in df.columns]
missing = [c for c in core_cols if c not in df.columns]

print("Keeping columns:", existing)
print("Missing columns (not found in file):", missing)

df_out = df[existing].copy()

# Basic sanity checks
print("\n=== SANITY CHECKS ===")
print("Rows:", df_out.shape[0])
print("Duplicates (by Date+HomeTeam+AwayTeam):",
      df_out.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum())
print("Missing core match fields:",
      df_out[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].isna().sum().to_dict())

df_out.to_csv(OUT_PATH, index=False)
print(f"\nSaved interim dataset to: {OUT_PATH}")
