import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "EPL_all_seasons_interim.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "merge_report.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

core_cols = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam", "Referee",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
    "HS", "AS", "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR",
    "AvgH", "AvgD", "AvgA",
    "MaxH", "MaxD", "MaxA",
]

files = sorted(RAW_DIR.glob("EPL_*.csv"))
if not files:
    raise FileNotFoundError(f"No EPL_*.csv files found in {RAW_DIR}")

all_dfs = []
report_rows = []

for f in files:
    season = f.stem.replace("EPL_", "")  # "2024-25"
    df = pd.read_csv(f)

    missing = [c for c in core_cols if c not in df.columns]
    present = [c for c in core_cols if c in df.columns]

    # keep only what exists, but keep consistent order
    df_out = df[present].copy()
    df_out["season"] = season
    df_out["source_file"] = f.name

    all_dfs.append(df_out)

    report_rows.append({
        "file": f.name,
        "season": season,
        "rows": df.shape[0],
        "cols_total": df.shape[1],
        "cols_present_core": len(present),
        "cols_missing_core": len(missing),
        "missing_core_cols": ";".join(missing),
    })

merged = pd.concat(all_dfs, ignore_index=True)

# Save merged interim
merged.to_csv(OUT_PATH, index=False)

# Save merge report
pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)

print("Files merged:", len(files))
print("Merged rows:", merged.shape[0])
print("Merged cols:", merged.shape[1])
print("Saved interim to:", OUT_PATH)
print("Saved report to:", REPORT_PATH)

# Quick sanity check: how many seasons / rows per season
print("\nRows per season:")
print(merged.groupby("season").size().sort_index())
