import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_all_seasons_processed.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_SEASONS = ["2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23"]
VAL_SEASONS   = ["2023-24"]
TEST_SEASONS  = ["2024-25"]
LIVE_SEASONS  = ["2025-26"]

df = pd.read_csv(IN_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff")

train = df[df["season"].isin(TRAIN_SEASONS)].copy()
val   = df[df["season"].isin(VAL_SEASONS)].copy()
test  = df[df["season"].isin(TEST_SEASONS)].copy()
live  = df[df["season"].isin(LIVE_SEASONS)].copy()

print("Train:", train.shape[0], "rows | seasons:", TRAIN_SEASONS)
print("Val:  ", val.shape[0],   "rows | seasons:", VAL_SEASONS)
print("Test: ", test.shape[0],  "rows | seasons:", TEST_SEASONS)
print("Live: ", live.shape[0],  "rows | seasons:", LIVE_SEASONS)

train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(OUT_DIR / "val.csv", index=False)
test.to_csv(OUT_DIR / "test.csv", index=False)
live.to_csv(OUT_DIR / "live.csv", index=False)

print("\nSaved:",
      OUT_DIR / "train.csv",
      OUT_DIR / "val.csv",
      OUT_DIR / "test.csv",
      OUT_DIR / "live.csv",
      sep="\n")
