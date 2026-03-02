import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_processed.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff").reset_index(drop=True)

split_idx = int(len(df) * 0.7)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print("Train size:", train.shape[0])
print("Test size:", test.shape[0])

print("\nTrain period:",
      train["kickoff"].min().date(), "→", train["kickoff"].max().date())
print("Test period:",
      test["kickoff"].min().date(), "→", test["kickoff"].max().date())
