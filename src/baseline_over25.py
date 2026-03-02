import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_processed.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff").reset_index(drop=True)

split_idx = int(len(df) * 0.7)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# === BASELINE PROBABILITY ===
p_over = train["over_2_5"].mean()
print(f"Baseline P(Over2.5) from train: {p_over:.3f}")

# apply to test
test["p_over"] = p_over

# === BRIER SCORE (binary) ===
brier = ((test["p_over"] - test["over_2_5"]) ** 2).mean()
print("Baseline Brier score (Over2.5):", round(brier, 4))

# === LOG LOSS (optional but very useful) ===
eps = 1e-15
p = test["p_over"].clip(eps, 1 - eps)
y = test["over_2_5"]
logloss = (-(y * p.apply(lambda x: __import__("math").log(x)) +
            (1 - y) * (1 - p).apply(lambda x: __import__("math").log(x)))).mean()
print("Baseline Log loss (Over2.5):", round(logloss, 4))
