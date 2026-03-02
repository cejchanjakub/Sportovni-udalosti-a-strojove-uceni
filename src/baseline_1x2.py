import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_processed.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff").reset_index(drop=True)

split_idx = int(len(df) * 0.7)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# === BASELINE PROBABILITIES ===
p_home = train["home_win"].mean()
p_draw = train["draw"].mean()
p_away = train["away_win"].mean()

print("Baseline probabilities (train):")
print(f"P(Home) = {p_home:.3f}")
print(f"P(Draw) = {p_draw:.3f}")
print(f"P(Away) = {p_away:.3f}")
print("Sum:", p_home + p_draw + p_away)

# === APPLY TO TEST ===
test["p_home"] = p_home
test["p_draw"] = p_draw
test["p_away"] = p_away

# === BRIER SCORE ===
def brier_multiclass(row):
    return (
        (row["p_home"] - row["home_win"])**2 +
        (row["p_draw"] - row["draw"])**2 +
        (row["p_away"] - row["away_win"])**2
    )

brier = test.apply(brier_multiclass, axis=1).mean()
print("\nBaseline Brier score (1X2):", round(brier, 4))
