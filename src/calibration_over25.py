import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_processed.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff").reset_index(drop=True)

split_idx = int(len(df) * 0.7)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# Baseline probability from train
p_over = train["over_2_5"].mean()
test["p_over"] = p_over

# --- Calibration bins ---
n_bins = 10
bins = np.linspace(0.0, 1.0, n_bins + 1)
test["bin"] = pd.cut(test["p_over"], bins=bins, include_lowest=True)

cal = test.groupby("bin").agg(
    mean_pred=("p_over", "mean"),
    frac_pos=("over_2_5", "mean"),
    count=("over_2_5", "size")
).reset_index()

print(cal)

# --- Plot reliability diagram ---
plt.figure()
plt.plot([0, 1], [0, 1])  # perfect calibration line
plt.scatter(cal["mean_pred"], cal["frac_pos"])
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability diagram: Over 2.5 (baseline)")
plt.show()
