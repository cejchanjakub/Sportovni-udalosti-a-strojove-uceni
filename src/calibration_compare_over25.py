import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = PROJECT_ROOT / "data" / "processed" / "over25_predictions.csv"

df = pd.read_csv(PRED_PATH, parse_dates=["kickoff"])

def calibration_table(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    b = pd.cut(p, bins=bins, include_lowest=True)
    cal = pd.DataFrame({"bin": b, "y": y, "p": p}).groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"),
        frac_pos=("y", "mean"),
        count=("y", "size")
    ).reset_index()
    # keep only non-empty bins
    cal = cal[cal["count"] > 0].copy()
    return cal

y = df["over_2_5"].astype(int)

cal_base = calibration_table(y, df["p_base"], n_bins=10)
cal_lr = calibration_table(y, df["p_logreg"], n_bins=10)

print("\n=== CALIBRATION TABLE: BASELINE ===")
print(cal_base)

print("\n=== CALIBRATION TABLE: LOGREG ===")
print(cal_lr)

plt.figure()
plt.plot([0, 1], [0, 1])  # perfect calibration

plt.scatter(cal_base["mean_pred"], cal_base["frac_pos"])
plt.scatter(cal_lr["mean_pred"], cal_lr["frac_pos"])

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability diagram: Over 2.5 (Baseline vs LogReg)")
plt.legend(["Perfect", "Baseline", "LogReg"])
plt.show()
