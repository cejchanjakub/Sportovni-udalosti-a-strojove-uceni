import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

IN_PATH = DATA_DIR / "match_odds_1x2_fair_test.csv"

df = pd.read_csv(IN_PATH)

# --- True outcome ---
# 0 = home win, 1 = draw, 2 = away win
y_true = np.where(
    df["lambda_home"] > df["lambda_away"],  # placeholder, will overwrite below
    0, 2
)

# skutečný výsledek ze zápasu
# FTHG / FTAG jsou v původním test setu, ale tady nemusí být
# pokud nejsou, znovu je načteme z team_aware file
if "FTHG" not in df.columns or "FTAG" not in df.columns:
    src = pd.read_csv(DATA_DIR / "team_aware_poisson_test_predictions.csv")
    df = df.merge(
        src[["match_id", "FTHG", "FTAG"]],
        on="match_id",
        how="left"
    )

y_true = np.where(
    df["FTHG"] > df["FTAG"], 0,
    np.where(df["FTHG"] == df["FTAG"], 1, 2)
)

# --- Predicted probabilities ---
probs = df[["p_home_win", "p_draw", "p_away_win"]].values

# --- Log loss ---
ll = log_loss(y_true, probs, labels=[0,1,2])
print(f"Multiclass LogLoss (1X2): {ll:.4f}")

# --- Brier score (multiclass) ---
y_onehot = np.zeros_like(probs)
y_onehot[np.arange(len(y_true)), y_true] = 1.0

brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
print(f"Multiclass Brier score (1X2): {brier:.4f}")
