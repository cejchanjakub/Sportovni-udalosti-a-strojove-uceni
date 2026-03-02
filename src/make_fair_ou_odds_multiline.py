import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

IN_PATH = DATA_DIR / "goals_poisson_test_predictions.csv"
OUT_PATH = DATA_DIR / "ou_fair_odds_test_0_5_to_5_5.csv"

LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
EPS = 1e-9

def over_prob_poisson(lmbda, line):
    k = int(np.floor(line))
    return 1.0 - poisson.cdf(k, lmbda)

df = pd.read_csv(IN_PATH)

# safety
df["lambda_goals"] = pd.to_numeric(df["lambda_goals"], errors="coerce")
df = df.dropna(subset=["lambda_goals"])

for L in LINES:
    key = str(L).replace(".", "_")

    p_over = df["lambda_goals"].apply(lambda l: over_prob_poisson(l, L)).clip(EPS, 1-EPS)
    p_under = (1.0 - p_over).clip(EPS, 1-EPS)

    df[f"p_over_{key}"] = p_over
    df[f"p_under_{key}"] = p_under

    # férové kurzy (bez marže)
    df[f"odds_over_{key}_fair"] = 1.0 / p_over
    df[f"odds_under_{key}_fair"] = 1.0 / p_under

# Ulož jen to, co budeš chtít používat dál (nebo klidně celý df)
keep_cols = [
    "match_id", "season", "HomeTeam", "AwayTeam", "total_goals", "lambda_goals",
]
for L in LINES:
    key = str(L).replace(".", "_")
    keep_cols += [
        f"p_over_{key}", f"p_under_{key}",
        f"odds_over_{key}_fair", f"odds_under_{key}_fair",
    ]

out = df[keep_cols].copy()
out.to_csv(OUT_PATH, index=False)
print("Saved fair O/U odds to:", OUT_PATH)
