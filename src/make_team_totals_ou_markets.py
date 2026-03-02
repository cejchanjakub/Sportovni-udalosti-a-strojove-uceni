import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

IN_PATH = DATA_DIR / "team_aware_poisson_test_predictions.csv"
OUT_PATH = DATA_DIR / "team_totals_ou_fair_odds_test.csv"

LINES = [0.5, 1.5, 2.5, 3.5]
EPS = 1e-9

def over_prob(lmbda, line):
    k = int(np.floor(line))
    return 1.0 - poisson.cdf(k, lmbda)

df = pd.read_csv(IN_PATH)

for side in ["home", "away"]:
    lam_col = f"lambda_{side}"

    for L in LINES:
        key = str(L).replace(".", "_")

        p_over = df[lam_col].apply(lambda l: over_prob(l, L)).clip(EPS, 1-EPS)
        p_under = (1.0 - p_over).clip(EPS, 1-EPS)

        df[f"p_{side}_over_{key}"] = p_over
        df[f"p_{side}_under_{key}"] = p_under

        df[f"odds_{side}_over_{key}_fair"] = 1.0 / p_over
        df[f"odds_{side}_under_{key}_fair"] = 1.0 / p_under

df.to_csv(OUT_PATH, index=False)
print("Saved team totals O/U fair odds to:", OUT_PATH)
