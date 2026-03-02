import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skellam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

IN_PATH = DATA_DIR / "team_aware_poisson_test_predictions.csv"
OUT_PATH = DATA_DIR / "match_odds_1x2_fair_test.csv"

EPS = 1e-9

df = pd.read_csv(IN_PATH)

# sanity
df["lambda_home"] = pd.to_numeric(df["lambda_home"], errors="coerce")
df["lambda_away"] = pd.to_numeric(df["lambda_away"], errors="coerce")
df = df.dropna(subset=["lambda_home", "lambda_away"])

# Skellam: D = H - A
mu1 = df["lambda_home"].values
mu2 = df["lambda_away"].values

p_draw = skellam.pmf(0, mu1, mu2)  # P(D=0)
p_home = 1.0 - skellam.cdf(0, mu1, mu2)  # P(D>=1) = 1 - P(D<=0)
p_away = skellam.cdf(-1, mu1, mu2)  # P(D<=-1)

# clip to avoid division issues
p_home = np.clip(p_home, EPS, 1-EPS)
p_draw = np.clip(p_draw, EPS, 1-EPS)
p_away = np.clip(p_away, EPS, 1-EPS)

# (volitelně) zkontroluj součet
df["p_home_win"] = p_home
df["p_draw"] = p_draw
df["p_away_win"] = p_away
df["p_sum"] = df["p_home_win"] + df["p_draw"] + df["p_away_win"]

# férové kurzy bez marže
df["odds_home_win_fair"] = 1.0 / df["p_home_win"]
df["odds_draw_fair"] = 1.0 / df["p_draw"]
df["odds_away_win_fair"] = 1.0 / df["p_away_win"]

keep_cols = [
    "match_id", "season", "HomeTeam", "AwayTeam",
    "lambda_home", "lambda_away",
    "p_home_win", "p_draw", "p_away_win", "p_sum",
    "odds_home_win_fair", "odds_draw_fair", "odds_away_win_fair"
]
# některé z těchto sloupců nemusí být v testu (pokud je nemáš), tak to ošetříme
keep_cols = [c for c in keep_cols if c in df.columns]

out = df[keep_cols].copy()
out.to_csv(OUT_PATH, index=False)
print("Saved 1X2 fair odds to:", OUT_PATH)

# quick sanity print
print("\nSanity check p_sum (min/mean/max):",
      out["p_sum"].min(), out["p_sum"].mean(), out["p_sum"].max())
