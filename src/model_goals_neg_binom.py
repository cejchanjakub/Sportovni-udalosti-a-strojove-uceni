import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_poisson_deviance
import statsmodels.api as sm
from scipy.stats import nbinom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

train = pd.read_csv(DATA_DIR / "train.csv")
val   = pd.read_csv(DATA_DIR / "val.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

TARGET = "total_goals"

FEATURES = [
    "goals_expected_total_proxy_last5",
    "goals_expected_total_proxy_last10",
    "gd_last5_home",
    "gd_last5_away",
    "elo_diff",
    "is_midweek",
    "days_rest_home",
    "days_rest_away",
]

def prepare(df):
    X = df[FEATURES].copy()

    # bezpečnost pro začátečníka: doplň NA, ať to nespadne
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # doporučeno: škálování pro numerickou stabilitu (nemění interpretaci v DP zásadně,
    # jen pomáhá optimalizaci; koeficienty jsou pak ve škálovaných jednotkách)
    for col in X.columns:
        if col != "is_midweek":
            std = X[col].std()
            if std and std > 0:
                X[col] = (X[col] - X[col].mean()) / std

    X = sm.add_constant(X)
    y = df[TARGET].values
    return X, y

X_train, y_train = prepare(train)
X_val, y_val     = prepare(val)
X_test, y_test   = prepare(test)

# ------------------------------------------------------------
# 1) Poisson fit (pro start + odhad disperze)
# ------------------------------------------------------------
pois_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
pois_res = pois_model.fit()

mu_train = pois_res.predict(X_train)

# Odhad alpha (NB2): Var(Y|X)=mu + alpha*mu^2
# metoda momentů: alpha = mean(((y-mu)^2 - y) / mu^2)
eps = 1e-8
alpha_hat = np.mean(((y_train - mu_train) ** 2 - y_train) / (mu_train ** 2 + eps))
alpha_hat = float(max(alpha_hat, 1e-6))  # musí být kladné

print(f"Estimated dispersion alpha (method-of-moments): {alpha_hat:.6f}")

# ------------------------------------------------------------
# 2) GLM Negative Binomial s alpha_hat (už žádný default 1.0)
# ------------------------------------------------------------
nb_model = sm.GLM(
    y_train,
    X_train,
    family=sm.families.NegativeBinomial(alpha=alpha_hat)
)
nb_res = nb_model.fit()

print(nb_res.summary())

# --- Predict expected goals (mu) ---
val_mu  = nb_res.predict(X_val)
test_mu = nb_res.predict(X_test)

# --- Deviance (pro porovnání s Poissonem) ---
val_dev  = mean_poisson_deviance(y_val, val_mu)
test_dev = mean_poisson_deviance(y_test, test_mu)

print("\nNegative Binomial deviance (fixed alpha_hat):")
print(f"Validation: {val_dev:.4f}")
print(f"Test:       {test_dev:.4f}")

# --- Over/Under probabilities z NB2 ---
def over_prob_nb(mu, alpha, line):
    r = 1.0 / alpha
    p = r / (r + mu)
    k = int(np.floor(line))
    return 1.0 - nbinom.cdf(k, r, p)

test_out = test.copy()
test_out["mu_goals"] = test_mu
test_out["P_over_2_5"] = [over_prob_nb(m, alpha_hat, 2.5) for m in test_mu]
test_out["P_under_2_5"] = 1.0 - test_out["P_over_2_5"]

out_path = DATA_DIR / "goals_neg_binom_test_predictions.csv"
test_out[[
    "match_id", "season", "HomeTeam", "AwayTeam",
    "total_goals", "mu_goals", "P_over_2_5", "P_under_2_5"
]].to_csv(out_path, index=False)

print("\nSaved predictions to:", out_path)
