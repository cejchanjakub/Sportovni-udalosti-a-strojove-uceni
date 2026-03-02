import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PRED_PATH = DATA_DIR / "goals_poisson_test_predictions.csv"

# ⚠️ Sem dej cestu na soubor s kurzy (tvůj dataset s odds)
# Musí obsahovat match_id + odds_over_X_Y a odds_under_X_Y pro linky
ODDS_PATH = DATA_DIR / "odds_ou_test.csv"

LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

def over_prob_poisson(lmbda, line):
    k = int(np.floor(line))
    return 1.0 - poisson.cdf(k, lmbda)

pred = pd.read_csv(PRED_PATH)
odds = pd.read_csv(ODDS_PATH)

# --- Join on match_id ---
df = pred.merge(odds, on="match_id", how="inner")

# --- Compute model probs for all lines ---
for L in LINES:
    key = str(L).replace(".", "_")
    df[f"p_over_{key}"] = df["lambda_goals"].apply(lambda l: over_prob_poisson(l, L))
    df[f"p_under_{key}"] = 1.0 - df[f"p_over_{key}"]

# --- Helper: de-vig fair probs for O/U pair ---
def devig_pair(odds_over, odds_under):
    if odds_over <= 1.0 or odds_under <= 1.0:
        return np.nan, np.nan
    p_over_raw = 1.0 / odds_over
    p_under_raw = 1.0 / odds_under
    s = p_over_raw + p_under_raw
    return p_over_raw / s, p_under_raw / s

bets = []

for L in LINES:
    key = str(L).replace(".", "_")
    o_col = f"odds_over_{key}"
    u_col = f"odds_under_{key}"

    # skip if odds columns not present
    if o_col not in df.columns or u_col not in df.columns:
        continue

    for side in ["over", "under"]:
        if side == "over":
            odds_col = o_col
            p_model_col = f"p_over_{key}"
        else:
            odds_col = u_col
            p_model_col = f"p_under_{key}"

        tmp = df[["match_id", "season", "HomeTeam", "AwayTeam", "total_goals", odds_col, p_model_col]].copy()
        tmp = tmp.rename(columns={odds_col: "odds", p_model_col: "p_model"})
        tmp["line"] = L
        tmp["side"] = side

        # fair implied probs (de-vig) for this match+line
        fair = df[[o_col, u_col]].apply(lambda r: devig_pair(r[o_col], r[u_col]), axis=1, result_type="expand")
        tmp["p_fair_over"] = fair[0].values
        tmp["p_fair_under"] = fair[1].values

        tmp["p_fair"] = tmp["p_fair_over"] if side == "over" else tmp["p_fair_under"]

        # edge + EV
        tmp["edge"] = tmp["p_model"] - tmp["p_fair"]
        tmp["ev"] = tmp["p_model"] * (tmp["odds"] - 1.0) - (1.0 - tmp["p_model"])

        # realized result (profit on stake=1)
        # win condition:
        # Over L -> total_goals >= floor(L)+1
        # Under L -> total_goals <= floor(L)
        thr = int(np.floor(L))
        if side == "over":
            win = tmp["total_goals"] >= (thr + 1)
        else:
            win = tmp["total_goals"] <= thr

        tmp["profit"] = np.where(win, tmp["odds"] - 1.0, -1.0)

        bets.append(tmp)

bets_df = pd.concat(bets, ignore_index=True).dropna(subset=["p_fair", "odds", "p_model"])

# --- Basic selection rules (start simple, then tighten later) ---
# Example: only positive EV
selected = bets_df[bets_df["ev"] > 0].copy()

# Sort by EV descending
selected = selected.sort_values("ev", ascending=False)

out_all = DATA_DIR / "value_bets_all_candidates.csv"
out_sel = DATA_DIR / "value_bets_selected_ev_gt_0.csv"

bets_df.to_csv(out_all, index=False)
selected.to_csv(out_sel, index=False)

print("Saved all candidates to:", out_all)
print("Saved selected value bets to:", out_sel)

# --- Simple backtest summary on selected bets ---
n = len(selected)
roi = selected["profit"].sum() / n if n > 0 else 0.0
avg_ev = selected["ev"].mean() if n > 0 else 0.0

print("\nBacktest (selected EV>0, stake=1):")
print("Bets:", n)
print(f"Total profit: {selected['profit'].sum():.3f}")
print(f"ROI:          {roi:.4f}")
print(f"Avg EV:       {avg_ev:.4f}")

# Optional: breakdown by line
by_line = selected.groupby("line").agg(
    bets=("profit","size"),
    profit=("profit","sum"),
    roi=("profit", lambda x: x.sum()/len(x) if len(x)>0 else 0.0),
    avg_ev=("ev","mean"),
).reset_index()

print("\nBreakdown by line (selected):")
print(by_line.to_string(index=False))
