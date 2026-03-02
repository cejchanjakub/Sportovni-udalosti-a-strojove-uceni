import pandas as pd
import re
from difflib import get_close_matches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MATCHES_PATH = DATA_DIR / "train.csv"
MANAGERS_PATH = DATA_DIR / "epl_main_managers_2015_2026.csv"

OUT_ALIASES = PROJECT_ROOT / "team_aliases_suggested.csv"
OUT_TRAIN = DATA_DIR / "train_norm.csv"
OUT_MANAGERS = DATA_DIR / "managers_norm.csv"

def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"&", "and", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # zkrácení běžných suffixů
    s = s.replace("football club", "").replace("fc", "").replace("afc", "")
    s = s.replace("association football club", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 1) načti data
train = pd.read_csv(MATCHES_PATH)
man = pd.read_csv(MANAGERS_PATH)

# předpoklad: v train jsou HomeTeam/AwayTeam (u tebe to tak typicky je)
home_col = "HomeTeam"
away_col = "AwayTeam"
if home_col not in train.columns or away_col not in train.columns:
    raise ValueError(f"V train.csv nevidím sloupce {home_col}/{away_col}. Uprav home_col/away_col podle reality.")

match_teams = sorted(set(train[home_col].dropna().astype(str)).union(set(train[away_col].dropna().astype(str))))
manager_teams = sorted(set(man["team"].dropna().astype(str)))

match_norm = {t: norm(t) for t in match_teams}
manager_norm = {t: norm(t) for t in manager_teams}

match_norm_set = set(match_norm.values())
manager_norm_set = set(manager_norm.values())

# 2) co se neshoduje
only_in_matches = sorted(match_norm_set - manager_norm_set)
only_in_managers = sorted(manager_norm_set - match_norm_set)

print("Unique teams (matches):", len(match_teams))
print("Unique teams (managers):", len(manager_teams))
print("Norm diff -> only_in_matches:", len(only_in_matches))
print("Norm diff -> only_in_managers:", len(only_in_managers))

# 3) navrhni fuzzy párování: pro každé jméno z matches najdi nejlepší z managers
suggestions = []
for raw in match_teams:
    raw_n = match_norm[raw]
    if raw_n in manager_norm_set:
        # už sedí
        suggestions.append((raw, raw, "exact_norm"))
        continue

    candidates = list(manager_norm_set)
    # nejdřív přímé podobnosti v normalizovaném prostoru
    best = get_close_matches(raw_n, candidates, n=1, cutoff=0.80)
    if best:
        # najdi původní team z managers, který má tuhle norm hodnotu (první výskyt)
        target_norm = best[0]
        target_raw = next(t for t, tn in manager_norm.items() if tn == target_norm)
        suggestions.append((raw, target_raw, "fuzzy_norm"))
    else:
        suggestions.append((raw, "", "unmatched"))

aliases = pd.DataFrame(suggestions, columns=["raw_match_team", "suggested_manager_team", "match_type"])
aliases.to_csv(OUT_ALIASES, index=False, encoding="utf-8")
print("Saved:", OUT_ALIASES)

print("\nDŮLEŽITÉ: Otevři team_aliases_suggested.csv a oprav řádky s match_type='unmatched' nebo špatné fuzzy návrhy.")
print("Pak teprve aplikuj mapping (viz další krok níže).")
