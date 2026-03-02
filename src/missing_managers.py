import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

df = pd.read_csv(DATA_DIR / "train_with_coaches.csv")
train = pd.read_csv(DATA_DIR / "train_norm.csv")
man = pd.read_csv(DATA_DIR / "managers_norm.csv")

# kolik chybí podle týmů
missing_home = df[df["HomeCoach"].isna()]
missing_away = df[df["AwayCoach"].isna()]

print("Missing HomeCoach rows:", len(missing_home))
print("Missing AwayCoach rows:", len(missing_away))

print("\nTop missing HOME teams:")
print(missing_home["HomeTeam_std"].value_counts().head(15))

print("\nTop missing AWAY teams:")
print(missing_away["AwayTeam_std"].value_counts().head(15))

# zkontroluj, jestli v managers nejsou NaN start_date
man["start_date"] = pd.to_datetime(man["start_date"], errors="coerce")
man["end_date"] = man["end_date"].replace("", pd.NA)
man["end_date"] = pd.to_datetime(man["end_date"], errors="coerce")

print("\nManagers with NaT start_date:", man["start_date"].isna().sum())
print("Managers with NaT end_date  :", man["end_date"].isna().sum())

print("\nExample rows with missing start_date:")
print(man[man["start_date"].isna()][["team_std","coach_name","start_date","end_date"]].head(10).to_string(index=False))
