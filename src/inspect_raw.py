import pandas as pd
from pathlib import Path

# === PATH SETUP ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "data" / "raw" / "EPL_2024-25.csv"

# === LOAD ===
df = pd.read_csv(FILE_PATH)

print("=== BASIC INFO ===")
print(f"Rows (matches): {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n=== FIRST 10 COLUMNS ===")
print(list(df.columns[:10]))

print("\n=== LAST 10 COLUMNS ===")
print(list(df.columns[-10:]))

print("\n=== MISSING VALUES (TOP 15) ===")
print(df.isna().sum().sort_values(ascending=False).head(15))

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== DATA TYPES ===")
print(df.dtypes)
