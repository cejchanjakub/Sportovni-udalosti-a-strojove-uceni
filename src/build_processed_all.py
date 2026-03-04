# src/build_processed_all.py

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "interim" / "EPL_all_seasons_interim.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_all_seasons_processed.csv"

SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.85  # 70–85 % = val, 85–100 % = test


def ensure_col(df: pd.DataFrame, col: str, default) -> None:
    if col not in df.columns:
        df[col] = default


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    tmp = pd.DataFrame({c: to_num(df[c]) for c in cols})
    return tmp.sum(axis=1)


def safe_eq_int(df: pd.DataFrame, col: str, value: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    return (df[col] == value).astype("Int64")


def main() -> None:
    df = pd.read_csv(IN_PATH, low_memory=False)

    # --- BASIC REQUIRED COLUMNS ---
    ensure_col(df, "Date", pd.NA)
    ensure_col(df, "Time", "00:00")
    ensure_col(df, "HomeTeam", "")
    ensure_col(df, "AwayTeam", "")

    # --- DATE & TIME ---
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Time"] = df["Time"].fillna("00:00").astype(str)

    date_str = df["Date"].dt.strftime("%Y-%m-%d")
    df["kickoff"] = pd.to_datetime(date_str + " " + df["Time"], errors="coerce")

    # --- MATCH ID ---
    kickoff_part = df["kickoff"].dt.strftime("%Y%m%d").fillna("unknown_date")
    home_part = df["HomeTeam"].astype(str).str.replace(" ", "", regex=False).str.lower()
    away_part = df["AwayTeam"].astype(str).str.replace(" ", "", regex=False).str.lower()
    df["match_id"] = kickoff_part + "_" + home_part + "_" + away_part

    # --- TARGETS ---
    df["home_win"] = safe_eq_int(df, "FTR", "H")
    df["draw"]     = safe_eq_int(df, "FTR", "D")
    df["away_win"] = safe_eq_int(df, "FTR", "A")

    ensure_col(df, "FTHG", pd.NA)
    ensure_col(df, "FTAG", pd.NA)
    df["total_goals"] = safe_sum(df, ["FTHG", "FTAG"]).astype("Float64")
    df["over_2_5"] = (df["total_goals"] > 2.5).astype("Int64")
    df.loc[df["total_goals"].isna(), "over_2_5"] = pd.NA

    df["total_corners"] = safe_sum(df, ["HC", "AC"]).astype("Float64")
    df["total_sot"]     = safe_sum(df, ["HST", "AST"]).astype("Float64")
    df["total_fouls"]   = safe_sum(df, ["HF", "AF"]).astype("Float64")

    ensure_col(df, "HY", pd.NA)
    ensure_col(df, "AY", pd.NA)
    if "HR" not in df.columns:
        df["HR"] = 0
    if "AR" not in df.columns:
        df["AR"] = 0

    df["total_cards"] = (
        to_num(df["HY"]).astype("Float64")
        + to_num(df["AY"]).astype("Float64")
        + to_num(df["HR"]).astype("Float64")
        + to_num(df["AR"]).astype("Float64")
    )
    df.loc[to_num(df["HY"]).isna() | to_num(df["AY"]).isna(), "total_cards"] = pd.NA

    # --- SORT ---
    df = df.sort_values("kickoff", na_position="last").reset_index(drop=True)

    # --- ULOŽENÍ CELÉHO DATASETU ---
    df.to_csv(OUT_PATH, index=False)
    print(f"Rows: {df.shape[0]}")
    print(f"Unique match_id: {df['match_id'].nunique()}")
    print(f"Missing kickoff: {int(df['kickoff'].isna().sum())}")

    if "season" in df.columns:
        print("\nRows per season:")
        print(df.groupby("season").size().sort_index())

    print(f"\nSaved processed dataset to: {OUT_PATH}")

    # --- TRAIN / VAL / TEST SPLIT ---
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    n = len(df)
    train_end = int(n * SPLIT_TRAIN)
    val_end   = int(n * SPLIT_VAL)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv",     index=False)
    test.to_csv(processed_dir / "test.csv",   index=False)

    print(f"\nSplits uloženy do {processed_dir}:")
    print(f"  train: {len(train)} řádků  ({train['kickoff'].min().date()} → {train['kickoff'].max().date()})")
    print(f"  val:   {len(val)} řádků  ({val['kickoff'].min().date()} → {val['kickoff'].max().date()})")
    print(f"  test:  {len(test)} řádků  ({test['kickoff'].min().date()} → {test['kickoff'].max().date()})")


if __name__ == "__main__":
    main()