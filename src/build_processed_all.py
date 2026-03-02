import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "interim" / "EPL_all_seasons_interim.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_all_seasons_processed.csv"


def ensure_col(df: pd.DataFrame, col: str, default) -> None:
    """Ensure a column exists; if not, create it with a default value."""
    if col not in df.columns:
        df[col] = default


def to_num(series: pd.Series) -> pd.Series:
    """Coerce series to numeric (float), invalid parsing -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def safe_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Sum columns safely:
    - If any required column is missing -> return NaN for all rows (float).
    - Otherwise, coerce to numeric and sum row-wise.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")

    tmp = pd.DataFrame({c: to_num(df[c]) for c in cols})
    return tmp.sum(axis=1)


def safe_eq_int(df: pd.DataFrame, col: str, value: str) -> pd.Series:
    """
    Create nullable int indicator for (df[col] == value).
    If col missing -> all NA.
    """
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    return (df[col] == value).astype("Int64")


def main() -> None:
    df = pd.read_csv(IN_PATH, low_memory=False)

    # --- BASIC REQUIRED COLUMNS (create if missing) ---
    ensure_col(df, "Date", pd.NA)
    ensure_col(df, "Time", "00:00")
    ensure_col(df, "HomeTeam", "")
    ensure_col(df, "AwayTeam", "")

    # --- DATE & TIME ---
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Time"] = df["Time"].fillna("00:00").astype(str)

    # Build kickoff datetime
    date_str = df["Date"].dt.strftime("%Y-%m-%d")
    df["kickoff"] = pd.to_datetime(date_str + " " + df["Time"], errors="coerce")

    # --- MATCH ID ---
    # Make it robust if kickoff is missing
    kickoff_part = df["kickoff"].dt.strftime("%Y%m%d").fillna("unknown_date")
    home_part = df["HomeTeam"].astype(str).str.replace(" ", "", regex=False).str.lower().fillna("unknown_home")
    away_part = df["AwayTeam"].astype(str).str.replace(" ", "", regex=False).str.lower().fillna("unknown_away")

    df["match_id"] = kickoff_part + "_" + home_part + "_" + away_part

    # --- TARGETS (nullable Int64 for robustness) ---
    df["home_win"] = safe_eq_int(df, "FTR", "H")
    df["draw"] = safe_eq_int(df, "FTR", "D")
    df["away_win"] = safe_eq_int(df, "FTR", "A")

    # Goals
    ensure_col(df, "FTHG", pd.NA)
    ensure_col(df, "FTAG", pd.NA)
    df["total_goals"] = safe_sum(df, ["FTHG", "FTAG"]).astype("Float64")
    df["over_2_5"] = (df["total_goals"] > 2.5).astype("Int64")
    # If total_goals is NA, over_2_5 should be NA (not 0)
    df.loc[df["total_goals"].isna(), "over_2_5"] = pd.NA

    # Corners / Cards / Shots / Fouls (safe)
    df["total_corners"] = safe_sum(df, ["HC", "AC"]).astype("Float64")
    df["total_sot"] = safe_sum(df, ["HST", "AST"]).astype("Float64")
    df["total_fouls"] = safe_sum(df, ["HF", "AF"]).astype("Float64")

    # Cards: treat reds as optional; if missing, they become NA and sum becomes NA.
    # Better: allow missing reds by treating them as 0 if yellow exists.
    # We'll do a pragmatic approach: HY/AY required, HR/AR optional (missing -> 0).
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

    # If HY/AY are missing/NA, cards should be NA (not just reds)
    df.loc[to_num(df["HY"]).isna() | to_num(df["AY"]).isna(), "total_cards"] = pd.NA

    # --- SORT ---
    df = df.sort_values("kickoff", na_position="last").reset_index(drop=True)

    # --- FINAL SANITY ---
    print("Rows:", df.shape[0])
    print("Unique match_id:", df["match_id"].nunique())
    print("Missing kickoff:", int(df["kickoff"].isna().sum()))

    if "season" in df.columns:
        print("\nRows per season:")
        print(df.groupby("season").size().sort_index())
    else:
        print("\n[WARN] Column 'season' not found in interim data; skipping season summary.")

    df.to_csv(OUT_PATH, index=False)
    print("\nSaved processed dataset to:", OUT_PATH)


if __name__ == "__main__":
    main()
