import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "val", "test", "live"]

MANAGERS_PATH = DATA_DIR / "managers_norm.csv"

HOME_TEAM_STD = "HomeTeam_std"
AWAY_TEAM_STD = "AwayTeam_std"

TEAM_COL_CAND = ["team_std", "team", "Team", "club", "Club"]
MANAGER_COL_CAND = ["coach_name", "coach", "Coach", "manager", "Manager", "manager_name", "name", "Name"]
START_COL_CAND = ["start_date", "start", "StartDate", "from", "From"]
END_COL_CAND = ["end_date", "end", "EndDate", "to", "To"]


def pick_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"V managers souboru nemůžu najít sloupec pro {label}. Kandidáti: {candidates}. "
        f"Sloupce: {df.columns.tolist()}"
    )


def parse_date_mixed(s: pd.Series) -> pd.Series:
    """
    Robustní parsování mixu:
    - YYYY-MM-DD -> dayfirst=False
    - DD/MM/YYYY -> dayfirst=True
    - fallback -> dayfirst=True
    """
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # prázdné / nan
    mask_empty = (s == "") | (s.str.lower() == "nan")
    s2 = s.mask(mask_empty, other=pd.NA)

    # ISO s pomlčkami
    mask_dash = s2.notna() & s2.str.contains("-", regex=False)
    out.loc[mask_dash] = pd.to_datetime(s2.loc[mask_dash], errors="coerce", dayfirst=False)

    # slashes DD/MM/YYYY
    mask_slash = s2.notna() & s2.str.contains("/", regex=False)
    out.loc[mask_slash] = pd.to_datetime(s2.loc[mask_slash], errors="coerce", dayfirst=True)

    # fallback pro zbytek
    mask_rest = s2.notna() & ~mask_dash & ~mask_slash
    out.loc[mask_rest] = pd.to_datetime(s2.loc[mask_rest], errors="coerce", dayfirst=True)

    return out


def build_intervals(man: pd.DataFrame, team_col: str, mgr_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    m = man.copy()

    m[team_col] = m[team_col].astype(str).str.strip()
    m[mgr_col] = m[mgr_col].astype(str).str.strip()

    m[start_col] = parse_date_mixed(m[start_col])
    m[end_col] = parse_date_mixed(m[end_col])

    # vyhoď bez start / team / coach
    m = m[~m[start_col].isna()].copy()
    m = m[(m[team_col] != "") & (m[mgr_col] != "")].copy()

    # seřadit
    m = m.sort_values([team_col, start_col]).reset_index(drop=True)

    # dopočítej end_date podle následujícího start_date (anti "Wenger forever")
    # end_effective = min(given_end, next_start - 1 day); pokud given_end chybí, použij next_start-1
    m["_next_start"] = m.groupby(team_col)[start_col].shift(-1)
    next_end = m["_next_start"] - pd.Timedelta(days=1)

    # kde end chybí, dosadíme next_end
    m[end_col] = m[end_col].where(~m[end_col].isna(), next_end)

    # pokud end existuje, ale přesahuje next_start, zkrátíme na next_end
    m[end_col] = m[end_col].where(m["_next_start"].isna() | (m[end_col] < m["_next_start"]), next_end)

    # poslední interval (bez next_start) necháme otevřený do budoucna
    m[end_col] = m[end_col].fillna(pd.Timestamp("2100-01-01"))

    m = m.drop(columns=["_next_start"])
    m = m.sort_values([team_col, start_col, end_col]).reset_index(drop=True)
    return m


def coach_for_team_on_date(intervals: pd.DataFrame, team: str, date: pd.Timestamp,
                           team_col: str, mgr_col: str, start_col: str, end_col: str):
    if pd.isna(date) or not team:
        return None, None

    sub = intervals[intervals[team_col] == team]
    if sub.empty:
        return None, None

    hit = sub[(sub[start_col] <= date) & (date <= sub[end_col])]
    if not hit.empty:
        row = hit.iloc[-1]
        return row[mgr_col], row[start_col]

    # fallback: poslední start před datem (kdyby end_date byla děravá)
    prev = sub[sub[start_col] <= date]
    if not prev.empty:
        row = prev.iloc[-1]
        return row[mgr_col], row[start_col]

    return None, None


def add_coaches(matches: pd.DataFrame, intervals: pd.DataFrame,
                team_col: str, mgr_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    df = matches.copy()

    # kickoff_dt
    if "kickoff" in df.columns:
        df["kickoff_dt"] = pd.to_datetime(df["kickoff"], errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce", dayfirst=True)
        df["Time"] = df.get("Time", "00:00").fillna("00:00").astype(str)
        df["kickoff_dt"] = pd.to_datetime(
            df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"], errors="coerce"
        )

    if HOME_TEAM_STD not in df.columns or AWAY_TEAM_STD not in df.columns:
        raise ValueError(f"Chybí {HOME_TEAM_STD}/{AWAY_TEAM_STD}. Spusť nejdřív appla_team_aliasses.py")

    home_coach, away_coach = [], []
    home_tenure, away_tenure = [], []

    for _, r in df.iterrows():
        d = r["kickoff_dt"]
        ht = str(r[HOME_TEAM_STD]).strip()
        at = str(r[AWAY_TEAM_STD]).strip()

        hc, hc_start = coach_for_team_on_date(intervals, ht, d, team_col, mgr_col, start_col, end_col)
        ac, ac_start = coach_for_team_on_date(intervals, at, d, team_col, mgr_col, start_col, end_col)

        home_coach.append(hc)
        away_coach.append(ac)

        if hc_start is None or pd.isna(d):
            home_tenure.append(pd.NA)
        else:
            home_tenure.append((d.normalize() - hc_start.normalize()).days)

        if ac_start is None or pd.isna(d):
            away_tenure.append(pd.NA)
        else:
            away_tenure.append((d.normalize() - ac_start.normalize()).days)

    df["HomeCoach"] = home_coach
    df["AwayCoach"] = away_coach
    df["HomeCoachTenureDays"] = pd.Series(home_tenure, dtype="Float64")
    df["AwayCoachTenureDays"] = pd.Series(away_tenure, dtype="Float64")
    return df


def main():
    if not MANAGERS_PATH.exists():
        raise FileNotFoundError(f"Chybí {MANAGERS_PATH}. Spusť nejdřív appla_team_aliasses.py")

    man = pd.read_csv(MANAGERS_PATH, low_memory=False)

    team_col = pick_col(man, TEAM_COL_CAND, "tým")
    mgr_col = pick_col(man, MANAGER_COL_CAND, "jméno trenéra")
    start_col = pick_col(man, START_COL_CAND, "datum začátku")
    end_col = pick_col(man, END_COL_CAND, "datum konce")

    intervals = build_intervals(man, team_col, mgr_col, start_col, end_col)

    print("Managers intervals:", len(intervals))
    print("Teams in managers:", intervals[team_col].nunique())
    print("Using cols:", {"team": team_col, "coach": mgr_col, "start": start_col, "end": end_col})

    for split in SPLITS:
        in_path = DATA_DIR / f"{split}_norm.csv"
        if not in_path.exists():
            print(f"⚠️ Přeskakuju {split}, neexistuje: {in_path}")
            continue

        matches = pd.read_csv(in_path, low_memory=False)
        out = add_coaches(matches, intervals, team_col, mgr_col, start_col, end_col)

        miss_home = out["HomeCoach"].isna().mean() * 100
        miss_away = out["AwayCoach"].isna().mean() * 100

        print(f"\n[{split}] Missing HomeCoach: {miss_home:.2f}%")
        print(f"[{split}] Missing AwayCoach: {miss_away:.2f}%")

        out_path = DATA_DIR / f"{split}_with_coaches.csv"
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{split}] Saved: {out_path}")

        cols_preview = [
            HOME_TEAM_STD, AWAY_TEAM_STD, "kickoff_dt",
            "HomeCoach", "AwayCoach",
            "HomeCoachTenureDays", "AwayCoachTenureDays"
        ]
        print(out[cols_preview].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
