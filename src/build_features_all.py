# src/build_features_all.py

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"

ROLL_WINDOWS   = [3, 5, 10]
MANAGERS_PATH  = PROCESSED_DIR / "managers_norm.csv"


# ============================================================
# UTIL
# ============================================================

def _sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kickoff_dt"] = pd.to_datetime(df["Date"])
    return df.sort_values("kickoff_dt").reset_index(drop=True)


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def _rolling_team_features(df: pd.DataFrame, home_stat: str, away_stat: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    for w in ROLL_WINDOWS:
        df[f"home_{prefix}_for_roll{w}"] = (
            df.groupby("HomeTeam")[home_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"home_{prefix}_against_roll{w}"] = (
            df.groupby("HomeTeam")[away_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"away_{prefix}_for_roll{w}"] = (
            df.groupby("AwayTeam")[away_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"away_{prefix}_against_roll{w}"] = (
            df.groupby("AwayTeam")[home_stat]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"diff_{prefix}_for_roll{w}"] = (
            df[f"home_{prefix}_for_roll{w}"] - df[f"away_{prefix}_for_roll{w}"]
        )
        df[f"diff_{prefix}_against_roll{w}"] = (
            df[f"home_{prefix}_against_roll{w}"] - df[f"away_{prefix}_against_roll{w}"]
        )
    return df


# ============================================================
# FORMA
# ============================================================

def _add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    if "FTHG" not in df.columns or "FTAG" not in df.columns:
        return df
    df = df.copy()
    conditions = [df["FTHG"] > df["FTAG"], df["FTHG"] == df["FTAG"]]
    df["_home_points"] = np.select(conditions, [3.0, 1.0], default=0.0)
    df["_away_points"] = np.select(
        [df["FTAG"] > df["FTHG"], df["FTHG"] == df["FTAG"]], [3.0, 1.0], default=0.0
    )
    mask_no_result = df["FTHG"].isna() | df["FTAG"].isna()
    df.loc[mask_no_result, ["_home_points", "_away_points"]] = np.nan
    for w in ROLL_WINDOWS:
        df[f"home_points_roll{w}"] = (
            df.groupby("HomeTeam")["_home_points"]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"away_points_roll{w}"] = (
            df.groupby("AwayTeam")["_away_points"]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"diff_points_roll{w}"] = df[f"home_points_roll{w}"] - df[f"away_points_roll{w}"]
    for w in [5, 10]:
        df[f"home_form_home_roll{w}"] = (
            df.groupby("HomeTeam")["_home_points"]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"away_form_away_roll{w}"] = (
            df.groupby("AwayTeam")["_away_points"]
            .transform(lambda x: x.shift().rolling(w).mean())
        )
        df[f"diff_form_ha_roll{w}"] = (
            df[f"home_form_home_roll{w}"] - df[f"away_form_away_roll{w}"]
        )
    df = df.drop(columns=["_home_points", "_away_points"], errors="ignore")
    return df


# ============================================================
# TABULKA
# ============================================================

def _add_table_position_features(df: pd.DataFrame) -> pd.DataFrame:
    if "FTHG" not in df.columns or "FTAG" not in df.columns:
        return df
    if "season" not in df.columns:
        return df
    df = df.copy()
    df["kickoff_dt"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["season", "kickoff_dt"]).reset_index(drop=True)
    home_pos_list = [None] * len(df)
    away_pos_list = [None] * len(df)
    home_pts_list = [None] * len(df)
    away_pts_list = [None] * len(df)
    for season, season_idx in df.groupby("season", sort=False).groups.items():
        season_df = df.loc[season_idx].sort_values("kickoff_dt")
        points: dict[str, int] = {}
        all_teams = set(season_df["HomeTeam"].tolist() + season_df["AwayTeam"].tolist())
        for t in all_teams:
            points[t] = 0
        for idx, row in season_df.iterrows():
            home = str(row["HomeTeam"])
            away = str(row["AwayTeam"])
            home_pts_list[idx] = points.get(home, 0)
            away_pts_list[idx] = points.get(away, 0)
            sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
            pos_map = {team: i + 1 for i, (team, _) in enumerate(sorted_teams)}
            home_pos_list[idx] = pos_map.get(home, len(all_teams))
            away_pos_list[idx] = pos_map.get(away, len(all_teams))
            hg = row.get("FTHG")
            ag = row.get("FTAG")
            if pd.isna(hg) or pd.isna(ag):
                continue
            if hg > ag:
                points[home] = points.get(home, 0) + 3
            elif hg == ag:
                points[home] = points.get(home, 0) + 1
                points[away] = points.get(away, 0) + 1
            else:
                points[away] = points.get(away, 0) + 3
    df["home_table_pos"]    = home_pos_list
    df["away_table_pos"]    = away_pos_list
    df["home_table_points"] = home_pts_list
    df["away_table_points"] = away_pts_list
    df["table_pos_diff"]    = pd.to_numeric(df["home_table_pos"], errors="coerce") - pd.to_numeric(df["away_table_pos"], errors="coerce")
    df["table_points_diff"] = pd.to_numeric(df["home_table_points"], errors="coerce") - pd.to_numeric(df["away_table_points"], errors="coerce")
    return df


# ============================================================
# DAYS REST
# ============================================================

def _add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kickoff_dt"] = pd.to_datetime(df["Date"])
    df = df.sort_values("kickoff_dt").reset_index(drop=True)
    last_match: dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []
    for _, row in df.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        d = row["kickoff_dt"]
        home_rest.append((d - last_match[home]).days if home in last_match else np.nan)
        away_rest.append((d - last_match[away]).days if away in last_match else np.nan)
        last_match[home] = d
        last_match[away] = d
    df["home_days_rest"] = home_rest
    df["away_days_rest"] = away_rest
    df["days_rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
    df["is_midweek"]     = df["kickoff_dt"].dt.dayofweek.isin([1, 2, 3]).astype(int)
    return df


# ============================================================
# COACH FEATURES
# ============================================================

def _load_manager_intervals() -> pd.DataFrame | None:
    if not MANAGERS_PATH.exists():
        print(f"[WARN] Chybí {MANAGERS_PATH} – coach featury budou NaN.")
        return None

    man = pd.read_csv(MANAGERS_PATH, low_memory=False)

    def _parse_dates(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        mask_empty = (s == "") | (s.str.lower() == "nan")
        s2 = s.mask(mask_empty, other=pd.NA)
        mask_dash  = s2.notna() & s2.str.contains("-", regex=False)
        mask_slash = s2.notna() & s2.str.contains("/", regex=False)
        mask_rest  = s2.notna() & ~mask_dash & ~mask_slash
        out.loc[mask_dash]  = pd.to_datetime(s2.loc[mask_dash],  errors="coerce", dayfirst=False)
        out.loc[mask_slash] = pd.to_datetime(s2.loc[mask_slash], errors="coerce", dayfirst=True)
        out.loc[mask_rest]  = pd.to_datetime(s2.loc[mask_rest],  errors="coerce", dayfirst=True)
        return out

    man["start_date"] = _parse_dates(man["start_date"])
    man["end_date"]   = _parse_dates(man["end_date"])
    man = man[man["start_date"].notna()].copy()
    man = man[(man["team_std"].astype(str).str.strip() != "") &
              (man["coach_name"].astype(str).str.strip() != "")].copy()
    man = man.sort_values(["team_std", "start_date"]).reset_index(drop=True)

    man["_next_start"] = man.groupby("team_std")["start_date"].shift(-1)
    next_end = man["_next_start"] - pd.Timedelta(days=1)
    man["end_date"] = man["end_date"].where(man["end_date"].notna(), next_end)
    man["end_date"] = man["end_date"].where(
        man["_next_start"].isna() | (man["end_date"] < man["_next_start"]), next_end
    )
    man["end_date"] = man["end_date"].fillna(pd.Timestamp("2100-01-01"))
    man = man.drop(columns=["_next_start"])
    return man


def _coach_for_team_on_date(intervals: pd.DataFrame, team: str, date: pd.Timestamp):
    if pd.isna(date) or not team:
        return None, None
    sub = intervals[intervals["team_std"] == team]
    if sub.empty:
        return None, None
    hit = sub[(sub["start_date"] <= date) & (date <= sub["end_date"])]
    if not hit.empty:
        row = hit.iloc[-1]
        return row["coach_name"], row["start_date"]
    prev = sub[sub["start_date"] <= date]
    if not prev.empty:
        row = prev.iloc[-1]
        return row["coach_name"], row["start_date"]
    return None, None


def _add_coach_features(df: pd.DataFrame, intervals: pd.DataFrame | None) -> pd.DataFrame:
    df = df.copy()

    # Kickoff datetime – ošetři timezone
    if "kickoff" in df.columns:
        kdt = pd.to_datetime(df["kickoff"], errors="coerce", utc=True)
        df["kickoff_dt"] = kdt.dt.tz_localize(None)
    else:
        df["kickoff_dt"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    if intervals is None:
        for col in ["HomeCoachTenureDays", "AwayCoachTenureDays", "CoachTenureDiff",
                    "HomeCoachTenure_log1p", "AwayCoachTenure_log1p",
                    "NewHomeCoach_30", "NewAwayCoach_30",
                    "NewHomeCoach_60", "NewAwayCoach_60",
                    "NewHomeCoach_90", "NewAwayCoach_90"]:
            df[col] = np.nan
        return df

    home_tenure, away_tenure = [], []
    for _, row in df.iterrows():
        d  = row["kickoff_dt"]
        ht = str(row["HomeTeam"]).strip()
        at = str(row["AwayTeam"]).strip()
        _, hc_start = _coach_for_team_on_date(intervals, ht, d)
        _, ac_start = _coach_for_team_on_date(intervals, at, d)
        if hc_start is None or pd.isna(d):
            home_tenure.append(np.nan)
        else:
            home_tenure.append((d.normalize() - hc_start.normalize()).days)
        if ac_start is None or pd.isna(d):
            away_tenure.append(np.nan)
        else:
            away_tenure.append((d.normalize() - ac_start.normalize()).days)

    df["HomeCoachTenureDays"] = pd.array(home_tenure, dtype="Float64")
    df["AwayCoachTenureDays"] = pd.array(away_tenure, dtype="Float64")
    df["HomeCoachTenure_log1p"] = np.log1p(pd.to_numeric(df["HomeCoachTenureDays"], errors="coerce"))
    df["AwayCoachTenure_log1p"] = np.log1p(pd.to_numeric(df["AwayCoachTenureDays"], errors="coerce"))
    df["CoachTenureDiff"] = (
        pd.to_numeric(df["HomeCoachTenureDays"], errors="coerce") -
        pd.to_numeric(df["AwayCoachTenureDays"], errors="coerce")
    )
    for d in [30, 60, 90]:
        df[f"NewHomeCoach_{d}"] = (pd.to_numeric(df["HomeCoachTenureDays"], errors="coerce") <= d).astype("Int64")
        df[f"NewAwayCoach_{d}"] = (pd.to_numeric(df["AwayCoachTenureDays"], errors="coerce") <= d).astype("Int64")
        df.loc[df["HomeCoachTenureDays"].isna(), f"NewHomeCoach_{d}"] = pd.NA
        df.loc[df["AwayCoachTenureDays"].isna(), f"NewAwayCoach_{d}"] = pd.NA
    return df


# ============================================================
# ELO
# ============================================================

def compute_elo_for_all(
    all_df: pd.DataFrame,
    base_elo: float = 1500.0,
    k: float = 20.0,
    home_adv: float = 50.0,
) -> pd.DataFrame:
    df = all_df.copy()
    df["kickoff_dt"] = pd.to_datetime(df["Date"])
    df = df.sort_values("kickoff_dt").reset_index(drop=True)

    def expected(a, b):
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    elo: dict[str, float] = {}
    elo_home_list, elo_away_list, elo_diff_list, match_ids = [], [], [], []

    for _, row in df.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        eh = elo.get(home, base_elo)
        ea = elo.get(away, base_elo)
        elo_home_list.append(eh)
        elo_away_list.append(ea)
        elo_diff_list.append((eh + home_adv) - ea)
        match_ids.append(row.get("match_id", f"{home}_{away}_{row['kickoff_dt']}"))
        hg = row.get("FTHG")
        ag = row.get("FTAG")
        if pd.isna(hg) or pd.isna(ag):
            continue
        s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        e_home = expected(eh + home_adv, ea)
        elo[home] = eh + k * (s_home - e_home)
        elo[away] = ea + k * ((1.0 - s_home) - (1.0 - e_home))

    return pd.DataFrame({
        "match_id": match_ids,
        "elo_home": elo_home_list,
        "elo_away": elo_away_list,
        "elo_diff": elo_diff_list,
    })


# ============================================================
# REFEREE FEATURES
# ============================================================

def compute_referee_features_for_all(
    all_df: pd.DataFrame,
    window: int = 20,
    k_prior: float = 10.0,
    cards_col: str = "total_cards",
    fouls_col: str = "total_fouls",
) -> pd.DataFrame:
    df = all_df.copy()
    df["kickoff_dt"] = pd.to_datetime(df["Date"])
    df = df.sort_values("kickoff_dt").reset_index(drop=True)
    for col in [cards_col, fouls_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    records = []
    ref_history: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        ref = str(row.get("Referee", "")).strip()
        match_id = row.get("match_id", None)
        is_unknown = ref in ("", "nan", "Unknown")
        all_so_far = [r for hist in ref_history.values() for r in hist]
        league_cards_median = float(np.median([r["cards"] for r in all_so_far if not np.isnan(r["cards"])])) if all_so_far else 0.0
        league_fouls_median = float(np.median([r["fouls"] for r in all_so_far if not np.isnan(r["fouls"])])) if all_so_far else 0.0
        if is_unknown or ref not in ref_history or len(ref_history[ref]) == 0:
            records.append({"match_id": match_id, "ref_matches_count_last20": 0.0,
                            "ref_cards_avg_last20": league_cards_median,
                            "ref_fouls_avg_last20": league_fouls_median, "ref_unknown": 1.0})
        else:
            last = ref_history[ref][-window:]
            n = float(len(last))
            raw_cards = float(np.nanmean([r["cards"] for r in last])) if last else league_cards_median
            raw_fouls = float(np.nanmean([r["fouls"] for r in last])) if last else league_fouls_median
            w = n / (n + k_prior)
            records.append({"match_id": match_id, "ref_matches_count_last20": n,
                            "ref_cards_avg_last20": w * raw_cards + (1 - w) * league_cards_median,
                            "ref_fouls_avg_last20": w * raw_fouls + (1 - w) * league_fouls_median,
                            "ref_unknown": 0.0})
        if not is_unknown:
            cards_val = row.get(cards_col, np.nan)
            fouls_val = row.get(fouls_col, np.nan)
            if pd.isna(cards_val) and pd.isna(fouls_val):
                continue
            if ref not in ref_history:
                ref_history[ref] = []
            ref_history[ref].append({
                "cards": float(cards_val) if not pd.isna(cards_val) else np.nan,
                "fouls": float(fouls_val) if not pd.isna(fouls_val) else np.nan,
            })
    return pd.DataFrame(records)



# ============================================================
# H2H FEATURES
# ============================================================

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočítá head-to-head (H2H) statistiky pro každý zápas.

    Pro každý zápas (HomeTeam, AwayTeam, Date) najde předchozí
    vzájemné zápasy (oba směry) a spočítá rolling statistiky.

    Výstupní featury (jako DataFrame s match_id + h2h sloupce):
      h2h_avg_yellow_last3   - průměr žlutých karet v posl. 3 H2H
      h2h_avg_yellow_last5   - průměr žlutých karet v posl. 5 H2H
      h2h_avg_goals_last3    - průměr gólů v posl. 3 H2H
      h2h_avg_goals_last5    - průměr gólů v posl. 5 H2H
      h2h_avg_corners_last3  - průměr rohů v posl. 3 H2H
      h2h_avg_corners_last5  - průměr rohů v posl. 5 H2H
      h2h_avg_fouls_last3    - průměr faulů v posl. 3 H2H
      h2h_matches_count      - počet dostupných H2H zápasů
    """
    df = _sort_matches(df.copy())

    h2h_cols = [
        "h2h_avg_yellow_last3", "h2h_avg_yellow_last5",
        "h2h_avg_goals_last3",  "h2h_avg_goals_last5",
        "h2h_avg_corners_last3","h2h_avg_corners_last5",
        "h2h_avg_fouls_last3",
        "h2h_matches_count",
    ]
    for col in h2h_cols:
        df[col] = np.nan

    has_yellow  = "HY" in df.columns and "AY" in df.columns
    has_goals   = "FTHG" in df.columns and "FTAG" in df.columns
    has_corners = "HC" in df.columns and "AC" in df.columns
    has_fouls   = "HF" in df.columns and "AF" in df.columns

    if "Date" not in df.columns:
        if "match_id" in df.columns:
            return df[["match_id"] + h2h_cols]
        return df[h2h_cols]

    df["_date_dt"] = pd.to_datetime(df["Date"], errors="coerce")

    for idx in df.index:
        row  = df.loc[idx]
        home = row.get("HomeTeam")
        away = row.get("AwayTeam")
        date = row.get("_date_dt")

        if pd.isna(date) or not home or not away:
            continue

        # Předchozí vzájemné zápasy – oba směry
        mask = (
            (
                ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
                ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
            ) &
            (df["_date_dt"] < date)
        )
        h2h = df[mask].sort_values("_date_dt", ascending=False)
        df.at[idx, "h2h_matches_count"] = float(len(h2h))

        if h2h.empty:
            continue

        # Žluté karty
        if has_yellow:
            tot = h2h["HY"].fillna(0) + h2h["AY"].fillna(0)
            valid = tot[h2h["HY"].notna() & h2h["AY"].notna()]
            if len(valid) >= 1:
                df.at[idx, "h2h_avg_yellow_last3"] = float(valid.head(3).mean())
                df.at[idx, "h2h_avg_yellow_last5"] = float(valid.head(5).mean())

        # Góly
        if has_goals:
            tot = h2h["FTHG"].fillna(0) + h2h["FTAG"].fillna(0)
            valid = tot[h2h["FTHG"].notna() & h2h["FTAG"].notna()]
            if len(valid) >= 1:
                df.at[idx, "h2h_avg_goals_last3"] = float(valid.head(3).mean())
                df.at[idx, "h2h_avg_goals_last5"] = float(valid.head(5).mean())

        # Rohy
        if has_corners:
            tot = h2h["HC"].fillna(0) + h2h["AC"].fillna(0)
            valid = tot[h2h["HC"].notna() & h2h["AC"].notna()]
            if len(valid) >= 1:
                df.at[idx, "h2h_avg_corners_last3"] = float(valid.head(3).mean())
                df.at[idx, "h2h_avg_corners_last5"] = float(valid.head(5).mean())

        # Fauly
        if has_fouls:
            tot = h2h["HF"].fillna(0) + h2h["AF"].fillna(0)
            valid = tot[h2h["HF"].notna() & h2h["AF"].notna()]
            if len(valid) >= 1:
                df.at[idx, "h2h_avg_fouls_last3"] = float(valid.head(3).mean())

    df = df.drop(columns=["_date_dt"], errors="ignore")

    # Vrať match_id + h2h sloupce jako lookup DataFrame
    if "match_id" in df.columns:
        return df[["match_id"] + h2h_cols]
    return df[h2h_cols]

# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    df: pd.DataFrame,
    elo_lookup: pd.DataFrame,
    referee_lookup: pd.DataFrame,
    manager_intervals: pd.DataFrame | None = None,
    h2h_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Hlavní funkce pro výpočet featur.
    manager_intervals: předpočítané intervaly z _load_manager_intervals().
    Pokud None, načte se automaticky z managers_norm.csv.
    """
    df = _sort_matches(df)
    numeric_cols = ["FTHG", "FTAG", "HF", "AF", "HC", "AC", "HY", "AY", "HST", "AST"]
    df = _ensure_numeric(df, numeric_cols)

    if "FTHG" in df.columns and "FTAG" in df.columns:
        df["total_goals"] = df["FTHG"] + df["FTAG"]
    if "HF" in df.columns and "AF" in df.columns:
        df["total_fouls"] = df["HF"] + df["AF"]
    if "HC" in df.columns and "AC" in df.columns:
        df["total_corners"] = df["HC"] + df["AC"]
    if "HY" in df.columns and "AY" in df.columns:
        df["total_cards"] = df["HY"] + df["AY"]
    if "HST" in df.columns and "AST" in df.columns:
        df["total_shots_on_target"] = df["HST"] + df["AST"]

    if elo_lookup is not None and "match_id" in df.columns:
        df = df.drop(columns=["elo_home", "elo_away", "elo_diff"], errors="ignore")
        df = df.merge(elo_lookup, on="match_id", how="left")

    if referee_lookup is not None and "match_id" in df.columns:
        ref_cols = ["ref_matches_count_last20", "ref_cards_avg_last20", "ref_fouls_avg_last20", "ref_unknown"]
        df = df.drop(columns=[c for c in ref_cols if c in df.columns], errors="ignore")
        df = df.merge(referee_lookup, on="match_id", how="left")

    if "FTHG" in df.columns and "FTAG" in df.columns:
        df = _rolling_team_features(df, "FTHG", "FTAG", "goals")
    if "HC" in df.columns and "AC" in df.columns:
        df = _rolling_team_features(df, "HC", "AC", "corners")
    if "HF" in df.columns and "AF" in df.columns:
        df = _rolling_team_features(df, "HF", "AF", "fouls")
    if "HY" in df.columns and "AY" in df.columns:
        df = _rolling_team_features(df, "HY", "AY", "yellow")
    if "HST" in df.columns and "AST" in df.columns:
        df = _rolling_team_features(df, "HST", "AST", "shotsot")

    df = _add_form_features(df)
    df = _add_table_position_features(df)
    df = _add_days_rest(df)

    # Coach featury – načti intervals pokud nebyly předány
    if manager_intervals is None:
        manager_intervals = _load_manager_intervals()
    df = _add_coach_features(df, manager_intervals)

    # H2H featury
    if h2h_lookup is not None and "match_id" in df.columns:
        h2h_cols = [c for c in h2h_lookup.columns if c != "match_id"]
        df = df.drop(columns=h2h_cols, errors="ignore")
        df = df.merge(h2h_lookup, on="match_id", how="left")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    all_splits = []
    for split in ["train", "val", "test", "live"]:
        coach_path = PROCESSED_DIR / f"{split}_with_coach_features.csv"
        base_path  = PROCESSED_DIR / f"{split}.csv"
        path = coach_path if coach_path.exists() else base_path
        if not path.exists():
            print(f"[SKIP] {path} neexistuje.")
            continue
        tmp = pd.read_csv(path, low_memory=False)
        tmp["_split"]  = split
        tmp["_source"] = path.name
        all_splits.append(tmp)

    if not all_splits:
        raise FileNotFoundError("Nenašel jsem žádné split soubory v data/processed/")

    all_df = pd.concat(all_splits, ignore_index=True)

    print("\nNačtené zdroje:")
    for split in ["train", "val", "test", "live"]:
        src = all_df[all_df["_split"] == split]["_source"].iloc[0] if (all_df["_split"] == split).any() else "—"
        print(f"  [{split}] {src}")

    print("\nPočítám ELO přes celý dataset...")
    elo_lookup = compute_elo_for_all(all_df)
    print(f"ELO spočítáno pro {len(elo_lookup)} zápasů.")

    print("\nPočítám Referee features přes celý dataset...")
    referee_lookup = compute_referee_features_for_all(all_df)
    print(f"Referee features spočítány pro {len(referee_lookup)} zápasů.")

    print("\nPočítám H2H featury přes celý dataset...")
    h2h_lookup = compute_h2h_features(all_df)
    print(f"H2H featury spočítány pro {len(h2h_lookup)} zápasů.")

    print("\nNačítám manager intervals...")
    manager_intervals = _load_manager_intervals()
    if manager_intervals is not None:
        print(f"Manager intervals načteny: {len(manager_intervals)} záznamů.")
    else:
        print("[WARN] managers_norm.csv nenalezen – coach featury budou NaN.")

    for split in ["train", "val", "test", "live"]:
        coach_path = PROCESSED_DIR / f"{split}_with_coach_features.csv"
        base_path  = PROCESSED_DIR / f"{split}.csv"
        in_path    = coach_path if coach_path.exists() else base_path
        if not in_path.exists():
            print(f"\n[SKIP] {in_path} neexistuje.")
            continue
        out_path = FEATURES_DIR / f"{split}_features.csv"
        df = pd.read_csv(in_path, low_memory=False)
        df = build_features(df, elo_lookup, referee_lookup, manager_intervals, h2h_lookup)
        df = df.drop(columns=["_split", "_source"], errors="ignore")
        df.to_csv(out_path, index=False)
        print(f"\n[{split}] Saved: {out_path} | rows: {len(df)} | cols: {len(df.columns)}")
        checks = {
            "Coach":   ["HomeCoachTenureDays", "CoachTenureDiff", "NewHomeCoach_30"],
            "Referee": ["ref_cards_avg_last20", "ref_fouls_avg_last20"],
            "Forma":   ["home_points_roll5", "home_form_home_roll5"],
            "Tabulka": ["home_table_pos", "table_pos_diff"],
            "Rest":    ["home_days_rest", "is_midweek"],
            "H2H":     ["h2h_avg_yellow_last3", "h2h_avg_goals_last3", "h2h_matches_count"],
        }
        for label, cols in checks.items():
            present = [c for c in cols if c in df.columns]
            if present:
                print(f"  ✓ {label}: {present}")
            else:
                print(f"  ⚠ Chybí {label} features")


if __name__ == "__main__":
    main()