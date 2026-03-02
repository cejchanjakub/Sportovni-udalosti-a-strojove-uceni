import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Splity, které chceme normalizovat
SPLITS = ["train", "val", "test", "live"]

# Vstupy
MANAGERS_PATH = DATA_DIR / "epl_main_managers_2015_2026.csv"

# Preferuj finální aliasy, pokud existují (už jednou ručně doplněné)
ALIASES_FINAL_PATH = DATA_DIR / "team_aliases_final.csv"
ALIASES_SUGGESTED_PATH = PROJECT_ROOT / "team_aliases_suggested.csv"

HOME_COL = "HomeTeam"
AWAY_COL = "AwayTeam"
MAN_TEAM_COL = "team"


def read_aliases(path: Path) -> pd.DataFrame:
    """Načti alias tabulku robustně (podpora , / ; a excelových Column1/2/3)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    aliases = pd.read_csv(path, sep=sep, engine="python")

    # když to vyjde jako 1 sloupec se středníky, zkus ; natvrdo
    if len(aliases.columns) == 1 and ";" in str(aliases.columns[0]):
        aliases = pd.read_csv(path, sep=";", engine="python")

    # Excel obecné názvy
    cols_lower = [str(c).lower().strip() for c in aliases.columns]
    if cols_lower == ["column1", "column2", "column3"]:
        aliases.columns = ["raw_match_team", "suggested_manager_team", "match_type"]

    needed = {"raw_match_team", "suggested_manager_team", "match_type"}
    if not needed.issubset(set(aliases.columns)):
        raise ValueError(
            f"Alias soubor musí mít sloupce {needed}, ale má: {aliases.columns.tolist()}"
        )

    aliases["raw_match_team"] = aliases["raw_match_team"].astype(str).str.strip()
    aliases["suggested_manager_team"] = (
        aliases["suggested_manager_team"].fillna("").astype(str).str.strip()
    )
    aliases["match_type"] = aliases["match_type"].astype(str).str.strip()

    # vyhoď prázdné řádky (pokud existují)
    aliases = aliases[aliases["raw_match_team"] != ""].copy()

    # validace: žádné prázdné mapování
    missing = aliases[aliases["suggested_manager_team"] == ""]
    if len(missing) > 0:
        print("❗ Nevyplněné mapování v alias tabulce (doplň suggested_manager_team):")
        print(missing[["raw_match_team", "match_type"]].to_string(index=False))
        raise SystemExit(1)

    return aliases


def apply_team_map(matches: pd.DataFrame, team_map: dict) -> pd.DataFrame:
    """Přidá HomeTeam_std a AwayTeam_std pomocí mapy."""
    for col in [HOME_COL, AWAY_COL]:
        if col not in matches.columns:
            raise ValueError(f"V matches chybí sloupec {col}.")
        matches[col + "_std"] = matches[col].map(team_map).fillna(matches[col])
    return matches


def main():
    # --- Load managers ---
    if not MANAGERS_PATH.exists():
        raise FileNotFoundError(f"Chybí managers soubor: {MANAGERS_PATH}")
    man = pd.read_csv(MANAGERS_PATH)

    if MAN_TEAM_COL not in man.columns:
        raise ValueError(f"V managers CSV chybí sloupec '{MAN_TEAM_COL}'.")

    # --- Choose aliases source ---
    aliases_path = ALIASES_FINAL_PATH if ALIASES_FINAL_PATH.exists() else ALIASES_SUGGESTED_PATH
    if not aliases_path.exists():
        raise FileNotFoundError(
            f"Chybí alias soubor. Očekávám buď {ALIASES_FINAL_PATH} nebo {ALIASES_SUGGESTED_PATH}"
        )

    aliases = read_aliases(aliases_path)

    # map: match_team -> managers_team (standard)
    team_map = dict(zip(aliases["raw_match_team"], aliases["suggested_manager_team"]))

    # managers standard column (pro jistotu)
    man["team_std"] = man[MAN_TEAM_COL].astype(str).str.strip()

    # --- Apply to all splits ---
    saved = []
    all_std_teams = set()

    for split in SPLITS:
        in_path = DATA_DIR / f"{split}.csv"
        if not in_path.exists():
            print(f"⚠️ Přeskakuju '{split}' (soubor neexistuje): {in_path}")
            continue

        df = pd.read_csv(in_path)
        df = apply_team_map(df, team_map)

        out_path = DATA_DIR / f"{split}_norm.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        saved.append(out_path)

        std_teams_split = set(df[HOME_COL + "_std"]).union(set(df[AWAY_COL + "_std"]))
        all_std_teams |= std_teams_split

        uniq_before = len(set(df[HOME_COL]).union(set(df[AWAY_COL])))
        uniq_after = len(std_teams_split)
        print(f"[{split}] Unique teams BEFORE: {uniq_before}")
        print(f"[{split}] Unique teams AFTER : {uniq_after}")

    # --- Cross-check against managers set ---
    std_teams_man = set(man["team_std"].astype(str).str.strip())
    still_diff = sorted(all_std_teams - std_teams_man)
    if still_diff:
        print("\n⚠️ Tyto týmy po mapování stále nejsou v managers standardu (zkontroluj aliasy):")
        for t in still_diff:
            print(" -", t)
    else:
        print("\n✅ Po mapování sedí týmy mezi (všemi splity) a managers (na úrovni množin).")

    # --- Save managers_norm + aliases_final ---
    out_man = DATA_DIR / "managers_norm.csv"
    out_aliases = DATA_DIR / "team_aliases_final.csv"

    man.to_csv(out_man, index=False, encoding="utf-8")
    aliases.to_csv(out_aliases, index=False, encoding="utf-8")

    print("\nSaved:")
    for p in saved:
        print(" -", p)
    print(" -", out_man)
    print(" -", out_aliases)


if __name__ == "__main__":
    main()
