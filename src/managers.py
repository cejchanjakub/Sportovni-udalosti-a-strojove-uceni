import re
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://en.wikipedia.org/wiki/List_of_Premier_League_managers"

# Rozsah pro sezóny 2015/16 až 2025/26 (pragmaticky datumově)
RANGE_START = pd.Timestamp("2015-07-01")
RANGE_END = pd.Timestamp("2026-06-30")

# --- HTTP (aby nebyl 403) ---
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://en.wikipedia.org/",
})

retry = Retry(
    total=5,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)


def fetch_html(url: str) -> str:
    time.sleep(0.8)
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def find_managers_table(tables):
    """
    Najde tabulku s hlavičkami: Name, Club, From, Until (a často i Years in League).
    """
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if all(x in cols for x in ["name", "club", "from", "until"]):
            return t
    raise RuntimeError("Nenašel jsem tabulku 'Managers' (Name/Club/From/Until). Wikipedia mohla změnit strukturu.")


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\[.*?\]", "", s)  # odstraní reference [1]
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_date(x):
    """
    Vrací pandas Timestamp nebo NaT.
    """
    s = clean_text(x)
    if not s or s in {"—", "-"}:
        return pd.NaT
    if "present" in s.lower():
        return pd.NaT
    # typicky "22 December 2019"
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def parse_years_in_league(x):
    """
    Z "2019–" nebo "2018–2019" udělá (start_year, end_year or None).
    """
    s = clean_text(x)
    if not s:
        return (None, None)
    s = s.replace("–", "-")
    m = re.match(r"^\s*(\d{4})(?:-(\d{4}))?\s*$", s)
    if not m:
        return (None, None)
    y1 = int(m.group(1))
    y2 = int(m.group(2)) if m.group(2) else None
    return (y1, y2)


def overlaps_years(y1, y2, lo=2015, hi=2026) -> bool:
    if y1 is None:
        return True
    if y2 is None:
        y2 = hi
    return not (y2 < lo or y1 > hi)


def main():
    html = fetch_html(URL)
    tables = pd.read_html(html)
    df = find_managers_table(tables).copy()

    # normalizace sloupců
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["Name", "Club", "From", "Until"]:
        if col not in df.columns:
            alt = [c for c in df.columns if c.strip().lower() == col.lower()]
            if alt:
                df.rename(columns={alt[0]: col}, inplace=True)

    # čistění textu
    df["Name_raw"] = df["Name"].map(clean_text)
    df["Club"] = df["Club"].map(clean_text)
    df["From_dt"] = df["From"].map(parse_date)
    df["Until_dt"] = df["Until"].map(parse_date)

    # 1) jen hlavní trenéři: vyhodíme caretaker/interim označené "‡"
    df = df[~df["Name_raw"].str.contains("‡", regex=False)].copy()

    # odstraň † (u aktuálních) a podobné značky
    df["coach_name"] = (
        df["Name_raw"]
        .str.replace("†", "", regex=False)
        .str.replace("‡", "", regex=False)
        .str.strip()
    )

    # 2) filtr období (sezóny 2015/16–2025/26)
    # pokud je "Years in League" dostupné, použijeme ho jako extra pojistku
    if "Years in League" in df.columns:
        y = df["Years in League"].map(clean_text)
        y1y2 = y.map(parse_years_in_league)
        df["y1"] = [a for a, b in y1y2]
        df["y2"] = [b for a, b in y1y2]
        df = df[df.apply(lambda r: overlaps_years(r["y1"], r["y2"]), axis=1)].copy()

    # 3) oříznutí přesně na náš datumový rozsah
    # start = max(From, RANGE_START)
    df["start_date"] = df["From_dt"].fillna(RANGE_START)
    df.loc[df["start_date"] < RANGE_START, "start_date"] = RANGE_START

    # end = min(Until, RANGE_END); když Until = NaT (Present), necháme prázdné, ale jen pokud je v rozsahu
    df["end_date"] = df["Until_dt"]
    df.loc[df["end_date"] > RANGE_END, "end_date"] = RANGE_END

    # vyhodíme řádky, které jsou celé mimo rozsah
    df = df[~((df["Until_dt"].notna()) & (df["Until_dt"] < RANGE_START))].copy()
    df = df[~((df["From_dt"].notna()) & (df["From_dt"] > RANGE_END))].copy()

    # finální sloupce
    out = df[["coach_name", "Club", "start_date", "end_date"]].copy()
    out.rename(columns={"Club": "team"}, inplace=True)

    # formát dat
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.date.astype(str)
    out["end_date"] = pd.to_datetime(out["end_date"]).dt.date.astype(str)

    # prázdné end_date pro aktuální trenéry (Present)
    out.loc[df["Until_dt"].isna(), "end_date"] = ""

    # odstranění duplicit (někdy se objeví stejné období 2×)
    out = out.drop_duplicates().sort_values(["team", "start_date", "coach_name"]).reset_index(drop=True)

    out.to_csv("epl_main_managers_2015_2026.csv", index=False, encoding="utf-8")
    print("OK -> epl_main_managers_2015_2026.csv | rows:", len(out))
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
