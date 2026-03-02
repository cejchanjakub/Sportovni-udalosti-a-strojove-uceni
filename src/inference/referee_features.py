from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

UNKNOWN_LABEL = "Unknown"


def _ensure_datetime(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Chybí sloupec '{date_col}' pro časový cutoff.")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def compute_referee_features(
    referee_name: str,
    cutoff_date,
    train_path: Path | str = DEFAULT_TRAIN_PATH,
    window: int = 20,
    *,
    yellow_cards_col: str = "total_cards",
    date_col: str = "Date",
) -> Dict[str, float]:
    """
    Vrátí featury:
      - ref_matches_count_last20
      - ref_yellow_avg_last20

    cutoff_date: datum zápasu (nebo "teď"); používáme data STRICTNĚ před cutoff (no leakage).
    yellow_cards_col: musí odpovídat sloupci, ze kterého chceš počítat průměr žlutých (nebo karet).
    """
    df = pd.read_csv(train_path)
    df = _ensure_datetime(df, date_col=date_col)

    if "Referee" not in df.columns:
        raise ValueError("Chybí sloupec 'Referee' v processed datech.")
    if yellow_cards_col not in df.columns:
        raise ValueError(
            f"Chybí sloupec '{yellow_cards_col}'. "
            f"Uprav yellow_cards_col podle toho, jak se u tebe jmenuje target v datech."
        )

    cutoff_ts = pd.to_datetime(cutoff_date)

    # jen historie před zápasem
    hist = df[df[date_col] < cutoff_ts].copy()

    # fallback hodnoty z celé historie (před cutoff)
    # median je robustní
    league_median_yellow = float(hist[yellow_cards_col].dropna().median()) if len(hist) else 0.0

    # pokud neznámý rozhodčí nebo nemáme historii
    if (referee_name is None) or (str(referee_name).strip() == "") or (referee_name == UNKNOWN_LABEL) or len(hist) == 0:
        return {
            "ref_matches_count_last20": float(min(window, len(hist))),  # klidně 0..20
            "ref_yellow_avg_last20": league_median_yellow,
            "ref_unknown": 1.0,
        }

    # historie konkrétního rozhodčího
    r = hist[hist["Referee"].astype(str).str.strip() == str(referee_name).strip()].copy()
    r = r.sort_values(date_col)

    last = r.tail(window)
    matches_count = float(len(last))
    yellow_avg = float(last[yellow_cards_col].dropna().mean()) if len(last) else league_median_yellow

    # pokud rozhodčí nemá dost historie, lehce shrin-kujem k ligovému medianu
    k = 10.0  # síla prioru
    w = matches_count / (matches_count + k) if matches_count > 0 else 0.0
    yellow_avg = w * yellow_avg + (1 - w) * league_median_yellow

    return {
        "ref_matches_count_last20": matches_count,
        "ref_yellow_avg_last20": yellow_avg,
        "ref_unknown": 0.0,
    }
