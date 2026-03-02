from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

UNKNOWN_LABEL = "Unknown"


def load_referee_list(train_path: Path | str = DEFAULT_TRAIN_PATH) -> List[str]:
    """
    Vrátí seznam rozhodčích z historických dat (Referee sloupec),
    vhodný pro Streamlit selectbox.
    """
    df = pd.read_csv(train_path)

    if "Referee" not in df.columns:
        raise ValueError("Ve vstupním souboru chybí sloupec 'Referee'.")

    refs = (
        df["Referee"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    refs = refs[refs != ""].unique().tolist()
    refs = sorted(refs)

    # Unknown na začátek
    return [UNKNOWN_LABEL] + refs
