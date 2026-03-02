# src/inference/model_loader.py
from pathlib import Path
import joblib
import json
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "v1_model_freeze"


# ==========================================================
# 1X2 LOADER
# ==========================================================

def load_1x2_model():
    artifact_dir = ARTIFACT_ROOT / "1x2"

    model = joblib.load(artifact_dir / "model.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


# ==========================================================
# GOALS TOTAL LOADER
# ==========================================================

def load_goals_total_model():
    artifact_dir = ARTIFACT_ROOT / "goals__total_goals"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


# ==========================================================
# CARDS TOTAL (YELLOW) LOADER
# ==========================================================

def load_cards_total_model():
    artifact_dir = ARTIFACT_ROOT / "yellow__total_cards"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_cards_home_model():
    artifact_dir = ARTIFACT_ROOT / "yellow__HY"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_cards_away_model():
    artifact_dir = ARTIFACT_ROOT / "yellow__AY"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


# ==========================================================
# GOALS HOME/AWAY LOADERS
# ==========================================================

def load_goals_home_model():
    artifact_dir = ARTIFACT_ROOT / "goals__FTHG"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_goals_away_model():
    artifact_dir = ARTIFACT_ROOT / "goals__FTAG"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


# ==========================================================
# CORNERS TOTAL LOADER
# ==========================================================

def load_corners_total_model():
    artifact_dir = ARTIFACT_ROOT / "corners__total_corners"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_corners_home_model():
    artifact_dir = ARTIFACT_ROOT / "corners__HC"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_corners_away_model():
    artifact_dir = ARTIFACT_ROOT / "corners__AC"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


# ==========================================================
# SHOTS ON TARGET (SOT) LOADERS
# ==========================================================

def load_sot_total_model():
    artifact_dir = ARTIFACT_ROOT / "sot__total_shots_on_target"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_sot_home_model():
    artifact_dir = ARTIFACT_ROOT / "sot__HST"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_sot_away_model():
    artifact_dir = ARTIFACT_ROOT / "sot__AST"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


# ==========================================================
# FOULS LOADERS (NEW)
# ==========================================================

def load_fouls_total_model():
    """
    Loader pro total fauly.

    Očekávaná složka:
      artifacts/v1_model_freeze/fouls__total_fouls/
        - model.sm
        - scaler.joblib
        - features.json
    """
    artifact_dir = ARTIFACT_ROOT / "fouls__total_fouls"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_fouls_home_model():
    """
    Loader pro fauly domácích (HF).

    Očekávaná složka:
      artifacts/v1_model_freeze/fouls__HF/
        - model.sm
        - scaler.joblib
        - features.json
    """
    artifact_dir = ARTIFACT_ROOT / "fouls__HF"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features


def load_fouls_away_model():
    """
    Loader pro fauly hostů (AF).

    Očekávaná složka:
      artifacts/v1_model_freeze/fouls__AF/
        - model.sm
        - scaler.joblib
        - features.json
    """
    artifact_dir = ARTIFACT_ROOT / "fouls__AF"

    model = sm.load(artifact_dir / "model.sm")
    scaler = joblib.load(artifact_dir / "scaler.joblib")

    with open(artifact_dir / "features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features