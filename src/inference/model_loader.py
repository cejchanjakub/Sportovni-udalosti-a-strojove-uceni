# src/inference/model_loader.py
from pathlib import Path
import joblib
import json
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "v1_model_freeze"


def _load_glm(subdir: str):
    """
    Načte statsmodels GLM model + features.json + alpha z meta.json.
    Vrací (model, features, alpha) kde alpha=None pro Poisson modely.
    """
    artifact_dir = ARTIFACT_ROOT / subdir
    model = sm.load(str(artifact_dir / "model.sm"))
    with open(artifact_dir / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)
    alpha = None
    meta_path = artifact_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        alpha = meta.get("alpha", None)
    return model, features, alpha


def load_1x2_model():
    artifact_dir = ARTIFACT_ROOT / "1x2"
    model = joblib.load(artifact_dir / "model.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    with open(artifact_dir / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)
    return model, scaler, features


def load_goals_total_model():  return _load_glm("goals__total_goals")
def load_goals_home_model():   return _load_glm("goals__FTHG")
def load_goals_away_model():   return _load_glm("goals__FTAG")

def load_corners_total_model(): return _load_glm("corners__total_corners")
def load_corners_home_model():  return _load_glm("corners__HC")
def load_corners_away_model():  return _load_glm("corners__AC")

def load_fouls_total_model():  return _load_glm("fouls__total_fouls")
def load_fouls_home_model():   return _load_glm("fouls__HF")
def load_fouls_away_model():   return _load_glm("fouls__AF")

def load_cards_total_model():  return _load_glm("yellow__total_cards")
def load_cards_home_model():   return _load_glm("yellow__HY")
def load_cards_away_model():   return _load_glm("yellow__AY")

def load_sot_total_model():    return _load_glm("shotsot__total_shots_on_target")
def load_sot_home_model():     return _load_glm("shotsot__HST")
def load_sot_away_model():     return _load_glm("shotsot__AST")