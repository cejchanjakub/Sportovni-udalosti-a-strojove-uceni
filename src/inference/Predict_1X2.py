import pandas as pd
import numpy as np
from src.inference.model_loader import load_1x2_model


def predict_1x2(feature_dict: dict):
    model, scaler, feature_order = load_1x2_model()

    # vytvoříme DF v přesném pořadí featur
    X = pd.DataFrame([feature_dict])
    X = X[feature_order]

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[0]

    p_home, p_draw, p_away = probs

    return {
        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
        "odds_home": float(1 / p_home) if p_home > 0 else None,
        "odds_draw": float(1 / p_draw) if p_draw > 0 else None,
        "odds_away": float(1 / p_away) if p_away > 0 else None,
    }
