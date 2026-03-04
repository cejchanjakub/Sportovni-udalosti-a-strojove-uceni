# src/api_main.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Multi-market orchestrace ---
from src.inference.registry import MarketRegistry
from src.inference.inference_service import InferenceService

# --- Services (match-mode / production) ---
from src.inference.services.one_x_two_service import OneXTwoService

from src.inference.services.goals_total_service import GoalsTotalService
from src.inference.services.goals_home_service import GoalsHomeService
from src.inference.services.goals_away_service import GoalsAwayService

from src.inference.services.cards_total_service import CardsTotalService
from src.inference.services.cards_home_service import CardsHomeService
from src.inference.services.cards_away_service import CardsAwayService

from src.inference.services.corners_total_service import CornersTotalService
from src.inference.services.corners_home_service import CornersHomeService
from src.inference.services.corners_away_service import CornersAwayService

from src.inference.services.shots_total_service import SOTTotalService
from src.inference.services.shots_home_service import SOTHomeService
from src.inference.services.shots_away_service import SOTAwayService

from src.inference.services.fouls_total_service import FoulsTotalService
from src.inference.services.fouls_home_service import FoulsHomeService
from src.inference.services.fouls_away_service import FoulsAwayService


app = FastAPI(title="dp_sazeni_ml inference")


# ==========================================================
# Cesty / meta
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_META_PATH = PROJECT_ROOT / "data" / "features" / "live_features_meta.json"


def _read_live_meta() -> Optional[Dict[str, Any]]:
    """
    Načte metadata posledního refresh live features, pokud existují.
    Vrací dict nebo None.
    """
    try:
        if not LIVE_META_PATH.exists():
            return None
        return json.loads(LIVE_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        # Nechceme, aby meta read shodilo API
        return None


def _lookup_error_detail(err: Exception) -> Dict[str, Any]:
    """
    Sestaví detail pro 404 (fixture nenalezen).
    """
    meta = _read_live_meta()

    hint_days = None
    if isinstance(meta, dict):
        hint_days = meta.get("days_ahead")

    # Pokud meta nemáme, dej konzervativní doporučení
    if not isinstance(hint_days, int):
        hint_days = 30

    return {
        "error": str(err),
        "reason": "fixture_not_found_in_live_features",
        "live_features_meta": meta,
        "hint": {
            "what_to_do": "Spusť refresh live features, aby se stáhly budoucí fixtures a přepočítaly featury.",
            "command": f"python -m src.inference.refresh_live_features_from_api --days-ahead {hint_days}",
            "meta_path": str(LIVE_META_PATH),
        },
    }


# ==========================================================
# Globální service instance (vytvoří se při startu aplikace)
# ==========================================================

registry = MarketRegistry()

# 1X2
registry.register("1x2", OneXTwoService())

# Goals
registry.register("goals_total", GoalsTotalService())
registry.register("goals_home", GoalsHomeService())
registry.register("goals_away", GoalsAwayService())

# Cards
registry.register("cards_total", CardsTotalService())
registry.register("cards_home", CardsHomeService())
registry.register("cards_away", CardsAwayService())

# Corners
registry.register("corners_total", CornersTotalService())
registry.register("corners_home", CornersHomeService())
registry.register("corners_away", CornersAwayService())

# Shots on target (SOT)
registry.register("sot_total", SOTTotalService())
registry.register("sot_home", SOTHomeService())
registry.register("sot_away", SOTAwayService())

# Fouls
registry.register("fouls_total", FoulsTotalService())
registry.register("fouls_home", FoulsHomeService())
registry.register("fouls_away", FoulsAwayService())

inference_service = InferenceService(registry)


# ==========================================================
# Request modely
# ==========================================================

class PredictRequest(BaseModel):
    match: Dict[str, Any]
    markets: List[str]


# ==========================================================
# Základní endpointy
# ==========================================================

@app.get("/")
def root():
    return {"message": "dp_sazeni_ml inference API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================================
# Multi-market endpoint
# ==========================================================

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return {
            "markets": inference_service.predict(match=req.match, markets=req.markets),
            "available_markets": registry.list_markets(),
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except LookupError as e:
        # NOVĚ: bohatší detail (včetně live_features_meta + hint na refresh)
        raise HTTPException(status_code=404, detail=_lookup_error_detail(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))