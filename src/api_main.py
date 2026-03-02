# src/api_main.py
from __future__ import annotations

from typing import Any, Dict, List

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
        raise HTTPException(status_code=404, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))