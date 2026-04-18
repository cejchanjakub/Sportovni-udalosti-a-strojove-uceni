# src/api_main.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Multi-market orchestrace ---
from src.inference.registry import MarketRegistry

# --- Services ---
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
from src.inference.services.shots_total_service import ShotsTotalService
from src.inference.services.shots_home_service import ShotsHomeService
from src.inference.services.shots_away_service import ShotsAwayService
from src.inference.services.fouls_total_service import FoulsTotalService
from src.inference.services.fouls_home_service import FoulsHomeService
from src.inference.services.fouls_away_service import FoulsAwayService

# --- Live fixtures provider ---
from src.inference.providers.football_data_provider import FootballDataProvider


app = FastAPI(title="dp_sazeni_ml inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_META_PATH = PROJECT_ROOT / "data" / "features" / "live_features_meta.json"
LIVE_FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "live_fixtures.csv"


def _read_live_meta() -> Optional[Dict[str, Any]]:
    try:
        if not LIVE_META_PATH.exists():
            return None
        return json.loads(LIVE_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _lookup_error_detail(err: Exception) -> Dict[str, Any]:
    meta = _read_live_meta()
    hint_days = meta.get("days_ahead") if isinstance(meta, dict) else 30
    if not isinstance(hint_days, int):
        hint_days = 30
    return {
        "error": str(err),
        "reason": "fixture_not_found_in_live_features",
        "live_features_meta": meta,
        "hint": {
            "what_to_do": "Spusť refresh live features, aby se stáhly budoucí fixtures a přepočítaly featury.",
            "command": f"python -m src.inference.refresh_live_features_from_api --days-ahead {hint_days}",
        },
    }


# ==========================================================
# Registry
# ==========================================================


def _serialize_1x2(result) -> dict:
    """Serializuje výsledek OneXTwoService – pravděpodobnosti jako 0-100."""
    d = result.__dict__ if hasattr(result, "__dict__") else dict(result)
    return {
        "mapped":      d.get("mapped", {}),
        "swapped":     d.get("swapped", False),
        "p_home":      round(float(d.get("p_home", 0)) * 100, 2),
        "p_draw":      round(float(d.get("p_draw", 0)) * 100, 2),
        "p_away":      round(float(d.get("p_away", 0)) * 100, 2),
        "odds_home":   round(float(d.get("odds_home", 0)), 2),
        "odds_draw":   round(float(d.get("odds_draw", 0)), 2),
        "odds_away":   round(float(d.get("odds_away", 0)), 2),
        "matched_row": d.get("matched_row", {}),
    }


def _serialize_ou(result) -> dict:
    """Serializuje výsledek O/U service – pravděpodobnosti jako 0-100, kurzy zaokrouhleny."""
    d = result.__dict__ if hasattr(result, "__dict__") else dict(result)
    lines_raw = d.get("lines", {})
    ou_rows = lines_raw.get("ou", []) if isinstance(lines_raw, dict) else []

    ou_formatted = []
    for row in ou_rows:
        ou_formatted.append({
            "line":            row.get("line"),
            "p_over":          round(float(row.get("p_over", 0)) * 100, 2),
            "p_under":         round(float(row.get("p_under", 0)) * 100, 2),
            "odds_over_fair":  round(float(row.get("odds_over_fair", 0)), 2),
            "odds_under_fair": round(float(row.get("odds_under_fair", 0)), 2),
            "odds_over":       round(float(row.get("odds_over", 0)), 2),
            "odds_under":      round(float(row.get("odds_under", 0)), 2),
        })

    return {
        "mapped":      d.get("mapped", {}),
        "swapped":     d.get("swapped", False),
        "mean":        round(float(d.get("mean", 0)), 4),
        "lines": {
            "main_line": lines_raw.get("main_line") if isinstance(lines_raw, dict) else None,
            "ou":        ou_formatted,
        },
        "matched_row": d.get("matched_row", {}),
        "referee":     d.get("referee"),
    }


def _run_predict(registry: MarketRegistry, match: dict, markets: list) -> dict:
    """Spustí predikci pro každý trh a vrátí výsledky."""
    results = {}
    utc_date  = match.get("utc_date") or match.get("kickoff", "")
    home_team = match.get("home_team") or match.get("HomeTeam", "")
    away_team = match.get("away_team") or match.get("AwayTeam", "")
    referee   = match.get("referee")
    margin    = float(match.get("margin", 0.0))

    # Pouze tyto markets přijímají referee parametr
    REFEREE_MARKETS = {"fouls_total", "fouls_home", "fouls_away",
                       "cards_total", "cards_home", "cards_away"}

    for market in markets:
        try:
            service = registry.get(market)
            if market == "1x2":
                result = service.predict_from_match(utc_date, home_team, away_team)
                results[market] = _serialize_1x2(result)
            else:
                kwargs = dict(margin=margin)
                if referee is not None and market in REFEREE_MARKETS:
                    kwargs["referee"] = referee
                result = service.predict_from_match(utc_date, home_team, away_team, **kwargs)
                results[market] = _serialize_ou(result)
        except Exception as e:
            results[market] = {"error": str(e)}
    return results


registry = MarketRegistry()
registry.register("1x2",           OneXTwoService())
registry.register("goals_total",   GoalsTotalService())
registry.register("goals_home",    GoalsHomeService())
registry.register("goals_away",    GoalsAwayService())
registry.register("cards_total",   CardsTotalService())
registry.register("cards_home",    CardsHomeService())
registry.register("cards_away",    CardsAwayService())
registry.register("corners_total", CornersTotalService())
registry.register("corners_home",  CornersHomeService())
registry.register("corners_away",  CornersAwayService())
registry.register("shots_total",   ShotsTotalService())
registry.register("shots_home",    ShotsHomeService())
registry.register("shots_away",    ShotsAwayService())
registry.register("fouls_total",   FoulsTotalService())
registry.register("fouls_home",    FoulsHomeService())
registry.register("fouls_away",    FoulsAwayService())


ALL_MARKETS = registry.list_markets()


# ==========================================================
# Request modely
# ==========================================================

class PredictRequest(BaseModel):
    match: Dict[str, Any]
    markets: List[str] = ALL_MARKETS  # default = všechny trhy


# ==========================================================
# Endpointy
# ==========================================================

@app.get("/")
def root():
    return {"message": "dp_sazeni_ml inference API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get(
    "/fixtures",
    summary="Seznam nadcházejících zápasů",
    description=(
        "Vrátí seznam nadcházejících EPL zápasů načtených z live_fixtures.csv. "
        "Hodnoty `home_team`, `away_team` a `utc_date` použij jako vstup do POST /predict."
    ),
)
def get_fixtures(days_ahead: int = 14):
    """
    Vrátí seznam nadcházejících zápasů.
    Parametr `days_ahead` určuje kolik dní dopředu se načtou zápasy (výchozí 14).
    """
    try:
        import pandas as pd
        if not LIVE_FIXTURES_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "live_fixtures.csv nenalezen",
                    "hint": "Spusť: python -m src.inference.refresh_live_features_from_api --days-ahead 14",
                },
            )
        df = pd.read_csv(LIVE_FIXTURES_PATH)

        # Filtruj podle days_ahead pokud existuje sloupec s datem
        # Zjisti název sloupce s datem (kickoff nebo utc_date)
        date_col = "kickoff" if "kickoff" in df.columns else "utc_date"

        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        df["_dt"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df[df["_dt"] >= now]
        df = df[df["_dt"] <= cutoff]
        df = df.sort_values("_dt")
        df = df.drop(columns=["_dt"])

        fixtures = []
        for _, row in df.iterrows():
            kickoff = str(row.get("kickoff") or row.get("utc_date", ""))
            fixtures.append({
                "home_team": str(row.get("HomeTeam") or row.get("home_team", "")),
                "away_team": str(row.get("AwayTeam") or row.get("away_team", "")),
                "utc_date":  kickoff,
                "competition": str(row.get("competition", "Premier League")),
            })

        return {
            "count": len(fixtures),
            "fixtures": fixtures,
            "tip": (
                "Zkopíruj home_team, away_team a utc_date do POST /predict. "
                "Pole 'markets' je nepovinné – bez něj se vrátí všechny trhy."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict",
    summary="Predikce kurzů pro zápas",
    description=(
        "Vrátí fair kurzy a pravděpodobnosti pro vybraný zápas. "
        "Pole `markets` je nepovinné – bez něj se vrátí všechny dostupné trhy. "
        "Hodnoty `home_team`, `away_team` a `utc_date` získáš z GET /fixtures."
    ),
)
def predict(req: PredictRequest):
    try:
        return {
            "markets": _run_predict(registry, req.match, req.markets),
            "available_markets": ALL_MARKETS,
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=_lookup_error_detail(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/referees",
    summary="Seznam rozhodčích s jejich statistikami",
    description="Vrátí seznam rozhodčích EPL s průměrnými statistikami za posledních 20 zápasů.",
)
def get_referees():
    """
    Vrátí seznam rozhodčích s jejich referee featury.
    Použij jméno rozhodčího jako parametr 'referee' v POST /predict.
    """
    try:
        import pandas as pd
        from pathlib import Path

        # Načti referee lookup ze souborů features
        features_path = PROJECT_ROOT / "data" / "features" / "train_features.csv"
        if not features_path.exists():
            raise HTTPException(status_code=404, detail="train_features.csv nenalezen")

        # Načti referee sloupce + Referee sloupec
        df = pd.read_csv(features_path)
        ref_cols = [c for c in df.columns if c.startswith("ref_")]

        if "Referee" not in df.columns or not ref_cols:
            raise HTTPException(status_code=404, detail="Referee sloupce nenalezeny")

        # Agreguj průměrné statistiky per rozhodčí
        referee_stats = (
            df[df["Referee"].notna()]
            .groupby("Referee")[ref_cols]
            .mean()
            .round(3)
            .reset_index()
        )

        # Počet zápasů per rozhodčí
        counts = df["Referee"].value_counts().reset_index()
        counts.columns = ["Referee", "total_matches"]
        referee_stats = referee_stats.merge(counts, on="Referee")

        # Seřaď podle počtu zápasů
        referee_stats = referee_stats.sort_values("total_matches", ascending=False)

        referees = []
        for _, row in referee_stats.iterrows():
            entry = {"name": row["Referee"], "total_matches": int(row["total_matches"])}
            for col in ref_cols:
                if col in row:
                    entry[col] = round(float(row[col]), 3)
            referees.append(entry)

        return {
            "count": len(referees),
            "referees": referees,
            "tip": "Předej jméno rozhodčího jako 'referee' v match objektu POST /predict",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """Slouží frontend HTML aplikaci."""
    ui_path = PROJECT_ROOT / "src" / "ui" / "index.html"
    if not ui_path.exists():
        return HTMLResponse("<h1>UI nenalezeno</h1><p>Zkopíruj index.html do src/ui/</p>", status_code=404)
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))