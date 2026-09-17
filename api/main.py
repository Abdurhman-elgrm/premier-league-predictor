"""
FastAPI Serving Endpoint for Premier League Match & Score Predictor
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to system path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from predict import predict_match

app = FastAPI(
    title="Premier League Match & Score Predictor API",
    description="Machine Learning API predicting Premier League exact scores, expected goals (xG), and 1X2 match probabilities.",
    version="2.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    home_team: str
    away_team: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Premier League Match & Score Predictor API is live!",
        "docs_url": "/docs"
    }


@app.post("/predict")
def predict_fixture(payload: MatchRequest):
    if payload.home_team.strip() == payload.away_team.strip():
        raise HTTPException(status_code=400, detail="Home team and Away team must be different.")
    
    try:
        results = predict_match(payload.home_team.strip(), payload.away_team.strip())
        return {
            "status": "success",
            "fixture": f"{payload.home_team} vs {payload.away_team}",
            "predicted_score": results["predicted_score"],
            "expected_goals": {
                payload.home_team: results["exp_home_goals"],
                payload.away_team: results["exp_away_goals"]
            },
            "outcome_probabilities": results["probabilities"],
            "top_likely_scores": [{"score": s, "probability": round(p, 4)} for s, p in results["top_scores"]]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
