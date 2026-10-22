"""FastAPI inference service. Set POKERPILOT_MODEL_PATH to the trained artifact."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import load_model

app = FastAPI(title="PokerPilot decision model", version="1.0.0")


class DecisionFeatures(BaseModel):
    features: dict[str, Any]


@lru_cache(maxsize=1)
def artifact() -> dict:
    path = Path(os.getenv("POKERPILOT_MODEL_PATH", "pokerpilot_model/artifacts/model.joblib"))
    return load_model(path)


@app.get("/health")
def health() -> dict:
    path = Path(os.getenv("POKERPILOT_MODEL_PATH", "pokerpilot_model/artifacts/model.joblib"))
    return {"status": "ready" if path.is_file() else "model_missing"}


@app.post("/predict")
def predict(payload: DecisionFeatures) -> dict:
    try:
        loaded = artifact()
        row = pd.DataFrame([{column: payload.features.get(column) for column in loaded["columns"]}])
        probabilities = loaded["pipeline"].predict_proba(row)[0]
        labels = loaded["pipeline"].named_steps["classifier"].classes_
        scores = {str(label): float(score) for label, score in zip(labels, probabilities)}
        return {"action": max(scores, key=scores.get), "probabilities": scores}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not available") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
