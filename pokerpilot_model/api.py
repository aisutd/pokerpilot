"""FastAPI endpoint for serving the trained PokerPilot behavior model with Uvicorn."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="PokerPilot XGBoost API", version="0.1.0")
MODEL_PATH = Path(os.environ.get("POKERPILOT_MODEL_PATH", Path(__file__).parent / "artifacts" / "poker_action_model.joblib"))
_bundle: dict | None = None


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(description="Decision state with feature names from poker_feature_dictionary.csv")


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail="Model not trained. Run train_model.py first.")
        _bundle = joblib.load(MODEL_PATH)  # Only load model files you trust.
    return _bundle


@app.get("/health")
def health():
    return {"status": "ready" if MODEL_PATH.is_file() else "model_missing", "model_path": str(MODEL_PATH)}


@app.get("/features")
def features():
    b = get_bundle()
    return {"features": b["features"], "classes": b["label_encoder"].classes_.tolist()}


@app.post("/predict")
def predict(request: PredictionRequest):
    b = get_bundle()
    missing = [feature for feature in b["features"] if feature not in request.features]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    x = pd.DataFrame([{k: request.features[k] for k in b["features"]}])
    for k in b["numeric"]:
        x[k] = pd.to_numeric(x[k], errors="coerce")
    for k in b["categorical"]:
        x[k] = x[k].astype("string").fillna("missing").astype(str)
    probs = b["pipeline"].predict_proba(x)[0]
    labels = b["label_encoder"].classes_
    idx = int(probs.argmax())
    return {
        "predicted_action": str(labels[idx]),
        "probabilities": {str(label): round(float(p), 6) for label, p in zip(labels, probs)},
        "interpretation": "Prediction of historical behavior, not optimal strategy",
    }
