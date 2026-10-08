"""Simple API for the saved credit scoring pipeline."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import joblib
import numpy as np
import pandas as pd

from src.api.models import CreditRequest, CreditResponse
from src.config import BEST_MODEL_PATH
from src.features.build_features import build_features


@lru_cache(maxsize=1)
def load_model():
    if not BEST_MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model is missing. Run make train.")
    return joblib.load(BEST_MODEL_PATH)


app = FastAPI(title="Credit Scoring API")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    load_model()
    return {"status": "ok"}


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    model = load_model()
    features = build_features(pd.DataFrame([request.model_dump()]))
    features = (
        features[list(model.feature_names_in_)]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    default_index = list(model.classes_).index(1)
    return CreditResponse(
        prediction=int(model.predict(features)[0]),
        default_probability=float(model.predict_proba(features)[0, default_index]),
    )
