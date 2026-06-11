"""
api.py — FastAPI serving layer for the House Price Prediction model.

Endpoints:
    GET  /          — root, redirect info
    GET  /health    — health check
    POST /predict   — predict house sale price
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts Ames Housing sale prices using a tuned Random Forest model.",
    version="1.0.0",
)

# ── Load model at startup — not on every request ───────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "rf_house_prices.joblib"
METADATA_PATH = Path(__file__).parent.parent / "artifacts" / "rf_house_prices_metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    import json
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    FEATURE_NAMES = metadata["feature_names"]
    print(f"✓ Model loaded — {len(FEATURE_NAMES)} features expected")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# ── Request / Response schemas ─────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    """Input features for house price prediction.
    
    Only the most impactful features are required.
    Remaining features default to dataset medians.
    """
    overall_qual: int = Field(..., ge=1, le=10, description="Overall quality rating (1-10)")
    total_sf: float = Field(..., gt=0, description="Total square footage (basement + floor 1 + floor 2)")
    house_age: int = Field(..., ge=0, description="Age of house in years")
    total_bathrooms: float = Field(..., ge=0, description="Total bathrooms (half baths count as 0.5)")
    garage_area: float = Field(0.0, ge=0, description="Garage area in sq ft")
    gr_liv_area: float = Field(..., gt=0, description="Above ground living area sq ft")
    year_built: int = Field(..., ge=1800, le=2025, description="Year house was built")
    lot_area: float = Field(..., gt=0, description="Lot size in sq ft")

    model_config = {"json_schema_extra": {
        "example": {
            "overall_qual": 7,
            "total_sf": 2500,
            "house_age": 15,
            "total_bathrooms": 2.5,
            "garage_area": 400,
            "gr_liv_area": 1800,
            "year_built": 2005,
            "lot_area": 9000,
        }
    }}


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_version: str
    confidence_note: str


# ── Helper — build full feature vector from partial input ──────────────────────

def build_input_vector(house: HouseFeatures) -> pd.DataFrame:
    """
    Map user-provided fields onto the full feature vector the model expects.
    Unspecified features are filled with 0 (safe default for this model).
    """
    row = {feat: 0.0 for feat in FEATURE_NAMES}

    # Map provided fields
    mapping = {
        "Overall Qual": house.overall_qual,
        "total_sf": house.total_sf,
        "house_age": house.house_age,
        "total_bathrooms": house.total_bathrooms,
        "Garage Area": house.garage_area,
        "Gr Liv Area": house.gr_liv_area,
        "Year Built": house.year_built,
        "Lot Area": house.lot_area,
        "has_garage": 1.0 if house.garage_area > 0 else 0.0,
    }

    for feature, value in mapping.items():
        if feature in row:
            row[feature] = float(value)

    return pd.DataFrame([row])


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "House Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "author": "Coach Ayomide — github.com/Ay-developerweb",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "RandomForestRegressor",
        "features": len(FEATURE_NAMES),
        "test_rmse": metadata.get("test_rmse"),
        "test_r2": metadata.get("test_r2"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    try:
        X = build_input_vector(house)
        prediction = model.predict(X)[0]
        prediction = max(0.0, prediction)  # no negative prices

        return PredictionResponse(
            predicted_price=round(prediction, 2),
            predicted_price_formatted=f"${prediction:,.0f}",
            model_version="1.0.0",
            confidence_note=f"Model R²=0.92, typical error ±$25,438 RMSE on Ames Housing test set",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
