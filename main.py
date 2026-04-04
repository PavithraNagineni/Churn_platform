"""
Churn Prediction - FastAPI Microservice
Real-time churn inference endpoint with health check and batch support
"""

import os
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging

from model import ChurnNet
from preprocessing import ChurnPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn prediction using deep learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Model State ───────────────────────────────────────────────────────
MODEL = None
PREPROCESSOR = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global MODEL, PREPROCESSOR
    model_path = os.getenv("MODEL_PATH", "artifacts/model/churn_model.pt")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    MODEL = ChurnNet(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()

    PREPROCESSOR = ChurnPreprocessor().load(preprocessor_path)
    logger.info("Model and preprocessor loaded successfully")


@app.on_event("startup")
async def startup_event():
    load_model()


# ─── Schemas ──────────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=844.2)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    high_risk_count: int


# ─── Helpers ─────────────────────────────────────────────────────────────────
def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def predict_single(features: dict) -> PredictionResponse:
    df = pd.DataFrame([features])
    X = PREPROCESSOR.transform(df)
    tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logit = MODEL(tensor)
        prob = torch.sigmoid(logit).item()

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prob > 0.5,
        risk_level=risk_level(prob),
        confidence=round(max(prob, 1 - prob), 4),
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predict_single(customer.dict())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        preds = [predict_single(c.dict()) for c in request.customers]
        high_risk = sum(1 for p in preds if p.risk_level == "HIGH")
        return BatchResponse(predictions=preds, total=len(preds), high_risk_count=high_risk)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    total_params = sum(p.numel() for p in MODEL.parameters())
    return {
        "architecture": "ChurnNet (MLP)",
        "total_parameters": total_params,
        "device": str(DEVICE),
        "features": PREPROCESSOR.feature_names if PREPROCESSOR else [],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
