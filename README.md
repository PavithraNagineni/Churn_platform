# Churn Prediction Platform

Deep learning churn prediction system with a FastAPI microservice for real-time inference.

## Architecture
```
data/ → preprocessing.py → model.py → train.py → artifacts/
                                                       ↓
                                               main.py (FastAPI)
                                                       ↓
                                             POST /predict (JSON)
```

## Project Structure
```
churn_prediction/
├── model.py          # ChurnNet (PyTorch MLP)
├── preprocessing.py  # Feature engineering & encoding pipeline
├── train.py          # Training loop with MLflow tracking + early stopping
├── main.py           # FastAPI microservice (single + batch predict)
├── Dockerfile        # Container definition
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Download Telco Customer Churn dataset from Kaggle and place at:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Train model
```bash
python train.py
```
Artifacts saved to `artifacts/model/` and `artifacts/preprocessor/`
MLflow UI: `mlflow ui` → http://localhost:5000

### 4. Run API server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Run with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /predict | Single customer prediction |
| POST | /predict/batch | Batch predictions |
| GET | /model/info | Model metadata |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
  }'
```

### Example Response
```json
{
  "churn_probability": 0.7821,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "confidence": 0.7821
}
```

## Model Details
- **Architecture**: 3-layer MLP (256 → 128 → 64) with BatchNorm + Dropout
- **Loss**: BCEWithLogitsLoss with class imbalance weighting
- **Optimizer**: AdamW with ReduceLROnPlateau scheduler
- **Early stopping**: Patience=8 on validation AUC
- **Tracking**: MLflow for experiment logging
