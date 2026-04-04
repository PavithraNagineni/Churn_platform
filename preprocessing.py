"""
Churn Prediction - Preprocessing Pipeline
Handles feature engineering, encoding, scaling for churn data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


CATEGORICAL_COLS = ["gender", "Partner", "Dependents", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod"]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"


class ChurnPreprocessor:
    """Full preprocessing pipeline: clean → encode → scale → split."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame):
        df = self._clean(df)
        df = self._encode(df, fit=True)
        X, y = self._split_xy(df)
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y.values

    def transform(self, df: pd.DataFrame):
        df = self._clean(df)
        df = self._encode(df, fit=False)
        X, _ = self._split_xy(df, has_target=False)
        # Ensure column order matches training
        X = X[self.feature_names]
        return self.scaler.transform(X)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Remove customerID if present
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)
        # TotalCharges may have spaces
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        return df

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).map(
                        lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                    )
        # Encode target
        if TARGET_COL in df.columns:
            df[TARGET_COL] = (df[TARGET_COL].str.strip().str.lower() == "yes").astype(int)
        return df

    def _split_xy(self, df, has_target=True):
        if has_target and TARGET_COL in df.columns:
            X = df.drop(columns=[TARGET_COL])
            y = df[TARGET_COL]
        else:
            X = df
            y = None
        return X, y

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(path, "label_encoders.pkl"))
        joblib.dump(self.feature_names, os.path.join(path, "feature_names.pkl"))

    def load(self, path: str):
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))
        self.label_encoders = joblib.load(os.path.join(path, "label_encoders.pkl"))
        self.feature_names = joblib.load(os.path.join(path, "feature_names.pkl"))
        return self


def make_train_val_test_split(X, y, val_size=0.1, test_size=0.1, random_state=42):
    """Stratified 80/10/10 split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (val_size + test_size),
        stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
