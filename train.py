"""
Churn Prediction - Training Pipeline
Trains, validates, and saves the ChurnNet model with MLflow tracking
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import mlflow
import mlflow.pytorch

from model import ChurnNet, ChurnDataset
from preprocessing import ChurnPreprocessor, make_train_val_test_split

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    "data_path": "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "model_save_dir": "artifacts/model",
    "preprocessor_save_dir": "artifacts/preprocessor",
    "hidden_dims": [256, 128, 64],
    "dropout": 0.3,
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 8,
    "mlflow_experiment": "churn_prediction",
}


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs.flatten())
        all_labels.extend(y_batch.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    preds_bin = (np.array(all_preds) > 0.5).astype(int)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, preds_bin),
        "roc_auc": roc_auc_score(all_labels, all_preds),
        "f1": f1_score(all_labels, preds_bin),
    }
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────
def train():
    # Load & preprocess
    df = pd.read_csv(CONFIG["data_path"])
    print(f"Loaded {len(df)} rows")

    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit_transform(df)
    preprocessor.save(CONFIG["preprocessor_save_dir"])

    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(X, y)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_ds = ChurnDataset(X_train, y_train)
    val_ds = ChurnDataset(X_val, y_val)
    test_ds = ChurnDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

    # Class imbalance weight
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    input_dim = X_train.shape[1]
    model = ChurnNet(input_dim, CONFIG["hidden_dims"], CONFIG["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    mlflow.set_experiment(CONFIG["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in CONFIG.items() if k not in ["data_path", "model_save_dir", "preprocessor_save_dir", "mlflow_experiment"]})

        best_val_auc, patience_counter = 0, 0
        best_state = None

        for epoch in range(1, CONFIG["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = evaluate(model, val_loader, criterion)
            scheduler.step(val_metrics["loss"])

            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["roc_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }, step=epoch)

            # Early stopping
            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_loader, criterion)
        print("\n=== Test Results ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
            mlflow.log_metric(f"test_{k}", v)

        # Save model
        os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": CONFIG["hidden_dims"],
            "dropout": CONFIG["dropout"],
        }, os.path.join(CONFIG["model_save_dir"], "churn_model.pt"))
        mlflow.pytorch.log_model(model, "churn_model")
        print(f"\nModel saved. Best Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    train()
