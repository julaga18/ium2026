import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
import mlflow.pytorch

model = mlflow.pytorch.load_model("models:/DiabetesNN/1")

X_test = pd.read_csv("X_test.csv").values
y_true = pd.read_csv("y_test.csv").values.flatten()

scaler = joblib.load("scaler.pkl")
X_test = scaler.transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32)

model.eval()

with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits).numpy().flatten()

thresholds = np.linspace(0.01, 0.99, 200)

f1_scores = []

for t in thresholds:
    preds = (probs > t).astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    f1_scores.append(f1)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nBest threshold (F1 optimized): {best_threshold:.4f}")

predictions = (probs > best_threshold).astype(int)

print("\nPrediction distribution:")
print(np.bincount(predictions))

print("\nAccuracy:")
print(accuracy_score(y_true, predictions))

print("\nROC AUC:")
print(roc_auc_score(y_true, probs))

print("\nClassification report:")
print(classification_report(y_true, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(y_true, predictions))

pd.DataFrame({
    "Prediction": predictions,
    "Probability": probs
}).to_csv("predictions.csv", index=False)

print("\nSaved predictions.csv")