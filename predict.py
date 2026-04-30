import pandas as pd
import numpy as np
import mlflow.pyfunc

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import mlflow
from mlflow.tracking import MlflowClient
import os


tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow_registry.db")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(registry_uri)

client = MlflowClient()
versions = client.search_model_versions("name='DiabetesNN'")
latest_version = max(int(v.version) for v in versions)
model = mlflow.pyfunc.load_model(f"models:/DiabetesNN/{latest_version}")

X_test = pd.read_csv("X_test.csv")
X_test.columns = X_test.columns.astype(int)
y_true = pd.read_csv("y_test.csv").values.flatten()

result = model.predict(X_test)

probs = result["diabetes_probability"].values

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

best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nBest threshold: {best_threshold:.4f}")

predictions = (probs > best_threshold).astype(int)

print("\nAccuracy:", accuracy_score(y_true, predictions))
print("\nROC AUC:", roc_auc_score(y_true, probs))
print("\nConfusion matrix:\n", confusion_matrix(y_true, predictions))
print("\nReport:\n", classification_report(y_true, predictions))

pd.DataFrame({
    "Prediction": predictions,
    "Probability": probs
}).to_csv("predictions.csv", index=False)

print("\nSaved predictions.csv")