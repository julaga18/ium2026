import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from torch.utils.data import DataLoader, TensorDataset

import joblib
import argparse

import mlflow
import mlflow.pyfunc

from mlflow.models import infer_signature
from model_wrapper import DiabetesPyFuncModel
from mlflow.tracking import MlflowClient
import os
from model_def import SimpleNN

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///mlflow_registry.db")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(registry_uri)
mlflow.set_experiment("Diabetes NN")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
args = parser.parse_args()

epochs = args.epochs

data = pd.read_csv("data/diabetes.csv")

X = data.drop("Diabetes_binary", axis=1).values
y = data["Diabetes_binary"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_raw = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test_raw, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True
)

model = SimpleNN()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015)

with mlflow.start_run():

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("lr", 0.0015)
    mlflow.log_param("batch_size", 128)

    best_auc = 0
    patience = 7
    counter = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:

            y_batch = y_batch * 0.9 + 0.05

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()

            acc = accuracy_score(y_test.numpy(), preds.numpy())
            auc = roc_auc_score(y_test.numpy(), probs.numpy())

        if auc > best_auc:
            best_auc = auc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

        mlflow.log_metric("loss", total_loss / len(train_loader), step=epoch)
        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("roc_auc", auc, step=epoch)

    artifact_dir = "artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    pd.DataFrame([{
        "accuracy": acc,
        "roc_auc": auc
    }]).to_csv(f"{artifact_dir}/metrics.csv", index=False)

    pd.DataFrame(X_test.numpy()).to_csv(f"{artifact_dir}/X_test.csv", index=False)
    pd.DataFrame(y_test.numpy()).to_csv(f"{artifact_dir}/y_test.csv", index=False)

    torch.save(model.state_dict(), f"{artifact_dir}/model.pth")
    joblib.dump(scaler, f"{artifact_dir}/scaler.pkl")

    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).numpy().flatten()

    input_example = pd.DataFrame(X_test.numpy()[:3].astype("float64"))

    example_output = pd.DataFrame({
        "diabetes_probability": probs[:3],
        "prediction": (probs[:3] > 0.5).astype(int)
    })

    signature = infer_signature(input_example, example_output)

    client = MlflowClient()

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=DiabetesPyFuncModel(),
        artifacts={
            "model": f"{artifact_dir}/model.pth",
            "scaler": f"{artifact_dir}/scaler.pkl"
        },
        signature=signature,
        input_example=input_example,
        registered_model_name="DiabetesNN"
    )

    model_card = f"""
    # DiabetesNN Model Card

    ## Cel
    Binary classification (diabetes prediction)

    ## Dane
    - Features: 21
    - Train/Test: 80/20

    ## Model
    - PyTorch MLP
    - BCEWithLogitsLoss

    ## Wyniki
    - Accuracy: {acc:.4f}
    - ROC AUC: {auc:.4f}
    """

    model_card_path = os.path.join(artifact_dir, "model_card.md")

    with open(model_card_path, "w") as f:
        f.write(model_card)

    mlflow.log_artifact(model_card_path)