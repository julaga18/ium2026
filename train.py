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
import mlflow.pytorch


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
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(21, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
   
model = SimpleNN()

criterion = nn.BCEWithLogitsLoss(reduction="mean")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0015,
    weight_decay=1e-4
)

with mlflow.start_run():

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("lr", 0.0005)
    mlflow.log_param("batch_size", 256)
    mlflow.log_param("task", "binary_classification")

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

        avg_loss = total_loss / len(train_loader)

        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("roc_auc", auc, step=epoch)

    result = mlflow.pytorch.log_model(model, "model")

    mlflow.register_model(
        result.model_uri,
        "DiabetesNN"
    )

    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")

torch.save(model.state_dict(), "model.pth")

pd.DataFrame(X_test.numpy()).to_csv("X_test.csv", index=False)
pd.DataFrame(y_test.numpy()).to_csv("y_test.csv", index=False)
