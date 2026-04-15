import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

X_test = pd.read_csv("X_test.csv").values
y_true = pd.read_csv("y_test.csv").values.flatten()

scaler = joblib.load("scaler.pkl")
X_test = scaler.transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32)

model = SimpleNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.argmax(outputs, dim=1)

print("Prediction distribution:")
print(torch.bincount(predictions))

print("Accuracy:")
print(accuracy_score(y_true, predictions.numpy()))

print("Classification report:")
print(classification_report(y_true, predictions.numpy()))

print("Confusion matrix:")
print(confusion_matrix(y_true, predictions.numpy()))

pd.DataFrame(predictions.numpy(), columns=["Prediction"]).to_csv(
    "predictions.csv",
    index=False
)

print("\nSaved predictions.csv")