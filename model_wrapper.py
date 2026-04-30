import mlflow.pyfunc
import torch
import pandas as pd
import joblib
from model_def import SimpleNN
class DiabetesPyFuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        self.scaler = joblib.load(context.artifacts["scaler"])

        self.model = SimpleNN()
        self.model.load_state_dict(torch.load(context.artifacts["model"]))
        self.model.eval()

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        X = self.scaler.transform(model_input.values)
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).numpy().flatten()

        return pd.DataFrame({
            "diabetes_probability": probs,
            "prediction": (probs > 0.5).astype(int)
        })