import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import MinMaxScaler
import os

input_path = "data/dataset.csv"

if not os.path.exists(input_path):
    print("BŁĄD: Brak pliku data/dataset.csv. Uruchom najpierw create_dataset.py")
    exit(1)

df = pd.read_csv(input_path)

print("\n--- STATYSTYKI OPISOWE (Diabetes Dataset) ---")
stats = df.describe().T[['mean', 'std', 'min', '50%', 'max']]
print(stats)

print("\n--- PRZYGOTOWANIE MODELU ---")
train_df, test_df = sklearn.model_selection.train_test_split(
    df, train_size=0.6, random_state=1
)
print(f"Wielkość zbioru treningowego: {len(train_df)}")

scaler = MinMaxScaler()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])

print("\n--- PIERWSZE 5 WIERSZY PO NORMALIZACJI ---")
print(train_df.head())