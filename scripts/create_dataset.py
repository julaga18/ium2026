import pandas as pd
from pathlib import Path
import os

Path("data").mkdir(exist_ok=True)

input_file = "data/diabetes.csv"

if os.path.exists(input_file):
    df = pd.read_csv(input_file)
    
    initial_count = len(df)
    df = df.dropna().drop_duplicates()
    
    # Zapisujemy yczyszczony zbiór
    df.to_csv("data/dataset.csv", index=False)
    
    print(f"Wczytano {initial_count} rekordów.")
    print(f"Oczyszczony zbiór ({len(df)} wierszy) zapisano w: data/dataset.csv")
else:
    print(f"BŁĄD: Nie znaleziono pliku {input_file}")
    exit(1)