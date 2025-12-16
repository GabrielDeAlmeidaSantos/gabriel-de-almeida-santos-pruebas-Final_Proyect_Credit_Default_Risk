from src.features.load_data import load_data
from src.features.build_features import build_features
import pandas as pd

# Cargar datos
data = load_data()
app = data["app"]

X = []
y = []

# 🔴 IMPORTANTE: sample pequeño para empezar
sample = app.sample(n=1000, random_state=42)

for i, (_, row) in enumerate(sample.iterrows()):
    sk_id = row["SK_ID_CURR"]

    feats = build_features(sk_id, data)
    X.append(feats)
    y.append(row["TARGET"])

    if i % 100 == 0:
        print(f"Procesados {i} clientes")

# Construir dataset final
X = pd.concat(X, ignore_index=True)
X["TARGET"] = y

# Guardar
X.to_parquet("data/processed/features.parquet")

print("Dataset final:", X.shape)
