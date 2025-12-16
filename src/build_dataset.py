from src.features.load_data import load_data
from src.features.build_features import build_features
import pandas as pd

data = load_data()
app = data["app"]

X = []
y = []

for _, row in app.iterrows():
    sk_id = row["SK_ID_CURR"]
    feats = build_features(sk_id, data)
    X.append(feats)
    y.append(row["TARGET"])

X = pd.concat(X, ignore_index=True)
y = pd.Series(y, name="TARGET")

X["TARGET"] = y

X.to_parquet("data/processed/features.parquet")

print(X.shape)