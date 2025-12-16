print("EJECUTANDO ESTE ARCHIVO:", __file__)

import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# === 1. Cargar dataset FULL ===
df = pd.read_parquet("data\processed\features.parquet")

y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

# object -> category (CLAVE)
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# === 2. Split (igual que entrenamiento) ===
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# === 3. Entrenar modelo (idéntico) ===
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# === 4. SHAP ===
explainer = shap.TreeExplainer(model)

# Muestra razonable (rápido y estable)
X_shap = X_val.sample(n=5000, random_state=42)

shap_values = explainer.shap_values(X_shap)

# === 5. SHAP Global ===
shap.summary_plot(shap_values, X_shap, max_display=20)
