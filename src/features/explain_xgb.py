import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# cargar datos y modelo
df = pd.read_parquet("data/processed/features_xgb.parquet")
X = df.drop(columns=["TARGET"])

model = XGBClassifier()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.iloc[:200])

# resumen global
shap.summary_plot(shap_values, X.iloc[:200])