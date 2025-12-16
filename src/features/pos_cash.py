import pandas as pd
import numpy as np

def pos_cash_features(pos, prev_ids):
    df = pos[pos["SK_ID_PREV"].isin(prev_ids)].copy()

    if df.empty:
        return pd.DataFrame({
            "pos_dpd_mean_all": [0],
            "pos_late_ratio_all": [0],
            "pos_dpd_mean_last_6m": [0],
            "pos_late_ratio_last_6m": [0],
            "pos_dpd_mean_older": [0],
            "pos_late_ratio_older": [0],
            "pos_dpd_trend": [0],
            "months_since_last_late": [np.nan],
        })

    # ordenar por tiempo (0 es más reciente)
    df = df.sort_values("MONTHS_BALANCE")

    # global
    dpd_all = df["SK_DPD"]
    late_all = dpd_all > 0

    # últimos 6 meses
    last_6m = df[df["MONTHS_BALANCE"] >= -6]
    older = df[df["MONTHS_BALANCE"] < -6]

    # cálculo seguro
    def safe_mean(x): 
        return x.mean() if not x.empty else 0

    def safe_ratio(x): 
        return (x > 0).mean() if not x.empty else 0

    # meses desde último retraso
    late_months = df.loc[df["SK_DPD"] > 0, "MONTHS_BALANCE"]
    months_since_last_late = (
        abs(late_months.max()) if not late_months.empty else np.nan
    )

    return pd.DataFrame({
        "pos_dpd_mean_all": [safe_mean(dpd_all)],
        "pos_late_ratio_all": [safe_ratio(dpd_all)],
        "pos_dpd_mean_last_6m": [safe_mean(last_6m["SK_DPD"])],
        "pos_late_ratio_last_6m": [safe_ratio(last_6m["SK_DPD"])],
        "pos_dpd_mean_older": [safe_mean(older["SK_DPD"])],
        "pos_late_ratio_older": [safe_ratio(older["SK_DPD"])],
        "pos_dpd_trend": [
            safe_mean(last_6m["SK_DPD"]) - safe_mean(older["SK_DPD"])
        ],
        "months_since_last_late": [months_since_last_late],
    })