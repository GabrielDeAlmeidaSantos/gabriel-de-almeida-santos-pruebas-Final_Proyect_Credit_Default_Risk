import pandas as pd
import numpy as np

def installments_features(inst, prev_ids):
    df = inst[inst["SK_ID_PREV"].isin(prev_ids)].copy()

    if df.empty:
        return pd.DataFrame({
            "inst_late_ratio_all": [0],
            "inst_dpd_mean_all": [0],
            "inst_dpd_max_all": [0],
            "inst_payment_ratio_all": [1],
            "inst_late_ratio_last_6m": [0],
            "inst_dpd_trend": [0],
            "months_since_last_late_inst": [np.nan],
        })

    # DPD por cuota (positivo si pagó tarde)
    dpd = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)

    # Global
    late_ratio_all = (dpd > 0).mean()
    dpd_mean_all = dpd.mean()
    dpd_max_all = dpd.max()

    # Payment ratio
    payment_ratio_all = (
        df["AMT_PAYMENT"].sum() / df["AMT_INSTALMENT"].sum()
        if df["AMT_INSTALMENT"].sum() > 0 else 1
    )

    # Ventana últimos 6 meses (más reciente: DAYS_INSTALMENT cerca de 0)
    last_6m = df[df["DAYS_INSTALMENT"] >= -180]
    dpd_last_6m = (last_6m["DAYS_ENTRY_PAYMENT"] - last_6m["DAYS_INSTALMENT"]).clip(lower=0)

    late_ratio_last_6m = (dpd_last_6m > 0).mean() if not last_6m.empty else 0

    # Tendencia
    inst_dpd_trend = (
        (dpd_last_6m.mean() if not last_6m.empty else 0) - dpd_mean_all
    )

    # Recencia último retraso
    late_rows = df.loc[dpd > 0, "DAYS_INSTALMENT"]
    months_since_last_late = (
        abs(late_rows.max()) // 30 if not late_rows.empty else np.nan
    )

    return pd.DataFrame({
        "inst_late_ratio_all": [late_ratio_all],
        "inst_dpd_mean_all": [dpd_mean_all],
        "inst_dpd_max_all": [dpd_max_all],
        "inst_payment_ratio_all": [payment_ratio_all],
        "inst_late_ratio_last_6m": [late_ratio_last_6m],
        "inst_dpd_trend": [inst_dpd_trend],
        "months_since_last_late_inst": [months_since_last_late],
    })
