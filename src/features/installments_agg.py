def build_installments_agg(inst, prev):

    inst.columns = inst.columns.str.strip()
    prev.columns = prev.columns.str.strip()

    inst = inst.merge(
        prev[["SK_ID_PREV", "SK_ID_CURR"]],
        on="SK_ID_PREV",
        how="left"
    )

    # 🔧 Normalizar SK_ID_CURR
    if "SK_ID_CURR_y" in inst.columns:
        inst["SK_ID_CURR"] = inst["SK_ID_CURR_y"]
    elif "SK_ID_CURR" in inst.columns:
        pass
    else:
        raise ValueError(
            f"SK_ID_CURR no encontrado tras merge. Columnas: {list(inst.columns)}"
        )

    # 🧹 Limpiar duplicadas
    inst = inst.drop(
        columns=[c for c in ["SK_ID_CURR_x", "SK_ID_CURR_y"] if c in inst.columns]
    )

    inst_agg = (
        inst
        .groupby("SK_ID_CURR")
        .agg({
            "AMT_INSTALMENT": ["mean", "sum"],
            "AMT_PAYMENT": ["mean", "sum"],
            "DAYS_ENTRY_PAYMENT": ["mean"]
        })
    )

    inst_agg.columns = [
        f"INST_{c[0]}_{c[1].upper()}" for c in inst_agg.columns
    ]

    return inst_agg.reset_index()
