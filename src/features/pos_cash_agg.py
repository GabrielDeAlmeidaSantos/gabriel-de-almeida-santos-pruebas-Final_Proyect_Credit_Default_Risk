def build_pos_cash_agg(pos, prev):

    pos.columns = pos.columns.str.strip()
    prev.columns = prev.columns.str.strip()

    pos = pos.merge(
        prev[["SK_ID_PREV", "SK_ID_CURR"]],
        on="SK_ID_PREV",
        how="left"
    )


    if "SK_ID_CURR_y" in pos.columns:
        pos["SK_ID_CURR"] = pos["SK_ID_CURR_y"]
    elif "SK_ID_CURR" in pos.columns:
        pass
    else:
        raise ValueError(f"No se encontró SK_ID_CURR tras merge: {list(pos.columns)}")


    pos = pos.drop(columns=[c for c in ["SK_ID_CURR_x", "SK_ID_CURR_y"] if c in pos.columns])

    pos_agg = (
        pos
        .groupby("SK_ID_CURR")
        .agg({
            "MONTHS_BALANCE": ["min", "max", "mean"],
            "SK_DPD": ["max", "mean"],
            "SK_DPD_DEF": ["max", "mean"]
        })
    )

    pos_agg.columns = [
        f"POS_{c[0]}_{c[1].upper()}" for c in pos_agg.columns
    ]

    return pos_agg.reset_index()
