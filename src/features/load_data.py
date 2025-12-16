import pandas as pd
from .pos_cash_agg import build_pos_cash_agg
from .installments_agg import build_installments_agg

def load_data():
    app = pd.read_csv("data/raw/application_tr.csv")
    prev = pd.read_csv("data/raw/previous_application.csv")
    pos = pd.read_csv("data/raw/POS_CASH_balance.csv")
    inst = pd.read_csv("data/raw/installments_payments.csv")

    # Pre-agregaciones a nivel cliente
    pos_agg = build_pos_cash_agg(pos, prev)
    inst_agg = build_installments_agg(inst, prev)

    return {
        "app": app,
        "prev": prev,
        "pos_agg": pos_agg,
        "inst_agg": inst_agg
    }
