import pandas as pd

def load_data():
    data = {
        "app": pd.read_csv("data/raw/application_tr.csv"),
        "prev": pd.read_csv("data/raw/previous_application.csv"),
        "pos": pd.read_csv("data/raw/POS_CASH_balance.csv"),
        "inst": pd.read_csv("data/raw/installments_payments.csv"),
    }
    return data
