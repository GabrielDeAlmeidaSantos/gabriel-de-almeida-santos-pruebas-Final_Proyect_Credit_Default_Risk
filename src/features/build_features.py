import pandas as pd
from .application import application_features
from .previous import previous_features
from .pos_cash import pos_cash_features

def build_features(sk_id, data):
    app_f = application_features(data["app"], sk_id)
    prev_f = previous_features(data["prev"], sk_id)

    prev_ids = data["prev"].loc[
        data["prev"]["SK_ID_CURR"] == sk_id, "SK_ID_PREV"
    ]

    pos_f = pos_cash_features(data["pos"], prev_ids)

    features = pd.concat([app_f, prev_f, pos_f], axis=1)
    return features