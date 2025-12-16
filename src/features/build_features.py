import pandas as pd

from .application import application_features
from .previous import previous_features
from .pos_cash import pos_cash_features
from .installments import installments_features


def build_features(sk_id, data):
    # Features base (cliente actual)
    app_f = application_features(data["app"], sk_id)

    # Features de préstamos previos
    prev_f = previous_features(data["prev"], sk_id)

    # IDs de préstamos previos
    prev_ids = data["prev"].loc[
        data["prev"]["SK_ID_CURR"] == sk_id, "SK_ID_PREV"
    ]

    # Comportamiento de pago (POS)
    pos_f = pos_cash_features(data["pos"], prev_ids)

    # Comportamiento de cuotas (installments)
    inst_f = installments_features(data["inst"], prev_ids)

    # Una sola fila final
    features = pd.concat(
        [app_f, prev_f, pos_f, inst_f],
        axis=1
    )

    return features
