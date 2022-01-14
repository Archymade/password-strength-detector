

""" Utilities for model instantiation, training, and prediction.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from data_ops import generate_features


def import_model():
    """ Import model class. """

    model = LogisticRegression
    return model


def train_model(model, X, y):
    """ Fit an instance of a model. """

    model.fit(X, y)
    return model


def get_predictions(passwords, model, transformer, return_features=False):
    """ Obtain strength prediction for password(s). """

    p = pd.Series(passwords)
    pattern = r'.*(qwert|qwer|rewq|wert|poiu|oiuy|bvcx|uytr|hgfd|iuyt|xcvb|sdfg|fghj|mnbv|jhgf|asdf|zxcv|poiuy|;lkj|lkjh|erty|rtyui|dfghj|cvbnm).*'

    features = generate_features(p, pattern)
    pred = model.predict(features)
    pred = transformer.inverse_transform(pred)
    
    features.insert(0, 'password', p)

    return (p, pred, features) if return_features else (p, pred)
