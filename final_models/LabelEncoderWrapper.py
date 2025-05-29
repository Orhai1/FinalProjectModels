from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class LabelEncoderWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrap any classifier so it can be trained with string labels.
    """
    def __init__(self, model):
        self.model = model
        self.le  = LabelEncoder()

    def fit(self, X, y):
        y_enc = self.le.fit_transform(y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.model.predict(X)
        return self.le.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        y_enc = self.le.transform(y)
        return self.model.score(X, y_enc)

    # Enable grid-search / cloning
    def get_params(self, deep=True):
        params = {"model": self.model}
        if deep:
            # prepend 'model__' to each inner param
            for k, v in self.model.get_params(deep=True).items():
                params[f"model__{k}"] = v
        return params

    def set_params(self, **params):
        if "model" in params:
            self.model = params.pop("model")
        model_params = {k.split("__", 1)[1]: v
                        for k, v in params.items() if k.startswith("model__")}
        if model_params:
            self.model.set_params(**model_params)
        return self
