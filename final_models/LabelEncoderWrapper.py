from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class LabelEncoderWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrap any classifier so it can be trained with string labels.
    """
    def __init__(self, clf):
        self.clf = clf
        self.le  = LabelEncoder()

    def fit(self, X, y):
        y_enc = self.le.fit_transform(y)
        self.clf.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.clf.predict(X)
        return self.le.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def score(self, X, y):
        y_enc = self.le.transform(y)
        return self.clf.score(X, y_enc)

    # Enable grid-search / cloning
    def get_params(self, deep=True):
        return {"clf": self.clf}

    def set_params(self, **params):
        if "clf" in params:
            self.clf = params["clf"]
        return self
