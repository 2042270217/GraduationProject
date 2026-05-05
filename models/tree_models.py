from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


class XGBoostMultiOutput:
    def __init__(self, **kwargs):
        self._kw = kwargs
        self.model = None

    def fit(self, X, y):
        base = XGBRegressor(**self._kw)
        self.model = MultiOutputRegressor(base, n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)


class GBMMultiOutput:
    def __init__(self, **kwargs):
        self._kw = kwargs
        self.model = None

    def fit(self, X, y):
        base = GradientBoostingRegressor(**self._kw)
        self.model = MultiOutputRegressor(base, n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("模型未训练")
        return self.model.predict(X)
