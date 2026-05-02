from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_arr, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("QuantileClipper not fitted")
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
