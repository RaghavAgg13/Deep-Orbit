"""Rung 2: Speed-driven linear Ridge filter.

Fits a Ridge-regularized linear model over the full speed lag profile
(96 lags + current speed) per horizon — a discretized linear impulse-response
function y(t+h) = sum_k w_k * Vsw(t-k) + b.
"""
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.harness import Forecaster


class LinearFilter(Forecaster):
    """Ridge regression over K speed lags (+ current), one model per horizon."""

    def __init__(self, K_steps=96, alpha=50.0):
        self.K_steps = K_steps
        self.alpha = alpha
        self.models_ = []  # one Ridge per horizon
        self._n_horizons = 3

    def _speed_matrix(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("LinearFilter requires a DataFrame with v_sw / v_lag columns.")
        # Columns: current speed + K lags, in chronological-from-present order.
        cols = ["v_sw"]
        cols += [f"v_lag_{k}" for k in range(1, self.K_steps + 1)
                 if f"v_lag_{k}" in X.columns]
        return X[cols].to_numpy(dtype=float)

    def fit(self, X, y, sample_weight=None):
        self._n_horizons = y.shape[1]
        S = self._speed_matrix(X)
        S = np.nan_to_num(S, nan=0.0)
        self.models_ = []
        for h in range(self._n_horizons):
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(S, y[:, h], sample_weight=sample_weight)
            self.models_.append(model)
        return self

    def predict(self, X):
        S = self._speed_matrix(X)
        S = np.nan_to_num(S, nan=0.0)
        preds = np.column_stack([m.predict(S) for m in self.models_])
        return preds
