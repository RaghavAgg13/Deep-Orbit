"""Rung 0a: Persistence baseline.

The persistence forecast asserts the future flux equals the most recent
observation: y_hat(t + h) = y(t). We approximate "y(t)" with the lagged flux
column ``flux_lag_1`` (the observation one step behind the current row), which
is the convention used by the pipeline's feature matrix.

IMPORTANT: the pipeline's StandardScaler normalises ``flux_lag_1`` to
zero-mean / unit-variance, but the target ``y`` is in *raw* log10 flux units.
A persistence forecast in scaled space is meaningless when scored against raw y.
When the scaler is provided (the normal path from ``main.py``), the model
inverse-transforms its prediction back to the raw scale before returning.
"""
import numpy as np
import pandas as pd

from src.harness import Forecaster


class Persistence(Forecaster):
    """Naive persistence baseline: forecast = last observed log-flux."""

    def __init__(self, flux_col_name="flux_lag_1", scaler=None):
        self.flux_col_name = flux_col_name
        self.scaler = scaler

    # Persistence is parameter-free; fit() is a no-op kept for interface parity.
    def fit(self, X, y, sample_weight=None):  # noqa: D401 - interface parity
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if self.flux_col_name not in X.columns:
                raise ValueError(
                    f"Persistence requires column '{self.flux_col_name}' in X."
                )
            last_flux = X[self.flux_col_name].to_numpy(dtype=float)
        else:
            last_flux = np.asarray(X, dtype=float)[:, 0]

        # Inverse-transform from standardised space back to raw log10 flux.
        # Without this the forecast is in (x - mean) / std units and scoring
        # against the unscaled target y produces meaningless PE / RMSE.
        if self.scaler is not None and hasattr(self.scaler, "mean_"):
            col_names = None
            if isinstance(X, pd.DataFrame):
                col_names = list(X.columns)
            if col_names is not None and self.flux_col_name in col_names:
                idx = col_names.index(self.flux_col_name)
                last_flux = last_flux * self.scaler.scale_[idx] + self.scaler.mean_[idx]

        # Broadcast the scalar-per-row forecast across all 3 horizons.
        return np.column_stack([last_flux, last_flux, last_flux])
