"""Rung 0b: Climatology baseline.

Predicts the historic mean log-flux conditional on the hour-of-day. We recover
the hour angle from the precomputed cyclic (sin, cos) features, then look up the
per-hour mean observed during training. A constant per-horizon lookup table is
sufficient because the climatology of the *future* state is the same hourly
distribution regardless of lead time for this baseline.

IMPORTANT: like Persistence, the hourly means are built from whichever flux
proxy is available in X (flux_lag_1 or log_flux). Since the pipeline standardises
these columns, we inverse-transform them back to raw log10 flux units so the
output is directly comparable to the unscaled target y.
"""
import numpy as np
import pandas as pd

from src.harness import Forecaster


class Climatology(Forecaster):
    """Hour-of-day conditional mean baseline."""

    def __init__(self, sin_col="hod_sin", cos_col="hod_cos", scaler=None):
        self.sin_col = sin_col
        self.cos_col = cos_col
        self.scaler = scaler
        self.hourly_mean = None  # index 0..23 -> mean log-flux (RAW scale)

    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Climatology requires a DataFrame with cyclic columns.")
        # Recover the continuous hour-of-day angle from the (sin, cos) encoding.
        angle = np.arctan2(X[self.sin_col].to_numpy(),
                           X[self.cos_col].to_numpy())
        hours = ((angle / (2.0 * np.pi)) * 24.0) % 24.0
        hours = np.round(hours).astype(int) % 24

        # Use the *current* observed flux, NOT the future target. y[:, 0] is
        # the 30-min-ahead TARGET (HORIZON_STEPS[0] = 2 steps); building the
        # climatology from it would be target leakage (the baseline would
        # learn the hourly distribution of the *future* flux we must predict).
        # The pipeline deliberately excludes log_flux from ml_features to
        # prevent copy-paste leakage in the ML models. The closest proxy
        # available in X is flux_lag_1 (the flux 15 min ago) — negligible
        # difference for a climatology baseline. Pass the full X matrix
        # (data['X_train']), not the ml_features subset, so flux_lag_1 is
        # present.
        col_name = None
        if "flux_lag_1" in X.columns:
            flux = X["flux_lag_1"].to_numpy(dtype=float)
            col_name = "flux_lag_1"
        elif "log_flux" in X.columns:
            flux = X["log_flux"].to_numpy(dtype=float)
            col_name = "log_flux"
        else:
            # No current-flux proxy is available. We MUST NOT fall back to
            # y[:, 0] (the 30-min-ahead target) — that would build the
            # climatology from the *future* state we are supposed to predict,
            # which is target leakage. Raise instead of leaking silently.
            raise ValueError(
                "Climatology requires a current-flux proxy (flux_lag_1 or "
                "log_flux) in X; neither is present. The pipeline excludes "
                "flux history from ml_features, so pass the full X matrix "
                "(data['X_train']), not the ml_features subset.")

        # Inverse-transform from standardised space to raw log10 flux so the
        # hourly means are in the same units as the target y.
        if self.scaler is not None and hasattr(self.scaler, "mean_"):
            col_names = list(X.columns)
            if col_name in col_names:
                idx = col_names.index(col_name)
                flux = flux * self.scaler.scale_[idx] + self.scaler.mean_[idx]

        df = pd.DataFrame({"h": hours, "f": flux})
        self.hourly_mean = df.groupby("h")["f"].mean()
        # Fallback to global mean for any unseen hour (shouldn't occur).
        global_mean = float(df["f"].mean())
        self.hourly_mean = self.hourly_mean.reindex(range(24), fill_value=global_mean)
        return self

    def predict(self, X):
        if self.hourly_mean is None:
            raise RuntimeError("Climatology model is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Climatology requires a DataFrame with cyclic columns.")
        angle = np.arctan2(X[self.sin_col].to_numpy(),
                           X[self.cos_col].to_numpy())
        hours = (((angle / (2.0 * np.pi)) * 24.0) % 24.0).astype(int) % 24
        forecast = self.hourly_mean.to_numpy()[hours]
        return np.column_stack([forecast, forecast, forecast])
