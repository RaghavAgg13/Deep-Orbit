"""Rung 3: Multi-driver impulse-response filter.

Ridge regression over the lag profiles of three solar-wind drivers — speed,
southward IMF (bz_s) and dynamic pressure (pdyn) — yielding a multi-channel
discrete impulse response. This is the physics backbone reused by the Rung-6
hybrid (linear backbone + ML residual corrector).

Per horizon h:
    y(t+h) = sum_k [ wV_k*Vsw(t-k) + wB_k*BzS(t-k) + wP_k*Pdyn(t-k) ] + b

The K_steps lags are downsampled to an exponential lag schedule inside
``_fit_columns`` so the Ridge can encode 24h of history with ~13 columns
per channel instead of 96 — regularising the linear backbone without
sacrificing the long-memory tail.

Fit/predict feature-contract safety
------------------------------------
``predict(X)`` always re-indexes ``X`` to the **exact same column set**
seen at fit time (``self._fit_columns``). This is REQUIRED because the
caller in ``main.py`` evaluates Hybrid/MultiFilter on the full 896-column
test matrix (sequence-aligned rows), while fit-time received only the
``ml_features`` subset. Re-indexing keeps the Ridge feature vector length
identical and treats any missing column as 0.0.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.harness import Forecaster


class MultiFilter(Forecaster):
    """Ridge regression over Vsw / bz_s / pdyn lags, one model per horizon."""

    # All (driver, lag_prefix) channels that can feed the filter ladder.
    # Base trio (speed, southward IMF, dynamic pressure) plus the enriched
    # coupling-function channels added in features.py. Channels that are
    # absent from a given feature matrix are silently zero-filled.
    DRIVER_CHANNELS = [
        ("v_sw",       "v_lag_"),
        ("bz_s",       "bz_lag_"),
        ("pdyn",       "pdyn_lag_"),
        ("clock_angle","clock_lag_"),
        ("sin4_theta2","sin4_lag_"),
        ("sckopke",    "sckopke_lag_"),
        ("viscous_proxy","visc_lag_"),
    ]

    # Exponential lag schedule: log2-spaced, 13 lags covering 0..96 steps
    # (i.e. 0-24 h). The 8-step schedule is what the R3 Ridge actually
    # consumes per channel -- roughly a 7x reduction in coefficient
    # dimension. NOTE: this is the LARGE-LAG schedule used for the Ridge
    # backbone; the K most-recent lags (k < schedule[0]) are skipped as
    # already representable through the instantaneous driver value.
    LAG_SCHEDULE = [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96]

    def __init__(self, K_steps=96, alpha=50.0, schedule=None):
        self.K_steps = K_steps
        self.alpha = alpha
        self.schedule = schedule if schedule is not None else self.LAG_SCHEDULE
        self.models_ = []
        self._n_horizons = 3
        # Coefficient matrices for diagnostics: {driver: (n_horizons, n_cols)}
        self._resp = {}
        self._fit_columns: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _build_fit_columns(self, X: pd.DataFrame) -> list[str]:
        """Driver + scheduled-lag columns that actually exist in X."""
        cols: list[str] = []
        for base, pref in self.DRIVER_CHANNELS:
            if base in X.columns:
                cols.append(base)
            for k in self.schedule:
                c = f"{pref}{k}"
                if c in X.columns:
                    cols.append(c)
        # Preserve caller column ordering (stable for coef slicing).
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _driver_matrix(self, X) -> np.ndarray:
        """Build the (n, len(_fit_columns)) float matrix for Ridge.

        At fit time this selects the scheduled columns from X; at predict
        time X is re-indexed to the same column set so the feature vector
        length is matched. Missing columns become 0.0 (harmless for the
        linear backbone -- the driver is assumed at background).
        """
        if isinstance(X, pd.DataFrame) and self._fit_columns is not None:
            return np.asarray(
                X.reindex(columns=self._fit_columns).fillna(0.0), dtype=float
            )
        if isinstance(X, pd.DataFrame):
            sub = X[[c for c in X.columns if any(
                c == b or c.startswith(p)
                for b, p in self.DRIVER_CHANNELS)]]
            return np.asarray(sub.fillna(0.0), dtype=float)
        return np.asarray(X, dtype=float)

    # ------------------------------------------------------------------ #
    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("MultiFilter.fit requires a DataFrame.")
        self._fit_columns = self._build_fit_columns(X)
        if not self._fit_columns:
            raise ValueError(
                "MultiFilter: no driver channels present in X. "
                f"X.columns[:10] = {list(X.columns)[:10]}")
        self._n_horizons = y.shape[1]
        M = np.nan_to_num(self._driver_matrix(X), nan=0.0)

        # Build the diagnostic snapshot {driver: (n_horizons, n_cols_slice)}.
        self._resp = {}
        driver_slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for base, pref in self.DRIVER_CHANNELS:
            cols_here = [c for c in self._fit_columns
                         if c == base or c.startswith(pref)]
            if not cols_here:
                self._resp[base] = np.zeros((self._n_horizons, 0))
                continue
            driver_slices[base] = (cursor, cursor + len(cols_here))
            self._resp[base] = np.zeros((self._n_horizons, len(cols_here)))
            cursor += len(cols_here)

        self.models_ = []
        for h in range(self._n_horizons):
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(M, y[:, h], sample_weight=sample_weight)
            self.models_.append(model)
            coeffs = model.coef_
            for base, (s, e) in driver_slices.items():
                self._resp[base][h, :] = coeffs[s:e]
        return self

    def predict(self, X):
        M = np.nan_to_num(self._driver_matrix(X), nan=0.0)
        return np.column_stack([m.predict(M) for m in self.models_])

    def get_impulse_responses(self):
        """Return per-driver coefficient matrices for diagnostic plotting."""
        return self._resp
