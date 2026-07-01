"""Rung 4: LightGBM gradient-boosted trees.

One independent LightGBM regressor per horizon. The caller supplies the
feature subset (``ml_features``); this module does not hard-code any
column list. High-flux storm samples (log10 flux >= 3.0) are up-weighted
so the optimisation penalises hazard-level misses more heavily. The class
defaults are conservative and are overridden by ``main.py`` in the live
pipeline:

- ``GBMForecaster`` (R4): ``n_estimators=500``, ``learning_rate=0.03``,
  ``num_leaves=51``
- ``HybridForecaster`` corrector (R6): ``n_estimators=500``,
  ``learning_rate=0.03``, ``num_leaves=15``
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.harness import Forecaster


class GBMForecaster(Forecaster):
    """Per-horizon LightGBM regressor with storm sample weighting."""

    def __init__(self, n_estimators=500, learning_rate=0.03, num_leaves=31):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.models_ = []
        self._n_horizons = 3
        self._feature_names = None  # captured at fit time from a DataFrame

    @staticmethod
    def _sample_weight(y, threshold=3.0, w_high=8.0):
        """Per-sample hazard weight using the PROJECT CONVENTION: a sample is
        a storm sample if flux at ANY horizon reaches the hazard threshold,
        i.e. ``max(y, axis=1)`` — not ``y[:, 0]``. A storm at any lead time
        up-weights the sample for ALL horizon heads equally. This matches the
        weighting applied by ``main.py`` (weights_train / weights_val) so a
        standalone ``GBMForecaster`` trained without explicit weights behaves
        identically to the in-pipeline one."""
        w = np.ones(len(y), dtype=float)
        mask = np.max(y, axis=1) >= threshold
        w[mask] = w_high
        return w

    @staticmethod
    def _to_frame(X, columns=None):
        """Ensure X is a clean, NaN-free DataFrame.

        LGBMRegressor records ``feature_name_`` when fitted with a DataFrame
        and later validates that predict() receives the same names. Passing
        a DataFrame at *both* fit and predict time eliminates the "X does not
        have valid feature names" warning that fires when fit receives numpy
        but predict receives a DataFrame (or vice versa).
        """
        if isinstance(X, pd.DataFrame):
            if columns is not None:
                X = X.reindex(columns=columns, fill_value=0.0)
            return X.fillna(0.0)
        arr = np.asarray(X, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        if columns is not None and len(columns) == arr.shape[1]:
            return pd.DataFrame(arr, columns=columns)
        return pd.DataFrame(arr)

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self._n_horizons = y.shape[1]
        if hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        X_df = self._to_frame(X, columns=self._feature_names)

        X_val_df = None
        y_val_arr = None
        if eval_set is not None:
            Xv, yv = eval_set
            X_val_df = self._to_frame(Xv, columns=self._feature_names)
            y_val_arr = yv

        self.models_ = []
        for h in range(self._n_horizons):
            # Default weight (used only when the caller passes no
            # sample_weight) follows the project convention: hazard at ANY
            # horizon up-weights the sample. When sample_weight IS supplied
            # (the normal in-pipeline path), it is reused verbatim for every
            # horizon head.
            w = (sample_weight if sample_weight is not None
                 else self._sample_weight(y))
            params = {
                "objective": "regression",
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "n_estimators": self.n_estimators,
                "metric": "rmse",
                "verbose": -1,
                "seed": 42,
            }
            model = lgb.LGBMRegressor(**params)
            # early_stopping(50): with num_leaves=63 and lr=0.03 the model
            # needs more rounds to converge than the old 30-round patience.
            # 50 rounds × 0.03 lr ≈ 1.5 leaves of additional refinement.
            kwargs = {"sample_weight": w, "callbacks": [lgb.early_stopping(50, verbose=False)]}
            if X_val_df is not None:
                kwargs["eval_set"] = [(X_val_df, y_val_arr[:, h])]
            else:
                # No validation set: disable early stopping callbacks to keep training
                # deterministic for the full n_estimator budget.
                kwargs.pop("callbacks", None)
            model.fit(X_df, y[:, h], **kwargs)
            self.models_.append(model)
        return self

    def predict(self, X):
        # Restrict to the exact columns seen at fit time so callers can safely
        # pass the full feature DataFrame (as main.py does). Converting to
        # DataFrame with the *same* columns the internal LGBMRegressor was
        # fitted on silences the sklearn feature-name validation warning.
        X_df = self._to_frame(X, columns=self._feature_names)
        return np.column_stack([m.predict(X_df) for m in self.models_])
