"""Rung 6: Physics-informed residual hybrid.

Composes a linear physics backbone (Rung 3 MultiFilter) with an ML residual
corrector (Rung 4 LightGBM). The backbone captures the dominant linear
solar-wind impulse response; the corrector learns what the linear filter
misses (non-linearities, dropouts, delayed recovery) from the residual::

    y_hybrid(t+h) = y_physics(t+h) + e_ml(t+h)
                  = backbone.predict(X) + corrector.predict(X)

Training is staged: fit backbone -> compute residual y - y_physics(train) ->
fit corrector on that residual.

Robustness: the hybrid remembers the feature columns the ML corrector was
trained on, so ``predict`` always presents the *exact same* feature set even
when the caller passes the full feature matrix (which is what ``main.py`` does
when it evaluates flat models against the sequence-aligned test rows).
"""
import numpy as np

from src.harness import Forecaster


class HybridForecaster(Forecaster):
    """Linear multi-driver backbone + ML residual corrector."""

    def __init__(self, physics_model, ml_model):
        self.physics_model = physics_model
        self.ml_model = ml_model
        self._feature_names = None  # captured at fit time from a DataFrame

    def _ml_input(self, X):
        """Restrict X to the columns the ML corrector was trained on."""
        if self._feature_names is None or not hasattr(X, "columns"):
            return X
        return X[self._feature_names]

    def fit(self, X, y, sample_weight=None, eval_set=None):
        if hasattr(X, "columns"):
            self._feature_names = list(X.columns)

        # 1. Fit physics backbone on the true target.
        self.physics_model.fit(X, y, sample_weight=sample_weight)

        # 2. Compute residuals that the linear filter fails to capture.
        y_phys = self.physics_model.predict(X)
        residuals = y - y_phys

        # 3. Fit the ML corrector on the residuals against the *same* feature
        # subset used at train time. The validation set, if provided, is
        # re-evaluated against its own residual with the physics component held
        # fixed, and features are restricted to the same subset.
        X_ml = self._ml_input(X)
        eval_set_corr = None
        if eval_set is not None:
            X_val, y_val = eval_set
            y_phys_val = self.physics_model.predict(X_val)
            eval_set_corr = (self._ml_input(X_val), y_val - y_phys_val)
        self.ml_model.fit(X_ml, residuals,
                          sample_weight=sample_weight,
                          eval_set=eval_set_corr)
        return self

    def predict(self, X, seq_idxs=None):
        y_phys = self.physics_model.predict(X)
        y_corr = self.ml_model.predict(self._ml_input(X))
        return y_phys + y_corr
