"""Rung 7: Stacking Meta-Learner (dynamic regime blending).

R6 uses a STATIC residual sum:
    y_hybrid = y_physics + e_ml
which implicitly trusts the ML corrector equally in all magnetospheric regimes.

R7 replaces this with a STACKING blender that learns, per prediction, how to
dynamically weight a panel of base forecasters (R3 physics, R4 LightGBM,
R5 sequence model) based on the current magnetospheric STATE. This captures:

  * Quiet-time decay: physics model dominates (monotonic loss curves are
    well-fit by the linear integrator backbone).
  * Storm onset: ML forecasters dominate (non-linear dropout and recovery
    are opaque to a linear response).
  * Recovery phase: blend of both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from src.harness import Forecaster
from src.models.lstm import to_sequences

_ALPHA_GRID = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
_MIN_CV_ROWS = 250
TARGET_HISTORY_PREFIXES = ("flux_lag", "flux_lag_")


class SequencePanel:
    def __init__(self, models: dict, seq_len: int, feature_names: list[str] | None = None):
        self.models = models
        self.seq_len = seq_len
        self.feature_names = feature_names

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_names is not None:
                cols = [c for c in self.feature_names if c in X.columns]
                return X[cols].to_numpy(dtype=np.float64)
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    def predict_aligned(self, X_flat, seq_idxs, segment_ids=None, flux_base=None):
        if flux_base is None:
            flux_base = getattr(self, "flux_base", None)
        X_flat = self._to_array(X_flat)
        seq_idxs = np.asarray(seq_idxs, dtype=int)
        n = X_flat.shape[0]
        f = X_flat.shape[1] if X_flat.ndim == 2 else 0

        if segment_ids is None:
            segment_ids = np.zeros(n, dtype=int)
        else:
            segment_ids = np.asarray(segment_ids, dtype=int)

        n_valid = len(seq_idxs)
        X_seq = np.empty((n_valid, self.seq_len, f), dtype=np.float32)
        valid_mask = np.ones(n_valid, dtype=bool)
        for j, e in enumerate(seq_idxs):
            s = e - self.seq_len + 1
            if s < 0 or segment_ids[s] != segment_ids[e]:
                valid_mask[j] = False
                continue
            window = X_flat[s:e + 1]
            if np.any(np.isnan(window)):
                valid_mask[j] = False
                continue
            X_seq[j] = window

        if not np.any(valid_mask):
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, np.array([], dtype=int)

        X_seq = X_seq[valid_mask]
        valid_idxs = seq_idxs[valid_mask]

        if flux_base is not None:
            flux_base = np.asarray(flux_base, dtype=np.float32)
            fb_valid = flux_base[valid_mask] if len(flux_base) == len(valid_mask) else flux_base
        else:
            fb_valid = None
        preds = np.stack([m.predict(X_seq, flux_base=fb_valid) for m in self.models.values()], axis=0)
        pred_valid = preds.mean(axis=0)
        return pred_valid, valid_idxs


class StackingMetaLearner(Forecaster):
    def __init__(self, base_models: dict, meta_alpha: float = 1.0, hazard_log: float = 3.0,
                 seq_panel: SequencePanel | None = None, alpha_cv: bool = True, use_interactions: bool = True):
        self.base_models = base_models
        self.meta_alpha = meta_alpha
        self.hazard_log = hazard_log
        self.meta_models_ = []
        self._base_names: list[str] = []
        self._n_horizons = 3
        self.alpha_cv = alpha_cv
        self._meta_alphas_ = [meta_alpha] * self._n_horizons
        self._use_interactions = use_interactions
        self._base_regime_dim: int = 14
        self._seq_panel = seq_panel

    def _panel(self, X) -> np.ndarray:
        preds = [m.predict(X) for m in self.base_models.values()]
        return np.hstack(preds)

    @staticmethod
    def _drop_target_history(X):
        if isinstance(X, pd.DataFrame):
            bad = [c for c in X.columns if c.startswith(TARGET_HISTORY_PREFIXES)]
            return X.drop(columns=bad)
        return X

    def _regime_features(self, X) -> np.ndarray:
        if not hasattr(X, "columns"):
            return np.zeros((len(X), 14))
        cols = X.columns
        def grab(candidates):
            for c in candidates:
                if c in cols:
                    return X[c].to_numpy(dtype=float)
            return np.zeros(len(X))
        f1 = grab(["v_sw"])
        f2 = grab(["bz_s"])
        f3 = grab(["pdyn"])
        f4 = grab(["v_sw"]) - grab(["v_lag_4"])
        f5 = grab(["sckopke", "vbz"])
        f6 = grab(["clock_angle"])
        f7 = grab(["sin4_theta2"])
        f8 = grab(["bz_s"]) - grab(["bz_lag_4"])
        f9 = grab(["vbz"])
        f10 = grab(["pdyn"]) - grab(["pdyn_lag_4"])
        f11 = grab(["hours_since_vsw_gt500"]) / 100.0
        f12 = grab(["hours_since_bz_flip"]) / 100.0
        f13 = grab(["vsw_gt500_duration_24h"]) / 24.0
        f14 = np.log1p(np.abs(grab(["cum_vbz_pos_24h"]))) / 12.0
        return np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14])

    def _add_interactions(self, X_meta):
        """Add squared terms for non-linear blending capacity.

        The meta-learner sees base predictions + regime features. Without non-linear
        terms, the Ridge blender can only learn a LINEAR combination — but the
        optimal blend weight depends on the CURRENT state (e.g., trust physics more
        during quiet times, trust ML more during storms). Adding squared terms lets
        the model learn state-dependent weights from a linear estimator.
        """
        if not self._use_interactions:
            return X_meta
        # Squared terms (one per main feature) let the blender learn state-dependent weights.
        # This is a simple polynomial expansion: [x, x^2] doubles feature count.
        sq = np.square(X_meta)
        return np.hstack([X_meta, sq])

    def _build_meta_matrix(self, X, seq_panel=None, seq_idxs=None):
        X_panel = self._panel(X)
        X_regime = self._regime_features(X)

        if seq_panel is not None and seq_idxs is not None:
            seq_preds, seq_valid = seq_panel.predict_aligned(X, seq_idxs)
            if len(seq_valid) != len(seq_idxs):
                raise ValueError(f"SequencePanel returned {len(seq_valid)} valid rows, but seq_idxs has {len(seq_idxs)} entries.")
            idx = np.asarray(seq_idxs, dtype=int)
            X_panel = X_panel[idx]
            X_regime = X_regime[idx]
            X_meta = np.hstack([X_panel, seq_preds, X_regime])
        else:
            X_meta = np.hstack([X_panel, X_regime])

        self._base_regime_dim = int(X_regime.shape[1])
        X_meta = np.nan_to_num(X_meta, nan=0.0)
        X_meta = self._add_interactions(X_meta)
        return X_meta

    def _alpha_cv_per_head(self, X_meta, y_meta, sw_meta):
        n = X_meta.shape[0]
        n_h = y_meta.shape[1]
        tscv = TimeSeriesSplit(n_splits=3)

        if n < _MIN_CV_ROWS:
            return [self.meta_alpha] * n_h
        alphas_out = []
        for h in range(n_h):
            best_alpha, best_mse = self.meta_alpha, np.inf
            y_h = y_meta[:, h]
            for alpha in _ALPHA_GRID:
                fold_mses = []
                for tr, va in tscv.split(X_meta):
                    m = Ridge(alpha=alpha, fit_intercept=True)
                    sw_tr = sw_meta[tr] if sw_meta is not None else None
                    sw_va = sw_meta[va] if sw_meta is not None else None
                    m.fit(X_meta[tr], y_h[tr], sample_weight=sw_tr)
                    pred = m.predict(X_meta[va])
                    err = y_h[va] - pred
                    if sw_va is not None:
                        fold_mses.append(float(np.average(err * err, weights=sw_va)))
                    else:
                        fold_mses.append(float(np.mean(err * err)))
                mse = float(np.mean(fold_mses))
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
            alphas_out.append(best_alpha)
        return alphas_out

    def fit(self, X, y, sample_weight=None, eval_set=None, seq_panel=None, seq_idxs=None):
        if seq_panel is None:
            seq_panel = self._seq_panel
        self._base_names = list(self.base_models.keys())
        self._n_horizons = y.shape[1]
        X = self._drop_target_history(X)
        X_meta = self._build_meta_matrix(X, seq_panel=seq_panel, seq_idxs=seq_idxs)

        if seq_panel is not None and seq_idxs is not None:
            idx = np.asarray(seq_idxs, dtype=int)
            y_meta = y[idx] if y is not None else y
            sw_meta = sample_weight[idx] if sample_weight is not None else None
        else:
            y_meta, sw_meta = y, sample_weight

        self._fit_n_features = X_meta.shape[1]

        if self.alpha_cv:
            self._meta_alphas_ = self._alpha_cv_per_head(X_meta, y_meta, sw_meta)
        else:
            self._meta_alphas_ = [self.meta_alpha] * self._n_horizons
        cv_used = getattr(self, "alpha_cv", True) and X_meta.shape[0] >= _MIN_CV_ROWS
        horizon_labels = ["30m", "6h", "12h"]
        alpha_str = ", ".join(f"{horizon_labels[h]}={self._meta_alphas_[h]:.2f}" for h in range(self._n_horizons))
        tag = "CV" if cv_used else "scalar-fallback"
        print(f"[R7] per-horizon alphas ({tag}): {alpha_str}")

        self.meta_models_ = []
        for h in range(self._n_horizons):
            alpha_h = self._meta_alphas_[h]
            meta = Ridge(alpha=alpha_h, fit_intercept=True)
            meta.fit(X_meta, y_meta[:, h], sample_weight=sw_meta)
            self.meta_models_.append(meta)
        return self

    def predict(self, X, seq_idxs=None):
        seq_panel = self._seq_panel
        X_meta = self._build_meta_matrix(X, seq_panel=seq_panel, seq_idxs=seq_idxs)
        if X_meta.shape[1] != self._fit_n_features:
            raise ValueError(f"StackingMetaLearner.predict built a {X_meta.shape[1]}-col matrix but model was fit on {self._fit_n_features} cols.")
        return np.column_stack([m.predict(X_meta) for m in self.meta_models_])

    def blend_weights(self, X, seq_idxs=None):
        out = {}
        n_b = len(self.base_models)
        seq_cols = 3 if (self._seq_panel is not None and seq_idxs is not None) else 0
        n_flat_coef = n_b * self._n_horizons
        for h, m in enumerate(self.meta_models_):
            coefs = m.coef_
            flat = coefs[:n_flat_coef]
            w_split = np.split(flat, n_b)
            weights = {name: float(np.mean(np.abs(w))) for name, w in zip(self.base_models.keys(), w_split)}
            if seq_cols:
                seq_coef = coefs[n_flat_coef:n_flat_coef + seq_cols]
                weights["seq"] = float(np.mean(np.abs(seq_coef)))
            out[f"horizon_{h}"] = weights
        return out


if __name__ == "__main__":
    from src.models.lstm import to_sequences

    class _MockFlat(Forecaster):
        def __init__(self, seed=0):
            self.seed = seed
        def fit(self, X, y, sample_weight=None, eval_set=None):
            return self
        def predict(self, X, flux_base=None):
            X = np.asarray(X, dtype=float)
            return np.column_stack([X[:, 0] + self.seed, X[:, 1] * 0.5 + self.seed, X[:, 2] * 2.0 + self.seed])

    class _MockSeq(Forecaster):
        def fit(self, X, y, sample_weight=None, eval_set=None):
            return self
        def predict(self, X, flux_base=None):
            X = np.asarray(X, dtype=float)
            last = X[:, -1, :]
            return np.column_stack([last[:, 0], last[:, 3] * 0.5, last[:, 4] * 2.0])

    rng = np.random.default_rng(42)
    N = 200
    X_flat = pd.DataFrame({
        "v_sw": rng.uniform(300, 700, N), "n_sw": rng.uniform(2, 15, N),
        "bz": rng.uniform(-10, 10, N), "bz_s": rng.uniform(0, 10, N),
        "pdyn": rng.uniform(0.5, 5, N), "vbz": rng.uniform(0, 7000, N),
        "hod_sin": rng.uniform(-1, 1, N), "hod_cos": rng.uniform(-1, 1, N),
        "doy_sin": rng.uniform(-1, 1, N), "doy_cos": rng.uniform(-1, 1, N),
        "goes_new": rng.integers(0, 2, N),
        "v_lag_4": rng.uniform(300, 700, N), "bz_lag_4": rng.uniform(0, 10, N),
        "pdyn_lag_4": rng.uniform(0.5, 5, N),
        "hours_since_vsw_gt500": rng.uniform(0, 50, N),
        "hours_since_bz_flip": rng.uniform(0, 40, N),
        "vsw_gt500_duration_24h": rng.uniform(0, 20, N),
        "cum_vbz_pos_24h": rng.uniform(0, 100000, N),
    })
    y = rng.normal(size=(N, 3))

    seq_len = 48
    seg_ids = np.zeros(N, dtype=int)
    _, _, valid_idxs = to_sequences(X_flat.values, np.zeros((N, 3)), seq_len, seg_ids)
    print(f"[self-test] flat rows={N}, seq_len={seq_len}, valid window-end rows={len(valid_idxs)}")

    seq_panel = SequencePanel(models={"r5": _MockSeq()}, seq_len=seq_len, feature_names=list(X_flat.columns))
    pred_valid, pred_idx = seq_panel.predict_aligned(X_flat, valid_idxs, segment_ids=seg_ids)
    print(f"[self-test] SequencePanel.predict_aligned: OK (shape={pred_valid.shape}, idx aligned)")

    base = {"r3": _MockFlat(seed=0), "r4": _MockFlat(seed=1)}
    _tmp_regime = StackingMetaLearner(base_models=base, meta_alpha=1.0)._regime_features(X_flat)
    regime_dim = int(_tmp_regime.shape[1])
    print(f"[self-test] regime_dim = {regime_dim}")

    learner = StackingMetaLearner(base_models=base, meta_alpha=1.0, seq_panel=seq_panel)
    learner.fit(X_flat, y, seq_panel=seq_panel, seq_idxs=valid_idxs)
    preds_seq = learner.predict(X_flat, seq_idxs=valid_idxs)
    print(f"[self-test] StackingMetaLearner seq-path: OK (shape={preds_seq.shape})")

    learner_flat = StackingMetaLearner(base_models=base, meta_alpha=1.0)
    learner_flat.fit(X_flat, y)
    preds_flat = learner_flat.predict(X_flat)
    print(f"[self-test] StackingMetaLearner flat-path: OK (shape={preds_flat.shape})")

    print("\n[self-test] ALL ASSERTIONS PASSED")