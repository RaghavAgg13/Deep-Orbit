"""Rung 1: Single-driver lagged linear regression.

For each horizon h independently, pick the (driver, lag) pair with the
highest *weighted* absolute correlation to y(t+h), then fit a univariate
weighted-least-squares line. Candidates are drawn from the three dominant
solar-wind drivers of MeV electron flux — speed (v_sw), southward IMF (bz_s)
and dynamic pressure (pdyn) — so the baseline can recover the dominant
delay per horizon from whichever channel is most informative.

Weighted WLS (instead of scipy linregress, which ignores weights) lets the
hazard sample-weighting flow into the baseline: storm-time rows matter more
than quiet-time rows when choosing the lag and fitting the line.
"""
import re  # used for column-name parsing in _driver_lags / predict
import numpy as np
import pandas as pd

from src.harness import Forecaster

# Driver channels to scan, in priority order. Each base name <-> lag-prefix
# pair produces a family of `prefix<k>` columns (v_lag_1, ..., v_lag_96).
DRIVER_CHANNELS = [
    ("v_sw",  "v_lag_"),
    ("bz_s",  "bz_lag_"),
    ("pdyn",  "pdyn_lag_"),
]


class LaggedLinear(Forecaster):
    """One (driver, lag) + intercept per horizon, fit by weighted LS."""

    def __init__(self):
        # Per-horizon: (driver_base, lag_prefix, lag_index, slope, intercept)
        self.coefs_ = {}

    def _driver_lags(self, X):
        """Return {base: (lag_indices, DataFrame)} for every channel present."""
        if not isinstance(X, pd.DataFrame):
            return {}
        out = {}
        for base, pref in DRIVER_CHANNELS:
            cols = []
            idxs = []
            for c in X.columns:
                m = re.match(rf"^{re.escape(pref)}(\d+)$", c)
                if m:
                    cols.append(c)
                    idxs.append(int(m.group(1)))
            if not cols:
                continue
            order = np.argsort(idxs)
            out[base] = ([idxs[i] for i in order], X[[cols[i] for i in order]])
        return out

    @staticmethod
    def _weighted_corr(a, b, w):
        """Weighted Pearson r between a and b (1-D arrays, w >= 0)."""
        w = np.asarray(w, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        sw = w.sum()
        if sw <= 0:
            return 0.0
        ma = np.sum(w * a) / sw
        mb = np.sum(w * b) / sw
        da = a - ma
        db = b - mb
        cov = np.sum(w * da * db)
        va = np.sum(w * da * da)
        vb = np.sum(w * db * db)
        denom = np.sqrt(va * vb)
        if denom == 0.0:
            return 0.0
        return float(cov / denom)

    @staticmethod
    def _wls(x, y, w):
        """Weighted least-squares fit of y = slope * x + intercept.
        Returns (slope, intercept). Solves the 2-parameter normal equations
        with diagonal weight matrix W.
        """
        w = np.asarray(w, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sw = w.sum()
        if sw <= 0:
            return 0.0, float(y.mean()) if len(y) else 0.0
        wx = w * x
        wy = w * y
        xhat = wx.sum() / sw
        yhat = wy.sum() / sw
        # slope = sum w_i (x_i - xbar)(y_i - ybar) / sum w_i (x_i - xbar)^2
        dx = x - xhat
        num = np.sum(w * dx * (y - yhat))
        den = np.sum(w * dx * dx)
        if den == 0.0:
            return 0.0, float(yhat)
        slope = float(num / den)
        intercept = float(yhat - slope * xhat)
        return slope, intercept

    def fit(self, X, y, sample_weight=None):
        channels = self._driver_lags(X)
        if not channels:
            raise ValueError(
                "LaggedLinear requires at least one of v_lag_*/bz_lag_*/pdyn_lag_* "
                "columns in X.")
        self._n_horizons = y.shape[1]
        n = len(y)
        if sample_weight is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)

        for h in range(self._n_horizons):
            y_h = y[:, h]
            # Pick the (driver, lag) with the greatest weighted |r|.
            best_r = -1.0
            best_key = None   # (base, pref, k, col)
            for base, (lag_idxs, lags_df) in channels.items():
                for col, k in zip(lags_df.columns, lag_idxs):
                    s = lags_df[col].to_numpy(dtype=float)
                    mask = np.isfinite(s) & np.isfinite(y_h)
                    if mask.sum() < 30:
                        continue
                    r = abs(self._weighted_corr(s[mask], y_h[mask], w[mask]))
                    if np.isnan(r):
                        continue
                    if r > best_r:
                        best_r = r
                        best_key = (base, DRIVER_CHANNELS[
                            [b for b, _ in DRIVER_CHANNELS].index(base)][1],
                            k, col)
            if best_key is None:
                raise ValueError(
                    f"LaggedLinear: no valid (driver, lag) pair for horizon {h}.")
            base, pref, k, col = best_key
            s = channels[base][1][col].to_numpy(dtype=float)
            mask = np.isfinite(s) & np.isfinite(y_h)
            slope, intercept = self._wls(s[mask], y_h[mask], w[mask])
            self.coefs_[h] = {
                "driver": base, "lag": k, "slope": slope, "intercept": intercept}
            print(f"  [LaggedLinear] Horizon {h}: best = {base} lag {k} steps "
                  f"(weighted r={best_r:.3f})")
        return self

    def predict(self, X):
        channels = self._driver_lags(X)
        n = len(X)
        out = np.zeros((n, self._n_horizons))
        for h in range(self._n_horizons):
            co = self.coefs_[h]
            base = co["driver"]
            k = co["lag"]
            col = None
            if base in channels:
                lag_idxs, lags_df = channels[base]
                if k in lag_idxs:
                    col = lags_df.columns[lag_idxs.index(k)]
            if col is not None:
                s = channels[base][1][col].to_numpy(dtype=float)
            else:
                s = np.zeros(n)
            out[:, h] = co["slope"] * np.nan_to_num(s) + co["intercept"]
        return out
