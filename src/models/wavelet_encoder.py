# src/models/wavelet_encoder.py
"""
Discrete Wavelet Transform driver feature encoder.

Decomposes the recent history of each solar-wind driver into sub-band energy
features that the tree / linear models can consume directly. PyWavelets is the
only third-party dependency (installed, v1.8.0).

For each driver base, the encoder produces for a lookback window [t-L:t]:
  * per-level energy: sum of squared detail coefficients at each scale
  * low-pass trend:   final approximation coefficient value
  * bandpower_ratio:  L2 energy of each detail band / total detail energy
  * spectral_entropy: Shannon entropy of the normalized bandpower vector
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pywt
    _HAS_PYWT = True
except Exception:  # pragma: no cover
    _HAS_PYWT = False

WAVELET = "sym4"
# pywt.dwt_max_level(192, pywt.Wavelet("sym4").dec_len) reports 4 (safe max).
DEFAULT_LEVEL = 4
DEFAULT_LOOKBACK = 48
DRIVER_BASES = ("v_sw", "bz_s", "pdyn", "clock_angle", "sin4_theta2",
                "sckopke", "viscous_proxy")


def bandpass_energy(window_1d, wavelet=WAVELET, level=DEFAULT_LEVEL):
    """Sub-band energy features for a 1-D driver lookback window.

    Parameters
    ----------
    window_1d : 1-D array-like, length L.
    wavelet : name passed to ``pywt.wavedec``.
    level   : decomposition level.

    Returns
    -------
    dict with keys ``energy``, ``trend``, ``bandpower_ratio``, ``spectral_entropy``.
    """
    x = np.asarray(window_1d, dtype=float).ravel()
    # Fill missing values before decomposition.
    x = np.nan_to_num(x, nan=0.0)

    min_len = 2 ** level
    if x.shape[0] < min_len:
        pad = min_len - x.shape[0]
        x = np.pad(x, (0, pad), mode="symmetric")

    coeffs = pywt.wavedec(x, wavelet, level=level)
    # coeffs = [cA, cD_level, cD_{level-1}, ..., cD_1]
    energy_j = np.array([float(np.sum(c * c)) for c in coeffs[1:]], dtype=float)
    trend = float(coeffs[0][-1])

    total = float(energy_j.sum())
    if total > 0.0:
        bandpower_ratio = (energy_j / total).astype(float)
    else:
        bandpower_ratio = np.zeros_like(energy_j)

    p = bandpower_ratio
    spectral_entropy = float(-np.sum(p * np.log2(p + 1e-12)))

    return {
        "energy": energy_j,
        "trend": trend,
        "bandpower_ratio": bandpower_ratio,
        "spectral_entropy": spectral_entropy,
    }


class WaveletEncoder:
    """Turn a driver matrix into per-row sub-band energy features.

    Stateless apart from recording which driver columns are present during
    ``fit``. ``transform`` / ``transform_rows`` operate on any DataFrame that
    contains a superset of those columns.
    """

    def __init__(self, driver_bases=DRIVER_BASES, lookback=DEFAULT_LOOKBACK,
                 wavelet=WAVELET, level=DEFAULT_LEVEL):
        self.driver_bases = tuple(driver_bases)
        self.lookback = int(lookback)
        self.wavelet = wavelet
        self.level = int(level)

    def fit(self, df, y=None):
        self.driver_bases_ = tuple(b for b in self.driver_bases if b in df.columns)
        self.n_features_ = len(self.driver_bases_) * (2 * self.level + 2)
        return self

    partial_fit = fit

    def feature_names(self):
        names = []
        for base in self.driver_bases_:
            names.append(f"{base}_trend")
            names.append(f"{base}_entropy")
            for j in range(1, self.level + 1):
                names.append(f"{base}_energy_{j}")
            for j in range(1, self.level + 1):
                names.append(f"{base}_bpr_{j}")
        return names

    def transform_rows(self, df):
        if not hasattr(self, "driver_bases_"):
            self.fit(df)

        lookback = self.lookback
        drivers = list(self.driver_bases_)
        n_drivers = len(drivers)

        vals = df[drivers].to_numpy(dtype=float)  # (N, n_drivers)
        N = vals.shape[0]
        if N <= lookback:
            return np.empty((0, self.n_features_), dtype=float), df.index[0:0]

        row_vectors = []
        for i in range(lookback, N):
            window = vals[i - lookback:i]  # (lookback, n_drivers)
            row = []
            for d in range(n_drivers):
                be = bandpass_energy(window[:, d], wavelet=self.wavelet,
                                     level=self.level)
                row.append(be["trend"])
                row.append(be["spectral_entropy"])
                row.extend(be["energy"].tolist())
                row.extend(be["bandpower_ratio"].tolist())
            row_vectors.append(np.array(row, dtype=float))

        return np.vstack(row_vectors), df.index[lookback:]

    def fit_transform(self, df, y=None):
        self.fit(df, y=y)
        return self.transform_rows(df)

    def transform(self, df):
        return self.transform_rows(df)


def augment_features_with_wavelet(df, encoder=None, lookback=DEFAULT_LOOKBACK):
    """Return df augmented with ``wavelet__*`` columns.

    Rows ``0 .. lookback-1`` are NaN in the wavelet columns.
    """
    if encoder is None:
        encoder = WaveletEncoder(lookback=lookback)
    X, idx = encoder.fit_transform(df)
    cols = [f"wavelet__{n}" for n in encoder.feature_names()]
    wdf = pd.DataFrame(X, index=idx, columns=cols)
    return df.join(wdf)


if __name__ == "__main__":
    if not _HAS_PYWT:
        raise RuntimeError("pywt required")
    np.random.seed(0)
    N = 1000
    idx = pd.date_range("2015-01-01", periods=N, freq="15min")
    df = pd.DataFrame({
        "v_sw": 400 + 100*np.sin(2*np.pi*np.arange(N)/50) + 20*np.random.randn(N),
        "bz_s": 5 + 3*np.sin(2*np.pi*np.arange(N)/30) + np.random.randn(N),
        "pdyn": 2 + 0.5*np.random.randn(N),
        "clock_angle": np.random.randn(N),
        "sin4_theta2": np.random.rand(N),
        "sckopke": 1e3 + 1e2*np.random.randn(N),
        "viscous_proxy": 10 + 2*np.random.randn(N),
    }, index=idx)
    enc = WaveletEncoder(lookback=48)
    X, idx_out = enc.fit_transform(df)
    print("X shape:", X.shape, "  expected rows:", N-48, " features:", enc.n_features_)
    print("feature_names[:12]:", enc.feature_names()[:12])
    print("idx_out[:3]:", idx_out[:3])
    # bandpass_energy on known input
    w = df["v_sw"].values[:48]
    be = bandpass_energy(w)
    print("bandpass_energy keys:", list(be.keys()))
    print("  energy shape:", be["energy"].shape, "  bpr sum:", be["bandpower_ratio"].sum(), "  entropy:", be["spectral_entropy"])
    # augment
    df_aug = augment_features_with_wavelet(df, lookback=48)
    print("augmented df shape:", df_aug.shape, "  wavelet__ cols:", len([c for c in df_aug.columns if c.startswith("wavelet__")]))
    print("NaN in first 48 rows (expected):", df_aug["wavelet__v_sw_trend"].iloc[:48].isna().all())
    print("VALID in row 48+ (expected):", df_aug["wavelet__v_sw_trend"].iloc[48:].notna().all())
    assert X.shape == (N-48, enc.n_features_), f"shape mismatch {X.shape} vs {(N-48, enc.n_features_)}"
    assert enc.n_features_ == 70, f"expected 70 features, got {enc.n_features_}"
    assert be["bandpower_ratio"].sum() > 0.99 and be["bandpower_ratio"].sum() < 1.01
    assert be["spectral_entropy"] > 0
    print("\nALL WAVELET SELF-TEST ASSERTIONS PASSED")
