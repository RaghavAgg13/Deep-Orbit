"""Forecasting model ladder (Rungs 0-6) for the ISRO space-weather pipeline.

Each model exposes the contrato used by ``main.py`` / ``src.harness``:

    model.fit(X, y, ...)            # train
    y_hat = model.predict(X)        # -> ndarray (n_samples, 3 horizons)

The three forecast horizons are fixed by ``src.config``: 30-min, 6-h, 12-h.
"""
from .persistence import Persistence
from .climatology import Climatology
from .lagged_linear import LaggedLinear
from .linear_filter import LinearFilter
from .multi_filter import MultiFilter
from .gbm import GBMForecaster
from .lstm import LSTMForecaster, to_sequences
from .tcn import TCNForecaster
from .hpo import run_hpo, make_objective, default_search_space, load_hpo_best
from .hybrid import HybridForecaster
from .stacking_meta_learner import StackingMetaLearner, SequencePanel
from .wavelet_encoder import (
    WaveletEncoder,
    bandpass_energy,
    augment_features_with_wavelet,
)

__all__ = [
    "Persistence",
    "Climatology",
    "LaggedLinear",
    "LinearFilter",
    "MultiFilter",
    "GBMForecaster",
    "LSTMForecaster",
    "TCNForecaster",
    "to_sequences",
    "run_hpo",
    "make_objective",
    "default_search_space",
    "load_hpo_best",
    "HybridForecaster",
    "StackingMetaLearner",
    "SequencePanel",
    "WaveletEncoder",
    "bandpass_energy",
    "augment_features_with_wavelet",
]
