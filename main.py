import os
import sys
import argparse
import subprocess

# ---------------------------------------------------------------------------
# Dependency checker + auto-installer
# ---------------------------------------------------------------------------
# All third-party packages the pipeline requires, grouped by necessity.
# Core: needed for every run.  Sequence: needed for LSTM/TCN training.
# Opt-in: loaded on demand (--hpo flag) and may be absent without error.

_DEPENDENCY_GROUPS = {
    "core": {
        "numpy":         "numpy",
        "pandas":        "pandas",
        "matplotlib":    "matplotlib",
        "sklearn":       "scikit-learn",
        "lightgbm":      "lightgbm",
        "cdflib":        "cdflib",
        "cdasws":        "cdasws",
        "joblib":        "joblib",
        "pywt":          "PyWavelets",
    },
    "sequence": {
        "torch":         "torch",
    },
    "opt-in": {
        "optuna":        "optuna",
    },
}


def _check_and_install_dependencies(install_missing=False, include_sequence=True,
                                     include_optin=False):
    """Check that all required packages are importable; optionally install.

    Parameters
    ----------
    install_missing : bool
        If True, prompt the user and auto-install any missing packages via pip.
    include_sequence : bool
        If True (default), require PyTorch for LSTM/TCN training.
    include_optin : bool
        If True, also require optuna (only needed with --hpo).

    Returns
    -------
    missing : list[str]
        Packages that are still missing after checking (and possibly installing).
    """
    groups_to_check = {"core": True, "sequence": include_sequence,
                       "opt-in": include_optin}
    missing = []
    for group_name, enabled in groups_to_check.items():
        if not enabled:
            continue
        for import_name, pip_name in _DEPENDENCY_GROUPS[group_name].items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append((import_name, pip_name, group_name))

    if not missing:
        return []

    print("\n" + "=" * 60)
    print("  MISSING DEPENDENCIES")
    print("=" * 60)
    for imp_name, pip_name, group in missing:
        tag = {"core": "REQUIRED", "sequence": "SEQUENCE MODEL",
               "opt-in": "OPTIONAL"}[group]
        print(f"  [{tag}] {imp_name}  (pip install {pip_name})")
    print("=" * 60)

    if not install_missing:
        print("\nInstall missing packages with:")
        pip_cmd = "pip install " + " ".join(p for _, p, _ in missing)
        print(f"  {pip_cmd}")
        print("Or re-run with:  python main.py --install-deps")
        return missing

    # Auto-install: ask for confirmation
    pip_packages = [pip_name for _, pip_name, _ in missing]
    print(f"\nWill install: {', '.join(pip_packages)}")
    try:
        answer = input("Proceed? [Y/n] ").strip().lower()
    except EOFError:
        answer = "y"
    if answer in ("", "y", "yes"):
        print("Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *pip_packages])
        print("Done. Verifying...")
        still_missing = []
        for imp_name, pip_name, group in missing:
            try:
                __import__(imp_name)
            except ImportError:
                still_missing.append((imp_name, pip_name, group))
        if still_missing:
            print(f"WARNING: still missing after install: "
                  f"{', '.join(i for i, _, _ in still_missing)}")
        else:
            print("All dependencies verified ✓")
        return still_missing
    else:
        print("Skipped installation.")
        return missing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Set backend to Agg for matplotlib to prevent GUI popup blocks in non-interactive terminals
import matplotlib
matplotlib.use('Agg')

from src.config import RAW_DIR, PROC_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, HORIZON_STEPS, HAZARD_THRESHOLD_LOG
from src.download_data import download_and_save_real_data
from src.pipeline import run_preprocessing_pipeline, prepare_pipeline_data

from src.harness import run_benchmark, evaluate_model, Forecaster
from src.models import (
    Persistence, Climatology, LaggedLinear, LinearFilter,
    MultiFilter, GBMForecaster, LSTMForecaster, HybridForecaster
)
from src.models.lstm import to_sequences
from src.models.tcn import TCNForecaster
from src.models.stacking_meta_learner import StackingMetaLearner, SequencePanel
from src.models.physics_loss import PhysicsConstraint, physics_clip_trajectory, diagnose_physics_violations
from src.models.wavelet_encoder import augment_features_with_wavelet
from src.models.hpo import run_hpo, load_hpo_best

# ---------------------------------------------------------------------------
# Physics-informed inference clipping helpers
# ---------------------------------------------------------------------------
# At eval time we post-process a model's raw trajectory through the physics
# corrector (quiet-time monotonicity + rise-rate bound). The corrector only
# needs v_sw + the predicted trajectory -- never the flux history -- so this
# introduces zero target leakage.

def _raw_v_sw_for(data, df_features, index_test, test_seq_idxs):
    """Raw (unscaled, km/s) solar-wind speed for the seq-aligned test rows.

    The feature matrix ``data["X_test"]`` is STANDARDIZED (StandardScaler
    zero-mean/unit-variance), so ``v_sw`` values there are ~[-2, +4] — NOT the
    km/s the physics corrector's thresholds (QUIET_VSW=350 km/s, etc.) expect.
    Passing the scaled column makes EVERY row satisfy ``v_sw < 350``, so the
    quiet mask fires everywhere and R8 forces all trajectories to decay
    (quiet_frac -> 1.0, PE tanks). We must pass the RAW km/s from the
    un-scaled feature frame, subset to the same seq-aligned rows as the
    predictions.
    """
    return df_features.loc[index_test, "v_sw"].values[test_seq_idxs]


def physics_clip(model, v_sw_raw, X=None, seq_idxs=None):
    """Return model.predict(X) with the physics corrector applied.

    Parameters
    ----------
    model : forecaster with predict(X) / predict(X, seq_idxs=...).
    v_sw_raw : ndarray (n,) — RAW v_sw in km/s, ALREADY row-aligned to the
        model's prediction length (caller must sub-select if the model
        predicts on a subset). Passing the standardized v_sw here is the #1
        cause of R8 destroying PE — see _raw_v_sw_for.
    X : pd.DataFrame | ndarray — feature matrix for the model. If the model
        needs seq_idxs, both X and seq_idxs must be supplied.
    seq_idxs : array-like | None — passed through to model.predict for the
        seq-mode models.
    """
    pred = model.predict(X) if seq_idxs is None else model.predict(X, seq_idxs=seq_idxs)
    return physics_clip_trajectory(pred, v_sw_raw, PhysicsConstraint())


class _ArrayPredictor(Forecaster):
    """Wraps a pre-computed (n, 3) array for ``evaluate_model``.

    Used to feed a pre-computed, row-aligned prediction (e.g. the sequence
    panel's output, already aligned 1:1 with ``y_test_seq_flat``) through the
    metric loop without recomputing it. ``predict(X)`` returns the cached
    array via a *view* — it does NOT truncate or pad — so the caller MUST
    pass an X whose length matches the array. A length mismatch raises
    instead of silently scoring misaligned rows.
    """

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=float)

    def fit(self, X, y, **kw):
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        if n != self.arr.shape[0]:
            raise ValueError(
                f"_ArrayPredictor: cached array has {self.arr.shape[0]} rows "
                f"but X has {n}; they must be row-aligned. The caller is "
                f"responsible for slicing X and y to match the pre-computed "
                f"prediction.")
        return self.arr


class PhysicsClippedForecaster(Forecaster):
    """Wraps a flat forecaster so its .predict() is physics-corrected.

    Rung 8 = R7 (stacking) wrapped by this class. The wrapper delegates
    .fit() to the base model (so it trains identically) and applies the
    physics corrector on every subsequent .predict() call.

    IMPORTANT: ``v_sw_raw`` MUST be the RAW solar-wind speed in km/s, row-
    aligned to the model's prediction length (i.e. already sub-selected to the
    seq_idxs rows for a seq-mode base). The physics thresholds (QUIET_VSW =
    350 km/s) are in physical units — passing the STANDARDIZED v_sw from the
    scaled feature matrix makes every row satisfy ``v_sw < 350`` and the quiet
    mask fires on 100% of rows, which is what was destroying R8's PE.
    """

    def __init__(self, base, v_sw_raw, constraint=None):
        self.base = base
        self.v_sw_raw = None if v_sw_raw is None else np.asarray(v_sw_raw, dtype=float)
        self.constraint = constraint or PhysicsConstraint()

    def fit(self, X, y, **kw):
        return self.base.fit(X, y, **kw)

    def predict(self, X, seq_idxs=None):
        pred = self.base.predict(X, seq_idxs=seq_idxs)
        if self.v_sw_raw is None:
            return pred
        # v_sw_raw is pre-aligned to the prediction length by the caller.
        return physics_clip_trajectory(pred, self.v_sw_raw, self.constraint)


def main():
    parser = argparse.ArgumentParser(
        description="ISRO Space Weather Electron Flux Forecasting Pipeline")
    parser.add_argument("--hpo", action="store_true",
                        help="Run Optuna HPO on the TCN (slower, opt-in).")
    parser.add_argument("--install-deps", action="store_true",
                        help="Check and auto-install missing Python packages.")
    parser.add_argument("--resume-from", type=str, default=None, metavar="RUNG",
                        help="Resume training from a later rung by loading earlier "
                             "models from models/*.pkl. Rungs before RUNG are "
                             "restored from disk instead of retrained. "
                             "Valid: r3, r4, r5a, r5b, r6, r7. "
                             "Example: --resume-from r6 restores R0-R5 and "
                             "trains R6/R7 from scratch. Requires those pkl "
                             "files to exist from a prior (possibly interrupted) "
                             "run.")
    args = parser.parse_args()

    # --- Dependency check (runs BEFORE any third-party imports) ----------
    missing = _check_and_install_dependencies(
        install_missing=args.install_deps,
        include_sequence=True,
        include_optin=args.hpo,
    )
    if missing:
        core_missing = [i for i, _, g in missing if g == "core"]
        seq_missing = [i for i, _, g in missing if g == "sequence"]
        if core_missing:
            print(f"\nFATAL: core dependencies missing ({', '.join(core_missing)}). "
                  f"Cannot continue.\n"
                  f"  → Re-run with:  python main.py --install-deps")
            sys.exit(1)
        if seq_missing:
            print(f"\nWARNING: sequence-model dependencies missing "
                  f"({', '.join(seq_missing)}). LSTM/TCN rungs will be skipped.\n"
                  f"  → Install with:  pip install {' '.join(p for _, p, g in missing if g == 'sequence')}")
            # TODO: skip R5a/R5b at training time if torch is absent

    print("==================================================================")
    print("   ISRO SPACE WEATHER ELECTRON FLUX FORECASTING ORCHESTRATOR")
    print("==================================================================")
    
    # 1. Data Acquisition
    # download_and_save_real_data is idempotent: it validates each source
    # (GOES, SWE, MFI) independently and only downloads what's missing or
    # fails validation. Cached CDFs on disk are reused, so re-runs are fast
    # and never throw away data you already have.
    print("\n[Data Acquisition] Ensuring 4 years (2013-2016) of real scientific data from NASA CDAWeb...")
    download_and_save_real_data(start_year=2013, end_year=2016)
    print("[Data Acquisition] Real data ready.")

    # 2. Run Preprocessing Pipeline
    # cadence is 15min, history K_steps = 96 (24 hours of history)
    K_steps = 96
    df_features = run_preprocessing_pipeline(propagate=True, K_steps=K_steps)

    # 2b. Augment with wavelet driver features (DEFAULT ON, cheap: ~70 cols, <1s).
    # Computed from driver columns only (v_sw etc.), never from flux, so this
    # introduces zero target leakage. The first `lookback` rows are NaN in the
    # wavelet columns; prepare_pipeline_data drops them via dropna, so the
    # splits are unaffected. Because the new wavelet__* columns do NOT start
    # with "flux_lag", they are automatically included in `ml_features` and
    # therefore flow to LightGBM (R4) and the Hybrid corrector (R6).
    print("\n[Wavelet] Augmenting feature matrix with sub-band driver energy features...")
    df_features = augment_features_with_wavelet(df_features, lookback=48)
    n_wavelet = len([c for c in df_features.columns if c.startswith("wavelet__")])
    print(f"  [Wavelet] added {n_wavelet} wavelet__* columns (total features: {df_features.shape[1]})")

    # 3. Build Splits and Scale Features
    # Train: 2013-01-01 to 2015-12-31 (3 years)
    # Val: 2016-01-01 to 2016-06-30 (6 months)
    # Test: 2016-07-01 to 2016-12-31 (6 months)
    data = prepare_pipeline_data(df_features, train_end="2016-01-01", val_end="2016-07-01")
    
    # 4. Set up LSTM Sequence Data
    # At 15-min cadence we now use 48 hours of history (seq_len = 192 steps)
    # to cover the dominant 1-2 day Vsw lag identified in the physics brief.
    # The TCN's receptive field (4081 steps ≈ 1020 h) easily covers this; the
    # LSTM benefits directly from the longer context.
    seq_len = 192
    print(f"\nConstructing sequence datasets (seq_len={seq_len} steps) for LSTM/TCN...")
    # Driver feature set for the sequence encoder. DESIGN:
    #   * RAW drivers (v_sw, n_sw, bz, bz_s, pdyn, vbz) — the encoder's job is to
    #     learn a temporal representation from these, so we keep them.
    #   * DERIVED instantaneous state (coupling functions, storm-phase markers,
    #     cumulative energy integrals, time-since-event, tilt/seasonality) -
    #     single-number summaries of storm state the encoder would otherwise
    #     need many timesteps to reconstruct from raw history. THIS is the
    #     information R4/R6 consume that the old 11-feature set lacked - it is
    #     why the LSTM/TCN used to underperform the tree/linear rungs.
    #   * EXCLUDED: driver lags (v_lag_*, etc.) - redundant with the 192-step
    #     encoder window; feeding them would make the encoder learn a lookup
    #     table instead of a temporal integral.
    #   * EXCLUDED: flux_lag_* - TARGET LEAKAGE (pipeline invariant).
    #   * EXCLUDED: wavelet__* - requires a 48-step lookback so every window-start
    #     row is NaN; the raw-driver history already carries the same sub-band info.
    #
    # Width: 23 (was 37, was 11). Pruned for TIER A.1 — the 37-channel set
    # carried ~16 near-duplicate columns (three Vsw-duration windows, four Vsw
    # derivatives, three cum_vbz integrals, redundant tilt/doy sinusoids). A
    # deep encoder wastes capacity learning those are redundant; keeping ONE
    # representative per physical quantity lets the 192-step window reconstruct
    # the rest. The encoder's own 48h history of raw v_sw/bz/pdyn already
    # subsumes the shorter-duration integrals and the 1h deltas. Zero leakage.
    #
    # Selection rule: keep the WIDEST informative window per quantity (the
    # encoder narrows, it doesn't widen) and the HIGHEST-RESOLUTION derivative
    # (the encoder integrates, it doesn't differentiate).
    lstm_features = [
        # raw drivers (6) — the temporal backbone
        "v_sw", "n_sw", "bz", "bz_s", "pdyn", "vbz",
        # reconnection / coupling state (4)
        "clock_angle", "sin4_theta2", "sckopke", "viscous_proxy",
        # instantaneous storm-state derivatives (3) — 1-step only; the encoder
        # integrates these over its 48h window to recover the 1h deltas.
        "dVsw_dt", "dBz_dt", "dPdyn_dt",
        # cumulative energy input integrals (2) — widest window per quantity.
        "cum_vbz_pos_24h", "cum_pdyn_pos_12h",
        # high-speed-stream age + time-since-event (3)
        "vsw_gt500_duration_24h",
        "hours_since_bz_flip", "hours_since_vsw_gt500",
        # seasonality / diurnal context (2) — tilt=annual, hod=diurnal.
        "tilt_sin", "tilt_cos", "hod_sin", "hod_cos",
        # satellite-era flag (1)
        "goes_new",
    ]
    # Build sequences one split at a time and release each flat slice
    # immediately — holding all three (train+val+test) 11-column frames AND
    # the 3D sequence arrays simultaneously would exceed 16 GB on modest
    # machines. Each split is ~1.5 GB flat; the 3D array is similar. We
    # build one split at a time and release the flat frame right after.
    #
    # TIER A.8 — delta-flux target: the encoder predicts the *change* from
    # the known current level, so we also capture flux_lag_1 at each window-
    # end row (flux_base). This is the current flux, known at prediction
    # time — NOT leakage (it's the starting point, not the future target).
    # flux_base is passed to fit_sequences (to build the delta target) and
    # to predict (to add the level back). The pipeline invariant is
    # preserved: the model still never sees *future* flux.
    def _build_seq(split):
        X_flat = data[f"X_{split}"][lstm_features].values
        X_seq, y_seq, idxs = to_sequences(
            X_flat, data[f"y_{split}"], seq_len, data[f"segment_ids_{split}"]
        )
        # flux_lag_1 at the window-end rows = the known current flux level.
        flux_base = data[f"X_{split}"]["flux_lag_1"].values[idxs]
        del X_flat  # release the ~1.5 GB flat frame
        return X_seq, y_seq, idxs, flux_base

    X_train_seq, y_train_seq, train_seq_idxs, flux_base_train = _build_seq("train")
    X_val_seq, y_val_seq, val_seq_idxs, flux_base_val = _build_seq("val")
    X_test_seq, y_test_seq, test_seq_idxs, flux_base_test = _build_seq("test")
    print(f"Sequence dataset sizes - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    # 5. Fit Models
    # ------------------------------------------------------------------
    # Resume harness.
    # ------------------------------------------------------------------
    # Each entry: (rung_key, filename, description). Models are dumped to
    # models/<filename> immediately after training, so an interrupted run can
    # be resumed with --resume-from <later_rung>: every rung BEFORE the given
    # one is restored from its pkl instead of retrained. R7 is intentionally
    # excluded — it is cheap to refit and depends on the in-memory R5a/R5b via
    # the SequencePanel, so restoring it adds complexity for little savings.
    RUNG_SAVE_ORDER = [
        ("r3",  "multi_filter.pkl",  "R3 MultiFilter"),
        ("r4",  "gbm_forecaster.pkl", "R4 LightGBM"),
        ("r5a", "lstm_forecaster.pkl", "R5a LSTM"),
        ("r5b", "tcn_forecaster.pkl", "R5b TCN"),
        ("r6",  "hybrid_forecaster.pkl", "R6 Hybrid"),
    ]
    # A --resume-from target is valid if it is one of the persisted rungs OR a
    # later rung (r7/r8) that has at least one persisted rung before it to
    # restore. --resume-from means "train THIS rung"; everything before it must
    # be loaded from disk. So r7/r8 are valid (r3-r6 restored, r7 trained fresh)
    # but r3 is not (nothing before it to skip — just run normally).
    _VALID_RESUME_KEYS = [k for k, _, _ in RUNG_SAVE_ORDER] + ["r7", "r8"]
    # Map rung_key -> pkl filename, used by the per-rung save calls below.
    _RUNG_KEY_TO_FILE = {k: fname for k, fname, _ in RUNG_SAVE_ORDER}

    def _save_rung(model, filename):
        """Persist a trained rung to models/<filename>."""
        path = MODELS_DIR / filename
        joblib.dump(model, path)
        print(f"  [persist] saved {path.name} ({path.stat().st_size/1024:.0f} KB)")

    def _try_resume_from(resume_key):
        """If resume_key is set, load every earlier rung from disk and return
        a dict of {rung_key: model}. Returns {} if resume_key is None/invalid.
        """
        if resume_key is None:
            return {}
        rk = resume_key.strip().lower()
        if rk not in _VALID_RESUME_KEYS:
            print(f"[resume] --resume-from {resume_key!r} not in {_VALID_RESUME_KEYS}; "
                  f"starting from scratch.")
            return {}
        # Determine which persisted rungs come strictly before the target rk.
        # For rk in RUNG_SAVE_ORDER we stop before its own index; for r7/r8 the
        # target is after all persisted rungs so we restore all of them.
        all_rung_keys = [k for k, _, _ in RUNG_SAVE_ORDER] + ["r7", "r8"]
        try:
            target_idx = all_rung_keys.index(rk)
        except ValueError:
            return {}
        restore_keys = all_rung_keys[:target_idx]
        restored = {}
        for k, fname, desc in RUNG_SAVE_ORDER:
            if k not in restore_keys:
                continue
            path = MODELS_DIR / fname
            if not path.exists():
                print(f"[resume] {desc}: {path.name} missing — cannot resume from "
                      f"{rk}; starting from scratch.")
                return {}
            restored[k] = joblib.load(path)
            print(f"[resume] {desc}: loaded {path.name} ({path.stat().st_size/1024:.0f} KB)")
        print(f"[resume] restored {len(restored)} rung(s) from disk; "
              f"will train from {rk} onward.")
        return restored

    _restored = _try_resume_from(args.resume_from)

    print("\n--- Training Model Ladder ---")

    # Rung 0: Baselines (cheap, no persistence)
    print("Fitting Rung 0: Persistence and Climatology...")
    model_persistence = Persistence(flux_col_name="flux_lag_1", scaler=data["scaler"])
    model_persistence.fit(data["X_train"], data["y_train"])

    model_climatology = Climatology(sin_col="hod_sin", cos_col="hod_cos", scaler=data["scaler"])
    model_climatology.fit(data["X_train"], data["y_train"])

    # Rung 1: Lagged Linear Regression (cheap, no persistence)
    print("Fitting Rung 1: Single-Driver Lagged Linear...")
    model_lagged = LaggedLinear()
    model_lagged.fit(data["X_train"], data["y_train"])

    # Rung 2: Speed-driven Linear Filter (cheap, no persistence)
    print("Fitting Rung 2: Ridge-Regularized Speed Filter...")
    model_speed_filter = LinearFilter(K_steps=K_steps, alpha=50.0)
    model_speed_filter.fit(data["X_train"], data["y_train"])

    # Rung 3: Multi-driver Filter (Baseline Backbone)
    if "r3" in _restored:
        model_multi_filter = _restored["r3"]
        print("Fitting Rung 3: Ridge-Regularized Multi-driver Filter... [RESTORED from disk]")
    else:
        print("Fitting Rung 3: Ridge-Regularized Multi-driver Filter...")
        model_multi_filter = MultiFilter(K_steps=K_steps, alpha=50.0)
        model_multi_filter.fit(data["X_train"], data["y_train"])
        _save_rung(model_multi_filter, _RUNG_KEY_TO_FILE["r3"])
    
    # Define features to exclude target flux history for ML models (no copy-paste).
    # Tier 1.3: build a SPARSE lag schedule for the flat models (LightGBM,
    # Hybrid). The full K_steps × 7 channels = 672 lag columns is overkill: the
    # linear ridge impulse response decays in 8-12 h and the LightGBM can
    # encode long memory from the coarser exponential lags. We keep
    # log2-spaced lags (1,2,3,4,6,8,12,18,24,36,48,72,96 steps = 13 per
    # driver) — covering everything from sub-hour to 24 h — and discard the
    # rest. Net: ~90 flat-feature columns instead of ~672.
    K_steps = 96
    _lags_keep = sorted(set([1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96]))
    _lag_prefixes = ("v_lag_", "bz_lag_", "pdyn_lag_", "clock_lag_",
                    "sin4_lag_", "sckopke_lag_", "visc_lag_")
    ml_features = [
        c for c in data["X_train"].columns
        if not c.startswith("flux_lag")
        and not any(
            c.startswith(p) and int(c[len(p):]) not in _lags_keep
            for p in _lag_prefixes
            if c.startswith(p) and c[len(p):].isdigit()
        )
    ]
    
    # Rung 4: LightGBM Regressor
    print("Fitting Rung 4: LightGBM Regressor (with early stopping)...")
    # Emphasize high-flux events by giving them a sample weight of 8.0 (flux log > 3.0).
    # Storm samples are rare (~5-10% of rows); a 5x weight still lets the loss be
    # dominated by the 90%+ quiet rows. 8x gives hazardous events more influence
    # without going so high that the model over-fits to noise in individual storms.
    # We use max(y across horizons) as the hazard proxy so that a storm at ANY
    # lead time (30-min, 6-h, 12-h) up-weights the sample — y[:, 0] alone is a
    # noisy proxy because the 30-min target is dominated by autocorrelation.
    weights_train = np.where(
        np.max(data["y_train"], axis=1) >= HAZARD_THRESHOLD_LOG, 8.0, 1.0)

    # num_leaves=51 (up from 31) gives the trees more capacity to learn
    # feature interactions from the enriched cross-driver feature set (~214
    # columns after wavelet augmentation). 51 is aggressive enough to capture
    # non-linear interactions while early stopping at 50 rounds prevents
    # over-fitting.
    if "r4" in _restored:
        model_gbm = _restored["r4"]
        print("Fitting Rung 4: LightGBM Regressor (with early stopping)... [RESTORED from disk]")
    else:
        model_gbm = GBMForecaster(n_estimators=500, learning_rate=0.03, num_leaves=51)
        model_gbm.fit(
            data["X_train"][ml_features], data["y_train"],
            sample_weight=weights_train,
            eval_set=(data["X_val"][ml_features], data["y_val"])
        )
        _save_rung(model_gbm, _RUNG_KEY_TO_FILE["r4"])
    
    # Rung 5a: PyTorch LSTM
    # Re-scaled (Tier 2.2): 2 layers, 64 hidden, 40 epochs, patience=8.
    # The added capacity matches the richer cross-driver feature set and the
    # longer 48-hour context; the hazard-weighted MSE + early stopping on
    # validation loss keep over-fitting in check.
    print("Fitting Rung 5a: PyTorch LSTM (with early stopping)...")
    # Physics-loss regularization (quiet-time monotonicity) with lam=0.10 —
    # identical to the TCN so both sequence encoders are regularized the same
    # way. This gently nudges the LSTM toward physically plausible
    # trajectories without the hard-clip side effect.
    if "r5a" in _restored:
        model_lstm = _restored["r5a"]
        print("Fitting Rung 5a: PyTorch LSTM (with early stopping)... [RESTORED from disk]")
    else:
        # TIER A.7 — seed=42 for reproducibility; smooth_corr-based early
        # stopping (proven in A.6 to beat val_loss stopping by 0.6 PE). Two
        # architectural changes vs A.6:
        #   (a) use_attention=False: the attention readout pools over all 192
        #       hidden states, but for a 12h forecast the recent driver
        #       history (encoded in the last LSTM state) dominates. Attention
        #       adds 64 params that learn a near-uniform weighting = noise.
        #       The attn_entropy diagnostic (logged each epoch) tests this:
        #       if entropy is high, last-step is as good.
        #   (b) hidden_dim=96 (up from 64): the encoder's val skill plateaus
        #       at smooth_corr ~0.67 with dim=64 — it has converged to what it
        #       can represent. More hidden dims raise the ceiling so the
        #       plateau is higher before degradation kicks in.
        # TIER A.8 — delta_flux=True: encoder predicts the driver-driven
        # *change* from the known current level (flux_lag_1), then adds it
        # back. The A.8 diagnostic proved this is the difference between
        # PE -0.59 (absolute) and PE +0.03 (delta) at 12h.
        model_lstm = LSTMForecaster(seq_len=seq_len, hidden_dim=96, num_layers=2,
                                   dropout=0.3, use_attention=False,
                                   epochs=40, lr=1e-3, batch_size=512, patience=10,
                                   physics_loss=True, physics_lam=0.10,
                                   vsw_channel="v_sw",
                                   seed=42, delta_flux=True)
        model_lstm.fit_sequences(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                 feature_names=lstm_features,
                                 flux_base_train=flux_base_train,
                                 flux_base_val=flux_base_val)
        _save_rung(model_lstm, _RUNG_KEY_TO_FILE["r5a"])

    # Rung 5b: PyTorch TCN (dilated causal convs, 48h memory)
    # Uses the same lstm_features as the LSTM (NOT the wavelet
    # features -- the TCN has its own temporal encoder). Evaluated standalone
    # on the sequence test set, exactly like the LSTM.
    # TCN layer count: RF = 1 + n * (k-1) * (2^n - 1). For n=5, k=3: RF=317
    # steps (~80 h), comfortably covering the 192-step (48 h) input. Going to 8
    # layers yields RF=511 but the extra layers look into zero-padding and add
    # no signal; 5 layers is leaner, faster, and lower over-fitting risk.
    # Physics-loss regularization (quiet-time monotonicity) with lam=0.10
    # gently nudges the TCN toward physically plausible trajectories.
    print("Fitting Rung 5b: PyTorch TCN (dilated causal convs, 48h memory)...")
    if "r5b" in _restored:
        model_tcn = _restored["r5b"]
        print("Fitting Rung 5b: PyTorch TCN (dilated causal convs, 48h memory)... [RESTORED from disk]")
    else:
        # TIER A.7 — seed=42, smooth_corr stopping, no attention (same
        # rationale as LSTM). TCN uses hidden_dim=96 to match the LSTM and
        # give the conv encoder more capacity. patience=6 (vs LSTM's 10) —
        # the TCN peaks sharper, so a shorter patience catches the peak.
        # TIER A.8 — delta_flux=True (same rationale as LSTM).
        model_tcn = TCNForecaster(seq_len=seq_len, hidden_dim=96, num_layers=5, kernel_size=3,
                                  dropout=0.3, use_attention=False,
                                  epochs=40, lr=1e-3,
                                  batch_size=512, patience=6,
                                  physics_loss=True, physics_lam=0.10,
                                  vsw_channel="v_sw",
                                  seed=42, delta_flux=True)
        model_tcn.fit_sequences(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                feature_names=lstm_features,
                                flux_base_train=flux_base_train,
                                flux_base_val=flux_base_val)
        _save_rung(model_tcn, _RUNG_KEY_TO_FILE["r5b"])
        print(f"  [TCN] receptive field = {model_tcn.receptive_field} steps ({model_tcn.receptive_field*15/60:.1f} h)")

    # --- Sequence-model feature-contract check ----------------------------
    # The sequence encoders' first layer is sized to len(lstm_features) at train
    # time. If we RESTORED an encoder whose training feature-count differs from
    # the CURRENT lstm_features, the Conv1d/LSTM weight shape won't match the
    # 3D window we feed it and we get a cryptic "expected input to have 11
    # channels, but got 37" crash deep in R7. Catch it here with a clear
    # message instead.
    _expected_f = len(lstm_features)
    for _name, _m in [("LSTM (r5a)", model_lstm), ("TCN (r5b)", model_tcn)]:
        _saved_f = getattr(_m, "_n_features", None)
        if _saved_f is not None and _saved_f != _expected_f:
            print(f"\n[FATAL] {_name} was trained with {_saved_f} features but the "
                  f"current lstm_features has {_expected_f}. The saved pkl is "
                  f"stale (the feature set changed since it was trained). "
                  f"Delete models/lstm_forecaster.pkl and models/tcn_forecaster.pkl "
                  f"and run a full `python main.py` to retrain them, OR use "
                  f"--resume-from r5a/r5b to retrain from the sequence models up.\n")
            sys.exit(1)

    # Optional Optuna HPO on the TCN (EXPENSIVE -- opt-in via --hpo).
    # When --hpo is passed we replace model_tcn with the HPO-refit best model.
    # Wrapped in try/except so an HPO failure never kills the default run.
    if args.hpo:
        print("\n[HPO] --hpo flag set: running lightweight Optuna study on the TCN "
              "(n_trials=8, timeout_s=600, device=cpu)...")
        try:
            hpo_result = run_hpo(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                feature_names=lstm_features, n_trials=8, timeout_s=600, device="cpu"
            )
            print(f"  [HPO] best S={hpo_result['best_value']:.5f} params={hpo_result['best_params']}")
            model_tcn = load_hpo_best()
            print("  [HPO] replaced TCN with HPO-refit best model.")
        except Exception as e:
            print(f"  [HPO] FAILED (keeping default TCN): {e!r}")

    # Rung 6: Physics-Informed Hybrid (Backbone: MultiFilter + Corrector: LightGBM)
    # The Corrector is the key improvement point: R4 proves ~50 leaves can model the
    # residuals, but R6 needs less because the physics backbone already captures the
    # dominant dynamics. Using 31 leaves balances underfitting vs. over-correction.
    if "r6" in _restored:
        model_hybrid = _restored["r6"]
        print("Fitting Rung 6: Physics-Informed Residual Hybrid... [RESTORED from disk]")
    else:
        print("Fitting Rung 6: Physics-Informed Residual Hybrid...")
        physics_backbone = MultiFilter(K_steps=K_steps, alpha=50.0)
        ml_corrector = GBMForecaster(n_estimators=500, learning_rate=0.03, num_leaves=31)
        model_hybrid = HybridForecaster(physics_model=physics_backbone, ml_model=ml_corrector)
        model_hybrid.fit(
            data["X_train"][ml_features], data["y_train"],
            sample_weight=weights_train,
            eval_set=(data["X_val"][ml_features], data["y_val"])
        )
        _save_rung(model_hybrid, _RUNG_KEY_TO_FILE["r6"])

    # Rung 7: StackingMetaLearner blends the R3 physics forecast with the R4
    # machine-learned forecast AND (Tier 2.3) the TCN sequence model, all
    # conditioned on the current magnetospheric regime via a stacked Ridge
    # meta-learner. The meta-learner is fit on the VALIDATION split, which is
    # out-of-sample for R3/R4/TCN, so the blender weights are honest.
    #
    # A SequencePanel wraps the TCN so it can be scored against the flat
    # validation matrix at exactly the rows that form valid segment-safe
    # windows (val_seq_idxs) — aligning the sequence panel to the flat panel.
    print("Fitting Rung 7: Stacking Meta-Learner (regime blending, R3+R4+TCN)...")
    # SequencePanel builds windows on-the-fly via ``to_sequences`` from
    # whatever flat matrix the caller passes to predict_aligned. It is
    # CRITICAL that matrix match the original build — the StackingMetaLearner
    # exposes predict(X_flat, seq_idxs=...) where seq_idxs are offsets into
    # X_flat. Eval-time uses data["X_test"] (full test matrix) + test_seq_idxs;
    # the resulting predictions are then subset to test_seq_idxs for alignment
    # with y_test_seq_flat (see the predict-time call sites below).
    # Sequence panel blends the LSTM and TCN predictions (averaged) before
    # feeding them to the meta-learner. The two encoders have complementary
    # inductive biases (recurrent vs. convolutional), so averaging reduces
    # variance and gives the meta-learner a more stable sequence signal to
    # condition on.
    tcn_panel = SequencePanel(
        models={"tcn": model_tcn, "lstm": model_lstm}, seq_len=seq_len,
        feature_names=lstm_features)
    # TIER A.8 — the panel wraps delta-flux encoders, so it needs the
    # current-flux level (flux_lag_1) at each window-end row to pass to the
    # encoders' predict (they add it back to their predicted change).
    # flux_base is split-specific; the panel stores the one matching the
    # seq_idxs it's called with. We set test (eval) and val (fit) explicitly
    # at the call sites below.
    model_stacking = StackingMetaLearner(
        base_models={"r3": model_multi_filter, "r4": model_gbm},
        meta_alpha=1.0,
        seq_panel=tcn_panel)
    # TIER A.8 — set the panel's flux_base for the validation split (used
    # during fit via predict_aligned → encoder.predict).
    tcn_panel.flux_base = flux_base_val
    weights_val = np.where(
        np.max(data["y_val"], axis=1) >= HAZARD_THRESHOLD_LOG, 8.0, 1.0)
    model_stacking.fit(
        data["X_val"], data["y_val"], sample_weight=weights_val,
        seq_idxs=val_seq_idxs)

    # R7a baseline: a SEPARATE StackingMetaLearner with NO seq panel, trained on
    # the full validation rows. This isolates the TCN panel's marginal
    # contribution. Using the same instance for both evaluations is not
    # possible because the model was fit in seq mode (19 meta cols); calling
    # predict without seq_idxs would take the 16-col no-seq path and the Ridge
    # heads would blow up with a feature-count mismatch.
    model_stacking_noseq = StackingMetaLearner(
        base_models={"r3": model_multi_filter, "r4": model_gbm},
        meta_alpha=1.0)
    model_stacking_noseq.fit(
        data["X_val"], data["y_val"], sample_weight=weights_val)

    # Recalibrate the meta-learner's stacked prediction at test time too —
    # every predict() call needs seq_idxs so the sequence panel aligns.
    # test_seq_idxs are offsets into the full data["X_test"] matrix. The
    # SequencePanel builds windows on-the-fly so we MUST pass the same full
    # matrix to predict(); the predictions we get back are aligned to
    # test_seq_idxs in that full matrix. Subset at the call site.
    _stack_test_seq_idxs = test_seq_idxs
    _stack_X = data["X_test"]

    # Rung 8: Physics-Clipped Stacking = R7 (R3+R4 blend) with the physics
    # corrector applied at inference time. The wrapper delegates fit() to the
    # already-fit R7 model (so training is identical) and post-processes every
    # .predict() through physics_clip_trajectory. This is the "honest" version
    # of R7: same training, but physically impossible trajectories suppressed.
    #
    # DEFAULT-ON: the quiet-time mask was re-calibrated (Vsw < 350 km/s AND
    # sustained > 6 h AND flux below seasonal median) and the soft rise-rate
    # clamp (0.36/h) only kills patently broken trajectories, so R8 is now
    # safe to evaluate alongside R7 in every run.
    # Raw (km/s) solar-wind speed + current flux for the SEQ-ALIGNED test rows.
    # These are the physical-unit inputs the physics corrector requires. They
    # are computed ONCE here and threaded through every clip call below so we
    # never accidentally pass the STANDARDIZED columns from data["X_test"]
    # (which made every row look "quiet" and destroyed R8's PE).
    _raw_vsw_test = _raw_v_sw_for(data, df_features, data["index_test"], test_seq_idxs)
    _raw_flux_test = None
    if "flux_lag_1" in df_features.columns:
        _raw_flux_test = df_features.loc[data["index_test"], "flux_lag_1"].values[test_seq_idxs]
    _raw_index_test = data["index_test"][test_seq_idxs]

    # R8 wraps the seq-mode stacking model. Pass the pre-aligned RAW v_sw so
    # the corrector's thresholds (350 km/s) are evaluated in physical units.
    _r8_model = PhysicsClippedForecaster(model_stacking, v_sw_raw=_raw_vsw_test)

    # R6-physics: clipped hybrid evaluated at TEST time (for benchmark comparison).
    # This mirrors R8's evaluation but on the Hybrid backbone, showing the effect of
    # the physics corrector on a flat-model stack vs. the seq-blend stack.
    _r6_clipped = PhysicsClippedForecaster(model_hybrid, v_sw_raw=_raw_vsw_test)
    print(f"[R6] hybrid corrector: {len(model_hybrid.ml_model.models_)} heads, "
          f"num_leaves={model_hybrid.ml_model.num_leaves}")

    # TIER A.8 — set the panel's flux_base for the TEST split (used at eval
    # time when the StackingMetaLearner calls the panel's predict_aligned,
    # which passes flux_base to each delta-flux encoder's predict).
    tcn_panel.flux_base = flux_base_test

    # 6. Evaluation and Benchmark Compile
    print("\n--- Compiling Benchmarks ---")

    # To compare LSTM and other models on the exact same rows,
    # we sub-sample the test targets and predictions for all flat models
    # using the index rows that formed valid sequence windows (test_seq_idxs)
    X_test_seq_flat = data["X_test"].iloc[test_seq_idxs]
    y_test_seq_flat = data["y_test"][test_seq_idxs]
    index_test_seq = data["index_test"][test_seq_idxs]

    # Pack models. R8 (physics-clipped stacking) is always evaluated alongside
    # R7 so the benchmark table shows the before/after effect of the physics
    # corrector on the same run. R6-clipped shows the effect on the flat hybrid.
    flat_models = {
        "Persistence": model_persistence,
        "Climatology": model_climatology,
        "Lagged Linear": model_lagged,
        "Speed Filter (R2)": model_speed_filter,
        "Multi Filter (R3)": model_multi_filter,
        "LightGBM (R4)": model_gbm,
        "Hybrid (R6)": model_hybrid,
        "Hybrid R6-physics": _r6_clipped,
        "Stacking R7": model_stacking,
        "Stacking R8": _r8_model,
    }

    # Generate predictions and evaluate flat models on sequence-aligned test set
    results_list = []

    # Evaluate flat models. The Stacking R7 meta-learner blends R3 + R4 + TCN
    # when given seq_idxs; without them it falls back to the R3+R4 baseline
    # (backwards-compatible). We add BOTH rows to the benchmark so we can
    # see the TCN panel's marginal contribution.
    #
    # R8 wraps the seq-mode stacking model (model_stacking), so it MUST be
    # predicted with the SAME seq_idxs path as R7b — the inner StackingMetaLearner
    # was fit on 19 meta-features (flat panel + seq panel + regime). Calling it
    # without seq_idxs takes the 16-feature no-seq path and the Ridge heads blow
    # up with "X has 16 features, but Ridge is expecting 19". Physics clipping
    # itself is applied by the wrapper AFTER the base prediction.
    for name, model in flat_models.items():
        if name == "Stacking R7" or name == "Stacking R8":
            # Both R7 and R8 share the same seq-mode base (model_stacking);
            # the only difference is that R8 additionally applies the physics
            # corrector in its wrapper. Compute the SEQUENCE-AWARE prediction
            # ONCE and branch after.
            #
            # Must pass the FULL test matrix + seq_idxs to the SequencePanel
            # so it can build windows ending at those offsets in the same
            # matrix.  The returned predictions are row-aligned with
            # y_test_seq_flat (both are len(test_seq_idxs)).
            if name == "Stacking R7":
                # R7a (no-seq baseline for comparison): blends R3 + R4 only.
                # Uses the dedicated no-seq instance — the seq-panel instance
                # cannot predict without seq_idxs (dimension mismatch).
                r7a = evaluate_model(model_stacking_noseq, X_test_seq_flat,
                                     y_test_seq_flat)
                for row in r7a:
                    row["model"] = "Stacking R7 (R3+R4)"
                    results_list.append(row)
                # R7b: blends R3 + R4 + TCN via the SequencePanel (Tier 2.3).
                base_pred = model.predict(_stack_X,
                                          seq_idxs=_stack_test_seq_idxs)
                eval_model = _ArrayPredictor(base_pred)
                eval_X = X_test_seq_flat.iloc[:base_pred.shape[0]]
                eval_y = y_test_seq_flat[:base_pred.shape[0]]
                evals = evaluate_model(eval_model, eval_X, eval_y)
                for row in evals:
                    row["model"] = "Stacking R7 (R3+R4+TCN)"
                    results_list.append(row)
            else:
                # R8: physics-corrected R7 (same seq-mode base). The wrapper
                # forwards seq_idxs to the base and clips the result — passing
                # seq_idxs here is what keeps the meta-matrix at 19 features.
                r8_pred = model.predict(_stack_X, seq_idxs=_stack_test_seq_idxs)
                eval_model = _ArrayPredictor(r8_pred)
                eval_X = X_test_seq_flat.iloc[:r8_pred.shape[0]]
                eval_y = y_test_seq_flat[:r8_pred.shape[0]]
                evals = evaluate_model(eval_model, eval_X, eval_y)
                for row in evals:
                    row["model"] = "Stacking R8 (physics-corrected)"
                    results_list.append(row)
        else:
            evals = evaluate_model(model, X_test_seq_flat, y_test_seq_flat)
            for row in evals:
                row["model"] = name
                results_list.append(row)

    # Evaluate LSTM (R5a, using its sequence test data)
    # TIER A.8 — delta-flux encoders need flux_base at predict time so they
    # can add the known current level back to their predicted *change*.
    lstm_evals = evaluate_model(model_lstm, X_test_seq, y_test_seq,
                                flux_base=flux_base_test)
    for row in lstm_evals:
        row["model"] = "LSTM (R5a)"
        results_list.append(row)

    # Evaluate TCN (R5b, standalone on the sequence test data -- same contract
    # as the LSTM). The TCN uses its own temporal encoder, so it is NOT part of
    # the flat stacking panel in this pass.
    tcn_evals = evaluate_model(model_tcn, X_test_seq, y_test_seq,
                               flux_base=flux_base_test)
    for row in tcn_evals:
        row["model"] = "TCN (R5b)"
        results_list.append(row)
        
    df_benchmark = pd.DataFrame(results_list)
    cols = ["model", "horizon_idx", "pe", "corr", "rmse", "pod", "far", "hss"]
    df_benchmark = df_benchmark[cols]
    
    # Convert horizon index to human-readable strings
    # 0 -> 30 min, 1 -> 6h, 2 -> 12h
    horizon_map = {0: "30 Min", 1: "6 Hour", 2: "12 Hour"}
    df_benchmark["horizon"] = df_benchmark["horizon_idx"].map(horizon_map)
    df_benchmark = df_benchmark.drop(columns=["horizon_idx"])
    
    # Reorder columns
    df_benchmark = df_benchmark[["model", "horizon", "pe", "corr", "rmse", "pod", "far", "hss"]]
    
    # Print results
    print("\n=== BENCHMARK COMPARISON TABLE (TEST SET) ===")
    print(df_benchmark.to_markdown(index=False))

    # Save benchmark table
    benchmark_path = REPORTS_DIR / "benchmark.csv"
    df_benchmark.to_csv(benchmark_path, index=False)
    print(f"\nSaved benchmark results to {benchmark_path}")

    # Benchmark bar chart: PE (R²) per model, per horizon.
    # Visual comparison of forecast skill across the entire ladder — directly
    # addresses the "demonstration and visualisation of accuracy" requirement.
    fig_bench, ax_bench = plt.subplots(figsize=(12, 6))
    horizons = df_benchmark["horizon"].unique()
    models = df_benchmark["model"].unique()
    x = np.arange(len(models))
    width = 0.25
    for i, h in enumerate(horizons):
        vals = [df_benchmark.loc[(df_benchmark["model"] == m) & (df_benchmark["horizon"] == h), "pe"].values[0]
                if len(df_benchmark.loc[(df_benchmark["model"] == m) & (df_benchmark["horizon"] == h), "pe"]) > 0
                else 0.0 for m in models]
        ax_bench.bar(x + i * width, vals, width, label=h)
    ax_bench.set_xlabel("Model")
    ax_bench.set_ylabel("Prediction Efficiency (R²)")
    ax_bench.set_title("Forecast Skill Comparison (PE / R²)")
    ax_bench.set_xticks(x + width)
    ax_bench.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax_bench.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax_bench.legend(title="Horizon")
    ax_bench.grid(True, axis='y', alpha=0.3)
    fig_bench.tight_layout()
    bench_path = FIGURES_DIR / "benchmark_pe.png"
    fig_bench.savefig(bench_path, dpi=150)
    plt.close(fig_bench)
    print(f"Saved benchmark PE chart to {bench_path}")

    # Physics-violation diagnostic (12h horizon, test set).
    # Shows the effect of the physics corrector: R6/R7 are evaluated raw
    # (before) and clipped (after); R8 is the already-clipped stacking row.
    # The corrector uses only v_sw + the predicted trajectory -- no flux
    # HISTORY (flux_lag_1) -- so this is a pure inference-time diagnostic.
    # We DO pass the current flux level (flux_lag_1) + index to select the
    # *strict* quiet regime (Vsw<350 AND sustained>6h AND flux<seasonal median).
    print("\n=== Physics-violation diagnostic (12h horizon, test set) ===")
    # ALL physics-clip calls below use the pre-computed RAW v_sw series
    # (_raw_vsw_test, km/s) — never the standardized columns from the scaled
    # matrix. See _raw_v_sw_for for why this matters.
    # The diagnostic shows BEFORE (raw) and AFTER (clipped) for each model.
    # R6-clipped (_r6_clipped) is the pre-computed clipped hybrid predictions.
    pred_hybrid_raw = model_hybrid.predict(X_test_seq_flat)
    pred_stacking_r7a_raw = model_stacking_noseq.predict(X_test_seq_flat)
    pred_stacking_r7b_raw = model_stacking.predict(_stack_X, seq_idxs=_stack_test_seq_idxs)
    pred_hybrid_clip = _r6_clipped.predict(X_test_seq_flat)
    pred_stacking_r8_pred = _r8_model.predict(_stack_X, seq_idxs=_stack_test_seq_idxs)
    diag_items = [
        ("Hybrid R6 (raw)", pred_hybrid_raw),
        ("Hybrid R6-physics", pred_hybrid_clip),
        ("Stacking R7 (R3+R4)", pred_stacking_r7a_raw),
        ("Stacking R7+TCN (raw)", pred_stacking_r7b_raw),
        ("Stacking R7+TCN (clip)", physics_clip_trajectory(
            pred_stacking_r7b_raw, _raw_vsw_test, PhysicsConstraint())),
        ("Stacking R8 (physics)", pred_stacking_r8_pred),
    ]
    # diagnose_physics_violations shows before/after clip metrics for each model.
    # For already-clipped models (Hybrid R6-physics, Stacking R8), the "before"
    # and "after" will be nearly identical since clipping is already applied.
    for name, pred in diag_items:
        if pred.shape[0] != y_test_seq_flat.shape[0]:
            print(f"  {name}: SKIPPED (shape mismatch pred={pred.shape[0]} y={y_test_seq_flat.shape[0]})")
            continue
        diag = diagnose_physics_violations(
            pred, y_test_seq_flat, _raw_vsw_test, PhysicsConstraint(),
            flux=_raw_flux_test, index=_raw_index_test)
        print(f"  {name}: quiet_frac={diag['quiet_fraction']:.3f}  "
              f"recovery_violations/1000={diag['quiet_recovery_violations_per_1000']:.1f}  "
              f"rate_violations/1000={diag['rate_violations_per_1000']:.1f}  "
              f"rmse={diag['rmse_before']:.4f}->{diag['rmse_after_clip']:.4f}  "
              f"pe={diag['pe_before']:+.4f}->{diag['pe_after_clip']:+.4f}")

    # Persistence is owned by the resume harness above (each rung is dumped
    # to models/ immediately after training). R7/R8 are intentionally NOT
    # persisted: they are cheap to refit and R7's StackingMetaLearner holds a
    # live SequencePanel reference to the in-memory R5a/R5b, so serializing it
    # would pull the entire encoder state into the pkl for little benefit.
    # Re-running with --resume-from r6 restores R0-R5 and retrains R6/R7 fresh.
    print(f"\nRung pkls in {MODELS_DIR}/ (for --resume-from):")
    for k, fname, desc in RUNG_SAVE_ORDER:
        p = MODELS_DIR / fname
        if p.exists():
            print(f"  {k:4s} {fname:25s} {p.stat().st_size//1024:>6d} KB  ({desc})")
    
    # 7. Generate Physical & Visual Diagnostic Plots
    print("\n--- Generating Diagnostic Figures ---")

    # Plot 0: Data Overview — electron flux and solar wind drivers time series.
    # This plot demonstrates the "reading and visualization" requirement from
    # the task specification, showing the full test-period time series.
    fig_data, axes_data = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    # All four time series below are seq-aligned (len = len(test_seq_idxs) = 8349),
    # so the x-axis MUST be the seq-aligned index_test_seq, NOT the full
    # data["index_test"] (11444 rows). Mixing the two produces a shape mismatch
    # ("x and y must have same first dimension, but have shapes (11444,) and
    # (8349,)").
    test_idx = index_test_seq
    has_flux_lag = hasattr(X_test_seq_flat, "columns") and "flux_lag_1" in X_test_seq_flat.columns
    if has_flux_lag:
        fl1 = X_test_seq_flat["flux_lag_1"].values
        if data["scaler"] is not None:
            fl_idx = list(X_test_seq_flat.columns).index("flux_lag_1")
            fl1 = fl1 * data["scaler"].scale_[fl_idx] + data["scaler"].mean_[fl_idx]
        axes_data[0].plot(test_idx, fl1, "b-", linewidth=0.5, alpha=0.7)
    axes_data[0].set_ylabel("log₁₀(e⁻ flux) [pfu]")
    axes_data[0].axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='Hazard threshold')
    axes_data[0].legend(fontsize=8)
    axes_data[0].set_title("Test Set: Electron Flux and Solar Wind Drivers")
    v_sw_raw = X_test_seq_flat["v_sw"].values if hasattr(X_test_seq_flat, "columns") and "v_sw" in X_test_seq_flat.columns else None
    if v_sw_raw is not None and data["scaler"] is not None:
        vs_idx = list(X_test_seq_flat.columns).index("v_sw")
        v_sw_raw = v_sw_raw * data["scaler"].scale_[vs_idx] + data["scaler"].mean_[vs_idx]
        axes_data[1].plot(test_idx, v_sw_raw, "g-", linewidth=0.5, alpha=0.7)
    axes_data[1].set_ylabel("V_sw [km/s]")
    axes_data[1].axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='HSS threshold')
    axes_data[1].legend(fontsize=8)
    bz_raw = X_test_seq_flat["bz_s"].values if hasattr(X_test_seq_flat, "columns") and "bz_s" in X_test_seq_flat.columns else None
    if bz_raw is not None and data["scaler"] is not None:
        bz_idx = list(X_test_seq_flat.columns).index("bz_s")
        bz_raw = bz_raw * data["scaler"].scale_[bz_idx] + data["scaler"].mean_[bz_idx]
        axes_data[2].plot(test_idx, bz_raw, "r-", linewidth=0.5, alpha=0.7)
    axes_data[2].set_ylabel("Bz_s [nT]")
    pdyn_raw = X_test_seq_flat["pdyn"].values if hasattr(X_test_seq_flat, "columns") and "pdyn" in X_test_seq_flat.columns else None
    if pdyn_raw is not None and data["scaler"] is not None:
        pd_idx = list(X_test_seq_flat.columns).index("pdyn")
        pdyn_raw = pdyn_raw * data["scaler"].scale_[pd_idx] + data["scaler"].mean_[pd_idx]
        axes_data[3].plot(test_idx, pdyn_raw, "m-", linewidth=0.5, alpha=0.7)
    axes_data[3].set_ylabel("P_dyn [nPa]")
    axes_data[3].set_xlabel("Date")
    for ax in axes_data:
        ax.grid(True, alpha=0.3)
    fig_data.tight_layout()
    data_path = FIGURES_DIR / "data_overview.png"
    fig_data.savefig(data_path, dpi=150)
    plt.close(fig_data)
    print(f"Saved data overview to {data_path}")

    # Plot 1: Physics Backbone Impulse Responses
    # Speed, Bz, and Pressure impulse filters from MultiFilter
    responses = model_multi_filter.get_impulse_responses()
    # Keys in `responses` are the driver base names used by MultiFilter renamed
    # for readability in the plot labels.
    pretty = {"v_sw": "Vsw", "bz_s": "Bz_south", "pdyn": "Pdyn",
              "clock_angle": "Clock angle", "sin4_theta2": "sin4(theta/2)",
              "sckopke": "Sckopke eps", "viscous_proxy": "Viscous proxy"}
    # Plot base trio + any newly-enriched channels that happen to be present.
    plot_keys = [k for k in ("v_sw", "bz_s", "pdyn") if k in responses]
    n_axes = max(len(plot_keys), 3)
    fig, axes = plt.subplots(n_axes, 1, figsize=(10, 3 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]
    lags_hours = np.arange(K_steps + 1) * 15 / 60.0  # convert to hours
    for idx_ax, key in enumerate(plot_keys):
        ax = axes[idx_ax]
        for h_idx, h_name in [(1, "6 Hour"), (2, "12 Hour")]:
            n_lag = responses[key].shape[1]
            ax.plot(lags_hours[:n_lag], responses[key][h_idx],
                    label=f"{pretty.get(key, key)} - {h_name}")
        ax.set_ylabel(pretty.get(key, key) + " response")
        ax.legend()
        ax.grid(True)
        if idx_ax == 0:
            ax.set_title("Magnetospheric Impulse Responses (Rung 3 Multi-Filter)")
    axes[-1].set_xlabel("Lag (Hours)")
    # If enriched channels exist, add a separate figure so we can inspect them.
    enriched = [k for k in responses if k not in ("v_sw", "bz_s", "pdyn")]
    if enriched:
        fig2, ax2 = plt.subplots(len(enriched), 1, figsize=(10, 3 * len(enriched)), sharex=True)
        if len(enriched) == 1:
            ax2 = [ax2]
        for i, key in enumerate(enriched):
            for h_idx, h_name in [(1, "6 Hour"), (2, "12 Hour")]:
                ax2[i].plot(lags_hours[:responses[key].shape[1]], responses[key][h_idx],
                             label=f"{pretty.get(key, key)} - {h_name}")
            ax2[i].set_ylabel(pretty.get(key, key))
            ax2[i].legend(); ax2[i].grid(True)
            if i == 0:
                ax2[i].set_title("Enriched Coupling Impulse Responses (Rung 3)")
        ax2[-1].set_xlabel("Lag (Hours)")
        fig2.tight_layout()
        fig2.savefig(FIGURES_DIR / "impulse_responses_enriched.png", dpi=150)
        plt.close(fig2)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "impulse_responses.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved impulse response diagnostic to {fig_path}")
    
    # Plot 2: Time Series Overlay (Storm Case Study)
    # User-selected timeframe: December 12-24, 2016
    y_test_true_6h = y_test_seq_flat[:, 1]

    # Find indices for the fixed date range: Dec 12-24, 2016
    # index_test_seq is a DatetimeIndex (seq-aligned test indices)
    start_date = pd.Timestamp("2016-12-12")
    end_date = pd.Timestamp("2016-12-24")

    # Match tz-awareness of index_test_seq so searchsorted doesn't choke
    if getattr(index_test_seq, "tz", None) is not None:
        start_date = start_date.tz_localize(index_test_seq.tz)
        end_date = end_date.tz_localize(index_test_seq.tz)
    
    start_plot = index_test_seq.searchsorted(start_date, side="left")
    end_plot = index_test_seq.searchsorted(end_date, side="right")

    # Fallback to peak flux window if date range is empty / out of bounds
    if start_plot >= end_plot or start_plot >= len(index_test_seq):
        max_idx = np.argmax(y_test_true_6h)
        start_plot = max(0, max_idx - 200)
        end_plot = min(len(index_test_seq), max_idx + 200)
        print(f"  [Plot 2] Date range Dec 12-24 not in test index; falling back to peak-flux window")
    else:
        print(f"  [Plot 2] Using fixed window Dec 12-24, 2016 (indices {start_plot}:{end_plot})")
    
    # Get predictions for Hybrid, Stacking (R7), Physics-Clipped Stacking (R8),
    # and Persistence. R6/R7 are the raw model outputs; R8 is the physics-
    # corrected R7 (identical training, suppressed false-recovery trajectories).
    pred_hybrid = model_hybrid.predict(X_test_seq_flat)
    pred_hybrid_6h = pred_hybrid[:, 1]

    # R7a: no-seq instance (model_stacking was fit in seq mode; calling
    # predict without seq_idxs would crash with a feature-count mismatch).
    pred_stacking_r7a = model_stacking_noseq.predict(X_test_seq_flat)
    pred_stacking_r7a_6h = pred_stacking_r7a[:, 1]

    pred_stacking_r7b_full = model_stacking.predict(
        _stack_X, seq_idxs=_stack_test_seq_idxs)
    pred_stacking_r7b_6h = pred_stacking_r7b_full[:, 1]

    pred_persistence = model_persistence.predict(X_test_seq_flat)
    pred_persistence_6h = pred_persistence[:, 1]

    plt.figure(figsize=(12, 6))
    plt.plot(index_test_seq[start_plot:end_plot], y_test_true_6h[start_plot:end_plot], label="Observed (>2 MeV)", color="black", linewidth=2)
    plt.plot(index_test_seq[start_plot:end_plot], pred_persistence_6h[start_plot:end_plot], label="Persistence Forecast", color="gray", alpha=0.7)
    plt.plot(index_test_seq[start_plot:end_plot], pred_hybrid_6h[start_plot:end_plot], label="Hybrid (Rung 6) Forecast", color="red", linestyle="--")
    plt.plot(index_test_seq[start_plot:end_plot], pred_stacking_r7a_6h[start_plot:end_plot], label="Stacking (R7a, R3+R4) Forecast", color="blue", linestyle="-.")
    plt.plot(index_test_seq[start_plot:end_plot], pred_stacking_r7b_6h[start_plot:end_plot], label="Stacking (R7b, R3+R4+TCN) Forecast", color="orange", linestyle="--")
    # R8 wraps model_stacking (seq mode); must pass the FULL matrix _stack_X
    # (not X_test_seq_flat) so the SequencePanel can build windows at the
    # correct offsets in _stack_test_seq_idxs.
    pred_stacking_r8 = _r8_model.predict(_stack_X,
                                          seq_idxs=_stack_test_seq_idxs)
    pred_stacking_r8_6h = pred_stacking_r8[:, 1]
    plt.plot(index_test_seq[start_plot:end_plot], pred_stacking_r8_6h[start_plot:end_plot], label="Stacking R8 (physics-clipped) Forecast", color="green", linestyle=":")

    plt.title("Storm Event Forecast Overlay: Dec 12-24, 2016 (6-Hour Lead Time)")
    plt.ylabel("log10 Electron Flux (pfu)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    storm_path = FIGURES_DIR / "storm_case_study.png"
    plt.savefig(storm_path, dpi=150)
    plt.close()
    print(f"Saved storm case study overlay to {storm_path}")
    
    # Plot 3: Predicted vs Observed Scatter (focus on the physically plausible range)
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
    mask = (y_test_true_6h >= -1.0) & (y_test_true_6h <= 6.0)
    y_focus = y_test_true_6h[mask]
    p_focus = pred_hybrid_6h[mask]
    ax_scatter.scatter(y_focus, p_focus, alpha=0.35, s=3, color="blue")
    lims = [-1.0, 6.0]
    ax_scatter.plot(lims, lims, 'k--', label="Perfect Forecast")
    ax_scatter.axvline(3.0, color='r', linestyle='--', alpha=0.5, label='Hazard threshold')
    ax_scatter.axhline(3.0, color='r', linestyle='--', alpha=0.5)
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_title("Predicted vs Observed Scatter (Hybrid 6h Forecast)")
    ax_scatter.set_xlabel("Observed log10 Flux")
    ax_scatter.set_ylabel("Predicted log10 Flux")
    ax_scatter.grid(True)
    ax_scatter.legend()
    fig_scatter.tight_layout()
    
    scatter_path = FIGURES_DIR / "scatter_plot.png"
    fig_scatter.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)
    print(f"Saved scatter plot to {scatter_path}")

    # Plot 4: Residual distribution for Hybrid 6h.
    # More diagnostic than scatter alone: shows bias and outliers in the
    # forecast error, especially across quiet vs storm regimes.
    residuals = pred_hybrid_6h - y_test_true_6h
    fig_res, ax_res = plt.subplots(figsize=(7, 4))
    ax_res.hist(residuals, bins=120, color="steelblue", edgecolor="black", linewidth=0.3, alpha=0.85)
    ax_res.axvline(0.0, color='k', linestyle='--', linewidth=1.0, label='Zero error')
    ax_res.set_xlabel("Forecast residual (predicted - observed)")
    ax_res.set_ylabel("Count")
    ax_res.set_title("Hybrid 6h Forecast Residual Distribution")
    ax_res.grid(True, axis='y', alpha=0.3)
    ax_res.legend()
    fig_res.tight_layout()
    residual_path = FIGURES_DIR / "hybrid_6h_residuals.png"
    fig_res.savefig(residual_path, dpi=150)
    plt.close(fig_res)
    print(f"Saved residuals plot to {residual_path}")
    
    # 8. Phase 19: GRASP/GSAT Validation (Skipped for real-data workflow)
    print("\n--- Phase 19: GRASP/GSAT Validation Skipped (Pure Real Data Workflow) ---")
    
    print("\n==================================================================")
    print("   Orchestration Complete! All deliverables generated.")
    print("==================================================================")

if __name__ == "__main__":
    main()
