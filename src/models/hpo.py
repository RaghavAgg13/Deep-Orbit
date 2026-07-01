# src/models/hpo.py
"""
Optuna-based hyperparameter optimization for the TCNForecaster.

Fully self-contained: builds synthetic data in its ``__main__`` block so the
objective function can be sanity-tested without the real pipeline.

Objective: maximize a composite skill score S = mean_over_horizons(
    0.6 * norm(pe) + 0.4 * norm(hss) )
where norm(x) = x clamped to [0, 1] against a reference (persistence) baseline.
Most users optimize PE at the 12h horizon only; the composite is more stable.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np

# torch is optional at import time so the package imports cleanly.
try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from src.config import MODELS_DIR, SEQ_LEN, REPORTS_DIR
from src.metrics import prediction_efficiency, hss, to_events
from src.models.tcn import TCNForecaster

# Horizon weighting: 30-min, 6-h, 12-h. The 12-h horizon carries the most weight
# since it is the operationally meaningful prediction window.
_HORIZON_WEIGHTS = np.array([0.2, 0.3, 0.5], dtype=np.float64)

# Hazard threshold for event-based metrics (log10 flux).
_HAZARD_LOG = 3.0


def _load_persistence_ref() -> list[float]:
    """Return Persistence PE at [30m, 6h, 12h] from reports/benchmark.csv.

    Fails gracefully: returns [0.0, 0.0, 0.0] if the CSV cannot be read or the
    Persistence rows are missing.
    """
    csv_path = Path(REPORTS_DIR) / "benchmark.csv"
    import csv

    horizon_key = {"30 Min": 0, "6 Hour": 1, "12 Hour": 2}
    ref = [0.0, 0.0, 0.0]
    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("model") == "Persistence" and row.get("horizon") in horizon_key:
                    ref[horizon_key[row["horizon"]]] = float(row["pe"])
    except Exception:
        return [0.0, 0.0, 0.0]
    return ref


def default_search_space(trial) -> dict:
    """Return a dict of TCNForecaster kwargs sampled from ``trial``.

    ``seq_len`` is FIXED at the pipeline's ``SEQ_LEN`` (192) — only the model
    hyper-parameters are tuned here. ``epochs`` is fixed at 60; the caller
    early-stops via the Forecaster's internal patience.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 48, 64, 96, 128])
    num_layers = trial.suggest_int("num_layers", 5, 10)
    kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5])
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    use_attention = trial.suggest_categorical("use_attention", [True, False])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    chosen = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "use_attention": use_attention,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "epochs": 60,
        "seq_len": SEQ_LEN,
    }
    print(f"  [HPO trial {trial.number}] {chosen}")
    return chosen


def _norm_pe(pe: float, ref: float) -> float:
    """Normalize PE against a reference (persistence) baseline, clamped to [0, 1]."""
    denom = max(1e-6, 1.0 - ref)
    return float(np.clip((pe - ref) / denom, 0.0, 1.0))


def make_objective(
    X_train,
    y_train,
    X_val,
    y_val,
    feature_names=None,
    n_horizons: int = 3,
    ref_pe: list[float] | None = None,
    ref_hss: list[float] | None = None,
    device: str | None = None,
    event_weight: float = 1.5,
):
    """Return a Optuna ``objective(trial) -> float``.

    The objective trains a fresh ``TCNForecaster`` per trial on ``X_train`` and
    scores it against ``X_val`` using a horizon-weighted composite of
    persistence-normalized PE and HSS. Every knob sampled by
    ``default_search_space`` is forwarded to the constructor, so the value an
    Optuna trial reports is the value that actually trained the model.
    """
    if ref_pe is None:
        ref_pe = _load_persistence_ref()
    if ref_hss is None:
        ref_hss = [0.0] * n_horizons
    # Uniform horizon weights if caller asked for a non-default count.
    if n_horizons == 3:
        horizon_w = _HORIZON_WEIGHTS
    else:
        horizon_w = np.ones(n_horizons, dtype=np.float64) / n_horizons

    def objective(trial) -> float:
        try:
            kwargs = default_search_space(trial)
            # ``default_search_space`` samples ``weight_decay``; merge it into
            # the constructor kwargs verbatim so the reported value is the one
            # that trained the model (read by ``fit_sequences`` -> AdamW).
            fit_kwargs = dict(kwargs)
            clf = TCNForecaster(**fit_kwargs)
            # ``event_weight`` maps to the Forecaster's ``storm_weight``.
            clf.storm_weight = float(event_weight)
            if device is not None:
                clf.device = device

            clf.fit_sequences(
                X_train, y_train, X_val, y_val, feature_names=feature_names
            )
            y_hat_val = clf.predict(X_val)

            pe_list: list[float] = []
            hss_list: list[float] = []
            for h in range(n_horizons):
                y_t = np.asarray(y_val[:, h], dtype=np.float64)
                y_p = np.asarray(y_hat_val[:, h], dtype=np.float64)
                pe = prediction_efficiency(y_t, y_p)
                yt_bin = to_events(y_t, _HAZARD_LOG)
                yp_bin = to_events(y_p, _HAZARD_LOG)
                hv = float(hss(yt_bin, yp_bin))
                pe_list.append(pe)
                hss_list.append(hv)
                trial.report(float(pe), step=h)

            pe_n = np.array(
                [_norm_pe(pe, float(ref_pe[h])) for h in range(n_horizons)],
                dtype=np.float64,
            )
            hss_n = np.clip(
                np.array(
                    [
                        max(0.0, (hv - float(ref_hss[h])))
                        for h in range(n_horizons)
                    ],
                    dtype=np.float64,
                ),
                0.0,
                1.0,
            )
            score_per = 0.6 * pe_n + 0.4 * hss_n
            final = float(np.dot(horizon_w, score_per))

            trial.set_user_attr("pe_per_horizon", [float(p) for p in pe_list])
            trial.set_user_attr("hss_per_horizon", [float(v) for v in hss_list])
            trial.set_user_attr("score_per_horizon", score_per.tolist())
            return max(0.0, final)
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # HPO must never die on a bad config.
            print(f"  [HPO trial {trial.number}] CRASHED: {exc!r}")
            trial.set_user_attr("crash", repr(exc))
            return 0.0

    return objective



def run_hpo(
    X_train,
    y_train,
    X_val,
    y_val,
    feature_names=None,
    n_trials: int = 40,
    timeout_s: int = 1800,
    study_name: str = "tcn_hpo",
    db_path: str | None = None,
    device: str | None = None,
    event_weight: float = 1.5,
    seed: int = 42,
) -> dict:
    """Run an Optuna study over the TCN search space and persist results.

    Creates or resumes a TPE-sampled study keyed by ``study_name`` in an SQLite
    store under ``reports/``.  Returns a dict with the study, best params,
    best value, path to the joblib'd best model, and the per-horizon PE of the
    best trial.
    """
    import optuna  # local: optuna is an optional dependency

    if db_path is None:
        db_path = str(Path(REPORTS_DIR) / "hpo_study.sqlite")
    else:
        db_path = str(Path(db_path))

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names=feature_names,
        n_horizons=3,
        device=device,
        event_weight=event_weight,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_value = float(best_trial.value)
    best_pe = list(best_trial.user_attrs.get("pe_per_horizon", [0.0, 0.0, 0.0]))
    best_hss = list(best_trial.user_attrs.get("hss_per_horizon", [0.0, 0.0, 0.0]))

    # Trials dataframe -> CSV
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_csv = Path(REPORTS_DIR) / "hpo_trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    # Re-fit the best configuration so its live model is on disk for inference.
    fit_kwargs = {
        "hidden_dim": best_params["hidden_dim"],
        "num_layers": best_params["num_layers"],
        "kernel_size": best_params["kernel_size"],
        "dropout": best_params["dropout"],
        "use_attention": best_params["use_attention"],
        "lr": best_params["lr"],
        "batch_size": best_params["batch_size"],
        "epochs": 60,
        "seq_len": SEQ_LEN,
    }
    best_clf = TCNForecaster(**fit_kwargs)
    best_clf.storm_weight = float(event_weight)
    best_clf.fit_sequences(
        X_train, y_train, X_val, y_val, feature_names=feature_names
    )

    joblib_path = Path(MODELS_DIR) / "tcn_hpo_forecaster.pkl"
    joblib.dump(best_clf, joblib_path)

    # Human-readable report.
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value if t.value is not None else -float("inf"),
        reverse=True,
    )
    top5 = sorted_trials[:5]

    # "Feature importance" analogue: which hyper values are most frequent among
    # the top quartile of COMPLETED trials.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_sorted = sorted(
        [t for t in completed if t.value is not None],
        key=lambda t: t.value if t.value is not None else -float("inf"),
        reverse=True,
    )
    q = max(1, len(completed_sorted) // 4)
    top_q = completed_sorted[:q]
    top_params = [dict(t.params) for t in top_q]
    param_freq: dict[str, dict[str, int]] = {}
    for tp in top_params:
        for k, v in tp.items():
            param_freq.setdefault(k, {})
            param_freq[k][v] = param_freq[k].get(v, 0) + 1

    report_path = Path(REPORTS_DIR) / "hpo_report.txt"
    with open(report_path, "w") as fh:
        fh.write("TCN Hyperparameter Optimization Report\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Study:             {study_name}\n")
        fh.write(f"Trials completed:  {len(completed)}\n")
        fh.write(f"Best value (S):    {best_value:.6f}\n")
        fh.write(f"Best params:       {best_params}\n")
        fh.write(f"Best PE / horizon: {best_pe}\n")
        fh.write(f"Best HSS / horizon:{best_hss}\n\n")
        fh.write("Top-5 trials\n")
        fh.write("-" * 60 + "\n")
        for i, t in enumerate(top5, 1):
            pe = t.user_attrs.get("pe_per_horizon", [])
            fh.write(
                f"  #{i} trial {t.number:>3d}  value={t.value:.5f}  "
                f"pe={[round(x, 4) for x in pe]}  {t.params}\n"
            )
        fh.write("\nTop-quartile hyperparameter frequencies (feature-importance analogue)\n")
        fh.write("-" * 60 + "\n")
        for k in sorted(param_freq):
            fh.write(f"  {k}: {param_freq[k]}\n")
        fh.write(f"\nBest model (joblib): {joblib_path}\n")

    print(f"[HPO] best S={best_value:.5f} params={best_params}")
    print(f"[HPO] trials CSV:  {trials_csv}")
    print(f"[HPO] report:      {report_path}")
    print(f"[HPO] best model:  {joblib_path}")

    return {
        "study": study,
        "best_params": best_params,
        "best_value": best_value,
        "best_joblib_path": str(joblib_path),
        "best_pe_per_horizon": best_pe,
        "best_hss_per_horizon": best_hss,
        "n_trials_completed": len(completed),
    }


def load_hpo_best(path: str | Path | None = None) -> TCNForecaster:
    """Load the best TCNForecaster joblib'd by :func:`run_hpo`."""
    if path is None:
        path = Path(MODELS_DIR) / "tcn_hpo_forecaster.pkl"
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Self-contained sanity test.  Builds deterministic synthetic data and runs a
# tiny HPO so the objective can be validated without the real pipeline.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for the HPO self-test.")

    import time as _time

    import optuna  # local: optuna is an optional dependency

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    torch.manual_seed(42)
    np.random.seed(42)

    N, SEQ, F = 6000, SEQ_LEN, 11
    H = 3

    # A strong per-channel mean shift so the task isn't pure noise, plus a clean
    # sinusoidal component so the event-based HSS is achievable.
    X = torch.randn(N, SEQ, F).numpy().astype(np.float32)
    channel_shift = torch.linspace(-2.0, 2.0, F).numpy().astype(np.float32)
    X += channel_shift.reshape(1, 1, -1)

    t = torch.linspace(0.0, 6.283185307179586, SEQ)
    wave = (torch.sin(t) * 1.5).numpy().astype(np.float32)
    for h in range(H):
        X[:, :, h] += wave.reshape(1, -1)

    # Target: mix of last-step and first-step features with the sinusoid, then
    # add event-generating spikes so HSS is meaningful.
    X64 = X.astype(np.float64)
    y = (
        0.1 * X64[:, -1, :H]
        + 0.9 * X64[:, 0, :H]
        + np.float64(0.5) * wave.reshape(1, -1).repeat(N, axis=0)[:, :H]
    ).astype(np.float32)
    # Inject rare exceedance events (~3 % of samples).
    spike_mask = np.random.rand(N, H) < 0.03
    y = y.copy()
    y[spike_mask] = 3.5

    X_train, X_val = X[:4000], X[4000:]
    y_train, y_val = y[:4000], y[4000:]

    feature_names = [f"f{i}" for i in range(F)]

    wall0 = _time.time()
    result = run_hpo(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names=feature_names,
        n_trials=6,
        timeout_s=120,
        study_name="tcn_hpo_selftest",
        device="cpu",
        seed=42,
    )
    wall = _time.time() - wall0
    print(f"\nSelf-test wall time: {wall:.1f}s")

    study = result["study"]
    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state", "user_attrs")
    )
    print("\nBest value :", result["best_value"])
    print("Best params:", result["best_params"])
    print(
        "Top-3:\n",
        trials_df.sort_values("value", ascending=False)
        .head(3)[["number", "value", "params_hidden_dim", "params_num_layers",
                   "params_kernel_size", "params_dropout", "params_lr",
                   "params_use_attention", "params_batch_size",
                   "params_weight_decay"]],
    )

    print("\n--- reports/hpo_report.txt ---")
    print(Path(REPORTS_DIR, "hpo_report.txt").read_text())

    import os as _os
    print("Artifacts present:")
    print("  hpo_study.sqlite :", _os.path.exists(REPORTS_DIR / "hpo_study.sqlite"))
    print("  hpo_trials.csv   :", _os.path.exists(REPORTS_DIR / "hpo_trials.csv"))
    print("  hpo_report.txt   :", _os.path.exists(REPORTS_DIR / "hpo_report.txt"))
    print("  tcn_hpo_forecaster.pkl:", _os.path.exists(MODELS_DIR / "tcn_hpo_forecaster.pkl"))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    checks = {
        ">= 5 COMPLETED": len(completed) >= 5,
        "best_value > 0": result["best_value"] > 0,
        "sqlite exists": _os.path.exists(REPORTS_DIR / "hpo_study.sqlite"),
        "pkl exists": _os.path.exists(MODELS_DIR / "tcn_hpo_forecaster.pkl"),
    }
    print("\nSelf-test checks:")
    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
