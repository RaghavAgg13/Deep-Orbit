"""Physics-informed regularization for the electron flux forecasters.

This module is PURE NUMPY -- no torch import at module level (torch is optional).
The sequence-based trainers (TCN) and the flat Ridge meta-learner both drive
their training loops in numpy, so the physics constraint must also live in
numpy to be composable across all rungs.

Two constraints implemented:

1. Quiet-time monotonicity: under low Vsw the predicted flux trajectory
   (30m, 6h, 12h) must be non-increasing (physical decay during quiet times).
   Radial diffusion and atmospheric losses dominate in the quiet magnetosphere;
   no physical source exists to drive a recovery during such interval.

2. Diffusion rate bound: d(log10 flux)/dt is bounded by known radial diffusion
   limits. Source and acceleration cannot raise log10 flux faster than
   ~+0.18 / h during enhancements; magnetopause shadowing can drop flux as
   fast as ~-0.5 / h during dropouts (asymmetric bound).

Quiet-regime definition (re-calibrated 2026-06-29)
--------------------------------------------------
The original ``QUIET_VSW = 380`` threshold was BROKEN on the real test set:
``quiet_frac`` came out as 1.000, meaning the monotonic-decay constraint was
applied to *every* sample and actively destroyed real flux rises at 6h/12h
(R8 PE went from +0.386 to -0.094).

The fix is a three-condition quiet mask that is much harder to enter:

    Vsw < 350 km/s                            (strictly quiet solar wind)
    AND sustained for >= 6 h                  (not a transient dip)
    AND flux below the seasonal median        (no ongoing enhancement)

Only samples satisfying ALL three are forced to decay. The diffusion-rate
bound (+0.18 / h) is a hard physical limit and is applied to ALL samples
regardless of regime -- that part of the corrector is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Physical limits of MeV electron flux variation at GEO (log10 pfu / hour).
# From relativistic electron diffusion-coefficient literature:
#   - source+acceleration cannot raise log10 flux faster than ~+0.18 / h
#   - magnetopause shadowing dropout can flux drop as fast as ~-0.5 / h
MAX_RISE_RATE = 0.18     # log10(pfu) / hour, enhancement ceiling
MAX_DROP_RATE = -0.5     # log10(pfu) / hour, dropout floor (negative)
QUIET_VSW = 350.0        # km/s, STRICT quiet threshold (was 380 -- too loose)
QUIET_SUSTAIN_STEPS = 24 # consecutive steps @15-min cadence = 6 h sustained
# R8 (inference clip) intervention threshold. 0.25/h fires on spurious rises
# while letting real storm onsets (+0.2..0.3/h) pass through mostly intact.
# This replaces the previous 0.36/h which was too high to ever trigger.
RISE_RATE_INTERVENTION = 0.25   # log10(pfu)/h - effective soft threshold


class PhysicsConstraint:
    """Physics-informed regularizer over a 3-horizon flux trajectory.

    Operates purely in numpy so it can be composed into any training loop
    (LightGBM custom objective, TCN loss term, Ridge penalty, etc.).

    Parameters
    ----------
    horizon_steps : tuple[int, int, int]
        Step counts for the 3 forecast horizons. Default (2, 24, 48) at
        15-min cadence gives 30-min, 6-h, 12-h.
    cadence_min : float
        Minutes per step. 15.0 for the standard pipeline cadence.
    max_rise : float
        Maximum physically plausible rise rate in log10(pfu) / hour.
    max_drop : float
        Minimum (most negative) physically plausible rate in log10(pfu) / hour.
    quiet_vsw : float
        Solar-wind speed threshold (km/s) below which the magnetosphere
        is considered quiet and flux must decay.
    lam_mono : float
        Penalty weight for the quiet-time monotonicity constraint.
    lam_rate : float
        Penalty weight for the diffusion-rate bound constraint.
    """

    def __init__(
        self,
        horizon_steps: tuple[int, int, int] = (2, 24, 48),
        cadence_min: float = 15.0,
        max_rise: float = MAX_RISE_RATE,
        max_drop: float = MAX_DROP_RATE,
        quiet_vsw: float = QUIET_VSW,
        lam_mono: float = 0.05,
        lam_rate: float = 0.05,
    ):
        if len(horizon_steps) != 3:
            raise ValueError("horizon_steps must have exactly 3 entries")
        self.horizon_steps = tuple(horizon_steps)
        self.cadence_min = float(cadence_min)
        self.max_rise = float(max_rise)
        self.max_drop = float(max_drop)
        self.quiet_vsw = float(quiet_vsw)
        self.quiet_sustain_steps = int(QUIET_SUSTAIN_STEPS)
        self.lam_mono = float(lam_mono)
        self.lam_rate = float(lam_rate)

        # Pre-compute the time (in hours) between consecutive horizons.
        h0, h1, h2 = self.horizon_steps
        self._dt01 = (h1 - h0) * self.cadence_min / 60.0
        self._dt12 = (h2 - h1) * self.cadence_min / 60.0

    def hours_per_step(self) -> float:
        """Cadence converted to hours (cadence_min / 60)."""
        return self.cadence_min / 60.0

    def _dt_between(self, k: int) -> float:
        """Hours between horizon k and horizon k+1."""
        if k == 0:
            return self._dt01
        if k == 1:
            return self._dt12
        raise ValueError("k must be 0 or 1 for the 2 forward differences")

    def trajectory_rates(self, y_pred: np.ndarray) -> np.ndarray:
        """Forward-difference rates (log10(pfu) / hour) across the trajectory.

        Parameters
        ----------
        y_pred : ndarray, shape (n, 3)
            Predicted log10 flux at the 3 horizons (30m, 6h, 12h).

        Returns
        -------
        rates : ndarray, shape (n, 2)
            rates[:, 0] = (y1 - y0) / dt01
            rates[:, 1] = (y2 - y1) / dt12
        """
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim != 2 or y_pred.shape[1] != 3:
            raise ValueError(f"y_pred must have shape (n, 3), got {y_pred.shape}")
        y0 = y_pred[:, 0]
        y1 = y_pred[:, 1]
        y2 = y_pred[:, 2]
        r01 = (y1 - y0) / self._dt01
        r12 = (y2 - y1) / self._dt12
        return np.column_stack([r01, r12])

    def quiet_mask(self, v_sw: np.ndarray,
                  flux: np.ndarray | None = None,
                  index=None) -> np.ndarray:
        """Boolean mask of samples in quiet magnetosphere.

        Three conditions must ALL hold:
        1. v_sw < quiet_vsw
        2. Sustained for >= QUIET_SUSTAIN_STEPS consecutive rows
        3. flux < seasonal median (when flux/index supplied)
        """
        v_sw = np.asarray(v_sw, dtype=float).ravel()
        n = v_sw.shape[0]
        mask = v_sw < self.quiet_vsw

        if mask.size > 0:
            sustained = np.zeros(n, dtype=bool)
            run_start = None
            for i in range(n + 1):
                in_run = i < n and mask[i]
                if in_run and run_start is None:
                    run_start = i
                if not in_run and run_start is not None:
                    run_len = i - run_start
                    if run_len >= self.quiet_sustain_steps:
                        sustained[run_start:i] = True
                    run_start = None
            mask = mask & sustained

        if flux is not None:
            flux = np.asarray(flux, dtype=float).ravel()
            if flux.shape[0] == n:
                if index is not None and hasattr(index, "dayofyear"):
                    med = _seasonal_median(flux, index, half_window_days=15)
                else:
                    med = _rolling_median(flux, window=2880)
                mask = mask & (flux < med)

        return mask

    def monotonicity_violation(
        self, y_pred: np.ndarray, quiet_mask: np.ndarray
    ) -> float:
        """Mean monotonicity violation over quiet samples."""
        y_pred = np.asarray(y_pred, dtype=float)
        quiet_mask = np.asarray(quiet_mask, dtype=bool).ravel()
        if y_pred.shape[0] != quiet_mask.shape[0]:
            raise ValueError(
                f"y_pred rows ({y_pred.shape[0]}) must match quiet_mask length "
                f"({quiet_mask.shape[0]})"
            )
        if not np.any(quiet_mask):
            return 0.0
        yq = y_pred[quiet_mask]
        v01 = np.maximum(yq[:, 1] - yq[:, 0], 0.0)
        v12 = np.maximum(yq[:, 2] - yq[:, 1], 0.0)
        return float(np.mean(v01 + v12))

    def rate_violation(self, y_pred: np.ndarray) -> float:
        """Mean squared rate-bound violation over ALL samples."""
        y_pred = np.asarray(y_pred, dtype=float)
        rates = self.trajectory_rates(y_pred)
        rise_viol = np.maximum(rates - self.max_rise, 0.0) ** 2
        drop_viol = np.maximum(self.max_drop - rates, 0.0) ** 2
        return float(np.mean(np.sum(rise_viol + drop_viol, axis=1)))

    def penalty(self, y_pred: np.ndarray, v_sw: np.ndarray,
                flux: np.ndarray | None = None, index=None) -> dict:
        """Full physics penalty breakdown."""
        y_pred = np.asarray(y_pred, dtype=float)
        v_sw = np.asarray(v_sw, dtype=float).ravel()
        qmask = self.quiet_mask(v_sw, flux=flux, index=index)
        mono = self.monotonicity_violation(y_pred, qmask)
        rate = self.rate_violation(y_pred)
        total = self.lam_mono * mono + self.lam_rate * rate
        quiet_frac = float(np.mean(qmask)) if qmask.size > 0 else 0.0
        return {
            "mono": float(mono),
            "rate": float(rate),
            "total": float(total),
            "quiet_frac": float(quiet_frac),
        }


def _rolling_median(a: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling median via pandas implementation."""
    s = pd.Series(a, copy=False)
    return s.rolling(window, min_periods=1, center=True).median().to_numpy()


def _seasonal_median(a: np.ndarray, index, half_window_days: int = 15) -> np.ndarray:
    """Per-row seasonal median = median of ``a`` in a ±half_window_days calendar bin."""
    s = pd.Series(a, index=index, copy=False)
    doy = s.index.dayofyear.to_numpy()
    win = half_window_days
    daily = s.groupby(doy).median()
    day_keys = daily.index.to_numpy()
    day_vals = daily.to_numpy(dtype=float)
    ext_days = np.concatenate([day_keys - 365, day_keys, day_keys + 365])
    ext_vals = np.tile(day_vals, 3)
    order = np.argsort(ext_days)
    ext_days = ext_days[order]
    ext_vals = ext_vals[order]
    lo = np.searchsorted(ext_days, doy - win, side="left")
    hi = np.searchsorted(ext_days, doy + win, side="right")
    med = np.array([np.median(ext_vals[l:h]) if h > l else a[i]
                    for i, (l, h) in enumerate(zip(lo, hi))], dtype=float)
    return med


def physics_penalized_residual(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    v_sw: np.ndarray,
    constraint: PhysicsConstraint,
    mse: float,
) -> tuple[float, dict]:
    """Physics-aware total loss = MSE + physics penalty."""
    pen = constraint.penalty(y_pred, v_sw)
    return mse + pen["total"], pen


def physics_clip_trajectory(
    y_pred: np.ndarray,
    v_sw: np.ndarray,
    constraint: PhysicsConstraint,
    flux: np.ndarray | None = None,
    index=None,
    mode: str = "soft",
) -> np.ndarray:
    """Hard post-hoc corrector that enforces physical bounds.

    Two corrections are applied on a *copy* of the input (never in-place):
    1. Quiet-time monotonicity: force non-increasing trajectory on quiet samples.
    2. Rise-rate bound: clamp rises exceeding the threshold.
    """
    y = np.array(y_pred, dtype=float, copy=True)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"y_pred must have shape (n, 3), got {y.shape}")
    v_sw = np.asarray(v_sw, dtype=float).ravel()
    if v_sw.shape[0] != y.shape[0]:
        raise ValueError(
            f"v_sw length ({v_sw.shape[0]}) must match y_pred rows "
            f"({y.shape[0]})"
        )

    quiet = constraint.quiet_mask(v_sw, flux=flux, index=index)

    if np.any(quiet):
        yq = y[quiet]
        yq[:, 1] = np.minimum(yq[:, 1], yq[:, 0])
        yq[:, 2] = np.minimum(yq[:, 2], yq[:, 1])
        y[quiet] = yq

    if mode == "strict":
        rise_limit = constraint.max_rise
    elif mode == "soft":
        rise_limit = RISE_RATE_INTERVENTION
    else:
        raise ValueError(
            f"physics_clip_trajectory mode must be 'soft' or 'strict', "
            f"got {mode!r}")

    max_inc_01 = y[:, 0] + rise_limit * constraint._dt01
    interv_01 = y[:, 1] > max_inc_01
    y[:, 1] = np.minimum(y[:, 1], max_inc_01)
    max_inc_12 = y[:, 1] + rise_limit * constraint._dt12
    interv_12 = y[:, 2] > max_inc_12
    y[:, 2] = np.minimum(y[:, 2], max_inc_12)

    n = y.shape[0]
    n_30m = int(np.sum(interv_01))
    n_6h = int(np.sum(interv_12))
    n_12h = int(np.sum(interv_01 | interv_12))
    if n > 0:
        pct_30m = n_30m / n * 100.0
        pct_6h = n_6h / n * 100.0
        pct_12h = n_12h / n * 100.0
    else:
        pct_30m = pct_6h = pct_12h = 0.0
    quiet_frac = float(np.mean(quiet)) if quiet.size > 0 else 0.0
    print(
        f"[R8 diag] per-horizon rise-rate interventions: "
        f"30m={n_30m}/{n} ({pct_30m:.2f}%), "
        f"6h={n_6h}/{n} ({pct_6h:.2f}%), "
        f"12h={n_12h}/{n} ({pct_12h:.2f}%); "
        f"quiet_frac={quiet_frac:.3f}"
    )

    return y


def diagnose_physics_violations(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    v_sw: np.ndarray,
    constraint: PhysicsConstraint,
    flux: np.ndarray | None = None,
    index=None,
) -> dict:
    """Validation-time diagnostic: violation counts before / after clipping."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    v_sw = np.asarray(v_sw, dtype=float).ravel()

    n = y_pred.shape[0]
    quiet = constraint.quiet_mask(v_sw, flux=flux, index=index)
    n_quiet = int(np.sum(quiet))
    quiet_frac = float(n_quiet / n) if n > 0 else 0.0

    quiet_rec_per_1000, rate_per_1000, mean_mono, mean_rate = (
        _count_violations(y_pred, quiet, constraint, n, n_quiet)
    )

    rmse_before = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pe_before = _prediction_efficiency(y_true, y_pred)

    y_clip = physics_clip_trajectory(y_pred, v_sw, constraint)
    quiet_rec_per_1000_after, rate_per_1000_after, _, _ = (
        _count_violations(y_clip, quiet, constraint, n, n_quiet)
    )
    rmse_after = float(np.sqrt(np.mean((y_true - y_clip) ** 2)))
    pe_after = _prediction_efficiency(y_true, y_clip)

    return {
        "quiet_fraction": quiet_frac,
        "quiet_recovery_violations_per_1000": quiet_rec_per_1000,
        "quiet_recovery_violations_per_1000_after": quiet_rec_per_1000_after,
        "rate_violations_per_1000": rate_per_1000,
        "rate_violations_per_1000_after": rate_per_1000_after,
        "mean_mono_violation": float(mean_mono),
        "mean_rate_violation": float(mean_rate),
        "rmse_before": rmse_before,
        "rmse_after_clip": rmse_after,
        "pe_before": float(pe_before),
        "pe_after_clip": float(pe_after),
    }


def _count_violations(
    y: np.ndarray,
    quiet: np.ndarray,
    constraint: PhysicsConstraint,
    n: int,
    n_quiet: int,
) -> tuple[float, float, float, float]:
    """Return (quiet_rec_per_1000, rate_per_1000, mean_mono, mean_rate)."""
    quiet_recovery_violations = 0.0
    if n_quiet > 0:
        yq = y[quiet]
        v01 = np.maximum(yq[:, 1] - yq[:, 0], 0.0)
        v12 = np.maximum(yq[:, 2] - yq[:, 1], 0.0)
        n_viol_samples = int(np.sum((v01 > 0) | (v12 > 0)))
        quiet_recovery_violations = float(n_viol_samples)

    rates = constraint.trajectory_rates(y)
    rise_viol = rates > constraint.max_rise
    drop_viol = rates < constraint.max_drop
    rate_violation_mat = rise_viol | drop_viol
    n_rate_viol_samples = int(np.sum(np.any(rate_violation_mat, axis=1)))
    rate_violations_total = float(n_rate_viol_samples)

    quiet_rec_per_1000 = (
        (quiet_recovery_violations / n_quiet * 1000.0) if n_quiet > 0 else 0.0
    )
    rate_per_1000 = (
        (rate_violations_total / n * 1000.0) if n > 0 else 0.0
    )
    mean_mono = constraint.monotonicity_violation(y, quiet)
    mean_rate = constraint.rate_violation(y)
    return quiet_rec_per_1000, rate_per_1000, mean_mono, mean_rate


def _prediction_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PE = 1 - MSE / Var(y_true), computed over the full (n, 3) arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


if __name__ == "__main__":
    import pandas as pd

    np.random.seed(42)

    N = 2000
    y_true = np.zeros((N, 3))
    y_true[:, 0] = np.random.uniform(0.5, 4.0, size=N)
    y_true[:, 1] = y_true[:, 0] + np.random.normal(0.0, 0.3, size=N)
    y_true[:, 2] = y_true[:, 1] + np.random.normal(0.0, 0.4, size=N)

    y_pred = y_true + np.random.normal(0.05, 0.25, size=(N, 3))

    v_sw = np.random.uniform(300.0, 700.0, size=N)
    v_sw[500:1000] = np.random.uniform(300.0, 345.0, size=500)

    fake_index = pd.date_range("2016-01-01", periods=N, freq="15min")
    flux_now = np.full(N, fill_value=3.0)
    flux_now[500:1000] = np.random.uniform(0.2, 0.8, size=500)

    con = PhysicsConstraint()
    print("=== PhysicsConstraint self-test ===")
    print(f"hours_per_step = {con.hours_per_step():.4f}")
    print(f"dt01 = {con._dt01:.2f} h, dt12 = {con._dt12:.2f} h")

    pen = con.penalty(y_pred, v_sw, flux=flux_now, index=fake_index)
    print(f"Penalty breakdown (strict): mono={pen['mono']:.4f}  "
          f"rate={pen['rate']:.4f}  total={pen['total']:.4f}  "
          f"quiet_frac={pen['quiet_frac']:.4f}")

    pen_loose = con.penalty(y_pred, v_sw)
    print(f"Penalty breakdown (loose):  mono={pen_loose['mono']:.4f}  "
          f"rate={pen_loose['rate']:.4f}  total={pen_loose['total']:.4f}  "
          f"quiet_frac={pen_loose['quiet_frac']:.4f}")

    diag = diagnose_physics_violations(
        y_true, y_pred, v_sw, con, flux=flux_now, index=fake_index)
    print("\n--- diagnose_physics_violations (strict mask) ---")
    for k, v in diag.items():
        print(f"  {k:44s} = {v:.4f}")

    y_clip = physics_clip_trajectory(
        y_pred, v_sw, con, flux=flux_now, index=fake_index)
    diag_after = diagnose_physics_violations(
        y_true, y_clip, v_sw, con, flux=flux_now, index=fake_index)
    print("\n--- after clip ---")
    print(f"  quiet_recovery_violations/1000: "
          f"{diag['quiet_recovery_violations_per_1000']:.2f} -> "
          f"{diag_after['quiet_recovery_violations_per_1000_after']:.2f}")
    print(f"  rate_violations/1000:          "
          f"{diag['rate_violations_per_1000']:.2f} -> "
          f"{diag_after['rate_violations_per_1000_after']:.2f}")
    print(f"  rmse: {diag['rmse_before']:.4f} -> "
          f"{diag_after['rmse_after_clip']:.4f}")
    print(f"  pe:   {diag['pe_before']:.4f} -> "
          f"{diag_after['pe_after_clip']:.4f}")

    assert (
        diag_after["quiet_recovery_violations_per_1000_after"]
        <= diag["quiet_recovery_violations_per_1000"] + 1e-9
    ), "clip should reduce quiet-time recovery violations"
    assert (
        diag_after["rate_violations_per_1000_after"]
        <= diag["rate_violations_per_1000"] + 1e-9
    ), "clip should reduce rate violations"

    assert pen["quiet_frac"] <= pen_loose["quiet_frac"], (
        "strict mask must be a subset of the loose mask")
    assert pen["quiet_frac"] > 0.0, (
        "strict mask must have found the injected quiet segment")

    mse = float(np.mean((y_true - y_pred) ** 2))
    total, breakdown = physics_penalized_residual(
        y_pred, y_true, v_sw, con, mse)
    assert abs(total - (mse + breakdown["total"])) < 1e-12

    y_pred_copy = y_pred.copy()
    _ = physics_clip_trajectory(y_pred, v_sw, con)
    assert np.array_equal(y_pred, y_pred_copy), "clip must not mutate input"

    assert np.all(np.isfinite(diag["pe_before"]))

    print("\nAll self-test assertions passed.")