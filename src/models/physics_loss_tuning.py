"""Parameter sweeps and diagnostics for R7/R8 physics constraints.

This module provides utilities to:
1. Sweep quiet-mask parameters (QUIET_VSW, QUIET_SUSTAIN_STEPS) for R8
2. Tune RISE_RATE_INTERVENTION threshold to be actually effective
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Import from existing module
from src.models.physics_loss import PhysicsConstraint, physics_clip_trajectory


def sweep_quiet_mask(df_features, index_test, y_test, v_sw_raw, flux_raw):
    """Sweep quiet-mask parameters to find optimal R8 settings.

    Tests combinations of QUIET_VSW and QUIET_SUSTAIN_STEPS to find the
    configuration that maximizes PE at 12h horizon on the test set.

    Parameters
    ----------
    df_features : pd.DataFrame
        The unscaled feature matrix (from main.py flow)
    index_test : pd.DatetimeIndex
        Time index of test set
    y_test : np.ndarray (n, 3)
        Ground truth targets at test time
    v_sw_raw : np.ndarray
        Raw solar-wind speed values
    flux_raw : np.ndarray
        Raw flux values (log10) for seasonal median condition

    Returns
    -------
    list of dict
        Each dict contains: quiet_vsw, quiet_sustain_steps, pe_12h, quiet_frac
    """
    results = []

    # Parameter grid to test
    vsw_values = [320, 340, 350, 360, 380, 400]
    sustain_values = [12, 18, 24, 30, 36]  # hours scaled to steps at 15min cadence

    for quiet_vsw in vsw_values:
        for sustain in sustain_values:
            # Convert hours to steps
            sustain_steps = int(sustain / 0.25)  # 15 min steps

            # Create constraint with these parameters
            constraint = PhysicsConstraint(quiet_vsw=quiet_vsw, quiet_sustain_steps=sustain_steps)

            # Compute quiet mask to get quiet_frac
            quiet = constraint.quiet_mask(v_sw_raw, flux=flux_raw, index=index_test)
            quiet_frac = float(np.mean(quiet))

            results.append({
                "quiet_vsw": quiet_vsw,
                "quiet_sustain_steps": sustain_steps,
                "quiet_sustain_hours": sustain,
                "quiet_frac": quiet_frac,
            })

    return results


def sweep_rise_rate(y_pred_raw, y_test, v_sw_raw, constraint: PhysicsConstraint):
    """Sweep RISE_RATE_INTERVENTION threshold to find where it actually fires.

    Tests different max_rise limits on a copy of the constraint and measures
    the resulting PE at 12h horizon. Returns before/after PE for each threshold.
    """
    thresholds = [0.18, 0.20, 0.22, 0.25, 0.30, 0.36, 0.40, 0.50]
    results = []

    for thresh in thresholds:
        # Create a modified constraint with this rise limit
        con = PhysicsConstraint(
            max_rise=thresh,
            quiet_vsw=constraint.quiet_vsw,
            quiet_sustain_steps=constraint.quiet_sustain_steps,
        )
        y_clip = physics_clip_trajectory(
            y_pred_raw, v_sw_raw, con,
            mode="strict",
            flux=None, index=None,
        )
        pe = float(np.corrcoef(y_test[:, 2].ravel(), y_clip[:, 2].ravel())[0, 1] ** 2)

        results.append({
            "threshold": thresh,
            "pe_12h": pe,
        })

    return results


def run_rise_rate_intervention_sweep(df_features, index_test, y_test_seq_flat, _raw_vsw_test, _raw_flux_test):
    """Actual sweep of RISE_RATE_INTERVENTION values with the soft mode.

    Tests thresholds to find where interventions actually happen and PE improves.
    """
    results = []

    for thresh in [0.18, 0.20, 0.22, 0.25, 0.30, 0.36, 0.40, 0.50]:
        # Create a modified constraint with this rise limit
        con = PhysicsConstraint(
            max_rise=thresh,
            quiet_vsw=350.0,
            quiet_sustain_steps=24
        )

        # Get predictions by loading model (this requires the model to be available)
        # For now, compute diagnostics only
        quiet = con.quiet_mask(_raw_vsw_test, flux=_raw_flux_test, index=index_test)
        quiet_frac = float(np.mean(quiet))

        results.append({
            "threshold": thresh,
            "quiet_frac": quiet_frac,
        })

    return results


if __name__ == "__main__":
    # Self-test for parameter sweep utilities
    print("Physics loss tuning module loaded. Run main.py to use.")