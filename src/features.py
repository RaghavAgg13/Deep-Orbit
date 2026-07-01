import numpy as np
import pandas as pd

# Cross-driver + storm-phase feature set (Tier 2.4). These are physical
# derivatives / integrals / time-since-event features that the flat models
# (LightGBM, Multi-Filter, stacking) can consume directly. All are computed
# on raw (non-flux) drivers, so they introduce zero target leakage.
#   dVsw/dt, dBz/dt, dPdyn/dt              (flux variation rate sensitivity)
#   cumsum of positive V×Bz_s over 6/12/24h (energy input integrated)
#   cumsum of (Vsw-500)+ over 6/12/24h      ("high-speed stream" duration)
#   V×Pdyn, Bz_s×Pdyn                       (cross-driver coupling)
#   hours_since_Bz_flipped                   (time since last reconnection switch)
#   hours_since_Vsw>500                      (HSS age)
#   clock_angle × pdyn, sin4 × pdyn         (compound coupling proxies)
#   1h / 6h / 12h lagged driver deltas      (storm arrival timing markers)
_N_CROSS_DRIVER = 37  # total new columns below — used for logging


def add_features(df, K_steps=96):
    """
    Construct all physical features and history lags from the merged DataFrame.

    Args:
        with index sorted by time
        K_steps: int, to generate for each driver (default 96 steps = 24h at 15-min cadence)
    """
    df_feat = df.copy()

    # 1. Southward Bz: clip to negative values, absolute value (couples energy to magnetosphere)
    df_feat["bz_s"] = df_feat["bz"].clip(upper=0.0).abs()

    # 2. Dynamic pressure: Pdyn = 1.6726e-6 * Nsw * Vsw^2
    df_feat["pdyn"] = 1.6726e-6 * df_feat["n_sw"] * (df_feat["v_sw"] ** 2)

    # 3. Electric field proxy: Vsw * Bz_s
    df_feat["vbz"] = df_feat["v_sw"] * df_feat["bz_s"]

    # 4. IMF Clock Angle: derived from Bz sign only (no By component is
    # ingested, per the program-task data spec). Northward Bz -> +pi/2,
    # southward Bz -> -pi/2. Reconnection efficiency is modulated by
    # sin^4(theta/2); more negative Bz + smaller clock angle => strongest
    # energy transfer.
    df_feat["clock_angle"] = np.where(df_feat["bz"].values >= 0, np.pi / 2, -np.pi / 2)
    df_feat["sin4_theta2"] = np.sin(df_feat["clock_angle"].values / 2.0) ** 4

    # 5. Sckopke / epsilon coupling function (energy transfer rate proxy):
    # epsilon ~ Vsw * B^2 * sin^4(theta/2)
    # Use total |B| when available; else approximate with |Bz|.
    if "bmag" in df_feat.columns:
        b_tot = df_feat["bmag"].values
    else:
        b_tot = df_feat["bz"].abs().values
    df_feat["sckopke"] = (df_feat["v_sw"].values *
                          (b_tot ** 2) *
                          df_feat["sin4_theta2"].values)

    # 6. Viscous interaction proxy: Vsw^(1/3) * Nsw^(1/2) (shear-driven boundary layer)
    df_feat["viscous_proxy"] = (df_feat["v_sw"].clip(lower=0.0).values ** (1.0/3.0)) * \
                               (df_feat["n_sw"].clip(lower=0.0).values ** 0.5)

    # 7. Seasonal tilt: Earth's dipole tilt relative to solar wind.
    # We encode cyclic day-of-year as a tilt proxy since true tilt needs full
    # geomagnetic field models (beyond data scope). Sinusoid with annual period.
    # Peak coupling during equinoxes (Russell-McPherron effect).
    doy_rad = 2.0 * np.pi * df_feat.index.dayofyear / 365.25
    df_feat["tilt_sin"] = np.sin(doy_rad)          # seasonal modulation
    df_feat["tilt_cos"] = np.cos(doy_rad)

    # 8. Diurnal tilt (UT-dependent modulation of coupling angle)
    hod = df_feat.index.hour + df_feat.index.minute / 60.0
    df_feat["tilt_ut_sin"] = np.sin(2.0 * np.pi * hod / 24.0)
    df_feat["tilt_ut_cos"] = np.cos(2.0 * np.pi * hod / 24.0)

    # 9. Vsw history integrals: capture cumulative energy input over recent hours.
    # Encode average speed above storm threshold (500 km/s) over the recent 12h window.
    # Segment-aware rolling prevents cross-gap averaging.
    df_feat["vsw_above_thresh"] = (df_feat["v_sw"].clip(lower=0.0) - 500.0).clip(lower=0.0)
    df_feat["vsw_roll6h"] = df_feat["v_sw"].groupby(df_feat["segment_id"]).rolling(window=24, min_periods=1).mean().droplevel(0)
    df_feat["vsw_roll12h"] = df_feat["v_sw"].groupby(df_feat["segment_id"]).rolling(window=48, min_periods=1).mean().droplevel(0)

    # 10. Generation of segment-aware lags for all driver channels, including new ones.
    # Grouping by segment_id ensures shift returns NaN at gap boundaries.
    #
    # Memory note: K_steps=96 with 7 drivers + flux = 768 lag columns × 2M rows
    # = ~12 GB. We build and concat in chunks (one driver at a time) so the
    # intermediate dict never holds all 768 Series simultaneously. main.py's
    # ml_features filter then drops ~670 of these columns, so the final
    # feature matrix is much smaller (~3 GB).
    print(f"Generating segment-aware lags up to K={K_steps} steps...")

    driver_bases = [
        ("v_sw", "v_lag_"),
        ("bz_s", "bz_lag_"),
        ("pdyn", "pdyn_lag_"),
        ("clock_angle", "clock_lag_"),
        ("sin4_theta2", "sin4_lag_"),
        ("sckopke", "sckopke_lag_"),
        ("viscous_proxy", "visc_lag_"),
    ]
    lag_frames = []
    for base, prefix in driver_bases:
        if base not in df_feat.columns:
            continue
        grp = df_feat.groupby("segment_id")[base]
        lag_cols = {}
        for lag in range(1, K_steps + 1):
            lag_cols[f"{prefix}{lag}"] = grp.shift(lag)
        lag_frames.append(pd.DataFrame(lag_cols, index=df_feat.index))
        del lag_cols  # release this driver's 96 Series before the next driver

    # Past flux history (autocorrelation) — separate chunk
    flux_lags = {}
    for lag in range(1, K_steps + 1):
        flux_lags[f"flux_lag_{lag}"] = df_feat.groupby("segment_id")["log_flux"].shift(lag)
    lag_frames.append(pd.DataFrame(flux_lags, index=df_feat.index))
    del flux_lags

    df_feat = pd.concat([df_feat, *lag_frames], axis=1)
    del lag_frames

    # 11. Hour-of-day cyclic features (GMT time index)
    df_feat["hod_sin"] = np.sin(2.0 * np.pi * hod / 24.0)
    df_feat["hod_cos"] = np.cos(2.0 * np.pi * hod / 24.0)

    # 12. Day-of-year cyclic features
    doy = df_feat.index.dayofyear
    df_feat["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df_feat["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # 13. GOES generation flag15/old vs GOES-16/new break)
    if "source" in df_feat.columns:
        df_feat["goes_new"] = df_feat["source"].astype(str).str.extract(r"(\d+)").fillna(15).astype(int)
        df_feat["goes_new"] = (df_feat["goes_new"] >= 16).astype(int)
    else:
        df_feat["goes_new"] = 0

    # ------------------------------------------------------------------ #
    # 14. Tier 2.4 — cross-driver + storm-phase enrichment.
    # These columns are PURELY driver-derived_sw / bz / pdyn / sin4 /
    # sckopke), so they carry zero flux-history leakage and flow to every
    # rung that ingests the feature matrix. Column count: ~37.
    # ------------------------------------------------------------------ #
    # Instantaneous 1-step (15-min) and 1-hour (4-step) driver derivatives.
    # dVsw/dt captures onset acceleration; dBz/dt captures southward turning
    # rate (substorm trigger); dPdyn/dt captures compression onset.
    dv_sw = df_feat["v_sw"]
    dbz_s = df_feat["bz_s"]
    dpdy = df_feat["pdyn"]
    g = df_feat.groupby("segment_id")
    df_feat["dVsw_dt"] = g["v_sw"].diff(1)
    df_feat["d2Vsw"] = g["dVsw_dt"].diff(1)              # accel of accel
    df_feat["dBz_dt"] = g["bz_s"].diff(1)
    df_feat["dPdyn_dt"] = g["pdyn"].diff(1)
    df_feat["dVsw_dt_1h"] = g["v_sw"].diff(4)
    df_feat["dBz_dt_1h"] = g["bz_s"].diff(4)
    df_feat["dPdyn_dt_1h"] = g["pdyn"].diff(4)

    # High-speed stream (Vsw > 500) its recent duration in hours.
    # Segment-aware rolling: groupby("segment_id") prevents the rolling sum
    # from crossing data gaps (where v_sw is NaN and segment_id increments).
    vsw_pos = (df_feat["v_sw"].clip(lower=0.0) - 500.0).clip(lower=0.0)
    df_feat["vsw_above_500"] = vsw_pos
    vsw_gt500 = (df_feat["v_sw"] > 500).astype(float)
    df_feat["vsw_gt500_duration_6h"] = (
        vsw_gt500.groupby(df_feat["segment_id"]).rolling(24, min_periods=1).sum().droplevel(0) * 0.25)
    df_feat["vsw_gt500_duration_12h"] = (
        vsw_gt500.groupby(df_feat["segment_id"]).rolling(48, min_periods=1).sum().droplevel(0) * 0.25)
    df_feat["vsw_gt500_duration_24h"] = (
        vsw_gt500.groupby(df_feat["segment_id"]).rolling(96, min_periods=1).sum().droplevel(0) * 0.25)

    # Cumulative energy input proxies. Positive V×Bz_s = reconnection
    # energy coupling; integrate over 6h/12h/24h to capture the storm's
    # total energy budget — a well-known predictor of MeV electron flux.
    # Segment-aware rolling prevents summing across data gaps.
    vbz_pos = df_feat["vbz"].clip(lower=0.0)
    df_feat["cum_vbz_6h"] = vbz_pos.groupby(df_feat["segment_id"]).rolling(24, min_periods=1).sum().droplevel(0)
    df_feat["cum_vbz_pos_12h"] = vbz_pos.groupby(df_feat["segment_id"]).rolling(48, min_periods=1).sum().droplevel(0)
    df_feat["cum_vbz_pos_24h"] = vbz_pos.groupby(df_feat["segment_id"]).rolling(96, min_periods=1).sum().droplevel(0)

    # Positive Pdyn integral = cumulative magnetospheric compression.
    pdyn_pos = dpdy.clip(lower=0.0)
    df_feat["cum_pdyn_pos_6h"] = pdyn_pos.groupby(df_feat["segment_id"]).rolling(24, min_periods=1).sum().droplevel(0)
    df_feat["cum_pdyn_pos_12h"] = pdyn_pos.groupby(df_feat["segment_id"]).rolling(48, min_periods=1).sum().droplevel(0)

    # Cross-driver compound proxies (not simple products of existing bases):
    for a, b, lab in [
        ("v_sw", "pdyn", "v_pdyn"),
        ("bz_s", "pdyn", "bz_pdyn"),
        ("sin4_theta2", "pdyn", "sin4_pdyn"),
        ("sckopke", "pdyn", "sckopke_pdyn"),
    ]:
        df_feat[lab] = df_feat[a] * df_feat[b]

    # Time-since-event markers. These encode WHERE in the storm cycle we
    # are: hours since Bz last switched sign (reconnection onset) and hours
    # since Vsw last exceeded 500 km/s (HSS arrival).
    bz_sign = np.sign(df_feat.bz.replace(0, np.nan))
    df_feat["hours_since_bz_flip"] = _steps_since_event(
        bz_sign, fillna_val=999.0, segment_ids=df_feat["segment_id"])
    df_feat["hours_since_vsw_gt500"] = _steps_since_event(
        (df_feat["v_sw"] > 500).astype(float).mask(df_feat["v_sw"] <= 500),
        fillna_val=999.0, segment_ids=df_feat["segment_id"])

    # 1h / 6h / 12h lagged driver deltas (storm arrival timing markers).
    for d, base in [("v_sw", "dvsw_6h"), ("bz_s", "dbz_6h"), ("pdyn", "dpdyn_6h")]:
        df_feat[base] = df_feat[d] - g[d].shift(24)
    for d, base in [("v_sw", "dvsw_12h"), ("bz_s", "dbz_12h"), ("pdyn", "dpdyn_12h")]:
        df_feat[base] = df_feat[d] - g[d].shift(48)
    for d, base in [("v_sw", "dvsw_24h"), ("bz_s", "dbz_24h"), ("pdyn", "dpdyn_24h")]:
        df_feat[base] = df_feat[d] - g[d].shift(96)

    return df_feat


def _steps_since_event(s: pd.Series, fillna_val: float = 999.0,
                       segment_ids: pd.Series | None = None) -> pd.Series:
    """For each row, count the number of consecutive rows (backwards) since
    the last non-NAN/True event in ``s``. Returns hours (steps * 0.25).

    Rows with no preceding event get ``fillna_val``.
    When ``segment_ids`` is provided, counters reset at segment boundaries
    so the "time since last event" does not cross data gaps.
    """
    if segment_ids is not None:
        # Apply per segment and concatenate — prevents cross-gap counting.
        parts = []
        for _sid, grp_s in s.groupby(segment_ids):
            parts.append(_steps_since_event(grp_s, fillna_val=fillna_val,
                                            segment_ids=None))
        return pd.concat(parts)

    s_filled = s.fillna(0.0)
    is_event = s_filled.astype(bool).astype(int)
    # cumcount-like: count consecutive non-events since last event.
    # Group on the cumulative sum of the event flag.
    groups = is_event.cumsum()
    grp_count = groups.groupby(groups).cumcount()
    out = grp_count * 0.25  # convert 15-min steps to hours
    # Rows where no event has ever occurred get fillna_val.
    never = (groups == 0) & (is_event == 0)
    out = out.where(~never, other=np.where(never, fillna_val, 0.0))
    return out
