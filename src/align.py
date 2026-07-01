import numpy as np
import pandas as pd
from src.config import CADENCE

def propagate_l1_to_earth(df_wind, speed_col="v_sw", distance_km=1.5e6):
    """
    Time-shift Wind spacecraft measurements from L1 to Earth bow shock.
    Uses speed-dependent propagation: delta_t = distance / Vsw.

    Args:
        df_wind: pd.DataFrame with index sorted by time
        speed_col: str, name of solar wind speed column (km/s)
        distance_km: float, distance of L1 in km
    Returns:
        pd.DataFrame with shifted timestamps, before resampling.
    """
    df_shifted = df_wind.copy()

    # Calculate delay in seconds. Vsw is in km/s.
    # If speed is missing, use median speed (approx 400 km/s) as fallback
    median_speed = df_shifted[speed_col].median()
    if np.isnan(median_speed):
        median_speed = 400.0

    speeds = df_shifted[speed_col].fillna(median_speed)
    # Clip speeds to prevent division by zero or infinite delay
    speeds = speeds.clip(lower=200.0, upper=1200.0)

    delays_sec = distance_km / speeds

    # Shift index timestamps
    # df_shifted.index is DatetimeIndex
    new_times = df_shifted.index + pd.to_timedelta(delays_sec, unit="s")
    df_shifted.index = new_times
    return df_shifted.sort_index()

def align_and_merge(df_goes, df_wind_swe, df_wind_mfi, cadence=CADENCE, propagate=True):
    """
    Align and merge GOES and Wind data streams onto a common time grid.

    1. Optionally propagates Wind SWE/MFI data to Earth.
    2. Resamples all streams to target cadence using mean.
    3. Merges using outer join to preserve data indices.
    4. Interpolates short gaps (up to 3 steps, i.e., 45 minutes).
    5. Flags valid rows and segments.
    """
    # 1. Propagation
    if propagate:
        if "v_sw" in df_wind_swe.columns:
            # We propagate SWE using its own speed variable
            df_swe_prop = propagate_l1_to_earth(df_wind_swe, speed_col="v_sw")

            # For MFI, we can propagate it using the speed from SWE by aligning them first,
            # or simply shift MFI by the average delay of SWE.
            # Shifting MFI by average delay (approx 50 minutes) or using aligned SWE speed:
            mfi_times_shifted = df_wind_mfi.index + pd.to_timedelta(50.0, unit="m")
            df_mfi_prop = df_wind_mfi.copy()
            df_mfi_prop.index = mfi_times_shifted
            df_mfi_prop = df_mfi_prop.sort_index()
        else:
            df_swe_prop = df_wind_swe
            df_mfi_prop = df_wind_mfi
    else:
        df_swe_prop = df_wind_swe
        df_mfi_prop = df_wind_mfi

    # 2. Sort indexes if needed (cheap; only reorders when propagation or
    # upstream bugs shuffled rows).
    if not df_goes.index.is_monotonic_increasing:
        df_goes = df_goes.sort_index()
    if not df_swe_prop.index.is_monotonic_increasing:
        df_swe_prop = df_swe_prop.sort_index()
    if not df_mfi_prop.index.is_monotonic_increasing:
        df_mfi_prop = df_mfi_prop.sort_index()

    # Downsample via groupby(Grouper) instead of resample() — the
    # Resampler constructor triggers a C-level segfault on pandas 3.0.x /
    # numpy 2.4.x / Python 3.14.x even on a clean, unique, sorted index.
    # groupby(Grouper).mean() produces the same result via a safer C path.
    goes_res = df_goes.groupby(pd.Grouper(freq=cadence)).mean()
    swe_res = df_swe_prop.groupby(pd.Grouper(freq=cadence)).mean()
    mfi_res = df_mfi_prop.groupby(pd.Grouper(freq=cadence)).mean()

    # 3. Outer Join on time index
    df_merged = goes_res.join([swe_res, mfi_res], how="outer").sort_index()

    # 4. Handle short gaps (linear interpolation up to 3 steps)
    # Gaps longer than 3 steps remain NaN
    df_merged = df_merged.interpolate(method="linear", limit=3, limit_direction="both")

    # 5. Build valid mask: all necessary columns must be non-NaN
    # Required features: log_flux, v_sw, n_sw, bz
    required_cols = ["log_flux", "v_sw", "n_sw", "bz"]

    # If any required column is missing, mask is False
    existing_required = [c for c in required_cols if c in df_merged.columns]
    if len(existing_required) == len(required_cols):
        df_merged["valid"] = df_merged[required_cols].notna().all(axis=1)
    else:
        df_merged["valid"] = False

    # 6. Segment IDs to prevent rolling windows or sequences crossing gaps
    # segment_id increments at every invalid block, so contiguous valid blocks have the same segment_id
    df_merged["segment_id"] = (~df_merged["valid"]).cumsum()

    return df_merged
