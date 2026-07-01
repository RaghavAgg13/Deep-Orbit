import numpy as np
import pandas as pd

def mad_despike(series, window=11, n_sigma=6):
    """
    Apply Median Absolute Deviation (MAD) filter to remove spikes.
    Outliers are replaced with NaN.
    """
    # Use rolling window to get local median and MAD
    med = series.rolling(window, center=True, min_periods=3).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=3).median()
    
    # 1.4826 scale factor matches MAD to standard deviation for normal distribution
    thresh = n_sigma * 1.4826 * mad
    # Add a tiny constant to prevent division by zero in quiet periods
    thresh = np.maximum(thresh, 1e-6)
    
    # Keep values within threshold, replace others with NaN
    clean_series = series.where((series - med).abs() <= thresh)
    return clean_series

def clean_goes_data(df, flux_col="e_flux", proton_col="p_flux"):
    """
    Clean GOES electron flux data.
    - Convert to log10, clipping values below 1.0.
    - Despike the log flux.
    - Mask intervals with strong solar proton event contamination.
    """
    df_clean = df.copy()
    
    if flux_col not in df_clean.columns:
        raise ValueError(f"Required column '{flux_col}' not found in GOES DataFrame")
        
    # 1. Clip and Log-transform
    df_clean["log_flux"] = np.log10(df_clean[flux_col].clip(lower=1.0))
    
    # 2. Despike log flux (operate in log space for symmetric scale)
    df_clean["log_flux"] = mad_despike(df_clean["log_flux"], window=11, n_sigma=6)
    
    # 3. Handle solar proton contamination if proton channel is present
    if proton_col in df_clean.columns:
        p_flux = df_clean[proton_col]
        # Identify strong proton events (typically fluxes exceeding SWPC threshold, e.g. 10 pfu for >10 MeV)
        # Or relative anomaly: top 99.9% percentile
        proton_thresh = p_flux.quantile(0.999)
        # Prevent threshold being set to NaN if entire series is NaN
        if not np.isnan(proton_thresh):
            proton_thresh = max(proton_thresh, 10.0) # 10 pfu is standard NOAA threshold
            proton_contamination = p_flux > proton_thresh
            # Mask electron flux during proton spikes
            df_clean.loc[proton_contamination, "log_flux"] = np.nan
            print(f"Masked {proton_contamination.sum()} rows of electron flux due to proton contamination (>{proton_thresh} pfu)")
            
    return df_clean

def clean_wind_data(df, speed_col="v_sw", density_col="n_sw", bz_col="bz"):
    """
    Clean Wind SWE and MFI data.
    - Caps speed, density and Bz to physically realistic limits.
    - Despikes the variables.
    """
    df_clean = df.copy()
    
    # 1. Apply physical caps
    if speed_col in df_clean.columns:
        # Solar wind speed is typically 200 to 1200 km/s. Outside that is instrument error.
        df_clean.loc[(df_clean[speed_col] < 200) | (df_clean[speed_col] > 1200), speed_col] = np.nan
        df_clean[speed_col] = mad_despike(df_clean[speed_col], window=11, n_sigma=6)
        
    if density_col in df_clean.columns:
        # Density cannot be negative. Solar wind density rarely exceeds 100 cm^-3.
        df_clean.loc[(df_clean[density_col] < 0) | (df_clean[density_col] > 100), density_col] = np.nan
        df_clean[density_col] = mad_despike(df_clean[density_col], window=11, n_sigma=6)
        
    if bz_col in df_clean.columns:
        # IMF Bz is typically in GSM coordinates, +/- 100 nT limit
        df_clean.loc[(df_clean[bz_col] < -100) | (df_clean[bz_col] > 100), bz_col] = np.nan
        df_clean[bz_col] = mad_despike(df_clean[bz_col], window=11, n_sigma=6)

    return df_clean

def clean_grasp_data(df, flux_col="e_flux"):
    """
    Clean GRASP/GSAT electron flux data.
    - Convert to log10, clipping values below 1.0.
    - Despike the log flux.
    """
    df_clean = df.copy()
    
    if flux_col not in df_clean.columns:
        raise ValueError(f"Required column '{flux_col}' not found in GRASP/GSAT DataFrame")
        
    # 1. Clip and Log-transform
    df_clean["log_flux"] = np.log10(df_clean[flux_col].clip(lower=1.0))
    
    # 2. Despike log flux
    df_clean["log_flux"] = mad_despike(df_clean["log_flux"], window=11, n_sigma=6)
    
    return df_clean

