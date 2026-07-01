import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.config import RAW_DIR, INTERIM_DIR, PROC_DIR, MODELS_DIR, REPORTS_DIR, CADENCE, HORIZON_STEPS, SEED
from src.io_cdf import read_cdf_directory
from src.clean import clean_goes_data, clean_wind_data, clean_grasp_data
from src.align import align_and_merge
from src.features import add_features
from src.dataset import build_xy
from src.splits import chrono_split, scale_features

def run_preprocessing_pipeline(propagate=True, K_steps=96, raw_dir=None, interim_dir=None, proc_dir=None):
    """
    Execute the data preprocessing pipeline from raw CDF files to processed features.
    
    1. Read raw CDF files from data/raw/ (or custom raw_dir)
    2. Clean GOES and Wind data
    3. Resample and align onto 15-min common grid with L1 time propagation
    4. Compute physical features and lags
    5. Save final features to data/processed/features.parquet (or custom proc_dir)
    """
    print("--- Starting Data Preprocessing Pipeline ---")
    
    # Define default directories if not provided
    r_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    i_dir = Path(interim_dir) if interim_dir is not None else INTERIM_DIR
    p_dir = Path(proc_dir) if proc_dir is not None else PROC_DIR
    
    # Ensure directories exist
    i_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)
    
    # Define CDF variable maps
    # Wind SWE: WI_H1_SWE (Proton speed and density)
    # Wind MFI: WI_H0_MFI (Magnetic field, GSM coordinates)
    # GOES-15: GOES15_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN
    
    # We will map column names to the variables in our downloaded CDFs
    goes_vars = {"e_flux": "e_flux", "p_flux": "p_flux"} # standard columns in our downloaded/generated CDFs
    swe_vars = {"v_sw": "v_sw", "n_sw": "n_sw"}
    mfi_vars = {"bz": "bz"}
    
    # 1. Read data from raw directories
    print(f"Reading raw CDF files from: {r_dir}")
    df_goes = read_cdf_directory(r_dir / "goes", time_var="Epoch", var_map=goes_vars)
    df_swe = read_cdf_directory(r_dir / "wind_swe", time_var="Epoch", var_map=swe_vars)
    df_mfi = read_cdf_directory(r_dir / "wind_mfi", time_var="Epoch", var_map=mfi_vars)
    
    if df_goes.empty or df_swe.empty or df_mfi.empty:
        raise ValueError("One or more raw datasets are empty. Please check your data downloads.")
        
    print(f"Loaded raw records - GOES: {len(df_goes)}, Wind SWE: {len(df_swe)}, Wind MFI: {len(df_mfi)}")
    
    # 2. Clean data
    print("Cleaning and despiking data...")
    df_goes_clean = clean_goes_data(df_goes, flux_col="e_flux", proton_col="p_flux")
    
    # Wind SWE and MFI cleaning
    df_swe_clean = clean_wind_data(df_swe, speed_col="v_sw", density_col="n_sw", bz_col=None)
    df_mfi_clean = clean_wind_data(df_mfi, speed_col=None, density_col=None, bz_col="bz")
    
    # Save clean interim parquet files
    df_goes_clean.to_parquet(i_dir / "goes_clean.parquet")
    df_swe_clean.to_parquet(i_dir / "wind_swe_clean.parquet")
    df_mfi_clean.to_parquet(i_dir / "wind_mfi_clean.parquet")
    
    # 3. Resample and Align
    print("Aligning and merging datasets onto a 15-minute grid...")
    df_merged = align_and_merge(
        df_goes_clean, df_swe_clean, df_mfi_clean,
        cadence=CADENCE, propagate=propagate
    )
    df_merged.to_parquet(i_dir / "merged.parquet")
    print(f"Aligned dataset size: {len(df_merged)} rows. Gaps handled.")
    
    # 4. Feature Engineering
    print("Engineering features and lags...")
    df_features = add_features(df_merged, K_steps=K_steps)
    
    # Save processed features
    features_path = p_dir / "features.parquet"
    df_features.to_parquet(features_path)
    print(f"Features saved to {features_path}. Shape: {df_features.shape}")
    
    return df_features

def prepare_pipeline_data(df_features, train_end="2016-12-31", val_end="2017-06-30"):
    """
    Constructs train/validation/test arrays from engineered features.
    Applies StandardScaling safely.
    """
    print("\n--- Preparing Model Datasets ---")
    
    # 1. Build X, y matrices based on the contract
    X, y, index = build_xy(df_features, target_col="log_flux", horizons=HORIZON_STEPS)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # 2. Chronological Split
    train_mask, val_mask, test_mask = chrono_split(index, train_end, val_end)
    print(f"Split sizes - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Extract segment IDs and index for splits (needed for LSTM sequence windowing)
    segment_ids = df_features.loc[X.index, "segment_id"].values
    
    # 3. Fit scaler and scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X, train_mask, val_mask, test_mask, scaler_path=MODELS_DIR / "scaler.pkl"
    )
    
    return {
        "X_train": X_train_scaled, "y_train": y[train_mask],
        "X_val": X_val_scaled, "y_val": y[val_mask],
        "X_test": X_test_scaled, "y_test": y[test_mask],
        "index_train": index[train_mask],
        "index_val": index[val_mask],
        "index_test": index[test_mask],
        "segment_ids_train": segment_ids[train_mask],
        "segment_ids_val": segment_ids[val_mask],
        "segment_ids_test": segment_ids[test_mask],
        "scaler": scaler,
        "raw_X": X,
        "raw_y": y,
        "raw_index": index
    }

def run_grasp_preprocessing_pipeline(propagate=True, K_steps=96):
    """
    Ingests raw GRASP/GSAT data and aligns it with Wind SWE/MFI drivers.
    Computes features using GRASP flux lags instead of GOES flux lags.
    """
    print("--- Starting GRASP Data Preprocessing Pipeline ---")
    grasp_vars = {"e_flux": "e_flux"}
    swe_vars = {"v_sw": "v_sw", "n_sw": "n_sw"}
    mfi_vars = {"bz": "bz"}
    
    df_grasp = read_cdf_directory(RAW_DIR / "grasp", time_var="Epoch", var_map=grasp_vars)
    df_swe = read_cdf_directory(RAW_DIR / "wind_swe", time_var="Epoch", var_map=swe_vars)
    df_mfi = read_cdf_directory(RAW_DIR / "wind_mfi", time_var="Epoch", var_map=mfi_vars)
    
    if df_grasp.empty:
        print("Warning: GRASP dataset is empty.")
        return pd.DataFrame()
        
    df_grasp["p_flux"] = 0.0 # Add dummy proton flux to match GOES feature signatures
    df_grasp_clean = clean_grasp_data(df_grasp, flux_col="e_flux")
    df_swe_clean = clean_wind_data(df_swe, speed_col="v_sw", density_col="n_sw", bz_col=None)
    df_mfi_clean = clean_wind_data(df_mfi, speed_col=None, density_col=None, bz_col="bz")
    
    df_merged = align_and_merge(
        df_grasp_clean, df_swe_clean, df_mfi_clean,
        cadence=CADENCE, propagate=propagate
    )
    
    df_features = add_features(df_merged, K_steps=K_steps)
    return df_features

