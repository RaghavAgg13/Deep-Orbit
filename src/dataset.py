import pandas as pd
import numpy as np
from src.config import HORIZON_STEPS

def build_xy(df, target_col="log_flux", horizons=HORIZON_STEPS):
    """
    Construct the feature matrix X and multi-horizon target matrix y.
    
    X is the feature matrix at time t.
    y has columns corresponding to the targets shifted by -h steps (future flux).
    Only returns rows where the input features and all targets are valid (non-NaN).
    
    Returns:
        X: pd.DataFrame (n_samples, n_features)
        y: np.ndarray (n_samples, n_horizons)
        index: pd.DatetimeIndex (n_samples,)
    """
    df_copy = df.copy()
    
    # 1. Build target columns using backward shift (negative shift pulls future value to current row)
    target_cols = []
    for h in horizons:
        col_name = f"target_t_plus_{h}"
        df_copy[col_name] = df_copy[target_col].shift(-h)
        target_cols.append(col_name)
    
    # 2. Define features (exclude raw/log flux, target columns, valid mask, segment_id, source)
    exclude_cols = [
        "e_flux", "log_flux", "valid", "segment_id", "source"
    ] + target_cols
    
    feature_cols = [c for c in df_copy.columns if c not in exclude_cols]
    
    # 3. Create validity masks:
    # - No NaNs in any feature column
    # - No NaNs in any target column
    # - The valid column itself must be True
    # - The segment_id must match between current time t and future time t+h to prevent gap crossing
    valid_mask = df_copy["valid"].values.copy()
    
    # Verify no gap crossing for target window
    for h in horizons:
        # At step t, the segment_id of t must equal segment_id of t+h
        segment_t = df_copy["segment_id"]
        segment_future = df_copy["segment_id"].shift(-h)
        valid_mask &= (segment_t == segment_future)
        # Also target must not be NaN
        valid_mask &= df_copy[f"target_t_plus_{h}"].notna()
        
    valid_mask &= df_copy[feature_cols].notna().all(axis=1)
    
    # Filter rows
    df_valid = df_copy[valid_mask]
    
    X = df_valid[feature_cols]
    y = df_valid[target_cols].values # np.ndarray
    index = df_valid.index
    
    return X, y, index
