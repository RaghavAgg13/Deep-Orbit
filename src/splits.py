import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import MODELS_DIR

def chrono_split(index, train_end, val_end):
    """
    Split the dataset index chronologically into train, validation, and test sets.
    Returns boolean masks.
    """
    t_end = pd.to_datetime(train_end)
    v_end = pd.to_datetime(val_end)
    
    # Handle timezone mismatch if the index is timezone-aware
    if index.tz is not None:
        if t_end.tz is None:
            t_end = t_end.tz_localize(index.tz)
        else:
            t_end = t_end.tz_convert(index.tz)
            
        if v_end.tz is None:
            v_end = v_end.tz_localize(index.tz)
        else:
            v_end = v_end.tz_convert(index.tz)
            
    train_mask = index < t_end
    val_mask = (index >= t_end) & (index < v_end)
    test_mask = index >= v_end
    return train_mask, val_mask, test_mask

def scale_features(X, train_mask, val_mask, test_mask, scaler_path=None):
    """
    Fit a StandardScaler on the training data only and apply it to all splits.
    This prevents feature leakage.
    Saves the scaler to scaler_path if provided.
    """
    scaler = StandardScaler()
    
    # Fit only on training data
    scaler.fit(X[train_mask])
    
    # Transform all splits, wrapping them back into DataFrames with correct column names and indices
    X_train_scaled = pd.DataFrame(
        scaler.transform(X[train_mask]),
        columns=X.columns,
        index=X[train_mask].index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X[val_mask]),
        columns=X.columns,
        index=X[val_mask].index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X[test_mask]),
        columns=X.columns,
        index=X[test_mask].index
    )
    
    # Save the scaler if path is specified
    if scaler_path is not None:
        joblib.dump(scaler, scaler_path)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
