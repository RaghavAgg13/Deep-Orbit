import numpy as np
import pandas as pd
from src.metrics import rmse, prediction_efficiency, correlation, to_events, pod, far, hss
from src.config import HAZARD_THRESHOLD_LOG

class Forecaster:
    """Base class for all forecasting models in the migration ladder."""
    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError("Subclasses must implement fit")
        
    def predict(self, X):
        """
        Generate predictions for all horizons.
        Args:
            X: np.ndarray (n_samples, n_features) or pd.DataFrame
        Returns:
            y_pred: np.ndarray (n_samples, n_horizons)
        """
        raise NotImplementedError("Subclasses must implement predict")

def evaluate_model(model, X, y, hazard_log=HAZARD_THRESHOLD_LOG, **predict_kw):
    """
    Evaluate a model on a given dataset across all horizons.
    Returns:
        list of dicts, each dict containing metrics for a single horizon.
    """
    pred = model.predict(X, **predict_kw)
    n_horizons = y.shape[1]
    results = []
    
    for h in range(n_horizons):
        y_true_h = y[:, h]
        y_pred_h = pred[:, h]
        
        # Binary event conversion for threshold metrics
        yt_bin = to_events(y_true_h, hazard_log)
        yp_bin = to_events(y_pred_h, hazard_log)
        
        results.append({
            "horizon_idx": h,
            "rmse": rmse(y_true_h, y_pred_h),
            "pe": prediction_efficiency(y_true_h, y_pred_h),
            "corr": correlation(y_true_h, y_pred_h),
            "pod": pod(yt_bin, yp_bin),
            "far": far(yt_bin, yp_bin),
            "hss": hss(yt_bin, yp_bin)
        })
        
    return results

def run_benchmark(models_dict, X_test, y_test, hazard_log=HAZARD_THRESHOLD_LOG):
    """
    Evaluate multiple models and compile the results into a tidy DataFrame.
    """
    all_results = []
    for name, model in models_dict.items():
        eval_results = evaluate_model(model, X_test, y_test, hazard_log)
        for row in eval_results:
            row["model"] = name
            all_results.append(row)
            
    df_res = pd.DataFrame(all_results)
    # Reorder columns for presentation
    cols = ["model", "horizon_idx", "pe", "corr", "rmse", "pod", "far", "hss"]
    return df_res[cols]
