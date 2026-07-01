import numpy as np

def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def prediction_efficiency(y_true, y_pred):
    """
    Compute Prediction Efficiency (equivalent to R^2).
    PE = 1 - MSE / Var(y_true)
    PE = 1 is a perfect forecast, PE = 0 is equivalent to predicting the mean, PE < 0 is worse.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def correlation(y_true, y_pred):
    """Compute Pearson linear correlation coefficient."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]

def to_events(log_flux, hazard_log=3.0):
    """Convert continuous log flux to binary events (1 if >= threshold, else 0)."""
    return (log_flux >= hazard_log).astype(int)

def pod(yt_bin, yp_bin):
    """Compute Probability of Detection (POD) = hits / (hits + misses)."""
    hits = np.sum((yt_bin == 1) & (yp_bin == 1))
    misses = np.sum((yt_bin == 1) & (yp_bin == 0))
    denom = hits + misses
    if denom == 0:
        return 1.0 if np.sum(yp_bin == 1) == 0 else 0.0
    return float(hits) / denom

def far(yt_bin, yp_bin):
    """Compute False Alarm Rate (FAR) = false_alarms / (hits + false_alarms)."""
    hits = np.sum((yt_bin == 1) & (yp_bin == 1))
    false_alarms = np.sum((yt_bin == 0) & (yp_bin == 1))
    denom = hits + false_alarms
    if denom == 0:
        return 0.0
    return float(false_alarms) / denom

def hss(yt_bin, yp_bin):
    """
    Compute Heidke Skill Score (HSS).
    Measures forecast skill relative to random chance.
    HSS = 1 is perfect, HSS = 0 is no skill, HSS < 0 is worse than chance.
    """
    a = np.sum((yt_bin == 1) & (yp_bin == 1)) # hits
    b = np.sum((yt_bin == 0) & (yp_bin == 1)) # false alarms
    c = np.sum((yt_bin == 1) & (yp_bin == 0)) # misses
    d = np.sum((yt_bin == 0) & (yp_bin == 0)) # correct negatives
    n = a + b + c + d
    if n == 0:
        return 0.0
    
    # Expected correct by chance
    exp = float((a + c) * (a + b) + (b + d) * (c + d)) / n
    num = (a + d) - exp
    den = n - exp
    if den == 0:
        return 0.0
    return num / den
