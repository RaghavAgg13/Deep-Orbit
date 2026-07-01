import os
from pathlib import Path

# Random seed for reproducibility
SEED = 42

# Project root path.
# Resolve to the repository root (two levels up from src/) so the pipeline
# works regardless of where the checkout lives, instead of a hardcoded
# developer-desktop path.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directory paths
RAW_DIR = ROOT_DIR / "data/raw"
INTERIM_DIR = ROOT_DIR / "data/interim"
PROC_DIR = ROOT_DIR / "data/processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for d in [RAW_DIR, INTERIM_DIR, PROC_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cadence and target horizons
# Resampling to 15-minute grid to allow exact 30-45 min forecasting
CADENCE = "15min"

# Prediction horizons in minutes: 30 minutes, 6 hours (360 minutes), 12 hours (720 minutes)
HORIZONS_MIN = [30, 360, 720]

# Convert horizons to number of steps at 15-minute cadence
# 30 min -> 2 steps, 6h -> 24 steps, 12h -> 48 steps
HORIZON_STEPS = [m // 15 for m in HORIZONS_MIN]

# Hazard threshold: integral flux > 1000 pfu (equivalent to log10(flux) > 3)
HAZARD_THRESHOLD_RAW = 1000.0
HAZARD_THRESHOLD_LOG = 3.0 # log10(1000.0)

# Memory length (history window) in steps
# At 15-min cadence, 48 hours of history is 48 * 4 = 192 steps
SEQ_LEN = 192
