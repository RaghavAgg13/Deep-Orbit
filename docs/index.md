# bah2026_hackathon Codebase Documentation Suite

This documentation suite provides a comprehensive, deep dive into the `bah2026_hackathon` (stacking-tcn-review branch) repository. It details the architecture, algorithms, data engineering, and modeling ladder used for physics-informed space weather forecasting.

## Table of Contents

### [Chapter 1: System Architecture & Data Flow](chapter1.md)
*   The "Modeling Ladder" (R0 - R8) approach.
*   Data ingestion and dependency management.
*   The orchestrator (`main.py`) and pipeline flow.
*   Invariants (Anti-Leakage and Delta-Flux).

### [Chapter 2: Data Engineering & Feature Construction](chapter2.md)
*   Primary solar wind drivers ($V_{sw}$, $N_{sw}$, $B_z$, $P_{dyn}$).
*   Engineered physical features (Viscous proxies, coupling functions).
*   Cumulative integrals and time-since-event state variables.
*   Wavelet spectrum augmentation.
*   The segment-safe sequence builder.

### [Chapter 3: The Flat Modeling Ladder (R0 - R4)](chapter3.md)
*   **R0a/0b**: Persistence and Climatology baselines.
*   **R1**: Lagged Linear Regression.
*   **R2/R3**: Single-driver and multi-driver discrete Ridge filters (the linear physics backbone).
*   **R4**: LightGBM tree regressor with storm-sample weighting.

### [Chapter 4: Sequence Encoders (R5)](chapter4.md)
*   Deep-learning feature extraction directly from driver histories.
*   **LSTMForecaster**: Multi-layer recurrent encoder with temporal attention.
*   **TCNForecaster**: Causal 1D convolutions with exponential dilation.
*   Shared training mechanics: Delta-flux targets, hazard-weighted MSE, smooth-correlation early stopping, and physics regularization.
*   Optuna Hyperparameter Optimization (`hpo.py`).

### [Chapter 5: Meta-Learning, Ensembling & Physics (R6 - R8)](chapter5.md)
*   **R6**: Physics-Informed Residual Hybrid (Linear backbone + ML corrector).
*   **R7**: Stacking Meta-Learner (dynamic weighting of R3, R4, R5 based on magnetospheric state and polynomial interactions).
*   **R8**: Hard physical post-hoc corrector (Quiet-time monotonicity and diffusion rate bounds).

### [Chapter 6: Evaluation & Metrics](chapter6.md)
*   The unified `Forecaster` harness.
*   Continuous metrics (PE, RMSE, Correlation).
*   Event-based alert metrics for hazard thresholds (POD, FAR, HSS).
