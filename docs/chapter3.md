# Chapter 3: The Flat Modeling Ladder (R0 - R4)

This chapter covers the mathematical formulations and algorithms for the flat baselines and machine learning models (rungs R0 to R4).

---

## Rung 0a: Persistence (`Persistence`)

The persistence baseline assumes that the future flux at any horizon $h$ is equal to the most recent observation:
$$\hat{y}_{t+h} = y_t$$
The pipeline uses the lagged flux column `flux_lag_1` (the observation 1 step behind the current row) as the proxy for $y_t$.

### Scale Recovery
During preprocessing, `flux_lag_1` is standardized by the scaler:
$$\text{flux\_lag\_1}_{\text{scaled}} = \frac{\text{flux\_lag\_1}_{\text{raw}} - \mu}{\sigma}$$
To prevent evaluating scaled predictions against raw targets, the model reconstructs the raw scale before returning predictions:
$$\hat{y}_{t+h} = \text{flux\_lag\_1}_{\text{scaled}} \times \sigma_{\text{flux\_lag\_1}} + \mu_{\text{flux\_lag\_1}}$$
The scalar forecast is broadcast across all three horizons:
$$\hat{\mathbf{y}}_t = [\hat{y}_{t+2}, \hat{y}_{t+24}, \hat{y}_{t+48}]$$

---

## Rung 0b: Climatology (`Climatology`)

The climatology baseline predicts the historical mean log-flux conditioned on the hour-of-day.

### Hour Recovery
The continuous hour-of-day value is reconstructed from its cyclic sine and cosine encodings:
$$\phi_t = \text{atan2}(\text{hod\_sin}_t, \text{hod\_cos}_t)$$
$$\text{Hour}_t = \text{round}\left( \frac{\phi_t}{2\pi} \times 24 \right) \pmod{24}$$

### Model Fitting
During training, the model maps each hour $k \in [0, 23]$ to its mean observed log-flux:
$$\bar{y}_k = \frac{1}{|I_k|} \sum_{i \in I_k} y^{\text{raw}}_i$$
Where $I_k$ is the set of training indices where $\text{Hour}_i == k$, and $y^{\text{raw}}_i$ is the unscaled log-flux. 
The prediction is:
$$\hat{y}_{t+h} = \bar{y}_{\text{Hour}_t}$$

---

## Rung 1: Lagged Linear Regression (`LaggedLinear`)

Fits a separate univariate Weighted Least Squares (WLS) model for each horizon $h$. It identifies the single best combination of solar-wind driver and lag.

### Candidate Search
The model scans three primary drivers: speed (`v_sw`), southward magnetic field (`bz_s`), and dynamic pressure (`pdyn`). It searches across all lags $j \in [1, 96]$:
$$\text{Candidates} = \{ \mathbf{x}_{j} = \text{driver\_lag\_}j \}$$

### Weighted Pearson Correlation
For each candidate feature $\mathbf{x}$, it computes the weighted Pearson correlation $r_w$ with the target $\mathbf{y}_h$ using sample weights $\mathbf{w}$:
$$\mu_{x} = \frac{\sum w_i x_i}{\sum w_i}, \quad \mu_{y} = \frac{\sum w_i y_i}{\sum w_i}$$
$$\text{Cov}_w(x, y) = \sum w_i (x_i - \mu_x)(y_i - \mu_y)$$
$$\text{Var}_w(x) = \sum w_i (x_i - \mu_x)^2, \quad \text{Var}_w(y) = \sum w_i (y_i - \mu_y)^2$$
$$r_w = \frac{\text{Cov}_w(x, y)}{\sqrt{\text{Var}_w(x) \text{Var}_w(y)}}$$
The candidate feature with the highest absolute weighted correlation $|r_w|$ is selected.

### Weighted Least Squares (WLS)
With the selected feature $\mathbf{x}^*$, the parameters are computed by solving the WLS normal equations:
$$\beta = \frac{\sum w_i (x^*_i - \mu_{x^*})(y_i - \mu_y)}{\sum w_i (x^*_i - \mu_{x^*})^2}$$
$$\alpha = \mu_y - \beta \mu_{x^*}$$
The prediction is:
$$\hat{y}_{t+h} = \beta x^*_t + \alpha$$

---

## Rungs 2 & 3: Discrete Linear Filters

These rungs model the forecast as a discrete linear impulse-response function.

### Rung 2: Speed-Driven Linear Filter (`LinearFilter`)
Ridge regression is fit over the solar wind speed lag profile:
$$\hat{y}_{t+h} = w_0 V_{sw, t} + \sum_{j=1}^{96} w_j V_{sw, t-j} + b$$
This results in 97 parameters per horizon.

### Rung 3: Multi-Driver Filter (`MultiFilter`)
Extends the linear filter to incorporate seven drivers:
$$\hat{y}_{t+h} = \sum_{d \in \text{Drivers}} \left( w_{d, 0} x_{d, t} + \sum_{k \in \text{Schedule}} w_{d, k} x_{d, t-k} \right) + b$$

To manage the feature space, the lags are downsampled using an exponential schedule:
$$\text{Schedule} = [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96]$$
This reduces the lag feature count from 96 to 13 per driver, limiting the parameter space.

### Ridge Regularization
The parameters $\mathbf{w}$ are estimated by minimizing the L2-regularized residual sum of squares:
$$\mathcal{L}_{\text{Ridge}} = \sum_{i} w_i \left( y_i - (\mathbf{w}^T \mathbf{x}_i + b) \right)^2 + \alpha_{\text{Ridge}} \|\mathbf{w}\|_2^2$$
Where $\alpha_{\text{Ridge}} = 50.0$.

---

## Rung 4: LightGBM Regressor (`GBMForecaster`)

Rung 4 uses gradient-boosted trees to capture non-linear relationships.

### Hyperparameters
*   `learning_rate`: $0.03$
*   `num_leaves`: $51$
*   `n_estimators`: $500$
*   `early_stopping_rounds`: $50$

### Target Hazard Weighting
To focus training on high-flux intervals, a sample weight is calculated for each row. If the true flux at *any* future horizon exceeds the hazard threshold ($3.0$), the sample weight is set to $8.0$:
$$w_i = \begin{cases} 8.0 & \text{if } \max_h(y_{i, h}) \ge 3.0 \\ 1.0 & \text{otherwise} \end{cases}$$
This weighting is applied uniformly across the models for all three horizons.
Validation early stopping uses RMSE computed on the validation set.
