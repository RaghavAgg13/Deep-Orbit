# Chapter 1: System Architecture & Data Flow

This chapter details the overarching system architecture, directories, parameter configurations, data acquisition pipeline, and the step-by-step orchestrator flow for the ISRO Space Weather Electron Flux Forecasting Pipeline.

## 1. System Overview & The Orchestrator Flow

The forecasting pipeline is designed to predict geostationary relativistic electron flux ($>2\text{ MeV}$) at three future lead times ($h$):
*   **30 Minutes** ($2$ steps of 15-minute cadence)
*   **6 Hours** ($24$ steps of 15-minute cadence)
*   **12 Hours** ($48$ steps of 15-minute cadence)

The main pipeline orchestrator is implemented in `main.py` and coordinates the system execution across nine distinct phases.

```mermaid
graph TD
    A[Start: python main.py] --> B[Phase 1: Dependency Check & CDF Ingestion]
    B --> C[Phase 2: Preprocessing Pipeline]
    C --> D[Phase 3: Wavelet Augmentation]
    D --> E[Phase 4: Chronological Splitting & Scaling]
    E --> F[Phase 5: Sequence Building 3D]
    F --> G[Phase 6: Modeling Ladder R0 - R8 Training]
    G --> H[Phase 7: Stacking Meta-Learner R7]
    H --> I[Phase 8: Physics Clipping R8 wrapper]
    I --> J[Phase 9: Unified Evaluation Harness & Diagnostics]
```

### The 9-Step Pipeline Execution Flow
1.  **Dependency Checking & Installation**: Scans the Python path for core, sequence, and optimization dependencies. Auto-installs missing packages from PyPI via `subprocess` if the `--install-deps` flag is active.
2.  **Data Ingestion**: Downloads solar-wind measurements and target geostationary electron flux from the NASA CDAWeb database for the years 2013–2016.
3.  **Preprocessing Pipeline**: Merges, despikes, and resamples GOES and Wind data streams onto a uniform 15-minute time grid. Shifts L1 observations to Earth using speed-dependent propagation.
4.  **Wavelet Transform Augmentation**: Automatically extracts sub-band energy components from the solar-wind drivers using a Discrete Wavelet Transform (DWT), appending features without target leakage.
5.  **Chronological Split & Scale**: Segments data into Train (2013-2015), Validation (H1 2016), and Test (H2 2016) splits. Fits a `StandardScaler` strictly on the train features and applies it to the validation and test sets.
6.  **3D Sequence Generation**: Constructs 3D temporal sequence arrays of size `(batch, seq_len, features)` (with `seq_len=192` steps or 48 hours) for the sequence models. Gaps are managed using a `segment_id` identifier.
7.  **Model Migration Ladder Training**: Sequentially trains and checkpoints each model level (R0 through R8) to the `models/` directory, enabling intermediate execution resuming using `--resume-from`.
8.  **Stacking Meta-Learner (Regime Blending)**: Re-fits base forecasters and trains a Ridge-based meta-blender that uses physical solar wind conditions as features to dynamically weight the baseline, flat, and sequence models.
9.  **Verification, Metrics & Diagnostics**: Compiles predictions, computes continuous and event-based performance metrics, generates benchmark summaries, and writes diagnostic plots.

---

## 2. Ingestion and Cleaning Pipeline (`src/clean.py` & `src/align.py`)

Raw CDF datasets contain instrumental gaps, cosmic ray noise spikes, and solar proton event (SPE) contamination. The data ingestion functions resolve these issues prior to feature engineering.

### Spike Removal: Median Absolute Deviation (MAD)
Spikes are identified using a rolling Median Absolute Deviation filter via `mad_despike()`. For a window size $W=11$ and threshold factor $K=6$:
$$\text{Median}_t = \text{median}(x_{t-W/2}, \dots, x_{t+W/2})$$
$$\text{MAD}_t = \text{median}(|x_{t-W/2} - \text{Median}_t|, \dots, |x_{t+W/2} - \text{Median}_t|)$$
$$\text{Threshold}_t = 6 \times 1.4826 \times \text{MAD}_t$$
Any value exceeding the threshold is marked as an outlier and replaced with `NaN`:
$$x_t = \begin{cases} x_t & \text{if } |x_t - \text{Median}_t| \le \text{Threshold}_t \\ \text{NaN} & \text{otherwise} \end{cases}$$
A safety floor of $10^{-6}$ is added to $\text{Threshold}_t$ to prevent division-by-zero during quiet periods.

### Proton Contamination Masking
Geostationary solid-state electron detectors are sensitive to high-energy protons. During Solar Proton Events (SPEs), MeV protons penetrate the electron channels, inflating the measured flux.
1.  The pipeline extracts the proton flux channel (`p_flux`).
2.  It determines the $99.9\%$ percentile threshold, defaulting to the NOAA standard threshold of $10\text{ pfu}$ (particle flux units) for $>10\text{ MeV}$ protons:
    $$\theta_{proton} = \max(\text{quantile}(p\_flux, 0.999), 10.0)$$
3.  If $p\_flux_t > \theta_{proton}$, the electron log-flux at time $t$ is replaced with `NaN`.

### L1 Spacecraft-to-Earth Propagation
The Wind spacecraft orbits at the L1 Lagrange point, approximately $1.5 \times 10^6\text{ km}$ upstream from Earth. The solar wind measurements must be time-shifted to represent conditions at the Earth's bow shock:
$$\Delta t_t = \frac{D}{V_{sw, t}}$$
Where $D = 1.5 \times 10^6\text{ km}$ and $V_{sw, t}$ is the solar wind speed in $\text{km/s}$ clamped between $200\text{ km/s}$ and $1200\text{ km/s}$. If $V_{sw}$ is missing, the training median (approx. $400\text{ km/s}$) is used as a fallback. 
The shifted timestamps are calculated as:
$$T_{\text{Earth}} = T_{\text{L1}} + \Delta t_t$$
After shifting, Wind SWE and MFI parameters are aligned with GOES-15 geostationary observations.

---

## 3. Data Alignment and Gaps (`src/align.py`)

1.  **Resampling**: Grouping is performed using `pd.Grouper(freq="15min")` to calculate the mean value over each interval, avoiding pandas downsampling issues.
2.  **Merging**: The streams are aligned using an outer join on the time index:
    ```python
    df_merged = goes_res.join([swe_res, mfi_res], how="outer").sort_index()
    ```
3.  **Interpolation**: Small gaps are filled using linear interpolation up to a limit of 3 steps ($45\text{ minutes}$). Gaps larger than 3 steps remain `NaN`.
4.  **Validity Flagging**: A boolean column `valid` is set to `True` for a row if and only if all critical variables are non-NaN:
    $$\text{valid}_t = (y_t \ne \text{NaN}) \land (V_{sw,t} \ne \text{NaN}) \land (N_{sw,t} \ne \text{NaN}) \land (Bz_t \ne \text{NaN})$$
5.  **Segment Identification**: To prevent rolling calculations and sequence windows from spanning data gaps, a `segment_id` increments at every invalid row:
    $$\text{segment\_id}_t = \sum_{i=1}^t \mathbb{I}(\neg \text{valid}_i)$$
    Contiguous valid sequences share the same integer ID.

---

## 4. Chronological Splitting & Scaling Invariants (`src/splits.py` & `src/dataset.py`)

To prevent temporal leakage, data splitting and scaling are strictly controlled:

### The Splits
*   **Training Set**: 2013-01-01 to 2015-12-31
*   **Validation Set**: 2016-01-01 to 2016-06-30
*   **Testing Set**: 2016-07-01 to 2016-12-31

### Scaling Invariant
Standardization scales features to zero mean and unit variance. To prevent validation/test set information from leaking into training:
1.  A `StandardScaler` is initialized.
2.  The scaler is fit **only** on the training features:
    $$\mu_{\text{train}} = \frac{1}{N} \sum X_{\text{train}}, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N} \sum (X_{\text{train}} - \mu_{\text{train}})^2}$$
3.  All three splits are transformed using these training statistics:
    $$X_{\text{scaled}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
4.  The fitted scaler object is saved to `models/scaler.pkl` to scale input features at test time.

### Target Alignment & Gap Safety (`src/dataset.py`)
The `build_xy()` function aligns input features $X_t$ with future targets $y_{t+h}$ for the three horizons.
To prevent target gap crossing:
1.  For each horizon step $h \in \{2, 24, 48\}$, it checks:
    $$\text{valid\_mask}_t = \text{valid\_mask}_t \land (\text{segment\_id}_t == \text{segment\_id}_{t+h}) \land (y_{t+h} \ne \text{NaN})$$
2.  If the segment ID changes between $t$ and $t+h$, indicating a data gap within the forecast horizon, the row is dropped.
3.  All columns starting with `flux_lag` are dropped from the ML feature matrix to prevent direct autocorrelation leaks.

# Chapter 2: Data Engineering & Feature Construction

This chapter outlines the mathematical formulations and algorithms used to construct physical solar-wind features, cumulative energy proxies, time-since-event markers, and wavelet-decomposed features.

## 1. Derived Physical Indicators

The features are calculated in `add_features()` from `src/features.py`.

### Southward IMF ($Bz_s$)
Northward magnetic fields shield the magnetosphere, while southward fields drive reconnection. Southward IMF ($Bz_s$) is isolated by clipping positive values to zero and taking the absolute value:
$$Bz_s = \max(-Bz, 0)$$

### Dynamic Pressure ($P_{dyn}$)
Calculated using the proton mass ($m_p \approx 1.6726 \times 10^{-27}\text{ kg}$), proton density ($N_{sw}$ in $\text{cm}^{-3}$), and solar wind speed ($V_{sw}$ in $\text{km/s}$):
$$P_{dyn} = 1.6726 \times 10^{-6} \times N_{sw} \times V_{sw}^2$$
Resulting units are in nanopascals ($\text{nPa}$).

### Electric Field Proxy ($V_e$)
The convection electric field is modeled as the product of the solar wind speed and the southward magnetic field component:
$$V_e = V_{sw} \times Bz_s$$

### IMF Clock Angle ($\theta$) and Reconnection Efficiency ($\sin^4(\theta/2)$)
Because the dataset contains only the $Bz$ component (with no $By$), the clock angle $\theta$ is modeled as a binary state:
$$\theta = \begin{cases} \pi/2 & \text{if } Bz \ge 0 \\ -\pi/2 & \text{if } Bz < 0 \end{cases}$$
The modulated magnetic reconnection efficiency proxy is:
$$\eta = \sin^4(\theta/2)$$

### Sckopke ($\epsilon$) Coupling Function
Estimates the solar wind energy transfer rate into the magnetosphere:
$$\epsilon = V_{sw} \times B_{tot}^2 \times \sin^4(\theta/2)$$
Where $B_{tot}$ is the total IMF magnetic field magnitude. If $B_{tot}$ is not present in the dataset, the absolute value $|Bz|$ is used as a proxy:
$$\epsilon \approx V_{sw} \times Bz^2 \times \sin^4(\theta/2)$$

### Viscous Interaction Proxy ($F_{visc}$)
Represents viscous coupling along the magnetopause boundary:
$$F_{visc} = V_{sw}^{1/3} \times N_{sw}^{1/2}$$

### Dipole Seasonal & UT Diurnal Tilt Proxies
To account for seasonal and diurnal variations in the Earth's dipole tilt relative to the solar wind:
*   **Seasonal**:
    $$\text{doy\_rad} = \frac{2\pi \times \text{day\_of\_year}}{365.25}$$
    $$\text{tilt\_sin} = \sin(\text{doy\_rad}), \quad \text{tilt\_cos} = \cos(\text{doy\_rad})$$
*   **Diurnal (Universal Time)**:
    $$\text{hod\_rad} = \frac{2\pi \times (\text{hour} + \text{minute}/60)}{24}$$
    $$\text{tilt\_ut\_sin} = \sin(\text{hod\_rad}), \quad \text{tilt\_ut\_cos} = \cos(\text{hod\_rad})$$

---

## 2. Segment-Aware Rolling Integrals

Rolling calculations are grouped by `segment_id` to prevent summing or averaging across data gaps. In pandas, this is implemented as:
```python
df["cum_vbz_pos_24h"] = vbz_pos.groupby(df["segment_id"]).rolling(96, min_periods=1).sum().droplevel(0)
```
The `.droplevel(0)` call removes the `segment_id` index level, aligning the output Series back with the original DataFrame index.

### High-Speed Stream (HSS) Duration
Calculates the duration in hours that the solar wind speed has exceeded $500\text{ km/s}$ over $6$, $12$, and $24$-hour windows:
$$\text{HSS\_flag}_t = \mathbb{I}(V_{sw, t} > 500)$$
$$\text{HSS\_duration\_24h}_t = 0.25 \times \sum_{i=0}^{95} \text{HSS\_flag}_{t-i}$$

### Cumulative Reconnection Energy
Integrates the positive electric field proxy over $6$, $12$, and $24$-hour windows to estimate the accumulated magnetospheric energy:
$$\text{cum\_vbz\_pos\_24h}_t = \sum_{i=0}^{95} \max(V_{sw, t-i} \times Bz_{s, t-i}, 0)$$

---

## 3. Time-Since-Event Markers

The `_steps_since_event()` helper function tracks the time elapsed since specific physical events.

### Algorithm: Steps Since Event
For a boolean series $S$ where $S_t = 1$ if the event occurs, and $S_t = 0$ otherwise:
1.  Compute the cumulative sum of the events:
    $$C_t = \sum_{i=1}^t S_i$$
2.  Group the rows by the cumulative sum $C_t$.
3.  Compute the cumulative count within each group:
    $$\text{steps\_since}_t = \text{cum\_count}(C_t)$$
4.  Convert steps to hours:
    $$\text{hours\_since}_t = \text{steps\_since}_t \times 0.25$$
5.  If no event has occurred yet ($C_t == 0$), assign a default fallback of $999.0\text{ hours}$.

This algorithm is applied to track:
*   `hours_since_bz_flip`: Hours since $Bz$ last crossed zero.
*   `hours_since_vsw_gt500`: Hours since $V_{sw}$ last exceeded $500\text{ km/s}$.

---

## 4. Discrete Wavelet Transform (DWT) Extraction

The `WaveletEncoder` in `src/models/wavelet_encoder.py` uses PyWavelets to decompose solar-wind drivers into sub-band energy features.

### Wavelet Setup
*   **Wavelet Type**: Symlets 4 (`sym4`), chosen for its near-symmetry and orthogonal properties.
*   **Decomposition Level**: $L = 4$.
*   **Lookback Window**: $W = 48$ steps ($12\text{ hours}$).

### Feature Extraction Step
For a window of driver values $x = [x_1, \dots, x_{48}]$:
1.  **Imputation**: Replaces any internal `NaN` with $0.0$.
2.  **Padding**: If the window is shorter than $2^L = 16$ steps, it is padded symmetrically to 16.
3.  **Decomposition**: Performs a multi-level 1D discrete wavelet decomposition:
    $$[cA_4, cD_4, cD_3, cD_2, cD_1] = \text{wavedec}(x, \text{wavelet='sym4'}, \text{level=4})$$
    Where $cA_4$ is the low-frequency approximation coefficient, and $cD_j$ are the detail coefficients at scale $j$.
4.  **Per-Level Energy**: Calculated as the sum of squared detail coefficients:
    $$E_j = \sum cD_j^2 \quad \text{for } j \in \{1, 2, 3, 4\}$$
5.  **Trend**: Isolated as the final value of the approximation band:
    $$\text{Trend} = cA_4[-1]$$
6.  **Bandpower Ratio (BPR)**: Measures the relative energy distribution across scales:
    $$\text{BPR}_j = \frac{E_j}{\sum_{k=1}^4 E_k}$$
7.  **Spectral Entropy**: Shannon entropy calculated over the BPR vector to measure spectral complexity:
    $$H_{spec} = -\sum_{j=1}^4 \text{BPR}_j \log_2(\text{BPR}_j + 10^{-12})$$

For each of the seven primary drivers, this yields $1\text{ Trend} + 1\text{ Entropy} + 4\text{ Energies} + 4\text{ BPRs} = 10$ features, resulting in $70$ total `wavelet__*` features.
These features are calculated over the lookback window $[t-48:t]$, meaning the first 48 rows of the dataset are `NaN` and are dropped prior to model fitting.

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

# Chapter 4: Sequence Encoders (R5)

This chapter covers the sequence-based deep learning models (Rung R5), detailing the sequence construction algorithms, model architectures, and training procedures.

---

## 1. Sequence Window Construction (`to_sequences`)

The `to_sequences()` function converts the 2D feature matrix $\mathbf{X} \in \mathbb{R}^{N \times F}$ and target matrix $\mathbf{y} \in \mathbb{R}^{N \times 3}$ into 3D sequences:
*   $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{M \times L \times F}$ (where $L = 192$ steps)
*   $\mathbf{y}_{\text{seq}} \in \mathbb{R}^{M \times 3}$

### Vectorized Sequence Validity Mask
To filter out invalid windows without looping over millions of rows, the function applies three vectorized conditions:
1.  **Segment Consistency**: The segment ID must remain constant across the window.
    $$\text{seg\_ok}_i = (\text{segment\_id}_i == \text{segment\_id}_{i - L + 1})$$
2.  **No NaNs in Input**: The input features within the window must contain no missing values.
    $$\text{nan\_ok}_i = \left( \sum_{j=i-L+1}^i \mathbb{I}(\mathbf{X}_j \text{ contains NaN}) \right) == 0$$
    This sum is computed using a cumulative sum array:
    $$\text{nan\_cumsum}_t = \sum_{k=1}^t \mathbb{I}(\mathbf{X}_k \text{ contains NaN})$$
    $$\text{nan\_ok}_i = (\text{nan\_cumsum}_i - \text{nan\_cumsum}_{i-L}) == 0$$
3.  **No NaNs in Target**: The target at the end step of the window must be valid.
    $$\text{y\_ok}_i = \neg \text{isnan}(\mathbf{y}_i)$$

A window ending at index $i$ is valid if all three conditions are met:
$$\text{valid}_i = \text{seg\_ok}_i \land \text{nan\_ok}_i \land \text{y\_ok}_i$$
For each valid ending index, the sequence window $\mathbf{X}[i-L+1 : i+1]$ is copied into the pre-allocated output array.

---

## 2. LSTM & Attention Architecture (`LSTMForecaster`)

### The LSTM Network
The LSTM processes the input sequence $\mathbf{x} = [x_1, \dots, x_L]$:
$$h_t, (c_t) = \text{LSTM}(x_t, h_{t-1}, c_{t-1})$$
Where $h_t \in \mathbb{R}^{H}$ is the hidden state (with $H = 64$ units) and $c_t$ is the cell state.

### Temporal Attention Readout
Instead of relying only on the final step's hidden state $h_L$, a temporal attention mechanism pools information across the entire sequence.
1.  **Score Calculation**: A linear layer projects each hidden state to an attention score:
    $$s_t = \mathbf{w}_{\text{attn}}^T h_t$$
2.  **Softmax Normalization**:
    $$\alpha_t = \frac{\exp(s_t)}{\sum_{i=1}^L \exp(s_i)}$$
3.  **Context Vector**: Computes the weighted sum of the hidden states:
    $$\mathbf{c} = \sum_{t=1}^L \alpha_t h_t$$
4.  **Readout Head**: Maps the context vector to the three output horizons:
    $$\hat{\mathbf{y}} = \mathbf{W}_{\text{head}} \mathbf{c} + \mathbf{b}_{\text{head}}$$
    Where $\mathbf{W}_{\text{head}} \in \mathbb{R}^{3 \times H}$.

---

## 3. Temporal Convolutional Network (`TCNForecaster`)

The TCN uses dilated causal 1D convolutions to capture temporal dependencies.

```
Dilated Causal Convolution (dilation d=4, kernel k=3):
Output:    o_t   o_{t-1}
          / | \   / | \
Input:   i_t ... i_{t-4} ... i_{t-8}
```

### The Causal Convolution and Chomp
To prevent the model from looking ahead in time, the convolutions are made causal. A standard convolution is padded on both sides:
$$\text{Padding} = (K_c - 1) \times d$$
Where $K_c = 3$ is the kernel size and $d$ is the dilation rate. The `_Chomp` layer then crops the rightmost $(K_c - 1) \times d$ values from the output tensor:
$$\text{Output}_t = \text{Conv1D}(x)_{t}$$
Ensuring the output at step $t$ depends only on inputs up to step $t$.

### Residual Block
Each block applies weight normalization, a causal convolution, ReLU activation, and dropout:
$$\text{block}(x) = \text{Dropout}(\text{ReLU}(\text{Chomp}(\text{WeightNorm}(\text{Conv1d}(x)))))$$
$$\text{Residual}(x) = \text{ReLU}(\text{block}(\text{block}(x)) + \mathbf{W}_{1\times 1} x)$$

### Receptive Field Calculation
The receptive field (RF) in steps is calculated as:
$$\text{RF} = 1 + N_{\text{layers}} \times (K_c - 1) \times \frac{D_b^{N_{\text{layers}}} - 1}{D_b - 1}$$
For $N_{\text{layers}} = 8$, $K_c = 3$, and dilation base $D_b = 2$:
$$\text{RF} = 1 + 8 \times 2 \times \frac{256 - 1}{1} = 4081 \text{ steps} \approx 1020 \text{ hours}$$
This allows the network to cover a long history.

---

## 4. Deep Learning Training Enhancements

### Delta-Flux Targeting
To stabilize training during high-flux events, the model is configured to predict the deviation from the current flux value (`flux_lag_1`) rather than the absolute value:
$$\mathbf{y}_{\Delta, t} = \mathbf{y}_t - \text{flux\_lag\_1}_t$$
During inference, the baseline is added back:
$$\hat{\mathbf{y}}_t = \hat{\mathbf{y}}_{\Delta, t} + \text{flux\_lag\_1}_t$$

### Hazard-Weighted MSE
The loss function scales the MSE using sample weights:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{i=1}^M w_i \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2^2$$
Where $w_i = 5.0$ (or $8.0$) for storm samples and $1.0$ otherwise.

### Physics-Loss Regularization
A regularization term is added to the loss to penalize violations of quiet-time decay:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_p \cdot \bar{w}_b \cdot \mathcal{L}_{\text{physics}}$$
Where $\mathcal{L}_{\text{physics}}$ represents the quiet-time monotonicity penalty, and $\bar{w}_b$ is the mean hazard weight of the batch. This scaling ensures the regularization remains balanced across different activity regimes.

### Validation Skill-Based Early Stopping
The early stopping logic tracks the forecast skill on the validation set, defined as the average Pearson correlation across the 6-hour and 12-hour horizons:
$$\text{Skill} = 0.5 \times (r_{\text{6h}} + r_{\text{12h}})$$
Training is terminated when this skill metric stops improving, and the model weights from the best-performing epoch are restored.

# Chapter 5: Meta-Learning, Ensembling & Physics (R6 - R8)

This chapter covers the ensembling methods, meta-learning algorithms, and physics-based constraints implemented in rungs R6 to R8.

---

## 1. Physics-Informed Residual Hybrid (`HybridForecaster`)

Rung R6 combines a linear physics model with an ML-based corrector.

### Residual Formulation
The forecasting target $y_t$ is decomposed into a linear physical component and a non-linear residual:
$$y_t = y_{\text{physics}, t} + e_{\text{ML}, t}$$

### Training Procedure
1.  **Backbone Fitting**: The R3 `MultiFilter` is fit on the raw target $y$:
    $$\hat{y}_{\text{physics}} = \text{MultiFilter}(X)$$
2.  **Residual Computation**: The residuals are calculated as the difference between the true values and the linear predictions:
    $$\mathbf{r}_{\text{train}} = y_{\text{train}} - \hat{y}_{\text{physics, train}}$$
3.  **ML Corrector Fitting**: The R4 `GBMForecaster` is trained to predict these residuals:
    $$\hat{e}_{\text{ML}} = \text{LightGBM}(X, \mathbf{r}_{\text{train}})$$

During inference, the final prediction combines both components:
$$\hat{y}_{\text{hybrid}} = \hat{y}_{\text{physics}} + \hat{e}_{\text{ML}}$$

---

## 2. Stacking Meta-Learner (`StackingMetaLearner`)

R7 uses stacking to dynamically blend predictions from the base models based on the state of the solar wind.

### Input Matrix
The meta-learner's input matrix $\mathbf{M}$ is constructed by combining:
1.  **Base Predictions**: $\mathbf{P} = [\hat{y}_{R3}, \hat{y}_{R4}, \hat{y}_{R5}]$
2.  **Regime Features**: $\mathbf{G} \in \mathbb{R}^{N \times 14}$, which represent physical solar-wind conditions (e.g., speed, pressure, reconnection rate, time since last shock).

$$\mathbf{M} = [\mathbf{P}, \mathbf{G}]$$

### State-Dependent Weighting (Polynomial Interactions)
To allow the weights to vary based on the physical regime, the feature space is expanded to include squared terms:
$$\mathbf{M}_{\text{expanded}} = [\mathbf{M}, \mathbf{M}^2]$$
This expansion allows a linear estimator like Ridge regression to model state-dependent weights:
$$\hat{y}_{t+h} = \mathbf{w}^T \mathbf{m}_{\text{expanded}, t} + b$$
This formulation allows the blending weights for the base models to adapt to changes in the regime features:
$$\text{Weight}_{\text{Base}}(G_t) \approx w_1 + 2 w_2 G_t$$

### Time-Series Cross-Validation for Regularization
The regularization parameter $\alpha_{\text{meta}}$ is selected using a 3-fold `TimeSeriesSplit` cross-validation:
$$\text{Split } k: \quad \text{Train } [1 : T_k], \quad \text{Val } [T_k+1 : T_k + V_k]$$
The model evaluates candidate alphas:
$$\alpha \in \{0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0\}$$
It selects the value of $\alpha$ that minimizes the weighted validation MSE.

---

## 3. Physics Loss and Clipping Engine (`src/models/physics_loss.py`)

Rung 8 enforces physical constraints on the predicted trajectories using two main rules.

### Constraint 1: Quiet-Time Monotonicity
During periods of low solar wind activity, the magnetosphere undergoes decay, and the predicted electron flux trajectory should be non-increasing over time:
$$\frac{d(\log_{10} \text{flux})}{dt} \le 0 \quad \implies \quad \hat{y}_{t+2} \ge \hat{y}_{t+24} \ge \hat{y}_{t+48}$$

#### Quiet-Time Mask
The monotonicity constraint is applied only when the magnetosphere is in a quiet state, defined by three conditions:
1.  **Low Speed**: $V_{sw} < 350\text{ km/s}$
2.  **Sustained duration**: The low-speed condition must persist for at least 6 hours (24 steps).
3.  **Low Flux**: The current flux must be below the seasonal median:
    $$\text{flux\_lag\_1}_t < \text{median}_{15\text{-day}}(\text{flux})$$

If all three conditions are met, the trajectory is clipped:
$$\hat{y}_{t+24} = \min(\hat{y}_{t+24}, \hat{y}_{t+2}), \quad \hat{y}_{t+48} = \min(\hat{y}_{t+48}, \hat{y}_{t+24})$$

### Constraint 2: Diffusion Rate Bound
Based on radial diffusion limits, there are physical constraints on how quickly the flux can change. The rate of change $R$ is bounded by:
$$R_{01} = \frac{\hat{y}_{t+24} - \hat{y}_{t+2}}{\Delta t_{01}}, \quad R_{12} = \frac{\hat{y}_{t+48} - \hat{y}_{t+24}}{\Delta t_{12}}$$
$$\text{Max Drop Rate} \le R \le \text{Max Rise Rate}$$
$$\text{where } \text{Max Drop Rate} = -0.5\text{ log}_{10}(\text{pfu})/\text{h}, \quad \text{Max Rise Rate} = +0.18\text{ log}_{10}(\text{pfu})/\text{h}$$

To prevent false alarms from instrument anomalies while still allowing for storm onsets, a soft threshold is used for the rise-rate constraint at inference time:
$$\text{Rise Rate Intervention} = 0.25\text{ log}_{10}(\text{pfu})/\text{h}$$
If a predicted rate exceeds this threshold, the corresponding prediction is clamped:
$$\hat{y}_{t+24} = \min(\hat{y}_{t+24}, \hat{y}_{t+2} + 0.25 \times \Delta t_{01})$$
$$\hat{y}_{t+48} = \min(\hat{y}_{t+48}, \hat{y}_{t+24} + 0.25 \times \Delta t_{12})$$
This clipping is applied to all predictions regardless of the activity regime.

# Chapter 6: Evaluation & Metrics

This chapter covers the mathematical definitions of the evaluation metrics used to score and benchmark the forecasting models.

---

## 1. Continuous Trajectory Metrics

These metrics evaluate the accuracy of the continuous log-flux predictions $\hat{y}$ against the true targets $y$ across all valid time steps.

### Root Mean Squared Error (RMSE)
Measures the average magnitude of the forecast error:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

### Prediction Efficiency (PE)
Also known as the Nash-Sutcliffe efficiency, PE measures the model's accuracy relative to a baseline that always predicts the mean of the observed data:
$$\text{PE} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
Where $\bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$ is the mean of the true targets.
*   $\text{PE} = 1.0$: Indicates a perfect forecast.
*   $\text{PE} = 0.0$: Indicates the model's predictions are as accurate as the mean of the observed data.
*   $\text{PE} < 0.0$: Indicates the model's predictions are less accurate than the mean.

### Pearson Correlation Coefficient ($r$)
Measures the linear correlation between the predicted and observed values:
$$r = \frac{\sum_{i=1}^N (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^N (y_i - \bar{y})^2 \sum_{i=1}^N (\hat{y}_i - \bar{\hat{y}})^2}}$$
Where $\bar{\hat{y}}$ is the mean of the predicted values.

---

## 2. Event-Based Alert Metrics

To evaluate model performance during high-flux events, the continuous predictions and targets are converted into binary classifications using a log-flux threshold of $3.0$:
$$Y_{\text{bin}} = \mathbb{I}(y \ge 3.0), \quad \hat{Y}_{\text{bin}} = \mathbb{I}(\hat{y} \ge 3.0)$$

These binary classifications are used to construct a contingency table:

| | Observed Event ($Y = 1$) | Observed Non-Event ($Y = 0$) |
|---|---|---|
| **Predicted Event ($\hat{Y} = 1$)** | Hit ($A$) | False Alarm ($B$) |
| **Predicted Non-Event ($\hat{Y} = 0$)** | Miss ($C$) | Correct Negative ($D$) |

### Probability of Detection (POD)
Also known as recall or sensitivity, POD measures the fraction of observed events that were correctly predicted:
$$\text{POD} = \frac{A}{A + C}$$
*   Values range from $0.0$ to $1.0$, where $1.0$ is a perfect score.

### False Alarm Rate (FAR)
Measures the fraction of predicted events that did not occur:
$$\text{FAR} = \frac{B}{A + B}$$
*   Values range from $0.0$ to $1.0$, where $0.0$ is a perfect score.

### Heidke Skill Score (HSS)
Measures the accuracy of the forecast relative to the accuracy expected by chance:
$$\text{HSS} = \frac{(A + D) - \text{Expected}}{N - \text{Expected}}$$
Where $N = A + B + C + D$ is the total number of samples, and the number of correct forecasts expected by chance is:
$$\text{Expected} = \frac{(A + C)(A + B) + (B + D)(C + D)}{N}$$
*   $\text{HSS} = 1.0$: Indicates a perfect forecast.
*   $\text{HSS} = 0.0$: Indicates the forecast is as accurate as expected by chance.
*   $\text{HSS} < 0.0$: Indicates the forecast is less accurate than expected by chance.

---

## 3. Local Validation Benchmark Results

The following tables show the model performance on the test split (from 2016-07-01 to 2016-12-31) after resolving the sequence encoder absolute-flux target issues.

### 12-Hour Horizon (Primary Target)

| Model | Before PE | After PE (Delta-Flux) | Before Correlation | After Correlation |
|---|---|---|---|---|
| **Persistence (R0a)** | -- | -0.590 | -- | 0.450 |
| **TCN (R5b)** | +0.055 | **+0.440** | 0.647 | **0.857** |
| **LSTM (R5a)** | −0.137 | **+0.445** | 0.526 | **0.844** |
| **R7 Stacking (No Panel)** | 0.314 | 0.314 | 0.749 | 0.749 |
| **R7 Stacking (+TCN Panel)** | 0.242 | **+0.759** | 0.708 | **0.878** |
| **R8 Physics-Corrected** | 0.192 | **+0.737** | 0.707 | **0.875** |

### 6-Hour Horizon

| Model | Prediction Efficiency (PE) | Correlation |
|---|---|---|
| **TCN (R5b)** | 0.629 | 0.896 |
| **LSTM (R5a)** | 0.609 | 0.893 |
| **R7 Stacking (+TCN Panel)** | 0.841 | 0.919 |

> [!IMPORTANT]
> The transition from predicting absolute flux to predicting delta-flux ($\Delta$flux) is the single most significant factor in performance gains, raising the 12-hour horizon Stacking Meta-Learner PE from `0.242` to `0.759`.