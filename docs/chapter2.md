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
