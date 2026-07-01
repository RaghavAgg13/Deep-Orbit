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
