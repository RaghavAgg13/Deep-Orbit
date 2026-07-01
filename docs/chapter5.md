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
