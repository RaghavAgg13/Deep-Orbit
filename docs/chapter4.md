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
