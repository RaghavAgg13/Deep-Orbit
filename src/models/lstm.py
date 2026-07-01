"""Rung 5: Deep driver-driven sequence model.

A multi-layer LSTM encoder processes a window of length ``seq_len`` of driver
features (NO flux history — strictly driver-driven to prevent copy-paste
leakage). A readout head maps the final encoder state to the 3 forecast
horizons. An optional Temporal-Attention readout aggregates the encoder's full
hidden-state trajectory instead of relying solely on the last step, which
substantially improves the 6-h / 12-h horizons where long-range memory matters.

Training uses:
  * AdamW + ReduceLROnPlateau schedule
  * gradient clipping (stable deep recurrence)
  * hazard-weighted MSE (storm samples up-weighted)
  * physics-loss regularization (quiet-time monotonicity penalty)
  * early stopping on validation loss

The module also exposes ``to_sequences`` which feature matrix
into segment-safe for the encoder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.harness import Forecaster
from src.models.physics_loss import PhysicsConstraint

# torch is an optional dependency at import time; the class is still defined
# so the package imports cleanly even without torch installed.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover - graceful degradation
    _HAS_TORCH = False


def to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int,
                 segment_ids: np.ndarray | None = None):
    """Convert flat (n, f) arrays into segment-safe 3D sequences.

    A valid sequence starting at row ``i`` requires ``seq_len`` consecutive
    rows within the same ``segment_id`` (so windows never cross data gaps) and
    a valid target at row ``i + seq_len - 1``.

    Returns
    -------
    X_seq : ndarray (n_valid, seq_len, n_features)
    y_seq : ndarray (n_valid, n_horizons)
    idxs  : ndarray (n_valid,) — the *end* row index of each window in X
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n, f = X.shape
    if segment_ids is None:
        segment_ids = np.zeros(n, dtype=int)

    # Vectorized validity mask — replaces a Python loop over ~2M rows with
    # numpy ops. Three conditions per window ending at row i:
    #   1. segment_id constant across [i-seq_len+1, i]  (no gap crossing)
    #   2. no NaN in X over the window
    #   3. no NaN in y at row i
    # Each is computed via cumulative sums / rolling comparisons in O(n).
    sl = seq_len

    # --- condition 1: segment constant across window ---
    # The loop checks seg[s] == seg[e] where s = i-sl+1, e = i. Vectorized:
    # compare seg[sl-1 : n] (endpoints) vs seg[0 : n-sl+1] (startpoints).
    seg_ok = (segment_ids[sl - 1 :] == segment_ids[: n - sl + 1])  # (n-sl+1,)

    # --- condition 2: no NaN in X across window ---
    row_nan = np.isnan(X).any(axis=1).astype(np.int32)          # (n,)
    nan_cs = np.concatenate([[0], row_nan.cumsum()])             # (n+1,)
    nan_ok = (nan_cs[sl:] - nan_cs[: n - sl + 1]) == 0           # (n-sl+1,)

    # --- condition 3: no NaN in y at the window-end row ---
    y_nan = np.isnan(y).any(axis=1)                              # (n,)
    y_ok = ~y_nan[sl - 1 :]                                      # (n-sl+1,)

    # Combine: valid window-end indices are rows [sl-1, n) where all hold.
    valid = seg_ok & nan_ok & y_ok
    ends = np.nonzero(valid)[0] + (sl - 1)                       # absolute row idxs
    n_valid = len(ends)
    if n_valid == 0:
        empty = np.empty((0, sl, f), dtype=np.float32)
        return empty, np.empty((0, y.shape[1]), dtype=np.float32), np.array([], dtype=int)

    # Build the 3D window array. Pre-allocate once and copy each window
    # slice in-place. The vectorized mask above (pass 1) is now instant; this
    # copy loop is the remaining cost (~18s on 2M rows) but is unavoidable
    # for sparse `ends` (NaN/segment-gap rows skipped). Stride tricks only
    # work for contiguous windows and torch.from_numpy requires contiguous
    # memory, so a copy is needed regardless. Pre-allocation avoids the ~16
    # GB temporary that np.stack over a Python list would create.
    X_seq = np.empty((n_valid, sl, f), dtype=np.float32)
    for j, e in enumerate(ends):
        X_seq[j] = X[e - sl + 1: e + 1]
    y_seq = y[ends]
    return X_seq, y_seq, ends.astype(int)


class _TemporalAttention(nn.Module):
    """Learnable temporal-pooling over the encoder's hidden-state sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        # h: (batch, seq_len, hidden)
        attn = torch.softmax(self.score(h).squeeze(-1), dim=1)  # (batch, seq_len)
        return torch.bmm(attn.unsqueeze(1), h).squeeze(1)        # (batch, hidden)


class _SeqEncoder(nn.Module):
    """Multi-layer LSTM encoder + (attention | last-step) readout head."""

    def __init__(self, n_features: int, hidden_dim: int = 64, num_layers: int = 2,
                 output_dim: int = 3, dropout: float = 0.2, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = _TemporalAttention(hidden_dim) if use_attention else None
        # TIER A.2 — single linear readout. The old 2-layer MLP head
        # (Linear(64)->GELU->Dropout->Linear(64->3)) has ~4.2K params that
        # over-fit the ~27K training windows before the encoder learns a
        # useful temporal representation. A single projection head(->output)
        # forces the encoder to do the work; the meta-learner (R7) is the one
        # that blends non-linearly, so the encoder doesn't need its own MLP.
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(x)                       # (batch, seq, hidden)
        feat = self.attn(h) if self.attn is not None else h[:, -1, :]
        return self.head(feat)                    # (batch, output_dim)

    def attn_entropy(self, x):
        """Return the mean entropy of the attention distribution over the
        sequence (diagnostic). Low entropy = attention concentrates on a few
        timesteps; high entropy = uniform. If entropy is high, attention is
        just noise and the last-step readout would be as good."""
        if self.attn is None:
            return None
        with torch.no_grad():
            h, _ = self.lstm(x)
            w = torch.softmax(self.attn.score(h).squeeze(-1), dim=1)
            ent = -(w * torch.log(w + 1e-12)).sum(dim=1).mean()
        return float(ent)


class LSTMForecaster(Forecaster):
    """Deep LSTM sequence model with hazard-weighted MSE and early stopping."""

    def __init__(self, seq_len: int = 96, hidden_dim: int = 64, num_layers: int = 2,
                 epochs: int = 40, lr: float = 1e-3, batch_size: int = 512,
                 dropout: float = 0.2, use_attention: bool = True,
                 hazard_log: float = 3.0, storm_weight: float = 5.0,
                 patience: int = 8, device: str | None = None,
                 physics_loss: bool = True,
                 physics_lam: float = 0.05,
                 vsw_channel: str = "v_sw",
                 weight_decay: float = 1e-4,
                 seed: int | None = None,
                 swa_last_n: int = 0,
                 delta_flux: bool = False):
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for LSTMForecaster. Install via `pip install torch`.")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_attention = use_attention
        self.hazard_log = hazard_log
        self.storm_weight = storm_weight
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.physics_loss = physics_loss
        self.physics_lam = physics_lam
        self.vsw_channel = vsw_channel
        self.weight_decay = weight_decay
        # TIER A.5 — reproducibility + stability. `seed` makes init + shuffle
        # deterministic so the benchmark is repeatable (the run-to-run PE
        # variance was ~0.4 with early stopping on a 5.4K-window val set).
        # `swa_last_n` averages the last N epochs' weights (SWA) instead of
        # restoring a single noisy "best" epoch — standard fix for val-loss
        # early stopping on a tiny validation set.
        self.seed = seed
        self.swa_last_n = int(swa_last_n)
        # TIER A.8 — delta-flux target. When True, the encoder predicts
        # y - flux_base (the driver-driven *change* from the known current
        # level) instead of absolute flux. flux_base = flux_lag_1 is the
        # current flux, known at prediction time (NOT leakage — it's the
        # starting point, not the future target). At predict time we add
        # flux_base back. The diagnostic proved this is the difference
        # between PE -0.59 (absolute) and PE +0.03 (delta) at 12h: without
        # the current level the model can't tell quiet (flux~2) from storm
        # (flux~5) from drivers alone, so it regresses to the mean.
        self.delta_flux = delta_flux
        self.model_: _SeqEncoder | None = None
        self._n_features: int | None = None
        self._n_horizons: int = 3
        self.feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _build(self, n_features: int, n_horizons: int):
        self._n_features = n_features
        self._n_horizons = n_horizons
        self.model_ = _SeqEncoder(
            n_features=n_features, hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, output_dim=n_horizons,
            dropout=self.dropout, use_attention=self.use_attention,
        ).to(self.device)

    @staticmethod
    def _hazard_weights(y: np.ndarray, hazard_log: float, storm_weight: float):
        """Per-sample weight: storm frames (any horizon >= threshold) up-weighted."""
        w = np.ones(len(y), dtype=np.float32)
        mask = np.any(y >= hazard_log, axis=1)
        w[mask] = storm_weight
        return torch.from_numpy(w)

    def _dataloader(self, X, y, shuffle: bool, weights: torch.Tensor | None = None):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        tensors = [Xt, yt]
        if weights is not None:
            tensors.append(weights)
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    def fit_sequences(self, X_train, y_train, X_val, y_val, feature_names=None,
                      physics_loss: bool = True, physics_lam: float = 0.05,
                      vsw_channel: str = "v_sw",
                      flux_base_train=None, flux_base_val=None):
        """Train on pre-built 3D sequences (see ``to_sequences``).

        Parameters
        ----------
        physics_loss : bool
            If True and feature_names is provided, add the physics penalty
            (quiet-time monotonicity) to the training loss.
        physics_lam : float
            Weight of the physics penalty relative to the MSE loss.
        vsw_channel : str
            Name of the solar-wind channel in feature_names used to compute
            the quiet-time mask for the physics penalty.
        flux_base_train : ndarray (n,) | None
            Current flux (flux_lag_1) aligned to y_train. When
            ``delta_flux=True`` the target becomes y - flux_base (the
            driver-driven change). Pass the known starting level so the
            encoder learns deviations from it.
        flux_base_val : ndarray (n,) | None
            Validation counterpart of ``flux_base_train``.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        # TIER A.8 — delta-flux target. Subtract the known current level so
        # the encoder predicts the driver-driven *change*. Without this the
        # model regresses to the mean (it can't tell quiet from storm using
        # drivers alone — see the A.8 diagnostic).
        self._flux_base_train = None
        self._flux_base_val = None
        if self.delta_flux:
            if flux_base_train is None or flux_base_val is None:
                raise ValueError(
                    "LSTMForecaster delta_flux=True requires flux_base_train "
                    "and flux_base_val (the flux_lag_1 values aligned to y).")
            self._flux_base_train = np.asarray(flux_base_train, dtype=np.float32)
            self._flux_base_val = np.asarray(flux_base_val, dtype=np.float32)
            y_train = y_train - self._flux_base_train[:, None]
            y_val = y_val - self._flux_base_val[:, None]

        # TIER A.5 — reproducibility. Seed torch (and cuda if used) so weight
        # init + dataloader shuffle are deterministic. Without this the same
        # code produces PE swings of ~0.4 on the sequence test set between
        # runs, because early stopping on a 5.4K-window val set is
        # noise-sensitive. A fixed seed makes the benchmark repeatable.
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self._build(n_features=X_train.shape[-1], n_horizons=y_train.shape[-1])
        self.feature_names = feature_names

        weights = self._hazard_weights(y_train, self.hazard_log, self.storm_weight)
        train_dl = self._dataloader(X_train, y_train, shuffle=True, weights=weights)
        val_dl = self._dataloader(X_val, y_val, shuffle=False)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # TIER A.5 — replace ReduceLROnPlateau (which couples the schedule to
        # the noisy val_loss) with a fixed step schedule. LR drops at 50% and
        # 75% of training — predictable, reproducible, no val-set dependency.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.5)
        mse = nn.MSELoss(reduction="none")

        # Physics-loss setup: find the v_sw column index in the feature list.
        physics_con = None
        vsw_idx = None
        if physics_loss and feature_names is not None:
            vsw_matches = [i for i, n in enumerate(feature_names)
                           if n == vsw_channel]
            if vsw_matches:
                vsw_idx = vsw_matches[-1]
                physics_con = PhysicsConstraint(quiet_vsw=350.0)

        # TIER A.6 — early stopping on SMOOTH_CORR (maximize), not val_loss.
        # The diagnostics proved val_loss and correlation DIVERGE: val_loss
        # keeps falling while smooth_corr plateaus then degrades, so
        # minimizing val_loss picks a degraded epoch. We track the epoch with
        # the highest smooth_corr (mean of 6h+12h val correlation — the
        # actual forecast-skill metric) and restore THAT epoch. No SWA: the
        # wide-plateau SWA experiment showed averaging the tail dilutes the
        # peak by ~0.05 corr, making test PE worse, not better.
        best_state = None
        best_corr = -float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            for batch in train_dl:
                xb, yb, wb = (t.to(self.device) for t in batch)
                pred = self.model_(xb)
                loss_per = mse(pred, yb).mean(dim=1)
                loss = (loss_per * wb).mean()
                # Physics-loss regularization: penalize quiet-time recovery
                # violations. Uses the Vsw value at the last timestep of each
                # window (the most recent observation) to determine the regime.
                #
                # TIER A.3 — the physics penalty is a BATCH MEAN (unweighted),
                # while the MSE is hazard-weighted. To keep the physics term a
                # consistent fraction of the total loss in every regime, scale
                # it by the batch's mean hazard weight. Otherwise the physics
                # term is a fixed scalar that dominates quiet batches (w=1.0)
                # and vanishes on storm batches (w=8.0), pushing the encoder
                # toward "always decay" on the 90% quiet rows.
                if physics_con is not None and vsw_idx is not None:
                    v_sw_batch = xb[:, -1, vsw_idx].detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    pen = physics_con.penalty(
                        pred_np, v_sw_batch,
                        flux=None, index=None)["total"]
                    w_scale = wb.mean().detach()
                    loss = loss + physics_lam * w_scale * torch.as_tensor(
                        pen, dtype=loss.dtype, device=loss.device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            # Validation
            self.model_.eval()
            val_loss = 0.0
            n_seen = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    l = mse(self.model_(xb), yb).mean(dim=1).sum().item()
                    val_loss += l
                    n_seen += xb.size(0)
            val_loss /= max(n_seen, 1)
            scheduler.step()

            # TIER A.4 — per-epoch diagnostic: per-horizon val RMSE + corr.
            # This is the feedback loop that was missing: without it you can't
            # tell whether the encoder is learning the 12h horizon or just
            # tracking the 30-min autocorrelation. Printed every epoch.
            vp_all = []
            with torch.no_grad():
                for xb, _ in val_dl:
                    vp_all.append(self.model_(xb.to(self.device)).cpu().numpy())
            vp_all = np.concatenate(vp_all, axis=0)
            yv = y_val[: vp_all.shape[0]]
            rmses = [float(np.sqrt(np.mean((yv[:, i] - vp_all[:, i]) ** 2))) for i in range(3)]
            corrs = [float(np.corrcoef(yv[:, i], vp_all[:, i])[0, 1]) for i in range(3)]
            # Smoothed val correlation = mean across the two long horizons
            # (6h+12h). This is the metric that tracks actual forecast skill;
            # val_loss tracks calibration and diverges from corr (see the
            # epoch-3-vs-6 TCN diagnostic). Used for the SWA window select.
            smooth_corr = 0.5 * (corrs[1] + corrs[2])
            print(f"  [LSTM] epoch {epoch:3d} val_loss={val_loss:.4f} "
                  f"rmse=[{rmses[0]:.3f},{rmses[1]:.3f},{rmses[2]:.3f}] "
                  f"corr=[{corrs[0]:+.3f},{corrs[1]:+.3f},{corrs[2]:+.3f}] "
                  f"smooth_corr={smooth_corr:+.3f}")

            # TIER A.5 — keep a rolling buffer of the last swa_last_n epochs.
            # Always track best_state too (used when swa_last_n=0 for
            # backwards compatibility).
            sd = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            # TIER A.6 — track best by smooth_corr (skill), not val_loss.
            # Restore this checkpoint; early stopping breaks on no-improve.
            if smooth_corr > best_corr + 1e-5:
                best_corr = smooth_corr
                best_state = sd
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= self.patience:
                print(f"  [LSTM] Early stopping at epoch {epoch} "
                      f"(best smooth_corr={best_corr:.3f})")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    # ``fit`` is kept for interface parity but the sequence model needs 3D input,
    # so we delegate to ``fit_sequences`` when given 3D arrays.
    def fit(self, X, y, sample_weight=None, eval_set=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X_val = np.asarray(eval_set[0], dtype=np.float32) if eval_set is not None else X
            y_val = np.asarray(eval_set[1], dtype=np.float32) if eval_set is not None else y
            return self.fit_sequences(X, y, X_val, y_val,
                                      physics_loss=self.physics_loss,
                                      physics_lam=self.physics_lam,
                                      vsw_channel=self.vsw_channel)
        raise ValueError("LSTMForecaster.fit expects 3D sequence input; use fit_sequences.")

    def predict(self, X, flux_base=None):
        """Predict absolute flux.

        Parameters
        ----------
        X : ndarray (n, seq, features) — driver window.
        flux_base : ndarray (n,) | None — current flux (flux_lag_1) for each
            window. Required when ``delta_flux=True``; added back to the
            predicted *change* to recover absolute flux.
        """
        if self.model_ is None:
            raise RuntimeError("LSTMForecaster is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self.device)
            out = self.model_(xb).cpu().numpy()
        # TIER A.8 — add the known starting level back to the predicted
        # change to recover absolute flux.
        if self.delta_flux:
            if flux_base is None:
                raise ValueError(
                    "LSTMForecaster delta_flux=True requires flux_base at "
                    "predict time (the flux_lag_1 values aligned to rows).")
            out = out + np.asarray(flux_base, dtype=np.float32)[:, None]
        return out
