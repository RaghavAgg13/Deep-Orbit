"""Rung 5b: Temporal Convolutional Network (TCN) sequence encoder.

Drop-in replacement for :class:`src.models.lstm.LSTMForecaster` with the
**exact same public API** so ``main.py`` can swap the encoder with a single
import change.

The TCN replaces the LSTM with a stack of dilated causal Conv1d blocks. The
receptive field grows *exponentially* with depth (``dilation_base ** i``) versus
the LSTM's linear-in-seq_len memory, so a modest number of layers covers a
48-72 h storm history with a parameter count that is essentially *independent*
of ``seq_len``.

Training loop, hazard-weighted MSE, AdamW + ReduceLROnPlateau, gradient
clipping and early stopping are copied verbatim from ``LSTMForecaster`` so the
two encoders are comparable. The optional temporal-attention readout and the
MLP head are reused from the LSTM module.
"""
from __future__ import annotations

import numpy as np

from src.harness import Forecaster
from src.models.physics_loss import PhysicsConstraint, physics_penalized_residual

# torch is an optional dependency at import time; the class is still defined
# so the package imports cleanly even without torch installed.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover - graceful degradation
    _HAS_TORCH = False

# Reuse the LSTM's temporal-attention readout and sequence builder verbatim.
from .lstm import _TemporalAttention, to_sequences  # noqa: F401,E402


class _Chomp(nn.Module):
    """Trim the extra right-pad so the convolution stays causal.

    A causal convolution with ``dilation=d`` needs ``(kernel_size - 1) * d`` pads
    on each side; we pad only on the left and drop the ``chomp_size`` extra
    positions the kernel produces on the right, so output[t] depends only on
    input[:t].
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (batch, channels, seq)
        return x[..., : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):
    """One TCN residual block.

    ``weight_norm(Conv1d) -> Chomp -> ReLU -> dropout`` applied twice, with a
    1x1 residual when in/out channel counts differ. Dilation doubles per block.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp1 = _Chomp(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.chomp2 = _Chomp(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_ch, seq)
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = self.downsample(x) if self.downsample is not None else x
        # Both paths are trimmed to the chomp output length; res is longer by
        # `pad` (the first block) but subsequent blocks keep length, so trim res
        # to match `out` to keep the residual addition well-defined.
        res = res[..., : out.size(-1)]
        return self.relu(out + res)


class _TCN(nn.Module):
    """Stack of :class:`_TemporalBlock` with exponentially growing dilation.

    Receptive field (in steps)::

        RF = 1 + num_layers * (kernel_size - 1) * (dilation_base ** num_layers - 1)
             / (dilation_base - 1)

    which for the defaults (``num_layers=8, kernel_size=3, dilation_base=2``)
    is 511 steps -- well over 192 (= 48 h at 15-min cadence).
    """

    def __init__(self, n_features: int, channels: list[int], kernel_size: int,
                 dropout: float, dilation_base: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = dilation_base ** i
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self._channels = channels
        self._kernel_size = kernel_size
        self._dilation_base = dilation_base
        self._num_layers = len(channels)

    def receptive_field(self) -> int:
        k = self._kernel_size
        d = self._dilation_base
        n = self._num_layers
        return 1 + n * (k - 1) * (d ** n - 1) // (d - 1)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        return self.network(x.transpose(1, 2))


class TCNForecaster(Forecaster):
    """Deep TCN sequence model with hazard-weighted MSE and early stopping.

    Drop-in replacement for :class:`src.models.lstm.LSTMForecaster`. Public API
    (constructor, ``fit_sequences``, ``fit``, ``predict``) is identical.
    """

    def __init__(self, seq_len: int = 192, hidden_dim: int = 64, num_layers: int = 8,
                 kernel_size: int = 3, dropout: float = 0.2, use_attention: bool = True,
                 hazard_log: float = 3.0, storm_weight: float = 5.0,
                 epochs: int = 60, lr: float = 1e-3, batch_size: int = 512,
                 patience: int = 10, device: str | None = None,
                 dilation_base: int = 2,
                 physics_loss: bool = True,
                 physics_lam: float = 0.05,
                 physics_quiet_vsw: float = 350.0,
                 vsw_channel: str = "v_sw",
                 weight_decay: float = 1e-4,
                 seed: int | None = None,
                 swa_last_n: int = 0,
                 delta_flux: bool = False):
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for TCNForecaster. Install via `pip install torch`.")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.use_attention = use_attention
        self.hazard_log = hazard_log
        self.storm_weight = storm_weight
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dilation_base = dilation_base
        self.physics_loss = physics_loss
        self.physics_lam = physics_lam
        self.physics_quiet_vsw = physics_quiet_vsw
        self.vsw_channel = vsw_channel
        self.weight_decay = weight_decay
        # TIER A.5 — reproducibility + SWA (see lstm.py for rationale). Seed
        # fixes weight init + shuffle; swa_last_n averages the last N epochs'
        # weights to smooth the early-stopping noise on the 5.4K val set.
        self.seed = seed
        self.swa_last_n = int(swa_last_n)
        # TIER A.8 — delta-flux target (see lstm.py for the full rationale).
        # Predicts the driver-driven *change* from the known current level;
        # flux_base is added back at predict time.
        self.delta_flux = delta_flux
        self.model_: nn.Module | None = None
        self._n_features: int | None = None
        self._n_horizons: int = 3
        self.feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _build(self, n_features: int, n_horizons: int):
        self._n_features = n_features
        self._n_horizons = n_horizons
        channels = [self.hidden_dim] * self.num_layers
        self.model_ = _SeqEncoder(
            n_features=n_features, channels=channels, kernel_size=self.kernel_size,
            output_dim=n_horizons, dropout=self.dropout,
            use_attention=self.use_attention, dilation_base=self.dilation_base,
        ).to(self.device)
        rf = self.model_.receptive_field()
        print(f"  [TCN] built: RF={rf} steps ({rf * 15} min = {rf * 15 / 60:.1f} h), "
              f"seq_len={self.seq_len}, n_params={self.model_.n_params:,}")

    @property
    def receptive_field(self) -> int:
        """Receptive field in steps (0 before the model is built)."""
        if self.model_ is None:
            return 0
        return self.model_.receptive_field()

    def summary(self) -> dict:
        """Return a logging dict with receptive field and size stats."""
        rf = self.receptive_field
        return {
            "receptive_field_steps": rf,
            "receptive_field_hours": rf * 15 / 60.0,
            "n_params": self.model_.n_params if self.model_ is not None else 0,
            "seq_len": self.seq_len,
        }

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
                      flux_base_train=None, flux_base_val=None):
        """Train on pre-built 3D sequences (see ``to_sequences``).

        flux_base_train / flux_base_val : ndarray (n,) | None
            Current flux (flux_lag_1) aligned to y. Required when
            ``delta_flux=True``; the target becomes y - flux_base.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        # TIER A.8 — delta-flux target (see lstm.py for the rationale).
        self._flux_base_train = None
        self._flux_base_val = None
        if self.delta_flux:
            if flux_base_train is None or flux_base_val is None:
                raise ValueError(
                    "TCNForecaster delta_flux=True requires flux_base_train "
                    "and flux_base_val (the flux_lag_1 values aligned to y).")
            self._flux_base_train = np.asarray(flux_base_train, dtype=np.float32)
            self._flux_base_val = np.asarray(flux_base_val, dtype=np.float32)
            y_train = y_train - self._flux_base_train[:, None]
            y_val = y_val - self._flux_base_val[:, None]

        # TIER A.5 — reproducibility (see lstm.py for the full rationale).
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
        # TIER A.5 — fixed step schedule instead of ReduceLROnPlateau (which
        # couples the schedule to the noisy val_loss). LR drops at 50% and 75%.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.epochs * 0.5), int(self.epochs * 0.75)],
            gamma=0.5)
        mse = nn.MSELoss(reduction="none")

        # Physics-informed training loss (Tier 2.1): add ``lam * physics_penalty``
        # to the MSE at each step. We need the per-row solar-wind speed to
        # compute the quiet/regime mask; ``vsw_channel`` is the name of the
        # driver column exposed on the LAST step of each window (so the
        # quiet-mask conditions hold on the same row as the forecast).
        physics_con = None
        vsw_idx = None
        if self.physics_loss and self.feature_names is not None:
            vsw_matches = [i for i, n in enumerate(self.feature_names)
                           if n == self.vsw_channel]
            if vsw_matches:
                vsw_idx = vsw_matches[-1]
                physics_con = PhysicsConstraint(quiet_vsw=self.physics_quiet_vsw)

        # TIER A.6 — early stopping on SMOOTH_CORR (maximize), not val_loss
        # (see lstm.py for the full rationale). The TCN peaks sharp at epoch
        # 6 then degrades, so we restore the best-smooth_corr epoch directly.
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
                # TIER A.3 — scale the unweighted batch-mean physics penalty by
                # the batch's mean hazard weight so it stays a consistent
                # fraction of the total loss across quiet/storm regimes (see
                # lstm.py for the full rationale).
                if physics_con is not None and vsw_idx is not None:
                    v_sw_batch = xb[:, -1, vsw_idx].detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    pen = physics_con.penalty(
                        pred_np, v_sw_batch,
                        flux=None, index=None)["total"]
                    w_scale = wb.mean().detach()
                    loss = loss + self.physics_lam * w_scale * torch.as_tensor(
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

            # TIER A.4 — per-epoch diagnostic (mirrors LSTM). Watch the 12h
            # horizon corr: if it stays near 0 while 30-min climbs, the encoder
            # is only tracking autocorrelation, not learning dynamics.
            with torch.no_grad():
                vp_all = []
                for xb, _ in val_dl:
                    vp_all.append(self.model_(xb.to(self.device)).cpu().numpy())
            vp_all = np.concatenate(vp_all, axis=0)
            yv = y_val[: vp_all.shape[0]]
            rmses = [float(np.sqrt(np.mean((yv[:, i] - vp_all[:, i]) ** 2))) for i in range(3)]
            corrs = [float(np.corrcoef(yv[:, i], vp_all[:, i])[0, 1]) for i in range(3)]
            smooth_corr = 0.5 * (corrs[1] + corrs[2])
            print(f"  [TCN] epoch {epoch:3d} val_loss={val_loss:.4f} "
                  f"rmse=[{rmses[0]:.3f},{rmses[1]:.3f},{rmses[2]:.3f}] "
                  f"corr=[{corrs[0]:+.3f},{corrs[1]:+.3f},{corrs[2]:+.3f}] "
                  f"smooth_corr={smooth_corr:+.3f}")

            sd = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            # TIER A.6 — track best by smooth_corr (skill), not val_loss.
            if smooth_corr > best_corr + 1e-5:
                best_corr = smooth_corr
                best_state = sd
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= self.patience:
                print(f"  [TCN] Early stopping at epoch {epoch} "
                      f"(best smooth_corr={best_corr:.3f})")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    # ``fit`` is kept for interface parity; the sequence model needs 3D input,
    # so we delegate to ``fit_sequences`` when given 3D arrays.
    def fit(self, X, y, sample_weight=None, eval_set=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            X_val = np.asarray(eval_set[0], dtype=np.float32) if eval_set is not None else X
            y_val = np.asarray(eval_set[1], dtype=np.float32) if eval_set is not None else y
            return self.fit_sequences(X, y, X_val, y_val)
        raise ValueError("TCNForecaster.fit expects 3D sequence input; use fit_sequences.")

    def predict(self, X, flux_base=None):
        """Predict absolute flux (see lstm.py predict for flux_base docs)."""
        if self.model_ is None:
            raise RuntimeError("TCNForecaster is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self.device)
            out = self.model_(xb).cpu().numpy()
        # TIER A.8 — add the known starting level back (see lstm.py).
        if self.delta_flux:
            if flux_base is None:
                raise ValueError(
                    "TCNForecaster delta_flux=True requires flux_base at "
                    "predict time (the flux_lag_1 values aligned to rows).")
            out = out + np.asarray(flux_base, dtype=np.float32)[:, None]
        return out


class _SeqEncoder(nn.Module):
    """TCN encoder + (attention | last-step) readout head.

    Wraps :class:`_TCN` with the same optional temporal-attention readout and
    MLP head as the LSTM's ``_SeqEncoder`` so the two are interchangeable.
    """

    def __init__(self, n_features: int, channels: list[int], kernel_size: int = 3,
                 output_dim: int = 3, dropout: float = 0.2, use_attention: bool = True,
                 dilation_base: int = 2):
        super().__init__()
        self.use_attention = use_attention
        self.tcn = _TCN(n_features, channels, kernel_size, dropout, dilation_base)
        hidden_dim = channels[-1]
        self.attn = _TemporalAttention(hidden_dim) if use_attention else None
        # TIER A.2 — single linear readout (see lstm.py _SeqEncoder for the
        # rationale). Forces the TCN encoder to learn the temporal
        # representation instead of passing the work to an over-parameterized
        # head. Mirrors the LSTM head so the two encoders stay interchangeable.
        self.head = nn.Linear(hidden_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters())
        self._hidden_dim = hidden_dim

    def receptive_field(self) -> int:
        return self.tcn.receptive_field()

    def forward(self, x):
        # x: (batch, seq, features)
        out = self.tcn(x)                       # (batch, hidden, seq)
        out = out.transpose(1, 2)               # (batch, seq, hidden)
        feat = self.attn(out) if self.attn is not None else out[:, -1, :]
        return self.head(feat)                  # (batch, output_dim)
