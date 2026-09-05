"""
src/attention_modules.py

Ablatable attention building blocks for the proposed model
(src/models/dual_attention_bigru.py). Each module is standalone and can be
switched on/off independently so the ablation story is clean:

    plain BiGRU (bigru_baseline)
    -> + one temporal attention axis (bilstm_attention_baseline)
    -> + channel attention axis, learned only (DualAttentionBiGRU, use_spectral_gate=False
       -- this reproduces the closed "channel+temporal dual attention" gap, e.g. Dual-INABA)
    -> + spectral gate on channel attention (DualAttentionBiGRU, use_spectral_gate=True
       -- the proposed novel component, see docs/research_gap_notes.md)

Novelty note: every dual-attention HAR paper found in the literature review
(Dual-INABA, TGA-HAR, MHAT-Net, CNN-BiGRU-Transformer, AttnFusionGRU) computes
channel attention purely from learned time-domain statistics. None inject a
frequency-domain prior. SpectralChannelGate below does exactly that, cheaply
(a single rFFT along the time axis -- no CNN, negligible FLOPs) and is the
concrete architectural novelty layered on top of this project's methodological
novelty (efficiency + evaluation-rigor accounting).
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Channel attention ("where to look" -- which sensor axis matters)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """
    Squeeze-and-excite style attention over raw sensor channels.

    Input:  x (batch, T, C)
    Output: weights (batch, C) in (0, 1), one scalar gate per sensor channel.

    Pools over time (mean AND max, as in CBAM) to get a per-channel summary,
    passes both through a shared bottleneck MLP, sums, and squashes with a
    sigmoid (independent per-channel gates, not a softmax distribution --
    softmax would force channels to compete for a fixed "attention budget",
    which is the wrong inductive bias when e.g. all 6 UCI-HAR channels can
    simultaneously be informative for a given activity).
    """

    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, num_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=1)            # (batch, C)
        max_pool = x.max(dim=1).values      # (batch, C)
        logits = self.mlp(avg_pool) + self.mlp(max_pool)
        return torch.sigmoid(logits)        # (batch, C)


# ---------------------------------------------------------------------------
# Spectral gate (novel component -- frequency-domain prior on channel attention)
# ---------------------------------------------------------------------------
class SpectralChannelGate(nn.Module):
    """
    Per-channel periodicity descriptor computed via rFFT along the time axis,
    mapped through a tiny MLP to a per-channel gate in (0, 1).

    Two hand-crafted, non-parametric descriptors per channel per window:
      1. low-band energy ratio: fraction of spectral power falling inside
         `low_band_hz` (default 0.3-3.5 Hz -- the human gait/limb-swing band).
         High for channels dominated by locomotion-driven periodicity
         (e.g. an ankle or wrist accelerometer axis during walking/jogging),
         low for channels that are mostly-DC (e.g. a gravity-aligned axis
         during sitting/standing/lying).
      2. normalized spectral entropy: low for a clean, dominant-frequency
         signal (strong periodic motion), high for noisy/broadband signals.

    These are cheap (one rFFT per forward pass, O(T log T), no learned
    parameters in the transform itself) and give the model a physically
    motivated prior that a purely learned channel-attention MLP has no direct
    way to access from a single time-domain window.

    The gate is combined multiplicatively with ChannelAttention's output in
    GatedChannelAttention below -- this module never runs standalone.
    """

    def __init__(self, num_channels: int, sampling_rate_hz: float,
                 low_band_hz: tuple = (0.3, 3.5), hidden: int = 8):
        super().__init__()
        self.sampling_rate_hz = sampling_rate_hz
        self.low_band_hz = low_band_hz
        # one shared tiny MLP applied per-channel: (ratio, entropy) -> gate
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, C)
        T = x.shape[1]
        eps = 1e-8

        X = torch.fft.rfft(x, dim=1)                     # (batch, F, C), complex
        power = X.real.pow(2) + X.imag.pow(2)             # (batch, F, C)
        total_power = power.sum(dim=1, keepdim=True) + eps  # (batch, 1, C)

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sampling_rate_hz).to(x.device)  # (F,)
        low, high = self.low_band_hz
        band_mask = (freqs >= low) & (freqs <= high)       # (F,)
        # guard against a band that collapses to nothing (e.g. very short T
        # at a low sampling rate) -- fall back to the full spectrum
        if band_mask.sum() == 0:
            band_mask = torch.ones_like(freqs, dtype=torch.bool)

        low_energy = power[:, band_mask, :].sum(dim=1, keepdim=True)  # (batch, 1, C)
        ratio = (low_energy / total_power).squeeze(1)       # (batch, C), in [0, 1]

        p = power / total_power                              # (batch, F, C)
        entropy = -(p * torch.log(p + eps)).sum(dim=1)       # (batch, C)
        entropy = entropy / math.log(power.shape[1] + eps)   # normalize to ~[0, 1]

        desc = torch.stack([ratio, entropy], dim=-1)         # (batch, C, 2)
        gate = torch.sigmoid(self.mlp(desc)).squeeze(-1)     # (batch, C)
        return gate


# ---------------------------------------------------------------------------
# Combined "where to look" module: learned channel attention x spectral gate
# ---------------------------------------------------------------------------
class GatedChannelAttention(nn.Module):
    """
    Learned channel attention optionally modulated by SpectralChannelGate.

    use_spectral_gate=True  -> proposed novel module (spectral-gated).
    use_spectral_gate=False -> plain learned channel attention only, i.e. the
                                already-closed "channel attention" half of
                                literature's channel+temporal dual attention
                                (e.g. Dual-INABA) -- kept as an ablation arm.

    Combining via elementwise product then re-normalizing by the mean keeps
    the average re-weighting factor near 1 regardless of how many gates are
    active, so turning the spectral gate on/off doesn't implicitly change the
    input scale the downstream BiGRU sees.
    """

    def __init__(self, num_channels: int, sampling_rate_hz: float,
                 reduction: int = 4, low_band_hz: tuple = (0.3, 3.5),
                 use_spectral_gate: bool = True):
        super().__init__()
        self.use_spectral_gate = use_spectral_gate
        self.channel_attention = ChannelAttention(num_channels, reduction=reduction)
        if use_spectral_gate:
            self.spectral_gate = SpectralChannelGate(
                num_channels, sampling_rate_hz, low_band_hz=low_band_hz,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.channel_attention(x)          # (batch, C)
        if self.use_spectral_gate:
            weights = weights * self.spectral_gate(x)  # (batch, C)
        weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-8)
        return weights


# ---------------------------------------------------------------------------
# Temporal attention ("when to look" -- which timestep matters)
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over the time dimension, applied to
    the BiGRU's output sequence.

    Input:  x (batch, T, hidden_dim)
    Output: context (batch, hidden_dim), weights (batch, T)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        scores = self.score(x).squeeze(-1)                        # (batch, T)
        weights = torch.softmax(scores, dim=1)                     # (batch, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)    # (batch, hidden_dim)
        return context, weights


if __name__ == "__main__":
    # Smoke test across the three datasets' native (T, C, sampling_rate) shapes.
    for T, C, sr, name in [(128, 6, 50, "ucihar"), (51, 3, 20, "wisdm"), (256, 12, 100, "pamap2")]:
        x = torch.randn(8, T, C)

        gated = GatedChannelAttention(C, sampling_rate_hz=sr, use_spectral_gate=True)
        w = gated(x)
        print(f"{name:8s} GatedChannelAttention(spectral=True)  weights {tuple(w.shape)} "
              f"mean={w.mean().item():.3f}")

        plain = GatedChannelAttention(C, sampling_rate_hz=sr, use_spectral_gate=False)
        w2 = plain(x)
        print(f"{name:8s} GatedChannelAttention(spectral=False) weights {tuple(w2.shape)} "
              f"mean={w2.mean().item():.3f}")

        temp_attn = TemporalAttention(hidden_dim=16)
        seq = torch.randn(8, T, 16)
        ctx, tw = temp_attn(seq)
        print(f"{name:8s} TemporalAttention context {tuple(ctx.shape)} weights {tuple(tw.shape)} "
              f"sum={tw.sum(dim=1)[0].item():.3f}")
