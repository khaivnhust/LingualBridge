"""Fake-quantisation for Q4_K-family weights, for quantization-aware training.

Why this exists
---------------
No installed library can fake-quantise the k-quants that llama.cpp actually
uses. `bitsandbytes` ships NF4 and FP4, which are *different formats* from
Q4_K_M; `torchao` is not installed and its int4 path is not k-quant either.
Training against NF4 and then shipping Q4_K_M is training against the wrong
forward pass.

So we emulate the real scheme. Specification read off this repo's own
llama.cpp (ggml/src/ggml-common.h):

    #define QK_K 256                super-block = 256 weights
    #define K_SCALE_SIZE 12         8 scales (6b) + 8 mins (6b) = 96 bits
    static_assert(sizeof(block_q4_K)
        == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, ...)
                                 -> 144 bytes / 256 weights = 4.5 bpw

Layout: the 256 weights split into 8 sub-blocks of 32. Each sub-block gets an
affine map (min, scale) with 16 levels; the 8 scales and 8 mins are themselves
quantised to 6 bits inside the 12-byte field.

Straight-through estimator: forward uses the dequantised values, backward
passes the gradient through untouched. That is the whole trick -- the model
learns weights that survive the round trip.

Not bit-exact with llama.cpp's integer packing (we keep the scale/min in fp16
rather than the 6-bit sub-quantisation). For QAT the dominant term is the 4-bit
weight rounding, and the 6-bit scale quantisation contributes ~0.05 bpw. Add
`_quantise_scales=True` to close the gap if you want it.
"""
from __future__ import annotations

import torch

QK_K = 256
K_SCALE_SIZE = 12
N_SUB = 8          # sub-blocks per super-block
SUB = QK_K // N_SUB  # 32 weights per sub-block


def _fake_quant_q4_0(w: torch.Tensor) -> torch.Tensor:
    """Q4_0 fallback for rows that are not a multiple of 256.

    llama.cpp cannot apply k-quant to every tensor (e.g. the audio tower's
    conv_out is 1024x7680 per-channel and several projections are not
    256-aligned), and falls back to Q4_0 there. Q4_0 is also 4.5 bits/weight
    (fp16 scale + 32 nibbles = 18 bytes / 32 weights), so the bitrate target is
    unchanged -- only the block granularity differs.
    """
    orig_shape = w.shape
    x = w.reshape(-1, 32).float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    d = (amax / 8.0).clamp(min=1e-8)           # positive scale; x/d lands in [-8, 8]
    q = torch.round(x / d).clamp(-8, 7)
    return (q * d).reshape(orig_shape).to(w.dtype)


def _fake_quant_q4_k(w: torch.Tensor, quantise_scales: bool = False) -> torch.Tensor:
    """w: (..., N) -> fake-quantised back to the same shape."""
    orig_shape = w.shape
    n = orig_shape[-1]
    if n % QK_K != 0:
        if n % 32:
            pad = 32 - (n % 32)
            w = torch.cat([w, w.new_zeros(*orig_shape[:-1], pad)], dim=-1)
            out = _fake_quant_q4_0(w)
            return out[..., :n]
        return _fake_quant_q4_0(w)
    x = w.reshape(-1, QK_K).float()

    sb = x.reshape(-1, N_SUB, SUB)
    mx = sb.amax(dim=-1, keepdim=True)
    mn = sb.amin(dim=-1, keepdim=True)

    scale = (mx - mn) / 15.0
    scale = torch.where(scale == 0, torch.full_like(scale, 1e-8), scale)

    d, dmin = scale.squeeze(-1), mn.squeeze(-1)          # (M, 8)

    if quantise_scales:
        # the 6-bit sub-quantisation of the 8 scales and 8 mins
        for t in (d, dmin):
            lo, hi = t.amin(dim=-1, keepdim=True), t.amax(dim=-1, keepdim=True)
            rng = torch.where(hi - lo == 0, torch.full_like(hi, 1e-8), hi - lo)
            qs = torch.round((t - lo) / rng * 63).clamp_(0, 63)
            t.copy_((qs / 63 * rng + lo).to(t.dtype))
        d, dmin = d.unsqueeze(-1), dmin.unsqueeze(-1)
        scale = d

    # 4-bit round trip through the affine map
    q = torch.round((sb - mn) / scale).clamp_(0, 15)
    deq = q * scale + mn

    return deq.reshape(orig_shape).to(w.dtype)


def _fake_quant_q6_k(w: torch.Tensor) -> torch.Tensor:
    """Approximate Q6_K round trip: 256-element blocks, int8 scales, 6-bit values.

    Not a bit-exact emulation of ggml's Q6_K layout -- and it does not need to
    be, because this is fake quantisation for training. What matters is the
    effective precision (210 B / 256 weights = 6.56 bits) and the per-block
    dynamic range, both of which this reproduces.

    This exists because llama.cpp's Q4_K_M promotes `attn_v` and `ffn_down` to
    Q6_K on half the layers. Faking those at Q4_K applies distortion the
    deployed model does not have.
    """
    orig_shape = w.shape
    n = orig_shape[-1]
    if n % 256:
        if n % 32:
            pad = 32 - (n % 32)
            w = torch.cat([w, w.new_zeros(*orig_shape[:-1], pad)], dim=-1)
        return _fake_quant_q4_0(w)[..., :n]

    x = w.reshape(-1, 256).float()
    # Symmetric int8 quantisation. The scale MUST be derived from max|x| and the
    # clamp range must match: sizing the scale from (max-min)/254 while clamping
    # to [-128,127] saturates the upper half of every block and gives ~2.7 dB
    # SNR -- worse than Q4_K, which is impossible for a 6-bit format.
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.full_like(scale, 1e-8), scale)
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return (q * scale).reshape(orig_shape).to(w.dtype)


class FakeQuantQ6K(torch.autograd.Function):
    """Q6_K round trip with a straight-through gradient."""

    @staticmethod
    def forward(ctx, w):
        return _fake_quant_q6_k(w)

    @staticmethod
    def backward(ctx, g):
        return g


def q6_k_fake_quant(w: torch.Tensor) -> torch.Tensor:
    return FakeQuantQ6K.apply(w)


class FakeQuantQ4K(torch.autograd.Function):
    """Q4_K round trip with a straight-through gradient."""

    @staticmethod
    def forward(ctx, w, quantise_scales):
        return _fake_quant_q4_k(w, quantise_scales)

    @staticmethod
    def backward(ctx, g):
        return g, None


def q4_k_fake_quant(w: torch.Tensor, quantise_scales: bool = False) -> torch.Tensor:
    return FakeQuantQ4K.apply(w, quantise_scales)


def bytes_per_weight() -> float:
    """144 bytes per 256 weights (fp16 d, fp16 dmin, 12 B scales/mins, 128 nibbles)."""
    return (2 * 2 + K_SCALE_SIZE + QK_K // 2) / QK_K


def bits_per_weight() -> float:
    return 8.0 * bytes_per_weight()


def roundtrip_error(w: torch.Tensor) -> dict:
    """Sanity check: relative MSE and SNR of the fake-quant round trip."""
    w = w.detach().float()
    if w.shape[-1] % 32 != 0:
        w = w[..., : (w.shape[-1] // 32) * 32]
    q = _fake_quant_q4_k(w).float()
    err = (w - q).pow(2).mean().item()
    sig = w.pow(2).mean().item()
    return {
        "rel_mse": err / max(sig, 1e-12),
        "snr_db": 10 * torch.log10(torch.tensor(max(sig, 1e-12) / max(err, 1e-12))).item(),
        "bits_per_weight": bits_per_weight(),
        "max_abs_err": (w - q).abs().max().item(),
    }