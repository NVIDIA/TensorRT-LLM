# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch module references for DeepSeek-V4.1-Flash.

This is rung 2 of the reference ladder. Rung 1 is the checkpoint's own
``inference/`` implementation run end to end (the accuracy anchor and the five
frozen greedy fixtures that ``anchor.py`` produced). Rung 3 is the staircase
target. Nothing can be compared against the target until something independent
of it is known to be right, and a benchmark that passes only proves agreement
with itself -- so every module below is written from the semantics of the
checkpoint's own forward and then *aligned* against that forward's live
activations by ``refcapture.py``. Only after a module aligns does it become a
reference the later module Goals may cite.

INDEPENDENCE IS THE POINT, so read the import list as a contract: this file
imports ``torch``, ``numpy`` and (for one function) ``tokenizers`` and nothing
else. It never imports ``model``, ``kernel``, ``engram`` or ``vision`` from the
reference tree, and never imports ``tensorrt_llm``. A reference that shared a
helper with the thing it checks could not separate "my reference is wrong" from
"my port is wrong", which is the entire reason this rung exists.

Everything here is expressed in native torch ops with fp32 accumulation
wherever the reference's tilelang kernels accumulate in fp32. The quantization
grids (E4M3, E2M1, E8M0) are reimplemented arithmetically rather than by
calling the same cast the kernel calls, except for E4M3 where torch's own dtype
is the primitive rather than a helper of the implementation under test.

Shapes follow the reference's own conventions: ``[b, s, ...]`` batched, hc
copies in dim 2, attention heads in dim 2 of ``[b, s, h, d]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefConfig:
    """The checkpoint geometry this reference is parameterized by.

    Built from ``inference/config.json`` by :func:`config_from_dict`, but kept
    as its own type rather than reusing the reference's ``ModelArgs``: the
    reference's dataclass carries defaults for a toy model and behaviour
    (``get_moe_config``) that would make this file depend on it.
    """

    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_heads: int
    n_routed_experts: int
    n_activated_experts: int
    score_func: str
    gate_temp: float
    norm_topk_prob: bool
    route_scale: float
    swiglu_limit: float
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    compress_rope_theta: float
    original_seq_len: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_id: int
    engram_compressed_vocab_size: int

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    def compress_ratio(self, layer_id: int) -> int:
        return self.compress_ratios[layer_id]


#: Field names this reference needs out of the checkpoint's ``inference``
#: config. Everything else there (vision, dspark, dtype selectors, runtime
#: limits) is deliberately not part of the module semantics.
_CONFIG_FIELDS = tuple(RefConfig.__dataclass_fields__)

#: The two routing knobs the checkpoint's ``inference/config.json`` does not
#: spell out, with where each value actually comes from. They are listed
#: explicitly rather than defaulted in the dataclass so that a *newly* missing
#: field is still an error instead of silently taking a toy value.
#:
#:   gate_temp       absent from both configs; the reference's ModelArgs
#:                   default is 1.0 and nothing overrides it.
#:   norm_topk_prob  absent from inference/config.json, but the HF
#:                   config.json's text_config carries ``true``, which is also
#:                   the ModelArgs default.
_OMITTED_DEFAULTS: dict = {"gate_temp": 1.0, "norm_topk_prob": True}


def config_from_dict(raw: dict) -> RefConfig:
    """Build a :class:`RefConfig` from the checkpoint's ``inference/config.json``.

    A missing key that is not in :data:`_OMITTED_DEFAULTS` is an error rather
    than a default: a reference that silently fell back to a toy value would
    align against nothing.
    """
    kwargs: dict[str, Any] = {}
    for name in _CONFIG_FIELDS:
        if name in raw:
            value = raw[name]
        elif name in _OMITTED_DEFAULTS:
            value = _OMITTED_DEFAULTS[name]
        else:
            raise KeyError(f"config is missing {name!r}, which the reference modules need")
        if isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return RefConfig(**kwargs)


# ---------------------------------------------------------------------------
# quantization grids
#
# The reference's kernels quantize with three grids. Each is reimplemented here
# arithmetically so that a disagreement between this file and the kernel is a
# measurable fact rather than two calls into the same converter.
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0
_FP4_MAX = 6.0

#: The eight magnitudes representable in E2M1: subnormal {0, 0.5} then three
#: binades of two values each. Used by :func:`round_to_e2m1`.
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pow2_ceil_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """``2 ** ceil(log2(amax * max_inv))`` -- the reference's ``fast_round_scale``.

    The kernel computes this by pulling the exponent out of the IEEE-754 bit
    pattern and adding one when the mantissa is nonzero, which is exactly
    ``ceil(log2(v))`` for a positive normal float and exact ``log2(v)`` for a
    power of two. Written here with ``frexp`` instead of bit surgery so the two
    formulations are genuinely different expressions of the same value.
    """
    v = (amax.float() * max_inv).clamp_min(torch.finfo(torch.float32).tiny)
    mantissa, exponent = torch.frexp(v)
    # frexp returns mantissa in [0.5, 1), so v == mantissa * 2**exponent and
    # ceil(log2(v)) == exponent when mantissa is exactly 0.5, else exponent.
    e = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return torch.ldexp(torch.ones_like(v), e)


def round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to the E4M3 grid, returning fp32. Inputs must already be in range."""
    return x.to(torch.float8_e4m3fn).float()


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid with ties to even, returning fp32.

    E2M1 encodes sign, a 2-bit exponent and a 1-bit mantissa, so the magnitudes
    are ``0, 0.5, 1, 1.5, 2, 3, 4, 6``. Ties land on the even encoding, which is
    the even *index* into that list because the list is the encoding order.
    Values outside +-6 are the caller's problem: every call site clamps first,
    matching the kernel.
    """
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=x.device)
    mag = x.float().abs().clamp(max=_FP4_MAX)
    # index of the first grid point >= mag, i.e. the upper neighbour
    hi = torch.searchsorted(grid, mag.contiguous().reshape(-1)).reshape(mag.shape)
    hi = hi.clamp(max=len(_E2M1_GRID) - 1)
    lo = (hi - 1).clamp(min=0)
    lo_v, hi_v = grid[lo], grid[hi]
    d_lo, d_hi = mag - lo_v, hi_v - mag
    # ties to even index; otherwise the nearer neighbour
    tie_pick_hi = (hi % 2) == 0
    pick_hi = torch.where(d_lo == d_hi, tie_pick_hi, d_hi < d_lo)
    out = torch.where(pick_hi, hi_v, lo_v)
    return torch.copysign(out, x.float())


def act_quant_ref(
    x: torch.Tensor,
    block_size: int,
    round_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's ``act_quant`` with ``inplace=False``: E4M3 values + scales.

    Returns ``(values_fp32_on_grid, scales_fp32)`` where the values are the
    *unscaled* E4M3 codes lifted back to fp32, i.e. what the GEMM multiplies
    before applying the scale. Keeping them in fp32 rather than an fp8 tensor
    means the pure-torch GEMM below can stay a plain matmul.
    """
    if x.size(-1) % block_size:
        raise ValueError(f"last dim {x.size(-1)} is not a multiple of block {block_size}")
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(dim=-1).clamp_min(1e-4)
    s = pow2_ceil_scale(amax, 1.0 / _FP8_MAX) if round_scale else amax / _FP8_MAX
    q = round_to_e4m3((xb / s.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX))
    return q.flatten(-2), s


def act_quant_dequant_ref(
    x: torch.Tensor, block_size: int, round_scale: bool = True
) -> torch.Tensor:
    """The reference's ``act_quant(..., inplace=True)``: quantize then expand back.

    The reference uses this to force a tensor onto the fp8 grid while keeping it
    in bf16 -- the sliding-window KV cache is stored that way.
    """
    q, s = act_quant_ref(x, block_size, round_scale)
    y = q.unflatten(-1, (-1, block_size)) * s.unsqueeze(-1)
    return y.flatten(-2).to(x.dtype)


def fp4_act_quant_dequant_ref(
    x: torch.Tensor,
    block_size: int,
    e4m3_scale: bool = False,
) -> torch.Tensor:
    """The reference's ``fp4_act_quant(..., inplace=True)``.

    Two scale formats, and the difference is not cosmetic. The indexer's keys
    and queries use power-of-two E8M0 scales with a subnormal-ish floor of
    ``6 * 2**-126``; the compressed KV uses E4M3 scales with a much higher floor
    of ``6 * 2**-9``, which the kernel's own comment ties to training keeping an
    all-zero group's scale nonzero.
    """
    if x.size(-1) % block_size:
        raise ValueError(f"last dim {x.size(-1)} is not a multiple of block {block_size}")
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(dim=-1)
    if e4m3_scale:
        amax = amax.clamp_min(_FP4_MAX * 2.0**-9)
        s = round_to_e4m3(amax / _FP4_MAX)
    else:
        amax = amax.clamp_min(_FP4_MAX * 2.0**-126)
        s = pow2_ceil_scale(amax, 1.0 / _FP4_MAX)
    q = round_to_e2m1((xb / s.unsqueeze(-1)).clamp(-_FP4_MAX, _FP4_MAX))
    y = q * s.unsqueeze(-1)
    return y.flatten(-2).to(x.dtype)


def unpack_e2m1_x2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``[..., K//2]`` ``float4_e2m1fn_x2`` storage into ``[..., K]`` fp32.

    Two E2M1 values share a byte, low nibble first along K. The decode is a
    table lookup on the 16 codes rather than bit arithmetic on a float, because
    the storage dtype has no arithmetic in torch.
    """
    codes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    raw = packed.view(torch.uint8)
    lo = codes[(raw & 0x0F).long()]
    hi = codes[(raw >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# linear
#
# The two weight-space dequantization helpers that used to sit above were
# removed: nothing drove them, and an unexercised reference is not a reference
# -- which is the whole premise of this rung. The scale layouts they encoded
# are not lost, because they are applied inside the two GEMM references below
# (the fp8 grid is one value per 32x32 tile, read at row ``n // 32``; the fp4
# grid is one per 32 input channels of each output channel) and those are
# aligned against the kernel on every run. A load-time weight check that wants
# them again belongs to the Goal that samples the checkpoint, with its own
# evidence.
# ---------------------------------------------------------------------------

#: Element budget for the per-K-block intermediate inside
#: :func:`_blocked_scaled_matmul`. 64M fp32 elements is 256 MiB, which keeps a
#: prefill-length reference GEMM off the allocator's back. Grouping changes only
#: the order in which disjoint K-block partials are summed in fp32 -- the same
#: freedom the kernel's own pipelining has -- and that reordering is orders of
#: magnitude below the bf16 rounding of the result.
_MATMUL_BLOCK_BUDGET = 64 * 1024 * 1024


def _blocked_scaled_matmul(
    a_q: torch.Tensor,
    a_s: torch.Tensor,
    b_q: torch.Tensor,
    b_s_expanded: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """``sum_k (A_k @ B_k^T) * sa_k * sb_k`` over K blocks, accumulating in fp32.

    The kernel accumulates one scaled partial per K block rather than scaling a
    single full-K product, and with per-block scales those are not the same
    expression. Mirroring the block structure keeps this reference honest about
    where the rounding happens.

    ``a_s`` is ``[M, K/block]`` and ``b_s_expanded`` is ``[N, K/block]`` -- the
    weight's scale grid already broadcast to one row per output channel, since
    the fp8 grid is per 32 output channels and the fp4 grid is per channel.
    """
    m, n = a_q.shape[0], b_q.shape[0]
    nb = a_q.shape[1] // block_size
    a = a_q.reshape(m, nb, block_size)
    b = b_q.reshape(n, nb, block_size)
    # Blocks are batched rather than looped one at a time, but only as many at
    # once as fit a fixed intermediate budget: the per-block product is
    # [group, m, n], which at prefill lengths is the largest tensor here.
    group = max(1, min(nb, _MATMUL_BLOCK_BUDGET // max(1, m * n)))
    out = torch.zeros(m, n, dtype=torch.float32, device=a_q.device)
    for lo in range(0, nb, group):
        hi = min(lo + group, nb)
        part = torch.einsum("mbk,nbk->bmn", a[:, lo:hi], b[:, lo:hi])
        scale = a_s[:, lo:hi].t().unsqueeze(-1) * b_s_expanded[:, lo:hi].t().unsqueeze(-2)
        out += (part * scale).sum(dim=0)
    return out


def linear_fp8_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``x @ W^T`` for an E4M3 weight with UE8M0 32x32 scales and a quantized activation.

    This is the reference's ``linear()`` fp8 branch: the activation is quantized
    on the spot with the same power-of-two block scales, then the GEMM applies
    both scale grids per K block.
    """
    shape = x.shape[:-1]
    xf = x.reshape(-1, x.size(-1))
    a_q, a_s = act_quant_ref(xf, block_size, round_scale=True)
    b_q = weight.float()
    n, k = weight.shape
    # weight scale is [ceil(n/32), k/32]; the GEMM reads row n // 32
    b_s = scale.float().repeat_interleave(block_size, dim=0)[:n]
    y = _blocked_scaled_matmul(a_q, a_s, b_q, b_s, block_size)
    return y.to(out_dtype).reshape(*shape, n)


def linear_fp4_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    act_block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``x @ W^T`` for a packed E2M1 weight with per-32 UE8M0 scales.

    The activation is E4M3 on the same 32-wide grid; the kernel widens the FP4
    weight to FP8 and runs an FP8 x FP8 GEMM, so the arithmetic below is exact
    in fp32 and the only precision left is the two quantization grids.
    """
    shape = x.shape[:-1]
    xf = x.reshape(-1, x.size(-1))
    a_q, a_s = act_quant_ref(xf, act_block_size, round_scale=True)
    b_q = unpack_e2m1_x2(weight)
    n = b_q.shape[0]
    b_s = scale.float()
    y = _blocked_scaled_matmul(a_q, a_s, b_q, b_s, 32)
    return y.to(out_dtype).reshape(*shape, n)


def linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dispatch on the stored weight dtype, exactly as the reference's ``linear()`` does."""
    if weight.dtype == torch.float4_e2m1fn_x2:
        assert scale is not None
        return linear_fp4_ref(x, weight, scale, out_dtype=out_dtype)
    if weight.dtype == torch.float8_e4m3fn:
        assert scale is not None
        return linear_fp8_ref(x, weight, scale, out_dtype=out_dtype)
    return F.linear(x, weight)


# ---------------------------------------------------------------------------
# norms, embedding, head
# ---------------------------------------------------------------------------


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, cast back to the input dtype. ``eps`` is 1e-20 here, not 1e-6."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight.float() * xf).to(dtype)


def embedding_shard_ref(
    ids: torch.Tensor,
    weight: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> torch.Tensor:
    """One rank's partial of a vocab-sharded embedding; off-rank ids contribute zero.

    The full embedding is the sum of these across ranks, which is what the
    reference's ``all_reduce`` computes.
    """
    mask = (ids < vocab_start) | (ids >= vocab_end)
    local = (ids - vocab_start).masked_fill(mask, 0)
    y = F.embedding(local, weight)
    return y.masked_fill(mask.unsqueeze(-1), 0)


def head_shard_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One rank's vocab slice of the language-head logits, in fp32."""
    return F.linear(x.float(), weight.float())


# ---------------------------------------------------------------------------
# rotary
# ---------------------------------------------------------------------------


def precompute_freqs_cis_ref(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Complex rotary frequencies, with YaRN when ``original_seq_len > 0``.

    Adjacent element pairs are the complex parts (``is_neox=False``), which is
    the pairing this checkpoint trained with; NeoX-style half-split pairing is a
    different rotation and is one of the controls the alignment run drives.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return (
                dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen, device=device), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope_ref(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent element pairs. Returns a new tensor; never writes into ``x``.

    ``inverse`` conjugates, which is how the attention output gets the query's
    rotation removed before the grouped output LoRA.
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:
        f = f.view(1, xc.size(1), xc.size(-1))
    else:
        f = f.view(1, xc.size(1), 1, xc.size(-1))
    return torch.view_as_real(xc * f).flatten(-2).to(x.dtype)


def apply_rope_tail_ref(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate only the trailing ``rope_head_dim`` channels, leaving the NoPE head intact."""
    head, tail = x[..., :-rope_head_dim], x[..., -rope_head_dim:]
    return torch.cat([head, apply_rope_ref(tail, freqs_cis, inverse)], dim=-1)


# ---------------------------------------------------------------------------
# hyper-connections
# ---------------------------------------------------------------------------


def hc_mix_projection_ref(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Project the flattened ``[b, s, hc*dim]`` stream to ``[b, s, (2+hc)*hc]`` mixes.

    The normalizing statistic is taken over the whole flattened stream -- one
    value per token, not one per hc copy -- and multiplies the projection
    afterwards rather than the input beforehand.
    """
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    return F.linear(xf, hc_fn.float()) * rsqrt


def hc_split_sinkhorn_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the mixes into pre/post/comb and make ``comb`` doubly stochastic.

    Three details the kernel makes explicit and a casual reading loses:
    ``pre`` is ``sigmoid(.) + eps`` while ``post`` is ``2 * sigmoid(.)``; the
    first normalization is a row softmax with ``+ eps`` applied *after* the
    division, whereas every later one divides by ``sum + eps``; and the loop
    runs ``sinkhorn_iters - 1`` times because the first column normalization
    happens before it.
    """
    hc = hc_mult
    s = hc_scale.float()
    base = hc_base.float()
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * s[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * s[1] + base[hc : 2 * hc])
    comb = m[..., 2 * hc :] * s[2] + base[2 * hc :]
    comb = comb.unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre_ref(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the hc copies into one sublayer input. ``[b,s,hc,d] x [b,s,hc] -> [b,s,d]``."""
    y = torch.sum(pre_mix.float().unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)


def hc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer output back to hc copies and mix the residual in through ``comb``.

    ``comb`` is contracted over dim 2 of the residual, i.e. ``comb[.., j, k]``
    weights source copy ``j`` into destination copy ``k``. The transpose is a
    different model and is one of the alignment controls.
    """
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def identity_pre_mix_ref(
    bsz: int, seqlen: int, hc_mult: int, device: torch.device | str
) -> torch.Tensor:
    """The initial one-hot stream selector the first block's attention consumes."""
    pre = torch.zeros(bsz, seqlen, hc_mult, dtype=torch.float32, device=device)
    pre[:, :, 0] = 1.0
    return pre


# ---------------------------------------------------------------------------
# engram
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin over the witnesses that are exact below 3.3e24."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def engram_primes_ref(
    n_layers: int,
    max_ngram_size: int,
    n_heads: int,
    engram_vocab_size: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """``[layer][ngram size][head]`` bucket moduli.

    Primes are drawn in one global sequence and never reused, so every
    ``(layer, n-gram size, head)`` bucket range is disjoint -- which is what
    lets one flat table hold all of them.
    """
    out = []
    seen: set[int] = set()
    for _ in range(n_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes = []
            current = engram_vocab_size - 1
            for _ in range(n_heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        out.append(tuple(per_ngram))
    return tuple(out)


def engram_offsets_ref(primes: tuple[tuple[tuple[int, ...], ...], ...]) -> torch.Tensor:
    """Row offset of every ``(layer, hash column)`` bucket: the exclusive prefix sum of its primes."""
    rows = []
    for layer in primes:
        flat = [p for per_ngram in layer for p in per_ngram]
        rows.append(np.cumsum([0, *flat[:-1]]))
    return torch.tensor(np.array(rows))


def engram_multipliers_ref(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab: int
) -> torch.Tensor:
    """One odd multiplier per ``(layer, lookback)``, bounded so int64 cannot overflow."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        values = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def compressed_token_map_ref(tokenizer) -> tuple[list[int], int]:
    """Collapse token ids that normalize alike onto a smaller id space.

    Hashing over these ids is what makes ``" The"``, ``"the"`` and ``"THE"``
    share an n-gram. The size of the collapsed space is not just a bound: every
    hash multiplier is derived from it, so a mismatch rehashes the whole table.
    """
    from tokenizers import Regex, normalizers

    # A private-use codepoint, so a token that is exactly one space survives
    # Strip() instead of collapsing to the empty string and merging with
    # unrelated tokens. Spelled as an escape because the character itself is
    # invisible and would not survive casual editing.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        # U+FFFD: a partial UTF-8 byte token has nothing to normalize, so it
        # is keyed by its raw form instead.
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


#: Sentinel the reference writes into its compressed-token history for a
#: position that must not take part in any n-gram (an image span).
ENGRAM_DEAD = -1


def ngram_hash_ref(
    history: torch.Tensor,
    start_pos: int,
    seqlen: int,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    max_ngram_size: int,
) -> torch.Tensor:
    """Hash ids for the n-grams ending at each position.

    ``history`` is the rank's compressed-token history ``[b, >= start+seqlen]``,
    already filled through ``start_pos + seqlen``. Look-back stops at the start
    of the sequence and at any dead position, so an n-gram never spans one; both
    cases substitute ``pad_id``, which is what training did.

    Returns ``[b, seqlen, n_engram_layers, (max_ngram_size - 1) * n_heads]``.
    """
    batch = history.size(0)
    device = history.device
    positions = torch.arange(start_pos, start_pos + seqlen, device=device).expand(batch, seqlen)
    tokens = []
    blocked = torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(max_ngram_size):
        source = history.gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == ENGRAM_DEAD)
        tokens.append(torch.where(blocked, pad_id, source))
    stacked = torch.stack(tokens, dim=-1)
    products = stacked.unsqueeze(2) * multipliers
    rolling = products[..., 0]
    hashes = []
    for i in range(1, max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


def engram_lookup_shard_ref(
    hash_ids: torch.Tensor,
    table: torch.Tensor,
    table_scale: torch.Tensor,
    row_start: int,
    row_end: int,
    block_size: int = 32,
) -> torch.Tensor:
    """One rank's partial of the sharded n-gram table lookup, dequantized to bf16.

    Rows another rank owns contribute exactly zero, so the sum over ranks is the
    full lookup. Dequantization is fp8 value x UE8M0 scale per 32 channels, done
    in fp32 before the bf16 cast -- the order matters at this dynamic range.
    """
    mask = (hash_ids < row_start) | (hash_ids >= row_end)
    local = (hash_ids - row_start).masked_fill(mask, 0)
    values = F.embedding(local, table)
    scales = F.embedding(local, table_scale)
    out = values.float().unflatten(-1, (-1, block_size)) * scales.float().unsqueeze(-1)
    out = out.flatten(-2).to(torch.bfloat16)
    return out.masked_fill(mask.unsqueeze(-1), 0)


def engram_gate_ref(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    dim: int,
    eps: float,
    clamp_value: float = 1e-6,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gate the n-gram value into the residual stream.

    ``x`` is ``[b,s,hc,dim]``, ``key`` ``[b,s,hc,dim]`` and ``value``
    ``[b,s,dim]``. The normalizing statistic is per ``(token, hc copy)`` over
    ``dim`` -- not joint over the copies -- and the gate takes a *signed square
    root* of the normalized dot before the sigmoid, matching the training
    kernel. ``q_weight`` and ``k_weight`` only ever appear as their product.
    """
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    k = key.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h * weight * k).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def window_topk_idxs_ref(window_size: int, bsz: int, seqlen: int, start_pos: int) -> torch.Tensor:
    """Which sliding-window ring slots each query may read; ``-1`` means "holds nothing".

    Prefill gives every query its own causal window. Decode has one query that
    sees the whole ring, listed oldest first -- and while the ring is still
    filling, the slots past ``start_pos`` are ``-1`` rather than stale.
    """
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def compressor_prefill_ref(
    x: torch.Tensor,
    ratio: int,
    wkv: torch.Tensor,
    wgate: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Pool complete groups of ``ratio`` tokens into one pre-RoPE KV latent.

    Returns ``(latent, kv_tail, score_tail)``. The tail is the incomplete
    trailing group, which the reference parks in state and completes on a later
    step; losing it is one of the failure modes the state canary exists for.

    At ratio 1 there is nothing to pool: no gate, no fp32 promotion, and the
    projection stays in the checkpoint's bf16. Keeping ratio 1 distinct from
    ratio 2 is a semantic difference, not an optimization.
    """
    if ratio == 1:
        kv = F.linear(x, wkv)
        return rms_norm_ref(kv, norm_weight, norm_eps), None, None

    xf = x.float()
    kv = F.linear(xf, wkv.float())
    assert wgate is not None
    score = F.linear(xf, wgate.float())
    seqlen = x.size(1)
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    kv_tail = score_tail = None
    if remainder:
        kv, kv_tail = kv.split([cutoff, remainder], dim=1)
        score, score_tail = score.split([cutoff, remainder], dim=1)
    if cutoff == 0:
        return None, kv_tail, score_tail
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio))
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    return rms_norm_ref(pooled.to(out_dtype), norm_weight, norm_eps), kv_tail, score_tail


def compressor_decode_ref(
    x: torch.Tensor,
    ratio: int,
    start_pos: int,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    """One decode step of the compressor; emits a latent only when a group just completed.

    ``kv_state`` / ``score_state`` are ``[ratio, head_dim]`` for the single row
    being decoded and are updated in place, mirroring the reference's persistent
    buffers. ``score_state`` must start at ``-inf`` so unfilled slots take no
    softmax mass.
    """
    if ratio == 1:
        return rms_norm_ref(F.linear(x, wkv), norm_weight, norm_eps)
    xf = x.float()
    kv = F.linear(xf, wkv.float())
    assert wgate is not None
    score = F.linear(xf, wgate.float())
    slot = start_pos % ratio
    kv_state[slot] = kv.squeeze(0).squeeze(0)
    score_state[slot] = score.squeeze(0).squeeze(0)
    if (start_pos + 1) % ratio != 0:
        return None
    pooled = (kv_state * score_state.softmax(dim=0)).sum(dim=0)
    pooled = pooled.view(1, 1, -1).to(out_dtype)
    return rms_norm_ref(pooled, norm_weight, norm_eps)


def select_candidate_blocks_ref(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the ``topk_blocks`` best blocks per query.

    A block scores as its best position. The block holding the query's newest
    position is pinned in with ``+inf`` because it is only partly filled and
    would otherwise lose to an older full block; dropping that pin is a control
    the alignment run drives. Positions already at ``-inf`` are unreachable, so
    a block that scores ``-inf`` is not yet visible and its top-k pick is
    discarded rather than kept.
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def indexer_scores_ref(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Rectified per-head index logits combined by the per-head weights.

    ``index_q`` is ``[b, s, h, d]`` post-RoPE and post-FP4-round-trip,
    ``index_k`` ``[b, t, d]`` likewise; ``weights`` is ``[b, s, h]`` already
    carrying the ``softmax_scale * n_heads ** -0.5`` factor. The ReLU is applied
    to the raw score *before* the head weighting, so a negative score
    contributes nothing rather than contributing negatively.
    """
    score = torch.einsum("bshd,btd->bsht", index_q.float(), index_k.float())
    return (score.relu() * weights.float().unsqueeze(-1)).sum(dim=2)


def indexer_topk_ref(
    index_score: torch.Tensor,
    compress_lens: torch.Tensor | int,
    index_topk: int,
    end_pos: int,
    ratio: int,
    offset: int,
) -> torch.Tensor:
    """Final selection: top-k by score, re-sorted into position order, shifted by ``offset``.

    Unreachable picks come back as ``-1`` rather than silently pointing at
    position 0, and ``offset`` is where the sliding-window rows end in the
    concatenated KV -- the compressed rows follow them.
    """
    topk = min(index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


#: The kernel's KV tile. The online softmax rescales once per tile and casts
#: its numerators to bf16 against the *running* max, so the tile width is part
#: of the arithmetic and not a performance knob: a reference that used the
#: global max instead rounds different values and lands about 1.2 bf16 ULP away
#: at this checkpoint's geometry, which is exactly what was measured before
#: this was made faithful.
_SPARSE_ATTN_BLOCK = 64


def sparse_attn_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Index-gathered MQA with one sink logit per head.

    ``q`` is ``[b, m, h, d]``, ``kv`` ``[b, n, d]`` (one shared latent head) and
    ``topk_idxs`` ``[b, m, topk]`` int32 with ``-1`` for "nothing here". The
    sink enters the denominator only, so a head with a large sink attenuates
    every value without contributing one of its own, and a row whose indices are
    all ``-1`` returns exactly zero rather than NaN -- the kernel gets that from
    a finite ``-1e30`` running-max floor instead of ``-inf``, and so does this.

    The tiled online softmax is reproduced rather than collapsed into one pass.
    It is not an optimization here: the kernel casts each tile's numerators to
    bf16 before the PV product, against the running max at that point, so the
    values being rounded depend on tile order. Computing the same thing with a
    global max is a *different* rounding, and it measured about 1.2 bf16 ULP
    away on the real checkpoint -- enough to miss a dtype-default gate and be
    mistaken for a semantic gap.
    """
    b, m, h, d = q.shape
    block = _SPARSE_ATTN_BLOCK
    topk = topk_idxs.size(-1)
    qf = q.float()
    acc_o = torch.zeros(b, m, h, d, dtype=torch.float32, device=q.device)
    sum_exp = torch.zeros(b, m, h, dtype=torch.float32, device=q.device)
    running_max = torch.full((b, m, h), -1e30, dtype=torch.float32, device=q.device)
    for lo in range(0, topk, block):
        idx = topk_idxs[..., lo : lo + block].long()
        valid = idx >= 0
        tile = kv.gather(1, idx.clamp_min(0).reshape(b, -1, 1).expand(-1, -1, d))
        tile = (tile.reshape(b, m, -1, d) * valid.unsqueeze(-1)).float()
        scores = torch.einsum("bmhd,bmkd->bmhk", qf, tile) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
        prev_max = running_max
        running_max = torch.maximum(prev_max, scores.amax(dim=-1))
        rescale = torch.exp(prev_max - running_max)
        p = torch.exp(scores - running_max.unsqueeze(-1))
        sum_exp = sum_exp * rescale + p.sum(dim=-1)
        acc_o = acc_o * rescale.unsqueeze(-1)
        acc_o = acc_o + torch.einsum("bmhk,bmkd->bmhd", p.to(torch.bfloat16).float(), tile)
    sum_exp = sum_exp + torch.exp(attn_sink.float().view(1, 1, h) - running_max)
    return (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)


def output_lora_a_ref(
    o: torch.Tensor,
    wo_a: torch.Tensor,
    n_local_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """The grouped half of the output LoRA: ``[b,s,h*d] -> [b,s,groups,rank]``.

    ``wo_a`` is stored flat as ``[n_local_groups * o_lora_rank, heads_per_group *
    head_dim]`` but is block-diagonal over groups -- group ``g`` projects only
    its own heads -- so it is an einsum and not a dense matmul. Treating it as a
    dense ``[groups*rank, h*d]`` projection is the V3-family misreading, and it
    is one of the alignment controls.

    The contraction runs in fp32 and the result is cast back to the checkpoint's
    bf16, which is what the reference's ``torch.einsum`` on bf16 inputs with an
    fp32 accumulator does.
    """
    b, s = o.shape[0], o.shape[1]
    og = o.reshape(b, s, n_local_groups, -1)
    a = wo_a.reshape(n_local_groups, o_lora_rank, -1)
    return torch.einsum("bsgd,grd->bsgr", og.float(), a.float()).to(o.dtype)


def output_lora_ref(
    o: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    n_local_groups: int,
    o_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The grouped output LoRA: eight block-diagonal A projections, then one B.

    Returns ``(partial_fp32, lora_a)``. The partial is this rank's share of the
    row-parallel B projection; the sum over ranks is the full output. It is
    rounded to the checkpoint's bf16 *before* the widening, because the
    reference's ``RowParallelLinear`` reduces the bf16 GEMM result rather than
    an fp32 one. ``lora_a`` comes back because the A half is the only part of
    this module with an observable of its own -- the reference computes it
    inline and feeds it straight to B -- so returning it here is what lets the
    caller compare both without running the projection twice.
    """
    lora_a = output_lora_a_ref(o, wo_a, n_local_groups, o_lora_rank)
    y = linear_ref(lora_a.flatten(2), wo_b, wo_b_scale, out_dtype=o.dtype)
    return y.float(), lora_a


# ---------------------------------------------------------------------------
# mixture of experts
# ---------------------------------------------------------------------------


def gate_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    score_func: str,
    gate_temp: float,
    norm_topk_prob: bool,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router scores, selection and weights -- all in fp32.

    ``sqrtsoftplus`` is this checkpoint's scoring function and is not the V3
    family's sigmoid: ``sqrt(softplus(s))`` is unbounded above and never
    saturates, so a sigmoid substitution changes both the selection and the
    weights. The correction bias steers *selection only*; the returned weights
    are gathered from the unbiased scores. The renormalizer adds ``1e-20``,
    which is a training constant here and not ``norm_eps``.
    """
    scores = F.linear(x.float(), weight.float()) / gate_temp
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    elif score_func == "sqrtsoftplus":
        scores = F.softplus(scores).sqrt()
    else:
        raise ValueError(f"unknown score_func {score_func!r}")
    indices = (scores + bias.float()).topk(topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if norm_topk_prob and topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * route_scale
    return weights, indices


def clamped_swiglu_ref(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """SiLU-and-multiply with this checkpoint's asymmetric clamp.

    The up branch is clamped on both sides and the gate branch only from above.
    Plain unclamped SwiGLU is a different function, and so is clamping both
    branches symmetrically; the asymmetry comes from training.
    """
    g, u = gate.float(), up.float()
    if limit > 0:
        u = torch.clamp(u, min=-limit, max=limit)
        g = torch.clamp(g, max=limit)
    return F.silu(g) * u


def expert_ref(
    x: torch.Tensor,
    w1: torch.Tensor,
    w1_s: torch.Tensor | None,
    w2: torch.Tensor,
    w2_s: torch.Tensor | None,
    w3: torch.Tensor,
    w3_s: torch.Tensor | None,
    limit: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """One SwiGLU expert. ``w1`` is the gate branch, ``w3`` the up branch.

    Swapping ``w1`` and ``w3`` is not symmetric because only one of them is
    clamped from below; it is one of the discriminating controls.
    """
    dtype = x.dtype
    gate = linear_ref(x, w1, w1_s, out_dtype=dtype).float()
    up = linear_ref(x, w3, w3_s, out_dtype=dtype).float()
    h = clamped_swiglu_ref(gate, up, limit)
    if weights is not None:
        h = weights * h
    return linear_ref(h.to(dtype), w2, w2_s, out_dtype=dtype)


def moe_local_partial_ref(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    expert_params,
    expert_start: int,
    expert_end: int,
    limit: float,
) -> torch.Tensor:
    """This rank's partial of the routed-expert sum, in fp32.

    ``expert_params`` maps a global expert id to
    ``(w1, w1_s, w2, w2_s, w3, w3_s)``. Only the rank's own contiguous window is
    computed and every other row contributes zero, so the sum over ranks is the
    full routed output -- which is what the reference's ``all_reduce``
    computes and what a reduce-scatter assembly has to reproduce.
    """
    y = torch.zeros_like(x, dtype=torch.float32)
    for expert_id in range(expert_start, expert_end):
        rows, slot = torch.where(indices == expert_id)
        if rows.numel() == 0:
            continue
        params = expert_params(expert_id)
        if params is None:
            continue
        w1, w1_s, w2, w2_s, w3, w3_s = params
        out = expert_ref(
            x[rows],
            w1,
            w1_s,
            w2,
            w2_s,
            w3,
            w3_s,
            limit,
            weights[rows, slot, None],
        )
        y[rows] += out.float()
    return y


# ---------------------------------------------------------------------------
# module-level compositions
#
# The primitives above are the vocabulary; these are the five modules the Goal
# split names, assembled from them. They live here rather than in the capture
# harness for one concrete reason: a later Goal compares its *target* module
# against these, in the target's own environment, and must not have to import a
# harness that reaches for the checkpoint's `inference/` tree to do it.
#
# Each takes the collective-free part and returns this rank's partial where the
# reference all-reduces, so the caller owns the exchange and its ordering. That
# split is deliberate -- the collective is a property of the parallel layout,
# not of the module's arithmetic.
# ---------------------------------------------------------------------------


def engram_module_ref(
    x: torch.Tensor,
    rows: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    hc_mult: int,
    dim: int,
    eps: float,
    clamp_value: float = 1e-6,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The Engram sublayer, from the already-gathered table rows to the stream.

    ``rows`` is the dequantized ``[b, s, n_hash_cols, head_dim]`` lookup -- the
    *summed* one, because the table is row-sharded and each rank holds only part
    of it. One projection turns it into ``hc_mult`` keys plus one shared value,
    and the gate is a per-(token, copy) normalized dot of the stream against its
    key. Nothing here is the n-gram hashing or the lookup; both are separate
    entries because both carry state the arithmetic does not.
    """
    kv = linear_ref(rows.flatten(-2), wkv, wkv_scale)
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.unflatten(-1, (hc_mult, dim))
    return engram_gate_ref(x, key, value, q_weight, k_weight, dim, eps, clamp_value, token_mask)


def moe_module_ref(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    expert_params,
    expert_start: int,
    expert_end: int,
    shared_params,
    topk: int,
    score_func: str,
    gate_temp: float,
    norm_topk_prob: bool,
    route_scale: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routing, this rank's routed partial, and the replicated shared expert.

    Returns ``(routed_partial_fp32, shared_fp32, weights, indices)``. The routed
    partial is *this rank's* contribution only; summing it across ranks and then
    adding the shared expert is the full output, and that order is the
    reference's -- the shared expert is added after the reduction, not inside
    it, so a rank that folded it in early would count it four times.
    """
    weights, indices = gate_ref(
        x, gate_weight, gate_bias, topk, score_func, gate_temp, norm_topk_prob, route_scale
    )
    routed = moe_local_partial_ref(
        x, weights, indices, expert_params, expert_start, expert_end, limit
    )
    w1, w1_s, w2, w2_s, w3, w3_s = shared_params
    shared = expert_ref(x, w1, w1_s, w2, w2_s, w3, w3_s, limit, None).float()
    return routed, shared, weights, indices


def attention_module_ref(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    freqs_cis: torch.Tensor,
    n_local_heads: int,
    head_dim: int,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    softmax_scale: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The attention sublayer end to end, from its normed input to this rank's partial.

    ``kv`` and ``topk_idxs`` are inputs rather than computed here, and the split
    is deliberate: they are the *concatenation* of the sliding-window ring and
    the shared compressed pool, both of which are cross-layer cache state owned
    by the parallel and caching layout rather than by this module's arithmetic.
    Every one of their constituents has its own boundary -- the window index
    map, the compressor, the indexer, the index keys -- so what is left here is
    exactly the part that belongs to attention: the q-LoRA, the rotation, the
    gathered MQA with sinks, the inverse rotation and the grouped output LoRA.

    Returns ``(partial_fp32, lora_a, o_before_inverse)``. The partial is this
    rank's contribution to the row-parallel B projection; summing it over ranks
    and casting back to the input dtype is the full sublayer output. The other
    two are separately comparable intermediates: ``lora_a`` is the grouped half
    of the output LoRA, and ``o_before_inverse`` is the attention result with
    the query's rotation still on it, which is what a caller builds the
    missing-inverse-rotation control from.
    """
    qr = rms_norm_ref(linear_ref(x, wq_a, wq_a_scale, out_dtype=x.dtype), q_norm_weight, norm_eps)
    q = linear_ref(qr, wq_b, wq_b_scale, out_dtype=x.dtype).unflatten(-1, (n_local_heads, head_dim))
    q = apply_rope_tail_ref(q, freqs_cis, rope_head_dim)
    o_raw = sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale)
    o = apply_rope_tail_ref(o_raw, freqs_cis, rope_head_dim, inverse=True)
    partial, lora_a = output_lora_ref(
        o.flatten(2), wo_a, wo_b, wo_b_scale, n_local_groups, o_lora_rank
    )
    return partial, lora_a, o_raw


def block_module_ref(
    x: torch.Tensor,
    pre_mix: torch.Tensor,
    attn_out: torch.Tensor,
    ffn_out: torch.Tensor,
    hc_attn: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    hc_ffn: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    attn_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    norm_eps: float,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    mid: torch.Tensor | None = None,
    immediate_pre: bool = False,
) -> dict[str, torch.Tensor]:
    """One block's Hyper-Connection wiring around two given sublayer results.

    ``attn_out`` and ``ffn_out`` are inputs for the same reason ``kv`` is an
    input above: both sublayers carry cache and collective state of their own
    and have their own boundaries. What this composition owns is the part a
    reading of the source can get wrong without any kernel being involved --
    **which** coefficients each sublayer consumes.

    The schedule is delayed by one sublayer. A sublayer's ``hc_mixes`` produces
    the ``pre`` the *next* one consumes, so attention takes the mix the previous
    layer's FFN produced (the caller's ``pre_mix``) while its own ``attn_pre``
    goes to this block's FFN. Its ``post`` and doubly-stochastic ``comb`` apply
    to the current result. ``immediate_pre=True`` builds the control: each
    sublayer consuming the mix it produced itself, which is the single most
    plausible misreading and has to land outside the gate.

    ``mid`` is the stream between the two sublayers. Pass the native one: the
    composed value is returned as ``mid_composed`` and compared separately, and
    driving the second half from it instead would compound a bf16 rounding
    difference through a 20,480-wide fp32 projection, where it no longer means
    anything about the wiring. Each link is checked against the source's own
    value of the previous link, which is what keeps the fp32 gates meaningful.

    ``hc_attn`` / ``hc_ffn`` are ``(fn, scale, base)``.
    """
    attn_fn, attn_scale, attn_base = hc_attn
    ffn_fn, ffn_scale, ffn_base = hc_ffn
    attn_pre, attn_post, attn_comb = hc_split_sinkhorn_ref(
        hc_mix_projection_ref(x, attn_fn, norm_eps),
        attn_scale,
        attn_base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
    )
    attn_input = rms_norm_ref(
        hc_pre_ref(x, attn_pre if immediate_pre else pre_mix), attn_norm_weight, norm_eps
    )
    mid_composed = hc_post_ref(attn_out, x, attn_post, attn_comb)
    mid = mid_composed if mid is None else mid
    ffn_pre, ffn_post, ffn_comb = hc_split_sinkhorn_ref(
        hc_mix_projection_ref(mid, ffn_fn, norm_eps),
        ffn_scale,
        ffn_base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
    )
    ffn_input = rms_norm_ref(
        hc_pre_ref(mid, pre_mix if immediate_pre else attn_pre), ffn_norm_weight, norm_eps
    )
    out = hc_post_ref(ffn_out, mid, ffn_post, ffn_comb)
    return {
        "attn_input": attn_input,
        "ffn_input": ffn_input,
        "mid_composed": mid_composed,
        "out": out,
        "next_pre_mix": ffn_pre,
        "attn_comb": attn_comb,
        "ffn_comb": ffn_comb,
        "attn_post": attn_post,
        "ffn_post": ffn_post,
    }


def index_k_ref(
    latent: torch.Tensor,
    wk: torch.Tensor,
    wk_scale: torch.Tensor | None,
    k_norm_weight: torch.Tensor,
    norm_eps: float,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """The indexer's shared key for one compressed position.

    Derived from the compressor's *pre-RoPE* latent, which is why the indexer
    has to run before attention rotates and requantizes that same storage. The
    key is then held in packed E2M1 with UE8M0 scales per 32 channels -- a
    different format from the compressed KV beside it, which uses E4M3 scales
    per 16, and the two are not interchangeable.
    """
    k = rms_norm_ref(linear_ref(latent, wk, wk_scale), k_norm_weight, norm_eps)
    k = apply_rope_tail_ref(k, freqs_cis, rope_head_dim)
    return fp4_act_quant_dequant_ref(k, 32, e4m3_scale=False)


def index_q_ref(
    qr: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    n_local_heads: int,
    index_head_dim: int,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """The indexer's per-head query, from the same q-LoRA the main path uses."""
    q = linear_ref(qr, wq_b, wq_b_scale).unflatten(-1, (n_local_heads, index_head_dim))
    q = apply_rope_tail_ref(q, freqs_cis, rope_head_dim)
    return fp4_act_quant_dequant_ref(q, 32, e4m3_scale=False)


def indexer_mask_ref(
    index_score: torch.Tensor,
    start_pos: int,
    seqlen: int,
    ratio: int,
    end_pos: int,
    candidates: torch.Tensor | None,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor | int]:
    """Mask the unreachable compressed positions, and the candidate blocks if any.

    A compressed block only becomes visible once the query has passed its last
    token, so prefill masks per query and decode is a single count. Returns the
    masked score and the visible length, because the final top-k needs both.
    """
    if start_pos == 0:
        lens: torch.Tensor | int = (torch.arange(1, seqlen + 1, device=device) // ratio).unsqueeze(
            -1
        )
        index_score = index_score.masked_fill(
            torch.arange(seqlen // ratio, device=device) >= lens, -torch.inf
        )
    else:
        lens = end_pos // ratio
    if candidates is not None:
        index_score = index_score.masked_fill(~candidates, -torch.inf)
    return index_score, lens
