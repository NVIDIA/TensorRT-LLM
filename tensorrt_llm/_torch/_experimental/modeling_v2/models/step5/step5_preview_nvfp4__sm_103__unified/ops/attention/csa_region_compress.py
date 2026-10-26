# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA region compression for the sparse-attention indexer: every complete
8-token region of indexer keys becomes one e4m3 summary vector.

Tier 4, composed from native torch (two widenings, one max, one exp, two
token-axis sums, one divide, the bf16 -> e4m3 rounding chain). Rejected at the
cheaper rungs:

- tier 1: ``torch.ops.trtllm.compressor_prefill_reduction`` and
  ``compressor_paged_kv_compress`` (DeepSeek-V4 CSA, ``compressorOp.cpp``) do a
  per-channel softmax-weighted sum, but only at compress ratios 2/4/128 and
  head dims 128/512, add a learned positional bias and overlap ratio-4 windows;
  this checkpoint compresses 8 tokens at width 256 with neither.
- tier 2: the qwen4_exp ``qsa_prefill_compress`` is a mean pool followed by a
  Gemma norm and RoPE; it has no logit operand.
- tier 3: FlashInfer ships no region compressor usable on sm_103.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Tokens per region (`sparse_config.region_block_size`).
REGION_SIZE = 8
#: Indexer key width (`sparse_config.proxy_dim`).
PROXY_DIM = 256
#: Largest finite e4m3fn magnitude; the summary saturates here.
E4M3_MAX = 448.0
#: Floor on a channel's softmax denominator, the oracle's guard against 0 / 0.
DENOMINATOR_FLOOR = 1e-20
#: Largest region count one call carries: max_seq_len 9216 / 8.
MAX_REGIONS = 1152
#: Output dtypes that hold every e4m3 value exactly.
EXACT_OUT_DTYPES = (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32)


class RegionSummaryExpectation(NamedTuple):
    """What `reference` returns: the exact-arithmetic answer and what a
    correct fp32 evaluation may differ from it by.

    `summary` is `unrounded` taken through the bf16 -> e4m3 chain, in the
    call's `out_dtype`. `magnitude` is, per channel,
    `sum_t w_t |k_t| / max(sum_t w_t, floor)`: the scale an fp32 evaluation's
    rounding error is proportional to (it equals `|unrounded|` when the
    region's keys share a sign, and exceeds it where they cancel).
    """

    summary: torch.Tensor
    unrounded: torch.Tensor
    magnitude: torch.Tensor


class CsaRegionCompress(OpWrapper):
    """For region `r`, token `t < 8`, channel `d`, with `m_r` the largest `z`
    in the region over both the token and the channel axis::

        w[r, t, d] = exp(z[r, t, d] - m_r)
        den[r, d]  = sum_t w[r, t, d]
        s[r, d]    = sum_t w[r, t, d] * k[r, t, d] / max(den[r, d], 1e-20)   (0 where den == 0)
        out[r, d]  = e4m3(clamp(bf16(s[r, d]), -448, 448))

    A per-channel softmax over the region's tokens, weighting the keys; all in
    fp32, then rounded to bf16 and from there to e4m3fn (round to nearest even
    at each step, saturating at 448). Returns a fresh contiguous `[R, D]` in
    `out_dtype`; every dtype `is_valid` admits holds the e4m3 value exactly,
    so a bf16 store and an fp8 store carry the same numbers.

    Fusion boundary: the softmax-weighted reduction and its rounding. Gathering
    a region's 8 tokens out of the paged index plane (and from the pending tail
    for a region a chunk completes), the scatter of summaries into their store,
    and which regions count as complete are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: regions (R); dist "ckpt" (keys at the measured LayerNorm+RoPE
    # scale, logits at the measured z scale; default), "spread" (channel d's
    # logits pushed d * 120/255 nats below the region maximum, through the
    # denominator floor into underflow), "midpoint" (uniform logits, keys
    # whose mean is exact in fp32 and rounds onto an e4m3 tie through bf16),
    # "saturate" (keys past 448); out_dtype (default float8_e4m3fn).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one region completes, checkpoint scale",
            spec=dict(regions=1),
        ),
        Cell(
            why="decode at max batch 128, every request completes a region",
            spec=dict(regions=128),
        ),
        Cell(
            why="decode at 128, summaries stored in a bf16 plane",
            spec=dict(regions=128, out_dtype=torch.bfloat16),
        ),
        Cell(
            why="chunked prefill: one 512-token chunk, 64 regions",
            spec=dict(regions=64),
        ),
        Cell(
            why="8192-token prefill chunk, 1024 regions",
            spec=dict(regions=1024),
        ),
        Cell(
            why="envelope: 1152 regions, max_seq_len 9216 / 8, bf16 out",
            spec=dict(regions=MAX_REGIONS, out_dtype=torch.bfloat16),
        ),
        Cell(
            why="channel logit gaps 0..120 nats: the floor and underflow",
            spec=dict(regions=128, dist="spread"),
        ),
        Cell(
            why="bf16 lands on an e4m3 tie: the double rounding is kept",
            spec=dict(regions=64, dist="midpoint"),
        ),
        Cell(
            why="keys past 448: summaries saturate at +-448",
            spec=dict(regions=16, dist="saturate"),
        ),
    )

    note = """
    Metadata consumed: none; stateless, every region independent. Call sites
    (both targets, the 23 sparse layers): the regions completed this step,
    gathered as `[R, 8, 256]` bf16 `k` (indexer key after LayerNorm and RoPE)
    and `[R, 8, 256]` bf16 `z` (raw indexer logit projection); R <= b at
    decode, at prefill <= 1024 per 8192-token chunk plus one per request whose
    carried tail the chunk completes; 1152 (max_seq_len 9216 / 8) is the
    largest cell. Out-of-place; R = 0 returns `[0, D]`.

    Semantics follow the StepFun reference compressor
    (`_csa_compress_regions_kernel` in the checkpoint's `inference/kernel.py`),
    including its two numerical choices, each certified by a cell:
    - one shift per region, the max over tokens AND channels. A channel whose
      logits sit more than ln(1e20) = 46 nats below the region maximum has
      `den < 1e-20` and is shrunk toward 0 by the floor (a per-channel shift
      would return that channel's softmax mean); past ~104 nats every weight
      underflows and the channel is exactly 0. Measured on this checkpoint
      (sparse layers 3 and 7, gsm8k prompts up to 6101 tokens): a channel's
      best logit sits at most 15.3 nats below its region maximum, so real
      inputs stay clear of the floor, and a per-channel shift gives identical
      bytes there.
    - rounding through bf16 before e4m3. Rounding fp32 straight to e4m3
      differs on 6082 of 195072 checkpoint codes (layer 3, 6101 tokens) and on
      8257 of 16384 in the `midpoint` cell.
    Saturation: the clamp to +-448 makes the summary saturate whatever the
    float -> float8_e4m3fn cast does past 448 (the CUDA cast measured here
    already saturates 449..500 to 448). |s| <= max |k|, and checkpoint keys
    stay below 8.4, so saturation is unreachable from real inputs.

    Measured against the reference compressor itself (its Triton kernel, on
    GB300) on checkpoint k/z built from the oracle's captured residuals:
    469504 summary codes over sparse layers 3 and 7 and three prompts,
    bit-identical, also at R 1 and 128; and bit-identical on the spread,
    midpoint and saturate constructions.

    Preconditions: `k` and `z` the same `[R, T, D]` shape (`is_valid`;
    measured, a `z` of `[R, T, 1]` or `[1, T, D]` broadcasts silently);
    `out_dtype` one that holds e4m3 exactly (`is_valid`; float8_e5m2 silently
    re-rounds, 458 of 1024 codes moved). Any T and D compute the same
    formula; certified only at T 8, D 256, bf16 inputs.

    CUDA graphs: capturable, replay bit-equal to eager (test). Cost, not
    gated: ~140-150 us eager at every certified R (launch-bound, about ten
    kernels), graph replay 35 us at R 128 and 47 us at R 1024.

    Hard limits: no index-typed argument. torch indexes with int64, so the
    2^31-element boundary (R = 2^31 / 2048 = 1048576 regions) is not a limit
    of this composition; the envelope's largest call is 1152 x 8 x 256 =
    2^21.2 elements.
    """

    def __call__(
        self,
        k: torch.Tensor,
        z: torch.Tensor,
        out_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        logits = z.float()
        shift = logits.amax(dim=(1, 2), keepdim=True)
        weights = torch.exp(logits - shift)
        denominator = weights.sum(dim=1)
        numerator = (weights * k.float()).sum(dim=1)
        summary = torch.where(
            denominator > 0.0, numerator / denominator.clamp_min(DENOMINATOR_FLOOR), 0.0
        )
        e4m3 = summary.to(torch.bfloat16).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return e4m3.to(out_dtype)

    def reference(
        self,
        k: torch.Tensor,
        z: torch.Tensor,
        out_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> RegionSummaryExpectation:
        """The same formula in float64, rounded once to bf16 and then to e4m3.

        float64 because the entry already computes in fp32, so an fp32
        reference would be the same arithmetic. The comparison band is derived
        from `magnitude` in the test.
        """
        logits = z.double()
        keys = k.double()
        weights = torch.exp(logits - logits.amax(dim=(1, 2), keepdim=True))
        denominator = weights.sum(dim=1)
        floored = denominator.clamp_min(DENOMINATOR_FLOOR)
        unrounded = torch.where(denominator > 0.0, (weights * keys).sum(dim=1) / floored, 0.0)
        magnitude = torch.where(denominator > 0.0, (weights * keys.abs()).sum(dim=1) / floored, 0.0)
        rounded = unrounded.to(torch.bfloat16).clamp(-E4M3_MAX, E4M3_MAX)
        summary = rounded.to(torch.float8_e4m3fn).to(out_dtype)
        return RegionSummaryExpectation(summary, unrounded, magnitude)

    def is_valid(
        self,
        k: torch.Tensor,
        z: torch.Tensor,
        out_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        # Measured: a z of [R, T, 1] or [1, T, D] beside a [R, T, D] k
        # broadcasts and returns [R, D] computed from the wrong logits.
        assert k.dim() == 3 and k.shape == z.shape, (
            f"k {tuple(k.shape)} and z {tuple(z.shape)} must both be [regions, tokens, "
            f"dim] and equal; a mismatch broadcasts silently instead of raising"
        )
        # Measured: float8_e5m2 re-rounds the e4m3 summary to 2 mantissa bits.
        assert out_dtype in EXACT_OUT_DTYPES, (
            f"out_dtype {out_dtype} does not hold every e4m3 value exactly; "
            f"use one of {EXACT_OUT_DTYPES}"
        )


csa_region_compress = CsaRegionCompress()
