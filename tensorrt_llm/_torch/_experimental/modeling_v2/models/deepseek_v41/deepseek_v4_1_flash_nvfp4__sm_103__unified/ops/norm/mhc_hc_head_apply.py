# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection collapse with caller-supplied mixes via the mHC hc_head_apply kernel.

The mixes and the RMS statistic are inputs, not derived from ``x``: they may come
from a different (earlier) stream than the one being collapsed, which is how
DeepSeek-V4.1's lagged ``hc_pre`` uses this op.

The op is `torch.ops.trtllm.mhc_hc_head_apply`, registered in `cpp/tensorrt_llm/thop/mhcOp.cpp`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The op is built with fast math and is not bit-identical to torch; measured on sm_103 between
#: 0.999863 and 1.0 of elements are, in every certified cell. Every wrong variant the test
#: rejects keeps at most 0.40.
_MIN_BITEXACT = 0.999

#: Row counts DeepSeek-V4.1 makes its call at, from the target's runtime-value dumps: the trunk's
#: ('mhc', T, tile) records (T 1..16384, engines up to max_batch_size 256 / max_num_tokens 16384,
#: CUDA-graph and eager, dep4 and tp4, 128K and 1M max_seq_len, the long-context envelope) and
#: the DSpark drafter's ('dspark.mhc', N, tile) records (N = 5G); plus the tops of the step range
#: (16384) and of the decoder bounded-replay pass (32768).
_TARGET_ROWS = tuple(
    sorted(
        {
            1, 2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 24, 29, 32, 34, 48, 60, 96, 108, 127, 128,
            129, 134, 144, 168, 192, 228, 240, 288, 330, 336, 384, 432, 462, 480, 528, 546, 576,
            597, 610, 624, 672, 720, 768, 780, 852, 888, 890, 1018, 1152, 1536, 1906, 1935, 2033,
            2038, 2048, 2049, 2252, 2778, 2784, 3870, 4208, 4245, 4388, 4394, 4559, 4591, 4868,
            7835, 7855, 7893, 7908, 8147, 8191, 8192, 8467, 9193, 9240, 9364, 9370, 9524, 10441,
            10444, 10447, 10451, 16298, 16308, 16316, 16320, 16322, 16327, 16328, 16329, 16333,
            16340, 16378, 16384,
        }
        | {
            5, 10, 15, 20, 40, 50, 80, 90, 110, 115, 120, 125, 140, 160, 190, 200, 240, 275, 280,
            320, 360, 370, 385, 400, 440, 455, 480, 520, 560, 580, 585, 600, 640, 650, 665, 710,
            715, 720, 960, 1280,
        }
        | {4095, 8191, 8193, 16383, 16385, 32767, 32768}
    )
)  # fmt: skip


def pre_weights(
    mixes: torch.Tensor,
    sqrsum: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    mult: int,
    K: int,
    norm_eps: float,
    eps: float,
) -> torch.Tensor:
    """The collapse weights `pre [rows, mult]` the op forms from its statistics, in fp32 torch."""
    rstd = torch.rsqrt(sqrsum / K + norm_eps)
    return torch.sigmoid(mixes * rstd[:, None] * scale[0] + base[:mult]) + eps


def collapse_error_ratio(out: torch.Tensor, exact: torch.Tensor, terms: torch.Tensor) -> float:
    """max(|out - exact| / bound) under the per-element gate of `MhcHcHeadApply.compare`.

    An element that is NaN in both `out` and `exact` (a row the op does not write, NaN-filled
    by the caller) counts as exact.
    """
    if not out.numel():
        return 0.0
    bound = 2.0**-7 * exact.abs() + 16 * 2.0**-24 * terms + torch.finfo(torch.float64).tiny
    err = (out.double() - exact).abs()
    err = torch.where(out.isnan() & exact.isnan(), torch.zeros_like(err), err)
    return (err / bound).max().item()


class MhcHcHeadApply(OpWrapper):
    """Collapse a `mult`-copy residual stream into one hidden vector per token, from given mixes.

    For every token `t < M`::

        rstd[t]   = rsqrt(sqrsum[t] / K + norm_eps)
        pre[t, j] = sigmoid(mixes[t, j] * rstd[t] * scale[0] + base[j]) + eps         j < mult
        out[t, :] = bf16( sum_{j < mult} pre[t, j] * x[t, j, :] )     fp32 accumulation, j in order

    `pre` is a function of `mixes[t]`, `sqrsum[t]`, `scale[0]`, `base[:mult]`, `K`, `norm_eps`
    and `eps` only -- `x` is never read to form it, so the statistics may describe a different
    (earlier) stream than the one being collapsed. That cross-stream use is what is certified.

    Fusion boundary: the call does the RMS scaling, the sigmoid, the `+ eps`, the weighted sum and
    the bf16 cast. The caller owns the mixes GEMM and sum of squares (and keeping them until the
    consuming sublayer), and any norm applied to the collapsed vector afterwards; `post` and
    `comb` are not computed here. Writes rows `[0, M)` of `out` in place and returns `out`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"lagged collapse M={m}: independent producer and consumer streams",
                spec=dict(M=m, eps=1e-6, saturated=False),
            )
            for m in (1, 5, 6, 32, 1000, 8192)
        )
        + (
            Cell(
                why="saturated sigmoid M=256: pre ~ eps and ~ 1 + eps on two copies",
                spec=dict(M=256, eps=1e-6, saturated=True),
            ),
            Cell(
                why="eps=0.25 M=64: the + eps after the sigmoid is visible in bf16",
                spec=dict(M=64, eps=0.25, saturated=False),
            ),
        )
        + tuple(
            Cell(
                why=f"target call M={m}: mult 4, hidden 5120, K 20480, eps 1e-6",
                spec=dict(M=m, eps=1e-6, saturated=False),
            )
            for m in _TARGET_ROWS
        )
    )

    note = """
    Every cell: mult 4, hidden_size 5120, K 20480, norm_eps 1e-20, fresh producer and consumer
    streams drawn independently, and (M >= 3) an all-zero padded row (sqrsum 0, so rstd 1e10
    and pre = sigmoid(base) + eps, finite) and an attention-sink row of stream norm ~2e6.

    DeepSeek-V4.1's lagged hc_pre. Pass the statistics of the producer stream through the
    PRODUCER sublayer's `hc_*_fn`, that sublayer's `scale` / `base`, and the current stream as
    `x`: the attention input of layer L >= 1 uses the stream entering layer L-1's FFN with
    `hc_ffn_*` of L-1 (x: the stream entering layer L, after the Engram add on 1 and 14); the
    FFN input of layer L uses the stream entering layer L's attention with `hc_attn_*` of L (x:
    the stream after L's attention hc_post); the final-norm input uses the stream entering layer
    39's FFN with `hc_ffn_*` of 39 (x: layer 39's output; the checkpoint has no hc_head
    tensors). Layer 0's attention input and each DSpark stage's first collapse use the one-hot
    pre [1, 0, 0, 0], which this op cannot produce: that is copy 0 of the stream, not a call.
    Each site was checked against the checkpoint's golden layer dumps. Wiring the consuming
    sublayer's `scale` / `base` instead gives row-relative error median 0.02 to 0.81.

    `mixes[t, j]` is the RAW product of the flattened producer row with row j of `hc_*_fn` (no
    rsqrt applied) and `sqrsum[t]` the SUM (not mean) of its squares: `y[:, :mult]` and `r` of
    `torch.ops.trtllm.mhc_gemm_sqrsum_fma`, with `y[:, :mult]` copied to a contiguous buffer.
    Sibling ops that collapse with the fresh `pre` of the same stream
    (`torch.ops.trtllm.mhc_big_fuse`'s `layer_input`, `torch.ops.trtllm.mhc_fused_hc`) compute
    the wrong collapse under V4.1's lag.

    Metadata consumed: none. Stateless; no allocation, no host synchronization; capturable.

    Preconditions: fp32 `mixes`, `sqrsum`, `scale`, `base`; bf16 `x`, `out` -- a wrong dtype
    raises RuntimeError before launch. The op reads no shape or stride; the layouts it silently
    miscomputes are in `is_valid`. A contiguous view with a storage offset is fine. Only
    `scale[0]` and `base[:mult]` are read (the checkpoint's [3] and [24] pass as is). Rows >= M
    of `out` are not written, no input is modified, and M = 0 touches nothing. CUDA tensors
    (a CPU `mixes` was accepted without error on GB300; not certified). hidden_size % 8 == 0
    with 16-byte-aligned rows (8-element vector accesses, not probed); mult <= 8 (an 8-entry
    shared array), only 4 certified.

    Arithmetic: one CTA of 256 threads per token, built with fast math (approximate exp,
    reciprocal, rsqrt), a 4-term fp32 sum in copy order, one RNE rounding to bf16. At V4.1's
    eps 1e-6 the `+ eps` sits below bf16 resolution (omitting it keeps 0.998 of elements
    bit-identical), so no output-level check confirms a caller passed it; its handling is
    certified at eps 0.25, where omitting it is rejected.

    Every M in 1..32768 of the V4.1 call: each prefix of a 32768-row call is bitwise equal to its
    first M rows; the call launches exactly one `mhcHcHeadApplyKernel`, compiled ahead of time,
    at every M; captured into a CUDA graph at every decode / DSpark-verify / drafter row count
    up to 1536 it replays bitwise equal to eager.

    Not certified: M > 32768; mult != 4, hidden_size != 5120, K != 20480, norm_eps != 1e-20;
    CPU inputs; torch.compile (no fake implementation is registered).
    """

    def raw_call(
        self,
        mixes: torch.Tensor,
        sqrsum: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        M: int,
        mult: int,
        hidden_size: int,
        K: int,
        norm_eps: float,
        eps: float,
    ) -> torch.Tensor:
        torch.ops.trtllm.mhc_hc_head_apply(
            mixes, sqrsum, x, out, scale, base, M, mult, hidden_size, K, norm_eps, eps
        )
        return out

    def is_valid(
        self,
        mixes: torch.Tensor,
        sqrsum: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        M: int,
        mult: int,
        hidden_size: int,
        K: int,
        norm_eps: float,
        eps: float,
    ) -> None:
        # The op addresses every tensor by raw pointer at the geometry the integer
        # arguments describe and never reads a shape or a stride. Each layout
        # rejected here was measured on sm_103 to return a wrong result without an
        # error (and, for a strided or short `out`, to write outside it).
        assert mixes.dim() == 2 and mixes.is_contiguous(), (
            "mixes must be a contiguous [>=M, mult] tensor"
        )
        assert mixes.shape[1] == mult and mixes.shape[0] >= M, (
            "mixes must be a contiguous [>=M, mult] tensor"
        )
        assert sqrsum.dim() == 1 and sqrsum.is_contiguous() and sqrsum.shape[0] >= M, (
            "sqrsum must be a contiguous [>=M] tensor"
        )
        assert x.dim() == 3 and x.is_contiguous(), (
            "x must be a contiguous [>=M, mult, hidden_size] tensor"
        )
        assert tuple(x.shape[1:]) == (mult, hidden_size) and x.shape[0] >= M, (
            "x must be a contiguous [>=M, mult, hidden_size] tensor"
        )
        assert out.dim() == 2 and out.is_contiguous(), (
            "out must be a contiguous [>=M, hidden_size] tensor"
        )
        assert out.shape[1] == hidden_size and out.shape[0] >= M, (
            "out must be a contiguous [>=M, hidden_size] tensor"
        )

    def reference(
        self,
        mixes: torch.Tensor,
        sqrsum: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        M: int,
        mult: int,
        hidden_size: int,
        K: int,
        norm_eps: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(out, exact, terms)` expected after the call, each the shape of `out`.

        `pre` in fp32 torch (`pre_weights`). `out` is the torch collapse: fp32 weighted sum over
        the copies, rounded once to bf16. `exact` is the same collapse in fp64, unrounded, and
        `terms` is `sum_j |pre_j * x_j|` in fp64: the scale of the kernel's fp32 work, which the
        gate in `compare` needs. Rows `>= M` keep the passed `out`'s contents (`terms` 0 there);
        nothing passed in is mutated. `hidden_size` is implied by `x` and not read.
        """
        pre = pre_weights(mixes[:M], sqrsum[:M], scale, base, mult, K, norm_eps, eps)
        expected = out.clone()
        expected[:M] = torch.sum(pre[:, :, None] * x[:M].float(), dim=1).to(torch.bfloat16)
        terms = pre.double()[:, :, None] * x[:M].double()
        exact = out.to(torch.float64, copy=True)
        exact[:M] = terms.sum(1)
        term_scale = torch.zeros_like(exact)
        term_scale[:M] = terms.abs().sum(1)
        return expected, exact, term_scale

    def compare(
        self,
        out: torch.Tensor,
        ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """A per-element term bound, plus at least 0.999 of elements bit-identical to torch.

        Per element: one bf16 ulp of the exact value (the final rounding, which a few-ulp fp32
        difference can push across a rounding midpoint) plus 16 fp32 ulps of
        S = sum_j |pre_j * x_j|. The kernel forms `pre` with fast-math exp, reciprocal and rsqrt,
        a few fp32 ulps from torch's, and sums the four products in fp32: both err in proportion
        to the terms, not to a result that cancels to near zero. The attention-sink row does
        cancel like that on a few elements, where torch's default bf16 relative tolerance
        misreads a correct result (at M = 15, 20 and 180 one element is 7.3e-4 to 9.8e-4 from
        the torch collapse, relative 0.018 to 0.041). An all-zero row must come out exactly 0.
        Measured on sm_103 the op's largest error is 0.498 of the bound; on 2048 rows the wrong
        variants land at 7.3e5 (another sublayer's scale / base) to 2.1e7 (the consumer's RMS
        statistic) times it.
        """
        expected, exact, terms = ref
        assert out.dtype == expected.dtype == torch.bfloat16, (out.dtype, expected.dtype)
        assert out.shape == expected.shape, (out.shape, expected.shape)
        ratio = collapse_error_ratio(out, exact, terms)
        assert ratio <= 1.0, f"an element exceeds the term bound: max err / bound = {ratio:.3f}"
        if out.numel():
            same = (out == expected) | (out.isnan() & expected.isnan())
            frac = same.float().mean().item()
            assert frac >= _MIN_BITEXACT, f"bit-exact fraction {frac} < {_MIN_BITEXACT}"


mhc_hc_head_apply = MhcHcHeadApply()
