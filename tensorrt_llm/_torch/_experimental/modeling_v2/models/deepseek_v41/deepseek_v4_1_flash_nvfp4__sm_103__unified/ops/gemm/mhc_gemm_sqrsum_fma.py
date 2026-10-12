# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixes GEMM: true-fp32 FMA product of a bf16 stream with fp32 weights, plus sum(x^2).

The op is `torch.ops.trtllm.mhc_gemm_sqrsum_fma`, registered in
`cpp/tensorrt_llm/thop/mhcOp.cpp`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HC_MULT, _HIDDEN = 4, 5120
#: The flattened [hc_mult, hidden] stream.
_K_HC = _HC_MULT * _HIDDEN
_U = 2.0**-24  # fp32 unit roundoff
#: Median of |y - y64| / ||x * w||_2 allowed over the output.
_FIDELITY = 2.0**-17

#: decode (1, 8, 32), DSpark draft/verify blocks (5, 6 per request: 30, 48, 192),
#: an odd count, prefill up to max_num_tokens (4096, 8192).
_HC_MS = (1, 5, 6, 8, 30, 32, 48, 192, 333, 4096, 8192)

#: The tile_n DeepSeek-V4.1's target passes by row count: (first M, last M, tile_n).
_BANDS = ((1, 128, 1), (129, 2048, 6), (2049, 32768, 24))
#: Each band's lower edge, interior points and upper edge: the tile-24 band's
#: upper edge is 16384 for a step (max_num_tokens) and 32768 for the decoder pass.
_BAND_POINTS = (1, 64, 128, 129, 1000, 2048, 2049, 5000, 8192, 12000, 16384, 24000, 32768)
#: The trunk's ('mhc', T, tile) records in the core target's runtime-value dumps
#: (boot gates on tep4/dep4/tp4, engines at max_batch_size 256 / max_num_tokens
#: 16384 with CUDA-graph, eager, dep4 and tp4 legs, the 128K and 1M long-context
#: engines and the long-context envelope runs). Every dumped tile equals its band's.
_DUMPED_TRUNK_M = (
    1, 2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 24, 29, 32, 34, 48, 60, 96, 108, 127, 128, 129, 134,
    144, 168, 192, 228, 240, 288, 330, 336, 384, 432, 462, 480, 528, 546, 576, 597, 610, 624,
    672, 720, 768, 780, 852, 888, 890, 1018, 1152, 1536, 1906, 1935, 2033, 2038, 2048, 2049,
    2252, 2778, 2784, 3870, 4208, 4245, 4388, 4394, 4559, 4591, 4868, 7835, 7855, 7893, 7908,
    8147, 8191, 8192, 8467, 9193, 9240, 9364, 9370, 9524, 10441, 10444, 10447, 10451, 16298,
    16308, 16316, 16320, 16322, 16327, 16328, 16329, 16333, 16340, 16378, 16384,
)  # fmt: skip
#: The DSpark drafter's ('dspark.mhc', N, tile) records in the same engines (N = 5G).
_DUMPED_DRAFT_M = (
    5, 10, 15, 20, 40, 50, 80, 90, 110, 115, 120, 125, 140, 160, 190, 200, 240, 275, 280, 320,
    360, 370, 385, 400, 440, 455, 480, 520, 560, 580, 585, 600, 640, 650, 665, 710, 715, 720,
    960, 1280,
)  # fmt: skip


def _target_tile(m: int) -> int:
    for lo, hi, tile in _BANDS:
        if lo <= m <= hi:
            return tile
    raise ValueError(m)


def _band_label(m: int) -> str:
    if m in _BAND_POINTS:
        return "a band edge or interior point"
    if m in _DUMPED_TRUNK_M:
        return "a trunk runtime-dump row count"
    return "a drafter runtime-dump row count (5G)"


def _depth(k: int) -> int:
    """Longest rounding chain through the kernel's reduction tree.

    `mhcGemmSqrsumFmaKernel` gives each of 256 threads a sequential FMA chain
    (4 per 1024-wide step, plus a scalar tail of up to one per 256 leftover
    columns), then a 5-level warp butterfly, then an 8-warp sum.
    """
    return 4 * (k // 1024) + -(-(k % 1024) // 256) + 5 + 8


def _gamma(h: int) -> float:
    return h * _U / (1 - h * _U)


class MhcGemmSqrsumFma(OpWrapper):
    """`y = x @ w.T` (true fp32 FMA) and `r = (x.float() ** 2).sum(-1)` in one CUDA-core launch.

    `x` bf16 `[M, K]`, `w` fp32 `[N, K]` (the checkpoint's `hc_*_fn` as stored,
    no transpose), `y` fp32 `[M, N]` and `r` fp32 `[M]`, all dense row-major.
    The bf16 -> fp32 widening of `x` is exact and every term is fused into an
    fp32 accumulator with one rounding (`fmaf`); no tensor cores, no TF32.

    Fusion boundary: the GEMM and the sum of squares are inside the call. The
    caller owns flattening the hyper-connection stream `[T, hc_mult, hidden]` to
    `[T, hc_mult * hidden]` copy-major (a view of the contiguous stream), the RMS
    factor `rsqrt(r / K + norm_eps)` and everything downstream (scale, base,
    sigmoid, Sinkhorn), and keeping `y[:, :hc_mult]` and `r` for a later
    sublayer. Overwrites `y` and `r` completely, writes nothing else, and
    returns `(y, r)`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"heuristic tile M={m} {flavor}: N=24 K=20480 hc mixes",
                spec=dict(M=m, K=_K_HC, flavor=flavor, tile_n=0, tile_m=0),
            )
            for m in _HC_MS
            for flavor in ("gauss", "hc_like")
        )
        + tuple(
            Cell(
                why=f"K={k} M={m}: {what}",
                spec=dict(M=m, K=k, flavor="gauss", tile_n=0, tile_m=0),
            )
            for k, what in (
                (999, "below 1024, the scalar tail loop only"),
                (1024, "exactly one main-loop step, no tail"),
                (5124, "main loop plus a 4-column tail"),
            )
            for m in (1, 7)
        )
        + tuple(
            Cell(
                why=f"M={m} tile_n={_target_tile(m)}: {_band_label(m)}",
                spec=dict(M=m, K=_K_HC, flavor="hc_like", tile_n=_target_tile(m), tile_m=0),
            )
            for m in sorted(set(_BAND_POINTS) | set(_DUMPED_TRUNK_M) | set(_DUMPED_DRAFT_M))
        )
    )

    note = """
    Metadata consumed: none. Stateless.

    DeepSeek-V4.1 calls this twice per layer at N = 24 (hc_mult 4: 4 pre, 4
    post, 16 comb mixes), K = 20480, with M the rows of the pass: up to
    `max_num_tokens` (16384), or up to 32768 in the decoder bounded-replay pass
    of layers 21-39; the DSpark drafter calls it at M = 5G. `y * rsqrt(r / K +
    1e-20)` is the reference model's `F.linear(x.flatten(2).float(), hc_fn) *
    rsqrt(x.square().mean(-1) + 1e-20)`; `pre` derived from this op's outputs
    matched the golden layer dumps to at most 1.192e-06 absolute on 8 sublayers.

    Preconditions: the op checks dtypes (RuntimeError) and rejects all-CPU
    arguments (NotImplementedError); a mixed-device call was not probed and,
    since GB300 can address host memory, should not be expected to fail. It
    checks neither layout nor sizes (see `is_valid`). Alignment: `x` and `w`
    data pointers 8- and 16-byte aligned and `K % 4 == 0` whenever K >= 1024,
    else a sticky CUDA misaligned-address error poisons the process; freshly
    allocated tensors and row slices of a contiguous `[T, K]` buffer with
    `K % 4 == 0` are aligned. `tile_n > 0` must divide N (the op raises; for
    N = 24 that is 1, 2, 3, 4, 6, 8, 12, 24); `tile_m` is not read. M = 0
    returns without launching.

    Deterministic, and bit-identical across every `tile_n` and `tile_m`. The
    longest rounding chain is 93 at K = 20480. Built with fast math: subnormal
    `x` elements are flushed to zero. Grid `(M, N / tile_n)` of 256-thread
    blocks, block `(m, j)` reads row `m` and writes row `m`; `tile_n` selects one
    of eight kernels compiled ahead of time. Every prefix `x[:M]`, M = 1..32768,
    at its band's `tile_n` returns the first M rows of the 32768-row result and
    writes nothing past row M - 1.

    Tactic: `tile_n` changes speed only. The built-in heuristic (`tile_n <= 0`)
    picks 1 for M <= 32 and 8 above, slow from M = 40 up; measured on sm_103 the
    fast choice is 1 for M <= 128, 6 for 129..2048 and 24 from 4096, which is
    what the target passes (24 from 2049). DeepGEMM's `tf32_hc_prenorm_gemm`
    computes the same pair (5.5x faster at 4096 rows, 6.9x slower at 1) but
    rounds the weight to TF32 (median error about 1000x this op's), so it does
    not qualify; the
    autotuner in `modules/mhc/mhc_cuda.py` can pick it, and calling this op
    directly pins the fp32 FMA path.

    Capturable in a CUDA graph (no host sync, no allocation); certified at the
    114 per-rank capture row counts (G, 5G, 6G over the captured batch sizes
    1..32, 40..128 step 8, 192, 256; at most 1536, so tile 1 or 6), each
    captured after an eager call at that M.

    Not certified: N other than 24; K other than 999, 1024, 5124 and 20480;
    M above 32768; the built-in heuristic at M values outside its own cells.
    """

    def raw_call(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        r: torch.Tensor,
        M: int,
        N: int,
        K: int,
        tile_n: int = 0,
        tile_m: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.mhc_gemm_sqrsum_fma(x, w, y, r, M, N, K, tile_n, tile_m)
        return y, r

    def reference(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        r: torch.Tensor,
        M: int,
        N: int,
        K: int,
        tile_n: int = 0,
        tile_m: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The float64 product and the per-element scales `compare` gates against.

        Returns `(y64, r64, bound, l2)`, all float64: `y64 = x @ w.T`,
        `r64 = sum x^2`, `bound = gamma_h * sum_k |x_k w_k|` with `h` the
        kernel's longest rounding chain at this K, and `l2 = ||x * w||_2` per
        element. A torch fp32 product is not an adequate stand-in even with TF32
        off: at M = 4096 and 8192 its largest error measured 10x to 40x this
        op's. `y`, `r`, `M`, `N`, `tile_n` and `tile_m` are ignored; nothing is
        mutated.
        """
        xd, wd = x.double(), w.double()
        y64 = xd @ wd.T
        r64 = (xd * xd).sum(-1)
        bound = (xd.abs() @ wd.abs().T) * _gamma(_depth(K))
        l2 = ((xd * xd) @ (wd * wd).T).sqrt()
        return y64, r64, bound, l2

    def is_valid(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        r: torch.Tensor,
        M: int,
        N: int,
        K: int,
        tile_n: int = 0,
        tile_m: int = 0,
    ) -> None:
        # The kernel indexes every operand as dense row-major from M, N, K alone and
        # never reads a stride or a shape: a strided operand, or an M/N/K that
        # disagrees with the tensors, is accepted and gives silently wrong results
        # (or writes past the views it was given).
        assert x.shape == (M, K) and x.is_contiguous(), "x must be a contiguous bf16 [M, K]"
        assert w.shape == (N, K) and w.is_contiguous(), "w must be a contiguous fp32 [N, K]"
        assert y.shape == (M, N) and y.is_contiguous(), "y must be a contiguous fp32 [M, N]"
        assert r.shape == (M,) and r.is_contiguous(), "r must be a contiguous fp32 [M]"

    def compare(
        self,
        out: tuple[torch.Tensor, torch.Tensor],
        ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """`y` against a rigorous bound and a fidelity gate; `r` at torch's fp32 defaults.

        * Every element satisfies `|y - y64| <= gamma_h * sum_k |x_k w_k|`, the
          forward-error bound of an fp32 dot product with longest rounding chain
          `h`. It cannot flake on a correct kernel and catches gross errors (a
          wrong row, column, layout or operand).
        * The median over elements of `|y - y64| / ||x * w||_2` is at most 2^-17.
          fp32 accumulation noise sits near 2^-23; measured on sm_103 the op lands
          at 8e-08 to 1.5e-07 on synthetic operands and 1.5e-07 to 2.3e-07 on the
          real streams (0.0155x to 0.0193x of the gate), while TF32-rounded
          weights land 17.2x to 19.1x above it and bf16-rounded weights 140x to
          158x.
        * `r` is a sum of non-negative terms: `assert_close` at fp32 defaults
          (largest relative error measured 3.3e-07).
        """
        (y, r), (y64, r64, bound, l2) = out, ref
        assert y.dtype == r.dtype == torch.float32, (y.dtype, r.dtype)
        assert y.shape == y64.shape and r.shape == r64.shape, (y.shape, r.shape)
        err = (y.double() - y64).abs()
        bound_ratio = (err / bound).max().item()
        assert bound_ratio <= 1.0, (
            f"an element exceeds the fp32 rounding-chain bound: {bound_ratio:.3f}x"
        )
        fidelity = (err / l2).median().item()
        assert fidelity <= _FIDELITY, f"median err/||x*w||_2 = {fidelity:.3e} > {_FIDELITY:.3e}"
        torch.testing.assert_close(r, r64.float())


mhc_gemm_sqrsum_fma = MhcGemmSqrsumFma()
