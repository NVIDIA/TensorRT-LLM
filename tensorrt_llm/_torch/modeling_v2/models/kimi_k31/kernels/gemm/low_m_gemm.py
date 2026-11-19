# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M BF16 GEMMs for Kimi K3.1 decode / verify projections: FlashInfer's CuTe DSL kernels by a measured plan.

``out = a @ weight.T`` for a row-major ``a [M, K]`` and a row-major ``weight [N, K]`` (an ``nn.Linear`` weight),
BF16 in and out, FP32 accumulation, one launch. Which kernel runs is a *plan* entry measured offline per
``(N, K)`` and M range (cold weights, CUDA-graph replay):

* ``("direct", block_size, outputs_per_block)`` -- ``dense_bf16_gemm_direct``: SIMT, each CTA prefetches its
  weight rows' whole K into registers; the row tile is the runtime M (M <= 32);
* ``("splitk", mma_m, mma_n, split_k, ab_stages)`` -- ``dense_bf16_gemm_sm100_splitk``: tcgen05 swap-AB with
  the split-K partials reduced over a cluster through distributed shared memory (no second launch; M <= 32);
* ``("splitkc", mma_m, mma_n, split_k, ab_stages)`` -- ``kernels/gemm/splitk_chunked``: the same cluster split-K
  over front-heavy K chunks (the first ``K tiles % split_k`` ranks take one K tile more; even chunks where ``split_k``
  divides the K tiles), which is cuBLAS's split-K partition, so where cuBLAS splits K that way it gives cuBLAS's bits;
  FlashInfer's token tiles at M <= 32, or a wide token tile (``mma_n`` up to 128) at any M, one CTA per K chunk of
  each (weight tile, token tile);
* ``("tgv", cta_m, cta_n, ab_stages, use_2cta)`` -- ``kernels/gemm/tgv_gemm``: FlashInfer's warp-specialized
  TMA / tcgen05 swap-AB TGV GEMM over several token tiles (any M), with the 2-CTA pair synchronized around its TMEM
  dealloc; any config the kernel class builds (``cta_m`` 64 / 128; ``cta_n`` a multiple of 8, of 16 with 2-CTA, up
  to 256), in FlashInfer's tactic table or not. Every TGV config accumulates the whole K in order, so all of them
  give the same bits;
* ``("tgvchunk", cta_n, ab_stages, b1, b2, ...)`` -- ``kernels/gemm/chunked_tgv_gemm``: the 2-CTA TGV GEMM
  (``cta_m`` 64) with the K tiles split into chunks at the given 128-wide K-tile bounds, each chunk accumulated
  in its own tensor-memory accumulator and the partials added in order before one BF16 rounding. That is the
  arithmetic of a cuBLAS split-K GEMM whose reduce adds its partials in split order, so a choice with cuBLAS's
  chunk map gives cuBLAS's bits at any M (``(1 + chunks) * cta_n <= 256``; ``cta_n`` covers the rows in one tile);
* ``("tgvchunkm", cta_m, cta_n, ab_stages, b1, b2, ...)`` -- the same chunked TGV GEMM on ``cta_m``-row weight tiles
  (64 or 128; N a multiple of ``2 * cta_m``): the same bits as ``tgvchunk`` with the same bounds;
* ``("default",)`` -- the M range stays on the caller's GEMM.

A plan maps ``(N, K)`` to ``((M upper bound, choice), ...)`` in ascending M; an M above the last bound, or an
``(N, K)`` the plan does not list, has no choice. The kernels are FlashInfer's (a dependency of the engine);
every one is JIT-compiled in-process on its first call per tactic (direct: per M, N, K too) and cached for the
process, so the first call of each shape has to run before CUDA-graph capture, which the engine's warmup does.

``pdl`` launches with programmatic dependent launch: each kernel waits on the previous grid
(``griddepcontrol.wait``) before it reads ``a``, so it is safe after any kernel and lets the weight prologue
overlap the producer's tail.

The launcher is :func:`k31_low_m_gemm`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

#: (N, K) -> ((M upper bound, choice), ...), ascending M. See the module docstring for the choice tuples.
LowMPlan = Dict[Tuple[int, int], Tuple[Tuple[int, tuple], ...]]

(
    _KIND_DIRECT,
    _KIND_SPLITK,
    _KIND_TGV,
    _KIND_SPLITK_CHUNKED,
    _KIND_TGV_CHUNKED,
    _KIND_TGV_CHUNKED_M,
) = range(6)
_KINDS = {
    "direct": _KIND_DIRECT,
    "splitk": _KIND_SPLITK,
    "tgv": _KIND_TGV,
    "splitkc": _KIND_SPLITK_CHUNKED,
    "tgvchunk": _KIND_TGV_CHUNKED,
    "tgvchunkm": _KIND_TGV_CHUNKED_M,
}


def plan_choice(plan: Optional[LowMPlan], m: int, n: int, k: int) -> Optional[tuple]:
    """The plan's kernel choice for an ``[M, K] x [K, N]`` product, or None for the caller's GEMM."""
    if not plan or m < 1:
        return None
    rows = plan.get((n, k))
    if not rows:
        return None
    choice = next((c for bound, c in rows if m <= bound), None)
    if choice is None or choice[0] == "default":
        return None
    return choice


def encode_choice(choice: tuple) -> Tuple[int, List[int]]:
    """``(kind, params)`` of a plan choice, the integer form :func:`k31_low_m_gemm` takes."""
    kind = _KINDS[choice[0]]
    return kind, [int(x) for x in choice[1:]]


def launch(
    kind: int,
    params: List[int],
    a: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    pdl: bool,
) -> torch.Tensor:
    """Run one plan choice into ``out``: ``a [M, K] @ weight[N, K].T``."""
    m = a.shape[0]
    w_t = weight.t()  # [K, N] column-major view, the kernels' B operand
    if kind == _KIND_DIRECT:
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense

        return run_direct_dense(a, w_t, out, pdl, DirectTactic(params[0], params[1], m))
    if kind == _KIND_SPLITK:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        return run_splitk_dense(a, w_t, None, out, pdl, SplitKTactic(*params))
    if kind == _KIND_SPLITK_CHUNKED:
        from .splitk_chunked import run_chunked_splitk

        return run_chunked_splitk(a, w_t, out, pdl, tuple(params))
    if kind == _KIND_TGV:
        from .tgv_gemm import run_tgv

        return run_tgv(a, w_t, out, pdl, (params[0], params[1], params[2], bool(params[3])))
    if kind == _KIND_TGV_CHUNKED:
        from .chunked_tgv_gemm import run_chunked_tgv

        run_chunked_tgv(a, [(weight, out, tuple(params[2:]))], params[0], params[1], pdl)
        return out
    if kind == _KIND_TGV_CHUNKED_M:
        from .chunked_tgv_gemm import run_chunked_tgv

        run_chunked_tgv(a, [(weight, out, tuple(params[3:]))], params[1], params[2], pdl, params[0])
        return out
    raise ValueError(f"unknown low-M kernel kind {kind}")


def k31_low_m_gemm(
    a: torch.Tensor, weight: torch.Tensor, kind: int, params: List[int], pdl: bool
) -> torch.Tensor:
    """``a @ weight.T`` (BF16, FP32 accumulation) with the low-M kernel ``(kind, params)`` (``encode_choice``).

    ``a`` is row-major ``[M, K]``, ``weight`` row-major ``[N, K]``, both BF16 and 32-byte aligned.
    """
    out = torch.empty((a.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=a.device)
    # DLPack export refuses tensors that require grad: module parameters, and activations outside inference mode.
    return launch(kind, params, a.detach(), weight.detach(), out, pdl)
