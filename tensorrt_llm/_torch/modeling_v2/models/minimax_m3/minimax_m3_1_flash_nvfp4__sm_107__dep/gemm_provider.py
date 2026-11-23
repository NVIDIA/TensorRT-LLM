# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_GEMM_PROVIDER: the four small-M MXFP8 GEMMs of a decode step (qkv, o_proj, shared gate_up, shared down) at
65..128 rows leave FlashInfer's pinned CuTe DSL tactic for cuBLASLt at one pinned algorithm (DEP_GEMM_CUBLASLT);
without it o_proj and the shared gate_up run TRT-LLM's SM107 persistent dense kernel (CuteDSLMXFP8RubinLinear) at
split-K 2.

DEP_GEMM_CUBLASLT (algo 66 tile 20 stages 36 split 1 cluster 3, no workspace): deterministic, rel-L2 <= 2.2e-5
against the pinned tactic (<= 4 bf16 ulp), not bitwise; its numerics gate is the natural acceptance length, like
split-K 2. ``TRTLLM_M31_DISABLE_DEP_GEMM_CUBLASLT=1`` returns the four to split-K 2 / the pinned tactic (o_proj and the
shared gate_up on split-K 2, qkv / shared down on the pinned tactic). Split-K 2 also zero-fills its output before every
call (the kernel reduces with TMA reduce-add stores); the cuBLASLt algorithm does not.

Split-K 2 (o_proj and the shared gate_up without DEP_GEMM_CUBLASLT):

At 65..128 decode rows per rank the two shapes, o_proj [6144, 8192] and the shared gate_up [6144, 6144] ([N, K]), stream
their weight through ceil(6144 / 64) x 2 CTAs instead of one CTA per output tile. Deterministic (the two K halves reduce
in a fixed order), but not bitwise the full-K tactic: rel-L2 2.9e-3 against it, about one bf16 rounding. The numerics
gate is the natural acceptance length.

The hand-off GEMMs (merge_mxfp8.DepMergeMxfp8LinearMethod and its subclasses) ask ``run`` first; None keeps the
pinned tactic (DEP_PROJ_CUTE) or the stock apply. Other shapes and row counts are untouched. The GEMMs are the
family's m31_mxfp8_cublaslt.run and m31_rubin_mxfp8.run (the latter compiles on its first eager call, the decode
warmup, before capture); a shape whose first call fails is declined for the process, with a warning.
``TRTLLM_M31_DISABLE_DEP_GEMM_PROVIDER=1`` restores the pinned tactic. Read at import.
"""

from __future__ import annotations

from typing import Optional

import torch

from .. import m31_mxfp8_cublaslt, m31_rubin_mxfp8
from .m31_switches import m31_disabled

DEP_GEMM_PROVIDER = not m31_disabled("DEP_GEMM_PROVIDER")
DEP_GEMM_CUBLASLT = not m31_disabled("DEP_GEMM_CUBLASLT")

_TAG = "[M3.1 DEP] DEP_GEMM_PROVIDER"
#: The decode rows the route takes; outside them the pinned tactic runs.
MIN_ROWS, MAX_ROWS = 65, 128
RUBIN, CUBLASLT = "rubin", "cublaslt"
#: CuteDSLMXFP8RubinLinear's base tactic: mma 128 x 64, cluster (1, 2), swap_ab, no prefetch, static scheduler,
#: raster along m, split-K 2.
RUBIN_SPLIT_K2 = ("base", (128, 64, 128), (128, 64, 64), (1, 2), True, False, "static", "m", 2)
#: cuBLASLt's nine CUBLASLT_ALGO_CONFIG attributes (id, tile, stages, split-K, reduction, CTA swizzle, custom option,
#: inner shape, cluster shape): one deterministic configuration for qkv / shared gate_up / shared down.
CUBLASLT_ALGO66 = (66, 20, 36, 1, 0, 0, 3, 0, 3)
#: (N, K) -> (provider, tactic or algorithm).
ROUTES = {
    (9856, 6144): (CUBLASLT, CUBLASLT_ALGO66),  # qkv_proj
    (6144, 8192): (CUBLASLT, CUBLASLT_ALGO66),  # o_proj
    (6144, 6144): (CUBLASLT, CUBLASLT_ALGO66),  # shared expert gate_up
    (6144, 3072): (CUBLASLT, CUBLASLT_ALGO66),  # shared expert down
}
#: Phase 1's route of a shape whose cuBLASLt route is off or declined (None keeps the pinned tactic).
PHASE1 = {(6144, 8192): (RUBIN, RUBIN_SPLIT_K2), (6144, 6144): (RUBIN, RUBIN_SPLIT_K2)}


def route(m: int, n: int, k: int):
    """(provider, tactic) for an [m, k] x [n, k]^T GEMM, or None to keep the pinned tactic."""
    if not DEP_GEMM_PROVIDER or not MIN_ROWS <= m <= MAX_ROWS:
        return None
    chosen = ROUTES.get((n, k))
    if chosen is not None and chosen[0] == CUBLASLT:
        if not DEP_GEMM_CUBLASLT or m31_mxfp8_cublaslt.declined(n, k):
            chosen = PHASE1.get((n, k))
    if chosen is not None and chosen[0] == RUBIN and m31_rubin_mxfp8.declined(n, k):
        return None
    return chosen


def run(
    act_e4m3: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The GEMM's bf16 output on the routed provider, or None (not routed, not bf16, or the runner unavailable)."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    chosen = route(int(act_e4m3.shape[0]), n, k)
    if chosen is None or out_dtype != torch.bfloat16:
        return None
    if chosen[0] == CUBLASLT:
        output = m31_mxfp8_cublaslt.run(act_e4m3, act_sf, weight, weight_scale, chosen[1], _TAG)
        if output is not None:
            return output
        chosen = PHASE1.get((n, k))
        if chosen is None or m31_rubin_mxfp8.declined(n, k):
            return None
    return m31_rubin_mxfp8.run(act_e4m3, act_sf, weight, weight_scale, chosen[1], _TAG)
