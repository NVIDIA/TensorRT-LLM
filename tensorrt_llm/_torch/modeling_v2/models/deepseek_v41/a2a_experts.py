# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP routed experts over the NVLink one-sided all-to-all, as the built-in model's MoE runs them with the
TRTLLM backend (``ConfigurableMoE`` + ``NVLinkOneSided`` + the TRTLLM-Gen NVFP4 runner): each rank routes and
quantizes its own rows in one launch (``dsv4_route_nvfp4_quant``), dispatches them to the ranks that hold their
experts, runs its experts over the rows it received and combines the partial sums back to the rows' owners.

Shared by the attention-DP generation targets for every MoE backend but MegaMoE; the experts themselves are the
caller's (``run_experts``), so the module holds no weights. Without it those targets all-gathered every rank's rows,
ran their experts over all of them and reduce-scattered the result (about twice the bytes and the expert rows).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.moe_a2a import (
    INVALID_EXPERT,
    MoeA2AWorkspace,
    moe_a2a_combine,
    moe_a2a_dispatch,
    moe_a2a_workspace,
)

_SF_VEC = 16


def a2a_applies(mapping) -> bool:
    """Attention DP with the experts split only across ranks (each rank a window of whole experts)."""
    return bool(
        mapping.enable_attention_dp
        and mapping.moe_tp_size == 1
        and mapping.moe_ep_size == mapping.world_size > 1
    )


def build_a2a_workspace(
    mapping, max_rows: int, hidden: int, top_k: int, num_experts: int
) -> Optional[MoeA2AWorkspace]:
    """The all-to-all workspace every layer shares (they run one after another), for dispatches of NVFP4 rows, their
    scales, int32 ids and bf16 routing weights, and combines of bf16 rows; None where `a2a_applies` does not or the
    platform has no MNNVL.
    **Collective**: every rank builds it once, after the meta-device construction, in the same order."""
    if not a2a_applies(mapping):
        return None
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    if not MnnvlMemory.supports_mnnvl():
        # As the engine's strategy selection (`NVLinkOneSided.is_platform_supported`): without MNNVL the experts run
        # over the all-gather / reduce-scatter.
        return None
    return moe_a2a_workspace(
        mapping,
        max_rows,
        top_k,
        [hidden // 2, hidden // _SF_VEC, 4 * top_k, 2 * top_k],
        2 * hidden,
        hidden,
        num_experts,
    )


def a2a_experts(
    a2a: MoeA2AWorkspace,
    x: torch.Tensor,
    counts: Sequence[int],
    num_experts: int,
    top_k: int,
    route_quant: Callable[[torch.Tensor], tuple],
    run_experts: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """The routed experts' output of `x`'s rows `[n, hidden]` bf16 over the all-to-all. `counts`: every rank's row
    count of the step (the same on every rank). `route_quant(x)` -> `(ids [n, k] int32 global, weights [n, k] bf16,
    x_fp4 [n, hidden / 2] uint8, x_sf [n, hidden / 16] uint8)`; `run_experts(ids, weights, x_fp4, x_sf)` -> this rank's
    finalized partial sum `[rows, hidden]` bf16 of the rows it received (ids of other ranks' experts and of dead rows,
    -1, contribute nothing). A rank without rows dispatches one row routed nowhere (the kernels take no empty side),
    as the built-in model does."""
    n, h, dev = x.shape[0], x.shape[1], x.device
    rows = max(max(1, int(c)) for c in counts)
    if n:
        ids, w, xq, xsf = route_quant(x)
    else:
        ids = torch.full((1, top_k), INVALID_EXPERT, dtype=torch.int32, device=dev)
        w = torch.zeros(1, top_k, dtype=torch.bfloat16, device=dev)
        xq = torch.zeros(1, h // 2, dtype=torch.uint8, device=dev)
        xsf = torch.zeros(1, h // _SF_VEC, dtype=torch.uint8, device=dev)
    payloads = [
        t.contiguous()
        for t in (xq.view(torch.uint8), xsf.reshape(-1, h // _SF_VEC).view(torch.uint8), ids, w)
    ]
    (rq, rsf, rids, rw), offset = moe_a2a_dispatch(
        payloads[2],
        payloads,
        2,
        a2a.workspace,
        a2a.metainfo,
        rows,
        a2a.ep_rank,
        a2a.ep_size,
        num_experts,
        a2a.sentinel_max_rows,
    )
    partial = run_experts(rids, rw, rq, rsf)
    out = moe_a2a_combine(
        partial,
        max(n, 1),
        a2a.workspace,
        a2a.metainfo,
        rows,
        a2a.ep_rank,
        a2a.ep_size,
        top_k,
        offset,
        a2a.sentinel_max_rows,
        False,
    )
    return out if n else out[:0]
