# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 head GEMV: ``x @ weight^T`` for at most 8 bf16 rows and a large weight (an lm_head vocabulary shard) on a
persistent CuTe DSL kernel, over a caller-owned :class:`K3HeadGemvWorkspace`.

``K3HeadGemvWorkspace`` is re-exported here so that a target creates it through the catalog: once per weight shape
and schedule, eagerly, before any CUDA-graph capture. The contract is the ``## State`` section of ``k3_head_gemv.md``.
"""

import torch

# Importing the op module also registers torch.ops.trtllm.k3_head_gemv.
from tensorrt_llm._torch.cute_dsl_kernels.k3_head_gemv.op import K3HeadGemvWorkspace

__all__ = ["K3HeadGemvWorkspace", "k3_head_gemv"]


def k3_head_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    workspace: K3HeadGemvWorkspace,
    keep_tiles: int = 0,
    ring: int = 6,
    prefetch: int = 16,
) -> torch.Tensor:
    """Return ``x @ weight^T`` as bf16 ``[M, N]`` (fp32 accumulation, one bf16 rounding) in one k3_head_gemv call over
    ``workspace``, whose schedule and ``chunk_tiles`` the call takes. The calls that share a workspace must be ordered
    on one stream."""
    return torch.ops.trtllm.k3_head_gemv(
        x,
        weight,
        workspace.partials,
        workspace.flags,
        workspace.claim,
        keep_tiles,
        workspace.chunk_tiles,
        ring,
        workspace.schedule,
        prefetch,
    )
