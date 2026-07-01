# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Torch entry point for the CuTe DSL block-sparse attention forward.

Blackwell (sm_100) fast path for VSA's fine stage. The kernel JIT-compiles
on first call and is cached per process; the caller
(CuTeDSLAttention._forward_vsa) falls back to dense SDPA when the
device/dtype/head_dim envelope is not met.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import cuda.bindings.driver as _cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from .block_sparse_attn_dsl_fwd import (
        VideoSparseAttentionForwardGroup2QInterleaveKV as VideoSparseAttentionForward,
    )

    CUTE_AVAILABLE = True
except ImportError:  # cuda-bindings / cutlass-dsl not installed
    _cuda = None
    cute = None
    from_dlpack = None
    VideoSparseAttentionForward = None
    CUTE_AVAILABLE = False


__all__ = [
    "CUTE_AVAILABLE",
    "is_cute_supported",
    "block_sparse_attn_from_indices_cute",
]


# JIT compile is multi-second; reuse aggressively.
_COMPILE_CACHE: dict = {}


def is_cute_supported(q: torch.Tensor) -> bool:
    """Capability check for the CuTe path."""
    # Kernel asserts head_dim==128, block_m==block_n==64, fp16/bf16, sm_100+.
    if not CUTE_AVAILABLE:
        return False
    if not q.is_cuda:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.shape[-1] != 128:
        return False
    cap = torch.cuda.get_device_capability(q.device)
    return cap[0] >= 10


def _to_cute_tensor(t: torch.Tensor):
    """Convert a 4-D BHSD tensor into a CuTe tensor with dynamic B/H/S strides."""
    # Head-dim mode left static (=128) since the kernel specializes on it.
    return (
        from_dlpack(t.detach(), assumed_align=128)
        .mark_compact_shape_dynamic(mode=0, stride_order=t.dim_order())
        .mark_compact_shape_dynamic(mode=1, stride_order=t.dim_order())
        .mark_compact_shape_dynamic(mode=2, stride_order=t.dim_order())
    )


@torch.compiler.disable
def block_sparse_attn_from_indices_cute(
    q: torch.Tensor,  # [B, H, S, D]  fp16/bf16, sm_100+
    k: torch.Tensor,  # [B, H, S, D]
    v: torch.Tensor,  # [B, H, S, D]
    q2k_idx: torch.Tensor,  # [B, H, num_q_blk, K]  int32
    q2k_num: torch.Tensor,  # [B, H, num_q_blk]     int32
    variable_block_sizes: torch.Tensor,  # [num_q_blk]  int32
    sm_scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run VSA fine-stage attention on Blackwell using the CuTe DSL kernel.

    Returns:
        out: [B, H, S, D] same dtype as Q.
        lse: [B, H, S] fp32.
    """
    # Disabled for torch.compile: cuda.bindings.driver.CUstream + cute.compile
    # are not Dynamo-traceable, and torch.cuda.current_stream() returns a
    # proxy without .cuda_stream inside compiled regions.
    if not CUTE_AVAILABLE:
        raise RuntimeError(
            "block_sparse_attn_from_indices_cute called but cuda.bindings or "
            "cutlass-dsl is not importable."
        )

    num_q_blk = variable_block_sizes.shape[0]
    if num_q_blk > VideoSparseAttentionForward.MAX_INDICES:
        raise ValueError(
            f"variable_block_sizes has {num_q_blk} entries but the CuTe kernel "
            f"supports at most {VideoSparseAttentionForward.MAX_INDICES} "
            "(SMEM-allocated sVariable_block_sizes buffer). Lower video "
            "resolution/length or fall back to dense SDPA."
        )

    B, H, T, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)
    lse = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

    cuda_stream = _cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    q_packed = _to_cute_tensor(q)
    k_packed = _to_cute_tensor(k)
    v_packed = _to_cute_tensor(v)
    o_packed = _to_cute_tensor(out)

    lse_packed = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=2)
    idx_packed = from_dlpack(q2k_idx.detach()).mark_layout_dynamic(leading_dim=3)
    num_packed = from_dlpack(q2k_num.detach()).mark_layout_dynamic(leading_dim=2)
    var_packed = from_dlpack(variable_block_sizes.detach()).mark_layout_dynamic(leading_dim=0)

    # Full q2k_idx shape is part of the key: mark_layout_dynamic(leading_dim=3)
    # only makes the innermost stride dynamic; B-loop bound and inner strides
    # are baked in at compile time, so each shape needs its own compiled kernel.
    compile_key = (D, q.dtype, float(sm_scale)) + tuple(q2k_idx.shape)
    compiled = _COMPILE_CACHE.get(compile_key)
    if compiled is None:
        fwd_kernel = VideoSparseAttentionForward(block_m=64, block_n=64, headdim=D)
        compiled = cute.compile(
            fwd_kernel,
            q_packed,
            k_packed,
            v_packed,
            sm_scale,
            o_packed,
            lse_packed,
            idx_packed,
            num_packed,
            var_packed,
            cuda_stream,
        )
        _COMPILE_CACHE[compile_key] = compiled

    compiled(
        q_packed,
        k_packed,
        v_packed,
        sm_scale,
        o_packed,
        lse_packed,
        idx_packed,
        num_packed,
        var_packed,
        cuda_stream,
    )
    return out, lse
