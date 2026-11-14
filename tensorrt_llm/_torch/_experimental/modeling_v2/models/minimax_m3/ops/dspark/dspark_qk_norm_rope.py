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
# =============================================================================
# DSpark / DFlash drafter block: q / k RMSNorm + partial NeoX RoPE in one launch
# =============================================================================
#
# It replaces, bit for bit, the three launches of DFlashForCausalLM.dflash_forward's flashinfer-RoPE branch at head_dim
# 128: flashinfer_rmsnorm on the q heads, flashinfer_rmsnorm on the k heads (both flashinfer's CuTe RMSNormKernel),
# then flashinfer_apply_rope_with_cos_sin_cache_inplace (NeoX, the cache's leading ROT dims rotated, the rest kept).
# The arithmetic is flashinfer's, made explicit (every op inline PTX, so nothing is contracted or reordered):
#   RMSNorm: 16 threads x 8 contiguous elements per row; each thread sums its x^2 sequentially from 0; the 16 sums are
#     added by the butterfly at offsets 1, 2, 4, 8 (adjacent pairs, then pairs of pairs); mean = sum * (1 / 128);
#     rstd = rsqrt.approx.ftz(mean + eps); y = bf16((x * rstd) * w).
#   RoPE (on the bf16 norm output, cos / sin fp32 from the cache): out_i = fma(x_i, cos_i, -x_{i+R/2} * sin_i) and
#     out_{i+R/2} = fma(x_{i+R/2}, cos_i, x_i * sin_i) for i < R / 2, rounded to bf16.
# Bitwise the stock launches.
#
# DRAFT_APPEND_FUSE (``qk_norm_rope_append``): the same launch also writes each row's K (the bits it returns) and V
# (copied from qkv) into the layer's paged HND cache [pages, 2, NKV, PS, HD] where flashinfer.page.append_paged_kv_cache
# would: page kv_indices[kv_indptr[b] + pos // PS], slot pos % PS, with b = batch_indices[row] and pos = positions[row].
# The caller then skips that append launch; the cache bits are the same.
#
# The launch is the CUDA C++ op family_ops().minimax_m31_dspark_qk_norm_rope (kernels/minimaxM31/dsparkGlueSm107.cu,
# No Triton; bit for bit the Triton kernel it replaced; PDL launch, 4 warps per CTA, vector loads when aligned,
# else the same kernel with element loads). Where the op is not taken (not an sm_107 device, the op not built,
# TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1) both entry points return None and the caller keeps the stock flashinfer path.

import torch

# The family-level DSPARK_GLUE_CPP gate (kill switch, sm_107, op registered) and its engagement lines.
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_advance import (
    _glue_cpp_op,
    _glue_log,
)

HEAD_DIM = 128
_OP = "minimax_m31_dspark_qk_norm_rope"


def _plain_rmsnorm(norm) -> bool:
    """The stock q_norm / k_norm takes flashinfer's plain (non-Gemma) RMSNorm on a bf16 input."""
    return (
        getattr(norm, "use_gemma", True) is False
        and not getattr(norm, "use_cuda_tile", False)
        and not getattr(norm, "is_nvfp4", False)
        and not getattr(norm, "return_hp_output", False)
        and getattr(norm, "weight", None) is not None
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.shape == (HEAD_DIM,)
        and norm.weight.is_contiguous()
    )


def qk_norm_rope(qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox):
    """q / k RMSNorm + RoPE of the drafter block in one launch: returns new contiguous (q [T, q_size], k [T, kv_size])
    bitwise equal to the stock flashinfer path, or None (nothing computed) for anything it does not reproduce."""
    if not qk_norm_rope_supported(
        qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox
    ):
        return None
    op = _glue_cpp_op(_OP, qkv.device)
    if op is None:
        return None
    return _launch(op, qkv, q_size, kv_size, q_norm, k_norm, positions, cos_sin_cache)


def qk_norm_rope_supported(
    qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox
) -> bool:
    """Whether the one-launch q / k RMSNorm + RoPE reproduces the stock path for these operands."""
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

    return bool(
        IS_FLASHINFER_AVAILABLE
        and is_neox
        and head_dim == HEAD_DIM
        and qkv.dtype == torch.bfloat16
        and qkv.dim() == 2
        and qkv.stride(1) == 1
        and qkv.shape[1] >= q_size + kv_size
        and q_size % head_dim == 0
        and kv_size % head_dim == 0
        and kv_size >= head_dim
        and positions.dtype == torch.int32
        and positions.dim() == 1
        and positions.shape[0] == qkv.shape[0]
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.stride(1) == 1
        and cos_sin_cache.shape[1] in (16, 32, 64, 128)
        and _plain_rmsnorm(q_norm)
        and _plain_rmsnorm(k_norm)
    )


def _launch(op, qkv, q_size, kv_size, q_norm, k_norm, positions, cos_sin_cache, append=None):
    T = qkv.shape[0]
    q = torch.empty(T, q_size, dtype=torch.bfloat16, device=qkv.device)
    k = torch.empty(T, kv_size, dtype=torch.bfloat16, device=qkv.device)
    if T == 0:
        return q, k
    _glue_log(_OP)
    cache, bidx, apos, kvi, kvp = append if append is not None else (None,) * 5
    op(
        qkv,
        positions.contiguous(),
        cos_sin_cache,
        q_norm.weight,
        k_norm.weight,
        q,
        k,
        float(q_norm.variance_epsilon),
        float(k_norm.variance_epsilon),
        q_size // HEAD_DIM,
        kv_size // HEAD_DIM,
        cache,
        bidx,
        apos,
        kvi,
        kvp,
        True,
    )
    return q, k


_INT = (torch.int32, torch.int64)


def append_supported(
    qkv, kv_size, paged_kv_cache, batch_indices, positions, kv_indices, kv_indptr
) -> bool:
    """Whether ``qk_norm_rope_append`` reproduces flashinfer.page.append_paged_kv_cache for these operands: a
    contiguous bf16 HND layer cache [pages, 2, NKV, PS, HEAD_DIM] with NKV = kv_size / HEAD_DIM, one append row per qkv
    row, integer index tensors on the same device."""
    c = paged_kv_cache
    T = qkv.shape[0]
    return (
        isinstance(c, torch.Tensor)
        and c.dim() == 5
        and c.dtype == torch.bfloat16
        and c.is_contiguous()
        and c.device == qkv.device
        and c.shape[1] == 2
        and c.shape[2] == kv_size // HEAD_DIM
        and c.shape[4] == HEAD_DIM
        and c.shape[3] >= 1
        and all(
            x.dtype in _INT and x.device == qkv.device and x.dim() == 1 and x.is_contiguous()
            for x in (batch_indices, positions, kv_indices, kv_indptr)
        )
        and batch_indices.shape[0] == T
        and positions.shape[0] == T
    )


def qk_norm_rope_append(qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox,
                        paged_kv_cache, append_batch_indices, append_positions, kv_indices, kv_indptr):  # fmt: skip
    """``qk_norm_rope`` that also appends the rows' K / V into ``paged_kv_cache`` exactly as
    flashinfer.page.append_paged_kv_cache(k, v, append_batch_indices, append_positions, paged_kv_cache, kv_indices,
    kv_indptr, ..., kv_layout="HND") would (same bits); None (nothing computed or written) when either part is not
    reproduced. The caller skips its append on a non-None return."""
    if not qk_norm_rope_supported(qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache,
                                  is_neox):  # fmt: skip
        return None
    if not append_supported(qkv, kv_size, paged_kv_cache, append_batch_indices, append_positions, kv_indices,
                            kv_indptr):  # fmt: skip
        return None
    if (
        qkv.shape[1] < q_size + 2 * kv_size
    ):  # the V heads the append copies (columns q_size + kv_size ...)
        return None
    op = _glue_cpp_op(_OP, qkv.device)
    if op is None:
        return None
    return _launch(op, qkv, q_size, kv_size, q_norm, k_norm, positions, cos_sin_cache,
                   (paged_kv_cache, append_batch_indices, append_positions, kv_indices, kv_indptr))  # fmt: skip
