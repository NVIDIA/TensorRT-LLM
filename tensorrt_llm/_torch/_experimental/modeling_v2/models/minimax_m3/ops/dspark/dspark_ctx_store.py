# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
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
# DSpark / DFlash drafter context K / V store after the fused K / V GEMM: one launch for every drafter layer
# =============================================================================
#
# Adapted from SGLang's DFlash KV materialization (_fused_norm_rope_kernel_stacked in
# python/sglang/kernels/ops/speculative/fused_kv_materialize.py, https://github.com/sgl-project/sglang, commit
# 6ed9843b57aaecbf0566f24a4eaf78c7c92ef25d, Apache-2.0): one program per (row, layer, kv head) reads the stacked
# [K | V] GEMM output, applies the layer's k_norm and NeoX RoPE to K. Changed here: the 0/1 accept mask on K and V, the
# paged HND store through the request's block table (instead of a dense output), and the stock path's rounding points.
#
# It replaces, bit for bit, what DFlashWorker does after DFlashForCausalLM.precompute_context_kv's GEMM. The k_norm
# stays torch's own F.rms_norm (as precompute_context_kv runs it: an in-kernel row sum did not reproduce torch's
# reduction order on every input); the bf16 weight multiply runs in the kernel (``fold_k_norm_weight``, TAILBK_SPEC F:
# fp32 product rounded to bf16, as torch's bf16 multiply) or as torch's own; this one launch does the rest:
#   k = rms_norm(k) (fp32, rounded to bf16) * k_norm weight (bf16)  ->  flashinfer cos/sin-cache NeoX RoPE over the
#   first ROT dims, ROT = the cache's width (the rotary dim; a partial rotary leaves dims ROT..HD as they are) (fp32
#   from the bf16 k, x*cos + (-/+)p*sin as fma(x, cos, (-/+)p*sin), rounded to bf16)  ->  k, v *= accept mask (bf16)  ->
#   flashinfer append_paged_kv_cache per layer (HND) at (block_table[row, col // page], col % page).
# Row r belongs to request r // (K+1) at block offset r % (K+1); it is accepted when that offset < num_accepted.
# Masked rows still write (zeros) at their clamped column, as the stock fixed-size write does.

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

# DSPARK_CTX_STORE_CPP (no Triton in served paths): the store is the CUDA C++ op
# family_ops().minimax_m31_dspark_ctx_store (the family's csrc/m31/dsparkCtxStore.cu), bit for bit the Triton kernel
# it replaces. Without the op, or with TRTLLM_M31_DISABLE_DSPARK_CTX_STORE_CPP=1 (the kill switch), ``fused_ctx_store``
# returns False and the stock store runs.
_CPP_OFF = m31_disabled("DSPARK_CTX_STORE_CPP")
_OP = "minimax_m31_dspark_ctx_store"
_logged: set = set()


def _cpp_op():
    """The C++ op, or None (killed or not registered), with the engagement line logged once per outcome."""
    op = None if _CPP_OFF else getattr(family_ops(), _OP, None)
    why = (
        None
        if op is not None
        else (
            "killed (TRTLLM_M31_DISABLE_DSPARK_CTX_STORE_CPP=1)"
            if _CPP_OFF
            else "the op is not registered"
        )
    )
    if why not in _logged:
        _logged.add(why)
        from tensorrt_llm.logger import logger

        if why is None:
            logger.info(f"[M3 dispatch] DSPARK_CTX_STORE_CPP taken: {_OP}")
        else:
            logger.info(f"[M3 dispatch] DSPARK_CTX_STORE_CPP not taken: {_OP}: {why}; stock store")
    return op


_INT = (torch.int32, torch.int64)


_fold_w = [
    False
]  # TAILBK_SPEC F: the k_norm weight multiply in the kernel (see fold_k_norm_weight)


def fold_k_norm_weight(enabled: bool = True) -> bool:
    """Run the stacked k_norm weight multiply inside the store kernel (one launch fewer per call) when the weight is a
    contiguous bf16 [L, HD] and rms_norm's output is bf16: the fp32 product rounded to bf16, bitwise torch's bf16
    multiply. Off by default; process-wide."""
    _fold_w[0] = bool(enabled)
    return _fold_w[0]


# Per-layer cache lists (the bound draft pool) -> their int64 address array, built once outside capture.
_PTR_ARRAYS: Dict[Tuple[int, ...], torch.Tensor] = {}


def _cache_arg(
    kv_buf: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> Optional[Tuple[torch.Tensor, bool, int]]:
    if isinstance(kv_buf, torch.Tensor):
        return kv_buf, False, kv_buf.stride(0)
    key = tuple(t.data_ptr() for t in kv_buf)
    arr = _PTR_ARRAYS.get(key)
    if arr is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # never seen eagerly: the caller keeps the stock store for this capture
        arr = torch.tensor(key, dtype=torch.int64, device=kv_buf[0].device)
        _PTR_ARRAYS[key] = arr
    return arr, True, 0


def _pow2(n: int) -> bool:
    return n >= 1 and n & (n - 1) == 0


def _rotary_dim(draft_model, hd: int) -> int:
    """The rotary dim the drafter's RoPE uses (the width of its fp32 cos/sin cache: flashinfer rotates that many leading
    dims of each head), or 0 when the kernel cannot reproduce it."""
    cos_sin = draft_model._get_cos_sin_cache()
    if cos_sin.dim() != 2 or cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous():
        return 0
    rot = cos_sin.shape[1]
    if rot % 2 or rot > hd or not _pow2(rot // 2) or (rot < hd and not _pow2(hd - rot)):
        return 0
    return rot


def ctx_store_supported(draft_model, kv_buf) -> bool:
    """Whether this drafter's context store is one the kernel reproduces: GQA K and V into a bf16 HND paged cache,
    stacked per-layer k_norm, NeoX RoPE over the leading rotary dims (whole or partial head) from the fp32 cos/sin
    cache, head_dim a power of two."""
    first = kv_buf if isinstance(kv_buf, torch.Tensor) else kv_buf[0]
    layer_shape = first.shape[1:] if isinstance(kv_buf, torch.Tensor) else first.shape
    hd = getattr(draft_model, "_head_dim", 0)
    return (
        getattr(draft_model, "_k_norm_stacked", None) is not None
        and getattr(draft_model, "_k_norm_eps", None) is not None
        and getattr(draft_model, "_is_neox", False)
        and getattr(draft_model, "_num_kv_heads", 0) >= 1
        and hd >= 16
        and hd & (hd - 1) == 0
        and first.dtype == torch.bfloat16
        and len(layer_shape) == 5
        and layer_shape[1] == 2
        and layer_shape[2] == draft_model._num_kv_heads
        and layer_shape[4] == hd
        and all(
            t.is_contiguous() for t in ([kv_buf] if isinstance(kv_buf, torch.Tensor) else kv_buf)
        )
        and _rotary_dim(draft_model, hd) > 0
        and _cpp_op() is not None
    )


def fused_ctx_store(
    draft_model,
    projected: torch.Tensor,
    positions: torch.Tensor,
    num_accepted: torch.Tensor,
    request_rows: torch.Tensor,
    cols: Optional[torch.Tensor],
    block_table: torch.Tensor,
    kv_buf: Union[torch.Tensor, Sequence[torch.Tensor]],
    tokens_per_request: int,
    *,
    capacity_counts: Optional[torch.Tensor] = None,
    max_col: Optional[int] = None,
) -> bool:
    """Store the context K / V of ``projected`` (fc + hidden_norm'd rows, request-major, ``tokens_per_request`` rows
    each) into the paged draft cache; returns False (nothing written) when the stock path must run instead.

    ``request_rows`` / ``num_accepted`` are per request, ``positions`` / ``cols`` per row. With ``cols`` None the kernel
    derives each row's column from its position as the stock path does: clamped to the request's pages on a bound pool
    (``capacity_counts``: pages per block-table row, int64) or to ``max_col`` on the private arena."""
    if not ctx_store_supported(draft_model, kv_buf):
        return False
    if cols is None and (
        (capacity_counts is None) == (max_col is None)
        or (capacity_counts is not None and capacity_counts.dtype != torch.int64)
    ):
        return False
    if (
        positions.dtype not in _INT
        or num_accepted.dtype not in _INT
        or request_rows.dtype not in _INT
        or (cols is not None and cols.dtype not in _INT)
        or block_table.dtype not in _INT
        or block_table.dim() != 2
        or (block_table.stride(1) != 1 and block_table.shape[1] != 1)
    ):
        return False
    cache = _cache_arg(kv_buf)
    if cache is None:
        return False
    cache_arg, ptr_list, cache_stride_layer = cache
    if draft_model._fused_kv_weight is None:
        draft_model._build_fused_kv_buffers()
    weight = draft_model._fused_kv_weight
    # precompute_context_kv's GEMM, unchanged.
    if getattr(draft_model, "_input_ln_eps", None) is not None:
        ph = projected.float()
        ph = ph * torch.rsqrt(ph.pow(2).mean(-1, keepdim=True) + draft_model._input_ln_eps)
        projected = ph.to(weight.dtype)
    elif projected.dtype != weight.dtype:
        projected = projected.to(weight.dtype)
    kv_flat = F.linear(projected, weight, draft_model._fused_kv_bias)
    n = kv_flat.shape[0]
    if n == 0:
        return True
    num_layers, nkv, hd = (
        draft_model._num_attn_layers,
        draft_model._num_kv_heads,
        draft_model._head_dim,
    )
    # precompute_context_kv's k_norm, unchanged: contiguous K, F.rms_norm, the stacked weight multiply.
    k = kv_flat.view(n, num_layers, 2, nkv, hd)[:, :, 0].contiguous()
    k = F.rms_norm(k, (hd,), eps=draft_model._k_norm_eps)
    w = draft_model._k_norm_stacked
    fold = (
        _fold_w[0]
        and k.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.is_contiguous()
        and w.numel() == num_layers * hd
        and w.device == k.device
    )
    if not fold:
        k = k * w.view(1, num_layers, 1, hd)
    first = kv_buf if isinstance(kv_buf, torch.Tensor) else kv_buf[0]
    cos_sin = draft_model._get_cos_sin_cache()
    positions = positions.contiguous()
    family_ops().minimax_m31_dspark_ctx_store(
        kv_flat,
        k,
        w,
        cos_sin,
        positions,
        num_accepted.contiguous(),
        request_rows.contiguous(),
        positions if cols is None else cols.contiguous(),
        capacity_counts if capacity_counts is not None else positions,
        int(max_col) if max_col is not None else 0,
        block_table,
        cache_arg,
        cache_stride_layer,
        tokens_per_request,
        num_layers,
        nkv,
        hd,
        first.shape[-2],
        ptr_list,
        cols is None,
        capacity_counts is not None,
        fold,
    )
    return True
