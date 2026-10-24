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
import triton
import triton.language as tl


@triton.jit
def _ctx_store_kernel(
    kv_ptr,  # [N, L * 2 * NKV * HD] the fused K / V GEMM output, per layer [K heads | V heads] (V is read here)
    k_ptr,  # [N, L, NKV, HD] K after torch's k_norm and (without FOLD_W) the weight multiply
    w_ptr,  # [L, HD] bf16 k_norm weight (FOLD_W)
    cs_ptr,  # [max_pos, ROT] fp32 cos | sin (flashinfer's cos/sin cache; ROT = the rotary dim)
    pos_ptr,  # [N] RoPE positions
    nacc_ptr,  # [N // KP1] accepted tokens per request
    rows_ptr,  # [N // KP1] block-table row per request
    col_ptr,  # [N] cache column per row (unused with COL_FROM_POS)
    counts_ptr,  # [rows] int64 pages per block-table row (COL_FROM_POS on a bound pool)
    max_col,  # the private arena's last column (COL_FROM_POS without a bound pool)
    table_ptr,  # [rows, pages_per_row] page ids
    cache_ptr,  # stacked [L, pages, 2, NKV, PS, HD] cache, or an int64 array of L per-layer cache addresses
    kv_stride_row,
    table_stride,
    cache_stride_layer,
    KP1: tl.constexpr,
    L: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    ROT: tl.constexpr,
    PS: tl.constexpr,
    PTR_LIST: tl.constexpr,
    COL_FROM_POS: tl.constexpr,  # derive the column from the position as the stock clamp does
    BOUND: tl.constexpr,  # COL_FROM_POS on a bound pool (clamp to the row's pages) vs the private arena
    FOLD_W: tl.constexpr,  # k_ptr holds rms_norm(k) (bf16); the bf16 weight multiply runs here
):
    r = tl.program_id(0)
    layer = tl.program_id(1)
    h = tl.program_id(2)
    half: tl.constexpr = HD // 2
    rhalf: tl.constexpr = ROT // 2
    o = tl.arange(0, half)
    ro = tl.arange(0, rhalf)
    req = r // KP1
    m = tl.where((r % KP1) < tl.load(nacc_ptr + req), 1.0, 0.0)
    vbase = kv_ptr + r * kv_stride_row + (layer * 2 * NKV + NKV + h) * HD
    kbase = k_ptr + ((r * L + layer) * NKV + h) * HD
    a = tl.load(kbase + ro)
    b = tl.load(kbase + rhalf + ro)
    if FOLD_W:  # torch's bf16 * bf16: the fp32 product rounded to bf16
        wbase = w_ptr + layer * HD
        a = (a.to(tl.float32) * tl.load(wbase + ro).to(tl.float32)).to(tl.bfloat16)
        b = (b.to(tl.float32) * tl.load(wbase + rhalf + ro).to(tl.float32)).to(tl.bfloat16)
    a = a.to(tl.float32)
    b = b.to(tl.float32)
    p = tl.load(pos_ptr + r)
    c = tl.load(cs_ptr + p * ROT + ro)
    s = tl.load(cs_ptr + p * ROT + rhalf + ro)
    ra = (tl.fma(a, c, -b * s).to(tl.bfloat16).to(tl.float32) * m).to(tl.bfloat16)
    rb = (tl.fma(b, c, a * s).to(tl.bfloat16).to(tl.float32) * m).to(tl.bfloat16)
    v0 = (tl.load(vbase + o).to(tl.float32) * m).to(tl.bfloat16)
    v1 = (tl.load(vbase + half + o).to(tl.float32) * m).to(tl.bfloat16)
    row = tl.load(rows_ptr + req)
    last = (r % KP1) == KP1 - 1
    if COL_FROM_POS:
        # The stock column: ctx_len + offset (= the position), clamped to the request's pages (bound pool:
        # min(col, pages * PS - 1).clamp(min=0)) or to the arena (min(col, max_ctx - 1)).
        p_nxt = tl.load(pos_ptr + tl.where(last, r, r + 1))
        if BOUND:
            cap = tl.load(counts_ptr + row) * PS - 1
            col = tl.maximum(tl.minimum(p, cap), 0)
            nxt = tl.maximum(tl.minimum(p_nxt, cap), 0)
        else:
            col = tl.minimum(p, max_col)
            nxt = tl.minimum(p_nxt, max_col)
    else:
        col = tl.load(col_ptr + r)
        nxt = tl.load(
            col_ptr + tl.where(last, r, r + 1)
        )  # the next row of the same request (itself for the last)
    page = tl.load(table_ptr + row * table_stride + col // PS).to(tl.int64)
    if PTR_LIST:
        cache = tl.load(cache_ptr + layer).to(tl.pointer_type(tl.bfloat16))
    else:
        cache = cache_ptr + layer.to(tl.int64) * cache_stride_layer
    kdst = cache + ((page * 2 * NKV + h) * PS + col % PS) * HD
    vdst = cache + ((page * 2 * NKV + NKV + h) * PS + col % PS) * HD
    # Clamped rows of one request share a column; the last of them writes. Stock's order there is a race between the
    # append's thread blocks; that no reader reads such a cell is shown in the S10c-A in-process gate.
    write = last | (nxt != col)
    tl.store(kdst + ro, ra, mask=write)
    tl.store(kdst + rhalf + ro, rb, mask=write)
    if ROT < HD:  # partial rotary: dims ROT..HD are not rotated, only masked
        po = ROT + tl.arange(0, HD - ROT)
        kp = tl.load(kbase + po)
        if FOLD_W:
            kp = (kp.to(tl.float32) * tl.load(w_ptr + layer * HD + po).to(tl.float32)).to(
                tl.bfloat16
            )
        kp = (kp.to(tl.float32) * m).to(tl.bfloat16)
        tl.store(kdst + po, kp, mask=write)
    tl.store(vdst + o, v0, mask=write)
    tl.store(vdst + half + o, v1, mask=write)


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
    _ctx_store_kernel[(n, num_layers, nkv)](
        kv_flat,
        k,
        w,
        cos_sin,
        positions.contiguous(),
        num_accepted.contiguous(),
        request_rows.contiguous(),
        positions if cols is None else cols.contiguous(),
        capacity_counts if capacity_counts is not None else positions,
        int(max_col) if max_col is not None else 0,
        block_table,
        cache_arg,
        kv_flat.stride(0),
        block_table.stride(0),
        cache_stride_layer,
        KP1=tokens_per_request,
        L=num_layers,
        NKV=nkv,
        HD=hd,
        ROT=cos_sin.shape[1],
        PS=first.shape[-2],
        PTR_LIST=ptr_list,
        COL_FROM_POS=cols is None,
        BOUND=capacity_counts is not None,
        FOLD_W=fold,
        num_warps=1,
    )
    return True
