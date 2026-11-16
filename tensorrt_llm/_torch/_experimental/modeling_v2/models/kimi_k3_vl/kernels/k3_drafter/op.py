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
"""Torch ops of the Kimi K3 DSpark drafter CTM kernels.

``trtllm::k3_drafter_attn``: the draft blocks of R <= 8 requests (T <= 8 tokens each, M = R T rows of ``qkv``), each
attending densely to its request's pages of the drafter's paged cache plus its own K / V (taken from ``qkv``, not
appended to the cache), GQA groups of 6 query heads per KV head, head dim 64, pages of 64 rows (HND).

Requests: R = ``ctx_len.numel()`` (int32 [R], the cached rows before each block) and T = M / R. ``page_table`` is
int32 with dense rows: [>= R, width] (any row stride, e.g. rows of the pool's block table) or, for R = 1, one row
[width]. Request r reads rows r T .. r T + T - 1 of ``qkv`` (and ``positions``), page-table row r and
``ctx_len[r]``, and writes rows r T .. of ``out``.

Compiled on the first call for its head counts, cache page stride and whether R > 1 (not for R or T otherwise),
which must happen outside CUDA-graph capture.
"""

from __future__ import annotations

import os
import threading
from typing import Dict

import torch

MAX_TOKENS = 8  # per request
MAX_REQUESTS = 8
HEADS_PER_KV = 6
HEAD_DIM = 64
PAGE = 64

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_arg_dummies: Dict[tuple, torch.Tensor] = {}


def _dummy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A tensor argument the build does not touch (the norm / RoPE inputs when the kernel does not apply them)."""
    key = (device.index, dtype)
    t = _arg_dummies.get(key)
    if t is None:
        t = _arg_dummies[key] = torch.zeros(MAX_TOKENS, dtype=dtype, device=device)
    return t


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def supports_attn(
    qkv: torch.Tensor, cache: torch.Tensor, num_heads: int, num_kv_heads: int, num_requests: int = 1
) -> bool:
    """Whether ``k3_drafter_attn`` takes the call: ``qkv`` [M = R T, (heads + 2 kv_heads) * 64] dense bf16 of
    R = ``num_requests`` <= 8 blocks of T <= 8 tokens, 6 query heads per KV head, and an HND cache
    [pages, 2, kv_heads, 64, 64] bf16 whose pages are dense (any page stride)."""
    if not (qkv.is_cuda and cache.is_cuda and qkv.dtype == cache.dtype == torch.bfloat16):
        return False
    if num_kv_heads < 1 or num_heads != HEADS_PER_KV * num_kv_heads:
        return False
    if not 0 < num_requests <= MAX_REQUESTS or qkv.dim() != 2 or not qkv.is_contiguous():
        return False
    if qkv.shape[0] % num_requests or not 0 < qkv.shape[0] // num_requests <= MAX_TOKENS:
        return False
    if qkv.shape[1] != (num_heads + 2 * num_kv_heads) * HEAD_DIM:
        return False
    if cache.dim() != 5 or tuple(cache.shape[1:]) != (2, num_kv_heads, PAGE, HEAD_DIM):
        return False
    inner = (num_kv_heads * PAGE * HEAD_DIM, PAGE * HEAD_DIM, HEAD_DIM, 1)
    return (
        tuple(cache.stride()[1:]) == inner
        and cache.stride(0) % 8 == 0
        and cache.data_ptr() % 16 == 0
    )


def _launch_attn(
    qkv, cache, page_table, ctx_len, num_heads, num_kv_heads, out, norm=None
) -> torch.Tensor:
    """``norm``: None (qkv already normalized and roped) or dict(q_w, k_w, positions, eps, base) (qkv raw)."""
    import cuda.bindings.driver as cuda_driver

    num_requests = ctx_len.numel()
    if not supports_attn(qkv, cache, num_heads, num_kv_heads, num_requests):
        raise ValueError(
            f"k3_drafter_attn: unsupported call qkv {tuple(qkv.shape)} {qkv.dtype} for {num_requests} requests, "
            f"cache {tuple(cache.shape)} {tuple(cache.stride())} {cache.dtype}, heads {num_heads} / {num_kv_heads}"
        )
    from . import k3_drafter_attn_kernel as kernel

    num_rows = qkv.shape[0]
    num_tokens = num_rows // num_requests
    if not (
        out.is_contiguous()
        and out.dtype == torch.bfloat16
        and out.numel() == num_rows * num_heads * HEAD_DIM
    ):
        raise ValueError(
            f"k3_drafter_attn: output {tuple(out.shape)} {out.dtype} is not a dense [M, heads * 64] bf16"
        )
    if page_table.dtype != torch.int32 or ctx_len.dtype != torch.int32:
        raise ValueError(
            f"k3_drafter_attn: page table {page_table.dtype} / lengths {ctx_len.dtype} not int32"
        )
    # Request r's pages start at element r * table_stride of the table (a view over the R rows, no copy).
    if (
        page_table.dim() == 2
        and page_table.shape[0] >= num_requests
        and (page_table.shape[1] == 1 or page_table.stride(1) == 1)
    ):
        table_stride = page_table.stride(0)
        table = page_table.as_strided(
            ((num_requests - 1) * table_stride + page_table.shape[1],), (1,)
        )
    elif page_table.dim() == 1 and num_requests == 1:
        table = page_table.reshape(-1)
        table_stride = table.numel()
    else:
        raise ValueError(
            f"k3_drafter_attn: page table {tuple(page_table.shape)} {tuple(page_table.stride())} is not "
            f"[>= {num_requests}, width] with dense rows (or one dense row for one request)"
        )
    norm_rope = norm is not None
    q_w = k_w = _dummy(qkv.device, torch.bfloat16)
    positions = _dummy(qkv.device, torch.int32)
    norm_eps, rope_base = 0.0, 1.0
    if norm_rope:
        q_w, k_w, positions = norm["q_w"], norm["k_w"], norm["positions"]
        norm_eps, rope_base = float(norm["eps"]), float(norm["base"])
        if not (
            q_w.dtype == k_w.dtype == torch.bfloat16
            and q_w.numel() == k_w.numel() == HEAD_DIM
            and q_w.is_contiguous()
            and k_w.is_contiguous()
            and positions.dtype in (torch.int32, torch.int64)
            and positions.numel() >= num_rows
        ):
            raise ValueError(
                "k3_drafter_attn: q/k norm weights must be bf16 [64], positions int32 / int64 [>= M]"
            )
    # The cache's first page carries the pointer; the tensor map gets the page count and stride separately.
    base = cache.as_strided((HEAD_DIM,), (1,))
    args = (
        _arg(qkv.view(-1)),
        _arg(base),
        _arg(table, align=4),  # a view at any row of a block table; read one int32 at a time
        _arg(ctx_len.reshape(-1)),
        _arg(out.view(-1)),
        _arg(q_w.reshape(-1)),
        _arg(k_w.reshape(-1)),
        _arg(positions.reshape(-1)),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(qkv.device).cuda_stream)
    use_pdl = _use_pdl()
    page_stride = cache.stride(0)
    total_pages = cache.shape[0]
    scale_log2 = kernel.softmax_scale_log2(HEAD_DIM)
    multi_request = num_requests > 1
    key = (
        "k3_drafter_attn",
        num_heads,
        num_kv_heads,
        page_stride,
        norm_rope,
        positions.dtype,
        multi_request,
        use_pdl,
    )
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_drafter_attn must run once outside CUDA-graph capture first (it compiles)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_drafter_attn, *args, num_tokens, num_requests, table_stride, scale_log2, total_pages,
                    norm_eps, rope_base, num_heads, num_kv_heads, page_stride, norm_rope, multi_request, use_pdl,
                    stream,
                )  # fmt: skip
    fn(*args, num_tokens, num_requests, table_stride, scale_log2, total_pages, norm_eps, rope_base,
       stream)  # fmt: skip
    return out


@torch.library.custom_op("trtllm::k3_drafter_attn", mutates_args=("out",))
def k3_drafter_attn(
    qkv: torch.Tensor,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    ctx_len: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    out: torch.Tensor,
) -> None:
    """``out`` [M, heads * 64] = softmax(q K^T / 8) V per query head of each of the R = ``ctx_len.numel()`` requests,
    dense over its ``ctx_len[r]`` cached rows of ``cache`` (pages: row r of ``page_table``) and its block's own
    T = M / R rows of k / v in ``qkv`` (q, k after RMSNorm + RoPE). See the module docstring for the layouts."""
    _launch_attn(qkv, cache, page_table, ctx_len, num_heads, num_kv_heads, out)


@k3_drafter_attn.register_fake
def _(qkv, cache, page_table, ctx_len, num_heads, num_kv_heads, out):
    return None


@torch.library.custom_op("trtllm::k3_drafter_attn_qknorm", mutates_args=("out",))
def k3_drafter_attn_qknorm(
    qkv: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    rope_base: float,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    ctx_len: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    out: torch.Tensor,
) -> None:
    """``k3_drafter_attn`` on the raw projection output: the per-head q / k RMSNorm (``q_w``, ``k_w``, ``eps``) and the
    NeoX RoPE (``rope_base``, ``positions`` [M], one per row of ``qkv``) are applied in the kernel with
    fused_qk_norm_rope's arithmetic."""
    norm = dict(q_w=q_w, k_w=k_w, positions=positions, eps=eps, base=rope_base)
    _launch_attn(qkv, cache, page_table, ctx_len, num_heads, num_kv_heads, out, norm=norm)


@k3_drafter_attn_qknorm.register_fake
def _(
    qkv,
    q_w,
    k_w,
    positions,
    eps,
    rope_base,
    cache,
    page_table,
    ctx_len,
    num_heads,
    num_kv_heads,
    out,
):
    return None


def _per_request(qkv: torch.Tensor, page_table: torch.Tensor, num_requests: int):
    """(rows of ``qkv``, page-table row) of each request."""
    tables = page_table.unsqueeze(0) if page_table.dim() == 1 else page_table
    t = qkv.shape[0] // num_requests
    return [(qkv[r * t : (r + 1) * t], tables[r]) for r in range(num_requests)]


def reference(qkv: torch.Tensor, cache: torch.Tensor, page_table: torch.Tensor, ctx, num_heads: int,
              num_kv_heads: int) -> torch.Tensor:  # fmt: skip
    """torch reference of ``k3_drafter_attn`` with host-known lengths ``ctx`` (an int for one request, else one per
    request), [M, heads * 64] fp32 (computed in float64: no TF32 even where cuBLAS is told to use it)."""
    ctxs = [ctx] if isinstance(ctx, int) else [int(c) for c in ctx]
    return torch.cat([
        _reference_one(rows, cache, table, c, num_heads, num_kv_heads)
        for (rows, table), c in zip(_per_request(qkv, page_table, len(ctxs)), ctxs)
    ]).float()  # fmt: skip


def _reference_one(qkv, cache, page_table, ctx, num_heads, num_kv_heads):
    m = qkv.shape[0]
    q = qkv[:, : num_heads * HEAD_DIM].double().view(m, num_heads, HEAD_DIM)
    k_blk = (
        qkv[:, num_heads * HEAD_DIM : (num_heads + num_kv_heads) * HEAD_DIM]
        .double()
        .view(m, num_kv_heads, HEAD_DIM)
    )
    v_blk = qkv[:, (num_heads + num_kv_heads) * HEAD_DIM :].double().view(m, num_kv_heads, HEAD_DIM)
    n_pages = (ctx + PAGE - 1) // PAGE
    pages = page_table[:n_pages].long()
    k_ctx = (
        cache[pages, 0].double().permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)[:, :ctx]
    )
    v_ctx = (
        cache[pages, 1].double().permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)[:, :ctx]
    )
    k_all = torch.cat([k_ctx, k_blk.permute(1, 0, 2)], dim=1)  # [kv, L, 64]
    v_all = torch.cat([v_ctx, v_blk.permute(1, 0, 2)], dim=1)
    group = num_heads // num_kv_heads
    k_h = k_all.repeat_interleave(group, dim=0)  # [heads, L, 64]
    v_h = v_all.repeat_interleave(group, dim=0)
    s = torch.einsum("thd,hld->thl", q, k_h) / (HEAD_DIM**0.5)
    p = torch.softmax(s, dim=-1)
    return torch.einsum("thl,hld->thd", p, v_h).reshape(m, num_heads * HEAD_DIM)


def reference_masked(qkv: torch.Tensor, cache: torch.Tensor, page_table: torch.Tensor, ctx_len: torch.Tensor,
                     num_heads: int, num_kv_heads: int) -> torch.Tensor:  # fmt: skip
    """torch reference of ``k3_drafter_attn`` with the lengths on the device (CUDA-graph safe): every page of a
    request's page-table row is gathered and its rows >= ``ctx_len[r]`` are masked (their V zeroed), [M, heads * 64]
    fp32 (computed in float64)."""
    lens = ctx_len.reshape(-1)
    return torch.cat([
        _reference_masked_one(rows, cache, table, lens[r : r + 1], num_heads, num_kv_heads)
        for r, (rows, table) in enumerate(_per_request(qkv, page_table, lens.numel()))
    ]).float()  # fmt: skip


def _reference_masked_one(qkv, cache, page_table, ctx_len, num_heads, num_kv_heads):
    m = qkv.shape[0]
    pages = page_table.reshape(-1).long().clamp(min=0, max=cache.shape[0] - 1)
    n_rows = pages.numel() * PAGE
    k_ctx = cache[pages, 0].double().permute(1, 0, 2, 3).reshape(num_kv_heads, n_rows, HEAD_DIM)
    v_ctx = cache[pages, 1].double().permute(1, 0, 2, 3).reshape(num_kv_heads, n_rows, HEAD_DIM)
    valid_ctx = torch.arange(n_rows, device=qkv.device) < ctx_len.long()
    k_blk = (
        qkv[:, num_heads * HEAD_DIM : (num_heads + num_kv_heads) * HEAD_DIM]
        .double()
        .view(m, num_kv_heads, HEAD_DIM)
    )
    v_blk = qkv[:, (num_heads + num_kv_heads) * HEAD_DIM :].double().view(m, num_kv_heads, HEAD_DIM)
    k_all = torch.cat([k_ctx, k_blk.permute(1, 0, 2)], dim=1)
    v_all = torch.cat([v_ctx, v_blk.permute(1, 0, 2)], dim=1)
    valid = torch.cat([valid_ctx, torch.ones(m, dtype=torch.bool, device=qkv.device)])
    v_all = torch.where(valid.view(1, -1, 1), v_all, torch.zeros_like(v_all))
    group = num_heads // num_kv_heads
    q = qkv[:, : num_heads * HEAD_DIM].double().view(m, num_heads, HEAD_DIM)
    s = torch.einsum("thd,hld->thl", q, k_all.repeat_interleave(group, dim=0)) / (HEAD_DIM**0.5)
    s = s.masked_fill(~valid.view(1, 1, -1), float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("thl,hld->thd", p, v_all.repeat_interleave(group, dim=0)).reshape(
        m, num_heads * HEAD_DIM
    )
