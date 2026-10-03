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
"""Torch ops of the Kimi K3 MLA decode CTM kernels.

A decode step is M = R x T tokens: R <= 8 requests of T <= 8 tokens each, request-major (the rows of request i are
i T .. i T + T - 1). Request i's KV pages are row i of ``page_table`` (int32 [R, W] with unit column stride and any row
stride, e.g. a slice of the attention metadata's kv_cache_block_offsets; [W] for one request) and its KV length,
including its T new tokens, is ``seq_len[i]`` (int32 [R]).

``trtllm::k3_mla_q``: the decode query path (M <= 64; 6 heads per rank at TP16, 24 at TP4): q_a RMSNorm, q_b projection
and k_b absorption in one launch, producing the attention's ``fused_q`` [M, heads * 576] = [q_nope @ W_kb^T | q_pe] per
head; ``trtllm::k3_mla_qkv`` also stores the KV half (kv_a RMSNorm, rope columns) into the paged latent cache.
``trtllm::k3_mla_attn`` and its ``_out`` / ``_vb_out`` forms: the attention of every request over its pages, over a
caller-owned workspace (:func:`make_attn_workspace`). Compiled on the first call for its shapes, which must happen
outside CUDA-graph capture.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

import torch

MAX_TOKENS = 64  # tokens of a k3_mla_q / k3_mla_qkv call
MAX_REQUESTS = 8
MAX_REQUEST_TOKENS = 8  # T
# k3_mla_q's tokens per chunk (the MMA's N) by call size, as (largest call, chunk): a chunk is a cluster of 6 CTAs per
# head, 23 such clusters are resident at once on GB200, and a wider chunk takes longer per CTA. 8 up to 24 tokens (18
# clusters), 16 up to 48 (18), 32 up to 64 (12).
CHUNK_TOKENS = ((24, 8), (48, 16), (64, 32))

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=t.dim() - 1)


def _use_pdl() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


_arg_dummies: Dict[tuple, torch.Tensor] = {}


def _dummy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A tensor argument the build does not touch (the kv arguments when the KV half is off)."""
    key = (device.index, dtype)
    t = _arg_dummies.get(key)
    if t is None:
        t = _arg_dummies[key] = torch.zeros(8, dtype=dtype, device=device)
    return t


def _pool_base(pool: torch.Tensor, row_stride: int) -> torch.Tensor:
    """The pool's first page as the kernels' pointer carrier: they index the pool with 64-bit offsets (or through a
    tensor map sized separately), and a flat view of a multi-GB pool would not fit a 32-bit dynamic shape."""
    return pool.view(-1)[: 64 * row_stride]


def _requests(page_table: torch.Tensor, seq_len: torch.Tensor, num_tokens: int):
    """``(rows, row_stride, R, T)`` of a step of ``num_tokens`` tokens over R = ``seq_len.numel()`` requests: the
    page-table rows as one flat int32 view (request i's row at ``i * row_stride``), or None when the arguments do not
    describe R requests of T = num_tokens / R tokens (see the module docstring)."""
    num_requests = seq_len.numel()
    if not (
        page_table.dtype == seq_len.dtype == torch.int32
        and page_table.device == seq_len.device
        and seq_len.dim() == 1
        and seq_len.is_contiguous()
        and 0 < num_requests <= num_tokens
        and num_tokens % num_requests == 0
        and page_table.dim() in (1, 2)
        and page_table.stride(-1) == 1
        and page_table.shape[-1] > 0
    ):
        return None
    if page_table.dim() == 1:
        return (page_table, 0, 1, num_tokens) if num_requests == 1 else None
    width = page_table.shape[1]
    row_stride = page_table.stride(0) if num_requests > 1 else width
    if page_table.shape[0] != num_requests or row_stride < width:
        return None
    rows = page_table.as_strided(((num_requests - 1) * row_stride + width,), (1,))
    return rows, row_stride, num_requests, num_tokens // num_requests


def supports_kv(ag: torch.Tensor, w_kv: torch.Tensor, pool: torch.Tensor, row_stride: int, page_table: torch.Tensor,
                seq_len: torch.Tensor) -> bool:  # fmt: skip
    """Whether ``k3_mla_qkv`` can store the KV half of the ``ag.shape[0]`` tokens: the latent (512) and rope (64)
    columns after q_a in ``ag``, a dense bf16 pool with rows of ``row_stride`` elements, int32 page-table rows and
    lengths of R requests (see the module docstring)."""
    from . import k3_mla_q_kernel as kernel

    return (
        ag.shape[1] >= kernel.Q_LORA + kernel.FUSED
        and tuple(w_kv.shape) == (kernel.LATENT,)
        and w_kv.dtype == torch.bfloat16
        and w_kv.is_contiguous()
        and pool.dtype == torch.bfloat16
        and pool.is_contiguous()
        and row_stride >= kernel.FUSED
        and row_stride % 8 == 0
        and pool.numel() >= 64 * row_stride
        and _requests(page_table, seq_len, ag.shape[0]) is not None
    )


def supports_q(
    ag: torch.Tensor, w_qa: torch.Tensor, w_qb: torch.Tensor, w_kb: torch.Tensor
) -> bool:
    """Whether ``k3_mla_q`` runs: q_a in the first 1536 columns of dense bf16 rows, the TP16 per-rank shapes."""
    from . import k3_mla_q_kernel as kernel

    return (
        ag.is_cuda
        and ag.dtype == w_qa.dtype == w_qb.dtype == w_kb.dtype == torch.bfloat16
        and ag.dim() == 2
        and 0 < ag.shape[0] <= MAX_TOKENS
        and ag.shape[1] >= kernel.Q_LORA
        and ag.shape[1] % 8 == 0
        and ag.is_contiguous()
        and tuple(w_qa.shape) == (kernel.Q_LORA,)
        and w_kb.dim() == 3
        and tuple(w_kb.shape[1:]) == (kernel.LATENT, kernel.NOPE)
        and 0 < w_kb.shape[0] * kernel.CLUSTER <= 148
        and tuple(w_qb.shape) == (w_kb.shape[0] * kernel.QK, kernel.Q_LORA)
        and w_qa.is_contiguous()
        and w_qb.is_contiguous()
        and w_kb.is_contiguous()
    )


def _launch_q(
    ag, w_qa, eps, w_qb, w_kb, trigger_early=True, single_hop=False, kv=None, cluster_rms=True
) -> torch.Tensor:
    """``kv``: None (query path only) or dict(w, eps) plus either pool, row_stride, page_table, page_offset, seq_len
    (the KV half into the paged pool) or out (into a dense [M, 576] tensor). ``cluster_rms``: each rank of a head's
    cluster reads only its 256 q_a columns and the ranks exchange per-token partial sums of squares (st.async; False:
    every CTA reads all 1536 columns). ``single_hop``: the single-hop k_b reduce (same bits, slower on GB200).
    The call runs in chunks of CHUNK_TOKENS' size for its token count when cluster_rms and two-hop (and, with the KV
    half, when the heads launch a CTA per token of a chunk), else in chunks of 8: every token's bits are the same."""
    import cuda.bindings.driver as cuda_driver

    if not supports_q(ag, w_qa, w_qb, w_kb):
        raise ValueError(
            f"k3_mla_q: unsupported call ag {tuple(ag.shape)} {ag.dtype}, w_qa {tuple(w_qa.shape)}, "
            f"w_qb {tuple(w_qb.shape)}, w_kb {tuple(w_kb.shape)}"
        )
    from . import k3_mla_q_kernel as kernel

    num_tokens, ag_cols = ag.shape
    heads = w_kb.shape[0]
    mma_n = next(n for limit, n in CHUNK_TOKENS if num_tokens <= limit)
    if single_hop or not cluster_rms or (kv is not None and heads * kernel.CLUSTER < mma_n):
        mma_n = kernel.MMA_N
    out = torch.empty(num_tokens, heads * kernel.FUSED, dtype=torch.bfloat16, device=ag.device)
    bf16_dummy, i32_dummy = _dummy(ag.device, torch.bfloat16), _dummy(ag.device, torch.int32)
    kv_mode, row_stride, kv_eps, page_offset = 0, kernel.FUSED, 0.0, 0
    tokens, pt_stride = num_tokens, 0
    w_kv, kv_pool, page_table, seq_len, kv_out = (
        bf16_dummy,
        bf16_dummy,
        i32_dummy,
        i32_dummy,
        bf16_dummy,
    )
    if kv is not None:
        w_kv, kv_eps = kv["w"], float(kv["eps"])
        if heads * kernel.CLUSTER < kernel.MMA_N:
            raise ValueError(
                f"k3_mla_q: the KV half takes {kernel.MMA_N} CTAs per 8 tokens, {heads} heads launch "
                f"{heads * kernel.CLUSTER}"
            )
        if "out" in kv:
            kv_mode, kv_out = 2, kv["out"]
            if not (
                kv_out.is_contiguous()
                and kv_out.dtype == torch.bfloat16
                and kv_out.numel() == num_tokens * kernel.FUSED
            ):
                raise ValueError(
                    f"k3_mla_q: kv out {tuple(kv_out.shape)} {kv_out.dtype} is not a dense [M, 576] bf16"
                )
        else:
            kv_mode, row_stride, page_offset = 1, int(kv["row_stride"]), int(kv["page_offset"])
            page_table, seq_len = kv["page_table"], kv["seq_len"]
            if not supports_kv(ag, w_kv, kv["pool"], row_stride, page_table, seq_len):
                raise ValueError(
                    f"k3_mla_q: unsupported KV half: ag {tuple(ag.shape)}, w_kv {tuple(w_kv.shape)} {w_kv.dtype}, pool "
                    f"{kv['pool'].dtype} rows of {row_stride}, page table {tuple(page_table.shape)} "
                    f"{page_table.dtype} strides {page_table.stride()}, lengths {tuple(seq_len.shape)} {seq_len.dtype}"
                )
            page_table, pt_stride, _, tokens = _requests(page_table, seq_len, num_tokens)
            kv_pool = _pool_base(kv["pool"], row_stride)
    args = (
        _arg(w_qb),
        _arg(w_kb.view(heads * kernel.LATENT, kernel.NOPE)),
        _arg(ag.view(-1)),
        _arg(w_qa),
        _arg(out.view(-1)),
        _arg(w_kv),
        _arg(kv_pool),
        # int32 page-table rows and lengths: views into the metadata buffers, read by scalar loads.
        _arg(page_table, 4),
        _arg(seq_len, 4),
        _arg(kv_out.view(-1)),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(ag.device).cuda_stream)
    use_pdl = _use_pdl()
    key = (
        "k3_mla_q",
        ag_cols,
        heads,
        trigger_early,
        single_hop,
        kv_mode,
        row_stride,
        cluster_rms,
        use_pdl,
        mma_n,
    )
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_mla_q must run once outside CUDA-graph capture first (it compiles its kernel)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_mla_q, *args, num_tokens, tokens, pt_stride, float(eps), kv_eps, page_offset, ag_cols,
                    heads, trigger_early, single_hop, kv_mode, row_stride, cluster_rms, use_pdl, mma_n, stream,
                )  # fmt: skip
    fn(*args, num_tokens, tokens, pt_stride, float(eps), kv_eps, page_offset, stream)
    return out


@torch.library.custom_op("trtllm::k3_mla_q", mutates_args=())
def k3_mla_q(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """``fused_q`` [M <= 64, heads * 576] bf16 from ``ag`` [M, C >= 1536] (q_a = ag[:, :1536]): per head,
    ``[bf16(q_nope @ w_kb[h]^T) | q_pe]`` with ``q = bf16(rmsnorm(q_a) @ w_qb^T)``; each bf16 rounding of the
    unfused RMSNorm -> q_b -> bmm chain is kept. Every 8-token chunk of rows computes as an M <= 8 call would."""
    return _launch_q(ag, w_qa, eps, w_qb, w_kb, trigger_early)


@k3_mla_q.register_fake
def _(ag, w_qa, eps, w_qb, w_kb, trigger_early=True):
    return ag.new_empty((ag.shape[0], w_kb.shape[0] * 576), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::k3_mla_qkv", mutates_args=("pool",))
def k3_mla_qkv(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    w_kv: torch.Tensor,
    kv_eps: float,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """``k3_mla_q`` plus the KV half in the same launch: the cache row ``[bf16(rmsnorm(ag[t, 1536:2048]) * w_kv) |
    ag[t, 2048:2112]]`` (K3 is NoPE) of token t = i T + u (token u of request i, see the module docstring) stored into
    the paged latent ``pool`` at position ``pos = seq_len[i] - T + u``, row ``(page_table[i][pos // 64] +
    page_offset) * 64 + pos % 64`` of ``row_stride`` elements (``seq_len[i] >= T``; a token with ``pos < 0`` is not
    stored). The page table and lengths are read before the grid dependency wait (written before the graph runs)."""
    kv = dict(w=w_kv, eps=kv_eps, pool=pool, row_stride=row_stride, page_table=page_table, page_offset=page_offset,
              seq_len=seq_len)  # fmt: skip
    return _launch_q(ag, w_qa, eps, w_qb, w_kb, trigger_early, kv=kv)


@k3_mla_qkv.register_fake
def _(
    ag,
    w_qa,
    eps,
    w_qb,
    w_kb,
    w_kv,
    kv_eps,
    pool,
    row_stride,
    page_table,
    page_offset,
    seq_len,
    trigger_early=True,
):
    return ag.new_empty((ag.shape[0], w_kb.shape[0] * 576), dtype=torch.bfloat16)


@torch.library.custom_op("trtllm::k3_mla_qkv_out", mutates_args=("kv_out",))
def k3_mla_qkv_out(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    w_kv: torch.Tensor,
    kv_eps: float,
    kv_out: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """``k3_mla_qkv`` with the cache rows stored densely into ``kv_out`` [M, 576] instead of the pool (checks)."""
    return _launch_q(
        ag, w_qa, eps, w_qb, w_kb, trigger_early, kv=dict(w=w_kv, eps=kv_eps, out=kv_out)
    )


@k3_mla_qkv_out.register_fake
def _(ag, w_qa, eps, w_qb, w_kb, w_kv, kv_eps, kv_out, trigger_early=True):
    return ag.new_empty((ag.shape[0], w_kb.shape[0] * 576), dtype=torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------------
# trtllm::k3_mla_attn: decode attention over the paged latent cache (R <= 8 requests of T <= 8 tokens, one cluster of
# 16 CTAs per request and 6 heads)
# ---------------------------------------------------------------------------------------------------------------
def attn_workspace_elems(groups: int) -> int:
    """fp16 elements of a ``k3_mla_attn`` workspace for calls of ``groups`` head groups (heads / 6): the per-CTA
    partials of MAX_REQUESTS x groups x 16 slots, then the no_cluster mode's (m, l) exchange and arrival counters."""
    from . import k3_mla_attn_kernel as kernel

    return (
        kernel.MAX_REQUESTS * groups * kernel.CLUSTER * kernel.WS_SLOT_ELEMS
        + kernel.ws_sync_elems(groups)
    )


def make_attn_workspace(device: torch.device, groups: int) -> torch.Tensor:
    """A new workspace for the ``k3_mla_attn`` calls of ``groups`` head groups on ``device``: fp16
    [attn_workspace_elems(groups)], the partial slots uninitialized (a call reads only the words it wrote) and the
    no_cluster tail zeroed (the arrival counters start at 0). It allocates, so it refuses to run under CUDA-graph
    capture; the zeroing is ordered on the device's current stream."""
    from . import k3_mla_attn_kernel as kernel

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("k3_mla_attn workspaces allocate: make them outside CUDA-graph capture.")
    if groups < 1:
        raise ValueError(f"k3_mla_attn workspace: {groups} head groups")
    ws = torch.empty(attn_workspace_elems(groups), dtype=torch.float16, device=device)
    ws[kernel.MAX_REQUESTS * groups * kernel.CLUSTER * kernel.WS_SLOT_ELEMS :].zero_()
    return ws


def _check_attn_workspace(workspace: torch.Tensor, device: torch.device, groups: int) -> None:
    elems = attn_workspace_elems(groups)
    if not (
        workspace.dtype == torch.float16
        and workspace.dim() == 1
        and workspace.is_contiguous()
        and workspace.numel() == elems
        and workspace.device == device
    ):
        raise ValueError(
            f"k3_mla_attn: workspace {tuple(workspace.shape)} {workspace.dtype} on {workspace.device} is not one for "
            f"{groups} head group(s) on {device}: fp16 [{elems}] (make_attn_workspace)"
        )


def supports_attn(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    seq_len: torch.Tensor,
) -> bool:
    """Whether ``k3_mla_attn`` takes the call: ``q`` [M = R T, heads * 576] bf16 with heads a multiple of 6, R <= 8
    requests of T <= 8 tokens (page-table rows and lengths as in the module docstring), a dense bf16 pool with rows of
    ``row_stride`` elements."""
    from . import k3_mla_attn_kernel as kernel

    if not (
        q.is_cuda
        and q.dtype == pool.dtype == torch.bfloat16
        and q.dim() == 2
        and q.shape[1] % (kernel.HEADS * kernel.QK) == 0
        and q.is_contiguous()
        and pool.is_contiguous()
        and row_stride >= kernel.QK
        and row_stride % 8 == 0
    ):
        return False
    requests = _requests(page_table, seq_len, q.shape[0])
    return (
        requests is not None
        and requests[2] <= kernel.MAX_REQUESTS
        and requests[3] <= kernel.MAX_TOKENS
    )


def _launch_attn(
    q,
    pool,
    row_stride,
    page_table,
    seq_len,
    softmax_scale,
    workspace,
    out=None,
    page_offset=0,
    w_vb=None,
    gate=None,
    gate_col0=0,
):
    import cuda.bindings.driver as cuda_driver

    from . import k3_mla_attn_kernel as kernel

    if not supports_attn(q, pool, row_stride, page_table, seq_len):
        raise ValueError(
            f"k3_mla_attn: unsupported call q {tuple(q.shape)} {q.dtype}, row_stride {row_stride}, page table "
            f"{tuple(page_table.shape)} {page_table.dtype} strides {page_table.stride()}, lengths "
            f"{tuple(seq_len.shape)} {seq_len.dtype}"
        )
    num_tokens = q.shape[0]
    page_rows, pt_stride, num_requests, tokens = _requests(page_table, seq_len, num_tokens)
    total_heads = q.shape[1] // kernel.QK
    fuse_vb = w_vb is not None
    if fuse_vb and not (
        w_vb.dtype == torch.bfloat16
        and w_vb.is_contiguous()
        and tuple(w_vb.shape) == (total_heads, kernel.V_DIM, kernel.LATENT)
    ):
        raise ValueError(
            f"k3_mla_attn: v_b weight {tuple(w_vb.shape)} {w_vb.dtype} is not a dense [heads, 128, 512] bf16"
        )
    width = kernel.V_DIM if fuse_vb else kernel.LATENT
    apply_gate = gate is not None
    if apply_gate and not (
        fuse_vb
        and gate.dtype == torch.bfloat16
        and gate.dim() == 2
        and gate.shape[0] == num_tokens
        and gate.stride(1) == 1
        and 0 <= gate_col0
        and gate_col0 + total_heads * kernel.V_DIM <= gate.shape[1]
    ):
        raise ValueError(
            f"k3_mla_attn: gate {tuple(gate.shape)} {gate.dtype} col0 {gate_col0} does not fit the v_b output"
        )
    groups = total_heads // kernel.HEADS
    _check_attn_workspace(workspace, q.device, groups)
    # More 16-CTA clusters than co-reside would run a second wave: launch without a cluster instead when every CTA fits
    # on the SMs at once (one per SM; its waits are spins).
    clusters = num_requests * groups
    no_cluster = (
        clusters > kernel.CLUSTER_WAVE
        and clusters * kernel.CLUSTER <= torch.cuda.get_device_properties(q.device).multi_processor_count
    )
    if out is None:
        out = torch.empty(num_tokens, total_heads * width, dtype=torch.bfloat16, device=q.device)
    elif not (
        out.is_contiguous()
        and out.dtype == torch.bfloat16
        and out.numel() == num_tokens * total_heads * width
    ):
        raise ValueError(
            f"k3_mla_attn: output {tuple(out.shape)} {out.dtype} is not a dense [M, heads * {width}] bf16"
        )
    total_rows = pool.numel() // row_stride
    gate_flat = gate.as_strided((gate.numel(),), (1,)) if apply_gate else q.view(-1)
    gate_ld = gate.stride(0) if apply_gate else 0
    args = (_arg(q.view(-1)), _arg(_pool_base(pool, row_stride)), _arg(page_rows, 4), _arg(seq_len, 4),
            _arg(workspace), _arg(out.view(-1)), _arg((w_vb if fuse_vb else q).view(-1)), _arg(gate_flat))  # fmt: skip
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    use_pdl = _use_pdl()
    scale_log2 = float(softmax_scale) * kernel.LOG2E
    key = ("k3_mla_attn", row_stride, total_heads, fuse_vb, apply_gate, use_pdl, no_cluster)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_mla_attn must run once outside CUDA-graph capture first (it compiles its kernel)."
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kernel.k3_mla_attn, *args, tokens, num_requests, pt_stride, scale_log2, total_rows,
                    int(page_offset), int(gate_col0), int(gate_ld), row_stride, total_heads, fuse_vb, apply_gate,
                    use_pdl, stream, no_cluster,
                )  # fmt: skip
    fn(
        *args,
        tokens,
        num_requests,
        pt_stride,
        scale_log2,
        total_rows,
        int(page_offset),
        int(gate_col0),
        int(gate_ld),
        stream,
    )
    return out


@torch.library.custom_op("trtllm::k3_mla_attn", mutates_args=("workspace",))
def k3_mla_attn(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    seq_len: torch.Tensor,
    softmax_scale: float,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """MLA decode attention of R <= 8 requests of T <= 8 tokens: ``q`` [M = R T, heads * 576] (``fused_q``, heads a
    multiple of 6, request-major) against the paged latent cache ``pool`` (flat bf16; row i of page p at ``(p * 64 +
    i) * row_stride``, 512 latent then 64 rope columns), request i's pages ``page_table[i]`` and length ``seq_len[i]``
    = L_i (rows including its T new ones; see the module docstring), causal bottom-right (token t of request i sees
    rows <= L_i - T + t). Returns ``[M, heads * 512]`` bf16. The page table and lengths are read before the grid
    dependency wait (they must be written before the CUDA graph runs); q and the pages holding rows >= L_i - T after
    it. Request i's rows are computed as the R = 1 call on its own rows, pages and length would compute them.
    ``workspace``: from :func:`make_attn_workspace` for this device and heads / 6 head groups. A call writes and reads
    its partials there, all after its grid dependency wait, and in the no_cluster mode (more than CLUSTER_WAVE
    clusters, all of whose CTAs fit on the SMs) adds 16 to the arrival counters of its requests' head groups; calls
    on one workspace must run one at a time."""
    return _launch_attn(q, pool, row_stride, page_table, seq_len, softmax_scale, workspace)


@torch.library.custom_op("trtllm::k3_mla_attn_out", mutates_args=("out", "workspace"))
def k3_mla_attn_out(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    """``k3_mla_attn`` into ``out`` (dense [M, heads * 512] bf16), with ``page_offset`` added to every page-table
    entry (the layer's slot in a layer-interleaved pool)."""
    _launch_attn(
        q,
        pool,
        row_stride,
        page_table,
        seq_len,
        softmax_scale,
        workspace,
        out=out,
        page_offset=page_offset,
    )


@torch.library.custom_op("trtllm::k3_mla_attn_vb_out", mutates_args=("out", "workspace"))
def k3_mla_attn_vb_out(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    softmax_scale: float,
    w_vb: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    gate_col0: int = 0,
) -> None:
    """``k3_mla_attn`` with v_b applied in the same launch: ``out`` [M, heads * 128] = per head
    ``bf16(bf16(o) @ w_vb[h]^T)`` for the attention output o, ``w_vb`` = v_b_proj [heads, 128, 512] bf16. With ``gate``
    (bf16 [M, C], sigmoid of head h's gate at columns ``gate_col0 + 128 h``) the output is ``bf16(y * s)``, the
    unfused output gate."""
    _launch_attn(
        q, pool, row_stride, page_table, seq_len, softmax_scale, workspace, out=out, page_offset=page_offset,
        w_vb=w_vb, gate=gate, gate_col0=gate_col0,
    )  # fmt: skip


@k3_mla_attn.register_fake
def _(q, pool, row_stride, page_table, seq_len, softmax_scale, workspace):
    return q.new_empty((q.shape[0], q.shape[1] // 576 * 512), dtype=torch.bfloat16)
