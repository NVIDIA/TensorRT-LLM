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
"""Opt-in DSA sparse MLA prefill on the PrimTS D576 kernel over the per-tensor FP8 latent pool (Kimi K3.1).

Replaces the FlashMLA sparse prefill of ``flash_mla_prefill`` (BF16 queries padded to 128 heads over a compact BF16
copy of the context KV) for the context rows of attention-DP layers and of ShardQ: the absorbed query is quantized
to E4M3 per tensor exactly like MLA generation preprocessing (``kv_scale_orig_quant``, saturating), the indexer's
request-local Top-K becomes rows of the unique primary pool (``convertReqIndexToGlobal``), and PrimTS attends the
FP8 rows in place with every head of the call natively (96 heads run on the 128-head 2-CTA tile). The KV gather,
the index rebase and the head padding disappear; the softmax and output scales carry the Q / KV descales.

Enabled by ``TRTLLM_KIMI_K31_SPARSE_MLA_PREFILL=prims_ts`` on SM100 / SM103 with an FP8 latent cache.

``nvfp4_ds_mla`` records (inline NVFP4 latents) are read in place by the PrimTS ``nvfp4.fp8rope`` loader, which turns
each record into E4M3 tiles for the ordinary FP8 MMA. On a Kimi K3.1 disaggregated context server
(``_nvfp4_prefill_dequant``) the context prefill instead converts the records of its requests once per layer into an
E4M3 D576 scratch with the loader's arithmetic and runs the native E4M3 kernel over the scratch: the same output bit
for bit, without the in-loop conversion. The scratch holds ``_DEQUANT_TOKENS`` rows (576 B each), reserved by the
first context prefill (the KV-cache memory profiling forward); a step whose context requests hold more tokens takes
the in-place path.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version, nvtx_range
from tensorrt_llm.logger import logger

from .metadata import DSAtrtllmAttentionMetadata
from .params import disagg_role, is_kimi_k31_dsa

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.mla import MLA

SPARSE_MLA_PREFILL_ENV = "TRTLLM_KIMI_K31_SPARSE_MLA_PREFILL"
# Rows of the E4M3 scratch of the nvfp4_ds_mla context prefill.
_DEQUANT_TOKENS = 1 << 20
_DEQUANT_SCRATCH: dict = {}
_LATENT = 512
_ROPE = 64
_QUERY = _LATENT + _ROPE
_FP8_MAX = 448.0
# Row capacities planned per head count: powers of two up to the largest call. A call runs on the smallest capacity
# that holds its rows, so the number of plans (and kernel variants) stays bounded however the chunk sizes vary.
_MAX_ROWS = 1 << 16
_PLANS: dict = {}
_WORKSPACE: dict = {}


def _nvfp4_prefill_dequant() -> bool:
    """Whether the nvfp4_ds_mla context prefill runs over the E4M3 scratch: Kimi K3.1 on a disaggregated context
    server (a generation server reserves no scratch)."""
    return is_kimi_k31_dsa() and disagg_role() == "context"


def prims_ts_prefill_requested() -> bool:
    """Whether the PrimTS sparse MLA prefill is requested."""
    return os.environ.get(SPARSE_MLA_PREFILL_ENV, "").strip().lower() == "prims_ts"


def prims_ts_prefill_supported(mla: "MLA", num_heads: int) -> bool:
    """PrimTS D576 prefill: SM100 / SM103, per-tensor FP8 latent pool, latent 512 + RoPE 64, at most 128 heads.

    Static per layer and head count; evaluated once and cached on the module."""
    cache = mla.__dict__.setdefault("_prims_ts_prefill_supported", {})
    supported = cache.get(num_heads)
    if supported is None:
        supported = _prims_ts_prefill_supported(mla, num_heads)
        cache[num_heads] = supported
    return supported


def _prims_ts_prefill_supported(mla: "MLA", num_heads: int) -> bool:
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
        get_prims_ts_toolchain_unsupported_reason,
    )

    if not (
        prims_ts_prefill_requested()
        and get_sm_version() in (100, 103)
        and int(mla.kv_lora_rank) == _LATENT
        and int(mla.qk_rope_head_dim) == _ROPE
        and 1 <= num_heads <= 128
        and mla.kv_cache_dtype != "fp8_ds_mla"
        and getattr(mla.mqa, "has_fp8_kv_cache", False)
        and not getattr(mla.mqa, "has_fp4_kv_cache", False)
        and mla.k_b_proj_trans.dtype == torch.bfloat16
        and mla.v_b_proj.dtype == torch.bfloat16
    ):
        return False
    return get_prims_ts_toolchain_unsupported_reason() is None


@triton.jit
def _quant_query_kernel(
    lat_ptr,  # bf16 [n, H, 512] (absorbed q_nope), last dim contiguous
    pe_ptr,  # bf16 [n, H, 64] (rope'd q_pe), last dim contiguous
    out_ptr,  # fp8 [n, H, 576] contiguous
    scale_ptr,  # fp32 [1] quantization scale (latent part)
    pe_scale_ptr,  # fp32 [1] quantization scale (RoPE part)
    num_rows,  # n * H
    heads,
    lat_tok_stride,
    lat_head_stride,
    pe_tok_stride,
    pe_head_stride,
    FP8_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = r < num_rows
    t = (r // heads).to(tl.int64)
    h = (r % heads).to(tl.int64)
    s = tl.load(scale_ptr)
    d = tl.arange(0, 512)
    x = tl.load(
        lat_ptr + (t * lat_tok_stride + h * lat_head_stride)[:, None] + d[None, :], mask=m[:, None]
    )
    x = tl.minimum(tl.maximum(x.to(tl.float32) * s, -FP8_MAX), FP8_MAX)
    o = r.to(tl.int64) * 576
    tl.store(out_ptr + o[:, None] + d[None, :], x.to(tl.float8e4nv), mask=m[:, None])
    e = tl.arange(0, 64)
    y = tl.load(
        pe_ptr + (t * pe_tok_stride + h * pe_head_stride)[:, None] + e[None, :], mask=m[:, None]
    )
    y = tl.minimum(tl.maximum(y.to(tl.float32) * tl.load(pe_scale_ptr), -FP8_MAX), FP8_MAX)
    tl.store(out_ptr + o[:, None] + 512 + e[None, :], y.to(tl.float8e4nv), mask=m[:, None])


def quantize_query_e4m3(
    q_latent: torch.Tensor,
    q_pe: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    pe_scale: Optional[torch.Tensor] = None,
) -> None:
    """``out[n, H, 576] = e4m3(clamp([q_latent * scale | q_pe * pe_scale], +-448))`` (saturating, round to
    nearest); ``pe_scale`` defaults to ``scale``."""
    n, heads, _ = q_latent.shape
    rows = n * heads
    if rows == 0:
        return
    block_r = 16
    _quant_query_kernel[(triton.cdiv(rows, block_r),)](
        q_latent,
        q_pe,
        out,
        scale,
        scale if pe_scale is None else pe_scale,
        rows,
        heads,
        q_latent.stride(0),
        q_latent.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        FP8_MAX=_FP8_MAX,
        BLOCK_R=block_r,
    )


def _dequant_scratch(device: torch.device) -> torch.Tensor:
    """The per-device E4M3 D576 scratch of the nvfp4_ds_mla context prefill (uint8 ``[tokens, 576]``), reserved once:
    the first call is the KV-cache memory profiling forward, so the KV pool is sized around it."""
    scratch = _DEQUANT_SCRATCH.get(device)
    if scratch is None:
        tokens = _DEQUANT_TOKENS
        scratch = torch.empty((tokens, _QUERY), dtype=torch.uint8, device=device)
        _DEQUANT_SCRATCH[device] = scratch
        logger.info(
            f"nvfp4_ds_mla context prefill: E4M3 scratch of {tokens} tokens ({tokens * _QUERY / 2**20:.0f} MiB) "
            "reserved"
        )
    return scratch


@triton.jit
def _rebase_topk_kernel(
    topk_ptr,  # int32 [n, K] request-local positions, < 0 = none
    req_ptr,  # int32 [n] context request of each row
    indptr_ptr,  # int64 [R + 1] first scratch row of each request
    out_ptr,  # int32 [n, K]
    top_k,
    topk_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < top_k
    base = tl.load(indptr_ptr + tl.load(req_ptr + row)).to(tl.int32)
    t = tl.load(topk_ptr + row.to(tl.int64) * topk_stride + c, mask=m, other=-1)
    tl.store(out_ptr + row.to(tl.int64) * top_k + c, tl.where(t >= 0, t + base, -1), mask=m)


def _rebase_topk(
    topk_local: torch.Tensor, req_idx: torch.Tensor, kv_indptr: torch.Tensor
) -> torch.Tensor:
    """Request-local Top-K positions -> rows of the request-packed scratch (``kv_indptr[req] + position``; -1 kept)."""
    n, top_k = topk_local.shape
    out = torch.empty((n, top_k), dtype=torch.int32, device=topk_local.device)
    if n:
        block = 1024
        _rebase_topk_kernel[(n, triton.cdiv(top_k, block))](
            topk_local, req_idx, kv_indptr, out, top_k, topk_local.stride(0), BLOCK=block
        )
    return out


def _capacity(rows: int) -> int:
    return max(1, 1 << (rows - 1).bit_length())


def _plan(device: torch.device, rows: int, heads: int, top_k: int, kv_cache_format: str = "native"):
    """The wrapper planned for ``rows``'s capacity bucket and its full-capacity ``qo_indptr``."""
    from tensorrt_llm._torch.attention.backends.prims_ts_next import mla_decode

    capacity = _capacity(rows)
    if capacity > _MAX_ROWS:
        raise ValueError(
            f"PrimTS sparse MLA prefill takes at most {_MAX_ROWS} rows per call, got {rows}."
        )
    key = (device, capacity, heads, top_k, kv_cache_format)
    plan = _PLANS.get(key)
    if plan is not None:
        return plan
    kwargs = dict(
        max_topk=top_k,
        max_q_len=1,
        packed_query=True,
        q_dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=_ROPE,
        kv_cache_format=kv_cache_format,
        return_lse=False,
    )
    need = int(
        mla_decode.get_prims_ts_sparse_mla_decode_workspace_size(device, capacity, heads, **kwargs)
    )
    workspace = _WORKSPACE.get(device)
    if workspace is None or workspace.numel() < need:
        workspace = torch.empty((need,), dtype=torch.uint8, device=device)
        _WORKSPACE[device] = workspace
    wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper(workspace)
    wrapper.plan(device, capacity, heads, **kwargs)
    plan = (wrapper, torch.arange(capacity + 1, dtype=torch.int32, device=device))
    _PLANS[key] = plan
    return plan


def _nvfp4_prefill_scratch(
    attn_metadata: DSAtrtllmAttentionMetadata, device: torch.device
) -> Optional[torch.Tensor]:
    """The scratch rows of this step's context requests, or None (they hold more tokens than the reservation)."""
    scratch = _dequant_scratch(device)
    total = int(attn_metadata.host_ctx_kv_indptr[attn_metadata.num_contexts])
    if total > scratch.shape[0]:
        logger.warning_once(
            f"nvfp4_ds_mla context prefill: {total} context KV tokens exceed the {scratch.shape[0]}-token E4M3 "
            "scratch; such steps read the records in place",
            key="nvfp4_prefill_dequant_overflow",
        )
        return None
    return scratch[:total]


def _host_scales(mla: "MLA") -> tuple[float, float, torch.Tensor]:
    """(softmax scale with the Q and KV descales, KV descale, device Q quantization scale); read once per layer."""
    cached = getattr(mla, "_prims_ts_prefill_scales", None)
    if cached is None:
        kv_dequant = float(mla.mqa.kv_scale_quant_orig.reshape(-1)[0].item())
        q_quant = mla.mqa.kv_scale_orig_quant.reshape(-1)[:1].to(torch.float32).contiguous()
        q_dequant = 1.0 / float(q_quant[0].item())
        for value in (kv_dequant, q_dequant):
            if not (math.isfinite(value) and value > 0):
                raise RuntimeError(
                    f"PrimTS sparse MLA prefill needs positive finite KV scales, got {value}."
                )
        cached = (float(mla.softmax_scale) * q_dequant * kv_dequant, kv_dequant, q_quant)
        mla._prims_ts_prefill_scales = cached
    return cached


def _nvfp4_pe_quant(mla: "MLA", q_quant: torch.Tensor, scales) -> torch.Tensor:
    """Device quantization scale of the query's RoPE part over ``nvfp4_ds_mla`` records; cached per layer."""
    cached = getattr(mla, "_prims_ts_prefill_nvfp4_pe_quant", None)
    if cached is None:
        cached = (q_quant * (scales.rope / scales.nope)).contiguous()
        mla._prims_ts_prefill_nvfp4_pe_quant = cached
    return cached


@nvtx_range("prims_ts_sparse_prefill_rows")
def prims_ts_sparse_prefill_rows(
    mla: "MLA",
    q_heads: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    topk_local: torch.Tensor,
    req_idx: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    out: torch.Tensor,
    q_latent: Optional[torch.Tensor] = None,
    kv_cache_format: str = "native",
) -> None:
    """Absorb, PrimTS sparse prefill over the FP8 pool and V projection for ``n`` context rows with all ``H`` heads.

    q_heads ``[n, H, qk_nope + qk_rope]`` (RoPE applied), w_uk ``[H, kv_lora, qk_nope]``, w_uv ``[H, v_head,
    kv_lora]``, topk_local ``[n, K]`` request-local Top-K positions (-1 = none) of the rows whose context requests
    are ``req_idx`` ``[n]`` (int32, index into the context block table); writes ``out`` ``[n, H, v_head]``.
    """
    n, heads, _ = q_heads.shape
    if n == 0:
        return
    device = q_heads.device
    q_nope, q_pe = q_heads.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1)
    if q_latent is None:
        q_latent = q_heads.new_empty(n, heads, _LATENT)
    mla._bmm_bf16_out(q_nope.transpose(0, 1), w_uk, w_uk.transpose(1, 2), q_latent.transpose(0, 1))
    softmax_scale, kv_dequant, q_quant = _host_scales(mla)
    record_bytes = _QUERY
    pe_quant = None
    if kv_cache_format == "nvfp4.fp8rope":
        # Inline NVFP4 records (nvfp4_ds_mla): the records hold x / g (NoPE) and x / g_rope (RoPE); the RoPE part
        # of the query carries g_rope / g, the softmax and output scales carry g.
        from ..nvfp4_ds_mla_kv import TOKEN_BYTES, record_scales

        scales = record_scales()
        softmax_scale *= scales.nope / kv_dequant
        kv_dequant = scales.nope
        record_bytes = TOKEN_BYTES
        pe_quant = _nvfp4_pe_quant(mla, q_quant, scales)
    query = torch.empty(n, heads, _QUERY, dtype=torch.float8_e4m3fn, device=device)
    quantize_query_e4m3(q_latent, q_pe, q_quant, query, pe_quant)

    attn_metadata._ensure_pool_view_cached()
    local_layer_idx = mla.mqa.get_local_layer_idx(attn_metadata)
    page_scale, layer_offset = attn_metadata.kv_cache_manager.get_primary_pool_page_index_params(
        local_layer_idx
    )
    tpb = attn_metadata._cached_tokens_per_block
    top_k = int(topk_local.shape[1])
    scratch_rows = None
    if kv_cache_format == "nvfp4.fp8rope" and _nvfp4_prefill_dequant():
        scratch_rows = _nvfp4_prefill_scratch(attn_metadata, device)
    if scratch_rows is not None:
        # Records of the context requests -> E4M3 rows of a request-packed scratch, attended by the native kernel.
        from ..nvfp4_ds_mla_kv import paged_records_to_e4m3

        num_contexts = attn_metadata.num_contexts
        kv_indptr = attn_metadata.ctx_kv_indptr[: num_contexts + 1]
        paged_records_to_e4m3(
            attn_metadata._cached_pool_view.view(torch.uint8).view(-1, record_bytes),
            attn_metadata._cached_block_table_ctx,
            kv_indptr,
            int(attn_metadata.max_ctx_kv_len),
            tpb,
            page_scale * tpb,
            layer_offset,
            scratch_rows,
        )
        indices = _rebase_topk(topk_local.contiguous(), req_idx.contiguous(), kv_indptr)
        pool = scratch_rows.view(torch.float8_e4m3fn)
        kv_cache_format = "native"
    else:
        indices = torch.ops.trtllm.convert_req_index_to_global(
            req_idx.contiguous(),
            attn_metadata._cached_block_table_ctx,
            topk_local.contiguous(),
            tpb,
            top_k,
            page_scale * tpb,
            layer_offset,
        ).view(n, top_k)
        if kv_cache_format == "native":
            pool = attn_metadata._cached_pool_view.view(torch.float8_e4m3fn).view(-1, _QUERY)
        else:
            pool = attn_metadata._cached_pool_view.view(torch.uint8).view(-1, record_bytes)

    wrapper, qo_full = _plan(device, n, heads, top_k, kv_cache_format)
    lengths = torch.full((n,), top_k, dtype=torch.int32, device=device)
    latent_out = q_latent  # the absorbed query is dead after quantization: reuse its storage
    wrapper.run(
        query,
        pool,
        indices,
        lengths,
        qo_indptr=qo_full.clamp(max=n),
        softmax_scale=softmax_scale,
        output_scale=kv_dequant,
        out=latent_out,
        validate=False,
    )
    mla._bmm_bf16_out(latent_out.transpose(0, 1), w_uv, w_uv.transpose(1, 2), out.transpose(0, 1))


@nvtx_range("forward_sparse_mla_nvfp4_ds_mla_ctx")
def forward_context_sparse_attn_nvfp4_ds_mla(
    mla: "MLA",
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    output: torch.Tensor,
    indexer_intermediates,
) -> torch.Tensor:
    """Context rows of an ``nvfp4_ds_mla`` layer: append the chunk's latent rows as inline NVFP4 records, then
    PrimTS sparse prefill over the records in place with this rank's heads."""
    from ...fmha.prims_ts_dsa_mla import _require_identity_rope, append_nvfp4_ds_mla_rows

    num_tokens = q.shape[0]
    heads = mla.num_heads_tp
    qk_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    _require_identity_rope(mla.mqa)
    attn_metadata._ensure_pool_view_cached()
    append_nvfp4_ds_mla_rows(
        mla.mqa,
        attn_metadata,
        latent_cache.view(num_tokens, -1),
        0,
        num_tokens,
        is_generation=False,
    )
    indexer = mla.mqa.indexer
    shared = attn_metadata.shared_topk_indices
    if indexer is None:
        topk = shared[:num_tokens]
    else:
        topk = indexer.forward_from_projected(
            attn_metadata, q, indexer_intermediates, is_generation=False
        )
        if shared is not None:
            shared[: topk.shape[0], : topk.shape[1]].copy_(topk)
    prims_ts_sparse_prefill_rows(
        mla,
        q.view(num_tokens, heads, qk_dim),
        mla.k_b_proj_trans,
        mla.v_b_proj,
        topk,
        attn_metadata._cached_req_idx_ctx,
        attn_metadata,
        output.view(num_tokens, heads, mla.v_head_dim),
        kv_cache_format="nvfp4.fp8rope",
    )
    return output


__all__ = [
    "forward_context_sparse_attn_nvfp4_ds_mla",
    "SPARSE_MLA_PREFILL_ENV",
    "prims_ts_prefill_requested",
    "prims_ts_prefill_supported",
    "prims_ts_sparse_prefill_rows",
    "quantize_query_e4m3",
]
