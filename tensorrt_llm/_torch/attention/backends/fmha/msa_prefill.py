# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 context-phase FMHA backed by MSA's fmha_sm100 kernel.

MsaPrefillFmha wraps the fmha_sm100 paged sparse GQA kernel and participates
in the standard TrtllmAttention.forward dispatch loop. The owning MiniMax-M3
MSA attention layer runs an MsaIndexer to select the per-query KV blocks and
publishes them on forward_args.sparse_runtime_params; this class attends over
them.

It serves the context phase alone. fmha_sm100 schedules a generation row like
a context row, and a mixed batch cannot split the two apart (see
_mixed_batch_split in fmha_sm100/api.py), so the generation phase is
MsaDecodeFmha's outright.

Every import of the kernels below is function-local. This module is on the
import path of every attention.backends.trtllm import, and the minimax_m3
package init reaches msa_backend, which subclasses TrtllmAttention, so a
module-scope import here would close a cycle.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version, mpi_rank, prefer_pinned
from tensorrt_llm.logger import logger

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

# SM107: the NVFP4 sparse prefill's combine writes the layer's attention output buffer directly instead of a fresh
# tensor that is then copied over (same kernel, same values). Kill switch: False restores the copy.
_SM107_NVFP4_SPARSE_OUT_IN_PLACE = os.environ.get("TRTLLM_MSA_NVFP4_OUT_IN_PLACE", "1") == "1"

# FP8 dense prefill (MiniMax-M3's dense layers) runs FA4 (flash_attn.cute) over the paged FP8 cache in place of
# fmha_sm100, about 1.6x faster at long context. The image's FA4 carries rescale_threshold 0.0 for FP8, so the
# row max is refreshed on every increase and P stays <= 2^8 (FA4's FP8 offset), inside E4M3's 448, instead of the
# lazy rescale that lets P saturate. Deterministic, and inside fmha_sm100's own 16-seed fp64 error band (numerics
# rule 1b). An FA4 without that change is never used. Kill switch: TRTLLM_MSA_DISABLE_DENSE_FA4=1 restores
# fmha_sm100.
_DENSE_PREFILL_FA4 = os.environ.get("TRTLLM_MSA_DISABLE_DENSE_FA4", "0") != "1"
# The FA4 shape this path serves: TMA page loads need the page to equal FA4's 128-wide n tile.
_DENSE_FA4_HEAD_DIM = 128
_DENSE_FA4_PAGE_SIZE = 128
# FA4 loads whole V pages and masks only the scores, so P @ V still multiplies the rows past kv_len on a partial
# last page by zero: finite stale bytes add nothing, but an FP8 NaN byte (0x7F / 0xFF) left in a recycled page
# turns the tile's output NaN. Zeroing those rows first leaves every finite result bitwise unchanged. Kill switch:
# TRTLLM_MSA_DENSE_FA4_V_TAIL_ZERO=0.
_DENSE_FA4_V_TAIL_ZERO = os.environ.get("TRTLLM_MSA_DENSE_FA4_V_TAIL_ZERO", "1") != "0"
# What the patched flash_fwd_sm100 holds for the FP8 rescale threshold.
_DENSE_FA4_PATCH_MARKER = "0.0 if const_expr(self.q_dtype.width == 8) else"
# SM107: the FP8-KV prefill's E4M3 query is cast from the strided query rows of the fused QKV output by one Triton pass
# (row by row, 16-byte accesses) instead of the aten dtype-converting copy, whose strided path runs at ~1.2 TB/s. The
# cast rounds to nearest even and saturates to +-448: the bits of the aten cast wherever that is finite. Kill:
# TRTLLM_CTX_Q_E4M3_ROWS=0 in the deployment env (read once at import) restores the aten cast.
_SM107_Q_E4M3_ROWS = os.environ.get("TRTLLM_CTX_Q_E4M3_ROWS", "1") == "1"
_SM107_Q_E4M3_ROWS_LOGGED = []


@triton.jit
def _rows_to_e4m3_kernel(x_ptr, y_ptr, row_stride, width, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask)
    tl.store(y_ptr + row * width + cols, x.to(tl.float32).to(tl.float8e4nv), mask=mask)


def _q_to_e4m3(q: torch.Tensor) -> torch.Tensor:
    """``q.to(torch.float8_e4m3fn)`` (contiguous); BF16 ``[rows, heads, head_dim]`` views whose rows are
    contiguous take the Triton row cast."""
    if not (
        _SM107_Q_E4M3_ROWS
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and q.dim() == 3
        and q.stride(2) == 1
        and q.stride(1) == q.shape[2]
        and q.shape[0] > 0
        and get_sm_version() == 107
    ):
        return q.to(torch.float8_e4m3fn)
    rows, width = q.shape[0], q.shape[1] * q.shape[2]
    out = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    if not _SM107_Q_E4M3_ROWS_LOGGED:
        _SM107_Q_E4M3_ROWS_LOGGED.append(True)
        logger.info(f"[M3 dispatch] rank {mpi_rank()}: q_e4m3_rows taken at {rows} rows")
    block = 1024
    _rows_to_e4m3_kernel[(rows, triton.cdiv(width, block))](
        q, out, q.stride(0), width, BLOCK=block, num_warps=4
    )
    return out


if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def run_msa_sparse_gqa(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    kv_block_indexes: Optional[torch.Tensor] = None,
    *,
    kv_indices: torch.Tensor,
    sm_scale: float,
    qo_lens_cpu: Optional[torch.Tensor] = None,
    kv_lens_cpu: Optional[torch.Tensor] = None,
    qo_offset_cpu: Optional[torch.Tensor] = None,
    causal: bool = True,
    head_dim: Optional[int] = None,
    plan: Optional[tuple] = None,
    out: Optional[torch.Tensor] = None,
    use_fp8: bool = False,
) -> None:
    """Run fmha_sm100 paged GQA (plan/run split).

    `kv_block_indexes`: if set, sparse top-k mode (fixed `kv_block_num=topk`);
    if None, dense mode attending all pages in `kv_indices`.
    `plan`: prebuilt execution plan; if None, built inline from the CPU length
    tensors (the prebuilt plan for a step prepare() staged, inline for a test).
    `head_dim`: defaults to the only head dimension the kernel supports.
    `out`: destination buffer the kernel writes in place.
    `use_fp8`: FP8 KV cache. The caller must pass FP8 `q` to match the FP8 paged
    K/V, since the kernel variant shares one dtype across q/k/v. Also selects the
    FP8 AOT kernels for an inline sparse-prefill plan.
    """
    from ..sparse.minimax_m3.kernels.msa_utils import MSA_REQUIRED_HEAD_DIM, require_msa_module

    if head_dim is None:
        head_dim = MSA_REQUIRED_HEAD_DIM
    fmha_sm100 = require_msa_module()

    if q.dim() != 3:
        raise ValueError(
            f"MSA paged GQA expects q [total_q, num_qo_heads, head_dim]; got {tuple(q.shape)}."
        )
    if q.shape[-1] != head_dim:
        raise NotImplementedError(f"MSA paged GQA supports head_dim={head_dim}; got {q.shape[-1]}.")
    if k_paged.dim() != 4 or v_paged.dim() != 4:
        raise ValueError(
            "MSA paged GQA expects paged KV [num_pages, num_kv_heads, page_size, head_dim]; "
            f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
        )
    if k_paged.shape != v_paged.shape:
        raise ValueError(
            f"MSA paged GQA requires k and v to share shape; "
            f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
        )

    if plan is None:
        # kv_block_num is planned only for the sparse (block-indexed) path;
        # dense paged GQA leaves it unset and attends the full page table.
        kv_block_num = int(kv_block_indexes.shape[-1]) if kv_block_indexes is not None else -1
        plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            int(q.shape[1]),  # num query heads.
            num_kv_heads=int(k_paged.shape[1]),
            qo_offset=qo_offset_cpu,
            page_size=int(k_paged.shape[2]),
            kv_block_num=kv_block_num,
            causal=causal,
            num_kv_splits=1,
            use_fp8_kvcache=use_fp8,
        )
    fmha_sm100.fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        out=out,
        sm_scale=sm_scale,
        output_maxscore=False,
    )


def _aligned_nvfp4_dequant_scales(
    attn: "TrtllmAttention", kv_scale_quant_orig: torch.Tensor, *, refresh: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable, separately 16-byte-aligned K/V dequant scales.

    M3 checkpoints expose the Q/K/V scales as one contiguous three-float
    tensor.  Slicing elements 1 and 2 leaves addresses four and eight bytes
    past the allocation base, while CuTe DSL requires every tensor argument
    to start on a 16-byte boundary.  Keep one padded two-row buffer per
    layer-attention object. Populate it during eager warmup, and refresh its
    contents after weight loading without changing captured tensor addresses.
    """
    if kv_scale_quant_orig.dtype != torch.float32 or kv_scale_quant_orig.numel() < 3:
        raise ValueError("MiniMax-M3 NVFP4 dequantization scales must be FP32 [Q, K, V]")

    cache = getattr(attn, "_msa_nvfp4_dequant_scales", None)
    source_ptr = int(kv_scale_quant_orig.data_ptr())
    if cache is None:
        if kv_scale_quant_orig.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MiniMax-M3 NVFP4 scale alignment buffer must be initialized during eager warmup"
            )
        cache = torch.empty((2, 4), dtype=torch.float32, device=kv_scale_quant_orig.device)
        attn._msa_nvfp4_dequant_scales = cache
        refresh = True
    if cache.device != kv_scale_quant_orig.device:
        raise ValueError("MiniMax-M3 NVFP4 scale reload must preserve the scale buffer device")
    if refresh or getattr(attn, "_msa_nvfp4_dequant_scale_source_ptr", None) != source_ptr:
        cache[:, 0].copy_(kv_scale_quant_orig[1:3])
        attn._msa_nvfp4_dequant_scale_source_ptr = source_ptr

    k_global_scale = cache[0, :1]
    v_global_scale = cache[1, :1]
    assert k_global_scale.data_ptr() % 16 == 0
    assert v_global_scale.data_ptr() % 16 == 0
    return k_global_scale, v_global_scale


def _nvfp4_sparse_out_kwargs(sparse: Any, out: torch.Tensor) -> dict[str, torch.Tensor]:
    """``out=`` for sparse_atten_nvfp4_kv_func when its combine can write ``out`` in place (see the kill switch)."""
    if (
        not _SM107_NVFP4_SPARSE_OUT_IN_PLACE
        or get_sm_version() != 107
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or not _nvfp4_sparse_takes_out(sparse)
    ):
        return {}
    logger.info_once(
        "MSA NVFP4 sparse prefill: combine writes the attention output in place (SM107)",
        key="minimax_m3_sm107_nvfp4_sparse_out_in_place",
    )
    return {"out": out}


def _nvfp4_sparse_takes_out(sparse: Any) -> bool:
    """Whether the installed fmha_sm100 sparse_atten_nvfp4_kv_func accepts ``out`` (resolved once)."""
    takes = getattr(_nvfp4_sparse_takes_out, "_cached", None)
    if takes is None:
        import inspect

        takes = "out" in inspect.signature(sparse.sparse_atten_nvfp4_kv_func).parameters
        _nvfp4_sparse_takes_out._cached = takes
    return takes


# Every sparse layer of a step builds its k2q CSR with the same sizes, so the
# non-identity build keeps one set of buffers keyed on the full step shape (reallocated when any size
# changes) and calls the fmha_sm100 extension directly, skipping the two Python validation layers and
# their six allocations. Same kernels in the same order on the same stream, and the stock builder also
# hands the kernels torch.empty buffers, so the CSR and schedule are unchanged. Eager only: a capturing
# stream takes the stock builder, and so does any stream other than the one the buffers were first built on
# (stream order is what keeps a step's later build from overwriting buffers an earlier attention still reads).
# SM107 only. Kill switch: TRTLLM_MSA_K2Q_WS=0.
_K2Q_WS = os.environ.get("TRTLLM_MSA_K2Q_WS", "1") != "0"
_K2Q_WS_BLK_KV = 128
_K2Q_WS_TOPK = (4, 8, 16, 32)
_k2q_ws: Optional[tuple] = None  # (key, q2k, row_ptr, q_idx, schedule)
_k2q_ws_stream: Optional[int] = None  # the CUDA stream the buffers belong to


def _k2q_csr_ws(
    kv_block_indexes: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    page_size: int,
    *,
    max_seqlen_k: int,
    max_seqlen_q: int,
    total_rows: int,
    qhead_per_kv: int,
) -> Optional[tuple]:
    """(k2q_row_ptr, k2q_q_indices, schedule) from per-shape buffers, or None for the stock builder."""
    global _k2q_ws, _k2q_ws_stream
    total_q, head_kv, topk = (int(v) for v in kv_block_indexes.shape)
    if (
        not _K2Q_WS
        or get_sm_version() != 107
        or page_size != _K2Q_WS_BLK_KV
        or topk not in _K2Q_WS_TOPK
        or kv_block_indexes.dtype != torch.int32
        or total_q == 0
        or head_kv == 0
        or total_rows <= 0
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    import sys

    utils = sys.modules.get("sparse_index_utils")
    sched_mod = sys.modules.get("src.sm100.prepare_scheduler")
    if utils is None or sched_mod is None:
        return None
    stream = torch.cuda.current_stream(kv_block_indexes.device).cuda_stream
    if _k2q_ws_stream is not None and stream != _k2q_ws_stream:
        return None
    builder = utils._K2Q_CSR_BUILDER
    max_kv_blocks = (max(max_seqlen_k, _K2Q_WS_BLK_KV) + _K2Q_WS_BLK_KV - 1) // _K2Q_WS_BLK_KV
    device = kv_block_indexes.device
    key = (
        device.index,
        head_kv,
        total_q,
        topk,
        total_rows,
        max_kv_blocks,
        max_seqlen_q,
        qhead_per_kv,
    )
    if _k2q_ws is None or _k2q_ws[0] != key:
        model = sched_mod.SPARSE_SCHEDULE_MODEL
        target = model.balanced_target_q_per_cta(
            total_q=total_q,
            topk=topk,
            blk_kv=_K2Q_WS_BLK_KV,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            device=device,
        )
        capacity = model.flat_schedule_capacity(
            total_rows=total_rows,
            total_q=total_q,
            topk=topk,
            head_kv=head_kv,
            target_q_per_cta=target,
        )
        q_idx = torch.empty((head_kv, total_q * topk), dtype=torch.int32, device=device)
        schedule = sched_mod.SparseAttentionSchedule(
            enabled=True,
            scheduler_metadata=torch.empty((capacity, 6), dtype=torch.int32, device=device),
            work_count=torch.empty((1,), dtype=torch.int32, device=device),
            qsplit_indices=torch.empty_like(q_idx),
            split_counts=torch.empty((total_q, head_kv), dtype=torch.int32, device=device),
            target_q_per_cta=target,
        )
        q2k = torch.empty((head_kv, total_q, topk), dtype=torch.int32, device=device)
        row_ptr = torch.empty((head_kv, total_rows + 1), dtype=torch.int32, device=device)
        _k2q_ws = (key, q2k, row_ptr, q_idx, schedule)
        _k2q_ws_stream = stream
    _, q2k_buf, row_ptr, q_idx, schedule = _k2q_ws
    # The indexer's storage is already head-major on pure-context steps, where the stock .contiguous() is a
    # no-op; only a strided (mixed-step) view is copied, into the reused buffer.
    q2k = kv_block_indexes.permute(1, 0, 2)
    if not q2k.is_contiguous():
        q2k = q2k_buf.copy_(q2k)
    builder._ensure_loaded()
    with torch.cuda.nvtx.range("SparseK2qCsr_Pipeline"):
        builder._run_with_schedule(
            q2k,
            cu_q,
            cu_kv,
            row_ptr,
            q_idx,
            schedule.scheduler_metadata,
            schedule.work_count,
            schedule.qsplit_indices,
            schedule.split_counts,
            topk,
            _K2Q_WS_BLK_KV,
            total_rows,
            max_kv_blocks,
            schedule.target_q_per_cta,
            schedule.work_capacity,
            max_seqlen_q,
        )
    logger.info_once(
        "MSA NVFP4 sparse prefill: k2q CSR build reuses per-shape buffers (TRTLLM_MSA_K2Q_WS)",
        key="minimax_m3_msa_k2q_ws",
    )
    return row_ptr, q_idx, schedule


def run_msa_nvfp4_sparse_gqa(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    scale_buffers: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    metadata: "TrtllmAttentionMetadata",
    *,
    sm_scale: float,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    num_rows: Optional[int] = None,
    out: torch.Tensor,
) -> None:
    """Run sparse attention over M3's packed NVFP4 cache with the MSA CSR kernel.

    This is the prefill-shaped path: it builds a k2q CSR worklist per call and
    tiles 128 query rows at a time. num_rows narrows it to the leading batch
    rows, so a mixed step can hand its generation suffix to the Triton NVFP4
    decode and size this worklist from the context prefix alone.
    """
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        MSA_REQUIRED_TOPK,
        require_msa_module,
    )

    fmha_sm100 = require_msa_module()
    sparse = getattr(fmha_sm100, "sparse", None)
    if sparse is None:
        try:
            from fmha_sm100 import sparse
        except ImportError as exc:
            raise RuntimeError("MiniMax-M3 NVFP4 KV cache requires the Fan MSA sparse API") from exc
    if not hasattr(sparse, "build_k2q_csr") or not hasattr(sparse, "sparse_atten_nvfp4_kv_func"):
        raise RuntimeError(
            "The loaded MSA build lacks build_k2q_csr or "
            "sparse_atten_nvfp4_kv_func; use the NVFP4-capable Fan revision"
        )

    for name, scale in (("K", k_global_scale), ("V", v_global_scale)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise ValueError(f"MiniMax-M3 NVFP4 {name} dequantization scale must be one FP32 value")
        if scale.data_ptr() % 16 != 0:
            raise ValueError(
                f"MiniMax-M3 NVFP4 {name} dequantization scale must be 16-byte aligned"
            )
    if scale_buffers.shape[:2] != k_paged.shape[:1] + (2,):
        raise ValueError(
            "MiniMax-M3 NVFP4 scale buffers must be [pages, 2, heads, page, D/16]; "
            f"got {tuple(scale_buffers.shape)} for K {tuple(k_paged.shape)}"
        )

    batch = int(getattr(metadata, "_msa_live_batch", 0))
    if batch <= 0:
        raise RuntimeError("MiniMax-M3 NVFP4 sparse attention metadata was not prepared")
    max_q_len = int(metadata._msa_max_q_len)
    max_kv_len = int(metadata._msa_max_kv_len_all)
    total_k = int(metadata._msa_total_k)
    total_k_rows = int(metadata._msa_total_k_rows)
    if num_rows is not None:
        if not 0 < num_rows <= batch:
            raise ValueError(f"MiniMax-M3 NVFP4 row limit ({num_rows}) must lie in (0, {batch}].")
        batch = num_rows
        # Staged by prepare() over the context rows, which are exactly the rows
        # a narrowed call covers; see _stage_step_fields.
        max_q_len, max_kv_len, total_k, total_k_rows = metadata._msa_context_prefix_bounds
        if total_k_rows <= 0:
            raise RuntimeError(
                "MiniMax-M3 NVFP4 attention was narrowed to a context prefix of "
                f"{num_rows} row(s), but prepare() staged no bounds for one."
            )
    cu_q = metadata.msa_cu_q_lens[: batch + 1]
    cu_kv = metadata.msa_cu_kv_lens[: batch + 1]
    topk = int(kv_block_indexes.shape[-1])
    page_size = int(k_paged.shape[2])
    if topk != MSA_REQUIRED_TOPK:
        raise ValueError(f"MiniMax-M3 MSA NVFP4 requires topK={MSA_REQUIRED_TOPK}, got {topk}")

    # A short pure-prefill step hands every layer the same identity selection (staged once per step by the
    # backend), so its k2q CSR worklist is layer-invariant: build it for the first layer and reuse it.
    identity = metadata._msa_prefill_identity
    csr_key = (batch, int(kv_block_indexes.shape[0]))
    cached = metadata._msa_identity_csr
    if (
        identity is not None
        and cached is not None
        and cached[0] is identity
        and cached[1] == csr_key
    ):
        k2q_row_ptr, k2q_q_indices, schedule = cached[2]
    elif (
        identity is None
        and (
            built := _k2q_csr_ws(
                kv_block_indexes,
                cu_q,
                cu_kv,
                page_size,
                max_seqlen_k=max_kv_len,
                max_seqlen_q=max_q_len,
                total_rows=total_k_rows,
                qhead_per_kv=int(q.shape[1]) // int(k_paged.shape[1]),
            )
        )
        is not None
    ):
        k2q_row_ptr, k2q_q_indices, schedule = built
    else:
        q2k = kv_block_indexes.permute(1, 0, 2).contiguous()
        k2q_row_ptr, k2q_q_indices, schedule = sparse.build_k2q_csr(
            q2k,
            cu_q,
            cu_kv,
            page_size,
            total_k=total_k,
            max_seqlen_k=max_kv_len,
            max_seqlen_q=max_q_len,
            total_rows=total_k_rows,
            qhead_per_kv=int(q.shape[1]) // int(k_paged.shape[1]),
            return_schedule=True,
        )
        if identity is not None:
            metadata._msa_identity_csr = (identity, csr_key, (k2q_row_ptr, k2q_q_indices, schedule))
    from .msa_combine_mxfp8 import combine_kwargs, publish

    out_kwargs = _nvfp4_sparse_out_kwargs(sparse, out)
    # Host CTX (b): the combine also writes the o_proj input as MXFP8 when the attention module armed it.
    mxfp8_kwargs = combine_kwargs(sparse, out, metadata, in_place=bool(out_kwargs))
    result = sparse.sparse_atten_nvfp4_kv_func(
        q,
        k_paged.view(torch.uint8),
        v_paged.view(torch.uint8),
        scale_buffers[:, 0].view(torch.uint8),
        scale_buffers[:, 1].view(torch.uint8),
        k_global_scale,
        v_global_scale,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        max_seqlen_q=max_q_len,
        max_seqlen_k=max_kv_len,
        blk_kv=page_size,
        causal=True,
        softmax_scale=sm_scale,
        partial_dtype=torch.bfloat16,
        return_softmax_lse=False,
        page_table=metadata.msa_block_table[:batch],
        seqused_k=metadata.msa_seq_lens_cuda[:batch],
        schedule=schedule,
        **out_kwargs,
        **(mxfp8_kwargs or {}),
    )
    if result is not out:
        out.copy_(result)
    if mxfp8_kwargs is not None:
        publish(out, mxfp8_kwargs)


@functools.lru_cache(maxsize=None)
def _dense_fa4_forward():
    """Return FA4's forward entry point, or None where the path is off or the FA4 lacks the FP8 rescale change."""
    if not _DENSE_PREFILL_FA4:
        return None
    try:
        import inspect

        from ..sparse.minimax_m3.kernels.msa_utils import _install_msa_cutlass_compatibility

        # flash_attn.cute (via quack) still names cute.core.ThrMma / ThrCopy, which this CUTLASS DSL moved to the
        # top level. The MSA import path installs those aliases; install them here too, so the import does not
        # depend on fmha_sm100 having been loaded first.
        _install_msa_cutlass_compatibility()
        from flash_attn.cute import flash_fwd_sm100
        from flash_attn.cute.interface import _flash_attn_fwd
    except Exception as exc:  # noqa: BLE001 - any import failure keeps fmha_sm100
        logger.warning(f"MiniMax-M3 dense prefill: FA4 unavailable ({exc!r}); keeping fmha_sm100")
        return None
    if _DENSE_FA4_PATCH_MARKER not in inspect.getsource(flash_fwd_sm100):
        logger.warning(
            "MiniMax-M3 dense prefill: FA4 lacks the FP8 rescale_threshold 0.0 change; keeping fmha_sm100"
        )
        return None
    return _flash_attn_fwd


def dense_prefill_fa4_enabled(use_fp8: bool, head_dim: int, page_size: int) -> bool:
    """Whether FP8 dense prefill runs FA4 rather than fmha_sm100.

    prepare() asks too, to skip the dense fmha_sm100 plan the FA4 path never reads.
    """
    return (
        use_fp8
        and head_dim == _DENSE_FA4_HEAD_DIM
        and page_size == _DENSE_FA4_PAGE_SIZE
        # Rubin only: FA4 rounds differently from fmha_sm100 (deterministic, not bitwise), so other SM10x keep it.
        and get_sm_version() == 107
        and _dense_fa4_forward() is not None
    )


def _v_tail_zero_safe(kv_cache_manager: Any) -> bool:
    """Whether every row's partial last page is its own, so zeroing past kv_len cannot touch another row's tokens.

    True without block reuse or partial reuse, or when a partially matched block is copied on reuse:
    KVCacheManagerV2 always copies it, and KVCacheManager does with copy_on_partial_reuse.
    """
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    if not getattr(kv_cache_manager, "enable_block_reuse", True) or not getattr(
        kv_cache_manager, "enable_partial_reuse", True
    ):
        return True
    if isinstance(kv_cache_manager, KVCacheManagerV2):
        return True
    return bool(getattr(kv_cache_manager, "copy_on_partial_reuse", False))


def _zero_partial_page_v_tail(
    v_paged: torch.Tensor, page_table: torch.Tensor, kv_rows: torch.Tensor
) -> None:
    """Zero the V rows at and past each row's kv_len on its partial last page.

    v_paged is the cache's HND V [pages, heads, page, d]; page_table the rows' block table; kv_rows the host
    kv_len per row. Those slots hold no token of the row yet and are rewritten when the sequence grows into them.
    """
    page_size = int(v_paged.shape[2])
    tails = [(row, n) for row, n in enumerate(kv_rows.tolist()) if n % page_size]
    if not tails:
        return
    host = torch.empty(3, len(tails), dtype=torch.int64, pin_memory=prefer_pinned())
    for i, (row, n) in enumerate(tails):
        host[0, i] = row
        host[1, i] = n // page_size
        host[2, i] = n % page_size
    dev = host.to(v_paged.device, non_blocking=True)
    pages = page_table[dev[0], dev[1]].long()
    v_bytes = v_paged.view(torch.uint8)
    tail_pages = v_bytes[pages]
    past = torch.arange(page_size, device=v_paged.device)[None, :] >= dev[2][:, None]
    tail_pages.masked_fill_(past[:, None, :, None], 0)
    v_bytes[pages] = tail_pages


_dense_fa4_warm: set = set()


def _run_dense_fa4(
    q_view: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    metadata: "TrtllmAttentionMetadata",
    *,
    row_first: int,
    num_rows: int,
    sm_scale: float,
    out_view: torch.Tensor,
) -> bool:
    """Run FP8 dense causal prefill with FA4 into out_view; False (nothing run) where it cannot serve the rows.

    K/V are the cache's HND pages viewed as FA4's (pages, page, heads, d), so nothing is gathered. Each row
    attends its first kv_len tokens through the 2-D block table, with its queries the last qo_len of them
    (FA4's bottom-right causal, the same alignment as the qo_offset = kv_len - qo_len the plans take).
    """
    qo_lens = metadata.msa_qo_lens_cpu
    kv_lens = metadata.msa_kv_lens_cpu
    if qo_lens is None or kv_lens is None or out_view.dtype != torch.bfloat16:
        return False
    if _DENSE_FA4_V_TAIL_ZERO and not _v_tail_zero_safe(metadata.kv_cache_manager):
        logger.warning_once(
            "MiniMax-M3 dense prefill: FA4 needs a private partial last page to zero its V tail, but this KV cache "
            "can share a partially reused page without copying it; keeping fmha_sm100 "
            "(TRTLLM_MSA_DENSE_FA4_V_TAIL_ZERO=0 runs FA4 without the zeroing)",
            key="minimax_m3_dense_prefill_fa4_v_tail_unsafe",
        )
        return False
    qo_rows = qo_lens[row_first : row_first + num_rows]
    kv_rows = kv_lens[row_first : row_first + num_rows]
    # One pinned host staging for both length tensors: cu_seqlens_q then seqused_k.
    host = torch.empty(2 * num_rows + 1, dtype=torch.int32, pin_memory=prefer_pinned())
    host[0] = 0
    torch.cumsum(qo_rows, 0, dtype=torch.int32, out=host[1 : num_rows + 1])
    if int(host[num_rows]) != int(q_view.shape[0]):
        return False
    host[num_rows + 1 :].copy_(kv_rows)
    lens = host.to(q_view.device, non_blocking=True)
    max_seqlen_q = int(qo_rows.max())
    fwd = _dense_fa4_forward()
    k = k_paged.permute(0, 2, 1, 3)
    v = v_paged.permute(0, 2, 1, 3)
    num_kv_heads = int(k_paged.shape[1])
    warm_key = (int(q_view.shape[1]), num_kv_heads, q_view.device.index)
    if warm_key not in _dense_fa4_warm:
        # FA4 compiles per variant on first use, and the variant depends on whether a row's packed queries
        # (max_seqlen_q x the GQA group) fill more than one 128-row tile. The first call is the server warmup,
        # so compile both there, on one-token requests over the first page, rather than in a served step.
        group = int(q_view.shape[1]) // num_kv_heads
        for rows_q in (1, 128 // group + 1):
            fwd(
                q_view.new_zeros(rows_q, q_view.shape[1], q_view.shape[2]),
                k,
                v,
                cu_seqlens_q=torch.tensor([0, rows_q], dtype=torch.int32, device=q_view.device),
                seqused_k=torch.tensor([rows_q], dtype=torch.int32, device=q_view.device),
                max_seqlen_q=rows_q,
                page_table=torch.zeros(1, 1, dtype=torch.int32, device=q_view.device),
                softmax_scale=sm_scale,
                causal=True,
            )
        _dense_fa4_warm.add(warm_key)
    logger.info_once(
        "MiniMax-M3 dense prefill: FA4 (flash_attn.cute, FP8 P without lazy rescale) over the paged FP8 KV cache",
        key="minimax_m3_dense_prefill_fa4",
    )
    page_table = metadata.msa_block_table[row_first : row_first + num_rows]
    if _DENSE_FA4_V_TAIL_ZERO:
        _zero_partial_page_v_tail(v_paged, page_table, kv_rows)
        logger.info_once(
            "MiniMax-M3 dense prefill: FA4 V tail past kv_len zeroed on partial pages",
            key="minimax_m3_dense_prefill_fa4_v_tail",
        )
    fwd(
        q_view,
        k,
        v,
        cu_seqlens_q=lens[: num_rows + 1],
        seqused_k=lens[num_rows + 1 :],
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=sm_scale,
        causal=True,
        out=out_view,
    )
    return True


def run_msa_prefill_gqa(
    attn: "TrtllmAttention",
    q: torch.Tensor,
    metadata: "TrtllmAttentionMetadata",
    output: torch.Tensor,
    *,
    kv_block_indexes: Optional[torch.Tensor],
    plan: Optional[tuple],
    row_first: int,
    num_rows: int,
    kv_scale_quant_orig: Optional[torch.Tensor] = None,
) -> None:
    """Run paged GQA over one row range into output in place.

    Shared by the sparse layers (kv_block_indexes is the per-query top-k table
    for these rows, with the sparse plan) and the dense layers
    (kv_block_indexes None, with the dense plan, attending the full page
    table).

    `q` and `output` are already the phase's token slice. `row_first` and
    `num_rows` are its batch rows, which narrow the host length tensors an
    inline plan would read.
    """
    from ..sparse.minimax_m3.kernels.msa_utils import msa_paged_kv

    head_dim = attn.head_dim
    num_tokens = int(q.shape[0])
    if num_tokens == 0:
        return
    q_view = q.view(num_tokens, attn.num_heads, head_dim)
    out_view = output.view(num_tokens, attn.num_heads, head_dim)
    k_paged, v_paged = msa_paged_kv(metadata.kv_cache_manager, attn.layer_idx)
    sm_scale = (head_dim**-0.5) / float(attn.q_scaling)

    if getattr(metadata.kv_cache_manager, "is_nvfp4_layer", lambda _: False)(attn.layer_idx):
        if kv_block_indexes is None or kv_scale_quant_orig is None:
            raise RuntimeError(
                "NVFP4 sparse prefill requires selected blocks and dequantization scales"
            )
        if row_first != 0:
            raise ValueError("NVFP4 sparse prefill must cover the context prefix")
        k_scale, v_scale = _aligned_nvfp4_dequant_scales(attn, kv_scale_quant_orig)
        run_msa_nvfp4_sparse_gqa(
            q_view,
            k_paged,
            v_paged,
            metadata.kv_cache_manager.get_block_scale_buffers(attn.layer_idx, "HND"),
            kv_block_indexes,
            metadata,
            sm_scale=sm_scale,
            k_global_scale=k_scale,
            v_global_scale=v_scale,
            num_rows=num_rows,
            out=out_view,
        )
        return

    # The fmha_sm100 variant is chosen from q.dtype and shares one dtype across
    # q/k/v, so q must be FP8 to match an FP8 paged K/V. MiniMax-M3 has no
    # KV-cache scales, so the scale is 1.0 and this is a plain E4M3 cast.
    use_fp8 = k_paged.dtype == torch.float8_e4m3fn
    if use_fp8:
        q_view = _q_to_e4m3(q_view)

    def rows_of(lens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Narrow a per-request host length tensor to this phase's rows.

        Slicing keeps the pinned backing, so an inline plan still stages these
        with non-blocking copies.
        """
        if lens is None:
            return None
        return lens[row_first : row_first + num_rows]

    if (
        kv_block_indexes is None
        and dense_prefill_fa4_enabled(use_fp8, head_dim, int(k_paged.shape[2]))
        and _run_dense_fa4(
            q_view,
            k_paged,
            v_paged,
            metadata,
            row_first=row_first,
            num_rows=num_rows,
            sm_scale=sm_scale,
            out_view=out_view,
        )
    ):
        return

    run_msa_sparse_gqa(
        q_view,
        k_paged,
        v_paged,
        kv_block_indexes,
        kv_indices=metadata.msa_kv_indices,
        sm_scale=sm_scale,
        qo_lens_cpu=rows_of(metadata.msa_qo_lens_cpu),
        kv_lens_cpu=rows_of(metadata.msa_kv_lens_cpu),
        qo_offset_cpu=rows_of(metadata.msa_qo_offset_cpu),
        causal=True,
        head_dim=head_dim,
        plan=plan,
        out=out_view,
        use_fp8=use_fp8,
    )


class MsaPrefillFmha(PhasedFmha):
    """SM100 paged GQA FMHA powered by MSA's fmha_sm100 kernel.

    Handles the context phase of every MiniMax-M3 MSA layer. Sparse layers pass
    the indexer's selected KV block indices on
    forward_args.sparse_runtime_params.sparse_attn_indices and attend those
    blocks; dense layers leave the indices None and attend the full page table.
    Requires head_dim 128 and 4-D HND paged K/V.

    The generation phase is MsaDecodeFmha's, so run_generation is left to the
    base class, which refuses it.
    """

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        from ..sparse.minimax_m3.kernels.msa_utils import is_msa_layer

        return is_msa_layer(attn)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        # A step's context rows are this library's and its generation rows
        # MsaDecodeFmha's, whatever the step looks like, so each library claims
        # its one phase and their partition of the phases is total. The
        # phase-less query asks for the whole step, which neither can serve
        # alone: a mixed step belongs to the two together through CombinedFmha.
        return phase is FmhaPhase.CONTEXT

    def run_context(self, params: FmhaParams) -> None:
        from ..sparse.minimax_m3.kernels.msa_utils import write_msa_phase_kv

        metadata = params.meta
        write_msa_phase_kv(
            params.attn,
            params.key_input,
            params.value_input,
            metadata,
            params.fwd.attention_input_type,
            token_offset=params.token_offset,
        )
        # Sparse layers attend the per-query top-k blocks with the sparse plan;
        # dense layers leave the indices None and attend the full page table
        # with the dense plan.
        kv_block_indexes = params.fwd.sparse_runtime_params.sparse_attn_indices
        is_sparse_layer = kv_block_indexes is not None
        if is_sparse_layer:
            kv_block_indexes = kv_block_indexes[
                params.token_offset : params.token_offset + params.num_tokens
            ]
        run_msa_prefill_gqa(
            params.attn,
            params.query_input,
            metadata,
            params.output,
            kv_block_indexes=kv_block_indexes,
            plan=(
                metadata.msa_prefill_gqa_plan
                if is_sparse_layer
                else metadata.msa_prefill_dense_plan
            ),
            row_first=params.seq_offset,
            num_rows=metadata.num_contexts,
            kv_scale_quant_orig=params.fwd.kv_scale_quant_orig,
        )


__all__ = [
    "MsaPrefillFmha",
    "dense_prefill_fa4_enabled",
    "run_msa_prefill_gqa",
    "run_msa_sparse_gqa",
]
