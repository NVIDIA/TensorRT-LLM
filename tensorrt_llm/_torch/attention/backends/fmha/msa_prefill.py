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
import inspect
import sys
from typing import TYPE_CHECKING, Callable, Optional

import torch

from tensorrt_llm.logger import logger

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

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


@functools.lru_cache(maxsize=1)
def _msa_sparse_api():
    """Resolve fmha_sm100's sparse API once; the package cannot change under a run."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        require_msa_module,
    )

    fmha_sm100 = require_msa_module()
    sparse = getattr(fmha_sm100, "sparse", None)
    if sparse is None:
        try:
            from .msa_kernels.fmha_sm100 import sparse
        except ImportError as exc:
            raise RuntimeError("MiniMax-M3 NVFP4 KV cache requires the Fan MSA sparse API") from exc
    if not hasattr(sparse, "build_k2q_csr") or not hasattr(sparse, "sparse_atten_nvfp4_kv_func"):
        raise RuntimeError(
            "The loaded MSA build lacks build_k2q_csr or "
            "sparse_atten_nvfp4_kv_func; use the NVFP4-capable Fan revision"
        )
    return sparse


@functools.lru_cache(maxsize=1)
def _k2q_csr_parts():
    """(run_with_schedule, schedule model, schedule type) behind sparse.build_k2q_csr, or None.

    Internals of the packaged MSA builder, resolved once; a build that moves
    them leaves every call on build_k2q_csr itself.
    """
    _msa_sparse_api()  # puts the CuTe-DSL sources on sys.path
    try:
        from sparse_index_utils import _K2Q_CSR_BUILDER
        from src.sm100.prepare_scheduler import SPARSE_SCHEDULE_MODEL, SparseAttentionSchedule

        _K2Q_CSR_BUILDER._ensure_loaded()
        run = _K2Q_CSR_BUILDER._run_with_schedule
    except (ImportError, AttributeError):
        return None
    if run is None:
        return None
    return run, SPARSE_SCHEDULE_MODEL, SparseAttentionSchedule


def _k2q_csr_with_schedule(
    sparse,
    q2k: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    page_size: int,
    *,
    total_k: int,
    max_seqlen_k: int,
    max_seqlen_q: int,
    total_rows: int,
    qhead_per_kv: int,
    metadata,
) -> tuple:
    """sparse.build_k2q_csr(..., return_schedule=True), into buffers reused across layers.

    The CSR, schedule and split buffers and the schedule sizing depend only on
    the step's geometry, so they are sized once and kept on the metadata; the
    builder kernels write them whole on every call, as they do the fresh
    torch.empty buffers build_k2q_csr allocates. Same kernels, same arguments,
    so the same bytes. Each layer's attention reads them on the stream before
    the next layer's build rewrites them.
    """
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
        PREFILL_HOST_TRIMS,
    )

    head_kv, total_q, topk = (int(x) for x in q2k.shape)
    parts = _k2q_csr_parts() if PREFILL_HOST_TRIMS else None
    if (
        parts is None
        or page_size != 128
        or topk not in (4, 8, 16, 32)
        or total_rows <= 0
        or total_q <= 0
        or head_kv <= 0
        or not q2k.is_contiguous()
    ):
        return sparse.build_k2q_csr(
            q2k,
            cu_q,
            cu_kv,
            page_size,
            total_k=total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            total_rows=total_rows,
            qhead_per_kv=qhead_per_kv,
            return_schedule=True,
        )
    run, model, schedule_type = parts
    device = q2k.device
    key = (device, head_kv, total_q, topk, total_rows, qhead_per_kv)
    cached = getattr(metadata, "_msa_csr_workspace", None)
    if cached is None or cached[0] != key:
        target_q_per_cta = model.balanced_target_q_per_cta(
            total_q=total_q,
            topk=topk,
            blk_kv=page_size,
            head_kv=head_kv,
            qhead_per_kv=qhead_per_kv,
            device=device,
        )
        capacity = model.flat_schedule_capacity(
            total_rows=total_rows,
            total_q=total_q,
            topk=topk,
            head_kv=head_kv,
            target_q_per_cta=target_q_per_cta,
        )
        q_indices = torch.empty((head_kv, total_q * topk), dtype=torch.int32, device=device)
        schedule = schedule_type(
            enabled=True,
            scheduler_metadata=torch.empty((capacity, 6), dtype=torch.int32, device=device),
            work_count=torch.empty((1,), dtype=torch.int32, device=device),
            qsplit_indices=torch.empty_like(q_indices),
            split_counts=torch.empty((total_q, head_kv), dtype=torch.int32, device=device),
            target_q_per_cta=target_q_per_cta,
        )
        row_ptr = torch.empty((head_kv, total_rows + 1), dtype=torch.int32, device=device)
        cached = (key, row_ptr, q_indices, schedule)
        metadata._msa_csr_workspace = cached
    _, row_ptr, q_indices, schedule = cached
    max_kv_blocks = (max(int(max_seqlen_k), page_size) + page_size - 1) // page_size
    run(
        q2k,
        cu_q,
        cu_kv,
        row_ptr,
        q_indices,
        schedule.scheduler_metadata,
        schedule.work_count,
        schedule.qsplit_indices,
        schedule.split_counts,
        topk,
        page_size,
        total_rows,
        max_kv_blocks,
        schedule.target_q_per_cta,
        schedule.work_capacity,
        int(max_seqlen_q),
    )
    return row_ptr, q_indices, schedule


@functools.lru_cache(maxsize=1)
def _nvfp4_attention_parts():
    """(validate inputs, validate schedule, forward launch, combine) behind
    sparse.sparse_atten_nvfp4_kv_func, or None.

    Internals of the packaged MSA entry point, resolved once; a build that
    moves them leaves every call on sparse_atten_nvfp4_kv_func itself.
    """
    _msa_sparse_api()  # puts the CuTe-DSL sources on sys.path
    interface = sys.modules.get("interface")
    names = (
        "_validate_csr_varlen_nvfp4_kv_inputs",
        "_validate_fwd_schedule",
        "_call_sparse_forward_sm100_csr_varlen_nvfp4_kv",
        "combine",
    )
    if interface is None or not all(callable(getattr(interface, name, None)) for name in names):
        return None
    return tuple(getattr(interface, name) for name in names)


def _tensor_geometry(tensor: Optional[torch.Tensor]) -> Optional[tuple]:
    if tensor is None:
        return None
    return (tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.is_contiguous())


def _nvfp4_sparse_attention_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    schedule,
    topk: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    blk_kv: int,
    softmax_scale: float,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    metadata,
) -> bool:
    """sparse_atten_nvfp4_kv_func(..., causal=True) with the result combined straight into `out`.

    The packaged entry point validates every input and allocates an output
    the caller then copies into place, on every layer. The inputs' geometry is
    fixed for a step, so this validates it once per geometry (recorded on the
    metadata) and hands `out` to the combine kernel as its destination. Same
    two kernels on the same arguments, so the same bytes.
    Returns False, having done nothing, where it does not apply.
    """
    parts = _nvfp4_attention_parts()
    if (
        parts is None
        or schedule is None
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or tuple(out.shape) != tuple(q.shape)
    ):
        return False
    validate_inputs, validate_schedule, launch_forward, combine = parts
    total_q, head_q, head_dim = (int(x) for x in q.shape)
    key = (
        q.device,
        topk,
        int(blk_kv),
        tuple(
            _tensor_geometry(t)
            for t in (
                q,
                k,
                v,
                k_scale,
                v_scale,
                k_global_scale,
                v_global_scale,
                k2q_row_ptr,
                k2q_q_indices,
                schedule.qsplit_indices,
                schedule.split_counts,
                cu_seqlens_q,
                cu_seqlens_k,
                page_table,
                seqused_k,
            )
        ),
    )
    cached = getattr(metadata, "_msa_nvfp4_attn_validated", None)
    if cached is None or cached[0] != key:
        _, head_kv = validate_inputs(
            q,
            k,
            v,
            k_scale,
            v_scale,
            k_global_scale,
            v_global_scale,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            blk_kv,
            page_table,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
        )
        validate_schedule(schedule, q=q, k2q_q_indices=k2q_q_indices, head_kv=head_kv)
        cached = (key, int(head_kv))
        metadata._msa_nvfp4_attn_validated = cached
    head_kv = cached[1]
    # Allocated per call as the entry point does: a long prefill's partials run
    # to gigabytes, which the layers between two attentions need back.
    device = q.device
    o_partial = torch.empty((topk, total_q, head_q, head_dim), dtype=torch.bfloat16, device=device)
    lse_partial = torch.empty((topk, total_q, head_q), dtype=torch.float32, device=device)
    lse_out = torch.empty((total_q, head_q), dtype=torch.float32, device=device)
    # The packaged entry point makes these contiguous on the way in (each a
    # no-op on the views production passes); so does this.
    launch_forward(
        q.contiguous(),
        k,
        v,
        k_scale,
        v_scale,
        k_global_scale.contiguous(),
        v_global_scale.contiguous(),
        k2q_row_ptr.contiguous(),
        k2q_q_indices.contiguous(),
        schedule.qsplit_indices.contiguous(),
        schedule.split_counts.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        page_table.contiguous(),
        seqused_k.contiguous(),
        o_partial,
        lse_partial,
        None,
        float(softmax_scale),
        1.0,
        False,
        int(k2q_row_ptr.shape[1] - 1),
        int(blk_kv),
        head_kv,
        int(max_seqlen_q),
        causal=True,
        schedule=schedule,
    )
    combine(
        o_partial,
        lse_partial,
        out,
        lse_out,
        lse_temperature_partial=None,
        lse_temperature_out=None,
        cu_seqlens=cu_seqlens_q,
        split_counts=schedule.split_counts,
        output_scale=None if q.dtype == torch.float8_e4m3fn else v_global_scale,
        use_pdl=True,
    )
    return True


_TP2CTX_Q8KV4_STATE: dict = {}


def _tp2ctx_q8kv4_run():
    """msa_kernels.prefill_q8kv4.run_prefill (forward_only) when TP2CTX_Q8KV4 may run, else None.

    Option value, opt-in only: TRTLLM_M31_ENABLE_TP2CTX_Q8KV4=1 turns it on, and
    TRTLLM_M31_DISABLE_TP2CTX_Q8KV4=1 (read per call) keeps it off regardless. The package probe is cached; that
    package is not vendored, so the probe finds none and the forward stays the default one.
    """
    import inspect
    import os

    if os.environ.get("TRTLLM_M31_ENABLE_TP2CTX_Q8KV4", "0") != "1":
        return None
    if os.environ.get("TRTLLM_M31_DISABLE_TP2CTX_Q8KV4", "0") == "1":
        return None
    if "run" not in _TP2CTX_Q8KV4_STATE:
        run = None
        if torch.cuda.get_device_capability() in ((10, 3), (10, 7)):
            try:
                from .msa_kernels.prefill_q8kv4 import run_prefill as run

                if "forward_only" not in inspect.signature(run).parameters:
                    run = None  # an overlay older than v3 would combine without C1 / C2
            except ImportError:
                run = None
        _TP2CTX_Q8KV4_STATE["run"] = run
    return _TP2CTX_Q8KV4_STATE["run"]


def _run_tp2ctx_q8kv4(
    run,
    q,
    k_paged,
    v_paged,
    scale_buffers,
    metadata,
    batch,
    cu_q,
    cu_kv,
    k2q_row_ptr,
    schedule,
    sm_scale,
    topk,
    k_global_scale,
    v_global_scale,
    out,
    in_place,
    mxfp8_sink,
) -> None:
    """TP2CTX_Q8KV4: the Q8KV4 forward into split partials, then the installed combine as the DSL path calls it."""
    from src.sm100.fwd.combine import combine

    if not _TP2CTX_Q8KV4_STATE.get("logged"):
        _TP2CTX_Q8KV4_STATE["logged"] = True
        logger.info(
            "[M3 dispatch] TP2CTX_Q8KV4: Q8KV4 C++ sparse prefill forward (E4M3 Q) + installed combine"
        )
    q8 = q if q.dtype == torch.float8_e4m3fn else q.to(torch.float8_e4m3fn)
    o_partial, lse_partial = run(
        q8,
        (k_paged.view(torch.uint8), v_paged.view(torch.uint8)),
        (scale_buffers[:, 0].view(torch.uint8), scale_buffers[:, 1].view(torch.uint8)),
        kv_indices=metadata.msa_block_table[:batch],
        kv_indptr=None,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        k2q_row_ptr=k2q_row_ptr,
        schedule=schedule,
        sm_scale=sm_scale,
        topk=topk,
        seqused_k=metadata.msa_seq_lens_cuda[:batch],
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        forward_only=True,
    )
    write_in_place = (
        in_place
        and out.dtype == torch.bfloat16
        and out.dim() == 3
        and out.is_contiguous()
        and not any(_overlaps(out, other) for other in (q, k_paged, v_paged))
    )
    o_out = (
        out
        if write_in_place
        else torch.empty(
            (int(q.shape[0]), int(q.shape[1]), int(q.shape[2])),
            dtype=torch.bfloat16,
            device=q.device,
        )
    )
    kwargs = dict(
        cu_seqlens=cu_q, split_counts=schedule.split_counts, use_pdl=True
    )  # no output_scale: vg is in K1
    mxfp8 = None
    if (
        mxfp8_sink is not None
        and int(out.shape[0]) >= getattr(mxfp8_sink, "min_rows", 0)
        and write_in_place
        and out.shape[-1] == 128
        and _combine_takes(combine, "o_fp8")
    ):
        rows, width = int(out.shape[0]), int(out.shape[1] * out.shape[2])
        rows_padded = (rows + 127) // 128 * 128
        # The combine writes the live rows' scales; the padding rows' scale bytes must be zero (MSA contract).
        # A mixed-step caller owns C+G buffers and supplies only their context slices here.
        # Returning None declines the sink (for example inside a breakable graph boundary).
        allocate = getattr(mxfp8_sink, "allocate", None)
        if allocate is not None:
            mxfp8 = allocate(out.view(rows, width))
        else:
            mxfp8 = (
                torch.empty((rows, width), dtype=torch.uint8, device=out.device),
                torch.zeros((rows_padded * width // 32,), dtype=torch.uint8, device=out.device),
            )
        if mxfp8 is not None:
            kwargs.update(o_fp8=mxfp8[0], sf_out=mxfp8[1])
    combine(o_partial, lse_partial, o_out, None, **kwargs)
    if not write_in_place:
        out.copy_(o_out)
    elif mxfp8 is not None:
        mxfp8_sink(out.view(out.shape[0], -1), mxfp8[0].view(torch.float8_e4m3fn), mxfp8[1])


@functools.lru_cache(maxsize=None)
def _combine_takes(fn, name: str) -> bool:
    """Whether the installed combine accepts keyword ``name`` (C1's o_fp8 / sf_out)."""
    import inspect

    return name in inspect.signature(fn).parameters


_TP2CTX_BF16X_STATE: dict = {"taken": 0, "skip": {}, "q8kv4_taken": 0, "e4m3x_taken": 0}

# The BF16X / E4M3X fork is a model opt-in (RuntimeHooks.msa_prefill_fork), armed by the executor (arm_prefill_fork).
_PREFILL_FORK_ARMED = False


def arm_prefill_fork() -> None:
    """Let the MSA NVFP4 sparse prefill take the BF16X / E4M3X fork in this process (RuntimeHooks.msa_prefill_fork)."""
    global _PREFILL_FORK_ARMED
    _PREFILL_FORK_ARMED = True


def _tp2ctx_bf16x_run():
    """(run, reason): fmha_sm100.prefill_bf16x.run_prefill (BF16X fork m3b) when TP2CTX_BF16X may run here, else None.

    Armed by a model opt-in; TRTLLM_M31_DISABLE_TP2CTX_BF16X=1 (read per call) keeps the CuTe-DSL forward. The probe
    is cached.
    """
    import os

    if not _PREFILL_FORK_ARMED:
        return None, "not_armed"
    if os.environ.get("TRTLLM_M31_DISABLE_TP2CTX_BF16X", "0") == "1":
        return None, "disabled_by_env"
    if "run" not in _TP2CTX_BF16X_STATE:
        run, reason = None, None
        if torch.cuda.get_device_capability() != (10, 7):  # certified on VR
            reason = "not_sm107"
        else:
            try:
                from .msa_kernels.prefill_bf16x import jit as _bf16x_jit
                from .msa_kernels.prefill_bf16x import run_prefill as run

                if not _bf16x_jit.prebuilt_available():
                    run, reason = None, "no_prefill_bf16x_prebuilt"
            except ImportError:
                run, reason = None, "no_prefill_bf16x_package"
        _TP2CTX_BF16X_STATE["run"] = (run, reason)
    return _TP2CTX_BF16X_STATE["run"]


def _tp2ctx_counters_atexit() -> None:
    """Register the exit-time final-count lines once per process."""
    if not _TP2CTX_BF16X_STATE.get("atexit"):
        import atexit

        _TP2CTX_BF16X_STATE["atexit"] = True
        atexit.register(_tp2ctx_bf16x_final)


def _tp2ctx_e4m3x_ok():
    """None when TP2CTX_E4M3X may take an fp8-q call, else the skip reason.

    TRTLLM_M31_DISABLE_TP2CTX_E4M3X=1 (read per call) keeps the CuTe-DSL fp8 forward. The loaded fork must be an E4M3X
    build (its E4M3 path the exact DSL transcription), else the upstream Q8KV4 kernel would run: probed once.
    """
    import os

    if os.environ.get("TRTLLM_M31_DISABLE_TP2CTX_E4M3X", "0") == "1":
        return "e4m3x_disabled_by_env"
    if "e4m3x_build" not in _TP2CTX_BF16X_STATE:
        try:
            from .msa_kernels.prefill_bf16x import jit as _bf16x_jit

            _TP2CTX_BF16X_STATE["e4m3x_build"] = bool(getattr(_bf16x_jit, "_E4M3X", False))
        except ImportError:
            _TP2CTX_BF16X_STATE["e4m3x_build"] = False
    return None if _TP2CTX_BF16X_STATE["e4m3x_build"] else "fork_not_e4m3x"


def _tp2ctx_e4m3x_count() -> None:
    """TP2CTX_E4M3X calls that returned, logged at counts 1, 2, 4, 8, ..."""
    _tp2ctx_counters_atexit()
    _TP2CTX_BF16X_STATE["e4m3x_taken"] += 1
    n = _TP2CTX_BF16X_STATE["e4m3x_taken"]
    if n & (n - 1) == 0:
        logger.info(f"[M3 dispatch] TP2CTX_E4M3X taken n={n}")


def _tp2ctx_q8kv4_count() -> None:
    """TP2CTX_Q8KV4 calls that returned, logged at counts 1, 2, 4, 8, ..."""
    _tp2ctx_counters_atexit()
    _TP2CTX_BF16X_STATE["q8kv4_taken"] += 1
    n = _TP2CTX_BF16X_STATE["q8kv4_taken"]
    if n & (n - 1) == 0:
        logger.info(f"[M3 dispatch] TP2CTX_Q8KV4 taken n={n}")


def _tp2ctx_bf16x_count(reason=None) -> None:
    """Taken (after the call returned) / skip counters, logged at counts 1, 2, 4, 8, ...

    The last power-of-two line is a lower bound of the final count; the exact one is the exit line, which a killed
    process may not log.
    """
    _tp2ctx_counters_atexit()
    if reason is None:
        _TP2CTX_BF16X_STATE["taken"] += 1
        n = _TP2CTX_BF16X_STATE["taken"]
        if n & (n - 1) == 0:
            logger.info(f"[M3 dispatch] TP2CTX_BF16X taken n={n}")
        return
    skips = _TP2CTX_BF16X_STATE["skip"]
    skips[reason] = n = skips.get(reason, 0) + 1
    if n & (n - 1) == 0:
        logger.info(f"[M3 skip] TP2CTX_BF16X reason={reason} n={n}")


def _tp2ctx_bf16x_final() -> None:
    """At process exit: the exact final taken / skip counts (the power-of-two lines only bound them)."""
    logger.info(
        f"[M3.1 prefill-fwd] TP2CTX_BF16X final taken={_TP2CTX_BF16X_STATE['taken']} "
        f"skip={dict(sorted(_TP2CTX_BF16X_STATE['skip'].items()))}"
    )
    logger.info(f"[M3.1 prefill-fwd] TP2CTX_Q8KV4 final taken={_TP2CTX_BF16X_STATE['q8kv4_taken']}")
    logger.info(f"[M3.1 prefill-fwd] TP2CTX_E4M3X final taken={_TP2CTX_BF16X_STATE['e4m3x_taken']}")


def _tp2ctx_fwd_path_log(q8kv4_run, bf16x_run, bf16x_reason) -> None:
    """Once per process: which sparse prefill forward this process runs by default."""
    if _TP2CTX_BF16X_STATE.get("path_logged"):
        return
    _TP2CTX_BF16X_STATE["path_logged"] = True
    if q8kv4_run is not None:
        path = "Q8KV4 (opt-in TRTLLM_M31_ENABLE_TP2CTX_Q8KV4=1)"
    elif bf16x_run is not None:
        path = (
            "BF16X for bf16 q / E4M3X for fp8 q (default; TRTLLM_M31_DISABLE_TP2CTX_BF16X=1 or "
            "TRTLLM_M31_DISABLE_TP2CTX_E4M3X=1 keeps the DSL)"
        )
    else:
        path = f"DSL (BF16X off: {bf16x_reason})"
    logger.info(f"[M3.1 prefill-fwd] active path: {path}")


def _run_tp2ctx_bf16x(
    run,
    q,
    k_paged,
    v_paged,
    scale_buffers,
    metadata,
    batch,
    cu_q,
    cu_kv,
    k2q_row_ptr,
    schedule,
    sm_scale,
    topk,
    k_global_scale,
    v_global_scale,
    out,
    write_in_place,
    mxfp8=None,
    mxfp8_sink=None,
) -> None:
    """TP2CTX_BF16X: the BF16X forward into split partials + the installed combine (output_scale = v_global_scale).

    ``mxfp8`` (TP2CTX_COMBINE_MXFP8 on the E4M3X / BF16X fork, in-place output only): the fork's run_prefill forwards
    ``o_fp8`` / ``sf_out`` to the combine, which also writes the MXFP8 copy of the final output (bitwise
    ``mxfp8_quantize(out.view(rows, -1), True)``); the pair is then handed to ``mxfp8_sink`` for o_proj."""
    result, _ = run(
        q,
        (k_paged.view(torch.uint8), v_paged.view(torch.uint8)),
        (scale_buffers[:, 0].view(torch.uint8), scale_buffers[:, 1].view(torch.uint8)),
        kv_indices=metadata.msa_block_table[:batch],
        kv_indptr=None,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        k2q_row_ptr=k2q_row_ptr,
        schedule=schedule,
        sm_scale=sm_scale,
        topk=topk,
        seqused_k=metadata.msa_seq_lens_cuda[:batch],
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        out=out if write_in_place else None,
        **({"o_fp8": mxfp8[0], "sf_out": mxfp8[1]} if mxfp8 is not None else {}),
    )
    if not write_in_place:
        out.copy_(result)
    elif mxfp8 is not None:
        mxfp8_sink(out.view(out.shape[0], -1), mxfp8[0].view(torch.float8_e4m3fn), mxfp8[1])


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
    in_place: bool = False,
    schedule_order: Optional[str] = None,
    mxfp8_sink: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]] = None,
) -> None:
    """Run sparse attention over M3's packed NVFP4 cache with the MSA CSR kernel.

    This is the prefill-shaped path: it builds a k2q CSR worklist per call and
    tiles 128 query rows at a time. num_rows narrows it to the leading batch
    rows, so a mixed step can hand its generation suffix to the Triton NVFP4
    decode and size this worklist from the context prefix alone.

    in_place (an attention layer opts in with ``msa_sparse_out_in_place``): the
    MSA combine writes the result into ``out`` itself when the loaded build
    accepts a caller-owned output, instead of returning a tensor this copies.

    schedule_order (an attention layer opts in with ``msa_k2q_schedule_order``): the order of the k2q CSR schedule's
    work items, passed to the MSA build_k2q_csr when it accepts it (e.g. ``"lpt"``: heaviest item first). The order
    only changes which CTA runs which item, never what an item computes or where it writes, so the output is the same.

    mxfp8_sink (an attention layer opts in with ``msa_sparse_out_mxfp8_sink``; requires the in-place output): the MSA
    combine also writes the output as MXFP8 (E4M3 [rows, heads * dim] + swizzled 128x4 K32 UE8M0 scales, bitwise
    ``mxfp8_quantize(out.view(rows, -1), True)``) when the loaded build accepts ``o_fp8`` / ``sf_out``, and the pair is
    handed to ``mxfp8_sink(out_2d, o_fp8, sf_out)`` so the MXFP8 o_proj can skip its own quantize (COMBINE_MXFP8).
    """
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        MSA_REQUIRED_TOPK,
    )

    sparse = _msa_sparse_api()

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
    else:
        q2k = kv_block_indexes.permute(1, 0, 2).contiguous()
        if schedule_order is None:
            k2q_row_ptr, k2q_q_indices, schedule = _k2q_csr_with_schedule(
                sparse,
                q2k,
                cu_q,
                cu_kv,
                page_size,
                total_k=total_k,
                max_seqlen_k=max_kv_len,
                max_seqlen_q=max_q_len,
                total_rows=total_k_rows,
                qhead_per_kv=int(q.shape[1]) // int(k_paged.shape[1]),
                metadata=metadata,
            )
        else:
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
                **(
                    {"schedule_order": schedule_order}
                    if schedule_order is not None and _csr_builder_takes_order(sparse.build_k2q_csr)
                    else {}
                ),
            )
        if identity is not None:
            metadata._msa_identity_csr = (identity, csr_key, (k2q_row_ptr, k2q_q_indices, schedule))
    k_data = k_paged.view(torch.uint8)
    v_data = v_paged.view(torch.uint8)
    k_scale = scale_buffers[:, 0].view(torch.uint8)
    v_scale = scale_buffers[:, 1].view(torch.uint8)
    page_table = metadata.msa_block_table[:batch]
    seqused_k = metadata.msa_seq_lens_cuda[:batch]
    q8kv4_run = _tp2ctx_q8kv4_run()
    bf16x_run, bf16x_reason = _tp2ctx_bf16x_run()
    _tp2ctx_fwd_path_log(q8kv4_run, bf16x_run, bf16x_reason)
    if (
        q8kv4_run is not None
        and page_size == 128
        and int(q.shape[1]) == 16 * int(k_paged.shape[1])
        and int(q.shape[2]) == 128
    ):
        _run_tp2ctx_q8kv4(
            q8kv4_run,
            q,
            k_paged,
            v_paged,
            scale_buffers,
            metadata,
            batch,
            cu_q,
            cu_kv,
            k2q_row_ptr,
            schedule,
            sm_scale,
            topk,
            k_global_scale,
            v_global_scale,
            out,
            in_place,
            mxfp8_sink,
        )
        _tp2ctx_q8kv4_count()
        return
    write_in_place = in_place and _sparse_func_takes_out(sparse.sparse_atten_nvfp4_kv_func)
    if in_place and not _SPARSE_OUT_LOGGED:
        _SPARSE_OUT_LOGGED.append(True)
        logger.info(
            "[M3.1 tp2ctx] SPARSE_OUT in-place: "
            f"{'on' if write_in_place else 'off (the loaded MSA build takes no out=)'}"
        )
    mxfp8 = None
    # TP2CTX_COMBINE_MXFP8: the old NVFP4 sparse func and the E4M3X / BF16X fork (run_prefill o_fp8 / sf_out,
    # forwarded to the combine) each take the MXFP8 sink only when their loaded build accepts it. With a fork loaded the
    # sink follows the fork: a fork without o_fp8 keeps the call on the fork (o_proj then quantizes as stock) instead of
    # falling off to the slower DSL path for the sink.
    sink_old = _sparse_func_takes(sparse.sparse_atten_nvfp4_kv_func, "o_fp8")
    sink_fork = bf16x_run is not None and _combine_takes(bf16x_run, "o_fp8")
    sink_ok = sink_fork if bf16x_run is not None else sink_old
    # A sink may set ``min_rows``: shorter calls skip the MXFP8 copy (the extra combine epilogue costs more than
    # o_proj's quantize saves there) and o_proj quantizes as stock.
    if (
        mxfp8_sink is not None
        and int(out.shape[0]) >= getattr(mxfp8_sink, "min_rows", 0)
        and write_in_place
        and out.dtype == torch.bfloat16
        and out.dim() == 3
        and out.shape[-1] == 128
        and out.is_contiguous()
        and sink_ok
    ):
        rows, width = int(out.shape[0]), int(out.shape[1] * out.shape[2])
        rows_padded = (rows + 127) // 128 * 128
        # The combine writes the live rows' scales; the padding rows' scale bytes must be zero (MSA contract).
        allocate = getattr(mxfp8_sink, "allocate", None)
        if allocate is not None:
            mxfp8 = allocate(out.view(rows, width))
        else:
            mxfp8 = (
                torch.empty((rows, width), dtype=torch.uint8, device=out.device),
                torch.zeros((rows_padded * width // 32,), dtype=torch.uint8, device=out.device),
            )
    if write_in_place:
        # The combine writes out after the forward has read q and the paged K/V; an out that shares their memory
        # (a view of a fused QKV buffer) would depend on that kernel order, so it is refused.
        for name, other in (("q", q), ("K", k_paged), ("V", v_paged)):
            if _overlaps(out, other):
                raise ValueError(f"MiniMax-M3 NVFP4 in-place sparse output aliases {name}")
    if bf16x_run is not None:
        if mxfp8 is not None and not sink_fork:
            bf16x_reason = "c1_mxfp8_sink"
        elif q.dtype == torch.float8_e4m3fn and _tp2ctx_e4m3x_ok() is not None:
            bf16x_reason = _tp2ctx_e4m3x_ok()
        elif q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            bf16x_reason = "q_dtype"
        elif page_size != 128 or int(q.shape[2]) != 128:
            bf16x_reason = "page_or_head_dim"
        elif int(q.shape[1]) != 16 * int(k_paged.shape[1]):
            bf16x_reason = "not_16_q_heads_per_kv"
        else:
            bf16x_reason = None
            _run_tp2ctx_bf16x(
                bf16x_run,
                q,
                k_paged,
                v_paged,
                scale_buffers,
                metadata,
                batch,
                cu_q,
                cu_kv,
                k2q_row_ptr,
                schedule,
                sm_scale,
                topk,
                k_global_scale,
                v_global_scale,
                out,
                write_in_place,
                mxfp8=mxfp8,
                mxfp8_sink=mxfp8_sink,
            )
            if q.dtype == torch.float8_e4m3fn:
                _tp2ctx_e4m3x_count()
            else:
                _tp2ctx_bf16x_count()
            return
    _tp2ctx_bf16x_count(bf16x_reason)
    if mxfp8 is not None and not sink_old:
        mxfp8 = None
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
        PREFILL_HOST_TRIMS,
    )

    if (
        PREFILL_HOST_TRIMS
        and mxfp8 is None
        and _nvfp4_sparse_attention_into(
            q,
            k_data,
            v_data,
            k_scale,
            v_scale,
            k_global_scale,
            v_global_scale,
            k2q_row_ptr,
            k2q_q_indices,
            schedule,
            topk,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_q_len,
            blk_kv=page_size,
            softmax_scale=sm_scale,
            page_table=page_table,
            seqused_k=seqused_k,
            out=out,
            metadata=metadata,
        )
    ):
        return
    result = sparse.sparse_atten_nvfp4_kv_func(
        q,
        k_data,
        v_data,
        k_scale,
        v_scale,
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
        page_table=page_table,
        seqused_k=seqused_k,
        schedule=schedule,
        **({"out": out} if write_in_place else {}),
        **({"o_fp8": mxfp8[0], "sf_out": mxfp8[1]} if mxfp8 is not None else {}),
    )
    if not write_in_place:
        out.copy_(result)
    elif mxfp8 is not None:
        mxfp8_sink(out.view(out.shape[0], -1), mxfp8[0].view(torch.float8_e4m3fn), mxfp8[1])


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the byte ranges a and b span intersect (host-side pointer arithmetic)."""

    def span(t):
        start = t.data_ptr()
        extent = (
            1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
            if t.numel()
            else 0
        )
        return start, start + extent * t.element_size()

    a0, a1 = span(a)
    b0, b1 = span(b)
    return a0 < b1 and b0 < a1


_SPARSE_OUT_LOGGED: list = []  # one engagement line per process (an attention layer opted in)
_SPARSE_FUNC_TAKES_OUT: dict = {}


_SPARSE_FUNC_TAKES: dict = {}


def _sparse_func_takes(func, name: str) -> bool:
    """Whether the loaded MSA sparse_atten_nvfp4_kv_func accepts the keyword ``name`` (decided once per function)."""
    key = (func, name)
    takes = _SPARSE_FUNC_TAKES.get(key)
    if takes is None:
        try:
            takes = name in inspect.signature(func).parameters
        except (TypeError, ValueError):
            takes = False
        _SPARSE_FUNC_TAKES[key] = takes
    return takes


def _sparse_func_takes_out(func) -> bool:
    """Whether the loaded MSA sparse_atten_nvfp4_kv_func accepts a caller-owned ``out`` (decided once per function)."""
    takes = _SPARSE_FUNC_TAKES_OUT.get(func)
    if takes is None:
        try:
            takes = "out" in inspect.signature(func).parameters
        except (TypeError, ValueError):
            takes = False
        _SPARSE_FUNC_TAKES_OUT[func] = takes
    return takes


_CSR_BUILDER_TAKES_ORDER: dict = {}


def _csr_builder_takes_order(func) -> bool:
    """Whether the loaded MSA build_k2q_csr accepts ``schedule_order`` (decided once per function)."""
    takes = _CSR_BUILDER_TAKES_ORDER.get(func)
    if takes is None:
        try:
            takes = "schedule_order" in inspect.signature(func).parameters
        except (TypeError, ValueError):
            takes = False
        _CSR_BUILDER_TAKES_ORDER[func] = takes
    return takes


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
            in_place=getattr(attn, "msa_sparse_out_in_place", False),
            schedule_order=getattr(attn, "msa_k2q_schedule_order", None),
            mxfp8_sink=getattr(attn, "msa_sparse_out_mxfp8_sink", None),
        )
        return

    # The fmha_sm100 variant is chosen from q.dtype and shares one dtype across
    # q/k/v, so q must be FP8 to match an FP8 paged K/V. MiniMax-M3 has no
    # KV-cache scales, so the scale is 1.0 and this is a plain E4M3 cast.
    use_fp8 = k_paged.dtype == torch.float8_e4m3fn
    if use_fp8:
        q_view = q_view.to(torch.float8_e4m3fn)

    def rows_of(lens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Narrow a per-request host length tensor to this phase's rows.

        Slicing keeps the pinned backing, so an inline plan still stages these
        with non-blocking copies.
        """
        if lens is None:
            return None
        return lens[row_first : row_first + num_rows]

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


__all__ = ["MsaPrefillFmha", "run_msa_prefill_gqa", "run_msa_sparse_gqa"]
