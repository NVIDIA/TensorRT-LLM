# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Drop-in adapter: fmha_sm100 API → sparse_atten_func backend.

Usage:
    from minfer.ops.sparse_fmha_adapter import sparse_fmha_plan, sparse_fmha

    plan_info = sparse_fmha_plan(qo_lens, kv_lens, num_qo_heads, ...)
    out, _ = sparse_fmha(q, k, v, plan_info=plan_info,
                         kv_indices=kv_indices,
                         kv_block_indexes=kv_block_indexes)
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

_MM_SPARSE_DIR = os.path.join(
    os.path.dirname(__file__), ".", "cute"
)
if os.path.isdir(_MM_SPARSE_DIR) and _MM_SPARSE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_MM_SPARSE_DIR))

from interface import sparse_atten_func
from sparse_index_utils import build_k2q_csr
from src.sm100.prepare_scheduler import SPARSE_SCHEDULE_MODEL
from src.common.aot_cache import _key_to_path


def _compute_aot_kernel_paths(head_dim, n_block_size, qhead_per_kv, topk,
                              causal,
                              dtype=torch.bfloat16, partial_dtype=torch.float32):
    fwd_key = (
        "sparse_forward_sm100_csr_varlen",
        head_dim, n_block_size, qhead_per_kv, dtype, partial_dtype,
        bool(causal), True, True, n_block_size,
        True, False,
    )
    k_block_size = 128 if head_dim > 64 else 64
    try:
        import cutlass
        cutlass_partial = cutlass.Float32
        cutlass_out = cutlass.BFloat16
    except ImportError:
        cutlass_partial = partial_dtype
        cutlass_out = dtype
    combine_key = (
        "combine", head_dim, k_block_size, 64, topk,
        cutlass_partial, cutlass_out,
        True, False, True, False, True, True,
    )
    fwd_path = _key_to_path(fwd_key) + ".o"
    combine_path = _key_to_path(combine_key) + ".o"
    return {
        "fwd_kernel_path": fwd_path if os.path.isfile(fwd_path) else "",
        "combine_kernel_path": combine_path if os.path.isfile(combine_path) else "",
        "fwd_func_name": str(fwd_key[0]),
        "combine_func_name": str(combine_key[0]),
    }


def sparse_fmha_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[torch.Tensor] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    causal: bool = True,
    usable_SM_count=-1,
    use_fp8_kvcache: bool = False,
) -> dict:
    """Build a reusable sparse-prefill plan for the MM-Sparse backend.

    This is the sparse prefill implementation used behind ``fmha_sm100_plan``
    when ``kv_block_num > 0`` and the selected sparse mode is prefill.

    Parameters
    ----------
    qo_segment_lens : torch.Tensor
        Shape ``[batch_size]``.  Per-request Q lengths.
    kv_segment_lens : torch.Tensor
        Shape ``[batch_size]``.  Per-request KV lengths.
    num_qo_heads : int
        Number of Q/O heads.
    num_kv_heads : int, optional
        Number of KV heads.  Required for GQA planning; ``num_qo_heads`` must
        be divisible by this value.
    qo_offset : torch.Tensor, optional
        Shape ``[batch_size]``.  Per-request causal offset.  If omitted,
        ``seqused_k`` is derived from ``kv_segment_lens``.
    num_kv_splits : int, optional
        Reserved for API compatibility with dense ``fmha_sm100_plan``.
    page_size : int, optional
        KV page/block size.  Sparse prefill requires paged KV, so this must be
        positive and is normally 128.
    output_maxscore : bool, optional
        Sparse prefill backend does not emit max-score tensors; must be False.
    kv_block_num : int, optional
        Number of selected KV blocks per query.  Supported values are
        ``4, 8, 16, 32``.
    causal : bool, optional
        Whether to apply causal masking.  Current backend requires True.
    usable_SM_count : int, optional
        Maximum number of SMs used by the sparse scheduler.  ``-1`` uses all
        available SMs.
    use_fp8_kvcache : bool, optional
        If True, compute AOT lookup keys for FP8 K/V cache kernels.

    Returns
    -------
    dict
        Plan dictionary consumed by ``sparse_fmha`` and by ``fmha_sm100``'s
        sparse-prefill path.
    """
    assert kv_block_num in {4,8,16,32}, f"kv_block_num={kv_block_num}"
    assert page_size >= 1, f"page_size={page_size}"
    assert output_maxscore == False
    assert causal == True
    
    batch = qo_segment_lens.shape[0]
    gpu_device = torch.device('cuda')

    # Stage the host lengths to the device once, asynchronously, and read every
    # scalar below from the host copy. Planning runs on the critical path of
    # input preparation, where a blocking copy or a device .item() would drain
    # the CUDA queue.
    qo_lens_gpu = qo_segment_lens.to(torch.int32).to(gpu_device, non_blocking=True)
    kv_lens_host = kv_segment_lens.to(torch.int32)
    kv_lens_gpu = kv_lens_host.to(gpu_device, non_blocking=True)

    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=gpu_device)
    cu_seqlens_q[1:] = torch.cumsum(qo_lens_gpu, dim=0)

    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=gpu_device)
    cu_seqlens_k[1:] = torch.cumsum(kv_lens_gpu, dim=0)

    max_seqlen_q = int(qo_segment_lens.max().item())
    max_seqlen_k = int(kv_segment_lens.max().item())
    total_k = int(kv_segment_lens.sum().item())
    blk_kv = page_size

    kv_lens = kv_segment_lens.tolist()
    total_rows = sum((int(kl) + blk_kv - 1) // blk_kv for kl in kv_lens)

    if qo_offset is not None:
        seqused_k = (qo_segment_lens + qo_offset).to(torch.int32).to(gpu_device, non_blocking=True)
    else:
        seqused_k = kv_lens_gpu.clone()

    # Equals cu_seqlens_q[-1], read on the host.
    total_q = int(qo_segment_lens.sum())
    qhead_per_kv = num_qo_heads // num_kv_heads if num_kv_heads > 0 else 1

    target_q_per_cta = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta(
        total_q=total_q,
        topk=kv_block_num,
        blk_kv=page_size,
        head_kv=num_kv_heads,
        qhead_per_kv=qhead_per_kv,
        device=gpu_device,
        usable_SM_count=usable_SM_count,
    )
    scheduler_metadata_capacity = SPARSE_SCHEDULE_MODEL.flat_schedule_capacity(
        total_rows=total_rows,
        total_q=total_q,
        topk=kv_block_num,
        head_kv=num_kv_heads,
        target_q_per_cta=target_q_per_cta,
    )

    return {
        "qo_segment_lens": qo_lens_gpu,
        "cu_seqlens_q": cu_seqlens_q,
        "qo_segment_offsets": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        # Host tensor: only the page-table builder consumes it, and it reads
        # per-request page counts on the host. Kernels take the lengths from
        # cu_seqlens_k and seqused_k.
        "kv_segment_lens": kv_lens_host,
        "seqused_k": seqused_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "total_k": total_k,
        "total_rows": total_rows,
        "num_qo_heads": num_qo_heads,
        "page_size": page_size,
        "blk_kv": blk_kv,
        "kv_block_num": kv_block_num,
        "causal": causal,
        "batch": batch,
        "MM-SA-Nv":True,
        "usable_SM_count":usable_SM_count,
        "num_kv_heads": num_kv_heads,
        "qhead_per_kv": qhead_per_kv,
        "target_q_per_cta": target_q_per_cta,
        "scheduler_metadata_capacity": scheduler_metadata_capacity,
        **_compute_aot_kernel_paths(
            head_dim=128,
            n_block_size=page_size,
            qhead_per_kv=qhead_per_kv,
            topk=kv_block_num,
            causal=causal,
            dtype=torch.float8_e4m3fn if use_fp8_kvcache else torch.bfloat16,
        ),
    }


def _convert_kv_block_indexes_to_q2k(
    kv_block_indexes: torch.Tensor,
    num_kv_heads: int,
    num_qo_heads: int,
    qhead_per_kv: int,
) -> torch.Tensor:
    """kv_block_indexes [total_q, H, topk] → q2k [H_kv, total_q, topk]."""
    h_dim = kv_block_indexes.shape[1]
    if h_dim == num_kv_heads:
        q2k = kv_block_indexes.permute(1, 0, 2).contiguous()
    elif h_dim == num_qo_heads and num_qo_heads != num_kv_heads:
        q2k = kv_block_indexes[:, ::qhead_per_kv, :].permute(1, 0, 2).contiguous()
    else:
        raise ValueError(
            f"kv_block_indexes head dim {h_dim} doesn't match "
            f"num_kv_heads={num_kv_heads} or num_qo_heads={num_qo_heads}"
        )
    return q2k.to(torch.int32)


def _build_page_table(
    kv_indices: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    page_size: int,
    batch: int,
) -> torch.Tensor:
    """Flat kv_indices → page_table [batch, max_pages_per_seq].

    Each request's pages are a contiguous run of kv_indices, so the padded table
    is kv_indices[start[b] + col] masked to the request's page count. One gather
    builds it: a copy per request would dominate host time once the batch is
    wide.
    """
    lens = kv_segment_lens if kv_segment_lens.device.type == "cpu" else kv_segment_lens.cpu()
    pages = (lens.to(torch.int64) + (page_size - 1)) // page_size
    max_pages = int(pages.max())
    starts = torch.cumsum(pages, 0) - pages
    # Both row descriptors travel in a single host-to-device copy.
    meta = torch.stack((pages, starts)).to(kv_indices.device, non_blocking=True)
    pages_dev = meta[0].unsqueeze(1)
    starts_dev = meta[1].unsqueeze(1)

    col = torch.arange(max_pages, device=kv_indices.device, dtype=torch.int64).unsqueeze(0)
    mask = col < pages_dev
    # Padded columns run past the end of a request's run, so clamp the gather
    # index into range and zero them afterwards.
    src = (starts_dev + col).clamp_(max=max(kv_indices.numel() - 1, 0))
    page_table = kv_indices.index_select(0, src.reshape(-1)).to(torch.int32)
    page_table = page_table.view(batch, max_pages).masked_fill_(~mask, 0)
    assert page_table.data_ptr() % 16 == 0, (
        f"_build_page_table failed to align: page_table=0x{page_table.data_ptr():x}"
    )
    return page_table


def _version_token(tensor: torch.Tensor):
    """Version counter for `tensor`, or None if it does not track one.

    Inference tensors have no version counter, so this is None on the serving
    path and the caller falls back to identity checks alone.
    """
    return None if torch.is_inference(tensor) else tensor._version


def _page_table_for_plan(
    plan_info: dict,
    kv_indices: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    page_size: int,
    batch: int,
) -> torch.Tensor:
    """Return this plan's page table, building it at most once per step.

    Every sparse layer of a step runs the same plan over the same kv_indices
    buffer, so they all build a byte-identical table. Cache it on the plan and
    reuse it while the inputs look unchanged.

    The cache holds a reference to kv_segment_lens so identity comparison is
    sound: a live object's id cannot be recycled. A caller that rebuilds its
    plan or restages these lengths therefore gets a fresh table. Version
    counters add a content check where tensors track one.
    """
    cached = plan_info.get("_page_table_cache")
    if cached is not None:
        lens, ptr, numel, versions, geometry, page_table = cached
        if (
            lens is kv_segment_lens
            and ptr == kv_indices.data_ptr()
            and numel == kv_indices.numel()
            and geometry == (page_size, batch)
            and versions
            == (_version_token(kv_indices), _version_token(kv_segment_lens))
        ):
            return page_table
    page_table = _build_page_table(kv_indices, kv_segment_lens, page_size, batch)
    plan_info["_page_table_cache"] = (
        kv_segment_lens,
        kv_indices.data_ptr(),
        kv_indices.numel(),
        (_version_token(kv_indices), _version_token(kv_segment_lens)),
        (page_size, batch),
        page_table,
    )
    return page_table


def sparse_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info: dict,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    o_scale: Optional[float] = None,
    kv_indices: Optional[torch.Tensor] = None,
    output_maxscore: bool = True,
    output_o: bool = True,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override = None,
    check_input_valid: bool = False,
) -> Tuple[torch.Tensor, None]:
    """Run sparse prefill through ``sparse_atten_func`` using an FMHA-style API.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_q, num_qo_heads, 128]``.  BF16 or FP8 E4M3.
    k : torch.Tensor
        Paged KV tensor with shape ``[total_pages, num_kv_heads, page_size, 128]``.
    v : torch.Tensor
        Same layout as ``k``.
    plan_info : dict
        Plan returned by ``sparse_fmha_plan``.
    out : torch.Tensor, optional
        Accepted for compatibility.  If supplied, the result is copied into it.
    max_score : torch.Tensor, optional
        Accepted for compatibility.  Sparse prefill does not produce max-score
        output and returns ``None`` for this slot.
    sm_scale : float, optional
        Softmax scale.  Defaults to ``1 / sqrt(head_dim)``.
    q_scale, k_scale, v_scale, o_scale : float, optional
        Accepted for FMHA API compatibility; only ``sm_scale`` is used by this
        backend.
    kv_indices : torch.Tensor, optional
        Flattened physical page table with dtype int32.  Required for paged KV.
    output_maxscore : bool, optional
        Accepted for compatibility; sparse prefill returns no max-score tensor.
    output_o : bool, optional
        Accepted for compatibility.  The sparse backend always computes O.
    kv_block_indexes : torch.Tensor
        Shape ``[total_q, num_kv_heads or num_qo_heads, topK]``.  Sparse KV
        block indices in ascending order with ``-1`` padding.
    q_offset_override : int or torch.Tensor, optional
        Runtime causal-offset override.  Tensor form has shape ``[batch_size]``.
    check_input_valid : bool, optional
        Reserved for compatibility with ``fmha_sm100``.

    Returns
    -------
    tuple[torch.Tensor, None]
        Output tensor and ``None`` for max-score output.
    """
    if kv_block_indexes is None:
        raise ValueError("sparse_fmha requires kv_block_indexes")
    

    qo_segment_lens = plan_info["qo_segment_lens"]
    cu_seqlens_q = plan_info["cu_seqlens_q"]
    cu_seqlens_k = plan_info["cu_seqlens_k"]
    num_qo_heads = plan_info["num_qo_heads"]
    num_kv_heads = k.shape[1]
    qhead_per_kv = num_qo_heads // num_kv_heads
    assert qhead_per_kv in {1,2,4,8,16}, f"qhead_per_kv={qhead_per_kv}"
    page_size = plan_info["page_size"]
    blk_kv = plan_info["blk_kv"]
    topk = plan_info["kv_block_num"]
    causal = plan_info["causal"]
    max_seqlen_q = plan_info["max_seqlen_q"]
    max_seqlen_k = plan_info["max_seqlen_k"]
    total_k = plan_info["total_k"]
    total_rows = plan_info["total_rows"]
    batch = plan_info["batch"]
    kv_segment_lens = plan_info["kv_segment_lens"]
    usable_SM_count = int(plan_info.get("usable_SM_count", -1))

    if isinstance(q_offset_override, int):
        q_offset_override = torch.full_like(qo_segment_lens, q_offset_override)
    elif q_offset_override is not None:
        assert q_offset_override.device == qo_segment_lens.device
    
    if q_offset_override is not None:
        seqused_k = (qo_segment_lens + q_offset_override).to(torch.int32).to(qo_segment_lens.device)
    else:
        seqused_k = plan_info["seqused_k"]

    q2k = _convert_kv_block_indexes_to_q2k(
        kv_block_indexes, num_kv_heads, num_qo_heads, qhead_per_kv,
    )

    is_paged = page_size > 0 and k.ndim == 4

    page_table = None
    if is_paged:

        if kv_indices is not None:
            page_table = _page_table_for_plan(
                plan_info, kv_indices, kv_segment_lens, page_size, batch,
            )

    # build_k2q_csr(return_schedule=True) builds schedule using hardware SM count internally
    # (build_k2q_csr_native.cu), which ignores usable_SM_count. When SM-limited, skip its
    # schedule and let prepare_scheduler build one that respects usable_SM_count.
    if usable_SM_count > 0:
        k2q_row_ptr, k2q_q_indices = build_k2q_csr(
            q2k, cu_seqlens_q, cu_seqlens_k, blk_kv,
            total_k=total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            total_rows=total_rows,
            qhead_per_kv=qhead_per_kv,
            return_schedule=False,
        )
        schedule = None
    else:
        k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
            q2k, cu_seqlens_q, cu_seqlens_k, blk_kv,
            total_k=total_k,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            total_rows=total_rows,
            qhead_per_kv=qhead_per_kv,
            return_schedule=True,
        )

    softmax_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5

    # Hand the caller's buffer to the kernel when its layout already matches
    # what `combine` produces, so the result does not have to be copied into
    # place afterwards. Anything else falls back to allocate-then-copy.
    writable_out = (
        out is not None
        and out.is_contiguous()
        and out.dtype == torch.bfloat16
        and out.device == q.device
        and tuple(out.shape) == tuple(q.shape)
    )

    # print(q.shape, k.shape)
    result = sparse_atten_func(
        q, k, v,
        k2q_row_ptr, k2q_q_indices, topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=False,
        page_table=page_table,
        seqused_k=seqused_k,
        schedule=schedule,
        usable_SM_count=usable_SM_count,
        out=out if writable_out else None,
    )

    if out is not None and not writable_out:
        out.copy_(result)
        return out, None
    return result, None
