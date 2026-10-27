# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Host-side q2k <-> k2q index conversion for sparse attention.

These utilities prepare sparse metadata on the Python side for tests,
benchmarks, and other offline preprocessing flows. They are not kernel
runtime helpers, so they intentionally live outside `src/common`.

Sparse attention pattern:
    - Each Q token independently selects up to topK KV blocks (blk_kv tokens each).
    - Under GQA, all Q heads in one group share the same sparsity pattern,
      so indices are defined at the head_kv level.

Shapes:
    q2k_indices: [batch, head_kv, Sq, topK]   int32, valid values in [0, num_kv_blocks),
                                              trailing unused slots padded with -1
    k2q_indices: [batch, head_kv, Nkv, Sq]    int32, padded with -1
    k2q_counts:  [batch, head_kv, Nkv]        int32

CSR reverse-index format:
    q2k_indices:   [head_kv, total_q, topK]   int32, values are batch-local kv_block indices
    k2q_row_ptr:   [head_kv, total_rows + 1]  int32
    k2q_q_indices: [head_kv, total_q * topK]  int32, values are batch-local q_idx
"""

from typing import Optional, Tuple

import torch

from src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100


def q2k_to_k2q(
    q2k_indices: torch.Tensor,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert q2k sparse indices to k2q representation.

    For each KV block, find which Q tokens attend to it.

    Args:
        q2k_indices: [batch, head_kv, Sq, topK] int32.
            For each Q token, the KV blocks it attends to. Unused slots must
            be padded with -1.
        num_kv_blocks: Total number of KV blocks (= Skv / blk_kv).

    Returns:
        k2q_indices: [batch, head_kv, num_kv_blocks, Sq] int32.
            For each KV block, the Q token indices that attend to it,
            left-packed and padded with -1. Last dim fixed to Sq (upper bound).
        k2q_counts: [batch, head_kv, num_kv_blocks] int32.
            Actual number of Q tokens per KV block.
    """
    B, H, Sq, topK = q2k_indices.shape
    device = q2k_indices.device
    N = Sq * topK

    kv_flat = q2k_indices.reshape(B, H, N).long()
    valid_flat = kv_flat >= 0
    q_flat = (
        torch.arange(Sq, device=device)
        .unsqueeze(-1)
        .expand(Sq, topK)
        .reshape(N)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, H, N)
    )

    k2q_counts = torch.zeros(B, H, num_kv_blocks, dtype=torch.int32, device=device)
    safe_kv_flat = torch.where(valid_flat, kv_flat, torch.zeros_like(kv_flat))
    k2q_counts.scatter_add_(
        2,
        safe_kv_flat,
        valid_flat.to(torch.int32),
    )

    sort_keys = torch.where(
        valid_flat,
        kv_flat,
        torch.full_like(kv_flat, num_kv_blocks),
    )
    sorted_kv, sort_idx = sort_keys.sort(dim=-1, stable=True)
    sorted_q = q_flat.gather(-1, sort_idx)
    sorted_valid = valid_flat.gather(-1, sort_idx)

    offsets = torch.zeros(B, H, num_kv_blocks, dtype=torch.int64, device=device)
    offsets[:, :, 1:] = k2q_counts[:, :, :-1].cumsum(dim=-1).long()

    global_pos = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0).expand(B, H, N)
    group_offset = offsets.gather(2, sorted_kv.clamp(max=num_kv_blocks - 1))
    pos_in_group = global_pos - group_offset

    k2q_indices = torch.full(
        (B, H, num_kv_blocks, Sq), -1, dtype=torch.int32, device=device
    )
    flat_k2q = k2q_indices.reshape(B, H, -1)
    flat_idx = sorted_kv.clamp(max=num_kv_blocks - 1) * Sq + pos_in_group
    for b in range(B):
        for h in range(H):
            valid = sorted_valid[b, h]
            flat_k2q[b, h, flat_idx[b, h, valid]] = sorted_q[b, h, valid].int()

    return k2q_indices, k2q_counts


def k2q_to_q2k(
    k2q_indices: torch.Tensor,
    k2q_counts: torch.Tensor,
    Sq: int,
    topK: int,
) -> torch.Tensor:
    """Convert dense k2q indices back to q2k representation.

    Parameters
    ----------
    k2q_indices : torch.Tensor
        Shape ``[batch, head_kv, num_kv_blocks, Sq]`` and dtype int32.  Values
        are Q token indices padded with ``-1``.
    k2q_counts : torch.Tensor
        Shape ``[batch, head_kv, num_kv_blocks]`` and dtype int32.  Number of
        valid Q indices per KV block.
    Sq : int
        Q sequence length per batch item in this dense reference format.
    topK : int
        Maximum number of KV blocks selected per Q token.

    Returns
    -------
    torch.Tensor
        Shape ``[batch, head_kv, Sq, topK]``, dtype int32.  Entries are sorted
        by KV block index with ``-1`` padding at the tail.
    """
    B, H, Nkv, _ = k2q_indices.shape
    device = k2q_indices.device

    q2k = torch.full((B, H, Sq, topK), -1, dtype=torch.int32, device=device)
    counters = torch.zeros(B, H, Sq, dtype=torch.int64, device=device)

    for b in range(B):
        for h in range(H):
            for kv_blk in range(Nkv):
                count = k2q_counts[b, h, kv_blk].item()
                for j in range(count):
                    qt = k2q_indices[b, h, kv_blk, j].item()
                    if qt < 0:
                        continue
                    p = counters[b, h, qt].item()
                    if p < topK:
                        q2k[b, h, qt, p] = kv_blk
                        counters[b, h, qt] += 1

    q2k_sort_key = torch.where(q2k < 0, torch.full_like(q2k, Nkv), q2k)
    _, sort_idx = q2k_sort_key.sort(dim=-1)
    q2k = q2k.gather(-1, sort_idx)
    return q2k


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, *, name: str) -> None:
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {tuple(cu_seqlens.shape)}")
    if cu_seqlens.numel() < 1:
        raise ValueError(f"{name} must have at least one element")
    if not cu_seqlens.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _rows_per_batch(cu_seqlens_k: torch.Tensor, kv_block_size: int) -> torch.Tensor:
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    return (seqlens_k + kv_block_size - 1) // kv_block_size


def _build_packed_row_map(rows_per_batch: torch.Tensor) -> tuple[torch.Tensor, int]:
    rows_per_batch_cpu = rows_per_batch.to("cpu", non_blocking=False).tolist()
    batch = len(rows_per_batch_cpu)
    max_rows = max(rows_per_batch_cpu, default=0)
    row_dtype = (
        torch.int32
        if sum(rows_per_batch_cpu) < torch.iinfo(torch.int32).max
        else torch.int64
    )
    row_map_cpu = torch.full((batch, max_rows), -1, dtype=row_dtype)
    row_linear = 0
    for kv_block_idx in range(max_rows):
        for batch_idx, row_count in enumerate(rows_per_batch_cpu):
            if kv_block_idx < row_count:
                row_map_cpu[batch_idx, kv_block_idx] = row_linear
                row_linear += 1
    return row_map_cpu.to(rows_per_batch.device), row_linear


def build_k2q_csr_torch_reference(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_block_size: int,
) -> tuple:
    """Torch reference for q2k -> k2q CSR conversion.

    Parameters
    ----------
    q2k_indices : torch.Tensor
        Shape ``[head_kv, total_q, topK]``, dtype int32.  Values are
        batch-local KV block indices padded with ``-1``.
    cu_seqlens_q : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of Q lengths.
    cu_seqlens_k : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of KV lengths.
    kv_block_size : int
        Number of KV tokens per sparse block.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(k2q_row_ptr, k2q_q_indices)`` where ``k2q_row_ptr`` has shape
        ``[head_kv, total_rows + 1]`` and ``k2q_q_indices`` has shape
        ``[head_kv, total_q * topK]``.
    """
    if kv_block_size <= 0:
        raise ValueError(f"kv_block_size must be > 0, got {kv_block_size}")
    if q2k_indices.dtype != torch.int32:
        raise TypeError(f"q2k_indices must be torch.int32, got {q2k_indices.dtype}")
    if q2k_indices.ndim != 3:
        raise ValueError(
            "q2k_indices must have shape [head_kv, total_q, topK], "
            f"got {tuple(q2k_indices.shape)}"
        )
    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q")
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape [B + 1]")
    if q2k_indices.device != cu_seqlens_q.device or q2k_indices.device != cu_seqlens_k.device:
        raise ValueError("q2k_indices, cu_seqlens_q, and cu_seqlens_k must be on the same device")

    head_kv, total_q, topk = q2k_indices.shape
    if total_q != int(cu_seqlens_q[-1].item()):
        raise ValueError(
            f"q2k_indices.shape[1] ({total_q}) must equal cu_seqlens_q[-1] "
            f"({int(cu_seqlens_q[-1].item())})"
        )

    rows_per_batch = _rows_per_batch(cu_seqlens_k, kv_block_size)
    row_map, total_rows = _build_packed_row_map(rows_per_batch)
    nnz_upper_bound = total_q * topk

    k2q_row_ptr = torch.zeros((head_kv, total_rows + 1), dtype=torch.int32, device=q2k_indices.device)
    k2q_q_indices = torch.full(
        (head_kv, nnz_upper_bound), -1, dtype=torch.int32, device=q2k_indices.device
    )
    if total_rows == 0 or total_q == 0 or topk == 0:
        return k2q_row_ptr, k2q_q_indices

    counts = torch.zeros((head_kv, total_rows), dtype=torch.int32, device=q2k_indices.device)
    total_entries = total_q * topk
    row_dtype = torch.int32 if total_rows < torch.iinfo(torch.int32).max else torch.int64
    row_all = torch.empty((head_kv, total_entries), dtype=row_dtype, device=q2k_indices.device)
    q_all = torch.empty((head_kv, total_entries), dtype=torch.int32, device=q2k_indices.device)
    valid_all = torch.empty((head_kv, total_entries), dtype=torch.bool, device=q2k_indices.device)
    rows_per_batch_cpu = rows_per_batch.to("cpu", non_blocking=False).tolist()
    q_cu_cpu = cu_seqlens_q.to("cpu", non_blocking=False).tolist()
    entry_cursor = 0

    for batch_idx, kv_rows in enumerate(rows_per_batch_cpu):
        q_start = q_cu_cpu[batch_idx]
        q_end = q_cu_cpu[batch_idx + 1]
        q_len = q_end - q_start
        if q_len == 0:
            continue
        num_entries = q_len * topk
        q2k_batch = q2k_indices[:, q_start:q_end, :]
        valid_batch = q2k_batch >= 0
        if valid_batch.any():
            max_valid_kv = int(q2k_batch[valid_batch].max().item())
            if max_valid_kv >= kv_rows:
                raise ValueError(
                    f"q2k_indices references kv_block {max_valid_kv} for batch {batch_idx}, "
                    f"but that batch only has {kv_rows} logical kv blocks"
                )
        kv_flat = q2k_batch.reshape(head_kv, num_entries).long()
        valid_flat = valid_batch.reshape(head_kv, num_entries)
        safe_kv_flat = torch.where(valid_flat, kv_flat, torch.zeros_like(kv_flat))
        row_map_batch = row_map[batch_idx]
        row_flat = row_map_batch[safe_kv_flat]
        q_flat = (
            torch.arange(q_len, device=q2k_indices.device, dtype=torch.int32)
            .view(1, q_len, 1)
            .expand(head_kv, q_len, topk)
            .reshape(head_kv, num_entries)
        )
        row_all[:, entry_cursor : entry_cursor + num_entries] = row_flat
        q_all[:, entry_cursor : entry_cursor + num_entries] = q_flat
        valid_all[:, entry_cursor : entry_cursor + num_entries] = valid_flat
        counts.scatter_add_(1, row_flat.to(torch.int64), valid_flat.to(torch.int32))
        entry_cursor += num_entries

    k2q_row_ptr[:, 1:] = counts.cumsum(dim=1, dtype=torch.int32)

    sort_stride = max(total_q, 1)
    invalid_key = total_rows * sort_stride
    max_sort_key = invalid_key + max(total_q - 1, 0)
    if max_sort_key < torch.iinfo(torch.int32).max:
        sort_keys = torch.full_like(row_all, invalid_key, dtype=torch.int32)
        sort_keys[valid_all] = row_all[valid_all] * sort_stride + q_all[valid_all]
    else:
        sort_keys = torch.full_like(row_all, invalid_key, dtype=torch.int64)
        sort_keys[valid_all] = (
            row_all[valid_all].to(torch.int64) * sort_stride
            + q_all[valid_all].to(torch.int64)
        )
    _, sort_idx = sort_keys.sort(dim=1, stable=True)
    sorted_q = q_all.gather(1, sort_idx)

    valid_counts = valid_all.sum(dim=1)
    write_mask = (
        torch.arange(total_entries, device=q2k_indices.device)
        .unsqueeze(0)
        .expand(head_kv, -1)
        < valid_counts.unsqueeze(1)
    )
    k2q_q_indices[write_mask] = sorted_q[write_mask]

    return k2q_row_ptr, k2q_q_indices


_K2Q_CSR_BUILDER = SparseK2qCsrBuilderSm100()


def build_k2q_csr(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kv_block_size: int,
    *,
    total_k: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    total_rows: Optional[int] = None,
    qhead_per_kv: int = 1,
    return_schedule: bool = False,
    schedule_order: str = "stock",
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, object]:
    """Build the public k2q CSR reverse index on GPU.

    Runtime construction does not read device-side ``cu_seqlens`` on the host,
    so callers must provide size hints such as ``total_k`` from already-known
    tensor shapes.

    Parameters
    ----------
    q2k_indices : torch.Tensor
        Shape ``[head_kv, total_q, topK]``, dtype int32, contiguous.  Values are
        batch-local KV block indices with trailing ``-1`` padding.
    cu_seqlens_q : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of Q lengths.
    cu_seqlens_k : torch.Tensor
        Shape ``[batch_size + 1]``, dtype int32.  Prefix sums of KV lengths.
    kv_block_size : int
        Number of KV tokens per sparse block.
    total_k : int
        Total KV token count.  Required; normally ``k.shape[0]`` for dense KV
        or ``sum(kv_segment_lens)`` for paged KV.
    max_seqlen_k : int, optional
        Maximum KV sequence length.  Passing this avoids recomputing a bound.
    max_seqlen_q : int, optional
        Maximum Q sequence length.
    total_rows : int, optional
        Total number of packed KV-block rows across the batch.  If omitted,
        the builder derives it from ``cu_seqlens_k`` and ``kv_block_size``.
    qhead_per_kv : int, optional
        Number of Q heads per KV head under GQA.
    return_schedule : bool, optional
        If True, also return the sparse forward schedule object produced by the
        SM100 builder.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or tuple[torch.Tensor, torch.Tensor, object]
        ``(k2q_row_ptr, k2q_q_indices)`` or
        ``(k2q_row_ptr, k2q_q_indices, schedule)``.  CSR tensors are int32 on
        the same CUDA device as ``q2k_indices``.
    """
    if total_k is None:
        raise ValueError("build_k2q_csr requires total_k from k.shape[0]")
    if kv_block_size <= 0:
        raise ValueError(f"kv_block_size must be > 0, got {kv_block_size}")
    if q2k_indices.dtype != torch.int32:
        raise TypeError(f"q2k_indices must be torch.int32, got {q2k_indices.dtype}")
    if q2k_indices.ndim != 3:
        raise ValueError(f"q2k_indices must be rank-3, got shape {tuple(q2k_indices.shape)}")
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous with layout [head_kv, total_q, topK]")
    _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q")
    _validate_cu_seqlens(cu_seqlens_k, name="cu_seqlens_k")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape [B + 1]")
    if q2k_indices.device != cu_seqlens_q.device or q2k_indices.device != cu_seqlens_k.device:
        raise ValueError("q2k_indices, cu_seqlens_q, and cu_seqlens_k must be on the same device")
    return _K2Q_CSR_BUILDER(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        total_k=int(total_k),
        blk_kv=int(kv_block_size),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=qhead_per_kv,
        return_schedule=return_schedule,
        schedule_order=schedule_order,
    )
