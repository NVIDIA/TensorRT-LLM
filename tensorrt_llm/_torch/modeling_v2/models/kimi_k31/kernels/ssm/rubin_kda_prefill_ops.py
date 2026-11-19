# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer BT16 KDA prefill and fused short convolution.

Inputs are raw BF16 Q/K/V, gate and beta logits. BT16 performs Q/K
normalization and gate/beta activation. Recurrent state is updated directly
in its resident BF16 or FP32 pool, with no FP32 staging/writeback.
"""

import math
from functools import lru_cache

import torch

# Token granularity of the optional mid-sequence state snapshots of kda_prefill.
KDA_PREFILL_SNAPSHOT_CHUNK = 64


def _kda_snapshot_checkpoints(
    cu_seqlens: torch.Tensor,
    snapshot_rows: torch.Tensor,
    snapshot_chunks: torch.Tensor,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BT16 checkpoint offsets and target rows for per-sequence state snapshots.

    With a ``KDA_PREFILL_SNAPSHOT_CHUNK`` checkpoint interval, sequence ``b`` owns
    ``len_b // KDA_PREFILL_SNAPSHOT_CHUNK`` checkpoint slots starting at
    ``cu[b]``; slot ``cu[b] + c - 1`` holds the state after the first
    ``KDA_PREFILL_SNAPSHOT_CHUNK * c`` tokens. Only the slot selected by
    ``snapshot_chunks[b]`` maps to ``snapshot_rows[b]``; every other slot is -1
    (not stored). Built on the device without a host sync.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = torch.div(lengths, KDA_PREFILL_SNAPSHOT_CHUNK, rounding_mode="floor")
    cu_ckpts = torch.zeros_like(cu_seqlens)
    torch.cumsum(counts, dim=0, out=cu_ckpts[1:])
    capacity = total_tokens // KDA_PREFILL_SNAPSHOT_CHUNK + 1
    # The last entry is a sink for sequences without a snapshot; it stays -1.
    indices = torch.full((capacity,), -1, dtype=torch.int32, device=cu_seqlens.device)
    chunks = snapshot_chunks.to(torch.int64)
    valid = (snapshot_rows >= 0) & (chunks > 0) & (chunks <= counts)
    sink = torch.full_like(chunks, capacity - 1)
    positions = torch.where(valid, cu_ckpts[:-1] + chunks - 1, sink)
    values = torch.where(valid, snapshot_rows, torch.full_like(snapshot_rows, -1))
    indices.index_put_((positions,), values.to(torch.int32))
    return cu_ckpts, indices


@lru_cache(maxsize=32)
def _compile_prefill(device: torch.device, state_dtype: torch.dtype, lower_bound: float):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm KDA prefill before CUDA graph capture")
    import cutlass

    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_chunked_bt16 import compile

    with torch.cuda.device(device):
        return compile(
            dtype=cutlass.BFloat16,
            state_dtype=cutlass.BFloat16 if state_dtype == torch.bfloat16 else cutlass.Float32,
            gate_dtype=cutlass.BFloat16,
            safe_gate=True,
            gate_lower_bound=lower_bound,
            has_state_in=True,
            has_state_out=True,
            has_state_indices=True,
            mode="auto",
        )


def kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    seq_order: torch.Tensor | None = None,
    cu_chunks: torch.Tensor | None = None,
    snapshot_rows: torch.Tensor | None = None,
    snapshot_chunks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update unique selected V-first state rows using packed raw Q/K/V.

    The scheduler must prepare monotone int64 offsets and unique, nonnegative
    int32 slots outside execution. Empty sequences are allowed. Fresh rows
    must be reset by the caller. Optional int32 ``seq_order`` (stable descending
    length order) and ``cu_chunks`` (prefix sums of ceil(length / 16) in request
    order) are prepared once per batch and shared across layers. Both must be
    refreshed before replay when packed lengths change, retaining their storage.
    Without them, BT16 regenerates metadata on the device each invocation.

    BT16 routes between its engine and its decomposition by occupancy. BT16
    decomposition consumes aligned, head-contiguous beta views directly; the
    engine requires compact beta storage.

    ``snapshot_rows`` / ``snapshot_chunks`` (int32 ``[N]``, both or neither):
    sequence ``b`` with ``snapshot_chunks[b] = c > 0`` also stores its state
    after its first ``KDA_PREFILL_SNAPSHOT_CHUNK * c`` tokens into
    ``state_pool[snapshot_rows[b]]`` (``-1``: no snapshot for ``b``). Snapshot
    rows must be distinct from ``state_indices``; BT16 stores them through its
    state checkpoints.
    """
    device = q.device
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3), (10, 7)):
        raise ValueError("FlashInfer KDA prefill requires SM100, SM103 or SM107")
    if not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("KDA prefill requires a finite negative gate bound")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or q.shape[1] < 1:
        raise ValueError("KDA prefill requires packed [1, T, H, 128] inputs")
    if any(
        t.shape != q.shape or t.dtype != torch.bfloat16 or not t.is_contiguous()
        for t in (q, k, v, g)
    ):
        raise ValueError("KDA Q/K/V and raw gates must be contiguous BF16")
    if beta.shape != q.shape[:-1] or beta.dtype != torch.bfloat16:
        raise ValueError("KDA beta logits must be BF16 [1, T, H]")
    n, heads = state_indices.numel(), q.shape[2]
    if (
        state_pool.dtype not in (torch.bfloat16, torch.float32)
        or state_pool.ndim != 4
        or state_pool.shape[1:] != (heads, 128, 128)
        or state_pool.stride()[1:] != (128 * 128, 128, 1)
        or state_pool.stride(0) < heads * 128 * 128
        or state_pool.stride(0) * state_pool.element_size() % 16
        or state_pool.data_ptr() % 16
    ):
        raise ValueError("KDA requires aligned BF16/FP32 V-first state rows")
    if (
        state_indices.dtype != torch.int32
        or state_indices.shape != (n,)
        or not state_indices.is_contiguous()
        or cu_seqlens.dtype != torch.int64
        or cu_seqlens.shape != (n + 1,)
        or not cu_seqlens.is_contiguous()
        or n == 0
    ):
        raise ValueError("Prepare contiguous int32 indices and int64 packed offsets")
    if (
        A_log.dtype != torch.float32
        or A_log.numel() != heads
        or not A_log.is_contiguous()
        or dt_bias.dtype != torch.float32
        or dt_bias.numel() != heads * 128
        or not dt_bias.is_contiguous()
    ):
        raise ValueError("KDA gate parameters must be contiguous FP32")
    if any(
        t.device != device
        for t in (k, v, g, beta, state_pool, state_indices, cu_seqlens, A_log, dt_bias)
    ):
        raise ValueError("KDA inputs and state must share one CUDA device")
    if any(t.data_ptr() % 16 for t in (q, k, v, g, A_log, dt_bias)):
        raise ValueError("KDA inputs and gate parameters must be 16-byte aligned")
    if (seq_order is None) != (cu_chunks is None):
        raise ValueError("Provide both KDA sequence order and chunk offsets")
    if seq_order is not None:
        for tensor, shape in ((seq_order, (n,)), (cu_chunks, (n + 1,))):
            if (
                tensor.device != device
                or tensor.dtype != torch.int32
                or tensor.shape != shape
                or not tensor.is_contiguous()
            ):
                raise ValueError("KDA scheduling metadata must be contiguous CUDA int32 [N]/[N+1]")
            if tensor.untyped_storage().data_ptr() == state_pool.untyped_storage().data_ptr():
                raise ValueError("KDA state must not alias its scheduling metadata")
    if any(
        state_pool.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        for t in (q, k, v, g, beta, state_indices, cu_seqlens, A_log, dt_bias)
    ):
        raise ValueError("KDA state must not alias its immutable inputs")
    has_snapshot = snapshot_rows is not None
    if (snapshot_chunks is not None) != has_snapshot:
        raise ValueError("Provide both KDA snapshot rows and snapshot chunks")
    if has_snapshot:
        for tensor in (snapshot_rows, snapshot_chunks):
            if (
                tensor.device != device
                or tensor.dtype != torch.int32
                or tensor.shape != (n,)
                or not tensor.is_contiguous()
            ):
                raise ValueError("KDA snapshot rows/chunks must be contiguous CUDA int32 [N]")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # Match BT16's occupancy router: the engine needs a packed beta.
    use_decomp = 2 * n * heads <= sm_count
    # The merged [f_a | beta] projection has unit head stride and a padded,
    # 16-byte-aligned token pitch. Preserve packing for other input layouts.
    strided_beta_supported = (
        use_decomp
        and beta.stride(-1) == 1
        and beta.stride(1) >= heads
        and beta.stride(1) * beta.element_size() % 16 == 0
    )
    aligned_beta = beta if strided_beta_supported else beta.contiguous()
    if aligned_beta.data_ptr() % 16:
        aligned_beta = aligned_beta.clone(memory_format=torch.contiguous_format)
    output = torch.empty_like(q)
    # Torch allocations belong to the CUDA graph's private memory pool during
    # capture. Keeping scratch per layer would otherwise retain O(layers*T*H)
    # memory; eager execution reuses allocator storage between layer calls.
    compiled = _compile_prefill(device, state_pool.dtype, lower_bound)
    nonempty = min(q.shape[1], n)
    capacity = nonempty + (q.shape[1] - nonempty) // 16
    size = compiled.workspace_size_from_total_chunks(n, heads, capacity, device)
    workspace = torch.empty(size, dtype=torch.uint8, device=device) if size else None
    generate_metadata = seq_order is None
    if generate_metadata:
        seq_order = torch.empty(n, dtype=torch.int32, device=device)
        cu_chunks = torch.empty(n + 1, dtype=torch.int32, device=device)
    snapshot_kwargs = {}
    if has_snapshot:
        cu_ckpts, ckpt_rows = _kda_snapshot_checkpoints(
            cu_seqlens, snapshot_rows, snapshot_chunks, q.shape[1]
        )
        # The compiled callable delegates to the checkpoint specialization.
        snapshot_kwargs = dict(
            state_ckpt=state_pool,
            checkpoint_cu_starts=cu_ckpts,
            ckpt_interval=KDA_PREFILL_SNAPSHOT_CHUNK,
            checkpoint_state_indices=ckpt_rows,
        )
    compiled(
        q,
        k,
        v,
        g,
        A_log,
        dt_bias.reshape(heads, 128),
        aligned_beta,
        cu_seqlens,
        state_pool,
        output,
        state_pool,
        workspace,
        torch.cuda.current_stream(device).cuda_stream,
        scale=scale,
        seq_order=seq_order,
        planned_cu_chunks=cu_chunks,
        planned_total_chunks=capacity,
        generate_planned_metadata=generate_metadata,
        state_indices=state_indices,
        **snapshot_kwargs,
    )
    return output


def kda_prefill_conv(
    src: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    starts: torch.Tensor,
    slots: torch.Tensor,
    has_init: torch.Tensor,
    max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse short convolution, SiLU and raw Q/K/V layout conversion."""
    from .rubin_fused_prefill_conv import fused_prefill_conv

    shape = (1, src.shape[0], weight.shape[0] // 384, 128)
    q, k, v = (src.new_empty(shape) for _ in range(3))
    with torch.cuda.device(src.device):
        fused_prefill_conv(src, weight, conv_state, starts, slots, has_init, max_seqlen, q, k, v)
    return q, k, v
