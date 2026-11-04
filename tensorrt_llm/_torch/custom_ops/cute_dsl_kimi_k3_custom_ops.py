# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer BT16/Small-BH KDA prefill and fused short convolution.

Inputs are raw BF16 Q/K/V, gate and beta logits. BT16 performs Q/K
normalization and gate/beta activation. Recurrent state is updated directly
in its resident BF16 or FP32 pool, with no FP32 staging/writeback.
"""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _compile_prefill(device: torch.device, state_dtype: torch.dtype, lower_bound: float, mode: str):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm KDA prefill before CUDA graph capture")
    import cutlass

    from ..attention.kernels.blackwell.kimi_k3_kda.kda_chunked_bt16 import compile

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
            mode=mode,
        )


@torch.library.custom_op("trtllm::kda_prefill", mutates_args=("state_pool",), device_types="cuda")
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
    mode: str = "auto",
    max_seqlen: int = 0,
    seq_order: torch.Tensor | None = None,
    cu_chunks: torch.Tensor | None = None,
    sm_reserve: int = 0,
) -> torch.Tensor:
    """Update unique selected V-first state rows using packed raw Q/K/V.

    The scheduler must prepare monotone int64 offsets and unique, nonnegative
    int32 slots outside execution. Empty sequences are allowed. Fresh rows
    must be reset by the caller. Optional int32 ``seq_order`` (stable descending
    length order) and ``cu_chunks`` (prefix sums of ceil(length / 16) in request
    order) are prepared once per batch and shared across layers. Both must be
    refreshed before replay when packed lengths change, retaining their storage.
    Without them, BT16 regenerates metadata on the device each invocation.

    Auto and ``bt16`` use BT16's engine/decomposition selection. ``small-bh``
    forces Small-BH, which auto never selects: with BF16 state its output
    diverges from the reference on long sequences whose heads mix near-unit
    and strong per-channel decay, until it overflows to NaN (a repeated token
    over about a thousand steps is enough). ``max_seqlen`` is an optional
    upper bound for all replays, or zero to use the total token capacity.
    Small-BH and BT16 decomposition consume aligned, head-contiguous beta
    views directly; the engine requires compact beta storage.

    ``sm_reserve`` SMs, held by a kernel resident for the whole step, are left
    out of the Small-BH persistent grid; blocks beyond the free SMs would wait
    behind the whole first wave. The routing above stays on the device SM count
    so it matches BT16's own router.
    """
    device = q.device
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3), (10, 7)):
        raise ValueError("FlashInfer KDA prefill requires SM100, SM103 or SM107")
    if (
        mode not in ("auto", "bt16", "small-bh", "engine", "decomp")
        or not math.isfinite(lower_bound)
        or lower_bound >= 0
    ):
        raise ValueError("KDA prefill requires a finite negative gate bound and a supported mode")
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
    if max_seqlen < 0 or max_seqlen > q.shape[1]:
        raise ValueError("max_seqlen must be zero or a sequence-length bound within total tokens")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    if not 0 <= sm_reserve < sm_count:
        raise ValueError(f"sm_reserve={sm_reserve} must leave at least one of {sm_count} SMs")
    small_bh_eligible = state_pool.dtype == torch.bfloat16 and q.shape[1] > 1
    if mode == "small-bh" and not small_bh_eligible:
        raise ValueError("Small-BH requires BF16 state and more than one packed token")
    # Auto stays on BT16 (see the docstring): Small-BH is opt-in only.
    use_small_bh = mode == "small-bh"
    # Match BT16's occupancy router. Forced engine calls must still pack beta.
    use_decomp = mode == "decomp" or (mode in ("auto", "bt16") and 2 * n * heads <= sm_count)
    # The merged [f_a | beta] projection has unit head stride and a padded,
    # 16-byte-aligned token pitch. Preserve packing for other input layouts.
    strided_beta_supported = (
        (use_small_bh or use_decomp)
        and beta.stride(-1) == 1
        and beta.stride(1) >= heads
        and beta.stride(1) * beta.element_size() % 16 == 0
    )
    aligned_beta = beta if strided_beta_supported else beta.contiguous()
    if aligned_beta.data_ptr() % 16:
        aligned_beta = aligned_beta.clone(memory_format=torch.contiguous_format)
    output = torch.empty_like(q)
    if use_small_bh:
        from ..attention.kernels.blackwell.kimi_k3_kda.kda_chunked_bt16_small_bh import (
            chunk_kda_fwd,
        )

        with torch.cuda.device(device):
            chunk_kda_fwd(
                q[0],
                k[0],
                v[0],
                g[0],
                aligned_beta[0],
                dt_bias.reshape(heads, 128),
                A_log.reshape(heads),
                scale=scale,
                gate_scale=lower_bound,
                gate_lower_bound=True,
                return_state=True,
                transpose_state=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen or q.shape[1],
                output=output[0],
                state_input=state_pool,
                state_output=state_pool,
                state_indices=state_indices,
                max_active_clusters=sm_count - sm_reserve,
            )
        return output
    # Torch allocations belong to the CUDA graph's private memory pool during
    # capture. Keeping scratch per layer would otherwise retain O(layers*T*H)
    # memory; eager execution reuses allocator storage between layer calls.
    compiled = _compile_prefill(
        device, state_pool.dtype, lower_bound, "auto" if mode == "bt16" else mode
    )
    nonempty = min(q.shape[1], n)
    capacity = nonempty + (q.shape[1] - nonempty) // 16
    size = compiled.workspace_size_from_total_chunks(n, heads, capacity, device)
    workspace = torch.empty(size, dtype=torch.uint8, device=device) if size else None
    generate_metadata = seq_order is None
    if generate_metadata:
        seq_order = torch.empty(n, dtype=torch.int32, device=device)
        cu_chunks = torch.empty(n + 1, dtype=torch.int32, device=device)
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
    )
    return output


@kda_prefill.register_fake
def _kda_prefill_fake(
    q,
    k,
    v,
    g,
    beta,
    state_pool,
    state_indices,
    cu_seqlens,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    mode="auto",
    max_seqlen=0,
    seq_order=None,
    cu_chunks=None,
    sm_reserve=0,
):
    return torch.empty_like(q)


@torch.library.custom_op(
    "trtllm::kda_prefill_conv", mutates_args=("conv_state",), device_types="cuda"
)
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
    from ..attention.kernels.blackwell.kimi_k3_kda.fused_prefill_conv import fused_prefill_conv

    shape = (1, src.shape[0], weight.shape[0] // 384, 128)
    q, k, v = (src.new_empty(shape) for _ in range(3))
    with torch.cuda.device(src.device):
        fused_prefill_conv(src, weight, conv_state, starts, slots, has_init, max_seqlen, q, k, v)
    return q, k, v


@kda_prefill_conv.register_fake
def _kda_prefill_conv_fake(src, weight, conv_state, starts, slots, has_init, max_seqlen):
    shape = (1, src.shape[0], weight.shape[0] // 384, 128)
    return tuple(src.new_empty(shape) for _ in range(3))
