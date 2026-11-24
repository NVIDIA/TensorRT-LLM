# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer BT16 KDA prefill and fused short convolution.

Inputs are raw BF16 Q/K/V, gate and beta logits. BT16 performs Q/K
normalization and gate/beta activation. Recurrent state is updated directly
in its resident BF16 or FP32 pool, with no FP32 staging/writeback.
"""

import math
import weakref
from functools import lru_cache
from typing import Sequence

import torch

# Token granularity of the optional mid-sequence state snapshots of kda_prefill.
KDA_PREFILL_SNAPSHOT_CHUNK = 64
# ``kda_prefill`` phases: the whole call, or its state-free half and then the rest.
KDA_PREFILL_PHASES = ("both", "prep", "chain")


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
def _compile_prefill(device: torch.device, state_dtype: torch.dtype, lower_bound: float, mode: str):
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
            mode=mode,
        )


@lru_cache(maxsize=8)
def _compile_halves(
    device: torch.device, state_dtype: torch.dtype, lower_bound: float, snapshots: bool
):
    """The BT16 decomposition's two hosts compiled on their own, ``(prep, chain)``: kernel 1 (the
    chunk factors from q / k / gate / beta, no state) and kernel 2 (the state recursion), from the
    fake operands of the single build (``_unified_fakes`` of the same specialization), with the
    constants ``_compile_prefill``'s build passes them."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm KDA prefill before CUDA graph capture")
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import kda_chunked_bt16 as bt16

    dtype = cutlass.BFloat16
    sdt = cutlass.BFloat16 if state_dtype == torch.bfloat16 else cutlass.Float32
    with torch.cuda.device(device):
        fakes, fckpt, fcuk, fcsi, _ = bt16._unified_fakes(
            dtype, sdt, True, True, snapshots, True, snapshots, dtype
        )
        fq, fk, fv, fgate, fa_log, fdt, fbeta, fcu, _, _, fsi, fcuc, _ = fakes[:13]
        fkd, fqd, fw, fqk, fdiag, finit, fout, ffinal = fakes[13:]
        ws = (fkd, fqd, fw, fqk, fdiag)
        prep = cute.compile(
            bt16.host_prep,
            fq,
            fk,
            fgate,
            fa_log,
            fdt,
            fbeta,
            fcu,
            fcuc,
            *ws,
            make_fake_stream(),
            0,  # num_ctas
            0,  # chunks_per_cta
            0,  # head_base
            0,  # launch_heads
            True,  # SAFE_GATE
            lower_bound * bt16.LOG2_E,
            bt16.THREADS_PER_CTA,
            dtype,  # gate dtype
            options="--enable-tvm-ffi",
        )
        chain = cute.compile(
            bt16.host_chain_dv2,
            fv,
            fcu,
            fcuc,
            *ws,
            fsi,
            finit,
            fout,
            ffinal,
            make_fake_stream(),
            0,  # head_base
            0,  # launch_heads
            bt16.DEFAULT_SCALE,
            fckpt,
            fcuk,
            fcsi,
            0,  # checkpoint_stride_chunks
            bt16.THREADS_PER_CTA,
            options="--enable-tvm-ffi",
        )
    return prep, chain


def _decomp_route(q: torch.Tensor, n: int) -> bool:
    """Whether BT16's occupancy rule sends a call of ``n`` sequences over ``q`` to its two-kernel
    decomposition (else the single-kernel engine)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import kda_chunked_bt16 as bt16

    return bt16._route_for_workspace(n, q.shape[2], q.device, "auto") == "decomp"


def _capacity(q: torch.Tensor, n: int) -> int:
    """The planned chunk capacity of a call over ``q`` with ``n`` sequences."""
    nonempty = min(q.shape[1], n)
    return nonempty + (q.shape[1] - nonempty) // 16


def kda_prefill_workspace(
    q: torch.Tensor, state_pool: torch.Tensor, state_indices: torch.Tensor, lower_bound: float
) -> torch.Tensor | None:
    """The workspace a ``"prep"`` / ``"chain"`` pair of ``kda_prefill`` calls over ``q`` and
    ``state_indices`` shares (None where the call runs the engine, which needs none)."""
    n = state_indices.numel()
    if not _decomp_route(q, n):
        return None
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import kda_chunked_bt16 as bt16

    size = bt16._decomp_ws_bytes(q.shape[2], _capacity(q, n))
    return torch.empty(size, dtype=torch.uint8, device=q.device)


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
    mode: str = "bt16",
    seq_order: torch.Tensor | None = None,
    cu_chunks: torch.Tensor | None = None,
    snapshot_rows: torch.Tensor | None = None,
    snapshot_chunks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    phase: str = "both",
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update unique selected V-first state rows using packed raw Q/K/V.

    The scheduler must prepare monotone int64 offsets and unique, nonnegative
    int32 slots outside execution. Empty sequences are allowed. Fresh rows
    must be reset by the caller. Optional int32 ``seq_order`` (stable descending
    length order) and ``cu_chunks`` (prefix sums of ceil(length / 16) in request
    order) are prepared once per batch and shared across layers. Both must be
    refreshed before replay when packed lengths change, retaining their storage.
    Without them, BT16 regenerates metadata on the device each invocation.

    ``bt16`` keeps BT16's own engine/decomposition routing; ``engine`` and
    ``decomp`` force one of the two. BT16 decomposition consumes aligned,
    head-contiguous beta views directly; the engine requires compact beta
    storage.

    ``snapshot_rows`` / ``snapshot_chunks`` (int32 ``[N]``, both or neither):
    sequence ``b`` with ``snapshot_chunks[b] = c > 0`` also stores its state
    after its first ``KDA_PREFILL_SNAPSHOT_CHUNK * c`` tokens into
    ``state_pool[snapshot_rows[b]]`` (``-1``: no snapshot for ``b``). Snapshot
    rows must be distinct from ``state_indices``; BT16 stores them through its
    state checkpoints.

    ``out``: a contiguous BF16 tensor shaped like ``q`` the output is written
    into (a new one when None); returned.

    ``phase`` (``KDA_PREFILL_PHASES``): ``"both"`` runs the call. ``"prep"``
    issues only its part that reads no state and ``"chain"`` the rest: on
    BT16's decomposition route its kernel 1 (the chunk factors) and kernel 2
    (the state recursion), launched through their own hosts with the operands
    the single build gives them; on the engine route ``"prep"`` issues nothing
    and ``"chain"`` the whole call. Called in that order with the same
    arguments, ``mode="bt16"``, the planned ``seq_order`` / ``cu_chunks`` and one
    ``workspace`` (``kda_prefill_workspace``), the two run the kernels of one
    ``"both"`` call on the same inputs. Between them the caller may run anything
    that leaves ``q``, ``k``, ``g``, ``beta``, the metadata and the workspace
    alone; the state may change. ``"both"`` allocates its own workspace.
    """
    device = q.device
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3), (10, 7)):
        raise ValueError("FlashInfer KDA prefill requires SM100, SM103 or SM107")
    if (
        mode not in ("bt16", "engine", "decomp")
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
    if phase not in KDA_PREFILL_PHASES:
        raise ValueError(f"KDA prefill phase {phase!r}: one of {KDA_PREFILL_PHASES}")
    if phase != "both" and (mode != "bt16" or seq_order is None):
        raise ValueError(
            "A KDA prefill phase needs mode bt16 and the planned seq_order / cu_chunks"
        )
    if phase == "both" and workspace is not None:
        raise ValueError("A whole KDA prefill call allocates its own workspace")
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
    # Match BT16's occupancy router. Forced engine calls must still pack beta.
    use_decomp = mode == "decomp" or (mode == "bt16" and 2 * n * heads <= sm_count)
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
    if out is not None and (
        out.shape != q.shape
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.device != device
        or out.data_ptr() % 16
    ):
        raise ValueError("KDA output must be a contiguous, 16-byte aligned BF16 tensor like q")
    output = torch.empty_like(q) if out is None else out
    if phase != "both":
        _kda_prefill_phase(
            phase,
            q,
            k,
            v,
            g,
            aligned_beta,
            state_pool,
            state_indices,
            cu_seqlens,
            A_log,
            dt_bias,
            scale,
            lower_bound,
            seq_order,
            cu_chunks,
            snapshot_rows,
            snapshot_chunks,
            output,
            workspace,
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


def _kda_prefill_phase(
    phase: str,
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
    seq_order: torch.Tensor,
    cu_chunks: torch.Tensor,
    snapshot_rows: torch.Tensor | None,
    snapshot_chunks: torch.Tensor | None,
    output: torch.Tensor,
    workspace: torch.Tensor | None,
) -> None:
    """``kda_prefill``'s ``"prep"`` or ``"chain"`` half (checked arguments; ``beta`` aligned as
    the whole call aligns it). On the decomposition route each half launches its kernel with the
    operands BT16's host passes it in the whole call; on the engine route ``"chain"`` is the whole
    call."""
    n, heads = state_indices.numel(), q.shape[2]
    if not _decomp_route(q, n):
        if workspace is not None:
            raise ValueError("A KDA prefill phase on the engine route takes no workspace")
        if phase == "chain":
            kda_prefill(
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
                "bt16",
                seq_order,
                cu_chunks,
                snapshot_rows,
                snapshot_chunks,
                output,
            )
        return
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import kda_chunked_bt16 as bt16

    capacity = _capacity(q, n)
    if workspace is None or workspace.numel() != bt16._decomp_ws_bytes(heads, capacity):
        raise ValueError("A KDA prefill phase needs its kda_prefill_workspace")
    has_snapshot = snapshot_rows is not None
    prep, chain = _compile_halves(q.device, state_pool.dtype, lower_bound, has_snapshot)
    parts = bt16._partition_workspace(workspace, heads, capacity)
    ws = (parts["kd"], parts["qd"], parts["w"], parts["qk"], parts["diag"])
    stream = torch.cuda.current_stream(q.device).cuda_stream
    if phase == "prep":
        num_ctas, cpc = bt16._k1_grid(capacity, heads)
        prep(
            q,
            k,
            g,
            A_log,
            dt_bias.reshape(heads, 128),
            beta,
            cu_seqlens,
            cu_chunks,
            *ws,
            stream,
            num_ctas,
            cpc,
            0,
            heads,
        )
        return
    ckpt = (None, None, None, 0)
    if has_snapshot:
        cu_ckpts, ckpt_rows = _kda_snapshot_checkpoints(
            cu_seqlens, snapshot_rows, snapshot_chunks, q.shape[1]
        )
        ckpt = (state_pool, cu_ckpts, ckpt_rows, KDA_PREFILL_SNAPSHOT_CHUNK // bt16.BT)
    chain(
        v,
        cu_seqlens,
        cu_chunks,
        *ws,
        state_indices,
        state_pool,
        output,
        state_pool,
        stream,
        0,
        heads,
        scale,
        *ckpt,
    )


def seed_cu_seqlens(cu_seqlens: torch.Tensor, contents: Sequence[int]) -> None:
    """Hand the BT16 engine the contents of ``cu_seqlens`` (a live device tensor the caller passes
    to every layer of a step), so that it reads them without a device-to-host copy (its memo keys
    on this tensor object and its version)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import kda_chunked_bt16

    try:
        version = cu_seqlens._version
    except RuntimeError:  # inference tensors track no version counter
        version = None
    kda_chunked_bt16._CU_CONTENTS_MEMO[cu_seqlens.data_ptr()] = (
        weakref.ref(cu_seqlens),
        version,
        tuple(int(x) for x in contents),
    )


# ``kda_prefill_segments`` ``seq_flags``: the initial state (zero, identity) and a zero v.
KDA_SEQ_INIT_ZERO = 1
KDA_SEQ_INIT_IDENTITY = 2
KDA_SEQ_ZERO_V = 4


@lru_cache(maxsize=8)
def _compile_segments(device: torch.device, lower_bound: float):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm KDA prefill before CUDA graph capture")
    import cutlass

    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.kda_chunked_bt16 import compile

    with torch.cuda.device(device):
        return compile(
            dtype=cutlass.BFloat16,
            state_dtype=cutlass.Float32,
            gate_dtype=cutlass.BFloat16,
            safe_gate=True,
            gate_lower_bound=lower_bound,
            has_state_in=True,
            has_state_out=True,
            has_state_indices=True,
            mode="engine",
            has_src_rows=True,
            has_seq_flags=True,
        )


def kda_prefill_segments(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    states: torch.Tensor,
    state_rows: torch.Tensor,
    cu_seqlens: torch.Tensor,
    src_rows: torch.Tensor,
    seq_flags: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    seq_order: torch.Tensor,
    cu_chunks: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    """BT16 engine over sequences that read shared input rows; returns ``[1, out_rows, H, 128]``.

    Sequence ``b`` reads q / k / g / beta rows ``[src_rows[b], src_rows[b] + len_b)``, v and the
    output rows per ``cu_seqlens``; its initial state is synthesized per ``seq_flags[b]``
    (``KDA_SEQ_INIT_ZERO`` / ``KDA_SEQ_INIT_IDENTITY``, ``KDA_SEQ_ZERO_V`` skips v) and its final
    state goes to the FP32 row ``states[state_rows[b]]``. Every sequence names a synthesized
    initial state, so no row of ``states`` is read. ``out_rows`` is ``cu_seqlens[-1]`` (host).
    """
    device = q.device
    n, heads = state_rows.numel(), q.shape[2]
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3), (10, 7)):
        raise ValueError("FlashInfer KDA prefill requires SM100, SM103 or SM107")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("KDA prefill requires packed [1, T, H, 128] inputs")
    if any(
        t.shape[0] != 1 or t.shape[2:] != q.shape[2:] or t.dtype != torch.bfloat16
        for t in (k, v, g)
    ) or any(not t.is_contiguous() for t in (q, k, v, g)):
        raise ValueError("KDA Q/K/V and raw gates must be contiguous BF16 [1, T, H, 128]")
    if k.shape != q.shape or g.shape != q.shape or beta.shape != q.shape[:-1]:
        raise ValueError("KDA q, k, g and beta must cover the same rows")
    if (
        states.dtype != torch.float32
        or states.ndim != 4
        or states.shape[1:] != (heads, 128, 128)
        or not states.is_contiguous()
    ):
        raise ValueError("KDA segment states must be contiguous FP32 [rows, H, 128, 128]")
    for tensor, dtype, shape in (
        (state_rows, torch.int32, (n,)),
        (src_rows, torch.int32, (n,)),
        (seq_flags, torch.int32, (n,)),
        (cu_seqlens, torch.int64, (n + 1,)),
        (seq_order, torch.int32, (n,)),
        (cu_chunks, torch.int32, (n + 1,)),
    ):
        if tensor.dtype != dtype or tensor.shape != shape or not tensor.is_contiguous():
            raise ValueError("KDA segment metadata must be contiguous int32 / int64 vectors")
    if n == 0 or out_rows < q.shape[1]:
        raise ValueError("KDA segments: no sequence, or fewer output rows than input rows")
    output = q.new_empty((1, out_rows) + tuple(q.shape[2:]))
    compiled = _compile_segments(device, lower_bound)
    nonempty = min(out_rows, n)
    capacity = nonempty + (out_rows - nonempty) // 16
    size = compiled.workspace_size_from_total_chunks(n, heads, capacity, device)
    workspace = torch.empty(size, dtype=torch.uint8, device=device) if size else None
    compiled(
        q,
        k,
        v,
        g,
        A_log,
        dt_bias.reshape(heads, 128),
        beta.contiguous(),
        cu_seqlens,
        states,
        output,
        states,
        workspace,
        torch.cuda.current_stream(device).cuda_stream,
        scale=scale,
        seq_order=seq_order,
        planned_cu_chunks=cu_chunks,
        planned_total_chunks=capacity,
        generate_planned_metadata=False,
        state_indices=state_rows,
        src_rows=src_rows,
        seq_flags=seq_flags,
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
