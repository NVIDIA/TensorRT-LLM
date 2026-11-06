# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM TP4 one-shot Lamport all-gather along the last dimension.

The GLM-5.2 vocabulary-parallel ``lm_head`` leaves each TP rank with a BF16
``[M, N]`` shard of the logits (``N = 38720`` at TP4).  The serving stack then
gathers the shards across the four ranks and concatenates them along the
vocabulary, and the logits processor widens the result to FP32.  This kernel
does the three steps in one launch:

    input_local[M, N] (BF16)
      -> one-shot Lamport exchange across WORLD_SIZE ranks
      -> output[M, WORLD_SIZE * N]  (BF16, or FP32 when OUT_FP32)
         output[r, s * N + c] = input_s[r, c]

It is the all-gather twin of the pure all-reduce in ``tp4_allreduce_fused_rms.py`` and runs on
the **same** workspace and signal state: the three-slot peer payload buffers
(``[3, WORLD_SIZE, workspace_numel]`` BF16, symmetric memory), the in-band
``-0.0`` sentinel readiness protocol, and the device-managed slot rotation.
Only the epilogue differs: the all-reduce sums the peer vectors, this kernel stores each
one at its source rank's column block.  Because the slot protocol is shared, an
all-gather launch interleaves with the all-reduce launches around it in one
CUDA graph without any extra state.

Protocol, per thread and per 8-element BF16 vector of the local shard:

1. read the slot counter from the device signal state; clear the slot that the
   launch two steps ahead will use (``clear_slot``) back to the sentinel, for
   as many vectors as its last user published;
2. publish the local vector into slot ``ping`` / region ``LOCAL_RANK`` of every
   peer's buffer (the local rank's own region is never used: its vector goes
   straight to the output);
3. poll the local buffer's ``ping`` regions of every peer until no lane holds
   the sentinel, then store each peer vector into the output.

The sentinel is bitwise BF16 ``-0.0``.  A published ``-0.0`` would read as not
ready, so every lane equal to ``-0.0`` is written as ``+0.0`` -- both to the
peers and to this rank's own output block, so the four ranks hold bitwise
identical outputs.  This is the only difference from an exact gather (NCCL
keeps the sign of zero); ``-0.0 == +0.0`` numerically, so argmax, softmax and
sampling are unaffected.

The workspace must hold the shard: ``1 <= M`` and ``M * N <= workspace_numel``
(an empty batch would be an empty grid, which CUDA rejects).  The caller checks
both on the host (every rank sees the same ``M``, so every rank takes the same
branch).  Readiness by separate ready flags is not implemented; compile
refuses ``use_ready_flags=True``.

"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from . import tp4_allreduce_fused_rms as _ar

# Shared with the all-reduce so the two kernels agree on the workspace and slot state.
_PING_PONG_SIZE = _ar._PING_PONG_SIZE
_BF16_VEC_SIZE = _ar._BF16_VEC_SIZE
_ALIGNMENT_BYTES = _ar._ALIGNMENT_BYTES
_READY_FLAG_ALIGNMENT_BYTES = _ar._READY_FLAG_ALIGNMENT_BYTES
_NEGATIVE_ZERO_BF16_BITS = _ar._NEGATIVE_ZERO_BF16_BITS
_SIGNAL_COUNTER_INDEX = _ar._SIGNAL_COUNTER_INDEX
_SIGNAL_VALUE_INDEX = _ar._SIGNAL_VALUE_INDEX
_SIGNAL_SLOT_COUNTS_INDEX = _ar._SIGNAL_SLOT_COUNTS_INDEX
_SIGNAL_STATE_SIZE = _ar._SIGNAL_STATE_SIZE

# The lm_head vocabulary shard at TP4: 154880 / 4.
LM_HEAD_SHARD_N = 38720
_FP32_PER_STORE = 4
_THREADS_PER_BLOCK = 256
_THREADS_PER_BLOCK_CHOICES = (128, 256, 512)
_PDL_RELEASE_POLICIES = ("output_safe", "entry")

CompiledKernel = Callable[..., object]


@cute.jit
def _sanitize_bf16x8(values):
    """Return ``values`` with every bitwise ``-0.0`` lane replaced by ``+0.0``."""

    neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
    clean = cutlass.Array(cutlass.BFloat16, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES)
    for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        lane_value = values[lane]
        if lane_value.bitcast(cutlass.Int16) == neg_zero_bits:
            lane_value = cutlass.BFloat16(0.0)
        clean[lane] = lane_value
    return clean


@cute.jit
def _store_output_vector(output_array, offset, values, OUT_FP32: cutlass.Constexpr[bool]):
    """Store one 8-lane BF16 vector at element ``offset`` of the output, widened
    to FP32 (two 16-byte stores) when ``OUT_FP32``."""

    if cutlass.const_expr(OUT_FP32):
        low = cutlass.Array(cutlass.Float32, _FP32_PER_STORE, alignment=_ALIGNMENT_BYTES)
        high = cutlass.Array(cutlass.Float32, _FP32_PER_STORE, alignment=_ALIGNMENT_BYTES)
        for lane in cutlass.range_constexpr(_FP32_PER_STORE):
            low[lane] = values[lane].to(cutlass.Float32)
            high[lane] = values[lane + _FP32_PER_STORE].to(cutlass.Float32)
        output_array.store(
            low[0:_FP32_PER_STORE, cutlass.align(_ALIGNMENT_BYTES)],
            offset,
            _FP32_PER_STORE,
            alignment=_ALIGNMENT_BYTES,
        )
        output_array.store(
            high[0:_FP32_PER_STORE, cutlass.align(_ALIGNMENT_BYTES)],
            offset + cutlass.Int64(_FP32_PER_STORE),
            _FP32_PER_STORE,
            alignment=_ALIGNMENT_BYTES,
        )
    else:
        output_array.store(
            values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)],
            offset,
            _BF16_VEC_SIZE,
            alignment=_ALIGNMENT_BYTES,
        )


@cute.kernel
def _glm_tp_allgather_bf16_oneshot_lamport_kernel(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    signal: cutlass.Int64,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    OUT_FP32: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_ENTRY: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    bdim, _, _ = cute.arch.block_dim()

    input_array = cutlass.make_array_view(input_tensor)
    output_array = cutlass.make_array_view(output_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]

    rows = cute.size(input_tensor, mode=[0])
    active_numel = rows * N
    workspace_numel_i64 = cutlass.Int64(cute.size(buffers[0], mode=[2]))
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_VALUE_INDEX)).llvm_ptr

    # The slot state and the input are both predecessor-owned.
    if cutlass.const_expr(USE_PDL):
        cute.arch.griddepcontrol_wait()
    # ``entry`` releases the successor right away: its own entry wait still
    # blocks until this grid has completed, so it cannot read the slot state
    # (or this grid's output) early -- it only gets its launch latency hidden.
    if cutlass.const_expr(USE_PDL and PDL_RELEASE_ENTRY):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()

    signal_i32 = cute.arch.load(state_signal_ptr, cutlass.Int32, sem="acquire", scope="gpu")
    signal_i64 = cutlass.Int64(signal_i32)
    ping_slot = signal_i64 % _PING_PONG_SIZE
    clear_slot = (signal_i64 + cutlass.Int64(1)) % _PING_PONG_SIZE
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64
    clear_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64

    neg_zero_vec = cutlass.Array(cutlass.BFloat16, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES)
    neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
    for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        neg_zero_vec[lane] = neg_zero_bits.bitcast(cutlass.BFloat16)

    # 1. Re-arm the slot two launches ahead.  Its last user (an all-reduce or
    #    an all-gather) published how many vectors its peers wrote into it.
    #    The local rank's own region is skipped: no kernel reads its own region
    #    across launches (the all-reduce reads back what the same thread just wrote).
    clear_count_ptr = (
        signal_state.iterator + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX) + cutlass.Int64(clear_slot)
    ).llvm_ptr
    clear_vectors = cute.arch.load(clear_count_ptr, cutlass.Int32, sem="acquire", scope="gpu")
    clear_vector_idx = bidx * bdim + tidx
    clear_stride = grid_x * bdim
    while clear_vector_idx < clear_vectors:
        clear_vector_idx_i64 = cutlass.Int64(clear_vector_idx)
        for source_rank in cutlass.range_constexpr(WORLD_SIZE):
            if source_rank != LOCAL_RANK:
                clear_offset = (
                    clear_base
                    + cutlass.Int64(source_rank) * workspace_numel_i64
                    + clear_vector_idx_i64 * cutlass.Int64(_BF16_VEC_SIZE)
                )
                local_buffer.store(
                    neg_zero_vec[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)],
                    clear_offset,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                    is_volatile=True,
                )
        clear_vector_idx = clear_vector_idx + clear_stride

    base_idx = (bidx * bdim + tidx) * _BF16_VEC_SIZE
    if base_idx < active_numel:
        base_idx_i64 = cutlass.Int64(base_idx)
        row_i64 = base_idx_i64 // cutlass.Int64(N)
        col_i64 = base_idx_i64 - row_i64 * cutlass.Int64(N)
        out_row_base = row_i64 * cutlass.Int64(WORLD_SIZE * N) + col_i64

        input_values = input_array.load(base_idx, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES)
        clean_values = _sanitize_bf16x8(input_values)

        # 2. Publish to every peer, in rank-rotated order.
        for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
            target_rank = PEER_ORDER[peer_ordinal]
            if target_rank != LOCAL_RANK:
                write_offset = (
                    ping_base + cutlass.Int64(LOCAL_RANK) * workspace_numel_i64 + base_idx_i64
                )
                buffer_arrays[target_rank].store(
                    clean_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)],
                    write_offset,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                    is_volatile=True,
                )

        # Own block: the same sanitized bits the peers will store for it.
        _store_output_vector(
            output_array,
            out_row_base + cutlass.Int64(LOCAL_RANK * N),
            clean_values,
            OUT_FP32,
        )

        # 3. Receive every peer's vector and place it at its column block.
        for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
            source_rank = PEER_ORDER[peer_ordinal]
            if source_rank != LOCAL_RANK:
                read_offset = (
                    ping_base + cutlass.Int64(source_rank) * workspace_numel_i64 + base_idx_i64
                )
                peer_values = local_buffer.load(
                    read_offset,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                    is_volatile=True,
                )
                while not _ar._all_bf16x8_lanes_ready(peer_values):
                    peer_values = local_buffer.load(
                        read_offset,
                        _BF16_VEC_SIZE,
                        alignment=_ALIGNMENT_BYTES,
                        is_volatile=True,
                    )
                peer_copy = cutlass.Array(
                    cutlass.BFloat16, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES
                )
                for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                    peer_copy[lane] = peer_values[lane]
                _store_output_vector(
                    output_array,
                    out_row_base + cutlass.Int64(source_rank * N),
                    peer_copy,
                    OUT_FP32,
                )

    # ``output_safe``: the successor may start once every output store of this
    # CTA has been issued.  Its entry wait still covers completion.
    if cutlass.const_expr(USE_PDL and not PDL_RELEASE_ENTRY):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()

    # 4. Advance the slot state: the last CTA to arrive publishes how many
    #    vectors the peers wrote into ``ping`` (for its future clear), resets the
    #    counter, and rotates the signal.  Every CTA has read the signal above.
    if tidx == 0:
        active_vectors = (cutlass.Int32(rows) * cutlass.Int32(N)) // cutlass.Int32(_BF16_VEC_SIZE)
        _ar._advance_signal_if_last_cta(
            state_counter_ptr,
            state_signal_ptr,
            signal_state,
            signal_i32,
            ping_slot,
            active_vectors,
            grid_x,
        )


@cute.jit
def _glm_tp_allgather_bf16_oneshot_lamport(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    signal: cutlass.Int64,
    stream: cuda.CUstream,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    OUT_FP32: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_ENTRY: cutlass.Constexpr[bool],
) -> None:
    rows = input_tensor.shape[0]
    num_blocks = cute.ceil_div(rows * N, THREADS_PER_BLOCK * _BF16_VEC_SIZE)
    _glm_tp_allgather_bf16_oneshot_lamport_kernel(
        input_tensor,
        output_tensor,
        buffers,
        ready_flags,
        signal_state,
        signal,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        PEER_ORDER,
        OUT_FP32,
        USE_PDL,
        PDL_RELEASE_ENTRY,
    ).launch(
        grid=(num_blocks, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


def compile_for_rank(
    local_rank: int = 0,
    world_size: int = 4,
    device_id: int = 0,
    n: int = LM_HEAD_SHARD_N,
    out_fp32: bool = True,
    threads_per_block: int = _THREADS_PER_BLOCK,
    use_ready_flags: bool = False,
    use_pdl: bool = False,
    pdl_release_policy: str = "output_safe",
    rotate_peer_order: bool = True,
) -> CompiledKernel:
    """Compile the all-gather for one rank, symbolic in the row count.

    The launch signature matches the pure all-reduce of ``tp4_allreduce_fused_rms`` with the
    output widened to ``[M, world_size * n]``:

        fn(input[M, n] bf16, output[M, world_size * n] bf16|fp32,
           peer_buffers, peer_ready_flags, signal_state, Int64 signal, stream)

    ``peer_ready_flags`` is accepted (so adapters pass the all-reduce's operand tail
    unchanged) and never read; ``signal`` is dead because the slot counter
    lives in ``signal_state``.
    """

    if use_ready_flags:
        raise ValueError(
            "the all-gather implements the in-band -0.0 sentinel protocol only; "
            "the workspace must be compiled and armed with use_ready_flags=False"
        )
    if n % _BF16_VEC_SIZE != 0:
        raise ValueError(f"n must be divisible by {_BF16_VEC_SIZE}, got {n}")
    if threads_per_block not in _THREADS_PER_BLOCK_CHOICES:
        raise ValueError(
            f"threads_per_block must be one of {_THREADS_PER_BLOCK_CHOICES}, "
            f"got {threads_per_block}"
        )
    if pdl_release_policy not in _PDL_RELEASE_POLICIES:
        raise ValueError(
            f"pdl_release_policy must be one of {_PDL_RELEASE_POLICIES}, got {pdl_release_policy!r}"
        )
    if not 0 <= local_rank < world_size:
        raise ValueError(f"local_rank {local_rank} outside world size {world_size}")
    cutlass.cuda.initialize_cuda_context(device_id)

    sym_m = cute.sym_int32()
    input_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    output_tensor = make_fake_compact_tensor(
        cutlass.Float32 if out_fp32 else cutlass.BFloat16,
        (sym_m, world_size * n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    workspace_numel = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    workspace_num_vectors = workspace_numel // _BF16_VEC_SIZE
    fake_buffers = []
    fake_ready_flags = []
    for _ in range(world_size):
        fake_buffers.append(
            make_fake_compact_tensor(
                cutlass.BFloat16,
                (_PING_PONG_SIZE, world_size, workspace_numel),
                stride_order=(2, 1, 0),
                assumed_align=_ALIGNMENT_BYTES,
            )
        )
        fake_ready_flags.append(
            make_fake_compact_tensor(
                cutlass.Int32,
                (_PING_PONG_SIZE, world_size, workspace_num_vectors),
                stride_order=(2, 1, 0),
                assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
            )
        )
    fake_signal_state = make_fake_compact_tensor(
        cutlass.Int32,
        (_SIGNAL_STATE_SIZE,),
        assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
    )
    peer_order = tuple(range(world_size))
    if rotate_peer_order:
        peer_order = tuple((local_rank + k) % world_size for k in range(world_size))
    return cute.compile(
        _glm_tp_allgather_bf16_oneshot_lamport,
        input_tensor,
        output_tensor,
        fake_buffers,
        fake_ready_flags,
        fake_signal_state,
        cutlass.Int64(0),
        make_fake_stream(),
        n,
        local_rank,
        world_size,
        threads_per_block,
        peer_order,
        out_fp32,
        use_pdl,
        pdl_release_policy == "entry",
        options="--enable-tvm-ffi",
    )
