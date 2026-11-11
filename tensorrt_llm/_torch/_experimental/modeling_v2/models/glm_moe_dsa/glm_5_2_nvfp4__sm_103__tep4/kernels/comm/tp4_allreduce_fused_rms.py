# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM TP allreduce and fused GLM residual/RMSNorm CuTe DSL examples.

This example models the GLM tensor-parallel output paths that follow local
projection GEMMs. Each TP rank owns a BF16 partial hidden vector for the same
token rows. The pure kernel only performs the TP sum. The fused kernel first
sums those partial vectors across TP ranks with a one-shot Lamport allreduce,
then immediately applies the residual add and the row-wise RMSNorm that consume
the reduced hidden state.

The pure kernel computes, per rank:

    input_local[M, 6144]
      -> one-shot Lamport allreduce across four TP ranks
      -> output[M, 6144]

The fused kernel computes, per rank:

    input_local[M, 6144]
      -> one-shot Lamport allreduce across four TP ranks
      -> + residual[M, 6144]
      -> residual_output[M, 6144]
      -> per-row RMSNorm with norm_weight[6144]
      -> norm_output[M, 6144]

The fused-out-quant variant keeps the same BF16 outputs and also emits the
linear-layout NVFP4 activation payload consumed by the GLM MoE backend:

    norm_output[M, 6144]
      -> fp4_output[M, 3072] + scale_output[M, 384]

The kernels run on four GPUs with symmetric peer memory. They compile through
TVM FFI, so the environment needs ``apache-tvm-ffi`` installed.

The default path keeps payload data and readiness state separate: BF16 vectors
move through the Lamport payload buffers, while int32 ready flags keyed by
``(slot, source_rank, vector_index)`` publish vector availability.  The split
row fused path can also compile an experimental in-band sentinel variant that
uses bitwise BF16 ``-0.0`` in the payload buffer as the not-ready marker, closer
to TensorRT-LLM's Lamport backend.

Runs on SM100+ (Blackwell) GPUs.  Rows are BF16 with a runtime M; the served
configuration is N=6144 with a TP world size of 4.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

try:
    from cutlass._mlir import ir as _ir
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import T as _T
    from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op
except ImportError:  # pragma: no cover - older DSL packages
    _ir = None
    _llvm = None
    _T = None
    _dsl_user_op = None


_DEFAULT_N = 6144
_THREADS_PER_BLOCK = 128
_WARPS_PER_BLOCK = _THREADS_PER_BLOCK // 32
_FUSED_CTAS_PER_ROW = 4
_SPLIT_ROW_THREADS_PER_BLOCK = 192
_PURE_THREADS_PER_BLOCK_CHOICES = (128, 192)
_FUSED_THREADS_PER_BLOCK_CHOICES = (128, 192, 256, 384)
_OUT_QUANT_THREADS_PER_BLOCK_CHOICES = (128, 192, 256, 384)
_PING_PONG_SIZE = 3
_BF16_VEC_SIZE = 8
_SCALE_BLOCK = 16
_ALIGNMENT_BYTES = 16
_READY_FLAG_ALIGNMENT_BYTES = 4
_NEGATIVE_ZERO_BF16_BITS = -32768
_RMS_EPS = 1e-6
_USES_READY_FLAGS = True
_USES_DEVICE_SIGNAL_STATE = True
_PDL_RELEASE_POLICIES = ("strict", "output_safe", "pre_norm", "entry")
_FUSED_PDL_RELEASE_POLICIES = (
    "strict",
    "output_safe",
    "pre_norm",
    "entry",
)
_SIGNAL_COUNTER_INDEX = 0
_SIGNAL_VALUE_INDEX = 1
_SIGNAL_SLOT_COUNTS_INDEX = 2
_SIGNAL_RELEASE_EPOCH_INDEX = _SIGNAL_SLOT_COUNTS_INDEX + _PING_PONG_SIZE
_SIGNAL_STATE_SIZE = _SIGNAL_RELEASE_EPOCH_INDEX + 1


def _validate_multicast_publish(
    multicast_publish: bool, use_ready_flags: bool, split_rows: bool
) -> None:
    if not multicast_publish:
        return
    if use_ready_flags:
        raise ValueError(
            "multicast publish requires sentinel readiness (the multicast store "
            "carries the payload and the not-ready marker in one 16 B write)"
        )
    if not split_rows:
        raise ValueError("multicast publish is implemented only for split-row fused")
    if _dsl_user_op is None or _llvm is None:
        raise RuntimeError("multicast publish requires cutlass._mlir.dialects.llvm")


def _validate_ctas_per_row(ctas_per_row: int) -> None:
    if ctas_per_row not in (2, 4, 8):
        raise ValueError(f"ctas_per_row must be one of 2, 4, or 8, got {ctas_per_row}")


def _validate_signal_advance_flags(
    last_cta_signal_advance: bool,
    aggregate_signal_at_cluster_root: bool,
    use_device_signal_state: bool,
    use_pdl: bool,
    pdl_release_policy: str,
) -> None:
    if not last_cta_signal_advance:
        return
    if not use_device_signal_state:
        raise ValueError("last-CTA signal advance requires device signal state")
    if aggregate_signal_at_cluster_root:
        raise ValueError(
            "last-CTA signal advance and cluster-root signal aggregation are mutually exclusive"
        )
    if use_pdl and pdl_release_policy == "strict":
        raise ValueError(
            "last-CTA signal advance does not implement the strict PDL "
            "release epoch; use output_safe, pre_norm, or entry"
        )


def _validate_pdl_release_policy(policy: str) -> None:
    if policy not in _PDL_RELEASE_POLICIES:
        raise ValueError(f"pdl_release_policy must be 'strict' or 'output_safe', got {policy!r}")


def _validate_fused_pdl_release_policy(policy: str) -> None:
    if policy not in _FUSED_PDL_RELEASE_POLICIES:
        raise ValueError(
            "fused RMSNorm pdl_release_policy must be 'strict', 'output_safe', "
            f"'pre_norm', or 'entry', got {policy!r}"
        )


@cute.jit
def _warp_sum_f32(value):
    value = value + cute.arch.shuffle_sync_bfly(value, 16)
    value = value + cute.arch.shuffle_sync_bfly(value, 8)
    value = value + cute.arch.shuffle_sync_bfly(value, 4)
    value = value + cute.arch.shuffle_sync_bfly(value, 2)
    return value + cute.arch.shuffle_sync_bfly(value, 1)


@cute.jit
def _advance_signal_if_last_cta(
    state_counter_ptr,
    state_signal_ptr,
    signal_state,
    signal_i32,
    ping_slot,
    active_vectors,
    grid_x,
):
    """Last-arrival Lamport state advance (replaces the bidx0 tail spin).

    Every CTA reads ``signal`` before calling this, then one thread per CTA
    performs a returning ``atom.add`` on the entry counter.  The CTA that
    observes ``grid_x - 1`` is the last one to enter, so every CTA of this
    launch has already snapshotted the current signal and the slot state can
    be advanced immediately.  No CTA ever spins on the counter and the scheme
    is independent of the launch grid size (M may change between launches).
    """
    entered_before = prims.atomicrmw(
        prims.AtomicOp.ADD,
        state_counter_ptr,
        cutlass.Int32(1),
        mem_order=prims.MemOrder.ACQ_REL,
        syncscope=prims.MemScope.GPU,
    )
    if entered_before == grid_x - cutlass.Int32(1):
        ping_count_ptr = (
            signal_state.iterator
            + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
            + cutlass.Int64(ping_slot)
        ).llvm_ptr
        next_signal = (signal_i32 + cutlass.Int32(1)) % cutlass.Int32(_PING_PONG_SIZE)
        cute.arch.store(
            ping_count_ptr,
            active_vectors,
            sem="release",
            scope="gpu",
        )
        cute.arch.store(
            state_counter_ptr,
            cutlass.Int32(0),
            sem="release",
            scope="gpu",
        )
        cute.arch.store(
            state_signal_ptr,
            next_signal,
            sem="release",
            scope="gpu",
        )


@cute.jit
def _publish_cta_partial_to_cluster(cluster_partials, cta_sum, cta_rank, CTAS_PER_ROW):
    """Store this CTA's sum-of-squares partial into slot ``cta_rank`` of every
    CTA in the cluster (including itself) through distributed shared memory.

    Each slot has exactly one writer, so no zero-initialisation or atomics are
    needed and a single release/acquire cluster barrier afterwards makes all
    partials visible everywhere.
    """
    for peer_cta_rank in cutlass.range_constexpr(CTAS_PER_ROW):
        for slot in cutlass.range_constexpr(CTAS_PER_ROW):
            if cta_rank == slot:
                peer_slot = prims.mapa(
                    cluster_partials.data_ptr() + slot,
                    cutlass.Int32(peer_cta_rank),
                )
                peer_slot.store(cta_sum)


@cute.jit
def _cluster_inv_rms_from_partials(cluster_partials, CTAS_PER_ROW, N, eps):
    """Every CTA sums the CTAS_PER_ROW partials in the same fixed order, so all
    CTAs of a row compute a bit-identical inverse RMS without a broadcast."""
    cluster_sum = cutlass.Float32(0.0)
    for slot in cutlass.range_constexpr(CTAS_PER_ROW):
        cluster_sum = cluster_sum + cluster_partials[slot]
    return cute.math.rsqrt(
        cluster_sum * (1.0 / cutlass.Float32(N)) + eps,
        fastmath=True,
    )


if _dsl_user_op is not None:

    @_dsl_user_op
    def _multimem_st_v4_b32(mc_addr, word0, word1, word2, word3, *, loc=None, ip=None):
        """``multimem.st.weak.global.v4`` of four b32 words to a multicast address.

        ``mc_addr`` is the 64-bit multicast virtual address (an ``Int64``); the
        NVSwitch fabric replicates the 16 B store into every rank's copy of the
        symmetric buffer, so one instruction replaces WORLD_SIZE-1 unicast
        remote stores. Register bits are moved as-is (``.f32`` is a bit carrier).
        """
        mc_ptr_int = mc_addr.ir_value(loc=loc, ip=ip)
        _llvm.inline_asm(
            None,
            [
                mc_ptr_int,
                word0.ir_value(loc=loc, ip=ip),
                word1.ir_value(loc=loc, ip=ip),
                word2.ir_value(loc=loc, ip=ip),
                word3.ir_value(loc=loc, ip=ip),
            ],
            "multimem.st.weak.global.v4.f32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

else:  # pragma: no cover - older DSL packages

    def _multimem_st_v4_b32(*args, **kwargs):
        raise RuntimeError("multimem store requires cutlass._mlir.dialects.llvm")


@cute.jit
def _multicast_publish_bf16x8(mc_addr, values):
    """Pack eight BF16 lanes into four b32 words and multicast-store them."""
    mask = cutlass.Int32(0xFFFF)
    words = [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)]
    for pair in cutlass.range_constexpr(_BF16_VEC_SIZE // 2):
        low = values[2 * pair].bitcast(cutlass.Int16).to(cutlass.Int32) & mask
        high = values[2 * pair + 1].bitcast(cutlass.Int16).to(cutlass.Int32) & mask
        words[pair] = low | (high << cutlass.Int32(16))
    _multimem_st_v4_b32(mc_addr, words[0], words[1], words[2], words[3])


@cute.jit
def _all_bf16x8_lanes_ready(values):
    ready = cutlass.Boolean(True)
    neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
    for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        ready = ready & (values[lane].bitcast(cutlass.Int16) != neg_zero_bits)
    return ready


@cute.jit
def _decode_positive_e4m3(code: cutlass.Int32) -> cutlass.Float32:
    """Decode a finite non-negative E4M3 byte to FP32."""
    exponent = (code >> cutlass.Int32(3)) & cutlass.Int32(0xF)
    mantissa = code & cutlass.Int32(0x7)
    value = cutlass.Float32(mantissa) * cutlass.Float32(1.0 / 512.0)
    if exponent != cutlass.Int32(0):
        power = cutlass.Int32(exponent + cutlass.Int32(120)) << cutlass.Int32(23)
        value = power.bitcast(cutlass.Float32) * (
            cutlass.Float32(1.0) + cutlass.Float32(mantissa) * cutlass.Float32(0.125)
        )
    return value


@cute.kernel
def _glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    signal: cutlass.Int64,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    USE_READY_FLAGS: cutlass.Constexpr[bool],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    ELIDE_SELF_RANK: cutlass.Constexpr[bool],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    FP32_ACCUMULATION: cutlass.Constexpr[bool],
    BATCH_POLL_PEERS: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_STRICT: cutlass.Constexpr[bool],
    LAST_CTA_SIGNAL_ADVANCE: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    bdim, _, _ = cute.arch.block_dim()

    input_array = cutlass.make_array_view(input_tensor)
    output_array = cutlass.make_array_view(output_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]
    ready_flag_arrays = [cutlass.make_array_view(ready_flag) for ready_flag in ready_flags]
    local_ready_flags = ready_flag_arrays[LOCAL_RANK]

    active_numel = cute.size(input_tensor, mode=[0]) * N
    workspace_numel = cute.size(buffers[0], mode=[2])
    workspace_numel_i64 = cutlass.Int64(workspace_numel)
    num_vectors = cute.size(ready_flags[0], mode=[2])
    num_vectors_i64 = cutlass.Int64(num_vectors)
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_VALUE_INDEX)).llvm_ptr
    release_epoch_ptr = (
        signal_state.iterator + cutlass.Int64(_SIGNAL_RELEASE_EPOCH_INDEX)
    ).llvm_ptr

    signal_i32 = cutlass.Int32(0)
    signal_i64 = signal
    entry_release_epoch = cutlass.Int32(0)
    if cutlass.const_expr(USE_PDL):
        cute.arch.griddepcontrol_wait()
    if USE_DEVICE_SIGNAL_STATE:
        if cutlass.const_expr(USE_PDL):
            if cutlass.const_expr(PDL_RELEASE_STRICT):
                entry_release_epoch = cute.arch.load(
                    release_epoch_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
        signal_i32 = cute.arch.load(
            state_signal_ptr,
            cutlass.Int32,
            sem="acquire",
            scope="gpu",
        )
        signal_i64 = cutlass.Int64(signal_i32)

    ping_slot = signal_i64 % _PING_PONG_SIZE
    clear_slot = (signal_i64 + cutlass.Int64(1)) % _PING_PONG_SIZE
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64
    clear_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64
    ping_ready_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64
    clear_ready_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64

    neg_zero_vec = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
    for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        neg_zero_vec[lane] = neg_zero_bits.bitcast(cutlass.BFloat16)
    ready_zero = cutlass.full((1,), 0, dtype=cutlass.Int32)

    if USE_DEVICE_SIGNAL_STATE:
        clear_count_ptr = (
            signal_state.iterator
            + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
            + cutlass.Int64(clear_slot)
        ).llvm_ptr
        clear_vectors = cute.arch.load(
            clear_count_ptr,
            cutlass.Int32,
            sem="acquire",
            scope="gpu",
        )
        clear_vector_idx = bidx * bdim + tidx
        clear_stride = grid_x * bdim
        while clear_vector_idx < clear_vectors:
            clear_vector_idx_i64 = cutlass.Int64(clear_vector_idx)
            if USE_READY_FLAGS:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_ready_offset = (
                            clear_ready_base
                            + cutlass.Int64(source_rank) * num_vectors_i64
                            + clear_vector_idx_i64
                        )
                        local_ready_flags.store(
                            ready_zero,
                            clear_ready_offset,
                            1,
                            alignment=_READY_FLAG_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
            else:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_offset = (
                            clear_base
                            + cutlass.Int64(source_rank) * workspace_numel_i64
                            + clear_vector_idx_i64 * cutlass.Int64(_BF16_VEC_SIZE)
                        )
                        local_buffer.store(
                            neg_zero_vec[
                                0:_BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ],
                            clear_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
            clear_vector_idx = clear_vector_idx + clear_stride

    base_idx = (bidx * bdim + tidx) * _BF16_VEC_SIZE
    if base_idx < active_numel:
        base_idx_i64 = cutlass.Int64(base_idx)
        vector_idx_i64 = base_idx_i64 // cutlass.Int64(_BF16_VEC_SIZE)

        if not USE_DEVICE_SIGNAL_STATE:
            if USE_READY_FLAGS:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_ready_offset = (
                            clear_ready_base
                            + cutlass.Int64(source_rank) * num_vectors_i64
                            + vector_idx_i64
                        )
                        local_ready_flags.store(
                            ready_zero,
                            clear_ready_offset,
                            1,
                            alignment=_READY_FLAG_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
            else:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_offset = (
                            clear_base
                            + cutlass.Int64(source_rank) * workspace_numel_i64
                            + base_idx_i64
                        )
                        local_buffer.store(
                            neg_zero_vec[
                                0:_BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ],
                            clear_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )

        input_values = input_array.load(
            base_idx,
            _BF16_VEC_SIZE,
            alignment=_ALIGNMENT_BYTES,
        )
        sanitized_input_values = cutlass.Array(
            cutlass.BFloat16,
            _BF16_VEC_SIZE,
            alignment=_ALIGNMENT_BYTES,
        )
        if not USE_READY_FLAGS:
            for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                lane_value = input_values[lane]
                if lane_value.bitcast(cutlass.Int16) == neg_zero_bits:
                    lane_value = cutlass.BFloat16(0.0)
                sanitized_input_values[lane] = lane_value

        for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
            target_rank = PEER_ORDER[peer_ordinal]
            if (not ELIDE_SELF_RANK) or target_rank != LOCAL_RANK:
                write_offset = (
                    ping_base + cutlass.Int64(LOCAL_RANK) * workspace_numel_i64 + base_idx_i64
                )
                if USE_READY_FLAGS:
                    ready_offset = (
                        ping_ready_base
                        + cutlass.Int64(LOCAL_RANK) * num_vectors_i64
                        + vector_idx_i64
                    )
                    buffer_arrays[target_rank].store(
                        input_values,
                        write_offset,
                        _BF16_VEC_SIZE,
                        alignment=_ALIGNMENT_BYTES,
                        is_volatile=True,
                    )
                    prims.red(
                        prims.ReductionOp.ADD,
                        prims.ReductionType.S32,
                        (ready_flags[target_rank].iterator + ready_offset).llvm_ptr,
                        cutlass.Int32(1),
                        mem_order=prims.MemOrder.RELEASE,
                        mem_scope=prims.MemScope.SYS,
                    )
                else:
                    buffer_arrays[target_rank].store(
                        sanitized_input_values[
                            0:_BF16_VEC_SIZE,
                            cutlass.align(_ALIGNMENT_BYTES),
                        ],
                        write_offset,
                        _BF16_VEC_SIZE,
                        alignment=_ALIGNMENT_BYTES,
                        is_volatile=True,
                    )

        # Counter atomic is issued after publication so its round trip
        # overlaps NVLink propagation instead of delaying the publish.
        if USE_DEVICE_SIGNAL_STATE and LAST_CTA_SIGNAL_ADVANCE:
            if tidx == 0:
                active_vectors_i32 = (
                    cutlass.Int32(cute.size(input_tensor, mode=[0])) * cutlass.Int32(N)
                ) // cutlass.Int32(_BF16_VEC_SIZE)
                _advance_signal_if_last_cta(
                    state_counter_ptr,
                    state_signal_ptr,
                    signal_state,
                    signal_i32,
                    ping_slot,
                    active_vectors_i32,
                    grid_x,
                )

        acc_f32 = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.Float32)
        acc_bf16 = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)

        if BATCH_POLL_PEERS:
            peer_values_all = cutlass.Array(
                cutlass.BFloat16,
                WORLD_SIZE * _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            all_peers_ready = cutlass.Boolean(False)
            while not all_peers_ready:
                all_peers_ready = cutlass.Boolean(True)
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        read_offset = (
                            ping_base
                            + cutlass.Int64(source_rank) * workspace_numel_i64
                            + base_idx_i64
                        )
                        polled_values = local_buffer.load(
                            read_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
                        peer_base = source_rank * _BF16_VEC_SIZE
                        peer_values_all[
                            peer_base:_BF16_VEC_SIZE,
                            cutlass.align(_ALIGNMENT_BYTES),
                        ] = polled_values
                        all_peers_ready = all_peers_ready & _all_bf16x8_lanes_ready(polled_values)

            for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                    peer_base = source_rank * _BF16_VEC_SIZE
                    peer_values = peer_values_all[
                        peer_base:_BF16_VEC_SIZE,
                        cutlass.align(_ALIGNMENT_BYTES),
                    ]
                    if FP32_ACCUMULATION:
                        acc_f32 = acc_f32 + peer_values.to(cutlass.Float32)
                    else:
                        acc_bf16 = acc_bf16 + peer_values
                if ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                    # Own partial enters the sum at its rank position so every
                    # rank accumulates 0 -> WORLD_SIZE-1 (bit-identical, as TRT-LLM).
                    if FP32_ACCUMULATION:
                        acc_f32 = acc_f32 + input_values.to(cutlass.Float32)
                    else:
                        acc_bf16 = acc_bf16 + input_values
        else:
            for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                    read_offset = (
                        ping_base + cutlass.Int64(source_rank) * workspace_numel_i64 + base_idx_i64
                    )
                    peer_values = cutlass.full(
                        (_BF16_VEC_SIZE,),
                        0.0,
                        dtype=cutlass.BFloat16,
                    )
                    if USE_READY_FLAGS:
                        read_ready_offset = (
                            ping_ready_base
                            + cutlass.Int64(source_rank) * num_vectors_i64
                            + vector_idx_i64
                        )
                        ready_value = cute.arch.load(
                            (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                            cutlass.Int32,
                            sem="acquire",
                            scope="sys",
                        )
                        while ready_value < cutlass.Int32(1):
                            ready_value = cute.arch.load(
                                (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )
                        peer_values = local_buffer.load(
                            read_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
                    else:
                        peer_values = local_buffer.load(
                            read_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
                        while not _all_bf16x8_lanes_ready(peer_values):
                            peer_values = local_buffer.load(
                                read_offset,
                                _BF16_VEC_SIZE,
                                alignment=_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )

                    if FP32_ACCUMULATION:
                        acc_f32 = acc_f32 + peer_values.to(cutlass.Float32)
                    else:
                        acc_bf16 = acc_bf16 + peer_values
                if ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                    # Own partial enters the sum at its rank position so every
                    # rank accumulates 0 -> WORLD_SIZE-1 (bit-identical, as TRT-LLM).
                    if FP32_ACCUMULATION:
                        acc_f32 = acc_f32 + input_values.to(cutlass.Float32)
                    else:
                        acc_bf16 = acc_bf16 + input_values
        output_values = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)
        if FP32_ACCUMULATION:
            output_values = acc_f32.to(cutlass.BFloat16)
        else:
            output_values = acc_bf16
        output_array.store(
            output_values,
            base_idx,
            _BF16_VEC_SIZE,
            alignment=_ALIGNMENT_BYTES,
        )

    # A successor can launch here, but its entry wait precedes its signal-state
    # load, so it cannot select a Lamport slot until this grid has published it.
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(not PDL_RELEASE_STRICT):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()

    if USE_DEVICE_SIGNAL_STATE and not LAST_CTA_SIGNAL_ADVANCE:
        if tidx == 0:
            prims.red(
                prims.ReductionOp.ADD,
                prims.ReductionType.S32,
                state_counter_ptr,
                cutlass.Int32(1),
                mem_order=prims.MemOrder.RELEASE,
                mem_scope=prims.MemScope.GPU,
            )
        if (bidx == 0) & (tidx == 0):
            completed_ctas = cute.arch.load(
                state_counter_ptr,
                cutlass.Int32,
                sem="acquire",
                scope="gpu",
            )
            while completed_ctas < grid_x:
                completed_ctas = cute.arch.load(
                    state_counter_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
            active_vectors = (
                cutlass.Int32(cute.size(input_tensor, mode=[0])) * cutlass.Int32(N)
            ) // cutlass.Int32(_BF16_VEC_SIZE)
            ping_count_ptr = (
                signal_state.iterator
                + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
                + cutlass.Int64(ping_slot)
            ).llvm_ptr
            next_signal = (signal_i32 + cutlass.Int32(1)) % cutlass.Int32(_PING_PONG_SIZE)
            cute.arch.store(
                ping_count_ptr,
                active_vectors,
                sem="release",
                scope="gpu",
            )
            cute.arch.store(
                state_signal_ptr,
                next_signal,
                sem="release",
                scope="gpu",
            )
            cute.arch.store(
                state_counter_ptr,
                cutlass.Int32(0),
                sem="release",
                scope="gpu",
            )
            if cutlass.const_expr(USE_PDL):
                if cutlass.const_expr(PDL_RELEASE_STRICT):
                    prims.red(
                        prims.ReductionOp.ADD,
                        prims.ReductionType.S32,
                        release_epoch_ptr,
                        cutlass.Int32(1),
                        mem_order=prims.MemOrder.RELEASE,
                        mem_scope=prims.MemScope.GPU,
                    )

    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(PDL_RELEASE_STRICT):
            if tidx == 0:
                if USE_DEVICE_SIGNAL_STATE:
                    published_epoch = cute.arch.load(
                        release_epoch_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="gpu",
                    )
                    while published_epoch == entry_release_epoch:
                        published_epoch = cute.arch.load(
                            release_epoch_ptr,
                            cutlass.Int32,
                            sem="acquire",
                            scope="gpu",
                        )
                cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _glm_tp_allreduce_bf16_oneshot_lamport_serving(
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
    USE_READY_FLAGS: cutlass.Constexpr[bool],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    ELIDE_SELF_RANK: cutlass.Constexpr[bool],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    FP32_ACCUMULATION: cutlass.Constexpr[bool],
    BATCH_POLL_PEERS: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_STRICT: cutlass.Constexpr[bool],
    LAST_CTA_SIGNAL_ADVANCE: cutlass.Constexpr[bool],
) -> None:
    M = input_tensor.shape[0]
    active_numel = M * N
    elems_per_cta = THREADS_PER_BLOCK * _BF16_VEC_SIZE
    num_blocks = cute.ceil_div(active_numel, elems_per_cta)
    _glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel(
        input_tensor,
        output_tensor,
        buffers,
        ready_flags,
        signal_state,
        signal,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        THREADS_PER_BLOCK,
        USE_READY_FLAGS,
        USE_DEVICE_SIGNAL_STATE,
        ELIDE_SELF_RANK,
        PEER_ORDER,
        FP32_ACCUMULATION,
        BATCH_POLL_PEERS,
        USE_PDL,
        PDL_RELEASE_STRICT,
        LAST_CTA_SIGNAL_ADVANCE,
    ).launch(
        grid=(num_blocks, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


@cute.kernel
def _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_kernel(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal: cutlass.Int64,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    input_array = cutlass.make_array_view(input_tensor)
    residual_array = cutlass.make_array_view(residual_tensor)
    norm_weight_array = cutlass.make_array_view(norm_weight_tensor)
    norm_output_array = cutlass.make_array_view(norm_output_tensor)
    residual_output_array = cutlass.make_array_view(residual_output_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]
    ready_flag_arrays = [cutlass.make_array_view(ready_flag) for ready_flag in ready_flags]
    local_ready_flags = ready_flag_arrays[LOCAL_RANK]

    row_values_per_thread = (N + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    row_values_padded = (
        (row_values_per_thread + _BF16_VEC_SIZE - 1) // _BF16_VEC_SIZE
    ) * _BF16_VEC_SIZE
    residual_values = cutlass.Array(
        cutlass.BFloat16,
        row_values_padded,
        alignment=_ALIGNMENT_BYTES,
    )
    weight_values = cutlass.Array(
        cutlass.BFloat16,
        row_values_padded,
        alignment=_ALIGNMENT_BYTES,
    )
    norm_values = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    warp_sums = cutlass.Array(
        cutlass.Float32,
        _WARPS_PER_BLOCK,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )

    lane_id = tidx % 32
    warp_id = tidx // 32
    numel = cute.size(buffers[0], mode=[2])
    row_base = row * N
    ping_slot = signal % _PING_PONG_SIZE
    pong_slot = (signal + cutlass.Int64(1)) % _PING_PONG_SIZE
    numel_i64 = cutlass.Int64(numel)
    num_vectors = cute.size(ready_flags[0], mode=[2])
    num_vectors_i64 = cutlass.Int64(num_vectors)
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * numel_i64
    ping_ready_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64
    pong_ready_base = cutlass.Int64(pong_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64

    ready_zero = cutlass.full((1,), 0, dtype=cutlass.Int32)
    local_sum_sq = cutlass.Float32(0.0)

    for j in cutlass.range_constexpr(0, row_values_per_thread, _BF16_VEC_SIZE):
        col = tidx * _BF16_VEC_SIZE + j * _THREADS_PER_BLOCK
        if col + _BF16_VEC_SIZE <= N:
            base_idx = row_base + col
            base_idx_i64 = cutlass.Int64(base_idx)
            vector_idx_i64 = base_idx_i64 // cutlass.Int64(_BF16_VEC_SIZE)

            for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                clear_ready_offset = (
                    pong_ready_base + cutlass.Int64(source_rank) * num_vectors_i64 + vector_idx_i64
                )
                local_ready_flags.store(
                    ready_zero,
                    clear_ready_offset,
                    1,
                    alignment=_READY_FLAG_ALIGNMENT_BYTES,
                    is_volatile=True,
                )

            input_values = input_array[row, :].load(
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )

            for target_rank in cutlass.range_constexpr(WORLD_SIZE):
                write_offset = ping_base + cutlass.Int64(LOCAL_RANK) * numel_i64 + base_idx_i64
                ready_offset = (
                    ping_ready_base + cutlass.Int64(LOCAL_RANK) * num_vectors_i64 + vector_idx_i64
                )
                buffer_arrays[target_rank].store(
                    input_values,
                    write_offset,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                    is_volatile=True,
                )
                prims.red(
                    prims.ReductionOp.ADD,
                    prims.ReductionType.S32,
                    (ready_flags[target_rank].iterator + ready_offset).llvm_ptr,
                    cutlass.Int32(1),
                    mem_order=prims.MemOrder.RELEASE,
                    mem_scope=prims.MemScope.SYS,
                )

            acc = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.Float32)

            for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                read_offset = ping_base + cutlass.Int64(source_rank) * numel_i64 + base_idx_i64
                read_ready_offset = (
                    ping_ready_base + cutlass.Int64(source_rank) * num_vectors_i64 + vector_idx_i64
                )
                ready_value = cute.arch.load(
                    (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="sys",
                )
                while ready_value < cutlass.Int32(1):
                    ready_value = cute.arch.load(
                        (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="sys",
                    )
                peer_values = local_buffer.load(
                    read_offset,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                    is_volatile=True,
                )

                acc = acc + peer_values.to(cutlass.Float32)

            residual_vec = residual_array[row, :].load(
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            residual_out_vec = (acc + residual_vec.to(cutlass.Float32)).to(cutlass.BFloat16)
            weight_vec = norm_weight_array.load(
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            residual_output_array[row, :].store(
                residual_out_vec,
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            residual_values[j:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = residual_out_vec
            weight_values[j:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = weight_vec

            residual_f32 = residual_out_vec.to(cutlass.Float32)
            local_sum_sq = local_sum_sq + (residual_f32 * residual_f32).reduce("add")

    warp_sum = _warp_sum_f32(local_sum_sq)
    if lane_id == 0:
        warp_sums[warp_id] = warp_sum

    prims.barrier_cta_sync(0)

    if tidx == 0:
        total_sum = cutlass.Float32(0.0)
        for warp_idx in cutlass.range_constexpr(_WARPS_PER_BLOCK):
            total_sum = total_sum + warp_sums[warp_idx]
        warp_sums[0] = cute.math.rsqrt(
            total_sum * (1.0 / cutlass.Float32(N)) + eps,
            fastmath=True,
        )

    prims.barrier_cta_sync(0)
    inv_rms = warp_sums[0]

    for j in cutlass.range_constexpr(0, row_values_per_thread, _BF16_VEC_SIZE):
        col = tidx * _BF16_VEC_SIZE + j * _THREADS_PER_BLOCK
        if col + _BF16_VEC_SIZE <= N:
            residual_vec = residual_values[
                j:_BF16_VEC_SIZE,
                cutlass.align(_ALIGNMENT_BYTES),
            ]
            weight_vec = weight_values[
                j:_BF16_VEC_SIZE,
                cutlass.align(_ALIGNMENT_BYTES),
            ]
            norm_f32 = residual_vec.to(cutlass.Float32) * inv_rms
            norm_f32 = norm_f32 * weight_vec.to(cutlass.Float32)
            norm_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = norm_f32.to(
                cutlass.BFloat16
            )
            norm_output_array[row, :].store(
                norm_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)],
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )


@cute.kernel
def _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    multicast_base: cutlass.Int64,
    signal: cutlass.Int64,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    CTAS_PER_ROW: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    USE_READY_FLAGS: cutlass.Constexpr[bool],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    ELIDE_SELF_RANK: cutlass.Constexpr[bool],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    PREFETCH_FUSED_INPUTS: cutlass.Constexpr[bool],
    DIRECT_CLUSTER_REDUCTION: cutlass.Constexpr[bool],
    POST_NORM_CLUSTER_BARRIER: cutlass.Constexpr[bool],
    FP32_ACCUMULATION: cutlass.Constexpr[bool],
    BATCH_POLL_PEERS: cutlass.Constexpr[bool],
    READY_MASK_POLL_PEERS: cutlass.Constexpr[bool],
    AGGREGATE_SIGNAL_AT_CLUSTER_ROOT: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_STRICT: cutlass.Constexpr[bool],
    PDL_RELEASE_PRE_NORM: cutlass.Constexpr[bool],
    PDL_RELEASE_AT_ENTRY: cutlass.Constexpr[bool],
    LAST_CTA_SIGNAL_ADVANCE: cutlass.Constexpr[bool],
    ALL_TO_ALL_CLUSTER_REDUCTION: cutlass.Constexpr[bool],
    EARLY_INPUT_LOAD: cutlass.Constexpr[bool],
    MULTICAST_PUBLISH: cutlass.Constexpr[bool],
    IKET_PHASES: cutlass.Constexpr[bool],
    ROTATED_POLL_ORDER: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    cta_rank = cute.arch.block_idx_in_cluster()
    row = bidx // CTAS_PER_ROW

    input_array = cutlass.make_array_view(input_tensor)
    residual_array = cutlass.make_array_view(residual_tensor)
    norm_weight_array = cutlass.make_array_view(norm_weight_tensor)
    norm_output_array = cutlass.make_array_view(norm_output_tensor)
    residual_output_array = cutlass.make_array_view(residual_output_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]
    ready_flag_arrays = [cutlass.make_array_view(ready_flag) for ready_flag in ready_flags]
    local_ready_flags = ready_flag_arrays[LOCAL_RANK]

    cols_per_cta = N // CTAS_PER_ROW
    chunk_values_per_thread = (cols_per_cta + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    chunk_values_padded = (
        (chunk_values_per_thread + _BF16_VEC_SIZE - 1) // _BF16_VEC_SIZE
    ) * _BF16_VEC_SIZE
    residual_values = cutlass.Array(
        cutlass.BFloat16,
        chunk_values_padded,
        alignment=_ALIGNMENT_BYTES,
    )
    weight_values = cutlass.Array(
        cutlass.BFloat16,
        chunk_values_padded,
        alignment=_ALIGNMENT_BYTES,
    )
    norm_values = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    warp_sums = cutlass.Array(
        cutlass.Float32,
        THREADS_PER_BLOCK // 32,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )
    row_sum = cutlass.Array(
        cutlass.Float32,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )
    # One slot per CTA of the cluster; written remotely via DSMEM by the
    # all-to-all reduction (each slot has exactly one writer).
    cluster_partials = cutlass.Array(
        cutlass.Float32,
        CTAS_PER_ROW,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )

    lane_id = tidx % 32
    warp_id = tidx // 32
    numel = cute.size(buffers[0], mode=[2])
    row_base = row * N
    cta_col_base = cta_rank * cutlass.Int32(cols_per_cta)
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_VALUE_INDEX)).llvm_ptr
    release_epoch_ptr = (
        signal_state.iterator + cutlass.Int64(_SIGNAL_RELEASE_EPOCH_INDEX)
    ).llvm_ptr
    signal_i32 = cutlass.Int32(0)
    signal_i64 = signal
    entry_release_epoch = cutlass.Int32(0)
    if IKET_PHASES:
        cute.experimental.iket.range_push("ar_entry")
    if cutlass.const_expr(USE_PDL):
        cute.arch.griddepcontrol_wait()
    # Issue the local input load before the signal-state acquire load so the two
    # global round trips overlap; publication then starts as soon as both
    # arrive instead of after their sum.
    early_input_values = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    if EARLY_INPUT_LOAD:
        early_col_in_cta = tidx * _BF16_VEC_SIZE
        if early_col_in_cta + _BF16_VEC_SIZE <= cols_per_cta:
            early_input_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = input_array[
                row, :
            ].load(
                cta_col_base + early_col_in_cta,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
    if USE_DEVICE_SIGNAL_STATE:
        if cutlass.const_expr(USE_PDL):
            if cutlass.const_expr(PDL_RELEASE_STRICT):
                entry_release_epoch = cute.arch.load(
                    release_epoch_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
        signal_i32 = cute.arch.load(
            state_signal_ptr,
            cutlass.Int32,
            sem="acquire",
            scope="gpu",
        )
        signal_i64 = cutlass.Int64(signal_i32)
        if (not AGGREGATE_SIGNAL_AT_CLUSTER_ROOT) and (not LAST_CTA_SIGNAL_ADVANCE) and tidx == 0:
            prims.red(
                prims.ReductionOp.ADD,
                prims.ReductionType.S32,
                state_counter_ptr,
                cutlass.Int32(1),
                mem_order=prims.MemOrder.RELEASE,
                mem_scope=prims.MemScope.GPU,
            )

    # This is deliberately after the device signal snapshot: a PDL successor
    # may start immediately, but its entry wait cannot observe stale slot state.
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(PDL_RELEASE_AT_ENTRY):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()
    ping_slot = signal_i64 % _PING_PONG_SIZE
    clear_slot = (signal_i64 + cutlass.Int64(1)) % _PING_PONG_SIZE
    numel_i64 = cutlass.Int64(numel)
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * numel_i64
    clear_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * numel_i64
    num_vectors = cute.size(ready_flags[0], mode=[2])
    num_vectors_i64 = cutlass.Int64(num_vectors)
    ping_ready_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64
    clear_ready_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64
    neg_zero_vec = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
    for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        neg_zero_vec[lane] = neg_zero_bits.bitcast(cutlass.BFloat16)

    ready_zero = cutlass.full((1,), 0, dtype=cutlass.Int32)
    local_sum_sq = cutlass.Float32(0.0)

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_clear")
    if USE_DEVICE_SIGNAL_STATE:
        clear_count_ptr = (
            signal_state.iterator
            + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
            + cutlass.Int64(clear_slot)
        ).llvm_ptr
        clear_vectors = cute.arch.load(
            clear_count_ptr,
            cutlass.Int32,
            sem="acquire",
            scope="gpu",
        )
        clear_vector_idx = bidx * THREADS_PER_BLOCK + tidx
        clear_stride = grid_x * THREADS_PER_BLOCK
        while clear_vector_idx < clear_vectors:
            clear_vector_idx_i64 = cutlass.Int64(clear_vector_idx)
            if USE_READY_FLAGS:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_ready_offset = (
                            clear_ready_base
                            + cutlass.Int64(source_rank) * num_vectors_i64
                            + clear_vector_idx_i64
                        )
                        local_ready_flags.store(
                            ready_zero,
                            clear_ready_offset,
                            1,
                            alignment=_READY_FLAG_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
            else:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_offset = (
                            clear_base
                            + cutlass.Int64(source_rank) * numel_i64
                            + clear_vector_idx_i64 * cutlass.Int64(_BF16_VEC_SIZE)
                        )
                        local_buffer.store(
                            neg_zero_vec[
                                0:_BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ],
                            clear_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
            clear_vector_idx = clear_vector_idx + clear_stride

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_load_publish")
    for j in cutlass.range_constexpr(0, chunk_values_per_thread, _BF16_VEC_SIZE):
        col_in_cta = tidx * _BF16_VEC_SIZE + j * THREADS_PER_BLOCK
        if col_in_cta + _BF16_VEC_SIZE <= cols_per_cta:
            col = cta_col_base + col_in_cta
            base_idx = row_base + col
            base_idx_i64 = cutlass.Int64(base_idx)
            vector_idx_i64 = base_idx_i64 // cutlass.Int64(_BF16_VEC_SIZE)

            if not USE_DEVICE_SIGNAL_STATE:
                if USE_READY_FLAGS:
                    for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                            clear_ready_offset = (
                                clear_ready_base
                                + cutlass.Int64(source_rank) * num_vectors_i64
                                + vector_idx_i64
                            )
                            local_ready_flags.store(
                                ready_zero,
                                clear_ready_offset,
                                1,
                                alignment=_READY_FLAG_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )
                else:
                    for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                            clear_offset = (
                                clear_base + cutlass.Int64(source_rank) * numel_i64 + base_idx_i64
                            )
                            local_buffer.store(
                                neg_zero_vec[
                                    0:_BF16_VEC_SIZE,
                                    cutlass.align(_ALIGNMENT_BYTES),
                                ],
                                clear_offset,
                                _BF16_VEC_SIZE,
                                alignment=_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )

            # Both arms assign into the same array so the frontend sees one
            # container structure before and after the branch.
            input_values_array = cutlass.Array(
                cutlass.BFloat16,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            if EARLY_INPUT_LOAD and j == 0:
                input_values_array[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = (
                    early_input_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)]
                )
            else:
                input_values_array[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = input_array[
                    row, :
                ].load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
            input_values = input_values_array[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)]
            sanitized_input_values = cutlass.Array(
                cutlass.BFloat16,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            if not USE_READY_FLAGS:
                neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
                for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                    lane_value = input_values[lane]
                    if lane_value.bitcast(cutlass.Int16) == neg_zero_bits:
                        lane_value = cutlass.BFloat16(0.0)
                    sanitized_input_values[lane] = lane_value

            if MULTICAST_PUBLISH:
                # One NVLS multicast store lands in slot[LOCAL_RANK] of every
                # rank (including this one); the fabric does the replication,
                # so egress bytes drop from (WORLD_SIZE-1)x to 1x. Sentinel
                # polling on the receiving side is unchanged.
                multicast_offset = ping_base + cutlass.Int64(LOCAL_RANK) * numel_i64 + base_idx_i64
                _multicast_publish_bf16x8(
                    multicast_base + multicast_offset * cutlass.Int64(2),
                    sanitized_input_values[
                        0:_BF16_VEC_SIZE,
                        cutlass.align(_ALIGNMENT_BYTES),
                    ],
                )
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                target_rank = PEER_ORDER[peer_ordinal]
                if (not MULTICAST_PUBLISH) and ((not ELIDE_SELF_RANK) or target_rank != LOCAL_RANK):
                    write_offset = ping_base + cutlass.Int64(LOCAL_RANK) * numel_i64 + base_idx_i64
                    if USE_READY_FLAGS:
                        ready_offset = (
                            ping_ready_base
                            + cutlass.Int64(LOCAL_RANK) * num_vectors_i64
                            + vector_idx_i64
                        )
                        buffer_arrays[target_rank].store(
                            input_values,
                            write_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
                        prims.red(
                            prims.ReductionOp.ADD,
                            prims.ReductionType.S32,
                            (ready_flags[target_rank].iterator + ready_offset).llvm_ptr,
                            cutlass.Int32(1),
                            mem_order=prims.MemOrder.RELEASE,
                            mem_scope=prims.MemScope.SYS,
                        )
                    else:
                        buffer_arrays[target_rank].store(
                            sanitized_input_values[
                                0:_BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ],
                            write_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )

            if IKET_PHASES:
                cute.experimental.iket.range_pop()
                cute.experimental.iket.range_push("ar_poll")
            # Counter atomic is issued after publication so its round trip
            # overlaps NVLink propagation instead of delaying the publish.
            if USE_DEVICE_SIGNAL_STATE and LAST_CTA_SIGNAL_ADVANCE and j == 0:
                if tidx == 0:
                    active_vectors_i32 = (
                        cutlass.Int32(cute.size(input_tensor, mode=[0])) * cutlass.Int32(N)
                    ) // cutlass.Int32(_BF16_VEC_SIZE)
                    _advance_signal_if_last_cta(
                        state_counter_ptr,
                        state_signal_ptr,
                        signal_state,
                        signal_i32,
                        ping_slot,
                        active_vectors_i32,
                        grid_x,
                    )

            residual_vec = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)
            weight_vec = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)
            if PREFETCH_FUSED_INPUTS:
                # Issue independent local loads after publication and before
                # polling peers so their latency overlaps NVLink propagation.
                residual_vec = residual_array[row, :].load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
                weight_vec = norm_weight_array.load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )

            acc_f32 = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.Float32)
            acc_bf16 = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)

            if READY_MASK_POLL_PEERS:
                # Cache each peer exactly once when its sentinel becomes ready.
                # Accumulation is deferred until every peer is present so the
                # arithmetic order remains PEER_ORDER, independent of arrival.
                peer_values_all = cutlass.Array(
                    cutlass.BFloat16,
                    WORLD_SIZE * _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
                peer_ready_mask = cutlass.Int32(0)
                expected_peer_mask = cutlass.Int32(0)
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        expected_peer_mask = expected_peer_mask | cutlass.Int32(1 << source_rank)

                while peer_ready_mask != expected_peer_mask:
                    for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                            source_rank_mask = cutlass.Int32(1 << source_rank)
                            if (peer_ready_mask & source_rank_mask) == cutlass.Int32(0):
                                read_offset = (
                                    ping_base
                                    + cutlass.Int64(source_rank) * numel_i64
                                    + base_idx_i64
                                )
                                polled_values = local_buffer.load(
                                    read_offset,
                                    _BF16_VEC_SIZE,
                                    alignment=_ALIGNMENT_BYTES,
                                    is_volatile=True,
                                )
                                if _all_bf16x8_lanes_ready(polled_values):
                                    peer_base = source_rank * _BF16_VEC_SIZE
                                    peer_values_all[
                                        peer_base:_BF16_VEC_SIZE,
                                        cutlass.align(_ALIGNMENT_BYTES),
                                    ] = polled_values
                                    peer_ready_mask = peer_ready_mask | source_rank_mask

                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        peer_base = source_rank * _BF16_VEC_SIZE
                        peer_values = peer_values_all[
                            peer_base:_BF16_VEC_SIZE,
                            cutlass.align(_ALIGNMENT_BYTES),
                        ]
                        if FP32_ACCUMULATION:
                            acc_f32 = acc_f32 + peer_values.to(cutlass.Float32)
                        else:
                            acc_bf16 = acc_bf16 + peer_values
                    if ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                        # Own partial enters the sum at its rank position so every
                        # rank accumulates 0 -> WORLD_SIZE-1 (bit-identical, as TRT-LLM).
                        if FP32_ACCUMULATION:
                            acc_f32 = acc_f32 + input_values.to(cutlass.Float32)
                        else:
                            acc_bf16 = acc_bf16 + input_values
            elif BATCH_POLL_PEERS:
                peer_values_all = cutlass.Array(
                    cutlass.BFloat16,
                    WORLD_SIZE * _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
                all_peers_ready = cutlass.Boolean(False)
                while not all_peers_ready:
                    all_peers_ready = cutlass.Boolean(True)
                    for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                            read_offset = (
                                ping_base + cutlass.Int64(source_rank) * numel_i64 + base_idx_i64
                            )
                            polled_values = local_buffer.load(
                                read_offset,
                                _BF16_VEC_SIZE,
                                alignment=_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )
                            peer_base = source_rank * _BF16_VEC_SIZE
                            peer_values_all[
                                peer_base:_BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ] = polled_values
                            all_peers_ready = all_peers_ready & _all_bf16x8_lanes_ready(
                                polled_values
                            )

                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        peer_base = source_rank * _BF16_VEC_SIZE
                        peer_values = peer_values_all[
                            peer_base:_BF16_VEC_SIZE,
                            cutlass.align(_ALIGNMENT_BYTES),
                        ]
                        if FP32_ACCUMULATION:
                            acc_f32 = acc_f32 + peer_values.to(cutlass.Float32)
                        else:
                            acc_bf16 = acc_bf16 + peer_values
                    if ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                        # Own partial enters the sum at its rank position so every
                        # rank accumulates 0 -> WORLD_SIZE-1 (bit-identical, as TRT-LLM).
                        if FP32_ACCUMULATION:
                            acc_f32 = acc_f32 + input_values.to(cutlass.Float32)
                        else:
                            acc_bf16 = acc_bf16 + input_values
            else:
                peer_values_all = cutlass.Array(
                    cutlass.BFloat16,
                    WORLD_SIZE * _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
                for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                    source_rank = PEER_ORDER[peer_ordinal] if ROTATED_POLL_ORDER else peer_ordinal
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        read_offset = (
                            ping_base + cutlass.Int64(source_rank) * numel_i64 + base_idx_i64
                        )
                        peer_values = cutlass.full(
                            (_BF16_VEC_SIZE,),
                            0.0,
                            dtype=cutlass.BFloat16,
                        )
                        if USE_READY_FLAGS:
                            read_ready_offset = (
                                ping_ready_base
                                + cutlass.Int64(source_rank) * num_vectors_i64
                                + vector_idx_i64
                            )
                            ready_value = cute.arch.load(
                                (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )
                            while ready_value < cutlass.Int32(1):
                                ready_value = cute.arch.load(
                                    (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                                    cutlass.Int32,
                                    sem="acquire",
                                    scope="sys",
                                )
                            peer_values = local_buffer.load(
                                read_offset,
                                _BF16_VEC_SIZE,
                                alignment=_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )
                        else:
                            peer_values = local_buffer.load(
                                read_offset,
                                _BF16_VEC_SIZE,
                                alignment=_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )
                            while not _all_bf16x8_lanes_ready(peer_values):
                                peer_values = local_buffer.load(
                                    read_offset,
                                    _BF16_VEC_SIZE,
                                    alignment=_ALIGNMENT_BYTES,
                                    is_volatile=True,
                                )

                        if ROTATED_POLL_ORDER:
                            peer_values_all[
                                source_rank * _BF16_VEC_SIZE : _BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ] = peer_values
                        elif FP32_ACCUMULATION:
                            acc_f32 = acc_f32 + peer_values.to(cutlass.Float32)
                        else:
                            acc_bf16 = acc_bf16 + peer_values
                    if (not ROTATED_POLL_ORDER) and ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                        # Own partial enters the sum at its rank position so every
                        # rank accumulates 0 -> WORLD_SIZE-1 (bit-identical, as TRT-LLM).
                        if FP32_ACCUMULATION:
                            acc_f32 = acc_f32 + input_values.to(cutlass.Float32)
                        else:
                            acc_bf16 = acc_bf16 + input_values
                if ROTATED_POLL_ORDER:
                    # Polled in PEER_ORDER (latency), summed in absolute rank order (numerics).
                    for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                            peer_values = peer_values_all[
                                source_rank * _BF16_VEC_SIZE : _BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ]
                            if FP32_ACCUMULATION:
                                acc_f32 = acc_f32 + peer_values.to(cutlass.Float32)
                            else:
                                acc_bf16 = acc_bf16 + peer_values
                        if ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                            if FP32_ACCUMULATION:
                                acc_f32 = acc_f32 + input_values.to(cutlass.Float32)
                            else:
                                acc_bf16 = acc_bf16 + input_values
            if IKET_PHASES:
                cute.experimental.iket.range_pop()
                cute.experimental.iket.range_push("ar_residual")
            if not PREFETCH_FUSED_INPUTS:
                residual_vec = residual_array[row, :].load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
                weight_vec = norm_weight_array.load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
            # The DSL requires a value used after staged control flow to be
            # defined before the branch, even when both constexpr arms assign it.
            residual_out_vec = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)
            if FP32_ACCUMULATION:
                residual_out_vec = (acc_f32 + residual_vec.to(cutlass.Float32)).to(cutlass.BFloat16)
            else:
                residual_out_vec = acc_bf16 + residual_vec
            residual_output_array[row, :].store(
                residual_out_vec,
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            residual_values[j:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = residual_out_vec
            weight_values[j:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = weight_vec

            residual_f32 = residual_out_vec.to(cutlass.Float32)
            local_sum_sq = local_sum_sq + (residual_f32 * residual_f32).reduce("add")

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_rms_reduce")
    warp_sum = _warp_sum_f32(local_sum_sq)
    if lane_id == 0:
        warp_sums[warp_id] = warp_sum

    prims.barrier_cta_sync(0)

    cta_sum = cutlass.Float32(0.0)
    if tidx == 0:
        for warp_idx in cutlass.range_constexpr(THREADS_PER_BLOCK // 32):
            cta_sum = cta_sum + warp_sums[warp_idx]
        if ALL_TO_ALL_CLUSTER_REDUCTION:
            _publish_cta_partial_to_cluster(cluster_partials, cta_sum, cta_rank, CTAS_PER_ROW)
        elif DIRECT_CLUSTER_REDUCTION:
            row_sum[0] = cta_sum
        elif cta_rank == 0:
            row_sum[0] = cutlass.Float32(0.0)

    inv_rms = cutlass.Float32(0.0)
    if cutlass.const_expr(ALL_TO_ALL_CLUSTER_REDUCTION):
        # Single cluster rendezvous: arrive.release publishes the DSMEM partial
        # stores, wait.acquire makes every slot visible; afterwards no CTA
        # touches another CTA's shared memory, so no lifetime barrier is needed.
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        if cutlass.const_expr(USE_PDL):
            if cutlass.const_expr(PDL_RELEASE_PRE_NORM):
                if tidx == 0:
                    cute.arch.griddepcontrol_launch_dependents()
        inv_rms = _cluster_inv_rms_from_partials(cluster_partials, CTAS_PER_ROW, N, eps)
    elif cutlass.const_expr(not DIRECT_CLUSTER_REDUCTION):
        prims.barrier_cta_sync(0)
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        if tidx == 0:
            root_row_sum = prims.mapa(row_sum.data_ptr(), cutlass.Int32(0))
            prims.red(
                prims.ReductionOp.ADD,
                prims.ReductionType.F32,
                root_row_sum,
                cta_sum,
                mem_order=prims.MemOrder.RELAXED,
                mem_scope=prims.MemScope.CLUSTER,
            )

        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        if (cta_rank == 0) & (tidx == 0):
            row_sum[0] = cute.math.rsqrt(
                row_sum[0] * (1.0 / cutlass.Float32(N)) + eps,
                fastmath=True,
            )

        prims.barrier_cta_sync(0)
    else:
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        if (cta_rank == 0) & (tidx == 0):
            cluster_sum = cutlass.Float32(0.0)
            for peer_cta_rank in cutlass.range_constexpr(CTAS_PER_ROW):
                peer_row_sum = prims.mapa(row_sum.data_ptr(), cutlass.Int32(peer_cta_rank))
                cluster_sum = cluster_sum + peer_row_sum.load()
            row_sum[0] = cute.math.rsqrt(
                cluster_sum * (1.0 / cutlass.Float32(N)) + eps,
                fastmath=True,
            )

    if cutlass.const_expr(not ALL_TO_ALL_CLUSTER_REDUCTION):
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        # This is the earliest existing cluster-safe release point: every CTA
        # has completed peer polling and residual/sumsq work, and no extra
        # barrier is added. The successor may overlap this kernel's RMS
        # normalization tail.
        if cutlass.const_expr(USE_PDL):
            if cutlass.const_expr(PDL_RELEASE_PRE_NORM):
                if tidx == 0:
                    cute.arch.griddepcontrol_launch_dependents()

        root_row_sum = prims.mapa(row_sum.data_ptr(), cutlass.Int32(0))
        inv_rms = root_row_sum.load()

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_tail_store")
    for j in cutlass.range_constexpr(0, chunk_values_per_thread, _BF16_VEC_SIZE):
        col_in_cta = tidx * _BF16_VEC_SIZE + j * THREADS_PER_BLOCK
        if col_in_cta + _BF16_VEC_SIZE <= cols_per_cta:
            col = cta_col_base + col_in_cta
            residual_vec = residual_values[
                j:_BF16_VEC_SIZE,
                cutlass.align(_ALIGNMENT_BYTES),
            ]
            weight_vec = weight_values[
                j:_BF16_VEC_SIZE,
                cutlass.align(_ALIGNMENT_BYTES),
            ]
            norm_f32 = residual_vec.to(cutlass.Float32) * inv_rms
            norm_f32 = norm_f32 * weight_vec.to(cutlass.Float32)
            norm_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = norm_f32.to(
                cutlass.BFloat16
            )
            norm_output_array[row, :].store(
                norm_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)],
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
    if POST_NORM_CLUSTER_BARRIER and not ALL_TO_ALL_CLUSTER_REDUCTION:
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()

    # A successor can launch here, but its entry wait precedes its signal-state
    # load, so it cannot select a Lamport slot until this grid has published it.
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(
            not PDL_RELEASE_STRICT and not PDL_RELEASE_PRE_NORM and not PDL_RELEASE_AT_ENTRY
        ):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()

    if USE_DEVICE_SIGNAL_STATE and not LAST_CTA_SIGNAL_ADVANCE:
        if AGGREGATE_SIGNAL_AT_CLUSTER_ROOT:
            if (cta_rank == 0) & (tidx == 0):
                prims.red(
                    prims.ReductionOp.ADD,
                    prims.ReductionType.S32,
                    state_counter_ptr,
                    cutlass.Int32(1),
                    mem_order=prims.MemOrder.RELEASE,
                    mem_scope=prims.MemScope.GPU,
                )
        if (bidx == 0) & (tidx == 0):
            if AGGREGATE_SIGNAL_AT_CLUSTER_ROOT:
                completed_clusters = cute.arch.load(
                    state_counter_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
                while completed_clusters < grid_x // CTAS_PER_ROW:
                    completed_clusters = cute.arch.load(
                        state_counter_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="gpu",
                    )
            else:
                entered_ctas = cute.arch.load(
                    state_counter_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
                while entered_ctas < grid_x:
                    entered_ctas = cute.arch.load(
                        state_counter_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="gpu",
                    )
            active_vectors = (
                cutlass.Int32(cute.size(input_tensor, mode=[0])) * cutlass.Int32(N)
            ) // cutlass.Int32(_BF16_VEC_SIZE)
            ping_count_ptr = (
                signal_state.iterator
                + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
                + cutlass.Int64(ping_slot)
            ).llvm_ptr
            next_signal = (signal_i32 + cutlass.Int32(1)) % cutlass.Int32(_PING_PONG_SIZE)
            cute.arch.store(
                ping_count_ptr,
                active_vectors,
                sem="release",
                scope="gpu",
            )
            cute.arch.store(
                state_signal_ptr,
                next_signal,
                sem="release",
                scope="gpu",
            )
            cute.arch.store(
                state_counter_ptr,
                cutlass.Int32(0),
                sem="release",
                scope="gpu",
            )
            if cutlass.const_expr(USE_PDL):
                if cutlass.const_expr(PDL_RELEASE_STRICT):
                    prims.red(
                        prims.ReductionOp.ADD,
                        prims.ReductionType.S32,
                        release_epoch_ptr,
                        cutlass.Int32(1),
                        mem_order=prims.MemOrder.RELEASE,
                        mem_scope=prims.MemScope.GPU,
                    )

    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(PDL_RELEASE_STRICT):
            if tidx == 0:
                if USE_DEVICE_SIGNAL_STATE:
                    published_epoch = cute.arch.load(
                        release_epoch_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="gpu",
                    )
                    while published_epoch == entry_release_epoch:
                        published_epoch = cute.arch.load(
                            release_epoch_ptr,
                            cutlass.Int32,
                            sem="acquire",
                            scope="gpu",
                        )
                cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def _glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    global_scale_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    fp4_output_tensor: cute.Tensor,
    scale_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    multicast_base: cutlass.Int64,
    signal: cutlass.Int64,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    CTAS_PER_ROW: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    USE_READY_FLAGS: cutlass.Constexpr[bool],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    ELIDE_SELF_RANK: cutlass.Constexpr[bool],
    AGGREGATE_SIGNAL_AT_CLUSTER_ROOT: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_STRICT: cutlass.Constexpr[bool],
    PDL_RELEASE_PRE_NORM: cutlass.Constexpr[bool],
    PDL_RELEASE_AT_ENTRY: cutlass.Constexpr[bool],
    LAST_CTA_SIGNAL_ADVANCE: cutlass.Constexpr[bool],
    ALL_TO_ALL_CLUSTER_REDUCTION: cutlass.Constexpr[bool],
    EARLY_INPUT_LOAD: cutlass.Constexpr[bool],
    MULTICAST_PUBLISH: cutlass.Constexpr[bool],
    IKET_PHASES: cutlass.Constexpr[bool],
    ROTATED_POLL_ORDER: cutlass.Constexpr[bool],
    PREFETCH_FUSED_INPUTS: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    cta_rank = cute.arch.block_idx_in_cluster()
    row = bidx // CTAS_PER_ROW

    input_array = cutlass.make_array_view(input_tensor)
    residual_array = cutlass.make_array_view(residual_tensor)
    norm_weight_array = cutlass.make_array_view(norm_weight_tensor)
    norm_output_array = cutlass.make_array_view(norm_output_tensor)
    residual_output_array = cutlass.make_array_view(residual_output_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]
    ready_flag_arrays = [cutlass.make_array_view(ready_flag) for ready_flag in ready_flags]
    local_ready_flags = ready_flag_arrays[LOCAL_RANK]

    cols_per_cta = N // CTAS_PER_ROW
    chunk_values_per_thread = (cols_per_cta + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    chunk_values_padded = (
        (chunk_values_per_thread + _BF16_VEC_SIZE - 1) // _BF16_VEC_SIZE
    ) * _BF16_VEC_SIZE
    residual_values = cutlass.Array(
        cutlass.BFloat16,
        chunk_values_padded,
        alignment=_ALIGNMENT_BYTES,
    )
    weight_values = cutlass.Array(
        cutlass.BFloat16,
        chunk_values_padded,
        alignment=_ALIGNMENT_BYTES,
    )
    norm_values = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    warp_sums = cutlass.Array(
        cutlass.Float32,
        (THREADS_PER_BLOCK // 32),
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )
    row_sum = cutlass.Array(
        cutlass.Float32,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )
    # One slot per CTA of the cluster; written remotely via DSMEM by the
    # all-to-all reduction (each slot has exactly one writer).
    cluster_partials = cutlass.Array(
        cutlass.Float32,
        CTAS_PER_ROW,
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )

    lane_id = tidx % 32
    warp_id = tidx // 32
    numel = cute.size(buffers[0], mode=[2])
    row_base = row * N
    cta_col_base = cta_rank * cutlass.Int32(cols_per_cta)
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_VALUE_INDEX)).llvm_ptr
    release_epoch_ptr = (
        signal_state.iterator + cutlass.Int64(_SIGNAL_RELEASE_EPOCH_INDEX)
    ).llvm_ptr
    signal_i32 = cutlass.Int32(0)
    signal_i64 = signal
    entry_release_epoch = cutlass.Int32(0)
    if IKET_PHASES:
        cute.experimental.iket.range_push("ar_entry")
    if cutlass.const_expr(USE_PDL):
        cute.arch.griddepcontrol_wait()
    # The NVFP4 global scale is a per-layer constant; issuing its load here lets
    # the round trip overlap the signal-state load and the Lamport protocol
    # instead of sitting on the critical path of the quant tail.
    global_scale_ptr = global_scale_tensor.iterator.raw_ptr()
    global_scale = cutlass.Float32(0.0)
    if PREFETCH_FUSED_INPUTS:
        global_scale = prims.load_ext(global_scale_ptr, dtype=cutlass.Float32)
    # Issue the local input load before the signal-state acquire load so the two
    # global round trips overlap; publication then starts as soon as both
    # arrive instead of after their sum.
    early_input_values = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    if EARLY_INPUT_LOAD:
        early_col_in_cta = tidx * _BF16_VEC_SIZE
        if early_col_in_cta + _BF16_VEC_SIZE <= cols_per_cta:
            early_input_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = input_array[
                row, :
            ].load(
                cta_col_base + early_col_in_cta,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
    if USE_DEVICE_SIGNAL_STATE:
        if cutlass.const_expr(USE_PDL):
            if cutlass.const_expr(PDL_RELEASE_STRICT):
                entry_release_epoch = cute.arch.load(
                    release_epoch_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
        signal_i32 = cute.arch.load(
            state_signal_ptr,
            cutlass.Int32,
            sem="acquire",
            scope="gpu",
        )
        signal_i64 = cutlass.Int64(signal_i32)
        if (not AGGREGATE_SIGNAL_AT_CLUSTER_ROOT) and (not LAST_CTA_SIGNAL_ADVANCE) and tidx == 0:
            prims.red(
                prims.ReductionOp.ADD,
                prims.ReductionType.S32,
                state_counter_ptr,
                cutlass.Int32(1),
                mem_order=prims.MemOrder.RELEASE,
                mem_scope=prims.MemScope.GPU,
            )

    # Deliberately after the device signal snapshot: a PDL successor may start
    # immediately, but its entry wait cannot observe stale slot state.
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(PDL_RELEASE_AT_ENTRY):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()
    ping_slot = signal_i64 % _PING_PONG_SIZE
    clear_slot = (signal_i64 + cutlass.Int64(1)) % _PING_PONG_SIZE
    numel_i64 = cutlass.Int64(numel)
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * numel_i64
    clear_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * numel_i64
    num_vectors = cute.size(ready_flags[0], mode=[2])
    num_vectors_i64 = cutlass.Int64(num_vectors)
    ping_ready_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64
    clear_ready_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64
    neg_zero_vec = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
    for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        neg_zero_vec[lane] = neg_zero_bits.bitcast(cutlass.BFloat16)

    ready_zero = cutlass.full((1,), 0, dtype=cutlass.Int32)
    local_sum_sq = cutlass.Float32(0.0)

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_clear")
    if USE_DEVICE_SIGNAL_STATE:
        clear_count_ptr = (
            signal_state.iterator
            + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
            + cutlass.Int64(clear_slot)
        ).llvm_ptr
        clear_vectors = cute.arch.load(
            clear_count_ptr,
            cutlass.Int32,
            sem="acquire",
            scope="gpu",
        )
        clear_vector_idx = bidx * THREADS_PER_BLOCK + tidx
        clear_stride = grid_x * THREADS_PER_BLOCK
        while clear_vector_idx < clear_vectors:
            clear_vector_idx_i64 = cutlass.Int64(clear_vector_idx)
            if USE_READY_FLAGS:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_ready_offset = (
                            clear_ready_base
                            + cutlass.Int64(source_rank) * num_vectors_i64
                            + clear_vector_idx_i64
                        )
                        local_ready_flags.store(
                            ready_zero,
                            clear_ready_offset,
                            1,
                            alignment=_READY_FLAG_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
            else:
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        clear_offset = (
                            clear_base
                            + cutlass.Int64(source_rank) * numel_i64
                            + clear_vector_idx_i64 * cutlass.Int64(_BF16_VEC_SIZE)
                        )
                        local_buffer.store(
                            neg_zero_vec[
                                0:_BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ],
                            clear_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
            clear_vector_idx = clear_vector_idx + clear_stride

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_load_publish")
    for j in cutlass.range_constexpr(0, chunk_values_per_thread, _BF16_VEC_SIZE):
        col_in_cta = tidx * _BF16_VEC_SIZE + j * THREADS_PER_BLOCK
        if col_in_cta + _BF16_VEC_SIZE <= cols_per_cta:
            col = cta_col_base + col_in_cta
            base_idx = row_base + col
            base_idx_i64 = cutlass.Int64(base_idx)
            vector_idx_i64 = base_idx_i64 // cutlass.Int64(_BF16_VEC_SIZE)

            if not USE_DEVICE_SIGNAL_STATE:
                if USE_READY_FLAGS:
                    for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                            clear_ready_offset = (
                                clear_ready_base
                                + cutlass.Int64(source_rank) * num_vectors_i64
                                + vector_idx_i64
                            )
                            local_ready_flags.store(
                                ready_zero,
                                clear_ready_offset,
                                1,
                                alignment=_READY_FLAG_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )
                else:
                    for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                            clear_offset = (
                                clear_base + cutlass.Int64(source_rank) * numel_i64 + base_idx_i64
                            )
                            local_buffer.store(
                                neg_zero_vec[
                                    0:_BF16_VEC_SIZE,
                                    cutlass.align(_ALIGNMENT_BYTES),
                                ],
                                clear_offset,
                                _BF16_VEC_SIZE,
                                alignment=_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )

            # Both arms assign into the same array so the frontend sees one
            # container structure before and after the branch.
            input_values_array = cutlass.Array(
                cutlass.BFloat16,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            if EARLY_INPUT_LOAD and j == 0:
                input_values_array[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = (
                    early_input_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)]
                )
            else:
                input_values_array[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = input_array[
                    row, :
                ].load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
            input_values = input_values_array[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)]
            sanitized_input_values = cutlass.Array(
                cutlass.BFloat16,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            if not USE_READY_FLAGS:
                neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
                for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                    lane_value = input_values[lane]
                    if lane_value.bitcast(cutlass.Int16) == neg_zero_bits:
                        lane_value = cutlass.BFloat16(0.0)
                    sanitized_input_values[lane] = lane_value

            if MULTICAST_PUBLISH:
                # One NVLS multicast store lands in slot[LOCAL_RANK] of every
                # rank (including this one); the fabric does the replication,
                # so egress bytes drop from (WORLD_SIZE-1)x to 1x. Sentinel
                # polling on the receiving side is unchanged.
                multicast_offset = ping_base + cutlass.Int64(LOCAL_RANK) * numel_i64 + base_idx_i64
                _multicast_publish_bf16x8(
                    multicast_base + multicast_offset * cutlass.Int64(2),
                    sanitized_input_values[
                        0:_BF16_VEC_SIZE,
                        cutlass.align(_ALIGNMENT_BYTES),
                    ],
                )
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                target_rank = PEER_ORDER[peer_ordinal]
                if (not MULTICAST_PUBLISH) and ((not ELIDE_SELF_RANK) or target_rank != LOCAL_RANK):
                    write_offset = ping_base + cutlass.Int64(LOCAL_RANK) * numel_i64 + base_idx_i64
                    if USE_READY_FLAGS:
                        ready_offset = (
                            ping_ready_base
                            + cutlass.Int64(LOCAL_RANK) * num_vectors_i64
                            + vector_idx_i64
                        )
                        buffer_arrays[target_rank].store(
                            input_values,
                            write_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
                        prims.red(
                            prims.ReductionOp.ADD,
                            prims.ReductionType.S32,
                            (ready_flags[target_rank].iterator + ready_offset).llvm_ptr,
                            cutlass.Int32(1),
                            mem_order=prims.MemOrder.RELEASE,
                            mem_scope=prims.MemScope.SYS,
                        )
                    else:
                        buffer_arrays[target_rank].store(
                            sanitized_input_values[
                                0:_BF16_VEC_SIZE,
                                cutlass.align(_ALIGNMENT_BYTES),
                            ],
                            write_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )

            # Defined before the constexpr branches so the value survives staged
            # control flow (DSL requirement, as in the fused RMSNorm kernel).
            residual_vec = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)
            weight_vec = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.BFloat16)
            if PREFETCH_FUSED_INPUTS:
                # Issue the residual and RMS-weight loads after publication and
                # before polling so their latency hides under NVLink propagation
                # (the fused RMSNorm kernel does the same).
                residual_vec = residual_array[row, :].load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
                weight_vec = norm_weight_array.load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
            if IKET_PHASES:
                cute.experimental.iket.range_pop()
                cute.experimental.iket.range_push("ar_poll")
            # Counter atomic is issued after publication so its round trip
            # overlaps NVLink propagation instead of delaying the publish.
            if USE_DEVICE_SIGNAL_STATE and LAST_CTA_SIGNAL_ADVANCE and j == 0:
                if tidx == 0:
                    active_vectors_i32 = (
                        cutlass.Int32(cute.size(input_tensor, mode=[0])) * cutlass.Int32(N)
                    ) // cutlass.Int32(_BF16_VEC_SIZE)
                    _advance_signal_if_last_cta(
                        state_counter_ptr,
                        state_signal_ptr,
                        signal_state,
                        signal_i32,
                        ping_slot,
                        active_vectors_i32,
                        grid_x,
                    )

            acc = cutlass.full((_BF16_VEC_SIZE,), 0.0, dtype=cutlass.Float32)

            peer_values_all = cutlass.Array(
                cutlass.BFloat16,
                WORLD_SIZE * _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                source_rank = PEER_ORDER[peer_ordinal] if ROTATED_POLL_ORDER else peer_ordinal
                if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                    read_offset = ping_base + cutlass.Int64(source_rank) * numel_i64 + base_idx_i64
                    peer_values = cutlass.full(
                        (_BF16_VEC_SIZE,),
                        0.0,
                        dtype=cutlass.BFloat16,
                    )
                    if USE_READY_FLAGS:
                        read_ready_offset = (
                            ping_ready_base
                            + cutlass.Int64(source_rank) * num_vectors_i64
                            + vector_idx_i64
                        )
                        ready_value = cute.arch.load(
                            (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                            cutlass.Int32,
                            sem="acquire",
                            scope="sys",
                        )
                        while ready_value < cutlass.Int32(1):
                            ready_value = cute.arch.load(
                                (ready_flags[LOCAL_RANK].iterator + read_ready_offset).llvm_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )
                        peer_values = local_buffer.load(
                            read_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
                    else:
                        peer_values = local_buffer.load(
                            read_offset,
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                            is_volatile=True,
                        )
                        while not _all_bf16x8_lanes_ready(peer_values):
                            peer_values = local_buffer.load(
                                read_offset,
                                _BF16_VEC_SIZE,
                                alignment=_ALIGNMENT_BYTES,
                                is_volatile=True,
                            )

                    if ROTATED_POLL_ORDER:
                        peer_values_all[
                            source_rank * _BF16_VEC_SIZE : _BF16_VEC_SIZE,
                            cutlass.align(_ALIGNMENT_BYTES),
                        ] = peer_values
                    else:
                        acc = acc + peer_values.to(cutlass.Float32)
                if (not ROTATED_POLL_ORDER) and ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                    # Own partial enters the sum at its rank position so every
                    # rank accumulates 0 -> WORLD_SIZE-1 (bit-identical, as TRT-LLM).
                    acc = acc + input_values.to(cutlass.Float32)
            if ROTATED_POLL_ORDER:
                # Polled in PEER_ORDER (latency), summed in absolute rank order (numerics).
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    if (not ELIDE_SELF_RANK) or source_rank != LOCAL_RANK:
                        peer_values = peer_values_all[
                            source_rank * _BF16_VEC_SIZE : _BF16_VEC_SIZE,
                            cutlass.align(_ALIGNMENT_BYTES),
                        ]
                        acc = acc + peer_values.to(cutlass.Float32)
                    if ELIDE_SELF_RANK and source_rank == LOCAL_RANK:
                        acc = acc + input_values.to(cutlass.Float32)
            if IKET_PHASES:
                cute.experimental.iket.range_pop()
                cute.experimental.iket.range_push("ar_residual")
            if not PREFETCH_FUSED_INPUTS:
                residual_vec = residual_array[row, :].load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
                weight_vec = norm_weight_array.load(
                    col,
                    _BF16_VEC_SIZE,
                    alignment=_ALIGNMENT_BYTES,
                )
            residual_out_vec = (acc + residual_vec.to(cutlass.Float32)).to(cutlass.BFloat16)
            residual_output_array[row, :].store(
                residual_out_vec,
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )
            residual_values[j:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = residual_out_vec
            weight_values[j:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = weight_vec

            residual_f32 = residual_out_vec.to(cutlass.Float32)
            local_sum_sq = local_sum_sq + (residual_f32 * residual_f32).reduce("add")

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_rms_reduce")
    warp_sum = _warp_sum_f32(local_sum_sq)
    if lane_id == 0:
        warp_sums[warp_id] = warp_sum

    prims.barrier_cta_sync(0)

    cta_sum = cutlass.Float32(0.0)
    if tidx == 0:
        for warp_idx in cutlass.range_constexpr((THREADS_PER_BLOCK // 32)):
            cta_sum = cta_sum + warp_sums[warp_idx]
        if ALL_TO_ALL_CLUSTER_REDUCTION:
            _publish_cta_partial_to_cluster(cluster_partials, cta_sum, cta_rank, CTAS_PER_ROW)
        elif cta_rank == 0:
            row_sum[0] = cutlass.Float32(0.0)

    inv_rms = cutlass.Float32(0.0)
    if cutlass.const_expr(ALL_TO_ALL_CLUSTER_REDUCTION):
        # Single cluster rendezvous; afterwards no CTA touches another CTA's
        # shared memory, so no lifetime barrier is needed before exit.
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        if cutlass.const_expr(USE_PDL):
            if cutlass.const_expr(PDL_RELEASE_PRE_NORM):
                if tidx == 0:
                    cute.arch.griddepcontrol_launch_dependents()
        inv_rms = _cluster_inv_rms_from_partials(cluster_partials, CTAS_PER_ROW, N, eps)
    else:
        prims.barrier_cta_sync(0)
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        if tidx == 0:
            root_row_sum = prims.mapa(row_sum.data_ptr(), cutlass.Int32(0))
            prims.red(
                prims.ReductionOp.ADD,
                prims.ReductionType.F32,
                root_row_sum,
                cta_sum,
                mem_order=prims.MemOrder.RELAXED,
                mem_scope=prims.MemScope.CLUSTER,
            )

        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        if (cta_rank == 0) & (tidx == 0):
            row_sum[0] = cute.math.rsqrt(
                row_sum[0] * (1.0 / cutlass.Float32(N)) + eps,
                fastmath=True,
            )

        prims.barrier_cta_sync(0)
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        if cutlass.const_expr(USE_PDL):
            if cutlass.const_expr(PDL_RELEASE_PRE_NORM):
                if tidx == 0:
                    cute.arch.griddepcontrol_launch_dependents()

        root_row_sum = prims.mapa(row_sum.data_ptr(), cutlass.Int32(0))
        inv_rms = root_row_sum.load()
    if IKET_PHASES:
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("ar_tail_store")
    fp4_ptr = fp4_output_tensor.iterator.raw_ptr()
    sf_ptr = scale_output_tensor.iterator.raw_ptr()
    if not PREFETCH_FUSED_INPUTS:
        global_scale = prims.load_ext(global_scale_ptr, dtype=cutlass.Float32)
    scale_blocks_per_row: cutlass.Constexpr = N // _SCALE_BLOCK

    for j in cutlass.range_constexpr(0, chunk_values_per_thread, _BF16_VEC_SIZE):
        col_in_cta = tidx * _BF16_VEC_SIZE + j * THREADS_PER_BLOCK
        if col_in_cta + _BF16_VEC_SIZE <= cols_per_cta:
            col = cta_col_base + col_in_cta
            residual_vec = residual_values[
                j:_BF16_VEC_SIZE,
                cutlass.align(_ALIGNMENT_BYTES),
            ]
            weight_vec = weight_values[
                j:_BF16_VEC_SIZE,
                cutlass.align(_ALIGNMENT_BYTES),
            ]
            norm_f32 = residual_vec.to(cutlass.Float32) * inv_rms
            norm_f32 = norm_f32 * weight_vec.to(cutlass.Float32)
            norm_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = norm_f32.to(
                cutlass.BFloat16
            )
            quant_f32 = norm_f32.to(cutlass.BFloat16).to(cutlass.Float32)
            norm_output_array[row, :].store(
                norm_values[0:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)],
                col,
                _BF16_VEC_SIZE,
                alignment=_ALIGNMENT_BYTES,
            )

            local_max = cutlass.Float32(0.0)
            for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                magnitude = cute.math.abs(quant_f32[lane])
                if magnitude > local_max:
                    local_max = magnitude

            block_max = local_max
            peer_max = cute.arch.shuffle_sync_bfly(block_max, 1)
            if peer_max > block_max:
                block_max = peer_max

            sf_unquantized = global_scale * block_max * cutlass.Float32(1.0 / 6.0)
            sf_packed = prims.cvt_f32x2_to_f8x2(
                sf_unquantized,
                sf_unquantized,
                cutlass.Float8E4M3FN,
                rnd=prims.FPRoundingMode.RN,
                sat=prims.SaturationMode.SATFINITE,
            )
            sf_code = cutlass.Int32(sf_packed) & cutlass.Int32(0xFF)
            half_block = (col // cutlass.Int32(_BF16_VEC_SIZE)) & cutlass.Int32(1)
            scale_block = col // cutlass.Int32(_SCALE_BLOCK)
            if half_block == cutlass.Int32(0):
                sf_index = row * cutlass.Int32(scale_blocks_per_row) + scale_block
                prims.store_ext(cutlass.Int8(sf_code), sf_ptr + sf_index)

            narrowed_sf = _decode_positive_e4m3(sf_code)
            output_scale = cutlass.Float32(0.0)
            if block_max != cutlass.Float32(0.0):
                output_scale = cute.arch.rcp_approx(
                    narrowed_sf * cute.arch.rcp_approx(global_scale)
                )

            packed_word = cutlass.Int32(0)
            for pair in cutlass.range_constexpr(_BF16_VEC_SIZE // 2):
                low = quant_f32[2 * pair] * output_scale
                high = quant_f32[2 * pair + 1] * output_scale
                packed_byte = prims.cvt_f32x2_to_f4x2(
                    high,
                    low,
                    cutlass.Float4E2M1FN,
                    rnd=prims.FPRoundingMode.RN,
                ) & cutlass.Int32(0xFF)
                packed_word = packed_word | (packed_byte << cutlass.Int32(pair * 8))
            fp4_index = (
                row * cutlass.Int32(N // 2)
                + scale_block * cutlass.Int32(_SCALE_BLOCK // 2)
                + half_block * cutlass.Int32(_BF16_VEC_SIZE // 2)
            )
            prims.store_ext(packed_word, fp4_ptr + fp4_index)

    if IKET_PHASES:
        cute.experimental.iket.range_pop()
    if cutlass.const_expr(not ALL_TO_ALL_CLUSTER_REDUCTION):
        # Keeps the root CTA's shared memory alive until every peer has read
        # inv_rms through DSMEM; the all-to-all path has no such read.
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()

    # A successor can launch here, but its entry wait precedes its signal-state
    # load, so it cannot select a Lamport slot until this grid has published it.
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(
            not PDL_RELEASE_STRICT and not PDL_RELEASE_PRE_NORM and not PDL_RELEASE_AT_ENTRY
        ):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()

    if USE_DEVICE_SIGNAL_STATE and not LAST_CTA_SIGNAL_ADVANCE:
        if AGGREGATE_SIGNAL_AT_CLUSTER_ROOT and (cta_rank == 0) and (tidx == 0):
            prims.red(
                prims.ReductionOp.ADD,
                prims.ReductionType.S32,
                state_counter_ptr,
                cutlass.Int32(CTAS_PER_ROW),
                mem_order=prims.MemOrder.RELEASE,
                mem_scope=prims.MemScope.GPU,
            )
        if (bidx == 0) & (tidx == 0):
            entered_ctas = cute.arch.load(
                state_counter_ptr,
                cutlass.Int32,
                sem="acquire",
                scope="gpu",
            )
            while entered_ctas < grid_x:
                entered_ctas = cute.arch.load(
                    state_counter_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="gpu",
                )
            active_vectors = (
                cutlass.Int32(cute.size(input_tensor, mode=[0])) * cutlass.Int32(N)
            ) // cutlass.Int32(_BF16_VEC_SIZE)
            ping_count_ptr = (
                signal_state.iterator
                + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
                + cutlass.Int64(ping_slot)
            ).llvm_ptr
            next_signal = (signal_i32 + cutlass.Int32(1)) % cutlass.Int32(_PING_PONG_SIZE)
            cute.arch.store(
                ping_count_ptr,
                active_vectors,
                sem="release",
                scope="gpu",
            )
            cute.arch.store(
                state_signal_ptr,
                next_signal,
                sem="release",
                scope="gpu",
            )
            cute.arch.store(
                state_counter_ptr,
                cutlass.Int32(0),
                sem="release",
                scope="gpu",
            )
            if cutlass.const_expr(USE_PDL):
                if cutlass.const_expr(PDL_RELEASE_STRICT):
                    prims.red(
                        prims.ReductionOp.ADD,
                        prims.ReductionType.S32,
                        release_epoch_ptr,
                        cutlass.Int32(1),
                        mem_order=prims.MemOrder.RELEASE,
                        mem_scope=prims.MemScope.GPU,
                    )

    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(PDL_RELEASE_STRICT):
            if tidx == 0:
                if USE_DEVICE_SIGNAL_STATE:
                    published_epoch = cute.arch.load(
                        release_epoch_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="gpu",
                    )
                    while published_epoch == entry_release_epoch:
                        published_epoch = cute.arch.load(
                            release_epoch_ptr,
                            cutlass.Int32,
                            sem="acquire",
                            scope="gpu",
                        )
                cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal: cutlass.Int64,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
) -> None:
    M = input_tensor.shape[0]
    _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_kernel(
        input_tensor,
        residual_tensor,
        norm_weight_tensor,
        norm_output_tensor,
        residual_output_tensor,
        buffers,
        ready_flags,
        signal,
        eps,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
    ).launch(
        grid=(M, 1, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
    )


@cute.jit
def _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    multicast_base: cutlass.Int64,
    signal: cutlass.Int64,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    CTAS_PER_ROW: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    USE_READY_FLAGS: cutlass.Constexpr[bool],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    ELIDE_SELF_RANK: cutlass.Constexpr[bool],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    PREFETCH_FUSED_INPUTS: cutlass.Constexpr[bool],
    DIRECT_CLUSTER_REDUCTION: cutlass.Constexpr[bool],
    POST_NORM_CLUSTER_BARRIER: cutlass.Constexpr[bool],
    FP32_ACCUMULATION: cutlass.Constexpr[bool],
    BATCH_POLL_PEERS: cutlass.Constexpr[bool],
    READY_MASK_POLL_PEERS: cutlass.Constexpr[bool],
    AGGREGATE_SIGNAL_AT_CLUSTER_ROOT: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_STRICT: cutlass.Constexpr[bool],
    PDL_RELEASE_PRE_NORM: cutlass.Constexpr[bool],
    PDL_RELEASE_AT_ENTRY: cutlass.Constexpr[bool],
    LAST_CTA_SIGNAL_ADVANCE: cutlass.Constexpr[bool],
    ALL_TO_ALL_CLUSTER_REDUCTION: cutlass.Constexpr[bool],
    EARLY_INPUT_LOAD: cutlass.Constexpr[bool],
    MULTICAST_PUBLISH: cutlass.Constexpr[bool],
    IKET_PHASES: cutlass.Constexpr[bool],
    ROTATED_POLL_ORDER: cutlass.Constexpr[bool],
) -> None:
    M = input_tensor.shape[0]
    _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel(
        input_tensor,
        residual_tensor,
        norm_weight_tensor,
        norm_output_tensor,
        residual_output_tensor,
        buffers,
        ready_flags,
        signal_state,
        multicast_base,
        signal,
        eps,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        CTAS_PER_ROW,
        THREADS_PER_BLOCK,
        USE_READY_FLAGS,
        USE_DEVICE_SIGNAL_STATE,
        ELIDE_SELF_RANK,
        PEER_ORDER,
        PREFETCH_FUSED_INPUTS,
        DIRECT_CLUSTER_REDUCTION,
        POST_NORM_CLUSTER_BARRIER,
        FP32_ACCUMULATION,
        BATCH_POLL_PEERS,
        READY_MASK_POLL_PEERS,
        AGGREGATE_SIGNAL_AT_CLUSTER_ROOT,
        USE_PDL,
        PDL_RELEASE_STRICT,
        PDL_RELEASE_PRE_NORM,
        PDL_RELEASE_AT_ENTRY,
        LAST_CTA_SIGNAL_ADVANCE,
        ALL_TO_ALL_CLUSTER_REDUCTION,
        EARLY_INPUT_LOAD,
        MULTICAST_PUBLISH,
        IKET_PHASES,
        ROTATED_POLL_ORDER,
    ).launch(
        grid=(M * CTAS_PER_ROW, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        cluster=(CTAS_PER_ROW, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


@cute.jit
def _glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    global_scale_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    fp4_output_tensor: cute.Tensor,
    scale_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    multicast_base: cutlass.Int64,
    signal: cutlass.Int64,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    CTAS_PER_ROW: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    USE_READY_FLAGS: cutlass.Constexpr[bool],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    ELIDE_SELF_RANK: cutlass.Constexpr[bool],
    AGGREGATE_SIGNAL_AT_CLUSTER_ROOT: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_STRICT: cutlass.Constexpr[bool],
    PDL_RELEASE_PRE_NORM: cutlass.Constexpr[bool],
    PDL_RELEASE_AT_ENTRY: cutlass.Constexpr[bool],
    LAST_CTA_SIGNAL_ADVANCE: cutlass.Constexpr[bool],
    ALL_TO_ALL_CLUSTER_REDUCTION: cutlass.Constexpr[bool],
    EARLY_INPUT_LOAD: cutlass.Constexpr[bool],
    MULTICAST_PUBLISH: cutlass.Constexpr[bool],
    IKET_PHASES: cutlass.Constexpr[bool],
    ROTATED_POLL_ORDER: cutlass.Constexpr[bool],
    PREFETCH_FUSED_INPUTS: cutlass.Constexpr[bool],
) -> None:
    M = input_tensor.shape[0]
    _glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel(
        input_tensor,
        residual_tensor,
        norm_weight_tensor,
        global_scale_tensor,
        norm_output_tensor,
        fp4_output_tensor,
        scale_output_tensor,
        residual_output_tensor,
        buffers,
        ready_flags,
        signal_state,
        multicast_base,
        signal,
        eps,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        CTAS_PER_ROW,
        THREADS_PER_BLOCK,
        USE_READY_FLAGS,
        USE_DEVICE_SIGNAL_STATE,
        PEER_ORDER,
        ELIDE_SELF_RANK,
        AGGREGATE_SIGNAL_AT_CLUSTER_ROOT,
        USE_PDL,
        PDL_RELEASE_STRICT,
        PDL_RELEASE_PRE_NORM,
        PDL_RELEASE_AT_ENTRY,
        LAST_CTA_SIGNAL_ADVANCE,
        ALL_TO_ALL_CLUSTER_REDUCTION,
        EARLY_INPUT_LOAD,
        MULTICAST_PUBLISH,
        IKET_PHASES,
        PREFETCH_FUSED_INPUTS,
        ROTATED_POLL_ORDER,
    ).launch(
        grid=(M * CTAS_PER_ROW, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        cluster=(CTAS_PER_ROW, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


# ---------------------------------------------------------------------------
# Prefill-class two-shot NVLS variant of the fused RMSNorm and out-quant kernels.
#
# The one-shot kernels above need every rank to receive the three other
# partials in full (ingress 3P), which is the algorithmic floor at prefill M
# (P = 100.7 MB at M=8192).  This variant keeps the fused epilogue local and
# changes only how the sum is formed:
#
#   step 1  every CTA copies its rows of the local partial into region
#           ``_TWOSHOT_INPUT_REGION`` of the current Lamport slot (local HBM
#           write) and release-adds the block's input-ready flag on the owner
#           rank (owner = block % WORLD_SIZE).
#   step 2  the owner CTA waits for WORLD_SIZE input-ready arrivals, then for
#           each of its rows issues ``multimem.ld_reduce`` on the multicast
#           address of the input region: the NVSwitch fetches the four
#           partials and returns their f32-accumulated BF16 sum.  The same
#           16 B words are pushed back with one ``multimem.st`` into region
#           ``_TWOSHOT_SUM_REGION`` of every rank, and the owner runs the
#           residual/RMSNorm(/NVFP4) epilogue on its rows from registers.
#           It then release-adds the block's sum-ready flag on each peer.
#   step 3  the three non-owner CTAs wait for the sum-ready flag and run the
#           same epilogue on the rows that arrived via multicast.
#
# Per rank and call this moves ~P out and ~P in instead of 3P in; the flags
# are reset by their single reader, so the 3-slot rotation, the clear phase
# and the device signal state are shared unchanged with the one-shot kernels
# (a decode launch may follow a prefill launch on the same workspace).
# ---------------------------------------------------------------------------
_TWOSHOT_INPUT_REGION = 0
_TWOSHOT_SUM_REGION = 1
_TWOSHOT_FLAG_INPUT_READY = 0
_TWOSHOT_FLAG_SUM_READY = 1
_TWOSHOT_THREADS_PER_BLOCK = 256
_TWOSHOT_THREADS_PER_BLOCK_CHOICES = (128, 192, 256, 384, 768)
_TWOSHOT_ROWS_PER_CTA = 4
# Diagnostic option, 0 in normal use: 3 runs the full two-shot protocol but skips the
# sentinel clear of the next slot.
_TWOSHOT_DEBUG_PHASE = 0
# Cap on the number of persistent two-shot CTAs (0 = one CTA per row-block group, no pipelining).
_TWOSHOT_MAX_CTAS = 256
# Number of dedicated CONSUMER CTAs (peer-block epilogues). 0 = every CTA does both roles.
# With C > 0 the grid is P producer CTAs (staging, ld_reduce, multicast, own epilogue) followed by
# C consumer CTAs, so reduce-scatter egress, all-gather ingress and the local staging copy proceed
# concurrently instead of phase by phase.
_TWOSHOT_CONSUMER_CTAS = 1024
# 1 (default): reduce-scatter / all-gather over direct peer loads (unicast NVLink reads, fp32 accumulation
# in fixed rank order by the block owner). 0 selects the multimem.ld_reduce / multimem.st (NVLS) transport,
# which needs a multicast_ptr and is slower than direct loads at prefill sizes.
_TWOSHOT_DIRECT = 1


if _dsl_user_op is not None:

    @_dsl_user_op
    def _multimem_ld_reduce_bf16x8_words(mc_addr, *, loc=None, ip=None):
        """``multimem.ld_reduce.acc::f32.v4.bf16x2`` of one 16 B vector.

        The NVSwitch fetches the eight BF16 lanes at ``mc_addr`` from every
        rank of the multicast group, accumulates them in f32 and returns the
        BF16 sum as four b32 words (raw MLIR values, like the store operand).
        """
        mc_ptr_int = mc_addr.ir_value(loc=loc, ip=ip)
        return_struct = _llvm.inline_asm(
            _ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
            [mc_ptr_int],
            "multimem.ld_reduce.weak.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
        return tuple(
            _llvm.extractvalue(_T.i32(), return_struct, [index], loc=loc, ip=ip)
            for index in range(4)
        )

else:  # pragma: no cover - older DSL packages

    def _multimem_ld_reduce_bf16x8_words(*args, **kwargs):
        raise RuntimeError("multimem ld_reduce requires cutlass._mlir.dialects.llvm")


@cute.jit
def _twoshot_row_epilogue(
    row,
    sum_values,
    residual_array,
    norm_weight_array,
    norm_output_array,
    residual_output_array,
    warp_sums,
    tidx,
    lane_id,
    warp_id,
    global_scale,
    fp4_ptr,
    sf_ptr,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    VECS_PER_THREAD: cutlass.Constexpr[int],
    EMIT_NVFP4: cutlass.Constexpr[bool],
):
    """Residual add, row RMSNorm and optional NVFP4 quant of one full row.

    ``sum_values`` holds this thread's ``VECS_PER_THREAD`` BF16x8 vectors of
    the reduced row (same lane layout as the input copy: vector ``v`` covers
    columns ``(tidx + v * THREADS_PER_BLOCK) * 8``).  Every thread of the CTA
    must call this for the same row; the whole CTA owns the row so the
    sum-of-squares reduction is a plain CTA reduction, no cluster needed.
    """
    residual_out_values = cutlass.Array(
        cutlass.BFloat16,
        VECS_PER_THREAD * _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    weight_values = cutlass.Array(
        cutlass.BFloat16,
        VECS_PER_THREAD * _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    local_sum_sq = cutlass.Float32(0.0)
    for v in cutlass.range_constexpr(VECS_PER_THREAD):
        col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(_BF16_VEC_SIZE)
        lane_base = v * _BF16_VEC_SIZE
        sum_vec = sum_values[lane_base:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)]
        residual_vec = residual_array[row, :].load(col, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES)
        weight_vec = norm_weight_array.load(col, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES)
        residual_out_vec = (sum_vec.to(cutlass.Float32) + residual_vec.to(cutlass.Float32)).to(
            cutlass.BFloat16
        )
        residual_output_array[row, :].store(
            residual_out_vec, col, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES
        )
        residual_out_values[lane_base:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = (
            residual_out_vec
        )
        weight_values[lane_base:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = weight_vec
        residual_f32 = residual_out_vec.to(cutlass.Float32)
        local_sum_sq = local_sum_sq + (residual_f32 * residual_f32).reduce("add")

    warp_sum = _warp_sum_f32(local_sum_sq)
    if lane_id == 0:
        warp_sums[warp_id] = warp_sum
    prims.barrier_cta_sync(0)
    # Every thread sums the warp partials in the same fixed order, so the row
    # gets one bit-identical inverse RMS without a broadcast.
    cta_sum = cutlass.Float32(0.0)
    for warp_idx in cutlass.range_constexpr(THREADS_PER_BLOCK // 32):
        cta_sum = cta_sum + warp_sums[warp_idx]
    inv_rms = cute.math.rsqrt(cta_sum * (1.0 / cutlass.Float32(N)) + eps, fastmath=True)
    # The next row reuses warp_sums; keep the slow readers from seeing the
    # next row's partials.
    prims.barrier_cta_sync(0)

    for v in cutlass.range_constexpr(VECS_PER_THREAD):
        col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(_BF16_VEC_SIZE)
        lane_base = v * _BF16_VEC_SIZE
        residual_vec = residual_out_values[
            lane_base:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)
        ]
        weight_vec = weight_values[lane_base:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)]
        norm_f32 = residual_vec.to(cutlass.Float32) * inv_rms
        norm_f32 = norm_f32 * weight_vec.to(cutlass.Float32)
        norm_bf16 = norm_f32.to(cutlass.BFloat16)
        norm_output_array[row, :].store(norm_bf16, col, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES)
        if cutlass.const_expr(EMIT_NVFP4):
            # Same linear-layout NVFP4 contract as the one-shot out-quant kernel:
            # lanes (tidx, tidx^1) hold the two 8-column halves of one
            # 16-column scale block, so a butterfly shuffle of distance 1
            # gives the block max.
            quant_f32 = norm_bf16.to(cutlass.Float32)
            local_max = cutlass.Float32(0.0)
            for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
                magnitude = cute.math.abs(quant_f32[lane])
                if magnitude > local_max:
                    local_max = magnitude
            block_max = local_max
            peer_max = cute.arch.shuffle_sync_bfly(block_max, 1)
            if peer_max > block_max:
                block_max = peer_max
            sf_unquantized = global_scale * block_max * cutlass.Float32(1.0 / 6.0)
            sf_packed = prims.cvt_f32x2_to_f8x2(
                sf_unquantized,
                sf_unquantized,
                cutlass.Float8E4M3FN,
                rnd=prims.FPRoundingMode.RN,
                sat=prims.SaturationMode.SATFINITE,
            )
            sf_code = cutlass.Int32(sf_packed) & cutlass.Int32(0xFF)
            half_block = (col // cutlass.Int32(_BF16_VEC_SIZE)) & cutlass.Int32(1)
            scale_block = col // cutlass.Int32(_SCALE_BLOCK)
            if half_block == cutlass.Int32(0):
                sf_index = row * cutlass.Int32(N // _SCALE_BLOCK) + scale_block
                prims.store_ext(cutlass.Int8(sf_code), sf_ptr + sf_index)
            narrowed_sf = _decode_positive_e4m3(sf_code)
            output_scale = cutlass.Float32(0.0)
            if block_max != cutlass.Float32(0.0):
                output_scale = cute.arch.rcp_approx(
                    narrowed_sf * cute.arch.rcp_approx(global_scale)
                )
            packed_word = cutlass.Int32(0)
            for pair in cutlass.range_constexpr(_BF16_VEC_SIZE // 2):
                low = quant_f32[2 * pair] * output_scale
                high = quant_f32[2 * pair + 1] * output_scale
                packed_byte = prims.cvt_f32x2_to_f4x2(
                    high,
                    low,
                    cutlass.Float4E2M1FN,
                    rnd=prims.FPRoundingMode.RN,
                ) & cutlass.Int32(0xFF)
                packed_word = packed_word | (packed_byte << cutlass.Int32(pair * 8))
            fp4_index = (
                row * cutlass.Int32(N // 2)
                + scale_block * cutlass.Int32(_SCALE_BLOCK // 2)
                + half_block * cutlass.Int32(_BF16_VEC_SIZE // 2)
            )
            prims.store_ext(packed_word, fp4_ptr + fp4_index)


@cute.jit
def _twoshot_stage_group(
    group,
    tidx,
    m_rows,
    input_array,
    local_buffer,
    ready_flags,
    stage_values,
    input_region_base,
    flag_in_base,
    N: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    ROWS_PER_CTA: cutlass.Constexpr[int],
    VECS_PER_THREAD: cutlass.Constexpr[int],
):
    """Copy this rank's rows of the WORLD_SIZE row blocks of ``group`` into the input region
    and release one input-ready flag per block on that block's owner rank.

    The loads of a whole block (ROWS_PER_CTA x VECS_PER_THREAD vectors per thread) are issued
    before its stores so the copy is not a chain of dependent 16 B load->store pairs.
    """
    row_block_rows = cutlass.Int32(ROWS_PER_CTA)
    for r in cutlass.range_constexpr(WORLD_SIZE):
        block = group * cutlass.Int32(WORLD_SIZE) + cutlass.Int32(r)
        row0 = block * row_block_rows
        if row0 < m_rows:
            for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                row = row0 + cutlass.Int32(rr)
                if row < m_rows:
                    for v in cutlass.range_constexpr(VECS_PER_THREAD):
                        col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(
                            _BF16_VEC_SIZE
                        )
                        base = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                        stage_values[base:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)] = (
                            input_array[row, :].load(
                                col, _BF16_VEC_SIZE, alignment=_ALIGNMENT_BYTES
                            )
                        )
            for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                row = row0 + cutlass.Int32(rr)
                if row < m_rows:
                    row_offset = input_region_base + cutlass.Int64(row) * cutlass.Int64(N)
                    for v in cutlass.range_constexpr(VECS_PER_THREAD):
                        col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(
                            _BF16_VEC_SIZE
                        )
                        base = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                        local_buffer.store(
                            stage_values[base:_BF16_VEC_SIZE, cutlass.align(_ALIGNMENT_BYTES)],
                            row_offset + cutlass.Int64(col),
                            _BF16_VEC_SIZE,
                            alignment=_ALIGNMENT_BYTES,
                        )
    prims.barrier_cta_sync(0)
    if tidx == 0:
        for r in cutlass.range_constexpr(WORLD_SIZE):
            block = group * cutlass.Int32(WORLD_SIZE) + cutlass.Int32(r)
            if block * row_block_rows < m_rows:
                prims.red(
                    prims.ReductionOp.ADD,
                    prims.ReductionType.S32,
                    (ready_flags[r].iterator + flag_in_base + cutlass.Int64(block)).llvm_ptr,
                    cutlass.Int32(1),
                    mem_order=prims.MemOrder.RELEASE,
                    mem_scope=prims.MemScope.SYS,
                )


@cute.kernel
def _glm_tp_allreduce_residual_rms_norm_twoshot_nvls_kernel(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    global_scale_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    fp4_output_tensor: cute.Tensor,
    scale_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    multicast_base: cutlass.Int64,
    signal: cutlass.Int64,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    ROWS_PER_CTA: cutlass.Constexpr[int],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_AT_ENTRY: cutlass.Constexpr[bool],
    EMIT_NVFP4: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    lane_id = tidx % 32
    warp_id = tidx // 32
    VECS_PER_THREAD = N // (_BF16_VEC_SIZE * THREADS_PER_BLOCK)

    m_rows = cutlass.Int32(cute.size(input_tensor, mode=[0]))
    input_array = cutlass.make_array_view(input_tensor)
    residual_array = cutlass.make_array_view(residual_tensor)
    norm_weight_array = cutlass.make_array_view(norm_weight_tensor)
    norm_output_array = cutlass.make_array_view(norm_output_tensor)
    residual_output_array = cutlass.make_array_view(residual_output_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]
    fp4_ptr = fp4_output_tensor.iterator.raw_ptr()
    sf_ptr = scale_output_tensor.iterator.raw_ptr()

    warp_sums = cutlass.Array(
        cutlass.Float32,
        (THREADS_PER_BLOCK // 32),
        space=cutlass.AddressSpace.smem,
        alignment=_ALIGNMENT_BYTES,
    )

    numel = cute.size(buffers[0], mode=[2])
    numel_i64 = cutlass.Int64(numel)
    num_vectors = cute.size(ready_flags[0], mode=[2])
    num_vectors_i64 = cutlass.Int64(num_vectors)
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_VALUE_INDEX)).llvm_ptr

    if cutlass.const_expr(USE_PDL):
        cute.arch.griddepcontrol_wait()
    global_scale = cutlass.Float32(0.0)
    if cutlass.const_expr(EMIT_NVFP4):
        global_scale = prims.load_ext(global_scale_tensor.iterator.raw_ptr(), dtype=cutlass.Float32)

    signal_i32 = cutlass.Int32(0)
    signal_i64 = signal
    if USE_DEVICE_SIGNAL_STATE:
        signal_i32 = cute.arch.load(
            state_signal_ptr,
            cutlass.Int32,
            sem="acquire",
            scope="gpu",
        )
        signal_i64 = cutlass.Int64(signal_i32)
    ping_slot = signal_i64 % _PING_PONG_SIZE
    clear_slot = (signal_i64 + cutlass.Int64(1)) % _PING_PONG_SIZE
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * numel_i64
    clear_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * numel_i64
    input_region_base = ping_base + cutlass.Int64(_TWOSHOT_INPUT_REGION) * numel_i64
    sum_region_base = ping_base + cutlass.Int64(_TWOSHOT_SUM_REGION) * numel_i64
    flag_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * num_vectors_i64
    active_vectors_i32 = (m_rows * cutlass.Int32(N)) // cutlass.Int32(_BF16_VEC_SIZE)
    if USE_DEVICE_SIGNAL_STATE:
        if tidx == 0:
            _advance_signal_if_last_cta(
                state_counter_ptr,
                state_signal_ptr,
                signal_state,
                signal_i32,
                ping_slot,
                active_vectors_i32,
                grid_x,
            )
    # Deliberately after the device signal snapshot (see the one-shot kernels).
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(PDL_RELEASE_AT_ENTRY):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()

    # Clear the slot the next launch will use back to the sentinel, exactly as
    # the one-shot kernels do, so a decode one-shot launch can follow.
    neg_zero_vec = cutlass.Array(
        cutlass.BFloat16,
        _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    neg_zero_bits = cutlass.Int16(_NEGATIVE_ZERO_BF16_BITS)
    for lane in cutlass.range_constexpr(_BF16_VEC_SIZE):
        neg_zero_vec[lane] = neg_zero_bits.bitcast(cutlass.BFloat16)
    if cutlass.const_expr(_TWOSHOT_DEBUG_PHASE != 3):
        if USE_DEVICE_SIGNAL_STATE:
            clear_count_ptr = (
                signal_state.iterator
                + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
                + cutlass.Int64(clear_slot)
            ).llvm_ptr
            clear_vectors = cute.arch.load(
                clear_count_ptr,
                cutlass.Int32,
                sem="acquire",
                scope="gpu",
            )
            clear_vector_idx = bidx * THREADS_PER_BLOCK + tidx
            clear_stride = grid_x * THREADS_PER_BLOCK
            while clear_vector_idx < clear_vectors:
                clear_vector_idx_i64 = cutlass.Int64(clear_vector_idx)
                for source_rank in cutlass.range_constexpr(WORLD_SIZE):
                    clear_offset = (
                        clear_base
                        + cutlass.Int64(source_rank) * numel_i64
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

    # Persistent CTAs over row-block GROUPS (group g = blocks g*WORLD_SIZE+r,
    # this rank owns block g*WORLD_SIZE+LOCAL_RANK). Each CTA walks groups bidx, bidx+grid, ...
    # and stages group g+1 BEFORE waiting on group g, so the local HBM copy of the next group
    # overlaps the switch round trips (ld_reduce / multicast / peer flags) of the current one.
    # With grid == number of groups (default) each CTA handles one group (no pipelining).
    group_rows = cutlass.Int32(ROWS_PER_CTA * WORLD_SIZE)
    num_groups = (m_rows + group_rows - cutlass.Int32(1)) // group_rows
    row_block_rows = cutlass.Int32(ROWS_PER_CTA)
    flag_in_base = flag_base + cutlass.Int64(_TWOSHOT_FLAG_INPUT_READY) * num_vectors_i64
    flag_sum_base = flag_base + cutlass.Int64(_TWOSHOT_FLAG_SUM_READY) * num_vectors_i64

    stage_values = cutlass.Array(
        cutlass.BFloat16,
        ROWS_PER_CTA * VECS_PER_THREAD * _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    sum_values = cutlass.Array(
        cutlass.BFloat16,
        VECS_PER_THREAD * _BF16_VEC_SIZE,
        alignment=_ALIGNMENT_BYTES,
    )
    # All ld_reduce results of the owned block are issued before any is consumed so the
    # switch round trips overlap (ROWS_PER_CTA x VECS_PER_THREAD x 4 words in flight).
    owner_words = cutlass.Array(
        cutlass.Int32,
        ROWS_PER_CTA * VECS_PER_THREAD * 4,
        alignment=_ALIGNMENT_BYTES,
    )

    if cutlass.const_expr(_TWOSHOT_CONSUMER_CTAS == 0):
        # Single-role schedule: every CTA stages, reduces its own block and then consumes the peers.
        group = bidx
        if group < num_groups:
            _twoshot_stage_group(
                group,
                tidx,
                m_rows,
                input_array,
                local_buffer,
                ready_flags,
                stage_values,
                input_region_base,
                flag_in_base,
                N,
                WORLD_SIZE,
                THREADS_PER_BLOCK,
                ROWS_PER_CTA,
                VECS_PER_THREAD,
            )
        while group < num_groups:
            next_group = group + grid_x
            if next_group < num_groups:
                _twoshot_stage_group(
                    next_group,
                    tidx,
                    m_rows,
                    input_array,
                    local_buffer,
                    ready_flags,
                    stage_values,
                    input_region_base,
                    flag_in_base,
                    N,
                    WORLD_SIZE,
                    THREADS_PER_BLOCK,
                    ROWS_PER_CTA,
                    VECS_PER_THREAD,
                )

            # ---- own block: wait for the WORLD_SIZE input-ready arrivals, switch-reduce,
            # multicast the sum to every rank, local fused epilogue, release sum-ready on peers.
            own_block = group * cutlass.Int32(WORLD_SIZE) + cutlass.Int32(LOCAL_RANK)
            own_row0 = own_block * row_block_rows
            if own_row0 < m_rows:
                own_in_flag_ptr = (
                    ready_flags[LOCAL_RANK].iterator + flag_in_base + cutlass.Int64(own_block)
                ).llvm_ptr
                if tidx == 0:
                    arrivals = cute.arch.load(
                        own_in_flag_ptr, cutlass.Int32, sem="acquire", scope="sys"
                    )
                    while arrivals < cutlass.Int32(WORLD_SIZE):
                        arrivals = cute.arch.load(
                            own_in_flag_ptr, cutlass.Int32, sem="acquire", scope="sys"
                        )
                    # Single reader: consume the arrivals so the slot is clean for its next use.
                    prims.red(
                        prims.ReductionOp.ADD,
                        prims.ReductionType.S32,
                        own_in_flag_ptr,
                        cutlass.Int32(-WORLD_SIZE),
                        mem_order=prims.MemOrder.RELAXED,
                        mem_scope=prims.MemScope.SYS,
                    )
                prims.barrier_cta_sync(0)
                if cutlass.const_expr(_TWOSHOT_DIRECT):
                    # Direct reduce-scatter: read the block rows of every rank (fixed order 0..W-1, fp32),
                    # keep the BF16 sums in registers, then store them once into the local sum region.
                    for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                        row = own_row0 + cutlass.Int32(rr)
                        if row < m_rows:
                            in_row_offset = input_region_base + cutlass.Int64(row) * cutlass.Int64(
                                N
                            )
                            for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(
                                    _BF16_VEC_SIZE
                                )
                                acc_f32 = (
                                    buffer_arrays[0]
                                    .load(
                                        in_row_offset + cutlass.Int64(col),
                                        _BF16_VEC_SIZE,
                                        alignment=_ALIGNMENT_BYTES,
                                        is_volatile=True,
                                    )
                                    .to(cutlass.Float32)
                                )
                                for src in cutlass.range_constexpr(1, WORLD_SIZE):
                                    acc_f32 = acc_f32 + buffer_arrays[src].load(
                                        in_row_offset + cutlass.Int64(col),
                                        _BF16_VEC_SIZE,
                                        alignment=_ALIGNMENT_BYTES,
                                        is_volatile=True,
                                    ).to(cutlass.Float32)
                                pbase = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                                stage_values[
                                    pbase:_BF16_VEC_SIZE,
                                    cutlass.align(_ALIGNMENT_BYTES),
                                ] = acc_f32.to(cutlass.BFloat16)
                    for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                        row = own_row0 + cutlass.Int32(rr)
                        if row < m_rows:
                            sum_row_offset = sum_region_base + cutlass.Int64(row) * cutlass.Int64(N)
                            for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(
                                    _BF16_VEC_SIZE
                                )
                                pbase = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                                local_buffer.store(
                                    stage_values[
                                        pbase:_BF16_VEC_SIZE,
                                        cutlass.align(_ALIGNMENT_BYTES),
                                    ],
                                    sum_row_offset + cutlass.Int64(col),
                                    _BF16_VEC_SIZE,
                                    alignment=_ALIGNMENT_BYTES,
                                )
                                for e in cutlass.range_constexpr(_BF16_VEC_SIZE):
                                    sum_values[v * _BF16_VEC_SIZE + e] = stage_values[pbase + e]
                            _twoshot_row_epilogue(
                                row,
                                sum_values,
                                residual_array,
                                norm_weight_array,
                                norm_output_array,
                                residual_output_array,
                                warp_sums,
                                tidx,
                                lane_id,
                                warp_id,
                                global_scale,
                                fp4_ptr,
                                sf_ptr,
                                eps,
                                N,
                                THREADS_PER_BLOCK,
                                VECS_PER_THREAD,
                                EMIT_NVFP4,
                            )
                else:
                    for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                        row = own_row0 + cutlass.Int32(rr)
                        if row < m_rows:
                            in_row_offset = input_region_base + cutlass.Int64(row) * cutlass.Int64(
                                N
                            )
                            for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(
                                    _BF16_VEC_SIZE
                                )
                                word0, word1, word2, word3 = _multimem_ld_reduce_bf16x8_words(
                                    multicast_base
                                    + (in_row_offset + cutlass.Int64(col)) * cutlass.Int64(2)
                                )
                                wbase = (rr * VECS_PER_THREAD + v) * 4
                                owner_words[wbase] = cutlass.Int32(word0)
                                owner_words[wbase + 1] = cutlass.Int32(word1)
                                owner_words[wbase + 2] = cutlass.Int32(word2)
                                owner_words[wbase + 3] = cutlass.Int32(word3)
                    for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                        row = own_row0 + cutlass.Int32(rr)
                        if row < m_rows:
                            sum_row_offset = sum_region_base + cutlass.Int64(row) * cutlass.Int64(N)
                            for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(
                                    _BF16_VEC_SIZE
                                )
                                wbase = (rr * VECS_PER_THREAD + v) * 4
                                w0 = owner_words[wbase]
                                w1 = owner_words[wbase + 1]
                                w2 = owner_words[wbase + 2]
                                w3 = owner_words[wbase + 3]
                                _multimem_st_v4_b32(
                                    multicast_base
                                    + (sum_row_offset + cutlass.Int64(col)) * cutlass.Int64(2),
                                    w0,
                                    w1,
                                    w2,
                                    w3,
                                )
                                words = [w0, w1, w2, w3]
                                lane_base = v * _BF16_VEC_SIZE
                                for pair in cutlass.range_constexpr(_BF16_VEC_SIZE // 2):
                                    word = words[pair]
                                    low_bits = (word << cutlass.Int32(16)) >> cutlass.Int32(16)
                                    high_bits = word >> cutlass.Int32(16)
                                    # Constructor narrowing (as cutlass.Int8(sf_code) in the out-quant kernel);
                                    # both halves were sign-extended by the arithmetic shifts.
                                    sum_values[lane_base + 2 * pair] = cutlass.Int16(
                                        low_bits
                                    ).bitcast(cutlass.BFloat16)
                                    sum_values[lane_base + 2 * pair + 1] = cutlass.Int16(
                                        high_bits
                                    ).bitcast(cutlass.BFloat16)
                            _twoshot_row_epilogue(
                                row,
                                sum_values,
                                residual_array,
                                norm_weight_array,
                                norm_output_array,
                                residual_output_array,
                                warp_sums,
                                tidx,
                                lane_id,
                                warp_id,
                                global_scale,
                                fp4_ptr,
                                sf_ptr,
                                eps,
                                N,
                                THREADS_PER_BLOCK,
                                VECS_PER_THREAD,
                                EMIT_NVFP4,
                            )
                prims.barrier_cta_sync(0)
                if tidx == 0:
                    for target_rank in cutlass.range_constexpr(WORLD_SIZE):
                        if cutlass.const_expr(target_rank != LOCAL_RANK):
                            prims.red(
                                prims.ReductionOp.ADD,
                                prims.ReductionType.S32,
                                (
                                    ready_flags[target_rank].iterator
                                    + flag_sum_base
                                    + cutlass.Int64(own_block)
                                ).llvm_ptr,
                                cutlass.Int32(1),
                                mem_order=prims.MemOrder.RELEASE,
                                mem_scope=prims.MemScope.SYS,
                            )

            # ---- the three peer blocks of the group: wait for each owner's multicast sum
            # (rotated start so the four ranks do not all wait on the same owner first).
            for peer_ordinal in cutlass.range_constexpr(1, WORLD_SIZE):
                peer_rank = (LOCAL_RANK + peer_ordinal) % WORLD_SIZE
                peer_block = group * cutlass.Int32(WORLD_SIZE) + cutlass.Int32(peer_rank)
                peer_row0 = peer_block * row_block_rows
                if peer_row0 < m_rows:
                    peer_sum_flag_ptr = (
                        ready_flags[LOCAL_RANK].iterator + flag_sum_base + cutlass.Int64(peer_block)
                    ).llvm_ptr
                    if tidx == 0:
                        ready = cute.arch.load(
                            peer_sum_flag_ptr, cutlass.Int32, sem="acquire", scope="sys"
                        )
                        while ready < cutlass.Int32(1):
                            ready = cute.arch.load(
                                peer_sum_flag_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )
                        prims.red(
                            prims.ReductionOp.ADD,
                            prims.ReductionType.S32,
                            peer_sum_flag_ptr,
                            cutlass.Int32(-1),
                            mem_order=prims.MemOrder.RELAXED,
                            mem_scope=prims.MemScope.SYS,
                        )
                    prims.barrier_cta_sync(0)
                    # Batched: issue every row's loads of the block before the first epilogue so the
                    # CTA is not one latency-bound row at a time.
                    for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                        row = peer_row0 + cutlass.Int32(rr)
                        if row < m_rows:
                            sum_row_offset = sum_region_base + cutlass.Int64(row) * cutlass.Int64(N)
                            for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                col = (tidx + cutlass.Int32(v * THREADS_PER_BLOCK)) * cutlass.Int32(
                                    _BF16_VEC_SIZE
                                )
                                pbase = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                                if cutlass.const_expr(_TWOSHOT_DIRECT):
                                    stage_values[
                                        pbase:_BF16_VEC_SIZE,
                                        cutlass.align(_ALIGNMENT_BYTES),
                                    ] = buffer_arrays[peer_rank].load(
                                        sum_row_offset + cutlass.Int64(col),
                                        _BF16_VEC_SIZE,
                                        alignment=_ALIGNMENT_BYTES,
                                        is_volatile=True,
                                    )
                                else:
                                    stage_values[
                                        pbase:_BF16_VEC_SIZE,
                                        cutlass.align(_ALIGNMENT_BYTES),
                                    ] = local_buffer.load(
                                        sum_row_offset + cutlass.Int64(col),
                                        _BF16_VEC_SIZE,
                                        alignment=_ALIGNMENT_BYTES,
                                        is_volatile=True,
                                    )
                    for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                        row = peer_row0 + cutlass.Int32(rr)
                        if row < m_rows:
                            for e in cutlass.range_constexpr(VECS_PER_THREAD * _BF16_VEC_SIZE):
                                sum_values[e] = stage_values[
                                    rr * VECS_PER_THREAD * _BF16_VEC_SIZE + e
                                ]
                            _twoshot_row_epilogue(
                                row,
                                sum_values,
                                residual_array,
                                norm_weight_array,
                                norm_output_array,
                                residual_output_array,
                                warp_sums,
                                tidx,
                                lane_id,
                                warp_id,
                                global_scale,
                                fp4_ptr,
                                sf_ptr,
                                eps,
                                N,
                                THREADS_PER_BLOCK,
                                VECS_PER_THREAD,
                                EMIT_NVFP4,
                            )

            group = next_group
    else:
        # Split-role schedule: CTAs [0, P) are producers, CTAs [P, P + C) consumers (C constexpr).
        num_producers = grid_x - cutlass.Int32(_TWOSHOT_CONSUMER_CTAS)
        if bidx < num_producers:
            group = bidx
            if group < num_groups:
                _twoshot_stage_group(
                    group,
                    tidx,
                    m_rows,
                    input_array,
                    local_buffer,
                    ready_flags,
                    stage_values,
                    input_region_base,
                    flag_in_base,
                    N,
                    WORLD_SIZE,
                    THREADS_PER_BLOCK,
                    ROWS_PER_CTA,
                    VECS_PER_THREAD,
                )
            while group < num_groups:
                next_group = group + num_producers
                if next_group < num_groups:
                    _twoshot_stage_group(
                        next_group,
                        tidx,
                        m_rows,
                        input_array,
                        local_buffer,
                        ready_flags,
                        stage_values,
                        input_region_base,
                        flag_in_base,
                        N,
                        WORLD_SIZE,
                        THREADS_PER_BLOCK,
                        ROWS_PER_CTA,
                        VECS_PER_THREAD,
                    )
                # ---- own block: wait for the WORLD_SIZE input-ready arrivals, switch-reduce,
                # multicast the sum to every rank, local fused epilogue, release sum-ready on peers.
                own_block = group * cutlass.Int32(WORLD_SIZE) + cutlass.Int32(LOCAL_RANK)
                own_row0 = own_block * row_block_rows
                if own_row0 < m_rows:
                    own_in_flag_ptr = (
                        ready_flags[LOCAL_RANK].iterator + flag_in_base + cutlass.Int64(own_block)
                    ).llvm_ptr
                    if tidx == 0:
                        arrivals = cute.arch.load(
                            own_in_flag_ptr, cutlass.Int32, sem="acquire", scope="sys"
                        )
                        while arrivals < cutlass.Int32(WORLD_SIZE):
                            arrivals = cute.arch.load(
                                own_in_flag_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )
                        # Single reader: consume the arrivals so the slot is clean for its next use.
                        prims.red(
                            prims.ReductionOp.ADD,
                            prims.ReductionType.S32,
                            own_in_flag_ptr,
                            cutlass.Int32(-WORLD_SIZE),
                            mem_order=prims.MemOrder.RELAXED,
                            mem_scope=prims.MemScope.SYS,
                        )
                    prims.barrier_cta_sync(0)
                    if cutlass.const_expr(_TWOSHOT_DIRECT):
                        # Direct reduce-scatter: read the block rows of every rank (fixed order 0..W-1, fp32),
                        # keep the BF16 sums in registers, then store them once into the local sum region.
                        for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                            row = own_row0 + cutlass.Int32(rr)
                            if row < m_rows:
                                in_row_offset = input_region_base + cutlass.Int64(
                                    row
                                ) * cutlass.Int64(N)
                                for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                    col = (
                                        tidx + cutlass.Int32(v * THREADS_PER_BLOCK)
                                    ) * cutlass.Int32(_BF16_VEC_SIZE)
                                    acc_f32 = (
                                        buffer_arrays[0]
                                        .load(
                                            in_row_offset + cutlass.Int64(col),
                                            _BF16_VEC_SIZE,
                                            alignment=_ALIGNMENT_BYTES,
                                            is_volatile=True,
                                        )
                                        .to(cutlass.Float32)
                                    )
                                    for src in cutlass.range_constexpr(1, WORLD_SIZE):
                                        acc_f32 = acc_f32 + buffer_arrays[src].load(
                                            in_row_offset + cutlass.Int64(col),
                                            _BF16_VEC_SIZE,
                                            alignment=_ALIGNMENT_BYTES,
                                            is_volatile=True,
                                        ).to(cutlass.Float32)
                                    pbase = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                                    stage_values[
                                        pbase:_BF16_VEC_SIZE,
                                        cutlass.align(_ALIGNMENT_BYTES),
                                    ] = acc_f32.to(cutlass.BFloat16)
                        for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                            row = own_row0 + cutlass.Int32(rr)
                            if row < m_rows:
                                sum_row_offset = sum_region_base + cutlass.Int64(
                                    row
                                ) * cutlass.Int64(N)
                                for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                    col = (
                                        tidx + cutlass.Int32(v * THREADS_PER_BLOCK)
                                    ) * cutlass.Int32(_BF16_VEC_SIZE)
                                    pbase = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                                    local_buffer.store(
                                        stage_values[
                                            pbase:_BF16_VEC_SIZE,
                                            cutlass.align(_ALIGNMENT_BYTES),
                                        ],
                                        sum_row_offset + cutlass.Int64(col),
                                        _BF16_VEC_SIZE,
                                        alignment=_ALIGNMENT_BYTES,
                                    )
                                    for e in cutlass.range_constexpr(_BF16_VEC_SIZE):
                                        sum_values[v * _BF16_VEC_SIZE + e] = stage_values[pbase + e]
                                _twoshot_row_epilogue(
                                    row,
                                    sum_values,
                                    residual_array,
                                    norm_weight_array,
                                    norm_output_array,
                                    residual_output_array,
                                    warp_sums,
                                    tidx,
                                    lane_id,
                                    warp_id,
                                    global_scale,
                                    fp4_ptr,
                                    sf_ptr,
                                    eps,
                                    N,
                                    THREADS_PER_BLOCK,
                                    VECS_PER_THREAD,
                                    EMIT_NVFP4,
                                )
                    else:
                        for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                            row = own_row0 + cutlass.Int32(rr)
                            if row < m_rows:
                                in_row_offset = input_region_base + cutlass.Int64(
                                    row
                                ) * cutlass.Int64(N)
                                for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                    col = (
                                        tidx + cutlass.Int32(v * THREADS_PER_BLOCK)
                                    ) * cutlass.Int32(_BF16_VEC_SIZE)
                                    word0, word1, word2, word3 = _multimem_ld_reduce_bf16x8_words(
                                        multicast_base
                                        + (in_row_offset + cutlass.Int64(col)) * cutlass.Int64(2)
                                    )
                                    wbase = (rr * VECS_PER_THREAD + v) * 4
                                    owner_words[wbase] = cutlass.Int32(word0)
                                    owner_words[wbase + 1] = cutlass.Int32(word1)
                                    owner_words[wbase + 2] = cutlass.Int32(word2)
                                    owner_words[wbase + 3] = cutlass.Int32(word3)
                        for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                            row = own_row0 + cutlass.Int32(rr)
                            if row < m_rows:
                                sum_row_offset = sum_region_base + cutlass.Int64(
                                    row
                                ) * cutlass.Int64(N)
                                for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                    col = (
                                        tidx + cutlass.Int32(v * THREADS_PER_BLOCK)
                                    ) * cutlass.Int32(_BF16_VEC_SIZE)
                                    wbase = (rr * VECS_PER_THREAD + v) * 4
                                    w0 = owner_words[wbase]
                                    w1 = owner_words[wbase + 1]
                                    w2 = owner_words[wbase + 2]
                                    w3 = owner_words[wbase + 3]
                                    _multimem_st_v4_b32(
                                        multicast_base
                                        + (sum_row_offset + cutlass.Int64(col)) * cutlass.Int64(2),
                                        w0,
                                        w1,
                                        w2,
                                        w3,
                                    )
                                    words = [w0, w1, w2, w3]
                                    lane_base = v * _BF16_VEC_SIZE
                                    for pair in cutlass.range_constexpr(_BF16_VEC_SIZE // 2):
                                        word = words[pair]
                                        low_bits = (word << cutlass.Int32(16)) >> cutlass.Int32(16)
                                        high_bits = word >> cutlass.Int32(16)
                                        # Constructor narrowing (as cutlass.Int8(sf_code) in the out-quant kernel);
                                        # both halves were sign-extended by the arithmetic shifts.
                                        sum_values[lane_base + 2 * pair] = cutlass.Int16(
                                            low_bits
                                        ).bitcast(cutlass.BFloat16)
                                        sum_values[lane_base + 2 * pair + 1] = cutlass.Int16(
                                            high_bits
                                        ).bitcast(cutlass.BFloat16)
                                _twoshot_row_epilogue(
                                    row,
                                    sum_values,
                                    residual_array,
                                    norm_weight_array,
                                    norm_output_array,
                                    residual_output_array,
                                    warp_sums,
                                    tidx,
                                    lane_id,
                                    warp_id,
                                    global_scale,
                                    fp4_ptr,
                                    sf_ptr,
                                    eps,
                                    N,
                                    THREADS_PER_BLOCK,
                                    VECS_PER_THREAD,
                                    EMIT_NVFP4,
                                )
                    prims.barrier_cta_sync(0)
                    if tidx == 0:
                        for target_rank in cutlass.range_constexpr(WORLD_SIZE):
                            if cutlass.const_expr(target_rank != LOCAL_RANK):
                                prims.red(
                                    prims.ReductionOp.ADD,
                                    prims.ReductionType.S32,
                                    (
                                        ready_flags[target_rank].iterator
                                        + flag_sum_base
                                        + cutlass.Int64(own_block)
                                    ).llvm_ptr,
                                    cutlass.Int32(1),
                                    mem_order=prims.MemOrder.RELEASE,
                                    mem_scope=prims.MemScope.SYS,
                                )

                group = next_group
        else:
            group = bidx - num_producers
            while group < num_groups:
                # ---- the three peer blocks of the group: wait for each owner's multicast sum
                # (rotated start so the four ranks do not all wait on the same owner first).
                for peer_ordinal in cutlass.range_constexpr(1, WORLD_SIZE):
                    peer_rank = (LOCAL_RANK + peer_ordinal) % WORLD_SIZE
                    peer_block = group * cutlass.Int32(WORLD_SIZE) + cutlass.Int32(peer_rank)
                    peer_row0 = peer_block * row_block_rows
                    if peer_row0 < m_rows:
                        peer_sum_flag_ptr = (
                            ready_flags[LOCAL_RANK].iterator
                            + flag_sum_base
                            + cutlass.Int64(peer_block)
                        ).llvm_ptr
                        if tidx == 0:
                            ready = cute.arch.load(
                                peer_sum_flag_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )
                            while ready < cutlass.Int32(1):
                                ready = cute.arch.load(
                                    peer_sum_flag_ptr,
                                    cutlass.Int32,
                                    sem="acquire",
                                    scope="sys",
                                )
                            prims.red(
                                prims.ReductionOp.ADD,
                                prims.ReductionType.S32,
                                peer_sum_flag_ptr,
                                cutlass.Int32(-1),
                                mem_order=prims.MemOrder.RELAXED,
                                mem_scope=prims.MemScope.SYS,
                            )
                        prims.barrier_cta_sync(0)
                        # Batched: issue every row's loads of the block before the first epilogue so the
                        # CTA is not one latency-bound row at a time.
                        for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                            row = peer_row0 + cutlass.Int32(rr)
                            if row < m_rows:
                                sum_row_offset = sum_region_base + cutlass.Int64(
                                    row
                                ) * cutlass.Int64(N)
                                for v in cutlass.range_constexpr(VECS_PER_THREAD):
                                    col = (
                                        tidx + cutlass.Int32(v * THREADS_PER_BLOCK)
                                    ) * cutlass.Int32(_BF16_VEC_SIZE)
                                    pbase = (rr * VECS_PER_THREAD + v) * _BF16_VEC_SIZE
                                    if cutlass.const_expr(_TWOSHOT_DIRECT):
                                        stage_values[
                                            pbase:_BF16_VEC_SIZE,
                                            cutlass.align(_ALIGNMENT_BYTES),
                                        ] = buffer_arrays[peer_rank].load(
                                            sum_row_offset + cutlass.Int64(col),
                                            _BF16_VEC_SIZE,
                                            alignment=_ALIGNMENT_BYTES,
                                            is_volatile=True,
                                        )
                                    else:
                                        stage_values[
                                            pbase:_BF16_VEC_SIZE,
                                            cutlass.align(_ALIGNMENT_BYTES),
                                        ] = local_buffer.load(
                                            sum_row_offset + cutlass.Int64(col),
                                            _BF16_VEC_SIZE,
                                            alignment=_ALIGNMENT_BYTES,
                                            is_volatile=True,
                                        )
                        for rr in cutlass.range_constexpr(ROWS_PER_CTA):
                            row = peer_row0 + cutlass.Int32(rr)
                            if row < m_rows:
                                for e in cutlass.range_constexpr(VECS_PER_THREAD * _BF16_VEC_SIZE):
                                    sum_values[e] = stage_values[
                                        rr * VECS_PER_THREAD * _BF16_VEC_SIZE + e
                                    ]
                                _twoshot_row_epilogue(
                                    row,
                                    sum_values,
                                    residual_array,
                                    norm_weight_array,
                                    norm_output_array,
                                    residual_output_array,
                                    warp_sums,
                                    tidx,
                                    lane_id,
                                    warp_id,
                                    global_scale,
                                    fp4_ptr,
                                    sf_ptr,
                                    eps,
                                    N,
                                    THREADS_PER_BLOCK,
                                    VECS_PER_THREAD,
                                    EMIT_NVFP4,
                                )

                group = group + cutlass.Int32(_TWOSHOT_CONSUMER_CTAS)

    # Output-safe release: every output row of this CTA is written.
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(not PDL_RELEASE_AT_ENTRY):
            if tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _glm_tp_allreduce_residual_rms_norm_bf16_twoshot_nvls(
    input_tensor: cute.Tensor,
    residual_tensor: cute.Tensor,
    norm_weight_tensor: cute.Tensor,
    norm_output_tensor: cute.Tensor,
    residual_output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    ready_flags: list[cute.Tensor],
    signal_state: cute.Tensor,
    multicast_base: cutlass.Int64,
    signal: cutlass.Int64,
    stream: cuda.CUstream,
    eps: cutlass.Constexpr[cutlass.Float32],
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    THREADS_PER_BLOCK: cutlass.Constexpr[int],
    ROWS_PER_CTA: cutlass.Constexpr[int],
    USE_DEVICE_SIGNAL_STATE: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
    PDL_RELEASE_AT_ENTRY: cutlass.Constexpr[bool],
) -> None:
    M = input_tensor.shape[0]
    grid_x = (M + ROWS_PER_CTA * WORLD_SIZE - 1) // (ROWS_PER_CTA * WORLD_SIZE)
    if cutlass.const_expr(_TWOSHOT_MAX_CTAS > 0):
        grid_x = cutlass.min(grid_x, cutlass.Int32(_TWOSHOT_MAX_CTAS))
    if cutlass.const_expr(_TWOSHOT_CONSUMER_CTAS > 0):
        grid_x = grid_x + cutlass.Int32(_TWOSHOT_CONSUMER_CTAS)
    # The fused RMSNorm kernel has no NVFP4 outputs; the placeholder tensors are never touched
    # because EMIT_NVFP4 is False (all uses sit under const_expr guards).
    _glm_tp_allreduce_residual_rms_norm_twoshot_nvls_kernel(
        input_tensor,
        residual_tensor,
        norm_weight_tensor,
        norm_weight_tensor,
        norm_output_tensor,
        norm_output_tensor,
        norm_output_tensor,
        residual_output_tensor,
        buffers,
        ready_flags,
        signal_state,
        multicast_base,
        signal,
        eps,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        THREADS_PER_BLOCK,
        ROWS_PER_CTA,
        USE_DEVICE_SIGNAL_STATE,
        USE_PDL,
        PDL_RELEASE_AT_ENTRY,
        False,
    ).launch(
        grid=(grid_x, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


def compile_pure_for_rank(
    local_rank: int = 0,
    world_size: int = 1,
    device_id: int = 0,
    n: int = _DEFAULT_N,
    pure_threads_per_block: int = _THREADS_PER_BLOCK,
    pure_elide_self_rank: bool = False,
    pure_rotate_peer_order: bool = False,
    pure_fp32_accumulation: bool = True,
    pure_batch_poll_peers: bool = False,
    use_ready_flags: bool = _USES_READY_FLAGS,
    use_device_signal_state: bool = _USES_DEVICE_SIGNAL_STATE,
    use_pdl: bool = False,
    pdl_release_policy: str = "strict",
    last_cta_signal_advance: bool = False,
):
    cutlass.cuda.initialize_cuda_context(device_id)
    _validate_signal_advance_flags(
        last_cta_signal_advance,
        False,
        use_device_signal_state,
        use_pdl,
        pdl_release_policy,
    )
    if pure_threads_per_block not in _PURE_THREADS_PER_BLOCK_CHOICES:
        raise ValueError(
            "pure_threads_per_block must be one of "
            f"{_PURE_THREADS_PER_BLOCK_CHOICES}, got {pure_threads_per_block}"
        )
    _validate_pdl_release_policy(pdl_release_policy)
    if pure_batch_poll_peers and (world_size != 4 or use_ready_flags):
        raise ValueError(
            "allreduce batch peer polling requires world_size=4 and sentinel readiness"
        )
    if n % _BF16_VEC_SIZE != 0:
        raise ValueError(f"n must be divisible by {_BF16_VEC_SIZE}, got {n}")

    sym_m = cute.sym_int32()
    input_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    output_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )

    workspace_numel = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    workspace_num_vectors = workspace_numel // _BF16_VEC_SIZE
    fake_buffers = []
    fake_ready_flags = []
    for _ in range(world_size):
        buffer_tensor = make_fake_compact_tensor(
            cutlass.BFloat16,
            (_PING_PONG_SIZE, world_size, workspace_numel),
            stride_order=(2, 1, 0),
            assumed_align=_ALIGNMENT_BYTES,
        )
        fake_buffers.append(buffer_tensor)
        ready_flag_tensor = make_fake_compact_tensor(
            cutlass.Int32,
            (_PING_PONG_SIZE, world_size, workspace_num_vectors),
            stride_order=(2, 1, 0),
            assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
        )
        fake_ready_flags.append(ready_flag_tensor)

    fake_signal_state = make_fake_compact_tensor(
        cutlass.Int32,
        (_SIGNAL_STATE_SIZE,),
        assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
    )
    peer_order = tuple(range(world_size))
    if pure_rotate_peer_order:
        peer_order = tuple(
            (local_rank + peer_ordinal) % world_size for peer_ordinal in range(world_size)
        )

    return cute.compile(
        _glm_tp_allreduce_bf16_oneshot_lamport_serving,
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
        pure_threads_per_block,
        use_ready_flags,
        use_device_signal_state,
        pure_elide_self_rank,
        peer_order,
        pure_fp32_accumulation,
        pure_batch_poll_peers,
        use_pdl,
        pdl_release_policy == "strict",
        last_cta_signal_advance,
        options="--enable-tvm-ffi",
    )


def compile_fused_for_rank(
    local_rank: int = 0,
    world_size: int = 1,
    device_id: int = 0,
    n: int = _DEFAULT_N,
    eps: float = _RMS_EPS,
    split_rows: bool = True,
    use_ready_flags: bool = _USES_READY_FLAGS,
    use_device_signal_state: bool = _USES_DEVICE_SIGNAL_STATE,
    elide_self_rank: bool = False,
    rotate_peer_order: bool = False,
    aggregate_signal_at_cluster_root: bool = False,
    ctas_per_row: int = _FUSED_CTAS_PER_ROW,
    fused_threads_per_block: int = _SPLIT_ROW_THREADS_PER_BLOCK,
    prefetch_fused_inputs: bool = False,
    fused_direct_cluster_reduction: bool = False,
    fused_post_norm_cluster_barrier: bool = True,
    fused_fp32_accumulation: bool = True,
    fused_batch_poll_peers: bool = False,
    fused_ready_mask_poll_peers: bool = False,
    use_pdl: bool = False,
    pdl_release_policy: str = "strict",
    last_cta_signal_advance: bool = False,
    all_to_all_cluster_reduction: bool = False,
    early_input_load: bool = False,
    multicast_publish: bool = False,
    iket_phases: bool = False,
    rotated_poll_order: bool = False,
):
    cutlass.cuda.initialize_cuda_context(device_id)
    _validate_ctas_per_row(ctas_per_row)
    _validate_signal_advance_flags(
        last_cta_signal_advance,
        aggregate_signal_at_cluster_root,
        use_device_signal_state,
        use_pdl,
        pdl_release_policy,
    )
    if fused_threads_per_block not in _FUSED_THREADS_PER_BLOCK_CHOICES:
        raise ValueError(
            "fused_threads_per_block must be one of "
            f"{_FUSED_THREADS_PER_BLOCK_CHOICES}, got {fused_threads_per_block}"
        )
    _validate_fused_pdl_release_policy(pdl_release_policy)
    if fused_batch_poll_peers and fused_ready_mask_poll_peers:
        raise ValueError("fused RMSNorm batch and ready-mask peer polling are mutually exclusive")
    if rotated_poll_order and (fused_batch_poll_peers or fused_ready_mask_poll_peers):
        raise ValueError(
            "rotated_poll_order applies to the sequential peer poll only; the batch "
            "and ready-mask variants already poll every peer each iteration"
        )
    if (fused_batch_poll_peers or fused_ready_mask_poll_peers) and (
        world_size != 4 or use_ready_flags
    ):
        raise ValueError(
            "fused RMSNorm batch and ready-mask peer polling require "
            "world_size=4 and sentinel readiness"
        )
    if n % _BF16_VEC_SIZE != 0:
        raise ValueError(f"n must be divisible by {_BF16_VEC_SIZE}, got {n}")
    split_row_alignment = ctas_per_row * _BF16_VEC_SIZE
    if split_rows and n % split_row_alignment != 0:
        raise ValueError(
            f"n must be divisible by {split_row_alignment} for split-row fused launch, got {n}"
        )
    if not use_ready_flags and not split_rows:
        raise ValueError("sentinel readiness is implemented only for split-row fused")
    if use_device_signal_state and not split_rows:
        raise ValueError("device signal state is implemented only for split-row fused")
    if use_pdl and not split_rows:
        raise ValueError("PDL is implemented only for split-row fused")
    _validate_multicast_publish(multicast_publish, use_ready_flags, split_rows)

    sym_m = cute.sym_int32()
    input_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    residual_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    norm_weight_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (n,),
        assumed_align=_ALIGNMENT_BYTES,
    )
    norm_output_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    residual_output_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )

    workspace_numel = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    workspace_num_vectors = workspace_numel // _BF16_VEC_SIZE
    fake_buffers = []
    fake_ready_flags = []
    for _ in range(world_size):
        buffer_tensor = make_fake_compact_tensor(
            cutlass.BFloat16,
            (_PING_PONG_SIZE, world_size, workspace_numel),
            stride_order=(2, 1, 0),
            assumed_align=_ALIGNMENT_BYTES,
        )
        fake_buffers.append(buffer_tensor)
        ready_flag_tensor = make_fake_compact_tensor(
            cutlass.Int32,
            (_PING_PONG_SIZE, world_size, workspace_num_vectors),
            stride_order=(2, 1, 0),
            assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
        )
        fake_ready_flags.append(ready_flag_tensor)

    fake_signal_state = make_fake_compact_tensor(
        cutlass.Int32,
        (_SIGNAL_STATE_SIZE,),
        assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
    )

    fused_host = (
        _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row
        if split_rows
        else _glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport
    )
    compile_args = [
        input_tensor,
        residual_tensor,
        norm_weight_tensor,
        norm_output_tensor,
        residual_output_tensor,
        fake_buffers,
        fake_ready_flags,
    ]
    if split_rows:
        compile_args.append(fake_signal_state)
        compile_args.append(cutlass.Int64(0))
    compile_args.extend(
        [
            cutlass.Int64(0),
            make_fake_stream(),
            cutlass.Float32(eps),
            n,
            local_rank,
            world_size,
        ]
    )
    if split_rows:
        peer_order = tuple(range(world_size))
        if rotate_peer_order:
            peer_order = tuple(
                (local_rank + peer_ordinal) % world_size for peer_ordinal in range(world_size)
            )
        compile_args.append(ctas_per_row)
        compile_args.append(fused_threads_per_block)
        compile_args.append(use_ready_flags)
        compile_args.append(use_device_signal_state)
        compile_args.append(elide_self_rank)
        compile_args.append(peer_order)
        compile_args.append(prefetch_fused_inputs)
        compile_args.append(fused_direct_cluster_reduction)
        compile_args.append(fused_post_norm_cluster_barrier)
        compile_args.append(fused_fp32_accumulation)
        compile_args.append(fused_batch_poll_peers)
        compile_args.append(fused_ready_mask_poll_peers)
        compile_args.append(aggregate_signal_at_cluster_root)
        compile_args.append(use_pdl)
        compile_args.append(pdl_release_policy == "strict")
        compile_args.append(pdl_release_policy == "pre_norm")
        compile_args.append(pdl_release_policy == "entry")
        compile_args.append(last_cta_signal_advance)
        compile_args.append(all_to_all_cluster_reduction)
        compile_args.append(early_input_load)
        compile_args.append(multicast_publish)
        compile_args.append(iket_phases)
        compile_args.append(rotated_poll_order)

    return cute.compile(
        fused_host,
        *compile_args,
        options="--enable-tvm-ffi",
    )


def compile_fused_out_quant_for_rank(
    local_rank: int = 0,
    world_size: int = 1,
    device_id: int = 0,
    n: int = _DEFAULT_N,
    eps: float = _RMS_EPS,
    split_rows: bool = True,
    use_ready_flags: bool = _USES_READY_FLAGS,
    use_device_signal_state: bool = _USES_DEVICE_SIGNAL_STATE,
    rotate_peer_order: bool = False,
    elide_self_rank: bool = False,
    aggregate_signal_at_cluster_root: bool = False,
    ctas_per_row: int = _FUSED_CTAS_PER_ROW,
    use_pdl: bool | None = None,
    pdl_release_policy: str = "strict",
    last_cta_signal_advance: bool = False,
    all_to_all_cluster_reduction: bool = False,
    threads_per_block: int = _SPLIT_ROW_THREADS_PER_BLOCK,
    early_input_load: bool = False,
    multicast_publish: bool = False,
    iket_phases: bool = False,
    prefetch_fused_inputs: bool = False,
    rotated_poll_order: bool = False,
):
    cutlass.cuda.initialize_cuda_context(device_id)
    if n % _SCALE_BLOCK != 0:
        raise ValueError(f"n must be divisible by {_SCALE_BLOCK}, got {n}")
    _validate_ctas_per_row(ctas_per_row)
    if threads_per_block not in _OUT_QUANT_THREADS_PER_BLOCK_CHOICES:
        raise ValueError(
            "threads_per_block must be one of "
            f"{_OUT_QUANT_THREADS_PER_BLOCK_CHOICES}, got {threads_per_block}"
        )
    _validate_pdl_release_policy(pdl_release_policy)
    _validate_multicast_publish(multicast_publish, use_ready_flags, split_rows)
    if use_pdl is None:
        use_pdl = True
    _validate_signal_advance_flags(
        last_cta_signal_advance,
        aggregate_signal_at_cluster_root,
        use_device_signal_state,
        use_pdl,
        pdl_release_policy,
    )
    split_row_alignment = ctas_per_row * _SCALE_BLOCK
    if not split_rows:
        raise ValueError(
            "NVFP4 out-quant fused allreduce is implemented only for the split-row serving launch."
        )
    if n % split_row_alignment != 0:
        raise ValueError(
            f"n must be divisible by {split_row_alignment} for split-row "
            f"NVFP4 out-quant launch, got {n}"
        )

    sym_m = cute.sym_int32()
    input_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    residual_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    norm_weight_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (n,),
        assumed_align=_ALIGNMENT_BYTES,
    )
    global_scale_tensor = make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    norm_output_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    fp4_output_tensor = make_fake_compact_tensor(
        cutlass.Int8,
        (sym_m, n // 2),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    scale_output_tensor = make_fake_compact_tensor(
        cutlass.Int8,
        (sym_m, n // _SCALE_BLOCK),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    residual_output_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )

    workspace_numel = cute.sym_int32(divisibility=_BF16_VEC_SIZE)
    workspace_num_vectors = workspace_numel // _BF16_VEC_SIZE
    fake_buffers = []
    fake_ready_flags = []
    for _ in range(world_size):
        buffer_tensor = make_fake_compact_tensor(
            cutlass.BFloat16,
            (_PING_PONG_SIZE, world_size, workspace_numel),
            stride_order=(2, 1, 0),
            assumed_align=_ALIGNMENT_BYTES,
        )
        fake_buffers.append(buffer_tensor)
        ready_flag_tensor = make_fake_compact_tensor(
            cutlass.Int32,
            (_PING_PONG_SIZE, world_size, workspace_num_vectors),
            stride_order=(2, 1, 0),
            assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
        )
        fake_ready_flags.append(ready_flag_tensor)

    fake_signal_state = make_fake_compact_tensor(
        cutlass.Int32,
        (_SIGNAL_STATE_SIZE,),
        assumed_align=_READY_FLAG_ALIGNMENT_BYTES,
    )

    peer_order = tuple(range(world_size))
    if rotate_peer_order:
        peer_order = tuple(
            (local_rank + peer_ordinal) % world_size for peer_ordinal in range(world_size)
        )

    return cute.compile(
        _glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row,
        input_tensor,
        residual_tensor,
        norm_weight_tensor,
        global_scale_tensor,
        norm_output_tensor,
        fp4_output_tensor,
        scale_output_tensor,
        residual_output_tensor,
        fake_buffers,
        fake_ready_flags,
        fake_signal_state,
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_stream(),
        cutlass.Float32(eps),
        n,
        local_rank,
        world_size,
        ctas_per_row,
        threads_per_block,
        use_ready_flags,
        use_device_signal_state,
        peer_order,
        elide_self_rank,
        aggregate_signal_at_cluster_root,
        use_pdl,
        pdl_release_policy == "strict",
        pdl_release_policy == "pre_norm",
        pdl_release_policy == "entry",
        last_cta_signal_advance,
        all_to_all_cluster_reduction,
        early_input_load,
        multicast_publish,
        iket_phases,
        prefetch_fused_inputs,
        rotated_poll_order,
        options="--enable-tvm-ffi",
    )


def _compile_twoshot_common(
    jit_entry,
    *,
    local_rank: int,
    world_size: int,
    device_id: int,
    n: int,
    eps: float,
    use_device_signal_state: bool,
    use_pdl: bool | None,
    pdl_release_policy: str,
    threads_per_block: int,
    rows_per_cta: int,
    emit_nvfp4: bool,
):
    cutlass.cuda.initialize_cuda_context(device_id)
    if threads_per_block not in _TWOSHOT_THREADS_PER_BLOCK_CHOICES:
        raise ValueError(
            "threads_per_block must be one of "
            f"{_TWOSHOT_THREADS_PER_BLOCK_CHOICES}, got {threads_per_block}"
        )
    if n % (_BF16_VEC_SIZE * threads_per_block) != 0:
        raise ValueError(
            "two-shot kernel needs n divisible by 8 x threads_per_block "
            f"(each thread owns whole 16 B vectors of a row), got n={n}, "
            f"threads_per_block={threads_per_block}"
        )
    if emit_nvfp4 and n % _SCALE_BLOCK != 0:
        raise ValueError(f"n must be divisible by {_SCALE_BLOCK}, got {n}")
    if rows_per_cta < 1:
        raise ValueError(f"rows_per_cta must be positive, got {rows_per_cta}")
    if pdl_release_policy not in ("entry", "output_safe"):
        raise ValueError(
            "two-shot pdl_release_policy must be 'entry' or 'output_safe', "
            f"got {pdl_release_policy!r}"
        )
    if use_pdl is None:
        use_pdl = True
    if not use_device_signal_state:
        raise ValueError("two-shot kernel requires device signal state")
    # The CTA-role options are module constants.
    # A negative consumer count would
    # make the producer stride grid_x - C exceed the producer count and skip
    # row-block groups, so reject it here for every caller.
    if _TWOSHOT_MAX_CTAS < 0 or _TWOSHOT_CONSUMER_CTAS < 0:
        raise ValueError(
            "two-shot CTA counts must be non-negative, got max_ctas="
            f"{_TWOSHOT_MAX_CTAS}, consumer_ctas={_TWOSHOT_CONSUMER_CTAS}"
        )
    if _TWOSHOT_DIRECT not in (0, 1):
        raise ValueError(f"_TWOSHOT_DIRECT must be 0 or 1, got {_TWOSHOT_DIRECT}")

    sym_m = cute.sym_int32()
    input_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    residual_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    norm_weight_tensor = make_fake_compact_tensor(
        cutlass.BFloat16, (n,), assumed_align=_ALIGNMENT_BYTES
    )
    norm_output_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
        stride_order=(1, 0),
        assumed_align=_ALIGNMENT_BYTES,
    )
    residual_output_tensor = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, n),
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
        cutlass.Int32, (_SIGNAL_STATE_SIZE,), assumed_align=_READY_FLAG_ALIGNMENT_BYTES
    )
    tail = (
        fake_buffers,
        fake_ready_flags,
        fake_signal_state,
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_stream(),
        cutlass.Float32(eps),
        n,
        local_rank,
        world_size,
        threads_per_block,
        rows_per_cta,
        use_device_signal_state,
        use_pdl,
        pdl_release_policy == "entry",
    )
    if emit_nvfp4:
        global_scale_tensor = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        fp4_output_tensor = make_fake_compact_tensor(
            cutlass.Int8,
            (sym_m, n // 2),
            stride_order=(1, 0),
            assumed_align=_ALIGNMENT_BYTES,
        )
        scale_output_tensor = make_fake_compact_tensor(
            cutlass.Int8,
            (sym_m, n // _SCALE_BLOCK),
            stride_order=(1, 0),
            assumed_align=_ALIGNMENT_BYTES,
        )
        return cute.compile(
            jit_entry,
            input_tensor,
            residual_tensor,
            norm_weight_tensor,
            global_scale_tensor,
            norm_output_tensor,
            fp4_output_tensor,
            scale_output_tensor,
            residual_output_tensor,
            *tail,
            options="--enable-tvm-ffi",
        )
    return cute.compile(
        jit_entry,
        input_tensor,
        residual_tensor,
        norm_weight_tensor,
        norm_output_tensor,
        residual_output_tensor,
        *tail,
        options="--enable-tvm-ffi",
    )


def compile_fused_twoshot_for_rank(
    local_rank: int = 0,
    world_size: int = 1,
    device_id: int = 0,
    n: int = _DEFAULT_N,
    eps: float = _RMS_EPS,
    use_device_signal_state: bool = _USES_DEVICE_SIGNAL_STATE,
    use_pdl: bool | None = None,
    pdl_release_policy: str = "output_safe",
    threads_per_block: int = _TWOSHOT_THREADS_PER_BLOCK,
    rows_per_cta: int = _TWOSHOT_ROWS_PER_CTA,
    **_ignored_oneshot_options,
):
    """Fused residual-add + RMSNorm (RESIDUAL_RMS_NORM) two-shot NVLS kernel; same
    launch signature as ``compile_fused_for_rank`` so adapters can hold both and pick by M.
    One-shot-only options passed by shared adapter code are accepted and ignored."""
    return _compile_twoshot_common(
        _glm_tp_allreduce_residual_rms_norm_bf16_twoshot_nvls,
        local_rank=local_rank,
        world_size=world_size,
        device_id=device_id,
        n=n,
        eps=eps,
        use_device_signal_state=use_device_signal_state,
        use_pdl=use_pdl,
        pdl_release_policy=pdl_release_policy,
        threads_per_block=threads_per_block,
        rows_per_cta=rows_per_cta,
        emit_nvfp4=False,
    )
