# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE round trip's two collectives as one Lamport launch each.

Under attention DP every rank holds its own rows. A MoE block gathers the group's rows
(`[R, H]` a rank -> `[W * R, H]`, rank-major), runs this rank's expert window over all of
them and reduce-scatters the window partials back (`[W * R, H]` a rank -> the sum of every
rank's rows `[r * R, (r + 1) * R)` on rank `r`). NCCL serves both with its ring kernels,
`W - 1` hops each. Here each is one launch over symmetric memory on one NVLink domain:

* `gather`: a thread publishes one 8-lane bf16 vector of this rank's rows into its region
  of every peer's slot, stores it into this rank's block of the output, then polls its own
  slot for the same vector of every peer and stores each at its rank's block.
* `reduce_scatter`: a thread publishes the vector of every peer's slice into that peer's
  slot, polls its own slot for the three peers' partials of its own slice, and sums.

The transport is GLM-5.2 TEP4's (`tp4_allreduce_fused_rms` / `tp4_allgather_lamport`):
three rotating slots of `[W, max_rows * H]` bf16 per rank, bitwise bf16 -0.0 as the in-band
not-ready marker, the slot counter in device memory read and advanced by the kernel, so a
CUDA graph captures every launch and still rotates slots on replay.

What differs from the NCCL pair, bit for bit:

* A -0.0 lane travels as +0.0 (it is the marker). The gathered rows therefore hold +0.0
  where a rank's row held -0.0, on every rank alike, and a reduce-scatter result is +0.0
  where every partial was -0.0.
* The reduce-scatter sums the four partials in NCCL's ring order for the owning rank
  (`SUM_ORDER`: the chain starts at the owner's successor and ends at the owner), rounding
  to bf16 after every add as NCCL's bf16 sum does.

Collective: every rank of the group launches the same kernel at the same point with the
same row count.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.comm import (
    tp4_allreduce_fused_rms as _ar,
)

_PING_PONG = _ar._PING_PONG_SIZE
_VEC = _ar._BF16_VEC_SIZE
_ALIGN = _ar._ALIGNMENT_BYTES
_NEG_ZERO = _ar._NEGATIVE_ZERO_BF16_BITS
_THREADS = 256
_raw_stream = torch._C._cuda_getCurrentRawStream


@cute.jit
def _sanitize(values):
    """`values` with every bitwise -0.0 lane as +0.0."""
    neg_zero_bits = cutlass.Int16(_NEG_ZERO)
    clean = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
    for lane in cutlass.range_constexpr(_VEC):
        lane_value = values[lane]
        if lane_value.bitcast(cutlass.Int16) == neg_zero_bits:
            lane_value = cutlass.BFloat16(0.0)
        clean[lane] = lane_value
    return clean


@cute.jit
def _poll(local_buffer, offset):
    """The vector at `offset` of this rank's slot, once no lane holds the marker."""
    values = local_buffer.load(offset, _VEC, alignment=_ALIGN, is_volatile=True)
    while not _ar._all_bf16x8_lanes_ready(values):
        values = local_buffer.load(offset, _VEC, alignment=_ALIGN, is_volatile=True)
    copy = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
    for lane in cutlass.range_constexpr(_VEC):
        copy[lane] = values[lane]
    return copy


@cute.kernel
def _round_trip_kernel(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    signal_state: cute.Tensor,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    SUM_ORDER: cutlass.Constexpr[tuple[int, ...]],
    REDUCE: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    bdim, _, _ = cute.arch.block_dim()

    input_array = cutlass.make_array_view(input_tensor)
    output_array = cutlass.make_array_view(output_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]

    # One rank's rows: the gather's input, the reduce-scatter's output.
    if cutlass.const_expr(REDUCE):
        rows = cute.size(output_tensor, mode=[0])
    else:
        rows = cute.size(input_tensor, mode=[0])
    active_numel = rows * N
    active_numel_i64 = cutlass.Int64(active_numel)
    workspace_numel_i64 = cutlass.Int64(cute.size(buffers[0], mode=[2]))
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_ar._SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_ar._SIGNAL_VALUE_INDEX)).llvm_ptr

    signal_i32 = cute.arch.load(state_signal_ptr, cutlass.Int32, sem="acquire", scope="gpu")
    signal_i64 = cutlass.Int64(signal_i32)
    ping_slot = signal_i64 % _PING_PONG
    clear_slot = (signal_i64 + cutlass.Int64(1)) % _PING_PONG
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64
    clear_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64

    neg_zero_vec = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
    neg_zero_bits = cutlass.Int16(_NEG_ZERO)
    for lane in cutlass.range_constexpr(_VEC):
        neg_zero_vec[lane] = neg_zero_bits.bitcast(cutlass.BFloat16)

    # Re-arm the slot the launch after next uses, for as many vectors as its last user's
    # peers wrote into it.
    clear_count_ptr = (
        signal_state.iterator
        + cutlass.Int64(_ar._SIGNAL_SLOT_COUNTS_INDEX)
        + cutlass.Int64(clear_slot)
    ).llvm_ptr
    clear_vectors = cute.arch.load(clear_count_ptr, cutlass.Int32, sem="acquire", scope="gpu")
    clear_vector_idx = bidx * bdim + tidx
    clear_stride = grid_x * bdim
    while clear_vector_idx < clear_vectors:
        clear_vector_idx_i64 = cutlass.Int64(clear_vector_idx)
        for source_rank in cutlass.range_constexpr(WORLD_SIZE):
            if source_rank != LOCAL_RANK:
                local_buffer.store(
                    neg_zero_vec[0:_VEC, cutlass.align(_ALIGN)],
                    clear_base
                    + cutlass.Int64(source_rank) * workspace_numel_i64
                    + clear_vector_idx_i64 * cutlass.Int64(_VEC),
                    _VEC,
                    alignment=_ALIGN,
                    is_volatile=True,
                )
        clear_vector_idx = clear_vector_idx + clear_stride

    base_idx = (bidx * bdim + tidx) * _VEC
    if base_idx < active_numel:
        base_idx_i64 = cutlass.Int64(base_idx)
        write_offset = ping_base + cutlass.Int64(LOCAL_RANK) * workspace_numel_i64 + base_idx_i64
        if cutlass.const_expr(REDUCE):
            # Publish every peer's slice of this rank's partials into that peer's slot.
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                target_rank = PEER_ORDER[peer_ordinal]
                if target_rank != LOCAL_RANK:
                    partial = input_array.load(
                        cutlass.Int64(target_rank) * active_numel_i64 + base_idx_i64,
                        _VEC,
                        alignment=_ALIGN,
                    )
                    clean = _sanitize(partial)
                    buffer_arrays[target_rank].store(
                        clean[0:_VEC, cutlass.align(_ALIGN)],
                        write_offset,
                        _VEC,
                        alignment=_ALIGN,
                        is_volatile=True,
                    )
            parts = cutlass.Array(cutlass.BFloat16, WORLD_SIZE * _VEC, alignment=_ALIGN)
            own = input_array.load(
                cutlass.Int64(LOCAL_RANK) * active_numel_i64 + base_idx_i64,
                _VEC,
                alignment=_ALIGN,
            )
            for lane in cutlass.range_constexpr(_VEC):
                parts[LOCAL_RANK * _VEC + lane] = own[lane]
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                source_rank = PEER_ORDER[peer_ordinal]
                if source_rank != LOCAL_RANK:
                    got = _poll(
                        local_buffer,
                        ping_base + cutlass.Int64(source_rank) * workspace_numel_i64 + base_idx_i64,
                    )
                    for lane in cutlass.range_constexpr(_VEC):
                        parts[source_rank * _VEC + lane] = got[lane]
            # The ring's chain: bf16 after every add.
            total = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
            for lane in cutlass.range_constexpr(_VEC):
                acc = parts[SUM_ORDER[0] * _VEC + lane]
                for k in cutlass.range_constexpr(1, WORLD_SIZE):
                    acc = (
                        acc.to(cutlass.Float32)
                        + parts[SUM_ORDER[k] * _VEC + lane].to(cutlass.Float32)
                    ).to(cutlass.BFloat16)
                total[lane] = acc
            output_array.store(
                total[0:_VEC, cutlass.align(_ALIGN)], base_idx_i64, _VEC, alignment=_ALIGN
            )
        else:
            values = input_array.load(base_idx_i64, _VEC, alignment=_ALIGN)
            clean = _sanitize(values)
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                target_rank = PEER_ORDER[peer_ordinal]
                if target_rank != LOCAL_RANK:
                    buffer_arrays[target_rank].store(
                        clean[0:_VEC, cutlass.align(_ALIGN)],
                        write_offset,
                        _VEC,
                        alignment=_ALIGN,
                        is_volatile=True,
                    )
            output_array.store(
                clean[0:_VEC, cutlass.align(_ALIGN)],
                cutlass.Int64(LOCAL_RANK) * active_numel_i64 + base_idx_i64,
                _VEC,
                alignment=_ALIGN,
            )
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                source_rank = PEER_ORDER[peer_ordinal]
                if source_rank != LOCAL_RANK:
                    got = _poll(
                        local_buffer,
                        ping_base + cutlass.Int64(source_rank) * workspace_numel_i64 + base_idx_i64,
                    )
                    output_array.store(
                        got[0:_VEC, cutlass.align(_ALIGN)],
                        cutlass.Int64(source_rank) * active_numel_i64 + base_idx_i64,
                        _VEC,
                        alignment=_ALIGN,
                    )

    # The last CTA to arrive records how many vectors the peers wrote into this slot and
    # rotates the signal; every CTA has read it above.
    if tidx == 0:
        _ar._advance_signal_if_last_cta(
            state_counter_ptr,
            state_signal_ptr,
            signal_state,
            signal_i32,
            ping_slot,
            cutlass.Int32(active_numel) // cutlass.Int32(_VEC),
            grid_x,
        )


@cute.jit
def _round_trip(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    signal_state: cute.Tensor,
    stream: cuda.CUstream,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    SUM_ORDER: cutlass.Constexpr[tuple[int, ...]],
    REDUCE: cutlass.Constexpr[bool],
) -> None:
    if cutlass.const_expr(REDUCE):
        rows = output_tensor.shape[0]
    else:
        rows = input_tensor.shape[0]
    num_blocks = cute.ceil_div(rows * N, _THREADS * _VEC)
    _round_trip_kernel(
        input_tensor,
        output_tensor,
        buffers,
        signal_state,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        PEER_ORDER,
        SUM_ORDER,
        REDUCE,
    ).launch(grid=(num_blocks, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def _compile(rank: int, world: int, device_id: int, hidden: int, reduce: bool, sum_order):
    cutlass.cuda.initialize_cuda_context(device_id)
    rows_in, rows_out = cute.sym_int32(), cute.sym_int32()
    fake_in = make_fake_compact_tensor(
        cutlass.BFloat16, (rows_in, hidden), stride_order=(1, 0), assumed_align=_ALIGN
    )
    fake_out = make_fake_compact_tensor(
        cutlass.BFloat16, (rows_out, hidden), stride_order=(1, 0), assumed_align=_ALIGN
    )
    numel = cute.sym_int32(divisibility=_VEC)
    fake_buffers = [
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (_PING_PONG, world, numel),
            stride_order=(2, 1, 0),
            assumed_align=_ALIGN,
        )
        for _ in range(world)
    ]
    fake_state = make_fake_compact_tensor(
        cutlass.Int32, (_ar._SIGNAL_STATE_SIZE,), assumed_align=_ar._READY_FLAG_ALIGNMENT_BYTES
    )
    return cute.compile(
        _round_trip,
        fake_in,
        fake_out,
        fake_buffers,
        fake_state,
        make_fake_stream(),
        hidden,
        rank,
        world,
        tuple((rank + k) % world for k in range(world)),
        tuple(sum_order),
        reduce,
        options="--enable-tvm-ffi",
    )


def nccl_ring_order(rank: int, world: int) -> tuple:
    """The order NCCL's ring reduce-scatter adds the partials of `rank`'s slice in: the
    chain starts at the owner's successor on the ring and ends at the owner."""
    return tuple((rank + 1 + k) % world for k in range(world))


@dataclass(frozen=True)
class DepRoundTrip:
    """One rank's peer buffers, slot state and the two compiled launches."""

    hidden: int
    max_rows: int
    rank: int
    world: int
    #: This rank's `[3, world, max_rows * hidden]` bf16 symmetric buffer and every rank's,
    #: mapped here, in rank order.
    buffer: torch.Tensor
    peer_buffers: list
    #: `[6]` int32: entry counter, current slot, per-slot vector counts, (unused) epoch.
    state: torch.Tensor
    gather_fn: object
    reduce_fn: object

    def takes(self, rows: int, x: torch.Tensor) -> bool:
        """Every rank answers the same: the padded row count and the dtype are the group's."""
        return (
            1 <= rows <= self.max_rows and x.dtype is torch.bfloat16 and x.shape[1] == self.hidden
        )

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """`[R, H]` on every rank -> the group's rows `[world * R, H]`, rank-major."""
        x = x.contiguous()
        out = torch.empty((self.world * x.shape[0], self.hidden), dtype=x.dtype, device=x.device)
        self.gather_fn(x, out, self.peer_buffers, self.state, _raw_stream(x.device.index))
        return out

    def reduce_scatter(self, y: torch.Tensor) -> torch.Tensor:
        """`[world * R, H]` partials on every rank -> this rank's `[R, H]` summed slice."""
        y = y.contiguous()
        rows = y.shape[0] // self.world
        assert rows * self.world == y.shape[0] and rows <= self.max_rows, (y.shape, self.max_rows)
        out = torch.empty((rows, self.hidden), dtype=y.dtype, device=y.device)
        self.reduce_fn(y, out, self.peer_buffers, self.state, _raw_stream(y.device.index))
        return out


def build(hidden: int, max_rows: int, rank: int, world: int, sum_order=None) -> DepRoundTrip:
    """Pin the peer buffers over the process group that is already up and compile the two
    launches. Collective: every rank of the group calls it with the same arguments."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    assert hidden % _VEC == 0, hidden
    assert dist.is_initialized() and dist.get_world_size() == world and dist.get_rank() == rank
    dev = torch.device("cuda", torch.cuda.current_device())
    buffer = symm_mem.empty(
        (_PING_PONG, world, max_rows * hidden), dtype=torch.bfloat16, device=dev
    )
    # Every slot starts not-ready, or the first exchange reads uninitialized memory as data.
    buffer.view(torch.int16).fill_(_NEG_ZERO)
    group = dist.group.WORLD
    handle = symm_mem.rendezvous(buffer, group=getattr(group, "group_name", group))
    peer_buffers = [handle.get_buffer(p, buffer.shape, buffer.dtype) for p in range(world)]
    state = torch.zeros(_ar._SIGNAL_STATE_SIZE, dtype=torch.int32, device=dev)
    order = nccl_ring_order(rank, world) if sum_order is None else tuple(sum_order)
    assert sorted(order) == list(range(world)), order
    built = DepRoundTrip(
        hidden=hidden,
        max_rows=max_rows,
        rank=rank,
        world=world,
        buffer=buffer,
        peer_buffers=peer_buffers,
        state=state,
        gather_fn=_compile(rank, world, dev.index, hidden, False, order),
        reduce_fn=_compile(rank, world, dev.index, hidden, True, order),
    )
    # Every rank holds its buffers and both launches before any of them exchanges a row.
    ok = torch.ones(1, dtype=torch.int32, device=dev)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    return built
