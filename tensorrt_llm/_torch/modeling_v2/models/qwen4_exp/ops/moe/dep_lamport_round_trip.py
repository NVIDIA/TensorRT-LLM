# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE round trip's two collectives as one Lamport launch each.

Under attention DP every rank holds its own rows. A MoE block gathers the group's rows
(`[R, H]` a rank -> `[W * R, H]`, rank-major), runs this rank's expert window over all of
them and reduce-scatters the window partials back (`[W * R, H]` a rank -> the sum of every
rank's rows `[r * R, (r + 1) * R)` on rank `r`). NCCL serves both with its ring kernels,
`W - 1` hops each. Here each is one launch over peer-mapped memory on one NVLink domain (torch
symmetric memory inside a node, `build`; any buffers every rank maps, such as
`distributed.peer_buffers.PeerBuffers` across the nodes of a rack, `build_on`):

* `gather`: a thread publishes one 8-lane bf16 vector of this rank's rows into its region
  of every peer's slot, stores it into this rank's block of the output, then polls its own
  slot for the same vector of every peer and stores each at its rank's block.
* `reduce_scatter`: a thread publishes the vector of every peer's slice into that peer's
  slot, polls its own slot for the three peers' partials of its own slice, and sums.

The transport is the one of GLM-5.2 TEP4's Lamport all-reduce (`glm_5_2_nvfp4__sm_103__tep4`
`kernels/comm/tp4_allreduce_fused_rms`, whose slot-state layout and last-arrival advance are
repeated below rather than imported across families): three rotating slots of
`[W, max_rows * H]` bf16 per rank, bitwise bf16 -0.0 as the in-band not-ready marker, the slot
counter in device memory read and advanced by the kernel, so a CUDA graph captures every launch
and still rotates slots on replay.

What differs from the NCCL pair, bit for bit:

* A -0.0 lane travels as +0.0 (it is the marker). The gathered rows therefore hold +0.0
  where a rank's row held -0.0, on every rank alike, and a reduce-scatter result is +0.0
  where every partial was -0.0.
* The reduce-scatter sums the four partials in NCCL's ring order for the owning rank
  (`SUM_ORDER`: the chain starts at the owner's successor and ends at the owner), rounding
  to bf16 after every add as NCCL's bf16 sum does.

Two slot protocols, chosen at build (`step`):

* device counter (the GLM kernel's): every launch reads the slot from the state with an acquire
  load, every CTA waits for all of its threads to hold it, and the last CTA to arrive records the
  slot's vector count and rotates the slot (acq_rel atomic, release stores).
* host index (`step=True`): the slot is `(state[1] + launch_index) mod 3`, `launch_index` the
  launch's ordinal on its workspace inside a region the host brackets (`LamportSteps`), baked into
  a captured graph; a 1-thread launch at the end of the region moves `state[1]` by the region's
  count (`advance`). The reader of a peer's vector stores the marker back right after it took it,
  so the slot is clean when the launch completes; CTA 0 records a vector count of 0 for the re-arm
  pass, which stays in every launch. No acquire load at entry, no atomic, no release store at
  exit. Every receive loop issues its first probe of every peer before it waits on any. The same
  bytes travel in the same slots in the same order: the results are the device counter's bit for
  bit.

Collective: every rank of the group launches the same kernel at the same point with the
same row count. A CTA publishes and then polls for the same CTA of every peer, so every CTA of a
launch must be resident at once on every rank (`MAX_ROWS`); a kernel on another stream that holds
SMs while this one waits can stall it. The sum order is a fixed chain per owning rank: it equals
NCCL's when NCCL's ring visits the ranks in rank order (one node); across nodes NCCL may order
the ring or tree otherwise, so the sum is then a same-precision reorder of NCCL's.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Rotating slots, bf16 lanes a vector (16 bytes), and the not-ready marker (bf16 -0.0).
_PING_PONG = 3
_VEC = 8
_ALIGN = 16
_NEG_ZERO = -32768
#: `[6]` int32 slot state: the entry counter, the current slot, the per-slot vector counts, and an
#: unused epoch word (the GLM kernel's layout).
_SIGNAL_COUNTER_INDEX = 0
_SIGNAL_VALUE_INDEX = 1
_SIGNAL_SLOT_COUNTS_INDEX = 2
_SIGNAL_STATE_SIZE = _SIGNAL_SLOT_COUNTS_INDEX + _PING_PONG + 1
_STATE_ALIGN = 4
_THREADS = 256
#: Rows a rank up to which a round trip runs here. A CTA publishes and then waits for the same
#: CTA of every peer, so every CTA of a launch has to be resident at once on every rank: 512
#: rows of 2560 are 640 CTAs. Above it the caller keeps NCCL.
MAX_ROWS = 512
_raw_stream = torch._C._cuda_getCurrentRawStream


@cute.jit
def _all_lanes_ready(values):
    """Whether no lane of the 8-lane vector holds the not-ready marker."""
    ready = cutlass.Boolean(True)
    neg_zero_bits = cutlass.Int16(_NEG_ZERO)
    for lane in cutlass.range_constexpr(_VEC):
        ready = ready & (values[lane].bitcast(cutlass.Int16) != neg_zero_bits)
    return ready


@cute.jit
def _advance_signal_if_last_cta(
    state_counter_ptr, state_signal_ptr, signal_state, signal_i32, ping_slot, active_vectors, grid_x
):
    """Thread 0 of every CTA counts the CTA in (acq_rel); the last one to arrive -- every CTA has
    read the signal by then -- records how many vectors this launch's peers wrote into the slot,
    resets the counter and rotates the signal."""
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
        next_signal = (signal_i32 + cutlass.Int32(1)) % cutlass.Int32(_PING_PONG)
        cute.arch.store(ping_count_ptr, active_vectors, sem="release", scope="gpu")
        cute.arch.store(state_counter_ptr, cutlass.Int32(0), sem="release", scope="gpu")
        cute.arch.store(state_signal_ptr, next_signal, sem="release", scope="gpu")


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
    while not _all_lanes_ready(values):
        values = local_buffer.load(offset, _VEC, alignment=_ALIGN, is_volatile=True)
    copy = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
    for lane in cutlass.range_constexpr(_VEC):
        copy[lane] = values[lane]
    return copy


@cute.jit
def _poll_from(local_buffer, offset, first):
    """`_poll` whose first load (`first`) was issued earlier, so the probes of every peer overlap."""
    values = first
    while not _all_lanes_ready(values):
        values = local_buffer.load(offset, _VEC, alignment=_ALIGN, is_volatile=True)
    copy = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
    for lane in cutlass.range_constexpr(_VEC):
        copy[lane] = values[lane]
    return copy


def _first_probes(
    local_buffer, ping_base, workspace_numel_i64, base_idx_i64, PEER_ORDER, WORLD_SIZE
):
    """The first load of every peer's vector, in `PEER_ORDER` from the first peer on (traced
    inline: a Python list of the loads, one a peer)."""
    return [
        local_buffer.load(
            ping_base + cutlass.Int64(PEER_ORDER[k]) * workspace_numel_i64 + base_idx_i64,
            _VEC,
            alignment=_ALIGN,
            is_volatile=True,
        )
        for k in range(1, WORLD_SIZE)
    ]


def _receive(local_buffer, offset, first, neg_zero_vec, STEP: bool):
    """A peer's vector at `offset` of this rank's slot (traced inline): the host-index form polls
    from its first probe and re-arms the vector (this thread is its one reader); the
    device-counter form polls."""
    if STEP:
        got = _poll_from(local_buffer, offset, first)
        local_buffer.store(
            neg_zero_vec[0:_VEC, cutlass.align(_ALIGN)],
            offset,
            _VEC,
            alignment=_ALIGN,
            is_volatile=True,
        )
    else:
        got = _poll(local_buffer, offset)
    return got


@cute.kernel
def _round_trip_kernel(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    signal_state: cute.Tensor,
    launch_index: cutlass.Int32,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    SUM_ORDER: cutlass.Constexpr[tuple[int, ...]],
    REDUCE: cutlass.Constexpr[bool],
    STEP: cutlass.Constexpr[bool],
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
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_VALUE_INDEX)).llvm_ptr

    if cutlass.const_expr(STEP):
        # The region's slot base, moved only by `advance` launches that completed before this grid
        # started, plus this launch's ordinal in the region.
        signal_i32 = (
            cute.arch.load(state_signal_ptr, cutlass.Int32, sem="relaxed", scope="gpu")
            + launch_index
        )
    else:
        signal_i32 = cute.arch.load(state_signal_ptr, cutlass.Int32, sem="acquire", scope="gpu")
        # Every thread of this CTA holds the signal before thread 0 can count the CTA as
        # arrived: the last arrival rotates the signal, and a thread that loaded it after that
        # would publish into the next slot.
        cute.arch.sync_threads()
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
        signal_state.iterator + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX) + cutlass.Int64(clear_slot)
    ).llvm_ptr
    if cutlass.const_expr(STEP):
        # Written two launches earlier on this workspace, by a completed grid.
        clear_vectors = cute.arch.load(clear_count_ptr, cutlass.Int32, sem="relaxed", scope="gpu")
    else:
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
            first = []
            if cutlass.const_expr(STEP):
                first = _first_probes(
                    local_buffer,
                    ping_base,
                    workspace_numel_i64,
                    base_idx_i64,
                    PEER_ORDER,
                    WORLD_SIZE,
                )
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                source_rank = PEER_ORDER[peer_ordinal]
                if source_rank != LOCAL_RANK:
                    got = _receive(
                        local_buffer,
                        ping_base + cutlass.Int64(source_rank) * workspace_numel_i64 + base_idx_i64,
                        first[peer_ordinal - 1] if STEP else None,
                        neg_zero_vec,
                        STEP,
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
            first = []
            if cutlass.const_expr(STEP):
                first = _first_probes(
                    local_buffer,
                    ping_base,
                    workspace_numel_i64,
                    base_idx_i64,
                    PEER_ORDER,
                    WORLD_SIZE,
                )
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                source_rank = PEER_ORDER[peer_ordinal]
                if source_rank != LOCAL_RANK:
                    got = _receive(
                        local_buffer,
                        ping_base + cutlass.Int64(source_rank) * workspace_numel_i64 + base_idx_i64,
                        first[peer_ordinal - 1] if STEP else None,
                        neg_zero_vec,
                        STEP,
                    )
                    output_array.store(
                        got[0:_VEC, cutlass.align(_ALIGN)],
                        cutlass.Int64(source_rank) * active_numel_i64 + base_idx_i64,
                        _VEC,
                        alignment=_ALIGN,
                    )

    if cutlass.const_expr(STEP):
        # Every vector was re-armed by its reader: the slot's count is 0, a plain store by CTA 0,
        # read two launches later by the re-arm pass.
        if bidx + tidx == 0:
            slot_count_ptr = (
                signal_state.iterator
                + cutlass.Int64(_SIGNAL_SLOT_COUNTS_INDEX)
                + cutlass.Int64(ping_slot)
            ).llvm_ptr
            cute.arch.store(slot_count_ptr, cutlass.Int32(0), sem="relaxed", scope="gpu")
    # Otherwise the last CTA to arrive records how many vectors the peers wrote into this slot
    # and rotates the signal; every CTA has read it above.
    elif tidx == 0:
        _advance_signal_if_last_cta(
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
    launch_index: cutlass.Int32,
    stream: cuda.CUstream,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    SUM_ORDER: cutlass.Constexpr[tuple[int, ...]],
    REDUCE: cutlass.Constexpr[bool],
    STEP: cutlass.Constexpr[bool],
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
        launch_index,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        PEER_ORDER,
        SUM_ORDER,
        REDUCE,
        STEP,
    ).launch(grid=(num_blocks, 1, 1), block=(_THREADS, 1, 1), stream=stream)


#: Compiled launches by `(rank, world, device, hidden, reduce, sum order, step)`: a rebuild of the
#: same workspace (the post-load walk reaches the build twice) compiles nothing new.
_COMPILED: dict = {}


def _compile(
    rank: int, world: int, device_id: int, hidden: int, reduce: bool, sum_order, step: bool = False
):
    key = (rank, world, device_id, hidden, reduce, tuple(sum_order), bool(step))
    if key not in _COMPILED:
        _COMPILED[key] = _compile_new(*key)
    return _COMPILED[key]


def _compile_new(
    rank: int, world: int, device_id: int, hidden: int, reduce: bool, sum_order, step: bool
):
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
        cutlass.Int32, (_SIGNAL_STATE_SIZE,), assumed_align=_STATE_ALIGN
    )
    return cute.compile(
        _round_trip,
        fake_in,
        fake_out,
        fake_buffers,
        fake_state,
        cutlass.Int32(0),
        make_fake_stream(),
        hidden,
        rank,
        world,
        tuple((rank + k) % world for k in range(world)),
        tuple(sum_order),
        reduce,
        step,
        options="--enable-tvm-ffi",
    )


@cute.kernel
def _advance_kernel(signal_state: cute.Tensor, n: cutlass.Int32):
    """`state[1] = (state[1] + n) mod 3`: the end of a host-index region. One thread."""
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        ptr = (signal_state.iterator + cutlass.Int64(_SIGNAL_VALUE_INDEX)).llvm_ptr
        value = cute.arch.load(ptr, cutlass.Int32, sem="relaxed", scope="gpu")
        cute.arch.store(ptr, (value + n) % cutlass.Int32(_PING_PONG), sem="relaxed", scope="gpu")


@cute.jit
def _advance(signal_state: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream) -> None:
    _advance_kernel(signal_state, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def _compile_advance(device_id: int):
    key = ("advance", device_id)
    if key not in _COMPILED:
        cutlass.cuda.initialize_cuda_context(device_id)
        fake_state = make_fake_compact_tensor(
            cutlass.Int32, (_SIGNAL_STATE_SIZE,), assumed_align=_STATE_ALIGN
        )
        _COMPILED[key] = cute.compile(
            _advance, fake_state, cutlass.Int32(0), make_fake_stream(), options="--enable-tvm-ffi"
        )
    return _COMPILED[key]


def nccl_ring_order(rank: int, world: int) -> tuple:
    """The order NCCL's ring reduce-scatter adds the partials of `rank`'s slice in: the
    chain starts at the owner's successor on the ring and ends at the owner."""
    return tuple((rank + 1 + k) % world for k in range(world))


class LamportSteps:
    """Host-side launch indices of host-index (`step`) workspaces: one count a workspace.

    `region()` brackets a forward: each launch on a workspace takes the next index (`next`), and
    when the outermost region closes one `advance` a workspace moves its slot base by the count
    mod 3 (none when that is 0), on the stream of the workspace's last launch. Nested regions
    count into the outermost. A launch outside any region takes index 0 and its own advance by 1.
    Nothing here reads device state, so a captured region records the launches and indices a
    replay runs. Every rank of the group makes the same launches in the same order (each launch is
    a collective), so the ranks hand out the same indices."""

    def __init__(self) -> None:
        self.depth = 0
        self.counts: dict = {}

    def next(self, trip: "DepRoundTrip", stream) -> int | None:
        """`trip`'s index in the open region, or None outside any region."""
        if self.depth == 0:
            return None
        entry = self.counts.setdefault(id(trip.state), [trip, 0, stream])
        index = entry[1]
        entry[1], entry[2] = index + 1, stream
        return index

    @contextlib.contextmanager
    def region(self):
        """One region; a captured region that raises records nothing that runs, so it advances
        nothing."""
        self.depth += 1
        if self.depth == 1:
            self.counts = {}
        try:
            yield
        except BaseException:
            self.depth -= 1
            if not self.depth:
                if not torch.cuda.is_current_stream_capturing():
                    self._advance_all()
                self.counts = {}
            raise
        self.depth -= 1
        if not self.depth:
            self._advance_all()
            self.counts = {}

    def _advance_all(self) -> None:
        for trip, count, stream in self.counts.values():
            if count % _PING_PONG:
                trip.advance(count, stream)


@dataclass(frozen=True)
class DepRoundTrip:
    """One rank's peer buffers, slot state and the two compiled launches."""

    hidden: int
    max_rows: int
    rank: int
    world: int
    #: This rank's `[3, world, max_rows * hidden]` bf16 buffer and every rank's, mapped here, in
    #: rank order.
    buffer: torch.Tensor
    peer_buffers: list
    #: `[6]` int32: entry counter, current slot, per-slot vector counts, (unused) epoch.
    state: torch.Tensor
    gather_fn: object
    reduce_fn: object
    #: The host-index form: the registry its launches take their indices from, and the compiled
    #: `advance`. None: the device-counter form.
    steps: LamportSteps | None = None
    advance_fn: object = None
    #: What owns the peer mappings (`build_on`'s caller's holder), kept alive with the workspace.
    keepalive: object = None

    def takes(self, rows: int, x: torch.Tensor) -> bool:
        """Every rank answers the same: the padded row count and the dtype are the group's."""
        return (
            1 <= rows <= self.max_rows and x.dtype is torch.bfloat16 and x.shape[1] == self.hidden
        )

    def advance(self, n: int, stream) -> None:
        """Move the host-index slot base by `n` mod 3 (one 1-thread launch on `stream`)."""
        self.advance_fn(self.state, int(n) % _PING_PONG, stream)

    def _launch(self, fn, a: torch.Tensor, out: torch.Tensor) -> None:
        stream = _raw_stream(a.device.index)
        index = None if self.steps is None else self.steps.next(self, stream)
        fn(a, out, self.peer_buffers, self.state, 0 if index is None else index, stream)
        if self.steps is not None and index is None:
            self.advance(1, stream)

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """`[R, H]` on every rank -> the group's rows `[world * R, H]`, rank-major."""
        x = x.contiguous()
        out = torch.empty((self.world * x.shape[0], self.hidden), dtype=x.dtype, device=x.device)
        self._launch(self.gather_fn, x, out)
        return out

    def reduce_scatter(self, y: torch.Tensor) -> torch.Tensor:
        """`[world * R, H]` partials on every rank -> this rank's `[R, H]` summed slice."""
        y = y.contiguous()
        rows = y.shape[0] // self.world
        assert rows * self.world == y.shape[0] and rows <= self.max_rows, (y.shape, self.max_rows)
        out = torch.empty((rows, self.hidden), dtype=y.dtype, device=y.device)
        self._launch(self.reduce_fn, y, out)
        return out


def workspace_shape(hidden: int, max_rows: int, world: int) -> tuple:
    """A rank's buffer: `[3, world, max_rows * hidden]` bf16 (`max_rows` capped at `MAX_ROWS`)."""
    return (_PING_PONG, world, min(max_rows, MAX_ROWS) * hidden)


def workspace_bytes(hidden: int, max_rows: int, world: int) -> int:
    """The bytes of a rank's buffer (`workspace_shape`)."""
    slots, ranks, numel = workspace_shape(hidden, max_rows, world)
    return 2 * slots * ranks * numel


def buffers_from_pointers(ptrs, hidden: int, max_rows: int, world: int, device_index: int) -> list:
    """Every rank's buffer, given as this process's device pointers to them (rank order), as bf16
    `workspace_shape` tensors on `cuda:<device_index>` (DLPack with the device stated: a mapping of
    another node's memory is this GPU's to touch). They do not own the memory.

    The memory is packed as int16 and viewed as bf16: `pack_strided_memory` states every float
    dtype with DLPack's IEEE float code, so a bf16 request would come back as float16."""
    from tensorrt_llm._dlpack_utils import pack_strided_memory

    shape = workspace_shape(hidden, max_rows, world)
    nbytes = workspace_bytes(hidden, max_rows, world)
    return [
        pack_strided_memory(int(ptr), nbytes, nbytes, 1, torch.int16, device_index)[0]
        .view(torch.bfloat16)
        .view(shape)
        for ptr in ptrs
    ]


def _assemble(
    hidden, max_rows, rank, world, buffer, peer_buffers, sum_order, steps, keepalive
) -> DepRoundTrip:
    dev = buffer.device
    state = torch.zeros(_SIGNAL_STATE_SIZE, dtype=torch.int32, device=dev)
    order = nccl_ring_order(rank, world) if sum_order is None else tuple(sum_order)
    assert sorted(order) == list(range(world)), order
    step = steps is not None
    return DepRoundTrip(
        hidden=hidden,
        max_rows=max_rows,
        rank=rank,
        world=world,
        buffer=buffer,
        peer_buffers=peer_buffers,
        state=state,
        gather_fn=_compile(rank, world, dev.index, hidden, False, order, step),
        reduce_fn=_compile(rank, world, dev.index, hidden, True, order, step),
        steps=steps,
        advance_fn=_compile_advance(dev.index) if step else None,
        keepalive=keepalive,
    )


def build(
    hidden: int, max_rows: int, rank: int, world: int, sum_order=None, steps=None
) -> DepRoundTrip:
    """Pin the peer buffers over the torch process group that is already up (symmetric memory,
    one node) and compile the two launches, in the host-index form with `steps` (a
    `LamportSteps`). Collective: every rank of the group calls it with the same arguments."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    assert hidden % _VEC == 0, hidden
    max_rows = min(max_rows, MAX_ROWS)
    assert dist.is_initialized() and dist.get_world_size() == world and dist.get_rank() == rank
    dev = torch.device("cuda", torch.cuda.current_device())
    buffer = symm_mem.empty(
        workspace_shape(hidden, max_rows, world), dtype=torch.bfloat16, device=dev
    )
    # Every slot starts not-ready, or the first exchange reads uninitialized memory as data.
    buffer.view(torch.int16).fill_(_NEG_ZERO)
    group = dist.group.WORLD
    handle = symm_mem.rendezvous(buffer, group=getattr(group, "group_name", group))
    peer_buffers = [handle.get_buffer(p, buffer.shape, buffer.dtype) for p in range(world)]
    built = _assemble(hidden, max_rows, rank, world, buffer, peer_buffers, sum_order, steps, None)
    # Every rank holds its buffers and both launches before any of them exchanges a row.
    ok = torch.ones(1, dtype=torch.int32, device=dev)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    return built


def build_on(
    hidden: int,
    max_rows: int,
    rank: int,
    world: int,
    peer_buffers: list,
    sum_order=None,
    steps=None,
    keepalive=None,
) -> DepRoundTrip:
    """The workspace over buffers the caller mapped (`peer_buffers`: every rank's
    `workspace_shape` bf16 buffer on this rank's device, rank order, e.g.
    `buffers_from_pointers` of `distributed.peer_buffers.PeerBuffers`), with `keepalive` the
    object that owns the mappings. Fills this rank's buffer with the marker and compiles the
    launches; rank-local. Before any rank launches, every rank must have returned from it (the
    caller's barrier, after a device synchronize)."""
    assert hidden % _VEC == 0, hidden
    max_rows = min(max_rows, MAX_ROWS)
    shape = workspace_shape(hidden, max_rows, world)
    assert len(peer_buffers) == world, (len(peer_buffers), world)
    assert all(tuple(b.shape) == shape and b.dtype is torch.bfloat16 for b in peer_buffers), shape
    buffer = peer_buffers[rank]
    # Every slot starts not-ready, or the first exchange reads uninitialized memory as data.
    buffer.view(torch.int16).fill_(_NEG_ZERO)
    return _assemble(
        hidden, max_rows, rank, world, buffer, list(peer_buffers), sum_order, steps, keepalive
    )


def _sanitized(x: torch.Tensor) -> torch.Tensor:
    """`x` with every bf16 -0.0 as +0.0 (the transport's marker)."""
    return torch.where(x.view(torch.int16) == _NEG_ZERO, torch.zeros_like(x), x)


class DepLamportRoundTrip(OpWrapper):
    """`dep_lamport_round_trip(x, trip, reduce)`, the attention-DP MoE round trip's two
    collectives over `trip` (`build`): `reduce=False` gathers every rank's `[R, H]` bf16 rows,
    rank-major, into `[W * R, H]`; `reduce=True` sums every rank's `[W * R, H]` partials of this
    rank's slice `[r * R, (r + 1) * R)` into `[R, H]`, in `trip`'s order with a bf16 rounding
    after every add (NCCL's ring order by default). A -0.0 lane travels as +0.0. Every rank calls
    it at the same point with the same `R` (`trip.takes`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(world=world, hidden=2560, rows=rows, step=step))
        for world, rows, step, why in (
            (4, 1, False, "dep4 decode: one row a rank (a padded idle rank)"),
            (4, 12, False, "dep4 MTP-3 verify, 3 requests a rank"),
            (4, 256, False, "dep4 MTP-3 verify, 64 requests a rank"),
            (4, MAX_ROWS, False, "the largest round trip a launch serves (640 CTAs, all resident)"),
            (4, 12, True, "dep4 MTP-3 verify, 3 requests a rank, host-index slots"),
            (8, 12, False, "dep8 MTP-3 verify, 3 requests a rank"),
            (8, 12, True, "dep8 MTP-3 verify, 3 requests a rank, host-index slots"),
            (8, 112, True, "dep8 MTP-3 verify, 28 requests a rank, host-index slots"),
            (8, MAX_ROWS, True, "dep8: the largest round trip, host-index slots"),
        )
    )

    note = """
    Source: two CuTe DSL kernels (`_round_trip_kernel`, gather and reduce-scatter) in two slot
    protocols (device counter; host index with `LamportSteps`), compiled per rank at build
    (`--enable-tvm-ffi`), over peer-mapped buffers: torch symmetric memory on the torch group of one
    node (`build`), or buffers the caller maps (`build_on`; the NVFP4 target's dep8 takes
    `distributed.peer_buffers.PeerBuffers`, MNNVL across the two nodes of one rack). Certified on
    GB300 at 4 ranks (one node), hidden 2560, 1 to `MAX_ROWS` rows; written for 8 ranks over the
    two nodes of one rack (the cells), whose prototype ran the same statements over CUDA fabric
    handles. Not certified: other world sizes, ranks of two racks, a stream holding SMs that a
    launch's resident CTAs wait for.
    """

    def __call__(self, x: torch.Tensor, trip: DepRoundTrip, reduce: bool) -> torch.Tensor:
        return trip.reduce_scatter(x) if reduce else trip.gather(x)

    def reference(
        self, xs_in_rank_order, trip_rank: int, reduce: bool, sum_order=None
    ) -> torch.Tensor:
        """Collective (the documented exception to mirroring `__call__`): every rank's input in
        rank order, and the rank whose result to form."""
        world = len(xs_in_rank_order)
        if not reduce:
            return torch.cat([_sanitized(x) for x in xs_in_rank_order])
        rows = xs_in_rank_order[0].shape[0] // world
        parts = [_sanitized(x[trip_rank * rows : (trip_rank + 1) * rows]) for x in xs_in_rank_order]
        order = nccl_ring_order(trip_rank, world) if sum_order is None else tuple(sum_order)
        acc = parts[order[0]]
        for k in order[1:]:
            acc = (acc.float() + parts[k].float()).to(torch.bfloat16)
        return acc

    def is_valid(self, x: torch.Tensor, trip: DepRoundTrip, reduce: bool) -> None:
        assert x.dtype is torch.bfloat16 and x.dim() == 2 and x.shape[1] == trip.hidden
        rows = x.shape[0] // trip.world if reduce else x.shape[0]
        assert (not reduce or rows * trip.world == x.shape[0]) and trip.takes(rows, x)


dep_lamport_round_trip = DepLamportRoundTrip()
