# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch low-latency exchanges over a tensor-parallel group (no NCCL).

``TPKeyExchange``: greedy argmax of a vocabulary-sharded LM head. Every rank packs the local best
candidate of each row into one int64 key

    key = (orderable_int32(value_fp32) << 32) | (0xFFFFFFFF - global_index)

(``orderable_int32(v) = bits(v) if bits(v) >= 0 else bits(v) ^ 0x7FFFFFFF``, with -0.0 taken as
+0.0), so the signed int64 maximum over the ranks is the global argmax and, among equal values, the
lowest global index wins -- ``torch.argmax`` over the concatenated vocabulary.
``TPGroupExchange``: all-gather and all-to-all + LSE merge within TP subgroups (KV-replica split attention). Its
words move as Lamport data: plain 32-bit words in receive slots that hold the sentinel 0xFFFFFFFF until written; a
reader polls until no word of its tile is the sentinel, then writes it back. Slot halves and epochs as below.

TPKeyExchange protocol (NCCL LL-protocol style): each 32-bit half of a key travels as one 8-byte
packet ``(epoch << 32) | word`` that the rank stores into every rank's receive slot; a reader polls
its own slot until the packets of every rank carry the current epoch, so data and flag arrive in
one store and no slot is ever cleared. Each channel has its own device epoch counter,
advanced by the launch at its end, and two slots chosen by the epoch's parity:

* consecutive launches of a channel use different slots, so the packets a fast rank pushes for
  launch ``e + 1`` never overwrite launch-``e`` packets that a slow rank is still polling;
* launch ``e + 2`` reuses launch ``e``'s slot only after every rank finished reading it: a rank
  enters ``e + 2`` after it received every rank's ``e + 1`` packets, which each rank pushed after it
  had finished launch ``e``.

The kernel never releases its dependents early (no ``griddepcontrol.launch_dependents``): every
later kernel starts after it exited, so the next launch of a channel always reads the epoch this
one wrote. Under PDL it may start early, but it reads the epoch and its inputs only after
``griddepcontrol.wait`` (emitted with a memory clobber so no load is hoisted above it). Rows and
channels must be used identically on every rank, as with any collective (same calls, same order,
same row counts), which also keeps the epochs equal across ranks; CUDA-graph replays keep advancing
them consistently.

Receive buffers: IPC memory inside a node, plain fabric memory (no multicast object) across the
nodes of one NVLink domain. Without either, ``create`` returns None and callers keep NCCL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..modules.fla.op import gdc_wait_with_memory_clobber

# Fewest rows (keys) per rank and launch a TPKeyExchange is built for.
_MIN_MAX_ROWS = 512
# Independent epoch channels of TPKeyExchange.
_NUM_CHANNELS = 16
# Programs per SM of the group all-gather (one 256-word block each, up to this many blocks per SM).
_GROUP_GRID = 2
# Programs per SM of alltoall_merge (one wave: every program is resident, so a program polling for a peer's rows
# never waits on a peer program that is not scheduled).
_MERGE_GRID = 8
# Warps of alltoall_merge; 0: up to 4, one per 32 words of a row's poll tile.
_MERGE_WARPS = 0
# With argmax_row_split, argmax_logits launches up to 32 programs of 2 warps per row up to this many rows (Gumbel
# rows: from _ARGMAX_ROW_SPLIT_GUMBEL_MIN_ROWS), else ~2 (greedy) / ~8 (Gumbel) programs per SM in total.
_ARGMAX_ROW_SPLIT_MAX_ROWS = 512
_ARGMAX_ROW_SPLIT_GUMBEL_MIN_ROWS = 15
# Constants read inside the Triton kernels must be constexpr globals.
_LO_MASK = tl.constexpr(0xFFFFFFFF)
_NEG_ZERO_BITS = tl.constexpr(-2147483648)  # 0x80000000 as int32 (-0.0f)
_INT64_MIN = tl.constexpr(-(2**63))
_INT32_MAX = tl.constexpr(2147483647)
# Lamport words: an empty receive word, and what a data word equal to it (only NaN data) is sent as instead (NaN in
# every FP8 byte, both BF16 halves and as FP32).
LAMPORT_EMPTY = -1
_LAMPORT_EMPTY = tl.constexpr(LAMPORT_EMPTY)
_LAMPORT_NAN = tl.constexpr(0x7FFFFFFF)


def _block_m(m: int, max_rows: int) -> int:
    """Rows tile of one launch: 64 up to 64 rows, the next power of two above (at most ``max_rows``)."""
    return min(max_rows, max(64, triton.next_power_of_2(m)))


@triton.jit
def _orderable_key_hi(v):
    """Signed int32 whose order is the float order of ``v`` (fp32), with -0.0 taken as +0.0."""
    bits = v.to(tl.int32, bitcast=True)
    bits = tl.where(bits == _NEG_ZERO_BITS, 0, bits)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _ll_exchange_words(
    hi_w,
    lo_w,
    rows,
    ok,
    m,
    max_out_ptr,
    gather_out_ptr,
    tok_out_ptr,
    state_ptr,
    pv,
    rank,
    NRANKS: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DO_MAX: tl.constexpr,
    DO_GATHER: tl.constexpr,
    DO_TOKENS: tl.constexpr,
):
    """Exchange this rank's keys (``hi_w`` / ``lo_w``: int64 [BLOCK_M] holding the 32-bit halves) with
    every rank of the group and write the requested outputs; advances the channel's epoch.

    ``state_ptr``: int32 epoch of the channel; ``peers_ptr``: int64 [NRANKS] addresses of the channel's
    region in every rank's receive buffer (two slots of NRANKS x WORDS 8-byte packets, source-rank
    major, (lo, hi) word pairs per row)."""
    epoch = tl.load(state_ptr, volatile=True)
    slot = (epoch & 1).to(tl.int64) * (NRANKS * WORDS)
    tag = epoch.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    lo_mask = tl.full([BLOCK_M], _LO_MASK, tl.int64)
    pk_lo = (tag | lo_w.to(tl.uint64, bitcast=True)).to(tl.int64, bitcast=True)
    pk_hi = (tag | hi_w.to(tl.uint64, bitcast=True)).to(tl.int64, bitcast=True)

    mine_off = slot + rank * WORDS + 2 * rows
    ar_n = tl.arange(0, NRANKS)
    for peer in tl.static_range(NRANKS):
        dst = tl.sum(tl.where(ar_n == peer, pv, 0)).to(tl.uint64).to(tl.pointer_type(tl.int64))
        tl.store(dst + mine_off, pk_lo, mask=ok)
        tl.store(dst + mine_off + 1, pk_hi, mask=ok)

    mine = tl.sum(tl.where(ar_n == rank, pv, 0)).to(tl.uint64).to(tl.pointer_type(tl.int64))
    src = tl.arange(0, NRANKS)
    offs = slot + src[:, None] * WORDS + 2 * rows[None, :]
    mask = (src[:, None] < NRANKS) & ok[None, :]
    got_lo = tl.load(mine + offs, mask=mask, other=0, volatile=True)
    got_hi = tl.load(mine + offs + 1, mask=mask, other=0, volatile=True)
    stale = ((got_lo >> 32).to(tl.int32) != epoch) | ((got_hi >> 32).to(tl.int32) != epoch)
    waiting = tl.sum((stale & mask).to(tl.int32))
    while waiting > 0:
        got_lo = tl.load(mine + offs, mask=mask, other=0, volatile=True)
        got_hi = tl.load(mine + offs + 1, mask=mask, other=0, volatile=True)
        stale = ((got_lo >> 32).to(tl.int32) != epoch) | ((got_hi >> 32).to(tl.int32) != epoch)
        waiting = tl.sum((stale & mask).to(tl.int32))

    keys_all = ((got_hi & lo_mask[None, :]) << 32) | (got_lo & lo_mask[None, :])
    if DO_GATHER:
        tl.store(gather_out_ptr + src[:, None] * m + rows[None, :], keys_all, mask=mask)
    if DO_MAX or DO_TOKENS:
        kmax = tl.max(tl.where(mask, keys_all, _INT64_MIN), axis=0)
        if DO_MAX:
            tl.store(max_out_ptr + rows, kmax, mask=ok)
        if DO_TOKENS:
            tok = lo_mask - (kmax & lo_mask)
            tl.store(tok_out_ptr + rows, tok.to(tl.int32), mask=ok)

    # Epoch 0 is never used: zeroed buffers hold epoch-0 packets.
    nxt = epoch + 1
    nxt = tl.where(nxt == 0, 2, nxt)
    tl.store(state_ptr, nxt)


@triton.jit(do_not_specialize=["m", "rank", "vocab_offset"])
def _tp_key_exchange_kernel(
    keys_ptr,
    vals_ptr,
    lidx_ptr,
    max_out_ptr,
    gather_out_ptr,
    tok_out_ptr,
    state_ptr,
    peers_ptr,
    m,
    rank,
    vocab_offset,
    NRANKS: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FROM_VALUES: tl.constexpr,
    DO_MAX: tl.constexpr,
    DO_GATHER: tl.constexpr,
    DO_TOKENS: tl.constexpr,
    WAIT: tl.constexpr,
    RESET_KEYS: tl.constexpr,
):
    """One CTA: this rank's keys (given, or packed from (value, local index)) -> exchange.

    ``RESET_KEYS``: after reading them, store INT64_MIN into the caller's keys, so a producer can
    fold its next keys into the same buffer with atomic max (no separate re-arm launch). Safe
    because this kernel never releases its dependents early: the producer's next atomics (after
    its own griddepcontrol.wait) come after this kernel exited."""
    pv = tl.load(peers_ptr + tl.arange(0, NRANKS))  # host-written table: ahead of the wait
    if WAIT:
        gdc_wait_with_memory_clobber()
    lo_mask = tl.full([BLOCK_M], _LO_MASK, tl.int64)
    rows = tl.arange(0, BLOCK_M)
    ok = rows < m
    if FROM_VALUES:
        v = tl.load(vals_ptr + rows, mask=ok, other=0.0).to(tl.float32)
        okey = _orderable_key_hi(v)
        gidx = tl.load(lidx_ptr + rows, mask=ok, other=0).to(tl.int64) + vocab_offset
        hi_w = okey.to(tl.int64) & lo_mask
        lo_w = (lo_mask - gidx) & lo_mask
    else:
        key = tl.load(keys_ptr + rows, mask=ok, other=0)
        if RESET_KEYS:
            tl.store(keys_ptr + rows, tl.full([BLOCK_M], _INT64_MIN, tl.int64), mask=ok)
        hi_w = (key >> 32) & lo_mask
        lo_w = key & lo_mask
    _ll_exchange_words(
        hi_w,
        lo_w,
        rows,
        ok,
        m,
        max_out_ptr,
        gather_out_ptr,
        tok_out_ptr,
        state_ptr,
        pv,
        rank,
        NRANKS,
        WORDS,
        BLOCK_M,
        DO_MAX,
        DO_GATHER,
        DO_TOKENS,
    )


@triton.jit
def fast_gumbel_key(
    logits_ptr,
    stride_m,
    v_shard,
    temp_ptr,
    seed_ptr,
    row,
    part,
    vocab_offset,
    vocab_full,
    greedy_threshold,
    split,
    BLOCK_V: tl.constexpr,
    SEED_OFFSET: tl.constexpr,
):
    """The exchange key of chunk ``part`` of ``row`` under the four-draws Gumbel noise of seed
    ``seed_ptr[0] + SEED_OFFSET`` (``_argmax_exchange_kernel``'s FAST_GUMBEL). ``split`` (programs per row) may be a
    runtime value: a caller that launches at many row counts keeps one compiled kernel."""
    chunk = tl.cdiv(tl.cdiv(v_shard, split), BLOCK_V) * BLOCK_V  # whole tiles
    start = part * chunk
    end = tl.minimum(start + chunk, v_shard)
    row_ptr = logits_ptr + row.to(tl.int64) * stride_m
    lanes = tl.arange(0, BLOCK_V)
    lane_v = tl.full([BLOCK_V], float("-inf"), tl.float32)
    lane_i = start + lanes
    temperature = tl.load(temp_ptr + row).to(tl.float32)
    if temperature >= greedy_threshold:
        seed = tl.load(seed_ptr) + SEED_OFFSET
        inv_t = 1.0 / temperature
        base = (row * vocab_full + vocab_offset) // 4
        q = tl.arange(0, BLOCK_V // 4)
        for off in range(start, end, BLOCK_V):
            cols = off + lanes
            x = tl.load(row_ptr + cols, mask=cols < end, other=float("-inf"))
            xs = x.to(tl.float32) * inv_t
            u0, u1, u2, u3 = tl.rand4x(seed, base + off // 4 + q)
            u = tl.reshape(tl.join(tl.join(u0, u1), tl.join(u2, u3)), [BLOCK_V])
            u = tl.minimum(tl.maximum(u, 1e-12), 0.99999994)  # strictly inside (0, 1)
            xs = xs - tl.log(-tl.log(u))
            take = xs > lane_v
            lane_v = tl.where(take, xs, lane_v)
            lane_i = tl.where(take, cols, lane_i)
    else:
        for off in range(start, end, BLOCK_V):
            cols = off + lanes
            x = tl.load(row_ptr + cols, mask=cols < end, other=float("-inf")).to(tl.float32)
            take = x > lane_v
            lane_v = tl.where(take, x, lane_v)
            lane_i = tl.where(take, cols, lane_i)
    best_v = tl.max(lane_v, axis=0)
    best_i = tl.min(tl.where(lane_v == best_v, lane_i, _INT32_MAX), axis=0)
    return (_orderable_key_hi(best_v).to(tl.int64) << 32) | (
        _LO_MASK - (best_i.to(tl.int64) + vocab_offset)
    )


@triton.jit
def argmax_exchange_finish(
    key,
    row,
    m,
    max_out_ptr,
    tok_out_ptr,
    scratch_ptr,
    ticket_ptr,
    state_ptr,
    pv,
    rank,
    split,
    NRANKS: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DO_MAX: tl.constexpr,
    DO_TOKENS: tl.constexpr,
):
    """Fold ``key`` into ``row``'s scratch key; the program that takes the last ticket swaps the keys
    out (resetting the scratch and the ticket) and runs the exchange. Returns whether this program
    took the last ticket. ``split`` (programs per row) may be a runtime value."""
    tl.atomic_max(scratch_ptr + row, key, sem="relaxed", scope="gpu")
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
    last = ticket == m * split - 1
    if last:
        rows = tl.arange(0, BLOCK_M)
        ok = rows < m
        lo_mask = tl.full([BLOCK_M], _LO_MASK, tl.int64)
        floor = tl.full([BLOCK_M], _INT64_MIN, tl.int64)
        keys = tl.atomic_xchg(scratch_ptr + rows, floor, mask=ok, sem="acquire", scope="gpu")
        tl.store(ticket_ptr, 0)
        _ll_exchange_words(
            (keys >> 32) & lo_mask,
            keys & lo_mask,
            rows,
            ok,
            m,
            max_out_ptr,
            tok_out_ptr,
            tok_out_ptr,
            state_ptr,
            pv,
            rank,
            NRANKS,
            WORDS,
            BLOCK_M,
            DO_MAX,
            False,
            DO_TOKENS,
        )
    return last


@triton.jit(
    do_not_specialize=[
        "m",
        "v_shard",
        "stride_m",
        "rank",
        "vocab_offset",
        "vocab_full",
        "greedy_threshold",
    ]
)
def _argmax_exchange_kernel(
    logits_ptr,
    stride_m,
    v_shard,
    max_out_ptr,
    tok_out_ptr,
    scratch_ptr,
    ticket_ptr,
    state_ptr,
    peers_ptr,
    temp_ptr,
    seed_ptr,
    m,
    rank,
    vocab_offset,
    vocab_full,
    greedy_threshold,
    SPLIT: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NRANKS: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DO_MAX: tl.constexpr,
    DO_TOKENS: tl.constexpr,
    WAIT: tl.constexpr,
    GUMBEL: tl.constexpr,
    FAST_GUMBEL: tl.constexpr = False,
):
    """Global greedy argmax (or Gumbel-max sample) of vocabulary-sharded logits in one launch.

    ``m * SPLIT`` programs: program ``p`` takes row ``p // SPLIT``, chunk ``p % SPLIT`` of the
    ``v_shard`` columns, packs its first maximum into a key and ``atomic_max``es it into
    ``scratch_ptr[row]`` (int64 [BLOCK_M], INT64_MIN between launches). The last program to take a
    ticket (``ticket_ptr``, 0 between launches) swaps the keys out (resetting the scratch), resets
    the ticket and runs the exchange. Per row: the largest value, lowest global index on ties --
    ``torch.argmax`` over the concatenated vocabulary.

    ``GUMBEL``: rows with ``temp[row] >= greedy_threshold`` maximize ``logit / T + G`` instead, with
    ``G = -log(-log(u))``, ``u = tl.rand(seed, row * vocab_full + global index)`` clamped into
    (0, 1) -- the noise of a vocabulary entry does not depend on the sharding, so the global argmax
    is a sample of softmax(logits / T); ``seed`` (int64 [1]) is read after the dependency wait.

    ``FAST_GUMBEL`` (TPKeyExchange fast_gumbel; DO_MAX off): the Gumbel rows draw one Philox call per four entries
    instead -- the noise of global entry ``4 * n + j`` of ``row`` is word ``(0, 2, 1, 3)[j]`` of ``philox(seed,
    row * vocab_full / 4 + n)``, so the winner does not depend on the sharding, SPLIT or BLOCK_V. Lanes keep running
    maxima; one reduction per program; SPLIT chunks are whole BLOCK_V tiles."""
    pv = tl.load(peers_ptr + tl.arange(0, NRANKS))  # host-written table: ahead of the wait
    if WAIT:
        gdc_wait_with_memory_clobber()
    pid = tl.program_id(0)
    row = pid // SPLIT
    part = pid % SPLIT
    if FAST_GUMBEL:
        key = fast_gumbel_key(
            logits_ptr,
            stride_m,
            v_shard,
            temp_ptr,
            seed_ptr,
            row,
            part,
            vocab_offset,
            vocab_full,
            greedy_threshold,
            SPLIT,
            BLOCK_V,
            0,
        )
    else:
        chunk = tl.cdiv(v_shard, SPLIT)
        start = part * chunk
        end = tl.minimum(start + chunk, v_shard)
        best_v = -float("inf")
        best_i = start
        if GUMBEL:
            temperature = tl.load(temp_ptr + row).to(tl.float32)
            sample = temperature >= greedy_threshold
            seed = tl.load(seed_ptr)
        for off in range(start, end, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            x = tl.load(
                logits_ptr + row.to(tl.int64) * stride_m + cols,
                mask=cols < end,
                other=-float("inf"),
            ).to(tl.float32)
            if GUMBEL:
                if sample:
                    u = tl.rand(seed, row * vocab_full + vocab_offset + cols)
                    u = tl.minimum(tl.maximum(u, 1e-12), 0.99999994)  # strictly inside (0, 1)
                    x = x / temperature - tl.log(-tl.log(u))
            v, i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
            take = v > best_v  # strict: an earlier tile keeps a tie
            best_i = tl.where(take, off + i, best_i)
            best_v = tl.where(take, v, best_v)
        key = (_orderable_key_hi(best_v).to(tl.int64) << 32) | (
            _LO_MASK - (best_i.to(tl.int64) + vocab_offset)
        )
    argmax_exchange_finish(
        key,
        row,
        m,
        max_out_ptr,
        tok_out_ptr,
        scratch_ptr,
        ticket_ptr,
        state_ptr,
        pv,
        rank,
        SPLIT,
        NRANKS,
        WORDS,
        BLOCK_M,
        DO_MAX,
        DO_TOKENS,
    )


# Tile width of the four-draws Gumbel kernel.
_FAST_GUMBEL_BLOCK_V = 1024


def _fast_gumbel_split(m: int, v_shard: int, num_sms: int) -> int:
    """Programs per row of the four-draws Gumbel kernel: ~8 programs per SM in total, at most 64 per row."""
    return 2 * max(
        1, min(32, (4 * num_sms) // max(m, 1), triton.cdiv(v_shard, _FAST_GUMBEL_BLOCK_V))
    )


@triton.jit
def _pack_argmax_keys_kernel(vals_ptr, lidx_ptr, keys_ptr, m, vocab_offset, BLOCK_M: tl.constexpr):
    """Reference packing of (value, local index) into exchange keys (tests / callers without fusion)."""
    rows = tl.arange(0, BLOCK_M)
    ok = rows < m
    v = tl.load(vals_ptr + rows, mask=ok, other=0.0).to(tl.float32)
    gidx = tl.load(lidx_ptr + rows, mask=ok, other=0).to(tl.int64) + vocab_offset
    key = (_orderable_key_hi(v).to(tl.int64) << 32) | (_LO_MASK - gidx)
    tl.store(keys_ptr + rows, key, mask=ok)


def pack_argmax_keys(
    values: torch.Tensor, local_index: torch.Tensor, vocab_offset: int
) -> torch.Tensor:
    """int64 keys ``(orderable(value) << 32) | (0xFFFFFFFF - (local_index + vocab_offset))``."""
    m = values.numel()
    keys = torch.empty(m, dtype=torch.int64, device=values.device)
    _pack_argmax_keys_kernel[(1,)](
        values.reshape(-1),
        local_index.reshape(-1),
        keys,
        m,
        vocab_offset,
        BLOCK_M=max(16, triton.next_power_of_2(m)),
        num_warps=1,
    )
    return keys


def unpack_argmax_tokens(keys: torch.Tensor) -> torch.Tensor:
    """Token ids (int32) of exchange keys (``0xFFFFFFFF - low word``)."""
    return (0xFFFFFFFF - (keys & 0xFFFFFFFF)).to(torch.int32)


class _PeerBuffers:
    """``nbytes`` per rank of a TP group, writable by every rank; filled with byte ``fill`` (default 0) before
    ``create`` returns.

    IPC memory inside a node; across the nodes of an NVLink domain unicast MNNVL memory (``TpMnnvlMemory``, no
    multicast object: NVLS multicast groups are a rack-wide NVSwitch resource and these exchanges only use the
    peer mappings). Setup is collective; a failure returns None on every rank (the caller keeps its NCCL path)."""

    def __init__(
        self,
        local_ptr: int,
        peer_ptrs: Sequence[int],
        nbytes: int,
        transport: str,
        keepalive: object,
    ):
        self.local_ptr = local_ptr
        self.peer_ptrs = list(peer_ptrs)
        self.nbytes = nbytes
        self.transport = transport
        self._keepalive = keepalive

    def close(self) -> None:
        """Release the memory now (every rank must be done with the buffers)."""
        keep, self._keepalive = self._keepalive, None
        del keep

    @classmethod
    def create(cls, mapping: Mapping, nbytes: int, fill: int = 0) -> Optional["_PeerBuffers"]:
        """Collective over the TP group (every rank must call it in step); None without peer memory."""
        from tensorrt_llm._ipc_utils import IpcMemory, can_access_peer
        from tensorrt_llm._utils import mpi_disabled

        from .mnnvl_memory import MnnvlMemory, TpMnnvlMemory

        if mpi_disabled():
            return None
        use_ipc = not mapping.is_multi_node() and can_access_peer(mapping)
        if not use_ipc and not (mapping.is_multi_node() and MnnvlMemory.supports_mnnvl()):
            return None
        # The TP group's communicator (ordered by TP rank) for the agreement steps.
        comm = TpMnnvlMemory.get_comm(mapping)
        if use_ipc:
            ipc = None
            try:
                ipc = IpcMemory(mapping, nbytes, True)  # cudaMalloc + memset 0 + handle all-gather
                ok = ipc.open_ipc
            except RuntimeError as e:
                logger.warning(f"[ll_exchange] IPC peer buffers failed: {e}")
                ok = False
            if not all(comm.allgather(ok)):
                del ipc
                return None
            error = None
            try:
                if fill:
                    from cuda.bindings import runtime as cudart

                    (err,) = cudart.cudaMemset(ipc.local_ptr, fill, nbytes)
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"cudaMemset: {err}")
                torch.cuda.synchronize()
            except RuntimeError as e:
                error = e
            # Agreed before the barrier: every rank's buffer is filled before any rank writes into it.
            if not cls._all_filled(comm, "IPC", error):
                del ipc
                return None
            return cls(ipc.local_ptr, ipc.peer_ptrs, nbytes, "IPC", ipc)
        try:
            mem = TpMnnvlMemory(mapping, nbytes)
        except RuntimeError as e:  # raised on every rank: the allocation agrees on failures
            logger.warning(f"[ll_exchange] MNNVL peer buffers failed: {e}")
            return None
        peers = [mem.ptr + r * mem.rank_stride for r in range(mapping.tp_size)]
        error = None
        try:
            mem.as_torch_strided_tensor(torch.uint8)[mapping.tp_rank].fill_(fill)
            torch.cuda.synchronize()
        except RuntimeError as e:
            error = e
        if not cls._all_filled(comm, "MNNVL", error):
            del mem
            return None
        return cls(peers[mapping.tp_rank], peers, nbytes, "MNNVL", mem)

    @staticmethod
    def failures(comm: Any, error: Optional[BaseException]) -> List[str]:
        """Collective over ``comm`` (``TpMnnvlMemory.get_comm``): "rank r: <cause>" for every rank that passed an
        error, the same list on every rank (empty when all succeeded)."""
        errors = comm.allgather(None if error is None else str(error))
        return [f"rank {r}: {e}" for r, e in enumerate(errors) if e is not None]

    @staticmethod
    def _all_filled(comm: Any, transport: str, error: Optional[BaseException]) -> bool:
        """Every rank's fill outcome, agreed over the TP group (the barrier the writes wait for); on any failure
        every rank logs each failed rank's cause and declines the buffers."""
        failed = _PeerBuffers.failures(comm, error)
        if failed:
            logger.warning(
                f"[ll_exchange] {transport} peer buffer fill failed ({'; '.join(failed)})"
            )
        return not failed


class TPKeyExchange:
    """Elementwise max / gather of ``M <= MAX_ROWS`` int64 keys per rank over a TP group, one launch."""

    fast_gumbel: bool = False
    argmax_row_split: bool = False

    def __init__(
        self,
        rank: int,
        nranks: int,
        buffers: _PeerBuffers,
        num_channels: int,
        max_rows: int,
        fast_gumbel: bool = False,
        argmax_row_split: bool = False,
    ):
        """``fast_gumbel``: argmax_logits' temperature rows draw their Gumbel noise four entries per Philox call
        (another noise stream than the default kernel's, so other tokens for a given seed).
        ``argmax_row_split``: argmax_logits splits rows by _ARGMAX_ROW_SPLIT_* (same tokens, other launch shape)."""
        if nranks not in (2, 4, 8, 16) or len(buffers.peer_ptrs) != nranks:
            raise ValueError(f"unsupported exchange group: rank {rank} of {nranks}")
        self.fast_gumbel = fast_gumbel
        self.argmax_row_split = argmax_row_split
        self.rank = rank
        self.nranks = nranks
        self.num_channels = num_channels
        self.MAX_ROWS = max_rows
        self._words = 2 * max_rows  # 32-bit words per rank: (lo, hi) per key
        self.transport = buffers.transport
        self._buffers = buffers
        device = torch.device("cuda", torch.cuda.current_device())
        self.channel_bytes = 2 * nranks * self._words * 8
        # Epoch 0 = never sent (the buffers start zeroed); the first launch of a channel uses epoch 1.
        self._state = torch.ones(num_channels, dtype=torch.int32, device=device)
        # argmax_logits: per-channel atomic-max key scratch (INT64_MIN between launches) and ticket.
        self._scratch = torch.full(
            (num_channels, max_rows), -(2**63), dtype=torch.int64, device=device
        )
        self._ticket = torch.zeros(num_channels, dtype=torch.int32, device=device)
        self._num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        table = [
            [p + c * self.channel_bytes for p in buffers.peer_ptrs] for c in range(num_channels)
        ]
        self._peers = torch.tensor(table, dtype=torch.int64, device=device)
        from ..flashinfer_utils import get_env_enable_pdl

        self._pdl = get_env_enable_pdl()

    @staticmethod
    def buffer_bytes(nranks: int, num_channels: int, max_rows: int) -> int:
        return num_channels * 2 * nranks * 2 * max_rows * 8

    def close(self) -> None:
        """Release the peer memory (collective use only: every rank calls it after its last launch)."""
        if self._buffers is not None:
            torch.cuda.synchronize()
            self._buffers.close()
            self._buffers = None

    @classmethod
    def create(
        cls,
        mapping,
        max_rows: int,
        num_channels: int = _NUM_CHANNELS,
        fast_gumbel: bool = False,
        argmax_row_split: bool = False,
    ) -> Optional["TPKeyExchange"]:
        """Collective over the TP group; None where peer memory is unavailable (caller keeps NCCL).

        ``max_rows``: the most keys one launch takes, rounded up to a power of two (at least 512)."""
        if (
            mapping.tp_size not in (2, 4, 8, 16)
            or mapping.enable_attention_dp
            or mapping.has_pp()
            or mapping.cp_size > 1
        ):
            return None
        max_rows = max(_MIN_MAX_ROWS, triton.next_power_of_2(max_rows))
        buffers = _PeerBuffers.create(
            mapping, cls.buffer_bytes(mapping.tp_size, num_channels, max_rows)
        )
        if buffers is None:
            return None
        return cls(
            mapping.tp_rank,
            mapping.tp_size,
            buffers,
            num_channels,
            max_rows,
            fast_gumbel,
            argmax_row_split,
        )

    def _check(self, m: int, channel: int):
        if not 1 <= m <= self.MAX_ROWS:
            raise ValueError(f"TPKeyExchange takes 1..{self.MAX_ROWS} rows, got {m}")
        if not 0 <= channel < self.num_channels:
            raise ValueError(f"channel {channel} out of range [0, {self.num_channels})")

    def fast_gumbel_launch(self, m: int, v_shard: int, channel: int) -> tuple:
        """For a kernel that runs ``fast_gumbel_key`` and ``argmax_exchange_finish`` itself (an
        ``argmax_logits`` sample with its own epilogue): ``(grid, split, block_v, scratch, ticket,
        state, peers, rank, nranks, words, block_m, pdl)`` of an ``m``-row call on ``channel``."""
        self._check(m, channel)
        split = _fast_gumbel_split(m, v_shard, self._num_sms)
        return (
            m * split,
            split,
            _FAST_GUMBEL_BLOCK_V,
            self._scratch[channel],
            self._ticket[channel:],
            self._state[channel:],
            self._peers[channel],
            self.rank,
            self.nranks,
            self._words,
            _block_m(m, self.MAX_ROWS),
            self._pdl,
        )

    def _launch(
        self,
        m,
        channel,
        keys=None,
        vals=None,
        lidx=None,
        vocab_offset=0,
        max_out=None,
        gather_out=None,
        tok_out=None,
        wait: Optional[bool] = None,
        reset_keys: bool = False,
    ):
        dummy = self._state  # never dereferenced when its constexpr switch is off
        _tp_key_exchange_kernel[(1,)](
            keys if keys is not None else dummy,
            vals if vals is not None else dummy,
            lidx if lidx is not None else dummy,
            max_out if max_out is not None else dummy,
            gather_out if gather_out is not None else dummy,
            tok_out if tok_out is not None else dummy,
            self._state[channel:],
            self._peers[channel],
            m,
            self.rank,
            vocab_offset,
            NRANKS=self.nranks,
            WORDS=self._words,
            BLOCK_M=_block_m(m, self.MAX_ROWS),
            FROM_VALUES=vals is not None,
            DO_MAX=max_out is not None,
            DO_GATHER=gather_out is not None,
            DO_TOKENS=tok_out is not None,
            WAIT=self._pdl if wait is None else wait,
            RESET_KEYS=reset_keys,
            num_warps=4 if m <= 64 else 8,
            launch_pdl=self._pdl,
        )

    def exchange(
        self,
        keys: torch.Tensor,
        max_out: Optional[torch.Tensor] = None,
        gather_out: Optional[torch.Tensor] = None,
        channel: int = 0,
        want_max: Optional[bool] = None,
        tokens_out: Optional[torch.Tensor] = None,
        reset_keys: bool = False,
        _wait: Optional[bool] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Elementwise max over the group (``max_out`` int64 [M]) and/or every rank's keys
        (``gather_out`` int64 [nranks, M], rank-major) of this rank's ``keys`` int64 [M].

        ``tokens_out`` (int32 [M]): also write the token ids of the maxima
        (``0xFFFFFFFF - low word``); ``want_max`` then defaults to False. ``reset_keys``: store
        INT64_MIN into ``keys`` after reading them (see ``_tp_key_exchange_kernel``)."""
        assert keys.dtype == torch.int64 and keys.dim() == 1 and keys.is_contiguous()
        m = keys.numel()
        self._check(m, channel)
        if want_max is None:
            want_max = tokens_out is None or max_out is not None
        if want_max and max_out is None:
            max_out = torch.empty_like(keys)
        self._launch(
            m,
            channel,
            keys=keys,
            max_out=max_out if want_max else None,
            gather_out=gather_out,
            tok_out=tokens_out,
            wait=_wait,
            reset_keys=reset_keys,
        )
        return max_out if want_max else None, gather_out

    def argmax_tokens(
        self,
        values: torch.Tensor,
        local_index: torch.Tensor,
        vocab_offset: int,
        tokens_out: Optional[torch.Tensor] = None,
        max_out: Optional[torch.Tensor] = None,
        channel: int = 0,
    ) -> torch.Tensor:
        """Global greedy token (int32 [M]) from each rank's local row maxima ``values`` [M] (any
        float dtype) at ``local_index`` [M] of its vocabulary shard starting at ``vocab_offset``."""
        m = values.numel()
        self._check(m, channel)
        values = values.reshape(-1)
        local_index = local_index.reshape(-1)
        if tokens_out is None:
            tokens_out = torch.empty(m, dtype=torch.int32, device=values.device)
        self._launch(
            m,
            channel,
            vals=values,
            lidx=local_index,
            vocab_offset=vocab_offset,
            max_out=max_out,
            tok_out=tokens_out,
        )
        return tokens_out

    def argmax_logits(
        self,
        logits: torch.Tensor,
        vocab_offset: int,
        tokens_out: Optional[torch.Tensor] = None,
        max_out: Optional[torch.Tensor] = None,
        channel: int = 1,
        split: Optional[int] = None,
        temperature: Optional[torch.Tensor] = None,
        seed: Optional[torch.Tensor] = None,
        greedy_threshold: float = 1e-4,
    ) -> torch.Tensor:
        """Global greedy tokens (int32 [M]) of this rank's vocabulary shard ``logits`` [M, V_shard]
        (bf16 / fp16 / fp32, unit stride in the vocabulary dimension) starting at global column
        ``vocab_offset``: local first maximum, packing and exchange in one launch. Same tokens as
        ``torch.argmax`` over the concatenated vocabulary. ``max_out`` (int64 [M]) gets the winning
        keys. Use a channel no other exchange uses concurrently (default 1).

        ``temperature`` (fp32 [M]) + ``seed`` (int64 [1], read in-kernel after the dependency wait):
        rows with T >= ``greedy_threshold`` (the sampler's GREEDY_TEMPERATURE_THRESHOLD) draw a
        temperature-only sample by Gumbel-max (see ``_argmax_exchange_kernel``); other rows stay
        greedy. Every rank must pass the same temperatures and seed value."""
        assert logits.dim() == 2 and logits.stride(-1) == 1
        m, v_shard = logits.shape
        self._check(m, channel)
        gumbel = temperature is not None
        if gumbel:
            assert seed is not None and seed.dtype == torch.int64 and temperature.numel() == m
        if (
            gumbel
            and self.fast_gumbel
            and split is None
            and max_out is None
            and v_shard % 4 == 0
            and vocab_offset % 4 == 0
        ):
            fsplit = _fast_gumbel_split(m, v_shard, self._num_sms)
            if tokens_out is None:
                tokens_out = torch.empty(m, dtype=torch.int32, device=logits.device)
            _argmax_exchange_kernel[(m * fsplit,)](
                logits,
                logits.stride(0),
                v_shard,
                self._state,
                tokens_out,
                self._scratch[channel],
                self._ticket[channel:],
                self._state[channel:],
                self._peers[channel],
                temperature.reshape(-1),
                seed,
                m,
                self.rank,
                vocab_offset,
                v_shard * self.nranks,
                float(greedy_threshold),
                SPLIT=fsplit,
                BLOCK_V=_FAST_GUMBEL_BLOCK_V,
                NRANKS=self.nranks,
                WORDS=self._words,
                BLOCK_M=_block_m(m, self.MAX_ROWS),
                DO_MAX=False,
                DO_TOKENS=True,
                WAIT=self._pdl,
                GUMBEL=True,
                FAST_GUMBEL=True,
                num_warps=4,
                launch_pdl=self._pdl,
            )
            return tokens_out
        num_warps = 4 if m <= 64 else 8
        block_v = None
        # Every split draws the same tokens: an entry's noise depends only on (seed, row, global index) and ties go
        # to the lowest index.
        if split is None and self.argmax_row_split:
            if m <= _ARGMAX_ROW_SPLIT_MAX_ROWS and not (
                gumbel and m < _ARGMAX_ROW_SPLIT_GUMBEL_MIN_ROWS
            ):
                split = max(1, min(32, triton.cdiv(v_shard, 256)))
                block_v = 512 if gumbel else None
                num_warps = 2
            elif gumbel:
                # Gumbel-max rows cost Philox + two logs per entry rather than bytes: ~8 programs per SM in total,
                # at most 32 per row, at least 1024 columns each.
                split = max(1, min(32, (8 * self._num_sms) // m, triton.cdiv(v_shard, 1024)))
        if split is None:
            # ~2 programs per SM in total, at least 1 and at most 32 per row.
            split = max(1, min(32, (2 * self._num_sms) // m, triton.cdiv(v_shard, 256)))
        if block_v is None:
            block_v = max(16, min(4096, triton.next_power_of_2(triton.cdiv(v_shard, split))))
        if tokens_out is None:
            tokens_out = torch.empty(m, dtype=torch.int32, device=logits.device)
        _argmax_exchange_kernel[(m * split,)](
            logits,
            logits.stride(0),
            v_shard,
            max_out if max_out is not None else self._state,
            tokens_out,
            self._scratch[channel],
            self._ticket[channel:],
            self._state[channel:],
            self._peers[channel],
            temperature.reshape(-1) if gumbel else self._state,
            seed if gumbel else self._state,
            m,
            self.rank,
            vocab_offset,
            v_shard * self.nranks,
            float(greedy_threshold),
            SPLIT=split,
            BLOCK_V=block_v,
            NRANKS=self.nranks,
            WORDS=self._words,
            BLOCK_M=_block_m(m, self.MAX_ROWS),
            DO_MAX=max_out is not None,
            DO_TOKENS=True,
            WAIT=self._pdl,
            GUMBEL=gumbel,
            num_warps=num_warps,
            launch_pdl=self._pdl,
        )
        return tokens_out


# The Lamport protocol of the group exchanges (allgather, ll_pos_gather, alltoall_merge; the CuTe producers that push
# into a merge slot follow it too): a sender stores its words into every member's receive slot, a word equal to the
# empty sentinel sent as _LAMPORT_NAN; a receiver polls its slot until no word is empty, then empties it again. Each
# position (block / row) has its own epoch, a plain counter whose parity picks one of the two slot halves, so the
# next write to a half is two launches of the channel away.


@triton.jit
def _ll_word(words):
    """``words`` as sent: the empty sentinel (only NaN data) goes out as another NaN."""
    return tl.where(words == _LAMPORT_EMPTY, _LAMPORT_NAN, words)


@triton.jit
def _ll_peer(pv, ar_g, d):
    """Member ``d``'s receive region (int32 words) from the group's peer table ``pv`` [G]."""
    return tl.sum(tl.where(ar_g == d, pv, 0)).to(tl.uint64).to(tl.pointer_type(tl.int32))


@triton.jit
def _ll_half(epoch, G: tl.constexpr, SLOT_WORDS: tl.constexpr):
    """Word offset of the slot half that ``epoch`` (its parity) selects."""
    return (epoch & 1).to(tl.int64) * (G * SLOT_WORDS)


@triton.jit
def _ll_empty_words(got, mask, MASKED: tl.constexpr):
    """How many of the polled words ``got`` (where ``mask``, if ``MASKED``) have not arrived."""
    empty = got == _LAMPORT_EMPTY
    if MASKED:
        empty = empty & mask
    return tl.sum(empty.to(tl.int32))


@triton.jit
def _ll_reset(ptr, offs, got, mask, MASKED: tl.constexpr):
    """Empty the received words again (``got``'s shape, where ``mask``, if ``MASKED``)."""
    empty = tl.full(got.shape, _LAMPORT_EMPTY, tl.int32)
    if MASKED:
        tl.store(ptr + offs, empty, mask=mask)
    else:
        tl.store(ptr + offs, empty)


@triton.jit
def _ll_load(ptr, offs, mask, MASKED: tl.constexpr):
    if MASKED:
        got = tl.load(ptr + offs, mask=mask, other=0, volatile=True)
    else:
        got = tl.load(ptr + offs, volatile=True)
    return got


@triton.jit
def _ll_poll(ptr, offs, mask, MASKED: tl.constexpr):
    """The words at ``ptr + offs`` (where ``mask``, if ``MASKED``) once every one arrived; their slot is emptied
    again."""
    got = _ll_load(ptr, offs, mask, MASKED)
    waiting = _ll_empty_words(got, mask, MASKED)
    while waiting > 0:
        got = _ll_load(ptr, offs, mask, MASKED)
        waiting = _ll_empty_words(got, mask, MASKED)
    _ll_reset(ptr, offs, got, mask, MASKED)
    return got


@triton.jit
def _ll_group_push(
    src_ptr,
    pv,
    ar_g,
    slot_base,
    block,
    epoch,
    n_words,
    grank,
    G: tl.constexpr,
    SLOT_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = _ll_half(epoch, G, SLOT_WORDS) + slot_base
    w = block * BLOCK + tl.arange(0, BLOCK)
    ok = w < n_words
    lw = _ll_word(tl.load(src_ptr + w, mask=ok, other=0))
    for d in tl.static_range(G):
        tl.store(_ll_peer(pv, ar_g, d) + slot + grank * SLOT_WORDS + w, lw, mask=ok)


@triton.jit
def _ll_group_poll(
    out_ptr,
    mine,
    block,
    epoch,
    n_words,
    row_words,
    out_row_stride,
    G: tl.constexpr,
    SLOT_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = _ll_half(epoch, G, SLOT_WORDS)
    src_rank = tl.arange(0, G)
    w = block * BLOCK + tl.arange(0, BLOCK)
    ok = w < n_words
    offs = slot + src_rank[:, None].to(tl.int64) * SLOT_WORDS + w[None, :]
    mask = (src_rank[:, None] < G) & ok[None, :]
    got = _ll_poll(mine, offs, mask, True)
    row = w // row_words
    col = w % row_words
    out_offs = (
        row[None, :].to(tl.int64) * out_row_stride + src_rank[:, None] * row_words + col[None, :]
    )
    tl.store(out_ptr + out_offs, got, mask=mask)


@triton.jit
def _ll_group_kernel(
    src_ptr,
    out_ptr,
    state_ptr,
    peers_ptr,
    n_words,
    row_words,
    out_row_stride,
    num_blocks,
    grank,
    G: tl.constexpr,
    SLOT_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
    WAIT: tl.constexpr,
    TRIGGER: tl.constexpr,
):
    """One launch of a Lamport all-gather of ``n_words`` 32-bit words per member.

    Words from member ``s`` land at ``out[(w // row_words) * out_row_stride + s * row_words +
    w % row_words]``. Every block of ``BLOCK`` words has its own epoch (``state_ptr[block]``) and slot
    parity, advanced by the one program that moves it, so no cross-program coordination is needed.
    Program ``p`` moves blocks ``p, p + grid, ...`` and pushes all before polling any; the grid stays
    within one wave. Only the host-written peer table is read before griddepcontrol.wait.

    ``TRIGGER``: 1 = release dependents at entry; 2 = right after this grid's wait (consumers may read
    the predecessors' outputs before their own wait, never the exchange's). Safe because consumers
    read the output after their own wait, and every launch touches epochs / slots only after its
    wait, so back-to-back exchanges stay ordered."""
    if TRIGGER == 1:
        tl.extra.cuda.gdc_launch_dependents()
    ar_g = tl.arange(0, G)
    pv = tl.load(peers_ptr + ar_g)
    mine = _ll_peer(pv, ar_g, grank)
    if WAIT:
        gdc_wait_with_memory_clobber()
    if TRIGGER == 2:
        tl.extra.cuda.gdc_launch_dependents()
    program = tl.program_id(0)
    stride = tl.num_programs(0)
    if num_blocks <= stride:
        # One block per program (the decode sizes): the epoch stays in a register.
        epoch = tl.load(state_ptr + program, volatile=True)
        _ll_group_push(
            src_ptr,
            pv,
            ar_g,
            0,
            program,
            epoch,
            n_words,
            grank,
            G,
            SLOT_WORDS,
            BLOCK,
        )
        _ll_group_poll(
            out_ptr,
            mine,
            program,
            epoch,
            n_words,
            row_words,
            out_row_stride,
            G,
            SLOT_WORDS,
            BLOCK,
        )
        tl.store(state_ptr + program, epoch + 1)
    else:
        for block in range(program, num_blocks, stride):
            epoch = tl.load(state_ptr + block, volatile=True)
            _ll_group_push(
                src_ptr,
                pv,
                ar_g,
                0,
                block,
                epoch,
                n_words,
                grank,
                G,
                SLOT_WORDS,
                BLOCK,
            )
        for block in range(program, num_blocks, stride):
            epoch = tl.load(state_ptr + block, volatile=True)
            _ll_group_poll(
                out_ptr,
                mine,
                block,
                epoch,
                n_words,
                row_words,
                out_row_stride,
                G,
                SLOT_WORDS,
                BLOCK,
            )
            tl.store(state_ptr + block, epoch + 1)


@triton.jit
def ll_pos_gather(
    words,
    pv,
    state_ptr,
    out_ptr,
    pos,
    row,
    head,
    grank,
    G: tl.constexpr,
    SLOT_WORDS: tl.constexpr,
    W: tl.constexpr,
    HQ: tl.constexpr,
):
    """Lamport all-gather of this member's W words of position ``pos`` (= row * HQ + head)
    inside a producer kernel (TPGroupExchange.rope_gather_args) into ``out`` [rows, G * HQ * W] words, member-major as
    allgather. One epoch per position; the caller's grid must stay within one wave (every program polls)."""
    epoch = tl.load(state_ptr + pos, volatile=True)
    half = _ll_half(epoch, G, SLOT_WORDS)
    w = tl.arange(0, W)
    ar_g = tl.arange(0, G)
    base = half + grank * SLOT_WORDS + pos.to(tl.int64) * W
    offs = half + ar_g[:, None].to(tl.int64) * SLOT_WORDS + pos.to(tl.int64) * W + w[None, :]
    mine = _ll_peer(pv, ar_g, grank)
    lw = _ll_word(words)
    for d in tl.static_range(G):
        tl.store(_ll_peer(pv, ar_g, d) + base + w, lw)
    got = _ll_poll(mine, offs, 0, False)
    out_offs = (
        row.to(tl.int64) * (G * HQ * W)
        + ar_g[:, None].to(tl.int64) * (HQ * W)
        + head * W
        + w[None, :]
    )
    tl.store(out_ptr + out_offs, got)
    tl.store(state_ptr + pos, epoch + 1)


# alltoall_merge receive layout inside each (parity, source) slot of a channel: the O words of row position pos at
# [pos * D/2, (pos + 1) * D/2), its LSE word at LSE_BASE + pos (the slot's last max_merge_rows words); one epoch
# per row position.


@triton.jit
def _bf16_lo(word):
    return (word << 16).to(tl.float32, bitcast=True)


@triton.jit
def _bf16_hi(word):
    return (word & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _bf16_pack(lo, hi):
    lo16 = lo.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    hi16 = hi.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    return lo16 | (hi16 << 16)


@triton.jit
def _ll_merge_kernel(
    o_ptr,  # int32 view of O [G, P, T, Hh, D] (BF16 pairs; Hh stride D/2 words, unit inner)
    o_sd,
    o_sp,
    o_st,
    lse_ptr,  # int32 view of LSE [G, P, T, Hh] (FP32, base 2; Hh stride 1)
    l_sd,
    l_sp,
    l_st,
    out_ptr,  # int32 view of out [T, Hh, D]
    out_st,
    out_sh,
    gate_ptr,  # int32 view of gate [T, Hh, D]
    g_st,
    g_sh,
    xo_ptr,  # int32 view of the extra (local) O [T, Hh, D]
    xo_st,
    xo_sh,
    xl_ptr,  # extra LSE [T, Hh] FP32 (Hh stride 1)
    xl_st,
    valid_ptr,  # int32 [G, P, T] (T stride 1): partial (source, segment, token) present
    v_sd,
    v_sp,
    state_ptr,  # int32 [max_merge_rows]: one epoch per slot position (row * P + p) of the channel
    peers_ptr,  # int64 [G]: the channel's region in every member's receive buffer
    o8_ptr,  # int16 view of the E4M3 out [T, Hh, D] (two elements per word, the lower one in the low byte)
    o8_st,
    o8_sh,
    qs_ptr,  # FP32 [1]: the consumer's static input scale s (OUT_FP8)
    rows,
    grank,
    HH: tl.constexpr,
    DW: tl.constexpr,  # D / 2 words per (row, partial)
    G: tl.constexpr,
    P: tl.constexpr,  # partials per (member, row): 1 or 2
    SLOT_WORDS: tl.constexpr,
    LSE_BASE: tl.constexpr,
    HAS_GATE: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_VALID: tl.constexpr,
    WAIT: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
    OUT_FP8: tl.constexpr,
    PUSH: tl.constexpr,
):
    """All-to-all of per-destination (O, LSE) partials + LSE-weighted merge (see alltoall_merge).

    Row r = t * HH + h; partial (source s, segment p) of row r sits at slot position r * P + p (O words
    at pos * DW, LSE word at LSE_BASE + pos), every position with its own epoch. Program q moves rows
    q, q + grid, ... (push all, then poll + merge each). OUT_FP8: the BF16 result goes out as E4M3 the
    way the consumer's static per-tensor quant writes it (scaleMatrixPerTensorVec: FP32(BF16) x (1 / s),
    cvt.rn.satfinite; 1 / s as div.rn)."""
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    ar_g = tl.arange(0, G)
    pv = tl.load(peers_ptr + ar_g)  # host-written table: ahead of the wait
    mine = _ll_peer(pv, ar_g, grank)
    if WAIT:
        gdc_wait_with_memory_clobber()
    program = tl.program_id(0)
    stride = tl.num_programs(0)
    w = tl.arange(0, DW)
    # PUSH False: the producer kernel already stored this member's words (merge_push_args)
    for r in range(program, rows if PUSH else 0, stride):
        t = r // HH
        h = r % HH
        for p in tl.static_range(P):
            pos = r * P + p
            epoch = tl.load(state_ptr + pos, volatile=True)
            slot = _ll_half(epoch, G, SLOT_WORDS) + grank * SLOT_WORDS
            for d in tl.static_range(G):
                dst = _ll_peer(pv, ar_g, d)
                ow = tl.load(o_ptr + d * o_sd + p * o_sp + t * o_st + h * DW + w)
                lw = tl.load(lse_ptr + d * l_sd + p * l_sp + t * l_st + h)
                tl.store(dst + slot + pos * DW + w, _ll_word(ow))
                tl.store(dst + slot + LSE_BASE + pos, _ll_word(lw))
    c = tl.arange(0, P * DW)  # columns of a row's poll tile: partial c // DW, word c % DW
    cp = c // DW
    first = (c % DW) == 0
    for r in range(program, rows, stride):
        t = r // HH
        h = r % HH
        # [G, P * DW] tiles of the row's O words and (column p * DW only) LSE words, one owner thread per word:
        # racing volatile loads of thread-replicated data could disagree and make the loop exit non-uniform
        # (barrier deadlock in the reductions). Masked-off LSE lanes read as "arrived".
        ecol = tl.load(state_ptr + r * P + cp, volatile=True)  # epoch of each column's position
        slotc = _ll_half(ecol, G, SLOT_WORDS)
        src = ar_g.to(tl.int64) * SLOT_WORDS
        o_offs = slotc[None, :] + src[:, None] + r * (P * DW) + c[None, :]
        l_offs = slotc[None, :] + src[:, None] + LSE_BASE + r * P + cp[None, :]
        lm = first[None, :] & (ar_g[:, None] < G)
        # arrived = not empty (masked-off LSE lanes read 0); read, then empty again. The next write to these words
        # is two launches of the channel away, after this rank's next launch has pushed.
        got_o = tl.load(mine + o_offs, volatile=True)
        got_l = tl.load(mine + l_offs, mask=lm, other=0, volatile=True)
        waiting = _ll_empty_words(got_o, 0, False) + _ll_empty_words(got_l, 0, False)
        while waiting > 0:
            got_o = tl.load(mine + o_offs, volatile=True)
            got_l = tl.load(mine + l_offs, mask=lm, other=0, volatile=True)
            waiting = _ll_empty_words(got_o, 0, False) + _ll_empty_words(got_l, 0, False)
        _ll_reset(mine, o_offs, got_o, 0, False)
        _ll_reset(mine, l_offs, got_l, lm, True)
        tl.store(state_ptr + r * P + cp, ecol + 1, mask=first)
        # every column of partial p carries that partial's LSE (column p * DW holds the word)
        lbits = tl.where(lm, got_l.to(tl.int32), 0)
        lse_full = tl.zeros([G, P * DW], dtype=tl.float32)
        for p in tl.static_range(P):
            lse_p = tl.sum(tl.where(cp[None, :] == p, lbits, 0), axis=1).to(
                tl.float32, bitcast=True
            )
            lse_full = tl.where(cp[None, :] == p, lse_p[:, None], lse_full)
        ok = lse_full > float("-inf")
        if HAS_VALID:
            vld = tl.load(valid_ptr + ar_g[:, None] * v_sd + cp[None, :] * v_sp + t)
            ok = ok & (vld != 0)
        m = tl.max(tl.max(tl.where(ok, lse_full, float("-inf")), axis=1), axis=0)
        if HAS_EXTRA:
            xl = tl.load(xl_ptr + t * xl_st + h)
            m = tl.maximum(m, xl)
        wt = tl.where(ok, tl.exp2(lse_full - m), 0.0)
        den = tl.sum(tl.sum(tl.where(first[None, :], wt, 0.0), axis=1), axis=0)
        ow32 = got_o.to(tl.int32)
        num_lo = tl.sum(tl.where(wt > 0, wt * _bf16_lo(ow32), 0.0), axis=0)  # [P * DW]
        num_hi = tl.sum(tl.where(wt > 0, wt * _bf16_hi(ow32), 0.0), axis=0)
        if P > 1:
            num_lo = tl.sum(tl.reshape(num_lo, (P, DW)), axis=0)
            num_hi = tl.sum(tl.reshape(num_hi, (P, DW)), axis=0)
        if HAS_EXTRA:
            wx = tl.where(xl > float("-inf"), tl.exp2(xl - m), 0.0)
            xw = tl.load(xo_ptr + t * xo_st + h * xo_sh + w)
            num_lo = num_lo + tl.where(wx > 0, wx * _bf16_lo(xw), 0.0)
            num_hi = num_hi + tl.where(wx > 0, wx * _bf16_hi(xw), 0.0)
            den = den + wx
        sden = tl.where(den > 0, den, 1.0)
        res_lo = tl.where(den > 0, num_lo / sden, 0.0)
        res_hi = tl.where(den > 0, num_hi / sden, 0.0)
        res_lo = res_lo.to(tl.bfloat16).to(tl.float32)
        res_hi = res_hi.to(tl.bfloat16).to(tl.float32)
        if HAS_GATE:
            gw = tl.load(gate_ptr + t * g_st + h * g_sh + w)
            res_lo = res_lo * tl.sigmoid(_bf16_lo(gw))
            res_hi = res_hi * tl.sigmoid(_bf16_hi(gw))
        if OUT_FP8:
            s = tl.load(qs_ptr + w * 0)
            f = tl.inline_asm_elementwise(
                "div.rn.f32 $0, 0f3F800000, $1;",
                "=r,r",
                [s],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            q_lo = res_lo.to(tl.bfloat16).to(tl.float32) * f
            q_hi = res_hi.to(tl.bfloat16).to(tl.float32) * f
            # cvt ... d, a, b puts a in the upper byte: (hi, lo) keeps the lower element in the low byte
            w8 = tl.inline_asm_elementwise(
                "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
                "=h,r,r",
                [q_lo, q_hi],
                dtype=tl.int16,
                is_pure=True,
                pack=1,
            )
            tl.store(o8_ptr + t * o8_st + h * o8_sh + w, w8)
        else:
            tl.store(out_ptr + t * out_st + h * out_sh + w, _bf16_pack(res_lo, res_hi))
    if not EARLY_TRIGGER:
        # PDL invariant: released only after a CTA barrier that follows this program's last
        # global access (EARLY_TRIGGER is an opt-in for consumers that only read static data before their wait)
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


class TPGroupExchange:
    """One-launch all-gather and all-to-all + merge within groups of ``group_size`` consecutive TP
    ranks (members ``[g * G, (g + 1) * G)``), e.g. the 4 ranks of a tray sharing a KV head under TP8.

    Data is moved as 32-bit Lamport words (per-row bytes must be a multiple of 4, see the module docstring). Every
    member must issue the same calls (same shapes, same channel order); channels are independent epoch counters for
    call sites that may overlap in time. Graph-safe; inputs are read after griddepcontrol.wait.
    Dependents are released at completion unless ``trigger`` says otherwise (see ``_ll_group_kernel``);
    a consumer always reads the exchange output only after its own griddepcontrol.wait."""

    # Head dim (bytes) of the in-kernel all-gathers (rope_gather_args); 0: none.
    gather_head_dim: int = 0

    def __init__(
        self,
        mapping,
        buffers: _PeerBuffers,
        group_size: int,
        max_words: int,
        max_merge_rows: int,
        num_channels: int,
        gather_head_dim: int = 0,
    ):
        self.group_size = group_size
        self.group = mapping.tp_rank // group_size
        self.grank = mapping.tp_rank % group_size
        self.max_words = max_words
        self.max_merge_rows = max_merge_rows
        self.num_channels = num_channels
        self.transport = buffers.transport
        self._buffers = buffers
        device = torch.device("cuda", torch.cuda.current_device())
        self.channel_bytes = self.buffer_bytes(group_size, max_words, 1)
        members = [self.group * group_size + i for i in range(group_size)]
        table = [
            [buffers.peer_ptrs[r] + c * self.channel_bytes for r in members]
            for c in range(num_channels)
        ]
        self._peers = torch.tensor(table, dtype=torch.int64, device=device)
        # One epoch per (channel, block of 256 words): a counter whose parity picks the slot half (starts at 1).
        self._state = torch.ones(
            (num_channels, triton.cdiv(max_words, 256)), dtype=torch.int32, device=device
        )
        # alltoall_merge: one epoch per (channel, row position). A channel serves either allgather (block epochs)
        # or alltoall_merge, never both (positions would alias across epochs).
        self._merge_state = torch.ones(
            (num_channels, max_merge_rows), dtype=torch.int32, device=device
        )
        # pos_gather (rope_gather_args): one epoch per (channel, position of gather_head_dim bytes) of an all-gather
        # done inside the producing kernel.
        self.gather_head_dim = gather_head_dim
        positions = max_words // (gather_head_dim // 4) if gather_head_dim >= 4 else 0
        self._pos_state = torch.ones(
            (num_channels, max(1, positions)), dtype=torch.int32, device=device
        )
        self._channel_kind: Dict[int, str] = {}
        self._num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        from ..flashinfer_utils import get_env_enable_pdl

        self._pdl = get_env_enable_pdl()

    @staticmethod
    def is_supported(mapping, group_size: int) -> bool:
        """Whether ``create`` can build the exchange for these groups (it still returns None without peer memory):
        TP groups of 2, 4 or 8 consecutive ranks, no attention DP, PP or CP, MPI (not Ray)."""
        from tensorrt_llm._utils import mpi_disabled

        return (
            mapping.tp_size % group_size == 0
            and group_size in (2, 4, 8)
            and not mapping.enable_attention_dp
            and not mapping.has_pp()
            and mapping.cp_size == 1
            and not mpi_disabled()
        )

    @staticmethod
    def buffer_bytes(group_size: int, max_words: int, num_channels: int) -> int:
        """Receive bytes per rank: per channel two slot halves of ``group_size`` source slots of ``max_words``
        words."""
        return num_channels * 2 * group_size * max_words * 4

    @classmethod
    def create(
        cls,
        mapping,
        group_size: int,
        max_bytes: int,
        max_merge_rows: int,
        num_channels: int = 4,
        gather_head_dim: int = 0,
    ) -> Optional["TPGroupExchange"]:
        """Collective over the whole TP group. ``max_bytes``: largest per-(source, destination) payload;
        ``max_merge_rows``: largest ``rows x P`` of one alltoall_merge (its LSE words share that payload);
        ``gather_head_dim``: head dim (bytes) of the in-kernel all-gathers (rope_gather_args), 0 without them."""
        if not cls.is_supported(mapping, group_size):
            return None
        max_words = max_bytes // 4
        buffers = _PeerBuffers.create(
            mapping,
            cls.buffer_bytes(group_size, max_words, num_channels),
            fill=LAMPORT_EMPTY & 0xFF,
        )
        if buffers is None:
            return None
        return cls(
            mapping,
            buffers,
            group_size,
            max_words,
            max_merge_rows,
            num_channels,
            gather_head_dim,
        )

    def close(self) -> None:
        """Release the peer memory (every rank calls it after its last launch)."""
        if self._buffers is not None:
            torch.cuda.synchronize()
            self._buffers.close()
            self._buffers = None

    def _use_channel(self, channel: int, kind: str) -> None:
        if not 0 <= channel < self.num_channels:
            raise ValueError(f"channel {channel} out of range ({self.num_channels} channels)")
        prev = self._channel_kind.setdefault(channel, kind)
        if prev != kind:
            raise ValueError(
                f"channel {channel} already carries {prev} exchanges; use another channel for {kind}"
            )

    def _launch(
        self,
        src_words,
        out_words,
        n_words,
        row_words,
        out_row_stride,
        channel,
        trigger=False,
    ):
        self._use_channel(channel, "gather")
        if not 0 < n_words <= self.max_words:
            raise ValueError(f"payload of {n_words} words exceeds {self.max_words}")
        block = 256
        num_blocks = triton.cdiv(n_words, block)
        _ll_group_kernel[(min(num_blocks, _GROUP_GRID * self._num_sms),)](
            src_words,
            out_words,
            self._state[channel],
            self._peers[channel],
            n_words,
            row_words,
            out_row_stride,
            num_blocks,
            self.grank,
            G=self.group_size,
            SLOT_WORDS=self.max_words,
            BLOCK=block,
            WAIT=self._pdl,
            TRIGGER=self._trigger_mode(trigger),
            num_warps=4,
            launch_pdl=self._pdl,
        )

    def _trigger_mode(self, trigger) -> int:
        """False / None: dependents released at completion; True / "entry": at entry; "after_wait": right
        after this grid's griddepcontrol.wait (every predecessor of the exchange has completed)."""
        if not self._pdl or trigger is None or trigger is False:
            return 0
        if trigger is True or trigger == "entry":
            return 1
        if trigger == "after_wait":
            return 2
        raise ValueError(f"unknown trigger mode {trigger!r}")

    def allgather(
        self, x: torch.Tensor, out: Optional[torch.Tensor] = None, channel: int = 0, trigger=False
    ) -> torch.Tensor:
        """``[rows, *rest]`` per member -> ``[rows, G * prod(rest)]`` (member-major along the last dim);
        e.g. q ``[T, 4, 256]`` -> ``[T, 16 * 256]`` (view it as ``[T, 16, 256]``). FP32 / FP16 / BF16 / E4M3 /
        E5M2 payloads only: a word equal to the empty-slot sentinel (all bits set: NaNs in each of these formats)
        arrives as another NaN; in other dtypes (integers, float64, the fnuz / E8M0 / FP4 formats) the rewrite
        would change a value."""
        if x.dtype not in _ALLGATHER_DTYPES:
            raise ValueError(
                f"allgather moves floating-point payloads whose all-ones words are NaN, got {x.dtype}"
            )
        assert x.is_contiguous()
        rows = x.shape[0]
        row_bytes = x[0].numel() * x.element_size() if rows else 0
        assert row_bytes % 4 == 0
        if out is None:
            out = torch.empty(
                (rows, self.group_size * x[0].numel()), dtype=x.dtype, device=x.device
            )
        row_words = row_bytes // 4
        self._launch(
            x.view(torch.int32),
            out.view(torch.int32),
            rows * row_words,
            row_words,
            self.group_size * row_words,
            channel,
            trigger,
        )
        return out

    def rope_gather_args(
        self, channel: int, positions: int, head_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """(peer table, epochs, member index, group size, slot words) for a kernel that all-gathers
        ``positions`` FP8 heads of ``head_dim`` bytes of its own output through ll_pos_gather (the RoPE / FP8 kernel
        pushing its q heads, fused_qkv_rope_gate_fp8_append_gather). The channel is reserved for this kind; the call
        rules of allgather apply (same calls and shapes on every member)."""
        self._use_channel(channel, "pos_gather")
        if head_dim != self.gather_head_dim:
            raise ValueError(
                f"head dim {head_dim}: the exchange was created for {self.gather_head_dim}"
            )
        if (
            not 0 < positions <= self._pos_state.shape[1]
            or positions * (head_dim // 4) > self.max_words
        ):
            raise ValueError(
                f"{positions} positions of head dim {head_dim} exceed the channel's "
                f"{self._pos_state.shape[1]} positions or {self.max_words} words"
            )
        return (
            self._peers[channel],
            self._pos_state[channel],
            self.grank,
            self.group_size,
            self.max_words,
        )

    def merge_push_args(
        self, channel: int, positions: int, head_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """(peer table, row epochs, member index, slot words, LSE base) for a producer kernel that
        stores its alltoall_merge partials itself (the KV-replica partial attention's combine, ll_push), so the
        following ``alltoall_merge(..., pushed=True)`` only polls and merges. ``positions``: rows x P of the call.
        Layout as the merge's own push: slot half epoch & 1, source slot = member index, O words of position pos at
        pos * D/2, its LSE at the LSE base + pos, as Lamport words."""
        self._use_channel(channel, "merge")
        lse_base = self.max_words - self.max_merge_rows
        # Checked before the producer pushes: alltoall_merge's own check comes after the remote writes.
        if not 0 < positions <= self.max_merge_rows or positions * (head_dim // 2) > lse_base:
            raise ValueError(
                f"{positions} positions of head dim {head_dim} exceed the channel's merge slot"
            )
        return (
            self._peers[channel],
            self._merge_state[channel],
            self.grank,
            self.max_words,
            lse_base,
        )

    def alltoall_merge(
        self,
        o: torch.Tensor,
        lse: torch.Tensor,
        out: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        channel: int = 1,
        early_trigger: bool = True,
        extra_o: Optional[torch.Tensor] = None,
        extra_lse: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        out_fp8: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
        pushed: bool = False,
    ) -> torch.Tensor:
        """All-to-all of LSE-weighted partials + their merge, one launch.

        With ``out_fp8`` [T, Hh, D] E4M3 and ``out_scale`` (the consumer's static input scale s, FP32, one
        element), the merged (gated) BF16 result is written there instead of ``out`` (which may then be None),
        byte for byte as the consumer's own static per-tensor quant of it would (``FP32(BF16) x (1 / s)``,
        cvt.rn.satfinite; 1 / s as div.rn), and ``out_fp8`` is returned.

        ``o`` [G, P, T, Hh, D] BF16 (or [G, T, Hh, D] for P = 1) and ``lse`` [G, P, T, Hh] FP32 (base 2): this
        rank's ``P`` (1 or 2) partials for every member ``d``; any G / P / T strides, Hh / D dense (strides D,
        1), LSE Hh stride 1. Member ``d`` merges, for its rows (t, h), the ``G * P`` partials it receives --
        skipping those with ``valid`` [G, P, T] int32 (local, T stride 1) == 0 -- and the local ``extra_o``
        [T, Hh, D] BF16 / ``extra_lse`` [T, Hh] FP32 partial (always valid) into ``out`` [T, Hh, D] BF16:
        ``m = max lse_c; w_c = lse_c > -inf ? exp2(lse_c - m) : 0; res = den > 0 ? sum(w_c > 0 ? w_c o_c : 0)
        / den : 0`` in FP32, rounded to BF16; with ``gate`` [T, Hh, D] BF16: ``BF16(float(res) *
        sigmoid(float(gate)))``. Token / head strides of out / gate / extra_o are free. The channel is reserved
        for this call kind. Dependents are released at entry (they read ``out`` only after their own
        ``griddepcontrol.wait``), or after the stores with ``early_trigger=False``."""
        self._use_channel(channel, "merge")
        G = self.group_size
        if o.dim() == 4:
            o = o.unsqueeze(1)
        if lse.dim() == 3:
            lse = lse.unsqueeze(1)
        assert o.dim() == 5 and o.shape[0] == G and o.dtype == torch.bfloat16, (
            "o: [G, P, T, Hh, D] bf16"
        )
        _, P, T, HH, D = o.shape
        assert P in (1, 2), "1 or 2 partials per (member, row)"
        if pushed and P != 1:
            # merge_push_args' producer pushes one partial per (member, row): other slots would never be written.
            raise ValueError(
                f"alltoall_merge(pushed=True) takes one partial per (member, row), got {P}"
            )
        assert o.stride(4) == 1 and o.stride(3) == D, "o: Hh / D dims dense"
        assert D % 2 == 0 and (D // 2) & (D // 2 - 1) == 0 and D >= 32, (
            "D / 2 must be a power of two >= 16"
        )
        assert (
            tuple(lse.shape) == (G, P, T, HH) and lse.dtype == torch.float32 and lse.stride(3) == 1
        )
        assert all(st % 2 == 0 for st in o.stride()[:3]), "o: even BF16 strides"
        fp8 = out_fp8 is not None
        if fp8:
            assert (
                tuple(out_fp8.shape) == (T, HH, D)
                and out_fp8.dtype == torch.float8_e4m3fn
                and out_fp8.stride(-1) == 1
                and all(st % 2 == 0 for st in out_fp8.stride()[:2])
            ), "out_fp8: [T, Hh, D] E4M3, D dense, even strides"
            assert (
                out_scale is not None
                and out_scale.dtype == torch.float32
                and out_scale.numel() == 1
            ), "out_scale: FP32, one element"
            o8_w = out_fp8.view(torch.int16)
        else:
            o8_w = None
        if out is not None:
            assert (
                tuple(out.shape) == (T, HH, D)
                and out.dtype == torch.bfloat16
                and out.stride(-1) == 1
            )
        else:
            assert fp8, "out: [T, Hh, D] BF16 (None only with out_fp8)"
        rows = T * HH
        dw = D // 2
        lse_base = self.max_words - self.max_merge_rows
        if rows * P > self.max_merge_rows or rows * P * dw > lse_base:
            raise ValueError(
                f"alltoall_merge payload of {rows} rows x {P} partials x {D} exceeds the channel slot"
            )
        if rows == 0:
            return out_fp8 if fp8 else out
        o_w = o.view(torch.int32)
        lse_w = lse.view(torch.int32)
        # with OUT_FP8 the BF16 store is compiled out: any int32 view serves as its pointer
        out_w = out.view(torch.int32) if out is not None else o_w
        dummy_i = out_w
        if o8_w is None:
            o8_w = o_w.view(torch.int16)  # unused (OUT_FP8 off)
        if gate is not None:
            gate = gate.view(T, HH, D)
            assert gate.dtype == torch.bfloat16 and gate.stride(-1) == 1
            gate_w = gate.view(torch.int32)
        else:
            gate_w = dummy_i
        has_extra = extra_o is not None
        if has_extra:
            extra_o = extra_o.view(T, HH, D)
            assert (
                extra_o.dtype == torch.bfloat16
                and extra_o.stride(-1) == 1
                and extra_lse is not None
            )
            assert (
                tuple(extra_lse.shape) == (T, HH)
                and extra_lse.dtype == torch.float32
                and extra_lse.stride(1) == 1
            )
            xo_w = extra_o.view(torch.int32)
            xl = extra_lse
        else:
            xo_w, xl = dummy_i, out_w.view(torch.float32)
        if valid is not None:
            assert (
                tuple(valid.shape) == (G, P, T)
                and valid.dtype == torch.int32
                and valid.stride(2) == 1
            )
        early = early_trigger
        merge_warps = _MERGE_WARPS or max(1, min(4, G * P * dw // 32))
        _ll_merge_kernel[(min(rows, _MERGE_GRID * self._num_sms),)](
            o_w,
            o_w.stride(0),
            o_w.stride(1),
            o_w.stride(2),
            lse_w,
            lse_w.stride(0),
            lse_w.stride(1),
            lse_w.stride(2),
            out_w,
            out_w.stride(0),
            out_w.stride(1),
            gate_w,
            gate_w.stride(0) if gate is not None else 0,
            gate_w.stride(1) if gate is not None else 0,
            xo_w,
            xo_w.stride(0) if has_extra else 0,
            xo_w.stride(1) if has_extra else 0,
            xl,
            xl.stride(0) if has_extra else 0,
            valid if valid is not None else dummy_i,
            valid.stride(0) if valid is not None else 0,
            valid.stride(1) if valid is not None else 0,
            self._merge_state[channel],
            self._peers[channel],
            o8_w,
            o8_w.stride(0) if fp8 else 0,
            o8_w.stride(1) if fp8 else 0,
            out_scale if fp8 else lse_w,
            rows,
            self.grank,
            HH=HH,
            DW=dw,
            G=G,
            P=P,
            SLOT_WORDS=self.max_words,
            LSE_BASE=lse_base,
            HAS_GATE=gate is not None,
            HAS_EXTRA=has_extra,
            HAS_VALID=valid is not None,
            WAIT=self._pdl,
            EARLY_TRIGGER=bool(early) and self._pdl,
            OUT_FP8=fp8,
            PUSH=not pushed,
            # no thread replication of the [G, P * D/2] poll tiles (see the kernel)
            num_warps=merge_warps,
            launch_pdl=self._pdl,
        )
        return out_fp8 if fp8 else out


_GROUP_EXCHANGES: Dict[tuple, Optional[TPGroupExchange]] = {}

# Payload dtypes of TPGroupExchange.allgather: those whose all-ones words are NaN.
_ALLGATHER_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


def get_tp_group_exchange(
    mapping,
    group_size: int,
    max_bytes: int,
    max_merge_rows: int,
    num_channels: int,
    gather_head_dim: int = 0,
) -> Optional[TPGroupExchange]:
    """The TP subgroup exchange, created by the first call (collective over the TP group, not during graph capture)
    with these sizes and channels; later calls with the same group, channel count and in-kernel gather head dim
    return it as created (each call checks its payload against the sizes it was created with)."""
    key = (
        tuple(mapping.tp_group),
        mapping.tp_rank,
        group_size,
        num_channels,
        gather_head_dim,
        torch.cuda.current_device(),
    )
    if key not in _GROUP_EXCHANGES:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            ex = TPGroupExchange.create(
                mapping, group_size, max_bytes, max_merge_rows, num_channels, gather_head_dim
            )
        except (RuntimeError, ValueError) as e:  # callers keep their path without the exchange
            logger.warning(f"[TPGroupExchange] setup failed: {e}")
            ex = None
        _GROUP_EXCHANGES[key] = ex
        logger.info(
            f"[TPGroupExchange] tp_size={mapping.tp_size} group_size={group_size} rank={mapping.tp_rank}: "
            f"{'LL exchange over ' + ex.transport if ex is not None else 'unavailable'}"
        )
    return _GROUP_EXCHANGES[key]


_EXCHANGES: Dict[tuple, Optional[TPKeyExchange]] = {}


def create_tp_key_exchange(
    mapping, max_rows: int, fast_gumbel: bool = False, argmax_row_split: bool = False
) -> Optional[TPKeyExchange]:
    """Build the TP group's exchange for launches of up to ``max_rows`` keys (collective: every rank of the group
    calls it in step, outside CUDA-graph capture); None where unsupported. A second call returns the first one's.
    ``fast_gumbel``, ``argmax_row_split``: see TPKeyExchange."""
    key = (
        tuple(mapping.tp_group),
        mapping.tp_rank,
        torch.cuda.current_device(),
        fast_gumbel,
        argmax_row_split,
    )
    if key not in _EXCHANGES:
        try:
            ex = TPKeyExchange.create(
                mapping, max_rows, fast_gumbel=fast_gumbel, argmax_row_split=argmax_row_split
            )
        except (RuntimeError, ValueError) as e:  # callers keep the NCCL all-gather
            logger.warning(f"[TPKeyExchange] setup failed, keeping NCCL all-gather: {e}")
            ex = None
        _EXCHANGES[key] = ex
        logger.info(
            f"[TPKeyExchange] tp_size={mapping.tp_size} rank={mapping.tp_rank}: "
            f"{'LL exchange over ' + ex.transport if ex is not None else 'unavailable (NCCL)'}"
        )
    return _EXCHANGES[key]
