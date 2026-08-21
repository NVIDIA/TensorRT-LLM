# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chunk planner: numpy port of the C++ ``BounceTransferPlan``.

Pure bin-packing of a transfer's (src, dst, size) descriptor triples into
chunks that each fit in one bounce region. No CUDA / NIXL / threads.

Packing rules (a chunk is cut when any holds), same as
BounceTransferPlan.cpp:
  - adding the next desc would push the packed extent past
    ``max_chunk_bytes``,
  - the chunk already holds ``max_descs_per_chunk`` descs,
  - the destination device id changes (the C++ rejects mixed dst devices;
    this port supports them by cutting a chunk at every device boundary).

Bounce offsets within a chunk are 32-byte aligned: enough for
memory-coalesced vectorized copies while imposing no stricter requirement
than the underlying registered memory already has. ``packed_bytes`` (the
region extent one RDMA write moves) is the last bounce offset plus its size —
alignment padding between descs is transferred, trailing padding is not.

The scatter-run coalescing reproduces the C++ greedy loop exactly (see
``_coalesce_scatter_runs``); the gather view stays per-desc.

DEVIATION from the C++: the in-place merging of gather descs whose src, dst
and bounce cursor all advance contiguously (``srcDstContig`` in
BounceTransferPlan.cpp) is not ported. It only shrank the C++ side's pinned
plan buffers; here the gather plan is numpy arrays consumed by the bound
batched-copy op, which re-splits runs internally anyway. Consequences: for
fully contiguous inputs this planner counts each input desc against
``max_descs_per_chunk`` individually (the C++ counted a merged desc once), so
desc-count cuts can occur earlier. For dense src+dst descs whose sizes are
NOT multiples of 32 the packing also differs: e.g. two contiguous 100 B descs
merge in C++ into one desc (packed 200 B), while this planner keeps them
separate with a 32-byte-aligned gap (packed 228 B) — the extra padding bytes
travel on the wire, and the scatter view becomes a count-2 strided run
instead of one dense run. Correctness is unaffected (fuzz-verified) and the
overhead is small; for the common 32-multiple sizes there is no gap and the
scatter-run coalescing still collapses dense destinations to single runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ALIGNMENT", "SCATTER_RUN_DTYPE", "BounceChunk", "Plan", "build_plan"]

#: Byte alignment of bounce offsets within a chunk region.
ALIGNMENT = 32

_U32_MAX = 0xFFFFFFFF

#: One scatter RUN of a DATA message: ``count`` equal pieces of ``piece_size``
#: bytes; piece p copies region[bounce_offset + p*bounce_stride ...] to
#: dst_addr + p*dst_stride. count == 1 is a single plain extent (strides 0).
#: Field layout matches the packed 36-byte C++ ``BounceScatterRun`` wire
#: struct, so a structured array serializes with ``tobytes()``.
SCATTER_RUN_DTYPE = np.dtype(
    [
        ("bounce_offset", "<u8"),
        ("dst_addr", "<u8"),
        ("dst_stride", "<u8"),
        ("bounce_stride", "<u4"),
        ("piece_size", "<u4"),
        ("count", "<u4"),
    ]
)
assert SCATTER_RUN_DTYPE.itemsize == 36


@dataclass
class BounceChunk:
    """One chunk: the descs packed into a single bounce region and moved with
    one RDMA write. Offsets are byte offsets within that region."""

    src_ptrs: np.ndarray  # [n] uint64 sender-local source addresses
    dst_ptrs: np.ndarray  # [n] uint64 receiver-local destination addresses
    sizes: np.ndarray  # [n] uint32 per-desc byte counts
    bounce_offsets: np.ndarray  # [n] uint64 per-desc offset within the region
    #: Coalesced scatter view (structured array, SCATTER_RUN_DTYPE): what goes
    #: on the wire in DATA. Adjacent descs merge when (bounce_offset, dst_ptr)
    #: advance contiguously or by a uniform stride, collapsing thousands of
    #: per-desc entries to a handful of runs and keeping the DATA message
    #: (which sits on the ACK critical path) tiny. The per-desc arrays above
    #: stay as-is for the gather (src layout is independent of dst).
    scatter_runs: np.ndarray
    total_bytes: int  # sum of desc sizes (payload only, excludes padding)
    packed_bytes: int  # region extent to RDMA-write: last offset + its size
    dst_device_id: int  # receiver device id (uniform within a chunk)

    @property
    def num_descs(self) -> int:
        return int(self.src_ptrs.shape[0])


@dataclass
class Plan:
    """The full chunking of one transfer request."""

    chunks: list[BounceChunk] = field(default_factory=list)
    total_bytes: int = 0
    total_descs: int = 0  # includes zero-length descs (which never pack)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def flat_gather(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flat per-desc gather view over ALL chunks, in chunk order — the
        one-call marshalling input of the C++ per-request plan handle
        (``BatchedCopyPool.register_plan``).

        Returns ``(srcs, bounce_offsets, sizes, chunk_starts)``: uint64
        source addresses, uint64 REGION-RELATIVE bounce offsets (the staging
        base is a per-launch argument, so the flat plan never references
        arena regions), uint32 sizes, and uint64 ``[num_chunks + 1]``
        desc-index boundaries (chunk c is ``[chunk_starts[c],
        chunk_starts[c + 1])``).
        """
        counts = np.array([c.num_descs for c in self.chunks], dtype=np.uint64)
        chunk_starts = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(counts)))
        if not self.chunks:
            return (
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.uint32),
                chunk_starts.astype(np.uint64),
            )
        srcs = np.concatenate([c.src_ptrs for c in self.chunks])
        offsets = np.concatenate([c.bounce_offsets for c in self.chunks])
        sizes = np.concatenate([c.sizes for c in self.chunks])
        return srcs, offsets, sizes, chunk_starts.astype(np.uint64)


def _block_end_of(cont_false_idx: np.ndarray, start: int, n: int) -> int:
    """Last index j >= start-1 such that ``cont`` holds for all of
    [start, j] (``cont_false_idx`` are the sorted indices where cont is
    False). Returns start-1 when cont[start] itself is False."""
    pos = int(np.searchsorted(cont_false_idx, start))
    nxt = int(cont_false_idx[pos]) if pos < cont_false_idx.shape[0] else n
    return nxt - 1


def _coalesce_scatter_runs(bounce: np.ndarray, dst: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Exact, mostly-vectorized reproduction of the C++ greedy run coalescing
    (``buildScatterRuns`` in BounceTransferPlan.cpp).

    The C++ walks descs once; per desc it tries, in order:
      (a) contiguous growth (run count == 1): bounce AND dst continue exactly
          where the accumulated piece ends -> grow piece_size in place;
      (b) stride latch (count == 1): same desc size, strictly forward dst and
          bounce, bounce step representable in u32 -> fix the strides,
          count = 2;
      (c) stride extension (count >= 2): the desc lands exactly one stride
          past the run's last piece -> count += 1.
    Irregular layouts simply break runs; correctness never depends on merging.

    Vectorization strategy: rule (a) forms maximal DENSE CHAINS, detected with
    one vector compare; rules (b)/(c) over the resulting pieces are an
    automaton whose regular stretches (uniform strides) are jumped over via
    precomputed constant-delta blocks. The Python loop below iterates once per
    OUTPUT RUN (plus once per rare multi-desc-chain boundary), never per desc,
    so a 20k-desc uniformly-strided chunk costs one iteration.

    One subtlety ported faithfully: when a run latches/extends into the HEAD
    desc of a multi-desc dense chain (rules (b)/(c) compare the next DESC's
    raw size), only that head joins the run; the chain's remainder starts a
    fresh run (in C++ the remainder re-merges via (a)). The u32 cap on dense
    growth can never fire here because a chunk's packed extent is already
    capped at max_chunk_bytes <= u32 max.
    """
    n = int(bounce.shape[0])
    if n == 0:
        return np.empty(0, dtype=SCATTER_RUN_DTYPE)

    # --- rule (a): maximal dense chains -> pieces ---
    if n > 1:
        dense = (bounce[1:] == bounce[:-1] + sizes[:-1]) & (dst[1:] == dst[:-1] + sizes[:-1])
        chain_start = np.concatenate(([True], ~dense))
    else:
        chain_start = np.array([True])
    starts = np.flatnonzero(chain_start)  # first desc index of each chain
    ends = np.concatenate((starts[1:], [n]))  # exclusive end desc index
    m = int(starts.shape[0])
    size_prefix = np.concatenate(([0], np.cumsum(sizes, dtype=np.uint64)))
    piece_size = size_prefix[ends] - size_prefix[starts]  # merged sizes (u64)
    piece_bounce = bounce[starts]
    piece_dst = dst[starts]
    head_size = sizes[starts]  # raw size of each chain's FIRST desc
    multi = (ends - starts) > 1

    # --- precomputed predicates for the (b)/(c) automaton over pieces ---
    # valid[i]: chain i's head can LATCH onto a fresh run made of chain i-1
    # (C++ (b): size equality vs the GROWN piece, strictly forward, u32 step).
    valid = np.zeros(m, dtype=bool)
    if m > 1:
        valid[1:] = (
            (head_size[1:] == piece_size[:-1])
            & (piece_dst[1:] > piece_dst[:-1])
            & (piece_bounce[1:] > piece_bounce[:-1])
            & ((piece_bounce[1:] - piece_bounce[:-1]) <= _U32_MAX)
        )
    delta_dst = np.zeros(m, dtype=np.uint64)
    delta_bounce = np.zeros(m, dtype=np.uint64)
    if m > 1:
        delta_dst[1:] = piece_dst[1:] - piece_dst[:-1]
        delta_bounce[1:] = piece_bounce[1:] - piece_bounce[:-1]
    # cont[i]: chain i EXTENDS (rule (c)) a run whose previous piece is the
    # single-desc chain i-1 and whose latched stride equals delta i-1: same
    # size, same delta -> exact position. Multi chains break blocks (their
    # heads may still join via the stepwise steal path below).
    cont = np.zeros(m, dtype=bool)
    if m > 2:
        cont[2:] = (
            valid[2:]
            & valid[1:-1]
            & (delta_dst[2:] == delta_dst[1:-1])
            & (delta_bounce[2:] == delta_bounce[1:-1])
            & ~multi[2:]
            & ~multi[1:-1]
        )
    cont_false_idx = np.flatnonzero(~cont)
    valid_idx = np.flatnonzero(valid)

    # --- output fragments (concatenated at the end) ---
    frag_bounce: list = []
    frag_dst: list = []
    frag_dstride: list = []
    frag_bstride: list = []
    frag_size: list = []
    frag_count: list = []

    def emit(b: int, d: int, ds: int, bs: int, ps: int, c: int) -> None:
        frag_bounce.append(b)
        frag_dst.append(d)
        frag_dstride.append(ds)
        frag_bstride.append(bs)
        frag_size.append(ps)
        frag_count.append(c)

    def batch_emit_count1(lo: int, hi: int) -> None:
        """Emit chains [lo, hi) as count-1 runs in one vectorized shot."""
        frag_bounce.extend(piece_bounce[lo:hi].tolist())
        frag_dst.extend(piece_dst[lo:hi].tolist())
        frag_dstride.extend([0] * (hi - lo))
        frag_bstride.extend([0] * (hi - lo))
        frag_size.extend(piece_size[lo:hi].tolist())
        frag_count.extend([1] * (hi - lo))

    # Automaton state: the current run start piece. cur_idx is its chain
    # index, or -1 for a SYNTHETIC piece (the remainder of a head-stolen
    # multi chain, logically sitting just before chain `nxt`).
    cur_idx = 0
    cur_b, cur_d, cur_s = int(piece_bounce[0]), int(piece_dst[0]), int(piece_size[0])
    nxt = 1
    while True:
        if nxt >= m:
            emit(cur_b, cur_d, 0, 0, cur_s, 1)
            break
        head_b = int(piece_bounce[nxt])
        head_d = int(piece_dst[nxt])
        can_latch = (
            int(head_size[nxt]) == cur_s
            and head_d > cur_d
            and head_b > cur_b
            and head_b - cur_b <= _U32_MAX
        )
        if not can_latch:
            if cur_idx == nxt - 1:
                # Fast-forward a stretch of un-latchable chains: every chain
                # up to the next valid[] index is its own count-1 run.
                pos = int(np.searchsorted(valid_idx, nxt + 1))
                j = int(valid_idx[pos]) if pos < valid_idx.shape[0] else m
                if j >= m:
                    batch_emit_count1(cur_idx, m)
                    break
                batch_emit_count1(cur_idx, j - 1)
                cur_idx = j - 1
                cur_b = int(piece_bounce[cur_idx])
                cur_d = int(piece_dst[cur_idx])
                cur_s = int(piece_size[cur_idx])
                nxt = j
            else:
                emit(cur_b, cur_d, 0, 0, cur_s, 1)
                cur_idx = nxt
                cur_b, cur_d, cur_s = head_b, head_d, int(piece_size[nxt])
                nxt += 1
            continue

        # Rule (b): latch. Strides come from the FIRST pair.
        dst_stride = head_d - cur_d
        bounce_stride = head_b - cur_b
        if multi[nxt]:
            # Head steal: only the multi chain's head joins; its remainder
            # (still a dense piece) starts the next run. The remainder can
            # never rule-(c) back into this run (that would require a dense
            # boundary between the chains, contradicting the chain split).
            emit(cur_b, cur_d, dst_stride, bounce_stride, cur_s, 2)
            first = int(starts[nxt]) + 1
            cur_idx = -1
            cur_b = int(bounce[first])
            cur_d = int(dst[first])
            cur_s = int(piece_size[nxt]) - int(head_size[nxt])
            nxt += 1
            continue

        # Rule (c): extend through the constant-delta block.
        count = 2
        last = nxt  # last chain joined so far
        if cur_idx == nxt - 1:
            # The latched stride equals delta[nxt]; cont[] chains directly.
            j = _block_end_of(cont_false_idx, nxt + 1, m)
            count += j - nxt
            last = j
        elif (
            nxt + 1 < m
            and not bool(multi[nxt + 1])
            and int(head_size[nxt + 1]) == cur_s
            and int(piece_dst[nxt + 1]) == head_d + dst_stride
            and int(piece_bounce[nxt + 1]) == head_b + bounce_stride
        ):
            # Synthetic run start: the first extension is checked manually
            # (precomputed deltas don't know the synthetic piece), then cont[]
            # blocks take over.
            j = _block_end_of(cont_false_idx, nxt + 2, m)
            count += 1 + (j - (nxt + 1))
            last = j

        # Steal-on-extension: the next chain after the block may be a MULTI
        # chain whose head lands exactly one stride further (C++ (c) compares
        # the raw desc size). The head joins; the remainder starts fresh.
        k = last + 1
        if (
            k < m
            and bool(multi[k])
            and int(head_size[k]) == cur_s
            and int(piece_dst[k]) == cur_d + count * dst_stride
            and int(piece_bounce[k]) == cur_b + count * bounce_stride
        ):
            count += 1
            emit(cur_b, cur_d, dst_stride, bounce_stride, cur_s, count)
            first = int(starts[k]) + 1
            cur_idx = -1
            cur_b = int(bounce[first])
            cur_d = int(dst[first])
            cur_s = int(piece_size[k]) - int(head_size[k])
            nxt = k + 1
            continue

        emit(cur_b, cur_d, dst_stride, bounce_stride, cur_s, count)
        if last + 1 >= m:
            break
        cur_idx = last + 1
        cur_b = int(piece_bounce[cur_idx])
        cur_d = int(piece_dst[cur_idx])
        cur_s = int(piece_size[cur_idx])
        nxt = last + 2

    runs = np.empty(len(frag_size), dtype=SCATTER_RUN_DTYPE)
    runs["bounce_offset"] = frag_bounce
    runs["dst_addr"] = frag_dst
    runs["dst_stride"] = frag_dstride
    runs["bounce_stride"] = frag_bstride
    runs["piece_size"] = frag_size
    runs["count"] = frag_count
    return runs


def build_plan(
    src_ptrs: np.ndarray,
    dst_ptrs: np.ndarray,
    sizes: np.ndarray,
    max_chunk_bytes: int,
    max_descs_per_chunk: int,
    dst_devs: "np.ndarray | int | None" = None,
) -> Plan:
    """Pack (src, dst, size) descriptor triples into bounce chunks.

    Args:
        src_ptrs: [n] uint64 sender-local source addresses.
        dst_ptrs: [n] uint64 receiver-local destination addresses.
        sizes: [n] uint32 per-desc byte counts (zero-length descs are counted
            but never packed, so they cannot force an empty chunk).
        max_chunk_bytes: Per-chunk byte cap (one RDMA write moves at most
            this many bytes; must fit u32 — chunk sizes are 32-bit on the
            wire).
        max_descs_per_chunk: Upper bound on descriptors per chunk (bounds the
            gather-plan size).
        dst_devs: Receiver device id(s): a scalar (uniform), an [n] array (a
            chunk is cut wherever the device changes), or None (device 0).

    Returns:
        The :class:`Plan`; empty input yields an empty plan.

    Raises:
        ValueError: On mismatched array lengths, non-positive
            ``max_chunk_bytes``/``max_descs_per_chunk``, ``max_chunk_bytes``
            above 4 GiB - 1, or a single desc larger than ``max_chunk_bytes``
            (or 4 GiB - 1).
    """
    if max_chunk_bytes <= 0 or max_descs_per_chunk <= 0:
        raise ValueError("build_plan: max_chunk_bytes/max_descs_per_chunk must be > 0")
    if max_chunk_bytes > _U32_MAX:
        raise ValueError(
            f"build_plan: max_chunk_bytes ({max_chunk_bytes}) must be <= 4 GiB - 1 "
            "(chunk size is 32-bit on the wire)"
        )

    src = np.ascontiguousarray(np.asarray(src_ptrs), dtype=np.uint64).reshape(-1)
    dst = np.ascontiguousarray(np.asarray(dst_ptrs), dtype=np.uint64).reshape(-1)
    raw_sizes = np.asarray(sizes).reshape(-1)
    n = int(src.shape[0])
    if dst.shape[0] != n or raw_sizes.shape[0] != n:
        raise ValueError(
            f"build_plan: src/dst/sizes length mismatch "
            f"({n} vs {dst.shape[0]} vs {raw_sizes.shape[0]})"
        )

    plan = Plan(total_descs=n)
    if n == 0:
        return plan

    sz = raw_sizes.astype(np.uint64)
    if bool((sz > _U32_MAX).any()):
        raise ValueError("build_plan: single desc exceeds 4 GiB - 1")
    if bool((sz > max_chunk_bytes).any()):
        worst = int(sz.max())
        raise ValueError(
            f"build_plan: single desc ({worst} B) exceeds max_chunk_bytes ({max_chunk_bytes} B)"
        )

    if dst_devs is None:
        devs = np.zeros(n, dtype=np.uint32)
    else:
        dev_arr = np.asarray(dst_devs, dtype=np.uint32)
        if dev_arr.ndim == 0:
            devs = np.full(n, int(dev_arr), dtype=np.uint32)
        else:
            devs = dev_arr.reshape(-1)
            if devs.shape[0] != n:
                raise ValueError(f"build_plan: dst_devs length mismatch ({devs.shape[0]} vs {n})")

    # Zero-length descs carry no data; drop them so they never force an empty
    # chunk (they are still counted in total_descs, like the C++).
    keep = sz > 0
    if not bool(keep.all()):
        src, dst, sz, devs = src[keep], dst[keep], sz[keep], devs[keep]
        n = int(src.shape[0])
        if n == 0:
            return plan

    plan.total_bytes = int(sz.sum())

    # Aligned exclusive prefix: because every bounce offset is 32-byte
    # aligned, desc i's offset within a chunk starting at desc s is simply
    # aligned_prefix[i] - aligned_prefix[s].
    aligned = (sz + (ALIGNMENT - 1)) // ALIGNMENT * ALIGNMENT
    aligned_prefix = np.concatenate(([0], np.cumsum(aligned, dtype=np.uint64)))
    # extent[i]: packed end of desc i relative to aligned_prefix (monotonic
    # non-decreasing because aligned[i] >= sz[i], so searchsorted applies).
    extent = aligned_prefix[:-1] + sz
    size_prefix = np.concatenate(([0], np.cumsum(sz, dtype=np.uint64)))

    # Device segments: a chunk never spans a dst-device change.
    seg_starts = np.flatnonzero(np.concatenate(([True], devs[1:] != devs[:-1])))
    seg_ends = np.concatenate((seg_starts[1:], [n]))

    for seg_lo, seg_hi in zip(seg_starts.tolist(), seg_ends.tolist()):
        lo = seg_lo
        while lo < seg_hi:
            base = int(aligned_prefix[lo])
            # First desc whose packed end would exceed the chunk cap.
            cut = lo + int(np.searchsorted(extent[lo:seg_hi], base + max_chunk_bytes, side="right"))
            cut = min(cut, lo + max_descs_per_chunk, seg_hi)
            # A single desc always fits (validated above), so cut > lo.
            offsets = aligned_prefix[lo:cut] - np.uint64(base)
            chunk_sizes = sz[lo:cut].astype(np.uint32)
            plan.chunks.append(
                BounceChunk(
                    src_ptrs=src[lo:cut].copy(),
                    dst_ptrs=dst[lo:cut].copy(),
                    sizes=chunk_sizes,
                    bounce_offsets=offsets,
                    scatter_runs=_coalesce_scatter_runs(offsets, dst[lo:cut], sz[lo:cut]),
                    total_bytes=int(size_prefix[cut] - size_prefix[lo]),
                    packed_bytes=int(extent[cut - 1]) - base,
                    dst_device_id=int(devs[lo]),
                )
            )
            lo = cut

    return plan
