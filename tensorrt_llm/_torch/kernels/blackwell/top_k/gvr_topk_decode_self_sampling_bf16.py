# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The BF16-only kernel and helpers of the self-sampling GVR top-K decode.

The device module, gvr_topk_decode_self_sampling.py, imports this module as ``_bf16``; this module
reaches the shared device helpers as ``gvr.<name>``. Names with a leading underscore are used only
in this module, as are RES_B, RES_M, RES_ABOVE and RES_TOT, copies of ``gvr.RES_*``. It holds:

- ``GvrPackedClampWindowKernel``, a register-resident kernel for rows of up to
  ``blk * (4 * vpt + 1)`` values. A sample of the row anchors an ``nbh``-bin window of 16-bit codes,
  every value is clamped into the window histogram, and a crossing scan turns the bins into emit
  cursors. When the K-th value does not land in a single-code bin of the window, the kernel falls
  back to an exact two-pass 8-bit code radix.
- The warp crossing scans that kernel uses for 256- and 512-bin histograms.
- The cluster push merge. ``push_histogram`` sends a CTA's histogram to every peer CTA with st.async
  (GvrClusKernel, GvrRegClusKernel); ``cluster_merge_scan_cross`` merges the cluster's histograms
  into the crossing bin and per-rank emit cursors (GvrRegClusKernel).
- ``refine_small_key_span``, the GvrRegClusKernel refine that ranks the crossing-bin candidates by a
  counting sort when their keys lie within 64 consecutive values.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as dsl_types
from cutlass.cutlass_dsl import T as mlir_types
from cutlass.cutlass_dsl import dsl_user_op

from . import gvr_topk_decode_self_sampling as gvr

# ---- small key-span refine ----


@cute.jit
def refine_small_key_span(
    keys,
    indices,
    histogram,
    warp_minimum,
    warp_maximum,
    output,
    count,
    needed,
    above,
    thread,
):
    """Ranks crossing-bin candidates by a 64-bin counting sort when ``max(key) - min(key) < 64``.

    Candidate ``t < count`` is ``(keys[t], indices[t])`` in shared memory (Uint32 keys). Returns 1
    after writing the candidate of rank ``r < needed`` (larger keys first, ties in any order) to
    ``output[above + r]``, or 0 without writing ``output`` when ``count < 64`` or
    ``max(key) - min(key) >= 64``. Every thread of a 1024-thread CTA must call it with the same
    ``count`` (CTA barriers inside; lane ``l`` of every warp reads the extremes warp ``l`` wrote),
    and ``count`` must not exceed 1024. ``histogram`` (64 words) and the 32-slot
    ``warp_minimum``/``warp_maximum`` are scratch. Used by the GvrRegClusKernel refine.
    """
    completed = cutlass.Int32(0)
    if count >= cutlass.Int32(64):
        lane = thread & cutlass.Int32(31)
        warp = thread >> cutlass.Int32(5)
        candidate_key = cutlass.Uint32(0)
        minimum = cutlass.Uint32(0xFFFFFFFF)
        maximum = cutlass.Uint32(0)
        if thread < count:
            candidate_key = cutlass.Uint32(keys[thread])
            minimum = candidate_key
            maximum = candidate_key
        minimum = cute.arch.warp_redux_sync(minimum, "min")
        maximum = cute.arch.warp_redux_sync(maximum, "max")
        if lane == cutlass.Int32(0):
            warp_minimum[warp] = minimum
            warp_maximum[warp] = maximum
        if thread < cutlass.Int32(64):
            histogram[thread] = cutlass.Int32(0)
        cute.arch.barrier()
        minimum = cutlass.Uint32(warp_minimum[lane])
        maximum = cutlass.Uint32(warp_maximum[lane])
        minimum = cute.arch.warp_redux_sync(minimum, "min")
        maximum = cute.arch.warp_redux_sync(maximum, "max")
        span = maximum - minimum
        if span < cutlass.Uint32(64):
            local_bin = cutlass.Int32(candidate_key - minimum)
            if thread < count:
                cute.arch.atomic_add(
                    histogram.iterator + local_bin,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
            cute.arch.barrier()
            if thread < cutlass.Int32(32):
                first_bin = lane << cutlass.Int32(1)
                lower_count = histogram[first_bin]
                higher_count = histogram[first_bin + cutlass.Int32(1)]
                inclusive = lower_count + higher_count
                for scan_step in cutlass.range_constexpr(5):
                    distance = 1 << scan_step
                    preceding = cute.arch.shuffle_sync_up(inclusive, distance, mask_and_clamp=0)
                    if lane >= cutlass.Int32(distance):
                        inclusive = inclusive + preceding
                total = cute.arch.shuffle_sync(inclusive, cutlass.Int32(31))
                greater = total - inclusive
                histogram[first_bin + cutlass.Int32(1)] = greater
                histogram[first_bin] = greater + higher_count
            cute.arch.barrier()
            if thread < count:
                position = cutlass.Int32(
                    cute.arch.atomic_add(
                        histogram.iterator + local_bin,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                )
                if position < needed:
                    output[above + position] = indices[thread]
            completed = cutlass.Int32(1)
    return completed


# ---- cluster push merge ----

RES_B = 0
RES_M = 1
RES_ABOVE = 2
RES_TOT = 3


@dsl_user_op
def _peer_count(local_address, peer_rank, *, loc=None, ip=None):
    """Returns the 32-bit word at ``local_address`` in cluster CTA ``peer_rank``.

    ``local_address`` is this CTA's shared address, mapped to the peer.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            mlir_types.i32(),
            [
                local_address.ir_value(loc=loc, ip=ip),
                peer_rank.ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b32 peer_address; "
            "mapa.shared::cluster.u32 peer_address, $1, $2; "
            "ld.shared::cluster.u32 $0, [peer_address]; }",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _push_peer_v4(local_address, peer_rank, mbar_address, v0, v1, v2, v3, *, loc=None, ip=None):
    """Stores four 32-bit words at ``local_address`` in cluster CTA ``peer_rank`` with st.async.

    ``local_address`` (16-byte aligned) and ``mbar_address`` are this CTA's shared addresses, mapped
    to the peer; the store completes 16 transaction bytes on the peer's mbarrier.
    """
    llvm.inline_asm(
        None,
        [
            local_address.ir_value(loc=loc, ip=ip),
            peer_rank.ir_value(loc=loc, ip=ip),
            mbar_address.ir_value(loc=loc, ip=ip),
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 peer_address, peer_barrier; "
        "mapa.shared::cluster.u32 peer_address, $0, $1; "
        "mapa.shared::cluster.u32 peer_barrier, $2, $1; "
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[peer_address], {$3, $4, $5, $6}, [peer_barrier]; }",
        "r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def push_histogram(
    histogram,
    receive_address,
    mbar_address,
    rank,
    thread_index,
    nb: cutlass.Constexpr,
    cs: cutlass.Constexpr,
    blk: cutlass.Constexpr,
    packed: cutlass.Constexpr = True,
):
    """Sends this CTA's ``nb``-bin histogram to every other CTA of the cluster with st.async.

    Peers receive it in slot ``rank`` of the receive area at ``receive_address`` (same 16-byte
    aligned shared offset in every CTA): ``nb // 2`` words of two 16-bit counts with ``packed``
    (counts below 65536), else ``nb`` words. The stores complete on the peer's mbarrier at
    ``mbar_address``, armed by the peer for ``(cs - 1) * nb * 2`` bytes (``* 4`` unpacked). Every
    CTA thread calls it, after the CTA barrier that makes ``histogram`` final, once every peer has
    initialized its mbarrier (a cluster wait after the init, such as the one before the first push)
    and no longer reads slot ``rank``. Used by GvrClusKernel and GvrRegClusKernel.
    """
    if cutlass.const_expr(not packed):
        chunks4 = nb // 4
        pairs4 = chunks4 * cs
        for round_index in cutlass.range_constexpr((pairs4 + blk - 1) // blk):
            pair = thread_index + cutlass.Int32(round_index * blk)
            peer = pair >> cutlass.Int32(chunks4.bit_length() - 1)
            chunk = pair & cutlass.Int32(chunks4 - 1)
            if pair < cutlass.Int32(pairs4) and peer != rank:
                base = chunk * cutlass.Int32(4)
                _push_peer_v4(
                    receive_address + (rank * cutlass.Int32(nb) + base) * cutlass.Int32(4),
                    peer,
                    mbar_address,
                    histogram[base],
                    histogram[base + cutlass.Int32(1)],
                    histogram[base + cutlass.Int32(2)],
                    histogram[base + cutlass.Int32(3)],
                )
        return
    chunks = nb // 8
    chunk_shift = chunks.bit_length() - 1
    pairs = chunks * cs
    for round_index in cutlass.range_constexpr((pairs + blk - 1) // blk):
        pair = thread_index + cutlass.Int32(round_index * blk)
        peer = pair >> cutlass.Int32(chunk_shift)
        chunk = pair & cutlass.Int32(chunks - 1)
        if pair < cutlass.Int32(pairs) and peer != rank:
            base = chunk * cutlass.Int32(8)
            words = []
            for word in cutlass.range_constexpr(4):
                words.append(
                    histogram[base + cutlass.Int32(2 * word)]
                    | (histogram[base + cutlass.Int32(2 * word + 1)] << cutlass.Int32(16))
                )
            _push_peer_v4(
                receive_address
                + (rank * cutlass.Int32(nb // 2) + chunk * cutlass.Int32(4)) * cutlass.Int32(4),
                peer,
                mbar_address,
                words[0],
                words[1],
                words[2],
                words[3],
            )


@cute.jit
def _warp_inclusive_add(value, lane):
    """Returns the inclusive prefix sum of ``value`` over lanes ``0 .. lane`` of the warp."""
    for scan_step in cutlass.range_constexpr(5):
        scan_offset = 1 << scan_step
        lower = cute.arch.shuffle_sync_up(value, scan_offset, mask_and_clamp=0)
        if lane >= cutlass.Int32(scan_offset):
            value = value + lower
    return value


@cute.jit
def cluster_merge_scan_cross(
    histogram,
    cursors,
    warp_sums,
    rank,
    target,
    thread_index,
    result,
    nb: cutlass.Constexpr,
    cs: cutlass.Constexpr,
    receive=None,
):
    """Merges the cluster's ``nb``-bin histograms into the crossing bin and this rank's cursors.

    Thread ``b < nb`` sums bin ``b`` over the ``cs`` ranks, from ``receive`` (packed counts sent by
    ``push_histogram``) or over DSMEM (after a releasing cluster barrier), and sets ``cursors[b]``
    to the count in higher bins plus bin ``b`` of lower ranks. The crossing bin (bin 0 if the total
    is below ``target``) writes RES_B, RES_M and RES_ABOVE; thread 0 writes RES_TOT.
    GvrRegClusKernel only: needs at least ``nb`` (512 or 1024) threads and ``nb // 32``
    ``warp_sums`` words. One CTA barrier inside and none after, so barrier before reading
    ``cursors`` or ``result``. With ``receive``, call after the mbarrier wait that lands every
    peer's push.
    """
    assert nb in (512, 1024)
    assert cs in (2, 4, 8, 16)
    lane = thread_index & cutlass.Int32(31)
    warp_index = thread_index >> cutlass.Int32(5)
    count = cutlass.Int32(0)
    rank_prefix = cutlass.Int32(0)
    within_warp = cutlass.Int32(0)
    if thread_index < cutlass.Int32(nb):
        if cutlass.const_expr(receive is not None):
            count = histogram[thread_index]
            for peer_rank in cutlass.range_constexpr(cs):
                peer_count = cutlass.Int32(0)
                if cutlass.Int32(peer_rank) != rank:
                    peer_count = (
                        receive[
                            cutlass.Int32(peer_rank * nb // 2) + (thread_index >> cutlass.Int32(1))
                        ]
                        >> ((thread_index & cutlass.Int32(1)) << cutlass.Int32(4))
                    ) & cutlass.Int32(0xFFFF)
                count = count + peer_count
                if cutlass.Int32(peer_rank) < rank:
                    rank_prefix = rank_prefix + peer_count
        else:
            local_address = histogram.iterator.toint() + thread_index * cutlass.Int32(4)
            peer_counts = []
            for peer_rank in cutlass.range_constexpr(cs):
                peer_counts.append(_peer_count(local_address, cutlass.Int32(peer_rank)))
            for peer_rank in cutlass.range_constexpr(cs):
                count = count + peer_counts[peer_rank]
                if cutlass.Int32(peer_rank) < rank:
                    rank_prefix = rank_prefix + peer_counts[peer_rank]
        within_warp = _warp_inclusive_add(count, lane)
        if lane == cutlass.Int32(31):
            warp_sums[warp_index] = within_warp
    cute.arch.barrier()
    if thread_index < cutlass.Int32(nb):
        warp_mass = cutlass.Int32(0)
        if lane < cutlass.Int32(nb // 32):
            warp_mass = warp_sums[lane]
        total = cutlass.Int32(cute.arch.warp_redux_sync(warp_mass, "add"))
        earlier_mass = cutlass.Int32(0)
        if lane < warp_index:
            earlier_mass = warp_mass
        lower_warps = cutlass.Int32(cute.arch.warp_redux_sync(earlier_mass, "add"))
        above = total - (lower_warps + within_warp)
        cursors[thread_index] = above + rank_prefix
        if thread_index == cutlass.Int32(0):
            result[RES_TOT] = total
        if above < target:
            if above + count >= target or thread_index == cutlass.Int32(0):
                result[RES_B] = thread_index
                result[RES_M] = count
                result[RES_ABOVE] = above


# ---- warp crossing scans ----


@cute.jit
def _scan_cross_warp(
    histogram,
    target,
    thread_index,
    result,
    nb: cutlass.Constexpr,
    vector: cutlass.Constexpr = True,
):
    """Crossing scan of an ``nb``-bin (256 or 512) Uint32 histogram by warp 0; no barrier inside.

    Lane ``l`` owns bins ``[l * nb // 32, (l + 1) * nb // 32)``. Each bin is replaced by its emit
    cursor, the count in higher bins. The crossing bin, the highest bin at which the count from the
    top reaches ``target`` (bin 0 if the total is below it), writes RES_B, RES_M and RES_ABOVE;
    lane 0 writes RES_TOT. ``vector`` moves each lane's bins with 16-byte shared accesses.
    """
    assert nb in (256, 512)
    if thread_index < cutlass.Int32(32):
        lane = cutlass.Int32(thread_index)
        bins_per_lane = nb // 32
        bin_base = lane * cutlass.Int32(bins_per_lane)
        target_count = cutlass.Uint32(target)
        counts = cute.make_rmem_tensor((bins_per_lane,), cutlass.Uint32)
        if cutlass.const_expr(vector):
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
            )
            histogram_span = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    histogram.iterator.toint() + bin_base * cutlass.Int32(4),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout((bins_per_lane,)),
            )
            cute.copy(copy_atom, histogram_span, counts)
        else:
            for local_bin in cutlass.range_constexpr(bins_per_lane):
                counts[local_bin] = cutlass.Uint32(histogram[bin_base + cutlass.Int32(local_bin)])
        lane_mass = cutlass.Uint32(0)
        for local_bin in cutlass.range_constexpr(bins_per_lane):
            lane_mass = lane_mass + cutlass.Uint32(counts[local_bin])
        lane_suffix = lane_mass
        for scan_step in cutlass.range_constexpr(5):
            scan_offset = 1 << scan_step
            higher_mass = cutlass.Uint32(
                cute.arch.shuffle_sync_down(lane_suffix, scan_offset, mask_and_clamp=31)
            )
            if lane + cutlass.Int32(scan_offset) < cutlass.Int32(32):
                lane_suffix = lane_suffix + higher_mass
        total_mass = cutlass.Uint32(cute.arch.shuffle_sync(lane_suffix, 0))
        if lane == cutlass.Int32(0):
            result[RES_TOT] = total_mass
        above = lane_suffix - lane_mass
        for local_bin in cutlass.range_constexpr(bins_per_lane - 1, -1, -1):
            bin_index = bin_base + cutlass.Int32(local_bin)
            mass = cutlass.Uint32(counts[local_bin])
            through = above + mass
            if cutlass.const_expr(vector):
                counts[local_bin] = above
            else:
                histogram[bin_index] = above
            if above < target_count:
                if through >= target_count or bin_index == cutlass.Int32(0):
                    result[RES_B] = cutlass.Uint32(bin_index)
                    result[RES_M] = mass
                    result[RES_ABOVE] = above
            above = through
        if cutlass.const_expr(vector):
            cute.copy(copy_atom, counts, histogram_span)


@cute.jit
def _scan_cross_warp_parts(
    histogram,
    target,
    thread,
    result,
    part,
    nb: cutlass.Constexpr,
    parts: cutlass.Constexpr,
    vector: cutlass.Constexpr = True,
):
    """Crossing scan by warp 0 of ``parts`` back-to-back ``nb``-bin sub-histograms.

    The crossing bin (bin 0 if the total is below ``target``), RES_B, RES_M, RES_ABOVE and RES_TOT
    come from the per-bin sums over all parts. The cursors (count in higher bins plus this bin's
    counts in parts ``0 .. part - 1``) are written into sub-histogram 0, so the ``parts`` CTAs of a
    row emit each bin into disjoint output ranges. Unlike the block-wide ``gvr.scan_cross_w_parts``,
    only warp 0 runs and there is no barrier.
    """
    assert nb in (256, 512) and parts in (1, 2, 4)
    if thread < cutlass.Int32(32):
        lane = cutlass.Int32(thread)
        lane_bins = nb // 32
        bin_base = lane * cutlass.Int32(lane_bins)
        target_count = cutlass.Uint32(target)
        counts = cute.make_rmem_tensor((lane_bins,), cutlass.Uint32)
        prefix = cute.make_rmem_tensor((lane_bins,), cutlass.Uint32)
        values = cute.make_rmem_tensor((lane_bins,), cutlass.Uint32)
        for local in cutlass.range_constexpr(lane_bins):
            counts[local] = cutlass.Uint32(0)
            prefix[local] = cutlass.Uint32(0)
        if cutlass.const_expr(vector):
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
            )
        for source in cutlass.range_constexpr(parts):
            offset = bin_base + cutlass.Int32(source * nb)
            if cutlass.const_expr(vector):
                span = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        histogram.iterator.toint() + offset * cutlass.Int32(4),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((lane_bins,)),
                )
                cute.copy(copy_atom, span, values)
            else:
                for local in cutlass.range_constexpr(lane_bins):
                    values[local] = cutlass.Uint32(histogram[offset + cutlass.Int32(local)])
            for local in cutlass.range_constexpr(lane_bins):
                mass = cutlass.Uint32(values[local])
                counts[local] = cutlass.Uint32(counts[local]) + mass
                if cutlass.Int32(source) < cutlass.Int32(part):
                    prefix[local] = cutlass.Uint32(prefix[local]) + mass
        lane_mass = cutlass.Uint32(0)
        for local in cutlass.range_constexpr(lane_bins):
            lane_mass = lane_mass + cutlass.Uint32(counts[local])
        suffix = lane_mass
        for scan_step in cutlass.range_constexpr(5):
            scan_offset = 1 << scan_step
            higher = cutlass.Uint32(
                cute.arch.shuffle_sync_down(suffix, scan_offset, mask_and_clamp=31)
            )
            if lane + cutlass.Int32(scan_offset) < cutlass.Int32(32):
                suffix = suffix + higher
        total_mass = cutlass.Uint32(cute.arch.shuffle_sync(suffix, 0))
        if lane == cutlass.Int32(0):
            result[RES_TOT] = total_mass
        above = suffix - lane_mass
        for local in cutlass.range_constexpr(lane_bins - 1, -1, -1):
            bin_index = bin_base + cutlass.Int32(local)
            mass = cutlass.Uint32(counts[local])
            through = above + mass
            cursor = above + cutlass.Uint32(prefix[local])
            if cutlass.const_expr(vector):
                counts[local] = cursor
            else:
                histogram[bin_index] = cursor
            if above < target_count:
                if through >= target_count or bin_index == cutlass.Int32(0):
                    result[RES_B] = cutlass.Uint32(bin_index)
                    result[RES_M] = mass
                    result[RES_ABOVE] = above
            above = through
        if cutlass.const_expr(vector):
            span = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    histogram.iterator.toint() + bin_base * cutlass.Int32(4),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout((lane_bins,)),
            )
            cute.copy(copy_atom, counts, span)


# ---- packed clamp window kernel ----


@dsl_user_op
def _clear_nan_signs(packed_pair: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Clears the sign bit of each NaN half of a packed BF16 pair; other halves are unchanged.

    NaNs then count as non-negative and their ordered keys rank above +inf, as torch.topk ranks NaN.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 nan_mask;\n\t"
            "set.nan.u32.bf16x2 nan_mask, $1, $1;\n\t"
            "lop3.b32 $0, $1, nan_mask, 0x80008000, 0x70;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sign_quad(
    lower_pair: cutlass.Uint32, upper_pair: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Returns the sign bits of the four BF16 values in two packed pairs.

    Value ``i`` is in bit 7 of byte ``i``; the other bits are 0.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [lower_pair.ir_value(loc=loc, ip=ip), upper_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs;\n\t"
            "prmt.b32 signs, $1, $2, 0x7531;\n\t"
            "and.b32 $0, signs, 0x80808080;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _positive_quad_counts(
    lower_pair: cutlass.Uint32, upper_pair: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Returns one byte per BF16 value of two packed pairs: 1 when its sign bit is clear, else 0.

    Callers add these words across vectors (each byte must stay below 256) and total the bytes with
    ``_sum_positive_counts``.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [lower_pair.ir_value(loc=loc, ip=ip), upper_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs;\n\t"
            "prmt.b32 signs, $1, $2, 0xfdb9;\n\t"
            "lop3.b32 $0, signs, 0x01010101, 0, 0x0c;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sum_positive_counts(packed_counts: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Int32:
    """Returns the sum of the four unsigned bytes of ``packed_counts``."""
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_counts.ir_value(loc=loc, ip=ip)],
            "dp4a.u32.u32 $0, $1, 0x01010101, 0;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ordered_key_pair(packed_pair: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Returns the ordered 16-bit keys of both halves of a packed 16-bit float pair.

    Negative halves flip all bits and the others flip the sign bit, so keys compare as unsigned in
    value order. The all-ones padding word maps to key 0, below every non-NaN value.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 sign_mask;\n\t"
            "prmt.b32 sign_mask, $1, 0, 0xbb99;\n\t"
            "lop3.b32 $0, $1, 0x80008000, sign_mask, 0x1e;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _packed_sample_accumulate(
    accumulator: cutlass.Uint32,
    ordered_pair: cutlass.Uint32,
    maximum: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Returns the per-half unsigned minimum of two packed 16-bit key pairs.

    With ``maximum``, returns the per-half unsigned maximum instead.
    """
    instruction = "min.u16x2 $0, $1, $2;"
    if maximum:
        instruction = "max.u16x2 $0, $1, $2;"
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [accumulator.ir_value(loc=loc, ip=ip), ordered_pair.ir_value(loc=loc, ip=ip)],
            instruction,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _packed_sample_collapse(
    accumulator: cutlass.Uint32,
    maximum: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Returns the minimum of the two 16-bit halves of ``accumulator``.

    With ``maximum``, returns their maximum instead.
    """
    instruction = "min.u16x2 pair, $1, other;\n\t"
    if maximum:
        instruction = "max.u16x2 pair, $1, other;\n\t"
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [accumulator.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 other, pair;\n\t"
            "prmt.b32 other, $1, 0, 0x1032;\n\t" + instruction + "and.b32 $0, pair, 0xffff;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half_code(
    packed_pair: cutlass.Uint32,
    upper: bool,
    signed: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Returns the ``upper`` or lower 16-bit half of ``packed_pair``.

    The half is zero-extended, or sign-extended with ``signed``.
    """
    if signed:
        selector = "0xbb32" if upper else "0x9910"
    else:
        selector = "0x4432" if upper else "0x4410"
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            f"prmt.b32 $0, $1, 0, {selector};",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_output_quad(
    destination_address: cutlass.Int64,
    first_index: cutlass.Int32,
    second_index: cutlass.Int32,
    third_index: cutlass.Int32,
    fourth_index: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Stores four Int32 values to global memory at the 16-byte aligned ``destination_address``."""
    llvm.inline_asm(
        res=None,
        operands_=[
            destination_address.ir_value(loc=loc, ip=ip),
            first_index.ir_value(loc=loc, ip=ip),
            second_index.ir_value(loc=loc, ip=ip),
            third_index.ir_value(loc=loc, ip=ip),
            fourth_index.ir_value(loc=loc, ip=ip),
        ],
        asm_string="st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        constraints="l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _copy_output_quad(source_address, destination_address, vector_index):
    """Copies the ``vector_index``-th group of four Int32 indices from shared to global memory.

    ``source_address`` (shared) and ``destination_address`` (global) are 16-byte aligned bases.
    """
    byte_offset = vector_index * cutlass.Int32(16)
    first_index, second_index, third_index, fourth_index = gvr._lds_v4_u32(
        source_address + byte_offset
    )
    _store_output_quad(
        destination_address + cutlass.Int64(byte_offset),
        first_index,
        second_index,
        third_index,
        fourth_index,
    )


@cute.jit
def _code_radix_cursor(key, crossing_high, threshold, high_address, low_address, tie_address):
    """Returns ``key``'s output slot in the two-pass 8-bit code radix.

    The result is 0x40000000 when ``key`` is below ``threshold``. The slot is an atomic increment of
    the cursor of the key's high byte, of its low byte when the high byte is ``crossing_high``, or
    of the tie counter at ``tie_address`` when it equals ``threshold``.
    """
    high_byte = key >> cutlass.Uint32(8)
    cursor_address = high_address + cutlass.Int32(high_byte << cutlass.Uint32(2))
    if high_byte == crossing_high:
        cursor_address = low_address + cutlass.Int32(
            (key & cutlass.Uint32(255)) << cutlass.Uint32(2)
        )
    if key == threshold:
        cursor_address = tie_address
    return gvr._atom_shared_add1_if_ge(cursor_address, key, threshold)


@dsl_user_op
def _packed_clamp_bins(
    packed_pair: cutlass.Uint32,
    flip: cutlass.Uint32,
    lower: cutlass.Uint32,
    upper: cutlass.Uint32,
    pivot: cutlass.Uint32,
    positive: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Returns the window bins of both halves of a packed BF16 pair, still packed.

    Each code is clamped to this lane's edges ``[lower, upper]`` and offset by ``pivot``, all
    replicated per half (see ``_window_controls``). The negative window first inverts the codes
    (``flip`` is all ones); the ``positive`` window clamps as signed 16-bit, so negative values land
    on the lower edge.
    """
    operands = (packed_pair, flip, lower, upper, pivot)
    instructions = (
        "xor.b32 code_pair, $1, $2;\n\t"
        "max.u16x2 code_pair, code_pair, $3;\n\t"
        "min.u16x2 code_pair, code_pair, $4;\n\t"
        "sub.u32 $0, code_pair, $5;\n}"
    )
    constraints = "=r,r,r,r,r,r"
    if positive:
        operands = (packed_pair, lower, upper, pivot)
        instructions = (
            "max.s16x2 code_pair, $1, $2;\n\t"
            "min.s16x2 code_pair, code_pair, $3;\n\t"
            "sub.u32 $0, code_pair, $4;\n}"
        )
        constraints = "=r,r,r,r,r"
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [value.ir_value(loc=loc, ip=ip) for value in operands],
            "{\n\t.reg .b32 code_pair;\n\t" + instructions,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _packed_clamp_cursor(
    histogram_address: cutlass.Int32,
    bin_index: cutlass.Int32,
    crossing_bin: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Returns window bin ``bin_index``'s next output slot, or 0x40000000 below ``crossing_bin``.

    The slot is an atomic increment of the bin's cursor in the histogram at ``histogram_address``.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [
                value.ir_value(loc=loc, ip=ip)
                for value in (histogram_address, bin_index, crossing_bin)
            ],
            "{\n\t.reg .b32 address;\n\t.reg .pred selected;\n\t"
            "setp.ge.u32 selected, $2, $3;\n\t"
            "mad.lo.u32 address, $2, 4, $1;\n\t"
            "mov.u32 $0, 0x40000000;\n\t"
            "@selected atom.relaxed.cta.shared.add.u32 $0, [address], 1;\n}",
            "=r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class GvrPackedClampWindowKernel:
    """Register-resident BF16 top-K decode kernel that selects through a sampled histogram window.

    Each row is served by ``parts`` CTAs of ``blk`` threads. Every CTA loads the whole row into
    registers as 8-byte vectors of four values (``vpt`` per thread) plus at most one tail value per
    thread, so a row holds at most ``blk * (4 * vpt + 1)`` values. NaN signs are cleared so NaNs
    rank above +inf, as in torch.topk.

    A sample anchors an ``nbh``-bin window of 16-bit codes. The sample is the first K values, or,
    when ``blk * 4 * vpt >= 16384``, the last value of every vector held by warp ``blk // 128``.
    With at least K values whose sign bit is clear, the window's top single-code bin (``nbh - 33``)
    holds the sample maximum unless the window base is clamped to the code range; with fewer, the
    window starts one code below the sample minimum. Clamp edges are staggered by lane, so
    out-of-window codes spread over 32 edge bins on each side and only bins ``[32, nbh - 32)`` hold
    a single code.

    When the crossing bin is one of these single-code bins (for a negative K-th value, also a
    negative code), each CTA emits its share through the bin cursors: ``parts`` splits only the
    emit, part ``p`` taking the ``p``-th of ``parts`` equal groups of each thread's ``4 * vpt``
    value slots (the tail goes with the last part). Otherwise part 0 reloads the row and runs an
    exact two-pass 8-bit code radix over the ordered keys. The selected set is exact, with ties at
    the K-th value taken in any order; the output order is unspecified. Part 0 writes rows of at
    most K values as ``0 .. len - 1`` followed by -1.

    Shared memory: a ``gvr.STATIC_WORDS``-word prelude (result slots in words 0-5, the code-radix
    tie counter in word 7, warp sums in words 16-47, sample extremes from word 48), then
    ``max(512, parts * nbh)`` histogram words (512 holds the code radix's two 256-bin histograms),
    then K words of staged output with ``stage_output``.

    Args:
        blk: Threads per CTA: 256, 512 or 1024.
        vpt: 8-byte vectors per thread, 1 to 10.
        nbh: Window bins: 256, 512, 1024 or 2048.
        next_n: Query rows per request; with ``varlen``, row ``r`` belongs to request
            ``r // next_n``.
        cr_shift: Right shift from a token count to a row length (log2 of the compression ratio).
        minb: Minimum resident CTAs per SM for the launch bounds; ``minb * blk <= 2048``.
        varlen: Row length ``(kv_lens[r // next_n] - next_n + r % next_n + 1) >> cr_shift``, clamped
            to ``[0, n]``. Without it every row has length ``n``.
        parts: CTAs per row: 1, 2, 4 or 8; must divide ``4 * vpt``.
        stage_output: Stage the output row in shared memory and copy it to ``out`` with 16-byte
            stores where aligned. Requires ``parts == 1``.
        packed_sample_bf16: Take the first-K sample extremes with packed 16-bit min/max. Requires
            ``K % 4 == 0`` and ``blk * 4 * vpt < 16384``.
        dtype: Must be ``cutlass.BFloat16``.
        cluster: With ``parts > 1``, launch each row's CTAs as one cluster (co-scheduling only; the
            kernel uses no cluster memory).
    """

    def __init__(
        self,
        blk: int,
        vpt: int,
        nbh: int = 1024,
        next_n: int = 1,
        cr_shift: int = 0,
        *,
        minb: int = 1,
        varlen: bool = True,
        parts: int = 1,
        stage_output: bool = False,
        packed_sample_bf16: bool = False,
        dtype=cutlass.BFloat16,
        cluster: bool = False,
    ) -> None:
        assert dtype == cutlass.BFloat16
        assert blk in (256, 512, 1024) and 1 <= vpt <= 10
        assert nbh in (256, 512, 1024, 2048)
        assert next_n >= 1 and 0 <= cr_shift < 31
        assert minb >= 1 and minb * blk <= 2048
        assert parts in (1, 2, 4, 8)
        assert not (stage_output and parts > 1)
        if packed_sample_bf16:
            assert blk * 4 * vpt < 16384
        self.blk = int(blk)
        self.vpt = int(vpt)
        self.nbh = int(nbh)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        self.minb = int(minb)
        self.varlen = bool(varlen)
        self.parts = int(parts)
        self.part_shift = self.parts.bit_length() - 1
        self.stage_output = bool(stage_output)
        self.cluster = bool(cluster) and self.parts > 1
        self.packed_sample_bf16 = bool(packed_sample_bf16)
        self.resident_sample = self.blk * 4 * self.vpt >= 16384
        self.sample_warp = self.blk // 128
        self.dtype = dtype
        self.capacity = self.blk * (4 * self.vpt + 1)
        self.histogram_words = max(512, self.parts * self.nbh)
        self.output_offset_words = gvr.STATIC_WORDS + self.histogram_words
        self.required_smem_bytes = self.output_offset_words * 4

    def _histogram_offset(self, scalar_fragment: int) -> int:
        """Returns the byte offset of the sub-histogram that counts scalar slot ``scalar_fragment``.

        Slot ``4 * f + i`` is value ``i`` of fragment ``f``; the ``4 * vpt`` slots split into
        ``parts`` equal contiguous groups, so ``parts`` must divide ``4 * vpt``.
        """
        return (scalar_fragment * self.parts // (4 * self.vpt)) * self.nbh * 4

    @cute.jit
    def _window_controls(self, window_base, thread_index, positive: cutlass.Constexpr):
        """Returns ``(flip, lower, upper, pivot)`` for ``_packed_clamp_bins``.

        The window base code (the inverted raw code for the negative window) is clamped to
        [0, 32768 - nbh] for the positive window and [0, 65536 - nbh] for the negative one, so all
        ``nbh`` bins stay in the 16-bit code range. ``lower``, ``upper`` and ``pivot`` repeat in
        both halves; this lane's clamp edges are ``base + lane`` and ``base + nbh - 32 + lane``,
        which spreads out-of-window codes over 32 edge bins per side.
        """
        code_base = window_base
        flip = cutlass.Uint32(0x80008000)
        maximum_base = cutlass.Int32(32768 - self.nbh)
        if cutlass.const_expr(not positive):
            code_base = cutlass.Int32(0xFFFF) - window_base
            flip = cutlass.Uint32(0xFFFFFFFF)
            maximum_base = cutlass.Int32(65536 - self.nbh)
        if code_base < cutlass.Int32(0):
            code_base = cutlass.Int32(0)
        if code_base > maximum_base:
            code_base = maximum_base
        lane_index = thread_index & cutlass.Int32(31)
        pivot = cutlass.Uint32(code_base) * cutlass.Uint32(65537)
        lower = cutlass.Uint32(code_base + lane_index) * cutlass.Uint32(65537)
        upper = cutlass.Uint32(
            code_base + cutlass.Int32(self.nbh - 32) + lane_index
        ) * cutlass.Uint32(65537)
        return flip, lower, upper, pivot

    @cute.jit
    def _fill_window_histogram(
        self,
        packed_fragments,
        raw_tail,
        histogram_address,
        window_base,
        thread_index,
        body_vectors,
        tail_count,
        positive: cutlass.Constexpr,
        full: cutlass.Constexpr = False,
    ):
        """Replaces valid ``packed_fragments`` pairs by their window bins and counts every value.

        Values count into their part's sub-histogram (``_histogram_offset``), the tail value into
        the last part's. Returns the tail value's bin, or 0 for threads without one.
        """
        flip, lower, upper, pivot = self._window_controls(window_base, thread_index, positive)
        for fragment_index in cutlass.range_constexpr(self.vpt):
            vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
            if cutlass.const_expr(full) or vector_index < body_vectors:
                for pair_index in cutlass.range_constexpr(2):
                    bins = _packed_clamp_bins(
                        packed_fragments[fragment_index][pair_index],
                        flip,
                        lower,
                        upper,
                        pivot,
                        positive,
                    )
                    packed_fragments[fragment_index][pair_index] = bins
                    for half_index in cutlass.range_constexpr(2):
                        bin_index = _half_code(bins, half_index == 1, False)
                        gvr._red_shared_add1__reg(
                            histogram_address
                            + cutlass.Int32(
                                self._histogram_offset(
                                    fragment_index * 4 + pair_index * 2 + half_index
                                )
                            )
                            + bin_index * cutlass.Int32(4)
                        )
        tail_bin = cutlass.Int32(0)
        if thread_index < tail_count:
            tail_bins = _packed_clamp_bins(raw_tail, flip, lower, upper, pivot, positive)
            tail_bin = _half_code(tail_bins, False, False)
            gvr._red_shared_add1__reg(
                histogram_address
                + cutlass.Int32((self.parts - 1) * self.nbh * 4)
                + tail_bin * cutlass.Int32(4)
            )
        return tail_bin

    @cute.jit
    def _emit_window(
        self,
        packed_fragments,
        tail_bin,
        output_row,
        thread_index,
        part_index,
        tail_index,
        histogram_address,
        window_base,
        crossing_bin,
        top_k,
        body_vectors,
        tail_count,
        positive: cutlass.Constexpr,
        full: cutlass.Constexpr = False,
    ):
        """Emits this CTA's share of the selected values through the window cursors.

        The share is value-slot group ``part_index`` of every thread, plus the tail value in the
        last part. A value whose bin is at or above ``crossing_bin`` takes an output slot from its
        bin's cursor and writes its row index to ``output_row`` when the slot is below ``top_k``.
        """
        for group_index in cutlass.range_constexpr(self.parts):
            if part_index == cutlass.Int32(group_index):
                for scalar_fragment in cutlass.range_constexpr(
                    group_index * 4 * self.vpt // self.parts,
                    (group_index + 1) * 4 * self.vpt // self.parts,
                ):
                    vector_index = thread_index + cutlass.Int32((scalar_fragment // 4) * self.blk)
                    if cutlass.const_expr(full) or vector_index < body_vectors:
                        bins = packed_fragments[scalar_fragment // 4][(scalar_fragment % 4) // 2]
                        bin_index = _half_code(bins, scalar_fragment % 2 == 1, False)
                        output_position = _packed_clamp_cursor(
                            histogram_address, bin_index, crossing_bin
                        )
                        if output_position < top_k:
                            selected_index = vector_index * cutlass.Int32(4) + cutlass.Int32(
                                scalar_fragment % 4
                            )
                            output_row[output_position] = selected_index
                if cutlass.const_expr(group_index == self.parts - 1):
                    if thread_index < tail_count:
                        output_position = _packed_clamp_cursor(
                            histogram_address, tail_bin, crossing_bin
                        )
                        if output_position < top_k:
                            output_row[output_position] = tail_index

    @cute.jit
    def _row_body(
        self,
        packed_fragments,
        envelope_raw_tail,
        envelope_body_vectors,
        row_address,
        shared_base,
        shared_histogram,
        histogram_address,
        output_row,
        thread_index,
        part_index,
        lane_index,
        warp_index,
        top_k,
        row_length,
        body_vectors,
        full: cutlass.Constexpr,
    ):
        """Selects the top-K of one row of ``row_length > top_k`` values in ``packed_fragments``.

        Clears NaN signs, counts values with a clear sign bit and takes the sample extremes, then
        fills and scans the window histogram and emits through its cursors when the crossing bin
        holds a single code; otherwise part 0 reloads the row and runs the two-pass 8-bit code
        radix. ``full`` (the row fills every vector) drops the per-vector bounds checks.
        """
        tail_begin = body_vectors << cutlass.Int32(2)
        tail_count = row_length - tail_begin
        tail_index = tail_begin + thread_index
        local_minimum = cutlass.Uint32(0xFFFF)
        local_maximum = cutlass.Uint32(0)
        positive_local = cutlass.Int32(0)
        if cutlass.const_expr(not self.resident_sample):
            packed_positive_count = cutlass.Uint32(0)
        if cutlass.const_expr(self.packed_sample_bf16):
            sample_vectors = top_k >> cutlass.Int32(2)
            packed_sample_minimum = cutlass.Uint32(0xFFFFFFFF)
            packed_sample_maximum = cutlass.Uint32(0)
        for fragment_index in cutlass.range_constexpr(self.vpt):
            vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
            if cutlass.const_expr(full) or vector_index < body_vectors:
                packed_fragments[fragment_index][0] = _clear_nan_signs(
                    packed_fragments[fragment_index][0]
                )
                packed_fragments[fragment_index][1] = _clear_nan_signs(
                    packed_fragments[fragment_index][1]
                )
                if cutlass.const_expr(not self.resident_sample):
                    packed_positive_count = packed_positive_count + _positive_quad_counts(
                        packed_fragments[fragment_index][0], packed_fragments[fragment_index][1]
                    )
                else:
                    positive_local = (
                        positive_local
                        + cutlass.Int32(4)
                        - cutlass.Int32(
                            cute.arch.popc(
                                cutlass.Int32(
                                    _sign_quad(
                                        packed_fragments[fragment_index][0],
                                        packed_fragments[fragment_index][1],
                                    )
                                )
                            )
                        )
                    )
                if cutlass.const_expr(self.packed_sample_bf16):
                    if vector_index < sample_vectors:
                        for pair_index in cutlass.range_constexpr(2):
                            sample_pair = _ordered_key_pair(
                                packed_fragments[fragment_index][pair_index]
                            )
                            packed_sample_minimum = _packed_sample_accumulate(
                                packed_sample_minimum, sample_pair, maximum=False
                            )
                            packed_sample_maximum = _packed_sample_accumulate(
                                packed_sample_maximum, sample_pair, maximum=True
                            )
                elif cutlass.const_expr(not self.resident_sample):
                    for pair_index in cutlass.range_constexpr(2):
                        raw_pair = packed_fragments[fragment_index][pair_index]
                        sample_begin = vector_index * cutlass.Int32(4) + cutlass.Int32(
                            pair_index * 2
                        )
                        if sample_begin < top_k:
                            sample_pair = _ordered_key_pair(raw_pair)
                            for half_index in cutlass.range_constexpr(2):
                                sample_index = sample_begin + cutlass.Int32(half_index)
                                if sample_index < top_k:
                                    sample_key = cutlass.Uint32(
                                        _half_code(sample_pair, half_index == 1, False)
                                    )
                                    if sample_key < local_minimum:
                                        local_minimum = sample_key
                                    if sample_key > local_maximum:
                                        local_maximum = sample_key
            else:
                packed_fragments[fragment_index][0] = cutlass.Uint32(0xFFFFFFFF)
                packed_fragments[fragment_index][1] = cutlass.Uint32(0xFFFFFFFF)
        if cutlass.const_expr(not self.resident_sample):
            positive_local = _sum_positive_counts(packed_positive_count)
        if cutlass.const_expr(self.packed_sample_bf16):
            if thread_index < sample_vectors:
                local_minimum = _packed_sample_collapse(packed_sample_minimum, maximum=False)
                local_maximum = _packed_sample_collapse(packed_sample_maximum, maximum=True)
        raw_tail = cutlass.Uint32(0xFFFF)
        if thread_index < tail_count:
            if body_vectors == envelope_body_vectors:
                raw_tail = envelope_raw_tail
            else:
                raw_tail = gvr._ld_g_nc_u16(
                    row_address + cutlass.Int64(tail_index) * cutlass.Int64(2)
                )
            raw_tail = _clear_nan_signs(raw_tail)
            positive_local = positive_local + cutlass.Int32(
                (raw_tail & cutlass.Uint32(0x8000)) == cutlass.Uint32(0)
            )
            if cutlass.const_expr(not self.resident_sample):
                if tail_index < top_k:
                    sample_key = cutlass.Uint32(gvr.fkey16_of_code16(raw_tail))
                    if sample_key < local_minimum:
                        local_minimum = sample_key
                    if sample_key > local_maximum:
                        local_maximum = sample_key

        shared_result = gvr._smem_view(cutlass.Int32, shared_base, 0, 6)
        shared_count = gvr._smem_view(cutlass.Int32, shared_base, 6, 2, align=4)
        shared_work = gvr._smem_view(cutlass.Int32, shared_base, 16, 32)
        positive_local = cutlass.Int32(cute.arch.warp_redux_sync(positive_local, "add"))
        if lane_index == cutlass.Int32(0):
            shared_work[warp_index] = positive_local
        if cutlass.const_expr(self.resident_sample):
            shared_samples = gvr._smem_view(cutlass.Uint32, shared_base, 48, 3)
            if warp_index == cutlass.Int32(self.sample_warp):
                local_minimum = cutlass.Uint32(0xFFFF)
                local_maximum = cutlass.Uint32(0)
                for fragment_index in cutlass.range_constexpr(self.vpt):
                    vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                    if cutlass.const_expr(full) or vector_index < body_vectors:
                        sample_pair = _ordered_key_pair(packed_fragments[fragment_index][1])
                        sample_key = cutlass.Uint32(_half_code(sample_pair, True, False))
                        if sample_key < local_minimum:
                            local_minimum = sample_key
                        if sample_key > local_maximum:
                            local_maximum = sample_key
                local_minimum = cute.arch.warp_redux_sync(local_minimum, "min")
                local_maximum = cute.arch.warp_redux_sync(local_maximum, "max")
                if lane_index == cutlass.Int32(0):
                    shared_samples[0] = local_minimum
                    shared_samples[1] = local_maximum
            cute.arch.barrier()
            sample_minimum = cutlass.Uint32(shared_samples[0])
            sample_maximum = cutlass.Uint32(shared_samples[1])
            positive_count = cutlass.Int32(0)
            if lane_index < cutlass.Int32(self.blk // 32):
                positive_count = shared_work[lane_index]
            positive_count = cutlass.Int32(cute.arch.warp_redux_sync(positive_count, "add"))
            negative_cut = positive_count < top_k
        else:
            shared_minimum = gvr._smem_view(cutlass.Uint32, shared_base, 48, 32)
            shared_maximum = gvr._smem_view(cutlass.Uint32, shared_base, 80, 32)
            local_minimum = cute.arch.warp_redux_sync(local_minimum, "min")
            local_maximum = cute.arch.warp_redux_sync(local_maximum, "max")
            if lane_index == cutlass.Int32(0):
                shared_minimum[warp_index] = local_minimum
                shared_maximum[warp_index] = local_maximum
            cute.arch.barrier()
            sample_minimum = cutlass.Uint32(0xFFFF)
            sample_maximum = cutlass.Uint32(0)
            positive_count = cutlass.Int32(0)
            if lane_index < cutlass.Int32(self.blk // 32):
                sample_minimum = cutlass.Uint32(shared_minimum[lane_index])
                sample_maximum = cutlass.Uint32(shared_maximum[lane_index])
                positive_count = cutlass.Int32(shared_work[lane_index])
            sample_minimum = cute.arch.warp_redux_sync(sample_minimum, "min")
            sample_maximum = cute.arch.warp_redux_sync(sample_maximum, "max")
            positive_count = cutlass.Int32(cute.arch.warp_redux_sync(positive_count, "add"))
            negative_cut = positive_count < top_k
        window_base = (
            cutlass.Int32(sample_maximum) - cutlass.Int32(0x8000) - cutlass.Int32(self.nbh - 33)
        )
        if negative_cut:
            window_base = cutlass.Int32(sample_minimum ^ cutlass.Uint32(0xFFFF)) + cutlass.Int32(1)
        elif window_base < cutlass.Int32(0):
            window_base = cutlass.Int32(0)
        tail_bin = cutlass.Int32(0)
        if negative_cut:
            tail_bin = self._fill_window_histogram(
                packed_fragments,
                raw_tail,
                histogram_address,
                window_base,
                thread_index,
                body_vectors,
                tail_count,
                positive=False,
                full=full,
            )
        else:
            tail_bin = self._fill_window_histogram(
                packed_fragments,
                raw_tail,
                histogram_address,
                window_base,
                thread_index,
                body_vectors,
                tail_count,
                positive=True,
                full=full,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.nbh < 512 and self.parts in (2, 4) and self.nbh >= self.blk):
            _scan_cross_warp_parts(
                shared_histogram,
                top_k,
                thread_index,
                shared_result,
                part_index,
                nb=self.nbh,
                parts=self.parts,
            )
        elif cutlass.const_expr(self.nbh <= 512 and self.parts == 1):
            _scan_cross_warp(
                shared_histogram,
                top_k,
                thread_index,
                shared_result,
                nb=self.nbh,
            )
        elif cutlass.const_expr(self.parts > 1):
            gvr.scan_cross_w_parts(
                shared_histogram,
                shared_work,
                top_k,
                thread_index,
                shared_result,
                part_index,
                blk=self.blk,
                nb=self.nbh,
                parts=self.parts,
            )
        else:
            gvr.scan_cross_w(
                shared_histogram,
                shared_work,
                top_k,
                thread_index,
                shared_result,
                blk=self.blk,
                nb=self.nbh,
            )
        cute.arch.barrier()
        crossing_bin = shared_result[gvr.RES_B]
        if (
            shared_result[gvr.RES_TOT] >= top_k
            and shared_result[gvr.RES_M] > cutlass.Int32(0)
            and crossing_bin >= cutlass.Int32(32)
            and crossing_bin < cutlass.Int32(self.nbh - 32)
            and (not negative_cut or window_base - crossing_bin >= cutlass.Int32(0x8000))
        ):
            if negative_cut:
                self._emit_window(
                    packed_fragments,
                    tail_bin,
                    output_row,
                    thread_index,
                    part_index,
                    tail_index,
                    histogram_address,
                    window_base,
                    crossing_bin,
                    top_k,
                    body_vectors,
                    tail_count,
                    positive=False,
                    full=full,
                )
            else:
                self._emit_window(
                    packed_fragments,
                    tail_bin,
                    output_row,
                    thread_index,
                    part_index,
                    tail_index,
                    histogram_address,
                    window_base,
                    crossing_bin,
                    top_k,
                    body_vectors,
                    tail_count,
                    positive=True,
                    full=full,
                )
        elif part_index == cutlass.Int32(0):
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if cutlass.const_expr(full) or vector_index < body_vectors:
                    fallback_lower, fallback_upper = gvr._ld_g_nc_v2_b32(
                        row_address + cutlass.Int64(vector_index) * cutlass.Int64(8)
                    )
                    packed_fragments[fragment_index][0] = _clear_nan_signs(fallback_lower)
                    packed_fragments[fragment_index][1] = _clear_nan_signs(fallback_upper)
            for fragment_index in cutlass.range_constexpr(self.vpt):
                for pair_index in cutlass.range_constexpr(2):
                    packed_fragments[fragment_index][pair_index] = _ordered_key_pair(
                        packed_fragments[fragment_index][pair_index]
                    )
            tail_key = cutlass.Uint32(0)
            if thread_index < tail_count:
                tail_key = cutlass.Uint32(gvr.fkey16_of_code16(raw_tail))
            shared_low = gvr._smem_view(cutlass.Int32, shared_base, gvr.STATIC_WORDS + 256, 256)
            low_address = histogram_address + cutlass.Int32(1024)
            for histogram_batch in cutlass.range_constexpr((512 + self.blk - 1) // self.blk):
                histogram_index = thread_index + cutlass.Int32(histogram_batch * self.blk)
                if histogram_index < cutlass.Int32(512):
                    shared_histogram[histogram_index] = cutlass.Int32(0)
            cute.arch.barrier()
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if cutlass.const_expr(full) or vector_index < body_vectors:
                    for pair_index in cutlass.range_constexpr(2):
                        key_pair = packed_fragments[fragment_index][pair_index]
                        for half_index in cutlass.range_constexpr(2):
                            key = cutlass.Uint32(_half_code(key_pair, half_index == 1, False))
                            gvr._red_shared_add1__reg(
                                histogram_address
                                + cutlass.Int32((key >> cutlass.Uint32(8)) << cutlass.Uint32(2))
                            )
            if thread_index < tail_count:
                gvr._red_shared_add1__reg(
                    histogram_address
                    + cutlass.Int32((tail_key >> cutlass.Uint32(8)) << cutlass.Uint32(2))
                )
            cute.arch.barrier()
            gvr.find_cross256_cursors(
                histogram_address,
                top_k,
                thread_index,
                shared_result,
                cutlass.Int32(0),
                shared_count,
                shared_low,
                cutlass.Int32(0),
                cancel_pad=False,
            )
            cute.arch.barrier()
            crossing_high = cutlass.Uint32(shared_result[gvr.RES_B])
            above_high = shared_result[gvr.RES_ABOVE]
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if cutlass.const_expr(full) or vector_index < body_vectors:
                    for pair_index in cutlass.range_constexpr(2):
                        key_pair = packed_fragments[fragment_index][pair_index]
                        for half_index in cutlass.range_constexpr(2):
                            key = cutlass.Uint32(_half_code(key_pair, half_index == 1, False))
                            gvr._red_shared_add1_if_eq(
                                low_address
                                + cutlass.Int32((key & cutlass.Uint32(255)) << cutlass.Uint32(2)),
                                key >> cutlass.Uint32(8),
                                crossing_high,
                            )
            if thread_index < tail_count:
                gvr._red_shared_add1_if_eq(
                    low_address
                    + cutlass.Int32((tail_key & cutlass.Uint32(255)) << cutlass.Uint32(2)),
                    tail_key >> cutlass.Uint32(8),
                    crossing_high,
                )
            cute.arch.barrier()
            gvr.find_cross256_cursors(
                low_address,
                top_k - above_high,
                thread_index,
                shared_result,
                above_high,
                shared_count,
                shared_low,
                cutlass.Int32(0),
                cancel_pad=False,
            )
            cute.arch.barrier()
            threshold = (crossing_high << cutlass.Uint32(8)) | cutlass.Uint32(
                shared_result[gvr.RES_B]
            )
            tie_address = shared_base + cutlass.Int32(7 * 4)
            for fragment_index in cutlass.range_constexpr(self.vpt):
                for pair_index in cutlass.range_constexpr(2):
                    key_pair = packed_fragments[fragment_index][pair_index]
                    for half_index in cutlass.range_constexpr(2):
                        key = cutlass.Uint32(_half_code(key_pair, half_index == 1, False))
                        output_position = _code_radix_cursor(
                            key,
                            crossing_high,
                            threshold,
                            histogram_address,
                            low_address,
                            tie_address,
                        )
                        if output_position < top_k:
                            selected_index = (
                                thread_index + cutlass.Int32(fragment_index * self.blk)
                            ) * cutlass.Int32(4) + cutlass.Int32(pair_index * 2 + half_index)
                            output_row[output_position] = selected_index
            output_position = _code_radix_cursor(
                tail_key, crossing_high, threshold, histogram_address, low_address, tie_address
            )
            if output_position < top_k:
                output_row[output_position] = tail_index

    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        envelope: cutlass.Int32,
    ):
        """Device body: CTA ``b`` serves part ``b & (parts - 1)`` of row ``b >> log2(parts)``.

        Loads up to ``envelope`` values of the row into registers, clears the histograms and
        computes the row length. Part 0 writes rows of at most K values directly; longer rows run
        ``_row_body``. With ``stage_output`` the staged row is copied to ``out`` at the end.
        Arguments as in ``__call__`` (``envelope`` is ``n``).
        """
        thread_index, _, _ = cute.arch.thread_idx()
        block_index, _, _ = cute.arch.block_idx()
        row_index = block_index >> cutlass.Int32(self.part_shift)
        part_index = block_index & cutlass.Int32(self.parts - 1)
        lane_index = thread_index & cutlass.Int32(31)
        warp_index = thread_index >> cutlass.Int32(5)
        top_k = cutlass.Int32(pre_idx.shape[1])
        shared_pointer = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
        shared_base = shared_pointer.toint()
        if cutlass.const_expr(self.stage_output):
            output_row = gvr._smem_view(
                cutlass.Int32, shared_base, self.output_offset_words, pre_idx.shape[1]
            )
        else:
            output_row = out[row_index, None]
        envelope_body_vectors = envelope >> cutlass.Int32(2)
        if envelope_body_vectors > cutlass.Int32(self.blk * self.vpt):
            envelope_body_vectors = cutlass.Int32(self.blk * self.vpt)
        envelope_tail_begin = envelope_body_vectors << cutlass.Int32(2)
        envelope_tail_count = envelope - envelope_tail_begin
        envelope_tail_index = envelope_tail_begin + thread_index
        row_address = logits[row_index, None].iterator.toint()
        packed_fragments = [cute.make_rmem_tensor((2,), cutlass.Uint32) for _ in range(self.vpt)]
        for fragment_index in cutlass.range_constexpr(self.vpt):
            packed_fragments[fragment_index][0] = cutlass.Uint32(0xFFFFFFFF)
            packed_fragments[fragment_index][1] = cutlass.Uint32(0xFFFFFFFF)
            vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
            if vector_index < envelope_body_vectors:
                lower_word, upper_word = gvr._ld_g_nc_v2_b32(
                    row_address + cutlass.Int64(vector_index) * cutlass.Int64(8)
                )
                packed_fragments[fragment_index][0] = lower_word
                packed_fragments[fragment_index][1] = upper_word
        envelope_raw_tail = cutlass.Uint32(0xFFFF)
        if thread_index < envelope_tail_count:
            envelope_raw_tail = gvr._ld_g_nc_u16(
                row_address + cutlass.Int64(envelope_tail_index) * cutlass.Int64(2)
            )
        shared_histogram = gvr._smem_view(
            cutlass.Int32, shared_base, gvr.STATIC_WORDS, self.histogram_words
        )
        histogram_address = gvr._smem_addr_reg__reg(
            shared_base + cutlass.Int32(gvr.STATIC_WORDS * 4)
        )
        for clear_batch in cutlass.range_constexpr(
            (self.parts * self.nbh // 4 + self.blk - 1) // self.blk
        ):
            clear_index = thread_index + cutlass.Int32(clear_batch * self.blk)
            if cutlass.const_expr((clear_batch + 1) * self.blk <= self.parts * self.nbh // 4):
                gvr._sts_v4_u32(
                    histogram_address + clear_index * cutlass.Int32(16),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                )
            elif clear_index < cutlass.Int32(self.parts * self.nbh // 4):
                gvr._sts_v4_u32(
                    histogram_address + clear_index * cutlass.Int32(16),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                )
        row_length = envelope
        if cutlass.const_expr(self.varlen):
            request_index = row_index // cutlass.Int32(self.next_n)
            query_position = row_index % cutlass.Int32(self.next_n)
            row_length = (
                kv_lens[request_index]
                - cutlass.Int32(self.next_n)
                + query_position
                + cutlass.Int32(1)
            ) >> cutlass.Int32(self.cr_shift)
        if row_length < cutlass.Int32(0):
            row_length = cutlass.Int32(0)
        if row_length > envelope:
            row_length = envelope
        if row_length <= top_k:
            if part_index == cutlass.Int32(0):
                output_index = thread_index
                while output_index < top_k:
                    selected_index = cutlass.Int32(-1)
                    if output_index < row_length:
                        selected_index = output_index
                    output_row[output_index] = selected_index
                    output_index = output_index + cutlass.Int32(self.blk)
        else:
            body_vectors = row_length >> cutlass.Int32(2)
            if body_vectors > cutlass.Int32(self.blk * self.vpt):
                body_vectors = cutlass.Int32(self.blk * self.vpt)
            if body_vectors == cutlass.Int32(self.blk * self.vpt):
                self._row_body(
                    packed_fragments,
                    envelope_raw_tail,
                    envelope_body_vectors,
                    row_address,
                    shared_base,
                    shared_histogram,
                    histogram_address,
                    output_row,
                    thread_index,
                    part_index,
                    lane_index,
                    warp_index,
                    top_k,
                    row_length,
                    body_vectors,
                    full=True,
                )
            else:
                self._row_body(
                    packed_fragments,
                    envelope_raw_tail,
                    envelope_body_vectors,
                    row_address,
                    shared_base,
                    shared_histogram,
                    histogram_address,
                    output_row,
                    thread_index,
                    part_index,
                    lane_index,
                    warp_index,
                    top_k,
                    row_length,
                    body_vectors,
                    full=False,
                )

        if cutlass.const_expr(self.stage_output):
            cute.arch.barrier()
            source_address = gvr._smem_addr_reg__reg(
                shared_base + cutlass.Int32(self.output_offset_words * 4)
            )
            destination_address = out[row_index, None].iterator.toint()
            scalar_begin = cutlass.Int32(0)
            if (destination_address & cutlass.Int64(15)) == cutlass.Int64(0) and out.stride[1] == 1:
                vector_count = top_k >> cutlass.Int32(2)
                if cutlass.const_expr(cute.is_static(pre_idx.shape[1])):
                    for copy_step in cutlass.range_constexpr(
                        (pre_idx.shape[1] // 4 + self.blk - 1) // self.blk
                    ):
                        vector_index = thread_index + cutlass.Int32(copy_step * self.blk)
                        if vector_index < vector_count:
                            _copy_output_quad(source_address, destination_address, vector_index)
                else:
                    vector_index = thread_index
                    while vector_index < vector_count:
                        _copy_output_quad(source_address, destination_address, vector_index)
                        vector_index = vector_index + cutlass.Int32(self.blk)
                scalar_begin = vector_count << cutlass.Int32(2)
            output_index = scalar_begin + thread_index
            while output_index < top_k:
                out[row_index, output_index] = output_row[output_index]
                output_index = output_index + cutlass.Int32(self.blk)

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        cmp_: cutlass.Int32,
        qc: cutlass.Int32,
        smem_bytes: cutlass.Int32,
        stream,
    ):
        """Launches ``rows * parts`` CTAs on ``stream``, as clusters of ``parts`` with ``cluster``.

        Args:
            logits: [rows, npad] BFloat16, row-major with 8-byte aligned rows; ``npad % 64 == 0`` is
                asserted when static.
            pre_idx: [rows, K] Int32. Only its width K is read; ``K % 4 == 0`` with
                ``packed_sample_bf16``.
            kv_lens: [requests] Int32 token counts, read only with ``varlen``.
            out: [rows, >= K] Int32 selected indices.
            n: Envelope: upper bound on every row length, at most ``blk * (4 * vpt + 1)``.
            cmp_: Not read; keeps the register-family launch signature.
            qc: Not read; keeps the register-family launch signature.
            smem_bytes: Not read; the launch uses ``required_smem_bytes``, plus K words with
                ``stage_output``.
            stream: CUDA stream.
        """
        assert logits.element_type == cutlass.BFloat16
        assert pre_idx.element_type == cutlass.Int32
        assert kv_lens.element_type == cutlass.Int32
        assert out.element_type == cutlass.Int32
        if cutlass.const_expr(cute.is_static(logits.shape[1])):
            assert logits.shape[1] % 64 == 0
        if cutlass.const_expr(self.packed_sample_bf16):
            assert pre_idx.shape[1] % 4 == 0
        shared_bytes = self.required_smem_bytes
        if cutlass.const_expr(self.stage_output):
            shared_bytes = shared_bytes + pre_idx.shape[1] * 4
        if cutlass.const_expr(self.cluster):
            self.kern(logits, pre_idx, kv_lens, out, n).launch(
                grid=(logits.shape[0] * self.parts, 1, 1),
                block=(self.blk, 1, 1),
                cluster=(self.parts, 1, 1),
                stream=stream,
                smem=shared_bytes,
                min_blocks_per_mp=self.minb,
            )
        else:
            self.kern(logits, pre_idx, kv_lens, out, n).launch(
                grid=(logits.shape[0] * self.parts, 1, 1),
                block=(self.blk, 1, 1),
                stream=stream,
                smem=shared_bytes,
                min_blocks_per_mp=self.minb,
            )


__all__ = [
    "GvrPackedClampWindowKernel",
    "cluster_merge_scan_cross",
    "push_histogram",
    "refine_small_key_span",
]
