# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin (cc 10.7) routed experts for TP16 verify steps above 64 token rows with each
expert's weights streamed once per ``width`` tokens (``RubinWideExpertsKernel``'s
capacity machinery; its 8-token units re-stream an expert per 8 tokens, 1.3-1.8x the
weight bytes at 256-512 rows).

- A **stream** is (expert, up to ``width`` tokens): k <= width / 8 consecutive 8-token
  **sub-units** of one expert. The partition, the producer, die matching and the FC2
  pool run over streams (an expert's weights once per stream); everything after the
  MMAs (accumulator reads, FC1 K-split partials, hidden images, staged FC2 rows) stays
  per sub-unit, as in the 64-row kernel.
- Each stage's MMAs take all k sub-units at once: N = 8k from the run-time instruction
  descriptor, the B operand (an FC1 stage's activation slice, an FC2 job's hidden
  images) as k 8-row 128B-swizzled images 2 KiB apart (descriptor SBO 2 KiB), its
  scale rows n at TMEM lane n (k <= 4: the 32x128b copies' rows), the accumulator a
  ``width``-column "big slot" (min(16, 384 / width) of them in columns 0-383; the
  weight, hidden and activation scale columns from 384).
- The two epilogue groups take every other big slot (the 64-row kernel's accumulator
  order): the owner reads the slot's k sub-units in order, releases it after the last
  read and handles each sub-unit as the 64-row kernel handles a unit.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int16, Int32, Int64, Uint32, Uint64

from ..common.primitives import (
    asm,
    bits_float,
    commit,
    copy_g2s,
    fence_tmem,
    float_bits,
    load_tmem,
    load_volatile,
    shared_address,
    smem_desc,
    store_shared_words,
    store_words,
    thread,
    view,
)
from .experts import PLAN_XSF, has_ff_byte, swizzled
from .experts_rubin import DESCRIPTOR_LIMIT
from .experts_wide import RubinWideExpertsKernel


@asm()
def stage_vw(mma, weight_scales, activation_scales, release, done, n8, *, kind, half):
    """``experts_rubin.stage16`` (packed K64 MMAs, whole stage) with N = 8 (``n8`` + 1)
    in the instruction descriptors' N field and the activation scales' second K128
    half ``half`` 16 B units past the first."""
    steps = 6 if kind == "fc2" else 8
    ks = [k for k in range(steps) if k % 2 == 0]
    cp = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    arrive = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = [
        "{ .reg .pred e,p0,pt,pc,pa,ph; .reg .b32 i<4>,nw; .reg .b64 as1,bs1;",
        f".reg .b32 af1,bf1; .reg .b64 xa<{steps}>,xb<{steps}>;",
        "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0;",
        "and.pred pc,pc,e; setp.ne.b32 pa,$10,0; and.pred pa,pa,e; shl.b32 nw,$11,17;",
    ]
    if kind == "fc1":
        code.append("setp.ne.b32 p0,$8,0;")
    else:
        code.append("setp.eq.b32 p0,0,1; setp.ne.b32 ph,$8,0; and.pred ph,ph,e;")
        code.append(f"@ph {arrive} [$8];")
    k64 = 1 << 31
    for k in range(4):
        code.append(f"mov.b32 i{k},{142738048 | k64 | k << 29 | k << 4:#x};")
        code.append(f"add.u32 i{k},i{k},nw;")
    code.append(f"add.s64 as1,$4,32; add.s64 bs1,$6,{half};")
    code.append("add.u32 af1,$3,4; add.u32 bf1,$5,4;")
    code += [f"@pc {cp} [$5],$6;", f"@pc {cp} [bf1],bs1;"]
    code += [f"@e {cp} [$3],$4;", f"@e {cp} [af1],as1;"]
    for k in ks:
        code.append(
            f"add.s64 xa{k},$1,{k}; add.s64 xb{k},$2,{k // 4 * 64 + k % 4 * 2};"
        )
    for k in ks:
        af, bf = ("$3", "$5") if k < 4 else ("af1", "bf1")
        code.append(
            "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            f"[$0],xa{k},xb{k},i{k % 4},[{af}],[{bf}],{'p0' if k == 0 else 'pt'};"
        )
    code += [f"@e {arrive} [$9];", f"@pa {arrive} [$10];", "}"]
    kinds = "rllrlrlrrrrr"
    operands = (*mma, *weight_scales, *activation_scales, *release, done, n8)
    values = [(Uint64 if c == "l" else Uint32)(v) for c, v in zip(kinds, operands)]
    return "\n".join(code), ",".join(kinds) + ",~{memory}", values


@cute.jit
def smem_desc_rows(pointer, sbo: cutlass.Constexpr[int]):
    """``primitives.smem_desc`` (128B-swizzled K-major) with stride byte offset
    ``sbo`` (16 B units) between 8-row groups."""
    address = shared_address(pointer)
    value = Uint64((address & 262143) >> 4) | Uint64(1) << 46
    value |= Uint64(1) << 16 | Uint64(sbo) << 32 | Uint64(2) << 61
    return value


class RubinVWExpertsKernel(RubinWideExpertsKernel):
    """``RubinWideExpertsKernel`` with streams of up to ``width`` (16 or 32) tokens."""

    def __init__(
        self, experts, grid, slots, capacity, width=32, publish=False, stages=None,
        trace=0, pool=(12, 6), dies=(), static_groups=False, fc2_base=1,
    ):  # fmt: skip
        assert width in (8, 16, 32)  # 8: the V2 machinery on 8-token units (a control)
        if stages is None:  # a 7th stage fits shared memory up to width 16
            stages = 7 if width <= 16 else 6
        # FC2 ranks split the jobs by weight, not count: a job weighs fc2_base + k (its
        # w2 tile, then the epilogue's k sub-units; at k = 2 the epilogue, not the
        # tile's bytes, bounds the job).
        self.fc2_base = fc2_base
        super().__init__(
            experts, grid, slots, capacity, publish, stages, trace, pool, dies,
            static_groups, tables=False,
        )  # fmt: skip
        self.wide, self.subs = (width, width // 8)
        # Accumulator big slots of ``width`` columns (one barrier pair each), then the
        # weight scales (8 columns a slot), the hidden scales (8 a buffer) and the
        # activation scales (8 a slot) from column 384.
        self.big = min(16, 384 // width)
        assert self.big >= 8  # ``fc2_mma``'s eight-ahead acquire
        self.col_w = 384
        self.col_h = self.col_w + 8 * stages
        self.col_x = self.col_h + 2 * self.w2_scale_columns
        assert self.col_x + 8 * stages <= 512
        # Streams: sum ceil(n_e / width) <= 16 capacity / width + experts.
        self.streams = (16 * capacity // width + experts + 31) // 32 * 32
        units = self.units  # sub-units (``RubinWideExpertsKernel``'s bound)
        self.xslot, self.hbuf = (self.subs * 2048, self.subs * self.h_image_bytes)
        self.xsf_stride = 32 * width  # [K128 half, width rows, 16 B]
        pad = max(0, 512 - 16 * width)  # the last half's 32-row copy reads past it
        self.hsf_stride = 32 * width + pad
        self.off_sfa = stages * self.slot_bytes
        self.off_x = self.off_sfa + stages * 1024
        self.off_h = self.off_x + stages * self.xslot
        self.off_xsf = self.off_h + 2 * self.hbuf
        self.off_hsf = self.off_xsf + stages * self.xsf_stride + pad
        self.off_misc = self.off_hsf + 2 * self.hsf_stride
        assert self.off_misc <= DESCRIPTOR_LIMIT
        self.off_meta = self.off_misc + 2560
        self.off_queue = self.off_misc + 4608
        self.off_tables = (self.off_misc + 5120 + 127) // 128 * 128
        self.off_sub0 = self.off_tables + 4 * self.streams
        self.off_rows = (self.off_sub0 + 4 * self.streams + 4 + 15) // 16 * 16
        # ``fc2_bounds``' owner table (Int16) in the activation slots while grouping.
        self.owners = fc2_base * self.streams + units
        end = self.off_rows + 16 * units
        # without the spill: bases, sbases, totals, bounds, then the transpose buffers
        whole = end + 8 * experts + 64 + 8 * (grid + 1)
        whole = (whole + 127) // 128 * 128 + 8 * 512
        if self.grouping_spill(stages, whole):  # grouping-only arrays below off_misc
            low = (2 * self.owners, 4 * experts, 4 * experts, 64)
            self.off_owner, self.off_bases, self.off_sbases, self.off_totals = (
                self.low_scratch(low)
            )
        else:
            self.off_bases = end
            self.off_sbases = self.off_bases + 4 * experts
            self.off_totals = self.off_sbases + 4 * experts
            self.off_owner = self.off_x
            assert 2 * self.owners <= stages * self.xslot
            end = self.off_totals + 64
        # FC2(A) / FC2(B) rank boundaries (``fc2_bounds``): [2, grid + 1] jobs.
        self.off_bounds = end
        end = self.off_bounds + 8 * (grid + 1)
        self.off_fc2s = (end + 127) // 128 * 128  # ``fc2``'s transpose buffers
        end = self.off_fc2s + 8 * 512
        # The token masks, then their running counts, in the weight ring while grouping
        # (below off_misc past the ring: ``grouping_spill``).
        self.off_masks = 0
        self.off_prefix = 4 * self.mask_stride * experts
        self.check_smem(end)
        # FC1 partials: one plane per sub-unit of a stream (plane 0: ``ws_partial``).
        self.plane_bytes = 2 * slots * 1024 * 4
        self.ws_planes = (self.workspace_bytes + 255) // 256 * 256
        self.workspace_bytes = self.ws_planes + (self.subs - 1) * self.plane_bytes
        # ``fc2_bounds``' rank x weight products stay in Int32.
        assert grid * 28 * (fc2_base * self.streams + units) < 2**31

    def tables(self, sm, window=True):
        """Streams' experts, then the sub-units' eight Int16 pair rows (as words)."""
        experts = view(sm.iterator, self.off_tables, Int32, self.streams)
        rows = view(sm.iterator, self.off_rows, Int16, 8 * self.units)
        return experts, rows, view(sm.iterator, self.off_rows, Uint32, 4 * self.units)

    @cute.jit
    def sub_starts(self, sm):
        """Each stream's first sub-unit; entry ``streams`` = the sub-unit count."""
        return view(sm.iterator, self.off_sub0, Int32, self.streams + 1)

    @cute.jit
    def group(self, sm, ids, workspace, batch: Int32, scalars=None, arrival=None):
        """``RubinWideExpertsKernel.group`` with streams: expert e's tokens fill
        ceil(n / 8) sub-units (rows as before) and ceil(n / width) streams of up to
        width / 8 of them. Misc scalars: (streams, call, grouped call); the sub-unit
        count past the streams' starts."""
        tid, warp, lane, cta = thread()
        words: cutlass.Constexpr = self.words
        stride: cutlass.Constexpr = self.mask_stride
        per: cutlass.Constexpr = self.subs
        masks = view(sm.iterator, self.off_masks, Uint32, stride * self.experts)
        base = view(sm.iterator, self.off_bases, Int32, self.experts)
        sbase = view(sm.iterator, self.off_sbases, Int32, self.experts)
        totals = view(sm.iterator, self.off_totals, Int32, self.threads // 32)
        prefix = view(sm.iterator, self.off_prefix, Int16, stride * self.experts)
        owner = view(sm.iterator, self.off_owner, Int16, self.owners)
        experts, rows, _ = self.tables(sm)
        sub0 = self.sub_starts(sm)
        scalars = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        threads: cutlass.Constexpr = self.threads
        rounds: cutlass.Constexpr = (16 * self.capacity + threads - 1) // threads
        mine = cute.make_rmem_tensor(rounds, Int32)
        ws, slots = (workspace.iterator, self.slots)
        calls = view(ws, self.ws_counter, Int32, slots)[cta]
        grouped = view(ws, self.ws_calls, Int32, slots)[cta]
        for k in cutlass.range_constexpr(rounds):
            mine[k] = Int32(0)
            if tid + threads * k < batch * 16:
                mine[k] = ids[tid + threads * k]
        self.clear(sm)  # while the ids are in flight
        if tid == 0:
            scalars[1], scalars[2] = (calls, grouped)
        cute.arch.sync_threads()
        if tid == 0:
            self.stamp(15)
        for k in cutlass.range_constexpr(rounds):
            pair = tid + threads * k
            if pair < batch * 16:
                token = pair >> 4
                bit = Uint32(1) << (token & 31)
                word = masks.iterator + stride * mine[k] + (token >> 5)
                cute.arch.atomic_or(word, bit, sem="relaxed", scope="cta")
        cute.arch.sync_threads()
        if tid == 0:
            self.stamp(16)
        span: cutlass.Constexpr = (self.experts + threads - 1) // threads
        counts = cute.make_rmem_tensor(span, Int32)  # streams << 16 | sub-units
        count = Int32(0)
        for j in cutlass.range_constexpr(span):
            counts[j], e = (Int32(0), span * tid + j)
            if e < self.experts:
                n = Int32(0)
                for w in cutlass.range_constexpr(words):
                    prefix[stride * e + w] = Int16(n)
                    n += Int32(cute.arch.popc(masks[stride * e + w]))
                counts[j] = (n + self.wide - 1) // self.wide << 16 | (n + 7) >> 3
            count += counts[j]
        total = count
        for shift in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(total, 1 << shift)
            if lane >= 1 << shift:
                total += other
        if lane == 31:
            totals[warp] = total
        cute.arch.sync_threads()
        if tid == 0:
            self.stamp(17)
        start = total - count
        for w in cutlass.range_constexpr(threads // 32 - 1):
            if warp > w:
                start += totals[w]
        if tid == threads - 1:
            whole = start + count
            scalars[0] = whole >> 16
            sub0[whole >> 16] = whole & 65535
        for j in cutlass.range_constexpr(span):
            e = span * tid + j
            if e < self.experts:
                first, stream = (start & 65535, start >> 16)
                base[e], sbase[e] = (first, stream)
                total_subs = counts[j] & 65535
                for v in cutlass.range(counts[j] >> 16, unroll=1):
                    experts[stream + v] = e
                    sub0[stream + v] = first + v * per
                    # ``fc2_bounds``' owner table: entries fc2_base x + sub0[x]
                    # .. + fc2_base + k - 1 name stream x.
                    sized = total_subs - v * per
                    if sized > per:
                        sized = Int32(per)
                    at = self.fc2_base * (stream + v) + first + v * per
                    for i in cutlass.range(self.fc2_base + sized, unroll=1):
                        owner[at + i] = Int16(stream + v)
            start += counts[j]
        if cutlass.const_expr(len(self.dies) > 0):
            self.arrive(sm, workspace, arrival)
        cute.arch.sync_threads()
        if tid == 0:
            self.stamp(18)
        self.fc2_bounds(sm, scalars[0])
        cute.arch.sync_threads()
        if tid == 0:
            self.stamp(20)
        if cutlass.const_expr(len(self.dies) > 0):
            self.home_bits(sm, experts, scalars[0])
            cute.arch.sync_threads()
            if warp == 0:
                self.matched(sm)
            cute.arch.sync_threads()
        for k in cutlass.range_constexpr(rounds):
            pair = tid + threads * k
            if pair < batch * 16:
                e, token = (mine[k], pair >> 4)
                at = stride * e + (token >> 5)
                below = (Uint32(1) << (token & 31)) - 1
                rank = Int32(prefix[at]) + Int32(cute.arch.popc(masks[at] & below))
                rows[(base[e] + (rank >> 3)) * 8 + (rank & 7)] = Int16(pair)
        if tid == 0:
            self.stamp(19)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

    @cute.jit
    def fc2_bounds(self, sm, streams):
        """Threads r <= grid: rank r's first FC2(A) / FC2(B) job. ``partition``'s
        stream halves, each split over the ranks by job weight (fc2_base + k): rank r
        starts at the first job whose weight prefix reaches r / grid of its half's.
        A stream x's 28 jobs span weights 28 (fc2_base x + sub0[x]) on, so the owner
        table (``group``: one entry per 28 weight) names the stream of any prefix."""
        tid = thread()[0]
        sub0 = self.sub_starts(sm)
        owner = view(sm.iterator, self.off_owner, Int16, self.owners)
        bounds = view(sm.iterator, self.off_bounds, Int32, 2 * self.grid + 2)
        base: cutlass.Constexpr = self.fc2_base
        if tid <= self.grid:
            middle = streams
            if streams > 40:
                middle = (streams * 50 + 99) // 100
            for g in cutlass.range_constexpr(2):
                start, end = (Int32(0), middle) if g == 0 else (middle, streams)
                origin = base * start + sub0[start]
                total = 28 * (base * end + sub0[end] - origin)
                w = tid * total // self.grid
                job = 28 * end
                if w < total:
                    x = Int32(owner[origin + w // 28])
                    prefix = 28 * (base * x + sub0[x] - origin)
                    weight = base + sub0[x + 1] - sub0[x]
                    job = 28 * x + (w - prefix + weight - 1) // weight
                bounds[(self.grid + 1) * g + tid] = job

    @cute.jit
    def rank_ranges(self, sm, units, rank):
        """``partition`` with the FC2 ranges from ``fc2_bounds``' table (ranks past
        the grid, ``home_bits``' spare lanes, read the last rank's)."""
        ranges = self.partition(units, rank)
        bounds = view(sm.iterator, self.off_bounds, Int32, 2 * self.grid + 2)
        r = rank
        if r >= self.grid:
            r = Int32(self.grid - 1)
        for g in cutlass.range_constexpr(2):
            ranges[g, 4] = bounds[(self.grid + 1) * g + r]
            ranges[g, 5] = bounds[(self.grid + 1) * g + r + 1]
        return ranges

    @cute.jit
    def fc2b_end(self, sm, group, rank):
        bounds = view(sm.iterator, self.off_bounds, Int32, 2 * self.grid + 2)
        return bounds[self.grid + 2 + rank]

    @cute.jit
    def zone_jobs(self, ranges):
        """``RubinGroupedExpertsKernel.zone_jobs`` for weighted FC2 ranges: a rank's
        FC2(B) range holds at least its weight share (each job >= fc2_base + 1) over
        the heaviest job's weight (fc2_base + width / 8)."""
        streams = ranges[1, 1] // (self.tiles * 14)
        share = 28 * (self.fc2_base + 1) * streams // self.grid
        least = share // (self.fc2_base + self.subs)
        zone = Int32(self.pool_zone)
        if least < zone:
            zone = least
        return zone // self.pool_grain * self.pool_grain

    @cute.jit
    def images_used(self, sm):
        """This call's hidden images (sub-units)."""
        streams = view(sm.iterator, self.off_misc + 1140, Int32, 1)[0]
        return self.sub_starts(sm)[streams]

    @cute.jit
    def sub_sources(self, rows, unit, sources):
        """``row_sources`` of sub-unit ``unit`` (the 64-row plan blocks)."""
        lane = thread()[2]
        for j in cutlass.range_constexpr(5):
            row = 2 * j + lane // 16 if j < 4 else lane & 7
            pair = Int32(rows[unit * 8 + row])
            if pair < 0:
                pair = Int32(rows[unit * 8])
            token = pair >> 4
            local = (token & 63) * (3584 if j < 4 else 448)
            sources[j] = (token >> 6) * (64 * (3584 + 448)) + local

    @cute.jit
    def gather_x(self, sm, plan, ranges, batch):
        """Each FC1 stage's activation slice of its stream's k sub-units (8-row images
        2 KiB apart) and their scale rows (row 8 s + r of each K128 half)."""
        lane, ws = (thread()[2], plan.iterator)
        stages, rows = (self.stages, self.tables(sm)[1])
        sub0 = self.sub_starts(sm)
        full, empty = (
            view(sm.iterator, self.off_misc + o, Uint64, stages)
            for o in (self.bar_full, self.bar_empty)
        )
        chunk, lanes = (lane & 15, lane // 16)
        image = [self.off_x + swizzled(chunk, 2 * j + lanes) for j in range(4)]
        scale = self.off_xsf + lane // 8 * (16 * self.wide) + (lane & 7) * 16
        sources = cute.make_rmem_tensor(5, Int32)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, r = (lo % 14, lo // 14 % 3)
            index = self.first_unit(ranges, 0, group)
            first, k = (Int32(0), Int32(0))
            if lo < hi:
                first, k = (sub0[index], sub0[index + 1] - sub0[index])
            for st in cutlass.range(lo, hi, unroll=1):
                if issued >= stages:
                    self.sync(empty.iterator + slot, phase ^ 1)
                for s in cutlass.range(k, unroll=1):
                    self.sub_sources(rows, first + s, sources)
                    dst = slot * self.xslot + s * 2048
                    for j in cutlass.range_constexpr(4):
                        src = sources[j] + ks * 256 + chunk * 16
                        copy_g2s(sm.iterator + image[j] + dst, ws + src)
                    if lane < 16:
                        src = PLAN_XSF + sources[4] + (2 * ks + lane // 8) * 16
                        dst = scale + slot * self.xsf_stride + s * 128
                        copy_g2s(sm.iterator + dst, ws + src)
                cute.arch.cp_async_mbarrier_arrive_noinc(full.iterator + slot)
                slot, issued = (slot + 1, issued + 1)
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)
                ks += 1
                if ks == 14:
                    ks, r = (Int32(0), r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        if st + 1 < hi:
                            first, k = (sub0[index], sub0[index + 1] - sub0[index])

    @cute.jit
    def acquire_big(self, empty, b, uses):
        if uses >= self.big:
            self.sync(empty.iterator + b, uses // self.big - 1 & 1, True)

    @cute.jit
    def consume(self, sm, tmem, ranges):
        """``RubinGroupedExpertsKernel.consume`` with N = 8k per stream and big-slot
        accumulators (one per FC1 tile chain / FC2 job, in the same order; one end
        marker per epilogue group)."""
        lane, stages = (thread()[2], self.stages)
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull)
        bars += (self.bar_accempty, self.bar_hfull, self.bar_hempty)
        full, empty, accfull, accempty, hfull, hempty = (
            view(sm.iterator, self.off_misc + o, Uint64, n)
            for o, n in zip(bars, (stages, stages, 16, 16, 2, 2))
        )
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        sub0 = self.sub_starts(sm)
        a0 = smem_desc_rows(sm.iterator, 64)
        asf0 = smem_desc(sm.iterator + self.off_sfa, False)
        x0 = smem_desc_rows(sm.iterator + self.off_x, 128)
        xsf0 = smem_desc(sm.iterator + self.off_xsf, False)
        h0 = smem_desc_rows(sm.iterator + self.off_h, 128)
        hsf0 = smem_desc(sm.iterator + self.off_hsf, False)
        uses, count = (Int32(0), Int32(0))
        for g in cutlass.range_constexpr(2):  # FC1(A), FC1(B): the producer's order
            lo, hi = (ranges[g, 2], ranges[g, 3])
            ks, r = (lo % 14, lo // 14 % 3)
            index = self.first_unit(ranges, 0, g)
            k = Int32(1)
            if lo < hi:
                k = sub0[index + 1] - sub0[index]
            for s in cutlass.range(lo, hi, unroll=1):
                slot = count % stages
                self.sync(full.iterator + slot, count // stages & 1, True)
                weights = (tmem + self.col_w + slot * 8, asf0 + slot * 64)
                released = shared_address(empty.iterator + slot)
                first, last = ((s == lo) | (ks == 0), (s == hi - 1) | (ks == 13))
                b = uses % self.big
                if first:
                    self.acquire_big(accempty, b, uses)
                cute.arch.fence_proxy("async.shared", space="cta")
                completion = Uint32(0)
                if last:
                    completion = shared_address(accfull.iterator + b)
                xsf = (
                    tmem + self.col_x + slot * 8,
                    xsf0 + slot * (self.xsf_stride // 16),
                )
                stage_vw(
                    (
                        tmem + b * self.wide,
                        a0 + slot * 1024,
                        x0 + slot * (self.xslot // 16),
                    ),
                    weights,
                    (*xsf, Uint32(1)),
                    (Uint32(not first), released),
                    completion,
                    k - 1,
                    kind="fc1",
                    half=self.wide,
                )
                cute.arch.sync_warp()
                if (lane == 0) & (count == 0):
                    self.stamp(6)
                if last:
                    uses += 1
                count += 1
                ks += 1
                if ks == 14:
                    ks, r = (Int32(0), r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        if s + 1 < hi:
                            k = sub0[index + 1] - sub0[index]
        mma = (tmem, a0, asf0, h0, hsf0, sub0)
        bars = (full, empty, accempty, accfull, hfull, hempty)
        slot, phase = (count % stages, count // stages & 1)
        state = (slot, phase, uses, Int32(-1), Int32(-1), Int32(0), Int32(1), Int32(0))
        zone = self.zone_jobs(ranges)
        if lane == 0:
            self.stamp(7)
        for g in cutlass.range_constexpr(2):
            end = ranges[g, 5] - (zone if g == 1 else 0)
            for j in cutlass.range(ranges[g, 4], end, unroll=1):
                self.sync(full.iterator + state[0], state[1], True)
                state = self.fc2_mma(state, mma, bars, j // self.jobs, j % self.jobs)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        pmeta = view(sm.iterator, self.off_misc + self.bar_pmeta, Uint64, 16)
        ended = False
        while not ended:
            self.sync(full.iterator + state[0], state[1], True)
            kind, a, b = (meta[4 * state[0] + i] for i in range(3))
            if kind != 1:  # the end marker
                ended = True
            else:
                state = self.fc2_mma(state, mma, bars, a, b, (accmeta, pmeta))
        slot, phase, uses, current, ordinal, hbuffer, _, _ = state
        for _ in cutlass.range(2, unroll=1):
            b = uses % self.big
            self.acquire_big(accempty, b, uses)
            self.signal(accmeta, pmeta, b, (Int32(2), Int32(0), Int32(0), Int32(0)))
            uses += 1
        if (lane == 0) & (current >= 0) & (hbuffer < 2):
            commit(hempty.iterator + hbuffer)

    @cute.jit
    def fc2_mma(self, state, mma, bars, unit, tile, pool=None):
        """One FC2 job (stream ``unit``, ``tile``) on the MMA warp, its slot waited:
        the stream's hidden images (a new stream releases the previous buffer through
        the MMAs' commit; its k read once), a big slot, N = 8k MMAs, the commit.
        Static jobs acquire eight big slots at a time, as the 64-row kernel's
        accumulators (each epilogue group frees its slots in order); ``pool``: one at a
        time, the job's metadata to the epilogue."""
        slot, phase, uses, current, ordinal, hbuffer, k, ahead = state
        tmem, a0, asf0, h0, hsf0, sub0 = mma
        full, empty, accempty, accfull, hfull, hempty = bars
        stages = self.stages
        weights = (tmem + self.col_w + slot * 8, asf0 + slot * 64)
        released = shared_address(empty.iterator + slot)
        copy_scales, release = (Uint32(0), Uint32(0))
        if unit != current:
            if (current >= 0) & (hbuffer < 2):
                release = shared_address(hempty.iterator + hbuffer)
            current = unit
            ordinal += 1
            hbuffer = ordinal & 1
            self.sync(hfull.iterator + hbuffer, ordinal >> 1 & 1, True)
            if (thread()[2] == 0) & (ordinal == 0):
                self.stamp(8)
            copy_scales = Uint32(1)
            k = sub0[unit + 1] - sub0[unit]
        hd = h0 + hbuffer * (self.hbuf // 16)
        hs = hsf0 + hbuffer * (self.hsf_stride // 16)
        sfbc = tmem + self.col_h + hbuffer * self.w2_scale_columns
        b = uses % self.big
        if cutlass.const_expr(pool is None):
            if ahead == 0:
                for i in cutlass.range_constexpr(6, 8):
                    self.acquire_big(accempty, (uses + i) % self.big, uses + i)
                ahead = Int32(8)
            ahead -= 1
        else:
            self.acquire_big(accempty, b, uses)
        stage_vw(
            (tmem + b * self.wide, a0 + slot * 1024, hd),
            weights,
            (sfbc, hs, copy_scales),
            (release, released),
            shared_address(accfull.iterator + b),
            k - 1,
            kind="fc2",
            half=self.wide,
        )
        cute.arch.sync_warp()
        if cutlass.const_expr(pool is not None):
            accmeta, pmeta = pool
            self.signal(accmeta, pmeta, b, (Int32(1), unit, tile, Int32(0)))
        uses += 1
        slot += 1
        if slot == stages:
            slot, phase = (Int32(0), phase ^ 1)
        return (slot, phase, uses, current, ordinal, hbuffer, k, ahead)

    @cute.jit
    def epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars):
        """``RubinGroupedExpertsKernel.epilogue`` over big slots: the items (an FC1
        tile chain, a static FC2 job, then the pool's jobs by metadata up to one end
        marker per group) come in ``consume``'s order, a big slot each, and the two
        epilogue groups take every other one. The owner reads the item's k sub-units
        in order (``take``: the slot released after the last read) and handles each
        as the 64-row kernel handles a unit."""
        eg = (thread()[1] - 4) // 4
        full = view(sm.iterator, self.off_misc + self.bar_accfull, Uint64, 16)
        empty = view(sm.iterator, self.off_misc + self.bar_accempty, Uint64, 16)
        sub0 = self.sub_starts(sm)
        zone = self.zone_jobs(ranges)
        offset = Int32(0)
        for kk in cutlass.range_constexpr(4):
            g = kk % 2
            if cutlass.const_expr(kk < 2):
                lo, hi = (ranges[g, 2], ranges[g, 3])
                start, count = (self.tiles * ranges[g, 0] + lo // 14, Int32(0))
                if lo < hi:
                    count = (hi - 1) // 14 - lo // 14 + 1
            else:
                start = ranges[g, 4]
                count = ranges[g, 5] - start - (zone if g == 1 else 0)
            for i in cutlass.range((eg ^ offset) & 1, count, 2, unroll=1):
                item, use = (start + i, offset + i)
                stream = item // (self.tiles if kk < 2 else self.jobs)
                first = sub0[stream]
                k = sub0[stream + 1] - first
                b = use % self.big
                self.sync(full.iterator + b, use // self.big & 1, True)
                for s in cutlass.range(k, unroll=1):
                    values = self.take(tmem, empty, b, s, s == k - 1)
                    if cutlass.const_expr(kk < 2):
                        if self.sub_partials(workspace, values, item, ranges, s):
                            tile = (first + s) * self.tiles + item % self.tiles
                            store = (sm, workspace, values, tile, parity)
                            self.store_hidden(*store, scalars[:3])
                    else:
                        unit, tile = (first + s, item % self.jobs)
                        self.fc2(sm, workspace, output, values, unit, tile, ranges)
            offset += count
        # The pool: this group's uses offset + m (m of its parity); pmeta[b]'s phase
        # counts the earlier pool uses of the slot (m // big).
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        pmeta = view(sm.iterator, self.off_misc + self.bar_pmeta, Uint64, 16)
        m, ended = ((eg ^ offset) & 1, False)
        while not ended:
            pool_use = offset + m
            slot = pool_use % self.big
            self.sync(pmeta.iterator + slot, m // self.big & 1)
            kind, unit0, tile0 = (accmeta[4 * slot + i] for i in range(3))
            if kind != 1:  # this group's end marker
                ended = True
            else:
                first = sub0[unit0]
                k = sub0[unit0 + 1] - first
                self.sync(full.iterator + slot, pool_use // self.big & 1, True)
                for s in cutlass.range(k, unroll=1):
                    values = self.take(tmem, empty, slot, s, s == k - 1)
                    self.fc2(sm, workspace, output, values, first + s, tile0, ranges)
                m += 2
        if cutlass.const_expr(self.publish):
            self.finalize(workspace, output, batch, scalars[3:], sm=sm)

    @cute.jit
    def take(self, tmem, empty, b, s, last):
        """``columns`` (sub-unit ``s`` of big slot ``b``); after the slot's last read,
        the slot released (this group's four warps: ``empty`` counts four)."""
        values = self.columns(tmem, b, s)
        if last:
            fence_tmem(before=True)
            cute.arch.sync_warp()
            if thread()[2] == 0:
                cute.arch.mbarrier_arrive(empty.iterator + b)
        return values

    @cute.jit
    def columns(self, tmem, b, s):
        """Sub-unit ``s``'s 8 FP32 columns of big slot ``b`` for this warp's 32 rows."""
        warp = thread()[1]
        column = b * self.wide + 8 * s
        raw = load_tmem(tmem + (Uint32(32 * (warp & 3)) << 16) + column, 8)
        values = cute.make_rmem_tensor(self.width, Float32)
        for t in cutlass.range_constexpr(self.width):
            values[t] = raw[t]
        return values

    @cute.jit
    def sub_partials(self, workspace, values, tile, ranges, s):
        """``RubinGroupedExpertsKernel.partials`` for sub-unit ``s`` of the tile's
        stream, on partial plane ``s`` (the same slots, owners, peers and order)."""
        ws = workspace.iterator
        _, warp, lane, cta = thread()
        row, slots = (32 * (warp & 3) + lane, self.slots)
        plane = Int64(self.ws_partial)
        if s > 0:
            plane = Int64(self.ws_planes) + Int64(s - 1) * self.plane_bytes
        group = Int32(tile >= self.tiles * ranges[1, 0])
        start = (tile - self.tiles * ranges[group, 0]) * 14
        mine = cta
        if cutlass.const_expr(len(self.dies) > 0):
            mine = ranges[group, 6]
        owner = start >= ranges[group, 2]
        if not owner:
            offset = plane + ((Int64(group) * slots + mine) * 1024 + row) * 4
            for t in cutlass.range_constexpr(self.width):
                store_words(ws + offset + t * 512, (float_bits(values[t]),))
        else:
            peer, size = (mine + 1, ranges[group, 1])
            while (peer < self.grid) & (peer * size // self.grid < start + 14):
                if (peer + 1) * size // self.grid != peer * size // self.grid:
                    index = (Int64(group) * slots + peer) * 1024 + row
                    source = ws + plane + index * 4
                    words = cute.make_rmem_tensor(self.width, Uint32)
                    ready = False
                    while not ready:
                        bad = False
                        for t in cutlass.range_constexpr(self.width):
                            words[t] = load_volatile(source + t * 512)[0]
                            bad |= words[t] == Uint32(4294967295)
                        ready = cute.arch.vote_all_sync(not bad)
                    for t in cutlass.range_constexpr(self.width):
                        values[t] += bits_float(words[t])
                        store_words(source + t * 512, (Uint32(4294967295),))
                peer += 1
        return owner

    @cute.jit
    def load_hidden(self, sm, workspace, ranges, parity):
        """The producer's queued streams in order, double-buffered: each stream's k
        hidden images 2 KiB apart and their scale rows 8 s + r."""
        lane = thread()[2]
        full = view(sm.iterator, self.off_misc + self.bar_hfull, Uint64, 2)
        empty = view(sm.iterator, self.off_misc + self.bar_hempty, Uint64, 2)
        sub0 = self.sub_starts(sm)
        ordinal = Int32(-1)
        unit = self.pop_unit(sm, Int32(0))
        while unit >= 0:
            ordinal += 1
            buffer = ordinal & 1
            if ordinal >= 2:
                self.sync(empty.iterator + buffer, (ordinal >> 1) - 1 & 1)
                cute.arch.fence_proxy("async.shared", space="cta")
            first = sub0[unit]
            k = sub0[unit + 1] - first
            for s in cutlass.range(k, unroll=1):
                source = self.image(workspace, parity, first + s)
                self.poll_sub(sm, source, buffer, s)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(full.iterator + buffer)
            unit = self.pop_unit(sm, ordinal + 1)

    @cute.jit
    def poll_sub(self, sm, source, buffer, s):
        """``ExpertsKernel.poll_hidden`` into sub-image ``s`` of ``buffer``."""
        lane = thread()[2]
        chunks: cutlass.Constexpr = self.h_image_bytes // 512
        values, scale = (cute.make_rmem_tensor((chunks, 4), Uint32), Uint32(0))
        valid_bytes = min(4, max(0, self.tiles * 2 - 4 * (lane & 3)))
        mask = Uint32(4294967295)
        if valid_bytes < 4:
            mask = (Uint32(1) << 8 * valid_bytes) - 1
        ready = False
        while not ready:
            bad = False
            for v in cutlass.range_constexpr(chunks):
                index = lane + 32 * v
                words = load_volatile(source + index * 16, count=4)
                for w in cutlass.range_constexpr(4):
                    values[v, w] = words[w]
                atom, token, position = (index >> 6, index >> 3 & 7, index & 7)
                if 8 * atom + (position ^ token) < self.tiles * 4:
                    for w in cutlass.range_constexpr(4):
                        bad |= has_ff_byte(words[w])
            scale = load_volatile(source + self.h_image_bytes + lane * 4)[0]
            if valid_bytes > 0:
                bad |= has_ff_byte(scale & mask)
            ready = cute.arch.vote_all_sync(not bad)
        image = sm.iterator + self.off_h + buffer * self.hbuf + s * self.h_image_bytes
        for v in cutlass.range_constexpr(chunks):
            words = tuple(values[v, w] for w in range(4))
            store_shared_words(image + (lane + 32 * v) * 16, words)
        word = cute.arch.shuffle_sync(scale, 4 * (lane & 7) + (lane >> 3))
        if lane < 8 * self.h_chunks:
            sf = sm.iterator + self.off_hsf + buffer * self.hsf_stride
            sf += (lane >> 3) * (16 * self.wide) + (8 * s + (lane & 7)) * 16
            store_shared_words(sf, (word,) * 4)
