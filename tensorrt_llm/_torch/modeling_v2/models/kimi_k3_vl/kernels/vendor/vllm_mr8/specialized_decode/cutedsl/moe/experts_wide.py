# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin (cc 10.7) routed experts for TP16 verify steps of more than 64 token rows
(``capacity`` 128-``WIDE_MAX``): ``RubinGroupedExpertsKernel``'s engine (units of
<= 8 tokens per expert, the packed weight stream, die matching, the FC2 pool) with the
grouping, unit tables, MXFP8 plan, hidden images and staged FC2 rows sized from the
capacity.

- Grouping: each expert's token mask spans ``capacity / 32`` words, with each word's
  running count before it (Int16); the masks (57 KiB at 512 rows) and the counts live
  in the weight ring while the CTA groups, so the producer starts streaming once the
  pairs are ranked (the 64-row kernel overlaps the ranking). A pair's rank is its
  word's count plus one popcount (a scan of up to 16 mask words per pair, on 16-way
  bank conflicts, took 17 us at 512 rows). Above 640 rows (7 stages) the masks and
  counts outgrow the ring: they, and the grouping's other grouping-only arrays (expert
  bases, warp totals, the VW owner table), then run on into the scale, activation and
  hidden slots below the misc block, all idle until the grouping's closing sync
  (``grouping_spill``). The CTA stays within the default 227 KB: Rubin's oversized
  shared-memory mode shrinks L1 and costs the wide kernels' epilogue ~25 us per call
  (M576 / M640 at 240-280 KB, 1 GPU).
- Plan: ``capacity / 64`` consecutive 64-row MXFP8 plans (``PLAN_BYTES`` each): token
  t sits at row t % 64 of plan t // 64.
- Workspace: past everything the 64-row kernels keep across calls (the 64-row
  workspace, die census, the M8 kernel's peer slots and generation), armed all-ones
  like it: this
  kernel's own call generation per slot, its last call's image count per parity, the
  staged rows' chunk homes (``finalize``), the hidden images ([2, the unit bound at
  capacity ``WIDE_MAX``]) and the staged FC2 rows at the same offsets for every
  capacity:
  token t's 16 rows as [28 units][16 slots][128 columns] BF16 at 112 KiB t, one 4 KiB
  page a unit (128 columns of the token's 16 slots). The images
  alternate on that own generation, so calls of other batch sizes in between change
  nothing; the shared grouped images and claim counters keep the
  64-row kernels' protocol (this call re-arms the last grouped call's, and reports no
  shared images of its own).
- Outputs: unpublished (``publish`` False: every pair's FC2 row), or ``"local"``: the
  units' finalize sums each token's 16 staged rows with its routing weights (the
  output argument) into this rank's partial latent row (BF16 [rows, 3584] at the
  address after the SiTU scalars): the stock experts' output contract, before the
  tail's reduction. With ``dies``, each die's warps finalize the units homed on
  their die.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int16, Int32, Int64, Uint8, Uint32

from ..common.primitives import (
    alloc,
    bf16,
    fence_tmem,
    fma,
    launch,
    load_volatile,
    load_words,
    pack_pair,
    release_tmem,
    sleep_ns,
    store_relaxed,
    store_words,
    thread,
    view,
)
from ..common.tensormap import GridMap, map_address
from .experts import PLAN_BYTES, ExpertsGeometry, ExpertsPlanKernel, plan_chunk
from .experts_rubin import (
    ROUTE_IDS,
    ROUTE_WORDS,
    generation_bytes,
    peers_bytes,
    peers_offset,
)
from .experts_rubin64 import CENSUS_BYTES, RubinGroupedExpertsKernel

# The most token rows a wide engine serves (DL3 q4 B192). Every wide engine's hidden
# images and staged rows sit at offsets derived from it (``image_units``).
WIDE_MAX = 768
# Shared memory a CTA may use: the default opt-in cap (module docstring: not the
# oversized mode).
SMEM_DEFAULT = 227 * 1024


class RubinWideExpertsKernel(RubinGroupedExpertsKernel):
    """``RubinGroupedExpertsKernel`` at ``capacity`` (a multiple of 64, at most
    ``WIDE_MAX``) token rows; ``publish`` False or "local" (module docstring)."""

    def __init__(
        self, experts, grid, slots, capacity, publish=False, stages=7, trace=0,
        pool=(12, 6), dies=(), static_groups=False, tables=True,
    ):  # fmt: skip
        """``tables`` False: a subclass lays out the shared-memory tables (and sizes
        the CTA) itself (``RubinVWExpertsKernel``)."""
        assert capacity % 64 == 0 and 64 < capacity <= WIDE_MAX
        # "twoshot" (axl wpush): the finalize pushes each token's latent columns into
        # their owners' two-shot receive slots of the wide tail (``tail_rows``'s
        # TailRowsBox at the deployment's capacity), as the 64-row engine's "twoshot"
        # does into the tuned tail's; the wide tail then only reduces.
        assert publish in (False, "local", "twoshot"), publish
        super().__init__(
            experts, grid, slots, publish, stages, trace, pool, dies, static_groups
        )
        self.capacity, self.blocks = (capacity, capacity // 64)
        self.words = capacity // 32  # token mask words per expert
        # Each expert's masks (and running counts) one word (count) apart beyond its
        # ``words``: an odd stride spreads a warp's experts over the shared memory banks
        # (at ``words`` = 4-16 they fell on 2-8 banks: 4-16-way conflicts in the
        # atomics, the running counts and the ranking).
        self.mask_stride = self.words + 1
        # Units of <= 8 tokens: sum ceil(n_e / 8) <= 2 capacity + 7 experts / 8.
        self.units = (2 * capacity + 7 * experts // 8 + 31) // 32 * 32
        self.off_masks = 0  # the weight ring, while grouping
        self.off_prefix = 4 * self.mask_stride * experts  # the counts, after them
        if tables:
            self.off_tables = (self.off_misc + 5120 + 127) // 128 * 128
            end = self.off_tables + 20 * self.units
            # without the spill: bases, totals, then the transpose buffers
            whole = (end + 4 * experts + 64 + 127) // 128 * 128 + 8 * 512
            if self.grouping_spill(stages, whole):
                self.off_bases, self.off_totals = self.low_scratch((4 * experts, 64))
            else:
                self.off_bases = end
                self.off_totals = self.off_bases + 4 * experts
                end = self.off_totals + 64
            # ``fc2``'s per-warp transpose buffers ([8 tokens, 64 B] per epilogue warp).
            self.off_fc2s = (end + 127) // 128 * 128
            end = self.off_fc2s + 8 * 512
            self.check_smem(end)
        shared = self.ws_steal + 2 * slots * 4  # ``ExpertsGeometry.workspace_bytes``
        # The route mailbox sits inside that workspace, below every region past it.
        assert self.ws_xflag + ROUTE_IDS + 4 * ROUTE_WORDS <= shared
        shared = peers_offset(shared, CENSUS_BYTES) + peers_bytes(slots)
        shared += generation_bytes(slots)  # the M8 kernel's own generation
        self.ws_wgen = (shared + 255) // 256 * 256
        self.ws_wlast = self.ws_wgen + 4 * slots
        # The home die bit of each 2 MiB chunk under the staged rows (``stage_homes``):
        # two words (chunks 0-31, 32-63).
        self.ws_homes = self.ws_wlast + 8
        self.ws_images = (self.ws_homes + 8 + 255) // 256 * 256
        # Every wide engine (any capacity or width) shares these regions of the
        # scratch, one call after another: the images' parity stride and the staged
        # rows' offset follow the largest capacity's unit bound, so a call re-arms the
        # images its predecessor used whatever its capacity, and no capacity's staged
        # rows overlap another's images.
        self.image_units = (2 * WIDE_MAX + 7 * experts // 8 + 31) // 32 * 32
        stage = self.ws_images + 2 * self.image_units * self.h_compact_bytes
        # Units 2i and 2i + 1 on one 8 KiB-aligned page pair (``finalize``).
        self.ws_stage = (stage + 8191) // 8192 * 8192
        self.workspace_bytes = self.ws_stage + 16 * capacity * 3584 * 2
        self.plan_bytes = self.blocks * PLAN_BYTES

    def grouping_spill(self, stages, whole):
        """Whether the masks and their running counts (Int16 words past the masks,
        ``off_prefix``) outgrow the weight ring (above 640 rows at 7 stages), or the
        layout with the grouping-only arrays past the tables (ending at ``whole``)
        passes the default shared memory (width 32 above 384 rows): the counts then
        run on below ``off_misc``, with the other grouping-only arrays after them
        (``low_scratch``)."""
        if whole > SMEM_DEFAULT:
            return True
        return 6 * self.mask_stride * self.experts > stages * self.slot_bytes

    def low_scratch(self, sizes):
        """Offsets of grouping-only arrays of ``sizes`` bytes (128 B-aligned) after the
        masks and counts, below ``off_misc`` (``grouping_spill``)."""
        at, out = 6 * self.mask_stride * self.experts, []
        for n in sizes:
            at = (at + 127) // 128 * 128
            out.append(at)
            at += n
        assert at <= self.off_misc, (at, self.off_misc)
        return tuple(out)

    def check_smem(self, end):
        """The CTA's shared memory (``smem_bytes``) for a layout ending at ``end``."""
        assert end <= SMEM_DEFAULT, end
        self.smem_bytes = max(end, 212096)

    def image(self, workspace, parity, unit):
        base = workspace.iterator + self.ws_images
        return base + (Int64(parity) * self.image_units + unit) * self.h_compact_bytes

    @cute.jit
    def images_used(self, sm):
        """This call's hidden images (units)."""
        return view(sm.iterator, self.off_misc + 1140, Int32, 1)[0]

    @cute.jit
    def rearm_wide(self, workspace, generation, wgen):
        """``ExpertsKernel.rearm`` on the shared grouped images (the last grouped
        call's, other parity) and claim counters, then this kernel's own images of its
        other generation parity (its previous call's)."""
        _, _, lane, cta = thread()
        ws, words = (workspace.iterator, self.h_compact_bytes // 16)
        last = view(ws, self.ws_calls + 4 * self.slots, Int32, 2)
        used = max(last[generation & 1 ^ 1], 0) * words
        lo, hi = (cta * used // self.grid, (cta + 1) * used // self.grid)
        other = ExpertsGeometry.image(self, workspace, generation & 1 ^ 1, 0)
        for i in range(lo + lane, hi, 32):
            store_words(other + i * 16, (Uint32(4294967295),) * 4)
        base = self.ws_steal + 4 * self.slots * (generation & 1 ^ 1)
        counters = view(ws, base, Int32, self.slots)
        if lane == 0:
            counters[cta] = Int32(-1)
            if cta < self.slots - self.grid:
                counters[self.grid + cta] = Int32(-1)
        own = view(ws, self.ws_wlast, Int32, 2)
        used = max(own[wgen & 1 ^ 1], 0) * words
        lo, hi = (cta * used // self.grid, (cta + 1) * used // self.grid)
        other = self.image(workspace, wgen & 1 ^ 1, 0)
        for i in range(lo + lane, hi, 32):
            store_words(other + i * 16, (Uint32(4294967295),) * 4)

    @cute.kernel
    def kernel_maps(self, args, w13_map: GridMap, w2_map: GridMap):
        self.body(args, (map_address(w13_map), map_address(w2_map)))

    @cute.jit
    def body(self, args, maps):
        """``RubinGroupedExpertsKernel.body`` with this kernel's own image generation
        (module docstring); no shared-row push."""
        x, ids, _, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        if tid == 0:
            self.stamp(0)
        tmem_base = self.initialize(sm, maps, batch)
        cute.arch.griddepcontrol_wait()
        if tid == 0:
            self.stamp(1)
        ws, slots = (workspace.iterator, self.slots)
        counters = view(ws, self.ws_counter, Int32, slots)
        wgens = view(ws, self.ws_wgen, Int32, slots)
        wgen = wgens[cta]  # in flight during the grouping
        routed = view(sm.iterator, self.off_misc + 1140, Int32, 3)
        arrival = None
        if cutlass.const_expr(len(self.dies) > 0):
            arrival = self.ticket(workspace)
        self.group(sm, ids, workspace, batch, arrival=arrival)
        if tid == 0:
            self.stamp(2)
        distinct, call_count, generation = (routed[0], routed[1], routed[2])
        experts = self.tables(sm)[0]
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(len(self.dies) > 0):
            ranges = self.matched_ranges(sm, distinct)
        else:
            ranges = self.rank_ranges(sm, distinct, cta)
        if tid == 0:
            self.stamp(11)
        if warp == 0:
            if lane == 0:
                weights = (sm, maps, w13_scales, w2_scales, ranges, experts)
                self.stream(*weights, (workspace, generation & 1))
        else:
            if warp == 1:
                self.consume(sm, tmem, ranges)
                if lane == 0:
                    self.stamp(9)
            elif warp == 2:
                self.load_hidden(sm, workspace, ranges, wgen & 1)
                if lane == 0:
                    self.stamp(12)
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:], sm=sm)
            elif warp == 3:
                self.gather_x(sm, x, ranges, batch)
                if lane == 0:
                    self.stamp(14)
                self.rearm_wide(workspace, generation, wgen)
                if lane == 0:
                    self.claims(sm, workspace, ranges, generation & 1)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.publish):
                    self.finalize(workspace, output, batch, scalars[3:], sm=sm)
            else:
                self.epilogue(
                    sm, workspace, output, tmem, ranges, batch, wgen & 1, scalars
                )
                if tid == 128:
                    self.stamp(10)
        release_tmem(tmem, 512, warp, 3)
        if tid == 0:
            self.stamp(13)
            grouped = view(ws, self.ws_calls, Int32, slots + 2)
            own = view(ws, self.ws_wlast, Int32, 2)
            counters[cta] = call_count + 1
            grouped[cta] = generation + 1
            wgens[cta] = wgen + 1
            if cta < slots - self.grid:
                counters[self.grid + cta] = call_count + 1
                grouped[self.grid + cta] = generation + 1
                wgens[self.grid + cta] = wgen + 1
            if cta == 0:
                grouped[slots + (generation & 1)] = Int32(0)  # no shared images
                own[wgen & 1] = self.images_used(sm)

    @cute.jit
    def row_sources(self, rows, index, sources, batch):
        """``ExpertsKernel.row_sources`` on the 64-row plan blocks."""
        lane = thread()[2]
        for j in cutlass.range_constexpr(5):
            row = 2 * j + lane // 16 if j < 4 else lane & 7
            pair = Int32(rows[index * 8 + row])
            if pair < 0:
                pair = Int32(rows[index * 8])
            token = pair >> 4
            local = (token & 63) * (3584 if j < 4 else 448)
            sources[j] = (token >> 6) * PLAN_BYTES + local

    @cute.jit
    def clear(self, sm):
        tid = thread()[0]
        words = self.mask_stride * self.experts
        masks = view(sm.iterator, self.off_masks, Uint32, words)
        row_words = self.tables(sm)[2]
        for i in range(tid, words, self.threads):
            masks[i] = Uint32(0)
        for i in range(tid, 4 * self.units, self.threads):
            row_words[i] = Uint32(4294967295)

    @cute.jit
    def group(self, sm, ids, workspace, batch: Int32, scalars=None, arrival=None):
        """``RubinGroupedExpertsKernel.group`` with ``words``-word token masks and their
        running counts in the weight ring: every warp ranks its pairs (ids still held),
        then the CTA syncs (and retires the masks' generic accesses) before any weight
        lands in the ring."""
        tid, warp, lane, cta = thread()
        words: cutlass.Constexpr = self.words
        stride: cutlass.Constexpr = self.mask_stride
        masks = view(sm.iterator, self.off_masks, Uint32, stride * self.experts)
        prefix = view(sm.iterator, self.off_prefix, Int16, stride * self.experts)
        base = view(sm.iterator, self.off_bases, Int32, self.experts)
        totals = view(sm.iterator, self.off_totals, Int32, self.threads // 32)
        experts, rows, _ = self.tables(sm)
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
        units = cute.make_rmem_tensor(span, Int32)
        count = Int32(0)
        for j in cutlass.range_constexpr(span):
            units[j], e = (Int32(0), span * tid + j)
            if e < self.experts:
                n = Int32(0)
                for w in cutlass.range_constexpr(words):
                    prefix[stride * e + w] = Int16(n)
                    n += Int32(cute.arch.popc(masks[stride * e + w]))
                units[j] = (n + 7) >> 3
            count += units[j]
        total = count
        for shift in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(total, 1 << shift)
            if lane >= 1 << shift:
                total += other
        if lane == 31:
            totals[warp] = total
        cute.arch.sync_threads()
        start = total - count
        for w in cutlass.range_constexpr(threads // 32 - 1):
            if warp > w:
                start += totals[w]
        if tid == threads - 1:
            scalars[0] = start + count
        if tid == 0:
            self.stamp(17)
        for j in cutlass.range_constexpr(span):
            e = span * tid + j
            if e < self.experts:
                base[e] = start
                for v in cutlass.range(units[j], unroll=1):
                    experts[start + v] = e
            start += units[j]
        if cutlass.const_expr(len(self.dies) > 0):  # the ticket has long returned
            self.arrive(sm, workspace, arrival)
        cute.arch.sync_threads()
        if tid == 0:
            self.stamp(18)
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
        # The masks' generic accesses before the weight ring's TMA writes.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

    @cute.jit
    def finalize(self, workspace, weights, batch, box, sm=None):
        """The staged rows' units (``unit``) into this rank's partial latent rows at
        ``box[0]`` (``"local"``), over ten warps a CTA (the hidden loader, the gather
        warp and the eight epilogue warps). Die oblivious: unit 10 cta + warp - 2,
        stride 10 grid. ``dies``: pages 2i and 2i + 1 sit on opposite dies (the home of
        a default allocation's 4 KiB page k of a 2 MiB chunk is parity(k & 0x1DB) ^ c,
        c per chunk in ``ws_homes``, ``common.dies``), so pair i gives each die one
        unit. Die d's job j holds its units of pairs j, j + 5 grid, ...; warp rank
        r = 10 k + warp - 2 (k: the CTA's census rank on its die, ``arrive``) takes job
        r of its die, past 5 grid the other die's job 10 grid - 1 - r (``matched``'s
        overflow). Every unit is finalized once whatever the home bits; they set only
        the locality (die-local reads and re-arms: ~1 us at 96-128 rows)."""
        _, warp, lane, cta = thread()
        stage = workspace.iterator + self.ws_stage
        if cutlass.const_expr(len(self.dies) > 0):
            info = view(sm.iterator, self.off_misc + 2688, Int32, 4)
            die, k = (info[0], info[3])
            homes = view(workspace.iterator, self.ws_homes, Uint32, 2)
            half: cutlass.Constexpr = self.grid * 5
            rank = k * 10 + warp - 2
            target, job = (die, rank)
            if rank >= half:
                target, job = (1 - die, 2 * half - 1 - rank)
            address = stage.toint()
            for i in range(job, batch * 14, half):
                page = address + Int64(i) * 8192
                bits = Uint32(Int32(page >> 12) & Int32(0x1DB))
                c = Int32((page >> 21) - (address >> 21))
                if cutlass.const_expr(self.capacity <= 512):  # 32 chunks at most
                    word = homes[0] >> Uint32(c)
                else:  # the staged rows span up to 44 chunks: two words
                    word = homes[c >> 5] >> Uint32(c & 31)
                h = (Int32(cute.arch.popc(bits)) ^ Int32(word)) & 1
                self.unit(stage, weights, box, 2 * i + Int32(h != target), lane)
        else:
            for unit in range(cta * 10 + warp - 2, batch * 28, self.grid * 10):
                self.unit(stage, weights, box, unit, lane)

    @cute.jit
    def unit(self, stage, weights, box, unit, lane):
        """``RubinGroupedExpertsKernel.finalize``'s line (the same polls, re-arm and FMA
        chains in slot order) on one unit, token unit // 28, columns 128 (unit % 28) +
        [0, 128): 16 slots x 256 B at stage + 4 KiB unit, two words a lane a slot."""
        token, column = (unit // 28, unit % 28 * 128)
        source = stage + Int64(unit) * 4096 + lane * 8
        words = cute.make_rmem_tensor(32, Uint32)
        scale = cute.make_rmem_tensor(16, Float32)
        ones = Uint32(4294967295)
        for slot in cutlass.range_constexpr(16):
            scale[slot] = Float32(weights[token * 16 + slot])
        ready = False
        while not ready:
            bad = False
            for slot in cutlass.range_constexpr(16):
                pair = load_volatile(source + slot * 256, count=2)
                words[2 * slot], words[2 * slot + 1] = pair
                bad |= (pair[0] == ones) | (pair[1] == ones)
            ready = cute.arch.vote_all_sync(not bad)
            if not ready:
                sleep_ns(64)
        low = [Float32(0), Float32(0)]
        high = [Float32(0), Float32(0)]
        for slot in cutlass.range_constexpr(16):
            store_words(source + slot * 256, (ones, ones))
            for v in cutlass.range_constexpr(2):
                low[v] = fma(bf16(words[2 * slot + v]), scale[slot], low[v])
                high[v] = fma(bf16(words[2 * slot + v], True), scale[slot], high[v])
        packed = [pack_pair(low[v], high[v], policy="latent") for v in range(2)]
        if cutlass.const_expr(self.publish == "twoshot"):
            # box: the receive slots [parities x TP, capacity, 3584 / TP], the owners'
            # bases, this rank's slot (parity x TP + rank): the words LatentRows would
            # push from the local row (``common.twoshot.push``; -0 is +0 already)
            recv, peers, rslot = box[0:3]
            width: cutlass.Constexpr = recv.shape[2]
            chunk = column + 4 * lane
            owner = chunk // width
            place = cute.crd2idx((rslot, token, chunk - owner * width), recv.layout)
            store_relaxed(Int64(peers[owner]) + Int64(place) * 2, packed)
        else:
            dst = Int64(box[0]) + (Int64(token) * 3584 + column + 4 * lane) * 2
            store_words(dst, packed)

    def staged_word(self, pair, tile, warp):
        """``RubinGroupedExpertsKernel.staged_word`` in the unit layout: pair p's slot
        p % 16 of unit (p // 16, tile)."""
        row = ((Int64(pair >> 4) * 28 + tile) * 16 + (pair & 15)) * 64
        return row + (warp & 3) * 16


class WidePlanKernel(ExpertsPlanKernel):
    """``ExpertsPlanKernel`` above 64 rows (no routing kernel in front): token t into
    row t % 64 of the 64-row plan t // 64, one CTA per token."""

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.batch, 256)

    @cute.kernel
    def kernel(self, args):
        x, plan = args
        tid, _, _, token = thread()
        block = view(plan.iterator, token // 64 * PLAN_BYTES, Uint8, PLAN_BYTES)
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:  # warp-uniform: chunks 256-447 are warps 0-5
                words = load_words(
                    x.iterator + token * 3584 + chunk * 8, count=4, mutable=True
                )
                plan_chunk(block, token % 64, chunk, words)
        cute.arch.griddepcontrol_launch_dependents()
