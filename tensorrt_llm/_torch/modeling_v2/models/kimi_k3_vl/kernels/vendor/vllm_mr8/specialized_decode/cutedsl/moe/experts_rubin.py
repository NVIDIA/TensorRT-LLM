# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin (cc 10.7) routed experts for TP16 capacity 8: ``ExpertsKernel``'s engine,
workspace and output rows, with the latent copied (bulk, each CTA's first FC1 stage
first) from an MXFP8 image the routing kernel writes in the experts' SMEM layout
instead of 212 CTAs loading and quantizing the BF16 latent, and a dynamic FC2(B) tail
(claimed tile by tile) instead of a fully static FC2 partition. The epilogue warps
stage the FC2 rows without shuffles or branches (one predicated BF16 half per lane),
reduce the hidden image's amax with ``redux.sync`` and select instead of branching: on
Rubin every warp shuffle and branch in the epilogue costs ~0.05-0.2 us, so their FC2
work kept a backlog behind the MMAs and set the kernel's tail. Each line's finalize
loads its staged rows while its CTA's last tiles drain, and the producer posts the
ring's end marker before its other bookkeeping. The MMA warp walks the FC1 stages from
the partition itself (only FC2 stages carry stage metadata): at TP16 B8 the stream is
bound by the bytes and by the MMA warp's per-stage work, not by the ring depth. The FP4
weights land packed (plain bytes) and feed Rubin's K64 block-scaled MMAs, which take
packed E2M1 from shared memory: half the slot bytes and half the MMAs of the K32
``mxf8f6f4`` steps on unpacked (16U4) weights."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int8, Int32, Int64, Uint8, Uint32, Uint64

from ..common.primitives import (
    acquire_tensormap,
    alloc,
    asm,
    bf16,
    bits_float,
    commit,
    copy_evict,
    copy_g2s,
    fence_tmem,
    float_bits,
    fma,
    fp8_bits,
    free_tmem,
    launch,
    load_tma3,
    load_tmem,
    load_volatile,
    load_words,
    multicast_words,
    mx_exponent,
    mx_rescale,
    named_barrier,
    pack_fp8,
    pack_pair,
    redux,
    shared_address,
    situ,
    sleep_ns,
    smem_desc,
    smem_words,
    store_shared_words,
    store_words,
    thread,
    unpack_bf16,
    view,
)
from ..common.tensormap import GridMap, encode_map, map_address, prefetch_map
from .experts import ExpertsKernel, claim, mx_block, swizzled

shuffle = cute.arch.shuffle_sync
# Image plan (capacity 8): the experts' SMEM latent image as written by the routing
# kernel, MXFP8 [14 stages][2 atoms][8 tokens][128 B, 128B-swizzled], then the scale
# image [28 words][8 tokens][16 B] (byte 0-3: four 32-blocks' E8M0).
IMAGE_XSF = 14 * 2048
IMAGE_BYTES = IMAGE_XSF + 28 * 128
# ``RubinExpertsKernel.fast_exchange``: one K-split peer piece's FP32 partial, 128 rows
# x 8 tokens row-major (32 B per row, two 16 B stores), in slots [call parity][group]
# [CTA] past the experts' workspace and the M64 die census; an owner loads its first
# PEER_BATCH peers' rows in one round.
PEER_SLOT = 4096
PEER_BATCH = 3


def peers_offset(workspace_bytes, census):
    """The peer slots' workspace offset (``experts_runtime.scratch`` sizes for them)."""
    return (workspace_bytes + 16 + census + 4095) // 4096 * 4096


def peers_bytes(slots):
    return 2 * 2 * slots * PEER_SLOT


def generation_bytes(slots):
    """``fast_exchange``'s own call generation per slot, past the peer slots."""
    return (4 * slots + 255) // 256 * 256


@cute.jit
def image_chunk(plan, token, chunk, words):
    """``experts.plan_chunk`` (bit-identical MXFP8) into the image plan: 8-element
    ``chunk`` of ``token``; lanes 4k..4k+3 hold one 32-element block."""
    lane, values = (thread()[2], cute.make_rmem_tensor(8, Float32))
    for i in cutlass.range_constexpr(4):
        pair = unpack_bf16(words[i])
        values[2 * i], values[2 * i + 1] = (pair[0], pair[1])
    top = abs(values[0])
    for i in cutlass.range_constexpr(1, 8):
        top = cute.arch.fmax(top, abs(values[i]))
    for width in (1, 2):
        top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, width))
    scale = mx_exponent(top)
    reciprocal = mx_rescale(scale)
    packed = [
        pack_fp8(tuple(values[4 * j + i] * reciprocal for i in range(4)))
        for j in range(2)
    ]
    wide = chunk >> 1
    offset = swizzled(wide, token) + (chunk & 1) * 8
    store_words(plan.iterator + offset, packed)
    word = Uint32(scale)
    for s in cutlass.range_constexpr(1, 4):
        word |= shuffle(Uint32(scale), lane + 4 * s) << 8 * s
    if chunk % 16 == 0:
        dst = plan.iterator + IMAGE_XSF + chunk // 16 * 128 + token * 16
        store_words(dst, (word, Uint32(0), Uint32(0), Uint32(0)))


class ImagePlanKernel:
    """Standalone image-plan writer (benchmarks, tests): one CTA per token."""

    def __init__(self, batch):
        self.batch = batch

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.batch, 256)

    @cute.kernel
    def kernel(self, args):
        x, plan = args
        tid, _, _, token = thread()
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range_constexpr(2):
            chunk = tid + 256 * i
            if chunk < 448:
                words = load_words(
                    x.iterator + token * 3584 + chunk * 8, count=4, mutable=True
                )
                image_chunk(plan, token, chunk, words)
        cute.arch.griddepcontrol_launch_dependents()


# Named barrier: the MMA and epilogue warps' last tcgen05 operation (warp 3 then frees
# the TMEM): warps 1, 3 and 4-11.
TMEM_DONE = 7
# Named barrier (``lean_route``): warp 0's unit tables are written (it arrives, every
# other warp waits).
ROUTED = 2

# Descriptor start addresses are 18 bits: every MMA / tcgen05.cp operand must sit in
# the first 256 KiB of the shared window (the dynamic base is 1 KiB in).
DESCRIPTOR_LIMIT = 262144 - 1024


@asm(Uint32)
def claim_last(pointer):
    """Acquire-release GPU-scope increment (orders this CTA's claims before it)."""
    code = "atom.acq_rel.gpu.global.add.u32 $0,[$1],1;"
    return code, "=r,l,~{memory}", (pointer,)


# Route mailbox (``RubinExpertsKernel.polled``), byte offsets in the experts'
# ``ws_xflag`` words (the FC2(B) pool uses the first two, ``dies`` the next two): the
# routing kernel's 128 expert ids, then the count of its copy CTAs that wrote their
# part of the image plan.
# The experts read it before their dependency wait, so it carries its own protocol:
# - Armed: all ones (ids never are; the count is -1). The workspace starts all ones.
# - Written once per MoE layer, only by a polled batch's routing kernel: the route
#   CTAs store the ids (each word valid on its own), each copy CTA adds 1 to the count
#   with a release after a CTA barrier that follows its image stores (count 2B - 1:
#   every image byte is visible to an acquire of it).
# - Consumed by every experts CTA of the same layer: warp 0 polls the ids until no
#   word is all ones; warp 3 acquires the count, then the image is copied after a
#   proxy fence. Anything else from routing or the shared expert (routing weights,
#   shared rows) is read only after ``griddepcontrol.wait``.
# - Re-armed by the experts' last CTA (the acq_rel claim the producer makes after its
#   ids poll and after warp 3's count read, through the xcount barrier, sees every
#   other CTA's claim). The next write is the next layer's routing kernel, which runs
#   after this grid completed: every kernel in between waits on its predecessor.
# Other batches never touch these words; a polled batch's experts always re-arm them.
ROUTE_IDS, ROUTE_IMAGE = 256, 768
ROUTE_WORDS = 132  # the mailbox view: ids, count, padding to 16 bytes


@asm()
def count_image(pointer):
    """Routing copy CTA, one thread after a CTA barrier: the CTA's part of the image
    plan is written (the release is cumulative over the barrier)."""
    return "red.release.gpu.global.add.u32 [$0],1;", "l,~{memory}", (pointer,)


@asm(Uint32)
def load_acquire(pointer):
    return "ld.acquire.gpu.global.u32 $0,[$1];", "=r,l,~{memory}", (pointer,)


@asm()
def fence_proxy_global():
    """Generic-proxy writes this thread observed, before its bulk copies read them."""
    return "fence.proxy.async.global;", "~{memory}", ()


@asm()
def store_stage_half(pointer, value, store):
    """``value`` as BF16 with ``pack_pair``'s "stage" NaN canonicalization (per half),
    stored as one 16-bit global word if ``store`` != 0 (predicated, no branch)."""
    code = "{ .reg .b32 w,bits; .reg .b16 h; .reg .pred p,q; setp.ne.u32 q,$2,0; "
    code += "cvt.rn.bf16x2.f32 w,0f00000000,$1; and.b32 bits,w,0x7fff; "
    code += "setp.gt.u32 p,bits,0x7f80; @p mov.b32 w,0x7fc0; cvt.u16.u32 h,w; "
    code += "@q st.global.b16 [$0],h; }"
    return code, "l,f,r,~{memory}", (pointer, Float32(value), Uint32(store))


@asm()
def store_shared_if(address, value, store, kind="u8"):
    """Predicated shared store of ``value``'s low ``kind`` bits (no branch)."""
    code = f"{{ .reg .pred q; setp.ne.u32 q,$2,0; @q st.shared.{kind} [$0],$1; }}"
    return code, "r,r,r,~{memory}", (address, Uint32(value), Uint32(store))


@asm(Uint32, pure=True)
def select_u32(cond, a, b):
    code = "{ .reg .pred q; setp.ne.u32 q,$1,0; selp.b32 $0,$2,$3,q; }"
    return code, "=r,r,r,r", (Uint32(cond), Uint32(a), Uint32(b))


@cute.jit
def mx_exponent_select(amax: Float32):
    """``mx_exponent`` (not floored) with a select instead of branches."""
    bits = Uint32(float_bits(amax))
    e = Int32(bits >> 23)
    rounded = min(max(e - 8 + Int32(bits & 8388607 > 6291456), 0), 254)
    return Int32(select_u32(Uint32(e != 0), Uint32(rounded), Uint32(0)))


@cute.jit
def mx_rescale_select(scale: Int32):
    """``mx_rescale`` with a select instead of a branch."""
    e = 254 - scale
    return bits_float(select_u32(Uint32(e >= 1), Uint32(e) << 23, Uint32(0)))


@cute.jit
def has_ff_half(word):
    """A half of a staged word still holds the 0xffff re-arm pattern: the rows land
    half by half, and a canonicalized BF16 half is never 0xffff."""
    return ((word & Uint32(65535)) == Uint32(65535)) | (word >> 16 == Uint32(65535))


def die_words(dies):
    """The SMs' dies (``common.dies.sm_domains``) as 32-bit words: bit i of word w is
    set when smid 32 w + i sits on die 1."""
    words = (len(dies) + 31) // 32
    bit = lambda i: 1 if i < len(dies) and dies[i] else 0
    return tuple(sum(bit(32 * w + i) << i for i in range(32)) for w in range(words))


@asm(Uint32)
def sm_id():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Uint32, pure=True)
def nth_set(word, n):
    """Bit position of the ``n``-th (0-based) set bit of ``word`` (``fns``)."""
    return "fns.b32 $0,$1,0,$2;", "=r,r,r", (Uint32(word), Int32(n + 1))


@cute.jit
def match_rank(bits, grid: cutlass.Constexpr[int], die, k):
    """The partition rank the k-th CTA (arrival order) on ``die`` runs: ``bits`` (one
    word per 32 ranks, bit set = the rank's middle weights sit on die 1); the k-th rank
    homed on its die, or past them (a spare) the other die's ranks from the end. The
    other die's CTAs take its ranks from the front, and a die's spares number exactly
    the other die's ranks without a CTA there: every rank runs once
    (``experts_rubin64.match_arrival``)."""
    words: cutlass.Constexpr = (grid + 31) // 32
    h1 = Int32(0)
    for w in cutlass.range_constexpr(words):
        valid = grid - 32 * w
        mask = Uint32(4294967295) if valid >= 32 else Uint32((1 << valid) - 1)
        h1 += Int32(cute.arch.popc(bits[w] & mask))
    own, other = (grid - h1, h1)
    if die == 1:
        own, other = (h1, grid - h1)
    target, index = (die, k)
    if k >= own:
        target, index = (1 - die, other - 1 - (k - own))
    rank = Int32(-1)
    for w in cutlass.range_constexpr(words):
        valid = grid - 32 * w
        mask = Uint32(4294967295) if valid >= 32 else Uint32((1 << valid) - 1)
        word = bits[w]
        if target == 0:
            word = ~word
        word &= mask
        count = Int32(cute.arch.popc(word))
        if (rank < 0) & (index < count):
            rank = 32 * w + Int32(nth_set(word, index))
        elif rank < 0:
            index -= count
    return rank


@asm()
def stamp_at(pointer, slot):
    """Benchmark trace: (%globaltimer, %smid) into 16 B ``slot`` of ``pointer``."""
    code = "{ .reg .u64 t; .reg .u32 s; mov.u64 t, %globaltimer; mov.u32 s, %smid; "
    code += "st.global.v2.u64 [$0], {t, 0}; st.global.u32 [$0+8], s; }"
    return code, "l,~{memory}", (pointer + 16 * slot,)


@asm()
def stage16(
    mma, weight_scales, activation_scales, release, done, *more, kind, early, part=None,
    packed=False,
):  # fmt: skip
    """``experts.stage`` for TP16 ("fc1": 8 K32 MMAs, "fc2": 6); ``early`` releases the
    weight slot when the MMAs have read A (their scales were copied before them).
    ``part`` 0 / 1: only the stage's first / second K128 box (MMAs 0-3 / 4-), with that
    box's scale copies; part 0 skips ``done``, part 1 the hidden release. ``more``:
    one more barrier released with the slot (a whole stage on a box-by-box ring).
    ``packed``: the weights are packed E2M1 rows (128 B = K256, 128B-swizzled) and the
    MMAs are Rubin's K64 ones (instruction descriptor bit 31), 32 B of A each, with two
    UE8M0 scales per row from bytes ``k % 4`` and ``k % 4 + 1`` of the scale column."""
    steps = 6 if kind == "fc2" else 8
    ks = range(steps) if part is None else range(4 * part, min(4 * part + 4, steps))
    ks = [k for k in ks if not packed or k % 2 == 0]  # K64: every other K32 step
    halves = (0, 1) if part is None else (part,)
    cp = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    arrive = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    free = arrive
    if early:
        free = "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        free += ".sync_restrict::shared::read::mma::a.b64"
    code = [
        "{ .reg .pred e,p0,pt,pc,pa,ph; .reg .b32 i<4>; .reg .b64 as1,bs1;",
        f".reg .b32 af1,bf1; .reg .b64 xa<{steps}>,xb<{steps}>;",
        "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0;",
        "and.pred pc,pc,e; setp.ne.b32 pa,$10,0; and.pred pa,pa,e;",
    ]
    if kind == "fc1":
        code.append("setp.ne.b32 p0,$8,0;")
    else:
        code.append("setp.eq.b32 p0,0,1; setp.ne.b32 ph,$8,0; and.pred ph,ph,e;")
        if part != 1:
            code.append(f"@ph {arrive} [$8];")
    k64 = 1 << 31 if packed else 0
    code += [f"mov.b32 i{k},{142738048 | k64 | k << 29 | k << 4:#x};" for k in range(4)]
    code.append("add.s64 as1,$4,32; add.s64 bs1,$6,8;")
    code.append("add.u32 af1,$3,4; add.u32 bf1,$5,4;")
    code += [(f"@pc {cp} [$5],$6;", f"@pc {cp} [bf1],bs1;")[h] for h in halves]
    code += [(f"@e {cp} [$3],$4;", f"@e {cp} [af1],as1;")[h] for h in halves]
    for k in ks:
        a = k // 4 * 1024 + k % 4 * 2 + (4 if kind == "fc2" and k >= 4 else 0)
        if packed:
            a = k
        b = k // 4 * 64 + k % 4 * 2
        code.append(f"add.s64 xa{k},$1,{a}; add.s64 xb{k},$2,{b};")
    # Packed: each K64 MMA sums two 32-element blocks the K32 path summed in two MMAs
    # (an accumulation-grouping reorder inside the MMA; same inputs, FP32 accumulator).
    for k in ks:
        af, bf = ("$3", "$5") if k < 4 else ("af1", "bf1")
        code.append(
            "@e tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
            f"[$0],xa{k},xb{k},i{k % 4},[{af}],[{bf}],{'p0' if k == 0 else 'pt'};"
        )
    code += [f"@e {free} [$9];"] + ([f"@e {free} [$11];"] if more else [])
    code += ([f"@pa {arrive} [$10];"] if part != 0 else []) + ["}"]
    kinds = "rllrlrlrrrr" + "r" * len(more)
    operands = (*mma, *weight_scales, *activation_scales, *release, done, *more)
    values = [(Uint64 if k == "l" else Uint32)(v) for k, v in zip(kinds, operands)]
    return "\n".join(code), ",".join(kinds) + ",~{memory}", values


class RubinExpertsKernel(ExpertsKernel):
    """TP16 capacity-8 ``ExpertsKernel`` (streamed producer, no grouping). Options:
    ``stages`` weight slots (6 fits only the oversized shared memory mode, which
    forfeits the PDL prologue overlap); ``early`` slot release at A read (no gain
    measured); ``rotate``d latent stages; ``grain``/``pool``: the FC2(B) pool (``pool``
    quarters of group B's units, claimed ``grain`` tiles at a time; 0 = static);
    ``planned``: x is the routing kernel's image plan; ``stamp``: benchmark
    timestamps at that workspace offset; ``publish``: ``ExpertsKernel``'s False / True
    (the WaveMoeTail latent) / "push" (and the shared rows, after the PDL wait);
    ``grid_maps``: grid-constant weight descriptors (``common.tensormap``) instead of
    the acquired global ones; ``halves`` (grid maps only): each weight slot is two K128
    boxes on their own full / empty barriers, consumed and released box by box, so the
    producer refills a slot's first half while the MMAs read its second;
    ``late_push`` (push mode): the rearm warp stores the shared rows once the CTA's
    first latent stage landed, instead of the epilogue warps right after the
    dependency wait. Published modes: once the producer stops claiming, each line's
    finalize warp loads the line's staged rows and routing weights before its group's
    last accumulators, so a line whose rows landed before its CTA's own last tiles is
    published without a poll round trip after the epilogue. ``packed``: the weights
    land packed (one 128 B x 128-row box per FC1 stage, the 96 data bytes of each w2
    row per FC2 tile, 16 KiB slots, whole stages) and feed K64 MMAs. Bitwise identical
    to ``ExpertsKernel`` without ``packed``; with it the MMAs sum K64 instead of K32
    groups (same inputs and FP32 accumulator, a different grouping inside the MMA).
    ``static_groups``: ``ExpertsKernel.group_sync``'s constant barrier ids.
    ``fast_exchange``: the K-split exchange through
    parity-buffered row-major peer slots (``owner_sums``): each peer publishes its rows
    with two 16 B stores, the owner loads its first ``PEER_BATCH`` peers' rows in one
    round (two 16 B loads each) and adds them in CTA order (``partials``' FP32 sums),
    and every CTA re-arms its own two slots of the other parity in the next call, as
    the hidden images are (no owner stores, nothing re-armed on the critical path).
    With it the images and peer slots alternate on this kernel's own call generation
    (a word per slot past the peer slots), not on the call counter every experts
    kernel advances: the workspace is shared across batch sizes, so an odd number of
    other experts calls between two of this kernel's would otherwise hand it a parity
    whose images and slots its previous call left written."""

    def __init__(
        self, experts, grid, slots, stages=None, early=False, rotate=True, stamp=0,
        grain=1, pool=1, planned=True, publish=False, grid_maps=False, halves=True,
        late_push=True, lean_route=True, xearly=0, polled=False, packed=True,
        dies=(), static_groups=False, fast_exchange=False,
    ):  # fmt: skip
        assert publish != "twoshot"
        self.static_groups, self.fast_exchange = static_groups, fast_exchange
        super().__init__(16, 8, experts, grid, publish, True, False, False, slots)
        if fast_exchange:
            from .experts_rubin64 import CENSUS_BYTES

            assert self.width == 8
            self.ws_peers = peers_offset(self.workspace_bytes, CENSUS_BYTES)
            self.ws_generation = self.ws_peers + peers_bytes(slots)
            self.workspace_bytes = self.ws_generation + generation_bytes(slots)
        # grid_maps: the w13 / w2 descriptors are encoded at launch from the weight
        # addresses (two Int64 arguments before the stream) and passed as
        # grid-constant parameters, so the prologue needs no tensormap proxy fence.
        self.grid_maps = grid_maps
        # Box-by-box ring (the half-box descriptors are encoded with the grid maps).
        self.packed = packed and grid_maps
        self.parts = 2 if halves and grid_maps and not self.packed else 1
        # Six packed slots: more bytes in flight only add memory-system queueing here.
        stages = stages or (6 if self.packed else 5)
        self.early, self.rotate, self.stamp = early, rotate, stamp
        # FC2(B) pool: the last half of group B's units, claimed in ``grain``-tile
        # grains through two counters (claims, finished CTAs) in the otherwise unused
        # ``ws_xflag`` words; the last CTA to finish re-arms both to -1.
        self.grain, self.pool, self.late_push = grain, pool, late_push
        # One finalize line per epilogue warp (``epilogue``).
        assert not publish or grid >= 56
        # planned: x is the routing kernel's image plan (``IMAGE_BYTES``: the SMEM
        # latent image; batch 8 only, as rows past the batch are not written).
        self.planned = planned
        # lean_route: warp 0 dedupes the routes alone (``route_warp``; the table is
        # cleared before the dependency wait, in the hidden image buffers) and its
        # producer lane issues without a CTA barrier; xearly (planned): the latent
        # image in four bulk copies on one barrier as soon as it is complete (1)
        # instead of warp 3's 28 per-stage copies after the partition (0).
        self.lean_route, self.xearly = lean_route, xearly if planned else 0
        assert not lean_route or self.h_image_bytes * 2 >= 4 * experts
        # polled: the routing kernel also writes the ids and its copy CTAs' count into
        # the route mailbox (``ROUTE_IDS``), so warp 0 polls the ids and dedupes,
        # partitions and streams before the dependency wait, and warp 3 copies the
        # image once the count is complete; only the shared-row push and the routing
        # weights' reads wait for the predecessors' completion.
        self.polled = polled
        if polled:
            assert lean_route and planned
            self.xearly = 1
            self.ws_route = self.ws_xflag + ROUTE_IDS
            assert 4 * slots >= ROUTE_IMAGE + 4
        assert lean_route or not self.xearly  # its barrier init needs route_init's sync
        assert grain == 0 or 28 % grain == 0
        self.stages = stages
        # dies (the SMs' dies): the routed w2 sits in 2 MiB chunks alternating die 0 /
        # 1 (``experts_runtime.place``); the k-th CTA of the call on die d (an arrival
        # ticket per die, ``match_fc2``) runs the k-th FC2(A) / static FC2(B) partition
        # rank whose middle tile sits on die d, the CTAs past a die's ranks the other
        # die's from the end (the hidden loader warp matches them during FC1; the
        # producer picks them up before FC2). Every rank runs once wherever the CTAs
        # land (two CTAs in turn on one SM, an SM without one). FC2 tiles are
        # independent: same outputs.
        self.dies = tuple(dies)
        if dies:
            assert len(dies) >= grid
            self.die_bits = die_words(dies)
        self.slot_bytes = 16384 if self.packed else 32768
        self.off_sfa = stages * self.slot_bytes
        self.off_x = self.off_sfa + stages * 1024
        self.off_h = self.off_x + 28672
        self.off_xsf = self.off_h + 2 * self.h_image_bytes
        self.off_hsf = self.off_xsf + 28 * 128 + 384
        self.off_misc = self.off_hsf + 2 * self.h_scale_bytes
        assert self.off_misc <= DESCRIPTOR_LIMIT
        # At least the unpacked five-slot size: one CTA per SM (two co-resident CTAs
        # would serialize in tcgen05.alloc while waiting on each other's partials).
        self.smem_bytes = max(self.off_misc + 5120, 212096)
        # ``ExpertsGeometry`` placed the unit queue for five stages; it follows misc.
        self.off_queue = self.off_misc + 4608
        ring = self.parts * stages
        self.bar_full, self.bar_empty, self.bar_accfull = (0, 8 * ring, 16 * ring)
        acc = self.bar_accfull
        self.bar_accempty, self.bar_hfull = (acc + 128, acc + 256)
        self.bar_hempty, self.bar_xready = (acc + 272, acc + 288)
        # + xcount (polled): warp 3 saw the routing kernel's image count.
        self.bar_xcount = acc + 400
        self.barriers = 2 * ring + 51
        assert 8 * self.barriers <= 600  # below the accumulator metadata
        # FC2 stage metadata after the epilogue's staging rows (misc 3200-4223).
        self.off_meta = self.off_misc + 4224
        # Producer done claiming (a word the finalize warps read; misc 4320).
        self.off_done = self.off_misc + 4224 + max(96, 16 * stages)
        assert self.off_done + 4 <= self.off_queue

    @cute.jit
    def initialize(self, sm, maps, batch):
        """``maps``: the (w13, w2) descriptor addresses."""
        tid, _, _ = cute.arch.thread_idx()
        barriers = view(sm.iterator, self.off_misc, Uint64, self.barriers)
        tmem_base = view(sm.iterator, self.off_misc + 856, Uint32, 1)
        if tid == 0:
            acc = 2 * self.parts * self.stages
            for i in cutlass.range_constexpr(self.barriers):
                count = 1 if not acc <= i < acc + 32 else 2 if i < acc + 16 else 4
                if acc + 36 <= i < acc + 50:
                    count = 1 if self.planned else batch * 8
                cute.arch.mbarrier_init(barriers.iterator + i, count)
            queue = view(sm.iterator, self.off_misc + self.bar_queue, Uint64, 16)
            for i in cutlass.range_constexpr(16):
                cute.arch.mbarrier_init(queue.iterator + i, 1)
            if cutlass.const_expr(len(self.dies) > 0):  # the FC2 ranks are matched
                ready = view(sm.iterator, self.off_misc + 4864, Uint64, 1)
                cute.arch.mbarrier_init(ready.iterator, 1)
            cute.arch.mbarrier_init_fence()
            store_shared_words(sm.iterator + self.off_done, (Uint32(0),))
            for descriptor in maps:
                if cutlass.const_expr(self.grid_maps):
                    prefetch_map(descriptor)  # parameter space: no proxy fence
                else:
                    acquire_tensormap(descriptor)
        if tid // 32 == 3:
            cute.arch.alloc_tmem(512, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        if cutlass.const_expr(not self.planned):
            self.clear_x(sm, tid, batch)
        cute.arch.fence_proxy("async.shared", space="cta")
        return tmem_base

    @cute.jit
    def route_init(self, sm, counters, generations=None):
        """``lean_route`` before the dependency wait: the first-occurrence table (in
        the hidden image buffers, which the hidden loader overwrites before any MMA
        reads them), the unit token slots and the call parity (the previous call's
        counter: that call completed before this launch, as every kernel since then
        waited on its predecessor); ``generations``: also this kernel's own call
        generation (misc 4880)."""
        tid, _, _, cta = thread()
        table = view(sm.iterator, self.off_h, Int32, self.experts)
        for i in range(tid, self.experts, 384):
            table[i] = Int32(2147483647)
        slot_words = view(sm.iterator, self.off_misc + 2176, Uint32, 256)
        if tid < 256:
            slot_words[tid] = Uint32(4294967295)
        if tid == 256:
            parity = view(sm.iterator, self.off_misc + 864, Int32, 1)
            parity[0] = counters[cta]
        if cutlass.const_expr(generations is not None):
            own = view(sm.iterator, self.off_misc + 4880, Int32, 1)
            if tid == 257:
                own[0] = generations[cta]
        cute.arch.sync_threads()

    @cute.jit
    def route_warp(self, sm, ids, batch):
        """``ExpertsKernel.route`` (capacity > 2) by warp 0 alone after ``route_init``,
        without CTA barriers: lane l holds routing pairs 4l..4l+3, so first occurrences
        rank in the same order (the same units, token slots and distinct count).
        Returns the distinct count."""
        lane = thread()[2]
        misc = self.off_misc
        table = view(sm.iterator, self.off_h, Int32, self.experts)
        expert, uof = (view(sm.iterator, misc + o, Int32, 128) for o in (1140, 1652))
        slots = view(sm.iterator, misc + 2176, Int8, 1024)
        base = 4 * lane
        valid = base < batch * 16
        mine = cute.make_rmem_tensor(4, Int32)
        for j in cutlass.range_constexpr(4):
            mine[j] = Int32(-1)
        if cutlass.const_expr(self.polled):  # Lamport: all ones until written
            ready = False
            while not ready:
                words = load_words(ids + base, count=4, scope="gpu")
                for j in cutlass.range_constexpr(4):
                    mine[j] = Int32(words[j])
                armed = (mine[0] == -1) | (mine[1] == -1)
                armed = armed | (mine[2] == -1) | (mine[3] == -1)
                ready = cute.arch.vote_all_sync((not valid) | (not armed))
                if not ready:
                    sleep_ns(32)
            if not valid:
                for j in cutlass.range_constexpr(4):
                    mine[j] = Int32(-1)
        elif valid:
            words = load_words(ids.iterator + base, count=4, mutable=True)
            for j in cutlass.range_constexpr(4):
                mine[j] = Int32(words[j])
        if valid:
            for j in cutlass.range_constexpr(4):
                cute.arch.atomic_min(
                    table.iterator + mine[j], base + j, sem="relaxed", scope="cta"
                )
        cute.arch.sync_warp()
        firstk = cute.make_rmem_tensor(4, Int32)
        rank, distinct = (Int32(0), Int32(0))
        below = (Uint32(1) << lane) - 1
        for j in cutlass.range_constexpr(4):
            firstk[j] = Int32(-1)
            if valid:
                firstk[j] = table[mine[j]]
            first = valid & (firstk[j] == base + j)
            ballot = Uint32(cute.arch.vote_ballot_sync(first))
            rank += Int32(cute.arch.popc(ballot & below))
            distinct += Int32(cute.arch.popc(ballot))
        for j in cutlass.range_constexpr(4):
            if valid & (firstk[j] == base + j):
                uof[base + j] = rank
                expert[rank] = mine[j]
                rank += 1
        cute.arch.sync_warp()
        if valid:
            for j in cutlass.range_constexpr(4):
                pair = base + j
                slots[uof[firstk[j]] * 8 + pair // 16] = Int8(pair % 16)
        return distinct

    @cute.jit
    def load_image(self, sm, plan, cta):
        """One lane (``xearly``): the whole image plan in four bulk copies on xready[0],
        from stage ``cta`` % 14 on, so the CTAs start on different lines."""
        barrier = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 1).iterator
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, IMAGE_BYTES)
        head = cta % 14
        for part in cutlass.range_constexpr(2):
            dst, src = (sm.iterator + self.off_x, plan.iterator)
            width = 2048
            if cutlass.const_expr(part == 1):
                dst, src = (sm.iterator + self.off_xsf, plan.iterator + IMAGE_XSF)
                width = 256
            cut = head * width
            copy_g2s(dst + cut, src + cut, 14 * width - cut, barrier)
            if head > 0:
                copy_g2s(dst, src, cut, barrier)

    @cute.jit
    def load_plan(self, sm, plan, shift):
        """Lanes 0-13 of a warp: latent stage (``shift`` + lane) % 14's image and
        scales from the image plan (two bulk copies); xready[stage] completes on
        them."""
        lane = thread()[2]
        if lane < 14:
            stage = shift + lane
            if stage >= 14:
                stage -= 14
            ready = view(sm.iterator, self.off_misc + self.bar_xready, Uint64, 14)
            barrier = ready.iterator + stage
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2304)
            image = sm.iterator + self.off_x + stage * 2048
            copy_g2s(image, plan.iterator + stage * 2048, 2048, barrier)
            scales = sm.iterator + self.off_xsf + stage * 256
            copy_g2s(scales, plan.iterator + IMAGE_XSF + stage * 256, 256, barrier)

    @cute.jit
    def owned(self, index: cutlass.Constexpr[int], batch, shift):
        """``ownership`` with stages rotated by ``shift`` (round 0 holds the CTA's
        first FC1 stage and the four after it)."""
        xtid = thread()[0] - 64
        stage = 5 * index + xtid // 64
        token, kblock = ((xtid & 63) // 8, xtid & 7)
        valid = (stage < 14) & (token < batch)
        stage += shift
        if stage >= 14:
            stage -= 14
        return (stage, token, kblock, valid)

    @cute.jit
    def load_x(self, x, raw, batch, shift):
        raw.fill(0)
        for index in cutlass.range_constexpr(self.rounds):
            stage, token, kblock, valid = self.owned(index, batch, shift)
            if valid:
                source = x.iterator + token * 3584 + stage * 256 + kblock * 32
                for fragment in cutlass.range_constexpr(4):
                    words = load_words(
                        source + fragment * 8, count=4, cache_policy="", mutable=True
                    )
                    for j in cutlass.range_constexpr(4):
                        raw[index, fragment, j] = words[j]

    @cute.jit
    def quantize_x(self, sm, raw, barriers, batch, shift):
        for index in cutlass.range_constexpr(self.rounds):
            stage, token, kblock, valid = self.owned(index, batch, shift)
            if valid:
                packed, scale = mx_block([raw[index, i // 4, i % 4] for i in range(16)])
                for half, words in enumerate((packed[:4], packed[4:])):
                    offset = swizzled(16 * stage + 2 * kblock + half, token, self.off_x)
                    store_shared_words(sm.iterator + offset, words)
                kb = 8 * stage + kblock
                sm[self.off_xsf + kb // 4 * 128 + token * 16 + kb % 4] = Uint8(scale)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(barriers + stage)

    @cute.jit
    def produce16(self, sm, maps, w13_scales, w2_scales, ranges, experts, workspace):
        """``stream`` for TP16 capacity 8: FC1(A), FC1(B) stages, the static FC2(A)
        and FC2(B) tiles, then claimed FC2(B) pool grains; every unit change is queued
        for the hidden loader (-1 ends). FC1 stages carry no metadata (``consume``
        walks them from the partition)."""
        stages, (w13, w2) = (self.stages, maps)
        barriers = view(sm.iterator, self.off_misc, Uint64, 2 * self.parts * stages)
        metadata = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        ring = (sm, barriers, metadata, w2_scales)
        slot, phase, issued = (Int32(0), Int32(0), Int32(0))
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, tile = (lo % 14, 3 * ranges[group, 0] + lo // 14)
            r, index = (tile % 3, self.first_unit(ranges, 0, group))
            expert = experts[index]
            for s in cutlass.range(lo, hi, unroll=1):
                row = expert * self.w13_rows + r * 128
                offset = Int64(expert) * self.w13_scale_bytes + r * 14336 + ks * 1024
                chunk = ks if self.packed else 2 * ks
                tma = (w13, w13_scales, row, chunk, offset, Int32(1024))
                slot, phase, issued = self.issue16(ring, slot, phase, issued, None, tma)
                ks += 1
                if ks == 14:
                    ks, tile, r = (Int32(0), tile + 1, r + 1)
                    if r == 3:
                        r, index = (Int32(0), index + 1)
                        expert = experts[index]
        pool = workspace.iterator + self.ws_xflag
        units = ranges[1, 1] // 42
        pooled = units * self.pool // 4 if self.grain else Int32(0)
        grain = max(self.grain, 1)
        base, grains = (28 * (ranges[1, 0] + units - pooled), 28 * pooled // grain)
        pending = claim(pool, 1)  # in flight during the static FC2 jobs
        if cutlass.const_expr(len(self.dies) > 0):
            ready = view(sm.iterator, self.off_misc + 4864, Uint64, 1)
            self.sync(ready.iterator, 0)
            ranks = view(sm.iterator, self.off_misc + 4856, Int32, 2)
            distinct = view(sm.iterator, self.off_misc + 860, Int32, 1)[0]
            for g in cutlass.range_constexpr(2):
                fc2 = self.partition(distinct, ranks[g])
                ranges[g, 4], ranges[g, 5] = (fc2[g, 4], fc2[g, 5])
        lo, hi, step = (ranges[0, 4], ranges[0, 5], Int32(0))
        pushed, current, expert = (Int32(0), Int32(-1), Int32(0))
        active = True
        while active:
            for j in cutlass.range(lo, hi, unroll=1):
                unit, tile = (j // 28, j % 28)
                if unit != current:
                    pushed = self.push_unit(sm, pushed, unit)
                    current, expert = (unit, experts[unit])
                row = expert * 3584 + tile * 128
                offset = Int64(expert) * self.w2_scale_bytes + tile * 1024
                tma = (w2, w2_scales, row, Int32(0), offset, Int32(1024))
                meta = (Int32(1), unit, tile, Int32(0))
                work = (ring, slot, phase, issued, meta, tma)
                slot, phase, issued = self.issue16(*work, whole=True)
            if step == 0:
                lo, hi = (ranges[1, 4], ranges[1, 5])
            else:
                index = Int32(pending + 1)
                if index < grains:
                    pending = claim(pool, 1)
                    lo = base + index * grain
                    hi = lo + grain
                else:
                    active = False
                    store_shared_words(sm.iterator + self.off_done, (Uint32(1),))
            step += 1
        # The ring's end marker first: neither the hidden loader's queue (the
        # producer can run 8 units ahead of it) nor the finished-CTA count (an acq_rel
        # round trip) delays the epilogue's end and the lines' finalize.
        parts = self.parts
        if issued >= stages:
            self.sync(barriers.iterator + parts * (stages + slot), phase ^ 1)
        for i in cutlass.range_constexpr(4):
            metadata[4 * slot + i] = Int32(2 if i == 0 else 0)
        cute.arch.mbarrier_arrive(barriers.iterator + parts * slot)
        self.push_unit(sm, pushed, Int32(-1))
        if cutlass.const_expr(self.polled):
            # Warp 3's read of the mailbox's image count precedes the release below
            # (this lane read the ids itself), so the last CTA may re-arm it.
            xcount = view(sm.iterator, self.off_misc + self.bar_xcount, Uint64, 1)
            self.sync(xcount.iterator, 0)
        if Int32(claim_last(pool + 4)) == self.grid - 2:
            if cutlass.const_expr(len(self.dies) > 0):  # + the die tickets
                store_words(pool, (Uint32(4294967295),) * 4)
            else:
                store_words(pool, (Uint32(4294967295), Uint32(4294967295)))
            if cutlass.const_expr(self.polled):  # every CTA has read the mailbox
                mailbox = workspace.iterator + self.ws_route
                for i in cutlass.range_constexpr(ROUTE_WORDS // 4):
                    store_words(mailbox + 16 * i, (Uint32(4294967295),) * 4)

    @cute.jit
    def issue16(self, ring, slot, phase, issued, meta, tma, whole=False):
        """``ExpertsKernel.issue`` for TP16 stages (1 KiB scales), box by box when
        ``parts`` is 2: each K128 box with its 512 scale bytes on its own barrier;
        a ``whole`` stage (FC2) is one box on the first barrier once both are free.
        ``meta`` None: no stage metadata (FC1)."""
        sm, barriers, metadata = ring[:3]
        desc, scales, row, chunk, offset, size = tma
        stages, parts = (self.stages, self.parts)
        if cutlass.const_expr(whole and parts == 2):
            if issued >= stages:
                for part in cutlass.range_constexpr(2):
                    empty = barriers.iterator + 2 * (stages + slot) + part
                    self.sync(empty, phase ^ 1)
            for i in cutlass.range_constexpr(4):
                metadata[4 * slot + i] = meta[i]
            full = barriers.iterator + 2 * slot
            cute.arch.mbarrier_arrive_and_expect_tx(full, 16384 + size)
            load_tma3(sm.iterator + slot * 32768, desc, full, row, chunk)
            sf = sm.iterator + self.off_sfa + slot * 1024
            copy_evict(sf, scales.iterator + offset, 1024, full)
            cute.arch.mbarrier_arrive(full + 1)  # completes the second box's phase
        for part in cutlass.range_constexpr(0 if whole and parts == 2 else parts):
            index = parts * slot + part
            if issued >= stages:
                self.sync(barriers.iterator + parts * stages + index, phase ^ 1)
            if cutlass.const_expr(part == 0 and meta is not None):
                for i in cutlass.range_constexpr(4):
                    metadata[4 * slot + i] = meta[i]
            full = barriers.iterator + index
            if cutlass.const_expr(parts == 1):
                box = 12288 if whole and self.packed else 16384
                cute.arch.mbarrier_arrive_and_expect_tx(full, box + size)
                self.fetch(sm, desc, scales, barriers, slot, row, chunk, offset, size)
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(full, 8192 + 512)
                dst = sm.iterator + slot * 32768 + part * 16384
                load_tma3(dst, desc, full, row, chunk + part)
                sf = sm.iterator + self.off_sfa + slot * 1024 + part * 512
                copy_evict(sf, scales.iterator + offset + part * 512, 512, full)
        slot += 1
        if slot == stages:
            slot, phase = (Int32(0), phase ^ 1)
        return slot, phase, issued + 1

    @cute.jit
    def fetch(self, sm, desc, scales, full, slot, row, chunk, offset, size, more=None):
        """``ExpertsKernel.fetch`` (one scale run) into ``slot_bytes`` slots."""
        dst = sm.iterator + slot * self.slot_bytes
        load_tma3(dst, desc, full.iterator + slot, row, chunk)
        dst = sm.iterator + self.off_sfa + slot * 1024
        copy_evict(dst, scales.iterator + offset, size, full.iterator + slot)

    @cute.jit
    def partition(self, distinct: Int32, cta: Int32):
        """``ExpertsKernel.partition`` with FC2(B) limited to its static units."""
        # Explicit base call: the DSL resolves a zero-argument super() through
        # type(self), which recurses in subclasses.
        ranges = ExpertsKernel.partition(self, distinct, cta)
        if cutlass.const_expr(self.grain > 0):
            start, count = (ranges[1, 0], ranges[1, 1] // 42)
            count -= count * self.pool // 4
            ranges[1, 4] = 28 * start + cta * 28 * count // self.grid
            ranges[1, 5] = 28 * start + (cta + 1) * 28 * count // self.grid
        return ranges

    @cute.jit
    def epilogue(self, sm, workspace, output, tmem, ranges, batch, parity, scalars):
        """``ExpertsKernel.epilogue``. Published modes: ``ExpertsKernel.finalize``'s
        line per warp (lines stride over the first CTAs' epilogue warps), with its 16
        staged-row words and routing weights loaded once the producer stops claiming,
        before the group's last accumulators; after the end marker the line is
        published at once if those words are all written, else polled as before."""
        _, warp, lane, cta = thread()
        ew, eg = (warp & 3, (warp - 4) // 4)
        full = view(sm.iterator, self.off_misc + self.bar_accfull, Uint64, 16)
        empty = view(sm.iterator, self.off_misc + self.bar_accempty, Uint64, 16)
        meta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        line = cta * 8 + warp - 4
        pending = line < batch * 56
        if cutlass.const_expr(not self.publish):
            pending = line < 0
        token, column = (line // 56, line % 56 * 64)
        source = workspace.iterator + self.ws_stage
        source += (Int64(token) * 16 * 1792 + column // 2 + lane) * 4
        words = cute.make_rmem_tensor(16, Uint32)
        scale = cute.make_rmem_tensor(16, Float32)
        loaded = Int32(0)
        acc, phase, ended = (eg, Int32(0), False)
        while not ended:
            if pending & (loaded == 0):
                done = load_words(sm.iterator + self.off_done, count=1, shared=True)
                if done[0] != Uint32(0):
                    if cutlass.const_expr(self.polled):
                        cute.arch.griddepcontrol_wait()  # the routing weights
                    for slot in cutlass.range_constexpr(16):
                        scale[slot] = Float32(output[token * 16 + slot])
                        words[slot] = load_volatile(source + slot * 7168)[0]
                    loaded = Int32(1)
            self.sync(full.iterator + acc, phase, True)
            kind, a, b = (meta[4 * acc], meta[4 * acc + 1], meta[4 * acc + 2])
            if kind == 2:
                ended = True
            else:
                raw = load_tmem(tmem + (Uint32(32 * ew) << 16) + acc * 8, 8)
                values = cute.make_rmem_tensor(self.width, Float32)
                for t in cutlass.range_constexpr(self.width):
                    values[t] = raw[t]
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty.iterator + acc)
                acc += 2
                if acc >= 16:
                    acc -= 16
                    phase ^= 1
                if kind == 0:
                    if self.exchange(workspace, values, a, ranges, batch, parity):
                        store = (sm, workspace, values, a, parity, scalars[:3])
                        self.store_hidden(*store)
                else:
                    self.fc2(sm, workspace, output, values, a, b, ranges)
        fence_tmem(before=True)
        named_barrier(TMEM_DONE, 320, arrive=True)
        if cutlass.const_expr(self.publish):
            if pending:
                box = scalars[3:6]
                self.publish_line(words, scale, loaded, line, source, output, box)

    @cute.jit
    def exchange(self, workspace, values, tile, ranges, batch, parity):
        """The K-split exchange of an FC1 accumulator: True for the tile's owner, whose
        ``values`` then hold the tile's sums."""
        if cutlass.const_expr(self.fast_exchange):
            owner = self.owner_sums(workspace, values, tile, ranges, parity)
        else:
            owner = self.partials(workspace, values, tile, ranges, batch)
        return owner

    @cute.jit
    def peer_slot(self, workspace, parity, group, cta):
        slot = (Int64(parity) * 2 + Int64(group)) * self.slots + Int64(cta)
        return workspace.iterator + self.ws_peers + slot * PEER_SLOT

    @cute.jit
    def owner_sums(self, workspace, values, tile, ranges, parity):
        """``fast_exchange``'s ``partials``: a peer stores its rows into its slot; the
        owner adds its peers' rows in CTA order, the first PEER_BATCH loaded at once,
        each polled until no word holds the 0xFFFFFFFF arm pattern (FP32 sums are
        never 0xFFFFFFFF; a row lands in two 16 B stores and every word is checked)."""
        _, warp, lane, cta = thread()
        group = Int32(tile >= self.tiles * ranges[1, 0])
        start = (tile - self.tiles * ranges[group, 0]) * 14
        owner = start >= ranges[group, 2]
        row = (32 * (warp & 3) + lane) * 32
        if not owner:
            slot = self.peer_slot(workspace, parity, group, cta) + row
            for half in cutlass.range_constexpr(2):
                words = tuple(
                    Uint32(float_bits(values[4 * half + i])) for i in range(4)
                )
                store_words(slot + 16 * half, words)
        else:
            size = ranges[group, 1]
            base = self.peer_slot(workspace, parity, group, 0) + row
            words = cute.make_rmem_tensor((PEER_BATCH, 8), Uint32)
            for j in cutlass.range_constexpr(PEER_BATCH):
                peer = cta + 1 + j
                begin = peer * size // self.grid
                real = (peer < self.grid) & (begin < start + 14)
                if real & ((peer + 1) * size // self.grid != begin):
                    source = base + Int64(peer) * PEER_SLOT
                    for half in cutlass.range_constexpr(2):
                        got = load_volatile(source + 16 * half, count=4)
                        for i in cutlass.range_constexpr(4):
                            words[j, 4 * half + i] = got[i]
            for j in cutlass.range_constexpr(PEER_BATCH):
                peer = cta + 1 + j
                begin = peer * size // self.grid
                real = (peer < self.grid) & (begin < start + 14)
                if real & ((peer + 1) * size // self.grid != begin):
                    source = base + Int64(peer) * PEER_SLOT
                    self.poll_row(words, j, source)
                    for t in cutlass.range_constexpr(8):
                        values[t] += bits_float(words[j, t])
            # Peers past the batch (small units), one at a time in CTA order.
            peer = cta + 1 + PEER_BATCH
            while (peer < self.grid) & (peer * size // self.grid < start + 14):
                if (peer + 1) * size // self.grid != peer * size // self.grid:
                    source = base + Int64(peer) * PEER_SLOT
                    for t in cutlass.range_constexpr(8):
                        words[0, t] = Uint32(4294967295)
                    self.poll_row(words, 0, source)
                    for t in cutlass.range_constexpr(8):
                        values[t] += bits_float(words[0, t])
                peer += 1
        return owner

    @cute.jit
    def poll_row(self, words, j: cutlass.Constexpr[int], source):
        """Reload row ``j``'s two 16 B chunks from ``source`` until the warp's rows hold
        no arm word."""
        bad = False
        for t in cutlass.range_constexpr(8):
            bad |= words[j, t] == Uint32(4294967295)
        ready = cute.arch.vote_all_sync(not bad)
        while not ready:
            bad = False
            for half in cutlass.range_constexpr(2):
                got = load_volatile(source + 16 * half, count=4)
                for i in cutlass.range_constexpr(4):
                    words[j, 4 * half + i] = got[i]
                    bad |= got[i] == Uint32(4294967295)
            ready = cute.arch.vote_all_sync(not bad)

    @cute.jit
    def rearm_peers(self, workspace, parity, cta):
        """``fast_exchange`` (rearm warp): this CTA's two peer slots of ``parity``, the
        previous call's (read by then, like its hidden images), back to 0xFFFFFFFF."""
        lane = thread()[2]
        for group in cutlass.range_constexpr(2):
            slot = self.peer_slot(workspace, parity, group, cta)
            for i in cutlass.range_constexpr(PEER_SLOT // 512):
                store_words(slot + (lane + 32 * i) * 16, (Uint32(4294967295),) * 4)

    @cute.jit
    def publish_line(self, words, scale, loaded, line, source, weights, box):
        """``finalize`` of ``line`` from the prepoll's words (``loaded``), polling
        again until all of its staged rows are written."""
        lane = thread()[2]
        token, column = (line // 56, line % 56 * 64)
        if loaded == 0:
            if cutlass.const_expr(self.polled):
                cute.arch.griddepcontrol_wait()
            for slot in cutlass.range_constexpr(16):
                scale[slot] = Float32(weights[token * 16 + slot])
        ready = loaded != 0
        if ready:
            bad = False
            for slot in cutlass.range_constexpr(16):
                bad |= has_ff_half(words[slot])
            ready = cute.arch.vote_all_sync(not bad)
        while not ready:
            bad = False
            for slot in cutlass.range_constexpr(16):
                words[slot] = load_volatile(source + slot * 7168)[0]
                bad |= has_ff_half(words[slot])
            ready = cute.arch.vote_all_sync(not bad)
        # Re-arm, the slot-order FMA sum (bitwise), each lane's latent word.
        low, high = Float32(0), Float32(0)
        for slot in cutlass.range_constexpr(16):
            store_words(source + slot * 7168, (Uint32(4294967295),))
            low = fma(bf16(words[slot]), scale[slot], low)
            high = fma(bf16(words[slot], True), scale[slot], high)
        word = pack_pair(low, high, policy="latent")
        mc, parity, rank = box[0:3]
        offset = ((parity * self.tp + rank) * 64 + token) * 3584 + column
        multicast_words(mc + Int64(offset + 2 * lane) * 2, (word,))

    @cute.jit
    def fc2(self, sm, workspace, output, values, unit, tile, ranges):
        """``ExpertsKernel.fc2``; published: the unit's eight token slots in two
        shared loads, and every lane stores its own row's BF16 half of the staged
        words (lanes m < 8 / 16-23 the low half of the word ``pack_pair`` and the
        gathering shuffles would assemble from rows m and m + 8, lanes 8-15 / 24-31
        the high half), at the same addresses (bitwise), predicated per token."""
        if cutlass.const_expr(not self.publish):
            ExpertsKernel.fc2(self, sm, workspace, output, values, unit, tile, ranges)
        else:
            _, warp, lane, _ = thread()
            slots = view(sm.iterator, self.off_misc + 2176, Uint32, 256)
            words8 = (slots[2 * unit], slots[2 * unit + 1])
            low = lane % 16
            pos = (lane - 8 * (low // 8)) % 16
            half = 4 * (4 * (pos // 2) + 2 * (pos % 2) + lane // 16) + 2 * (low // 8)
            for t in cutlass.range_constexpr(8):
                slot = words8[t // 4] >> 8 * (t % 4) & Uint32(255)
                row = (Int64(t) * 16 + Int32(slot)) * 1792
                row += tile * 64 + (warp & 3) * 16
                stage = workspace.iterator + self.ws_stage + row * 4 + half
                store_stage_half(stage, values[t], Uint32(slot != Uint32(255)))

    @cute.jit
    def store_hidden(self, sm, workspace, values, tile, parity, scalars):
        """``ExpertsKernel.store_hidden`` with each token's warp amax by one
        ``redux.sync.max`` on the bits of |h| (non-negative floats order as their
        bits) instead of five shuffle-fmax steps (NaN lanes count as 0, which the
        fmax tree also returns: half the lanes (gate) hold 0), and selects and
        predicated shared stores instead of lane-divergent branches."""
        _, warp, lane, _ = thread()
        ew, eg = (warp & 3, (warp - 4) // 4)
        expert, chunk = (tile // self.tiles, tile % self.tiles)
        up = Uint32(lane & 8 == 0)
        h = cute.make_rmem_tensor(self.width, Float32)
        maxima = cute.make_rmem_tensor(self.width, Float32)
        shared_max = view(sm.iterator, self.off_misc + 884, Float32, 64)
        for t in cutlass.range_constexpr(self.width):
            gate = cute.arch.shuffle_sync_bfly(values[t], 8)
            g = float_bits(situ(gate, values[t], *scalars))
            h[t] = bits_float(select_u32(up, g, 0))
        # Every reduction, then every shared store: a store between them would order
        # the reductions (and the peer loads below) behind it.
        for t in cutlass.range_constexpr(self.width):
            bits = Uint32(float_bits(abs(h[t])))
            bits = select_u32(Uint32(bits > Uint32(2139095040)), 0, bits)
            maxima[t] = bits_float(redux(bits, "max"))
        for t in cutlass.range_constexpr(self.width):
            own = sm.iterator + self.off_misc + 884 + 4 * ((eg * 4 + ew) * 8 + t)
            store_shared_if(own, float_bits(maxima[t]), Uint32(lane == 0), "b32")
        self.group_sync(eg)
        stage_base = self.off_misc + 3200 + (eg * 4 + ew) * 128
        scale_token = Int32(0)
        # The peer maxima first, then the scales and FP8 values, then the stores: the
        # same arithmetic per token, no store between a load and its use.
        peers = cute.make_rmem_tensor(self.width, Float32)
        for t in cutlass.range_constexpr(self.width):
            peers[t] = shared_max[(eg * 4 + (ew ^ 1)) * 8 + t]
        fp8 = cute.make_rmem_tensor(self.width, Uint32)
        for t in cutlass.range_constexpr(self.width):
            scale = mx_exponent_select(cute.arch.fmax(maxima[t], peers[t]))
            chosen = select_u32(Uint32(lane == t), Uint32(scale), Uint32(scale_token))
            scale_token = Int32(chosen)
            fp8[t] = Uint32(fp8_bits(h[t] * mx_rescale_select(scale)))
        for t in cutlass.range_constexpr(self.width):
            dst = sm.iterator + stage_base + t * 16 + 2 * (lane % 8) + lane // 16
            store_shared_if(dst, fp8[t], up)
        cute.arch.sync_warp()
        if lane < 8:
            kc = 4 * chunk + ew
            data = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            if lane < self.width:
                data = smem_words(sm.iterator + stage_base + lane * 16)
            image = self.image(workspace, parity, expert)
            store_words(image + swizzled(kc, lane), data)
            if ew & 1 == 0:
                byte = view(image, 0, Uint8, self.h_compact_bytes)
                byte[self.h_image_bytes + lane * 16 + 2 * chunk + ew // 2] = Uint8(
                    scale_token
                )
        self.group_sync(eg)

    @cute.jit
    def match_fc2(self, sm, workspace, experts, distinct):
        """The hidden loader warp (``dies``), during FC1: this CTA's arrival ticket on
        its SM's die (``ws_xflag`` words 2 / 3, armed to -1 like the pool counters and
        re-armed with them by the call's last CTA), the die of the middle w2 tile of
        every rank's FC2(A) and static FC2(B) range (seven ballots of 32 ranks), then
        this CTA's two matched ranks into misc 4856, and the producer's barrier."""
        lane = thread()[2]
        words: cutlass.Constexpr = (self.grid + 31) // 32
        smid = sm_id()
        word = Uint32(0)
        for w in cutlass.range_constexpr(words):
            if smid >> 5 == w:
                word = Uint32(self.die_bits[w])
        die = Int32(word >> (smid & 31) & 1)
        k = Int32(0)
        if lane == 0:  # in flight during the ballots
            tickets = workspace.iterator + self.ws_xflag + 8 + 4 * Int64(die)
            k = Int32(claim(tickets, 1)) + 1
        bits = view(sm.iterator, self.off_misc + 4800, Uint32, 2 * words)
        for i in cutlass.range_constexpr(words):
            rank = lane + 32 * i
            ranges = self.partition(distinct, rank)
            for g in cutlass.range_constexpr(2):
                home = Int32(0)
                lo, hi = (ranges[g, 4], ranges[g, 5])
                if (rank < self.grid) & (lo < hi):
                    mid = (lo + hi) // 2
                    row = experts[mid // 28] * 3584 + mid % 28 * 128 + 64
                    home = Int32(Int64(row) * 128 >> 21 & 1)
                ballot = cute.arch.vote_ballot_sync(home != 0)
                if lane == 0:
                    bits[g * words + i] = Uint32(ballot)
        cute.arch.sync_warp()
        k = cute.arch.shuffle_sync(k, 0)
        rows = view(
            sm.iterator, self.off_misc + 4800 + 4 * words * (lane & 1), Uint32, words
        )
        rank = match_rank(rows, self.grid, die, k)
        a, b = (cute.arch.shuffle_sync(rank, 0), cute.arch.shuffle_sync(rank, 1))
        if lane == 0:
            ranks = view(sm.iterator, self.off_misc + 4856, Int32, 2)
            ranks[0], ranks[1] = (a, b)
            ready = view(sm.iterator, self.off_misc + 4864, Uint64, 1)
            cute.arch.mbarrier_arrive(ready.iterator)

    @cute.jit
    def load_hidden16(self, sm, workspace, parity):
        """Hidden images of the producer's queued units, in order, double-buffered."""
        lane = thread()[2]
        full = view(sm.iterator, self.off_misc + self.bar_hfull, Uint64, 2)
        empty = view(sm.iterator, self.off_misc + self.bar_hempty, Uint64, 2)
        ordinal = Int32(-1)
        unit = self.pop_unit(sm, Int32(0))
        while unit >= 0:
            ordinal += 1
            buffer = ordinal & 1
            if ordinal >= 2:
                self.sync(empty.iterator + buffer, (ordinal >> 1) - 1 & 1)
                cute.arch.fence_proxy("async.shared", space="cta")
            self.poll_hidden(sm, self.image(workspace, parity, unit), buffer)
            if lane == 0:
                cute.arch.mbarrier_arrive(full.iterator + buffer)
            unit = self.pop_unit(sm, ordinal + 1)

    @cute.jit
    def consume(self, sm, tmem, trace, ranges):
        """``ExpertsKernel.consume`` for TP16 (one FC2 stage per tile): the FC1 stages
        in ``produce16``'s order from ``ranges`` (no metadata loads or kind dispatch per
        stage), then the FC2 stages and the end marker from their metadata."""
        lane, stages, parts = (thread()[2], self.stages, self.parts)
        bars: tuple = (self.bar_full, self.bar_empty, self.bar_accfull)
        bars += (self.bar_accempty, self.bar_hfull, self.bar_hempty, self.bar_xready)
        ring = parts * stages
        full, empty, accfull, accempty, hfull, hempty, xready = (
            view(sm.iterator, self.off_misc + o, Uint64, n)
            for o, n in zip(bars, (ring, ring, 16, 16, 2, 2, 14))
        )
        meta = view(sm.iterator, self.off_meta, Int32, 4 * stages)
        accmeta = view(sm.iterator, self.off_misc + 600, Int32, 64)
        offsets = (0, self.off_sfa, self.off_x, self.off_xsf, self.off_h, self.off_hsf)
        slot16 = self.slot_bytes // 16  # descriptor units per weight slot
        a0, asf0, x0, xsf0, h0, hsf0 = (
            smem_desc(sm.iterator + o, i % 2 == 0) for i, o in enumerate(offsets)
        )
        xmask, ended = (Uint32(0), False)
        slot, uses, acc, count = (Int32(0), Int32(0), Int32(0), Int32(0))
        current, ordinal, hbuffer = (Int32(-1), Int32(-1), Int32(0))
        phase = Int32(0)
        for group in cutlass.range_constexpr(2):
            lo, hi = (ranges[group, 2], ranges[group, 3])
            ks, tile = (lo % 14, 3 * ranges[group, 0] + lo // 14)
            for s in cutlass.range(lo, hi, unroll=1):
                first, last = ((s == lo) | (ks == 0), (s == hi - 1) | (ks == 13))
                self.sync(full.iterator + parts * slot, phase, True)
                if cutlass.const_expr(self.stamp > 0):
                    if (lane == 0) & (uses + (slot + phase) == 0):
                        stamp_at(trace, 7)
                weights = (tmem + 128 + slot * 8, asf0 + slot * 64)
                if first:
                    self.acquire(accempty, acc, uses)
                copy_scales = Uint32(0)
                if xmask >> ks & 1 == 0:
                    if cutlass.const_expr(self.xearly > 0):
                        self.sync(xready.iterator, 0, True)  # the whole image
                    else:
                        self.sync(xready.iterator + ks, 0, True)
                    xmask |= Uint32(1) << ks
                    copy_scales = Uint32(1)
                completion = Uint32(0)
                if last:
                    completion = shared_address(accfull.iterator + acc)
                for part in cutlass.range_constexpr(parts):
                    if cutlass.const_expr(part == 1):
                        self.sync(full.iterator + parts * slot + 1, phase, True)
                    released = shared_address(empty.iterator + parts * slot + part)
                    stage16(
                        (tmem + acc * 8, a0 + slot * slot16, x0 + ks * 128),
                        weights,
                        (tmem + 256 + ks * 8, xsf0 + ks * 16, copy_scales),
                        (Uint32(not first), released),
                        completion,
                        kind="fc1",
                        early=self.early,
                        packed=self.packed,
                        part=None if parts == 1 else part,
                    )
                cute.arch.sync_warp()
                if cutlass.const_expr(self.stamp > 0):
                    if (lane == 0) & (uses + (slot + phase) == 0):
                        stamp_at(trace, 5)
                if last:
                    c = Int32(first) | Int32(last) << 1
                    self.signal(accmeta, accfull, acc, (Int32(0), tile, ks, c))
                    uses += 1
                    acc = uses % 16
                slot += 1
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)
                ks += 1
                if ks == 14:
                    ks, tile = (Int32(0), tile + 1)
        while not ended:
            self.sync(full.iterator + parts * slot, phase, True)
            kind, a, b, c = (meta[4 * slot + i] for i in range(4))
            if kind == 2:
                if (lane == 0) & (current >= 0) & (hbuffer < 2):
                    commit(hempty.iterator + hbuffer)
                for _ in cutlass.range(2, unroll=1):
                    self.acquire(accempty, acc, uses)
                    self.signal(accmeta, accfull, acc, (kind, a, b, c), 2)
                    uses += 1
                    acc = uses % 16
                ended = True
            else:
                weights = (tmem + 128 + slot * 8, asf0 + slot * 64)
                copy_scales, release = (Uint32(0), Uint32(0))
                if a != current:
                    if (current >= 0) & (hbuffer < 2):
                        release = shared_address(hempty.iterator + hbuffer)
                    current = a
                    ordinal += 1
                    hbuffer = ordinal & 1
                    self.sync(hfull.iterator + hbuffer, ordinal >> 1 & 1, True)
                    copy_scales = Uint32(1)
                hd = h0 + hbuffer * (self.h_image_bytes // 16)
                hs = hsf0 + hbuffer * (self.h_scale_bytes // 16)
                sfbc = tmem + 128 + 8 * stages + hbuffer * self.w2_scale_columns
                self.acquire(accempty, acc, uses)
                # FC2 stages are whole boxes (both of the slot's barriers).
                boxes = [empty.iterator + parts * slot + p for p in range(parts)]
                stage16(
                    (tmem + acc * 8, a0 + slot * slot16, hd),
                    weights,
                    (sfbc, hs, copy_scales),
                    (release, shared_address(boxes[0])),
                    shared_address(accfull.iterator + acc),
                    *[shared_address(box) for box in boxes[1:]],
                    kind="fc2",
                    early=self.early,
                    packed=self.packed,
                )
                cute.arch.sync_warp()
                self.signal(accmeta, accfull, acc, (kind, a, b, c))
                uses += 1
                acc = uses % 16
                slot += 1
                if slot == stages:
                    slot, phase = (Int32(0), phase ^ 1)

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(self.grid_maps):
            maps = self.encode_maps(args[-3], args[-2])
            options = dict(grid=(self.grid, 1, 1), block=(384, 1, 1), cluster=(1, 1, 1))
            kernel = self.kernel_maps(args[:-3], *maps)
            kernel.launch(stream=args[-1], use_pdl=True, **options)
        else:
            launch(self.kernel, args, self.grid, 384, cluster=1)

    def encode_maps(self, w13, w2):
        """``PersistentExperts``' TP16 w13 / w2 descriptors (same bytes)."""
        types, swizzles = (cuda.CUtensorMapDataType, cuda.CUtensorMapSwizzle)
        promotion = cuda.CUtensorMapL2promotion
        fp4 = types.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
        swizzle = swizzles.CU_TENSOR_MAP_SWIZZLE_128B
        box = (128, 128, 3 - self.parts)  # FC1: one or two K128 boxes per stage
        rows = (self.experts * self.w13_rows, self.experts * 3584)
        if cutlass.const_expr(self.packed):  # plain bytes: 128 B rows, w2 data only
            u8 = types.CU_TENSOR_MAP_DATA_TYPE_UINT8
            return (
                encode_map(
                    w13, u8, (128, rows[0], 14), (1792, 128), (128, 128, 1), swizzle,
                    promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                ),
                encode_map(
                    w2, u8, (128, rows[1], 1), (128, 128 * rows[1]), (96, 128, 1),
                    swizzle, promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
                ),
            )  # fmt: skip
        return (
            encode_map(
                w13, fp4, (128, rows[0], 28), (1792, 64), box, swizzle,
                promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            ),
            encode_map(
                w2, fp4, (128, rows[1], 2), (128, 32), (128, 128, 2), swizzle,
                promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
            ),
        )  # fmt: skip

    @cute.kernel
    def kernel(self, args):
        maps = args[2]
        self.body(args, (maps.iterator, maps.iterator + 128))

    @cute.kernel
    def kernel_maps(self, args, w13_map: GridMap, w2_map: GridMap):
        self.body(args, (map_address(w13_map), map_address(w2_map)))

    @cute.jit
    def body(self, args, maps):
        """The kernel; ``maps``: the (w13, w2) descriptor addresses."""
        x, ids, _, w13_scales, w2_scales, workspace, output, batch, *scalars = args
        tid, warp, lane, cta = thread()
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        tmem_base = self.initialize(sm, maps, batch)
        counters = view(workspace.iterator, self.ws_counter, Int32, self.slots)
        generations = None
        if cutlass.const_expr(self.fast_exchange):
            ws_generation = self.ws_generation
            generations = view(workspace.iterator, ws_generation, Int32, self.slots)
        if cutlass.const_expr(self.lean_route):
            self.route_init(sm, counters, generations)
        source = ids
        if cutlass.const_expr(self.polled):
            source = view(workspace.iterator, self.ws_route, Int32, 129).iterator
        else:
            cute.arch.griddepcontrol_wait()
        trace = workspace.iterator + self.stamp + cta * 128
        if cutlass.const_expr(self.stamp > 0):
            if tid == 0:
                stamp_at(trace, 4)
        if cutlass.const_expr(self.polled):
            if tid == 96:
                # Every copy CTA's image part is written (acquire); the producer waits
                # for this arrival before its claim, so the read precedes the re-arm.
                ready = False
                while not ready:
                    ready = load_acquire(source + 128) == Uint32(2 * self.capacity - 1)
                    if not ready:
                        sleep_ns(64)
                xcount = view(sm.iterator, self.off_misc + self.bar_xcount, Uint64, 1)
                cute.arch.mbarrier_arrive(xcount.iterator)
                fence_proxy_global()  # the image, before this thread's bulk copies
        if cutlass.const_expr(self.xearly == 1):
            if tid == 96:
                self.load_image(sm, x, cta)
        if cutlass.const_expr(self.lean_route):
            shared_distinct = view(sm.iterator, self.off_misc + 860, Int32, 1)
            distinct = Int32(0)
            if warp == 0:
                distinct = self.route_warp(sm, source, batch)
                if lane == 0:
                    shared_distinct[0] = distinct
                named_barrier(ROUTED, 384, arrive=True)
            else:
                named_barrier(ROUTED, 384)
                distinct = shared_distinct[0]
            call_count = view(sm.iterator, self.off_misc + 864, Int32, 1)[0]
            experts = view(sm.iterator, self.off_misc + 1140, Int32, 128)
        else:
            distinct, call_count, experts = self.route(sm, ids, counters, batch)
        # The images' and peer slots' alternation: the call counter, or (fast exchange)
        # this kernel's own generation.
        own = call_count
        if cutlass.const_expr(self.fast_exchange):
            own = view(sm.iterator, self.off_misc + 4880, Int32, 1)[0]
        if cutlass.const_expr(self.stamp > 0):
            if tid == 0:
                stamp_at(trace, 0)
        ranges = self.partition(distinct, cta)
        shift = Int32(0)
        if cutlass.const_expr(self.rotate):
            shift = ranges[0, 2] % 14
        layout = cute.make_layout((self.rounds, 4, 4), stride=(16, 4, 1))
        raw_x = cute.make_rmem_tensor(layout, Uint32)
        if cutlass.const_expr(self.planned):
            if cutlass.const_expr(self.xearly == 0):
                if warp == 3:
                    self.load_plan(sm, x, shift)
        else:
            if warp >= 2:
                self.load_x(x, raw_x, batch, shift)
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        if warp == 0:
            if lane == 0:
                weights = (sm, maps, w13_scales, w2_scales, ranges, experts)
                self.produce16(*weights, workspace)
                counters[cta] = call_count + 1
                if cta < self.slots - self.grid:
                    counters[self.grid + cta] = call_count + 1
                if cutlass.const_expr(self.fast_exchange):
                    generations[cta] = own + 1
                    if cta < self.slots - self.grid:
                        generations[self.grid + cta] = own + 1
        else:
            if cutlass.const_expr(not self.planned):
                if warp >= 2:
                    ready = sm.iterator + self.off_misc + self.bar_xready
                    ready = cute.recast_ptr(ready, dtype=Uint64)
                    self.quantize_x(sm, raw_x, ready, batch, shift)
                    if cutlass.const_expr(self.stamp > 0):
                        if tid == 64:
                            stamp_at(trace, 6)
            if warp == 1:
                self.consume(sm, tmem, trace, ranges)
                fence_tmem(before=True)
                named_barrier(TMEM_DONE, 320, arrive=True)
            elif warp == 2:
                if cutlass.const_expr(len(self.dies) > 0):
                    self.match_fc2(sm, workspace, experts, distinct)
                self.load_hidden16(sm, workspace, own & 1)
            elif warp == 3:
                self.rearm(workspace, own)
                if cutlass.const_expr(self.fast_exchange):
                    self.rearm_peers(workspace, own & 1 ^ 1, cta)
                if cutlass.const_expr(self.polled):
                    cute.arch.griddepcontrol_wait()  # the shared rows
                if cutlass.const_expr(self.publish == "push" and self.late_push):
                    # After the CTA's first latent stage landed (a one-shot barrier):
                    # the stores stay out of the first weight loads' way.
                    ready = self.off_misc + self.bar_xready
                    ready = view(sm.iterator, ready, Uint64, 14)
                    if cutlass.const_expr(self.xearly > 0):
                        self.sync(ready.iterator, 0)
                    else:
                        self.sync(ready.iterator + shift, 0)
                    self.push_shared(scalars[3:9], batch, lane, cta, 32)
                # TMEM teardown once the MMA and epilogue warps are past their last
                # tcgen05 operation; the lines' finalize and the CTA's exit do not wait.
                named_barrier(TMEM_DONE, 320)
                fence_tmem()
                free_tmem(tmem, 512)
            else:
                if cutlass.const_expr(self.publish == "push" and not self.late_push):
                    if cutlass.const_expr(self.polled):
                        cute.arch.griddepcontrol_wait()  # the shared rows
                    self.push_shared(scalars[3:9], batch, tid - 128, cta, 256)
                self.epilogue(
                    sm, workspace, output, tmem, ranges, batch, own & 1, scalars
                )
        if cutlass.const_expr(self.stamp > 0):
            if (lane == 0) & ((warp < 2) | (warp == 4)):
                stamp_at(trace, 1 + warp - 2 * Int32(warp == 4))
