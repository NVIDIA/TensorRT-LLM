# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent NVFP4 expert kernel for M <= 8 decode rows (CuTe DSL, SM100 family).

Wide builds (``ntok`` 16): the same kernel with ``ntok`` token columns -- MMA N, the B
operand stages, the compact SFB chunk, the FC1 staging tile and the tagged hand-off words
scale with it (``geometry``); everything below is written for the ``ntok = 8`` build,
whose constants are the module's ``NTOK`` ones.

One launch per MoE layer replaces TRTLLM-gen's routing-indices kernel, FC1
and FC2 for the rows of one expert-parallel window of ``L`` local experts
(32 = this rank's own block, 64 = the ring-buddy stack, 96 = the
three-window stack). The active local experts are a bit set of ``L / 32``
words (two words for ``L <= 64``); a step selects at most 64 distinct experts,
so the slot tables keep 64 slots for every window.

Work decomposition. Every CTA reads the ``[M, 8]`` window ids, builds the
sorted list of the ``E`` distinct local experts ("slots") and the same static
tile list::

    position p <  64 E : FC1 tile (slot p // 64, rows 128 (p % 64) ..)
    position p >= 64 E : FC2 tile (slot q // 32, rows 128 (q % 32) ..), q = p - 64 E

A slot is local when its id is in ``[0, L)`` and no earlier slot of the same
token names the same id (a repeated expert is routed once, through its first
slot, as TRTLLM-gen does). CTA ``c`` of a grid of ``G`` runs positions ``c, c +
G, c + 2 G, ...`` in that order. A tile is 128 weight rows by the whole K (4096): 16 k-blocks of 256
through a TMA ring into ``tcgen05`` block-scaled MMAs (M = 128 weight rows,
N = 8 token columns, K = 64 per instruction, one accumulator, K ascending; no
split-K, so each output is TRTLLM-gen's accumulation). All eight token
columns are computed for every active expert: a column depends only on its
own token, so the routed columns are the ones TRTLLM-gen computes, and no
permutation is needed.

FC1 -> FC2 hand-off. The FC1 epilogue of tile ``mt`` applies SwiGLU and the
NVFP4 quantization of the intermediate to its 64 intermediate columns
(TRTLLM-gen's arithmetic in its order, ``_swiglu_quant_to_stage``) and
stores them, for all 8 tokens, into the slot's
operand image in global memory -- the exact shared-memory image of the FC2 B
operand (16 KB FP4, swizzled) and its compact block scales (8 KB) -- then
publishes the tile with a per-tile flag holding the launch epoch. FC2 k-block
``kb`` reads intermediate columns ``[256 kb, 256 kb + 256)``, i.e. FC1 tiles
``4 kb .. 4 kb + 3`` of its slot only; the fetch warps poll the 64 flags
(one per lane) and bulk-copy each k-block's operand bytes into one of two
image buffers as soon as its own four flags are published, so an FC2 tile
streams its weights through the ring while the late intermediate columns are
still being produced.

Epochs. Every CTA keeps its own launch counter in the workspace; all CTAs of
a launch read the same value without synchronizing (each launch increments
every CTA's counter once), so flags never need clearing between launches or
graph replays. The grid size of a workspace is therefore fixed. A reader
accepts a word only when its tag equals the launch epoch, so a stale tag must
never equal it: the epoch skips 0 (the zero-initialized workspace's tag),
and every launch re-tags one inactive slot in rotation, so no tag is older
than 65 launches while the 32-bit counter repeats only every 2^32 - 1.

Warp roles: 0-3 epilogue (TMEM lanes 0-127), 4 MMA, 5 weight TMA, 6-7 load
the activation operand, then fetch the FC2 operand images (k-blocks 0-7 and
8-15).

Early ids (``early_expert_ids``, ``early_ids != 0``; compiled out otherwise). The
kernel's same-stream predecessor (``ops/comm/pool_gather_route``) publishes
the window ids as 64 single-copy-atomic 64-bit words ``(id, tag)`` at the
compile-time address ``early_ids``, the tag being the epoch this launch will
use (``ops/moe/early_expert_ids.py``), and triggers its dependents right after
its own dependency wait. This kernel then does not wait at entry: each CTA
reads its epoch, polls its ids until their tags equal it, routes, and the
weight TMA warp starts streaming at once and never waits; every other warp
executes ``griddepcontrol.wait`` after the routing, before its first access
to an activation, an output or the hand-off workspace. A word still stale at
the deadline makes its thread wait and read the plain ids (the same values)
and count a fallback.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from . import ptx

# Expert geometry (MuseSpark Open experts after the pool projection).
HIDDEN = 4096  # FC1 K, FC2 output rows
INTER = 4096  # FC2 K; FC1 has 2 * INTER gate/up rows
NTOK = 8  # MMA N: every token column of a decode step
TOP_K = 8
KTILE = 256  # K per ring stage (4 MMA instructions of K = 64)
FC1_KB = HIDDEN // KTILE  # 16
FC2_KB = INTER // KTILE  # 16
FC1_MT = 2 * INTER // 128  # 64 FC1 tiles per expert
FC2_MT = HIDDEN // 128  # 32 FC2 tiles per expert
TILES_PER_EXPERT = FC1_MT + FC2_MT
MAX_SLOTS = NTOK * TOP_K  # distinct experts a step can select

A_STAGE_BYTES = 128 * KTILE // 2
SFA_STAGE_BYTES = 128 * KTILE // 16
X_ROW_BYTES = HIDDEN // 2
X_SF_ROW_BYTES = HIDDEN // 16

# B operand stages of 8 rows x 128 FP4 bytes (one k-block each): 16 for the
# activations (FC1), then two FC2 image buffers of 16.
B_STAGE_BYTES = NTOK * KTILE // 2
X_STAGE0 = 0
IMG_STAGE0 = FC1_KB
IMG_BUFS = 2
B_STAGES = FC1_KB + IMG_BUFS * FC2_KB
# Compact SFB: only token rows 0..7 of each 128-row scale chunk are read by
# the N = 8 MMA, so consecutive 64-K chunks are stored 128 B apart
# (overlapping the unused rows) instead of 512 B.
SFB_CHUNK_STRIDE = 128
SFB_STAGE_STRIDE = 4 * SFB_CHUNK_STRIDE
SFB_BYTES = (B_STAGES - 1) * SFB_STAGE_STRIDE + 3 * SFB_CHUNK_STRIDE + 512

# One slot's FC2 operand image in global memory: B stages then compact SFB.
IMG_B_BYTES = FC2_KB * B_STAGE_BYTES
IMG_SF_BYTES = FC2_KB * SFB_STAGE_STRIDE
IMG_BYTES = IMG_B_BYTES + IMG_SF_BYTES
IMG_KB_TX = B_STAGE_BYTES + SFB_STAGE_STRIDE
# Tagged hand-off (``handoff=1``): each FC1 tile's 72 output words (64 code
# words, then 8 scale words) as 64-bit (payload, launch epoch) pairs.
TAG_TILE_WORDS = 72
TAG_TILE_BYTES = TAG_TILE_WORDS * 8
TAG_KB_BYTES = 4 * TAG_TILE_BYTES  # FC2 k-block kb reads FC1 tiles 4 kb .. 4 kb + 3
TAG_SLOT_BYTES = 64 * TAG_TILE_BYTES
TAG_LANE_WORDS = TAG_KB_BYTES // 8 // 32  # words per lane per k-block (9)
TAG_PASS_KB = 4  # k-blocks one fetch warp polls together
# FC1 tile output staged in shared memory: 8 rows x 32 FP4 bytes, then 8 x 4 scales.
HSTAGE_SF = NTOK * 32
HSTAGE_BYTES = HSTAGE_SF + NTOK * 4

#: Workspace bytes per slot, enough for either hand-off.
WS_SLOT_BYTES = max(IMG_BYTES, TAG_SLOT_BYTES)

#: Token columns a build is compiled for: the 8-row decode step, and the wide build of a
#: TP4 decode step of two requests.
NTOKS = (8, 16)


class Geometry:
    """The ``ntok``-dependent sizes of a build; ``Geometry(NTOK)`` is the module constants.

    B operand stages of ``ntok`` rows x 128 FP4 bytes; the compact SFB keeps token rows
    ``0 .. ntok - 1`` of each 128-row scale chunk (``16 ntok`` bytes apart); the FC1 tile's
    staged output is ``ntok`` rows x 32 code bytes then ``ntok`` x 4 scale bytes; a tagged
    tile is those ``9 ntok`` words.
    """

    def __init__(self, ntok: int):
        assert ntok in NTOKS, ntok
        self.ntok = ntok
        self.route = ntok * TOP_K  # (token, k) routing lanes
        self.b_stage_bytes = ntok * KTILE // 2
        self.sfb_chunk = 16 * ntok
        self.sfb_stage = 4 * self.sfb_chunk
        self.sfb_bytes = (B_STAGES - 1) * self.sfb_stage + 3 * self.sfb_chunk + 512
        self.img_b_bytes = FC2_KB * self.b_stage_bytes
        self.img_sf_bytes = FC2_KB * self.sfb_stage
        self.img_bytes = self.img_b_bytes + self.img_sf_bytes
        self.img_kb_tx = self.b_stage_bytes + self.sfb_stage
        self.tag_code_words = 8 * ntok
        self.tag_tile_words = 9 * ntok
        self.tag_tile_bytes = self.tag_tile_words * 8
        self.tag_kb_bytes = 4 * self.tag_tile_bytes
        self.tag_slot_bytes = 64 * self.tag_tile_bytes
        self.tag_lane_words = self.tag_kb_bytes // 8 // 32
        self.hstage_sf = ntok * 32
        self.hstage_bytes = self.hstage_sf + ntok * 4
        self.ws_slot_bytes = max(self.img_bytes, self.tag_slot_bytes)


_G8 = Geometry(NTOK)
assert (
    _G8.b_stage_bytes,
    _G8.sfb_chunk,
    _G8.sfb_stage,
    _G8.sfb_bytes,
    _G8.img_bytes,
    _G8.img_kb_tx,
    _G8.tag_tile_words,
    _G8.tag_lane_words,
    _G8.hstage_sf,
    _G8.hstage_bytes,
    _G8.ws_slot_bytes,
) == (
    B_STAGE_BYTES,
    SFB_CHUNK_STRIDE,
    SFB_STAGE_STRIDE,
    SFB_BYTES,
    IMG_BYTES,
    IMG_KB_TX,
    TAG_TILE_WORDS,
    TAG_LANE_WORDS,
    HSTAGE_SF,
    HSTAGE_BYTES,
    WS_SLOT_BYTES,
)

# Warp roles
EPI_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5
FETCH_WARP0 = 6  # warps 6-7 fetch the FC2 operand images (k-blocks 0-7, 8-15)
X_WARP0 = 6  # warps 6-7 load the activation operand first
NUM_WARPS = 8
NUM_THREADS = 32 * NUM_WARPS
BAR_EPI = 1  # named barrier of the epilogue warps
BAR_X = 2  # named barrier of the activation loaders
BAR_ROUTE = 3  # named barrier of every warp but the weight TMA warp (routing)

# mbarriers (besides the ring and accumulator pipelines)
X_BAR = 0
IMG_FULL_BAR = 1  # IMG_BUFS x FC2_KB, one per (buffer, k-block)
IMG_EMPTY_BAR = IMG_FULL_BAR + IMG_BUFS * FC2_KB  # one per buffer
NUM_BAR = IMG_EMPTY_BAR + IMG_BUFS

TMEM_COLS = 128
TMEM_SFA_COL = 64
TMEM_SFB_COL = 80

# L2 policy of the weight stream (read once per launch): evict first, so the
# activations, operand images and flags stay in L2.
L2_EVICT_FIRST = 0x12F0000000000000

# FP32 constants of TRTLLM-gen's FC1 epilogue, exactly as its SASS holds them:
# log2(e) = 0x3FB8AA3B and 1/6 = 0x3E2AAAAB (a constant, not a runtime rcp).
LOG2E_F32 = 1.44269502162933349609375
RCP6_F32 = 0.16666667163372039794921875

# Workspace words per CTA (launch epochs)
MAX_CTAS = 1024

# Stress-test delay sites (``delay`` bits; compiled out when 0). A
# pseudo-random third of the CTAs, re-drawn every launch, spins 1..6 units of
# ``delay_ns`` on the global timer at each enabled site.
DELAY_ENTRY = 1  # whole CTA, right after the epoch is read
DELAY_PRODUCER = 2  # FC1 epilogue, before the tile's flag is published
DELAY_CONSUMER = 4  # fetch warp, before it polls a slot's flags

# Timeline mode (``trace``; compiled out when 0): per CTA, TRACE_WORDS global
# timer stamps (ns) in the workspace's trace buffer. ``trace >= 2`` also stamps
# TR_W0, for which the MMA warp waits for its first weight stage before the
# activation barrier (a different wait order from ``trace == 1`` and the
# untraced kernel).
TRACE_WORDS = 256
TR_ENTRY = 0  # before the dependency wait
TR_WAITED = 1  # after the wait and the epoch read
TR_ROUTED = 2  # slot tables built
TR_X = 3  # activation operand in shared memory
TR_EXIT = 4  # epilogue warps done
TR_TILES = 5  # (value) tiles of this CTA
TR_IDS = 6  # early ids: thread 0 saw its id word fresh (or fell back)
TR_W0 = 7  # (trace >= 2) the first weight stage of the CTA's first tile landed
TR_TILE0 = 8  # + 4 j: TMA starts tile j, MMA's first stage landed, MMA's last issue, epilogue done
TR_MAX_TILES = 24
TR_FETCH0 = TR_TILE0 + 4 * TR_MAX_TILES  # + j2: FC2 tile j2's operand image requested
TR_EPI0 = TR_FETCH0 + TR_MAX_TILES  # + 3 j: FC1 epilogue: accumulator ready, staged, stored
TR_FETCH1 = TR_EPI0 + 3 * TR_MAX_TILES  # + j2: as TR_FETCH0, for the second fetch warp


class PersistentExpertsKernel:
    """CuTe DSL kernel object; one compiled instance per configuration."""

    def __init__(
        self,
        num_ctas: int,
        num_local: int,
        num_stages: int = 8,
        acc_stages: int = 4,
        delay: int = 0,
        delay_ns: int = 2000,
        trace: int = 0,
        balance: int = 0,
        order: int = 0,
        publish: int = 0,
        handoff: int = 0,
        fastroute: int = 0,
        early_ids: int = 0,
        early_fallbacks: int = 0,
        early_deadline_ns: int = 100_000,
        early_poll_ns: int = 32,
        ntok: int = NTOK,
    ):
        assert num_ctas <= MAX_CTAS
        assert num_local in (32, 64, 96)
        # fastroute's warp-local bit set is two words wide
        assert num_local <= 64 or fastroute == 0, "fastroute supports windows up to 64"
        self.g = Geometry(ntok)
        self.ntok = ntok
        assert acc_stages * ntok <= TMEM_SFA_COL
        # A wide build's slot table keeps 64 slots: at most 64 distinct local experts.
        assert ntok == NTOK or (num_local <= 64 and early_ids == 0)
        if early_ids != 0 and (fastroute != 0 or early_fallbacks == 0):
            # The fast route reads the ids from global memory in the TMA warp,
            # which never waits in early mode.
            raise ValueError(
                "the persistent kernel's early-ids variant needs a fallback counter and the "
                "routing through shared memory (fastroute 0)"
            )
        self.num_ctas = num_ctas
        self.num_local = num_local
        # Words of the active-expert bit set and entries of the expert -> slot
        # table: 2 and 64 for every window up to 64 experts.
        self.bitmap_words = 3 if num_local > 64 else 2
        self.expert_slots = max(MAX_SLOTS, num_local)
        self.num_stages = num_stages
        self.acc_stages = acc_stages
        self.delay = delay
        self.delay_ns = delay_ns
        self.trace = trace
        self.balance = balance
        self.order = order
        self.publish = publish
        self.handoff = handoff
        self.fastroute = fastroute
        self.early_ids = early_ids
        self.early_fallbacks = early_fallbacks
        self.early_deadline_ns = early_deadline_ns
        self.early_poll_ns = early_poll_ns
        self.sf_vec_size = 16
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_inst_mn = (128, ntok)
        self.mma_tiler = (128, ntok, KTILE)

    def _sw(self, r):
        """The 128-byte swizzle's XOR row of operand row ``r`` (rows 8 apart share one).

        A plain Python method: the ``ntok = 8`` build traces ``r`` itself.
        """
        if self.ntok == NTOK:
            return r
        return r % 8

    def _setup(self):
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E4M3FN,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_mn,
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (self.tiled_mma.thr_id.shape,)
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, self.num_stages
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, B_STAGES
        )
        # Standard SFB layout (defines the TMEM layout) and the compact SMEM
        # layout actually used for storage (see SFB_CHUNK_STRIDE and ``Geometry``).
        self.sfb_smem_layout_std = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler, self.sf_vec_size, B_STAGES
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            self.sfb_smem_layout_std.shape,
            stride=((((16, 4), 0), (0, 1)), 0, self.g.sfb_chunk, self.g.sfb_stage),
        )

    @cute.jit
    def run(
        self,
        w13_ptr: cute.Pointer,
        w13_sf_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2_sf_ptr: cute.Pointer,
        x_addr: Int64,
        xsf_addr: Int64,
        ids_addr: Int64,
        scale_c_addr: Int64,
        alpha1_addr: Int64,
        alpha2_addr: Int64,
        rows_addr: Int64,
        index_addr: Int64,
        epoch_addr: Int64,
        flag_addr: Int64,
        img_addr: Int64,
        trace_addr: Int64,
        num_tokens: Int32,
        stream,
    ):
        self._setup()
        w13 = cute.make_tensor(
            w13_ptr,
            cute.make_ordered_layout((2 * INTER, HIDDEN, self.num_local), order=(1, 0, 2)),
        )
        w2 = cute.make_tensor(
            w2_ptr, cute.make_ordered_layout((HIDDEN, INTER, self.num_local), order=(1, 0, 2))
        )
        w13_sf = cute.make_tensor(
            w13_sf_ptr, blockscaled_utils.tile_atom_to_shape_SF(w13.shape, self.sf_vec_size)
        )
        w2_sf = cute.make_tensor(
            w2_sf_ptr, blockscaled_utils.tile_atom_to_shape_SF(w2.shape, self.sf_vec_size)
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_a1, tma_a1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w13, a_smem_layout, self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_sfa1, tma_sfa1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w13_sf,
            sfa_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_a2, tma_a2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w2, a_smem_layout, self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_sfa2, tma_sfa2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w2_sf,
            sfa_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        @cute.struct
        class SharedStorage:
            ring_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.acc_stages]
            b_mbar: cute.struct.MemRange[cutlass.Int64, NUM_BAR]
            tmem_holding: cutlass.Int32
            hstage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.g.hstage_bytes // 8], 16
            ]
            misc: cute.struct.MemRange[cutlass.Int32, 8]
            bitmap: cute.struct.MemRange[cutlass.Uint32, self.bitmap_words]
            slot_ids: cute.struct.MemRange[cutlass.Int32, self.g.route]
            slot_expert: cute.struct.MemRange[cutlass.Int32, MAX_SLOTS]
            slot_mask: cute.struct.MemRange[cutlass.Uint32, MAX_SLOTS]
            slot_k: cute.struct.MemRange[cutlass.Int32, MAX_SLOTS * self.ntok]
            expert_slot: cute.struct.MemRange[cutlass.Int32, self.expert_slots]
            slot_scalars: cute.struct.MemRange[cutlass.Float32, 3 * MAX_SLOTS]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E4M3FN, cute.cosize(self.sfa_smem_layout_staged)
                ],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, self.g.sfb_bytes], 1024
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            tma_atom_a1,
            tma_a1,
            tma_atom_sfa1,
            tma_sfa1,
            tma_atom_a2,
            tma_a2,
            tma_atom_sfa2,
            tma_sfa2,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfb_smem_layout_std,
            x_addr,
            xsf_addr,
            ids_addr,
            scale_c_addr,
            alpha1_addr,
            alpha2_addr,
            rows_addr,
            index_addr,
            epoch_addr,
            flag_addr,
            img_addr,
            trace_addr,
            num_tokens,
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[NUM_THREADS, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_atom_sfa1: cute.CopyAtom,
        mSFA1: cute.Tensor,
        tma_atom_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_atom_sfa2: cute.CopyAtom,
        mSFA2: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfb_smem_layout_staged: cute.Layout,
        sfb_smem_layout_std: cute.Layout,
        x_addr: Int64,
        xsf_addr: Int64,
        ids_addr: Int64,
        scale_c_addr: Int64,
        alpha1_addr: Int64,
        alpha2_addr: Int64,
        rows_addr: Int64,
        index_addr: Int64,
        epoch_addr: Int64,
        flag_addr: Int64,
        img_addr: Int64,
        trace_addr: Int64,
        num_tokens: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        cta = Int32(cta)
        grid = Int32(grid)
        grid0 = grid

        if warp_idx == TMA_WARP:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa2)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ring = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ring_mbar.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=A_STAGE_BYTES + SFA_STAGE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, EPI_WARPS),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        b_mbar = storage.b_mbar.data_ptr()
        if warp_idx == MMA_WARP:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(NUM_BAR):
                    cute.arch.mbarrier_init(b_mbar + i, 1)
        cute.arch.mbarrier_init_fence()
        if warp_idx == 0:
            cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding)

        s_misc = storage.misc.get_tensor(cute.make_layout(8))
        s_bitmap = storage.bitmap.get_tensor(cute.make_layout(self.bitmap_words))
        s_ids = storage.slot_ids.get_tensor(cute.make_layout(self.g.route))
        s_slot_expert = storage.slot_expert.get_tensor(cute.make_layout(MAX_SLOTS))
        s_slot_mask = storage.slot_mask.get_tensor(cute.make_layout(MAX_SLOTS))
        s_slot_k = storage.slot_k.get_tensor(cute.make_layout(MAX_SLOTS * self.ntok))
        s_expert_slot = storage.expert_slot.get_tensor(cute.make_layout(self.expert_slots))
        if tidx < MAX_SLOTS:
            s_slot_mask[tidx] = Uint32(0)
            s_expert_slot[tidx] = Int32(-1)
        if cutlass.const_expr(self.expert_slots > MAX_SLOTS):
            if (tidx >= MAX_SLOTS) & (tidx < self.expert_slots):
                s_expert_slot[tidx] = Int32(-1)
        for i in cutlass.range_constexpr(MAX_SLOTS * self.ntok // NUM_THREADS):
            s_slot_k[tidx + i * NUM_THREADS] = Int32(TOP_K)  # above every slot index (atomic min)
        if tidx < self.bitmap_words:
            s_bitmap[tidx] = Uint32(0)
        cute.arch.sync_threads()
        if tidx == 0:
            self._stamp(trace_addr, cta, TR_ENTRY)

        # Everything below reads tensors written by earlier kernels or touches
        # the workspace shared with the previous launch. With early ids only the
        # epoch (this grid's own workspace word; the previous launch completed
        # before the predecessor's wait returned, and the predecessor triggers
        # after that wait) and the tagged ids are read before the wait below.
        if cutlass.const_expr(self.early_ids == 0):
            cute.arch.griddepcontrol_wait()
        # With ``fastroute`` the weight TMA warp routes on its own
        # (``_tma_producer``) and skips the CTA's routing barriers, so its first
        # loads go out one ids load after the wait.
        epoch = Uint32(0)
        if cutlass.const_expr(self.fastroute == 0):
            epoch = self._next_epoch(epoch_addr, cta)
        else:
            if warp_idx != TMA_WARP:
                epoch = self._next_epoch(epoch_addr, cta)
        if cutlass.const_expr(self.early_ids == 0):
            if tidx == 0:
                self._stamp(trace_addr, cta, TR_WAITED)
        if cutlass.const_expr(self.delay & DELAY_ENTRY):
            self._delay_spin(DELAY_ENTRY * 53, cta)

        # -------- window ids -> sorted slots of the active local experts --------
        # Warps 0-1, one lane per (token, slot).
        if tidx < self.g.route:
            t = tidx // TOP_K
            gid = Int32(-1)
            if t < num_tokens:
                if cutlass.const_expr(self.early_ids != 0):
                    gid = self._early_id(tidx, ids_addr, epoch, trace_addr, cta)
                else:
                    gid = Int32(ptx.ld_u32(ids_addr + tidx * 4))
            local = (gid >= 0) & (gid < self.num_local)
            lid = Int32(-1)
            if local:
                lid = gid
                ptx.atom_or_shared_u32(
                    Int32(storage.bitmap.data_ptr().toint()) + (gid // 32) * 4,
                    Uint32(1) << Uint32(gid % 32),
                )
            s_ids[tidx] = lid
        self._route_barrier(warp_idx)
        # Every thread has read its epoch word: advance it for the next launch.
        if tidx == 0:
            ptx.st_relaxed_gpu_u32(epoch_addr + cta * 4, epoch)
            s_misc[0] = ptx.popc_u32(s_bitmap[0]) + ptx.popc_u32(s_bitmap[1])
            if cutlass.const_expr(self.bitmap_words > 2):
                s_misc[0] = s_misc[0] + ptx.popc_u32(s_bitmap[2])
        if tidx < self.num_local:
            e = tidx
            word = s_bitmap[e // 32]
            if ((word >> Uint32(e % 32)) & Uint32(1)) != Uint32(0):
                below = word & ((Uint32(1) << Uint32(e % 32)) - Uint32(1))
                prefix = ptx.popc_u32(below)
                if e >= 32:
                    prefix = prefix + ptx.popc_u32(s_bitmap[0])
                if cutlass.const_expr(self.bitmap_words > 2):
                    if e >= 64:
                        prefix = prefix + ptx.popc_u32(s_bitmap[1])
                s_slot_expert[prefix] = e
                s_expert_slot[e] = prefix
        self._route_barrier(warp_idx)
        if tidx < self.g.route:
            lid = s_ids[tidx]
            if lid >= 0:
                t = tidx // TOP_K
                sl = s_expert_slot[lid]
                # A token that names one local expert in several slots is routed
                # to it once, through its first such slot (TRTLLM-gen's routing
                # also lays out one row per (token, expert)): the smallest k wins.
                ptx.atom_min_shared_s32(
                    Int32(storage.slot_k.data_ptr().toint()) + (sl * self.ntok + t) * 4,
                    tidx % TOP_K,
                )
                ptx.atom_or_shared_u32(
                    Int32(storage.slot_mask.data_ptr().toint()) + sl * 4, Uint32(1) << Uint32(t)
                )
        self._route_barrier(warp_idx)
        # (with fastroute the TMA warp reads unsynchronized values here and
        # never uses them)
        num_slots = s_misc[0]
        grid, total, n_early = self._schedule(num_slots, cta, grid0)
        if cutlass.const_expr(self.trace != 0):
            if tidx == 0:
                self._stamp(trace_addr, cta, TR_ROUTED)
                mine = Int32(0)
                if cta < total:
                    mine = (total - cta + grid - 1) // grid
                ptx.st_u64(trace_addr + (Int64(cta) * TRACE_WORDS + TR_TILES) * 8, Int64(mine))
        if cutlass.const_expr(self.early_ids != 0):
            # Early ids: the weight TMA warp streams now and never waits (it reads
            # only weights); every other warp waits for the predecessor here,
            # before any activation, output or hand-off access.
            if warp_idx != TMA_WARP:
                cute.arch.griddepcontrol_wait()
            if tidx == 0:
                self._stamp(trace_addr, cta, TR_WAITED)

        if warp_idx == TMA_WARP:
            self._tma_producer(
                tma_atom_a1,
                mA1,
                tma_atom_sfa1,
                mSFA1,
                tma_atom_a2,
                mA2,
                tma_atom_sfa2,
                mSFA2,
                a_smem_layout_staged,
                sfa_smem_layout_staged,
                tiled_mma,
                cluster_layout_vmnk,
                storage,
                ring,
                ids_addr,
                num_tokens,
                cta,
                grid0,
                s_slot_expert,
                num_slots,
                grid,
                n_early,
                total,
                trace_addr,
            )
        if warp_idx == MMA_WARP:
            self._mma_issuer(
                tiled_mma,
                storage,
                ring,
                acc_pipe,
                b_mbar,
                a_smem_layout_staged,
                sfa_smem_layout_staged,
                b_smem_layout_staged,
                sfb_smem_layout_staged,
                sfb_smem_layout_std,
                cta,
                grid,
                num_slots,
                n_early,
                total,
                trace_addr,
            )
        if warp_idx >= X_WARP0:
            self._load_x(tidx - 32 * X_WARP0, x_addr, xsf_addr, num_tokens, storage, b_mbar)
            if tidx == 32 * X_WARP0:
                self._stamp(trace_addr, cta, TR_X)
            if warp_idx >= FETCH_WARP0:
                if cutlass.const_expr(self.handoff == 0):
                    self._fetch_images(
                        warp_idx - FETCH_WARP0,
                        lane,
                        storage,
                        b_mbar,
                        flag_addr,
                        img_addr,
                        epoch,
                        cta,
                        grid,
                        num_slots,
                        n_early,
                        total,
                        trace_addr,
                    )
                else:
                    self._fetch_tagged(
                        warp_idx - FETCH_WARP0,
                        lane,
                        storage,
                        b_mbar,
                        img_addr,
                        epoch,
                        cta,
                        grid,
                        num_slots,
                        n_early,
                        total,
                        trace_addr,
                    )
        if warp_idx < EPI_WARPS:
            # off the critical path and the tail: the epilogue warps are idle
            # until the first accumulator
            if cta == 0:
                self._write_row_map(tidx, index_addr, num_tokens, s_ids, s_expert_slot, s_slot_k)
            self._retag_rotation_slot(tidx, flag_addr, img_addr, epoch, cta, grid0, num_slots)
            self._epilogue(
                tiled_mma,
                tidx,
                warp_idx,
                lane,
                storage,
                acc_pipe,
                s_slot_expert,
                s_slot_mask,
                s_slot_k,
                num_tokens,
                scale_c_addr,
                alpha1_addr,
                alpha2_addr,
                rows_addr,
                flag_addr,
                img_addr,
                epoch,
                cta,
                grid,
                num_slots,
                n_early,
                total,
                trace_addr,
            )
            # the successor (the finalize all-reduce) may start launching
            cute.arch.griddepcontrol_launch_dependents()
            # every epilogue warp is done with TMEM before it is released
            ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
            if tidx == 0:
                self._stamp(trace_addr, cta, TR_EXIT)
            if warp_idx == 0:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding
                )
                cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)

    # ------------------------------------------------------------------
    @cute.jit
    def _next_epoch(self, epoch_addr, cta):
        """This launch's epoch: the CTA's stored counter plus one, never 0.

        Tag value 0 is what a zero-initialized workspace holds, so it is
        never a launch epoch; the counter goes ``0xFFFFFFFF -> 1``.
        """
        epoch = ptx.ld_relaxed_gpu_u32(epoch_addr + cta * 4) + Uint32(1)
        if epoch == Uint32(0):
            epoch = Uint32(1)
        return epoch

    @cute.jit
    def _early_id(self, tidx, ids_addr, epoch, trace_addr, cta):
        """Early ids: slot ``tidx``'s window id, from its tagged word once the tag is ``epoch``.

        The word is one 64-bit ``(id, tag)`` the predecessor stored with a
        single-copy-atomic relaxed store, so a fresh tag implies a fresh id. A
        word still stale after ``early_deadline_ns`` (the predecessor did not
        publish for this launch) makes this thread wait for the predecessor and
        read the plain ids instead, and counts one fallback.
        """
        word = Int64(self.early_ids) + Int64(tidx) * 8
        lo, hi = ptx.ld_relaxed_gpu_u64_pair(word)
        if hi != epoch:
            t0 = ptx.globaltimer_u64()
            t = t0
            while (hi != epoch) & (t - t0 < Int64(self.early_deadline_ns)):
                if cutlass.const_expr(self.early_poll_ns > 0):
                    ptx.nanosleep(self.early_poll_ns)
                lo, hi = ptx.ld_relaxed_gpu_u64_pair(word)
                t = ptx.globaltimer_u64()
        gid = Int32(lo)
        if hi != epoch:
            cute.arch.griddepcontrol_wait()
            gid = Int32(ptx.ld_u32(ids_addr + tidx * 4))
            ptx.red_add_relaxed_gpu_u32(Int64(self.early_fallbacks), Uint32(1))
        if tidx == 0:
            self._stamp(trace_addr, cta, TR_IDS)
        return gid

    @cute.jit
    def _write_row_map(self, tidx, index_addr, num_tokens, s_ids, s_expert_slot, s_slot_k):
        """CTA 0's epilogue warps: ``index[t, k] = t * 8 + k`` for a slot routed here, else -1.

        A slot is routed here when its id is local and it is the first slot of
        its token naming that expert, i.e. the ``k`` the slot table kept, so
        every valid entry names a row the FC2 epilogue writes. Runs after the
        routing barriers; the finalize reads the map only after this grid has
        completed.
        """
        if tidx < self.g.route:
            t = tidx // TOP_K
            if t < num_tokens:
                out = Int32(-1)
                lid = s_ids[tidx]
                if lid >= 0:
                    if s_slot_k[s_expert_slot[lid] * self.ntok + t] == tidx % TOP_K:
                        out = tidx
                ptx.st_u32(index_addr + tidx * 4, Uint32(out))

    @cute.jit
    def _retag_rotation_slot(self, tidx, flag_addr, img_addr, epoch, cta, grid0, num_slots):
        """Epilogue warps: stamp this launch's epoch on one inactive slot's hand-off words.

        The rotation slot is ``epoch % 64``. When it is not active in this
        launch, nothing reads it now, and the CTAs of the full grid together
        overwrite all of its tags (tagged words, or flags with ``handoff=0``)
        with the current epoch. Active slots are rewritten by their FC1
        producers. So every slot's tags are rewritten at least once every 65
        launches (64 residues plus the skipped epoch 0), a tag is never more
        than 65 launches old, and a stale tag can equal the current epoch only
        after a full trip of the 32-bit counter, which is impossible within
        65 launches.
        """
        r = Int32(epoch % Uint32(MAX_SLOTS))
        if r >= num_slots:
            words = FC1_MT
            if cutlass.const_expr(self.handoff != 0):
                words = FC1_MT * self.g.tag_tile_words
            # word w of the slot goes to CTA w % grid0, epilogue thread (w // grid0) % 128
            i = Int32(tidx)
            while i * grid0 + cta < words:
                w = i * grid0 + cta
                if cutlass.const_expr(self.handoff != 0):
                    ptx.st_relaxed_gpu_u64_pair(
                        img_addr + Int64(r) * self.g.tag_slot_bytes + Int64(w) * 8,
                        Uint32(0),
                        epoch,
                    )
                else:
                    ptx.st_relaxed_gpu_u32(flag_addr + (r * FC1_MT + w) * 4, epoch)
                i = i + 32 * EPI_WARPS

    @cute.jit
    def _stamp(self, trace_addr, cta, word):
        """Timeline mode: the global timer into this CTA's trace word ``word``."""
        if cutlass.const_expr(self.trace != 0):
            ptx.st_u64(
                trace_addr + (Int64(cta) * TRACE_WORDS + Int64(word)) * 8, ptx.globaltimer_u64()
            )

    @cute.jit
    def _tile_stamp(self, trace_addr, cta, j, which):
        """Timeline mode: stamp ``which`` of this CTA's ``j``-th tile."""
        if cutlass.const_expr(self.trace != 0):
            if j < TR_MAX_TILES:
                self._stamp(trace_addr, cta, TR_TILE0 + 4 * j + which)

    # ------------------------------------------------------------------
    @cute.jit
    def _schedule(self, num_slots, cta, grid):
        """(working grid, this CTA's tile-list length, early slots) of a launch."""
        total = num_slots * TILES_PER_EXPERT
        if cutlass.const_expr(self.balance != 0):
            # Keep the number of rounds the full grid needs, on as few CTAs as
            # that allows, so every working CTA runs the same number of tiles.
            rounds = (total + grid - 1) // grid
            if rounds > 0:
                wgrid = (total + rounds - 1) // rounds
                if cta >= wgrid:
                    total = Int32(0)
                grid = wgrid
        n_early = Int32(0)
        if cutlass.const_expr(self.order != 0):
            n_early = self._early_slots(num_slots, grid)
        return grid, total, n_early

    @cute.jit
    def _ids_bitmap(self, lane, ids_addr, num_tokens):
        """One warp: the 64-bit set of active local experts (two words), on every lane.

        Both id loads go out before either is used (indices past the last
        row are clamped to a valid word and then ignored).
        """
        n_ids = num_tokens * TOP_K
        gids = []
        for i in cutlass.range_constexpr(self.g.route // 32):
            idx = lane + 32 * i
            src = idx
            if idx >= n_ids:
                src = n_ids - 1
            gids.append(Int32(ptx.ld_u32(ids_addr + src * 4)))
        lo = Uint32(0)
        hi = Uint32(0)
        for i in cutlass.range_constexpr(self.g.route // 32):
            gid = gids[i]
            if (lane + 32 * i < n_ids) & (gid >= 0) & (gid < self.num_local):
                if gid < 32:
                    lo = lo | (Uint32(1) << Uint32(gid))
                else:
                    hi = hi | (Uint32(1) << Uint32(gid - 32))
        return ptx.redux_or_u32(lo), ptx.redux_or_u32(hi)

    @cute.jit
    def _route_barrier(self, warp_idx):
        """A routing barrier: every warp, or every warp but the TMA warp (fastroute)."""
        if cutlass.const_expr(self.fastroute != 0):
            if warp_idx != TMA_WARP:
                ptx.bar_sync(BAR_ROUTE, NUM_THREADS - 32)
        else:
            cute.arch.sync_threads()

    @cute.jit
    def _nth_expert(self, lo, hi, slot):
        """The local expert of ``slot``: the ``slot``-th set bit of the 64-bit set."""
        c0 = ptx.popc_u32(lo)
        e = Int32(0)
        if slot < c0:
            e = ptx.fns_u32(lo, slot + 1)
        else:
            e = 32 + ptx.fns_u32(hi, slot - c0 + 1)
        return e

    @cute.jit
    def _early_slots(self, num_slots, grid):
        """Order 1: how many leading slots get all their FC1 tiles before the
        round that mixes FC1 and FC2 tiles.

        Those are the slots whose FC2 tiles fall into that mixed round; their
        FC1 tiles come first, slot-major. The other slots' FC1 tiles follow
        tile-row-major across slots, so the last FC1 round holds only the
        highest rows, which FC2 reads in its last k-blocks. When the mixed
        round is the last round (or there is none) every slot is row-major:
        then the FC2 tiles that wait do so only for their last k-blocks.
        """
        fc1_total = num_slots * FC1_MT
        rounds = (num_slots * TILES_PER_EXPERT + grid - 1) // grid
        early = Int32(0)
        if fc1_total > 0:
            r_mixed = fc1_total // grid
            in_mixed = grid * (r_mixed + 1) - fc1_total
            if (in_mixed < grid) & (r_mixed + 1 < rounds):
                early = (in_mixed + FC2_MT - 1) // FC2_MT
                if early > num_slots:
                    early = num_slots
                cap = (grid * r_mixed) // FC1_MT
                if early > cap:
                    early = cap
        return early

    @cute.jit
    def _decode(self, p, num_slots, n_early):
        """Tile position -> (is FC2, slot, 128-row tile index within the slot)."""
        fc1_total = num_slots * FC1_MT
        is_fc2 = Int32(0)
        slot = p // FC1_MT
        mt = p - slot * FC1_MT
        if cutlass.const_expr(self.order != 0):
            # early slots slot-major, then the rest tile-row-major
            if p >= n_early * FC1_MT:
                if p < fc1_total:
                    q = p - n_early * FC1_MT
                    rest = num_slots - n_early
                    mt = q // rest
                    slot = n_early + (q - mt * rest)
        if p >= fc1_total:
            q = p - fc1_total
            is_fc2 = Int32(1)
            slot = q // FC2_MT
            mt = q - slot * FC2_MT
        return is_fc2, slot, mt

    # ------------------------------------------------------------------
    @cute.jit
    def _tma_producer(
        self,
        tma_atom_a1,
        mA1,
        tma_atom_sfa1,
        mSFA1,
        tma_atom_a2,
        mA2,
        tma_atom_sfa2,
        mSFA2,
        a_smem_layout_staged,
        sfa_smem_layout_staged,
        tiled_mma,
        cluster_layout_vmnk,
        storage,
        ring,
        ids_addr,
        num_tokens,
        cta,
        grid0,
        s_slot_expert,
        num_slots,
        grid,
        n_early,
        total,
        trace_addr,
    ):
        lane = cute.arch.lane_idx()
        lo = Uint32(0)
        hi = Uint32(0)
        if cutlass.const_expr(self.fastroute != 0):
            # this warp's own routing: the same slots and schedule the other
            # warps derive through shared memory
            lo, hi = self._ids_bitmap(lane, ids_addr, num_tokens)
            num_slots = ptx.popc_u32(lo) + ptx.popc_u32(hi)
            grid, total, n_early = self._schedule(num_slots, cta, grid0)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        thr_mma = tiled_mma.get_slice(0)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        gA1 = cute.local_tile(mA1, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFA1 = cute.local_tile(
            mSFA1, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gA2 = cute.local_tile(mA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFA2 = cute.local_tile(
            mSFA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        tAsA, tAgA1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a1,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA1), 0, 3),
        )
        tAsSFA, tAgSFA1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa1,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gSFA1), 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA1 = cute.filter_zeros(tAgSFA1)
        _, tAgA2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a2,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA2), 0, 3),
        )
        _, tAgSFA2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa2,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gSFA2), 0, 3),
        )
        tAgSFA2 = cute.filter_zeros(tAgSFA2)

        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        p = cta
        j = Int32(0)
        while p < total:
            is_fc2, slot, mt = self._decode(p, num_slots, n_early)
            expert = Int32(0)
            if cutlass.const_expr(self.fastroute != 0):
                expert = self._nth_expert(lo, hi, slot)
            else:
                expert = s_slot_expert[slot]
            kb = Int32(0)
            while kb < FC1_KB:
                ring.producer_acquire(state)
                if cutlass.const_expr(self.trace != 0):
                    if (kb == 0) & (lane == 0):
                        self._tile_stamp(trace_addr, cta, j, 0)
                bar = ring.producer_get_barrier(state)
                if is_fc2 != 0:
                    cute.copy(
                        tma_atom_a2,
                        tAgA2[(None, mt, kb, expert)],
                        tAsA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=Int64(L2_EVICT_FIRST),
                    )
                    cute.copy(
                        tma_atom_sfa2,
                        tAgSFA2[(None, mt, kb, expert)],
                        tAsSFA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=Int64(L2_EVICT_FIRST),
                    )
                else:
                    cute.copy(
                        tma_atom_a1,
                        tAgA1[(None, mt, kb, expert)],
                        tAsA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=Int64(L2_EVICT_FIRST),
                    )
                    cute.copy(
                        tma_atom_sfa1,
                        tAgSFA1[(None, mt, kb, expert)],
                        tAsSFA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=Int64(L2_EVICT_FIRST),
                    )
                state.advance()
                kb = kb + 1
            p = p + grid
            j = j + 1
        ring.producer_tail(state)

    # ------------------------------------------------------------------
    def _s2t_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), cutlass.Float8E4M3FN
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def _mma_issuer(
        self,
        tiled_mma,
        storage,
        ring,
        acc_pipe,
        b_mbar,
        a_smem_layout_staged,
        sfa_smem_layout_staged,
        b_smem_layout_staged,
        sfb_smem_layout_staged,
        sfb_smem_layout_std,
        cta,
        grid,
        num_slots,
        n_early,
        total,
        trace_addr,
    ):
        lane = cute.arch.lane_idx()
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding
        )
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + TMEM_SFA_COL, dtype=cutlass.Float8E4M3FN)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + TMEM_SFB_COL, dtype=cutlass.Float8E4M3FN)
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfb_smem_layout_std, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
        copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t_partition(sSFA, tCtSFA)
        copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t_partition(sSFB, tCtSFB)

        ring_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.acc_stages
        )
        num_kphases = cute.size(tCrA, mode=[2])

        if cutlass.const_expr(self.trace >= 2):
            # timeline: the first weight stage landed, whatever the activations do
            if cta < total:
                ring.consumer_wait(ring_state)
                if lane == 0:
                    self._stamp(trace_addr, cta, TR_W0)
        # the activation operand (FC1 B) is in shared memory
        cute.arch.mbarrier_wait(b_mbar + X_BAR, 0)
        j2 = Int32(0)  # FC2 tiles of this CTA so far
        p = cta
        while p < total:
            is_fc2, slot, mt = self._decode(p, num_slots, n_early)
            buf = j2 % IMG_BUFS
            use_phase = (j2 // IMG_BUFS) % 2
            b0 = Int32(X_STAGE0)
            if is_fc2 != 0:
                b0 = Int32(IMG_STAGE0) + buf * FC2_KB
            acc_pipe.producer_acquire(acc_state)
            tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            kb = Int32(0)
            while kb < FC1_KB:
                if is_fc2 != 0:
                    # this k-block's intermediate columns have landed
                    cute.arch.mbarrier_wait(b_mbar + IMG_FULL_BAR + buf * FC2_KB + kb, use_phase)
                ring.consumer_wait(ring_state)
                if cutlass.const_expr(self.trace != 0):
                    if (kb == 0) & (lane == 0):
                        self._tile_stamp(trace_addr, cta, (p - cta) // grid, 1)
                cute.copy(
                    copy_s2t_sfa,
                    tCsSFA_s2t[(None, None, None, None, ring_state.index)],
                    tCtSFA_s2t,
                )
                cute.copy(copy_s2t_sfb, tCsSFB_s2t[(None, None, None, None, b0 + kb)], tCtSFB_s2t)
                for kp in cutlass.range_constexpr(num_kphases):
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, ring_state.index)],
                        tCrB[(None, None, kp, b0 + kb)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ring.consumer_release(ring_state)
                ring_state.advance()
                kb = kb + 1
            if cutlass.const_expr(self.trace != 0):
                if lane == 0:
                    self._tile_stamp(trace_addr, cta, (p - cta) // grid, 2)
            if is_fc2 != 0:
                # the image buffer is free once these MMAs have completed
                with cute.arch.elect_one():
                    tcgen05.commit(b_mbar + IMG_EMPTY_BAR + buf)
                j2 = j2 + 1
            acc_pipe.producer_commit(acc_state)
            acc_state.advance()
            p = p + grid
        acc_pipe.producer_tail(acc_state)

    # ------------------------------------------------------------------
    @cute.jit
    def _load_x(self, q, x_addr, xsf_addr, num_tokens, storage, b_mbar):
        """Warps 6-7: the activation rows into the FC1 B stages and compact SFB.

        B stage ``kb`` holds row ``n``'s FP4 bytes ``[128 kb, 128 kb + 128)`` at
        ``kb * 1024 + n * 128``, 16-byte chunk ``c`` at ``(c ^ n) * 16`` (the
        128-byte swizzle of the MMA descriptor). Scale word ``j`` (64-K chunk)
        of row ``n`` goes to ``j * 128 + n * 16``. Rows at or beyond
        ``num_tokens`` get zero codes and zero scales.
        """
        sb = Int32(storage.sB.data_ptr().toint())
        ssfb = Int32(storage.sSFB.data_ptr().toint())
        # Every load goes out before the first store (rows past the last token
        # load a valid row and store zeros), so the operand lands in about one
        # L2 round trip instead of one per load.
        last = num_tokens - 1
        codes = []
        for i in cutlass.range_constexpr(X_ROW_BYTES * self.ntok // 16 // 64):
            c = q + 64 * i
            n = c // (X_ROW_BYTES // 16)
            cc = c - n * (X_ROW_BYTES // 16)
            nn = n
            if n > last:
                nn = last
            codes.append(ptx.ld_v4_u32(x_addr + nn * X_ROW_BYTES + cc * 16))
        scales = []
        for i in cutlass.range_constexpr(X_SF_ROW_BYTES * self.ntok // 4 // 64):
            w = q + 64 * i
            n = w // (X_SF_ROW_BYTES // 4)
            j = w - n * (X_SF_ROW_BYTES // 4)
            nn = n
            if n > last:
                nn = last
            scales.append(ptx.ld_u32(xsf_addr + nn * X_SF_ROW_BYTES + j * 4))
        for i in cutlass.range_constexpr(X_ROW_BYTES * self.ntok // 16 // 64):
            c = q + 64 * i
            n = c // (X_ROW_BYTES // 16)
            cc = c - n * (X_ROW_BYTES // 16)
            kb = cc // 8
            c16 = cc - kb * 8
            d = codes[i]
            d0 = d[0]
            d1 = d[1]
            d2 = d[2]
            d3 = d[3]
            if n > last:
                d0 = Uint32(0)
                d1 = Uint32(0)
                d2 = Uint32(0)
                d3 = Uint32(0)
            ptx.sts_v4_u32(
                sb + (X_STAGE0 + kb) * self.g.b_stage_bytes + n * 128 + ((c16 ^ self._sw(n)) * 16),
                d0,
                d1,
                d2,
                d3,
            )
        for i in cutlass.range_constexpr(X_SF_ROW_BYTES * self.ntok // 4 // 64):
            w = q + 64 * i
            n = w // (X_SF_ROW_BYTES // 4)
            j = w - n * (X_SF_ROW_BYTES // 4)
            v = scales[i]
            if n > last:
                v = Uint32(0)
            ptx.sts_u32(ssfb + X_STAGE0 * self.g.sfb_stage + j * self.g.sfb_chunk + n * 16, v)
        ptx.fence_proxy_async_shared()
        ptx.bar_sync(BAR_X, 64)
        if q == 0:
            cute.arch.mbarrier_arrive(b_mbar + X_BAR)

    # ------------------------------------------------------------------
    @cute.jit
    def _fetch_images(
        self,
        half,
        lane,
        storage,
        b_mbar,
        flag_addr,
        img_addr,
        epoch,
        cta,
        grid,
        num_slots,
        n_early,
        total,
        trace_addr,
    ):
        """Warps 6-7: the FC2 operand of each of this CTA's FC2 tiles, k-block by k-block.

        Warp ``half`` serves k-blocks ``8 half .. 8 half + 7``; lane ``l``
        acquires the flag of FC1 tile ``32 half + l`` of the slot (k-block
        ``kb`` reads the intermediate columns of FC1 tiles ``4 kb .. 4 kb + 3``).
        The warp polls in lockstep: every round each lane whose flag is not
        yet seen issues one ``ld.acquire``, a ballot tells which k-blocks
        have all four flags, and for each new one lane ``kb % 8`` bulk-copies
        the k-block's B stage and compact SFB stage from the slot's image into
        buffer ``j % 2``, completing on that (buffer, k-block) mbarrier. So a
        k-block is requested as soon as its own producers have published,
        independent of the others. A buffer is refilled only after the MMAs
        of its previous tile have completed.
        """
        sb = Int32(storage.sB.data_ptr().toint())
        ssfb = Int32(storage.sSFB.data_ptr().toint())
        full0 = Int32((b_mbar + IMG_FULL_BAR).toint())
        kb_base = half * (FC2_KB // 2)
        j2 = Int32(0)
        p = cta
        while p < total:
            is_fc2, slot, mt = self._decode(p, num_slots, n_early)
            if is_fc2 != 0:
                buf = j2 % IMG_BUFS
                use = j2 // IMG_BUFS
                if use > 0:
                    cute.arch.mbarrier_wait(b_mbar + IMG_EMPTY_BAR + buf, (use - 1) % 2)
                if cutlass.const_expr(self.delay & DELAY_CONSUMER):
                    self._delay_spin(DELAY_CONSUMER * 53, cta)
                fa = flag_addr + (slot * FC1_MT + 4 * kb_base + lane) * 4
                seen = Int32(0)
                issued = Uint32(0)
                while issued != Uint32(0xFF):
                    if seen == 0:
                        if ptx.ld_acquire_gpu_u32(fa) == epoch:
                            seen = Int32(1)
                    votes = ptx.ballot_u32(seen)
                    # bit 4 j of `quad` is set when flags 4 j .. 4 j + 3 are all seen
                    quad = (
                        votes & (votes >> Uint32(1)) & (votes >> Uint32(2)) & (votes >> Uint32(3))
                    )
                    done = Uint32(0)
                    for j in cutlass.range_constexpr(FC2_KB // 2):
                        done = done | (((quad >> Uint32(4 * j)) & Uint32(1)) << Uint32(j))
                    new = done & (issued ^ Uint32(0xFFFFFFFF))
                    # the lanes' acquires happen-before the issuing lane's copy
                    cute.arch.sync_warp()
                    if lane < FC2_KB // 2:
                        if ((new >> Uint32(lane)) & Uint32(1)) != Uint32(0):
                            kb = kb_base + lane
                            # the image was written by generic stores of other
                            # CTAs; the bulk copy reads it through the async proxy
                            ptx.fence_proxy_async_global()
                            bar = full0 + (buf * FC2_KB + kb) * 8
                            ptx.mbar_arrive_expect_tx(bar, Int32(self.g.img_kb_tx))
                            stage = IMG_STAGE0 + buf * FC2_KB + kb
                            src = img_addr + Int64(slot) * self.g.img_bytes
                            ptx.bulk_g2s(
                                sb + stage * self.g.b_stage_bytes,
                                src + kb * self.g.b_stage_bytes,
                                Int32(self.g.b_stage_bytes),
                                bar,
                            )
                            ptx.bulk_g2s(
                                ssfb + stage * self.g.sfb_stage,
                                src + self.g.img_b_bytes + kb * self.g.sfb_stage,
                                Int32(self.g.sfb_stage),
                                bar,
                            )
                    issued = issued | new
                if cutlass.const_expr(self.trace != 0):
                    if (lane == 0) & (j2 < TR_MAX_TILES):
                        if half == 0:
                            self._stamp(trace_addr, cta, TR_FETCH0 + j2)
                        else:
                            self._stamp(trace_addr, cta, TR_FETCH1 + j2)
                j2 = j2 + 1
            p = p + grid

    # ------------------------------------------------------------------
    @cute.jit
    def _fetch_tagged(
        self,
        half,
        lane,
        storage,
        b_mbar,
        img_addr,
        epoch,
        cta,
        grid,
        num_slots,
        n_early,
        total,
        trace_addr,
    ):
        """Warps 6-7, tagged hand-off: the FC2 operand of each FC2 tile, k-block by k-block.

        Warp ``half`` serves k-blocks ``8 half .. 8 half + 7`` in two passes of
        four. A k-block is the 4 x 72 tagged words of FC1 tiles
        ``4 kb .. 4 kb + 3``; lane ``l`` reads words ``l + 32 i`` (``i < 9``)
        with ``ld.relaxed.gpu`` 64-bit loads (single-copy atomic, so a word
        whose tag is this launch's epoch carries this launch's payload). Each
        polling round reloads the pass's undelivered k-blocks; a k-block whose
        288 words all carry the epoch (warp ballot) is unpacked into the
        swizzled B stage and the compact SFB stage of buffer ``j % 2`` with
        shared stores, made visible to the tensor core
        (``fence.proxy.async.shared::cta``, warp barrier) and released to the
        MMA warp with one arrive on its (buffer, k-block) mbarrier.
        """
        sb = Int32(storage.sB.data_ptr().toint())
        ssfb = Int32(storage.sSFB.data_ptr().toint())
        # this lane's 9 words of a k-block: source offset and destination
        src_off = []
        dst_off = []
        is_sf = []
        for i in cutlass.range_constexpr(self.g.tag_lane_words):
            gw = lane + 32 * i
            t4 = gw // self.g.tag_tile_words
            w = gw - t4 * self.g.tag_tile_words
            r = w // 8
            wr = w - r * 8
            c16 = 2 * t4 + wr // 4
            off_b = r * 128 + ((c16 ^ self._sw(r)) * 16) + (wr % 4) * 4
            off_s = t4 * self.g.sfb_chunk + (w - self.g.tag_code_words) * 16
            sf = Int32(0)
            off = off_b
            if w >= self.g.tag_code_words:
                sf = Int32(1)
                off = off_s
            src_off.append(gw * 8)
            dst_off.append(off)
            is_sf.append(sf)
        j2 = Int32(0)
        p = cta
        while p < total:
            is_fc2, slot, mt = self._decode(p, num_slots, n_early)
            if is_fc2 != 0:
                buf = j2 % IMG_BUFS
                use = j2 // IMG_BUFS
                if use > 0:
                    cute.arch.mbarrier_wait(b_mbar + IMG_EMPTY_BAR + buf, (use - 1) % 2)
                if cutlass.const_expr(self.delay & DELAY_CONSUMER):
                    self._delay_spin(DELAY_CONSUMER * 53, cta)
                slot_base = img_addr + Int64(slot) * self.g.tag_slot_bytes
                for ps in cutlass.range_constexpr(FC2_KB // 2 // TAG_PASS_KB):
                    kb0 = half * (FC2_KB // 2) + ps * TAG_PASS_KB
                    delivered = Int32(0)
                    while delivered != Int32((1 << TAG_PASS_KB) - 1):
                        payload = []
                        valid = []
                        for q in cutlass.range_constexpr(TAG_PASS_KB):
                            src = slot_base + Int64((kb0 + q) * self.g.tag_kb_bytes)
                            ok = Int32(1)
                            pl = []
                            for i in cutlass.range_constexpr(self.g.tag_lane_words):
                                lo, hi = ptx.ld_relaxed_gpu_u64_pair(src + Int64(src_off[i]))
                                pl.append(lo)
                                if hi != epoch:
                                    ok = Int32(0)
                            payload.append(pl)
                            valid.append(ok)
                        for q in cutlass.range_constexpr(TAG_PASS_KB):
                            if ((delivered >> q) & 1) == 0:
                                if ptx.ballot_u32(valid[q]) == Uint32(0xFFFFFFFF):
                                    stage = IMG_STAGE0 + buf * FC2_KB + kb0 + q
                                    for i in cutlass.range_constexpr(self.g.tag_lane_words):
                                        if is_sf[i] != 0:
                                            ptx.sts_u32(
                                                ssfb + stage * self.g.sfb_stage + dst_off[i],
                                                payload[q][i],
                                            )
                                        else:
                                            ptx.sts_u32(
                                                sb + stage * self.g.b_stage_bytes + dst_off[i],
                                                payload[q][i],
                                            )
                                    ptx.fence_proxy_async_shared()
                                    cute.arch.sync_warp()
                                    if lane == 0:
                                        cute.arch.mbarrier_arrive(
                                            b_mbar + IMG_FULL_BAR + buf * FC2_KB + kb0 + q
                                        )
                                    delivered = delivered | (1 << q)
                if cutlass.const_expr(self.trace != 0):
                    if (lane == 0) & (j2 < TR_MAX_TILES):
                        if half == 0:
                            self._stamp(trace_addr, cta, TR_FETCH0 + j2)
                        else:
                            self._stamp(trace_addr, cta, TR_FETCH1 + j2)
                j2 = j2 + 1
            p = p + grid

    # ------------------------------------------------------------------
    @cute.jit
    def _epilogue(
        self,
        tiled_mma,
        tidx,
        warp_idx,
        lane,
        storage,
        acc_pipe,
        s_slot_expert,
        s_slot_mask,
        s_slot_k,
        num_tokens,
        scale_c_addr,
        alpha1_addr,
        alpha2_addr,
        rows_addr,
        flag_addr,
        img_addr,
        epoch,
        cta,
        grid,
        num_slots,
        n_early,
        total,
        trace_addr,
    ):
        hst = Int32(storage.hstage.data_ptr().toint())
        # per-slot scalars, loaded once per launch, long before the first
        # accumulator is ready: [alpha1 | scale_c | alpha2] x MAX_SLOTS
        s_scal = storage.slot_scalars.get_tensor(cute.make_layout(3 * MAX_SLOTS))
        if tidx < num_slots:
            e_t = s_slot_expert[tidx]
            s_scal[tidx] = ptx.ld_nc_f32(alpha1_addr + e_t * 4)
            s_scal[MAX_SLOTS + tidx] = ptx.ld_nc_f32(scale_c_addr + e_t * 4)
            s_scal[2 * MAX_SLOTS + tidx] = ptx.ld_nc_f32(alpha2_addr + e_t * 4)
        ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        lane_off = (warp_idx * 32) << 16
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.acc_stages
        )
        v = cute.make_rmem_tensor((self.ntok,), Float32)
        # FC2: physical row 32 w + lane of a tile holds hidden index
        # 32 w + 4 (lane % 8) + lane // 8 (TRTLLM-gen's 32-row shuffle).
        hid_in_tile = warp_idx * 32 + (lane % 8) * 4 + lane // 8
        p = cta
        while p < total:
            is_fc2, slot, mt = self._decode(p, num_slots, n_early)
            mask = s_slot_mask[slot]
            acc_pipe.consumer_wait(acc_state)
            if cutlass.const_expr(self.trace != 0):
                if (tidx == 0) & (is_fc2 == 0):
                    jt = (p - cta) // grid
                    if jt < TR_MAX_TILES:
                        self._stamp(trace_addr, cta, TR_EPI0 + 3 * jt)
            taddr = (
                Int32(tCtAcc_base[(None, None, None, acc_state.index)].iterator.toint()) + lane_off
            )
            if cutlass.const_expr(self.ntok == NTOK):
                vals = ptx.tmem_ld_32x32b_x8(taddr)
            else:
                vals = ptx.tmem_ld_32x32b_xn(taddr, self.ntok)
            for n in cutlass.range_constexpr(self.ntok):
                v[n] = vals[n]
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipe.consumer_release(acc_state)
            acc_state.advance()
            if is_fc2 != 0:
                # TRTLLM-gen's FC2 epilogue: FMUL2 (IEEE) then cvt.rn.satfinite.bf16
                a2 = s_scal[2 * MAX_SLOTS + slot]
                col = mt * 128 + hid_in_tile
                for n in cutlass.range_constexpr(self.ntok):
                    if ((mask >> Uint32(n)) & Uint32(1)) != Uint32(0):
                        row = n * TOP_K + s_slot_k[slot * self.ntok + n]
                        ptx.st_u16(
                            rows_addr + (Int64(row) * HIDDEN + col) * 2,
                            ptx.bf16_bits(ptx.mul_rn(v[n], a2)),
                        )
            else:
                s_gate = s_scal[slot]
                s_c = s_scal[MAX_SLOTS + slot]
                self._swiglu_quant_to_stage(v, warp_idx, lane, mask, s_gate, s_c, num_tokens, hst)
                ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
                if cutlass.const_expr(self.trace != 0):
                    if tidx == 0:
                        jt = (p - cta) // grid
                        if jt < TR_MAX_TILES:
                            self._stamp(trace_addr, cta, TR_EPI0 + 3 * jt + 1)
                if cutlass.const_expr(self.delay & DELAY_PRODUCER):
                    self._delay_spin(DELAY_PRODUCER * 53, cta)
                if cutlass.const_expr(self.handoff != 0):
                    # tagged words: a reader takes a word exactly when its tag
                    # is this launch's epoch, so there is no flag and no fence
                    if cutlass.const_expr(self.g.tag_tile_words <= 32 * EPI_WARPS):
                        if tidx < self.g.tag_tile_words:
                            word = ptx.lds_u32(hst + tidx * 4)
                            ptx.st_relaxed_gpu_u64_pair(
                                img_addr
                                + Int64(slot) * self.g.tag_slot_bytes
                                + Int64(mt * self.g.tag_tile_bytes + tidx * 8),
                                word,
                                epoch,
                            )
                    else:
                        for part in cutlass.range_constexpr(
                            (self.g.tag_tile_words + 32 * EPI_WARPS - 1) // (32 * EPI_WARPS)
                        ):
                            wi = tidx + part * 32 * EPI_WARPS
                            if wi < self.g.tag_tile_words:
                                word = ptx.lds_u32(hst + wi * 4)
                                ptx.st_relaxed_gpu_u64_pair(
                                    img_addr
                                    + Int64(slot) * self.g.tag_slot_bytes
                                    + Int64(mt * self.g.tag_tile_bytes + wi * 8),
                                    word,
                                    epoch,
                                )
                    # the staging buffer is free for the next tile
                    ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
                    if cutlass.const_expr(self.trace != 0):
                        if tidx == 0:
                            jt = (p - cta) // grid
                            if jt < TR_MAX_TILES:
                                self._stamp(trace_addr, cta, TR_EPI0 + 3 * jt + 2)
                else:
                    img = img_addr + Int64(slot) * self.g.img_bytes
                    kb = mt // 4
                    if tidx < 2 * self.ntok:
                        # 16-byte chunk c16 = 2 (mt % 4) + half of row r, swizzled as in sB
                        r = tidx // 2
                        half = tidx - r * 2
                        c16 = 2 * (mt - kb * 4) + half
                        d = ptx.lds_v4_u32(hst + r * 32 + half * 16)
                        ptx.st_v4_u32(
                            img + kb * self.g.b_stage_bytes + r * 128 + ((c16 ^ self._sw(r)) * 16),
                            d[0],
                            d[1],
                            d[2],
                            d[3],
                        )
                        ptx.fence_proxy_async_global()
                    if (tidx >= 2 * self.ntok) & (tidx < 3 * self.ntok):
                        # the tile's four scale bytes of row r: compact SFB chunk mt
                        r = tidx - 2 * self.ntok
                        w1 = ptx.lds_u32(hst + self.g.hstage_sf + r * 4)
                        ptx.st_u32(img + self.g.img_b_bytes + mt * self.g.sfb_chunk + r * 16, w1)
                        ptx.fence_proxy_async_global()
                    # all image stores (and staging reads) precede the flag; the
                    # staging buffer is free for the next tile after this barrier
                    ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
                    if tidx == 0:
                        if cutlass.const_expr(self.trace != 0):
                            jt = (p - cta) // grid
                            if jt < TR_MAX_TILES:
                                self._stamp(trace_addr, cta, TR_EPI0 + 3 * jt + 2)
                        if cutlass.const_expr(self.publish == 0):
                            ptx.fence_acq_rel_gpu()
                            ptx.st_relaxed_gpu_u32(flag_addr + (slot * FC1_MT + mt) * 4, epoch)
                        else:
                            ptx.st_release_gpu_u32(flag_addr + (slot * FC1_MT + mt) * 4, epoch)
            if cutlass.const_expr(self.trace != 0):
                if tidx == 0:
                    self._tile_stamp(trace_addr, cta, (p - cta) // grid, 3)
            p = p + grid

    @cute.jit
    def _swiglu_quant_to_stage(self, v, warp_idx, lane, mask, s_gate, s_c, num_tokens, hst):
        """SwiGLU + NVFP4 quantization of one 128-row FC1 tile into the staging
        buffer: 8 rows x 32 FP4 bytes, then 8 rows x 4 scale bytes.

        Row ``32 w + q`` of the stock layout holds intermediate index
        ``64 mt + 16 w + j`` with ``j = 2 (q % 8) + q // 16`` as the up row
        when ``q % 16 < 8`` and its gate row at lane ``q + 8``
        (``reorder_rows_for_gated_act_gemm`` then the 32-row
        ``shuffle_matrix_a``), so one warp owns exactly one 16-element scale
        block of the intermediate. Tokens not routed to the expert get zero
        bytes and zero scales.

        The arithmetic is TRTLLM-gen's FC1 SwiGLU / NVFP4-output epilogue
        operation for operation, in its order and with its flush modes (read
        from the SASS of the ``Bmm_E2m1_E2m1E2m1_..._swiGlu_..._sm100f``
        cubins; FMUL.FTZ / MUFU are flushing, the packed FMUL2 / FADD2 /
        FFMA2 are IEEE). With ``sG = output1_scale_gate_scalar`` (alpha1),
        ``sC = output1_scale_scalar`` (scale_c), ``g`` / ``u`` the raw gate /
        up accumulators and no clamp, alpha or beta::

            t    = ftz(sG * log2 e)                    (negated here: g * -t == -(g * t))
            sig  = rcp.ftz(ex2.ftz(-(g * t)) + 1)
            h    = (u * sG + 0) * (g * sig)
            sf   = e4m3.rn.satfinite((amax|h| * sC) * f32(1/6))
            code = e2m1.rn.satfinite(ftz(h * rcp.ftz(sf)) * sC)    (0 when sf == 0)

        ``sC`` therefore enters after the block maximum and after the
        reciprocal of the scale, not into ``h``. Any other association is
        mathematically equal but moves some FP32 values by one ulp, which
        flips the FP4 code or block scale of values that sit on a rounding
        midpoint. Tokens go in pairs through the packed instructions where
        TRTLLM-gen uses them (``*_x2``: the same IEEE rounding per lane).
        """
        is_up = (lane % 16) < 8
        t_neg = ptx.mul_ftz(s_gate, Float32(-LOG2E_F32))
        one = Float32(1.0)
        rcp6 = Float32(RCP6_F32)
        for p in cutlass.range_constexpr(self.ntok // 2):
            n0 = 2 * p
            g0 = cute.arch.shuffle_sync_down(v[n0], 8)
            g1 = cute.arch.shuffle_sync_down(v[n0 + 1], 8)
            x = ptx.mul_rn_x2(g0, g1, t_neg, t_neg)
            d = ptx.add_rn_x2(ptx.ex2_approx(x[0]), ptx.ex2_approx(x[1]), one, one)
            gs = ptx.mul_rn_x2(g0, g1, ptx.rcp_approx_ftz(d[0]), ptx.rcp_approx_ftz(d[1]))
            u = ptx.fma_rn_zero_x2(v[n0], v[n0 + 1], s_gate, s_gate)
            hh = ptx.mul_rn_x2(u[0], u[1], gs[0], gs[1])
            h0 = Float32(0.0)
            h1 = Float32(0.0)
            if is_up:
                h0 = hh[0]
                h1 = hh[1]
            # (amax|h|) * sC == max(|h| * sC): rounding is monotone and sC > 0
            a = ptx.mul_rn_x2(self._block_amax(h0), self._block_amax(h1), s_c, s_c)
            q = ptx.mul_rn_x2(a[0], a[1], rcp6, rcp6)
            self._quant_store(h0, q[0], s_c, n0, mask, num_tokens, warp_idx, lane, hst)
            self._quant_store(h1, q[1], s_c, n0 + 1, mask, num_tokens, warp_idx, lane, hst)

    @cute.jit
    def _block_amax(self, h):
        """``max |h|`` over the 16 up lanes of a warp (one NVFP4 scale block)."""
        am = cute.math.absf(h)
        am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 1))
        am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 2))
        am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 4))
        am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 16))
        return am

    @cute.jit
    def _quant_store(self, h, q, s_c, n, mask, num_tokens, warp_idx, lane, hst):
        """Token ``n``: block scale ``e4m3(q)``, codes ``e2m1(ftz(h * rcp(sf)) * sC)``
        (0 when ``sf`` is 0), staged; zero bytes and scale when ``n`` is not routed."""
        sf = ptx.cvt_e4m3_rn_sat(q)
        sf_f = ptx.e4m3_to_f32(sf)
        osc = Float32(0.0)
        if sf_f != Float32(0.0):
            osc = ptx.rcp_approx_ftz(sf_f)
        partner = cute.arch.shuffle_sync_down(h, 16)
        c = ptx.mul_rn_x2(ptx.mul_ftz(h, osc), ptx.mul_ftz(partner, osc), s_c, s_c)
        byte = ptx.cvt_e2m1x2(c[0], c[1])
        routed = (((mask >> Uint32(n)) & Uint32(1)) != Uint32(0)) & (n < num_tokens)
        if not routed:
            byte = Uint32(0)
            sf = Uint32(0)
        if lane < 8:
            ptx.sts_u8(hst + n * 32 + warp_idx * 8 + lane, byte)
        if lane == 0:
            ptx.sts_u8(hst + self.g.hstage_sf + n * 4 + warp_idx, sf)

    @cute.jit
    def _delay_spin(self, salt, cta):
        """Stress testing: a pseudo-random third of the CTAs (re-drawn per
        launch from the global timer) spins 1..6 units of ``delay_ns``."""
        t0 = ptx.globaltimer_u64()
        seed = Uint32(Int32(t0 >> Int64(12)) & Int32(0xFFFF))
        h = (Uint32(cta) * Uint32(37) + seed * Uint32(101) + Uint32(salt)) % Uint32(17)
        if h < Uint32(6):
            d = Int64(h + Uint32(1)) * Int64(self.delay_ns)
            t = t0
            while t - t0 < d:
                t = ptx.globaltimer_u64()
