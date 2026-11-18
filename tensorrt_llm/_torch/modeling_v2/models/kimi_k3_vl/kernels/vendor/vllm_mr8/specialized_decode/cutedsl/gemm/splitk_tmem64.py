# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin split-K projection of 64 activation rows (spec verify B8 x q8) with the
weights on chip before x lands.

Each CTA of a ``split`` cluster owns one K slice of one 128-row weight tile, one CTA
per SM. Rubin co-schedules at most 28 clusters of 6 (24 of 7, 22 of 8) such CTAs
(``notes/gemm64.md``), so the KDA rows (26 tiles) take split 6 and the MLA rows
(23 tiles) split 7; the 112 K tiles (64 wide) are spread over the ranks as evenly as
they go (rank r < 112 % split gets one more). A slice's first 14 K tiles are staged
in tensor memory before the PDL wait (the weight warp streams them through a ring of
16 KB stages and the MMA warp copies each with ``tcgen05.cp`` as it lands); the rest
stay in their own shared-memory slots. After the PDL wait, once the ring has
drained, x's K tiles (64 rows x 64, 8 KB each) are loaded into ``slots`` 8 KB slots
of the ring's memory, each on its own barrier (a tile beyond the slots waits for
the MMAs of the tile before it in its slot). The M128 N64 K16 MMAs read A from
tensor memory (35.6 cycles vs 49.8 from shared memory, ``notes/gemm64.md``).

Reduce-scatter over the cluster in one DSMEM hop: once every rank's MMAs are done
(a cluster barrier: the owners' mailboxes reuse the A slots and x's slots), each
epilogue thread pushes its accumulator row (one TMEM lane, 64 token columns) to the
owner of its 128 / ``owners`` row share (16-byte ``st.async``); owners sum the
partials in rank order, ((p0 + p1) + p2) + ..., round once to BF16 and store their
rows. Versus ``SplitKGemm`` at (64, 64, 2, stages) this reorders the FP32
accumulation (``split`` partials over K / ``split`` instead of two over K / 2)."""

from typing import Any

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05

from ..attn_res.kernel import BESIDE_SMEM
from ..common.primitives import (
    asm,
    at,
    bits_float,
    float_bits,
    load_tmem,
    named_barrier,
    push_float,
    store_words,
)
from ..kda.head import _copy_to_tmem
from .splitk import (
    SplitKGemm,
    SplitKTactic,
    _acquire_tensor_map,
    _fence_tcgen05,
    _smem,
    _wait_cluster,
)

__all__ = ["N_TILES", "SplitKTmemGemm64", "acc_columns"]

# Token columns per N tile: 64 / 128 (the shipped bins), 80 / 96 (65-96 rows: one
# tile of 16-row granularity instead of the generic split-K above 64 rows).
N_TILES = (64, 80, 96, 128)


def acc_columns(n):
    """Tensor-memory columns of an N-column accumulator: whole 32-column A tiles."""
    return (n + 31) // 32 * 32


def _load_row(address, n):
    """``n`` FP32 TMEM columns of this lane from ``address``, as ``tcgen05.ld``
    32x32b loads of descending power-of-two counts (a plain trace-time helper)."""
    values, column, rest = (), 0, n
    while rest:
        count = 1 << (rest.bit_length() - 1)
        part = load_tmem(address + column, count)
        values += part if count > 1 else (part,)
        column, rest = column + count, rest - count
    return values

A_STAGE, B_TILE = 128 * 64 * 2, 64 * 64 * 2  # bytes: one A box, one x tile
TMEM_TILES = 14  # A tiles beside the 64-column accumulator in 512 TMEM columns
OWNERS = 4  # ranks that reduce and store a 32-row share (default)
SHARE = 128 // OWNERS
# Epilogue shared memory (bytes into the A slots + x's slots): the owners' mailboxes
# [split][16 quads][share rows][4] FP32 at 0, an owner's [64 tokens][share]
# output tile at TILE_OFFSET.
TILE_OFFSET = 64 * 1024
# kind::f16 instruction descriptor: D FP32, A and B BF16, both K-major, N 64, M 128
IDESC_M128_N64 = 0x08000490 | (64 >> 3) << 17
# Shared-memory descriptor high word: SBO 1024 B, version 1, 128-byte swizzle
DESC_HIGH = 0x40004040
# Shared memory requested per CTA: above half of Rubin's ~328 KB per SM (front4), so
# no two CTAs of this kernel share an SM (the second waits for the first's tensor
# memory: in a split-8 prototype behind a 112 x 200 KB predecessor, half the K
# splits landed their first weight tile 10 us late).
ALONE_SMEM = 170 * 1024
# Stamp slots (``stamps``, benchmarks only): 0 entry, 1 every A tile requested,
# 2 ring drained, 3 PDL wait returned (x warp), 4 / 5 first / last x tile landed,
# 6 MMAs done, 7 cluster barrier passed, 8 every partial received, 9 output stored,
# 10 / 11 first / last ring A tile landed, 12 %smid.
STAMPS = 13


@asm()
def _tma_box(dst, barrier, tensor_map, k0, row0):
    """One {64, 128} box of A at (k0, row0) to shared address ``dst``, L2
    evict_first, completing on ``barrier``."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    operands = (Int32(dst), tensor_map, Int32(k0), Int32(row0), barrier)
    return code, "r,l,r,r,r,~{memory}", operands


@asm()
def _tma_box3(dst, barrier, tensor_map, k0, t0, b0):
    """One {64, 8, 8} box of a 3-D x map (K, tokens per request, requests) at
    (k0, t0, b0) to shared address ``dst``, completing on ``barrier``: row 8 b + t of
    the 64-row tile, so each request takes 8 rows whatever its token count (the map's
    rows past it read as zeros)."""
    code = "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes [$0], [$1, {$2, $3, $4}], [$5];"
    operands = (Int32(dst), tensor_map, Int32(k0), Int32(t0), Int32(b0), barrier)
    return code, "r,l,r,r,r,r,~{memory}", operands


@asm()
def _mma_tmem64(d, a, b_desc, accumulate, idesc=IDESC_M128_N64):
    """D[tmem] (+)= A[tmem] B[smem]^T, M128 N64 K16 (``idesc``: N128)."""
    code = "{ .reg .pred p; .reg .b64 bd; .reg .b32 bh; setp.ne.b32 p, $3, 0; "
    code += f"mov.b32 bh, {DESC_HIGH}; mov.b64 bd, {{$2, bh}}; "
    code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], bd, {idesc}, p; }}"
    operands = (Int32(d), Int32(a), Int32(b_desc), Int32(accumulate))
    return code, "r,r,r,r,~{memory}", operands


@asm()
def _mma_smem64(d, a_desc, b_desc, accumulate, idesc=IDESC_M128_N64):
    """D[tmem] (+)= A[smem] B[smem]^T, M128 N64 K16 (low descriptor words;
    ``idesc``: N128)."""
    code = "{ .reg .pred p; .reg .b64 ad, bd; .reg .b32 h; setp.ne.b32 p, $3, 0; "
    code += f"mov.b32 h, {DESC_HIGH}; mov.b64 ad, {{$1, h}}; mov.b64 bd, {{$2, h}}; "
    code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], ad, bd, {idesc}, p; }}"
    operands = (Int32(d), Int32(a_desc), Int32(b_desc), Int32(accumulate))
    return code, "r,r,r,r,~{memory}", operands


@asm()
def _st_shared_v4(address, words):
    code = "st.shared.v4.b32 [$0], {$1, $2, $3, $4};"
    return code, "r,r,r,r,r,~{memory}", (Int32(address), *(Int32(w) for w in words))


@asm((Uint32,) * 4)
def _ld_shared_v4(address):
    code = "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (Int32(address),)


@asm()
def _st_shared_bf16(address, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.shared.b16 [$0], h; }"
    return code, "r,f,~{memory}", (Int32(address), Float32(value))


@asm()
def _cluster_arrive():
    return "barrier.cluster.arrive.release;", "~{memory}", ()


@asm()
def _cluster_wait():
    return "barrier.cluster.wait.acquire;", "~{memory}", ()


@asm()
def _cluster_phases():
    """Arrive on and pass a cluster barrier phase (threads without epilogue work)."""
    code = "barrier.cluster.arrive.release; barrier.cluster.wait.acquire;"
    return code, "~{memory}", ()


@asm()
def _globaltimer_store(address, when):
    code = "{ .reg .u64 t; .reg .pred p; setp.ne.b32 p, $1, 0; "
    code += "mov.u64 t, %globaltimer; @p st.global.u64 [$0], t; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


@asm()
def _smid_store(address, when):
    code = "{ .reg .u32 s; .reg .u64 w; .reg .pred p; setp.ne.b32 p, $1, 0; "
    code += "mov.u32 s, %smid; cvt.u64.u32 w, s; @p st.global.u64 [$0], w; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


class SplitKTmemGemm64(SplitKGemm):
    """``tactic`` (128, N, split, slots), N 64 or 128 token columns (N128: 12 A tiles
    in tensor memory beside a 128-column accumulator, 16 KB x tiles, mailboxes and
    the owner's tile sized by N): ``slots`` x's tiles (also the pre-wait ring, of
    16 KB stages); ``k``: the reduction length. Arguments
    as ``SplitKGemm`` (``bias`` unused, or with ``stamps`` (benchmarks only) Int64
    [CTAs, 16] %globaltimer stamps)."""

    def __init__(
        self, *, tactic: SplitKTactic, k: int, stamps: bool = False, alone: bool = True,
        owners: int = OWNERS,
    ):  # fmt: skip
        super().__init__(tactic=tactic)
        assert tactic.mma_m == 128 and tactic.mma_n in N_TILES, "M128 N64-N128"
        # the owners' sums read whole quads: N x share / 128 tokens per thread, a
        # multiple of 4 (N80 needs share 32, i.e. 4 owners)
        assert tactic.mma_n * (128 // owners) // 128 % 4 == 0, "N x share / 128 % 4"
        # N token columns: the accumulator's TMEM columns (``acc_columns``: N rounded
        # up to whole 32-column A tiles), the TMEM-staged A tiles beside it (14 at
        # N64, 13 at N80 / N96, 12 at N128), x's tile bytes per K tile.
        self.n = tactic.mma_n
        self.acc_columns = acc_columns(self.n)
        self.tmem_tiles = (512 - self.acc_columns) // 32
        self.b_tile = self.n * 64 * 2
        self.idesc = 0x08000490 | (self.n >> 3) << 17
        self.cta_k = 64
        self.k_total = k // self.cta_k
        assert self.k_total * self.cta_k == k
        split = self.split_k
        self.k_tiles = -(-self.k_total // split)  # the longest rank's tiles
        self.extra = self.k_total % split  # ranks below it hold k_tiles
        self.owners, self.share = owners, 128 // owners
        tmem_tiles = self.tmem_tiles
        assert self.k_tiles >= tmem_tiles and split >= owners and self.share % 4 == 0
        self.smem_tiles = self.k_tiles - tmem_tiles  # A tiles kept in shared memory
        self.slots = tactic.ab_stages  # x's tiles in flight (8 KB each)
        assert self.slots * self.b_tile % A_STAGE == 0 and self.slots <= self.k_tiles
        self.num_ab_stage = self.slots * self.b_tile // A_STAGE  # 16 KB ring stages
        self.stamps = stamps
        self.stamp_base: Any = None
        # Shared memory: A slots, x's slots (the ring before), barriers; the owners'
        # mailboxes [split][16 quads][share rows][4] FP32 reuse the first two.
        self.mailbox_bytes = split * self.share * self.n * 4
        self.tile_offset = TILE_OFFSET  # the owner's output tile past the mailboxes
        if self.n != 64:
            self.tile_offset = (self.mailbox_bytes + 1023) // 1024 * 1024
        data = self.smem_tiles * A_STAGE + self.slots * self.b_tile
        assert self.mailbox_bytes <= self.tile_offset
        assert data >= self.tile_offset + self.n * self.share * 4
        used = data + (2 * self.num_ab_stage + 2 * self.k_tiles + 6) * 8 + 4
        self.smem_bytes = (used + 2048 + 1023) // 1024 * 1024
        if alone:  # ``alone`` False: few CTAs, beside small-smem predecessors
            self.smem_bytes = max(self.smem_bytes, ALONE_SMEM)
        assert self.smem_bytes <= BESIDE_SMEM  # the in-projections beside an AttnRes

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, bias, a_map).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mS: cute.Tensor,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major."""
        self.body(mA, mB, mS, a_map, (mC,))

    @cute.experimental.jit
    def body(
        self, mA: cute.Tensor, mB: cute.Tensor, mS: cute.Tensor, a_map: cutlass.Int64,
        out, x_map=None,
    ):  # fmt: skip
        """The kernel; ``out``: the epilogue's arguments (``finish``); ``x_map``: a
        3-D x tensor map (K, tokens per request, requests) instead of ``mB``'s 2-D
        tiles (``x_warp64``): request r of the N tile in rows 8 r .. 8 r + 7."""
        stages, k_tiles = self.num_ab_stage, self.k_tiles
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        bidx, by, l_idx = cute.arch.block_idx()
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = by // self.split_k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if cutlass.const_expr(self.stamps):
            grid_x, _, _ = cute.arch.grid_dim()
            cta = bidx + grid_x * by
            self.stamp_base = mS.iterator.toint() + Int64(cta) * 128
        # This rank's K tiles: [first, first + tiles)
        base_tiles = self.k_total // self.split_k
        tiles = base_tiles + Int32(split_rank < self.extra)
        over = Int32(split_rank > self.extra) * (split_rank - self.extra)
        first = split_rank * (base_tiles + 1) - over

        # Shared memory: [A slots][x's slots] (x's slots are the pre-wait ring).
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, self.slots)
        b_layout = sm100_utils.make_smem_layout_b(*b_shape)
        if cutlass.const_expr(self.smem_tiles > 0):
            a_slots = _smem(cutlass.Uint8, self.smem_tiles * A_STAGE, 1024)
            data = Int32(a_slots.iterator.toint())
        sB = _smem(ab_dtype, None, 1024, b_layout)
        ring = Int32(sB.iterator.toint())
        if cutlass.const_expr(self.smem_tiles == 0):
            data = ring
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        a_full = _smem(cutlass.Int64, max(self.smem_tiles, 1), 8).iterator
        b_full = _smem(cutlass.Int64, k_tiles, 8).iterator
        b_empty = _smem(cutlass.Int64, k_tiles, 8).iterator
        bar_drained = _smem(cutlass.Int64, 1, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 1)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                for i in range(max(self.smem_tiles, 1)):
                    cute.arch.mbarrier_init(a_full + i, 1)
                for i in range(k_tiles):
                    cute.arch.mbarrier_init(b_full + i, 1)
                    cute.arch.mbarrier_init(b_empty + i, 1)
                cute.arch.mbarrier_init(bar_drained, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic
        cute.arch.cluster_wait()
        if warp_idx == 0:
            self.stamp(0, cute.arch.lane_idx() == 0)
            if cutlass.const_expr(self.stamps):
                _smid_store(self.stamp_base + 12 * 8, cute.arch.lane_idx() == 0)

        # Warp 3 is idle; warps 4-7 run the epilogue. One more cluster barrier phase:
        # the mailboxes are free (the epilogue's arrival follows its CTA's MMAs);
        # warps 0-3 pass it after their roles.
        if warp_idx == 0:
            self.weight_warp64(bar_full, bar_empty, a_full, ring, data, a_map,
                               bidx * self.cta_m, first)  # fmt: skip
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(bar_drained)
            _cluster_phases()
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            cute.arch.mbarrier_wait(bar_drained, 0)
            if cutlass.const_expr(x_map is None):
                self.x_warp64(b_full, b_empty, b_tile, sB, b_map, first, tiles)
            else:
                self.x_warp3d(b_full, b_empty, ring, x_map, first, tiles, n_idx)
            _cluster_phases()
        elif warp_idx == 2:
            self.staged_mma_warp64(
                bar_full, bar_empty, a_full, b_full, b_empty, bar_mma_epilog,
                bar_tmem_alloc, ring, data, tmem_base_ptr, tiles,
            )  # fmt: skip
            _cluster_phases()
        elif warp_idx == 3:
            _cluster_phases()
        else:
            self.reduce_scatter64(
                cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc,
                tmem_base_ptr, data, bar_reduce, split_rank, out,
                bidx * self.cta_m, n_idx * self.cta_n, l_idx,
            )  # fmt: skip

    def issue_a(self, dst, barrier, a_map, k0, row0):
        """One A K tile (128 rows from ``row0``, 64 columns from ``k0``) to ``dst``."""
        _tma_box(dst, barrier, a_map, k0, row0)

    def owner_of(self, row, row0):
        """(owner rank, row in its share) of accumulator row ``row``."""
        return row // self.share, row % self.share

    def stamp(self, slot, when):
        """%globaltimer into this CTA's stamp ``slot`` where ``when`` (``stamps``)."""
        if self.stamps:
            _globaltimer_store(self.stamp_base + slot * 8, Int32(when))

    @cute.experimental.jit
    def weight_warp64(
        self, bar_full, bar_empty, a_full, ring: Int32, data: Int32,
        a_map: cutlass.Int64, row0: cutlass.Int32, first: cutlass.Int32,
    ):  # fmt: skip
        """A, before the PDL wait: one evict_first {64, 128} box per K tile, the
        first TMEM_TILES into ring stage k % stages, the rest into their A slots; the
        dependents' trigger once every tile is requested; then the ring's drain
        (every ring tile copied to tensor memory)."""
        stages = self.num_ab_stage
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(self.tmem_tiles, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, A_STAGE)
                k0 = (first + k_tile) * self.cta_k
                self.issue_a(ring + stage * A_STAGE, bar_full + stage, a_map, k0, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
            if cutlass.const_expr(self.smem_tiles > 0):  # noqa: SIM102
                if k_tile == 0:  # the A slots' tiles right behind the first stage
                    with cute.arch.elect_one():
                        for s in cutlass.range_constexpr(self.smem_tiles):
                            k_a = (first + self.tmem_tiles + s) * self.cta_k
                            dst = data + s * A_STAGE
                            cute.arch.mbarrier_arrive_and_expect_tx(a_full + s, A_STAGE)
                            self.issue_a(dst, a_full + s, a_map, k_a, row0)
        self.stamp(1, cute.arch.lane_idx() == 0)
        cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, self.tmem_tiles)
        self.stamp(2, cute.arch.lane_idx() == 0)

    @cute.experimental.jit
    def x_warp64(
        self, b_full, b_empty, g_tile: cute.Tensor, s_tile: cute.Tensor,
        cta_v_map: cute.Layout, first: cutlass.Int32, tiles: cutlass.Int32,
    ):  # fmt: skip
        """Loads x's K tiles after the PDL wait, each on its own barrier; tile
        k >= slots waits for the MMAs of tile k - slots (its slot's previous tile)."""
        tile_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        self.stamp(3, cute.arch.lane_idx() == 0)
        for k_tile in cutlass.range(tiles, unroll=1):
            slot = k_tile % self.slots
            if k_tile >= self.slots:
                cute.arch.mbarrier_wait(b_empty + k_tile - self.slots, 0)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + k_tile, tile_bytes)
            cute_ext.tma_load(
                g_tile[None, None, first + k_tile],
                s_tile[None, None, None, slot],
                (b_full + k_tile).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def x_warp3d(
        self, b_full, b_empty, ring: Int32, x_map: cutlass.Int64,
        first: cutlass.Int32, tiles: cutlass.Int32, n_idx: cutlass.Int32,
    ):  # fmt: skip
        """``x_warp64`` through a 3-D x map: K tile k of the N tile's 8 requests as one
        {64, 8, 8} box (8 KB, the same 128-byte-swizzled rows as a 64-row 2-D box)."""
        assert self.n == 64, "3-D x boxes fill a 64-row tile"
        _acquire_tensor_map(x_map)
        cute.arch.griddepcontrol_wait()
        self.stamp(3, cute.arch.lane_idx() == 0)
        for k_tile in cutlass.range(tiles, unroll=1):
            slot = k_tile % self.slots
            if k_tile >= self.slots:
                cute.arch.mbarrier_wait(b_empty + k_tile - self.slots, 0)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + k_tile, B_TILE)
                k0 = (first + k_tile) * self.cta_k
                _tma_box3(
                    ring + slot * B_TILE, b_full + k_tile, x_map, k0, 0, 8 * n_idx
                )

    @cute.experimental.jit
    def staged_mma_warp64(
        self, bar_full, bar_empty, a_full, b_full, b_empty, bar_mma_epilog,
        bar_tmem_alloc, ring: Int32, data: Int32, tmem_base_ptr,
        tiles: cutlass.Int32,
    ):  # fmt: skip
        """The ring's tiles to tensor memory as they land (each stage released for
        the next tile), then every tile's four MMAs in K order once x's tile has
        landed: A from tensor memory for the first TMEM_TILES, then from the A
        slots."""
        stages, k_blocks = self.num_ab_stage, self.cta_k // 16
        a_step, b_step = A_STAGE // 16, self.b_tile // 16  # descriptor units
        tmem_tiles, acc_columns = self.tmem_tiles, self.acc_columns
        columns = self.cta_k // 2  # tensor-memory columns of one A tile
        cute.arch.alloc_tmem(512, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        ring_desc = ((ring >> 4) & 0x3FFF) | 0x10000
        data_desc = ((data >> 4) & 0x3FFF) | 0x10000

        for k in cutlass.range_constexpr(tmem_tiles):
            stage, phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + stage, phase)
            if cutlass.const_expr(k == 0 or k == tmem_tiles - 1):
                self.stamp(10 if k == 0 else 11, cute.arch.lane_idx() == 0)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    source = ring_desc + stage * a_step + 2 * kb
                    _copy_to_tmem(base + acc_columns + columns * k + 8 * kb, source)
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
        for k in cutlass.range_constexpr(self.k_tiles):
            if k < tiles:
                cute.arch.mbarrier_wait(b_full + k, 0)
                if cutlass.const_expr(k == 0):
                    self.stamp(4, cute.arch.lane_idx() == 0)
                if cutlass.const_expr(k >= tmem_tiles):
                    cute.arch.mbarrier_wait(a_full + k - tmem_tiles, 0)
                _fence_tcgen05()
                with cute.arch.elect_one():
                    for kb in cutlass.range_constexpr(k_blocks):
                        b = ring_desc + (k % self.slots) * b_step + 2 * kb
                        accumulate = Int32(0 if k == 0 and kb == 0 else 1)
                        if cutlass.const_expr(k < tmem_tiles):
                            a = base + acc_columns + columns * k + 8 * kb
                            _mma_tmem64(base, a, b, accumulate, idesc=self.idesc)
                        else:
                            a = data_desc + (k - tmem_tiles) * a_step + 2 * kb
                            _mma_smem64(base, a, b, accumulate, idesc=self.idesc)
                    if cutlass.const_expr(k + self.slots < self.k_tiles):
                        tcgen05.commit(b_empty + k, None, self.cta_group)
        self.stamp(5, cute.arch.lane_idx() == 0)
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        cute.arch.dealloc_tmem(tmem_ptr, 512, is_two_cta=False)

    @cute.experimental.jit
    def reduce_scatter64(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        data: Int32, bar_reduce, split_rank: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """Accumulator row ``epi_tid`` (64 FP32 token columns) -> slot
        ``split_rank`` of its owner's mailbox [split][16 quads][SHARE rows][4]
        (``st.async`` after the cluster barrier); owners sum the slots in rank order
        and ``finish``."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        quadrant = epi_tid // 32  # warps 4-7 read TMEM lanes 32 (warp % 4) ..
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        self.stamp(6, epi_tid == 0)
        _fence_tcgen05()
        values = _load_row(base + (quadrant * 32 << 16), self.n)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        split, share = self.split_k, self.share
        quads: cutlass.Constexpr = self.n // 4  # a row's token quads
        slot_bytes = quads * share * 16
        if (epi_tid == 0) & (split_rank < self.owners):
            expected = (split - 1) * slot_bytes
            cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
        _cluster_arrive()  # this CTA's MMAs are done: its slots may take partials
        _cluster_wait()
        self.stamp(7, epi_tid == 0)
        # The epilogue's own inputs, loaded under the reduce-scatter (None: none;
        # after the cluster barrier, whose release would wait for the loads).
        early = self.finish_inputs(epi_tid, out, row0 + split_rank * self.share, col0)
        finish_out = out if early is None else (*out, early)
        owner, r = self.owner_of(epi_tid, row0)
        slot = data + (split_rank * quads * share + r) * 16  # at the owner's address
        mailbox = cute.make_ptr(Float32, slot, cute.AddressSpace.smem, assumed_align=16)
        if owner == split_rank:
            for i in cutlass.range_constexpr(quads):
                bits = tuple(float_bits(values[4 * i + j]) for j in range(4))
                _st_shared_v4(slot + i * share * 16, bits)
        else:
            for i in cutlass.range_constexpr(quads):
                quad = tuple(values[4 * i + j] for j in range(4))
                push_float(mailbox + i * share * 4, quad, bar_reduce, owner)
        if split_rank < self.owners:
            named_barrier(3, 128)  # this CTA's own slot
            _wait_cluster(bar_reduce, Int32(0))
            self.stamp(8, epi_tid == 0)
            # thread -> (row rr, tokens width g .. width (g + 1) - 1)
            width: cutlass.Constexpr = self.n * share // 128
            rr, g = epi_tid % share, epi_tid // share
            totals = cute.make_rmem_tensor(width, Float32)
            for s in cutlass.range_constexpr(split):
                for v in cutlass.range_constexpr(width // 4):
                    offset = ((s * quads + width // 4 * g + v) * share + rr) * 16
                    words = _ld_shared_v4(data + offset)
                    for j in cutlass.range_constexpr(4):
                        part = bits_float(words[j])
                        if cutlass.const_expr(s == 0):
                            totals[4 * v + j] = part
                        else:
                            totals[4 * v + j] = totals[4 * v + j] + part
            row = row0 + split_rank * share
            tile = data + self.tile_offset
            self.finish(epi_tid, totals, tile, finish_out, row, col0, l_idx)
            self.stamp(9, epi_tid == 0)

    def finish_inputs(self, epi_tid, out, row0, col0):
        """Epilogue inputs an owner of rows [row0, row0 + share) loads before the
        reduce-scatter, passed to ``finish`` as the last item of ``out``; None:
        nothing."""
        return None

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """An owner's rows [row0, row0 + SHARE) of tokens col0 ..: thread (row rr,
        tokens width g .. width g + width - 1, width = N / 4) holds ``totals``; rounded
        to BF16, transposed through the shared-memory ``tile`` [N tokens][SHARE] and
        stored as 16-byte rows of C (``out`` = (C,)). Default owners only (share =
        SHARE)."""
        mC = out[0]
        rr, g = epi_tid % SHARE, epi_tid // SHARE
        width: cutlass.Constexpr = self.n * SHARE // 128  # tokens per thread
        for t in cutlass.range_constexpr(width):
            _st_shared_bf16(tile + ((width * g + t) * SHARE + rr) * 2, totals[t])
        named_barrier(3, 128)
        rows, columns = mC.layout.shape[0], mC.layout.shape[1]
        if cutlass.const_expr(self.n == 64):
            token, half = epi_tid // 2, epi_tid % 2
            row = row0 + 16 * half
            if (row < rows) & (col0 + token < columns):
                for c in cutlass.range_constexpr(2):
                    source = tile + (token * SHARE + 16 * half + 8 * c) * 2
                    words = _ld_shared_v4(source)
                    store_words(at(mC, row + 8 * c, col0 + token, l_idx), words)
        else:  # N128: a token per thread, its SHARE rows in 8-row stores
            token = epi_tid
            if col0 + token < columns:
                for c in cutlass.range_constexpr(SHARE // 8):
                    if row0 + 8 * c < rows:  # a last tile's share may hold fewer rows
                        words = _ld_shared_v4(tile + (token * SHARE + 8 * c) * 2)
                        store_words(at(mC, row0 + 8 * c, col0 + token, l_idx), words)
