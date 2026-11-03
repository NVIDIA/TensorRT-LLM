# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused 3x3 conv (padding 1) for Blackwell (sm100): 1-D Winograd F(2,3) along W, direct taps
along H. The input is channels-last bf16 or fp16; the output element type (bf16 or fp16)
follows the destination tensor. Launches on the caller's torch stream.

Per output tile of 2 px along W, the 4 Winograd coordinates j give 4 GEMMs
  M_j[tile, cout] = sum_{r in 0..2, c} V_j[input row h+r-1, tile, c] * U_j[r][cout, c]
with V = B^T d (input transform along W, fp16) and U = G g (filter transform, fp16), and
  y[2t] = M0 + M1 + M2,   y[2t+1] = M1 - M2 - M3.

The output transform is folded into the accumulators so that only TWO fp32 accumulators of
the full Cout = 256 width are needed (2 x 256 = all 512 TMEM columns):
  pass 0 (coords 1, 2, all Cin):  A = M1, B = M2
  butterfly in TMEM (epilogue warps): A <- A + B, B <- A - B
  pass 1 (coords 0, 3, all Cin):  A += M0 = y[2t],  B += U3 (d3 - d1) = -M3 -> y[2t+1]
Every MMA is cta_group::2 with M = 256. Most Cin chunks run at N = 256 (256 x 256 x 16); the
last chunk of pass 0 and the first chunk of pass 1 run as two N = 128 halves, so the butterfly
of one Cout half overlaps the MMAs of the other.

One CTA pair owns 2 x 128 tiles (each CTA: 8 rows x 32 px) and all 256 Cout. Raw input boxes
are TMA-loaded into SMEM and transformed by 4 transform warps into V (fp16, UMMA K-major SW64)
in SMEM (A operand; the H taps are 16-row offsets into V). U is TMA-loaded as the 2-CTA split
B operand, multicast between the two pairs of a 4-CTA cluster (no multicast when the launch
falls back to 2-CTA clusters). V and M never leave the SM; the epilogue converts the outputs
in registers and writes them through a per-warp SMEM box with TMA stores.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Vector
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import Int32, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ..common import CHANNELS_PER_VECTOR

RC = 8  # output rows per CTA block
WT = 16  # Winograd tiles per row (2 px each)
BOXW = 2 * WT + 3  # 35 px: 34 needed (+1 so row stride = 64 mod 128 B)
BOXH = RC + 2  # 10 input rows
KC = 32  # input channels per chunk (64 B rows, SW64)
NACC = 2  # accumulators (coordinate pair per pass)
NTAP = 3
NPASS = 2
VROWS = BOXH * WT  # 160 V rows per coordinate
MT = RC * WT  # 128 tiles per CTA (MMA M per CTA)
NCO = 256  # Cout per MMA (N) = all output channels
RAW_BYTES = BOXH * BOXW * KC * 2
VCOORD_BYTES = VROWS * KC * 2
VSTAGE_BYTES = NACC * VCOORD_BYTES
SLOT_CTAS = 4  # CTAs per slot: one 4-CTA cluster or two 2-CTA clusters

NUM_U_STAGE = 6
U_GROUP = 2  # (coord, tap) U tiles per pipeline stage
NUM_V_STAGE = 3
NUM_RAW_STAGE = 2
EPI_COLS = 64  # TMEM columns per butterfly step
DR_COLS = 16  # TMEM columns per drain step (= Cout per TMA store box)
EPI_BYTES = 2 * 2 * WT * DR_COLS * 2  # 2 KB: per-warp store box (2 rows x 32 px x 16 Cout)
STORE_STEPS = NCO // DR_COLS // 2  # 16-Cout TMA stores per Cout half (8)
DRAIN_LOADS = STORE_STEPS // 2  # 32-column TMEM loads per Cout half and accumulator (4)

FILTER_TILE = 16  # filter-transform block: 16 Cout x 16 Cin
FILTER_THREADS = FILTER_TILE * FILTER_TILE


@cute.struct
class _Barriers:
    """Shared memory of the pipeline mbarriers and the TMEM allocator."""

    u_mbar: cute.struct.MemRange[cutlass.Int64, NUM_U_STAGE * 2]
    v_mbar: cute.struct.MemRange[cutlass.Int64, NUM_V_STAGE * 2]
    x_mbar: cute.struct.MemRange[cutlass.Int64, NUM_RAW_STAGE * 2]
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    acc_b_free_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    acc_a_final_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    pass0_lo_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    pass0_hi_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def supports(cin: int, cout: int, h: int, w: int) -> bool:
    """Shapes the fused conv covers: 256 output channels, at least two 32-channel input chunks
    (the MMA schedule peels a first and a last chunk per pass), ``h`` a multiple of 8 and ``w``
    a multiple of 32."""
    return cout == NCO and cin % KC == 0 and cin >= 2 * KC and h % RC == 0 and w % (2 * WT) == 0


def _f16x2_cvt_asm(regs, in_half):
    # PTX converting the packed input pairs p<i> to f16x2 in q<i>: bf16 pairs through fp32,
    # fp16 pairs as they are.
    s = ""
    for i in regs:
        if in_half:
            s += f"mov.b32 q{i}, p{i};\n\t"
        else:
            s += (
                f"shl.b32 lo, p{i}, 16;\n\tand.b32 hi, p{i}, 0xffff0000;\n\t"
                f"cvt.rn.f16x2.f32 q{i}, hi, lo;\n\t"
            )
    return s


def _wino_item_asm(in_half, pass_idx):
    # src: px d0 of 4 consecutive px (64 B apart), 8 bf16/fp16 channels each, loaded into
    # p<4 * px> .. p<4 * px + 3>; dst: coord slot 0, slot 1 at +VCOORD_BYTES.
    #   pass 0: V1 = d1 + d2, V2 = d2 - d1
    #   pass 1: V0 = d0 - d2, V3' = d3 - d1 (= -V3, folds the minus sign of M3 into V)
    pixels = (1, 2) if pass_idx == 0 else (0, 1, 2, 3)
    loads = ""
    for px in pixels:
        off = f"+{px * KC * 2}" if px else ""
        loads += f"ld.shared.v4.b32 {{p{4 * px},p{4 * px + 1},p{4 * px + 2},p{4 * px + 3}}}"
        loads += f", [$0{off}];\n\t"
    arith = ""
    for i in range(4):
        if pass_idx == 0:
            arith += f"add.f16x2 v{i}, q{4 + i}, q{8 + i};\n\t"
            arith += f"sub.f16x2 w{i}, q{8 + i}, q{4 + i};\n\t"
        else:
            arith += f"sub.f16x2 v{i}, q{i}, q{8 + i};\n\t"
            arith += f"sub.f16x2 w{i}, q{12 + i}, q{4 + i};\n\t"
    regs = [4 * px + k for px in pixels for k in range(4)]
    return (
        "{\n\t.reg .b32 p<16>, q<16>, v<4>, w<4>, lo, hi;\n\t"
        + loads
        + _f16x2_cvt_asm(regs, in_half)
        + arith
        + "st.shared.v4.b32 [$1], {v0,v1,v2,v3};\n\t"
        f"st.shared.v4.b32 [$1+{VCOORD_BYTES}], {{w0,w1,w2,w3}};\n\t"
        "}\n"
    )


@dsl_user_op
def _wino_item(src: Int32, dst: Int32, in_half: bool, pass_idx: int, *, loc=None, ip=None):
    # One input-transform item of pass ``pass_idx`` (see _wino_item_asm).
    llvm.inline_asm(
        None,
        [Int32(src).ir_value(loc=loc, ip=ip), Int32(dst).ir_value(loc=loc, ip=ip)],
        _wino_item_asm(in_half, pass_idx),
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _bfly2(a0, a1, b0, b1):
    """``(a + b, a - b)`` on two fp32 lanes: one packed f32x2 add and one packed sub."""
    s = cute.arch.add_packed_f32x2((a0, a1), (b0, b1))
    d = cute.arch.sub_packed_f32x2(
        Vector.from_elements((a0, a1), cutlass.Float32),
        Vector.from_elements((b0, b1), cutlass.Float32),
    )
    return [s[0], s[1], cutlass.Float32(d[0]), cutlass.Float32(d[1])]


@dsl_user_op
def _fence_proxy_async_smem(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _Sync(NamedTuple):
    """Pipelines, TMEM allocator and barriers of one CTA, shared by its warp roles."""

    u: pipeline.PipelineTmaUmma  # U tiles: U TMA warp -> MMA warp
    v: pipeline.PipelineAsyncUmma  # V stages: transform warps -> MMA warp
    x: pipeline.PipelineTmaAsync  # raw x boxes: x TMA warp -> transform warps
    # Pass-0 result per Cout half (MMA -> butterfly warps of that half); the butterfly warps'
    # release lets the pass-1 MMAs of that half start.
    pass0_lo: pipeline.PipelineUmmaAsync
    pass0_hi: pipeline.PipelineUmmaAsync
    # Pass-1 result (accumulator B final) -> drain of B. The epilogue releases it right after
    # the wait; the next block's MMAs are gated by acc_a_final and acc_b_free instead.
    acc: pipeline.PipelineUmmaAsync
    # Accumulator B drained: the epilogue's commit lets the next block's coord-2 MMAs (into B)
    # start.
    acc_b_free: pipeline.PipelineAsyncUmma
    # Accumulator A final: committed after the last coord-0 MMAs of pass 1, so the epilogue
    # drains y0 while the tensor core is still on coord 3 (accumulator B); the epilogue's
    # release lets the next block's coord-1 MMAs (into A) start.
    acc_a_final: pipeline.PipelineUmmaAsync
    tmem: utils.TmemAllocator
    epi_all: pipeline.NamedBarrier  # all epilogue threads, before TMEM is freed


class _MmaOperands(NamedTuple):
    """MMA warp operands: V (A operand) source, accumulators and U (B operand) fragments."""

    acc: cute.Tensor  # (MMA, 1, 1, NACC) N = 256 accumulators A, B
    acc_h: cute.Tensor  # (MMA, 1, 1, 2 * NACC) N = 128 views A_lo, A_hi, B_lo, B_hi
    v_ptr: cute.Pointer  # V stages
    a_layout: cute.ComposedLayout
    b: cute.Tensor  # (MMA, 1, 2, stage tiles) N = 256 U fragments
    b_h: cute.Tensor  # (MMA, 1, 2, 2 * stage tiles) N = 128 U fragments


class _EpiCopies(NamedTuple):
    """Epilogue TMEM <-> register copies and register buffers of one thread."""

    t2r: cute.TiledCopy  # butterfly loads, EPI_COLS columns of A and B
    r2t: cute.TiledCopy  # butterfly stores
    tTR: cute.Tensor
    tRT: cute.Tensor
    rAB: cute.Tensor  # butterfly buffer: A and B column blocks
    d32: cute.TiledCopy  # drain loads, 2 * DR_COLS columns
    tD32: cute.Tensor
    rD32: cute.Tensor
    rY: list  # converted outputs: rY[2 * c + acc] = drain load c of accumulator acc
    lay16: cute.Layout  # register layout of one DR_COLS store step


class _EpiLane(NamedTuple):
    """Position of one epilogue thread in the output block."""

    warp: Int32  # warp index = SMEM store box index
    cout_half: Int32  # Cout half (128 columns of each accumulator)
    row_pair: Int32  # lane quadrant = 2-row group
    lane: Int32
    row: Int32  # row within the warp's 2-row box
    col: Int32  # Winograd tile within the row


class Wino23Conv:
    """The fused conv for one shape: ``n_img`` channels-last ``h x w`` images, ``cin`` -> ``cout``
    channels (see :func:`supports`). The grid is persistent: ``nslots`` (at most the number of
    work units) slots of 4 CTAs each walk the units ``slot, slot + nslots, ...``, one unit
    being 4 CTA blocks of 8 rows x 32 px. ``in_half`` / ``out_half``: the input / output
    element type is fp16 instead of bf16. ``pdl``: launch with programmatic dependent launch
    (the prologue overlaps the previous kernel's tail; every role that touches global memory
    waits for the previous grid); without it the kernel has no grid-dependency waits."""

    def __init__(self, n_img, h, w, cin, cout, nslots, in_half=False, out_half=False, pdl=True):
        assert supports(cin, cout, h, w)
        self.pdl = pdl
        # trace-time constants: the input (bf16/fp16) and output element types
        self.in_dtype = cutlass.Float16 if in_half else cutlass.BFloat16
        self.out_dtype = cutlass.Float16 if out_half else cutlass.BFloat16
        self.nkc = cin // KC
        self.hb = h // RC
        self.wb = w // (2 * WT)
        self.nblk = n_img * self.hb * self.wb
        self.nunit = (self.nblk + SLOT_CTAS - 1) // SLOT_CTAS
        self.nslots = min(nslots, self.nunit)

        # warps: [0-7] epilogue (4 lane quadrants x 2 column halves), [8-11] transform,
        # 12 MMA, 13 U-TMA, 14 x-TMA, 15 spare
        self.num_epi_warps = 8
        self.tr_warp0 = 8
        self.num_tr_warps = 4
        self.mma_warp = 12
        self.utma_warp = 13
        self.xtma_warp = 14
        self.spare_warp = 15
        self.threads = 32 * (self.spare_warp + 1)
        # input-transform items (one 8-channel group of one tile at one input row) per thread
        self.tr_items_per_thread = VROWS * (KC // CHANNELS_PER_VECTOR) // (32 * self.num_tr_warps)

    # ------------------------------------------------------------------ host side

    @staticmethod
    def _make_mma(n):
        return sm100_utils.make_trivial_tiled_mma(
            cutlass.Float16,
            cutlass.Float16,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            cutlass.Float32,
            tcgen05.CtaGroup.TWO,
            (2 * MT, n),
        )

    @staticmethod
    def _make_u_tmas(tiled_mma, n, mU, num_tiles, cluster_vmnk_4cta, cluster_vmnk_2cta):
        """SMEM layout of ``num_tiles`` U tiles of ``tiled_mma`` (N = ``n``), the bytes of one
        tile per CTA, and the U TMA loads (atom, tensor) for 4-CTA and 2-CTA clusters."""
        mma_tiler = (2 * MT, n, KC)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, cutlass.Float16, num_tiles)
        b_tile = cute.slice_(b_layout, (None, None, None, 0))
        tmas = [
            cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils.cluster_shape_to_tma_atom_B((ctas, 1), tiled_mma.thr_id),
                mU,
                b_tile,
                mma_tiler,
                tiled_mma,
                cluster.shape,
            )
            for ctas, cluster in ((4, cluster_vmnk_4cta), (2, cluster_vmnk_2cta))
        ]
        return b_layout, cute.size_in_bytes(cutlass.Float16, b_tile), tmas[0], tmas[1]

    @staticmethod
    def _make_x_tma(mX):
        """Raw input TMA load: box (1 img, BOXH rows, BOXW px, KC ch)."""
        raw_layout = cute.make_layout(
            (1, BOXH, BOXW, KC), stride=(BOXH * BOXW * KC, BOXW * KC, KC, 1)
        )
        atom, tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX, raw_layout, (1, BOXH, BOXW, KC)
        )
        return raw_layout, atom, tensor

    @staticmethod
    def _make_y_tma(mY):
        """Output TMA store: per-warp box (1 img, 2 rows, 32 px, 16 Cout)."""
        epi_layout = cute.make_layout(
            (1, 2, 2 * WT, DR_COLS), stride=(2 * 2 * WT * DR_COLS, 2 * WT * DR_COLS, DR_COLS, 1)
        )
        atom, tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mY, epi_layout, (1, 2, 2 * WT, DR_COLS)
        )
        return epi_layout, atom, tensor

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mU: cute.Tensor,
        mY: cute.Tensor,
        stream: cuda.CUstream,
    ):
        tiled_mma = self._make_mma(NCO)
        # N = 128 MMA for the two chunks around the mid-block butterfly, so the butterfly of
        # one Cout half overlaps the MMAs of the other. Its U tiles (64 Cout per CTA) alias the
        # same SMEM stages; D column j still holds Cout (half * 128 + j).
        tiled_mma_h = self._make_mma(NCO // 2)
        cluster_vmnk_4cta = cute.tiled_divide(
            cute.make_layout((4, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        cluster_vmnk_2cta = cute.tiled_divide(
            cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, (2 * MT, NCO, KC), cutlass.Float16, 1)
        b_layout, u_tile_bytes, u_4cta, u_2cta = self._make_u_tmas(
            tiled_mma, NCO, mU, NUM_U_STAGE * U_GROUP, cluster_vmnk_4cta, cluster_vmnk_2cta
        )
        b_layout_h, _, u_half_4cta, u_half_2cta = self._make_u_tmas(
            tiled_mma_h,
            NCO // 2,
            mU,
            NUM_U_STAGE * U_GROUP * 2,
            cluster_vmnk_4cta,
            cluster_vmnk_2cta,
        )
        # TMA bytes per U tile: the pair's leader CTA barrier counts the loads of both CTAs
        u_bytes = u_tile_bytes * 2
        raw_layout, tma_atom_x, tma_tensor_x = self._make_x_tma(mX)
        epi_layout, tma_atom_y, tma_tensor_y = self._make_y_tma(mY)

        # Grid = (4, 1, nslots). Each slot of 4 CTAs is either one 4-CTA cluster (2 pairs that
        # share U via TMA multicast) or two 2-CTA fallback clusters (2 plain pairs).
        self.kernel(
            tiled_mma,
            tiled_mma_h,
            u_4cta[0],
            u_4cta[1],
            u_2cta[0],
            u_2cta[1],
            u_half_4cta[0],
            u_half_4cta[1],
            u_half_2cta[0],
            u_half_2cta[1],
            tma_atom_x,
            tma_tensor_x,
            tma_atom_y,
            tma_tensor_y,
            cluster_vmnk_4cta,
            cluster_vmnk_2cta,
            a_layout,
            b_layout,
            b_layout_h,
            raw_layout,
            epi_layout,
            u_bytes,
        ).launch(
            grid=(SLOT_CTAS, 1, self.nslots),
            block=(self.threads, 1, 1),
            cluster=(4, 1, 1),
            fallback_cluster=(2, 1, 1),
            smem_merge_branch_allocs=True,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
            stream=stream,
        )

    # ------------------------------------------------------------------ device side

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_h: cute.TiledMma,
        tma_atom_u_4cta: cute.CopyAtom,
        mU_4cta: cute.Tensor,
        tma_atom_u_2cta: cute.CopyAtom,
        mU_2cta: cute.Tensor,
        tma_atom_u_half_4cta: cute.CopyAtom,
        mU_half_4cta: cute.Tensor,
        tma_atom_u_half_2cta: cute.CopyAtom,
        mU_half_2cta: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        tma_atom_y: cute.CopyAtom,
        mY: cute.Tensor,
        cluster_vmnk_4cta: cute.Layout,
        cluster_vmnk_2cta: cute.Layout,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        b_layout_h: cute.ComposedLayout,
        raw_layout: cute.Layout,
        epi_layout: cute.Layout,
        u_bytes: cutlass.Constexpr,
    ):
        cluster_x, _, _ = cute.arch.block_in_cluster_dim()
        if cluster_x == 4:
            self._cta_main(
                tiled_mma,
                tiled_mma_h,
                (tma_atom_u_4cta, mU_4cta),
                (tma_atom_u_half_4cta, mU_half_4cta),
                (tma_atom_x, mX),
                (tma_atom_y, mY),
                cluster_vmnk_4cta,
                (a_layout, b_layout, b_layout_h, raw_layout, epi_layout),
                u_bytes,
                2,
            )
        else:
            self._cta_main(
                tiled_mma,
                tiled_mma_h,
                (tma_atom_u_2cta, mU_2cta),
                (tma_atom_u_half_2cta, mU_half_2cta),
                (tma_atom_x, mX),
                (tma_atom_y, mY),
                cluster_vmnk_2cta,
                (a_layout, b_layout, b_layout_h, raw_layout, epi_layout),
                u_bytes,
                1,
            )

    @cute.jit
    def _cta_main(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_h: cute.TiledMma,
        u_tma: tuple,
        u_half_tma: tuple,
        x_tma: tuple,
        y_tma: tuple,
        cluster_layout_vmnk: cute.Layout,
        layouts: tuple,
        u_bytes: cutlass.Constexpr,
        pairs_per_cluster: cutlass.Constexpr,
    ):
        """One CTA: setup, then one warp role per warp. ``*_tma`` are (atom, tensor) pairs;
        ``pairs_per_cluster`` CTA pairs share U through TMA multicast."""
        tma_atom_u, mU = u_tma
        tma_atom_u_half, mU_half = u_half_tma
        tma_atom_x, mX = x_tma
        tma_atom_y, mY = y_tma
        a_layout, b_layout, b_layout_h, raw_layout, epi_layout = layouts

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
        cta_v = bidx % 2
        is_leader = cta_v == 0
        slot, cta_in_slot = bidz, bidx

        self._prefetch_descriptors(warp_idx, tma_atom_u, tma_atom_u_half, tma_atom_x, tma_atom_y)
        smem = utils.SmemAllocator()
        storage = smem.allocate(_Barriers)
        sync = self._make_pipelines(storage, cluster_layout_vmnk, u_bytes)
        pipeline_init_arrive(cluster_shape_mn=(2 * pairs_per_cluster, 1), is_relaxed=True)

        sU, sU_h, sRaw, sV, sEpi = self._alloc_smem(
            smem, b_layout, b_layout_h, raw_layout, epi_layout
        )
        frags = self._mma_fragments(tiled_mma, tiled_mma_h, sU, sU_h)
        u_loads, u_half_loads, u_mcast_mask = self._partition_u(
            tiled_mma,
            tiled_mma_h,
            cta_v,
            cluster_layout_vmnk,
            (tma_atom_u, mU, sU),
            (tma_atom_u_half, mU_half, sU_h),
        )
        pipeline_init_wait(cluster_shape_mn=(2 * pairs_per_cluster, 1))

        if warp_idx == self.spare_warp:
            cute.arch.setmaxregister_decrease(56)
        rot = slot % self.nkc
        if warp_idx == self.utma_warp:
            self._u_producer(sync.u, u_loads, u_half_loads, u_mcast_mask, slot, rot)
        if warp_idx == self.xtma_warp:
            self._x_producer(sync.x, tma_atom_x, mX, sRaw, slot, cta_in_slot, rot)
        if warp_idx >= self.tr_warp0 and warp_idx < self.tr_warp0 + self.num_tr_warps:
            self._transform_warps(sync.x, sync.v, sRaw, sV, tidx, slot)
        if warp_idx == self.mma_warp:
            self._mma_warp(sync, tiled_mma, tiled_mma_h, frags, a_layout, sV, is_leader, slot)
        if warp_idx < self.num_epi_warps:
            self._epilogue(sync, frags[2], tma_atom_y, mY, sEpi, warp_idx, tidx, slot, cta_in_slot)

    # ---------------------------------------------------------------- CTA setup

    @cute.jit
    def _prefetch_descriptors(self, warp_idx, tma_atom_u, tma_atom_u_half, tma_atom_x, tma_atom_y):
        """Each TMA descriptor is prefetched by the warp that issues its copies."""
        if warp_idx == self.utma_warp:
            cpasync.prefetch_descriptor(tma_atom_u)
            cpasync.prefetch_descriptor(tma_atom_u_half)
        if warp_idx == self.xtma_warp:
            cpasync.prefetch_descriptor(tma_atom_x)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_y)

    @staticmethod
    def _mma_fragments(tiled_mma, tiled_mma_h, sU, sU_h):
        """U (B operand) fragments of the N = 256 and N = 128 MMAs, and the accumulator
        fragments: A, B (N = 256) and A_lo, A_hi, B_lo, B_hi (N = 128)."""
        tCrB = tiled_mma.make_fragment_B(sU)  # (MMA, 1, 2, STAGE)
        tCrBh = tiled_mma_h.make_fragment_B(sU_h)  # (MMA, 1, 2, 2 * STAGE)
        acc_shape = tiled_mma.partition_shape_C((2 * MT, NCO))
        acc_layout_frag = tiled_mma.make_fragment_C(cute.append(acc_shape, NACC))
        acc_shape_h = tiled_mma_h.partition_shape_C((2 * MT, NCO // 2))
        acc_h_layout_frag = tiled_mma_h.make_fragment_C(cute.append(acc_shape_h, 2 * NACC))
        return tCrB, tCrBh, acc_layout_frag, acc_h_layout_frag

    def _make_pipelines(self, storage, cluster_layout_vmnk, u_bytes):
        u_pipe, v_pipe, x_pipe = self._make_operand_pipelines(storage, cluster_layout_vmnk, u_bytes)
        pass0_lo_pipe, pass0_hi_pipe, acc_pipe, acc_b_free_pipe, acc_a_final_pipe = (
            self._make_acc_pipelines(storage, cluster_layout_vmnk)
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * (self.num_epi_warps + 1)
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        epi_all_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * self.num_epi_warps)
        return _Sync(
            u=u_pipe,
            v=v_pipe,
            x=x_pipe,
            pass0_lo=pass0_lo_pipe,
            pass0_hi=pass0_hi_pipe,
            acc=acc_pipe,
            acc_b_free=acc_b_free_pipe,
            acc_a_final=acc_a_final_pipe,
            tmem=tmem,
            epi_all=epi_all_barrier,
        )

    def _make_operand_pipelines(self, storage, cluster_layout_vmnk, u_bytes):
        """U, V and raw x pipelines."""
        u_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.u_mbar.data_ptr(),
            num_stages=NUM_U_STAGE,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=u_bytes * U_GROUP,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        )
        v_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.v_mbar.data_ptr(),
            num_stages=NUM_V_STAGE,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.num_tr_warps),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        x_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_mbar.data_ptr(),
            num_stages=NUM_RAW_STAGE,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_tr_warps),
            tx_count=RAW_BYTES,
            defer_sync=True,
        )
        return u_pipe, v_pipe, x_pipe

    def _make_acc_pipelines(self, storage, cluster_layout_vmnk):
        """MMA <-> epilogue accumulator pipelines: pass-0 lo / hi, acc, acc_b_free, acc_a_final."""
        pass0_lo_pipe = self._umma_to_epilogue(
            storage.pass0_lo_mbar, cluster_layout_vmnk, self.num_epi_warps
        )
        pass0_hi_pipe = self._umma_to_epilogue(
            storage.pass0_hi_mbar, cluster_layout_vmnk, self.num_epi_warps
        )
        acc_pipe = self._umma_to_epilogue(
            storage.acc_mbar, cluster_layout_vmnk, 2 * self.num_epi_warps
        )
        acc_b_free_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.acc_b_free_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.num_epi_warps),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_a_final_pipe = self._umma_to_epilogue(
            storage.acc_a_final_mbar, cluster_layout_vmnk, 2 * self.num_epi_warps
        )
        return pass0_lo_pipe, pass0_hi_pipe, acc_pipe, acc_b_free_pipe, acc_a_final_pipe

    @staticmethod
    def _umma_to_epilogue(mbar, cluster_layout_vmnk, consumer_count):
        # one-stage MMA -> epilogue signal, released by `consumer_count` epilogue arrivals
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_count),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    def _alloc_smem(self, smem, b_layout, b_layout_h, raw_layout, epi_layout):
        """SMEM buffers: U stages (N = 256 and N = 128 views), raw x stages, V stages and the
        per-warp output boxes."""
        sUraw = smem.allocate_tensor(
            element_type=cutlass.Float16,
            layout=cute.make_layout(NUM_U_STAGE * U_GROUP * (NCO // 2) * KC),
            byte_alignment=1024,
        )
        sU = cute.make_tensor(
            cute.recast_ptr(sUraw.iterator, b_layout.inner, dtype=cutlass.Float16), b_layout.outer
        )
        # half-N view of the same U stages: slot 4 * stage + 2 * g + h = Cout half h of tile g
        sU_h = cute.make_tensor(
            cute.recast_ptr(sUraw.iterator, b_layout_h.inner, dtype=cutlass.Float16),
            b_layout_h.outer,
        )
        sRaw = smem.allocate_tensor(
            element_type=self.in_dtype,
            layout=cute.append(
                raw_layout, cute.make_layout(NUM_RAW_STAGE, stride=BOXH * BOXW * KC)
            ),
            byte_alignment=1024,
        )
        sV = smem.allocate_tensor(
            element_type=cutlass.Float16,
            layout=cute.make_layout(NUM_V_STAGE * VSTAGE_BYTES // 2),
            byte_alignment=1024,
        )
        # epilogue staging: one (1, 2, 32px, 16co) box per epilogue warp
        sEpi = smem.allocate_tensor(
            element_type=self.out_dtype,
            layout=cute.append(
                epi_layout, cute.make_layout(self.num_epi_warps, stride=EPI_BYTES // 2)
            ),
            byte_alignment=1024,
        )
        return sU, sU_h, sRaw, sV, sEpi

    @staticmethod
    def _partition_u(tiled_mma, tiled_mma_h, cta_v, cluster_layout_vmnk, u_full, u_half):
        """TMA partitions of the N = 256 and N = 128 U loads as (atom, SMEM, GMEM) triples, and
        the multicast mask to the other pair of the cluster. ``u_full`` / ``u_half``: (atom,
        GMEM tensor, SMEM tensor)."""
        tma_atom_u, mU, sU = u_full
        tma_atom_uh, mUh, sU_h = u_half
        # U TMA partition: (N=Cout, K=Cin, L=12) tiled by (256, 32)
        gU = cute.local_tile(
            mU, cute.slice_((2 * MT, NCO, KC), (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(cta_v)
        tCgU = thr_mma.partition_B(gU)
        cta_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        blk_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_in_cluster)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tUsU, tUgU = cpasync.tma_partition(
            tma_atom_u,
            blk_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sU, 0, 3),
            cute.group_modes(tCgU, 0, 3),
        )
        u_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, blk_coord_vmnk, mcast_mode=1
        )
        gUh = cute.local_tile(
            mUh, cute.slice_((2 * MT, NCO // 2, KC), (0, None, None)), (None, None, None)
        )
        thr_mma_h = tiled_mma_h.get_slice(cta_v)
        tCgUh = thr_mma_h.partition_B(gUh)
        tUsUh, tUgUh = cpasync.tma_partition(
            tma_atom_uh,
            blk_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sU_h, 0, 3),
            cute.group_modes(tCgUh, 0, 3),
        )
        return (tma_atom_u, tUsU, tUgU), (tma_atom_uh, tUsUh, tUgUh), u_mcast_mask

    def _rotated_chunk(self, kc, rot):
        # Cin chunk loaded at step kc of a pass: the chunk order is rotated by `rot` per slot so
        # CTAs do not all fetch the same U/x lines at once.
        if isinstance(kc, int) and kc == 0:
            return rot % self.nkc
        return (kc + rot) % self.nkc

    def _split_block(self, blk):
        # (image, block within the image) of CTA block `blk`; blocks are row-major per image
        n = blk // (self.hb * self.wb)
        return n, blk - n * (self.hb * self.wb)

    # ---------------------------------------------------------------- U TMA producer

    @cute.jit
    def _u_producer(self, u_pipe, u_loads, u_half_loads, mcast_mask, slot, rot):
        cute.arch.setmaxregister_decrease(56)
        # Under programmatic dependent launch every role that touches global memory waits for
        # the previous grid first (the wait is per thread).
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_U_STAGE)
        unit = slot
        while unit < self.nunit:
            # pass 0: chunks 0 .. nkc-2 full width, chunk nkc-1 as two Cout halves
            for kc in cutlass.range(self.nkc - 1, unroll=1):
                state = self._load_u_chunk(u_pipe, state, u_loads, mcast_mask, kc, rot, 0)
            state = self._load_u_chunk_halves(
                u_pipe, state, u_half_loads, mcast_mask, self.nkc - 1, rot, 0
            )
            # pass 1: chunk 0 as two Cout halves, chunks 1 .. nkc-1 full width
            state = self._load_u_chunk_halves(
                u_pipe, state, u_half_loads, mcast_mask, 0, rot, NACC * NTAP
            )
            for kc in cutlass.range(1, self.nkc, unroll=1):
                state = self._load_u_chunk(u_pipe, state, u_loads, mcast_mask, kc, rot, NACC * NTAP)
            unit += self.nslots
        u_pipe.producer_tail(state)

    def _load_u_chunk(self, u_pipe, state, u_loads, mcast_mask, kc, rot, row0):
        """Load the NACC * NTAP N = 256 U tiles (U rows ``row0 ..``) of Cin chunk step ``kc``,
        U_GROUP tiles per stage. Returns ``state`` advanced past the stages."""
        atom, tUsU, tUgU = u_loads
        for g in range(NACC * NTAP // U_GROUP):
            u_pipe.producer_acquire(state)
            for gi in range(U_GROUP):
                cute.copy(
                    atom,
                    tUgU[(None, 0, self._rotated_chunk(kc, rot), row0 + g * U_GROUP + gi)],
                    tUsU[(None, state.index * U_GROUP + gi)],
                    tma_bar_ptr=u_pipe.producer_get_barrier(state),
                    mcast_mask=mcast_mask,
                )
            state.advance()
        return state

    def _load_u_chunk_halves(self, u_pipe, state, u_half_loads, mcast_mask, kc, rot, row0):
        """:meth:`_load_u_chunk` for the N = 128 MMA: each tile as its two Cout halves."""
        atom, tUsUh, tUgUh = u_half_loads
        for g in range(NACC * NTAP // U_GROUP):
            u_pipe.producer_acquire(state)
            for gi in range(U_GROUP):
                for hh in range(2):
                    cute.copy(
                        atom,
                        tUgUh[(None, hh, self._rotated_chunk(kc, rot), row0 + g * U_GROUP + gi)],
                        tUsUh[(None, state.index * 2 * U_GROUP + 2 * gi + hh)],
                        tma_bar_ptr=u_pipe.producer_get_barrier(state),
                        mcast_mask=mcast_mask,
                    )
            state.advance()
        return state

    # ---------------------------------------------------------------- raw x TMA producer

    def _partition_x(self, tma_atom_x, mX, sRaw, blk):
        """TMA partitions (SMEM, GMEM) of the raw x boxes of block ``blk``, one per Cin chunk:
        rows h0-1 .. h0+RC, px w0-1 .. w0+2*WT+1 (out-of-range rows and px are zero-filled)."""
        n, rem = self._split_block(blk)
        h0 = (rem // self.wb) * RC
        w0 = (rem % self.wb) * (2 * WT)
        gX = cute.local_tile(
            cute.domain_offset((n, h0 - 1, w0 - 1, 0), mX),
            (1, BOXH, BOXW, KC),
            (0, 0, 0, None),
        )
        return cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sRaw, 0, 4),
            cute.group_modes(gX, 0, 4),
        )

    @cute.jit
    def _x_producer(self, x_pipe, tma_atom_x, mX, sRaw, slot, cta_in_slot, rot):
        cute.arch.setmaxregister_decrease(56)
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # x is written by the previous grid
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_RAW_STAGE)
        unit = slot
        while unit < self.nunit:
            blk = unit * SLOT_CTAS + cta_in_slot
            # a CTA past the last block still runs with its pair; it reloads the last block and
            # the epilogue skips its stores
            if blk >= self.nblk:
                blk = self.nblk - 1
            tXsX, tXgX = self._partition_x(tma_atom_x, mX, sRaw, blk)
            for _ in cutlass.range_constexpr(NPASS):
                for kc in cutlass.range(self.nkc, unroll=1):
                    x_pipe.producer_acquire(state)
                    cute.copy(
                        tma_atom_x,
                        tXgX[(None, self._rotated_chunk(kc, rot))],
                        tXsX[(None, state.index)],
                        tma_bar_ptr=x_pipe.producer_get_barrier(state),
                    )
                    state.advance()
            unit += self.nslots
        x_pipe.producer_tail(state)

    # ---------------------------------------------------------------- transform warps

    def _transform_offsets(self, twarp, lane):
        """Byte offsets of this thread's input-transform items into a raw x stage and a V stage.

        An item is one 8-channel group of one Winograd tile at one input row. Lane bits 0-1
        pick the channel group, lane bit 2 the even or odd row of a row pair (odd rows take the
        next tile, mod WT); lane bits 3-4, the warp and the item index enumerate the (row pair,
        tile) slots. The V offset applies the UMMA 64-byte swizzle (SW64): the 16-byte group
        index XOR bits 1-2 of the V row."""
        group = lane & 3
        odd = (lane >> 2) & 1
        quad = lane >> 3
        src_off = []
        dst_off = []
        for p in range(self.tr_items_per_thread):
            item = (p * self.num_tr_warps + twarp) * 4 + quad
            pair = item // WT
            t0 = item % WT
            row = 2 * pair + odd
            tile = (t0 + odd) % WT
            v_row = row * WT + tile
            src_off.append((row * BOXW + 2 * tile) * (KC * 2) + group * 16)
            dst_off.append(v_row * (KC * 2) + ((group ^ ((v_row >> 1) & 3)) << 4))
        return src_off, dst_off

    @cute.jit
    def _transform_warps(self, x_pipe, v_pipe, sRaw, sV, tidx, slot):
        cute.arch.setmaxregister_decrease(96)
        in_half = self.in_dtype is cutlass.Float16
        tr_tid = tidx - 32 * self.tr_warp0
        src_off, dst_off = self._transform_offsets(tr_tid // 32, tr_tid % 32)
        raw_base = sRaw.iterator.toint()
        v_base = sV.iterator.toint()
        x_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_RAW_STAGE)
        v_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_V_STAGE)
        unit = slot
        while unit < self.nunit:
            for pass_idx in cutlass.range_constexpr(NPASS):
                for _ in cutlass.range(self.nkc, unroll=1):
                    x_pipe.consumer_wait(x_state)
                    v_pipe.producer_acquire(v_state)
                    raw_stage = raw_base + x_state.index * RAW_BYTES
                    v_stage = v_base + v_state.index * VSTAGE_BYTES
                    for p in cutlass.range_constexpr(self.tr_items_per_thread):
                        _wino_item(raw_stage + src_off[p], v_stage + dst_off[p], in_half, pass_idx)
                    _fence_proxy_async_smem()
                    cute.arch.sync_warp()
                    x_pipe.consumer_release(x_state)
                    with cute.arch.elect_one():
                        v_pipe.producer_commit(v_state)
                    x_state.advance()
                    v_state.advance()
            unit += self.nslots

    # ---------------------------------------------------------------- MMA warp

    @staticmethod
    def _v_operand(mma, ops, v_index, acc_idx, tap):
        # A operand of one tap: the V rows of the pass's coordinate `acc_idx` in V stage
        # `v_index`, starting at input row `tap` (WT V rows per input row)
        vptr = cute.recast_ptr(
            ops.v_ptr + (v_index * (VSTAGE_BYTES // 2) + (acc_idx * VROWS + tap * WT) * KC),
            ops.a_layout.inner,
            dtype=cutlass.Float16,
        )
        return mma.make_fragment_A(cute.make_tensor(vptr, ops.a_layout.outer))

    @staticmethod
    def _b_slot(u_state, ut, half):
        # SMEM slot of the U tile of tap `ut` (`half`: its Cout half, for the N = 128 MMA)
        if half is None:
            return u_state.index * U_GROUP + ut % U_GROUP
        return u_state.index * 2 * U_GROUP + 2 * (ut % U_GROUP) + half

    def _issue_tap(self, ops, mma, u_pipe, u_state, v_index, ut, half=None, zero_init=False):
        """Leader-CTA MMAs of tap ``ut`` (accumulator ut // NTAP, H tap ut % NTAP) over one Cin
        chunk, waiting for / releasing the U stage at the group boundaries. ``half``: Cout half
        of the N = 128 MMA, ``None`` for N = 256. ``zero_init``: tap 0 overwrites the
        accumulator. Returns ``mma`` (its accumulate field is runtime state: callers inside a
        dynamic ``if`` must rebind it)."""
        acc_idx, tap = ut // NTAP, ut % NTAP
        if ut % U_GROUP == 0 and half in (None, 0):
            u_pipe.consumer_wait(u_state)
        a_frag = self._v_operand(mma, ops, v_index, acc_idx, tap)
        acc, b_frag, acc_slot = (
            (ops.acc, ops.b, acc_idx) if half is None else (ops.acc_h, ops.b_h, 2 * acc_idx + half)
        )
        for kb in range(KC // 16):
            mma.set(tcgen05.Field.ACCUMULATE, not (zero_init and tap == 0 and kb == 0))
            cute.gemm(
                mma,
                acc[(None, None, None, acc_slot)],
                a_frag[(None, None, kb, 0)],
                b_frag[(None, None, kb, self._b_slot(u_state, ut, half))],
                acc[(None, None, None, acc_slot)],
            )
        if ut % U_GROUP == U_GROUP - 1 and half in (None, 1):
            u_pipe.consumer_release(u_state)
        return mma

    @cute.jit
    def _mma_acc_taps(
        self, tiled_mma, ops, u_pipe, u_state, v_index, acc_idx, zero_init, is_leader
    ):
        """The NTAP N = 256 taps of accumulator ``acc_idx`` over one Cin chunk."""
        for tap in cutlass.range_constexpr(NTAP):
            ut = acc_idx * NTAP + tap
            if is_leader:
                tiled_mma = self._issue_tap(
                    ops, tiled_mma, u_pipe, u_state, v_index, ut, zero_init=zero_init
                )
            if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                u_state.advance()
        return tiled_mma, u_state

    @cute.jit
    def _mma_first_chunk(self, tiled_mma, ops, sync, u_state, v_state, b_free_state, is_leader):
        """Pass 0, chunk 0: coord 1 into A, then coord 2 into B once the epilogue has drained B
        of the previous block. Both overwrite their accumulator."""
        v_pipe = sync.v
        acc_b_free_pipe = sync.acc_b_free
        if is_leader:
            v_pipe.consumer_wait(v_state)
        tiled_mma, u_state = self._mma_acc_taps(
            tiled_mma, ops, sync.u, u_state, v_state.index, 0, True, is_leader
        )
        if is_leader:
            acc_b_free_pipe.consumer_wait(b_free_state)
        tiled_mma, u_state = self._mma_acc_taps(
            tiled_mma, ops, sync.u, u_state, v_state.index, 1, True, is_leader
        )
        if is_leader:
            acc_b_free_pipe.consumer_release(b_free_state)
        b_free_state.advance()
        if is_leader:
            v_pipe.consumer_release(v_state)
        v_state.advance()
        return tiled_mma, u_state, v_state, b_free_state

    @cute.jit
    def _mma_middle_chunks(self, tiled_mma, ops, sync, u_state, v_state, is_leader):
        """Chunks 1 .. nkc-2 of a pass, N = 256."""
        v_pipe = sync.v
        for _ in cutlass.range(1, self.nkc - 1, unroll=1):
            if is_leader:
                v_pipe.consumer_wait(v_state)
            for acc_idx in cutlass.range_constexpr(NACC):
                tiled_mma, u_state = self._mma_acc_taps(
                    tiled_mma, ops, sync.u, u_state, v_state.index, acc_idx, False, is_leader
                )
            if is_leader:
                v_pipe.consumer_release(v_state)
            v_state.advance()
        return tiled_mma, u_state, v_state

    @cute.jit
    def _mma_cout_half(
        self, tiled_mma_h, ops, sync, u_state, v_index, lo_state, hi_state, is_leader, pass_idx, hh
    ):
        """Cout half ``hh`` of a split chunk (see :meth:`_mma_split_chunk`). ``u_state`` is not
        advanced: both halves read the same U stages."""
        pass0_lo_pipe = sync.pass0_lo
        pass0_hi_pipe = sync.pass0_hi
        if cutlass.const_expr(pass_idx == 1):
            if is_leader:
                if cutlass.const_expr(hh == 0):
                    pass0_lo_pipe.producer_acquire(lo_state)
                else:
                    pass0_hi_pipe.producer_acquire(hi_state)
        half_state = u_state.clone()
        for ut in cutlass.range_constexpr(NACC * NTAP):
            if is_leader:
                tiled_mma_h = self._issue_tap(
                    ops, tiled_mma_h, sync.u, half_state, v_index, ut, half=hh
                )
            if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                half_state.advance()
        if cutlass.const_expr(pass_idx == 0):
            if is_leader:
                if cutlass.const_expr(hh == 0):
                    pass0_lo_pipe.producer_commit(lo_state)
                else:
                    pass0_hi_pipe.producer_commit(hi_state)
        return tiled_mma_h

    @cute.jit
    def _mma_split_chunk(
        self, tiled_mma_h, ops, sync, u_state, v_state, lo_state, hi_state, is_leader, pass_idx
    ):
        """A Cin chunk as two N = 128 Cout halves (both read the same U stages). Pass 0 (last
        chunk): each half's pass-0 result is committed to its butterfly warps as soon as its
        MMAs are issued, lo first. Pass 1 (chunk 0): each half starts once its butterfly
        released it."""
        v_pipe = sync.v
        if is_leader:
            v_pipe.consumer_wait(v_state)
        for hh in cutlass.range_constexpr(2):
            tiled_mma_h = self._mma_cout_half(
                tiled_mma_h,
                ops,
                sync,
                u_state,
                v_state.index,
                lo_state,
                hi_state,
                is_leader,
                pass_idx,
                hh,
            )
        if cutlass.const_expr(pass_idx == 0):
            lo_state.advance()
            hi_state.advance()
        for _ in cutlass.range_constexpr(NACC * NTAP // U_GROUP):
            u_state.advance()
        if is_leader:
            v_pipe.consumer_release(v_state)
        v_state.advance()
        return tiled_mma_h, u_state, v_state, lo_state, hi_state

    @cute.jit
    def _mma_last_chunk(self, tiled_mma, ops, sync, u_state, v_state, a_final_state, is_leader):
        """Pass 1, last chunk: coord 0 into A, then A (= y0, final) is committed to the
        epilogue before coord 3 into B."""
        v_pipe = sync.v
        acc_a_final_pipe = sync.acc_a_final
        if is_leader:
            v_pipe.consumer_wait(v_state)
        tiled_mma, u_state = self._mma_acc_taps(
            tiled_mma, ops, sync.u, u_state, v_state.index, 0, False, is_leader
        )
        if is_leader:
            acc_a_final_pipe.producer_commit(a_final_state)
        a_final_state.advance()
        tiled_mma, u_state = self._mma_acc_taps(
            tiled_mma, ops, sync.u, u_state, v_state.index, 1, False, is_leader
        )
        if is_leader:
            v_pipe.consumer_release(v_state)
        v_state.advance()
        return tiled_mma, u_state, v_state, a_final_state

    @cute.jit
    def _mma_block(
        self,
        sync,
        tiled_mma,
        tiled_mma_h,
        ops,
        u_state,
        v_state,
        acc_state,
        b_free_state,
        a_final_state,
        lo_state,
        hi_state,
        is_leader,
    ):
        """All MMAs of one block: pass 0 with chunk 0 peeled and the last chunk split into Cout
        halves, pass 1 with chunk 0 split and the last chunk peeled, then the commit of the
        final accumulators. Returns both MMAs and every pipeline state, advanced."""
        acc_a_final_pipe = sync.acc_a_final
        acc_pipe = sync.acc
        # coord-1 MMAs into A only need y0 of the previous block drained (not the whole
        # previous pass 1 retired), so the tensor pipe never empties between blocks
        if is_leader:
            acc_a_final_pipe.producer_acquire(a_final_state)
        tiled_mma, u_state, v_state, b_free_state = self._mma_first_chunk(
            tiled_mma, ops, sync, u_state, v_state, b_free_state, is_leader
        )
        tiled_mma, u_state, v_state = self._mma_middle_chunks(
            tiled_mma, ops, sync, u_state, v_state, is_leader
        )
        for pass_idx in cutlass.range_constexpr(NPASS):
            tiled_mma_h, u_state, v_state, lo_state, hi_state = self._mma_split_chunk(
                tiled_mma_h, ops, sync, u_state, v_state, lo_state, hi_state, is_leader, pass_idx
            )
        tiled_mma, u_state, v_state = self._mma_middle_chunks(
            tiled_mma, ops, sync, u_state, v_state, is_leader
        )
        tiled_mma, u_state, v_state, a_final_state = self._mma_last_chunk(
            tiled_mma, ops, sync, u_state, v_state, a_final_state, is_leader
        )
        if is_leader:
            acc_pipe.producer_acquire(acc_state)
            acc_pipe.producer_commit(acc_state)
        acc_state.advance()
        return (
            tiled_mma,
            tiled_mma_h,
            u_state,
            v_state,
            acc_state,
            b_free_state,
            a_final_state,
            lo_state,
            hi_state,
        )

    @staticmethod
    def _mma_operands(sync, frags, a_layout, sV):
        """Wait for the TMEM allocation and gather the MMA operands."""
        tmem = sync.tmem
        tCrB, tCrBh, acc_layout_frag, acc_h_layout_frag = frags
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        return _MmaOperands(
            acc=cute.make_tensor(tmem_ptr, acc_layout_frag.layout),
            acc_h=cute.make_tensor(tmem_ptr, acc_h_layout_frag.layout),
            v_ptr=sV.iterator,
            a_layout=a_layout,
            b=tCrB,
            b_h=tCrBh,
        )

    @cute.jit
    def _mma_warp(self, sync, tiled_mma, tiled_mma_h, frags, a_layout, sV, is_leader, slot):
        """Issues the MMAs of every block of the slot (see :meth:`_mma_block`). Only the leader
        CTA of the pair issues MMAs; both CTAs walk the pipeline states. ``frags``: see
        :meth:`_mma_fragments`."""
        pass0_lo_pipe = sync.pass0_lo
        pass0_hi_pipe = sync.pass0_hi
        acc_a_final_pipe = sync.acc_a_final
        acc_pipe = sync.acc
        cute.arch.setmaxregister_decrease(56)
        ops = self._mma_operands(sync, frags, a_layout, sV)
        consumer, producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        u_state = pipeline.make_pipeline_state(consumer, NUM_U_STAGE)
        v_state = pipeline.make_pipeline_state(consumer, NUM_V_STAGE)
        acc_state = pipeline.make_pipeline_state(producer, 1)
        b_free_state = pipeline.make_pipeline_state(consumer, 1)
        a_final_state = pipeline.make_pipeline_state(producer, 1)
        lo_state = pipeline.make_pipeline_state(producer, 1)
        hi_state = pipeline.make_pipeline_state(producer, 1)
        if is_leader:
            pass0_lo_pipe.producer_acquire(lo_state)
            pass0_hi_pipe.producer_acquire(hi_state)
        unit = slot
        while unit < self.nunit:
            (
                tiled_mma,
                tiled_mma_h,
                u_state,
                v_state,
                acc_state,
                b_free_state,
                a_final_state,
                lo_state,
                hi_state,
            ) = self._mma_block(
                sync,
                tiled_mma,
                tiled_mma_h,
                ops,
                u_state,
                v_state,
                acc_state,
                b_free_state,
                a_final_state,
                lo_state,
                hi_state,
                is_leader,
            )
            unit += self.nslots
        if is_leader:
            acc_a_final_pipe.producer_tail(a_final_state)
            acc_pipe.producer_tail(acc_state)

    # ---------------------------------------------------------------- epilogue warps

    def _epi_tmem_copies(self, tmem_ptr, acc_layout_frag, epi_tid):
        tAcc = cute.make_tensor(tmem_ptr, acc_layout_frag.layout)
        # TMEM address: lane in the upper 16 bits, column in the lower 16
        tAcc_flat = cute.make_tensor(
            tAcc.iterator, cute.make_layout((MT, NCO, NACC), stride=(65536, 1, NCO))
        )
        tAcc_div = cute.flat_divide(tAcc_flat, (MT, EPI_COLS))  # (128,64,1,4,2)
        t2r_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
        r2t_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
        t2r = tcgen05.make_tmem_copy(t2r_atom, tAcc_div[(None, None, 0, 0, 0)])
        r2t = tcgen05.make_tmem_copy(r2t_atom, tAcc_div[(None, None, 0, 0, 0)])
        thr_t2r = t2r.get_slice(epi_tid)
        thr_r2t = r2t.get_slice(epi_tid)
        tTR = thr_t2r.partition_S(tAcc_div)
        tRT = thr_r2t.partition_D(tAcc_div)
        tTRc = thr_t2r.partition_D(cute.make_identity_tensor((MT, EPI_COLS)))
        # A and B column blocks loaded/stored by one copy each (one TMEM wait per block)
        rAB = cute.make_rmem_tensor(cute.append(tTRc.shape, NACC), cutlass.Float32)
        # register layout of one 16-Cout store step
        tAcc_dd = cute.flat_divide(tAcc_flat, (MT, DR_COLS))  # (128,16,1,16,2)
        d2r_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
        d2r = tcgen05.make_tmem_copy(d2r_atom, tAcc_dd[(None, None, 0, 0, 0)])
        thr_d2r = d2r.get_slice(epi_tid)
        tDRc = thr_d2r.partition_D(cute.make_identity_tensor((MT, DR_COLS)))
        lay16 = cute.make_rmem_tensor(tDRc.shape, self.out_dtype).layout
        # drain in 32-column TMEM loads: half the round trips of the 16-column store steps
        tAcc_d32 = cute.flat_divide(tAcc_flat, (MT, 2 * DR_COLS))  # (128,32,1,8,2)
        d32_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
        d32 = tcgen05.make_tmem_copy(d32_atom, tAcc_d32[(None, None, 0, 0, 0)])
        thr_d32 = d32.get_slice(epi_tid)
        tD32 = thr_d32.partition_S(tAcc_d32)
        tD32c = thr_d32.partition_D(cute.make_identity_tensor((MT, 2 * DR_COLS)))
        rD32 = cute.make_rmem_tensor(tD32c.shape, cutlass.Float32)
        rY = [cute.make_rmem_tensor(tD32c.shape, self.out_dtype) for _ in range(2 * DRAIN_LOADS)]
        return _EpiCopies(t2r, r2t, tTR, tRT, rAB, d32, tD32, rD32, rY, lay16)

    @staticmethod
    def _butterfly_half(ec, cout_half):
        # (A, B) <- (A + B, A - B) in TMEM on the 128 columns of this warp's Cout half
        steps = NCO // EPI_COLS // 2
        rAB = ec.rAB
        for cc in range(steps):
            ccg = cout_half * steps + cc
            cute.copy(ec.t2r, ec.tTR[(None, None, None, 0, ccg, None)], rAB)
            for i in range(EPI_COLS // 2):
                s0, s1, d0, d1 = _bfly2(
                    rAB[2 * i],
                    rAB[2 * i + 1],
                    rAB[EPI_COLS + 2 * i],
                    rAB[EPI_COLS + 2 * i + 1],
                )
                rAB[2 * i] = s0
                rAB[2 * i + 1] = s1
                rAB[EPI_COLS + 2 * i] = d0
                rAB[EPI_COLS + 2 * i + 1] = d1
            cute.copy(ec.r2t, rAB, ec.tRT[(None, None, None, 0, ccg, None)])

    def _drain_acc(self, ec, acc_idx, cout_half):
        # accumulator `acc_idx` (0: A = y0, 1: B = y1) of this warp's Cout half from TMEM,
        # converted to the output type into rY[2 * cc + acc_idx]
        for cc in range(DRAIN_LOADS):
            cute.copy(
                ec.d32,
                ec.tD32[(None, None, None, 0, cout_half * DRAIN_LOADS + cc, acc_idx)],
                ec.rD32,
            )
            ec.rY[2 * cc + acc_idx].store(ec.rD32.load().to(self.out_dtype))

    @staticmethod
    def _fill_store_box(sEpi, ec, lane, cc):
        # store step `cc`: the thread's 16 Cout of px 2 * col (y0) and 2 * col + 1 (y1) into
        # the warp's SMEM box
        c32 = cc // 2
        off = (cc % 2) * DR_COLS
        for px in range(2):
            sy = sEpi[(0, lane.row, 2 * lane.col + px, None, lane.warp)]
            cute.autovec_copy(
                cute.make_tensor(ec.rY[2 * c32 + px].iterator + off, ec.lay16),
                cute.make_tensor(sy.iterator, ec.lay16),
            )

    @cute.jit
    def _store_outputs(self, tma_atom_y, mY, sEpi, ec, lane, n, rem, blk):
        """Write this warp's 2 rows x 32 px x 128 Cout of block ``blk`` from registers through its
        SMEM box with TMA stores, one 16-Cout box at a time (skipped for padding blocks)."""
        gYt = cute.local_tile(
            mY,
            (1, 2, 2 * WT, DR_COLS),
            (n, (rem // self.wb) * (RC // 2) + lane.row_pair, rem % self.wb, None),
        )
        sEw = sEpi[(None, None, None, None, lane.warp)]
        tYs, tYg = cpasync.tma_partition(
            tma_atom_y,
            0,
            cute.make_layout(1),
            cute.group_modes(sEw, 0, 4),
            cute.group_modes(gYt, 0, 4),
        )
        for cc in cutlass.range_constexpr(STORE_STEPS):
            ccg = lane.cout_half * STORE_STEPS + cc
            if lane.lane == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            self._fill_store_box(sEpi, ec, lane, cc)
            _fence_proxy_async_smem()
            cute.arch.sync_warp()
            if lane.lane == 0:
                if blk < self.nblk:
                    cute.copy(tma_atom_y, tYs, tYg[(None, ccg)])
                cute.arch.cp_async_bulk_commit_group()

    @staticmethod
    def _epi_lane(warp_idx, tidx):
        """This epilogue thread's :class:`_EpiLane` and its thread index within its lane
        quadrant group (0 .. 127)."""
        epi_tid = tidx % 128
        lane_e = tidx % 32
        lane = _EpiLane(
            warp=warp_idx,
            cout_half=warp_idx // 4,
            row_pair=warp_idx % 4,
            lane=lane_e,
            row=lane_e // WT,
            col=epi_tid % WT,
        )
        return lane, epi_tid

    @cute.jit
    def _epi_butterfly(self, sync, ec, cout_half, lo_state, hi_state):
        """Butterfly of this warp's Cout half as soon as pass 0 has finished on that half, then
        the release that lets pass 1 start on it."""
        pass0_lo_pipe = sync.pass0_lo
        pass0_hi_pipe = sync.pass0_hi
        if cout_half == 0:
            pass0_lo_pipe.consumer_wait(lo_state)
        else:
            pass0_hi_pipe.consumer_wait(hi_state)
        self._butterfly_half(ec, cout_half)
        cute.arch.fence_view_async_tmem_store()
        with cute.arch.elect_one():
            if cout_half == 0:
                pass0_lo_pipe.consumer_release(lo_state)
            else:
                pass0_hi_pipe.consumer_release(hi_state)
        lo_state.advance()
        hi_state.advance()
        return lo_state, hi_state

    @cute.jit
    def _epi_drain(self, sync, ec, cout_half, acc_state, a_final_state, b_free_state):
        """Drain y0 = A as soon as the last coord-0 MMAs are done (the tensor core is still on
        coord 3), then y1 = B after pass 1; TMEM is then free for the next block's MMAs."""
        acc_a_final_pipe = sync.acc_a_final
        acc_pipe = sync.acc
        acc_b_free_pipe = sync.acc_b_free
        acc_a_final_pipe.consumer_wait(a_final_state)
        self._drain_acc(ec, 0, cout_half)
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            acc_a_final_pipe.consumer_release(a_final_state)
        a_final_state.advance()
        acc_pipe.consumer_wait(acc_state)
        with cute.arch.elect_one():
            acc_pipe.consumer_release(acc_state)
        acc_state.advance()
        acc_b_free_pipe.producer_acquire(b_free_state)
        self._drain_acc(ec, 1, cout_half)
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            acc_b_free_pipe.producer_commit(b_free_state)
        b_free_state.advance()
        return acc_state, a_final_state, b_free_state

    @cute.jit
    def _epilogue_block(
        self,
        sync,
        ec,
        lane,
        y_out,
        acc_state,
        a_final_state,
        lo_state,
        hi_state,
        b_free_state,
        blk,
    ):
        """Butterfly, drain and TMA stores of block ``blk``. ``y_out``: (TMA atom, output
        tensor, SMEM store boxes). Returns every pipeline state, advanced."""
        tma_atom_y, mY, sEpi = y_out
        n, rem = self._split_block(blk)
        lo_state, hi_state = self._epi_butterfly(sync, ec, lane.cout_half, lo_state, hi_state)
        acc_state, a_final_state, b_free_state = self._epi_drain(
            sync, ec, lane.cout_half, acc_state, a_final_state, b_free_state
        )
        self._store_outputs(tma_atom_y, mY, sEpi, ec, lane, n, rem, blk)
        return acc_state, a_final_state, lo_state, hi_state, b_free_state

    @cute.jit
    def _epilogue(
        self, sync, acc_layout_frag, tma_atom_y, mY, sEpi, warp_idx, tidx, slot, cta_in_slot
    ):
        """8 warps: 4 lane quadrants x 2 Cout halves. Per block: butterfly after pass 0, drain
        of y0 and y1, then the TMA stores (overlapping the next block's MMAs)."""
        acc_b_free_pipe = sync.acc_b_free
        epi_all = sync.epi_all
        tmem = sync.tmem
        cute.arch.setmaxregister_increase(176)
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        lane, epi_tid = self._epi_lane(warp_idx, tidx)
        ec = self._epi_tmem_copies(tmem_ptr, acc_layout_frag, epi_tid)

        consumer, producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        acc_state = pipeline.make_pipeline_state(consumer, 1)
        a_final_state = pipeline.make_pipeline_state(consumer, 1)
        lo_state = pipeline.make_pipeline_state(consumer, 1)
        hi_state = pipeline.make_pipeline_state(consumer, 1)
        b_free_state = pipeline.make_pipeline_state(producer, 1)
        # accumulator B starts out free
        acc_b_free_pipe.producer_acquire(b_free_state)
        with cute.arch.elect_one():
            acc_b_free_pipe.producer_commit(b_free_state)
        b_free_state.advance()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # y may still be read by the previous grid
        unit = slot
        while unit < self.nunit:
            acc_state, a_final_state, lo_state, hi_state, b_free_state = self._epilogue_block(
                sync,
                ec,
                lane,
                (tma_atom_y, mY, sEpi),
                acc_state,
                a_final_state,
                lo_state,
                hi_state,
                b_free_state,
                unit * SLOT_CTAS + cta_in_slot,
            )
            unit += self.nslots
        if lane.lane == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        epi_all.arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


# ---------------------------------------------------------------------- filter transform


@cute.kernel
def _wt_kernel(mW: cute.Tensor, mU: cute.Tensor):
    # mW: (Cout, Cin*9) bf16 with (c, r, s) innermost; mU: (12, Cin, Cout) fp16 with row
    # l = pass*6 + jj*3 + r (pass 0: coords 1, 2; pass 1: coords 0, 3), Cout contiguous.
    # One block = 16 Cout x 16 Cin staged via SMEM so weight reads and U writes are coalesced.
    cute.arch.griddepcontrol_launch_dependents()
    bx, by, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    k0 = bx * FILTER_TILE
    c0 = by * FILTER_TILE
    smem = utils.SmemAllocator()
    sW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((FILTER_TILE, FILTER_TILE * 9), stride=(FILTER_TILE * 9 + 1, 1)),
        byte_alignment=16,
    )
    for it in cutlass.range_constexpr(FILTER_TILE * FILTER_TILE * 9 // FILTER_THREADS):
        idx = tidx + FILTER_THREADS * it
        kk = idx // (FILTER_TILE * 9)
        q = idx % (FILTER_TILE * 9)
        sW[kk, q] = mW[k0 + kk, c0 * 9 + q].to(cutlass.Float32)
    cute.arch.barrier()
    kk = tidx % FILTER_TILE
    cc = tidx // FILTER_TILE
    for r in cutlass.range_constexpr(NTAP):
        g0 = sW[kk, cc * 9 + 3 * r]
        g1 = sW[kk, cc * 9 + 3 * r + 1]
        g2 = sW[kk, cc * 9 + 3 * r + 2]
        mU[r, c0 + cc, k0 + kk] = ((g0 + g1 + g2) * 0.5).to(cutlass.Float16)
        mU[3 + r, c0 + cc, k0 + kk] = ((g0 - g1 + g2) * 0.5).to(cutlass.Float16)
        mU[6 + r, c0 + cc, k0 + kk] = g0.to(cutlass.Float16)
        mU[9 + r, c0 + cc, k0 + kk] = g2.to(cutlass.Float16)


@cute.jit
def _wt_launch(
    mW: cute.Tensor,
    mU: cute.Tensor,
    cout: cutlass.Constexpr,
    cin: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    # dynamic SMEM is sized by the kernel's SmemAllocator
    _wt_kernel(mW, mU).launch(
        grid=(cout // FILTER_TILE, cin // FILTER_TILE, 1),
        block=(FILTER_THREADS, 1, 1),
        stream=stream,
    )


_compiled_fns = {}


def _compiled(key, compile_fn):
    # compile_fn() on the first call with `key`, the cached function afterwards. The kernels
    # are compiled from the real tensors (static shapes and strides), so `key` names everything
    # that varies between calls: kernel, device, shapes and dtypes.
    fn = _compiled_fns.get(key)
    if fn is None:
        fn = _compiled_fns[key] = compile_fn()
    return fn


def _weight_transform(weight):
    cout, cin = weight.shape[0], weight.shape[1]
    u = torch.empty((NPASS * NACC * NTAP, cin, cout), dtype=torch.float16, device=weight.device)
    w2 = weight.reshape(cout, cin * 9)
    fn = _compiled(
        ("filter", weight.device, cout, cin, weight.dtype),
        lambda: cute.compile(
            _wt_launch,
            from_dlpack(w2, enable_tvm_ffi=True),
            from_dlpack(u, enable_tvm_ffi=True),
            cout,
            cin,
            make_fake_stream(use_tvm_ffi_env_stream=True),  # the caller's torch stream
            options="--enable-tvm-ffi",
        ),
    )
    fn(w2, u)
    return u


@torch.no_grad()
def prepare(weight):
    """The transformed filter for :func:`run_prepared`. ``weight``: ``[Cout, Cin, 3, 3]`` bf16,
    contiguous. Returns a ``[Cout, Cin, 12]`` fp16 view (Cout contiguous); index ``pass * 6 +
    acc * 3 + tap`` of the last dim holds coords 1, 2 (pass 0) and 0, 3 (pass 1). Launches on
    the current torch stream."""
    u = _weight_transform(weight)  # (12, Cin, Cout) fp16
    return u.permute(2, 1, 0)


def run_prepared(x, ut, y, *, pdl):
    """``y = conv3x3(x, weight)`` (padding 1) with ``ut`` from :func:`prepare`; x, y
    channels-last ``[N, H, W, C]`` (bf16 or fp16 x, y in its own dtype), shapes as accepted
    by :func:`supports`; launches on the current torch stream, with programmatic dependent
    launch when ``pdl`` (a compile-time choice: one compiled kernel per value)."""
    n_img, h, w, cin = x.shape
    cout = ut.shape[0]
    assert supports(cin, cout, h, w)

    def compile_conv():
        # one slot per two co-resident 2-CTA clusters
        nslots = utils.HardwareInfo().get_max_active_clusters(2) // 2
        return cute.compile(
            Wino23Conv(
                n_img,
                h,
                w,
                cin,
                cout,
                nslots,
                x.dtype == torch.float16,
                y.dtype == torch.float16,
                pdl,
            ),
            from_dlpack(x, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(ut, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(y, assumed_align=16, enable_tvm_ffi=True),
            make_fake_stream(use_tvm_ffi_env_stream=True),  # the caller's torch stream
            options="--enable-tvm-ffi",
        )

    key = ("conv", x.device, n_img, h, w, cin, cout, x.dtype, y.dtype, pdl)
    _compiled(key, compile_conv)(x, ut, y)
