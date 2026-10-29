# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fused 1D-Winograd F(2,3) (along W) x direct (along H) 3x3 conv, channels-last bf16 or fp16
# input; the output element type follows the destination; launches on the caller's stream.
#
# Per output tile of 2 px along W, the 4 Winograd coordinates j give 4 GEMMs
#   M_j[tile, cout] = sum_{r in 0..2, c} V_j[input row h+r-1, tile, c] * U_j[r][cout, c]
# with V = B^T d (input transform along W, fp16) and U = G g (filter transform, fp16), and
#   y[2t] = M0 + M1 + M2,   y[2t+1] = M1 - M2 - M3.
#
# The output transform is folded into the accumulators so that only TWO accumulators of the
# full Cout = 256 width are needed (2 x 256 = all 512 TMEM columns), which lets every MMA run at
# the most SMEM-efficient shape (cta_group::2, 256 x 256 x 16):
#   pass 0 (coords 1, 2, all Cin):  A = M1, B = M2
#   butterfly in TMEM (epilogue warps): A <- A + B, B <- A - B
#   pass 1 (coords 0, 3, all Cin):  A += M0 = y[2t],  B += U3 (d3 - d1) = -M3 -> y[2t+1]
#
# One CTA pair owns 2 x 128 tiles (each CTA: 8 rows x 32 px) and all 256 Cout.  Raw bf16
# input boxes are TMA-loaded into SMEM, transformed by 4 transform warps into V (fp16, UMMA
# K-major SW64) in SMEM (A operand; the H taps are 16-row offsets into V).  U is TMA-loaded
# as the 2-CTA split B operand, multicast between the two pairs of a 4-CTA cluster.  V and M
# never leave the SM; the epilogue writes the final bf16 outputs straight from registers.

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import Int32, T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

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
RAW_BYTES = BOXH * BOXW * KC * 2  # 22400
VCOORD_BYTES = VROWS * KC * 2  # 10240
VSTAGE_BYTES = NACC * VCOORD_BYTES  # 20480

NUM_U_STAGE = 6
U_GROUP = 2  # (coord, tap) U tiles per pipeline stage
NUM_V_STAGE = 3
NUM_RAW_STAGE = 2
EPI_COLS = 64  # TMEM columns per butterfly step
DR_COLS = 16  # TMEM columns per drain step (= Cout per TMA store box)
EPI_BYTES = 2 * 2 * WT * DR_COLS * 2  # 2 KB: per-warp store box (2 rows x 32 px x 16 Cout)


def _cvt4(regs_in, regs_out, in_half=False):
    # Converts packed input pairs to f16x2: bf16 pairs through fp32, fp16 pairs as they are.
    s = ""
    for pi, qi in zip(regs_in, regs_out):
        if in_half:
            s += f"mov.b32 q{qi}, p{pi};\n\t"
        else:
            s += (
                f"shl.b32 lo, p{pi}, 16;\n\tand.b32 hi, p{pi}, 0xffff0000;\n\t"
                f"cvt.rn.f16x2.f32 q{qi}, hi, lo;\n\t"
            )
    return s


@dsl_user_op
def _wino_item_p0(src: Int32, dst: Int32, in_half: bool, *, loc=None, ip=None):
    # pass 0: V1 = d1 + d2, V2 = d2 - d1.  src: px d0 of 4 consecutive px (64 B apart),
    # 8 bf16/fp16 channels each; dst: coord slot 0, slot 1 at +VCOORD_BYTES.
    cvt = _cvt4(range(4, 12), range(4, 12), in_half)
    ar = ""
    for i in range(4):
        ar += f"add.f16x2 v{i}, q{4 + i}, q{8 + i};\n\t"
        ar += f"sub.f16x2 w{i}, q{8 + i}, q{4 + i};\n\t"
    cs = VCOORD_BYTES
    asm = (
        "{\n\t.reg .b32 p<16>, q<16>, v<4>, w<4>, lo, hi;\n\t"
        "ld.shared.v4.b32 {p4,p5,p6,p7}, [$0+64];\n\t"
        "ld.shared.v4.b32 {p8,p9,p10,p11}, [$0+128];\n\t"
        + cvt
        + ar
        + "st.shared.v4.b32 [$1], {v0,v1,v2,v3};\n\t"
        f"st.shared.v4.b32 [$1+{cs}], {{w0,w1,w2,w3}};\n\t"
        "}\n"
    )
    llvm.inline_asm(
        None,
        [Int32(src).ir_value(loc=loc, ip=ip), Int32(dst).ir_value(loc=loc, ip=ip)],
        asm,
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _wino_item_p1(src: Int32, dst: Int32, in_half: bool, *, loc=None, ip=None):
    # pass 1: V0 = d0 - d2, V3' = d3 - d1 (= -V3, folds the minus sign of M3 into V).
    cvt = _cvt4(range(16), range(16), in_half)
    ar = ""
    for i in range(4):
        ar += f"sub.f16x2 v{i}, q{i}, q{8 + i};\n\t"
        ar += f"sub.f16x2 w{i}, q{12 + i}, q{4 + i};\n\t"
    cs = VCOORD_BYTES
    asm = (
        "{\n\t.reg .b32 p<16>, q<16>, v<4>, w<4>, lo, hi;\n\t"
        "ld.shared.v4.b32 {p0,p1,p2,p3}, [$0];\n\t"
        "ld.shared.v4.b32 {p4,p5,p6,p7}, [$0+64];\n\t"
        "ld.shared.v4.b32 {p8,p9,p10,p11}, [$0+128];\n\t"
        "ld.shared.v4.b32 {p12,p13,p14,p15}, [$0+192];\n\t"
        + cvt
        + ar
        + "st.shared.v4.b32 [$1], {v0,v1,v2,v3};\n\t"
        f"st.shared.v4.b32 [$1+{cs}], {{w0,w1,w2,w3}};\n\t"
        "}\n"
    )
    llvm.inline_asm(
        None,
        [Int32(src).ir_value(loc=loc, ip=ip), Int32(dst).ir_value(loc=loc, ip=ip)],
        asm,
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bfly2(a0, a1, b0, b1, *, loc=None, ip=None):
    # (a, b) <- (a + b, a - b) on two fp32 lanes with packed f32x2 add/sub (sm_100)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            cutlass.Float32(a0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(a1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b1).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b64 ra, rb, rs, rd;\n\t"
        "mov.b64 ra, {$4, $5};\n\tmov.b64 rb, {$6, $7};\n\t"
        "add.rn.f32x2 rs, ra, rb;\n\tsub.rn.f32x2 rd, ra, rb;\n\t"
        "mov.b64 {$0, $1}, rs;\n\tmov.b64 {$2, $3}, rd;\n\t}",
        "=f,=f,=f,=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


DBG = False  # per-role cycle instrumentation (diagnostic builds only)


def _clk():
    if DBG:
        return cute.arch.clock64()
    return cutlass.Int64(0)


def _gtm():
    if DBG:
        return _globaltimer()
    return cutlass.Int64(0)


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
    )


class Wino23Conv:
    def __init__(self, n_img, h, w, cin, cout, nslots, in_half=False, out_half=False):
        # trace-time constants: the input (bf16/fp16) and output element types
        self.in_dtype = cutlass.Float16 if in_half else cutlass.BFloat16
        self.out_dtype = cutlass.Float16 if out_half else cutlass.BFloat16
        assert cout == NCO and cin % KC == 0
        # Grid = (4, 1, nslots).  Each slot of 4 CTAs is either one 4-CTA cluster (2 pairs that
        # share U via TMA multicast) or two 2-CTA fallback clusters (2 plain pairs).
        self.n_img, self.h, self.w, self.cin, self.cout = n_img, h, w, cin, cout
        self.nkc = cin // KC
        self.hb = (h + RC - 1) // RC
        self.wb = (w + 2 * WT - 1) // (2 * WT)
        self.nblk = n_img * self.hb * self.wb
        self.nunit = (self.nblk + 3) // 4  # slot units: quads of blocks
        self.nslots = min(nslots, self.nunit)

        # warps: [0-7] epilogue (4 lane quadrants x 2 column halves), [8-11] transform,
        # 12 MMA, 13 U-TMA, 14 x-TMA, 15 spare
        self.num_epi_warps = 8
        self.tr_warp0 = 8
        self.num_tr_warps = 4
        self.mma_warp = 12
        self.utma_warp = 13
        self.xtma_warp = 14
        self.threads = 32 * 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mU: cute.Tensor,
        mY: cute.Tensor,
        mD: cute.Tensor,
        stream: cuda.CUstream,
    ):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float16,
            cutlass.Float16,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            cutlass.Float32,
            tcgen05.CtaGroup.TWO,
            (2 * MT, NCO),
        )
        mma_tiler = (2 * MT, NCO, KC)
        cl4 = cute.tiled_divide(cute.make_layout((4, 1, 1)), (tiled_mma.thr_id.shape,))
        cl2 = cute.tiled_divide(cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,))

        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, cutlass.Float16, 1)
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, cutlass.Float16, NUM_U_STAGE * U_GROUP
        )
        b_layout_1 = cute.slice_(b_layout, (None, None, None, 0))
        tma_atom_u4, tma_tensor_u4 = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((4, 1), tiled_mma.thr_id),
            mU,
            b_layout_1,
            mma_tiler,
            tiled_mma,
            cl4.shape,
        )
        tma_atom_u2, tma_tensor_u2 = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((2, 1), tiled_mma.thr_id),
            mU,
            b_layout_1,
            mma_tiler,
            tiled_mma,
            cl2.shape,
        )
        u_bytes = cute.size_in_bytes(cutlass.Float16, b_layout_1) * 2

        # half-width (N = 128) MMA used for the two chunks around the mid-block butterfly, so the
        # butterfly of one Cout half overlaps the MMAs of the other.  Its U tiles (64 Cout per
        # CTA) alias the same SMEM stages; D column j still holds Cout (half * 128 + j).
        tiled_mma_h = sm100_utils.make_trivial_tiled_mma(
            cutlass.Float16,
            cutlass.Float16,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            cutlass.Float32,
            tcgen05.CtaGroup.TWO,
            (2 * MT, NCO // 2),
        )
        mma_tiler_h = (2 * MT, NCO // 2, KC)
        b_layout_h = sm100_utils.make_smem_layout_b(
            tiled_mma_h, mma_tiler_h, cutlass.Float16, NUM_U_STAGE * U_GROUP * 2
        )
        b_layout_h_1 = cute.slice_(b_layout_h, (None, None, None, 0))
        tma_atom_uh4, tma_tensor_uh4 = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((4, 1), tiled_mma_h.thr_id),
            mU,
            b_layout_h_1,
            mma_tiler_h,
            tiled_mma_h,
            cl4.shape,
        )
        tma_atom_uh2, tma_tensor_uh2 = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((2, 1), tiled_mma_h.thr_id),
            mU,
            b_layout_h_1,
            mma_tiler_h,
            tiled_mma_h,
            cl2.shape,
        )

        # raw input TMA: box (1 img, BOXH rows, BOXW px, KC ch)
        raw_layout = cute.make_layout(
            (1, BOXH, BOXW, KC), stride=(BOXH * BOXW * KC, BOXW * KC, KC, 1)
        )
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mX, raw_layout, (1, BOXH, BOXW, KC)
        )

        # output TMA store: per-warp box (1 img, 2 rows, 32 px, 16 Cout)
        epi_layout = cute.make_layout(
            (1, 2, 2 * WT, DR_COLS), stride=(2 * 2 * WT * DR_COLS, 2 * WT * DR_COLS, DR_COLS, 1)
        )
        tma_atom_y, tma_tensor_y = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mY, epi_layout, (1, 2, 2 * WT, DR_COLS)
        )

        self.kernel(
            tiled_mma,
            tiled_mma_h,
            tma_atom_u4,
            tma_tensor_u4,
            tma_atom_u2,
            tma_tensor_u2,
            tma_atom_uh4,
            tma_tensor_uh4,
            tma_atom_uh2,
            tma_tensor_uh2,
            tma_atom_x,
            tma_tensor_x,
            tma_atom_y,
            tma_tensor_y,
            cl4,
            cl2,
            a_layout,
            b_layout,
            b_layout_h,
            raw_layout,
            epi_layout,
            u_bytes,
            mD,
        ).launch(
            grid=(4, 1, self.nslots),
            block=(self.threads, 1, 1),
            cluster=(4, 1, 1),
            fallback_cluster=(2, 1, 1),
            smem_merge_branch_allocs=True,
            min_blocks_per_mp=1,
            use_pdl=True,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_h: cute.TiledMma,
        tma_atom_u4: cute.CopyAtom,
        mU4: cute.Tensor,
        tma_atom_u2: cute.CopyAtom,
        mU2: cute.Tensor,
        tma_atom_uh4: cute.CopyAtom,
        mUh4: cute.Tensor,
        tma_atom_uh2: cute.CopyAtom,
        mUh2: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        tma_atom_y: cute.CopyAtom,
        mY: cute.Tensor,
        cl4: cute.Layout,
        cl2: cute.Layout,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        b_layout_h: cute.ComposedLayout,
        raw_layout: cute.Layout,
        epi_layout: cute.Layout,
        u_bytes: cutlass.Constexpr,
        mD: cute.Tensor,
    ):
        cdx, cdy, cdz = cute.arch.block_in_cluster_dim()
        if cdx == 4:
            self.body(
                tiled_mma,
                tiled_mma_h,
                tma_atom_u4,
                mU4,
                tma_atom_uh4,
                mUh4,
                tma_atom_x,
                mX,
                tma_atom_y,
                mY,
                cl4,
                a_layout,
                b_layout,
                b_layout_h,
                raw_layout,
                epi_layout,
                u_bytes,
                2,
                mD,
            )
        else:
            self.body(
                tiled_mma,
                tiled_mma_h,
                tma_atom_u2,
                mU2,
                tma_atom_uh2,
                mUh2,
                tma_atom_x,
                mX,
                tma_atom_y,
                mY,
                cl2,
                a_layout,
                b_layout,
                b_layout_h,
                raw_layout,
                epi_layout,
                u_bytes,
                1,
                mD,
            )

    @cute.jit
    def _dbg(self, mD, k: cutlass.Constexpr, v):
        if cutlass.const_expr(DBG):
            if cute.arch.lane_idx() == 0:
                cute.arch.atomic_add((mD.iterator + k).llvm_ptr, v)

    @cute.jit
    def body(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_h: cute.TiledMma,
        tma_atom_u: cute.CopyAtom,
        mU: cute.Tensor,
        tma_atom_uh: cute.CopyAtom,
        mUh: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mX: cute.Tensor,
        tma_atom_y: cute.CopyAtom,
        mY: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        b_layout_h: cute.ComposedLayout,
        raw_layout: cute.Layout,
        epi_layout: cute.Layout,
        u_bytes: cutlass.Constexpr,
        pairs: cutlass.Constexpr,
        mD: cute.Tensor,
    ):
        in_half = self.in_dtype is cutlass.Float16

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
        cta_v = bidx % 2
        is_leader = cta_v == 0
        cid = bidz  # slot index
        cta_rank = bidx  # block offset within the slot quad

        if warp_idx == self.utma_warp:
            cpasync.prefetch_descriptor(tma_atom_u)
            cpasync.prefetch_descriptor(tma_atom_uh)
        if warp_idx == self.xtma_warp:
            cpasync.prefetch_descriptor(tma_atom_x)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_y)

        @cute.struct
        class SharedStorage:
            u_mbar: cute.struct.MemRange[cutlass.Int64, NUM_U_STAGE * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, NUM_V_STAGE * 2]
            x_mbar: cute.struct.MemRange[cutlass.Int64, NUM_RAW_STAGE * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            af_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            p0lo_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            p0hi_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

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
        # pass-0 result per Cout half (MMA -> butterfly warps of that half); the butterfly warps'
        # release lets the pass-1 MMAs of that half start
        p0lo_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.p0lo_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        p0hi_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.p0hi_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # pass-1 result (-> drain).  One stage: the next block's MMAs wait for the release.
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.num_epi_warps),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # "accumulator B drained": lets the next block's coord-1 MMAs (into A) start while the
        # epilogue is still draining B
        b_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.num_epi_warps),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # "accumulator A final": committed after the last coord-0 MMAs of pass 1 so the epilogue
        # drains y0 while the tensor core is still finishing coord 3 (accumulator B)
        af_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.af_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.num_epi_warps),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
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

        pipeline_init_arrive(cluster_shape_mn=(2 * pairs, 1), is_relaxed=True)

        # SMEM buffers
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

        # MMA fragments
        tCrB = tiled_mma.make_fragment_B(sU)  # (MMA, 1, 2, STAGE)
        tCrBh = tiled_mma_h.make_fragment_B(sU_h)  # (MMA, 1, 2, 2 * STAGE)
        acc_shape = tiled_mma.partition_shape_C((2 * MT, NCO))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, NACC))
        # half-N accumulators: A_lo, A_hi, B_lo, B_hi (128 columns each)
        acc_shape_h = tiled_mma_h.partition_shape_C((2 * MT, NCO // 2))
        tCtAccH_fake = tiled_mma_h.make_fragment_C(cute.append(acc_shape_h, 2 * NACC))

        # U TMA partition: (N=Cout, K=Cin, L=12) tiled by (256, 32)
        gU = cute.local_tile(
            mU, cute.slice_((2 * MT, NCO, KC), (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(cta_v)
        tCgU = thr_mma.partition_B(gU)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        blk_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
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

        pipeline_init_wait(cluster_shape_mn=(2 * pairs, 1))

        if warp_idx == self.xtma_warp + 1:
            cute.arch.setmaxregister_decrease(56)

        hb = self.hb
        wb = self.wb
        nblk = self.nblk
        nkc = self.nkc
        nslots = self.nslots
        # rotate the Cin-chunk order per slot so CTAs do not all fetch the same U/x lines at once
        rot = cid % nkc

        # =================== U TMA producer ===================
        if warp_idx == self.utma_warp:
            cute.arch.setmaxregister_decrease(56)
            # U is produced by the preceding filter-transform kernel (programmatic launch)
            cute.arch.griddepcontrol_wait()
            st = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_U_STAGE)
            unit = cid
            while unit < self.nunit:
                # pass 0: chunks 0 .. nkc-2 full width, chunk nkc-1 as two Cout halves
                for kc in cutlass.range(nkc - 1, unroll=1):
                    for g in cutlass.range_constexpr(NACC * NTAP // U_GROUP):
                        u_pipe.producer_acquire(st)
                        for gi in cutlass.range_constexpr(U_GROUP):
                            cute.copy(
                                tma_atom_u,
                                tUgU[(None, 0, (kc + rot) % nkc, g * U_GROUP + gi)],
                                tUsU[(None, st.index * U_GROUP + gi)],
                                tma_bar_ptr=u_pipe.producer_get_barrier(st),
                                mcast_mask=u_mcast_mask,
                            )
                        st.advance()
                for g in cutlass.range_constexpr(NACC * NTAP // U_GROUP):
                    u_pipe.producer_acquire(st)
                    for gi in cutlass.range_constexpr(U_GROUP):
                        for hh in cutlass.range_constexpr(2):
                            cute.copy(
                                tma_atom_uh,
                                tUgUh[(None, hh, (nkc - 1 + rot) % nkc, g * U_GROUP + gi)],
                                tUsUh[(None, st.index * 2 * U_GROUP + 2 * gi + hh)],
                                tma_bar_ptr=u_pipe.producer_get_barrier(st),
                                mcast_mask=u_mcast_mask,
                            )
                    st.advance()
                # pass 1: chunk 0 as two Cout halves, chunks 1 .. nkc-1 full width
                for g in cutlass.range_constexpr(NACC * NTAP // U_GROUP):
                    u_pipe.producer_acquire(st)
                    for gi in cutlass.range_constexpr(U_GROUP):
                        for hh in cutlass.range_constexpr(2):
                            cute.copy(
                                tma_atom_uh,
                                tUgUh[(None, hh, rot % nkc, NACC * NTAP + g * U_GROUP + gi)],
                                tUsUh[(None, st.index * 2 * U_GROUP + 2 * gi + hh)],
                                tma_bar_ptr=u_pipe.producer_get_barrier(st),
                                mcast_mask=u_mcast_mask,
                            )
                    st.advance()
                for kc in cutlass.range(1, nkc, unroll=1):
                    for g in cutlass.range_constexpr(NACC * NTAP // U_GROUP):
                        u_pipe.producer_acquire(st)
                        for gi in cutlass.range_constexpr(U_GROUP):
                            cute.copy(
                                tma_atom_u,
                                tUgU[(None, 0, (kc + rot) % nkc, NACC * NTAP + g * U_GROUP + gi)],
                                tUsU[(None, st.index * U_GROUP + gi)],
                                tma_bar_ptr=u_pipe.producer_get_barrier(st),
                                mcast_mask=u_mcast_mask,
                            )
                        st.advance()
                unit += nslots
            u_pipe.producer_tail(st)

        # =================== raw x TMA producer ===================
        if warp_idx == self.xtma_warp:
            cute.arch.setmaxregister_decrease(56)
            st = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_RAW_STAGE)
            tw = cutlass.Int64(0)
            unit = cid
            while unit < self.nunit:
                blk = unit * 4 + cta_rank
                if blk >= nblk:
                    blk = nblk - 1
                n = blk // (hb * wb)
                rem = blk - n * (hb * wb)
                h0 = (rem // wb) * RC
                w0 = (rem % wb) * (2 * WT)
                gX = cute.local_tile(
                    cute.domain_offset((n, h0 - 1, w0 - 1, 0), mX),
                    (1, BOXH, BOXW, KC),
                    (0, 0, 0, None),
                )
                tXsX, tXgX = cpasync.tma_partition(
                    tma_atom_x,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sRaw, 0, 4),
                    cute.group_modes(gX, 0, 4),
                )
                for pss in cutlass.range_constexpr(NPASS):
                    for kc in cutlass.range(nkc, unroll=1):
                        q0 = _clk()
                        x_pipe.producer_acquire(st)
                        tw += _clk() - q0
                        cute.copy(
                            tma_atom_x,
                            tXgX[(None, (kc + rot) % nkc)],
                            tXsX[(None, st.index)],
                            tma_bar_ptr=x_pipe.producer_get_barrier(st),
                        )
                        st.advance()
                unit += nslots
            self._dbg(mD, 21, tw)
            x_pipe.producer_tail(st)

        # =================== transform warps ===================
        if warp_idx >= self.tr_warp0 and warp_idx < self.tr_warp0 + self.num_tr_warps:
            cute.arch.setmaxregister_decrease(96)
            ttid = tidx - 32 * self.tr_warp0
            twarp = ttid // 32
            lane = ttid % 32
            c = lane & 3
            e = (lane >> 2) & 1
            qw = lane >> 3
            raw_base = sRaw.iterator.toint()
            v_base = sV.iterator.toint()
            src_off = []
            dst_off = []
            for p in cutlass.range_constexpr(5):
                pi = (p * 4 + twarp) * 4 + qw
                a = pi // WT
                t0 = pi % WT
                ir = 2 * a + e
                t = (t0 + e) % WT
                m = ir * WT + t
                src_off.append((ir * BOXW + 2 * t) * (KC * 2) + c * 16)
                dst_off.append(m * (KC * 2) + ((c ^ ((m >> 1) & 3)) << 4))
            xst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_RAW_STAGE)
            vst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, NUM_V_STAGE)
            tx = cutlass.Int64(0)
            tv = cutlass.Int64(0)
            tc = cutlass.Int64(0)
            tt0 = _clk()
            unit = cid
            while unit < self.nunit:
                for pss in cutlass.range_constexpr(NPASS):
                    for kc in cutlass.range(nkc, unroll=1):
                        q0 = _clk()
                        x_pipe.consumer_wait(xst)
                        q1 = _clk()
                        v_pipe.producer_acquire(vst)
                        q2 = _clk()
                        tx += q1 - q0
                        tv += q2 - q1
                        rb = raw_base + xst.index * RAW_BYTES
                        vb = v_base + vst.index * VSTAGE_BYTES
                        for p in cutlass.range_constexpr(5):
                            if cutlass.const_expr(pss == 0):
                                _wino_item_p0(rb + src_off[p], vb + dst_off[p], in_half)
                            else:
                                _wino_item_p1(rb + src_off[p], vb + dst_off[p], in_half)
                        _fence_proxy_async_smem()
                        tc += _clk() - q2
                        cute.arch.sync_warp()
                        x_pipe.consumer_release(xst)
                        with cute.arch.elect_one():
                            v_pipe.producer_commit(vst)
                        xst.advance()
                        vst.advance()
                unit += nslots
            self._dbg(mD, 10, tx)
            self._dbg(mD, 11, tv)
            self._dbg(mD, 12, tc)
            self._dbg(mD, 13, _clk() - tt0)

        # =================== MMA warp ===================
        if warp_idx == self.mma_warp:
            cute.arch.setmaxregister_decrease(56)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tAccH = cute.make_tensor(tmem_ptr, tCtAccH_fake.layout)
            v_raw_ptr = sV.iterator
            ust = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_U_STAGE)
            vst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, NUM_V_STAGE)
            ast = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            bst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            afst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            lost = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            host = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            if is_leader:
                p0lo_pipe.producer_acquire(lost)
                p0hi_pipe.producer_acquire(host)
            # Per block: pass 0 (coords 1, 2 -> A, B) with chunk 0 peeled (coord-2 MMAs wait until
            # the epilogue drained B of the previous block) and the last chunk split into Cout
            # halves (lo half committed first so its butterfly overlaps the hi-half MMAs); pass 1
            # (coords 0, 3) with chunk 0 split (lo MMAs start once the lo butterfly is done) and
            # the last chunk peeled (after its coord-0 MMAs A = y0 is final).
            unit = cid
            while unit < self.nunit:
                # coord-1 MMAs into A only need y0 of the previous block drained (not the whole
                # previous pass 1 retired), so the tensor pipe never empties between blocks
                if is_leader:
                    af_pipe.producer_acquire(afst)
                # ---- pass 0, chunk 0 ----
                if is_leader:
                    v_pipe.consumer_wait(vst)
                for ut in cutlass.range_constexpr(NACC * NTAP):
                    jj = ut // NTAP
                    r = ut % NTAP
                    if cutlass.const_expr(ut == NTAP):
                        if is_leader:
                            b_pipe.consumer_wait(bst)
                    if is_leader:
                        if cutlass.const_expr(ut % U_GROUP == 0):
                            u_pipe.consumer_wait(ust)
                        vptr = cute.recast_ptr(
                            v_raw_ptr
                            + (vst.index * (VSTAGE_BYTES // 2) + (jj * VROWS + r * WT) * KC),
                            a_layout.inner,
                            dtype=cutlass.Float16,
                        )
                        sA = cute.make_tensor(vptr, a_layout.outer)
                        tCrA = tiled_mma.make_fragment_A(sA)
                        for kb in cutlass.range_constexpr(KC // 16):
                            if cutlass.const_expr(r == 0 and kb == 0):
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            else:
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(
                                tiled_mma,
                                tAcc[(None, None, None, jj)],
                                tCrA[(None, None, kb, 0)],
                                tCrB[(None, None, kb, ust.index * U_GROUP + ut % U_GROUP)],
                                tAcc[(None, None, None, jj)],
                            )
                        if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                            u_pipe.consumer_release(ust)
                    if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                        ust.advance()
                    if cutlass.const_expr(ut == NACC * NTAP - 1):
                        if is_leader:
                            b_pipe.consumer_release(bst)
                        bst.advance()
                if is_leader:
                    v_pipe.consumer_release(vst)
                vst.advance()
                # ---- pass 0, chunks 1 .. nkc-2 ; pass 1, chunks 1 .. nkc-2 ----
                for pss in cutlass.range_constexpr(NPASS):
                    if cutlass.const_expr(pss == 1):
                        # ---- pass 1, chunk 0: Cout halves, each after its butterfly ----
                        if is_leader:
                            v_pipe.consumer_wait(vst)
                        uh0 = ust.clone()
                        for hh in cutlass.range_constexpr(2):
                            if is_leader:
                                if cutlass.const_expr(hh == 0):
                                    p0lo_pipe.producer_acquire(lost)
                                else:
                                    p0hi_pipe.producer_acquire(host)
                            uh = uh0.clone()
                            for ut in cutlass.range_constexpr(NACC * NTAP):
                                jj = ut // NTAP
                                r = ut % NTAP
                                if is_leader:
                                    if cutlass.const_expr(hh == 0 and ut % U_GROUP == 0):
                                        u_pipe.consumer_wait(uh)
                                    vptr = cute.recast_ptr(
                                        v_raw_ptr
                                        + (
                                            vst.index * (VSTAGE_BYTES // 2)
                                            + (jj * VROWS + r * WT) * KC
                                        ),
                                        a_layout.inner,
                                        dtype=cutlass.Float16,
                                    )
                                    sA = cute.make_tensor(vptr, a_layout.outer)
                                    tCrA = tiled_mma_h.make_fragment_A(sA)
                                    for kb in cutlass.range_constexpr(KC // 16):
                                        tiled_mma_h.set(tcgen05.Field.ACCUMULATE, True)
                                        cute.gemm(
                                            tiled_mma_h,
                                            tAccH[(None, None, None, 2 * jj + hh)],
                                            tCrA[(None, None, kb, 0)],
                                            tCrBh[
                                                (
                                                    None,
                                                    None,
                                                    kb,
                                                    uh.index * 2 * U_GROUP
                                                    + 2 * (ut % U_GROUP)
                                                    + hh,
                                                )
                                            ],
                                            tAccH[(None, None, None, 2 * jj + hh)],
                                        )
                                    if cutlass.const_expr(hh == 1 and ut % U_GROUP == U_GROUP - 1):
                                        u_pipe.consumer_release(uh)
                                if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                                    uh.advance()
                        for g in cutlass.range_constexpr(NACC * NTAP // U_GROUP):
                            ust.advance()
                        if is_leader:
                            v_pipe.consumer_release(vst)
                        vst.advance()
                    for kc in cutlass.range(1, nkc - 1, unroll=1):
                        if is_leader:
                            v_pipe.consumer_wait(vst)
                        for ut in cutlass.range_constexpr(NACC * NTAP):
                            jj = ut // NTAP
                            r = ut % NTAP
                            if is_leader:
                                if cutlass.const_expr(ut % U_GROUP == 0):
                                    u_pipe.consumer_wait(ust)
                                vptr = cute.recast_ptr(
                                    v_raw_ptr
                                    + (
                                        vst.index * (VSTAGE_BYTES // 2) + (jj * VROWS + r * WT) * KC
                                    ),
                                    a_layout.inner,
                                    dtype=cutlass.Float16,
                                )
                                sA = cute.make_tensor(vptr, a_layout.outer)
                                tCrA = tiled_mma.make_fragment_A(sA)
                                for kb in cutlass.range_constexpr(KC // 16):
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    cute.gemm(
                                        tiled_mma,
                                        tAcc[(None, None, None, jj)],
                                        tCrA[(None, None, kb, 0)],
                                        tCrB[(None, None, kb, ust.index * U_GROUP + ut % U_GROUP)],
                                        tAcc[(None, None, None, jj)],
                                    )
                                if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                                    u_pipe.consumer_release(ust)
                            if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                                ust.advance()
                        if is_leader:
                            v_pipe.consumer_release(vst)
                        vst.advance()
                    if cutlass.const_expr(pss == 0):
                        # ---- pass 0, last chunk: Cout halves; lo committed before hi ----
                        if is_leader:
                            v_pipe.consumer_wait(vst)
                        uh0 = ust.clone()
                        for hh in cutlass.range_constexpr(2):
                            uh = uh0.clone()
                            for ut in cutlass.range_constexpr(NACC * NTAP):
                                jj = ut // NTAP
                                r = ut % NTAP
                                if is_leader:
                                    if cutlass.const_expr(hh == 0 and ut % U_GROUP == 0):
                                        u_pipe.consumer_wait(uh)
                                    vptr = cute.recast_ptr(
                                        v_raw_ptr
                                        + (
                                            vst.index * (VSTAGE_BYTES // 2)
                                            + (jj * VROWS + r * WT) * KC
                                        ),
                                        a_layout.inner,
                                        dtype=cutlass.Float16,
                                    )
                                    sA = cute.make_tensor(vptr, a_layout.outer)
                                    tCrA = tiled_mma_h.make_fragment_A(sA)
                                    for kb in cutlass.range_constexpr(KC // 16):
                                        tiled_mma_h.set(tcgen05.Field.ACCUMULATE, True)
                                        cute.gemm(
                                            tiled_mma_h,
                                            tAccH[(None, None, None, 2 * jj + hh)],
                                            tCrA[(None, None, kb, 0)],
                                            tCrBh[
                                                (
                                                    None,
                                                    None,
                                                    kb,
                                                    uh.index * 2 * U_GROUP
                                                    + 2 * (ut % U_GROUP)
                                                    + hh,
                                                )
                                            ],
                                            tAccH[(None, None, None, 2 * jj + hh)],
                                        )
                                    if cutlass.const_expr(hh == 1 and ut % U_GROUP == U_GROUP - 1):
                                        u_pipe.consumer_release(uh)
                                if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                                    uh.advance()
                            if is_leader:
                                if cutlass.const_expr(hh == 0):
                                    p0lo_pipe.producer_commit(lost)
                                else:
                                    p0hi_pipe.producer_commit(host)
                        lost.advance()
                        host.advance()
                        for g in cutlass.range_constexpr(NACC * NTAP // U_GROUP):
                            ust.advance()
                        if is_leader:
                            v_pipe.consumer_release(vst)
                        vst.advance()
                # ---- pass 1, last chunk ----
                if is_leader:
                    v_pipe.consumer_wait(vst)
                for ut in cutlass.range_constexpr(NACC * NTAP):
                    jj = ut // NTAP
                    r = ut % NTAP
                    if is_leader:
                        if cutlass.const_expr(ut % U_GROUP == 0):
                            u_pipe.consumer_wait(ust)
                        vptr = cute.recast_ptr(
                            v_raw_ptr
                            + (vst.index * (VSTAGE_BYTES // 2) + (jj * VROWS + r * WT) * KC),
                            a_layout.inner,
                            dtype=cutlass.Float16,
                        )
                        sA = cute.make_tensor(vptr, a_layout.outer)
                        tCrA = tiled_mma.make_fragment_A(sA)
                        for kb in cutlass.range_constexpr(KC // 16):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(
                                tiled_mma,
                                tAcc[(None, None, None, jj)],
                                tCrA[(None, None, kb, 0)],
                                tCrB[(None, None, kb, ust.index * U_GROUP + ut % U_GROUP)],
                                tAcc[(None, None, None, jj)],
                            )
                        if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                            u_pipe.consumer_release(ust)
                    if cutlass.const_expr(ut % U_GROUP == U_GROUP - 1):
                        ust.advance()
                    if cutlass.const_expr(ut == NTAP - 1):
                        if is_leader:
                            af_pipe.producer_commit(afst)
                        afst.advance()
                if is_leader:
                    v_pipe.consumer_release(vst)
                vst.advance()
                if is_leader:
                    acc_pipe.producer_acquire(ast)
                    acc_pipe.producer_commit(ast)
                ast.advance()
                unit += nslots
            if is_leader:
                af_pipe.producer_tail(afst)
                acc_pipe.producer_tail(ast)

        # =================== epilogue warps (8: 4 lane quadrants x 2 column halves) =====
        if warp_idx < self.num_epi_warps:
            cute.arch.setmaxregister_increase(176)
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tAcc_flat = cute.make_tensor(
                tAcc.iterator, cute.make_layout((MT, NCO, NACC), stride=(65536, 1, NCO))
            )
            tAcc_div = cute.flat_divide(tAcc_flat, (MT, EPI_COLS))  # (128,32,1,8,2)
            t2r_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32
            )
            r2t_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition.x64), cutlass.Float32
            )
            t2r = tcgen05.make_tmem_copy(t2r_atom, tAcc_div[(None, None, 0, 0, 0)])
            r2t = tcgen05.make_tmem_copy(r2t_atom, tAcc_div[(None, None, 0, 0, 0)])
            sh = warp_idx // 4  # column half (128 cols of each accumulator)
            etid = tidx % 128
            thr_t2r = t2r.get_slice(etid)
            thr_r2t = r2t.get_slice(etid)
            tTR = thr_t2r.partition_S(tAcc_div)
            tRT = thr_r2t.partition_D(tAcc_div)
            cid_t = cute.make_identity_tensor((MT, EPI_COLS))
            tTRc = thr_t2r.partition_D(cid_t)
            # A and B column blocks loaded/stored by one copy each (one TMEM wait per block)
            rAB = cute.make_rmem_tensor(cute.append(tTRc.shape, NACC), cutlass.Float32)
            tAcc_dd = cute.flat_divide(tAcc_flat, (MT, DR_COLS))  # (128,16,1,16,2)
            d2r_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
            )
            d2r = tcgen05.make_tmem_copy(d2r_atom, tAcc_dd[(None, None, 0, 0, 0)])
            thr_d2r = d2r.get_slice(etid)
            tDRc = thr_d2r.partition_D(cute.make_identity_tensor((MT, DR_COLS)))
            NDR = NCO // DR_COLS // 2  # 16-Cout store steps per column half (8)
            lay16 = cute.make_rmem_tensor(tDRc.shape, self.out_dtype).layout
            # drain in 32-column TMEM loads: half the round trips of the 16-column store steps
            tAcc_d32 = cute.flat_divide(tAcc_flat, (MT, 2 * DR_COLS))  # (128,32,1,8,2)
            d32_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
            )
            d32 = tcgen05.make_tmem_copy(d32_atom, tAcc_d32[(None, None, 0, 0, 0)])
            thr_d32 = d32.get_slice(etid)
            tD32 = thr_d32.partition_S(tAcc_d32)
            tD32c = thr_d32.partition_D(cute.make_identity_tensor((MT, 2 * DR_COLS)))
            rD32 = cute.make_rmem_tensor(tD32c.shape, cutlass.Float32)
            ND32 = NDR // 2
            rY = [cute.make_rmem_tensor(tD32c.shape, self.out_dtype) for _ in range(2 * ND32)]

            m = etid
            tcol = m % WT
            lane_e = tidx % 32
            lrow = lane_e // WT  # row within the warp's 2-row box
            qd = warp_idx % 4  # lane quadrant = 2-row group

            ast = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            afst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            lcst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            hcst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            bpst = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            # accumulator B starts out free
            b_pipe.producer_acquire(bpst)
            with cute.arch.elect_one():
                b_pipe.producer_commit(bpst)
            bpst.advance()
            ea = cutlass.Int64(0)
            eb = cutlass.Int64(0)
            ec = cutlass.Int64(0)
            ed = cutlass.Int64(0)
            ee = cutlass.Int64(0)
            unit = cid
            while unit < self.nunit:
                blk = unit * 4 + cta_rank
                n = blk // (hb * wb)
                rem = blk - n * (hb * wb)
                # ---- phase A: butterfly (A, B) <- (A + B, A - B) of this warp's Cout half, as
                # soon as pass 0 has finished on that half ----
                q0 = _clk()
                if sh == 0:
                    p0lo_pipe.consumer_wait(lcst)
                else:
                    p0hi_pipe.consumer_wait(hcst)
                q1 = _clk()
                ea += q1 - q0
                for cc in cutlass.range_constexpr(NCO // EPI_COLS // 2):
                    ccg = sh * (NCO // EPI_COLS // 2) + cc
                    cute.copy(t2r, tTR[(None, None, None, 0, ccg, None)], rAB)
                    for i in cutlass.range_constexpr(EPI_COLS // 2):
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
                    cute.copy(r2t, rAB, tRT[(None, None, None, 0, ccg, None)])
                cute.arch.fence_view_async_tmem_store()
                with cute.arch.elect_one():
                    if sh == 0:
                        p0lo_pipe.consumer_release(lcst)
                    else:
                        p0hi_pipe.consumer_release(hcst)
                lcst.advance()
                hcst.advance()
                q2 = _clk()
                eb += q2 - q1
                # ---- phase B: drain y0 = A as soon as the last coord-0 MMAs are done (the
                # tensor core is still on coord 3), then y1 = B after pass 1 ----
                af_pipe.consumer_wait(afst)
                q3 = _clk()
                ec += q3 - q2
                for cc in cutlass.range_constexpr(ND32):
                    cute.copy(d32, tD32[(None, None, None, 0, sh * ND32 + cc, 0)], rD32)
                    rY[2 * cc].store(rD32.load().to(self.out_dtype))
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    af_pipe.consumer_release(afst)
                afst.advance()
                acc_pipe.consumer_wait(ast)
                with cute.arch.elect_one():
                    acc_pipe.consumer_release(ast)
                ast.advance()
                b_pipe.producer_acquire(bpst)
                for cc in cutlass.range_constexpr(ND32):
                    cute.copy(d32, tD32[(None, None, None, 0, sh * ND32 + cc, 1)], rD32)
                    rY[2 * cc + 1].store(rD32.load().to(self.out_dtype))
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    b_pipe.producer_commit(bpst)
                bpst.advance()
                q4 = _clk()
                ed += q4 - q3
                # TMEM is free for the next block's MMAs; write the outputs from registers via a
                # per-warp SMEM box and TMA stores (overlaps the next block's MMAs)
                gYt = cute.local_tile(
                    mY, (1, 2, 2 * WT, DR_COLS), (n, (rem // wb) * (RC // 2) + qd, rem % wb, None)
                )
                sEw = sEpi[(None, None, None, None, warp_idx)]
                tYs, tYg = cpasync.tma_partition(
                    tma_atom_y,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sEw, 0, 4),
                    cute.group_modes(gYt, 0, 4),
                )
                for cc in cutlass.range_constexpr(NDR):
                    ccg = sh * NDR + cc
                    if lane_e == 0:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.sync_warp()
                    c32 = cc // 2
                    off = (cc % 2) * DR_COLS
                    sy = sEpi[(0, lrow, 2 * tcol, None, warp_idx)]
                    cute.autovec_copy(
                        cute.make_tensor(rY[2 * c32].iterator + off, lay16),
                        cute.make_tensor(sy.iterator, lay16),
                    )
                    sy = sEpi[(0, lrow, 2 * tcol + 1, None, warp_idx)]
                    cute.autovec_copy(
                        cute.make_tensor(rY[2 * c32 + 1].iterator + off, lay16),
                        cute.make_tensor(sy.iterator, lay16),
                    )
                    _fence_proxy_async_smem()
                    cute.arch.sync_warp()
                    if lane_e == 0:
                        if blk < nblk:
                            cute.copy(tma_atom_y, tYs, tYg[(None, ccg)])
                        cute.arch.cp_async_bulk_commit_group()
                ee += _clk() - q4
                unit += nslots
            if lane_e == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            if warp_idx == 0:
                self._dbg(mD, 30, ea)
                self._dbg(mD, 31, eb)
                self._dbg(mD, 32, ec)
                self._dbg(mD, 33, ed)
                self._dbg(mD, 34, ee)
            epi_all_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


_cache = {}
_wt_cache = {}


WT_T = 16  # filter-transform tile: 16 Cout x 16 Cin per block


@cute.kernel
def _wt_kernel(mW: cute.Tensor, mU: cute.Tensor):
    # mW: (Cout, Cin*9) bf16 with (c, r, s) innermost; mU: (12, Cin, Cout) fp16 with row
    # l = pass*6 + jj*3 + r (pass 0: coords 1, 2; pass 1: coords 0, 3), Cout contiguous.
    # One block = 16 Cout x 16 Cin staged via SMEM so weight reads and U writes are coalesced.
    cute.arch.griddepcontrol_launch_dependents()
    bx, by, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    k0 = bx * WT_T
    c0 = by * WT_T
    smem = utils.SmemAllocator()
    sW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((WT_T, WT_T * 9), stride=(WT_T * 9 + 1, 1)),
        byte_alignment=16,
    )
    for it in cutlass.range_constexpr(WT_T * WT_T * 9 // 256):
        idx = tidx + 256 * it
        kk = idx // (WT_T * 9)
        q = idx % (WT_T * 9)
        sW[kk, q] = mW[k0 + kk, c0 * 9 + q].to(cutlass.Float32)
    cute.arch.barrier()
    kk = tidx % WT_T
    cc = tidx // WT_T
    for r in cutlass.range_constexpr(NTAP):
        g0 = sW[kk, cc * 9 + 3 * r]
        g1 = sW[kk, cc * 9 + 3 * r + 1]
        g2 = sW[kk, cc * 9 + 3 * r + 2]
        mU[r, c0 + cc, k0 + kk] = ((g0 + g1 + g2) * 0.5).to(cutlass.Float16)
        mU[3 + r, c0 + cc, k0 + kk] = ((g0 - g1 + g2) * 0.5).to(cutlass.Float16)
        mU[6 + r, c0 + cc, k0 + kk] = g0.to(cutlass.Float16)
        mU[9 + r, c0 + cc, k0 + kk] = g2.to(cutlass.Float16)


@cute.jit
def _wt_launch(mW: cute.Tensor, mU: cute.Tensor, cout: cutlass.Constexpr, cin: cutlass.Constexpr):
    _wt_kernel(mW, mU).launch(
        grid=(cout // WT_T, cin // WT_T, 1),
        block=(WT_T * WT_T, 1, 1),
        smem=WT_T * (WT_T * 9 + 1) * 4 + 64,
    )


def _weight_transform(weight):
    cout, cin = weight.shape[0], weight.shape[1]
    u = torch.empty((NPASS * NACC * NTAP, cin, cout), dtype=torch.float16, device=weight.device)
    w2 = weight.reshape(cout, cin * 9)
    key = (cout, cin)
    if key not in _wt_cache:
        _wt_cache[key] = cute.compile(
            _wt_launch,
            from_dlpack(w2, enable_tvm_ffi=True),
            from_dlpack(u, enable_tvm_ffi=True),
            cout,
            cin,
            options="--enable-tvm-ffi",
        )
    _wt_cache[key](w2, u)
    return u


@torch.no_grad()
def prepare(weight):
    """The transformed filter for :func:`run_prepared`: ``[Cout, Cin, 12]`` fp16, Cout contiguous."""
    u = _weight_transform(weight)  # (12, Cin, Cout) fp16
    return u.permute(2, 1, 0)


def run_prepared(x, ut, y):
    """``y = conv3x3(x, weight)`` with ``ut`` from :func:`prepare`; x, y channels-last
    ``[N, H, W, C]`` (bf16 or fp16 x, y in its own dtype); launches on the current torch stream."""
    n_img, h, w, cin = x.shape
    cout = ut.shape[0]
    assert h % RC == 0 and w % (2 * WT) == 0 and cout == NCO and cin % KC == 0
    key = (n_img, h, w, cin, cout, x.dtype, y.dtype)
    if key not in _cache:
        nslots = utils.HardwareInfo().get_max_active_clusters(2) // 2
        _cache[key] = cute.compile(
            Wino23Conv(
                n_img, h, w, cin, cout, nslots, x.dtype == torch.float16, y.dtype == torch.float16
            ),
            from_dlpack(x, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(ut, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(y, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(y, assumed_align=16, enable_tvm_ffi=True),
            make_fake_stream(use_tvm_ffi_env_stream=True),  # the caller's torch stream
            options="--enable-tvm-ffi",
        )
    _cache[key](x, ut, y, y)


def run(x, weight, y):
    run_prepared(x, prepare(weight), y)
