# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE router GEMM of decode steps: BF16 x [M, K] @ W [E, K].T -> BF16 logits [M, E] in one launch.

tcgen05 MMA with the router weight as the M operand (128 experts per CTA, "swap AB") and token tiles of 32 or 64 as
N; a cluster of ``SPLITS`` CTAs splits K and sums the partials in split order (``MoeRouterGemmKernel``), so a token's
logits do not depend on the batch it runs in. Weight TMA loads go out before ``griddepcontrol.wait``, x after it.
The compile-time ``route_top_k`` epilogue feeds a MoE engine that routes itself: per token the top ``route_top_k``
keys of each 128-expert tile, the NVFP4 quantization of x, and the PDL release right after the dependency wait.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from ...cute_dsl_utils import smem_optin_bytes
from .moe_decode_front import (
    _add,
    _asm,
    _gdc_launch,
    _gdc_wait,
    _i32,
    _ld_f32,
    _ld_shared,
    _max_u32,
    _nvfp4_block,
    _redux_max_u32,
    _sel_eq,
    _smem_addr,
    _st_b32,
    _st_shared,
    _st_u8,
    _st_v2,
    _struct,
    _unpack,
)
from .moe_route_front import (
    _bulk_commit,
    _bulk_push,
    _cluster_arrive_relaxed,
    _cluster_arrive_release,
    _cluster_wait_acquire,
    _keys2,
)
from .ptx_helpers import _pack_bf16x2_rn

EXPERT_TILE = 128  # UMMA M (cta_group 1): TMEM lane = expert
K_TILE = 64  # one 128-byte swizzle atom of BF16 per row
SPLITS = 8  # K ranges = CTAs per cluster
TOKEN_TILES = (32, 64)  # UMMA N
# The ``route_top_k`` epilogue: one lane stores each key of a token's tile, and each expert tile's CTA quantizes
# this many rows of its token tile (threads / 16 lanes per row).
ROUTE_MAX_TOP_K = 32
QUANT_ROWS_PER_TILE = 8


@dsl_user_op
def _tmem_ld16(taddr: Int32, *, loc=None, ip=None):
    """16 consecutive TMEM columns of this thread's lane (32x32b shape), as raw 32-bit words."""
    regs = ", ".join(f"${i}" for i in range(16))
    out = _asm(
        _struct(16),
        [_i32(taddr, loc, ip)],
        f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [$16];\n\ttcgen05.wait::ld.sync.aligned;",
        ",".join(["=r"] * 16) + ",r,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 16, loc, ip)


@dsl_user_op
def _ld_shared_v4_f32(addr: Int32, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_i32(addr, loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_shared_v4_b32(addr: Int32, *, loc=None, ip=None):
    out = _asm(
        _struct(4),
        [_i32(addr, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _tcgen05_after_sync(*, loc=None, ip=None) -> None:
    _asm(None, [], "tcgen05.fence::after_thread_sync;", "~{memory}", loc=loc, ip=ip)


class MoeRouterGemmKernel:
    """logits [M, E] (BF16, row-major, contiguous) = x [M, K] (BF16, row stride ``a_stride``) @ w [E, K].T.

    Grid: ``E / 128`` expert tiles x ``ceil(M / token_tile)`` token tiles x ``SPLITS`` K ranges; clusters of
    ``SPLITS`` CTAs along the K ranges (block index = split + SPLITS * (expert tile + E / 128 * token tile)).
    128 threads: warp 0 issues the TMA loads, warp 1 allocates TMEM and issues the MMAs, all four warps run the
    epilogue (thread i = TMEM lane i = expert i of the tile).

    Split-K reduction: each CTA stages its FP32 partial tile [token][expert] in the shared memory that held x
    (the MMAs are done with it), then pushes token slice o (``token_tile / SPLITS`` tokens) to CTA o with one
    bulk copy that completes bytes on CTA o's receive mbarrier; each CTA waits for its ``SPLITS`` slices and
    sums them in split order 0 .. SPLITS - 1.

    ``route_top_k`` (compile time; 0 is the plain GEMM): the epilogue a routing MoE engine reads.

    * Each token's top ``route_top_k`` reduceTopK keys of every expert tile, descending (``cand`` [M, E / 128,
      route_top_k] int32: twiddled BF16 logit << 16 | 65535 - expert, as the route front's).
    * The NVFP4 quantization of x (``xq`` [M, K / 2] u8, linear ``xsf`` [M, K / 16] u8, global scale ``gs``) from
      the x tiles in shared memory, bitwise ``fp4_quantize``: CTA (e, s) quantizes rows 8 e .. 8 e + 7 of its token
      tile over its K range, the K tiles the staging tile overwrites before it is written.
    * The PDL dependents are released right after the dependency wait."""

    threads = 128

    @staticmethod
    def _shape_error(
        *,
        num_rows: int,
        k_extent: int,
        num_experts: int,
        a_stride: int,
        token_tile: int,
        route_top_k: int = 0,
    ) -> Optional[str]:
        if num_experts % EXPERT_TILE:
            return f"num_experts={num_experts} must be a multiple of {EXPERT_TILE}"
        if token_tile not in TOKEN_TILES:
            return f"token_tile must be one of {TOKEN_TILES}"
        if k_extent % (SPLITS * K_TILE):
            return f"k_extent={k_extent} must split into {SPLITS} ranges of whole {K_TILE} tiles"
        if a_stride % 8 or a_stride < k_extent:
            return "x rows must be 16-byte aligned"
        if k_extent // SPLITS // K_TILE < 4:
            return "the staging tile reuses the x buffers: at least 4 K tiles per split"
        smem_bytes = (k_extent // SPLITS // K_TILE) * (EXPERT_TILE + token_tile) * K_TILE * 2
        if smem_bytes + token_tile * EXPERT_TILE * 4 + 2048 > smem_optin_bytes():
            return "shared memory over budget"
        if route_top_k and not 0 < route_top_k <= ROUTE_MAX_TOP_K:
            return f"route_top_k must be in 1..{ROUTE_MAX_TOP_K}"
        if route_top_k and (
            token_tile != QUANT_ROWS_PER_TILE * (num_experts // EXPERT_TILE)
            or (k_extent // SPLITS // K_TILE) % 4
        ):
            return f"route_top_k takes {QUANT_ROWS_PER_TILE} token rows per expert tile and K tiles in 4s"
        return None

    @classmethod
    def supports(cls, **shape: int) -> bool:
        """Whether the constructor takes these shape arguments (the integer keyword arguments of ``__init__`` other
        than ``use_pdl``)."""
        return cls._shape_error(**shape) is None

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        num_experts: int,
        a_stride: int,
        token_tile: int,
        use_pdl: bool = True,
        route_top_k: int = 0,
    ) -> None:
        err = self._shape_error(
            num_rows=num_rows,
            k_extent=k_extent,
            num_experts=num_experts,
            a_stride=a_stride,
            token_tile=token_tile,
            route_top_k=route_top_k,
        )
        if err is not None:
            raise ValueError(err)
        self.num_rows = num_rows
        self.k_extent = k_extent
        self.num_experts = num_experts
        self.a_stride = a_stride
        self.splits = SPLITS
        self.tn = token_tile
        self.token_tiles = -(-num_rows // token_tile)
        self.use_pdl = use_pdl
        self.route_top_k = route_top_k
        self.route = route_top_k > 0
        self.n_etiles = num_experts // EXPERT_TILE
        self.k_tiles = k_extent // SPLITS // K_TILE
        self.tps = token_tile // SPLITS  # tokens of a CTA's output slice
        self.slice_bytes = self.tps * EXPERT_TILE * 4
        self.ctas = self.n_etiles * self.token_tiles * SPLITS
        self.tmem_cols = max(32, token_tile)
        self.smem_bytes = (
            self.k_tiles * (EXPERT_TILE + token_tile) * K_TILE * 2 + token_tile * EXPERT_TILE * 4
        )

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        out_ptr: Int64,
        stream,
        cand_ptr: Optional[Int64] = None,
        gs_ptr: Optional[Int64] = None,
        xq_ptr: Optional[Int64] = None,
        xsf_ptr: Optional[Int64] = None,
    ) -> None:
        # route_top_k only: the candidates (cand_ptr), global scale and quantized x (gs_ptr / xq_ptr / xsf_ptr).
        # Swap AB: MMA A = router weight (M = experts), MMA B = x (N = tokens); both K-major. L = 1.
        mA = cute.make_tensor(
            w_ptr,
            cute.make_layout(
                (self.num_experts, self.k_extent, 1),
                stride=(self.k_extent, 1, self.num_experts * self.k_extent),
            ),
        )
        mB = cute.make_tensor(
            x_ptr,
            cute.make_layout(
                (self.num_rows, self.k_extent, 1),
                stride=(self.a_stride, 1, self.num_rows * self.a_stride),
            ),
        )
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            a_major,
            b_major,
            Float32,
            tcgen05.CtaGroup.ONE,
            (EXPERT_TILE, self.tn),
        )
        mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_tiler = (EXPERT_TILE, self.tn, K_TILE)
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_smem_staged = utils.sm100.make_smem_layout_a(tiled_mma, mma_tiler, BFloat16, self.k_tiles)
        b_smem_staged = utils.sm100.make_smem_layout_b(tiled_mma, mma_tiler, BFloat16, self.k_tiles)
        a_smem_one = cute.slice_(a_smem_staged, (None, None, None, 0))
        b_smem_one = cute.slice_(b_smem_staged, (None, None, None, 0))
        a_op = utils.sm100.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        a_bytes = cute.size_in_bytes(BFloat16, a_smem_one)
        b_bytes = cute.size_in_bytes(BFloat16, b_smem_one)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            out_ptr,
            a_smem_staged,
            b_smem_staged,
            a_bytes + b_bytes,
            mma_inst_k,
            cand_ptr,
            gs_ptr,
            xq_ptr,
            xsf_ptr,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.splits, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _quant_phase(
        self,
        sB,
        full_mbar,
        gs_ptr: Int64,
        xq_ptr: Int64,
        xsf_ptr: Int64,
        s,
        e_tile,
        t_tile,
        tidx,
        phase: cutlass.Constexpr,
    ) -> None:
        """NVFP4 block 16 phase + tidx % 16 of this CTA's K range (K tile = block // 4) for row 8 e_tile + tidx // 16
        of the token tile. sB is K-major with the 128-byte swizzle: row r is a 128-byte line of a K tile, its 16-byte
        chunk c at c ^ (r % 8)."""
        tn: cutlass.Constexpr = self.tn
        r_loc = e_tile * Int32(QUANT_ROWS_PER_TILE) + tidx // Int32(16)
        cbl = Int32(16 * phase) + tidx % Int32(16)
        kt = cbl // Int32(4)
        row = t_tile * Int32(tn) + r_loc
        if row < Int32(self.num_rows):
            gs = _ld_f32(gs_ptr)
            cute.arch.mbarrier_wait(full_mbar + kt, 0)
            c0 = (cbl % Int32(4)) * Int32(2)
            line = _smem_addr(sB.iterator) + kt * Int32(tn * K_TILE * 2) + r_loc * Int32(128)
            sw = r_loc % Int32(8)
            w0, w1, w2, w3 = _ld_shared_v4_b32(line + ((c0 ^ sw) * Int32(16)))
            w4, w5, w6, w7 = _ld_shared_v4_b32(line + (((c0 + Int32(1)) ^ sw) * Int32(16)))
            lo, hi, sf = _nvfp4_block(w0, w1, w2, w3, w4, w5, w6, w7, gs)
            cb = s * Int32(self.k_tiles * 4) + cbl
            _st_v2(xq_ptr + Int64(row) * Int64(self.k_extent // 2) + Int64(cb) * Int64(8), lo, hi)
            _st_u8(xsf_ptr + Int64(row) * Int64(self.k_extent // 16) + Int64(cb), sf)

    @cute.jit
    def _tile_candidates(self, cand_ptr: Int64, row, e_tile, e4, lane, p01, p23) -> None:
        """The warp's token row: its top ``route_top_k`` keys among this expert tile's 128 BF16 logits (4 per lane,
        packed in p01 / p23), descending, lane r storing key r."""
        k0, k1 = _keys2(p01, e_tile * Int32(EXPERT_TILE) + e4)
        k2, k3 = _keys2(p23, e_tile * Int32(EXPERT_TILE) + e4 + Int32(2))
        zero = Int32(0)
        mine = zero
        for r in cutlass.range_constexpr(self.route_top_k):
            # keys are unique and never 0, so clearing the picked one takes it out
            g = _redux_max_u32(_max_u32(_max_u32(k0, k1), _max_u32(k2, k3)))
            k0 = _sel_eq(k0, g, zero, k0)
            k1 = _sel_eq(k1, g, zero, k1)
            k2 = _sel_eq(k2, g, zero, k2)
            k3 = _sel_eq(k3, g, zero, k3)
            mine = _sel_eq(lane, Int32(r), g, mine)
        if lane < Int32(self.route_top_k):
            base = Int64(row) * Int64(self.n_etiles) + Int64(e_tile)
            _st_b32(cand_ptr + (base * Int64(self.route_top_k) + Int64(lane)) * Int64(4), mine)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        out_ptr: Int64,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        tx_bytes: cutlass.Constexpr,
        mma_inst_k: cutlass.Constexpr,
        cand_ptr: Optional[Int64] = None,
        gs_ptr: Optional[Int64] = None,
        xq_ptr: Optional[Int64] = None,
        xsf_ptr: Optional[Int64] = None,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        splits: cutlass.Constexpr = self.splits
        k_tiles: cutlass.Constexpr = self.k_tiles
        tps: cutlass.Constexpr = self.tps
        tn: cutlass.Constexpr = self.tn
        slice_bytes: cutlass.Constexpr = self.slice_bytes
        s = bidx % Int32(splits)  # rank in the cluster (1-D clusters along x)
        tile = bidx // Int32(splits)
        e_tile = tile % Int32(self.n_etiles)
        t_tile = tile // Int32(self.n_etiles)

        if warp == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            full_mbar: cute.struct.MemRange[cutlass.Int64, k_tiles]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            recv_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(
            element_type=BFloat16,
            layout=a_smem_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=BFloat16,
            layout=b_smem_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_staged.inner,
        )
        recv = smem.allocate_tensor(
            Float32, cute.make_layout((tn * EXPERT_TILE,)), byte_alignment=128
        )
        full_mbar = storage.full_mbar.data_ptr()
        acc_mbar = storage.acc_mbar.data_ptr()
        recv_mbar = storage.recv_mbar.data_ptr()

        if warp == 0:
            with cute.arch.elect_one():
                for kt in cutlass.range_constexpr(k_tiles):
                    cute.arch.mbarrier_init(full_mbar + kt, 1)
                cute.arch.mbarrier_init(acc_mbar, 1)
                cute.arch.mbarrier_init(recv_mbar, 1)
                # This CTA's own arrive: one slice from every split of the cluster.
                cute.arch.mbarrier_arrive_and_expect_tx(recv_mbar, splits * slice_bytes)
            cute.arch.mbarrier_init_fence()
        if warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, storage.tmem_holding_buf)
        # Peers push into the receive buffer only after this arrive (barrier inits released).
        _cluster_arrive_release()
        cute.arch.sync_threads()

        # Tile views: A (weights) tile e_tile, B (x) tile t_tile, K tiles [s * k_tiles, (s + 1) * k_tiles).
        gA = cute.local_tile(
            mA, cute.slice_((EXPERT_TILE, tn, K_TILE), (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_((EXPERT_TILE, tn, K_TILE), (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAgA_t = tAgA[(None, e_tile, None, 0)]
        tBgB_t = tBgB[(None, t_tile, None, 0)]
        k0 = s * Int32(k_tiles)

        # Producer: static weights before the dependency wait, x after it.
        if warp == 0:
            for kt in cutlass.range_constexpr(k_tiles):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full_mbar + kt, tx_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA_t[(None, k0 + kt)],
                    tAsA[(None, kt)],
                    tma_bar_ptr=full_mbar + kt,
                )
        if const_expr(self.use_pdl):
            _gdc_wait()
            if const_expr(self.route):
                _gdc_launch()
        if warp == 0:
            for kt in cutlass.range_constexpr(k_tiles):
                cute.copy(
                    tma_atom_b,
                    tBgB_t[(None, k0 + kt)],
                    tBsB[(None, kt)],
                    tma_bar_ptr=full_mbar + kt,
                )

        # MMA warp: K tiles in order, 4 instructions each, one accumulator.
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, storage.tmem_holding_buf)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((EXPERT_TILE, tn))
        tCtAcc = cute.make_tensor(tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        if warp == 1:
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kt in cutlass.range_constexpr(k_tiles):
                cute.arch.mbarrier_wait(full_mbar + kt, 0)
                for kb in cutlass.range_constexpr(K_TILE // mma_inst_k):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kb, kt)],
                        tCrB[(None, None, kb, kt)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(acc_mbar)

        if const_expr(self.route):
            # K tiles 0-3 of x now: the staging tile overwrites them next
            self._quant_phase(sB, full_mbar, gs_ptr, xq_ptr, xsf_ptr, s, e_tile, t_tile, tidx, 0)
            cute.arch.sync_threads()
        # Epilogue: thread = TMEM lane = expert. Stage the partial as [token][expert] FP32 over the x buffers.
        cute.arch.mbarrier_wait(acc_mbar, 0)
        _tcgen05_after_sync()
        tmem_base = _ld_shared(_smem_addr(storage.tmem_holding_buf.ptr))
        taddr = tmem_base + ((warp * Int32(32)) << Int32(16))
        stage_addr = _smem_addr(sB.iterator)
        recv_addr = _smem_addr(recv.iterator)
        for c in cutlass.range_constexpr(tn // 16):
            words = _tmem_ld16(taddr + Int32(c * 16))
            for j in cutlass.range_constexpr(16):
                _st_shared(
                    stage_addr + (Int32((c * 16 + j) * EXPERT_TILE) + tidx) * Int32(4), words[j]
                )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        _cluster_wait_acquire()  # every peer's receive mbarrier is initialized
        # Token slice o of the staging tile goes to slot s of CTA o's receive buffer.
        if warp == 0:
            with cute.arch.elect_one():
                for o in cutlass.range_constexpr(splits):
                    _bulk_push(
                        recv_addr + s * Int32(slice_bytes),
                        _smem_addr(recv_mbar),
                        stage_addr + Int32(o * slice_bytes),
                        Int32(o),
                        Int32(slice_bytes),
                    )
                _bulk_commit()
        if const_expr(self.route):
            # the K tiles the staging tile left alone, while the split-K slices are in flight
            for phase in cutlass.range_constexpr(1, k_tiles // 4):
                self._quant_phase(
                    sB, full_mbar, gs_ptr, xq_ptr, xsf_ptr, s, e_tile, t_tile, tidx, phase
                )
        cute.arch.mbarrier_wait(recv_mbar, 0)
        _cluster_arrive_relaxed()  # this CTA's slices have arrived (waited at the end)

        # Reduce this CTA's slice (tokens [s tps, (s + 1) tps) of the tile, all 128 experts) in split order.
        lane = tidx % Int32(32)
        e4 = lane * Int32(4)
        for i in cutlass.range_constexpr(-(-tps // 4)):
            lt = warp + Int32(4 * i)
            if lt < Int32(tps):
                base = recv_addr + (lt * Int32(EXPERT_TILE) + e4) * Int32(4)
                a0, a1, a2, a3 = _ld_shared_v4_f32(base)
                for p in cutlass.range_constexpr(1, splits):
                    b0, b1, b2, b3 = _ld_shared_v4_f32(base + Int32(p * slice_bytes))
                    a0 = _add(a0, b0)
                    a1 = _add(a1, b1)
                    a2 = _add(a2, b2)
                    a3 = _add(a3, b3)
                row = t_tile * Int32(tn) + s * Int32(tps) + lt
                if row < Int32(self.num_rows):
                    dst = out_ptr + (
                        Int64(row) * Int64(self.num_experts)
                        + Int64(e_tile * Int32(EXPERT_TILE) + e4)
                    ) * Int64(2)
                    p01 = _pack_bf16x2_rn(a0, a1)
                    p23 = _pack_bf16x2_rn(a2, a3)
                    _st_v2(dst, p01, p23)
                    if const_expr(self.route):
                        self._tile_candidates(cand_ptr, row, e_tile, e4, lane, p01, p23)
        # Every peer received its slices, so no outgoing copy still reads the staging tile (the bulk wait_group does
        # not track copies that complete on an mbarrier).
        _cluster_wait_acquire()
        cute.arch.sync_threads()
        if warp == 1:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
        if const_expr(self.use_pdl and not self.route):
            _gdc_launch()


def router_gemm_token_tile(m: int) -> Optional[int]:
    """Token tile for m rows, or None where the stock GEMM stays: 32 while it takes at most 3 per expert tile, so
    the clusters launch early (PDL) beside the producer in one wave, then 64 up to 128 rows."""
    if m <= 3 * 32:
        return 32
    if m <= 2 * 64:
        return 64
    return None
