# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-expert gate/up split-K GEMM with SiTU in its epilogue (one kernel instead of
gate/up -> SiTU, so the down GEMM depends on the GEMM directly).

``SplitKGemm`` with M tile 64 = 32 gate rows + their 32 up rows: the weight warp loads
two {64, 32} boxes per 64-wide K atom (gate rows ``32 m``, up rows ``half + 32 m``)
into the halves of the swizzled A atom, so every output row keeps the MMA K order,
the cluster K split and the owner's partial-sum order of the 64-row gate/up tactic
with the same split (bitwise-identical FP32 gate/up). The owner then stages the
reduced tile in shared memory and its 128 epilogue threads apply SiTU exactly as the
routing kernel's SiTU role (gate, up rounded to BF16; SiTU in FP32; BF16 h).

``SplitKSiluMul``: the same GEMM with vLLM's silu_and_mul in the epilogue (the DSpark
drafter's dense MLP): gate and up rounded to BF16, silu(gate) = gate / (1 + exp(-gate))
(IEEE exp and divide) rounded to BF16, times up in FP32, BF16 h."""

from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.host import architecture
from ..common.host import stream as current_stream
from ..common.primitives import div_rn, named_barrier
from ..moe.shared import situ
from .projection import matrix_view, weight_map
from .splitk import (
    ATOM_K,
    OWNER_RANK,
    SplitKGemm,
    SplitKTactic,
    _acquire_tensor_map,
    _fence_tcgen05,
    _smem,
    _store_shared_remote_v4,
    _tma_load_evict_first,
    _wait_cluster,
)

SITU_BARRIER = 2  # epilogue warps 4-7 only (128 threads)


class SplitKSitu(SplitKGemm):
    """h [B, half] = SiTU(x W_gate^T, x W_up^T) for a [2 half, K] gate/up weight;
    ``a_map`` must have {64, 32} boxes (``weight_map(weight, 32)``)."""

    def __init__(self, *, tactic, half):
        super().__init__(tactic=tactic)
        assert tactic.mma_m == 64 and half % 32 == 0, "32 gate + 32 up rows per tile"
        self.half = half

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, h: cute.Tensor, a_map: cutlass.Int64,
        beta: Float32, linear_beta: Float32, stream: _cuda.CUstream,
    ):  # fmt: skip
        # h is (half, N, 1) M-major: grid-x walks 32-column h tiles.
        self.kernel(a, b, h, a_map, beta, linear_beta).launch(
            grid=(
                self.half // 32,
                cute.ceil_div(h.layout.shape[1], self.cta_n) * self.split_k,
                h.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mH: cute.Tensor, a_map: cutlass.Int64,
        beta: Float32, linear_beta: Float32,
    ):  # fmt: skip
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        bar_tma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        mailbox, bar_reduce = sA, bar_mma_epilog  # split 1: compile-time dead
        if cutlass.const_expr(self.split_k > 1):
            mailbox = _smem(cutlass.Float32, self.mailbox_elements, 128)
            bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        # The owner's reduced [64, N] tile (M-major), read back by SiTU.
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        staged = _smem(cutlass.Float32, None, 128, tile_layout)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        tiles = (k_tile_start, k_tile_count)
        if warp_idx == 0:
            self.pair_warp(bar_full, bar_empty, sA, a_map, bidx * 32, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            self.situ_epilog(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, staged,
                cute.arch.thread_idx()[0] - 128, mailbox, bar_reduce, split_rank,
                mH, bidx, n_idx, beta, linear_beta,
            )  # fmt: skip

    @cute.experimental.jit
    def pair_warp(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        col0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """A: per K atom, the gate box (rows ``col0``) and the up box (rows ``half +
        col0``) into the atom's two 32-row halves, before the PDL wait (weights)."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        up0 = col0 + self.half
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            with cute.arch.elect_one():
                k0 = (k_tile_start + k_tile) * self.cta_k
                tile = s_tile[None, None, None, stage].iterator
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    box, mbar = tile + atom * self.cta_m * ATOM_K, bar_full + stage
                    k = k0 + atom * ATOM_K
                    _tma_load_evict_first(box, mbar, a_map, k, col0)
                    _tma_load_evict_first(box + 32 * ATOM_K, mbar, a_map, k, up0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def situ_epilog(
        self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, staged: cute.Tensor, epi_tid: cutlass.Int32,
        mailbox, bar_reduce, split_rank: cutlass.Int32, mH: cute.Tensor,
        bidx: cutlass.Int32, n_idx: cutlass.Int32, beta: Float32,
        linear_beta: Float32,
    ):  # fmt: skip
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        d_layout = utils.LayoutEnum.from_tensor(mH)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, cutlass.Float32,
            self.acc_dtype, epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        sD_epi = cute.flat_divide(staged, epi_tile)

        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, sD_epi, epi_tid)
        rmem = cute.AddressSpace.rmem
        rAcc = cute_ext.allocate(self.acc_dtype, rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            vpt = cutlass.const_expr(cute.size(rmem_layout))
            vpp = cutlass.const_expr(self.epilog_threads * vpt)
            if split_rank != OWNER_RANK:
                for i in cutlass.range_constexpr(0, vpt, 4):
                    words = (rAcc[i], rAcc[i + 1], rAcc[i + 2], rAcc[i + 3])
                    slot = mailbox.iterator + (split_rank - Int32(1)) * vpp
                    slot = slot + epi_tid * vpt + i
                    _store_shared_remote_v4(*words, slot, bar_reduce, Int32(OWNER_RANK))
            else:
                if epi_tid == 0:
                    expected = self.expected_transaction_bytes
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
                _wait_cluster(bar_reduce, Int32(0))
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for i in cutlass.range_constexpr(vpt):
                        rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]

        if split_rank == OWNER_RANK:
            cute_ext.partition_and_copy(thr_t2r, rAcc, sD_epi[None, None, 0, 0])
            named_barrier(SITU_BARRIER, 128)
            # Thread t: h column 32 bidx + t % 32, tokens t // 32 + 4 j.
            col, first = epi_tid % 32, epi_tid // 32
            tokens = mH.layout.shape[1]
            for j in cutlass.range_constexpr(self.cta_n // 4):
                local = first + 4 * j
                token = n_idx * self.cta_n + local
                if token < tokens:
                    gate = Float32(cutlass.BFloat16(staged[col, local]))
                    up = Float32(cutlass.BFloat16(staged[32 + col, local]))
                    value = self.activate(gate, up, beta, linear_beta)
                    mH[bidx * 32 + col, token, 0] = cutlass.BFloat16(value)

    @cute.jit
    def activate(self, gate, up, beta, linear_beta):
        return situ(gate, up, beta, linear_beta)


class SplitKSiluMul(SplitKSitu):
    """``SplitKSitu`` with vLLM's silu_and_mul (``beta`` / ``linear_beta`` unused)."""

    @cute.jit
    def activate(self, gate, up, beta, linear_beta):
        one = Float32(1)
        silu = div_rn(gate, one + cute.math.exp(-gate, fastmath=False))
        return Float32(cutlass.BFloat16(silu)) * (up + Float32(0))


@cache
def compile_situ(tactic, half, arch, silu=False):
    """Code-only cache, as ``projection.compile_projection`` (``silu``: SiluMul)."""

    def fake(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kind = SplitKSiluMul if silu else SplitKSitu
    kernel = kind(tactic=SplitKTactic(*tactic), half=half)
    args = (fake(1), fake(1), fake(0), cutlass.Int64(0), Float32(1), Float32(0))
    args += (make_fake_stream(),)
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class PreparedSitu:
    """Gate/up projection + SiTU (``silu``: silu_and_mul) per batch tactic (M tile 64,
    or with ``silu`` the 128-row tensor-memory tactics of 33-64 rows:
    ``splitk_situ64``); ``run`` writes h."""

    def __init__(self, gate_up, policies, beta, linear_beta, silu=False):
        self.weight, self.device = gate_up, gate_up.device
        self.half = gate_up.shape[0] // 2
        self.policies = dict(policies)
        self.scalars = (Float32(beta), Float32(linear_beta))
        self.weight_view = matrix_view(gate_up)
        self.map = weight_map(gate_up, 32)  # {64, 32} boxes: gate and up halves
        self.extra = (cutlass.Int64(self.map.data_ptr()),)
        self.map64 = self.extra64 = None
        arch = architecture(self.device)
        self.kernels = {}
        with torch.accelerator.device_index(self.device.index):
            for b, t in self.policies.items():
                if len(t) > 4:  # SplitKTmemSiluMul64: {64, 64} boxes
                    from .splitk_situ64 import compile_situ64

                    assert silu, "the tensor-memory gate/up has silu_and_mul only"
                    if self.map64 is None:
                        self.map64 = weight_map(gate_up, 64)
                        self.extra64 = (cutlass.Int64(self.map64.data_ptr()),)
                    k = gate_up.shape[1]
                    self.kernels[b] = compile_situ64(t, k, self.half, arch)
                else:
                    self.kernels[b] = compile_situ(t, self.half, arch, silu)

    def run(self, x, h):
        rows = x.shape[0]
        code = self.kernels[rows]
        out = matrix_view(h, output=True)
        stream = current_stream(self.device)
        if len(self.policies[rows]) > 4:
            code(self.weight_view, matrix_view(x), out, out, *self.extra64, stream)
        else:
            args = (*self.extra, *self.scalars, stream)
            code(self.weight_view, matrix_view(x), out, *args)
        return h
