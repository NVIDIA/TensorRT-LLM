# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rubin variants of ``SplitKGemm`` (the dense first layer's MLP).

``FitSplitKGemm`` launches with the dynamic shared memory its tiles use instead of the
full 227 KiB (same code). ``RingSplitKGemm``'s output is this rank's one-shot all-reduce
contribution.

Instead of storing C, the owner CTA stages its FP32 tile in shared memory, rounds it
to BF16 once (as ``SplitKGemm`` does; negative zeros made positive, the Lamport
sentinel) and multicasts 16-byte rows fragments into slot ``rank`` of every TP rank's
[TP, cap, width] mailbox. The consumer (``ClusterAttnRes`` with ``slots`` = TP) polls
the TP slots, sums them in FP32 in rank order, rounds once and re-arms them. Used by
the Rubin dense MLP down projection; GB300 keeps ``SplitKGemm`` + the framework
all-reduce."""

from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int64
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import _smem_bytes

from ..common.host import architecture
from ..common.host import stream as current_stream
from ..common.primitives import load_words, multicast_words, pack_pair
from .projection import PreparedProjection, matrix_view
from .splitk import OWNER_RANK, SplitKGemm, SplitKTactic, _smem


def fit_smem(tactic):
    """Shared memory the vendored kernel uses (FlashInfer's layout mirror) plus 1 KiB
    for sB's 1024 B alignment and 1 KiB for the dynamic-smem base alignment."""
    return _smem_bytes(tactic, tactic.ab_stages) + 2048


class FitSplitKGemm(SplitKGemm):
    """``SplitKGemm`` launched with ``fit_smem`` bytes of dynamic shared memory."""

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
            smem=cute.Int64(self.fit),
            stream=stream,
            use_pdl=True,
        )


class RingSplitKGemm(SplitKGemm):
    """``ring``: the multicast address of this rank's slot [cap, width] (BF16)."""

    def __init__(self, *, tactic: SplitKTactic, trigger_after_wait: bool = False):
        super().__init__(tactic=tactic)
        assert self.cta_n * self.cta_m // 8 <= self.epilog_threads, "one fragment each"
        # ``trigger_after_wait``: the dependents (a deferred site that reads older
        # buffers before its own PDL wait) launch only once this grid is past its
        # wait, i.e. once every kernel before it has completed.
        self.trigger_after_wait = trigger_after_wait

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, ring: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, bias, a_map, ring).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
        a_map: cutlass.Int64, ring: cutlass.Int64,
    ):  # fmt: skip
        """``SplitKGemm.kernel`` with the epilogue tile in shared memory, then the
        ring publish."""
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
        bar_tma_epilog = _smem(cutlass.Int64, 1, 8).iterator  # FlashInfer's bias
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        # The owner's FP32 tile, M-major like C.
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        staged = _smem(Float32, None, 16, tile_layout)
        mailbox, bar_reduce = sA, bar_mma_epilog  # split 1: compile-time dead
        if cutlass.const_expr(self.split_k > 1):
            mailbox = _smem(cutlass.Float32, self.mailbox_elements, 128)
            bar_reduce = _smem(cutlass.Int64, 1, 8).iterator

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
            self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m, *tiles)
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
            epi_tid = cute.arch.thread_idx()[0] - 128
            self.epilog_warp(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, staged,
                epi_tid, Float32, utils.LayoutEnum.from_tensor(mC), mailbox,
                bar_reduce, split_rank,
            )  # fmt: skip
            if split_rank == OWNER_RANK:
                self.publish(staged, mC, ring, bidx, n_idx, epi_tid)

    @cute.experimental.jit
    def publish(
        self, staged: cute.Tensor, mC: cute.Tensor, ring: cutlass.Int64,
        bidx: cutlass.Int32, n_idx: cutlass.Int32, epi_tid: cutlass.Int32,
    ):  # fmt: skip
        """Each epilogue thread multicasts one 8-column fragment of one token row."""
        cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
        chunks = self.cta_m // 8
        token = epi_tid // chunks
        chunk = epi_tid - token * chunks
        width, rows = mC.layout.shape[0], mC.layout.shape[1]
        n, m = n_idx * self.cta_n + token, bidx * self.cta_m + chunk * 8
        if token < self.cta_n and n < rows and m < width:
            source = staged.iterator + token * self.cta_m + chunk * 8
            low = load_words(source, count=4, shared=True, result_type=Float32,
                             word_type="f32")  # fmt: skip
            high = load_words(source + 4, count=4, shared=True, result_type=Float32,
                              word_type="f32")  # fmt: skip
            values = (*low, *high)
            words = tuple(
                pack_pair(values[2 * j], values[2 * j + 1], policy="latent")
                for j in range(4)
            )
            offset = Int64(n) * Int64(width) + Int64(m)
            multicast_words(ring + offset * 2, words, kind="bf16x2")


def _fake(leading, dtype=cutlass.BFloat16):
    shape = tuple(cute.sym_int() for _ in range(3))
    stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
    return make_fake_tensor(dtype, shape, stride, assumed_align=16)


@cache
def compile_fit_projection(tactic, arch):
    split = SplitKTactic(*tactic)
    kernel = FitSplitKGemm(tactic=split)
    kernel.fit = fit_smem(split)
    y = _fake(0)
    args = (_fake(1), _fake(1), y, y, Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


@cache
def compile_ring_projection(tactic, arch):
    """``tactic``: the split-K tactic, then optionally 1 (``trigger_after_wait``)."""
    y = _fake(0)
    after = len(tactic) > 4 and bool(tactic[4])
    kernel = RingSplitKGemm(tactic=SplitKTactic(*tactic[:4]), trigger_after_wait=after)
    args = (_fake(1), _fake(1), y, y, Int64(0), Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


class FitProjection(PreparedProjection):
    """``PreparedProjection`` on ``FitSplitKGemm`` (BF16 output)."""

    def prepare(self):
        arch = architecture(self.device)
        if self.output_dtype != torch.bfloat16:
            raise ValueError("fit projections store BF16 rows")
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in self.policies.items():
                self.kernels[batch] = compile_fit_projection(tactic, arch)
                self.extra[batch] = self.weight_map(tactic[0])
        return self


class RingProjection(PreparedProjection):
    """``PreparedProjection`` whose ``run`` publishes into a one-shot ring slot."""

    def prepare(self):
        arch = architecture(self.device)
        if self.output_dtype != torch.bfloat16:
            raise ValueError("ring projections publish BF16 rows")
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in self.policies.items():
                self.kernels[batch] = compile_ring_projection(tactic, arch)
                self.extra[batch] = self.weight_map(tactic[0])
        return self

    def run_ring(self, x, out, ring):
        """``out`` [B, N] only shapes the grid (it is not written); ``ring`` is the
        multicast address of this rank's slot."""
        y = matrix_view(out, output=True)
        b, stream = x.shape[0], current_stream(self.device)
        if b not in self.kernels or x.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f"ring projection input {tuple(x.shape)}: prepared batches "
                f"{sorted(self.kernels)}, K {self.weight.shape[1]}"
            )
        code, extra = self.kernels[b], self.extra[b]
        code(self.weight_view, matrix_view(x), y, y, *extra, Int64(ring), stream)
        return out
