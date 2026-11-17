# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
CUTE DSL implementation of the TGV GEMM kernel.

This is a low-latency Blackwell GEMM kernel: C = A * B
- A (M, K, L) with K contiguous (UmmaMajor::K)
- B (N, K, L) with K contiguous
- C (M, N, L) with M contiguous

Features:
- TMA loads for A and B matrices (GMEM -> SMEM)
- Matrix multiply-accumulate in tensor memory
- Multi-stage pipeline for overlapping TMA and MMA
- Non-TMA-store epilogue (TMEM -> RMEM -> GMEM with direct store)
- 1 SM mode (no 2CTA instructions)
- 1x1 cluster (no multicast)
- PDL (Programmatic Dependent Launch) support

Default config: CTA_M=64, CTA_N=8, CTA_K=128, DMA_Stage=8
  TypeA=bf16, TypeB=bf16, TypeC=bf16, AccType=float
  UmmaMajorA=Major::K, UmmaMajorB=Major::K

Warp assignment (256 threads, 8 warps):
  Warp 0: DMA_A - loads A tiles via TMA
  Warp 1: DMA_B - loads B tiles via TMA
  Warp 2: MMA - performs matrix multiply-accumulate
  Warp 3: unused
  Warps 4-7: EPILOG - TMEM->RMEM->GMEM store
"""

from functools import lru_cache
from typing import NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.testing
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes.
    For non-persistent static scheduler, CTA id is the work tile info."""

    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32  # kblock range [K_idx_start, K_idx_end)
    K_idx_end: cutlass.Int32


class TgvGemmKernel:
    """
    Low-latency Blackwell GEMM kernel.

    Uses staged barriers for synchronization with 8-warp specialization:
    warp 0 = DMA_A, warp 1 = DMA_B, warp 2 = MMA, warps 4-7 = EPILOG.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = 128,
        num_ab_stage: int = 8,
        use_pdl: bool = False,
        pdl_count: int = -1,
        split_k: int = 1,
        use_bias: bool = False,
    ):
        self.acc_dtype = acc_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_pdl = use_pdl
        self.pdl_count = pdl_count
        self.split_k = split_k
        self.use_bias = use_bias

        # The distributed shared memory mailbox transports each accumulator
        # value in Float32 for split-K reduction.
        self._mailbox_elems_per_thread = (cta_m * cta_n) // 128
        # distributed reduction: each CTA reduces 1/split_k of the output tile
        # shard size in elements per thread (must divide evenly)
        self._shard_elems_per_thread = self._mailbox_elems_per_thread // max(split_k, 1)
        if split_k > 1:
            assert self._mailbox_elems_per_thread % split_k == 0, (
                f"elems_per_thread={self._mailbox_elems_per_thread} must be divisible by split_k={split_k}"
            )
        # per-CTA mailbox: receives (split_k-1) shards from peer CTAs
        self._mailbox_total_elems = max(split_k - 1, 0) * 128 * self._shard_elems_per_thread
        # TX bytes: each peer sends shard_elems_per_thread * 128 * 4B to this CTA
        self._mailbox_tx_per_sender = 128 * self._shard_elems_per_thread * 4
        self._mailbox_tx_total = max(split_k - 1, 0) * self._mailbox_tx_per_sender

        # Fixed configuration
        self.threads_per_cta = 256  # 8 warps
        self.use_2cta_instrs = False  # 1 SM mode
        self.cluster_shape_mn = (1, 1)  # No multicast, 1x1 cluster
        self.cta_group = tcgen05.CtaGroup.ONE

    @staticmethod
    def can_implement(
        problem_mnkl: tuple[int, int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler_mnk: tuple[int, int, int] = (64, 8, 128),
        num_ab_stage: int = 8,
        split_k: int = 1,
    ) -> bool:
        """Return whether the kernel supports the problem and configuration."""
        supported_dtypes = {
            (
                cutlass.Float16,
                cutlass.Float16,
                cutlass.Float16,
                cutlass.Float16,
            ),
            (
                cutlass.Float16,
                cutlass.Float16,
                cutlass.Float16,
                cutlass.Float32,
            ),
            (
                cutlass.BFloat16,
                cutlass.BFloat16,
                cutlass.BFloat16,
                cutlass.Float32,
            ),
        }
        if (a_dtype, b_dtype, c_dtype, acc_dtype) not in supported_dtypes:
            return False

        m, n, k, batch = problem_mnkl
        if min(m, n, k, batch) <= 0:
            return False

        cta_m, cta_n, cta_k = mma_tiler_mnk
        if (cta_m, cta_n) != (64, 8) or cta_k <= 0 or k % cta_k != 0:
            return False
        if num_ab_stage <= 0 or split_k <= 0:
            return False

        output_values_per_thread = (cta_m * cta_n) // 128
        return cta_m * cta_n % 128 == 0 and output_values_per_thread % split_k == 0

    def _setup_attributes(self):
        """Set up derived config."""
        mma_tiler_mn = (self.cta_m, self.cta_n)
        # tiled_mma just contains a single mma atom/instruction, not stacking anything
        # For 1SM mode, tiled_mma.thr_id has shape (_1,), meaning 1 "thread" in the MMA partition
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            mma_tiler_mn,
        )

        # Derive mma_inst_tile_k = NumMma_K = CTA_K / Mma_K
        # For bf16: Mma_K = 16, so CTA_K=128 -> mma_inst_tile_k = 8
        # This is the number of MMA instructions along the K dimension per CTA_K tile
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        # Maps a CTA's linear rank within its cluster to 4D (V, M, N, K) coordinates.
        # Used by TMA partition to determine which CTAs share data for multicast:
        #   V = CTA index within one tiled_mma (2SM: V=2, 1SM: V=1)
        #   M, N = CTA position along cluster M/N dimensions
        #   K = always 1 (no cluster tiling along K)
        # A is multicast along N (CTAs with same M share A tiles) →
        #   tma_partition uses the N-slice and block_in_cluster_coord_vmnk[2]
        # B is multicast along M (CTAs with same N share B tiles) →
        #   tma_partition uses the M-slice and block_in_cluster_coord_vmnk[1]
        # For 1SM + 1x1xS cluster: K dimension = split_k, CTAs along K share M/N tile.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, self.split_k)),
            (tiled_mma.thr_id.shape,),
        )

        # Create SMEM layouts for A and B with swizzle
        # For bf16 K-major: swizzle atom is Sw<3,4,3> with shape (M8, K64) = 8x128B
        # The atom is stacked/replicated to fill ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        # sA_layout: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.a_dtype,
            self.num_ab_stage,
        )
        # sB_layout: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.b_dtype,
            self.num_ab_stage,
        )

        # Accumulator shape for TMEM allocation size calculation
        # acc_shape: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = cutlass.memory.get_num_tmem_alloc_cols(tCtAcc_fake)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K or M contiguous
        b: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K or N contiguous
        c: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M contiguous
        bias: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch."""
        assert self.use_bias == (bias is not None)
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.use_bias):
            assert bias is not None
            self.bias_dtype = bias.element_type
        else:
            assert bias is None
            self.bias_dtype = self.c_dtype

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)  # 1 for 1SM

        # Setup TMA load for A
        # tma_tensor_a is the TMA coordinate tensor: (Gemm_M, Gemm_K, Gemm_L)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        # a_smem_layout: ((Mma_M, Mma_K), NumMma_M, NumMma_K) — 1 stage slice
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        # tma_tensor_b is the TMA coordinate tensor: (Gemm_N, Gemm_K, Gemm_L)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        # b_smem_layout: ((Mma_N, Mma_K), NumMma_N, NumMma_K) — 1 stage slice
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Calculate total bytes that TMA will transfer per stage to track completion
        # A and B have separate barriers, so we track bytes separately
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.tma_bytes_a = a_copy_size * atom_thr_size
        self.tma_bytes_b = b_copy_size * atom_thr_size

        # Grid: dim3{ceil_div(M, CTA_M), ceil_div(N, CTA_N), L * split_k}
        # Each CTA processes one (CTA_M, CTA_N) output tile (split across K)
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_m),
            cute.ceil_div(c.layout.shape[1], self.cta_n),
            c.layout.shape[2] * self.split_k,
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,  # tma_tensor_a is mA_mkl: (Gemm_M, Gemm_K, Gemm_L)
            tma_atom_b,
            tma_tensor_b,  # tma_tensor_b is mB_nkl: (Gemm_N, Gemm_K, Gemm_L)
            c,  # mC_mnl: (Gemm_M, Gemm_N, Gemm_L)
            bias,  # mBias_mnl: (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
            self.b_smem_layout_staged,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],  # 256 threads = 8 warps
            cluster=(*self.cluster_shape_mn, self.split_k),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor for A
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor for B
        mC_mnl: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        mBias_mnl: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        b_smem_layout_staged: cute.ComposedLayout,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
    ):
        """
        GPU device kernel.

        Slim kernel: only SMEM alloc + barrier init + warp dispatch.
        All tensor partitioning is done inside warp functions.

        256 threads, 8 warps:
          Warp 0: DMA_A  - loads A tiles via TMA
          Warp 1: DMA_B  - loads B tiles via TMA
          Warp 2: MMA    - performs matrix multiply-accumulate
          Warp 3: unused
          Warps 4-7: EPILOG - TMEM->RMEM->GMEM store
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        # Prefetch TMA descriptors (warp 0 only)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # ============================================================
        # CGA split-k: compute split_rank and l_idx from cluster topology
        # ============================================================
        _, _, split_rank = cute.arch.block_in_cluster_idx()
        _, _, l_idx = cute.arch.cluster_idx()

        # ============================================================
        # WorkTileInfo with per-split K-range
        # ============================================================
        total_k_tiles = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_k)
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        k_tiles_per_split = cute.ceil_div(total_k_tiles, self.split_k)
        k_start = split_rank * k_tiles_per_split
        if k_start > total_k_tiles:
            k_start = total_k_tiles
        k_end = k_start + k_tiles_per_split
        if k_end > total_k_tiles:
            k_end = total_k_tiles
        work_tile_info = WorkTileInfo(
            M_idx=bidx // cute.size(tiled_mma.thr_id.shape),
            N_idx=bidy,
            L_idx=l_idx,
            K_idx_start=k_start,
            K_idx_end=k_end,
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # block_in_cluster_coord_vmnk: this CTA's (V, M, N, K) coordinate within its cluster.
        # Used by tma_partition to set up TMA multicast masks:
        #   [2] (N) → A's tma_partition (A is multicast along N)
        #   [1] (M) → B's tma_partition (B is multicast along M)
        # For 1x1 cluster: always (0, 0, 0, 0), trivially no multicast.
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # ============================================================
        # SMEM allocation (SharedStorage struct for barriers + TMEM ptr)
        # ============================================================
        DMA_Stage = self.num_ab_stage
        MailboxElems = self._mailbox_total_elems

        # Shared storage struct
        # There are 2 kinds of barriers used here:
        #   1. Transaction barriers (tma_mma_full): 64-bit in SMEM, support transaction byte
        #      counting for TMA completion tracking. set_barrier_transaction_bytes arrives and
        #      sets the expected TX count; TMA increments TX bytes as data arrives in SMEM.
        #      When both arrival count and TX count are met, the barrier phase flips.
        #   2. Cluster barriers (all others): 64-bit in SMEM, simpler arrive/wait semantics.
        #      Used for thread-to-thread synchronization within/across warps.
        @cute.struct
        class SharedStorage:
            # Barrier between TMA and MMA: TMA tells MMA the SMEM tile is ready/full
            tma_mma_full_barrier: cute.struct.MemRange[cutlass.Int64, DMA_Stage]
            # Barrier between MMA and TMA: MMA tells TMA the SMEM tile is consumed/empty
            tma_mma_empty_barrier: cute.struct.MemRange[cutlass.Int64, DMA_Stage]
            # Barrier between TMA_B and epilog: TMA_B tells epilog all activation loads are issued
            tma_epilog_full_barrier: cutlass.Int64
            # Barrier between MMA and epilog: MMA tells epilog the accumulator is ready
            mma_epilog_full_barrier: cutlass.Int64
            # Barrier between MMA and epilog: sync TMEM allocation/deallocation status
            tmem_allocation_result_barrier: cutlass.Int64
            # Base pointer for TMEM allocation, MMA will write the allocated address here
            tmem_base: cutlass.Int32
            # DSMEM mailbox: transaction barrier for split-k scatter completion
            dsmem_mailbox_barrier: cutlass.Int64
            # DSMEM mailbox: float32 accumulator partials from (split_k-1) remote CTAs
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, MailboxElems]

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Extract named barrier pointers from struct (DSL can't pass struct objects
        # as @cute.jit function arguments — they fail tree flattening to IR values)
        tma_mma_full_bar = storage.tma_mma_full_barrier.data_ptr()
        tma_mma_empty_bar = storage.tma_mma_empty_barrier.data_ptr()
        tma_epilog_full_bar = storage.tma_epilog_full_barrier.ptr
        mma_epilog_full_bar = storage.mma_epilog_full_barrier.ptr
        tmem_alloc_result_bar = storage.tmem_allocation_result_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base.ptr
        dsmem_mailbox_mbar = storage.dsmem_mailbox_barrier.ptr
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()

        # ============================================================
        # Barrier initialization — ALL threads reach here, elect_one inits
        #
        # Phase initializes to 0, expected arrival count is set below.
        # For transaction barriers (tma_mma_full): set_barrier_transaction_bytes will
        # arrive and set TX count; TMA DRAM requests increment TX bytes. When both
        # arrival count and TX count are met, the barrier phase flips.
        # ============================================================
        if warp_idx == 0:
            with cute.arch.elect_one():
                # Transaction barrier: 2 arrivals (DMA_A + DMA_B each set TX bytes)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(tma_mma_full_bar + i, 2)
                # Cluster barrier: 1 arrival (MMA warp signals SMEM slot consumed)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(tma_mma_empty_bar + i, 1)

                # 32 threads/1 warp (DMA_B warp) arrive to signal epilog
                cute.arch.mbarrier_init(tma_epilog_full_bar, 32)
                # One MMA thread signals the epilogue.
                cute.arch.mbarrier_init(mma_epilog_full_bar, 1)
                # 32 (MMA warp) + 128 (EPILOG warps 4-7) = 160 arrivals for TMEM alloc sync
                cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128)

                # DSMEM mailbox transaction barrier (split_k > 1 only)
                if self.split_k > 1:
                    cute.arch.mbarrier_init(dsmem_mailbox_mbar, 1)

        # Barrier initialization needs to be visible to all warps before proceeding
        cute.arch.mbarrier_init_fence()
        if self.split_k > 1:
            # Make every CTA's barrier initialization visible across the cluster.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ============================================================
        # SMEM Tensor Allocation
        # ============================================================
        # sA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) with swizzle
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # sB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) with swizzle
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        # ============================================================
        # Warp dispatch
        # Each warp specializes in a different role:
        #   Warp 0 (threads 0-31):    DMA_A — loads A tiles via TMA
        #   Warp 1 (threads 32-63):   DMA_B — loads B tiles via TMA
        #   Warp 2 (threads 64-95):   MMA   — performs matrix multiply-accumulate
        #   Warp 3 (threads 96-127):  unused
        #   Warps 4-7 (threads 128-255): EPILOG — TMEM->RMEM->GMEM store
        # ============================================================
        if warp_idx == 0:
            self.dma_a_warp(
                tma_mma_full_bar,
                tma_mma_empty_bar,
                tma_atom_a,
                mA_mkl,
                sA,
                tiled_mma,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_info,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                tma_mma_full_bar,
                tma_mma_empty_bar,
                tma_epilog_full_bar,
                tma_atom_b,
                mB_nkl,
                sB,
                tiled_mma,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_info,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                tma_mma_full_bar,
                tma_mma_empty_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                tiled_mma,
                sA,
                sB,
                tmem_base_smem_ptr,
                mma_tile_coord_v,
                k_tile_count,
            )
        elif warp_idx >= 4:
            # Epilog tid is from 128 to 255, need to offset by -128
            # when getting the per-thread slice of the tiled_copy_t2r
            epi_tid = tidx - 128
            self.epilog_warp(
                tma_epilog_full_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                tiled_mma,
                mC_mnl,
                mBias_mnl,
                tmem_base_smem_ptr,
                dsmem_mailbox_mbar,
                dsmem_mailbox_ptr,
                epi_tid,
                mma_tile_coord_v,
                work_tile_info,
                split_rank,
                k_tile_count,
            )

        # Final sync
        cute.arch.barrier()
        return

    @cute.jit
    def dma_a_warp(
        self,
        tma_mma_full_bar,
        tma_mma_empty_bar,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: Tuple,  # WorkTileInfo: (M_idx, N_idx, L_idx, K_idx_start, K_idx_end)
        k_tile_count: cutlass.Int32,
    ):
        """DMA_A warp: loads A tiles via TMA."""
        DMA_Stage = self.num_ab_stage

        # ---- Self-contained tensor partitioning ----
        # Partitioned tensors use tXgY naming convention:
        #   tX = partitioning pattern applied to tensor gY
        #   tC = tensor partitioned into the MMA shape, i.e. ((Mma_M, Mma_K), ...)
        #   tA = tensor partitioned into the TMA shape, i.e. (TMA, ...)
        #   g = gmem, s = smem, t = tmem, r = rmem

        # Tile mA_mkl (Gemm_M, Gemm_K, Gemm_L) into CTA-level tiles
        # gA_mkl: (CTA_M, CTA_K, Tiles_M, Tiles_K, Gemm_L) — all tiles
        gA_mkl = cute.local_tile(mA_mkl, (self.cta_m, self.cta_k), (None, None, None))
        # Partition gmem tensor into the MMA shape
        # In 1SM mode, tiled_mma has 1 "thread", so get_slice(0) gives the only partition
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, Tiles_M, Tiles_K, Gemm_L)
        tCgA = thr_mma.partition_A(gA_mkl)

        # ---- TMA partition for cluster-level multicast ----
        # tma_partition sets up TMA descriptors that can multicast data to multiple CTAs
        # in the same cluster that share the same tile. It takes 3 key arguments:
        #   1. This CTA's coordinate within the multicast group
        #   2. The multicast group layout (how many CTAs participate)
        #   3. The SMEM/GMEM tensors to partition
        #
        # For A: all CTAs along the N dimension of the cluster share the same A tile
        # (same M, same K, different N). So the multicast group is the N-slice of the
        # cluster layout. For a 2x3 cluster, 3 CTAs along N would share each A tile.
        #
        # a_cta_layout: layout of CTAs that share A data = N-dimension of cluster.
        #   Extracted by slicing cluster_layout_vmnk at V=0, M=0, K=0, keeping N free.
        #   For 1x1 cluster: shape (_1,) — just this CTA, no multicast.
        # block_in_cluster_coord_vmnk[2]: this CTA's position within that N multicast group.
        #   For 1x1 cluster: always 0.
        #
        # group_modes groups ((Mma_M, Mma_K), NumMma_M, NumMma_K) into a single mode
        # so the TMA handles the entire SMEM tile in one copy call.
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],  # this CTA's N coord within cluster
            a_cta_layout,
            cute.group_modes(sA, 0, 3),  # (((Mma_M, Mma_K), NumMma_M, NumMma_K), DMA_Stage)
            cute.group_modes(
                tCgA, 0, 3
            ),  # (((Mma_M, Mma_K), NumMma_M, NumMma_K), Tiles_M, Tiles_K, Gemm_L)
        )
        # tAsA: ((TMA, NumTma_K), DMA_Stage) — SMEM destination for each pipeline stage
        # tAgA: ((TMA, NumTma_K), Tiles_M, Tiles_K, Gemm_L) — GMEM source tiles

        # Slice to this CTA's M tile and batch index, keep K tiles and TMA modes free
        # tAgA after slice: ((TMA, NumTma_K), Tiles_K)
        tAgA = tAgA[(None, work_tile_info.M_idx, None, work_tile_info.L_idx)]

        # ---- K-loop ----
        # The initial empty-barrier phase denotes that shared storage is available.
        # We wait on the "old" phase bit of 1: when it flips to 0 (the initial value), the slot is empty.
        # Example for DMA_Stage=2, barriers a and b:
        #   kblock 0: old phase=1, wait for a to flip to 0 (initial), slot a empty
        #   kblock 1: old phase=1, wait for b to flip to 0 (initial), slot b empty
        #   kblock 2: old phase=0, wait for a to flip to 1 (flipped once), slot a empty
        #   kblock 3: old phase=0, wait for b to flip to 1 (flipped once), slot b empty
        #   ...
        tma_mma_empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        # Iterate over k-blocks
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            # Wait for MMA to signal that SMEM slot is empty, ready for next TMA load
            cute.arch.mbarrier_wait(
                tma_mma_empty_bar + stage,
                tma_mma_empty_phase,
            )

            # Set the barrier transaction bytes for tracking TMA completion
            # Must be done by a single thread (elect_one)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    tma_mma_full_bar + stage,
                    self.tma_bytes_a,
                )
            # Issue TMA load: load A tile (CTA_M, CTA_K) from GMEM into SMEM
            # The TMA partition elects its issuing thread. A second election
            # would prevent the participating threads from making progress.
            # for split-k: offset k_tile by K_idx_start to load correct K-range
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile_global)],  # GMEM source: ((TMA, NumTma_K),) for k_tile
                tAsA[(None, stage)],  # SMEM destination: ((TMA, NumTma_K),) for stage
                tma_bar_ptr=tma_mma_full_bar + stage,
            )

            # For every DMA_Stage iterations, flip the phase bit to reuse barriers
            if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                tma_mma_empty_phase = tma_mma_empty_phase ^ 1

            # PDL: do preexit at a specific k_tile count
            if self.use_pdl:
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        # PDL: if pdl_count is -1, do preexit at the end
        if self.use_pdl:
            cute.arch.griddepcontrol_launch_dependents()

        # Producer tail: keep shared memory alive until MMA releases every stage.
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(tma_mma_empty_bar + stage, tma_mma_empty_phase)
            if stage == (DMA_Stage - 1):
                tma_mma_empty_phase = tma_mma_empty_phase ^ 1

    @cute.jit
    def dma_b_warp(
        self,
        tma_mma_full_bar,
        tma_mma_empty_bar,
        tma_epilog_full_bar,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: Tuple,  # WorkTileInfo: (M_idx, N_idx, L_idx, K_idx_start, K_idx_end)
        k_tile_count: cutlass.Int32,
    ):
        """DMA_B warp: loads B tiles via TMA."""
        DMA_Stage = self.num_ab_stage

        # ---- Self-contained tensor partitioning (similar to DMA_A_warp) ----
        # Tile mB_nkl (Gemm_N, Gemm_K, Gemm_L) into CTA-level tiles
        # gB_nkl: (CTA_N, CTA_K, Tiles_N, Tiles_K, Gemm_L)
        gB_nkl = cute.local_tile(mB_nkl, (self.cta_n, self.cta_k), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, Tiles_N, Tiles_K, Gemm_L)
        tCgB = thr_mma.partition_B(gB_nkl)

        # Same pattern as A, but for B the multicast group is along M.
        # All CTAs along M share the same B tile (same N, same K, different M).
        # b_cta_layout: layout of CTAs that share B data = M-dimension of cluster.
        #   Extracted by slicing cluster_layout_vmnk at V=0, N=0, K=0, keeping M free.
        #   For 1x1 cluster: shape (_1,) — just this CTA, no multicast.
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],  # this CTA's M coord within cluster
            b_cta_layout,
            cute.group_modes(sB, 0, 3),  # (((Mma_N, Mma_K), NumMma_N, NumMma_K), DMA_Stage)
            cute.group_modes(
                tCgB, 0, 3
            ),  # (((Mma_N, Mma_K), NumMma_N, NumMma_K), Tiles_N, Tiles_K, Gemm_L)
        )
        # tBsB: ((TMA, NumTma_K), DMA_Stage)
        # tBgB: ((TMA, NumTma_K), Tiles_N, Tiles_K, Gemm_L)

        # Slice to this CTA's N tile and batch index
        # tBgB after slice: ((TMA, NumTma_K), Tiles_K)
        tBgB = tBgB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]

        # PDL: wait on the predecessor grid (griddepcontrol.wait) only before loading B,
        # B is activation
        if self.use_pdl:
            cute.arch.griddepcontrol_wait()

        # ---- K-loop (same phase tracking as DMA_A_warp) ----
        tma_mma_empty_phase = cutlass.Int32(1)

        # Iterate over k-blocks
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            # Wait for MMA to signal SMEM slot is empty
            cute.arch.mbarrier_wait(
                tma_mma_empty_bar + stage,
                tma_mma_empty_phase,
            )

            # Set transaction bytes and issue TMA load for B tile (CTA_N, CTA_K)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    tma_mma_full_bar + stage,
                    self.tma_bytes_b,
                )
            # CRITICAL: cute.copy for TMA must be OUTSIDE elect_one()
            # for split-k: offset k_tile by K_idx_start to load correct K-range
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile_global)],  # GMEM source: ((TMA, NumTma_K),) for k_tile
                tBsB[(None, stage)],  # SMEM destination: ((TMA, NumTma_K),) for stage
                tma_bar_ptr=tma_mma_full_bar + stage,
            )

            # Flip phase bit every DMA_Stage iterations
            if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                tma_mma_empty_phase = tma_mma_empty_phase ^ 1

        # All activation/B loads are issued, signal the epilog warp
        cute.arch.mbarrier_arrive(tma_epilog_full_bar)

        # Producer tail: keep shared memory alive until MMA releases every stage.
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(tma_mma_empty_bar + stage, tma_mma_empty_phase)
            if stage == (DMA_Stage - 1):
                tma_mma_empty_phase = tma_mma_empty_phase ^ 1

    @cute.jit
    def mma_warp(
        self,
        tma_mma_full_bar,
        tma_mma_empty_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tmem_base_smem_ptr,
        mma_tile_coord_v: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        """Perform matrix multiply-accumulate for the assigned K tiles."""
        DMA_Stage = self.num_ab_stage

        # ---- MMA Fragment Allocation ----
        # We allocate "fragments" which are SMEM descriptors that serve as inputs to cute.gemm.
        # The MMA operands use shared memory descriptor fragments:
        #   - Matrices A and B are sourced from SMEM
        #   - tCrA and tCrB provide descriptor views of sA and sB respectively
        #   - The first mode of each descriptor represents the SMEM for a single MMA instruction
        # tCrA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — SMEM descriptors for A
        tCrA = tiled_mma.make_fragment_A(sA)
        # tCrB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — SMEM descriptors for B
        tCrB = tiled_mma.make_fragment_B(sB)

        # TMEM Accumulator shape
        # On SM100, accumulators are stored exclusively in tensor memory (TMEM).
        # make_fragment_C creates a TMEM tensor view with the appropriate layout.
        # tCtAcc is a view of the accumulator tensor; its tmem base ptr is unset until alloc_tmem.
        # acc_shape / tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        # For bf16 M64 N8 K16: (((_16,_4),_8),_1,_1) — Mma_M=(16,4)=(Mma_M_per_subp, NumSubp), Mma_N=8
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # ---- TMEM Allocation ----
        # Only use half of TMEM to allow overlapping with next CTA
        # TMEM has 128 lanes, 512 columns, each word is 4B, 256KB total
        # Our accumulator uses CTA_N * sizeof(AccType) columns
        num_tmem_cols = 256  # SM100_TMEM_CAPACITY_COLUMNS / 2
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_smem_ptr)

        # Notify epilog warp that TMEM allocation is complete
        # 32 threads (MMA warp) arrive; NO WAIT here — just arrives and continues
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # Relinquish TMEM allocation lock early so that the next prefetch CTA can be launched
        cute.arch.relinquish_tmem_alloc_permit()

        # Retrieve TMEM ptr from SMEM and create the accumulator tensor view
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        # tCtAcc: (((Mma_M_per_subp, NumSubp), Mma_N), NumMma_M, NumMma_N) — TMEM accumulator view
        # For bf16 M64 N8: (((16,4),8),1,1):(((65536,2097152),1),0,0)
        # N-major (N stride=1); dp lane stride=65536, subpartition stride=2097152
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # ---- MMA mainloop with try_wait/wait overlap ----
        # This follows the CUTLASS pattern to overlap barrier waits with MMA instructions:
        #   for each kblock:
        #     if not waitComplete: blocking wait
        #     update stage_idx
        #     try_wait for next iteration (non-blocking, returns predicate)
        #     execute MMA
        # The nonblocking wait allows independent MMA setup to proceed while the
        # next shared-memory stage becomes available.
        #
        # Initial phase bit for tma_mma_full_barrier is 0:
        #   kblock 0: old phase=0, wait for phase to flip to 1 (slot full)
        #   kblock 1: old phase=0, wait for phase to flip to 1 (slot full)
        #   ... after DMA_Stage iterations, phase flips
        tma_mma_full_phase = cutlass.Int32(0)
        # Explicitly manage stage_idx instead of k_tile % DMA_Stage to avoid
        # expensive modulo when DMA_Stage is non-power-of-2 (tighter mainloop)
        stage_idx = cutlass.Int32(0)
        old_stage_idx = cutlass.Int32(0)

        # Peel off the first iteration: try_wait for the first stage (non-blocking)
        waitComplete = cute.arch.mbarrier_try_wait(tma_mma_full_bar + stage_idx, tma_mma_full_phase)

        # Iterate over k-blocks
        for k_tile in cutlass.range(k_tile_count):
            # If try_wait failed, do a blocking wait
            if ~waitComplete:
                cute.arch.mbarrier_wait(tma_mma_full_bar + stage_idx, tma_mma_full_phase)

            # Update stage index
            old_stage_idx = stage_idx
            stage_idx = stage_idx + 1
            # Flip phase bit every DMA_Stage iterations
            if stage_idx == DMA_Stage:
                tma_mma_full_phase = tma_mma_full_phase ^ 1
                stage_idx = cutlass.Int32(0)

            # Try wait for the next iteration (non-blocking, overlaps with MMA below)
            if k_tile < (k_tile_count - 1):
                waitComplete = cute.arch.mbarrier_try_wait(
                    tma_mma_full_bar + stage_idx, tma_mma_full_phase
                )

            # Manually unroll the NumMma_K loop so that each cute.gemm issues exactly
            # one MMA instruction, and we can control the ACCUMULATE flag per instruction.
            # First instruction of first k_tile: ACCUMULATE=False (C = A*B, clears TMEM)
            # All subsequent instructions: ACCUMULATE=True (C += A*B)
            #        cute::gemm(tiled_mma, tCrA(_,_,k_block,stage), tCrB(_,_,k_block,stage), tCtAcc)
            for k_block in range(self.mma_inst_tile_k):
                if k_block == 0:
                    # First k_block of each k_tile: accumulate only if not the very first k_tile
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    # Subsequent k_blocks always accumulate (C += A*B)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # Execute a single Mma_M x Mma_N x Mma_K GEMM instruction
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[(None, None, k_block, old_stage_idx)],
                    tCrB[(None, None, k_block, old_stage_idx)],
                    tCtAcc,
                )

            # Release the consumed shared-memory stage. Exactly one MMA thread
            # signals completion for the warp.
            with cute.arch.elect_one():
                tcgen05.commit(
                    tma_mma_empty_bar + old_stage_idx,
                    None,
                    self.cta_group,
                )

        # Notify the epilog warp that MMA is done, the TMEM accumulator is ready to consume
        with cute.arch.elect_one():
            tcgen05.commit(
                mma_epilog_full_bar,
                None,
                self.cta_group,
            )

        # ---- TMEM dealloc protocol ----
        # Wait for tmem deallocation signal from epilog warp
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        # Initial phase bit = 1 since it's already flipped once for TMEM allocation
        # It will flip to 0 when TMEM can be deallocated, so we wait for old phase of 1
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 1)
        # Deallocate TMEM
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols)

    @cute.jit
    def epilog_warp(
        self,
        tma_epilog_full_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        tiled_mma: cute.TiledMma,
        mC_mnl: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        mBias_mnl: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        tmem_base_smem_ptr,
        dsmem_mailbox_mbar,  # transaction barrier for DSMEM mailbox
        dsmem_mailbox_ptr,  # SMEM pointer for DSMEM stores and reads
        epi_tid: cutlass.Int32,  # thread id within epilog warps (0-127)
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: Tuple,  # WorkTileInfo: (M_idx, N_idx, L_idx, K_idx_start, K_idx_end)
        split_rank: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        """EPILOG warp: TMEM -> RMEM -> type convert -> GMEM."""

        # ---- Self-contained tensor partitioning ----
        # Get the local tile of C for this CTA
        # gC_mnl: (CTA_M, CTA_N, Tiles_M, Tiles_N, Gemm_L)
        gC_mnl = cute.local_tile(mC_mnl, (self.cta_m, self.cta_n), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Since tCtAcc is a view of the accumulator tensor, it's safe to create a new view
        # in the epilog warp too. The layout is the same as in the MMA warp.
        # acc_shape / tCtAcc_fake: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # ---- Arrive + wait tmem_allocation_result_barrier phase 0 ----
        # 128 threads from EPILOG warps (4-7) arrive
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        # Wait for MMA warp to complete TMEM allocation
        # Initial phase=0, it will flip to 1 when TMEM is allocated, so we wait for old phase 0
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        # Update TMEM base ptr of the accumulator tensor view
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        # tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N) — TMEM accumulator view
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # ---- Create tiled copy for TMEM -> RMEM ----
        # Select the tensor-to-register copy shape for this CTA and output type.
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            (self.cta_m, self.cta_n),
            self.use_2cta_instrs,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[((None, None), 0, 0)])
        # Epilog tid is 0-127 (threads 128-255 offset by -128)
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # tD describes the per-thread tensor-to-register partition.
        # tDtAcc: per-subpartition view of the accumulator tensor (TMEM source)
        # Shape: (CpyS, NumCpy_M, NumCpy_N)
        # For bf16 M64 N8, CpyS = (CpyS_N, CpyS_M) = (8, 16).
        #   NumCpy_M = 1, NumCpy_N = CTA_N / CpyS_N = 1
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc[((None, None), 0, 0)])

        # Partition tCgC for 2 reasons:
        # 1. Get post-partition shape for allocating rmem space for the accumulator
        # 2. Partition it for storing the result back to gmem (per-thread slice)
        # tDgC represents which values in tCgC are stored in this thread's rmem
        # after copying the accumulator from tensor memory to registers
        # Shape: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # For bf16 M64 N8, CpyD = (2,2), or four registers per thread.
        # DSL tCgC has extra grid dims; partition with them, create rmem before slicing
        tDgC = thr_copy_t2r.partition_D(tCgC[((None, None), 0, 0, None, None, None)])

        # Allocate per-thread rmem space for the accumulator
        # tDrAcc and tDrC have the same shape as the single-tile slice of tDgC
        # Shape: (CpyD, NumCpy_M, NumCpy_N)
        # IMPORTANT: Create rmem tensors BEFORE slicing tDgC with work_tile_info,
        # because slicing reduces rank and breaks shape extraction
        epi_frag_shape = tDgC[(None, None, None, 0, 0, 0)].shape
        # tDrAcc: (CpyD, NumCpy_M, NumCpy_N) — per-thread accumulator in rmem (AccType)
        tDrAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)
        # tDrC: (CpyD, NumCpy_M, NumCpy_N) — per-thread converted output in rmem (TypeC)
        tDrC = cute.make_rmem_tensor(epi_frag_shape, self.c_dtype)

        # Slice gmem partition to this CTA's output tile
        # tDgC after slice: (CpyD, NumCpy_M, NumCpy_N) — per-thread GMEM store destinations
        tDgC = tDgC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]

        # ---- Bounds-check predicate ----
        # An identity tensor maps (m, n, l) coordinates to the payload tuple (m, n, l).
        # We use this to check if each thread's store coordinates are within bounds,
        # because the problem shape may not be a multiple of the tile size.
        # For TMA loads, OOB is handled automatically; for direct stores, we need explicit predicates.
        coordC = cute.make_identity_tensor(mC_mnl.shape)  # (M, N, L) -> (m, n, l)
        # Create the local tile of coordC, same tiling as gC
        gCcoord = cute.local_tile(coordC, (self.cta_m, self.cta_n), (None, None, None))
        # tCcC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCcC = thr_mma.partition_C(gCcoord)
        # tDcC: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # — per-thread coordinate tensor, same shape as tDgC but payload is (m,n,l) coord
        tDcC = thr_copy_t2r.partition_D(tCcC[((None, None), 0, 0, None, None, None)])
        # Slice to this CTA's tile: (CpyD, NumCpy_M, NumCpy_N)
        tDcC = tDcC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]
        # Construct predicate tensor: compare each coordinate with problem shape (M, N, L)
        # tDpC(t) = true if coordinate is in-bounds, false if out-of-bounds
        tDpC = cute.make_rmem_tensor(tDcC.shape, cutlass.Boolean)  # (CpyD, NumCpy_M, NumCpy_N)
        for i in range(cute.size(tDpC)):
            tDpC[i] = cute.elem_less(tDcC[i], mC_mnl.shape)

        # Partition the M-broadcast bias with C's register mapping.
        if cutlass.const_expr(self.use_bias):
            gBias_mnl = cute.local_tile(mBias_mnl, (self.cta_m, self.cta_n), (None, None, None))
            tCgBias = thr_mma.partition_C(gBias_mnl)
            tDgBias = thr_copy_t2r.partition_D(tCgBias[((None, None), 0, 0, None, None, None)])
            tDgBias = tDgBias[
                (
                    None,
                    None,
                    None,
                    work_tile_info.M_idx,
                    work_tile_info.N_idx,
                    work_tile_info.L_idx,
                )
            ]
            tDrBias = cute.make_rmem_tensor(epi_frag_shape, self.bias_dtype)
            tDrBiasAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)

        # ---- Wait for TMA_B to finish ----
        # Initial phase=0, it flips to 1 when TMA_B warp is done
        cute.arch.mbarrier_wait(tma_epilog_full_bar, 0)

        # Load bias before the MMA wait to hide memory latency.
        if cutlass.const_expr(self.use_bias):
            cute.basic_copy_if(tDpC, tDgBias, tDrBias)
            tDrBiasAcc.store(tDrBias.load().to(self.acc_dtype))

        # ---- Wait for MMA to finish ----
        # Initial phase=0, it flips to 1 when MMA warp is done
        cute.arch.mbarrier_wait(mma_epilog_full_bar, 0)

        # Copy the accumulator from tensor memory to the epilogue registers.
        cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)

        # ---- Signal MMA warp that TMEM read is done ----
        # Wait for asynchronous tensor-memory reads before releasing the allocation.
        cute.arch.fence_view_async_tmem_load()

        # Empty K partitions contribute zero to the split-K reduction.
        if cutlass.const_expr(self.split_k > 1):
            if k_tile_count == 0:
                tDrAcc.fill(0.0)

        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # ---- Type conversion and GMEM store (with optional DSMEM split-k reduction) ----
        elems_per_thread = self._mailbox_elems_per_thread
        if self.split_k == 1:
            # Accumulate bias in FP32 before output conversion.
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())
            # direct path: convert AccType -> TypeC and store to GMEM
            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            cute.basic_copy_if(tDpC, tDrC, tDgC)
        else:
            # CGA split-k via DSMEM scatter with DISTRIBUTED REDUCTION:
            # the output tile is sharded across all split_k CTAs by flat element index.
            # each CTA sends shard-targeted partials to every peer, reduces its own
            # shard from all received partials + its own, and writes its shard to GMEM.
            shard_ept = self._shard_elems_per_thread

            # every CTA arms its own mailbox transaction barrier BEFORE the scatter
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(dsmem_mailbox_mbar, self._mailbox_tx_total)

            # intra-CTA epilog sync: all 128 threads must finish TMEM->RMEM before scatter
            cute.arch.barrier(barrier_id=15, number_of_threads=128)

            # scatter: send each peer's shard of our partial accumulator to that peer
            scatter_val = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
            for peer in range(self.split_k):
                if peer != split_rank:
                    shard_start = peer * shard_ept
                    sender_idx = split_rank - (1 if split_rank > cutlass.Int32(peer) else 0)
                    mailbox_base = sender_idx * 128 * shard_ept
                    for i in range(shard_ept):
                        scatter_val[0] = tDrAcc[shard_start + i].to(cutlass.Float32)
                        cute.arch.store_async_dsmem(
                            dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid,
                            scatter_val[0].bitcast(cutlass.Int32),
                            dsmem_mailbox_mbar,
                            cutlass.Int32(peer),
                        )

            # wait for all peer shards to arrive in our local mailbox
            cute.arch.mbarrier_wait(dsmem_mailbox_mbar, 0)

            # reduce: accumulate received shards with our own partial
            my_shard_start = split_rank * shard_ept
            for s in range(self.split_k - 1):
                mailbox_base = s * 128 * shard_ept
                for i in range(shard_ept):
                    tDrAcc[my_shard_start + i] = tDrAcc[my_shard_start + i] + (
                        dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid
                    ).load().to(self.acc_dtype)

            # mask out non-shard predicates so basic_copy_if only writes our shard
            for i in range(elems_per_thread):
                if i < my_shard_start or i >= my_shard_start + shard_ept:
                    tDpC[i] = cutlass.Boolean(False)

            # Apply bias once after the split-K reduction.
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())

            # convert and write our shard to GMEM
            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            cute.basic_copy_if(tDpC, tDrC, tDgC)


# ============================================================
# BMM wrapper
# ============================================================
# This wrapper reorders tensor modes from PyTorch's (L, M/N, K/N) layout
# to the CuTe layout expected by the kernel: (M, K, L), (N, K, L), (M, N, L).
#
# PyTorch tensors come in as:
#   a: (L, M, K) — select modes [1, 2, 0] -> (M, K, L)
#   b: (L, K, N), seen as (L, N, K) after .permute(0, 2, 1); the mode
#                   reordering depends on the underlying strides
#   c: (L, M, N) — select modes [1, 2, 0] -> (M, N, L)
#
# The kernel expects:
#   A: (Gemm_M, Gemm_K, Gemm_L) with K or M contiguous
#   B: (Gemm_N, Gemm_K, Gemm_L) with K or N contiguous
#   C: (Gemm_M, Gemm_N, Gemm_L) with M contiguous
@cute.jit
def bmm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (L, M, K) from PyTorch
    b: cute.Tensor,  # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,  # (L, M, N) from PyTorch (permuted)
    bias: Optional[cute.Tensor],  # (M,), or None
    stream: cuda.CUstream,
):
    # Reorder modes to match kernel's expected (M, K, L), (N, K, L), (M, N, L)
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))  # (L,M,K) -> (M,K,L)
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))  # (L,K,N) -> (N,K,L)
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))  # (L,M,N) -> (M,N,L)
    if cutlass.const_expr(bias is not None):
        # Expand row bias to (M, N, L):(1, 0, 0).
        bias = cute.make_tensor(
            bias.iterator,
            cute.make_layout(c.layout.shape, stride=(1, 0, 0)),
        )
    gemm_op(a, b, c, bias, stream)


# ============================================================
# Compilation and execution helpers
# ============================================================
@lru_cache(maxsize=4)
def compile_tgv_gemm(
    mnkl: Tuple[int, int, int, int],
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    acc_dtype: Type[cutlass.Numeric],
    cta_m: int = 64,
    cta_n: int = 8,
    cta_k: int = 128,
    num_ab_stage: int = 8,
    use_pdl: bool = False,
    pdl_count: int = -1,
    split_k: int = 1,
    bias: Optional[cute.Tensor] = None,
):
    from cutlass.cute.runtime import make_fake_stream

    use_bias = bias is not None
    gemm = TgvGemmKernel(
        acc_dtype=acc_dtype,
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cta_k,
        num_ab_stage=num_ab_stage,
        use_pdl=use_pdl,
        pdl_count=pdl_count,
        split_k=split_k,
        use_bias=use_bias,
    )

    stream = make_fake_stream()
    return cute.compile(bmm, gemm, a, b, c, bias, stream)
