# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent dense BF16/FP16 GEMM kernel for Rubin (SM107).

Ports the Blackwell (SM100) BF16 dense GEMM to Rubin by changing the target
architecture to sm_107.  The BF16 MMA instructions (tcgen05) are shared
between SM100 and SM107, so the Blackwell ``utils.sm100.make_trivial_tiled_mma``
helper is reused directly. The only material changes are:

* ``self.arch`` is set to ``"sm_107"`` so the compiler emits SM107 code and
  the runtime picks up SM107's larger shared memory (328 KiB vs 228 KiB).
* SMEM capacity is re-queried for the new arch, which lets the kernel
  automatically compute more pipeline stages when SMEM allows.

The constructor, ``wrapper``, ``wrapper_strided``, ``can_implement``, and
``__call__`` interfaces are **identical** to the Blackwell version, so existing
test scripts work unchanged.
"""

from typing import Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings.driver import CUstream
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_gemm_persistent import (
    PersistentDenseGemmKernel as BlackwellPersistentDenseGemmKernel,
)


class PersistentDenseGemmKernel(BlackwellPersistentDenseGemmKernel):
    """Persistent dense BF16/FP16 GEMM kernel targeting Rubin (SM107).

    Inherits the full Blackwell (SM100) implementation and only overrides the
    target architecture so the kernel compiles for SM107 and benefits from its
    larger shared-memory capacity (328 KiB vs 228 KiB).

    All public APIs (``wrapper``, ``wrapper_strided``, ``can_implement``,
    ``__call__``) remain identical.

    Supported data types (same as Blackwell):
        - A/B: Float16, BFloat16
        - Accumulator: Float32
        - C: Float16, BFloat16

    Example::

        gemm = PersistentDenseGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=False,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 4),
            use_tma_store=True,
        )
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool = True,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
        max_num_ab_stage: int = 0,
        split_k_slices: int = 1,
    ):
        super().__init__(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=use_tma_store,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
            split_k_slices=split_k_slices,
            use_direct_split_k_reduce=True,
        )
        # Override architecture for Rubin – everything else is inherited.
        # ``self.arch`` is read only by ``_compute_num_tmem_alloc_cols``
        # (blackwell/dense_gemm_persistent.py:261 -> :1037), which forwards it
        # to ``utils.get_num_tmem_alloc_cols``; it never selects MMA
        # instructions, so the base "sm_107" name is correct here and matches
        # the sibling Rubin kernels (rubin/dense_gemm_persistent.py:181).
        self.arch = "sm_107"
        # Optional cap on A/B pipeline stages.  Rubin's larger SMEM (328 KiB)
        # causes the auto-computed num_ab_stage to be ~9, but profiling shows
        # that 6 stages (matching Blackwell) can be faster due to reduced
        # synchronization overhead and lower SMEM pressure.  Set to 0 (default)
        # to use the auto-computed value.
        self._max_num_ab_stage = max_num_ab_stage

    @staticmethod
    def _format_mn_pair(shape_mn: Tuple[int, int]) -> str:
        return f"{shape_mn[0]}x{shape_mn[1]}"

    def _kernel_tactic_name(self) -> str:
        return (
            f"base_2cta{int(self.use_2cta_instrs)}"
            f"_tile{self._format_mn_pair(self.mma_tiler_mn)}"
            f"_cluster{self._format_mn_pair(self.cluster_shape_mn)}"
            f"_stage{self._max_num_ab_stage}"
            f"_split{self.split_k_slices}"
            f"_direct{int(self.use_direct_split_k_reduce)}"
        )

    def __repr__(self) -> str:
        return self._kernel_tactic_name()

    __str__ = __repr__

    def _setup_attributes(self):
        """Set up configurations, optionally capping A/B pipeline stages."""
        super()._setup_attributes()
        if self._max_num_ab_stage > 0 and self.num_ab_stage > self._max_num_ab_stage:
            self.num_ab_stage = self._max_num_ab_stage
            # Recompute staged SMEM layouts with the capped stage count.
            tiled_mma = self._create_tiled_mma()
            self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
            )
            self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
            )

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        batch_size: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """Rubin (SM107) feasibility: the inherited Blackwell checks plus an
        M-vs-cluster_m guard.

        A cluster that is ``cluster_m`` wide in M but spans fewer than
        ``cluster_m`` real M CTA-tiles launches phantom M-CTAs whose
        cluster-multicast peers are never produced. On SM107 those phantom CTAs
        issue out-of-bounds TMA accesses (illegal memory access / hang) --
        observed when the autotuner profiles the M=1 decode MLA absorb BMM with
        ``cluster_m=4`` (base ``(4,1)/(4,4)`` and preferred ``(4,2)``). This
        mirrors the CGA-tile guard in the NVFP4 mixed-clusters kernel
        (``Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.can_implement``).
        The per-CTA M-tile is ``mma_tiler_mn[0]`` for 1-CTA MMA but
        ``mma_tiler_mn[0] // 2`` for 2-CTA MMA (the grid is built from the
        per-CTA tile), so the CTA-tile count uses the halved tile when
        ``use_2cta_instrs`` -- otherwise valid 2-CTA tactics are pruned. Only
        the M axis is gated; the kernel tolerates N over-padding.
        """
        if not BlackwellPersistentDenseGemmKernel.can_implement(
            ab_dtype,
            acc_dtype,
            c_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            batch_size,
            a_major,
            b_major,
            c_major,
        ):
            return False
        cta_tile_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
        ctas_m = (m + cta_tile_m - 1) // cta_tile_m
        if ctas_m < cluster_shape_mn[0]:
            return False
        return True


# Preferred-cluster variant kept in this file with the base Rubin BF16 kernel.
class PersistentDenseGemmKernelPreferredCluster(PersistentDenseGemmKernel):
    """This class implements batched matrix multiplication (C = A x B) with preferred cluster shape support.

    Extends PersistentDenseGemmKernel with support for preferred and fallback cluster shapes,
    enabling mixed cluster size optimization using static tile scheduling.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param preferred_cluster_shape_mn: Preferred cluster dimensions (M,N) for optimal performance
    :type preferred_cluster_shape_mn: Tuple[int, int]
    :param fallback_cluster_shape_mn: Fallback cluster dimensions (M,N) for parallel processing
    :type fallback_cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
    :type use_tma_store: bool

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Fallback/Preferred cluster M must be multiple of 2 if use_2cta_instrs=True
        - Fallback/Preferred cluster M/N must be positive and power of 2, total cluster size <= 16
        - Preferred cluster M/N must be multiple of fallback cluster M/N

    Example:
        >>> gemm = PersistentDenseGemmKernelPreferredCluster(
        ...     acc_dtype=cutlass.Float32,
        ...     use_2cta_instrs=True,
        ...     mma_tiler_mn=(128, 128),
        ...     preferred_cluster_shape_mn=(4, 2),
        ...     fallback_cluster_shape_mn=(2, 1),
        ... )
        >>> gemm(
        ...     a_tensor,
        ...     b_tensor,
        ...     c_tensor,
        ...     max_active_preferred_clusters,
        ...     max_active_fallback_clusters,
        ...     stream,
        ... )
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool = True,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
        max_num_ab_stage: int = 0,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel with preferred cluster support.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param preferred_cluster_shape_mn: Preferred cluster shape for optimal performance.
        :type preferred_cluster_shape_mn: Tuple[int, int]
        :param fallback_cluster_shape_mn: Tuple (ClusterM, ClusterN) fallback shape of the cluster.
        :type fallback_cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Use TMA or normal store for output C tensor. Defaults to True.
        :type use_tma_store: bool
        :param swizzle_size: Swizzle size for CTA scheduling. Defaults to 1.
        :type swizzle_size: int
        :param raster_along: Raster direction for CTA scheduling. Defaults to "m".
        :type raster_along: Literal["m", "n"]
        """
        # Initialize base class with fallback cluster shape
        super().__init__(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=fallback_cluster_shape_mn,
            use_tma_store=use_tma_store,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
            max_num_ab_stage=max_num_ab_stage,
        )

        # Add preferred cluster specific attributes
        self.preferred_cluster_shape_mn = preferred_cluster_shape_mn
        self.fallback_cluster_shape_mn = fallback_cluster_shape_mn

    def _kernel_tactic_name(self) -> str:
        return (
            f"preferred_cluster_2cta{int(self.use_2cta_instrs)}"
            f"_tile{self._format_mn_pair(self.mma_tiler_mn)}"
            f"_preferred{self._format_mn_pair(self.preferred_cluster_shape_mn)}"
            f"_fallback{self._format_mn_pair(self.fallback_cluster_shape_mn)}"
            f"_stage{self._max_num_ab_stage}"
        )

    def _setup_attributes(self):
        """Set up configurations for preferred cluster GEMM.

        Extends the base _setup_attributes to add:
        - Preferred cluster layout computation
        - Multicast CTA counts for both preferred and fallback clusters
        """
        # Call parent class setup for basic attributes
        super()._setup_attributes()

        # Compute preferred cluster layout
        tiled_mma = self._create_tiled_mma()
        self.preferred_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.fallback_cluster_layout_vmnk = self.cluster_layout_vmnk

        # Calculate multicast CTA counts for preferred cluster
        self.num_preferred_mcast_ctas_a = cute.size(self.preferred_cluster_layout_vmnk.shape[2])
        self.num_preferred_mcast_ctas_b = cute.size(self.preferred_cluster_layout_vmnk.shape[1])
        self.is_preferred_a_mcast = self.num_preferred_mcast_ctas_a > 1
        self.is_preferred_b_mcast = self.num_preferred_mcast_ctas_b > 1

        # Store fallback values (already computed in parent)
        self.num_fallback_mcast_ctas_a = self.num_mcast_ctas_a
        self.num_fallback_mcast_ctas_b = self.num_mcast_ctas_b
        self.is_fallback_a_mcast = self.is_a_mcast
        self.is_fallback_b_mcast = self.is_b_mcast

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_preferred_clusters: cutlass.Constexpr,
        max_active_fallback_clusters: cutlass.Constexpr,
        stream: CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation with preferred cluster support.

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param max_active_preferred_clusters: Maximum number of active preferred clusters
        :type max_active_preferred_clusters: cutlass.Constexpr
        :param max_active_fallback_clusters: Maximum number of active fallback clusters
        :type max_active_fallback_clusters: cutlass.Constexpr
        :param stream: CUDA stream for execution
        :type stream: CUstream
        :param epilogue_op: Optional elementwise function for epilogue
        :type epilogue_op: cutlass.Constexpr
        """
        # Setup static attributes
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        tiled_mma = self._create_tiled_mma()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup smem layouts
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))

        # Setup preferred TMA load for A
        a_op_preferred = sm100_utils.cluster_shape_to_tma_atom_A(
            self.preferred_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_a_preferred, tma_tensor_a_preferred = cute.nvgpu.make_tiled_tma_atom_A(
            a_op_preferred,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup fallback TMA load for A
        a_op_fallback = sm100_utils.cluster_shape_to_tma_atom_A(
            self.fallback_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_a_fallback, tma_tensor_a_fallback = cute.nvgpu.make_tiled_tma_atom_A(
            a_op_fallback,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup preferred TMA load for B
        b_op_preferred = sm100_utils.cluster_shape_to_tma_atom_B(
            self.preferred_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_preferred, tma_tensor_b_preferred = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_preferred,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        # Setup fallback TMA load for B
        b_op_fallback = sm100_utils.cluster_shape_to_tma_atom_B(
            self.fallback_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_fallback, tma_tensor_b_fallback = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_fallback,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                self.epi_tile,
            )

        # Compute the single (preferred) tile scheduler params and launch grid
        self.tile_sched_params, preferred_grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.preferred_cluster_shape_mn,
            max_active_preferred_clusters,
            self.fallback_cluster_shape_mn,
            max_active_fallback_clusters,
            self.swizzle_size,
            self.raster_along,
        )

        # Launch the kernel
        self.kernel(
            tiled_mma,
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.preferred_cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=preferred_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.preferred_cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )
        return

    @cute.jit
    def cluster_specific_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        num_tma_producer: int,
        effective_is_a_mcast: bool,
        effective_is_b_mcast: bool,
        cluster_shape_mn: Tuple[int, int],
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        This is essentially the same as parent's kernel but parameterized for cluster-specific settings.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup CTA/thread coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Define shared storage
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop AB pipeline
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize accumulator pipeline
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster synchronization
        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

        # Setup SMEM tensors
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        # Compute multicast masks
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(effective_is_a_mcast or effective_is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # Partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # TMA partitions
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # MMA fragments
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Wait for cluster synchronization
        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)

        # TMA load warp
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        # MMA warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

                        handle.release()

                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        # Epilogue warps
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            if cutlass.const_expr(self.use_tma_store):
                assert tma_atom_c is not None and sC is not None
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # Pre-advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.use_tma_store):
                    # (EPI_TILE_M, EPI_TILE_N, STAGE)
                    # sC = storage.sC.get_tensor(
                    #     c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
                    # )

                    acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        tCgC,
                        epi_tile,
                        num_tiles_executed,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                else:
                    acc_consumer_state = utils.gemm.sm100.epilogue(
                        self,
                        tidx,
                        tCtAcc_base,
                        tCgC,
                        epi_tile,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                    )

            if cutlass.const_expr(self.use_tma_store):
                # Wait for C store complete
                c_pipeline.producer_tail()
            else:
                # Synchronize before TMEM dealloc (done by the caller)
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a_preferred: cute.CopyAtom,
        mA_mkl_preferred: cute.Tensor,
        tma_atom_a_fallback: cute.CopyAtom,
        mA_mkl_fallback: cute.Tensor,
        tma_atom_b_preferred: cute.CopyAtom,
        mB_nkl_preferred: cute.Tensor,
        tma_atom_b_fallback: cute.CopyAtom,
        mB_nkl_fallback: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        preferred_cluster_layout_vmnk: cute.Layout,
        fallback_cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        # Get cluster coordinates to determine if this is a preferred cluster
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.preferred_cluster_shape_mn[0]
            and cbdim_y == self.preferred_cluster_shape_mn[1]
            and cbdim_z == 1
        )

        # as for now, only support preferred cluster kernel via the mega-kernel approach
        # mega-kernel approach has 2 mutually exclusive code branches, only one path runs per launch,
        # specify `smem_merge_branch_allocs=True` at launch to enables shared memory reuse between two paths
        if is_preferred_cluster:
            self.cluster_specific_kernel(
                tiled_mma,
                tma_atom_a_preferred,
                mA_mkl_preferred,
                tma_atom_b_preferred,
                mB_nkl_preferred,
                tma_atom_c,
                mC_mnl,
                preferred_cluster_layout_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                c_smem_layout_staged,
                epi_tile,
                tile_sched_params,
                epilogue_op,
                self.num_preferred_mcast_ctas_a + self.num_preferred_mcast_ctas_b - 1,
                self.is_preferred_a_mcast,
                self.is_preferred_b_mcast,
                self.preferred_cluster_shape_mn,
            )
        else:
            self.cluster_specific_kernel(
                tiled_mma,
                tma_atom_a_fallback,
                mA_mkl_fallback,
                tma_atom_b_fallback,
                mB_nkl_fallback,
                tma_atom_c,
                mC_mnl,
                fallback_cluster_layout_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                c_smem_layout_staged,
                epi_tile,
                # Same (preferred) tile_sched_params as the preferred branch:
                # the flexible-cluster launch only downgrades the cluster SHAPE
                # per cluster (8->2 CTAs, changing barrier scope + TMA
                # multicast), NOT the launched grid or the persistent tile
                # partition. Scheduling a fallback-shaped cluster over a
                # separate fallback partition (the previous behavior) left whole
                # cluster work-items unvisited -> output tiles kept stale memory.
                tile_sched_params,
                epilogue_op,
                self.num_fallback_mcast_ctas_a + self.num_fallback_mcast_ctas_b - 1,
                self.is_fallback_a_mcast,
                self.is_fallback_b_mcast,
                self.fallback_cluster_shape_mn,
            )

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        max_active_preferred_clusters: cutlass.Constexpr,
        max_active_fallback_clusters: cutlass.Constexpr,
        stream: CUstream,
    ):
        """Wrapped GEMM entrypoint for TVM-FFI/raw-pointer callers."""
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, batch_size), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, batch_size), order=(1, 0, 2)),
        )

        self(
            a_tensor,
            b_tensor,
            c_tensor,
            max_active_preferred_clusters,
            max_active_fallback_clusters,
            stream,
        )

    @cute.jit
    def wrapper_strided(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        a_stride_m: cutlass.Int32,
        a_stride_batch: cutlass.Int32,
        b_stride_n: cutlass.Int32,
        b_stride_batch: cutlass.Int32,
        max_active_preferred_clusters: cutlass.Constexpr,
        max_active_fallback_clusters: cutlass.Constexpr,
        stream: CUstream,
    ):
        """Wrapped BMM entrypoint supporting non-contiguous A/B strides.

        K stride is assumed to be 1 for both operands (K innermost);
        callers must reject K-strided (e.g. transposed-view) operands.
        """
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_layout(
                (m, k, batch_size),
                stride=(a_stride_m, 1, a_stride_batch),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_layout(
                (n, k, batch_size),
                stride=(b_stride_n, 1, b_stride_batch),
            ),
        )

        self(
            a_tensor,
            b_tensor,
            c_tensor,
            max_active_preferred_clusters,
            max_active_fallback_clusters,
            stream,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        preferred_max_active_clusters: cutlass.Constexpr,
        fallback_cluster_shape_mn: Tuple[int, int],
        fallback_max_active_clusters: cutlass.Constexpr,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ) -> Tuple[
        utils.PersistentTileSchedulerParams,
        Tuple[int, int, int],
    ]:
        """Compute the single (preferred) tile scheduler params and launch grid.

        The kernel is launched once with the preferred cluster shape and the
        fallback cluster shape as the hardware fallback. The persistent tile
        scheduler partitions the output using the PREFERRED geometry, and that
        single partition is used by both the preferred- and fallback-shaped
        clusters at runtime (the fallback shape only changes per-cluster
        barrier scope and TMA multicast, not tile assignment). The fallback
        grid is computed solely to bound max_preferred_cluster_count so the
        preferred clusters never spill into an extra scheduling wave; it is not
        returned. Mirrors the upstream reference
        dense_gemm_persistent_mixed_clusters.py.
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape

        # Compute scheduler params for preferred cluster
        preferred_cluster_shape_mnl = (*preferred_cluster_shape_mn, 1)
        preferred_tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, preferred_cluster_shape_mnl, swizzle_size, raster_along == "m"
        )
        preferred_grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            preferred_tile_sched_params, preferred_max_active_clusters
        )

        # Fallback grid is used ONLY to bound the preferred cluster count
        # below (so preferred clusters do not spill into an extra wave). The
        # fallback tile-scheduler params are intentionally NOT propagated to
        # the kernel: both the preferred- and fallback-shaped clusters must
        # schedule over the SAME preferred tile partition (see the kernel
        # dispatch). Plumbing a separate fallback partition is what previously
        # left output tiles unwritten.
        fallback_cluster_shape_mnl = (*fallback_cluster_shape_mn, 1)
        fallback_tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, fallback_cluster_shape_mnl, swizzle_size, raster_along == "m"
        )
        fallback_grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            fallback_tile_sched_params, fallback_max_active_clusters
        )

        # Align preferred grid to cluster shape
        preferred_grid = cute.round_up(preferred_grid, preferred_cluster_shape_mnl)

        # Compute max preferred clusters: total blocks <= fallback total, and is multiple of cluster size
        preferred_cluster_size_mn = preferred_cluster_shape_mn[0] * preferred_cluster_shape_mn[1]
        max_ctas_for_fallback_cluster = fallback_grid[0] * fallback_grid[1] * fallback_grid[2]
        max_preferred_cluster_count = max_ctas_for_fallback_cluster // preferred_cluster_size_mn
        preferred_grid = (
            preferred_grid[0],
            preferred_grid[1],
            max_preferred_cluster_count,
        )

        return preferred_tile_sched_params, preferred_grid

    def check_mma_tiler_and_cluster_shape(self) -> bool:
        """Check if mma tiler, fallback and preferred cluster shapes are valid."""
        if not self.is_valid_mma_tiler_and_cluster_shape(
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.fallback_cluster_shape_mn,
        ):
            raise testing.CantImplementError(
                "Invalid fallback cluster shape "
                f"{self.fallback_cluster_shape_mn} for mma_tiler_mn={self.mma_tiler_mn}"
            )

        # Validate preferred cluster shape
        if not self.is_valid_mma_tiler_and_cluster_shape(
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.preferred_cluster_shape_mn,
        ):
            raise testing.CantImplementError(
                "Invalid preferred cluster shape "
                f"{self.preferred_cluster_shape_mn} for mma_tiler_mn={self.mma_tiler_mn}"
            )

        # Check preferred is multiple of fallback
        if (
            self.preferred_cluster_shape_mn[0] % self.fallback_cluster_shape_mn[0] != 0
            or self.preferred_cluster_shape_mn[1] % self.fallback_cluster_shape_mn[1] != 0
        ):
            raise testing.CantImplementError(
                f"Preferred cluster shape {self.preferred_cluster_shape_mn} must be "
                f"integer multiple of fallback cluster shape {self.fallback_cluster_shape_mn}"
            )
        return True


@cute.jit
def bmm_preferred_cluster(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (l, m, k)
    b: cute.Tensor,  # (l, k, n)
    c: cute.Tensor,  # (l, m, n)
    max_active_preferred_clusters: cutlass.Constexpr,
    max_active_fallback_clusters: cutlass.Constexpr,
    stream: CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    """Wrapper API for persistent GEMM kernel with preferred cluster."""
    # (l,m,k) -> (m,k,l)
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    # (l,k,n) -> (n,k,l)
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    # (l,m,n) -> (m,n,l)
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))

    gemm_op(
        a,
        b,
        c,
        max_active_preferred_clusters,
        max_active_fallback_clusters,
        stream,
        epilogue_op,
    )
