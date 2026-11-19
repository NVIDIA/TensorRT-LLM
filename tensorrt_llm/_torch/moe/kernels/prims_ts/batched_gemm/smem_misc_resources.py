# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WorkQueue, work-throttle barrier, and cluster proxy resources."""

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.experimental import primitives as prims
from cutlass.pipeline.sm100 import PipelineClcFetchAsync, PipelineTmaUmma

from cutlass.experimental.task_scheduling.enums import PipelineType, TileSchedulerType
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    WorkQueue,
    consumer_work,
)
from cutlass.utils.dynamic_persistent_tile_scheduler import (
    ClcDynamicPersistentTileScheduler,
)

from .batched_gemm_config import BatchedGemmConfig
from .batched_gemm_cga import (
    active_cga_coords,
    active_clc_consumer_threads,
    has_multiple_umma_atoms,
    host_clc_consumer_threads,
    is_flexible_cga,
)

Constexpr = cutlass.Constexpr


class MixedCgaClcDynamicPersistentTileScheduler(ClcDynamicPersistentTileScheduler):
    """Decode the shared launch grid for preferred and fallback CGAs.

    Both physical cluster shapes use the preferred cluster's launch grid.
    N rasterization packs preferred-cluster tiles into grid Z; fallback
    cluster leaders additionally carry X/Y offsets within those tiles.
    """

    @cute.jit
    def _swizzle_and_rasterize(
        self,
        x_idx,
        y_idx,
        z_idx,
        *,
        loc=None,
        ip=None,
    ):
        if cutlass.const_expr(self.params.swizzle_size != 1):
            raise ValueError("mixed-CGA BatchedGemm requires swizzle_size=1")

        if cutlass.const_expr(self.params._raster_along_m):
            return x_idx, y_idx, z_idx

        # get_grid_shape() packs Z using the preferred shape for the entire
        # launch. Decode that same grid for both physical cluster shapes.
        cluster_minor_batch, cluster_major = divmod(
            z_idx, self.params.cluster_shape_major_fdd
        )
        batch_l, cluster_minor = divmod(
            cluster_minor_batch, self.params.cluster_shape_minor_fdd
        )
        cluster_shape_m, cluster_shape_n = self.params.cluster_shape_mn
        # Preferred leaders have x_idx=y_idx=0. Fallback leaders may start
        # inside that preferred tile; preserve their offsets. The caller adds
        # each CTA's coordinates within its physical cluster afterward.
        return (
            cluster_minor * cluster_shape_m + x_idx,
            cluster_major * cluster_shape_n + y_idx,
            batch_l,
        )


@dataclass(kw_only=True)
class WorkThrottleBarrierResource(MemoryResource):
    """One-shot per-work-tile throttle before issuing the next CLC work id.

    Clustered persistent kernels use a CpAsync pipeline
    signal from the active TMA load task to the WorkId task. This prevents the
    scheduler warp from running ahead of the mainloop start for each work tile.
    """

    is_barrier: Constexpr[bool] = True
    producer_state_owned_by_signaling_ctas_only: Constexpr[bool] = True
    # WorkSchedule is physical-CGA CTA0-only. The stock CtaLeader predicate is
    # based on the static 2-CTA UMMA atom and would also select ranks 2, 4, ...
    # in a wider CGA, creating local throttle barriers that no task releases.
    global_cga_producer_cta0: Constexpr[bool] = True


@dataclass(kw_only=True)
class BatchedGemmWorkQueue(WorkQueue):
    """WorkQueue with batched-GEMM CUDA-graph early-exit knowledge."""

    fast_drain_rate: Constexpr[int] = 4
    cfg: Constexpr[BatchedGemmConfig]
    num_non_exiting_ctas_tensor: Any = None
    num_non_exiting_ctas_value: Any = None
    fast_drain_response_ptr: Any = None
    fast_drain_mbar_ptr: Any = None
    _alloc_fast_drain_response: Constexpr[Optional[SmemAllocation]] = None
    _alloc_fast_drain_mbar: Constexpr[Optional[SmemAllocation]] = None

    def __init__(
        self,
        tile_scheduler_config,
        cfg,
        num_non_exiting_ctas_tensor=None,
        num_non_exiting_ctas_value=None,
        fast_drain_response_ptr=None,
        fast_drain_mbar_ptr=None,
        **kwargs,
    ) -> None:
        super().__init__(
            tile_scheduler_config=tile_scheduler_config,
            **kwargs,
        )
        self.cfg = cfg
        self.num_non_exiting_ctas_tensor = num_non_exiting_ctas_tensor
        self.num_non_exiting_ctas_value = num_non_exiting_ctas_value
        self.fast_drain_response_ptr = fast_drain_response_ptr
        self.fast_drain_mbar_ptr = fast_drain_mbar_ptr
        object.__setattr__(self, "_alloc_fast_drain_response", None)
        object.__setattr__(self, "_alloc_fast_drain_mbar", None)
        if self._uses_fast_drain() and fast_drain_mbar_ptr is None:
            object.__setattr__(
                self,
                "_alloc_fast_drain_response",
                SmemAllocation(
                    f"{self.name}_fast_drain_response",
                    dtype=cutlass.Int128,
                    count=self.fast_drain_rate,
                    alignment=16,
                ),
            )
            object.__setattr__(
                self,
                "_alloc_fast_drain_mbar",
                SmemAllocation(
                    f"{self.name}_fast_drain_mbar",
                    dtype=cutlass.Int64,
                    alignment=8,
                ),
            )

    def _uses_fast_drain(self) -> bool:
        return (
            getattr(self.cfg, "is_persistent", False)
            and getattr(self.cfg, "use_early_exit", False)
            and getattr(self.cfg, "use_clc_fast_drain", False)
        )

    def get_smem_requirements(self):
        requirements = []
        if self._alloc_fast_drain_response is not None:
            requirements.append(self._alloc_fast_drain_response)
        if self._alloc_fast_drain_mbar is not None:
            requirements.append(self._alloc_fast_drain_mbar)
        return requirements

    @cute.jit
    def _init_fast_drain_smem_state(self, stage_info: StageInfo) -> None:
        if cutlass.const_expr(not self._uses_fast_drain()):
            return
        if cutlass.const_expr(self._alloc_fast_drain_response is None):
            # Caller-supplied storage; the pointers are already set.
            return
        assert self._alloc_fast_drain_mbar is not None
        context = stage_info.context
        self.fast_drain_response_ptr = cute.make_ptr(
            cutlass.Int128,
            context.smem_base.data_ptr() + self._alloc_fast_drain_response.offset,
            mem_space=cutlass.AddressSpace.smem,
        )
        self.fast_drain_mbar_ptr = cute.make_ptr(
            cutlass.Int64,
            context.smem_base.data_ptr() + self._alloc_fast_drain_mbar.offset,
            mem_space=cutlass.AddressSpace.smem,
        )

    def create_pipeline(self, pipeline_config):
        if (
            not has_multiple_umma_atoms(self.cfg)
            or pipeline_config.pipeline_type != PipelineType.ClcFetchAsync
        ):
            return super().create_pipeline(pipeline_config)

        barrier_ptr = pipeline_config.barrier_ptr
        if barrier_ptr is None:
            barrier_storage = cutlass.Array(
                cutlass.Int64,
                pipeline_config.num_stages * 2,
                space=cutlass.AddressSpace.smem,
                alignment=8,
            )
            barrier_ptr = cute.make_ptr(
                cutlass.Int64,
                barrier_storage.data_ptr(),
                mem_space=cutlass.AddressSpace.smem,
            )

        sync_object_full = PipelineTmaUmma._make_sync_object(
            barrier_ptr.align(min_align=8),
            pipeline_config.num_stages,
            (pipeline.PipelineOp.ClcLoad, pipeline_config.producer_group),
            pipeline_config.num_bytes,
        )
        if is_flexible_cga(self.cfg):
            # Flexible launch: the barrier count must follow the device-selected
            # preferred/fallback shape.
            consumer_threads = active_clc_consumer_threads(self.cfg)
        else:
            # Fixed wider launch: materialize the exact preferred-shape count;
            # no runtime dimension query is needed.
            dim_x, dim_y, _ = self.cfg.preferred_cluster_shape
            consumer_threads = cutlass.Int32(
                host_clc_consumer_threads(self.cfg, dim_x, dim_y)
            )
        sync_object_empty = PipelineTmaUmma._make_sync_object(
            barrier_ptr.align(min_align=8) + pipeline_config.num_stages,
            pipeline_config.num_stages,
            (
                pipeline.PipelineOp.AsyncThread,
                pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    consumer_threads,
                ),
            ),
        )

        tidx, _, _ = cute.arch.thread_idx()
        dim_x, dim_y, _, _, _, _ = active_cga_coords(self.cfg)
        dst_rank = tidx % cutlass.Int32(32)
        is_signaling_thread = dst_rank < dim_x * dim_y
        return PipelineClcFetchAsync(
            sync_object_full,
            sync_object_empty,
            pipeline_config.num_stages,
            dst_rank,
            cutlass.Int32(0),
            is_signaling_thread,
        )

    def create_tile_scheduler(self):
        if (
            not has_multiple_umma_atoms(self.cfg)
            or self.tile_scheduler_config.tile_scheduler_type
            != TileSchedulerType.ClcDynamicPersistent
        ):
            return super().create_tile_scheduler()

        params = self.tile_scheduler_config.tile_scheduler_params
        bidx, bidy, bidz = cute.arch.block_idx()
        # Fixed wider launches use compile-time preferred dimensions here;
        # flexible launches query only the active preferred/fallback shape.
        dim_x, dim_y, _, _, _, _ = active_cga_coords(self.cfg)
        cta_id_in_cluster = (
            cutlass.Int32(bidx % dim_x),
            cutlass.Int32(bidy % dim_y),
            cutlass.Int32(0),
        )
        return MixedCgaClcDynamicPersistentTileScheduler(
            params,
            cta_id_in_cluster,
            cutlass.Int32(0),
            self.tile_scheduler_config.response_ptr,
            (bidx, bidy, bidz),
        )

    @cute.jit
    def should_skip_work_tile(self, work_tile) -> bool:
        # For CUDA-graph reuse, persistent tasks still fetch the padded
        # work ids, but skip the full task body for token CTAs past the active
        # batch limit and only run the WorkQueue tail to advance the scheduler.
        if cutlass.const_expr(
            (not self.cfg.use_early_exit)
            or (
                self.num_non_exiting_ctas_value is None
                and self.num_non_exiting_ctas_tensor is None
            )
        ):
            return cutlass.Boolean(False)

        tile_coord_m, tile_coord_n, _ = work_tile.tile_idx
        if cutlass.const_expr(self.cfg.is_swap_ab):
            token_cta_idx = tile_coord_n
        else:
            token_cta_idx = tile_coord_m
        token_cga_idx = token_cta_idx // cutlass.Int32(self.cfg.token_cluster_dim)

        if cutlass.const_expr(self.num_non_exiting_ctas_value is not None):
            num_non_exiting_ctas = self.num_non_exiting_ctas_value
        else:
            num_non_exiting_ctas_view = cutlass.make_array_view(
                self.num_non_exiting_ctas_tensor
            )
            num_non_exiting_ctas = num_non_exiting_ctas_view.load(
                idx=cutlass.Int32(0), vector_size=1
            )[0]
        return token_cga_idx >= num_non_exiting_ctas

    @cute.jit
    def _clc_fast_drain_batch(self, parity: cutlass.Int32) -> cutlass.Int32:
        assert self.fast_drain_response_ptr is not None
        assert self.fast_drain_mbar_ptr is not None

        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                self.fast_drain_mbar_ptr,
                16 * self.fast_drain_rate,
                relaxed=True,
            )
            for drain_idx in cutlass.range_constexpr(self.fast_drain_rate):
                cute.arch.issue_clc_query(
                    self.fast_drain_mbar_ptr,
                    self.fast_drain_response_ptr + drain_idx,
                    multicast=False,
                )

        while not prims.mbarrier_try_wait_parity(self.fast_drain_mbar_ptr, parity):
            pass

        canceled_count = cutlass.Int32(0)
        for drain_idx in cutlass.range_constexpr(self.fast_drain_rate):
            _, _, _, is_canceled_i32 = cute.arch.clc_response(
                self.fast_drain_response_ptr + drain_idx
            )
            canceled_count += is_canceled_i32
        return canceled_count

    @cute.jit
    def _clc_fast_drain_all(self) -> None:
        assert self.fast_drain_mbar_ptr is not None
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(self.fast_drain_mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()

        parity = cutlass.Int32(0)
        drained = cutlass.Int32(1)
        while drained > cutlass.Int32(0):
            drained = self._clc_fast_drain_batch(parity)
            parity = cutlass.Int32(1) - parity

    @cute.jit
    def _maybe_fast_drain_skipped_tile(self) -> None:
        # Fast-drain cancels every cluster that has not launched yet. Early-exit
        # padding sits on the token axis only, so that is a suffix of the CLC
        # order solely when the token axis is the slow raster axis. Along-N on
        # swap-AB hits the first M cluster's token padding while later M
        # clusters are still queued; draining there drops those tiles.
        if cutlass.const_expr(
            self._uses_fast_drain() and self.cfg.clc_fast_drain_skip_is_suffix
        ):
            skip_work_tile = self._get_consumer_var_from_ts("skip_work_tile")
            if skip_work_tile:
                self._clc_fast_drain_all()

    @cute.jit
    def _fetch_work_tile_impl(self, stage_info: StageInfo) -> None:
        self._init_fast_drain_smem_state(stage_info)
        if cutlass.const_expr(
            self.tile_scheduler_config.tile_scheduler_type
            == TileSchedulerType.ClcDynamicPersistent
        ):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            if cta_rank_in_cluster == 0:
                self._maybe_fast_drain_skipped_tile()
                mbarrier_addr = self.pipeline.producer_get_barrier(self.producer_state)
                stage_response_ptr = self._get_stage_response_ptr(stage_info.stage_idx)
                with cute.arch.elect_one():
                    cute.arch.issue_clc_query(mbarrier_addr, stage_response_ptr)
            assert self.tile_scheduler is not None
            self.tile_scheduler._num_tiles_executed += cutlass.Int32(1)


@dataclass(kw_only=True)
class ProxyClusterBarrierResource(MemoryResource):
    """Virtual resource for cross-CTA data-readiness signaling.

    Carries no SMEM data — pure barrier. SyncTask is the producer
    (cross-CTA arrive). MmaTask is the consumer (waits for both CTAs'
    data to be ready, releases via tcgen05.commit CTA_2).

    Pipeline: TS ``AsyncUmma`` (CUTLASS ``PipelineAsyncUmma``)
      Full barrier: AsyncThread producer with 32*umma_cta_count arrivals into the
      leader CTA's full barrier.
      Empty barrier: TCGen05Mma consumer release with CTA_2 peer signaling.
    """

    cfg: Constexpr[BatchedGemmConfig]
    consumer_stage_idx: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def create_pipeline(self, pipeline_config):
        result = super().create_pipeline(pipeline_config)
        if has_multiple_umma_atoms(self.cfg):
            from .mixed_cga_ts import override_umma_atom_pipeline

            return override_umma_atom_pipeline(result)
        return result

    def __post_init__(self) -> None:
        self.consumer_stage_idx = TaskLocalVariable(
            dtype=cutlass.Int32,
            default=cutlass.Int32(0),
            docs="Pipeline stage consumed by the cross-CTA proxy.",
        )

    @cute.jit
    def producer_work(self, stage_info: StageInfo) -> None:
        """Publish SMEM writes before the cross-CTA proxy arrive.

        SyncTask reaches this point only after waiting on the gather/TMA
        producer barriers.  The proxy barrier is a second handoff to the UMMA
        task, so use the same cluster-scoped async proxy release fence as the
        generated low-latency kernels before ProducerCommit releases MMA.
        """
        prims.fence_proxy_async_release_sync_restrict()

    @consumer_work(returns=consumer_stage_idx)
    @cute.jit
    def consumer_work(self, stage_info: StageInfo) -> cutlass.Int32:
        """Make cross-CTA SMEM writes visible and publish the ready stage."""
        prims.fence_proxy_async_acquire_sync_restrict()
        return stage_info.stage_idx
