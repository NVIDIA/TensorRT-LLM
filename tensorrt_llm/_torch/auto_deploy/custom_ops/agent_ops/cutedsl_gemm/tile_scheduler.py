from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr

from tensorrt_llm._torch.auto_deploy.custom_ops.agent_ops.cutedsl_gemm import utils

from .cute_dsl_utils import ArgumentsBase, ParamsBase
from .fast_math import FastDivmod
from .pipeline import PipelineStateWAdvance


class RasterOrderOption(IntEnum):
    AlongM = 0
    AlongN = 1
    Heuristic = 2  # Pick AlongM if tiles_n > tiles_m, else AlongN


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


@cute.jit
def get_raster_order_from_option(
    raster_order_option: RasterOrderOption,
    problem_shape_ncluster_mn: cute.Shape,
    group_size: Int32,
) -> RasterOrder:
    raster_order = (
        RasterOrder.AlongM
        if raster_order_option == RasterOrderOption.AlongM
        else RasterOrder.AlongN
    )
    if raster_order_option == RasterOrderOption.Heuristic:
        problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
        problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
        raster_order = (
            RasterOrder.AlongM if problem_blocks_n > problem_blocks_m else RasterOrder.AlongN
        )
    return raster_order


# Grouping arguments together that should be passed to __call__
@dataclass
class TileSchedulerOptions(ArgumentsBase):
    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption] = RasterOrderOption.Heuristic
    max_swizzle_size: Int32 = Int32(8)
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None


@dataclass
class TileSchedulerArguments(ArgumentsBase):
    problem_shape_ntile_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None
    is_persistent: cutlass.Constexpr[bool] = False


class TileScheduler:
    @dataclass
    class Params(ParamsBase):
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_divmod: FastDivmod
        num_groups_regular: Int32
        group_size_divmod: FastDivmod
        group_size_tail_divmod: FastDivmod
        num_clusters_in_group_divmod: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        batch_idx_permute: Optional[cute.Tensor]
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        is_persistent: cutlass.Constexpr[bool]

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "TileScheduler.Params":
            assert args.cluster_shape_mnk[2] == 1
            cluster_shape_mn = const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            raster_order = get_raster_order_from_option(
                args.raster_order, problem_shape_ncluster_mn, args.group_size
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            return TileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod.create(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod.create(group_size),
                # Don't divide by 0
                FastDivmod.create(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod.create(num_clusters_in_group),
                args.tile_count_semaphore if const_expr(args.is_persistent) else None,
                args.batch_idx_permute,
                cluster_shape_mn,
                args.is_persistent,
            )

    def __init__(
        self,
        current_work_linear_idx: Int32,
        num_tiles_executed: Int32,
        tile_count: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_linear_idx = current_work_linear_idx
        self.num_tiles_executed = num_tiles_executed
        self._tile_count = tile_count
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        tile_count: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TileScheduler":
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        stages = 0
        if const_expr(not params.is_persistent):
            cidx, cidy, _ = cute.arch.cluster_idx()
            cdimx, _, _ = cute.arch.cluster_dim()
            cluster_id = cidx + cidy * cdimx
            current_work_linear_idx = Int32(cluster_id)
        else:
            _, _, bidz = cute.arch.block_idx()
            current_work_linear_idx = Int32(bidz)
            if const_expr(params.tile_count_semaphore is not None):
                assert tile_count is not None
                assert scheduler_pipeline is not None
                stages = const_expr(cute.size(tile_count))
        return TileScheduler(
            current_work_linear_idx,
            Int32(0),  # num_tiles_executed
            tile_count,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1 if is_scheduler_warp else 0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        num_ctas_mnl = tuple(
            x * y for x, y in zip(params.problem_shape_ncluster_mnl, params.cluster_shape_mn)
        ) + (params.problem_shape_ncluster_mnl[2],)
        if const_expr(not params.is_persistent):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_id, id_in_group = params.num_clusters_in_group_divmod.divmod(cluster_id_in_problem)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = params.group_size_divmod.divmod(id_in_group)
        else:  # tail part
            cid_slow, cid_fast_in_group = params.group_size_tail_divmod.divmod(id_in_group)
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_divmod.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        if const_expr(not params.is_persistent):
            cluster_id_in_problem = self._current_work_linear_idx
            _, _, bidz = cute.arch.block_idx()
        else:
            bidz, cluster_id_in_problem = params.num_clusters_per_problem_divmod.divmod(
                self._current_work_linear_idx
            )
        cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * params.cluster_shape_mn[1] + bidx_in_cluster[1]
        batch_idx = (
            bidz if const_expr(params.batch_idx_permute is None) else params.batch_idx_permute[bidz]
        )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        if const_expr(not params.is_persistent):
            is_valid = self.num_tiles_executed == 0
        else:
            is_valid = self._current_work_linear_idx < cute.size(params.problem_shape_ncluster_mnl)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def fetch_next_work(self, is_scheduler_warp: bool | Boolean = False, *, loc=None, ip=None):
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        params = self.params
        if const_expr(params.is_persistent and params.tile_count_semaphore is not None):
            current_work_linear_idx = self._current_work_linear_idx
            if is_scheduler_warp:
                if cute.arch.lane_idx() == 0:
                    num_persistent_clusters = cute.arch.grid_dim()[2]
                    current_work_linear_idx = num_persistent_clusters + utils.atomic_inc_i32(
                        cute.size(params.problem_shape_ncluster_mnl) - 1,
                        params.tile_count_semaphore,
                    )
                # lane 0 already has the right tile_idx, just need to broadcast
                current_work_linear_idx = cute.arch.shuffle_sync(current_work_linear_idx, 0)
            self._current_work_linear_idx = current_work_linear_idx

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        params = self.params
        if const_expr(params.is_persistent):
            num_persistent_clusters = cute.arch.grid_dim()[2]
            if const_expr(params.tile_count_semaphore is None):  # Static persistent
                self._current_work_linear_idx += advance_count * Int32(num_persistent_clusters)
            else:  # Dynamic persistent
                if const_expr(advance_count > 1):
                    self._pipeline_state.advance_iters(advance_count - 1)
                current_work_linear_idx = self._current_work_linear_idx
                if is_scheduler_warp:
                    self._scheduler_pipeline.producer_acquire(self._pipeline_state)
                    lane_idx = cute.arch.lane_idx()
                    if lane_idx < cute.size(params.cluster_shape_mn):
                        # cute.printf("Producer bidx = {}, bidz = {}, tidx = {},
                        # after empty wait, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                        if const_expr(cute.size(params.cluster_shape_mn) == 1):
                            self._tile_count[self._pipeline_state.index] = current_work_linear_idx
                            self._scheduler_pipeline.producer_commit(self._pipeline_state)
                        else:
                            peer_cta_rank_in_cluster = lane_idx
                            mbar_ptr = self._scheduler_pipeline.producer_get_barrier(
                                self._pipeline_state
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_ptr, 4, peer_cta_rank_in_cluster
                            )
                            utils.store_shared_remote(
                                val=current_work_linear_idx,
                                smem_ptr=self._tile_count.iterator + self._pipeline_state.index,
                                mbar_ptr=mbar_ptr,
                                peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                            )
                        # cute.printf("Producer bidx = {}, bidz = {}, tidx = {}, after full arrive", bidx, bidz, tidx)
                else:
                    # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {},
                    # before full wait, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                    self._scheduler_pipeline.consumer_wait(self._pipeline_state)
                    # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {},
                    # after full wait, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                    current_work_linear_idx = self._tile_count[self._pipeline_state.index]
                    # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {},
                    # after smem read, idx = {}", bidx, bidz, tidx, current_work_linear_idx)
                    # Need this fence since the STAS from the producer is using the async proxy.
                    # Without this, we get race condition / deadlock.
                    if const_expr(cute.size(params.cluster_shape_mn) > 1):
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        # if tidx % 32 == 0: cute.printf("bidx = {}, bidz = {}, tidx = {},
                        # before empty arrive", bidx, bidz, tidx)
                        self._scheduler_pipeline.consumer_release(self._pipeline_state)
                        # if tidx == 320: cute.printf("bidx = {}, bidz = {}, tidx = {}, idx = {},
                        # after empty arrive", bidx, bidz, tidx, current_work_linear_idx)
                    # if tidx == 320: cute.printf("bidx = {}, bidz = {}, tidx = {}, idx = {},
                    # after empty arrive", bidx, bidz, tidx, current_work_linear_idx)
                self._current_work_linear_idx = current_work_linear_idx
                self._pipeline_state.advance()
        self.num_tiles_executed += Int32(advance_count)

    def producer_tail(self):
        if const_expr(self.params.is_persistent and self.params.tile_count_semaphore is not None):
            self._scheduler_pipeline.producer_tail(self._pipeline_state)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_linear_idx,
            self.num_tiles_executed,
            self._tile_count,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_linear_idx,
                self.num_tiles_executed,
                self._tile_count,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


@cute.jit
def triangular_idx_to_coord(idx: Int32) -> Tuple[Int32, Int32]:
    """
    Convert a triangular index to 2D coordinates.
    This is used to convert the linear index to 2D coordinates for triangular matrices.
    """
    row = utils.ceil((utils.sqrt(2 * idx + 2.25) - 0.5)) - 1
    col = idx - (row * (row + 1)) // 2
    return row, col


class TriangularTileScheduler(TileScheduler):
    """We assume the tile size per cluster is square (e.g., 128 x 256 per CTA, with cluster 2 x 1)"""

    @dataclass
    class Params(ParamsBase):
        problem_shape_ncluster_mnl: cute.Shape
        num_clusters_per_problem_divmod: FastDivmod
        group_size_inv_f32: cutlass.Float32
        num_groups_regular: Int32
        group_size_divmod: FastDivmod
        group_size_tail_divmod: FastDivmod
        group_size_mul_group_size_divmod: FastDivmod
        group_size_tail_mul_group_size_divmod: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        is_persistent: cutlass.Constexpr[bool]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "TriangularTileScheduler.Params":
            assert args.cluster_shape_mnk[2] == 1
            cluster_shape_mn = const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            cluster_m = problem_shape_ncluster_mn[0]
            # Assume that each cluster is responsible for a square tile
            num_clusters_per_problem = cluster_m * (cluster_m + 1) // 2
            group_size = min(args.group_size, cluster_m)
            group_size_tail = cluster_m % group_size
            num_groups_regular = cluster_m // group_size
            return TriangularTileScheduler.Params(
                problem_shape_ncluster_mnl,
                FastDivmod.create(num_clusters_per_problem),
                cutlass.Float32(1.0 / group_size),
                num_groups_regular,
                FastDivmod.create(group_size),
                # Don't divide by 0
                FastDivmod.create(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod.create(group_size * group_size),
                FastDivmod.create((group_size_tail if group_size_tail > 0 else 1) * group_size),
                args.tile_count_semaphore if const_expr(args.is_persistent) else None,
                cluster_shape_mn,
                args.is_persistent,
            )

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TriangularTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        tile_count: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TriangularTileScheduler":
        stages = 0
        if const_expr(not params.is_persistent):
            cluster_id, _, _ = cute.arch.cluster_idx()
            current_work_linear_idx = Int32(cluster_id)
        else:
            _, _, bidz = cute.arch.block_idx()
            current_work_linear_idx = Int32(bidz)
            if const_expr(params.tile_count_semaphore is not None):
                assert tile_count is not None
                assert scheduler_pipeline is not None
                stages = const_expr(cute.size(tile_count))
        return TriangularTileScheduler(
            current_work_linear_idx,
            Int32(0),  # num_tiles_executed
            tile_count,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1 if is_scheduler_warp else 0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        clusters = (
            params.num_clusters_per_problem_divmod.divisor,
            1,
            params.problem_shape_ncluster_mnl[2],
        )
        num_ctas_mnl = tuple(x * y for x, y in zip(clusters, params.cluster_shape_mn)) + (
            params.problem_shape_ncluster_mnl[2],
        )
        if const_expr(not params.is_persistent):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        if const_expr(not params.is_persistent):
            cluster_id_in_problem = self._current_work_linear_idx
            _, _, bidz = cute.arch.block_idx()
        else:
            bidz, cluster_id_in_problem = params.num_clusters_per_problem_divmod.divmod(
                self._current_work_linear_idx
            )
        # CTA Swizzle to promote L2 data reuse
        group_size = params.group_size_divmod.divisor
        group_id = (
            utils.ceil(
                (utils.sqrt(2 * cluster_id_in_problem + 2.25) - 0.5) * params.group_size_inv_f32
            )
            - 1
        )
        cid_m_start = group_id * group_size
        id_in_group = cluster_id_in_problem - (cid_m_start * (cid_m_start + 1)) // 2
        group_size_actual = (
            group_size
            if group_id < params.num_groups_regular
            else params.group_size_tail_divmod.divisor
        )
        group_col, group_remainder = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            group_col, group_remainder = params.group_size_mul_group_size_divmod.divmod(id_in_group)
        else:  # tail part
            group_col, group_remainder = params.group_size_tail_mul_group_size_divmod.divmod(
                id_in_group
            )
        cid_m_in_group, cid_n_in_group = Int32(0), Int32(0)
        if id_in_group >= group_size_actual * group_size * group_id:  # triangular tail
            cid_m_in_group, cid_n_in_group = triangular_idx_to_coord(group_remainder)
        else:
            if group_id < params.num_groups_regular:
                cid_n_in_group, cid_m_in_group = params.group_size_divmod.divmod(group_remainder)
            else:
                cid_n_in_group, cid_m_in_group = params.group_size_tail_divmod.divmod(
                    group_remainder
                )
        cid_m = cid_m_start + cid_m_in_group
        cid_n = group_col * group_size + cid_n_in_group

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * params.cluster_shape_mn[1] + bidx_in_cluster[1]
        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        if const_expr(not params.is_persistent):
            is_valid = self.num_tiles_executed == 0
        else:
            is_valid = (
                self._current_work_linear_idx
                < params.num_clusters_per_problem_divmod.divisor
                * params.problem_shape_ncluster_mnl[2]
            )
        # bidx, bidy, bidz = cute.arch.block_idx()
        # tidx, _, _ = cute.arch.thread_idx()
        # if tidx == 0:
        #     cute.printf("bidx = {}, bidy = {}, group_id = {}, id_in_group = {}, group_size_actual = {},
        #     group_col = {}, group_remainder = {}, cid_n_in_group = {}, cid_m_in_group = {},
        #     cid_m = {}, cid_n = {}, cid_m_in_group = {}, cid_n_in_group = {}, is_valid = {}",
        #     bidx, bidy, group_id, id_in_group, group_size_actual, group_col,
        #     group_remainder, cid_n_in_group, cid_m_in_group, cid_m, cid_n, is_valid)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)


@dataclass
class VarlenMTileSchedulerArguments(ParamsBase):
    problem_shape_ntile_mnl: cute.Shape
    total_m: Int32
    cu_seqlens_m: cute.Tensor
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    tile_shape_mn: cutlass.Constexpr[cute.Shape]
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    is_persistent: cutlass.Constexpr[bool] = False


class VarlenMTileScheduler(TileScheduler):
    @dataclass
    class Params(ParamsBase):
        problem_shape_ncluster_mnl: cute.Shape
        total_m: Int32
        cu_seqlens_m: cute.Tensor
        raster_order: cutlass.Constexpr[RasterOrder]
        group_size: Int32
        group_size_divmod: Optional[FastDivmod]
        group_size_tail_divmod: Optional[FastDivmod]
        num_clusters_in_group_divmod: FastDivmod
        tile_shape_mn: cutlass.Constexpr[cute.Shape]
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        is_persistent: cutlass.Constexpr[bool]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "VarlenMTileScheduler.Params":
            assert args.cluster_shape_mnk[2] == 1
            cluster_shape_mn = const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            # problem_shape_ntile_mnl[0] will be None for VarlenM
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                None,
                cute.ceil_div(problem_shape_ntile_mn[1], cluster_shape_mn[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            raster_order = const_expr(
                RasterOrder.AlongM
                if args.raster_order == RasterOrderOption.AlongM
                else RasterOrder.AlongN  # For Heuristic we also use AlongN
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            if const_expr(ncluster_fast is not None):
                group_size = min(args.group_size, ncluster_fast)
                group_size_tail = ncluster_fast % group_size
            else:
                group_size, group_size_tail = args.group_size, None
            if const_expr(ncluster_slow is not None):
                num_clusters_in_group = group_size * ncluster_slow
            else:
                num_clusters_in_group = None
            return VarlenMTileScheduler.Params(
                problem_shape_ncluster_mnl,
                args.total_m,
                args.cu_seqlens_m,
                raster_order,
                group_size,
                FastDivmod.create(group_size) if ncluster_fast is not None else None,
                # Don't divide by 0
                (
                    FastDivmod.create(group_size_tail if group_size_tail > 0 else 1)
                    if group_size_tail is not None
                    else None
                ),
                (
                    FastDivmod.create(num_clusters_in_group)
                    if num_clusters_in_group is not None
                    else None
                ),
                args.tile_shape_mn,
                args.tile_count_semaphore if const_expr(args.is_persistent) else None,
                cluster_shape_mn,
                args.is_persistent,
            )

    def __init__(
        self,
        current_work_linear_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        tile_count: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_linear_idx = current_work_linear_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._tile_count = tile_count
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return VarlenMTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        tile_count: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenMTileScheduler":
        stages = 0
        _, _, bidz = cute.arch.block_idx()
        current_work_linear_idx = Int32(bidz)
        if const_expr(params.tile_count_semaphore is not None):
            assert tile_count is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(tile_count))
        return VarlenMTileScheduler(
            current_work_linear_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            tile_count,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1 if is_scheduler_warp else 0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mn[0]
        num_batch = params.problem_shape_ncluster_mnl[2]
        total_clusters_m_max = (params.total_m + num_batch * (block_size - 1)) // block_size
        total_clusters_max = total_clusters_m_max * params.problem_shape_ncluster_mnl[1]
        if const_expr(not params.is_persistent):
            return (*params.cluster_shape_mn, total_clusters_max)
        else:
            num_persistent_clusters = cutlass.min(max_active_clusters, total_clusters_max)
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def _get_num_m_blocks(
        self, lane: Int32, bidb_start: Int32, block_size: cutlass.Constexpr[int]
    ) -> Int32:
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        batch_idx = lane + bidb_start
        cur_cu_seqlen = Int32(0)
        if batch_idx <= num_batch:
            cur_cu_seqlen = self.params.cu_seqlens_m[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        return (
            cute.ceil_div(seqlen, block_size)
            if batch_idx < num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, num_clusters_m: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        params = self.params
        # CTA Swizzle to promote L2 data reuse
        if const_expr(params.num_clusters_in_group_divmod is not None):
            group_id, id_in_group = params.num_clusters_in_group_divmod.divmod(
                cluster_id_in_problem
            )
            num_clusters_in_group = params.num_clusters_in_group_divmod.divisor
        else:
            assert params.raster_order == RasterOrder.AlongN
            num_clusters_in_group = params.group_size * num_clusters_m
            group_id = cluster_id_in_problem // num_clusters_in_group
            id_in_group = cluster_id_in_problem - group_id * num_clusters_in_group
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if const_expr(
            params.group_size_divmod is not None and params.group_size_tail_divmod is not None
        ):
            num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
            if (group_id + 1) * num_clusters_in_group <= num_clusters:
                cid_slow, cid_fast_in_group = params.group_size_divmod.divmod(id_in_group)
            else:  # tail part
                cid_slow, cid_fast_in_group = params.group_size_tail_divmod.divmod(id_in_group)
        else:
            assert params.raster_order == RasterOrder.AlongM
            group_size_actual = cutlass.min(
                params.group_size, num_clusters_m - group_id * params.group_size
            )
            cid_slow = id_in_group // group_size_actual
            cid_fast_in_group = id_in_group - cid_slow * group_size_actual
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else num_clusters_m
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mn[0]
        batch_idx = self._current_batch_idx
        num_clusters_m = self._get_num_m_blocks(
            lane_idx, bidb_start=batch_idx, block_size=block_size
        )
        num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
        num_clusters_cumulative = utils.warp_prefix_sum(num_clusters, lane_idx)
        # Total number of blocks for the next 31 problems, same for all lanes
        clusters_in_problems = cute.arch.shuffle_sync(
            num_clusters_cumulative, cute.arch.WARP_SIZE - 1
        )
        problems_end_tile = self._num_work_idx_before_cur_batch + clusters_in_problems
        # if cute.arch.thread_idx()[0] == 128 + 31:
        # cute.printf("SingleTileVarlenScheduler: tile_idx=%d, problems_end_tile = %d, num_clusters_m=%d,
        # num_clusters_cumulative = %d, problems_end_tile = %d", self._tile_idx, problems_end_tile,
        # num_clusters_m, num_clusters_cumulative, problems_end_tile)
        cid_m, cid_n = Int32(0), Int32(0)
        next_tile_idx = self._current_work_linear_idx
        while problems_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= num_batch:
                batch_idx = Int32(num_batch)
                problems_end_tile = next_tile_idx + 1
            else:
                num_clusters_m = self._get_num_m_blocks(
                    lane_idx, bidb_start=batch_idx, block_size=block_size
                )
                num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
                num_clusters_cumulative = utils.warp_prefix_sum(num_clusters, lane_idx)
                clusters_in_problems = cute.arch.shuffle_sync(
                    num_clusters_cumulative, cute.arch.WARP_SIZE - 1
                )
                problems_end_tile += clusters_in_problems
        # Just a placeholder value in case batch_idx >= num_batch
        num_work_idx_before_cur_batch = problems_end_tile - clusters_in_problems
        if batch_idx >= num_batch:
            cid_m, cid_n, batch_idx = Int32(0), Int32(0), Int32(num_batch)
        else:
            problems_start_tile = problems_end_tile - clusters_in_problems
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler:
            # tile_idx=%d, problems_end_tile = %d, num_clusters_m=%d, batch_idx = %d", self._tile_idx,
            # problems_end_tile, num_clusters_m, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_problems = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    problems_start_tile + num_clusters_cumulative <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_problems
            num_clusters_prev_lane = (
                0
                if batch_idx_in_problems == 0
                else cute.arch.shuffle_sync(num_clusters_cumulative, batch_idx_in_problems - 1)
            )
            num_clusters_m = cute.arch.shuffle_sync(num_clusters_m, batch_idx_in_problems)
            num_work_idx_before_cur_batch = problems_start_tile + num_clusters_prev_lane
            cluster_id_in_problem = next_tile_idx - num_work_idx_before_cur_batch
            # cid_n = cluster_id_in_problem // num_clusters_m
            # cid_m = cluster_id_in_problem - cid_n * num_clusters_m
            # if cute.arch.thread_idx()[0] == 128: cute.printf("SingleTileVarlenScheduler:
            # tile_idx=%d, batch_idx=%d, cid_n=%d, cid_m=%d, is_valid = %d",
            # self._tile_idx, batch_idx, cid_n, cid_m, is_valid)
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, num_clusters_m, loc=loc, ip=ip)
        self._current_batch_idx = batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * params.cluster_shape_mn[1] + bidx_in_cluster[1]
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        if const_expr(not params.is_persistent):
            is_valid = self.num_tiles_executed == 0 and batch_idx < num_batch
        else:
            is_valid = batch_idx < num_batch
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def fetch_next_work(self, is_scheduler_warp: bool | Boolean = False, *, loc=None, ip=None):
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        if const_expr(self.params.tile_count_semaphore is not None):
            params = self.params
            current_work_linear_idx = self._current_work_linear_idx
            if is_scheduler_warp:
                if cute.arch.lane_idx() == 0:
                    # cute.printf("before atomicadd, tidx = {}, bidz = {}, idx = {}",
                    # cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_linear_idx)
                    num_persistent_clusters = cute.arch.grid_dim()[2]
                    current_work_linear_idx = num_persistent_clusters + utils.atomic_add_i32(
                        1, params.tile_count_semaphore
                    )
                    # cute.printf("after atomicadd, tidx = {}, bidz = {}, idx = {}",
                    # cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_linear_idx)
                # lane 0 already has the right tile_idx, just need to broadcast
                current_work_linear_idx = cute.arch.shuffle_sync(current_work_linear_idx, 0)
            self._current_work_linear_idx = current_work_linear_idx

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_linear_idx,
            self.num_tiles_executed,
            self._current_batch_idx,
            self._num_work_idx_before_cur_batch,
            self._tile_count,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_linear_idx,
                self.num_tiles_executed,
                self._current_batch_idx,
                self._num_work_idx_before_cur_batch,
                self._tile_count,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
