# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Decode-specific tile scheduler for paged fp8 attention.

The pre-schedule step builds a dense worklist over decode KV chunks.  Static
persistent scheduling walks a flattened ``(work_idx, head_kv_idx)`` task id.
CLC scheduling keeps BSA's hardware grid shape, ``(work_idx, head_kv_idx, 1)``,
and maps the canceled CTA coordinate back to the same logical task space.
"""

from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor

from quack.cute_dsl_utils import ParamsBase

from src.common.tile_scheduler import SchedulingMode, WorkTileInfo


@dataclass
class DecodeTileSchedulerArguments(ParamsBase):
    work_capacity: Int32
    num_heads_kv: Int32
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)


class DecodeTileScheduler:
    """Persistent scheduler over decode ``(work_idx, head_kv_idx)`` tasks."""

    @dataclass
    class Params(ParamsBase):
        work_capacity: Int32
        num_heads_kv: Int32
        num_heads_kv_divmod: FastDivmodDivisor
        total_tasks: Int32
        cluster_shape_m: cutlass.Constexpr[int] = 1
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC

    def __init__(
        self,
        params: Params,
        task_idx: Int32,
        clc_scheduler=None,
        clc_pipeline=None,
        clc_consumer_state=None,
        clc_response_ptr=None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._task_idx = task_idx
        self._clc_scheduler = clc_scheduler
        self._clc_pipeline = clc_pipeline
        self._clc_consumer_state = clc_consumer_state
        self._clc_response_ptr = clc_response_ptr
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: DecodeTileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        loc=None,
        ip=None,
    ) -> Params:
        assert args.cluster_shape_mn[1] == 1, "Decode scheduler requires cluster N == 1"
        total_tasks = args.work_capacity * args.num_heads_kv
        return DecodeTileScheduler.Params(
            args.work_capacity,
            args.num_heads_kv,
            FastDivmodDivisor(args.num_heads_kv),
            total_tasks,
            cluster_shape_m=args.cluster_shape_mn[0],
            scheduling_mode=scheduling_mode,
        )

    @staticmethod
    def _clc_grid_shape(params: Params):
        return (
            cute.round_up(params.work_capacity, params.cluster_shape_m),
            params.num_heads_kv,
            Int32(1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        clc_response_ptr=None,
        *,
        loc=None,
        ip=None,
    ) -> "DecodeTileScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            from cutlass.utils import (
                ClcDynamicPersistentTileScheduler,
                ClcDynamicPersistentTileSchedulerParams,
            )

            cutlass_params = ClcDynamicPersistentTileSchedulerParams(
                problem_shape_ntile_mnl=DecodeTileScheduler._clc_grid_shape(params),
                cluster_shape_mnk=(params.cluster_shape_m, 1, 1),
            )
            block_idx = cute.arch.block_idx()
            grid_dim = cute.arch.grid_dim()
            clc_scheduler = ClcDynamicPersistentTileScheduler.create(
                cutlass_params,
                block_idx,
                grid_dim,
                clc_response_ptr,
            )
            return DecodeTileScheduler(
                params,
                block_idx[0],
                clc_scheduler,
                clc_response_ptr=clc_response_ptr,
                loc=loc,
                ip=ip,
            )

        if const_expr(params.cluster_shape_m == 1):
            task_idx = cute.arch.block_idx()[0]
        else:
            task_idx = cute.arch.cluster_idx()[0]
        return DecodeTileScheduler(params, task_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            return DecodeTileScheduler._clc_grid_shape(params)
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        max_ctas = (sm_count // params.cluster_shape_m) * params.cluster_shape_m
        grid_x = cutlass.min(max_ctas, params.total_tasks * params.cluster_shape_m)
        return (grid_x, Int32(1), Int32(1))

    @cute.jit
    def _task_to_work(self, task_idx: Int32, is_valid) -> WorkTileInfo:
        work_idx, head_kv_idx = divmod(task_idx, self.params.num_heads_kv_divmod)
        return WorkTileInfo(
            (Int32(work_idx), Int32(head_kv_idx), Int32(0), Int32(0)),
            is_valid,
        )

    @cute.jit
    def _clc_work_to_coords(self, work) -> WorkTileInfo:
        work_idx = work.tile_idx[0]
        if const_expr(self.params.cluster_shape_m > 1):
            work_idx = work_idx // self.params.cluster_shape_m
        return WorkTileInfo(
            (
                Int32(work_idx),
                Int32(work.tile_idx[1]),
                Int32(0),
                Int32(0),
            ),
            work.is_valid_tile,
        )

    @cute.jit
    def _clc_response_to_work(
        self,
        response_stage: Int32,
        *,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        # CLC responses are 16B opaque records.  The scheduler warp can query
        # the next stage before all consumer warps have read the current one,
        # so each pipeline stage needs its own response slot.  The slot
        # array is typed Int128 in SharedStorage, so the stage index is also
        # the element index and the 16B alignment survives the offset.
        response_ptr = self._clc_response_ptr + response_stage
        m_idx, n_idx, l_idx, is_valid = cute.arch.clc_response(
            response_ptr, loc=loc, ip=ip)
        cute.arch.fence_proxy("async.shared", space="cta")
        cta_idx_in_cluster = cute.arch.block_idx()[0] % Int32(
            self.params.cluster_shape_m)
        return WorkTileInfo(
            (
                Int32(m_idx) + cta_idx_in_cluster,
                Int32(n_idx),
                Int32(l_idx),
                Int32(0),
            ),
            is_valid,
        )

    @cute.jit
    def get_current_work(
        self,
        response_stage: Int32 = Int32(0),
        *,
        loc=None,
        ip=None,
    ) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._clc_response_to_work(
                response_stage, loc=loc, ip=ip)
            self._task_idx = (
                work.tile_idx[0] * self.params.num_heads_kv
                + work.tile_idx[1]
            )
            return self._clc_work_to_coords(work)
        is_valid = self._task_idx < self.params.total_tasks
        return self._task_to_work(self._task_idx, is_valid)

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._clc_scheduler.initial_work_tile_info()
            self._task_idx = (
                work.tile_idx[0] * self.params.num_heads_kv
                + work.tile_idx[1]
            )
            return self._clc_work_to_coords(work)
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(
        self,
        *,
        loc=None,
        ip=None,
        mbarrier_addr=None,
        response_stage: Int32 = Int32(0),
    ):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            assert mbarrier_addr is not None
            response_ptr = self._clc_response_ptr + response_stage
            with cute.arch.elect_one():
                cute.arch.issue_clc_query(
                    mbarrier_addr, response_ptr, loc=loc, ip=ip)
        else:
            assert mbarrier_addr is None
            if const_expr(self.params.cluster_shape_m == 1):
                self._task_idx += cute.arch.grid_dim()[0]
            else:
                self._task_idx += cute.arch.cluster_dim()[0]

    def consumer_advance(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            response_stage = self._clc_consumer_state.index
            self._clc_pipeline.consumer_wait(self._clc_consumer_state)
            work_tile = self.get_current_work(response_stage=response_stage)
            self._clc_pipeline.consumer_release(self._clc_consumer_state)
            self._clc_consumer_state.advance()
            return work_tile
        self.advance_to_next_work()
        return self.get_current_work()

    def set_clc_pipeline(self, clc_pipeline, clc_consumer_state):
        self._clc_pipeline = clc_pipeline
        self._clc_consumer_state = clc_consumer_state

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._task_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [
                self._clc_scheduler,
                self._clc_pipeline,
                self._clc_consumer_state,
                self._clc_response_ptr,
            ]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._task_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [
                self._clc_scheduler,
                self._clc_pipeline,
                self._clc_consumer_state,
                self._clc_response_ptr,
            ]
        for obj, n_items in zip(objs, self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return DecodeTileScheduler(*obj_list, loc=self._loc)
