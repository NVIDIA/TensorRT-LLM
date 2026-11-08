# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass, fields
from functools import partial
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir import ir
from cutlass.cute import FastDivmodDivisor

try:
    from typing import override
except ImportError:
    from typing_extensions import override


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """
    It includes block, head, batch, and is_valid_tile
    """

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 4
        new_tile_idx = cutlass.new_from_mlir_values(self.tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self.is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class TileSchedulerParams(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    headdim: Int32
    headdim_v: Int32


class StaticPersistentScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks: Int32

        @staticmethod
        def create(
            args: TileSchedulerParams,
            *,
            loc=None,
            ip=None,
        ) -> "StaticPersistentScheduler.Params":
            total_blocks = args.num_block * args.num_head * args.num_batch
            return StaticPersistentScheduler.Params(
                FastDivmodDivisor(args.num_block), FastDivmodDivisor(args.num_head), total_blocks
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.tile_idx = tile_idx
        self.loc = loc
        self.ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerParams, *, loc=None, ip=None) -> Params:
        return StaticPersistentScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "StaticPersistentScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return StaticPersistentScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        sm_count: Optional[Int32] = None,
        occupancy: Int32 = 1,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if sm_count is None:
            hardware_info = cutlass.utils.HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
        vacancies = sm_count * occupancy
        return (cutlass.min(vacancies, params.total_blocks), Int32(1), Int32(1))

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = divmod(self.tile_idx, self.params.num_block_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self.tile_idx < self.params.total_blocks
        return WorkTileInfo((Int32(block_idx), Int32(head_idx), Int32(batch_idx)), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self.tile_idx += cute.arch.grid_dim()[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self.tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self.tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentScheduler(*(tuple(obj_list)), loc=self.loc)


if __name__ == "__main__":

    class TestScheduler:
        def __init__(self, scheduler_cls):
            self.scheduler_cls = scheduler_cls

        @cute.jit
        def __call__(self):
            scheduler_params = TileSchedulerParams(
                num_block=Int32(5),
                num_head=Int32(4),
                num_batch=Int32(3),
                headdim=Int32(128),
                headdim_v=Int32(64),
            )
            params = self.scheduler_cls.to_underlying_arguments(scheduler_params)

            grid_dim = self.scheduler_cls.get_grid_shape(params)
            print(f"grid_dim: {grid_dim}")

            self.kernel(params).launch(grid=grid_dim, block=[32, 2, 1], min_blocks_per_mp=1)

        @cute.kernel
        def kernel(self, params: ParamsBase):
            TileSchedulerCls = partial(self.scheduler_cls.create, params)

            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            bidx, _, _ = cute.arch.block_idx()

            if warp_idx == 0:
                lane_idx = cute.arch.lane_idx()

                scheduler = TileSchedulerCls()

                work_tile = scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    m_block, head_idx, batch_idx = work_tile.tile_idx

                    if lane_idx == 0:
                        cute.printf(
                            "block_idx: {}, warp_idx: {}, m_block: {}, head_idx: {}, batch_idx: {}",
                            bidx,
                            warp_idx,
                            m_block,
                            head_idx,
                            batch_idx,
                        )

                    scheduler.prefetch_next_work()
                    scheduler.advance_to_next_work()
                    work_tile = scheduler.get_current_work()

            elif warp_idx == 1:
                scheduler = TileSchedulerCls()

    test_scheduler = TestScheduler(StaticPersistentScheduler)
    test_scheduler()
