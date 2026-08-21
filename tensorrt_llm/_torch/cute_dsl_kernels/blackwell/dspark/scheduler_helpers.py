# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import cutlass
import cutlass.cute as cute


class DSparkPersistentTileSchedulerParams:
    def __init__(
        self,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisor = None,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        """Parameters for the persistent DSpark attention tile scheduler."""
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        if cutlass.const_expr(problem_shape_b_fdd is None):
            self.problem_shape_b_fdd = cute.fast_divmod_create_divisor(
                problem_shape_b, loc=loc, ip=ip
            )
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor(
                problem_shape_s, loc=loc, ip=ip
            )
        self.loc = loc
        self.ip = ip

    def _dynamic_fields(self):
        return (
            self.problem_shape_b,
            self.problem_shape_s,
            self.problem_shape_b_fdd,
            self.problem_shape_s_fdd,
        )

    def __extract_mlir_values__(self):
        values = []
        for field in self._dynamic_fields():
            values += cutlass.extract_mlir_values(field)
        return values

    def __new_from_mlir_values__(self, values):
        # Slice per field by its own value count: a field that is a
        # compile-time constant contributes zero values, so mixing static and
        # dynamic scalars round-trips correctly (the original positional
        # values[0..5] indexing required every field to be dynamic).
        rebuilt = []
        values = list(values)
        for field in self._dynamic_fields():
            count = len(cutlass.extract_mlir_values(field))
            rebuilt.append(cutlass.new_from_mlir_values(field, tuple(values[:count])))
            values = values[count:]
        problem_shape_b, problem_shape_s, b_fdd, s_fdd = rebuilt
        return DSparkPersistentTileSchedulerParams(
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            problem_shape_b_fdd=b_fdd,
            problem_shape_s_fdd=s_fdd,
            loc=self.loc,
        )


def create_dspark_persistent_tile_scheduler_params(
    problem_shape_b: cute.Int32,
    problem_shape_s: cute.Int32,
    cluster_shape_mnk: cute.Shape,
) -> DSparkPersistentTileSchedulerParams:
    return DSparkPersistentTileSchedulerParams(
        problem_shape_b,
        problem_shape_s,
        cluster_shape_mnk,
    )


class WorkTileInfo:
    def __init__(self, blk_coord: cute.Coord, is_valid: bool):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        return values

    def __new_from_mlir_values__(self, values):
        new_tile_idx = cutlass.new_from_mlir_values(self.blk_coord, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self.is_valid, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


class DSparkPersistentTileScheduler:
    def __init__(
        self,
        params: DSparkPersistentTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        """Persistent scheduler for the fixed, unsplit DSpark attention work tiles."""
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        self.persistent_blk_layout = cute.make_layout(
            (
                params.cluster_shape_mnk[0],
                params.problem_shape_s,
                params.problem_shape_b,
            ),
            loc=loc,
            ip=ip,
        )
        self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: DSparkPersistentTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        # called by host
        total_blocks = params.cluster_shape_mnk[0] * params.problem_shape_b * params.problem_shape_s
        return (
            cutlass.min(
                max_active_clusters * cute.size(params.cluster_shape_mnk),
                total_blocks,
            ),
            1,
            1,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = self.current_work_linear_idx < self.num_blocks
        current_work_cluster_batch, cluster_idx = (
            self.current_work_linear_idx // self.params.cluster_shape_mnk[0],
            self.current_work_linear_idx % self.params.cluster_shape_mnk[0],
        )
        current_work_s_batch, s_idx = divmod(
            current_work_cluster_batch, self.params.problem_shape_s_fdd
        )
        _, b_idx = divmod(current_work_s_batch, self.params.problem_shape_b_fdd)

        # Keep a zero split coordinate for the kernel's existing tensor slicing.
        return WorkTileInfo((cluster_idx, s_idx, b_idx, 0), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        self.current_work_linear_idx += advance_count * self.num_persistent_sm

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        # Slice per component by its own value count so static and dynamic
        # scheduler fields round-trip correctly.
        values = list(values)
        rebuilt = []
        for component in (self.params, self.current_work_linear_idx, self.blk_coord):
            count = len(cutlass.extract_mlir_values(component))
            rebuilt.append(cutlass.new_from_mlir_values(component, values[:count]))
            values = values[count:]
        new_params, new_current_work_linear_idx, new_blk_coord = rebuilt
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values)
        return DSparkPersistentTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_dspark_persistent_tile_scheduler(
    params: DSparkPersistentTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> DSparkPersistentTileScheduler:
    return DSparkPersistentTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


LOG2_E = 1.4426950408889634074
