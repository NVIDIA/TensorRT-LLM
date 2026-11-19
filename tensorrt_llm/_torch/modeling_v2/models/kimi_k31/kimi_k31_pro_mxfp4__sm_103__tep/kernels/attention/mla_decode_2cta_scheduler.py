# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
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
"""MLA static tile scheduler helpers used by TS decode kernels.

The high-throughput MLA TS kernel schedules work in
``(cluster_idx, seq_q_idx, batch_idx, split_kv_idx)`` coordinates.  CUTLASS's
generic static scheduler does not expose that decomposition, so the MLA example
keeps this small scheduler locally instead of importing the bare-metal MLA
runner.
"""

import cutlass
from cutlass import cute


def _is_positive_power_of_two(value: int) -> bool:
    """Return whether a host-known scheduler extent is a power of two."""

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and (value & (value - 1)) == 0
    )


@cute.jit
def divmod_constexpr_power_of_two_or_fdd(
    dividend,
    constexpr_divisor: cutlass.Constexpr[int],
    fallback_divisor: cute.FastDivmodDivisor,
):
    """Use shift/mask for a power-of-two constexpr, otherwise the existing FDD."""

    if cutlass.const_expr(_is_positive_power_of_two(constexpr_divisor)):
        shift = constexpr_divisor.bit_length() - 1
        return (
            dividend >> cutlass.Int32(shift),
            dividend & cutlass.Int32(constexpr_divisor - 1),
        )
    return divmod(dividend, fallback_divisor)


class MLAStaticTileSchedulerParams:
    """Static scheduler parameters for MLA split-KV work tiles."""

    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisor = None,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        """Initialize static scheduler dimensions and fast-divmod divisors."""
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
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

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.problem_shape_s)
        values += cutlass.extract_mlir_values(self.split_kv)
        values += cutlass.extract_mlir_values(self.problem_shape_b_fdd)
        values += cutlass.extract_mlir_values(self.problem_shape_s_fdd)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(self.problem_shape_b, (values[0],))
        problem_shape_s = cutlass.new_from_mlir_values(self.problem_shape_s, (values[1],))
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[2],))
        problem_shape_b_fdd = cutlass.new_from_mlir_values(self.problem_shape_b_fdd, (values[3],))
        problem_shape_s_fdd = cutlass.new_from_mlir_values(self.problem_shape_s_fdd, (values[4],))
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            split_kv,
            problem_shape_b_fdd=problem_shape_b_fdd,
            problem_shape_s_fdd=problem_shape_s_fdd,
            loc=self.loc,
            ip=self.ip,
        )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    problem_shape_s: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    """Create MLA static scheduler parameters from host/runtime dimensions."""
    return MLAStaticTileSchedulerParams(
        is_persistent, problem_shape_b, problem_shape_s, cluster_shape_mnk, split_kv
    )


class MLAStaticTileScheduler:
    """Static and persistent MLA tile scheduler.

    Persistent mode grid-strides through the logical MLA tile space by SM count.
    Non-persistent mode maps each CTA directly to one logical work tile.
    """

    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        is_valid: bool = True,
        loc=None,
        ip=None,
    ):
        """Initialize scheduler state for the current CTA."""
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        if params.is_persistent:
            self.persistent_blk_layout = cute.make_layout(
                (
                    params.cluster_shape_mnk[0],
                    params.problem_shape_s,
                    params.problem_shape_b,
                    params.split_kv,
                ),
                loc=loc,
                ip=ip,
            )
            self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
            self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        else:
            self.is_valid = cutlass.Boolean(is_valid)
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: MLAStaticTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        """Return launch grid shape for static or persistent scheduling."""
        grid_shape = (
            params.cluster_shape_mnk[0],
            params.problem_shape_b * params.problem_shape_s,
            params.split_kv,
        )
        if params.is_persistent:
            return (
                cutlass.min(
                    max_active_clusters * cute.size(params.cluster_shape_mnk),
                    cute.size(grid_shape, loc=loc, ip=ip),
                ),
                1,
                1,
            )
        return grid_shape
