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

"""Device-bound persistent tile scheduler for MoE kernels.

When the actual tile count is only known on the device side (e.g., written
by a prior routing kernel), the host launches with an upper-bound. This
scheduler reads the actual value on the device and recomputes the problem
layout so that is_valid_tile correctly bounds the persistent loop.
"""

from typing import Tuple

import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Integer, dsl_user_op

from .static_persistent_tile_scheduler import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)


class DeviceBoundPersistentTileScheduler(StaticPersistentTileScheduler):
    """Persistent tile scheduler where actual tile count is read from
    device memory.

    Host launches with upper-bound max tiles. Device create() recomputes
    problem layout and FastDivmod from the actual value so is_valid_tile
    correctly reflects the real work bound.

    Use create() when the M dimension is runtime (standard kernels).
    Use create_n() when the N dimension is runtime (swapab kernels).
    """

    @staticmethod
    @dsl_user_op
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        actual_tiles_m: Int32,
        *,
        loc=None,
        ip=None,
    ):
        """Create scheduler with device-side actual_tiles_m override.

        Only recomputes M-dependent fields. N tiles, N clusters,
        batch_fdd, and the N-dependent raster fdd are unchanged (L=1).
        """
        sched = StaticPersistentTileScheduler.create(params, block_idx, grid_dim, loc=loc, ip=ip)

        n_tiles = params.problem_shape_ntile_mnl[1]
        sched.params.problem_shape_ntile_mnl = (actual_tiles_m, n_tiles, Int32(1))

        cluster_m = params.cluster_shape_mn[0]
        n_clusters = params.problem_layout_ncluster_mnl.shape[1]
        actual_clusters_m = (actual_tiles_m + Int32(cluster_m) - Int32(1)) // Int32(cluster_m)
        sched.params.problem_layout_ncluster_mnl = cute.make_layout(
            (actual_clusters_m, n_clusters, Int32(1)),
            loc=loc,
            ip=ip,
        )

        # Only recompute the fast_divmod for the M-dependent raster dimension.
        actual_clusters_m_fdd = cute.fast_divmod_create_divisor(actual_clusters_m, loc=loc, ip=ip)
        if params.raster_along_m:
            sched.params.cluster_shape_major_fdd = actual_clusters_m_fdd
        else:
            sched.params.cluster_shape_minor_fdd = actual_clusters_m_fdd

        return sched

    @staticmethod
    @dsl_user_op
    def create_n(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        actual_tiles_n: Int32,
        *,
        loc=None,
        ip=None,
    ):
        """Create scheduler with device-side actual_tiles_n override (swapab).

        Only recomputes N-dependent fields. M tiles, M clusters,
        batch_fdd, and the M-dependent raster fdd are unchanged (L=1).
        """
        sched = StaticPersistentTileScheduler.create(params, block_idx, grid_dim, loc=loc, ip=ip)

        m_tiles = params.problem_shape_ntile_mnl[0]
        sched.params.problem_shape_ntile_mnl = (m_tiles, actual_tiles_n, Int32(1))

        cluster_n = params.cluster_shape_mn[1]
        m_clusters = params.problem_layout_ncluster_mnl.shape[0]
        actual_clusters_n = (actual_tiles_n + Int32(cluster_n) - Int32(1)) // Int32(cluster_n)
        sched.params.problem_layout_ncluster_mnl = cute.make_layout(
            (m_clusters, actual_clusters_n, Int32(1)),
            loc=loc,
            ip=ip,
        )

        # Only recompute the fast_divmod for the N-dependent raster dimension.
        actual_clusters_n_fdd = cute.fast_divmod_create_divisor(actual_clusters_n, loc=loc, ip=ip)
        if params.raster_along_m:
            sched.params.cluster_shape_minor_fdd = actual_clusters_n_fdd
        else:
            sched.params.cluster_shape_major_fdd = actual_clusters_n_fdd

        return sched
