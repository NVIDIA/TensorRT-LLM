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

import inspect
from typing import Any, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Integer,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo


class ClcDynamicPersistentTileSchedulerParams:
    """A class to represent parameters for a dynamic persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    :ivar cluster_shape_mn: Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
    :type cluster_shape_mn: tuple
    """

    @dsl_user_op
    def __init__(
        self,
        problem_shape_ntile_mnl: cute.Shape,
        cluster_shape_mnk: cute.Shape,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """
        Initializes the ClcDynamicPersistentTileSchedulerParams with the given parameters.

        :param problem_shape_ntile_mnl: The shape of the problem in terms of
            number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
        :type problem_shape_ntile_mnl: cute.Shape
        :param cluster_shape_mnk: The shape of the cluster in (m, n) dimensions.
        :type cluster_shape_mnk: cute.Shape
        :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
        :type swizzle_size: int
        :param raster_along_m: Rasterization order of clusters. Only used when swizzle_size > 1.
            True means along M, false means along N.
        :type raster_along_m: bool

        :raises ValueError: If cluster_shape_k is not 1.
        """

        if cluster_shape_mnk[2] != 1:  # type: ignore[index]
            raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")  # type: ignore[index]
        if swizzle_size < 1:
            raise ValueError(f"expect swizzle_size >= 1, but get {swizzle_size}")

        self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
        # cluster_shape_mnk is kept for reconstruction
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]  # type: ignore[index]
        self.swizzle_size = swizzle_size
        self._raster_along_m = raster_along_m
        self.cluster_shape_major_fdd = None
        self.cluster_shape_minor_fdd = None
        self._loc = loc

        # By default, we follow m major (col-major) raster order, so make a col-major layout
        self.problem_layout_ncluster_mnl = cute.make_layout(
            cute.ceil_div(
                self.problem_shape_ntile_mnl,
                cluster_shape_mnk[:2],  # type: ignore[index]
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        )

        if swizzle_size == 1 and not raster_along_m:
            cluster_count_major = self.problem_layout_ncluster_mnl.shape[1]
            cluster_count_minor = self.problem_layout_ncluster_mnl.shape[0]
            self.cluster_shape_major_fdd = cute.fast_divmod_create_divisor(
                cluster_count_major, loc=loc, ip=ip
            )
            self.cluster_shape_minor_fdd = cute.fast_divmod_create_divisor(
                cluster_count_minor, loc=loc, ip=ip
            )

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values, self._values_pos = [], []
        for obj in [
            self.problem_shape_ntile_mnl,
            self._cluster_shape_mnk,
            self.swizzle_size,
            self._raster_along_m,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))

        # Add FastDivmod divisors to MLIR values for Host->Device transfer
        # Only add non-None values to avoid MLIR type errors
        fastdivmod_values = []
        fastdivmod_indices = []  # Track which FastDivmod objects are present

        for i, (fdd_name, fdd_obj) in enumerate(
            [
                ("cluster_shape_major_fdd", self.cluster_shape_major_fdd),
                ("cluster_shape_minor_fdd", self.cluster_shape_minor_fdd),
            ]
        ):
            if fdd_obj is not None:
                # Extract MLIR values from FastDivmodDivisor objects
                fdd_values = extract_mlir_values(fdd_obj)
                fastdivmod_values.extend(fdd_values)
                fastdivmod_indices.append(i)

        values += fastdivmod_values
        self._values_pos.append(
            len(fastdivmod_indices)
        )  # Store count of FastDivmod objects, not values
        self._fastdivmod_indices = fastdivmod_indices  # Store for reconstruction

        return values

    def __new_from_mlir_values__(
        self, values: list[ir.Value]
    ) -> "ClcDynamicPersistentTileSchedulerParams":
        obj_list = []
        values_copy = list(values)  # Make a copy to avoid modifying original

        # Reconstruct original objects from MLIR values
        for obj, n_items in zip(
            [
                self.problem_shape_ntile_mnl,
                self._cluster_shape_mnk,
                self.swizzle_size,
                self._raster_along_m,
            ],
            self._values_pos[:-1],  # Exclude FastDivmod count
        ):
            obj_list.append(new_from_mlir_values(obj, values_copy[:n_items]))
            values_copy = values_copy[n_items:]

        new_params = ClcDynamicPersistentTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

        # Restore FastDivmod divisors from remaining values
        fdd_names = [
            "cluster_shape_major_fdd",
            "cluster_shape_minor_fdd",
        ]

        if hasattr(self, "_fastdivmod_indices") and len(self._fastdivmod_indices) > 0:
            # Override the FastDivmod divisors created by __init__ with reconstructed ones
            for j, original_index in enumerate(self._fastdivmod_indices):
                fdd_name = fdd_names[original_index]
                # Get the original FastDivmodDivisor object
                original_fdd = getattr(self, fdd_name)
                if original_fdd is not None and j < len(values_copy):
                    # Each FastDivmodDivisor has 1 MLIR value
                    reconstructed_fdd = new_from_mlir_values(original_fdd, [values_copy[j]])
                    setattr(new_params, fdd_name, reconstructed_fdd)

        return new_params

    @dsl_user_op
    def get_grid_shape(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Integer, Integer, Integer]:
        """
        Computes the grid shape based on the problem shape and cluster shape.

        :return: the grid is the CTA numbers that has aligned with cluster shape.
        """

        problem_ceiling_cta_mnl = cute.round_up(
            self.problem_shape_ntile_mnl, self._cluster_shape_mnk
        )

        if self.swizzle_size == 1 and self._raster_along_m:
            return problem_ceiling_cta_mnl  # type: ignore[return-value]
        else:
            # If swizzling is enabled or raster_along_n,
            # we are going to map from a linear idx to tile id manually.
            num_problem_cta_count = cute.size(problem_ceiling_cta_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(self.cluster_shape_mn, loc=loc, ip=ip)
            return (
                *self.cluster_shape_mn,
                num_problem_cta_count // num_ctas_per_cluster,
            )


# Set explicit signature for Sphinx documentation to avoid issues with @dsl_user_op decorator
ClcDynamicPersistentTileSchedulerParams.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
    [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
)


class ClcDynamicPersistentTileScheduler:
    """A scheduler for dynamic persistent tile execution in CUTLASS/CuTe kernels.

    :ivar params: Tile schedule related params, including cluster shape.
    :type params: ClcDynamicPersistentTileSchedulerParams
    :ivar cta_id_in_cluster: ID of the CTA within its cluster
    :type cta_id_in_cluster: cute.Coord
    :ivar _num_tiles_executed: Counter for executed tiles
    :type _num_tiles_executed: Int32
    """

    def __init__(
        self,
        params: ClcDynamicPersistentTileSchedulerParams,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        clc_response_ptr: cute.Pointer,
        block_idx: Tuple[Integer, Integer, Integer],
        insert_fence: bool = True,
    ):
        """
        Initializes the ClcDynamicPersistentTileScheduler with the given parameters.

        :param params: Tile schedule related params, including cluster shape.
        :type params: ClcDynamicPersistentTileSchedulerParams
        :param cta_id_in_cluster: ID of the CTA within its cluster.
        :type cta_id_in_cluster: cute.Coord
        :param num_tiles_executed: Counter for executed tiles.
        :type num_tiles_executed: Int32
        :param clc_response_ptr: Pointer of the clc rsponse.
        :type clc_response_ptr: cute.Pointer
        :param block_idx: The block index.
        :type block_idx: Tuple[Integer, Integer, Integer]
        :param insert_fence: Whether to insert a fence to ensure generic-async proxy order.
            CLC issue is in async proxy while loading the response from shared memory is in
            generic proxy. A cross-proxy fence is needed to ensure producer's next issue
            won't race with consumer's current loading. Therefore the scheduler inserts a
            fence by default after loading the response.
            Developers may insert the fence in pipeline acquire/release functions. In that case,
            the fence here can be omitted.
        :type insert_fence: bool
        """
        self.params = params
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed
        self._clc_response_ptr = clc_response_ptr
        self._block_idx = block_idx
        self.insert_fence = insert_fence

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.cta_id_in_cluster)
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self._clc_response_ptr))
        values.extend(extract_mlir_values(self._block_idx))
        return values

    def __new_from_mlir_values__(
        self, values: list[ir.Value]
    ) -> "ClcDynamicPersistentTileScheduler":
        assert len(values) == 8
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[0:3])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[3]])
        new_clc_response_ptr = new_from_mlir_values(self._clc_response_ptr, [values[4]])
        new_block_idx = new_from_mlir_values(self._block_idx, values[5:8])
        return ClcDynamicPersistentTileScheduler(
            self.params,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            new_clc_response_ptr,
            new_block_idx,
            self.insert_fence,
        )

    @dsl_user_op
    @staticmethod
    def create(
        params: ClcDynamicPersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        clc_response_ptr: cute.Pointer,
        insert_fence: bool = True,
        physical_cluster_shape_mn: Optional[Tuple[int, int]] = None,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "ClcDynamicPersistentTileScheduler":
        """Initialize the dynamic persistent tile scheduler.

        :param params: Parameters for the persistent
            tile scheduler.
        :type params: ClcDynamicPersistentTileSchedulerParams
        :param block_idx: The 3d block index in the format (bidx, bidy, bidz).
        :type block_idx: Tuple[Integer, Integer, Integer]
        :param grid_dim: The 3d grid dimensions for kernel launch.
        :type grid_dim: Tuple[Integer, Integer, Integer]

        :return: A ClcDynamicPersistentTileScheduler object.
        :rtype: ClcDynamicPersistentTileScheduler
        """
        params = params

        bidx, bidy, bidz = block_idx

        # CTA id in the cluster.  This is a PHYSICAL property: it must
        # be taken modulo the cluster the hardware actually formed, which
        # under mixed CGA is not the params (preferred) shape.  A
        # fallback body passes physical_cluster_shape_mn so it can keep
        # decoding with the preferred params (see _swizzle_and_rasterize)
        # while offsetting by its own shape.
        cluster_mn = (
            params.cluster_shape_mn
            if physical_cluster_shape_mn is None
            else physical_cluster_shape_mn
        )
        cta_id_in_cluster = (
            Int32(bidx % cluster_mn[0]),
            Int32(bidy % cluster_mn[1]),
            Int32(0),
        )

        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        # Initialize clc response pointer
        clc_response_ptr = clc_response_ptr
        # The block index
        block_idx = block_idx

        return ClcDynamicPersistentTileScheduler(
            params,
            cta_id_in_cluster,
            num_tiles_executed,
            clc_response_ptr,
            block_idx,
            insert_fence,
        )

    # called by host
    @dsl_user_op
    @staticmethod
    def get_grid_shape(
        params: ClcDynamicPersistentTileSchedulerParams,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Integer, Integer, Integer]:
        """Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        :param params: Parameters for grid shape calculation.
        :type params: ClcDynamicPersistentTileSchedulerParams

        :return: The calculated 3d grid shape.
        :rtype: Tuple[Integer, Integer, Integer]
        """

        return params.get_grid_shape(loc=loc, ip=ip)

    @cute.jit
    def _swizzle_and_rasterize(
        self,
        x_idx: Int32,
        y_idx: Int32,
        z_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Int32, Int32, Int32]:
        """Swizzle and rasterize the given coordinates for leader CTA of the cluster.

        x_idx / y_idx are the leader's offsets INSIDE the params cluster
        footprint and z_idx selects the footprint.  The upstream version
        dropped x_idx / y_idx on the two z-form paths, which is sound only
        when every launched cluster fills the footprint (then they are
        provably 0).  MIXED CGA breaks that: a z slice that cannot form the
        preferred cluster is tiled by several smaller fallback clusters
        whose leaders differ ONLY in x/y, so dropping them aliases them all
        onto the same tiles and orphans the rest (silicon: intermittent
        256-row zero blocks and out-of-bounds tile ids).  Adding the offsets
        back is a no-op for the non-mixed case.
        """
        if cutlass.const_expr(self.params.swizzle_size == 1):
            if cutlass.const_expr(self.params._raster_along_m):
                return x_idx, y_idx, z_idx
            else:
                # Decode linear index using FastDivmod objects.
                # First, get cluster_major using cluster_shape_major_fdd
                cluster_minor_batch, cluster_major = divmod(  # type: ignore[operator]
                    z_idx, self.params.cluster_shape_major_fdd
                )
                # Then decode cluster_minor_batch to get cluster_minor and batch_l using FastDivmod
                batch_l, cluster_minor = divmod(
                    cluster_minor_batch, self.params.cluster_shape_minor_fdd
                )
                cluster_m = cluster_minor
                cluster_n = cluster_major

                return (
                    cluster_m * self.params.cluster_shape_mn[0] + x_idx,
                    cluster_n * self.params.cluster_shape_mn[1] + y_idx,
                    batch_l,
                )
        else:
            # Swizzled raster over the exact cluster count: groups of
            # swizzle_size clusters along the swizzled dimension are visited in
            # turn and the last group is narrower when the count is not a
            # swizzle multiple. Decoding through a layout padded to whole groups
            # would leave the trailing real tiles beyond the launched grid.
            num_cluster_m, num_cluster_n, _ = self.params.problem_layout_ncluster_mnl.shape
            swizzle = Int32(self.params.swizzle_size)
            clusters_per_batch = num_cluster_m * num_cluster_n
            batch_l = z_idx // clusters_per_batch
            linear = z_idx - batch_l * clusters_per_batch
            if cutlass.const_expr(self.params._raster_along_m):
                group = linear // (swizzle * num_cluster_m)
                width = cutlass.min(swizzle, num_cluster_n - group * swizzle)
                local = linear - group * swizzle * num_cluster_m
                cluster_m = local // width
                cluster_n = group * swizzle + local - cluster_m * width
            else:
                group = linear // (swizzle * num_cluster_n)
                width = cutlass.min(swizzle, num_cluster_m - group * swizzle)
                local = linear - group * swizzle * num_cluster_n
                cluster_n = local // width
                cluster_m = group * swizzle + local - cluster_n * width
            return (
                cluster_m * self.params.cluster_shape_mn[0] + x_idx,
                cluster_n * self.params.cluster_shape_mn[1] + y_idx,
                batch_l,
            )

    @dsl_user_op
    def work_tile_info_from_clc_response(
        self,
        result_addr: cute.Pointer,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> WorkTileInfo:
        """
        Simulates parsing CLC response data in Python.
        result_addr: 16-byte response data (simulating shared memory access)
        """
        m_idx, n_idx, l_idx, vld = cute.arch.clc_response(result_addr, loc=loc, ip=ip)

        # Cross-proxy fence to ensure generic proxy access are seen by later
        # async proxy operations.
        if self.insert_fence:
            cute.arch.fence_proxy("async.shared", space="cta", loc=loc, ip=ip)

        m_idx, n_idx, l_idx = self._swizzle_and_rasterize(m_idx, n_idx, l_idx, loc=loc, ip=ip)
        cta_idx_in_cluster, cta_idy_in_cluster, _ = self.cta_id_in_cluster  # type: ignore[misc]
        cur_tile_coord = (m_idx + cta_idx_in_cluster, n_idx + cta_idy_in_cluster, l_idx)

        return WorkTileInfo(cur_tile_coord, vld)

    @dsl_user_op
    @cute.jit
    def get_current_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> WorkTileInfo:
        smem_addr = self._clc_response_ptr
        work_tile = self.work_tile_info_from_clc_response(smem_addr, loc=loc, ip=ip)
        return work_tile

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(self, *, loc: Any = None, ip: Any = None) -> WorkTileInfo:
        bidx, bidy, bidz = self._block_idx
        # Subtract cta_id_in_cluster from block_idx because swizzle_and_rasterize expects coordinates to be
        # those of the leader CTA in the cluster.
        cta_idx_in_cluster, cta_idy_in_cluster, _ = self.cta_id_in_cluster  # type: ignore[misc]
        m_idx = bidx - cta_idx_in_cluster
        n_idx = bidy - cta_idy_in_cluster
        l_idx = bidz
        m_idx, n_idx, l_idx = self._swizzle_and_rasterize(m_idx, n_idx, l_idx, loc=loc, ip=ip)
        cur_tile_coord = (m_idx + cta_idx_in_cluster, n_idx + cta_idy_in_cluster, l_idx)
        return WorkTileInfo(cur_tile_coord, Boolean(True))

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(
        self,
        mbarrier_addr: cute.Pointer,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        # Query new work tile
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(mbarrier_addr, self._clc_response_ptr, loc=loc, ip=ip)
        self._num_tiles_executed += Int32(1)

    @property
    @cute.jit
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed
