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

# ruff: noqa: I001, E501

import inspect
from typing import Optional, Tuple

from cutlass.cutlass_dsl import (
    Boolean,
    Integer,
    Int32,
    min,
    extract_mlir_values,
    new_from_mlir_values,
    dsl_user_op,
    const_expr,
)
from cutlass._mlir import ir
import cutlass.cute as cute

##############################################################################
# Static persistent tile scheduler
##############################################################################


class WorkTileInfo:
    """A class to represent information about a work tile.

    :ivar tile_idx: The index of the tile.
    :type tile_idx: cute.Coord
    :ivar is_valid_tile: Whether the tile is valid.
    :type is_valid_tile: Boolean
    """

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean):
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.tile_idx)
        values.extend(extract_mlir_values(self.is_valid_tile))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 4
        new_tile_idx = new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    @cute.jit
    def is_valid_tile(self) -> Boolean:
        """Check latest tile returned by the scheduler is valid or not. Any scheduling
        requests after all tasks completed will return an invalid tile.

        :return: The validity of the tile.
        :rtype: Boolean
        """
        return self._is_valid_tile

    @property
    @cute.jit
    def tile_idx(self) -> cute.Coord:
        """
        Get the index of the tile.

        :return: The index of the tile.
        :rtype: cute.Coord
        """
        return self._tile_idx


class PersistentTileSchedulerParams:
    """A class to represent parameters for a persistent tile scheduler.

    This class is designed to manage and compute the layout of clusters and tiles
    in a batched gemm problem.

    :ivar cluster_shape_mn: Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
    :type cluster_shape_mn: tuple
    :ivar problem_layout_ncluster_mnl: Layout of the problem in terms of
        number of clusters in (m, n, l) dimensions.
    :type problem_layout_ncluster_mnl: cute.Layout
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
        Initializes the PersistentTileSchedulerParams with the given parameters.

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
        self.raster_along_m = raster_along_m
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

        # Apply swizzle if swizzle_size > 1
        if swizzle_size > 1:
            problem_shape_ncluster_mnl = cute.round_up(
                self.problem_layout_ncluster_mnl.shape,
                (1, swizzle_size, 1) if raster_along_m else (swizzle_size, 1, 1),
            )

            if raster_along_m:
                self.problem_layout_ncluster_mnl = cute.make_layout(
                    (
                        problem_shape_ncluster_mnl[0],  # type: ignore[index]
                        (swizzle_size, problem_shape_ncluster_mnl[1] // swizzle_size),  # type: ignore[index, operator]
                        problem_shape_ncluster_mnl[2],  # type: ignore[index]
                    ),
                    stride=(
                        swizzle_size,
                        (1, swizzle_size * problem_shape_ncluster_mnl[0]),  # type: ignore[index]
                        problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],  # type: ignore[index, operator]
                    ),
                    loc=loc,
                    ip=ip,
                )
            else:
                self.problem_layout_ncluster_mnl = cute.make_layout(
                    (
                        (swizzle_size, problem_shape_ncluster_mnl[0] // swizzle_size),  # type: ignore[index, operator]
                        problem_shape_ncluster_mnl[1],  # type: ignore[index]
                        problem_shape_ncluster_mnl[2],  # type: ignore[index]
                    ),
                    stride=(
                        (1, swizzle_size * problem_shape_ncluster_mnl[1]),  # type: ignore[index]
                        swizzle_size,
                        problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],  # type: ignore[index, operator]
                    ),
                    loc=loc,
                    ip=ip,
                )

        # Create FastDivmod divisors (only when swizzle_size == 1 for correctness)
        # FastDivmod assumes simple col-major layout, incompatible with swizzled layouts
        if swizzle_size == 1:
            _problem_layout_size = cute.size(self.problem_layout_ncluster_mnl, loc=loc, ip=ip)
            cluster_count_m = self.problem_layout_ncluster_mnl.shape[0]
            cluster_count_n = self.problem_layout_ncluster_mnl.shape[1]

            if raster_along_m:
                cluster_count_major = cluster_count_m
                cluster_count_minor = cluster_count_n
            else:
                cluster_count_major = cluster_count_n
                cluster_count_minor = cluster_count_m

            # cluster_shape_major_fdd: Used to decode work_unit_id to cluster coordinates
            self.cluster_shape_major_fdd = cute.fast_divmod_create_divisor(
                cluster_count_major, loc=loc, ip=ip
            )

            # cluster_shape_minor_fdd: Used for the second level decomposition
            self.cluster_shape_minor_fdd = cute.fast_divmod_create_divisor(
                cluster_count_minor, loc=loc, ip=ip
            )
        else:
            # FastDivmod not applicable with swizzling, set to None
            self.cluster_shape_major_fdd = None
            self.cluster_shape_minor_fdd = None

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values, self._values_pos = [], []
        for obj in [
            self.problem_shape_ntile_mnl,
            self._cluster_shape_mnk,
            self.swizzle_size,
            self.raster_along_m,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))

        # Add FastDivmod divisors to MLIR values for Host->Device transfer
        # Only add non-None values to avoid MLIR type errors
        fastdivmod_values = []
        fastdivmod_indices = []  # Track which FastDivmod objects are present
        fastdivmod_lengths = []  # Track serialized MLIR value count per FDD

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
                fastdivmod_lengths.append(len(fdd_values))

        values += fastdivmod_values
        self._values_pos.append(
            len(fastdivmod_indices)
        )  # Store count of FastDivmod objects, not values
        self._fastdivmod_indices = fastdivmod_indices  # Store for reconstruction
        self._fastdivmod_lengths = fastdivmod_lengths  # Per-FDD value count

        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "PersistentTileSchedulerParams":
        obj_list = []
        values_copy = list(values)  # Make a copy to avoid modifying original

        # Reconstruct original objects from MLIR values
        for obj, n_items in zip(
            [
                self.problem_shape_ntile_mnl,
                self._cluster_shape_mnk,
                self.swizzle_size,
                self.raster_along_m,
            ],
            self._values_pos[:-1],  # Exclude FastDivmod count
        ):
            obj_list.append(new_from_mlir_values(obj, values_copy[:n_items]))
            values_copy = values_copy[n_items:]

        # Create new params object by calling __init__ with reconstructed values
        # This properly recreates layouts and other derived attributes in the device context
        new_params = PersistentTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

        # Restore FastDivmod divisors from remaining values
        fdd_names = ["cluster_shape_major_fdd", "cluster_shape_minor_fdd"]

        if hasattr(self, "_fastdivmod_indices") and len(self._fastdivmod_indices) > 0:
            # Override the FastDivmod divisors created by __init__ with reconstructed ones.
            # FastDivmodDivisor now emits multiple MLIR values per object (see issue #3243);
            # use the per-FDD length recorded during __extract_mlir_values__ to slice the
            # tail of values_copy without re-emitting IR.
            fdd_tail_offset = 0
            for original_index, n_fdd in zip(self._fastdivmod_indices, self._fastdivmod_lengths):
                fdd_name = fdd_names[original_index]
                original_fdd = getattr(self, fdd_name)
                if original_fdd is None:
                    continue
                end = fdd_tail_offset + n_fdd
                if end > len(values_copy):
                    raise ValueError(
                        f"FastDivmod values out of range for {fdd_name}: "
                        f"need {end}, have {len(values_copy)}"
                    )
                reconstructed_fdd = new_from_mlir_values(
                    original_fdd, values_copy[fdd_tail_offset:end]
                )
                setattr(new_params, fdd_name, reconstructed_fdd)
                fdd_tail_offset = end
            if fdd_tail_offset != len(values_copy):
                raise ValueError(
                    "Unexpected trailing FastDivmod MLIR values: "
                    f"consumed {fdd_tail_offset}, have {len(values_copy)}"
                )

        return new_params

    @dsl_user_op
    def get_grid_shape(
        self,
        max_active_clusters: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Integer, Integer, Integer]:
        """
        Computes the grid shape based on the maximum active clusters allowed.

        :param max_active_clusters: The maximum number of active clusters that
            can run in one wave.
        :type max_active_clusters: Int32

        :return: A tuple containing the grid shape in (m, n, persistent_clusters).
            - m: self.cluster_shape_m.
            - n: self.cluster_shape_n.
            - persistent_clusters: Number of persistent clusters that can run.
        """

        # Total ctas in problem size
        num_ctas_mnl = tuple(
            cute.size(x) * y
            for x, y in zip(self.problem_layout_ncluster_mnl.shape, self.cluster_shape_mn)
        ) + (self.problem_layout_ncluster_mnl.shape[2],)

        num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)

        num_ctas_per_cluster = cute.size(self.cluster_shape_mn, loc=loc, ip=ip)
        # Total ctas that can run in one wave
        num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster

        num_persistent_ctas = min(num_ctas_in_problem, num_ctas_per_wave)
        num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster

        return (*self.cluster_shape_mn, num_persistent_clusters)


# Set explicit signature for Sphinx documentation to avoid issues with @dsl_user_op decorator
PersistentTileSchedulerParams.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
    [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
)


class StaticPersistentTileScheduler:
    """A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

    :ivar params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
    :type params: PersistentTileSchedulerParams
    :ivar num_persistent_clusters: Number of persistent clusters that can be launched
    :type num_persistent_clusters: Int32
    :ivar cta_id_in_cluster: ID of the CTA within its cluster
    :type cta_id_in_cluster: cute.Coord
    :ivar _num_tiles_executed: Counter for executed tiles
    :type _num_tiles_executed: Int32
    :ivar _current_work_linear_idx: Current cluster index
    :type _current_work_linear_idx: Int32
    """

    def __init__(
        self,
        params: PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
    ):
        """
        Initializes the StaticPersistentTileScheduler with the given parameters.

        :param params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
        :type params: PersistentTileSchedulerParams
        :param num_persistent_clusters: Number of persistent clusters that can be launched.
        :type num_persistent_clusters: Int32
        :param current_work_linear_idx: Current cluster index.
        :type current_work_linear_idx: Int32
        :param cta_id_in_cluster: ID of the CTA within its cluster.
        :type cta_id_in_cluster: cute.Coord
        :param num_tiles_executed: Counter for executed tiles.
        :type num_tiles_executed: Int32
        """
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))

        # CRITICAL: Also extract FastDivmod divisors from params
        values.extend(extract_mlir_values(self.params))

        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "StaticPersistentTileScheduler":
        assert len(values) >= 6
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])

        # Reconstruct params with FastDivmod divisors
        params_values = values[6:]  # Remaining values are from params
        new_params = new_from_mlir_values(self.params, params_values)

        return StaticPersistentTileScheduler(
            new_params,  # Use reconstructed params with FastDivmod divisors
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "StaticPersistentTileScheduler":
        """Initialize the static persistent tile scheduler.

        :param params: Parameters for the persistent
            tile scheduler.
        :type params: PersistentTileSchedulerParams
        :param block_idx: The 3d block index in the format (bidx, bidy, bidz).
        :type block_idx: Tuple[Integer, Integer, Integer]
        :param grid_dim: The 3d grid dimensions for kernel launch.
        :type grid_dim: Tuple[Integer, Integer, Integer]

        :return: A StaticPersistentTileScheduler object.
        :rtype: StaticPersistentTileScheduler
        """

        # Calculate the number of persistent clusters by dividing the total grid size
        # by the number of CTAs per cluster
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx

        # Initialize workload index equals to the cluster index in the grid
        current_work_linear_idx = Int32(bidz)

        # CTA id in the cluster
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        return StaticPersistentTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: PersistentTileSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Integer, Integer, Integer]:
        """Calculates the grid shape to be launched on GPU using problem shape,
        threadblock shape, and active cluster size.

        :param params: Parameters for grid shape calculation.
        :type params: PersistentTileSchedulerParams
        :param max_active_clusters: Maximum active clusters allowed.
        :type max_active_clusters: Int32

        :return: The calculated 3d grid shape.
        :rtype: Tuple[Integer, Integer, Integer]
        """

        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    # private method
    def _get_current_work_for_linear_idx(
        self,
        current_work_linear_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> WorkTileInfo:
        """Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

        :param current_work_linear_idx: The linear index of the current work.
        :type current_work_linear_idx: Int32

        :return: An object containing information about the current tile coordinates
            and validity status.
        :rtype: WorkTileInfo
        """

        is_valid = current_work_linear_idx < cute.size(
            self.params.problem_layout_ncluster_mnl, loc=loc, ip=ip
        )

        # Choose coordinate calculation method based on swizzle configuration
        if self.params.swizzle_size == 1:
            # Use FastDivmod optimization for non-swizzled layouts
            cur_cluster_coord = self._get_cluster_work_idx_with_fastdivmod(
                current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            # Use get_flat_coord for swizzled layouts (FastDivmod doesn't support them)
            cur_cluster_coord = self.params.problem_layout_ncluster_mnl.get_flat_coord(
                current_work_linear_idx, loc=loc, ip=ip
            )

        cur_tile_coord = tuple(
            cute.arch.make_warp_uniform(Int32(x) * Int32(z) + Int32(y))
            for x, y, z in zip(
                cur_cluster_coord,
                self.cta_id_in_cluster,  # type: ignore[arg-type]
                (*self.params.cluster_shape_mn, Int32(1)),
            )
        )

        return WorkTileInfo(cur_tile_coord, cute.arch.make_warp_uniform(is_valid))

    def _get_cluster_work_idx_with_fastdivmod(
        self,
        current_work_linear_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Int32, Int32, Int32]:
        """
        FastDivmod optimized CLUSTER coordinate calculation.

        CRITICAL: This should mimic problem_layout_ncluster_mnl.get_hier_coord()
        which returns CLUSTER coordinates, not tile coordinates!

        :param current_work_linear_idx: Linear index in the work space
        :type current_work_linear_idx: Int32
        :return: Cluster coordinates (m, n, l) or None if FastDivmod not available
        :rtype: Tuple[Int32, Int32, Int32] or None
        """

        # Step 1: Decode current_work_linear_idx using FastDivmod objects
        # The layout structure is: problem_layout_ncluster_mnl has shape (cluster_count_m, cluster_count_n, batch_count)
        # current_work_linear_idx needs to be decomposed into (batch_l, cluster_minor, cluster_major) in little-endian order

        # First, get cluster_major using cluster_shape_major_fdd
        cluster_minor_batch, cluster_major = divmod(
            current_work_linear_idx, self.params.cluster_shape_major_fdd
        )

        # Then decode cluster_minor_batch to get cluster_minor and batch_l using FastDivmod
        batch_l, cluster_minor = divmod(cluster_minor_batch, self.params.cluster_shape_minor_fdd)

        if self.params.raster_along_m:
            cluster_m = cluster_major
            cluster_n = cluster_minor
        else:
            cluster_m = cluster_minor
            cluster_n = cluster_major

        return (cluster_m, cluster_n, batch_l)

    @dsl_user_op
    @cute.jit
    def get_current_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> WorkTileInfo:
        return self._get_current_work_for_linear_idx(self._current_work_linear_idx, loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def initial_work_tile_info(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(
        self,
        *,
        advance_count: int = 1,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        self._current_work_linear_idx += Int32(advance_count) * Int32(self.num_persistent_clusters)
        self._num_tiles_executed += Int32(1)

    @property
    @cute.jit
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


class StaticPersistentRuntimeTileScheduler(StaticPersistentTileScheduler):
    """A scheduler for static persistent runtime tile execution in CUTLASS/CuTe kernels.
    This scheduler will always launch all the SMs and the scheduler will generate the real tile info for each SM.

    :ivar params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
    :type params: PersistentTileSchedulerParams
    :ivar num_persistent_clusters: Number of persistent clusters that can be launched
    :type num_persistent_clusters: Int32
    :ivar cta_id_in_cluster: ID of the CTA within its cluster
    :type cta_id_in_cluster: cute.Coord
    :ivar _num_tiles_executed: Counter for executed tiles
    :type _num_tiles_executed: Int32
    :ivar _current_work_linear_idx: Current cluster index
    :type _current_work_linear_idx: Int32
    """

    def __init__(
        self,
        params: PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        inner_mode: int = 1,
    ):
        """
        Initializes the StaticPersistentRuntimeTileScheduler with the given parameters.

        :param params: Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
        :type params: PersistentTileSchedulerParams
        :param num_persistent_clusters: Number of persistent clusters that can be launched.
        :type num_persistent_clusters: Int32
        :param current_work_linear_idx: Current cluster index.
        :type current_work_linear_idx: Int32
        :param cta_id_in_cluster: ID of the CTA within its cluster.
        :type cta_id_in_cluster: cute.Coord
        :param num_tiles_executed: Counter for executed tiles.
        :type num_tiles_executed: Int32
        :param inner_mode: The inner mode along which the linear index will be decomposed first.
        :type inner_mode: int
        """
        super().__init__(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
        )
        if inner_mode not in [0, 1]:
            raise ValueError(
                f"inner_mode must be 0(for M mode) or 1(for N mode), but got {inner_mode}"
            )
        self.inner_mode = inner_mode

    def __new_from_mlir_values__(
        self, values: list[ir.Value]
    ) -> "StaticPersistentRuntimeTileScheduler":
        assert len(values) >= 6
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])

        # Reconstruct params with FastDivmod divisors (same as parent class)
        params_values = values[6:]  # Remaining values are from params
        new_params = new_from_mlir_values(self.params, params_values)

        return StaticPersistentRuntimeTileScheduler(
            new_params,  # Use reconstructed params with FastDivmod divisors
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            self.inner_mode,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        inner_mode: int = 1,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "StaticPersistentRuntimeTileScheduler":
        """Initialize the static persistent tile scheduler.

        :param params: Parameters for the persistent
            tile scheduler.
        :type params: PersistentTileSchedulerParams
        :param block_idx: The 3d block index in the format (bidx, bidy, bidz).
        :type block_idx: Tuple[Integer, Integer, Integer]
        :param grid_dim: The 3d grid dimensions for kernel launch.
        :type grid_dim: Tuple[Integer, Integer, Integer]
        :param inner_mode: The inner mode along which the linear index will be decomposed first.
        :type inner_mode: int

        :return: A StaticPersistentRuntimeTileScheduler object.
        :rtype: StaticPersistentRuntimeTileScheduler
        """

        # Calculate the number of persistent clusters by dividing the total grid size
        # by the number of CTAs per cluster
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx

        # Initialize workload index equals to the cluster index in the grid
        current_work_linear_idx = Int32(bidz)

        # CTA id in the cluster
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        # Initialize number of tiles executed to zero
        num_tiles_executed = Int32(0)
        return StaticPersistentRuntimeTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
            inner_mode,
        )

    # private method
    def _get_current_work_for_linear_idx(
        self,
        current_work_linear_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> WorkTileInfo:
        """Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

        :param current_work_linear_idx: The linear index of the current work.
        :type current_work_linear_idx: Int32

        :return: An object containing information about the current tile coordinates
            and validity status.
        :rtype: WorkTileInfo
        """
        ntile_shape = self.params.problem_layout_ncluster_mnl.shape
        int_max = 2147483647
        if const_expr(self.inner_mode == 1):
            ntile_layout = cute.make_layout((int_max, ntile_shape[1]), stride=(ntile_shape[1], 1))
        else:
            ntile_layout = cute.make_layout((ntile_shape[0], int_max), stride=(1, ntile_shape[0]))
        cluster_tile_coord_mn = ntile_layout.get_hier_coord(current_work_linear_idx)
        cur_tile_coord = (
            cluster_tile_coord_mn[0],
            cluster_tile_coord_mn[1],
            Int32(0),
        )

        # it is determined by kernel implementation
        is_valid = Boolean(True)

        return WorkTileInfo(cur_tile_coord, is_valid)
