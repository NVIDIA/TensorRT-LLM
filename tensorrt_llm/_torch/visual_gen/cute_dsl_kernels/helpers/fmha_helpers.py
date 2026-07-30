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

# ruff: noqa: I001, E501, F841

import enum
from typing import Tuple, Optional
import cutlass
from cutlass.cute.typing import Boolean
from cutlass._mlir.dialects import llvm, vector

from cutlass.cutlass_dsl import (
    Int32,
    Float32,
    T,
    min,
    extract_mlir_values,
    new_from_mlir_values,
    dsl_user_op,
)
from cutlass.utils.hardware_info import HardwareInfo
from .static_persistent_tile_scheduler import WorkTileInfo
import cutlass.cute as cute


##############################################################################
# Fmha static tile scheduler
##############################################################################


class FmhaStaticTileSchedulerParams:
    """A class to represent parameters for the FMHA (Fused Multi-Head Attention) static tile scheduler.

    This class holds the configuration parameters needed to initialize and configure
    the tile scheduler for FMHA operations.

    :ivar is_persistent: Whether to use persistent kernel mode.
    :type is_persistent: bool
    :ivar problem_shape_mhb: Problem shape in (M, H, B) format.
    :type problem_shape_mhb: cute.Shape
    """

    def __init__(
        self,
        is_persistent: bool,
        problem_shape_mhb: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        """
        Initializes the FmhaStaticTileSchedulerParams with the given parameters.

        :param is_persistent: Whether to use persistent kernel mode.
        :type is_persistent: bool
        :param problem_shape_mhb: Problem shape in (M, H, B) format.
        :type problem_shape_mhb: cute.Shape
        """
        self.is_persistent = is_persistent
        self.problem_shape_mhb = problem_shape_mhb
        self._loc = loc
        self._ip = ip

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.problem_shape_mhb]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.problem_shape_mhb], self._values_pos):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FmhaStaticTileSchedulerParams(self.is_persistent, *(tuple(obj_list)), loc=self._loc)


class FmhaStaticTileScheduler:
    """A static tile scheduler for FMHA (Fused Multi-Head Attention) operations.

    This class manages the scheduling of work tiles for FMHA kernels, supporting
    both persistent and non-persistent kernel modes. It tracks the current work
    position and advances through the problem space efficiently.

    :ivar _params: Scheduler parameters.
    :type _params: FmhaStaticTileSchedulerParams
    :ivar _blk_coord: Block coordinates.
    :type _blk_coord: cute.Coord
    :ivar _grid_shape: Grid shape for the kernel.
    :type _grid_shape: cute.Shape
    :ivar _is_persistent: Whether to use persistent kernel mode.
    :type _is_persistent: bool
    :ivar _current_work_linear_idx: Current linear work index.
    :type _current_work_linear_idx: Int32
    :ivar _problem_shape_mhb: Problem shape in (M, H, B) format.
    :type _problem_shape_mhb: cute.Layout
    :ivar _num_blocks: Number of blocks in the problem.
    :type _num_blocks: Int32
    :ivar _is_first_block: Whether this is the first block.
    :type _is_first_block: bool
    :ivar num_persistent_sm: Number of persistent SMs.
    :type num_persistent_sm: Int32
    """

    def __init__(
        self,
        params: FmhaStaticTileSchedulerParams,
        current_work_linear_idx: Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        """
        Initializes the FmhaStaticTileScheduler with the given parameters.

        :param params: Scheduler parameters.
        :type params: FmhaStaticTileSchedulerParams
        :param current_work_linear_idx: Current linear work index.
        :type current_work_linear_idx: Int32
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param grid_shape: Grid shape for the kernel.
        :type grid_shape: cute.Shape
        """
        self._params = params
        self._blk_coord = blk_coord
        self._grid_shape = grid_shape
        self._is_persistent = params.is_persistent
        self._current_work_linear_idx = current_work_linear_idx
        self._problem_shape_mhb = cute.make_layout(params.problem_shape_mhb, loc=loc, ip=ip)
        self._num_blocks = cute.size(self._problem_shape_mhb, loc=loc, ip=ip)
        self._is_first_block = True
        self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        self._loc = loc
        self._ip = ip

    # called by host
    @staticmethod
    def get_grid_shape(
        params: FmhaStaticTileSchedulerParams,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        """
        Determine the grid shape for the FMHA kernel.

        For persistent kernels, the grid shape is limited by the number of SMs
        (Streaming Multiprocessors) available on the device. For non-persistent
        kernels, the grid shape matches the problem shape.

        :param params: Scheduler parameters.
        :type params: FmhaStaticTileSchedulerParams

        :return: Grid shape as (M, H, B) tuple.
        :rtype: cute.Shape
        """
        if params.is_persistent:
            hardware_info = HardwareInfo()
            sm_count = hardware_info.get_device_multiprocessor_count()
            return (
                min(sm_count, cute.size(params.problem_shape_mhb, loc=loc, ip=ip)),
                1,
                1,
            )
        else:
            return params.problem_shape_mhb

    @staticmethod
    def check_valid_work_for_seqlen_q(
        q_tiler: int,
        current_idx: Int32,
        seqlen_q: Int32,
    ) -> Boolean:
        """
        Check if the current work index is valid for the given query sequence length.

        This method verifies that the current work tile index multiplied by the
        query tiler size is within the bounds of the query sequence length.

        :param q_tiler: Query tiler size.
        :type q_tiler: int
        :param current_idx: Current work index.
        :type current_idx: Int32
        :param seqlen_q: Query sequence length.
        :type seqlen_q: Int32

        :return: True if the work is valid, False otherwise.
        :rtype: Boolean
        """
        return current_idx * q_tiler < seqlen_q

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        """
        Get information about the current work tile.

        Determines if the current work is valid and computes the tile coordinates
        based on whether the kernel is persistent or non-persistent.

        :return: WorkTileInfo containing tile coordinates and validity flag.
        :rtype: WorkTileInfo
        """
        is_valid = (
            self._current_work_linear_idx < self._num_blocks
            if self._is_persistent
            else self._is_first_block
        )

        blk_coord = (0, 0, 0)
        if self._is_persistent:
            blk_coord = self._problem_shape_mhb.get_hier_coord(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = self._blk_coord

        # cur_tile_coord is (mid, 0, (bid, hid))
        cur_tile_coord = (
            blk_coord[0],
            0,
            (blk_coord[1], blk_coord[2]),
        )

        return WorkTileInfo(cur_tile_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        """
        Get the initial work tile information.

        :return: Initial WorkTileInfo.
        :rtype: WorkTileInfo
        """
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        """
        Advance to the next work tile.

        For persistent kernels, advances by the number of persistent SMs.
        For non-persistent kernels, marks that the first block has been processed.

        :param advance_count: Number of steps to advance (default: 1).
        :type advance_count: int
        """
        if self._is_persistent:
            self._current_work_linear_idx += advance_count * self.num_persistent_sm
        self._is_first_block = False

    def __extract_mlir_values__(self):
        # Only pass mutable per-iteration state as scf.while block arguments.
        # _params and _grid_shape are loop-invariant and captured from outer
        # scope, keeping block arg count low (4 instead of 10).
        values = extract_mlir_values(self._current_work_linear_idx)
        values.extend(extract_mlir_values(self._blk_coord))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 4
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[0]]
        )
        new_blk_coord = new_from_mlir_values(self._blk_coord, values[1:4])
        return FmhaStaticTileScheduler(
            self._params, new_current_work_linear_idx, new_blk_coord, self._grid_shape
        )


def create_fmha_static_tile_scheduler(
    params: FmhaStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> FmhaStaticTileScheduler:
    """
    Create a new FMHA static tile scheduler.

    :param params: Scheduler parameters.
    :type params: FmhaStaticTileSchedulerParams
    :param blk_coord: Block coordinates.
    :type blk_coord: cute.Coord
    :param grid_shape: Grid shape.
    :type grid_shape: cute.Shape

    :return: New FmhaStaticTileScheduler instance.
    :rtype: FmhaStaticTileScheduler
    """
    return FmhaStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


def create_fmha_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_mhb: cute.Shape,
) -> FmhaStaticTileSchedulerParams:
    """
    Create FMHA static tile scheduler parameters.

    :param is_persistent: Whether to use persistent kernel mode.
    :type is_persistent: bool
    :param problem_shape_mhb: Problem shape in (M, H, B) format.
    :type problem_shape_mhb: cute.Shape

    :return: New FmhaStaticTileSchedulerParams instance.
    :rtype: FmhaStaticTileSchedulerParams
    """
    return FmhaStaticTileSchedulerParams(is_persistent, problem_shape_mhb)


def compute_grid(
    o_shape: cute.Shape,
    cta_tiler: Tuple[int, int, int],
    is_persistent: bool,
    is_2cta: bool = False,
) -> Tuple[FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
    """
    Compute grid parameters for FMHA operation.

    This function calculates the appropriate grid shape and scheduler parameters
    based on the output tensor shape, CTA (Cooperative Thread Array) tiler,
    and whether to use persistent kernel mode.

    The output tensor o has shape (s, d, ((h_r, h_k), b)) where:
    - s: sequence length
    - d: head dimension
    - h_r: number of heads for query
    - h_k: number of heads for key
    - b: batch size

    :param o_shape: Output tensor shape for grid computation.
    :type o_shape: cute.Shape
    :param cta_tiler: CTA tiler dimensions (M, N, K).
    :type cta_tiler: Tuple[int, int, int]
    :param is_persistent: Whether to use persistent kernel mode.
    :type is_persistent: bool
    :param is_2cta: Whether to use 2CTA mode.
    :type is_2cta: bool

    :return: Tuple of (scheduler_params, grid_shape).
    :rtype: Tuple[FmhaStaticTileSchedulerParams, Tuple[int, int, int]]
    """
    tile_sched_params = create_fmha_static_tile_scheduler_params(
        is_persistent,
        (
            cute.round_up(cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]), 2 if is_2cta else 1),
            cute.size(o_shape[2][0]),
            cute.size(o_shape[2][1]),
        ),
    )
    grid = FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)

    return tile_sched_params, grid


##############################################################################
# Fused Mask
##############################################################################


class MaskEnum(enum.Enum):
    """Enumeration of mask types for FMHA operations.

    - RESIDUAL_MASK: Residual mask for handling variable sequence lengths
    - WINDOW_MASK: Window mask for attention which also includes causal and no mask
    - WINDOW_MASK_INFERENCE: Same as the window mask, but has the limitation that the end of q is aligned with the end of k
    - WINDOW_MASK_BWD: Window mask for backward pass
    - WINDOW_MASK_BWD_INFERENCE: Same as the window mask for backward pass, but has the limitation that the end of q is aligned with the end of k
    """

    RESIDUAL_MASK = enum.auto()
    RESIDUAL_MASK_BWD = enum.auto()
    WINDOW_MASK = enum.auto()
    WINDOW_MASK_INFERENCE = enum.auto()
    WINDOW_MASK_BWD = enum.auto()
    WINDOW_MASK_BWD_INFERENCE = enum.auto()


class FusedMask:
    """A fused mask implementation for FMHA operations.

    This class handles different types of attention masks including no mask,
    residual mask for variable sequence lengths, and causal mask for
    autoregressive attention patterns.

    The class provides methods to:
    - Calculate trip counts for different mask types
    - Apply masks to attention scores
    - Handle masked and unmasked trip calculations
    """

    def get_trip_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of trips needed for the current block.

        The trip count depends on the mask type and the block coordinates.
        For causal masks, it considers the autoregressive constraint.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of trips needed.
        :rtype: Int32
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(mask_type == MaskEnum.RESIDUAL_MASK):
            result = cute.ceil_div(seqlen_k, tile_shape[1])
        if cutlass.const_expr(mask_type is MaskEnum.RESIDUAL_MASK_BWD):
            result = cute.ceil_div(seqlen_q, tile_shape[0])
        if cutlass.const_expr(
            mask_type == MaskEnum.WINDOW_MASK or mask_type == MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is None):
                result = cute.ceil_div(seqlen_k, tile_shape[1])
            else:
                max_idx_q = (blk_coord[0] + 1) * tile_shape[0]
                idx_k = max_idx_q + offset + window_size_right
                tmp_blocks_k = cute.ceil_div(idx_k, tile_shape[1])
                max_blocks_k = cute.ceil_div(seqlen_k, tile_shape[1])
                result = min(max_blocks_k, tmp_blocks_k)
        if cutlass.const_expr(
            mask_type == MaskEnum.WINDOW_MASK_BWD or mask_type == MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is None):
                result = cute.ceil_div(seqlen_q, tile_shape[0])
            else:
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1]
                idx_k = max_idx_k + offset + window_size_left
                tmp_blocks_q = cute.ceil_div(idx_k, tile_shape[0])
                max_blocks_q = cute.ceil_div(seqlen_q, tile_shape[0])
                result = min(max_blocks_q, tmp_blocks_q)
        start_block = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        result = result - start_block
        return result

    @cute.jit
    def get_trip_start(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Get the start of the trip for the current block.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        """
        result = 0
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK or mask_type is MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = blk_coord[0] * tile_shape[0]
                idx_k = min_idx_q + offset - window_size_left
                tmp_blocks_k = idx_k // tile_shape[1]
                result = max(tmp_blocks_k, result)
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK_BWD or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = blk_coord[1] * tile_shape[1]
                idx_q = min_idx_k + offset - window_size_right
                tmp_blocks_q = idx_q // tile_shape[0]
                result = max(tmp_blocks_q, result)
        return result

    @cute.jit
    def get_leading_mask_id(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Int32, Int32]:
        """
        Get the begin and end tile idx for the leading mask.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Tuple of (begin, end) tile idx for the leading mask.
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        leading_mask_begin = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        leading_mask_end = leading_mask_begin
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK or mask_type is MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_left is not None):
                min_idx_q = (blk_coord[0] + 1) * tile_shape[0] + offset - window_size_left
                leading_mask_end = min(
                    cute.ceil_div(min_idx_q, tile_shape[1]) - 1,
                    trip_count + leading_mask_begin - 1,
                )
            else:
                leading_mask_end = leading_mask_begin - 1
        elif cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK_BWD or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset - window_size_right
                leading_mask_end = cute.ceil_div(min_idx_k, tile_shape[0]) - 1
            else:
                leading_mask_end = leading_mask_begin - 1
        return leading_mask_begin, leading_mask_end

    @cute.jit
    def get_trailing_mask_id(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Tuple[Optional[Int32], Optional[Int32]]:
        """
        Get the begin and end tile idx for the trailing mask.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Tuple of (begin, end) tile idx for the trailing mask.
        :rtype: Tuple[Int32, Int32]
        """
        offset = 0
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_INFERENCE):
            offset = seqlen_k - seqlen_q
        if cutlass.const_expr(mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE):
            offset = seqlen_q - seqlen_k
        trip_start = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trip_count = FusedMask.get_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        trailing_mask_begin, trailing_mask_end = None, None
        if cutlass.const_expr(
            mask_type is MaskEnum.WINDOW_MASK or mask_type is MaskEnum.WINDOW_MASK_INFERENCE
        ):
            if cutlass.const_expr(window_size_right is not None):
                min_idx_q = blk_coord[0] * tile_shape[0] + offset + window_size_right
                trailing_mask_begin = min(min_idx_q // tile_shape[1], trip_count + trip_start - 1)
                trailing_mask_end = trip_count + trip_start - 1
            else:
                # last tile, we always apply mask on it regardless whether it's a residual tile
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1
        else:
            if cutlass.const_expr(window_size_left is not None):
                min_idx_k = blk_coord[1] * tile_shape[1] + offset + window_size_left + 1
                max_idx_k = (blk_coord[1] + 1) * tile_shape[1] + offset + window_size_left
                trailing_mask_begin = min(
                    cute.ceil_div(min_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
                trailing_mask_end = min(
                    cute.ceil_div(max_idx_k, tile_shape[0]) - 1,
                    trip_count + trip_start - 1,
                )
            else:
                # last tile, we always apply mask on it regardless whether it's a residual tile
                trailing_mask_begin = trip_count + trip_start - 1
                trailing_mask_end = trip_count + trip_start - 1

        return trailing_mask_begin, trailing_mask_end

    @cute.jit
    def get_masked_leading_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of masked trips for the leading mask.

        This is used for blocks that need special handling due to masking.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of masked trips.
        :rtype: Int32
        """
        result = 0
        if cutlass.const_expr(
            mask_type is not MaskEnum.RESIDUAL_MASK and mask_type is not MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                leading_mask_begin, leading_mask_end = FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                result = max(leading_mask_end - leading_mask_begin + 1, 0)

        return result

    @cute.jit
    def get_masked_trailing_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        rem_count: Optional[Int32] = 0,
    ) -> Int32:
        """
        Calculate the number of masked trips for the trailing mask.

        This is used for blocks that need special handling due to masking.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param rem_count: Remaining count from previous calculations.
        :type rem_count: Int32

        :return: Number of masked trips.
        :rtype: Int32
        """
        result = 0

        if cutlass.const_expr(
            mask_type is not MaskEnum.RESIDUAL_MASK and mask_type is not MaskEnum.RESIDUAL_MASK_BWD
        ):
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                trailing_mask_begin, trailing_mask_end = FusedMask.get_trailing_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                leading_mask_begin, leading_mask_end = FusedMask.get_leading_mask_id(
                    mask_type,
                    blk_coord,
                    tile_shape,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                if cutlass.const_expr(
                    trailing_mask_begin is not None and trailing_mask_end is not None
                ):
                    if trailing_mask_begin <= leading_mask_end:
                        result = max(trailing_mask_end - leading_mask_end, 0)
                    else:
                        result = max(trailing_mask_end - trailing_mask_begin + 1, 0)
        else:
            if seqlen_k % tile_shape[1] != 0:
                result = 1
            else:
                result = 0

        return result + rem_count

    @cute.jit
    def get_unmasked_trip_count(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ) -> Int32:
        """
        Calculate the number of unmasked trips for the current block.

        This represents the number of trips that don't require special
        masking treatment.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]

        :return: Number of unmasked trips.
        :rtype: Int32
        """
        result = (
            FusedMask.get_trip_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - FusedMask.get_masked_leading_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            - FusedMask.get_masked_trailing_count(
                mask_type,
                blk_coord,
                tile_shape,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                0,
            )
        )
        return result

    @cute.jit
    def get_masked_info(
        mask_type: MaskEnum,
        blk_coord: cute.Coord,
        tile_shape: cute.Shape,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        rem_count: Optional[Int32] = 0,
    ) -> Tuple[Int32, Int32, Int32, Int32, Int32]:
        """
        Calculate the number of masked trips for the trailing mask.

        This is used for blocks that need special handling due to masking.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param blk_coord: Block coordinates.
        :type blk_coord: cute.Coord
        :param tile_shape: Shape of the tile.
        :type tile_shape: cute.Shape
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Int32
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param rem_count: Remaining count from previous calculations.
        :type rem_count: Int32

        :return: Number of masked info.
        :rtype: Tuple[Int32, Int32, Int32, Int32, Int32]
        """
        start_count = FusedMask.get_trip_start(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        leading_mask_count = FusedMask.get_masked_leading_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        unmask_count = FusedMask.get_unmasked_trip_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        trailing_mask_count = FusedMask.get_masked_trailing_count(
            mask_type,
            blk_coord,
            tile_shape,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
            rem_count,
        )
        end_count = start_count + leading_mask_count + unmask_count + trailing_mask_count
        return (
            start_count,
            end_count,
            leading_mask_count,
            unmask_count,
            trailing_mask_count,
        )

    @cute.jit
    def apply_mask(
        mask_type: MaskEnum,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        seqlen_k: Int32,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        """
        Apply the appropriate mask to the attention scores.

        This method modifies the attention scores (acc_qk) based on the mask type
        and the positions in the index tensor.

        :param mask_type: Type of mask to use
        :type mask_type: utils.MaskEnum
        :param acc_qk: Accumulated QK attention scores tensor.
        :type acc_qk: cute.Tensor
        :param index_qk: Index tensor containing position information.
        :type index_qk: cute.Tensor
        :param seqlen_k: Key sequence length for attention computation.
        :type seqlen_k: Int32
        :param seqlen_q: Query sequence length for attention computation.
        :type seqlen_q: Optional[int]
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[int]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[int]
        """

        tidx, tidy, tidx = cute.arch.thread_idx()
        offset = 0
        offset = (
            seqlen_k - seqlen_q
            if cutlass.const_expr(
                mask_type is MaskEnum.WINDOW_MASK_INFERENCE
                or mask_type is MaskEnum.WINDOW_MASK_BWD_INFERENCE
            )
            else 0
        )
        for i in cutlass.range(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_transform(*index_qk[i])
            if cutlass.const_expr(window_size_left is not None or window_size_right is not None):
                if cutlass.const_expr(window_size_left is None):
                    if index_q + offset + window_size_right < index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf
                elif cutlass.const_expr(window_size_right is None):
                    if index_q + offset - window_size_left > index_k:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf
                else:
                    max_K_index = min(index_q + offset + window_size_right, seqlen_k)
                    min_K_index = max(0, index_q + offset - window_size_left)
                    if index_k > max_K_index or index_k < min_K_index:
                        acc_qk[i] = -Float32.inf
                    if index_k >= seqlen_k or index_q >= seqlen_q:  # residual mask
                        acc_qk[i] = -Float32.inf

            if cutlass.const_expr(
                mask_type == MaskEnum.RESIDUAL_MASK or mask_type == MaskEnum.RESIDUAL_MASK_BWD
            ):
                if index_k >= seqlen_k or index_q >= seqlen_q:
                    acc_qk[i] = -Float32.inf


@dsl_user_op
def ex2_emulation_packed_f32x2(
    x: Float32, y: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    # Clamp the xy so they fit within the FP32 exponent range
    # The upper side is ensured by the (s - row_max)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))

    fp32_round_int = float(2**23 + 2**22)
    # |  0   | 10010110 | 10000000000000000000000 |
    # | sign | exponent |        mantissa         |
    #                                           ^
    #                                           |
    #                                  digit of ones place
    # During FP32 addition, any number that smaller than this will be
    # aligned the exponent to 2^23, so that the digit of ones
    # is at the rightest place
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm")
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = cute.arch.sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = cute.arch.sub_packed_f32x2(xy_clamped, xy_rounded_back)

    @dsl_user_op
    @cute.jit
    def polynomial_deg3_packed_f32x2(
        x: Float32, y: Float32, *, loc=None, ip=None
    ) -> Tuple[Float32, Float32]:
        # 2^x ~= (0.077 * x + 0.228) * x + 0.695) * x + 1, for x in [0, 1)
        coeff = (
            1.0,  # coeff of deg0
            0.695146143436431884765625,  # coeff of deg1
            0.227564394474029541015625,  # coeff of deg2
            0.077119089663028717041015625,  # coeff of deg3
        )
        deg = len(coeff) - 1  # started with highest degree
        out = (coeff[deg], coeff[deg])
        for i in cutlass.range_constexpr(deg - 1, -1, -1):
            out = cute.arch.fma_packed_f32x2(out, (x, y), (coeff[i], coeff[i]), loc=loc, ip=ip)
        return out

    xy_frac_ex2 = polynomial_deg3_packed_f32x2(*xy_frac, loc=loc, ip=ip)

    @dsl_user_op
    def combine_int_frac_ex2(
        x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None
    ) -> Float32:
        return cutlass.Float32(
            llvm.inline_asm(
                T.f32(),
                [
                    Float32(x_rounded).ir_value(loc=loc, ip=ip),
                    Float32(frac_ex2).ir_value(loc=loc, ip=ip),
                ],
                "{\n\t"
                ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
                "mov.b32 x_rounded_i, $1;\n\t"
                "mov.b32 frac_ex_i, $2;\n\t"
                "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
                "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
                "mov.b32 $0, out_i;\n\t"
                "}\n",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)

    return x_out, y_out


@cute.jit
def cvt_f32x4_to_f8x4_pack_i32(fp32x4, fp8_type, *, loc=None, ip=None):
    fp32x4 = fp32x4.load()
    src_vec4 = fp32x4.ir_value(loc=loc, ip=ip) if hasattr(fp32x4, "ir_value") else fp32x4

    src0 = Float32(vector.extract(src_vec4, [], [0])).ir_value(loc=loc, ip=ip)
    src1 = Float32(vector.extract(src_vec4, [], [1])).ir_value(loc=loc, ip=ip)
    src2 = Float32(vector.extract(src_vec4, [], [2])).ir_value(loc=loc, ip=ip)
    src3 = Float32(vector.extract(src_vec4, [], [3])).ir_value(loc=loc, ip=ip)

    cvt_instruction = ""
    if cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
        cvt_instruction = "cvt.rn.satfinite.e4m3x2.f32"
    else:
        assert False, "Unsupported fp8 element type"

    asm_tmpl = (
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        f"  {cvt_instruction} lo, $2, $1;\n"
        f"  {cvt_instruction} hi, $4, $3;\n"
        "  mov.b32 $0, {lo, hi};\n"
        "}"
    )
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [src0, src1, src2, src3],
        asm_tmpl,
        "=r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return packed_i32


@cute.jit
def cvt_f32x4_to_f8x4(fp32x4, fp8x4, *, loc=None, ip=None):
    packed_i32 = cvt_f32x4_to_f8x4_pack_i32(fp32x4, fp8x4.element_type)
    fp8x4_i32 = cute.recast_tensor(fp8x4, cutlass.Int32)
    fp8x4_i32[0] = cutlass.Int32(packed_i32)
    return
