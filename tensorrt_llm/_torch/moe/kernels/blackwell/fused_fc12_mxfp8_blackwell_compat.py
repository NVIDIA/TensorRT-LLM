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
"""SM100 adapters for the helpers the fused FC12 kernel shares with Rubin.

The kernel calls these with the Rubin helper signatures; they forward to
``cutlass.utils.blackwell_helpers`` and reject the Rubin-only options (operand
collector reuse, MMA atom layouts and permutations).
"""

from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as blackwell
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp


def make_blockscaled_trivial_tiled_mma(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    a_major: tcgen05.OperandMajorMode,
    b_major: tcgen05.OperandMajorMode,
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    cta_group: tcgen05.CtaGroup,
    mma_shape: Tuple[int, ...],
    *,
    a_collector_op: CollectorOp = CollectorOp.DISCARD,
    b_collector_op: CollectorOp = CollectorOp.DISCARD,
    atom_layout_mnk: Tuple[int, int, int] = (1, 1, 1),
    permutation_mnk: Tuple[int, int, int] = (1, 1, 1),
) -> cute.TiledMma:
    """Block-scaled tiled MMA for the (M, N) of ``mma_shape``; K comes from the dtype."""
    if (
        a_collector_op != CollectorOp.DISCARD
        or b_collector_op != CollectorOp.DISCARD
        or atom_layout_mnk != (1, 1, 1)
        or permutation_mnk != (1, 1, 1)
    ):
        raise ValueError(
            "SM100 port does not support Rubin operand-collector reuse or MMA permutations"
        )
    return blackwell.make_blockscaled_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_major,
        b_major,
        sf_dtype,
        sf_vec_size,
        cta_group,
        mma_shape[:2],
    )


def compute_epilogue_tile_shape(
    mma_op: object,
    cta_tile: Tuple[int, int, int],
    use_2cta: bool,
    output_layout: utils.LayoutEnum,
    output_dtype: Type[cutlass.Numeric],
) -> cute.Tile:
    """Epilogue tile of the SM100 helper; ``mma_op`` (needed on Rubin) is unused."""
    return blackwell.compute_epilogue_tile_shape(cta_tile, use_2cta, output_layout, output_dtype)
