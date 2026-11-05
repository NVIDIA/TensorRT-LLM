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
#
# Copied from the dynamic-kernel-generator repository
# (mxf8-opt/blackwell-port/source/blackwell_compat.py): SM100 helper adapter
# for the fused FC12 kernel (rejects Rubin-only operand-collector reuse).
"""SM100 helper adapter for the restricted single-CTA FC12 port."""

import cutlass.utils.blackwell_helpers as blackwell
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp


def make_blockscaled_trivial_tiled_mma(
    a_dtype,
    b_dtype,
    a_major,
    b_major,
    sf_dtype,
    sf_vec_size,
    cta_group,
    mma_shape,
    *,
    a_collector_op=CollectorOp.DISCARD,
    b_collector_op=CollectorOp.DISCARD,
    atom_layout_mnk=(1, 1, 1),
    permutation_mnk=(1, 1, 1),
):
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


def compute_epilogue_tile_shape(mma_op, cta_tile, use_2cta, output_layout, output_dtype):
    return blackwell.compute_epilogue_tile_shape(cta_tile, use_2cta, output_layout, output_dtype)
