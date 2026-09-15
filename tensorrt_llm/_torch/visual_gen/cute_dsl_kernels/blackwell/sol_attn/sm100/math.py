# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/NVlabs/Sana (Apache-2.0); see
# THIRD_PARTY_NOTICES.md in this directory for the pin and scope.
"""Small tensor-core helpers used by the Blackwell mainloop."""

import cutlass
import cutlass.cute as cute
from cutlass import Boolean
from cutlass.cute.nvgpu import tcgen05


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    accumulator: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None:
    mma = cute.make_mma_atom(tiled_mma.op)
    for k in cutlass.range_constexpr(cute.size(a.shape[2])):
        mma.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
        cute.gemm(
            mma,
            accumulator,
            a[None, None, k],
            b[None, None, k],
            accumulator,
        )


__all__ = ["gemm"]
