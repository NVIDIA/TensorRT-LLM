# Copyright (c) 2026 by FlashInfer team.
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


"""Trace-time placeholder constructors shared by the FMHA TS kernels."""

import cutlass
from cutlass.experimental import primitives as prims


def _shape_tuple(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize scalar and tuple shapes for ``cutlass.Array`` construction."""
    if isinstance(shape, tuple):
        return shape
    return (shape,)


def _placeholder_smem_array(
    dtype: type, shape: int | tuple[int, ...] = 1
) -> cutlass.Array | None:
    """Build a typed shared-memory-view placeholder only when an MLIR context exists."""
    try:
        return cutlass.Array(
            cutlass.Int64(0),
            dtype=dtype,
            shape=_shape_tuple(shape),
            addrspace=3,
        )
    except (RuntimeError, ValueError):
        return None


def _placeholder_local_array(
    dtype: type, shape: int | tuple[int, ...] = 1, alignment: int | None = None
) -> cutlass.Array | None:
    """Build a typed local-memory placeholder only when an MLIR context exists."""
    try:
        if alignment is None:
            return cutlass.Array(dtype, shape, space=cutlass.AddressSpace.rmem)
        return cutlass.Array(
            dtype, shape, space=cutlass.AddressSpace.rmem, alignment=alignment
        )
    except (RuntimeError, ValueError):
        return None


def _placeholder_tmem_ptr() -> cutlass.Array | None:
    """Build a typed tensor-memory pointer placeholder only when an MLIR context exists."""
    try:
        return prims.make_tmem_ptr(cutlass.Int32(0), cutlass.Int8)
    except RuntimeError:
        return None
