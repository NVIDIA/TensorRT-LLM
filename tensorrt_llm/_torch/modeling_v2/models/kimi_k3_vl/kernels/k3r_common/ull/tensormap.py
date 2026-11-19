# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TMA descriptors as ``__grid_constant__`` kernel parameters.

A descriptor in global memory written by the host (``host.tensor_maps``) must be
acquired by every thread that issues TMAs with it
(``fence.proxy.tensormap::generic.acquire``) before its first use in each launch.
On Rubin that fence costs ~0.9 us per descriptor, which is on the critical path
whenever a kernel's prologue is not hidden behind its predecessor (a launch gated
by a full dependency). A descriptor passed as a grid-constant parameter needs no
fence (CUDA programming guide, "Using TMA to transfer multi-dimensional arrays"):
the driver places it in parameter space with the launch.

Usage, in a ``@cute.jit`` host function and its ``@cute.kernel``:

    weights = encode_map(address, encoding, dims, strides, box, swizzle, l2)
    self.kernel(args, weights).launch(...)

    @cute.kernel
    def kernel(self, args, weights: GridMap):
        descriptor = map_address(weights)    # generic address for TMA asm
        prefetch_map(descriptor)             # optional, no fence needed

``encode_map`` runs ``cuTensorMapEncodeTiled`` in the launch's host code with the
same arguments ``host.tensor_maps`` takes, so the descriptor bytes are identical to
a host-encoded map of the same spec; under CUDA graph capture the encoded
descriptor is captured with the kernel's parameters.
"""

import cutlass
import cutlass._mlir.ir as ir
from cutlass import Int32, Int64

from .primitives import asm

try:  # CuTe DSL >= 4.8 (the Rubin image); older DSLs keep global descriptors
    from cutlass._mlir.dialects import cuda as cuda_dialect
    from cutlass.base_dsl.typing import GridConstant
    from cutlass.experimental.cuda import TensorMap

    # Kernel parameter annotation for a descriptor built by ``encode_map``.
    GridMap = GridConstant[TensorMap]
    AVAILABLE = True
except ImportError:
    GridMap, AVAILABLE = None, False


def encode_map(address, encoding, dims, strides, box, swizzle, l2):
    """Tiled TMA descriptor of ``address`` (host code of a ``@cute.jit`` function).
    ``encoding``, ``swizzle`` and ``l2``: CUDA driver enum values
    (``CUtensorMapDataType`` / ``CUtensorMapSwizzle`` / ``CUtensorMapL2promotion``);
    ``dims`` in elements, innermost first; ``strides`` in bytes (one per outer
    dimension); ``box`` in elements; no interleave, no OOB fill, unit element
    strides (``host.tensor_maps``)."""
    rank = len(dims)
    assert len(strides) == rank - 1 and len(box) == rank
    pointer_type = ir.Type.parse("!ptr.ptr<#ptr.generic_space>")
    pointer = cuda_dialect.cast(pointer_type, Int64(address).ir_value())
    space = cutlass.AddressSpace.rmem
    shape = cutlass.Array(Int64, rank, space=space)
    step = cutlass.Array(Int64, max(rank - 1, 1), space=space)
    tile = cutlass.Array(Int32, rank, space=space)
    unit = cutlass.Array(Int32, rank, space=space)
    for i in range(rank):
        shape[i], tile[i], unit[i] = (dims[i], box[i], 1)
    for i in range(rank - 1):
        step[i] = Int64(strides[i])
    result, descriptor = cuda_dialect.tensor_map_encode_tiled(
        Int32(int(encoding)).ir_value(),
        Int32(rank).ir_value(),
        pointer,
        shape.ir_value(),
        step.ir_value(),
        tile.ir_value(),
        unit.ir_value(),
        Int32(0).ir_value(),  # CU_TENSOR_MAP_INTERLEAVE_NONE
        Int32(int(swizzle)).ir_value(),
        Int32(int(l2)).ir_value(),
        Int32(0).ir_value(),  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        results=[cuda_dialect.ResultType.get(), TensorMap._get_mlir_type()],
    )
    cuda_dialect.result_assert_success(result)
    return TensorMap(descriptor)


def map_address(descriptor):
    """Generic address of a ``GridMap`` parameter, for TMA instructions."""
    return descriptor.get_ptr().toint()


@asm()
def prefetch_map(address):
    """``prefetch.tensormap`` of a grid-constant descriptor (no proxy fence)."""
    return "prefetch.tensormap [$0];", "l", (Int64(address),)
