# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TMA prefetch of the CSA2 output-projection weights into L2."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import Int64, T, dsl_user_op


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class Csa2WeightPrefetch:
    """Prefetch two E4M3 weights ``[rows, k]`` into L2 in 128x128-byte TMA boxes, in the order the
    output GEMMs read them; a CTA stops issuing once its time budget is spent."""

    def __init__(self, first: tuple[int, int], second: tuple[int, int]):
        if any(size % 128 for size in (*first, *second)):
            raise ValueError("CSA2 weight prefetch needs 128-aligned weights")
        self.shapes = (first, second)

    @cute.jit
    def run(
        self,
        first: cute.Pointer,
        second: cute.Pointer,
        anchor: cute.Tensor,
        ctas: cutlass.Int32,
        budget_ns: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        """``anchor`` only carries the TVM FFI stream."""
        box = cute.make_layout((128, 128), stride=(128, 1))
        atoms = []
        for pointer, (rows, k) in zip((first, second), self.shapes):
            tensor = cute.make_tensor(pointer, cute.make_layout((rows, k), stride=(k, 1)))
            atoms.append(
                cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(), tensor, box, (128, 128)
                )
            )

        @cute.struct
        class SharedStorage:
            box: cute.struct.Align[cute.struct.MemRange[first.dtype, 128 * 128], 1024]

        self.shared_storage = SharedStorage
        self.kernel(*atoms[0], *atoms[1], ctas, budget_ns).launch(
            grid=[ctas, 1, 1],
            block=[32, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        ctas: cutlass.Int32,
        budget_ns: cutlass.Int64,
    ):
        bidx, _, _ = cute.arch.block_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        # Only shapes the partitions; prefetches write no shared memory.
        box = cute.group_modes(
            storage.box.get_tensor(cute.make_layout((128, 128), stride=(128, 1))), 0, 2
        )
        partitions = []
        for atom, tensor in ((atom_a, mA), (atom_b, mB)):
            tiles = cute.local_tile(tensor, (128, 128), (None, None))
            partitions.append(
                cpasync.tma_partition(
                    atom, 0, cute.make_layout(1), box, cute.group_modes(tiles, 0, 2)
                )[1]
            )
        tiles_a = self.shapes[0][0] // 128
        tiles_b = self.shapes[1][0] // 128
        units_a = tiles_a * (self.shapes[0][1] // 128)
        units = units_a + tiles_b * (self.shapes[1][1] // 128)
        count = (units - bidx + ctas - 1) // ctas
        with cute.arch.elect_one():
            start = _globaltimer()
            for group in cutlass.range((count + 15) // 16, unroll=1):
                if _globaltimer() - start < budget_ns:
                    for j in cutlass.range_constexpr(16):
                        i = group * 16 + j
                        if i < count:
                            unit = bidx + i * ctas
                            if unit < units_a:
                                cute.prefetch(
                                    atom_a, partitions[0][(None, unit % tiles_a, unit // tiles_a)]
                                )
                            else:
                                rest = unit - units_a
                                cute.prefetch(
                                    atom_b, partitions[1][(None, rest % tiles_b, rest // tiles_b)]
                                )
