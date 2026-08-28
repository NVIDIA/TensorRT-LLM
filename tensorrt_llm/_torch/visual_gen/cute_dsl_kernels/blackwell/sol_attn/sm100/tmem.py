# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from https://github.com/NVlabs/Sana (Apache-2.0); see
# THIRD_PARTY_NOTICES.md in this directory for the pin and scope.
"""TMEM load helpers used by the SM100 mainloop."""

from __future__ import annotations

import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm

M = 64
D = 128
O_OFFSET = 128


@cute.jit
def tcgen05_wait_ld() -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.wait::ld.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def tcgen05_wait_st() -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.wait::st.sync.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _zero_based_tmem_tensor(element_type, layout):
    return cute.make_tensor(
        cute.make_ptr(
            element_type,
            Int32(0),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        layout,
    )


@cute.jit
def _add_physical_tmem_base(
    relative: cute.Tensor,
    physical_address: Int32,
):
    return cute.make_tensor(
        cute.make_ptr(
            relative.element_type,
            physical_address + relative.iterator.toint(),
            cute.AddressSpace.tmem,
            assumed_align=16,
        ),
        relative.layout,
    )


@cute.jit
def _o_copy_views(
    o_template: cute.Tensor,
    pv_thread: cute.ThrMma,
):
    assert o_template.element_type == Float32
    assert cute.size(o_template) == M * D
    relative = _zero_based_tmem_tensor(Float32, o_template.layout)
    coordinates = pv_thread.partition_C(cute.make_identity_tensor((M, D)))
    tiler = (
        (
            cute.size(relative, mode=[0, 0]),
            cute.size(relative, mode=[0, 1]),
        ),
    )
    return (
        cute.zipped_divide(relative, tiler),
        cute.zipped_divide(coordinates, tiler),
    )


@cute.jit
def load_m64_o_fp32_256b(
    o_template: cute.Tensor,
    pv_thread: cute.ThrMma,
    physical_tmem_base: Int32,
    thread_idx: Int32,
):
    relative, coordinates = _o_copy_views(o_template, pv_thread)
    atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x8),
        Float32,
    )
    tiled_copy = tcgen05.make_tmem_copy(
        atom,
        relative[None, Int32(0)],
    )
    thread_copy = tiled_copy.get_slice(thread_idx)
    source = _add_physical_tmem_base(
        thread_copy.partition_S(relative),
        physical_tmem_base + Int32(O_OFFSET),
    )
    register_coordinates = thread_copy.partition_D(coordinates)[None, None, Int32(0)]
    registers = cute.make_rmem_tensor(
        register_coordinates.shape,
        Float32,
    )
    cute.copy(
        tiled_copy,
        source[None, None, Int32(0)],
        registers,
    )
    tcgen05_wait_ld()
    cute.arch.fence_view_async_tmem_load()
    return registers, register_coordinates


__all__ = [
    "_add_physical_tmem_base",
    "_zero_based_tmem_tensor",
    "load_m64_o_fp32_256b",
    "tcgen05_wait_ld",
    "tcgen05_wait_st",
]
