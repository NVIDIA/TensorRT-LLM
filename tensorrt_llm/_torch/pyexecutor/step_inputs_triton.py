# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step-input pack's Triton scatter kernels (``step_inputs``), imported only when a pack takes a Triton path: the
eager parallel scatter, or the in-graph prologue without a model ``prologue_op``. A pack with ``cpp_scatter`` takes
neither, so this module (and Triton) is never imported on its step path."""

import triton
import triton.language as tl


# One compiled variant for every descriptor offset: specializing ``first`` (== 1, divisible by 16) would compile a
# new kernel when a step's pack first holds a new number of copies, after the warm-up.
@triton.jit(do_not_specialize=["first"])
def _parallel_scatter_kernel(
    stage_u8, stage_i64, first, DEVICE_SRC: tl.constexpr, BLOCK: tl.constexpr
):
    """Program i writes copy ``first + i`` ([dst, src, bytes] descriptors after the 2-word header). ``src`` is an offset
    into the staged buffer for a host copy, a device address for a device copy (``DEVICE_SRC``)."""
    d = 2 + 3 * (first + tl.program_id(0))
    dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
    src_word = tl.load(stage_i64 + d + 1)
    if DEVICE_SRC:
        src = src_word.to(tl.pointer_type(tl.uint8))
    else:
        src = stage_u8 + src_word
    nbytes = tl.load(stage_i64 + d + 2)
    for start in range(0, nbytes, BLOCK):
        o = start + tl.arange(0, BLOCK)
        tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)


@triton.jit
def _graph_scatter_kernel(
    stage_u8,
    stage_i64,
    epoch,
    DEVICE_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
    BUMP_EPOCH: tl.constexpr,
):
    """Descriptor slot i of the host copies (``DEVICE_SRC`` False) or of the device copies, with the counts read from
    the staged header; a slot past its count writes nothing. ``BUMP_EPOCH``: program 0 also increments ``epoch[0]``."""
    i = tl.program_id(0)
    if BUMP_EPOCH:
        if i == 0:
            tl.store(epoch, tl.load(epoch) + 1)
    num_h2d = tl.load(stage_i64)
    if DEVICE_SRC:
        count = tl.load(stage_i64 + 1)
        d = 2 + 3 * (num_h2d + i)
    else:
        count = num_h2d
        d = 2 + 3 * i
    if i < count:
        dst = tl.load(stage_i64 + d).to(tl.pointer_type(tl.uint8))
        src_word = tl.load(stage_i64 + d + 1)
        if DEVICE_SRC:
            src = src_word.to(tl.pointer_type(tl.uint8))
        else:
            src = stage_u8 + src_word
        nbytes = tl.load(stage_i64 + d + 2)
        for start in range(0, nbytes, BLOCK):
            o = start + tl.arange(0, BLOCK)
            tl.store(dst + o, tl.load(src + o, mask=o < nbytes), mask=o < nbytes)
