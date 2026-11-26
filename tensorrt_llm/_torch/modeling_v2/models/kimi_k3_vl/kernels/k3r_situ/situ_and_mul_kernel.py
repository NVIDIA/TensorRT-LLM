# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused SiTU gated activation kernel (Triton).

``out[i, :] = beta * tanh(g / beta) * sigmoid(g) * up`` for ``[g | up] = x[i, :]``, with
``up = linear_beta * tanh(up / linear_beta)`` when ``linear_beta`` is set; fp32 math, one rounding
to the output dtype. One program covers ``BLOCK_SIZE`` columns of one row. With
``LAUNCH_WITH_PDL`` it waits on the producer before its first load and releases its dependents
after its store.

The kernel source is the one ``trtllm::situ_and_mul`` launches (``_torch/modules/situ.py``);
``test_k3r_situ.py`` pins it.
"""

import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
import triton.language.extra.libdevice as tldevice  # type: ignore[import]


@triton.jit
def situ_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    d,
    beta,
    linear_beta,
    BLOCK_SIZE: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    GATE_UP_INTERLEAVE: tl.constexpr = 0,
) -> None:
    """SiTU on packed ``[gate | up]`` rows or interleaved gate/up blocks.

    With ``GATE_UP_INTERLEAVE == 0`` each input row is ``[g0 .. g(d-1), u0 .. u(d-1)]``.
    With ``GATE_UP_INTERLEAVE == G`` it alternates blocks of ``G`` gate and ``G`` up
    columns, ``[g0 .. g(G-1), u0 .. u(G-1), gG .. g(2G-1), uG .. u(2G-1), ...]``, and
    ``d`` must be a multiple of ``G`` (the launcher checks it). The output row is
    ``d`` wide either way.
    """
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    if GATE_UP_INTERLEAVE:
        gate_offsets = (offsets // GATE_UP_INTERLEAVE) * (
            2 * GATE_UP_INTERLEAVE
        ) + offsets % GATE_UP_INTERLEAVE
        up_offsets = gate_offsets + GATE_UP_INTERLEAVE
    else:
        gate_offsets = offsets
        up_offsets = offsets + d
    gate = tl.load(x_row_ptr + gate_offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + up_offsets, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    tl.store(o_row_ptr + offsets, result, mask=mask)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()
