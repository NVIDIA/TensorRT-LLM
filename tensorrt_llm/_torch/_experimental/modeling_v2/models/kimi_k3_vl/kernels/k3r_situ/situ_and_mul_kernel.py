# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused SiTU gated activation kernel (Triton).

``out[i, :] = beta * tanh(g / beta) * sigmoid(g) * up`` for ``[g | up] = x[i, :]``, with
``up = linear_beta * tanh(up / linear_beta)`` when ``linear_beta`` is set; fp32 math, one rounding
to the output dtype. One program covers ``BLOCK_SIZE`` columns of one row. With
``LAUNCH_WITH_PDL`` it waits on the producer before its first load and releases its dependents
after its store.

The kernel source is the one ``trtllm::situ_and_mul`` launches (``_torch/modules/situ.py``), the
SiTU of the reference Kimi K3 SM107 runs; ``test_k3r_situ.py`` pins it.
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
) -> None:
    """Fused :class:`SituAndMul` on a packed ``[gate | up]`` row layout."""
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    tl.store(o_row_ptr + offsets, result, mask=mask)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()
