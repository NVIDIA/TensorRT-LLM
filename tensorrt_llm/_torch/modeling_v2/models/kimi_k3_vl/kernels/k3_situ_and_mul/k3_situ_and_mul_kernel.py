# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton kernel of ``trtllm::k3_situ_and_mul``: SiTU on the gate half of a packed ``[gate | up]`` row times the
(optionally soft-capped) up half.

The same kernel as ``situ_and_mul_kernel`` in ``tensorrt_llm/_torch/modules/situ.py``, which the built-in Kimi K3
model keeps using; the two give the same bits.
"""

import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
import triton.language.extra.libdevice as tldevice  # type: ignore[import]


@triton.jit
def k3_situ_and_mul_kernel(
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
    """Row ``i``, column block ``j``: ``o[i, c] = beta*tanh(g/beta)*sigmoid(g) * up``, ``g, up = x[i, c], x[i, d + c]``.

    ``up`` is first soft-capped to ``linear_beta*tanh(up/linear_beta)`` when ``HAS_LINEAR_BETA``.
    """
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
