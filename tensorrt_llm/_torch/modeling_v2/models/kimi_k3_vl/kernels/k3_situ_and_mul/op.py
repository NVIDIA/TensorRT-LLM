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
"""``trtllm::k3_situ_and_mul``: fused SiTU on a packed ``[num_tokens, 2 * d]`` input, the Kimi K3 targets' copy of
``trtllm::situ_and_mul`` (``tensorrt_llm/_torch/modules/situ.py``) under its own name, so both can be registered in
one process.

PDL is used when ``TRTLLM_ENABLE_PDL`` is unset or 1 and the GPU is SM 9.0 or newer, read on every call.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import torch
import triton  # type: ignore[import]

from tensorrt_llm._utils import get_sm_version

from .k3_situ_and_mul_kernel import k3_situ_and_mul_kernel


@torch.library.custom_op("trtllm::k3_situ_and_mul", mutates_args=())
def k3_situ_and_mul(
    x: torch.Tensor, beta: float, linear_beta: Optional[float] = None
) -> torch.Tensor:
    """Run fused SiTU on a packed ``[num_tokens, 2 * d]`` input whose last dimension is dense."""
    b, n = x.shape

    assert n % 2 == 0
    if x.stride(1) != 1:
        raise ValueError("k3_situ_and_mul requires a contiguous last dimension")
    d = n // 2
    output = torch.empty((b, d), dtype=x.dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    launch_with_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1" and get_sm_version() >= 90
    k3_situ_and_mul_kernel[grid](
        o_ptr=output,
        o_stride=output.stride(0),
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        beta=float(beta),
        linear_beta=float(linear_beta) if linear_beta is not None else 1.0,
        BLOCK_SIZE=1024,
        HAS_LINEAR_BETA=linear_beta is not None,
        LAUNCH_WITH_PDL=launch_with_pdl,
        launch_pdl=launch_with_pdl,
    )
    return output


@k3_situ_and_mul.register_fake
def _(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    b, n = x.shape
    assert n % 2 == 0
    return x.new_empty((b, n // 2))
