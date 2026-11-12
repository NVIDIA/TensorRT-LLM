# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``torch.ops.mv2_kimi_k3_vl.situ_and_mul``: the fused SiTU gated activation on ``[M, 2 * d]``.

Same launch as ``trtllm::situ_and_mul``: grid ``(M, cdiv(d, 1024))``, ``BLOCK_SIZE`` 1024, PDL when
the family's SM107 launch constant allows it (SM90+).
"""

from typing import Mapping, Optional

import torch
import triton  # type: ignore[import]

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of
from tensorrt_llm._utils import get_sm_version

from ...ops.common.sm107_launch import USE_PDL
from .situ_and_mul_kernel import situ_and_mul_kernel

_NS = namespace_of(__name__)


@torch.library.custom_op(f"{_NS}::situ_and_mul", mutates_args=())
def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    """Run fused SiTU on a packed ``[num_tokens, 2 * d]`` input."""
    b, n = x.shape

    assert n % 2 == 0
    if x.stride(1) != 1:
        raise ValueError("situ_and_mul requires a contiguous last dimension")
    d = n // 2
    output = torch.empty((b, d), dtype=x.dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    launch_with_pdl = USE_PDL and get_sm_version() >= 90
    situ_and_mul_kernel[grid](
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


@situ_and_mul.register_fake
def _(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    b, n = x.shape
    assert n % 2 == 0
    return x.new_empty((b, n // 2))
