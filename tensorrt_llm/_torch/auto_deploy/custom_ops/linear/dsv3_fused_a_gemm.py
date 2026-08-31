# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""AutoDeploy wrapper for ``trtllm::dsv3_fused_a_gemm_op``.

Provides a rank-agnostic ``auto_deploy::dsv3_fused_a_gemm`` op that flattens
input to 2D, calls the trtllm custom op (with the required column-major
weight stride), and reshapes the output back to the input's rank.  This
lets the FX-graph transform emit a single op call instead of explicit
view/reshape nodes around the trtllm cpp op (which requires 2D inputs).
"""

import torch


@torch.library.custom_op("auto_deploy::dsv3_fused_a_gemm", mutates_args=())
def dsv3_fused_a_gemm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fused q_a + kv_a_with_mqa projection for DeepSeek-V3 MLA.

    Args:
        input: Activation of shape ``(..., 7168)``.  Any rank; flattened to 2D
            internally.  Must be bf16.
        weight: Concatenated weight of shape ``(2112, 7168)`` row-major
            (``q_a_proj.weight`` over ``kv_a_proj_with_mqa.weight`` on dim 0).
            Must be bf16.

    Returns:
        Output of shape ``(..., 2112)``.  Caller is expected to split into
        ``q_a`` (1536) and ``kv_a_with_mqa`` (576) via ``narrow``/``split``.
    """
    leading = input.shape[:-1]
    in_features = input.shape[-1]
    flat = input.reshape(-1, in_features).contiguous()
    # mat_b must be column-major: weight.t() gives a (in, out) column-major view.
    out_2d = torch.ops.trtllm.dsv3_fused_a_gemm_op(flat, weight.t(), None, None)
    return out_2d.reshape(*leading, out_2d.shape[-1])


@dsv3_fused_a_gemm.register_fake
def _dsv3_fused_a_gemm_fake(input, weight):
    leading = list(input.shape[:-1])
    return input.new_empty(leading + [weight.shape[0]], dtype=input.dtype)
