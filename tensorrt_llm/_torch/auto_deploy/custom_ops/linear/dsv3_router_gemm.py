# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""AutoDeploy wrapper for ``trtllm::dsv3_router_gemm_op`` (MoE gate/router).

The reference model computes the router logits in fp32 (``hidden.float() @
gate_weight.float().t()``), which AutoDeploy lowers to a slow fp32 cuBLAS GEMV.
The PyTorch backend instead runs a bf16 GEMM with fp32 output via the dedicated
``trtllm::dsv3_router_gemm_op`` kernel. This wrapper mirrors that: bf16 input ×
bf16 weight -> fp32 logits, flattening to 2D and reshaping back.
"""

import torch


@torch.library.custom_op("auto_deploy::dsv3_router_gemm", mutates_args=())
def dsv3_router_gemm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """MoE gate/router projection for DeepSeek-V3.

    Args:
        input: Activation ``(..., hidden)``, bf16. Flattened to 2D internally.
        weight: Router weight ``(num_experts, hidden)`` row-major, bf16.

    Returns:
        Router logits ``(..., num_experts)`` in fp32.
    """
    leading = input.shape[:-1]
    in_features = input.shape[-1]
    flat = input.reshape(-1, in_features).contiguous()
    # mat_b must be column-major: weight.t() gives a (hidden, num_experts) view.
    out_2d = torch.ops.trtllm.dsv3_router_gemm_op(flat, weight.t(), None, torch.float32)
    return out_2d.reshape(*leading, out_2d.shape[-1])


@dsv3_router_gemm.register_fake
def _dsv3_router_gemm_fake(input, weight):
    leading = list(input.shape[:-1])
    return input.new_empty(leading + [weight.shape[0]], dtype=torch.float32)
