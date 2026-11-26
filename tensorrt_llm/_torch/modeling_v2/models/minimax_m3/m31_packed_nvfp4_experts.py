# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The global scales of the compressed-tensors NVFP4 routed experts of the MiniMax-M3.1-Flash checkpoint.

Per projection the checkpoint stores ``weight_global_scale`` / ``input_global_scale`` as the FP32 multiplier
2688 / amax (2688 = E4M3 max 448 * E2M1 max 6); the NVFP4 MoE kernels take the divisor amax / 2688. The targets' own
weight loaders convert each scale here and reject one that would disable an expert.
"""

import torch
from torch import Tensor


def nvfp4_weight_scale_2(weight_global_scale: Tensor) -> Tensor:
    """Return the NVFP4 MoE ``weight_scale_2`` of a stored global scale: its FP32 reciprocal, ``amax / 2688``."""
    return torch.reciprocal(weight_global_scale.float())


def _read_global_scale(name: str, value) -> Tensor:
    """Materialize one global scale, rejecting values that would disable an expert."""
    scale = (value if isinstance(value, Tensor) else value[...]).to(torch.float32)
    if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError(
            f"Checkpoint tensor {name} must hold one finite positive global scale, "
            f"got {scale.flatten()[:4].tolist()}."
        )
    return scale


def reciprocal_global_scale(name: str, value: Tensor) -> Tensor:
    """Validate a packed expert scale and return its FP32 modelopt divisor."""
    return nvfp4_weight_scale_2(_read_global_scale(name, value))
