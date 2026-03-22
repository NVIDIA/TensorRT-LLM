# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Template-based Triton kernel generation from fused MLIR ops.

Two operating modes:
    - ``preexisting``: Map fused ops to existing custom ops (FlashInfer/Triton).
    - ``generate``: Generate Triton kernels from MLIR op semantics, registered
      as proper ``torch.library.custom_op`` for FX graph compatibility.
"""

from typing import Callable, Literal, Optional

import torch

from ..dialect import AdFusedAddRMSNorm

# Cache for registered generated ops (module-level to survive across instances)
_generated_op_cache: dict = {}


def _get_triton_fused_add_rmsnorm_op():
    """Register and return the Triton-generated fused_add_rmsnorm as a custom op.

    The op is registered once and cached. It has a proper schema and fake
    implementation so FX tracing, shape propagation, and ``operator.getitem``
    on the tuple result all work correctly.
    """
    cache_key = "triton_fused_add_rmsnorm"
    if cache_key in _generated_op_cache:
        return _generated_op_cache[cache_key]

    from .templates.fused_add_rmsnorm import fused_add_rmsnorm as triton_kernel

    # Register as a torch custom op with explicit schema
    @torch.library.custom_op("auto_deploy::mlir_triton_fused_add_rmsnorm", mutates_args=())
    def mlir_triton_fused_add_rmsnorm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return triton_kernel(x, residual, weight, eps)

    @mlir_triton_fused_add_rmsnorm.register_fake
    def _(x, residual, weight, eps):
        return torch.empty_like(x), torch.empty_like(x)

    # Build a wrapper matching the flashinfer_fused_add_rms_norm calling convention:
    # call the custom op and return the tuple.
    def wrapper(x, residual, weight, eps):
        return torch.ops.auto_deploy.mlir_triton_fused_add_rmsnorm(x, residual, weight, eps)

    _generated_op_cache[cache_key] = wrapper
    return wrapper


class TritonCodegen:
    """Generate or select kernels for fused MLIR ops.

    Args:
        mode: ``"preexisting"`` to use existing custom ops, ``"generate"`` to emit
            Triton kernels from templates.
    """

    def __init__(self, mode: Literal["preexisting", "generate"] = "preexisting"):
        self.mode = mode

    def get_fused_add_rmsnorm_impl(self, fused_op: Optional[AdFusedAddRMSNorm] = None) -> Callable:
        """Return a callable implementing fused add + rmsnorm.

        In ``preexisting`` mode, returns the existing FlashInfer wrapper.
        In ``generate`` mode, returns the Triton-generated kernel launcher
        registered as a proper torch custom op.
        """
        if self.mode == "preexisting":
            return self._get_preexisting_impl()
        else:
            return _get_triton_fused_add_rmsnorm_op()

    def _get_preexisting_impl(self) -> Callable:
        """Return the existing flashinfer_fused_add_rms_norm wrapper."""
        from ...custom_ops.normalization.flashinfer_fused_add_rms_norm import (
            flashinfer_fused_add_rms_norm,
        )

        return flashinfer_fused_add_rms_norm
