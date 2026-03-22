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
    - ``generate``: Generate Triton kernels from MLIR op semantics.
"""

from typing import Callable, Literal, Optional

import torch

from ..dialect import AdFusedAddRMSNorm


class TritonCodegen:
    """Generate or select kernels for fused MLIR ops.

    Args:
        mode: ``"preexisting"`` to use existing custom ops, ``"generate"`` to emit
            Triton kernels from templates.
    """

    def __init__(self, mode: Literal["preexisting", "generate"] = "preexisting"):
        self.mode = mode
        self._generated_ops: dict = {}

    def get_fused_add_rmsnorm_impl(self, fused_op: Optional[AdFusedAddRMSNorm] = None) -> Callable:
        """Return a callable implementing fused add + rmsnorm.

        In ``preexisting`` mode, returns the existing FlashInfer wrapper.
        In ``generate`` mode, returns the Triton-generated kernel launcher.
        """
        if self.mode == "preexisting":
            return self._get_preexisting_impl()
        else:
            return self._get_generated_impl()

    def _get_preexisting_impl(self) -> Callable:
        """Return the existing flashinfer_fused_add_rms_norm wrapper."""
        from ...custom_ops.normalization.flashinfer_fused_add_rms_norm import (
            flashinfer_fused_add_rms_norm,
        )

        return flashinfer_fused_add_rms_norm

    def _get_generated_impl(self) -> Callable:
        """Return the Triton-generated fused_add_rmsnorm kernel launcher.

        The generated kernel is registered as a PyTorch custom op on first call.
        """
        cache_key = "fused_add_rmsnorm"
        if cache_key in self._generated_ops:
            return self._generated_ops[cache_key]

        from .templates.fused_add_rmsnorm import fused_add_rmsnorm as triton_impl

        # Register as a torch custom op for FX graph compatibility
        impl = self._register_as_custom_op(triton_impl)
        self._generated_ops[cache_key] = impl
        return impl

    def _register_as_custom_op(self, kernel_fn: Callable) -> Callable:
        """Register a generated kernel as a PyTorch custom op.

        Uses ``create_derived_custom_op`` from the AutoDeploy utils when available,
        otherwise returns the raw callable (suitable for direct invocation in tests).
        """
        try:
            from ...utils._graph import create_derived_custom_op

            def make_impl(orig_impl):
                def impl(x, residual, weight, eps):
                    return kernel_fn(x, residual, weight, eps)

                return impl

            def make_fake(orig_fake):
                def fake(x, residual, weight, eps):
                    norm_out = torch.empty_like(x)
                    add_out = torch.empty_like(x)
                    return norm_out, add_out

                return fake

            # Derive from the flashinfer op schema for compatibility
            from ...custom_ops.normalization.flashinfer_fused_add_rms_norm import (
                flashinfer_fused_add_rms_norm,
            )

            derived_op = create_derived_custom_op(
                flashinfer_fused_add_rms_norm,
                suffix="_triton_generated",
                make_impl=make_impl,
                make_fake=make_fake,
            )
            return derived_op
        except Exception:
            # Fallback: return raw callable (works for testing)
            return kernel_fn
