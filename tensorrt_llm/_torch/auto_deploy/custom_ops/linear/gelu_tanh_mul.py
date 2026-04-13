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

"""Fused GELU-tanh+Mul custom operation for graph transformation.

Replaces the pattern: gelu_tanh(narrow(x, 0, half)) * narrow(x, half, half)
with a single fused kernel that avoids materializing the narrow views.

Targets models using gelu_pytorch_tanh activation (e.g. Gemma4 dense MLP).

The custom op is only registered when FlashInfer is available.
"""

import torch

try:
    from flashinfer.activation import gelu_tanh_and_mul as _flashinfer_gelu_tanh_and_mul

    HAS_FUSED_GELU_TANH_AND_MUL = True
except ImportError:
    _flashinfer_gelu_tanh_and_mul = None
    HAS_FUSED_GELU_TANH_AND_MUL = False

if HAS_FUSED_GELU_TANH_AND_MUL:

    @torch.library.custom_op("auto_deploy::gelu_tanh_and_mul", mutates_args=())
    def gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
        """Fused GELU-tanh+Mul: split x in half, apply gelu_tanh to first half, multiply.

        Equivalent to: gelu(x[..., :half], approximate='tanh') * x[..., half:]
        """
        return _flashinfer_gelu_tanh_and_mul(x)

    @gelu_tanh_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        """Fake implementation for tracing."""
        assert x.shape[-1] % 2 == 0, (
            f"gelu_tanh_and_mul requires even last dimension, got {x.shape[-1]}"
        )
        half_size = x.shape[-1] // 2
        output_shape = list(x.shape[:-1]) + [half_size]
        return x.new_empty(output_shape, dtype=x.dtype)
