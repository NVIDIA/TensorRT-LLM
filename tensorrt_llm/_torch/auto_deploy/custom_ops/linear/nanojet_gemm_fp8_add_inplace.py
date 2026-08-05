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

"""nanojet FP8 GEMM for projections whose activation is already e4m3."""

import torch

from ....nanojet_utils import is_nanojet_available

_REGISTERED = False


def register() -> bool:
    """Define the ops, importing nanojet only now. Idempotent; returns availability."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if not is_nanojet_available():
        return False
    _REGISTERED = True

    from nanojet_kernels import ops

    def _check_activation(name: str, activation: torch.Tensor, weight: torch.Tensor) -> None:
        """Reject what the kernel cannot take, while there is still a Python frame to see."""
        if activation.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"{name} expects an e4m3 activation, got {activation.dtype}. "
                "The producer was expected to quantize in its epilogue."
            )
        if activation.shape[-1] != weight.shape[1]:
            raise RuntimeError(
                f"{name} shape mismatch: activation {tuple(activation.shape)} against weight "
                f"{tuple(weight.shape)} — the activation was probably taken before a reshape "
                "that flattens the head dimension."
            )

    @torch.library.custom_op("auto_deploy::nanojet_gemm_fp8_add_inplace", mutates_args={"residual"})
    def nanojet_gemm_fp8_add_inplace(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        input_scale: float,
        weight_scale: float,
    ) -> None:
        """``residual += (x @ weight^T) * input_scale * weight_scale``, in the epilogue."""
        activation = hidden_states.reshape(-1, hidden_states.shape[-1])
        accumulator = residual.view(-1, residual.shape[-1])
        _check_activation("nanojet_gemm_fp8_add", activation, weight)
        if accumulator.shape[0] != activation.shape[0] or accumulator.shape[-1] != weight.shape[0]:
            raise RuntimeError(
                f"nanojet_gemm_fp8_add accumulator {tuple(accumulator.shape)} does not match "
                f"the projection output [{activation.shape[0]}, {weight.shape[0]}]."
            )
        ops.gemm_fp8_add(accumulator, activation, weight, input_scale, weight_scale)

    @nanojet_gemm_fp8_add_inplace.register_fake
    def _(hidden_states, weight, residual, input_scale, weight_scale):
        return

    return True
