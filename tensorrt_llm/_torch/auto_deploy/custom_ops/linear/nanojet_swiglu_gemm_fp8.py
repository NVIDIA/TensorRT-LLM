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

"""nanojet gated SwiGLU GEMM.

Both projections, the activation and the multiply in one CUTLASS gated GEMM, emitting e4m3
so ``down_proj`` needs no quantize either — four kernels collapse to one. This is the shape
nanojet runs natively; TensorRT LLM has no equivalent, because its own SwiGLU fusion needs
gate and up to come from one already-fused GEMM and ``fuse_gemms`` is disabled upstream.
"""

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

    @torch.library.custom_op("auto_deploy::nanojet_swiglu_gemm_fp8", mutates_args=())
    def nanojet_swiglu_gemm_fp8(
        hidden_states: torch.Tensor,
        input_scale_tensor: torch.Tensor,
        gate_up_weight: torch.Tensor,
        input_scale: float,
        up_weight_scale: float,
        gate_weight_scale: float,
        output_scale: float,
    ) -> torch.Tensor:
        """``e4m3(silu(x @ gate^T) * (x @ up^T))`` in one launch.

        ``gate_up_weight`` is ``[up; gate]`` stacked — up first, which is the order nanojet's
        kernel indexes. ``hidden_states`` is already e4m3, quantized by the RMSNorm epilogue
        that produced it. All scales are host constants, folded at graph-build time.
        """
        shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, shape[-1])
        if flattened.dtype != torch.float8_e4m3fn:
            flattened, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                flattened, input_scale_tensor
            )
        output = ops.swiglu(
            flattened, gate_up_weight, input_scale, up_weight_scale, gate_weight_scale, output_scale
        )
        return output.view(*shape[:-1], output.shape[-1])

    @nanojet_swiglu_gemm_fp8.register_fake
    def _nanojet_swiglu_gemm_fp8_fake(
        hidden_states: torch.Tensor,
        input_scale_tensor: torch.Tensor,
        gate_up_weight: torch.Tensor,
        input_scale: float,
        up_weight_scale: float,
        gate_weight_scale: float,
        output_scale: float,
    ) -> torch.Tensor:
        intermediate = gate_up_weight.shape[0] // 2
        return torch.empty(
            *hidden_states.shape[:-1],
            intermediate,
            dtype=torch.float8_e4m3fn,
            device=hidden_states.device,
        )

    return True
