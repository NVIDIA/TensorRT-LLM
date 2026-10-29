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
"""Small-M BF16 dense MLP for the dense decoder layers (``dense_mlp``).

Replaces the rank-local compute of a BF16 ``GatedMLP`` with plain SwiGLU at
decode row counts: ``gate_up_proj`` GEMM, SwiGLU and ``down_proj`` GEMM become
two weight-streaming kernels (``native_ops.dense_mlp``). The result is
the same rank-local partial that native ``down_proj`` returns when its
all-reduce is disabled; the caller's all-reduce, residual and normalization
stay native. Intermediate rounding follows the native path: gate and up are
rounded to BF16 before SwiGLU and the SwiGLU output is rounded to BF16.

Every eligibility check reads only module attributes and tensor metadata, so a
call that is not eligible returns ``None`` before any device work and the
caller runs the native module.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ... import native_ops

# Largest activation row count of the kernels (MMA N side).
MAX_ROWS = 8
# Contraction sizes must be multiples of this for the default tilings
# (warps x 32 contraction elements per step x steps per chunk = 1024 for both).
CONTRACTION_MULTIPLE = 1024
# Output-column granularity of the gate/up (8) and down (16) CTA tiles.
GATE_UP_COL_MULTIPLE = 8
DOWN_ROW_MULTIPLE = 16
# Tiling variants of the gate/up and down kernels (see denseMlp.cu) for
# the GLM-5.2 GEN shapes (M = 6, H = 6144, rank-local I = 3072): gate/up
# 8 warps x 4 steps at three CTAs per SM, down 4 warps x 8 steps, both with a
# one-chunk L2 prefetch.
GATE_UP_VARIANT = 1
DOWN_VARIANT = 0


def _linear_is_plain_bf16(linear) -> bool:
    from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

    weight = getattr(linear, "weight", None)
    return (
        isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod)
        and weight is not None
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_contiguous()
        and getattr(linear, "bias", None) is None
    )


def _output_allreduce_requested(mlp, final_all_reduce_params) -> bool:
    down = mlp.down_proj
    if not getattr(down, "reduce_output", False):
        return False
    if down.mapping.tp_size == 1:
        return False
    return final_all_reduce_params is None or final_all_reduce_params.enable_allreduce


def check_eligibility(
    mlp, x, final_all_reduce_params=None, lora_params: Optional[dict] = None
) -> Tuple[bool, str]:
    """Decide on the host whether ``mlp(x)`` can run on the dense_mlp kernels.

    Returns:
        ``(eligible, reason)``; ``reason`` names the first failed condition.
    """
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    if not isinstance(mlp, GatedMLP):
        return False, "not a GatedMLP"
    if getattr(getattr(mlp, "mapping", None), "enable_attention_dp", False):
        # With attention DP the dense MLP runs unsharded (4x the rank-local
        # intermediate of the validated TP8 GEN shapes); outside the domain.
        return False, "attention data parallelism"
    if lora_params:
        return False, "LoRA"
    if not isinstance(x, torch.Tensor):
        return False, "input is not a plain tensor"
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2:
        return False, "input is not a 2-D BF16 CUDA tensor"
    if not x.is_contiguous() or x.data_ptr() % 16 != 0:
        return False, "input is not contiguous and 16-byte aligned"
    num_rows, hidden = x.shape
    if num_rows < 1 or num_rows > MAX_ROWS:
        return False, f"row count outside [1, {MAX_ROWS}]"
    if mlp.split_gate_up:
        return False, "split gate/up projections"
    if mlp.activation is not F.silu or not mlp._is_plain_swiglu() or mlp.swiglu_limit is not None:
        return False, "activation is not plain SwiGLU"
    if not (_linear_is_plain_bf16(mlp.gate_up_proj) and _linear_is_plain_bf16(mlp.down_proj)):
        return False, "projections are not unquantized BF16 without bias"
    if _output_allreduce_requested(mlp, final_all_reduce_params):
        return False, "down_proj all-reduce requested"
    gate_up_w = mlp.gate_up_proj.weight
    down_w = mlp.down_proj.weight
    inter = down_w.shape[1]
    if gate_up_w.shape[0] != 2 * inter or gate_up_w.shape[1] != hidden or down_w.shape[0] != hidden:
        return False, "weight shapes do not match the input"
    if (
        hidden % CONTRACTION_MULTIPLE != 0
        or inter % CONTRACTION_MULTIPLE != 0
        or inter % GATE_UP_COL_MULTIPLE != 0
        or hidden % DOWN_ROW_MULTIPLE != 0
    ):
        return False, "hidden/intermediate sizes not supported by the tiling"
    if gate_up_w.device != x.device or down_w.device != x.device:
        return False, "weights on a different device"
    return True, ""


def dense_mlp_forward(
    mlp, x, final_all_reduce_params=None, lora_params: Optional[dict] = None
) -> Optional[torch.Tensor]:
    """Run the dense MLP on the dense_mlp kernels, or return ``None`` if ineligible.

    The returned tensor is the rank-local ``down_proj`` output (no all-reduce),
    exactly what ``mlp(x, final_all_reduce_params=...)`` returns when its
    all-reduce is disabled.
    """
    from ... import dispatch

    eligible, reason = check_eligibility(mlp, x, final_all_reduce_params, lora_params)
    dispatch.log_capability("dense_mlp", f"dense_mlp:{reason or 'engaged'}", eligible, reason)
    if not eligible:
        return None
    return native_ops.dense_mlp(
        x, mlp.gate_up_proj.weight, mlp.down_proj.weight, GATE_UP_VARIANT, DOWN_VARIANT
    )
