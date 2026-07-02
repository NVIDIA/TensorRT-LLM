# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Wan 2.2 fused DiT kernel helpers.

Provides thin wrappers around the trtllm::fused_dit_layernorm_shift_scale[_quant]
custom ops used for the three LayerNorm sites in WanBlock.
"""

from typing import Optional, Union

import torch

from tensorrt_llm._torch.utils import Fp4QuantizedTensor


def get_nvfp4_input_scale(linear) -> Optional[torch.Tensor]:
    """Return the calibrated NVFP4 input_scale for a Linear, or None if not applicable.

    Returns None when the linear is not NVFP4-quantized, uses a non-16 group size,
    has an AWQ pre_quant_scale that must be folded into bf16 input first, or uses
    dynamic quantization (which recomputes input_scale per-forward and cannot
    consume a statically pre-quantized FP4 tensor).
    """
    if linear is None:
        return None
    scale = getattr(linear, "input_scale", None)
    if scale is None:
        return None
    if getattr(linear, "scaling_vector_size", None) != 16:
        return None
    if getattr(linear, "pre_quant_scale", None) is not None:
        return None
    if getattr(linear, "force_dynamic_quantization", False):
        return None
    return scale


def apply_fused_layernorm_adaln_quant(
    x: torch.Tensor,
    scale_msa: torch.Tensor,
    shift_msa: torch.Tensor,
    seq_len_per_batch: int,
    fp4_input_scale: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Union[torch.Tensor, "Fp4QuantizedTensor"]:
    """Fused LayerNorm + AdaLN (y = (1 + scale_msa) * x_hat + shift_msa) + optional NVFP4 quant.

    Used for norm1 and norm3 in WanBlock (no learned affine params; modulation from timestep emb).
    Returns Fp4QuantizedTensor when fp4_input_scale is provided, else a bf16 tensor.
    """
    # .contiguous() handles non-contiguous views (chunk/squeeze/reshape patterns) and
    # transposed layouts injected by torch.compile's inductor memory planner.
    x = x.contiguous()
    scale_msa = scale_msa.to(dtype=x.dtype).contiguous()
    shift_msa = shift_msa.to(dtype=x.dtype).contiguous()
    if fp4_input_scale is not None:
        y_fp4, sf_out = torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
            x, None, None, scale_msa, shift_msa, fp4_input_scale, seq_len_per_batch, eps
        )
        return Fp4QuantizedTensor(y_fp4, sf_out)
    out = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
        x, None, None, scale_msa, shift_msa, seq_len_per_batch, eps
    )
    return out


def apply_fused_layernorm_affine_quant(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fp4_input_scale: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Union[torch.Tensor, "Fp4QuantizedTensor"]:
    """Fused LayerNorm + affine (learned weight/bias) + optional NVFP4 quant.

    Used for norm2 in WanBlock (learned LN params; no AdaLN modulation).
    Returns Fp4QuantizedTensor when fp4_input_scale is provided, else a bf16 tensor.
    """
    x = x.contiguous()
    # seq_len_per_batch is unused for the affine path; pass M as a safe value.
    seq_len_per_batch = x.shape[0]
    if fp4_input_scale is not None:
        y_fp4, sf_out = torch.ops.trtllm.fused_dit_layernorm_shift_scale_quant(
            x,
            ln_weight.to(x.dtype),
            ln_bias.to(x.dtype),
            None,
            None,
            fp4_input_scale,
            seq_len_per_batch,
            eps,
        )
        return Fp4QuantizedTensor(y_fp4, sf_out)
    out = torch.ops.trtllm.fused_dit_layernorm_shift_scale(
        x, ln_weight.to(x.dtype), ln_bias.to(x.dtype), None, None, seq_len_per_batch, eps
    )
    return out
