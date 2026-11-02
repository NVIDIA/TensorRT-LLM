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

"""FP8 per-tensor GEMMs for the MiniMax-H3 VAE ViT decoder (opt-in, ``H3_L14_VAE_FP8=1``).

The decoder's 36 transformer blocks run once per clip in fp16 autocast and are GEMM-bound
(fp16 ``nvjet`` kernels are 57 % of the decode on R200). Every ``nn.Linear`` under the
decoder blocks is replaced by :class:`Fp8Linear`: the weight is quantized once to e4m3 with a
static per-tensor scale, the activation is quantized per call with a dynamic per-tensor scale,
and ``torch._scaled_mm`` produces the fp16 output with the original bias. This changes the
decoded pixels (a format change, gated by the quality worker), not the latents.
"""

import torch
from torch import nn

from tensorrt_llm.logger import logger

E4M3_MAX = 448.0


class Fp8Linear(nn.Module):
    """``nn.Linear`` replacement: e4m3 weight (static per-tensor scale), e4m3 activation (dynamic
    per-tensor scale), fp32 accumulation, fp16 output via ``torch._scaled_mm``."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach().to(torch.float32)
        amax = weight.abs().amax().clamp(min=1e-12)
        w_scale = (amax / E4M3_MAX).to(torch.float32).reshape(())
        w_fp8 = (weight / w_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("weight_fp8", w_fp8.contiguous())  # (out, in) row-major
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(torch.float16).contiguous())
        else:
            self.bias = None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        x_scale = (x.abs().amax().to(torch.float32).clamp(min=1e-12) / E4M3_MAX).reshape(())
        x_fp8 = (x.to(torch.float32) / x_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # (in, out) column-major, as cuBLASLt FP8 requires
            scale_a=x_scale,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.float16,
        )
        return out.reshape(*shape[:-1], self.out_features)


def convert_decoder_linears_fp8(decoder: nn.Module) -> int:
    """Replace every ``nn.Linear`` under ``decoder.transformer_blocks`` with :class:`Fp8Linear`.

    ``proj_in`` (K = 24) and ``proj_out`` run once per decoder call and stay fp16. Returns the
    number of replaced modules. Call after ``_prepare_vae_decoder`` and before ``torch_compile``.
    """
    replaced = 0
    for block in decoder.transformer_blocks:
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Fp8Linear(child))
                    replaced += 1
    logger.info(
        f"MiniMax-H3 VAE decoder: {replaced} Linears converted to FP8 per-tensor"
    )
    return replaced
