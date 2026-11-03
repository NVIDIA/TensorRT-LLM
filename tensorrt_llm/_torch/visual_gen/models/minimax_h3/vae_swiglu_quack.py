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
"""MiniMax-H3 ViT VAE decoder FFN SwiGLU in the ``ff.net[0].proj`` GEMM epilogue (SM100 QuACK).

Diffusers' ``SwiGLU`` writes the ``[M, 2I]`` projection and a second kernel reads it back for
``hidden * silu(gate)``. QuACK's gated SM100 GEMM (``gemm_act``) adds the bias to the FP32
accumulators, applies the activation there and stores only the ``[M, I]`` product, rounded once.

``fuse_vae_swiglu`` stores each fused projection's rows as ``[gate; hidden]`` (Diffusers stores
``[hidden; gate]``) so that QuACK's ``swiglu`` reads the gate from the first half.
"""

from typing import Optional

import torch
from diffusers.models.activations import SwiGLU

from tensorrt_llm._utils import get_sm_version

from .gated_mlp_quack import gemm_act_config


def _fused(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    from quack.gemm_interface import gemm_act

    if x.shape[0] == 0:
        return x.new_empty(0, weight.shape[0] // 2)
    _, out = gemm_act(
        x,
        weight.t(),
        # An fp32 bias keeps the epilogue's 32-bit vector loads on its interleaved view.
        bias=None if bias is None else bias.float(),
        activation="swiglu",
        store_preact=False,
        config=gemm_act_config(torch.cuda.get_device_capability(x.device)[0]),
        concat_layout=("B",) if bias is None else ("B", "bias"),
    )
    return out


def vae_swiglu_shape_ok(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> bool:
    """fp16/bf16 ``[2I, K]`` weight (``[2I]`` bias) with K, I multiples of 8 (16-byte TMA rows)."""
    n, k = weight.shape
    return (
        weight.dtype in (torch.float16, torch.bfloat16)
        and n % 16 == 0
        and k % 8 == 0
        and (bias is None or (bias.dtype == weight.dtype and bias.shape == (n,)))
    )


@torch.library.custom_op(
    "trtllm::minimax_h3_vae_swiglu_quack",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def vae_swiglu_quack(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """``hidden * silu(gate)`` for ``x @ W^T + b = [gate, hidden]``.

    Args:
        x: ``[..., K]`` activations in the weight's dtype. The op flattens the leading dimensions
            itself: a flatten in the caller makes Inductor split the producing norm kernel in two.
        weight: ``[2I, K]``, gate rows first (``fuse_vae_swiglu``'s order).
        bias: ``[2I]`` or None.

    Returns:
        Contiguous ``[..., I]``.
    """
    if (
        x.ndim < 2
        or x.dtype != weight.dtype
        or x.shape[-1] != weight.shape[1]
        or not vae_swiglu_shape_ok(weight, bias)
    ):
        raise ValueError(
            "minimax_h3_vae_swiglu_quack needs x [..., K] and weight [2I, K] (bias [2I]) of one "
            f"fp16/bf16 dtype with K, I multiples of 8; got x {tuple(x.shape)} {x.dtype}, weight "
            f"{tuple(weight.shape)} {weight.dtype}, "
            f"bias {None if bias is None else (tuple(bias.shape), bias.dtype)}"
        )
    out = _fused(x.reshape(-1, x.shape[-1]).contiguous(), weight.contiguous(), bias)
    return out.view(*x.shape[:-1], out.shape[-1])


@vae_swiglu_quack.register_fake
def _(x, weight, bias):
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


class QuackSwiGLU(SwiGLU):
    """Diffusers ``SwiGLU`` run as ``trtllm::minimax_h3_vae_swiglu_quack``; ``proj`` rows are ``[gate; hidden]``."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.proj.weight
        return torch.ops.trtllm.minimax_h3_vae_swiglu_quack(
            hidden_states.to(weight.dtype), weight, self.proj.bias
        )


def _gate_first(t: torch.Tensor) -> torch.Tensor:
    hidden, gate = t.chunk(2, dim=0)
    return torch.cat((gate, hidden))


@torch.no_grad()
def fuse_vae_swiglu(decoder: torch.nn.Module) -> int:
    """Turn each eligible Diffusers ``SwiGLU`` under ``decoder`` into a ``QuackSwiGLU``.

    Each fused module keeps its ``proj`` parameter names with the rows reordered to ``[gate; hidden]``.
    Returns how many changed.
    """
    if not (torch.cuda.is_available() and get_sm_version() in (100, 103)):
        return 0
    fused = 0
    for module in decoder.modules():
        if type(module) is SwiGLU and vae_swiglu_shape_ok(module.proj.weight, module.proj.bias):
            module.proj.weight.copy_(_gate_first(module.proj.weight))
            if module.proj.bias is not None:
                module.proj.bias.copy_(_gate_first(module.proj.bias))
            module.__class__ = QuackSwiGLU
            fused += 1
    return fused
