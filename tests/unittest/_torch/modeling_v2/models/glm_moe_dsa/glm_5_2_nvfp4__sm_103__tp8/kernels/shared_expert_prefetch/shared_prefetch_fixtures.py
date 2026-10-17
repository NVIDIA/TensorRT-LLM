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
"""Real GLM-5.2 shared-expert fixtures for the shared_expert_prefetch tests.

* ``build_shared_mlp(layer, tp_rank)``: the served BF16 shared expert of one
  MoE layer (one TP8 rank shard), built through ``GatedMLP``'s constructor
  with the served model configuration and loaded from the checkpoint's
  ``shared_experts.{gate,up,down}_proj`` tensors through its own loaders (the
  rank slicing is the module's).
* ``load_real_shared(mlp, layer)``: the same loading into an existing module
  (the chain bench's MoE modules carry random shared-expert weights).
* ``reference(mlp, x)``: the shared expert with the native rounding points and
  exact (FP64) GEMM sums: ``h = bf16(x attention_output_into_hop^T)``, ``a`` = the native Triton
  SiLU-and-mul of ``h``, ``y = bf16(a W2^T)``.
"""

import copy
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "routed_moe"))
import routed_moe_fixtures as fx  # noqa: E402

TARGET_LAYERS = tuple(range(3, 78))


def _shared_tensors(layer: int):
    rd = fx._Reader()
    pre = f"model.layers.{layer}.mlp.shared_experts"
    return {
        p: rd.full(f"{pre}.{p}.weight").to(torch.bfloat16)
        for p in ("gate_proj", "up_proj", "down_proj")
    }


def load_real_shared(mlp, layer: int) -> None:
    w = _shared_tensors(layer)
    dev = mlp.gate_up_proj.weight.device
    mlp.gate_up_proj.load_weights([{"weight": w["gate_proj"]}, {"weight": w["up_proj"]}])
    mlp.down_proj.load_weights([{"weight": w["down_proj"]}])
    fx._walk_post_load(mlp)
    assert mlp.gate_up_proj.weight.device == dev


def build_shared_mlp(layer: int, tp_rank: int):
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm.models.modeling_utils import QuantConfig

    mc = fx._served_model_config(layer, tp_rank)
    smc = copy.copy(mc)
    smc.quant_config = QuantConfig()
    mlp = GatedMLP(
        hidden_size=fx.HIDDEN,
        intermediate_size=fx.INTER,
        bias=False,
        dtype=torch.bfloat16,
        config=smc,
        overridden_tp_size=fx.TP_SIZE,
        reduce_output=False,
    ).cuda()
    load_real_shared(mlp, layer)
    return mlp


def reference(mlp, x: torch.Tensor) -> torch.Tensor:
    """FP64 GEMM sums with the native BF16 rounding points (see the module docstring)."""
    w1 = mlp.gate_up_proj.weight.double()
    w2 = mlp.down_proj.weight.double()
    h = (x.double() @ w1.t()).to(torch.bfloat16)
    a = torch.ops.trtllm.silu_and_mul(h)
    return (a.double() @ w2.t()).to(torch.bfloat16)


def native_in_batch(mlp, x: torch.Tensor, batch: int = 64) -> torch.Tensor:
    """The native module on ``x`` placed in a ``batch``-row input (another cuBLAS
    algorithm than at R rows): native algorithm-to-algorithm spread."""
    big = torch.zeros(batch, x.shape[1], dtype=x.dtype, device=x.device)
    big[: x.shape[0]] = x
    return mlp(big)[: x.shape[0]].contiguous()


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|ordinal(a) - ordinal(b)| of BF16 values (sign-magnitude to ordinal)."""

    def ordinal(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordinal(a) - ordinal(b)).abs()
