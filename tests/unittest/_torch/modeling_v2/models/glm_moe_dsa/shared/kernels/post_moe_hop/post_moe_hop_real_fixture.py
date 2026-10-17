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
"""Real-weight native MoE fixture for one GLM-5.2 NVFP4 target layer.

Builds, on every TP rank, the native TRT-LLM pieces that feed the post-MoE
hop of the served GEN role (TP8/EP1, ``moe_backend="TRTLLM"``):

* ``DeepseekV3Gate`` with the layer's real router weight and correction bias;
* the native NVFP4 MoE (``create_moe`` -> TRTLLM-gen NVFP4 leaf) with the
  layer's real 256 routed experts (TP-sliced by the module's own loader);
* the rank's TP slice of the real BF16 shared expert (evaluated with torch);
* the real RMSNorm weight of the next layer (``input_layernorm`` of
  ``layer + 1``, or ``model.norm`` after the last layer).

Weights are real; activations are synthetic (a unit-RMS random hidden state
scaled by the layer's real ``post_attention_layernorm`` weight, and a random
residual), identical on every rank as TP requires.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from utils.llm_data import llm_models_root

CKPT = os.environ.get(
    "GLM_MOE_DSA_POST_MOE_HOP_GLM52_CKPT",
    os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4"),
)
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 6144
MOE_INTER = 2048
NUM_LAYERS = 78
ROUTED_SCALING = 2.5
EPS = 1e-5


def _load(names: List[str]) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: Dict[str, List[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    out = {}
    for shard, shard_names in sorted(by_shard.items()):
        with safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu") as f:
            for name in shard_names:
                out[name] = f.get_tensor(name)
    return out


@dataclass
class RealMoELayer:
    layer: int
    gate: torch.nn.Module
    moe: torch.nn.Module
    shared_gate: torch.Tensor
    shared_up: torch.Tensor
    shared_down: torch.Tensor
    post_attn_gamma: torch.Tensor
    next_gamma: torch.Tensor
    expert_input_scale: float

    def shared_output(self, x: torch.Tensor) -> torch.Tensor:
        """Rank-local TP partial of the BF16 shared expert (no reduction)."""
        gate = torch.nn.functional.linear(x, self.shared_gate)
        up = torch.nn.functional.linear(x, self.shared_up)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, self.shared_down)

    def hidden_states(self, rows: int, seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(rows, HIDDEN, device="cuda", generator=gen)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        return (x * self.post_attn_gamma.float()).bfloat16()

    @staticmethod
    def residual(rows: int, seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cuda").manual_seed(seed + 1)
        return (torch.randn(rows, HIDDEN, device="cuda", generator=gen) * 4.0).bfloat16()

    def run_moe(self, x: torch.Tensor, router_logits: torch.Tensor, do_finalize: bool):
        return self.moe.forward(
            x, router_logits, do_finalize=do_finalize, output_dtype=torch.bfloat16
        )


def build_real_moe_layer(mapping, layer: int, max_num_tokens: int = 64) -> RealMoELayer:
    """Load layer ``layer`` of the checkpoint into the native modules on this rank."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
    from tensorrt_llm._torch.moe.fused_moe import create_moe
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    prefix = f"model.layers.{layer}"
    next_norm = (
        "model.norm.weight"
        if layer + 1 >= NUM_LAYERS
        else f"model.layers.{layer + 1}.input_layernorm.weight"
    )
    names = [
        f"{prefix}.mlp.gate.weight",
        f"{prefix}.mlp.gate.e_score_correction_bias",
        f"{prefix}.post_attention_layernorm.weight",
        next_norm,
    ] + [f"{prefix}.mlp.shared_experts.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    hf_to_native = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    for expert in range(NUM_EXPERTS):
        for proj in hf_to_native:
            names += [f"{prefix}.mlp.experts.{expert}.{proj}.{s}" for s in suffixes]
    tensors = _load(names)

    pretrained = PretrainedConfig()
    pretrained.num_experts = NUM_EXPERTS
    pretrained.n_routed_experts = NUM_EXPERTS
    pretrained.hidden_size = HIDDEN
    pretrained.intermediate_size = MOE_INTER
    pretrained.moe_intermediate_size = MOE_INTER
    pretrained.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained,
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        moe_backend="TRTLLM",
        max_num_tokens=max_num_tokens,
    )

    with torch.device("cuda"):
        gate = DeepseekV3Gate(
            HIDDEN,
            NUM_EXPERTS,
            top_k=TOP_K,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=ROUTED_SCALING,
            dtype=torch.bfloat16,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend="TRTLLM",
        )
        gate.load_weights(
            [
                {
                    "weight": tensors[f"{prefix}.mlp.gate.weight"],
                    "e_score_correction_bias": tensors[
                        f"{prefix}.mlp.gate.e_score_correction_bias"
                    ],
                }
            ]
        )
        moe = create_moe(
            routing_method=gate.routing_method,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size=MOE_INTER,
            dtype=torch.bfloat16,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer,
            allow_backend_degradation=False,
        )
        weights = {}
        for expert in range(NUM_EXPERTS):
            for proj, native in hf_to_native.items():
                for s in suffixes:
                    weights[f"{expert}.{native}.{s}"] = tensors[
                        f"{prefix}.mlp.experts.{expert}.{proj}.{s}"
                    ]
        moe.load_weights([weights])
        moe.post_load_weights()
    moe.cuda()

    tp, rank = mapping.tp_size, mapping.tp_rank
    width = MOE_INTER // tp
    rows = slice(rank * width, (rank + 1) * width)
    shared_gate = tensors[f"{prefix}.mlp.shared_experts.gate_proj.weight"][rows].cuda()
    shared_up = tensors[f"{prefix}.mlp.shared_experts.up_proj.weight"][rows].cuda()
    shared_down = (
        tensors[f"{prefix}.mlp.shared_experts.down_proj.weight"][:, rows].contiguous().cuda()
    )
    input_scale = float(tensors[f"{prefix}.mlp.experts.0.gate_proj.input_scale"])
    return RealMoELayer(
        layer,
        gate,
        moe,
        shared_gate,
        shared_up,
        shared_down,
        tensors[f"{prefix}.post_attention_layernorm.weight"].cuda(),
        tensors[next_norm].cuda(),
        input_scale,
    )
