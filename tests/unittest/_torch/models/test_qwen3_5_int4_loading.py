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

import pytest
import torch
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


@pytest.mark.skipif(get_sm_version() < 80, reason="W4A16 AWQ requires Ampere or newer")
@pytest.mark.parametrize("has_pre_quant_scale", [False, True])
def test_int4_mapper_to_gdn_tp2(monkeypatch: pytest.MonkeyPatch, has_pre_quant_scale: bool) -> None:
    """Load mapper output through real GDN projections and check each TP rank's outputs."""
    hidden_size, tp_size = 256, 2
    # The unused embedding/MLP collectives must not allocate multi-GPU workspaces.
    monkeypatch.setattr(
        "tensorrt_llm._torch.distributed.ops.get_allreduce_workspace", lambda _: None
    )
    prefix = "model.layers.0.linear_attn"
    generator = torch.Generator().manual_seed(6562942)
    weights, reference_weights = {}, {}
    for name, rows in (("q", 2048), ("k", 2048), ("v", 6144), ("z", 6144), ("b", 48), ("a", 48)):
        # Two nonzero inputs per row keep the independent reference exactly representable.
        logical = torch.zeros(rows, hidden_size, dtype=torch.int8)
        row_ids = torch.arange(rows)
        for group in range(2):
            columns = group * 128 + torch.randint(128, (rows,), generator=generator)
            logical[row_ids, columns] = torch.randint(
                -8, 8, (rows,), generator=generator, dtype=torch.int8
            )
        if name in ("a", "b"):
            weights[f"{prefix}.in_proj_{name}.weight"] = logical.to(torch.bfloat16)
            reference_weights[name] = logical.float()
        else:
            scales = 2.0 ** torch.randint(-3, 1, (rows, 2), generator=generator)
            weights[f"{prefix}.in_proj_{name}.weight"] = (
                (logical[0::2] & 15) | ((logical[1::2] & 15) << 4)
            ).to(torch.uint8)
            weights[f"{prefix}.in_proj_{name}.weight_scale"] = scales
            reference_weights[name] = logical.float() * scales.repeat_interleave(128, dim=1)
    for suffix in ("weight", "weight_scale"):
        weights[f"{prefix}.in_proj_qkv.{suffix}"] = torch.cat(
            [weights.pop(f"{prefix}.in_proj_{name}.{suffix}") for name in ("q", "k", "v")]
        )
    input_scale = torch.tensor([0.5, 2.0], dtype=torch.bfloat16).repeat_interleave(128)
    if has_pre_quant_scale:
        for name in ("qkv", "z"):
            weights[f"{prefix}.in_proj_{name}.pre_quant_scale"] = input_scale.clone()
    x = torch.randint(-2, 3, (3, hidden_size), generator=generator).to("cuda", torch.bfloat16)

    # Projection forwards have no collectives, so both real TP mappings run on one GPU.
    for rank in range(tp_size):
        config = Qwen3NextConfig(
            architectures=["Qwen3_5ForCausalLM"],
            hidden_size=hidden_size,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=0,
            vocab_size=256,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            layer_types=["linear_attention"],
            torch_dtype=torch.bfloat16,
        )
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=tp_size, tp_size=tp_size, rank=rank),
            quant_config=QuantConfig(
                quant_algo=QuantAlgo.W4A16_AWQ,
                group_size=128,
                pre_quant_scale=True,
                exclude_modules=[f"{prefix}.in_proj_a", f"{prefix}.in_proj_b"],
            ),
        )
        model = AutoModelForCausalLM.from_config(model_config)
        mapper = Qwen3_5MoeHfWeightMapper()
        mapper.init_model_and_config(model, model_config)
        mapped = mapper.preprocess_weights(weights)
        gdn = model.model.layers[0].linear_attn
        assert model_config.skip_create_weights_in_init
        assert gdn.in_proj_ba.quant_config.quant_algo is None
        for projection, names in (("qkvz", ("q", "k", "v", "z")), ("ba", ("b", "a"))):
            linear = getattr(gdn, f"in_proj_{projection}")
            key_prefix = f"{prefix}.in_proj_{projection}."
            linear.load_weights(
                [
                    {
                        key.removeprefix(key_prefix): value
                        for key, value in mapped.items()
                        if key.startswith(key_prefix)
                    }
                ]
            )
            linear.post_load_weights()
            linear.cuda()
            reference_input = x
            if has_pre_quant_scale and projection == "qkvz":
                reference_input = x * input_scale.cuda()
            expected = torch.cat(
                [
                    torch.nn.functional.linear(
                        reference_input.float(), reference_weights[name].cuda()
                    ).chunk(tp_size, dim=-1)[rank]
                    for name in names
                ],
                dim=-1,
            ).to(torch.bfloat16)
            torch.testing.assert_close(linear(x), expected, rtol=0, atol=0)
