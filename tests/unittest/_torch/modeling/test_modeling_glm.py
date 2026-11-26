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

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_glm import Glm4DecoderLayer, Glm4MoE, Glm4MoeForCausalLM
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class


@torch.no_grad()
def test_glm4_moe_single_token_context_logits():
    config = PretrainedConfig(
        architectures=["Glm4MoeForCausalLM"],
        vocab_size=64,
        hidden_size=512,
        intermediate_size=512,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_bias=False,
        use_qk_norm=False,
        rms_norm_eps=1e-6,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        num_nextn_predict_layers=0,
        torch_dtype=torch.float16,
        tie_word_embeddings=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
    )
    model_class = get_registered_model_class("Glm4MoeForCausalLM")
    assert model_class is Glm4MoeForCausalLM
    model_config = ModelConfig(
        pretrained_config=config,
        attn_backend="VANILLA",
        moe_backend="CUTLASS",
        max_num_tokens=2,
        moe_disable_finalize_fusion=True,
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        model = model_class(model_config).cuda().eval()
        assert isinstance(model, Glm4MoeForCausalLM)
        assert len(model.model.layers) == 1
        layer = model.model.layers[0]
        assert isinstance(layer, Glm4DecoderLayer)
        assert isinstance(layer.mlp, Glm4MoE)

        weights = {}

        def matrix(name, rows, columns):
            weights[name] = torch.randn(rows, columns, device="cuda", dtype=torch.float16) * 0.02

        matrix("model.embed_tokens.weight", 64, 512)
        matrix("lm_head.weight", 64, 512)
        prefix = "model.layers.0"
        for name in (
            "model.norm.weight",
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        ):
            weights[name] = torch.ones(512, device="cuda", dtype=torch.float16)
        for name, rows in (("q_proj", 512), ("k_proj", 256), ("v_proj", 256)):
            matrix(f"{prefix}.self_attn.{name}.weight", rows, 512)
        matrix(f"{prefix}.self_attn.o_proj.weight", 512, 512)
        matrix(f"{prefix}.mlp.gate.weight", 8, 512)
        # Separate selection scores so reduced-precision routing cannot create ties.
        weights[f"{prefix}.mlp.gate.e_score_correction_bias"] = (
            torch.arange(8, device="cuda", dtype=torch.float32) * 2
        )
        expert_prefixes = [f"{prefix}.mlp.experts.{i}" for i in range(config.n_routed_experts)]
        shared_prefix = f"{prefix}.mlp.shared_experts"
        for expert_prefix in [*expert_prefixes, shared_prefix]:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                matrix(f"{expert_prefix}.{projection}.weight", 512, 512)

        model.load_weights(weights)
        layer.mlp.experts.post_load_weights()
        model.post_load_weights()

        input_ids = torch.tensor([3, 11], device="cuda", dtype=torch.int)
        metadata = VanillaAttentionMetadata(
            max_num_requests=2,
            max_num_tokens=2,
            seq_lens=torch.tensor([1, 1], dtype=torch.int),
            num_contexts=2,
            request_ids=[0, 1],
        )
        metadata.prepare()
        logits = model(
            input_ids=input_ids,
            position_ids=torch.zeros((1, 2), device="cuda", dtype=torch.long),
            attn_metadata=metadata,
            return_context_logits=True,
        )

        def project(x, name):
            return F.linear(x, weights[f"{name}.weight"].float())

        def normalize(x, name):
            return (
                x
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + config.rms_norm_eps)
                * weights[f"{name}.weight"].float()
            )

        def expert(x, name):
            return project(
                F.silu(project(x, f"{name}.gate_proj")) * project(x, f"{name}.up_proj"),
                f"{name}.down_proj",
            )

        residual = weights["model.embed_tokens.weight"][input_ids.long()].float()
        hidden = normalize(residual, f"{prefix}.input_layernorm")
        # A singleton request's attention probability is exactly one.
        values = project(hidden, f"{prefix}.self_attn.v_proj")
        values = values.reshape(2, 2, 128).repeat_interleave(2, dim=1).reshape(2, 512)
        residual = residual + project(values, f"{prefix}.self_attn.o_proj")
        hidden = normalize(residual, f"{prefix}.post_attention_layernorm")
        scores = project(hidden, f"{prefix}.mlp.gate").sigmoid()
        bias = weights[f"{prefix}.mlp.gate.e_score_correction_bias"]
        indices = (scores + bias).topk(2, dim=-1).indices
        scales = scores.gather(1, indices)
        scales = scales / scales.sum(-1, keepdim=True)
        expert_outputs = torch.stack([expert(hidden, p) for p in expert_prefixes], dim=1)
        selected = expert_outputs.gather(1, indices.unsqueeze(-1).expand(-1, -1, 512))
        routed = (selected * scales.unsqueeze(-1)).sum(1)
        hidden = normalize(residual + routed + expert(hidden, shared_prefix), "model.norm")
        expected = project(hidden, "lm_head")
        assert logits.shape == (2, 64)
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, expected, atol=0.02, rtol=0.02)
