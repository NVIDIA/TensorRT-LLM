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
"""Numerical forward coverage for the Gemma4 shared-KV assistant on Blackwell."""

import unittest

import torch
from transformers import Gemma4TextConfig

import tensorrt_llm
from tensorrt_llm._torch.attention.backends import FlashInferAttention, FlashInferAttentionMetadata
from tensorrt_llm._torch.configs.gemma4 import Gemma4AssistantConfig
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4AssistantForCausalLM, Gemma4ForCausalLM
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


class TestGemma4Assistant(unittest.TestCase):
    @torch.no_grad()
    def test_assistant_forward_matches_reference(self):
        torch.manual_seed(0)
        text_config = {
            "vocab_size": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "global_head_dim": 128,
            "num_global_key_value_heads": 1,
            "hidden_activation": "gelu_pytorch_tanh",
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "sliding_window": 4,
            "attention_k_eq_v": True,
            "enable_moe_block": False,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": 0,
            "use_double_wide_mlp": False,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 10000.0,
                },
            },
            "torch_dtype": "bfloat16",
            "tie_word_embeddings": True,
        }
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        target_config = ModelConfig(
            pretrained_config=Gemma4TextConfig(**text_config),
            mapping=mapping,
            attn_backend="FLASHINFER",
        )
        target = Gemma4ForCausalLM(target_config).cuda().eval()
        assistant_config = ModelConfig(
            pretrained_config=Gemma4AssistantConfig(
                text_config={**text_config, "num_kv_shared_layers": 1},
                backbone_hidden_size=256,
                use_ordered_embeddings=False,
                num_centroids=16,
                centroid_intermediate_top_k=2,
                tie_word_embeddings=True,
                dtype="bfloat16",
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            ),
            mapping=mapping,
            attn_backend="FLASHINFER",
        )
        model_class = get_registered_model_class("Gemma4AssistantForCausalLM")
        self.assertIs(model_class, Gemma4AssistantForCausalLM)
        assistant = model_class(assistant_config).cuda().eval()
        for parameter in target.parameters():
            if parameter.ndim == 2:
                torch.nn.init.normal_(parameter, std=0.02)
            else:
                parameter.fill_(1)
        for parameter in assistant.parameters():
            if parameter.ndim == 2:
                torch.nn.init.normal_(parameter, std=0.02)
            else:
                parameter.fill_(1)
        assistant.load_weights_from_target_model(target)
        layer = assistant.model.layers[0]
        self.assertTrue(layer.is_kv_shared_layer)

        manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=128, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=128,
            tokens_per_block=32,
            max_seq_len=128,
            max_batch_size=1,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        self.addCleanup(manager.shutdown)
        requests = manager.add_dummy_requests([0], token_nums=[3])
        self.assertIsNotNone(requests)
        manager.get_buffers(0).zero_()
        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.tensor([3], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            workspace_buffer=torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            max_num_requests=1,
            max_num_tokens=3,
            kv_cache_manager=manager,
            request_ids=[0],
        )
        metadata.prepare()
        num_heads, head_dim, num_kv_heads = 4, 128, 1
        keys = torch.randn(3, num_kv_heads * head_dim, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        cache_writer = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            flashinfer_backend="trtllm-gen",
        )
        cache_writer.forward(
            torch.zeros(3, num_heads * head_dim, device="cuda", dtype=torch.bfloat16),
            keys,
            values,
            metadata,
        )
        draft_metadata = metadata.get_draft_metadata()
        cached_kv = manager.get_buffers(0).clone()
        input_ids = torch.tensor([5], device="cuda", dtype=torch.int)
        position_ids = torch.tensor([[2]], device="cuda", dtype=torch.int)
        recurrent = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)

        def linear(x, weight):
            return torch.nn.functional.linear(x, weight)

        def norm(x, weight, eps):
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            return (normalized * weight.float()).to(x.dtype)

        # The prefix is already in cache coordinates; only the assistant Q needs RoPE.
        embeddings = (
            torch.nn.functional.embedding(input_ids.long(), target.model.embed_tokens.weight)
            * target.model.embed_tokens.embed_scale
        )
        assistant_inputs = linear(
            torch.cat([embeddings, recurrent], dim=-1), assistant.pre_projection.weight
        )
        eps = text_config["rms_norm_eps"]
        normalized = norm(assistant_inputs, layer.input_layernorm.weight, eps)
        query = linear(normalized, layer.self_attn.qkv_proj.weight).reshape(1, num_heads, head_dim)
        query = norm(query, layer.self_attn.q_norm.weight, eps)
        angles = 2 * 10000.0 ** (
            -torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim
        )
        cos = torch.cat([angles.cos(), angles.cos()])
        sin = torch.cat([angles.sin(), angles.sin()])
        first, second = query.float().chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        query = (query.float() * cos + rotated * sin).to(torch.bfloat16)
        reference_keys = keys.reshape(3, num_kv_heads, head_dim).repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        reference_values = values.reshape(3, num_kv_heads, head_dim).repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        scores = torch.einsum("qhd,khd->hqk", query.float(), reference_keys.float())
        attention = (
            torch.einsum("hqk,khd->qhd", scores.softmax(-1), reference_values.float())
            .reshape(1, -1)
            .to(torch.bfloat16)
        )
        attention = linear(attention, layer.self_attn.o_proj.weight)
        hidden = assistant_inputs + norm(attention, layer.post_attention_layernorm.weight, eps)
        residual = hidden
        hidden = norm(hidden, layer.pre_feedforward_layernorm.weight, eps)
        gate, up = linear(hidden, layer.mlp.gate_up_proj.weight).chunk(2, dim=-1)
        hidden = (torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()).to(
            torch.bfloat16
        )
        hidden = linear(hidden, layer.mlp.down_proj.weight)
        hidden = residual + norm(hidden, layer.post_feedforward_layernorm.weight, eps)
        hidden = (hidden * layer.layer_scalar).to(torch.bfloat16)
        hidden = norm(hidden, assistant.model.norm.weight, eps)
        expected_logits = torch.nn.functional.linear(hidden, assistant.lm_head.weight).float()
        expected_recurrent = torch.nn.functional.linear(hidden, assistant.post_projection.weight)

        logits, projected_hidden_states = assistant.forward_draft_step(
            input_ids=input_ids,
            position_ids=position_ids,
            recurrent_hidden_states=recurrent,
            attn_metadata=draft_metadata,
        )
        torch.testing.assert_close(logits, expected_logits, atol=0.05, rtol=0.05)
        torch.testing.assert_close(
            projected_hidden_states, expected_recurrent, atol=0.05, rtol=0.05
        )
        forward_logits = assistant(
            attn_metadata=draft_metadata,
            position_ids=position_ids,
            inputs_embeds=assistant_inputs,
            return_context_logits=True,
        )
        torch.testing.assert_close(forward_logits, expected_logits, atol=0.05, rtol=0.05)
        torch.testing.assert_close(manager.get_buffers(0), cached_kv, atol=0, rtol=0)
