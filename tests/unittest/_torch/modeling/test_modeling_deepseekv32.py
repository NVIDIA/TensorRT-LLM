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

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig


@torch.inference_mode()
def test_deepseek_v32_context_forward():
    from contextlib import contextmanager

    from transformers import PretrainedConfig

    from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.models.modeling_deepseekv3 import (
        DeepseekV3ForCausalLM,
        DeepseekV32Attention,
    )
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._utils import torch_dtype_to_binding
    from tensorrt_llm.bindings.internal.batch_manager import CacheType

    # Reference MLA geometry keeps context (192) distinct from latent (576)
    # heads: native dispatch identifies generation MLA by head-size equality.
    config = PretrainedConfig(
        architectures=["DeepseekV32ForCausalLM"],
        model_type="deepseek_v32",
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        moe_intermediate_size=128,
        n_routed_experts=None,
        n_shared_experts=1,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=None,
        torch_dtype=torch.bfloat16,
    )
    model_config = ModelConfig(
        pretrained_config=config, attn_backend="TRTLLM", max_num_tokens=32, max_seq_len=64
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17)
        assert get_registered_model_class("DeepseekV32ForCausalLM") is DeepseekV3ForCausalLM
        model = DeepseekV3ForCausalLM(model_config).cuda().eval()
        assert type(model) is DeepseekV3ForCausalLM
        assert len(model.model.layers) == 1
        layer = model.model.layers[0]
        assert type(layer.self_attn) is DeepseekV32Attention
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0.0, std=0.05)
        model.setup_aliases()
        assert layer.next_layer_layernorm is model.model.norm
        # Use MLA's eager path without executor-owned custom-op lookup state.
        layer.self_attn.register_to_config = False

        @contextmanager
        def fresh_metadata(length):
            cache_manager = KVCacheManager(
                KvCacheConfig(max_tokens=64, enable_block_reuse=False),
                CacheType.SELFKONLY,
                num_layers=config.num_hidden_layers,
                num_kv_heads=1,
                head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
                tokens_per_block=64,
                max_seq_len=64,
                max_batch_size=1,
                mapping=model_config.mapping,
                dtype=torch_dtype_to_binding(config.torch_dtype),
            )
            try:
                requests = cache_manager.add_dummy_requests([0], [length])
                assert requests is not None and len(requests) == 1
                cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
                metadata = get_attention_backend("TRTLLM").Metadata(
                    max_num_requests=1,
                    max_num_tokens=32,
                    mapping=model_config.mapping,
                    kv_cache_manager=cache_manager,
                    kv_cache_params=cache_params,
                    request_ids=[0],
                    prompt_lens=[length],
                    seq_lens=torch.tensor([length], dtype=torch.int32),
                    num_contexts=1,
                    kv_layout="HND",
                )
                metadata.prepare()
                yield metadata
            finally:
                torch.cuda.synchronize()
                cache_manager.shutdown()

        input_ids = torch.arange(32, device="cuda", dtype=torch.long)
        position_ids = torch.arange(32, device="cuda").unsqueeze(0)
        embeddings = model.model.embed_tokens(input_ids).clone()
        with fresh_metadata(32) as metadata:
            logits = model(
                attn_metadata=metadata,
                input_ids=input_ids.clone(),
                position_ids=position_ids.clone(),
                return_context_logits=True,
            ).clone()
        assert logits.shape == (32, config.vocab_size)
        assert logits.is_cuda and logits.is_floating_point()
        assert torch.isfinite(logits).all()

        with fresh_metadata(32) as metadata:
            repeated_logits = model(
                attn_metadata=metadata,
                input_ids=input_ids.clone(),
                position_ids=position_ids.clone(),
                return_context_logits=True,
            ).clone()
        torch.testing.assert_close(
            logits,
            repeated_logits,
            rtol=2e-2,
            atol=2e-3,
        )
        with fresh_metadata(32) as metadata:
            embedded_logits = model(
                attn_metadata=metadata,
                inputs_embeds=embeddings.clone(),
                position_ids=position_ids.clone(),
                return_context_logits=True,
            ).clone()
        torch.testing.assert_close(
            logits,
            embedded_logits,
            rtol=2e-2,
            atol=2e-3,
        )
        with fresh_metadata(24) as metadata:
            prefix_logits = model(
                attn_metadata=metadata,
                input_ids=input_ids[:24].clone(),
                position_ids=position_ids[:, :24].clone(),
                return_context_logits=True,
            ).clone()
        torch.testing.assert_close(
            logits[:24],
            prefix_logits,
            rtol=2e-2,
            atol=2e-3,
        )

        changed_ids = input_ids.clone()
        changed_ids[0] = 33
        with fresh_metadata(32) as metadata:
            changed_logits = model(
                attn_metadata=metadata,
                input_ids=changed_ids.clone(),
                position_ids=position_ids.clone(),
                return_context_logits=True,
            ).clone()
        assert torch.isfinite(changed_logits).all()
        assert (logits[-1] - changed_logits[-1]).abs().max() > 1e-4

        normalized = embeddings.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        normalized = normalized.to(embeddings.dtype) * model.model.norm.weight
        bypass_logits = torch.nn.functional.linear(normalized, model.lm_head.weight)
        assert not torch.allclose(logits, bypass_logits.float(), rtol=2e-2, atol=2e-3)
