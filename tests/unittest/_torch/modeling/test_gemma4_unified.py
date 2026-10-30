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
from transformers import Gemma4ForCausalLM as HFGemma4
from transformers import Gemma4TextConfig

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.configs import Gemma4UnifiedConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import Gemma4HfWeightMapper
from tensorrt_llm._torch.models.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class


@torch.no_grad()
def test_unified_text_forward_matches_hf(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
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
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "use_double_wide_mlp": False,
        "final_logit_softcapping": 30.0,
        "use_bidirectional_attention": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000.0,
            },
        },
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": True,
    }
    config = Gemma4UnifiedConfig(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        text_config=text_config,
        vision_config=None,
        audio_config=None,
        image_token_id=60,
        audio_token_id=None,
        video_token_id=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    model_class = get_registered_model_class("Gemma4UnifiedForConditionalGeneration")
    assert model_class is Gemma4UnifiedForConditionalGeneration
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        reference = HFGemma4(Gemma4TextConfig(**text_config)).to(torch.bfloat16).cuda().eval()
        model = (
            model_class(ModelConfig(pretrained_config=config, attn_backend="TRTLLM")).cuda().eval()
        )

    mapper = Gemma4HfWeightMapper()
    mapper.init_model_and_config(model.llm, model.llm.model_config)
    model.llm.load_weights(reference.state_dict(), mapper)

    input_ids = torch.tensor([5, 9, 13], dtype=torch.int32, device="cuda")
    position_ids = torch.arange(3, dtype=torch.int32, device="cuda").unsqueeze(0)
    metadata = TrtllmAttentionMetadata(
        seq_lens=torch.tensor([3], dtype=torch.int),
        num_contexts=1,
        max_num_requests=1,
        max_num_tokens=3,
        request_ids=[0],
    )
    metadata.max_seq_len = 128
    expected = (
        reference.forward(
            input_ids=input_ids.unsqueeze(0),
            position_ids=position_ids,
            use_cache=False,
        )
        .logits.squeeze(0)
        .float()
    )
    metadata.prepare()
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attn_metadata=metadata,
        return_context_logits=True,
    )
    assert logits.shape == (3, 64)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, expected, atol=0.05, rtol=0.05)
    inputs_embeds = model.llm.model.embed_tokens(input_ids)
    metadata.prepare()
    embedded_logits = model(
        input_ids=input_ids + 7,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attn_metadata=metadata,
        return_context_logits=True,
    )
    torch.testing.assert_close(embedded_logits, expected, atol=0.05, rtol=0.05)
