# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_qwen3_for_causal_lm_forward():
    config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=32,
        rope_scaling=None,
        attention_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
    )
    model_class = get_registered_model_class("Qwen3ForCausalLM")
    assert model_class is Qwen3ForCausalLM

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        reference = HFQwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
        model = (
            model_class(
                ModelConfig(
                    pretrained_config=config,
                    attn_backend="TRTLLM",
                    max_seq_len=32,
                    max_num_tokens=8,
                )
            )
            .cuda()
            .eval()
        )
        model.load_weights(reference.state_dict())

    assert isinstance(model, Qwen3ForCausalLM)
    assert len(model.model.layers) == 2
    assert all(isinstance(layer, Qwen3DecoderLayer) for layer in model.model.layers)

    input_ids = torch.tensor([[1, 7, 11, 13], [1, 17, 19, 23]], device="cuda", dtype=torch.long)
    position_ids = torch.arange(4, device="cuda").repeat(2)
    metadata = TrtllmAttentionMetadata(
        max_num_requests=2,
        max_num_tokens=8,
        num_contexts=2,
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        request_ids=[0, 1],
        position_ids=position_ids,
    )
    metadata.max_seq_len = 32
    metadata.prepare()

    actual_logits = model(
        attn_metadata=metadata,
        input_ids=input_ids.flatten(),
        position_ids=position_ids,
        return_context_logits=True,
    )
    reference_logits = reference(input_ids=input_ids, use_cache=False).logits.reshape(8, 64)

    assert actual_logits.shape == (8, 64)
    assert actual_logits.dtype == torch.float32
    assert torch.isfinite(actual_logits).all()
    torch.testing.assert_close(actual_logits, reference_logits.float(), atol=1e-2, rtol=2e-3)
