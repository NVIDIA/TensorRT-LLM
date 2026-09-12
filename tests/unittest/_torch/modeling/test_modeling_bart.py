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

"""BART architecture parity using the native stateless attention backend."""

import pytest
import torch
from transformers import BartConfig
from transformers import BartForConditionalGeneration as HFBartForConditionalGeneration

from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bart import BartForConditionalGeneration
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class


def _context_metadata(length: int) -> VanillaAttentionMetadata:
    metadata = VanillaAttentionMetadata(
        max_num_requests=1,
        max_num_tokens=length,
        kv_cache_manager=None,
        kv_cache_params=KVCacheParams(use_cache=False),
    )
    metadata.seq_lens = torch.tensor([length], dtype=torch.int32)
    metadata.num_contexts = 1
    metadata.request_ids = [0]
    metadata.prompt_lens = [length]
    metadata.prepare()
    return metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The production BART layers require CUDA")
def test_bart_for_conditional_generation_construction_and_forward() -> None:
    torch.manual_seed(0)
    config = BartConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=32,
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        scale_embedding=False,
        tie_word_embeddings=True,
        torch_dtype=torch.float32,
    )
    # Use independent eager HF mathematics and local random weights; no download.
    config._attn_implementation = "eager"
    reference = HFBartForConditionalGeneration(config).cuda().float().eval()
    model_cls = get_registered_model_class("BartForConditionalGeneration")
    assert model_cls is BartForConditionalGeneration
    model = model_cls(ModelConfig(pretrained_config=config, attn_backend="VANILLA")).cuda().eval()
    assert isinstance(model, BartForConditionalGeneration)
    model.load_weights(
        {name: value.detach().clone() for name, value in reference.state_dict().items()}
    )

    encoder_input_ids = torch.tensor([3, 7, 11, 5, 9], dtype=torch.int32, device="cuda")
    decoder_input_ids = torch.tensor([2, 4, 6], dtype=torch.int32, device="cuda")
    encoder_metadata = _context_metadata(encoder_input_ids.numel())
    decoder_metadata = _context_metadata(decoder_input_ids.numel())
    cross_metadata = decoder_metadata.create_cross_metadata([encoder_input_ids.numel()])
    cross_metadata.prepare()
    assert cross_metadata.is_cross

    # The production entry point expects explicit learned-position indices.
    encoder_positions = (
        torch.arange(encoder_input_ids.numel(), dtype=torch.int32, device="cuda") + 2
    )
    decoder_positions = (
        torch.arange(decoder_input_ids.numel(), dtype=torch.int32, device="cuda") + 2
    )
    with torch.inference_mode():
        logits = model.forward(
            attn_metadata=decoder_metadata,
            input_ids=decoder_input_ids,
            position_ids=decoder_positions,
            encoder_input_ids=encoder_input_ids,
            encoder_position_ids=encoder_positions,
            encoder_attn_metadata=encoder_metadata,
            cross_attn_metadata=cross_metadata,
        )
        expected = reference(
            input_ids=encoder_input_ids.long().unsqueeze(0),
            decoder_input_ids=decoder_input_ids.long().unsqueeze(0),
            use_cache=False,
        ).logits[:, -1, :]

    assert logits.shape == (1, config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    # Both implementations use float32 and identical small weights. Allow
    # accumulation-order differences between SDPA and eager HF attention.
    torch.testing.assert_close(logits, expected, rtol=2e-4, atol=2e-5)
