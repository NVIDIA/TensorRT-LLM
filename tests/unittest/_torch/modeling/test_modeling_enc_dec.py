# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the PyTorch-flow encoder-decoder modules (step 2).

Tests that modules can be constructed and run forward passes on dummy tensors.
These tests use the VANILLA attention backend (no TRTLLM C++ dependency) and
run on a single GPU.
"""

import unittest
from copy import deepcopy

import torch
from transformers import BartConfig, T5Config

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bart import BartDecoderLayer, BartEncoderLayer, BartModel
from tensorrt_llm._torch.models.modeling_t5 import (
    T5DecoderLayer,
    T5Encoder,
    T5EncoderLayer,
    T5Model,
)
from tensorrt_llm._torch.modules.cross_attention import CrossAttention


def _make_vanilla_metadata(seq_lens, device="cuda"):
    """Create a minimal VanillaAttentionMetadata for testing."""
    metadata_cls = get_attention_backend("VANILLA").Metadata
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    total_tokens = sum(seq_lens)
    num_requests = len(seq_lens)
    metadata = metadata_cls(
        max_num_requests=num_requests,
        max_num_tokens=total_tokens,
        kv_cache_manager=None,
        request_ids=list(range(num_requests)),
        prompt_lens=seq_lens,
        seq_lens=seq_lens_tensor,
        num_contexts=num_requests,
    )
    metadata.max_seq_len = max(seq_lens)
    metadata.prepare()
    return metadata


# Small T5 config for fast testing
SMALL_T5_CONFIG = {
    "architectures": ["T5ForConditionalGeneration"],
    "d_model": 64,
    "d_kv": 8,
    "d_ff": 128,
    "num_heads": 8,
    "num_layers": 2,
    "num_decoder_layers": 2,
    "vocab_size": 100,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "relu",
    "is_encoder_decoder": True,
    "is_gated_act": False,
    "model_type": "t5",
    "decoder_start_token_id": 0,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "torch_dtype": "bfloat16",
}

# Small BART config for fast testing
SMALL_BART_CONFIG = {
    "architectures": ["BartForConditionalGeneration"],
    "d_model": 64,
    "encoder_ffn_dim": 128,
    "decoder_ffn_dim": 128,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "encoder_attention_heads": 8,
    "decoder_attention_heads": 8,
    "vocab_size": 100,
    "max_position_embeddings": 128,
    "activation_function": "gelu",
    "is_encoder_decoder": True,
    "model_type": "bart",
    "decoder_start_token_id": 2,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "bos_token_id": 0,
    "torch_dtype": "bfloat16",
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCrossAttention(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)

    def test_cross_attention_forward(self):
        """CrossAttention projects K/V from encoder and outputs correct shape."""
        device = torch.device("cuda")
        dtype = torch.bfloat16
        hidden_size = 64
        num_heads = 8
        num_tokens_decoder = 4
        num_tokens_encoder = 8

        t5_cfg = deepcopy(SMALL_T5_CONFIG)
        t5_cfg["torch_dtype"] = "bfloat16"
        config = ModelConfig(
            pretrained_config=T5Config.from_dict(t5_cfg),
            attn_backend="VANILLA",
        )
        cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            encoder_hidden_size=hidden_size,
            bias=False,
            layer_idx=0,
            dtype=dtype,
            config=config,
        ).to(device)

        decoder_hs = torch.randn(num_tokens_decoder, hidden_size, device=device, dtype=dtype)
        encoder_hs = torch.randn(num_tokens_encoder, hidden_size, device=device, dtype=dtype)
        metadata = _make_vanilla_metadata([num_tokens_decoder])

        with torch.inference_mode():
            output = cross_attn(
                hidden_states=decoder_hs,
                encoder_hidden_states=encoder_hs,
                attn_metadata=metadata,
                skip_cross_kv_projection=False,
            )
        self.assertEqual(output.shape, (num_tokens_decoder, hidden_size))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestT5Modules(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        self.model_config = ModelConfig(
            pretrained_config=self.hf_config,
            attn_backend="VANILLA",
        )

    def test_t5_encoder_layer_forward(self):
        """Single T5 encoder layer produces correct output shape."""
        layer = T5EncoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_tokens = 6
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens], self.device)

        output = layer(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_t5_decoder_layer_forward(self):
        """Single T5 decoder layer with cross-attention produces correct shape."""
        layer = T5DecoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_dec = 4
        num_enc = 8
        decoder_hs = torch.randn(
            num_dec, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        encoder_hs = torch.randn(
            num_enc, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_dec], self.device)

        output = layer(
            position_ids=torch.arange(num_dec, device=self.device),
            hidden_states=decoder_hs,
            attn_metadata=metadata,
            encoder_hidden_states=encoder_hs,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (num_dec, self.hf_config.d_model))

    def test_t5_encoder_stack_forward(self):
        """T5 encoder stack runs all layers and applies final norm."""
        encoder = T5Encoder(self.model_config).to(self.device)
        num_tokens = 10
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens], self.device)

        output = encoder(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_t5_model_forward(self):
        """T5Model encoder-decoder body runs end-to-end with encoder_input_ids."""
        model = T5Model(self.model_config).to(self.device)
        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, self.hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, self.hf_config.vocab_size, (dec_len,), device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        dec_metadata = _make_vanilla_metadata([dec_len], self.device)

        output = model(
            attn_metadata=dec_metadata,
            input_ids=decoder_ids,
            encoder_input_ids=encoder_ids,
            encoder_attn_metadata=enc_metadata,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (dec_len, self.hf_config.d_model))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBartModules(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.hf_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        self.model_config = ModelConfig(
            pretrained_config=self.hf_config,
            attn_backend="VANILLA",
        )

    def test_bart_encoder_layer_forward(self):
        """Single BART encoder layer produces correct output shape."""
        layer = BartEncoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_tokens = 6
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens], self.device)

        output = layer(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_bart_decoder_layer_forward(self):
        """Single BART decoder layer with cross-attention produces correct shape."""
        layer = BartDecoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_dec = 4
        num_enc = 8
        decoder_hs = torch.randn(
            num_dec, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        encoder_hs = torch.randn(
            num_enc, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_dec], self.device)

        output = layer(
            position_ids=torch.arange(num_dec, device=self.device),
            hidden_states=decoder_hs,
            attn_metadata=metadata,
            encoder_hidden_states=encoder_hs,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (num_dec, self.hf_config.d_model))

    def test_bart_model_forward(self):
        """BartModel encoder-decoder body runs end-to-end."""
        model = BartModel(self.model_config).to(self.device)
        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, self.hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, self.hf_config.vocab_size, (dec_len,), device=self.device)
        # BART position IDs start at offset 2 (padding_idx + 1) per HF convention.
        # The runtime (Step 9) will handle this; here we use the correct offset
        # so the test exercises valid embedding indices.
        offset = 2
        enc_positions = torch.arange(offset, offset + enc_len, device=self.device)
        dec_positions = torch.arange(offset, offset + dec_len, device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        dec_metadata = _make_vanilla_metadata([dec_len], self.device)

        output = model(
            attn_metadata=dec_metadata,
            input_ids=decoder_ids,
            encoder_input_ids=encoder_ids,
            encoder_position_ids=enc_positions,
            position_ids=dec_positions,
            encoder_attn_metadata=enc_metadata,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (dec_len, self.hf_config.d_model))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestModelRegistration(unittest.TestCase):
    def test_t5_registered(self):
        """T5ForConditionalGeneration is discoverable via MODEL_CLASS_MAPPING."""
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("T5ForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_bart_registered(self):
        """BartForConditionalGeneration is discoverable."""
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("BartForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_mbart_registered(self):
        """MBartForConditionalGeneration is discoverable."""
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("MBartForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_model_config_enc_dec_flag(self):
        """ModelConfig.is_encoder_decoder is True for T5/BART configs."""
        t5_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        mc = ModelConfig(pretrained_config=t5_config)
        self.assertTrue(mc.is_encoder_decoder)

        bart_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        mc = ModelConfig(pretrained_config=bart_config)
        self.assertTrue(mc.is_encoder_decoder)


if __name__ == "__main__":
    unittest.main()
