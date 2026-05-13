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
"""Unit tests for PyTorch encoder-decoder model foundations.

These tests intentionally use the VANILLA attention backend. TRTLLM backend,
dual-pool KV cache, scheduler, and public LLM API coverage live in later PRs
in the encoder-decoder stack.
"""

import os
import unittest
from copy import deepcopy
from pathlib import Path

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


def _make_vanilla_metadata(seq_lens):
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
        device = torch.device("cuda")
        dtype = torch.bfloat16
        hidden_size = 64
        num_heads = 8
        num_tokens_decoder = 4
        num_tokens_encoder = 8

        t5_cfg = deepcopy(SMALL_T5_CONFIG)
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
        layer = T5EncoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_tokens = 6
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens])

        output = layer(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_t5_decoder_layer_forward(self):
        layer = T5DecoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_dec = 4
        num_enc = 8
        decoder_hs = torch.randn(
            num_dec, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        encoder_hs = torch.randn(
            num_enc, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_dec])

        output = layer(
            position_ids=torch.arange(num_dec, device=self.device),
            hidden_states=decoder_hs,
            attn_metadata=metadata,
            encoder_hidden_states=encoder_hs,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (num_dec, self.hf_config.d_model))

    def test_t5_encoder_stack_forward(self):
        encoder = T5Encoder(self.model_config).to(self.device)
        num_tokens = 10
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens])

        output = encoder(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_t5_model_forward(self):
        model = T5Model(self.model_config).to(self.device)
        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, self.hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, self.hf_config.vocab_size, (dec_len,), device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len])
        dec_metadata = _make_vanilla_metadata([dec_len])

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
        layer = BartEncoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_tokens = 6
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens])

        output = layer(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_bart_decoder_layer_forward(self):
        layer = BartDecoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_dec = 4
        num_enc = 8
        decoder_hs = torch.randn(
            num_dec, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        encoder_hs = torch.randn(
            num_enc, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_dec])

        output = layer(
            position_ids=torch.arange(num_dec, device=self.device),
            hidden_states=decoder_hs,
            attn_metadata=metadata,
            encoder_hidden_states=encoder_hs,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (num_dec, self.hf_config.d_model))

    def test_bart_model_forward(self):
        model = BartModel(self.model_config).to(self.device)
        self.assertEqual(model.position_id_offset, 2)
        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, self.hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, self.hf_config.vocab_size, (dec_len,), device=self.device)
        offset = 2
        enc_positions = torch.arange(offset, offset + enc_len, device=self.device)
        dec_positions = torch.arange(offset, offset + dec_len, device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len])
        dec_metadata = _make_vanilla_metadata([dec_len])

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
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("T5ForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_bart_registered(self):
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("BartForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_mbart_registered(self):
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("MBartForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_model_config_enc_dec_flag(self):
        t5_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        mc = ModelConfig(pretrained_config=t5_config)
        self.assertTrue(mc.is_encoder_decoder)

        bart_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        mc = ModelConfig(pretrained_config=bart_config)
        self.assertTrue(mc.is_encoder_decoder)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestT5WeightLoading(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_t5_load_weights_and_encoder_parity(self):
        import transformers

        hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        hf_model = transformers.T5ForConditionalGeneration(hf_config).to(self.device).to(self.dtype)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration as TllmT5

        tllm_model = TllmT5(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)
        tllm_model.eval()

        enc_len = 8
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(input_ids=encoder_ids).last_hidden_state.squeeze(0)

        enc_metadata = _make_vanilla_metadata([enc_len])
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0))

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
            )

        max_diff = (hf_enc_out.to(self.dtype) - tllm_enc_out.to(self.dtype)).abs().max().item()
        self.assertLess(max_diff, 1e-3)

    def test_t5_load_weights_runs_forward(self):
        import transformers

        hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        hf_model = transformers.T5ForConditionalGeneration(hf_config)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration as TllmT5

        tllm_model = TllmT5(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)
        tllm_model.eval()

        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, hf_config.vocab_size, (dec_len,), device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len])
        dec_metadata = _make_vanilla_metadata([dec_len])

        with torch.inference_mode():
            tllm_out = tllm_model(
                attn_metadata=dec_metadata,
                input_ids=decoder_ids,
                encoder_input_ids=encoder_ids,
                encoder_attn_metadata=enc_metadata,
                skip_cross_kv_projection=False,
            )

        self.assertEqual(tllm_out.shape[-1], hf_config.vocab_size)
        self.assertTrue(torch.isfinite(tllm_out).all())

    def test_t5_for_conditional_generation_load_weights(self):
        import transformers

        hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        hf_model = transformers.T5ForConditionalGeneration(hf_config)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration

        tllm_model = T5ForConditionalGeneration(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBartWeightLoading(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_bart_load_weights_and_encoder_parity(self):
        import transformers

        hf_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        hf_model = transformers.BartModel(hf_config).to(self.device).to(self.dtype)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_bart import (
            BartForConditionalGeneration as TllmBart,
        )
        from tensorrt_llm._torch.models.modeling_bart import _convert_hf_bart_weights

        tllm_model = TllmBart(model_config).to(self.device)
        tllm_weights = _convert_hf_bart_weights(hf_weights, hf_config)
        loaded_count = 0
        for name, module in tllm_model.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            if name not in tllm_weights:
                continue
            weights = tllm_weights[name]
            if hasattr(module, "load_weights"):
                module.load_weights(weights=weights)
            else:
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name in weights[0]:
                        param.data.copy_(weights[0][param_name][:])
            loaded_count += 1

        self.assertGreater(loaded_count, 0)
        tllm_model.eval()

        enc_len = 8
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        embed_scale = hf_config.d_model**0.5
        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(
                inputs_embeds=hf_model.shared(encoder_ids) * embed_scale,
            ).last_hidden_state.squeeze(0)

        offset = 2
        enc_positions = torch.arange(offset, offset + enc_len, device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len])
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0)) * embed_scale

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
                position_ids=enc_positions,
            )

        max_diff = (hf_enc_out.to(self.dtype) - tllm_enc_out.to(self.dtype)).abs().max().item()
        self.assertLess(max_diff, 1e-4)

    def test_bart_for_conditional_generation_load_weights(self):
        import transformers

        hf_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        hf_model = transformers.BartForConditionalGeneration(hf_config)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_bart import BartForConditionalGeneration

        tllm_model = BartForConditionalGeneration(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)


def _get_llm_models_root():
    root = Path("/home/scratch.trt_llm_data/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    return root if root.exists() else None


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestT5SmallRealWeights(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        models_root = _get_llm_models_root()
        if models_root is None:
            self.skipTest("LLM_MODELS_ROOT not found")
        self.model_path = str(models_root / "t5-small")
        if not os.path.isdir(self.model_path):
            self.skipTest(f"t5-small not found at {self.model_path}")

    def test_t5_small_encoder_parity(self):
        import transformers

        hf_model = (
            transformers.T5ForConditionalGeneration.from_pretrained(self.model_path)
            .to(self.device)
            .to(self.dtype)
        )
        hf_model.eval()
        hf_config = hf_model.config
        hf_weights = hf_model.state_dict()

        hf_config.torch_dtype = self.dtype
        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration as TllmT5

        tllm_model = TllmT5(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)
        tllm_model.eval()

        for name, param in tllm_model.named_parameters():
            self.assertEqual(param.dtype, self.dtype, name)

        enc_len = 16
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(input_ids=encoder_ids).last_hidden_state.squeeze(0)

        enc_metadata = _make_vanilla_metadata([enc_len])
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0))

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
            )

        max_diff = (hf_enc_out - tllm_enc_out).abs().max().item()
        self.assertLess(max_diff, 0.05)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBartLargeCNNRealWeights(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        models_root = _get_llm_models_root()
        if models_root is None:
            self.skipTest("LLM_MODELS_ROOT not found")
        self.model_path = str(models_root / "bart-large-cnn")
        if not os.path.isdir(self.model_path):
            self.skipTest(f"bart-large-cnn not found at {self.model_path}")

    def test_bart_large_cnn_encoder_parity(self):
        import transformers

        hf_model = (
            transformers.BartModel.from_pretrained(self.model_path).to(self.device).to(self.dtype)
        )
        hf_model.eval()
        hf_config = hf_model.config
        hf_weights = hf_model.state_dict()

        hf_config.torch_dtype = self.dtype
        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_bart import (
            BartForConditionalGeneration as TllmBart,
        )
        from tensorrt_llm._torch.models.modeling_bart import _convert_hf_bart_weights

        tllm_model = TllmBart(model_config).to(self.device)
        tllm_weights = _convert_hf_bart_weights(hf_weights, hf_config, dtype=self.dtype)
        for name, module in tllm_model.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            if name not in tllm_weights:
                continue
            weights = tllm_weights[name]
            if hasattr(module, "load_weights"):
                module.load_weights(weights=weights)
            else:
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name in weights[0]:
                        param.data.copy_(weights[0][param_name][:])
        tllm_model.eval()

        for name, param in tllm_model.named_parameters():
            self.assertEqual(param.dtype, self.dtype, name)

        enc_len = 16
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        embed_scale = hf_config.d_model**0.5
        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(
                inputs_embeds=hf_model.shared(encoder_ids) * embed_scale,
            ).last_hidden_state.squeeze(0)

        offset = 2
        enc_positions = torch.arange(offset, offset + enc_len, device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len])
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0)) * embed_scale

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
                position_ids=enc_positions,
            )

        max_diff = (hf_enc_out - tllm_enc_out).abs().max().item()
        self.assertLess(max_diff, 0.1)


if __name__ == "__main__":
    unittest.main()
