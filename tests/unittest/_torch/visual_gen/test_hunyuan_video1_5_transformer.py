# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen import DiffusionModelConfig
from tensorrt_llm.llmapi import QuantConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.visual_gen import AttentionConfig

CONFIG = {
    "in_channels": 32,
    "text_embed_dim": 3584,
    "text_embed_2_dim": 1472,
    "hidden_size": 2048,
    "image_embed_dim": 1152,
    "torch_dtype": "bfloat16",
}


def _create_model_config(config_dict: dict, backend: str = "VANILLA") -> DiffusionModelConfig:
    """Create DiffusionModelConfig from config dict."""
    pretrained_config = SimpleNamespace(**config_dict)
    return DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )


def _make_inputs(config, dtype, device):
    batch_size = 1
    num_frames = 1
    height, width = 32, 32

    sequence_length = 6
    text_embed_dim = config["text_embed_dim"]
    text_embed_2_dim = config["text_embed_2_dim"]
    num_channels = config["in_channels"]
    image_embed_dim = config["image_embed_dim"]

    generator = torch.Generator(device=device).manual_seed(42)

    hidden_states = torch.randn(
        batch_size,
        num_channels,
        num_frames,
        height,
        width,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    encoder_hidden_states = torch.randn(
        batch_size, sequence_length, text_embed_dim, device=device, dtype=dtype, generator=generator
    )

    encoder_hidden_states_2 = torch.randn(
        batch_size,
        sequence_length,
        text_embed_2_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    image_embeds = torch.randn(
        batch_size,
        sequence_length,
        image_embed_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    encoder_attention_mask = torch.ones(batch_size, sequence_length, device=device, dtype=dtype)

    encoder_attention_mask_2 = torch.ones(batch_size, sequence_length, device=device, dtype=dtype)

    timestep = torch.tensor([1], device=device, dtype=dtype)

    return (
        hidden_states,
        encoder_hidden_states,
        image_embeds,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        timestep,
    )


class TestHunyuanVideo15Transformer(unittest.TestCase):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hunyuan_video1_5_structure(self):
        from tensorrt_llm._torch.visual_gen.models.hunyuan_video1_5.transformer_hunyuan_video1_5 import (
            HunyuanVideo15Transformer3DModel,
        )

        config = deepcopy(CONFIG)
        config["num_layers"] = 1
        model_config = _create_model_config(config)
        model = HunyuanVideo15Transformer3DModel(model_config)

        # Check components are present in model
        components = [
            "x_embedder",
            "image_embedder",
            "context_embedder",
            "context_embedder_2",
            "time_embed",
            "cond_type_embed",
            "rope",
            "transformer_blocks",
            "norm_out",
            "proj_out",
        ]

        for component in components:
            self.assertTrue(hasattr(model, component))

        self.assertEqual(len(model.transformer_blocks), 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hunyuan_video1_5_forward_sanity(self):
        from tensorrt_llm._torch.visual_gen.models.hunyuan_video1_5.transformer_hunyuan_video1_5 import (
            HunyuanVideo15Transformer3DModel,
        )

        config = deepcopy(CONFIG)
        config["num_layers"] = 1
        model_config = _create_model_config(config)
        dtype = torch.bfloat16

        model = HunyuanVideo15Transformer3DModel(model_config).to(self.DEVICE, dtype=dtype).eval()

        (
            hidden_states,
            encoder_hidden_states,
            image_embeds,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            timestep,
        ) = _make_inputs(config, dtype, self.DEVICE)

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_2=encoder_hidden_states_2,
                image_embeds=image_embeds,
                encoder_attention_mask_2=encoder_attention_mask_2,
            )

        # Model returns {"sample": tensor}
        if isinstance(output, dict):
            output = output["sample"]

        # Note: With random weights, NaN can occur. For unit tests, we only check shape.
        # Full numerical correctness is tested in TestFluxHuggingFaceComparison.
        self.assertEqual(output.shape, hidden_states.shape)


class TestHunyuanVideo15HuggingFaceComparison(unittest.TestCase):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hunyuan_video1_5_allclose_to_hf(self):
        try:
            from diffusers import (
                HunyuanVideo15Transformer3DModel as HFHunyuanVideo15Transformer3DModel,
            )
        except ImportError:
            self.skipTest("diffusers not installed")

        from tensorrt_llm._torch.visual_gen.models.hunyuan_video1_5.transformer_hunyuan_video1_5 import (
            HunyuanVideo15Transformer3DModel,
        )

        torch.manual_seed(42)

        # Create TRT-LLM Model
        config = deepcopy(CONFIG)
        config["num_layers"] = 1
        model_config = _create_model_config(config)
        dtype = torch.bfloat16
        trtllm_model = (
            HunyuanVideo15Transformer3DModel(model_config).to(self.DEVICE, dtype=dtype).eval()
        )

        hf_model = (
            HFHunyuanVideo15Transformer3DModel(
                in_channels=config["in_channels"], num_layers=config["num_layers"]
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Copy weights from HF to TRT-LLM
        hf_state_dict = hf_model.state_dict()
        trtllm_model.load_weights(hf_state_dict)

        # Create inputs
        (
            hidden_states,
            encoder_hidden_states,
            image_embeds,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            timestep,
        ) = _make_inputs(config, dtype, self.DEVICE)

        with torch.no_grad():
            hf_output = hf_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_2=encoder_hidden_states_2,
                image_embeds=image_embeds,
                encoder_attention_mask_2=encoder_attention_mask_2,
                return_dict=False,
            )[0]

            trtllm_output = trtllm_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_2=encoder_hidden_states_2,
                image_embeds=image_embeds,
                encoder_attention_mask_2=encoder_attention_mask_2,
            )

        # Model returns {"sample": tensor}
        if isinstance(trtllm_output, dict):
            trtllm_output = trtllm_output["sample"]

        # Compare outputs
        hf_output = hf_output.float()
        trtllm_output = trtllm_output.float()

        cos_sim = F.cosine_similarity(
            hf_output.flatten().unsqueeze(0), trtllm_output.flatten().unsqueeze(0)
        ).item()

        max_diff = (hf_output - trtllm_output).abs().max().item()

        print("\n[HunyuanVideo1.5 HF Comparison]")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max diff: {max_diff:.6f}")

        self.assertGreater(cos_sim, 0.99, f"Cosine similarity too low: {cos_sim}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
