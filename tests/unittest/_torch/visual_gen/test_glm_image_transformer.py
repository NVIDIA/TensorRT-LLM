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
    "text_embed_dim": 1472,
    "in_channels": 16,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
}

NUM_TRAIN_TIMESTEPS = 1000


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
    height, width = 32, 32

    sequence_length = 6
    text_embed_dim = config["text_embed_dim"]
    num_channels = config["in_channels"]

    generator = torch.Generator(device=device).manual_seed(42)

    hidden_states = torch.randn(
        batch_size,
        num_channels,
        height,
        width,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    encoder_hidden_states = torch.randn(
        batch_size, sequence_length, text_embed_dim, device=device, dtype=dtype, generator=generator
    )

    prior_token_id = torch.randint(0, 64, size=(batch_size,), generator=generator, device=device)
    prior_token_drop = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Normalized timestep in [0, 1]; 0.5 is exact in bf16 so it round-trips cleanly.
    timestep = torch.tensor([0.5] * batch_size, device=device, dtype=dtype)
    target_size = torch.tensor([[height, width]] * batch_size, dtype=torch.float32, device=device)
    crop_coords = torch.tensor([[0, 0]] * batch_size, dtype=torch.float32, device=device)

    return (
        hidden_states,
        encoder_hidden_states,
        prior_token_id,
        prior_token_drop,
        target_size,
        crop_coords,
        timestep,
    )


class TestGlmImageTransformer(unittest.TestCase):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_glm_image_structure(self):
        from tensorrt_llm._torch.visual_gen.models.glm_image import GlmImageTransformer2DModel

        config = deepcopy(CONFIG)
        config["num_layers"] = 1
        model_config = _create_model_config(config)
        model = GlmImageTransformer2DModel(model_config)

        # Check components are present in model
        components = [
            "rope",
            "image_projector",
            "glyph_projector",
            "prior_token_embedding",
            "prior_projector",
            "time_condition_embed",
            "transformer_blocks",
            "norm_out",
            "proj_out",
        ]

        for component in components:
            self.assertTrue(hasattr(model, component))

        self.assertEqual(len(model.transformer_blocks), 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_glm_image_forward_sanity(self):
        from tensorrt_llm._torch.visual_gen.models.glm_image import GlmImageTransformer2DModel

        config = deepcopy(CONFIG)
        config["num_layers"] = 1
        model_config = _create_model_config(config)
        dtype = torch.bfloat16

        model = GlmImageTransformer2DModel(model_config).to(self.DEVICE, dtype=dtype).eval()

        (
            hidden_states,
            encoder_hidden_states,
            prior_token_id,
            prior_token_drop,
            target_size,
            crop_coords,
            timestep,
        ) = _make_inputs(config, dtype, self.DEVICE)

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                prior_token_id=prior_token_id,
                prior_token_drop=prior_token_drop,
                target_size=target_size,
                crop_coords=crop_coords,
            )

        # Model returns {"sample": tensor}
        if isinstance(output, dict):
            output = output["sample"]

        # Note: With random weights, NaN can occur. For unit tests, we only check shape.
        # Full numerical correctness is tested in TestFluxHuggingFaceComparison.
        self.assertEqual(output.shape, hidden_states.shape)


class TestGlmImageHuggingFaceComparison(unittest.TestCase):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_glm_image_allclose_to_hf(self):
        try:
            from diffusers import GlmImageTransformer2DModel as HFGlmImageTransformer2DModel
        except ImportError:
            self.skipTest("diffusers not installed")

        from tensorrt_llm._torch.visual_gen.models.glm_image import GlmImageTransformer2DModel

        torch.manual_seed(42)

        # Create TRT-LLM Model
        config = deepcopy(CONFIG)
        config["num_layers"] = 1
        model_config = _create_model_config(config)
        dtype = torch.bfloat16
        trtllm_model = GlmImageTransformer2DModel(model_config).to(self.DEVICE, dtype=dtype).eval()

        hf_model = (
            HFGlmImageTransformer2DModel(
                in_channels=config["in_channels"],
                num_layers=config["num_layers"],
                attention_head_dim=config["attention_head_dim"],
                num_attention_heads=config["num_attention_heads"],
            )
            .to(self.DEVICE, dtype=dtype)
            .eval()
        )

        # Copy weights from HF to TRT-LLM
        hf_state_dict = hf_model.state_dict()
        for name, _ in hf_state_dict.items():
            print(f"{name}")

        print("\n")
        for name, _ in trtllm_model.named_modules():
            print(f"{name}")

        trtllm_model.load_weights(hf_state_dict)

        # Create inputs
        (
            hidden_states,
            encoder_hidden_states,
            prior_token_id,
            prior_token_drop,
            target_size,
            crop_coords,
            timestep,
        ) = _make_inputs(config, dtype, self.DEVICE)

        with torch.no_grad():
            # HF consumes the raw timestep; TRT-LLM consumes the normalized one.
            hf_output = hf_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep * NUM_TRAIN_TIMESTEPS,
                prior_token_id=prior_token_id,
                prior_token_drop=prior_token_drop,
                target_size=target_size,
                crop_coords=crop_coords,
                return_dict=False,
            )[0]

            trtllm_output = trtllm_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                prior_token_id=prior_token_id,
                prior_token_drop=prior_token_drop,
                target_size=target_size,
                crop_coords=crop_coords,
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

        print("\n[GlmImage HF Comparison]")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max diff: {max_diff:.6f}")

        self.assertGreater(cos_sim, 0.99, f"Cosine similarity too low: {cos_sim}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
