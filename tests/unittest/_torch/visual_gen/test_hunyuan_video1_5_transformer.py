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

    # Normalized timestep in [0, 1]; 0.5 is exact in bf16 so it round-trips cleanly.
    timestep = torch.tensor([0.5], device=device, dtype=dtype)

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
        # Full numerical correctness is tested in TestHunyuanVideo15HuggingFaceComparison.
        self.assertEqual(output.shape, hidden_states.shape)


def test_deco_recipe_builds_deco_layers(require_deco_kernels):
    """Every attention layer of a model under a DeCo-Attention recipe is a DeCo layer."""
    from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import DeCoAttention
    from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state
    from tensorrt_llm._torch.visual_gen.models.hunyuan_video1_5 import (
        transformer_hunyuan_video1_5 as hunyuan,
    )
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    require_deco_kernels()
    model_config = _create_model_config(CONFIG | {"num_layers": 1}, backend="CUTEDSL")
    model_config.attention = AttentionConfig(
        backend="CUTEDSL",
        quant_attention_config=QuantAttentionConfig(
            qk_dtype="nvfp4", v_dtype="fp8", deco_attention=True
        ),
    )
    model_config.attention_metadata_state = create_attention_metadata_state()
    model = hunyuan.HunyuanVideo15Transformer3DModel(model_config)

    layers = [m.attn for m in model.modules() if isinstance(m, hunyuan.Attention)]
    assert layers and all(type(layer) is DeCoAttention for layer in layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_joint_blocks_share_the_token_tags_and_the_padding_mask(monkeypatch):
    """The joint blocks of a forward hand their backend one tags tensor and one padding mask."""
    from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
    from tensorrt_llm._torch.visual_gen.models.hunyuan_video1_5 import (
        transformer_hunyuan_video1_5 as hunyuan,
    )

    config = CONFIG | {"num_layers": 2}
    dtype = torch.bfloat16
    model = hunyuan.HunyuanVideo15Transformer3DModel(_create_model_config(config))
    model = model.to("cuda", dtype=dtype).eval()

    inputs = [t.repeat(2, *[1] * (t.dim() - 1)) for t in _make_inputs(config, dtype, "cuda")]
    hidden_states, text, image_embeds, text_mask, text_2, text_mask_2, timestep = inputs
    text_mask[1, 2:] = 0  # a CFG batch: the second sample has a shorter text
    text_mask_2[1, 4:] = 0
    num_text = (text_mask.sum(dim=1) + text_mask_2.sum(dim=1)).long()
    num_image = image_embeds.shape[1]

    calls: list[dict] = []
    backend_forward = VanillaAttention.forward

    def record(self, q, k=None, v=None, **kwargs):
        if kwargs.get("token_tags") is not None:  # the token refiner passes none
            calls.append(kwargs)
        return backend_forward(self, q, k, v, **kwargs)

    monkeypatch.setattr(VanillaAttention, "forward", record)

    def forward(image_embeds):
        """Run the model and return the tags and the mask that its blocks passed."""
        calls.clear()
        with torch.no_grad():
            model(
                hidden_states=hidden_states,
                encoder_hidden_states=text,
                timestep=timestep,
                encoder_attention_mask=text_mask,
                encoder_hidden_states_2=text_2,
                image_embeds=image_embeds,
                encoder_attention_mask_2=text_mask_2,
            )
        tags, mask = calls[0]["token_tags"], calls[0]["key_padding_mask"]
        assert len(calls) == len(model.transformer_blocks)
        assert all(c["token_tags"] is tags and c["key_padding_mask"] is mask for c in calls)
        assert all(c.get("valid_seq_len") is None for c in calls)
        return tags, mask

    # Text to video: tag 0 on the video rows, 1 after them; the image rows hold no token.
    tags, mask = forward(torch.zeros_like(image_embeds))
    num_video = tags.numel() - text.shape[1] - text_2.shape[1] - num_image
    positions = torch.arange(tags.numel(), device="cuda")
    assert torch.equal(tags, (positions >= num_video).to(tags.dtype))
    assert torch.equal(mask, positions < (num_video + num_text)[:, None])
    assert forward(torch.zeros_like(image_embeds))[0] is tags

    # With image embeddings the image rows follow the video rows and share their tag.
    tags, mask = forward(image_embeds)
    assert torch.equal(tags, (positions >= num_video + num_image).to(tags.dtype))
    assert torch.equal(mask, positions < (num_video + num_image + num_text)[:, None])


class TestHunyuanVideo15HuggingFaceComparison(unittest.TestCase):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_hunyuan_video1_5_allclose_to_hf(self):
        from diffusers import HunyuanVideo15Transformer3DModel as HFHunyuanVideo15Transformer3DModel

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
            # HF consumes the raw timestep; TRT-LLM consumes the normalized one.
            hf_output = hf_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep * NUM_TRAIN_TIMESTEPS,
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
