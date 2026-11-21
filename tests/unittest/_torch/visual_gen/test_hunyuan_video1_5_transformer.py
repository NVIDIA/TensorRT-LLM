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


class TestHunyuanVideo15DeCoAttention(unittest.TestCase):
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_joint_blocks_pass_one_mask_and_the_token_tags(self):
        """Under a DeCo-Attention recipe the joint attention of a CFG batch with two text
        lengths reaches the kernels with the token count of every sample and with basis rows
        drawn from that sample's video and text tokens; the blocks of a forward share one
        mask tensor."""
        from unittest import mock

        from tensorrt_llm._torch.visual_gen.attention.backends import DeCoAttention
        from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import deco as deco_backend
        from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention
        from tensorrt_llm._torch.visual_gen.models.hunyuan_video1_5 import (
            transformer_hunyuan_video1_5 as hunyuan,
        )
        from tensorrt_llm.visual_gen.args import QuantAttentionConfig

        try:
            deco_attention.pv_formats()
        except NotImplementedError:
            pytest.skip("DeCo-Attention kernels need an SM100, SM103, SM107 or SM120 GPU.")
        config = deepcopy(CONFIG)
        config["num_layers"] = 2
        model_config = _create_model_config(config, backend="CUTEDSL")
        model_config.attention = AttentionConfig(
            backend="CUTEDSL",
            quant_attention_config=QuantAttentionConfig(
                qk_dtype="nvfp4", v_dtype="fp8", deco_attention=True
            ),
        )
        model_config.attention_metadata_state = create_attention_metadata_state()
        dtype = torch.bfloat16
        model = hunyuan.HunyuanVideo15Transformer3DModel(model_config)
        model = model.to("cuda", dtype=dtype).eval()

        inputs = [t.repeat(2, *[1] * (t.dim() - 1)) for t in _make_inputs(config, dtype, "cuda")]
        hidden_states, text, image_embeds, text_mask, text_2, text_mask_2, timestep = inputs
        text_mask[1, 2:] = 0  # the second sample has a shorter text
        text_mask_2[1, 4:] = 0
        num_text = [text_mask[b].sum() + text_mask_2[b].sum() for b in range(2)]

        plans, launches = [], []
        real_plan = DeCoAttention._plan

        def plan(self, key_padding_mask, valid_seq_len, token_tags, *shape):
            if token_tags is not None:
                plans.append((key_padding_mask, token_tags))
            return real_plan(self, key_padding_mask, valid_seq_len, token_tags, *shape)

        def launch(self, q, k, v, rows=None, valid_seq_lens=None, order=None):
            if rows is not None:
                launches.append((q.shape[1], rows, valid_seq_lens, order))
            return torch.zeros_like(q)

        with (
            mock.patch.object(deco_backend, "MIN_SEQ_LEN", 1),
            mock.patch.object(DeCoAttention, "_plan", plan),
            mock.patch.object(DeCoAttention, "_launch", launch),
            torch.no_grad(),
        ):
            model(
                hidden_states=hidden_states,
                encoder_hidden_states=text,
                timestep=timestep,
                encoder_attention_mask=text_mask,
                encoder_hidden_states_2=text_2,
                image_embeds=torch.zeros_like(image_embeds),
                encoder_attention_mask_2=text_mask_2,
            )

        self.assertEqual(len(launches), len(model.transformer_blocks))
        self.assertTrue(all(mask is plans[0][0] and tags is plans[0][1] for mask, tags in plans))
        seq_len, rows, valid_seq_lens, order = launches[0]
        num_video = seq_len - text.shape[1] - text_2.shape[1] - image_embeds.shape[1]
        self.assertIsNone(order)
        self.assertEqual(valid_seq_lens.tolist(), [num_video + int(n) for n in num_text])
        tags = plans[0][1]
        self.assertTrue((tags[:num_video] == 0).all() and (tags[num_video:] > 0).all())
        # Every sample draws from its own tokens, video and text.
        self.assertEqual(rows.shape[0], 2)
        self.assertTrue((rows < valid_seq_lens[:, None]).all())
        self.assertTrue(((rows >= num_video).sum(dim=1) > 0).all())
        self.assertTrue(((rows < num_video).sum(dim=1) > 0).all())

        # With image embeddings the image rows follow the video rows and share their tag.
        plans.clear()
        with (
            mock.patch.object(deco_backend, "MIN_SEQ_LEN", 1),
            mock.patch.object(DeCoAttention, "_plan", plan),
            mock.patch.object(DeCoAttention, "_launch", launch),
            torch.no_grad(),
        ):
            model(
                hidden_states=hidden_states,
                encoder_hidden_states=text,
                timestep=timestep,
                encoder_attention_mask=text_mask,
                encoder_hidden_states_2=text_2,
                image_embeds=image_embeds,
                encoder_attention_mask_2=text_mask_2,
            )
        mask, tags = plans[0]
        num_image = image_embeds.shape[1]
        self.assertTrue((tags[: num_video + num_image] == 0).all())
        self.assertTrue((tags[num_video + num_image :] > 0).all())
        self.assertTrue(mask[:, : num_video + num_image].all())


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
