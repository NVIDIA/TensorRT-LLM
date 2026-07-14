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
"""Tests for Kimi K2.5 multimodal model (KimiK25ForConditionalGeneration).

TestKimiK25Structure — structural tests (no forward pass, no checkpoint needed):
  - Composite config parsing (text_config + vision_config)
  - VLM model creation (LLM backbone + vision encoder + projector)
  - Decorator stack (@register_auto_model, @register_input_processor, etc.)
  - Input processor token counting for images, multi-image, and video
  - Multi-image and video input processing through the full __call__ path

TestKimiK25AutoModelRegistration — model registry verification:
  - KimiK25ForConditionalGeneration is registered in MODEL_CLASS_MAPPING

TestKimiK25E2ESmoke — E2E smoke test (requires GPU + checkpoint):
  - Text, image, and video generation through the full multimodal pipeline
  - Uses the NVFP4 checkpoint with 4x GPU

NOTE: TestKimiK25Structure does not include forward-pass tests because
DeepSeek-V3's MLA attention uses fused CUDA kernels with hard-coded tile
sizes that require production-scale dimensions (hidden_size=7168, 64 heads).
A model this large cannot be instantiated with random weights in a
unit-test context without risking C++ kernel aborts.
"""

import os
import tempfile
import unittest
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, PretrainedConfig
from utils.llm_data import llm_models_root
from utils.util import skip_pre_blackwell_unittest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_k25 import (
    KimiK25ForConditionalGeneration,
    KimiK25InputProcessor,
    KimiK25VisionModel,
    _frames_to_chunks,
)
from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm.inputs import create_input_processor

# ---------------------------------------------------------------------------
# Model path — needed for tokenizer / processor loading
# ---------------------------------------------------------------------------
_KIMI_K25_MODEL_DIR = str(os.path.join(llm_models_root(), "Kimi-K2.5-NVFP4"))

# ---------------------------------------------------------------------------
# Small composite config (matches real config.json structure)
# ---------------------------------------------------------------------------
KIMI_K25_SMALL_CONFIG = {
    "architectures": ["KimiK25ForConditionalGeneration"],
    "model_type": "kimi_k25",
    "torch_dtype": "bfloat16",
    "media_placeholder_token_id": 163605,
    "pad_token_id": 163839,
    "bos_token_id": 163584,
    "eos_token_id": 163585,
    "use_unified_vision_chunk": True,
    "video_placeholder": "<|kimi_k25_video_placeholder|>",
    "tie_word_embeddings": False,
    "text_config": {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "torch_dtype": "bfloat16",
        "hidden_size": 7168,
        "num_hidden_layers": 2,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "intermediate_size": 1024,
        "vocab_size": 163840,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        # MLA parameters (production dimensions required by kernel)
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        # MoE (minimal)
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 0,
        "tie_word_embeddings": False,
        "rope_scaling": None,
        "_name_or_path": _KIMI_K25_MODEL_DIR,
    },
    "vision_config": {
        "vt_hidden_size": 128,
        "vt_num_hidden_layers": 2,
        "vt_num_attention_heads": 4,
        "vt_intermediate_size": 512,
        "mm_hidden_size": 128,
        "text_hidden_size": 7168,
        "patch_size": 14,
        "merge_kernel_size": [2, 2],
        "merge_type": "sd2_tpool",
        "mm_projector_type": "patchmerger",
        "projector_hidden_act": "gelu",
        "projector_ln_eps": 1e-5,
        "video_attn_type": "spatial_temporal",
        "pos_emb_type": "divided_fixed",
        "init_pos_emb_height": 64,
        "init_pos_emb_width": 64,
        "init_pos_emb_time": 4,
    },
    "_name_or_path": _KIMI_K25_MODEL_DIR,
}


def _create_temp_video(
    num_frames: int = 8, size: Tuple[int, int] = (224, 224), fps: int = 2
) -> Optional[str]:
    """Create a temporary mp4 video file with random frames.

    The TRT-LLM input processor pre-decodes video files into frames using
    decord or cv2 before passing them to the HF processor. This helper
    creates a valid mp4 file for testing that pipeline.

    Tries cv2 (OpenCV) first, then imageio as fallback.

    Returns:
        Path to the temporary mp4 file, or None if no video writer is available.
    """
    h, w = size
    frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(num_frames)]
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    # Try cv2 (OpenCV) — most likely available in NVIDIA environments
    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            # cv2 expects BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        # Verify the file is non-empty
        if os.path.getsize(path) > 0:
            return path
    except Exception:
        pass

    # Fallback: try imageio + pyav
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=fps, plugin="pyav")
        return path
    except Exception:
        pass

    try:
        import imageio as iio

        iio.imwrite(path, np.stack(frames), fps=fps)
        return path
    except Exception:
        pass

    os.unlink(path)
    return None


def _build_hf_config():
    """Build the composite HF config with nested text_config object."""
    config_dict = deepcopy(KIMI_K25_SMALL_CONFIG)
    text_dict = config_dict.pop("text_config")
    vision_dict = config_dict.pop("vision_config")
    text_config = PretrainedConfig.from_dict(text_dict)
    config = PretrainedConfig.from_dict(config_dict)
    config.text_config = text_config
    config.vision_config = vision_dict
    return config


class TestKimiK25Structure(unittest.TestCase):
    """Structural tests for Kimi K2.5 — no forward pass."""

    def test_config_parsing(self):
        """Composite config has correct nested structure."""
        config = _build_hf_config()

        self.assertEqual(config.model_type, "kimi_k25")
        self.assertEqual(config.media_placeholder_token_id, 163605)
        self.assertTrue(hasattr(config, "text_config"))
        self.assertTrue(hasattr(config, "vision_config"))

        # text_config is a PretrainedConfig object (not dict)
        self.assertIsInstance(config.text_config, PretrainedConfig)
        self.assertEqual(config.text_config.hidden_size, 7168)
        self.assertEqual(config.text_config.vocab_size, 163840)
        self.assertEqual(config.text_config.num_hidden_layers, 2)
        self.assertFalse(config.text_config.tie_word_embeddings)

        # vision_config is a plain dict
        self.assertIsInstance(config.vision_config, dict)
        self.assertEqual(config.vision_config["patch_size"], 14)

    def test_model_instantiation(self):
        """KimiK25ForConditionalGeneration creates LLM backbone + vision encoder."""
        config = _build_hf_config()
        model_config = ModelConfig(pretrained_config=config)

        model = KimiK25ForConditionalGeneration(model_config)

        # LLM backbone exists
        self.assertTrue(hasattr(model, "llm"))
        self.assertTrue(hasattr(model.llm, "model"))
        self.assertTrue(hasattr(model.llm.model, "embed_tokens"))
        self.assertTrue(hasattr(model.llm.model, "layers"))
        self.assertEqual(len(model.llm.model.layers), 2)

        # Vision encoder exists
        self.assertTrue(hasattr(model, "mm_encoder"))
        self.assertIsInstance(model.mm_encoder, KimiK25VisionModel)

        # Projector exists on vision encoder
        self.assertTrue(hasattr(model.mm_encoder, "mm_projector"))

        # Media placeholder token ID
        self.assertEqual(model._media_placeholder_token_id, 163605)

    def test_vision_encoder_uses_trtllm_modules(self):
        """Vision blocks reuse TRT-LLM normalization, attention, linear, and MLP modules."""
        config = _build_hf_config()
        model_config = ModelConfig(pretrained_config=config)
        model = KimiK25ForConditionalGeneration(model_config)

        block = model.mm_encoder.encoder.blocks[0]
        self.assertIsInstance(block.norm0, LayerNorm)
        self.assertIsInstance(block.norm1, LayerNorm)
        self.assertIsInstance(block.attn, Attention)
        self.assertIsInstance(block.attn.qkv_proj, Linear)
        self.assertIsInstance(block.attn.o_proj, Linear)
        self.assertIsInstance(block.mlp, MLP)
        self.assertIsInstance(model.mm_encoder.encoder.final_layernorm, LayerNorm)
        self.assertIsInstance(model.mm_encoder.mm_projector.pre_norm, LayerNorm)
        self.assertIsInstance(model.mm_encoder.mm_projector.proj[0], Linear)
        self.assertIsInstance(model.mm_encoder.mm_projector.proj[2], Linear)

    def test_vision_encoder_deferred_under_meta_init(self):
        """MetaInitMode must not retain a vision encoder with meta tensors."""
        config = _build_hf_config()
        model_config = ModelConfig(pretrained_config=config)

        with MetaInitMode():
            model = KimiK25ForConditionalGeneration(model_config)

        self.assertIsNone(model.mm_encoder)

    def test_frames_to_chunks_uses_original_frame_indices_for_timestamps(self):
        """Pre-sampled VideoData frames keep timestamps from the original video."""
        from PIL import Image

        frames = [Image.new("RGB", (16, 16), color=(i, i, i)) for i in range(4)]

        _, prompt = _frames_to_chunks(
            frames,
            temporal_merge_kernel_size=2,
            fps=10.0,
            frame_indices=[0, 5, 9, 12],
        )

        self.assertIn("00:00:00.000<|media_begin|>video", prompt)
        self.assertIn("00:00:00.900<|media_begin|>video", prompt)
        self.assertNotIn("00:00:00.200<|media_begin|>video", prompt)

    def test_frames_to_chunks_rejects_invalid_timestamp_mode(self):
        """Timestamp formatting matches HF behavior by rejecting unknown modes."""
        from PIL import Image

        frames = [Image.new("RGB", (16, 16), color=(0, 0, 0))]

        with self.assertRaisesRegex(ValueError, "Invalid timestamp mode"):
            _frames_to_chunks(frames, timestamp_mode="invalid")

    def test_config_alignment(self):
        """__init__ aligns model.config with the LLM backbone's text_config."""
        config = _build_hf_config()
        model_config = ModelConfig(pretrained_config=config)
        model = KimiK25ForConditionalGeneration(model_config)

        # After __init__, model.config should be the text_config
        self.assertEqual(model.config.hidden_size, 7168)
        self.assertEqual(model.config.vocab_size, 163840)

    def test_multimodal_data_device_paths(self):
        """multimodal_data_device_paths returns expected keys for image and video."""
        config = _build_hf_config()
        model_config = ModelConfig(pretrained_config=config)
        model = KimiK25ForConditionalGeneration(model_config)

        paths = model.multimodal_data_device_paths
        # Image paths
        self.assertIn("image.pixel_values", paths)
        self.assertIn("image.image_grid_thw", paths)
        # Video paths (MoonViT3d supports spatial-temporal attention)
        self.assertIn("video.pixel_values_videos", paths)
        self.assertIn("video.video_grid_thw", paths)

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_creation(self):
        """Input processor can be created from the checkpoint directory."""
        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        self.assertIsInstance(processor, KimiK25InputProcessor)
        self.assertEqual(processor.get_vocab_size(), 163840)

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_text_only(self):
        """Input processor handles text-only inputs without error."""
        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        token_ids, extra = processor(
            {"prompt": "Hello, world!"},
            sampling_params=None,
        )
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_get_mm_token_ids(self):
        """get_mm_token_ids returns the media_placeholder_token_id."""
        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        mm_ids = processor.get_mm_token_ids()
        self.assertIsNotNone(mm_ids)
        self.assertEqual(mm_ids.tolist(), [163605])

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_num_tokens_per_image(self):
        """get_num_tokens_per_image computes correct token count."""
        from PIL import Image

        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        # 224x224 image with patch_size=14, merge_kernel_size=2:
        # patches = (224/14)^2 = 256, after merge = 256/4 = 64
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        num_tokens = processor.get_num_tokens_per_image(image=img)
        self.assertEqual(num_tokens, 64)

        # 448x448 image: patches = (448/14)^2 = 1024, after merge = 256
        img2 = Image.new("RGB", (448, 448), color=(128, 128, 128))
        num_tokens2 = processor.get_num_tokens_per_image(image=img2)
        self.assertEqual(num_tokens2, 256)

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_num_tokens_various_resolutions(self):
        """Token count for non-square and large images exercises NaViT resize."""
        from PIL import Image

        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        # Non-square: wide image (896x448)
        img_wide = Image.new("RGB", (896, 448), color=(100, 150, 200))
        tokens_wide = processor.get_num_tokens_per_image(image=img_wide)
        self.assertGreater(tokens_wide, 0)

        # Non-square: tall image (448x896)
        img_tall = Image.new("RGB", (448, 896), color=(200, 150, 100))
        tokens_tall = processor.get_num_tokens_per_image(image=img_tall)
        self.assertGreater(tokens_tall, 0)

        # Wide and tall should produce the same token count (symmetric)
        self.assertEqual(tokens_wide, tokens_tall)

        # Large image: should be downscaled by NaViT resize
        # in_patch_limit=16384, patch_limit_on_one_side=512
        img_large = Image.new("RGB", (2048, 2048), color=(50, 50, 50))
        tokens_large = processor.get_num_tokens_per_image(image=img_large)
        self.assertGreater(tokens_large, 0)
        # After NaViT downscale + merge, should not exceed the patch limit
        # Max patches per side = 512, after merge = 256 per side
        # So max tokens = 256 * 256 = 65536 (theoretical upper bound)
        self.assertLessEqual(tokens_large, 256 * 256)

        # Small image: 28x28 (minimum: 1 patch after merge)
        img_tiny = Image.new("RGB", (28, 28), color=(255, 255, 255))
        tokens_tiny = processor.get_num_tokens_per_image(image=img_tiny)
        self.assertGreaterEqual(tokens_tiny, 1)

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_multi_image(self):
        """Input processor handles multi-image input and expands placeholders correctly."""
        from PIL import Image

        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        img1 = Image.new("RGB", (224, 224), color=(128, 0, 0))
        img2 = Image.new("RGB", (448, 448), color=(0, 128, 0))

        expected_tokens_img1 = processor.get_num_tokens_per_image(image=img1)
        expected_tokens_img2 = processor.get_num_tokens_per_image(image=img2)

        # Build prompt with two images via multi_modal_data
        prompt = (
            "<|media_begin|><|media_pad|><|media_end|>"
            "<|media_begin|><|media_pad|><|media_end|>"
            "Describe both images."
        )
        token_ids, extra = processor(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [img1, img2],
                },
            },
            sampling_params=None,
        )

        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 0)

        # Verify multimodal_data contains image data
        mm_data = extra.get("multimodal_data", {})
        self.assertIn("image", mm_data)
        self.assertIn("pixel_values", mm_data["image"])

        # Count how many placeholder tokens appear in the expanded output.
        # Each image's single <|media_pad|> should be expanded to N copies.
        placeholder_count = sum(1 for t in token_ids if t == 163605)
        self.assertEqual(
            placeholder_count,
            expected_tokens_img1 + expected_tokens_img2,
            f"Expected {expected_tokens_img1} + {expected_tokens_img2} = "
            f"{expected_tokens_img1 + expected_tokens_img2} placeholder tokens "
            f"for two images, got {placeholder_count}",
        )

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_video(self):
        """Input processor handles video input and produces video multimodal data.

        MoonViT3d uses spatial-temporal attention for video. The TRT-LLM
        input processor pre-decodes video files into temporal chunks using
        decord or cv2, then passes them to the HF processor as video_chunk
        items (bypassing the HF processor's mecord dependency).
        """
        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        video_path = _create_temp_video(num_frames=8, size=(224, 224))
        if video_path is None:
            self.skipTest("cv2/imageio not available to create temp video")

        try:
            # Use the video_placeholder token. The input processor's
            # _decode_video_to_chunks splits the video into temporal chunks
            # and replaces the placeholder with per-chunk timestamp prompts,
            # each containing <|media_pad|>.
            prompt = "<|kimi_k25_video_placeholder|>Describe this video."
            token_ids, extra = processor(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "video": [video_path],
                    },
                },
                sampling_params=None,
            )

            self.assertIsInstance(token_ids, list)
            self.assertGreater(len(token_ids), 0)

            # Verify multimodal_data contains pixel data under "image" key.
            # MoonViT3d processes images and video chunks identically, so
            # all media (including video) goes under the "image" key.
            mm_data = extra.get("multimodal_data", {})
            self.assertIn(
                "image",
                mm_data,
                "Expected 'image' key in multimodal_data for video input (MoonViT3d unified path)",
            )
            self.assertIn("pixel_values", mm_data["image"])

            # The expanded token_ids should contain placeholder tokens
            placeholder_count = sum(1 for t in token_ids if t == 163605)
            self.assertGreater(
                placeholder_count, 0, "Video input should expand placeholders to vision tokens"
            )
        finally:
            os.unlink(video_path)

    @unittest.skipUnless(
        os.path.isdir(_KIMI_K25_MODEL_DIR), f"Model dir not found: {_KIMI_K25_MODEL_DIR}"
    )
    def test_input_processor_mixed_image_and_video(self):
        """Input processor handles mixed image + video input in one request."""
        from PIL import Image

        tokenizer = AutoTokenizer.from_pretrained(_KIMI_K25_MODEL_DIR, trust_remote_code=True)
        processor = create_input_processor(_KIMI_K25_MODEL_DIR, tokenizer=tokenizer)

        # One image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # One short video file
        video_path = _create_temp_video(num_frames=8, size=(224, 224))
        if video_path is None:
            self.skipTest("cv2/imageio not available to create temp video")

        try:
            # Image uses the standard media placeholder; video uses the
            # video_placeholder which gets expanded to per-chunk prompts.
            prompt = (
                "<|media_begin|><|media_pad|><|media_end|>"
                "<|kimi_k25_video_placeholder|>"
                "Describe the image and the video."
            )
            token_ids, extra = processor(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": [img],
                        "video": [video_path],
                    },
                },
                sampling_params=None,
            )

            self.assertIsInstance(token_ids, list)
            self.assertGreater(len(token_ids), 0)

            mm_data = extra.get("multimodal_data", {})
            # All media (images + video chunks) are unified under "image"
            # key because MoonViT3d processes them identically.
            self.assertIn("image", mm_data, "Expected 'image' key for mixed input")
            self.assertIn("pixel_values", mm_data["image"])

            # Placeholder tokens should be expanded for both modalities
            placeholder_count = sum(1 for t in token_ids if t == 163605)
            self.assertGreater(
                placeholder_count, 0, "Mixed input should produce expanded placeholder tokens"
            )
        finally:
            os.unlink(video_path)


class TestKimiK25AutoModelRegistration(unittest.TestCase):
    """Verify KimiK25ForConditionalGeneration is registered in MODEL_CLASS_MAPPING."""

    def test_auto_model_registered(self):
        """KimiK25ForConditionalGeneration must be in the model registry."""
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        cls = MODEL_CLASS_MAPPING.get("KimiK25ForConditionalGeneration")
        self.assertIsNotNone(cls, "KimiK25ForConditionalGeneration not in MODEL_CLASS_MAPPING")
        self.assertIs(cls, KimiK25ForConditionalGeneration)


# ---------------------------------------------------------------------------
# E2E Smoke Test — requires GPU + model checkpoint
# ---------------------------------------------------------------------------
# Quick end-to-end verification that Kimi-K2.5 NVFP4 loads and generates
# correctly with text, image, and video inputs. Not an accuracy benchmark.


def _save_solid_image(color: tuple, width: int = 224, height: int = 224) -> str:
    """Create a solid-colour image and return the temp file path."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=color)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return path


def _build_text_input(model_path: str, prompt: str, tokenizer=None, processor=None) -> dict:
    """Build a text-only input with chat template applied."""
    from transformers import AutoProcessor as _AutoProcessor

    from tensorrt_llm.inputs import prompt_inputs
    from tensorrt_llm.inputs.utils import apply_chat_template
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    if tokenizer is None:
        tokenizer = ModelLoader.load_hf_tokenizer(model_path, use_fast=True)
    if processor is None:
        processor = _AutoProcessor.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )
    text = apply_chat_template(
        model_type="kimi_k25",
        tokenizer=tokenizer,
        processor=processor,
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}],
    )
    return prompt_inputs({"prompt": text})


def _build_image_input(model_path: str, prompt: str, image_paths: list) -> dict:
    """Build a multimodal input dict using the framework's standard loader."""
    from tensorrt_llm.inputs import prompt_inputs
    from tensorrt_llm.inputs.utils import default_multimodal_input_loader

    modality = "image" if len(image_paths) == 1 else "multiple_image"
    inputs = default_multimodal_input_loader(
        tokenizer=None,
        model_dir=model_path,
        model_type="kimi_k25",
        modality=modality,
        prompts=[prompt],
        media=image_paths,
        image_data_format="pil",
    )
    return prompt_inputs(inputs[0])


@pytest.mark.threadleak(enabled=False)
@unittest.skipUnless(
    os.path.isdir(_KIMI_K25_MODEL_DIR),
    f"Model dir not found: {_KIMI_K25_MODEL_DIR}",
)
@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 4,
    "Requires at least 4 GPUs",
)
@skip_pre_blackwell_unittest
class TestKimiK25E2ESmoke(unittest.TestCase):
    """E2E smoke test for Kimi-K2.5 NVFP4 (requires 4x Blackwell GPU + checkpoint).

    Sends text-only, image, and video requests through the multimodal
    pipeline to verify model loading, vision encoding, and text generation
    work end-to-end.
    """

    MODEL_PATH = _KIMI_K25_MODEL_DIR

    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor as _AutoProcessor

        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig
        from tensorrt_llm.llmapi.llm_utils import ModelLoader

        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.80,
        )
        cls.llm = LLM(
            cls.MODEL_PATH,
            max_num_tokens=16384,
            kv_cache_config=kv_cache_config,
            tensor_parallel_size=4,
            trust_remote_code=True,
        )
        cls._tokenizer = ModelLoader.load_hf_tokenizer(cls.MODEL_PATH, use_fast=True)
        cls._processor = _AutoProcessor.from_pretrained(
            cls.MODEL_PATH, use_fast=True, trust_remote_code=True
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "llm") and cls.llm is not None:
            cls.llm.shutdown()

    def test_text_arithmetic(self):
        """Text-only: basic arithmetic."""
        from tensorrt_llm.llmapi import SamplingParams

        inp = _build_text_input(
            self.MODEL_PATH,
            "What is 2 + 3? Answer with a single number.",
            self._tokenizer,
            self._processor,
        )
        result = self.llm.generate(inp, sampling_params=SamplingParams(max_tokens=1024))
        text_out = result.outputs[0].text
        print(f"[text-arithmetic] {text_out!r}")
        self.assertGreater(len(text_out), 0, "Text generation produced empty output")

    def test_text_reasoning(self):
        """Text-only: reasoning (9.11 vs 9.9)."""
        from tensorrt_llm.llmapi import SamplingParams

        inp = _build_text_input(
            self.MODEL_PATH,
            "which one is bigger, 9.11 or 9.9? think carefully.",
            self._tokenizer,
            self._processor,
        )
        result = self.llm.generate(inp, sampling_params=SamplingParams(max_tokens=1024))
        text_out = result.outputs[0].text
        print(f"[text-reasoning] {text_out!r}")
        self.assertGreater(len(text_out), 0, "Reasoning produced empty output")
        self.assertIn("9.9", text_out, f"Expected '9.9' as the bigger number, got: {text_out!r}")

    def test_single_image(self):
        """Single image: solid red."""
        from tensorrt_llm.llmapi import SamplingParams

        red_path = _save_solid_image((255, 0, 0))
        try:
            inp = _build_image_input(
                self.MODEL_PATH, "What color is this image? Answer in one word.", [red_path]
            )
            result = self.llm.generate(inp, sampling_params=SamplingParams(max_tokens=1024))
            img_out = result.outputs[0].text
            print(f"[single-image-red] {img_out!r}")
            self.assertIn("red", img_out.lower(), f"Expected 'red', got: {img_out!r}")
        finally:
            os.unlink(red_path)

    def test_batch_images(self):
        """Batch images: red, green, blue."""
        from tensorrt_llm.llmapi import SamplingParams

        colors = [((255, 0, 0), "red"), ((0, 255, 0), "green"), ((0, 0, 255), "blue")]
        color_paths = [_save_solid_image(rgb) for rgb, _ in colors]
        try:
            prompts = [
                _build_image_input(
                    self.MODEL_PATH, "What color is this image? Answer in one word.", [p]
                )
                for p in color_paths
            ]
            results = self.llm.generate(prompts, sampling_params=SamplingParams(max_tokens=1024))
            for res, (_, name) in zip(results, colors):
                out = res.outputs[0].text
                print(f"[batch-{name}] {out!r}")
                self.assertIn(name, out.lower(), f"Expected '{name}', got: {out!r}")
        finally:
            for p in color_paths:
                os.unlink(p)

    def test_multi_image(self):
        """Multi-image: red + blue (different sizes)."""
        from tensorrt_llm.llmapi import SamplingParams

        red_path = _save_solid_image((255, 0, 0), 224, 224)
        blue_path = _save_solid_image((0, 0, 255), 448, 448)
        try:
            inp = _build_image_input(
                self.MODEL_PATH, "What colors are in these two images?", [red_path, blue_path]
            )
            result = self.llm.generate(inp, sampling_params=SamplingParams(max_tokens=1024))
            multi_out = result.outputs[0].text
            print(f"[multi-image] {multi_out!r}")
            self.assertGreater(len(multi_out), 0, "Multi-image produced empty output")
        finally:
            os.unlink(red_path)
            os.unlink(blue_path)

    def test_url_image(self):
        """URL image: Kimi logo from HuggingFace."""
        import urllib.request

        from tensorrt_llm.llmapi import SamplingParams

        _LOGO_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"
        logo_path = None
        try:
            fd, logo_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            urllib.request.urlretrieve(_LOGO_URL, logo_path)
            inp = _build_image_input(self.MODEL_PATH, "Describe this image in detail.", [logo_path])
            result = self.llm.generate(inp, sampling_params=SamplingParams(max_tokens=1024))
            logo_out = result.outputs[0].text
            print(f"[url-image-logo] {logo_out!r}")
            self.assertGreater(len(logo_out), 0, "Logo image produced empty output")
        except urllib.error.URLError as e:
            self.skipTest(f"URL download failed: {e}")
        finally:
            if logo_path and os.path.isfile(logo_path):
                os.unlink(logo_path)

    def test_url_video(self):
        """URL video: demo from HuggingFace, passed as file path to input processor."""
        import urllib.request

        from tensorrt_llm.inputs import prompt_inputs
        from tensorrt_llm.llmapi import SamplingParams

        _VIDEO_URL = (
            "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/demo_video.mp4"
        )
        video_tmp = None
        try:
            fd, video_tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            urllib.request.urlretrieve(_VIDEO_URL, video_tmp)
        except urllib.error.URLError as e:
            if video_tmp and os.path.isfile(video_tmp):
                os.unlink(video_tmp)
            self.skipTest(f"Video download failed: {e}")

        try:
            # Build prompt with chat template directly via tokenizer,
            # bypassing default_multimodal_input_loader which doesn't
            # support video file paths. The input processor's __call__
            # will handle decoding via _decode_video_to_chunks.
            text = self._tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "<|kimi_k25_video_placeholder|>Describe the video in detail.",
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            inp = prompt_inputs(
                {
                    "prompt": text,
                    "multi_modal_data": {"video": [video_tmp]},
                }
            )
            # This test passes a video file path (str) rather than pre-decoded
            # PIL frames, on purpose: it exercises the same code path used by
            # trtllm-serve when a client uploads a raw video to the OpenAI API
            # (KimiK25InputProcessor._decode_video_to_chunks handles file paths
            # / bytes natively).
            #
            # The framework's multimodal hashing path
            # (tensorrt_llm/inputs/multimodal.py:785, find_mm_token_lengths)
            # supports VideoData and List[PIL.Image] but not file paths — it
            # asserts the video item is a list. Once an earlier image test in
            # this shared-LLM class sets multimodal_hashing_supported to True,
            # the framework would skip the first-try fallback and raise on
            # this assert. Resetting to None puts the framework back in the
            # first-try state, so it falls back (registry.py:929) to calling
            # KimiK25InputProcessor directly, which handles the file path.
            saved = self.llm.input_processor.multimodal_hashing_supported
            self.llm.input_processor.multimodal_hashing_supported = None
            try:
                result = self.llm.generate(inp, sampling_params=SamplingParams(max_tokens=4096))
            finally:
                self.llm.input_processor.multimodal_hashing_supported = saved
            video_out = result.outputs[0].text
            print(f"[url-video] {video_out!r}")
            self.assertGreater(len(video_out), 0, "Video produced empty output")
        finally:
            if video_tmp and os.path.isfile(video_tmp):
                os.unlink(video_tmp)
