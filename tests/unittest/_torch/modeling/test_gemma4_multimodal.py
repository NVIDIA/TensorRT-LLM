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
"""Unit tests for the Gemma4 multimodal model components.

Tests Gemma4MultimodalEmbedder, Gemma4ForConditionalGeneration,
and Gemma4InputProcessor with HF reference comparison.
"""

import unittest

import pytest
import torch

# Gemma4 requires transformers>=5.5.0 (native Gemma4 config/model classes).
pytest.importorskip(
    "transformers", minversion="5.5.0", reason="Gemma4 requires transformers>=5.5.0"
)

from transformers import AutoModel, Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig  # noqa: E402

from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_gemma4mm import (  # noqa: E402
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from tensorrt_llm.mapping import Mapping  # noqa: E402

# Small vision config for testing
SMALL_VISION_CONFIG = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 1024,
    "model_type": "gemma4_vision",
}

SMALL_TEXT_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "global_head_dim": 32,
    "num_global_key_value_heads": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": None,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
}


def _get_model_path():
    import os

    llm_models_root = os.environ.get("LLM_MODELS_ROOT")
    if llm_models_root:
        return os.path.join(llm_models_root, "gemma4/gemma-4-26B-A4B-it")
    return None


MODEL_26B_PATH = _get_model_path()


def _model_available():
    import os

    if MODEL_26B_PATH is None:
        return False
    return os.path.isfile(os.path.join(MODEL_26B_PATH, "config.json"))


class TestGemma4InputProcessor(unittest.TestCase):
    """Tests for Gemma4InputProcessor with real model tokenizer/processor files."""

    @classmethod
    def setUpClass(cls):
        if not _model_available():
            raise unittest.SkipTest(f"Gemma4 model not found at {MODEL_26B_PATH}")

        from transformers import AutoConfig, AutoTokenizer

        cls.config = AutoConfig.from_pretrained(MODEL_26B_PATH)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_26B_PATH)

    def _make_processor(self):
        from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4InputProcessor

        return Gemma4InputProcessor(
            model_path=MODEL_26B_PATH,
            config=self.config,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
        )

    def test_instantiation(self):
        """Input processor initializes with AutoProcessor."""
        proc = self._make_processor()
        self.assertIsNotNone(proc._processor)
        self.assertIsNotNone(proc._image_processor)
        self.assertEqual(proc.model_path, MODEL_26B_PATH)

    def test_text_only_processing(self):
        """Text-only input returns input_ids without multimodal data."""
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        inputs = {"prompt": "Hello, how are you?"}
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsInstance(input_ids, list)
        self.assertGreater(len(input_ids), 0)
        self.assertIsNone(mm_data)

    def test_image_processing(self):
        """Image input returns input_ids + pixel_values in multimodal data."""
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # Use chat template to properly format the prompt
        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsInstance(input_ids, list)
        self.assertGreater(len(input_ids), 0)
        self.assertIsNotNone(mm_data)
        self.assertIn("multimodal_data", mm_data)
        self.assertIn("image", mm_data["multimodal_data"])
        img_data = mm_data["multimodal_data"]["image"]
        self.assertIn("pixel_values", img_data)
        self.assertEqual(img_data["pixel_values"].dim(), 3)  # [num_images, patches, channels]

    def test_image_token_expansion(self):
        """Chat template expands <image> placeholder to image tokens."""
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, _ = proc(inputs, sp)

        image_token_id = self.config.image_token_id
        image_count = sum(1 for t in input_ids if t == image_token_id)
        # Image should be expanded to multiple image tokens
        self.assertGreater(image_count, 0, "Expected image tokens in expanded input_ids")

    def test_multiple_images(self):
        """Multiple images are handled correctly."""
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "Compare these images."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img1, img2]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsNotNone(mm_data)
        pv = mm_data["multimodal_data"]["image"]["pixel_values"]
        # Two images should give batch dim of 2
        self.assertEqual(pv.shape[0], 2)

    def test_torch_tensor_image(self):
        """Torch tensor images disable rescaling."""
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        # Create a float tensor image (already rescaled)
        img_tensor = torch.randn(3, 224, 224)

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img_tensor]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsNotNone(mm_data)
        self.assertIn("pixel_values", mm_data["multimodal_data"]["image"])

    def test_processor_properties(self):
        """Verify processor property accessors."""
        proc = self._make_processor()
        self.assertIs(proc.config, self.config)
        self.assertIs(proc.tokenizer, self.tokenizer)
        self.assertEqual(proc.model_path, MODEL_26B_PATH)
        self.assertIsNotNone(proc.processor)

    def test_audio_processing(self):
        """Audio input returns input_ids + audio_features in multimodal data."""
        import numpy as np

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        # Synthetic 1-second mono audio at 16 kHz.
        audio = np.random.randn(16000).astype(np.float32)

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": "Transcribe the audio."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsInstance(input_ids, list)
        self.assertIsNotNone(mm_data)
        self.assertIn("multimodal_data", mm_data)
        self.assertIn("audio", mm_data["multimodal_data"])
        audio_data = mm_data["multimodal_data"]["audio"]
        self.assertIn("audio_features", audio_data)
        # (batch, frames, mel_bins)
        self.assertEqual(audio_data["audio_features"].dim(), 3)

    def test_audio_token_expansion(self):
        """Audio placeholder expands to audio soft-token positions."""
        import numpy as np

        from tensorrt_llm.sampling_params import SamplingParams

        audio_token_id = getattr(self.config, "audio_token_id", None)
        if audio_token_id is None:
            self.skipTest("Model config has no audio_token_id")

        proc = self._make_processor()
        audio = np.random.randn(16000).astype(np.float32)

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": "Transcribe the audio."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, _ = proc(inputs, sp)

        audio_count = sum(1 for t in input_ids if t == audio_token_id)
        self.assertGreater(audio_count, 0, "Expected audio soft tokens in expanded input_ids")

    def test_audio_input_normalization(self):
        """_normalize_audio_inputs downmixes stereo and resamples."""
        import numpy as np

        from tensorrt_llm._torch.models.modeling_gemma4mm import _normalize_audio_inputs

        # Stereo input at 48 kHz -> mono at 16 kHz.
        sr_src = 48000
        dur_sec = 0.5
        stereo = np.random.randn(int(sr_src * dur_sec), 2).astype(np.float32)
        normalized = _normalize_audio_inputs([(stereo, sr_src)], target_sr=16000)
        self.assertEqual(len(normalized), 1)
        arr = normalized[0]
        self.assertEqual(arr.ndim, 1)
        self.assertEqual(arr.dtype, np.float32)
        # 0.5 sec @ 16 kHz ≈ 8000 samples (resample_poly may differ by 1).
        self.assertAlmostEqual(arr.size, 8000, delta=4)

    def test_audio_input_torch_tensor(self):
        """_normalize_audio_inputs accepts torch tensors."""
        import numpy as np

        from tensorrt_llm._torch.models.modeling_gemma4mm import _normalize_audio_inputs

        tensor = torch.randn(8000)
        normalized = _normalize_audio_inputs([tensor], target_sr=16000)
        self.assertEqual(normalized[0].dtype, np.float32)
        self.assertEqual(normalized[0].ndim, 1)
        self.assertEqual(normalized[0].size, 8000)

    def test_unknown_mm_modality_rejected(self):
        """Passing an unknown modality key raises KeyError."""
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        inputs = {
            "prompt": "<|placeholder|>",
            "multi_modal_data": {"point_cloud": object()},
        }
        sp = SamplingParams(max_tokens=10)
        with self.assertRaises(KeyError):
            proc(inputs, sp)

    def test_video_processing_returns_pixel_values(self):
        """Video input bypasses HF Gemma4Processor and yields per-frame
        pixel_values + expanded video soft tokens in input_ids.

        Regression for the ``merged_typed_dict.__init__() got an unexpected
        keyword argument 'video_sizes'`` failure where calling
        ``Gemma4Processor`` with ``videos=`` triggers strict-typed-dict
        validation in transformers 5.5.x.  Our input processor sidesteps
        this by calling the underlying ``video_processor`` directly and
        expanding the placeholder manually.
        """
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        if not hasattr(proc._processor, "video_processor"):
            self.skipTest("Processor has no video_processor")

        # 32 dummy 224x224 frames (matches Gemma4VideoProcessor.num_frames).
        frames = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(32)
        ]
        # Dataclass-like wrapper to exercise the ``getattr(v, 'frames', ...)``
        # path used by the TRT-LLM video loader.
        VideoData = type("VideoData", (), {})
        vd = VideoData()
        vd.frames = frames

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"video": [vd]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsNotNone(mm_data)
        self.assertIn("video", mm_data["multimodal_data"])
        video_data = mm_data["multimodal_data"]["video"]
        self.assertIn("pixel_values", video_data)
        # Per-frame pixel_values: (num_frames, num_patches, channels)
        self.assertEqual(video_data["pixel_values"].dim(), 3)
        self.assertEqual(video_data["pixel_values"].shape[0], 32)

        # The video_token id should appear in the expanded input_ids.
        video_token_id = self._video_token_id()
        if video_token_id is not None:
            self.assertGreater(
                sum(1 for t in input_ids if t == video_token_id),
                0,
                "Expected video soft tokens in expanded input_ids",
            )

    def _video_token_id(self):
        return getattr(self.config, "video_token_id", None)


class TestGemma4MultimodalEmbedder(unittest.TestCase):
    """Test the multimodal embedder projection layer."""

    def test_embedder_instantiation(self):
        """Embedder creates norm + projection with correct dimensions."""
        embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=64,
            text_hidden_size=128,
            eps=1e-6,
            dtype=torch.bfloat16,
        )
        self.assertEqual(embedder.embedding_projection.weight.shape, torch.Size([128, 64]))

    @torch.no_grad()
    def test_embedder_forward(self):
        """Embedder forward: norm → linear, output matches expected shape."""
        embedder = (
            Gemma4MultimodalEmbedder(
                mm_hidden_size=64,
                text_hidden_size=128,
                dtype=torch.bfloat16,
            )
            .to("cuda")
            .to(torch.bfloat16)
        )
        # Initialize with non-zero weights to avoid zero-output edge cases
        torch.nn.init.xavier_uniform_(embedder.embedding_projection.weight.data)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = embedder(x)
        self.assertEqual(out.shape, torch.Size([4, 128]))
        self.assertFalse(out.isnan().any())

    @torch.no_grad()
    def test_embedder_matches_hf(self):
        """Compare embedder output with HF Gemma4MultimodalEmbedder."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder as HFEmbedder,
        )

        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        dtype = torch.bfloat16
        device = "cuda"

        hf_embedder = HFEmbedder(vision_cfg, text_cfg).to(dtype).to(device).eval()

        trt_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_cfg.hidden_size,
            eps=vision_cfg.rms_norm_eps,
            dtype=dtype,
        ).to(device)

        # Copy HF weights to TRT
        trt_embedder.embedding_projection.weight.data.copy_(
            hf_embedder.embedding_projection.weight.data
        )

        x = torch.randn(4, vision_cfg.hidden_size, device=device, dtype=dtype)
        with torch.inference_mode():
            hf_out = hf_embedder(x)
            trt_out = trt_embedder(x)

        self.assertTrue(
            torch.allclose(hf_out, trt_out, atol=1e-2),
            f"Embedder max diff: {(hf_out - trt_out).abs().max()}",
        )


class TestGemma4VisionTower(unittest.TestCase):
    """Test the vision tower (native transformers AutoModel)."""

    def test_vision_tower_creation(self):
        """Vision tower can be created from config."""
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        tower = AutoModel.from_config(vision_cfg)
        self.assertIsNotNone(tower)
        params = sum(p.numel() for p in tower.parameters())
        self.assertGreater(params, 0)

    @torch.no_grad()
    def test_vision_tower_forward(self):
        """Vision tower forward pass produces valid output with correct shape."""
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        dtype = torch.bfloat16
        device = "cuda"

        tower = AutoModel.from_config(vision_cfg).to(dtype).to(device).eval()

        # 6x6 = 36 patches, pixel_values shape: [B, N_patches, patch_size^2 * 3]
        B, side = 1, 6
        N_patches = side * side
        C_in = vision_cfg.patch_size**2 * 3  # 16*16*3 = 768
        pixel_values = torch.randn(B, N_patches, C_in, device=device, dtype=dtype)
        position_ids = torch.stack(
            torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(1, -1, 2)

        pooling_k2 = vision_cfg.pooling_kernel_size**2
        output_length = N_patches // pooling_k2

        with torch.inference_mode():
            out = tower(pixel_values, position_ids, output_length=output_length)

        hidden = out.last_hidden_state
        self.assertEqual(hidden.shape, torch.Size([output_length, vision_cfg.hidden_size]))
        self.assertFalse(hidden.isnan().any(), "Vision tower output contains NaN")

    @torch.no_grad()
    def test_vision_pipeline_matches_hf(self):
        """Full vision pipeline (tower → embedder) matches HF reference."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder as HFEmbedder,
        )
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel as HFVisionModel

        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        dtype = torch.bfloat16
        device = "cuda"

        # --- HF reference pipeline ---
        hf_tower = HFVisionModel(vision_cfg).to(dtype).to(device).eval()
        hf_embedder = HFEmbedder(vision_cfg, text_cfg).to(dtype).to(device).eval()

        # --- TRT-LLM pipeline (same tower class + our embedder) ---
        trt_tower = AutoModel.from_config(vision_cfg).to(dtype).to(device).eval()
        trt_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_cfg.hidden_size,
            eps=vision_cfg.rms_norm_eps,
            dtype=dtype,
        ).to(device)

        # Copy weights: tower
        trt_tower.load_state_dict(hf_tower.state_dict())
        # Copy weights: embedder
        trt_embedder.embedding_projection.weight.data.copy_(
            hf_embedder.embedding_projection.weight.data
        )

        # Dummy input: 6x6 = 36 patches
        side = 6
        N_patches = side * side
        C_in = vision_cfg.patch_size**2 * 3
        pixel_values = torch.randn(1, N_patches, C_in, device=device, dtype=dtype)
        position_ids = torch.stack(
            torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(1, -1, 2)

        pooling_k2 = vision_cfg.pooling_kernel_size**2
        output_length = N_patches // pooling_k2

        with torch.inference_mode():
            # HF pipeline
            hf_hidden = hf_tower(pixel_values, position_ids, output_length=output_length)
            hf_out = hf_embedder(hf_hidden.last_hidden_state)

            # TRT-LLM pipeline
            trt_hidden = trt_tower(pixel_values, position_ids, output_length=output_length)
            trt_out = trt_embedder(trt_hidden.last_hidden_state.unsqueeze(0)).squeeze(0)

        self.assertEqual(hf_out.shape, trt_out.shape)
        self.assertTrue(
            torch.allclose(hf_out, trt_out, atol=1e-2),
            f"Vision pipeline max diff: {(hf_out - trt_out).abs().max():.6f}",
        )


class TestGemma4ForConditionalGeneration(unittest.TestCase):
    """Test the multimodal VLM wrapper."""

    def test_instantiation_with_vision(self):
        """VLM wrapper creates LLM + vision tower + embedder."""
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        config = Gemma4Config(
            text_config=text_cfg,
            vision_config=vision_cfg,
            audio_config=None,
        )

        mc = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend="FLASHINFER",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIsNotNone(model.llm)
        self.assertIsNotNone(model.vision_tower)
        self.assertIsNotNone(model.embed_vision)
        self.assertIsNone(model.audio_tower)
        self.assertIsNone(model.embed_audio)

    def test_instantiation_without_vision(self):
        """VLM wrapper works text-only when vision_config is None."""
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        config = Gemma4Config(
            text_config=text_cfg,
            vision_config=None,
            audio_config=None,
        )

        mc = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend="FLASHINFER",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIsNotNone(model.llm)
        self.assertIsNone(model.vision_tower)
        self.assertIsNone(model.embed_vision)


if __name__ == "__main__":
    unittest.main()
