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
"""Unit tests for Gemma 4 12B *Unified* (encoder-free multimodal).

Gemma 4 12B uses HF `architectures = ["Gemma4UnifiedForConditionalGeneration"]`
/ `model_type = "gemma4_unified"`. Its text backbone is a standard dense Gemma 4
text model nested under `text_config`; the multimodal front-end is encoder-free
(lightweight projections). Covered here: config parsing, the vendored
preprocessing components, architecture registration, and the wrapper's
weight-key routing. (Full forward/accuracy parity is covered by the end-to-end
smoke on the real 12B checkpoint.)
"""

from types import SimpleNamespace

import numpy as np
import torch

from tensorrt_llm._torch.configs import (
    Gemma4UnifiedAudioConfig,
    Gemma4UnifiedConfig,
    Gemma4UnifiedVisionConfig,
)
from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import Gemma4HfWeightMapper
from tensorrt_llm._torch.models.gemma4_unified_processing import (
    Gemma4UnifiedAudioFeatureExtractor,
    Gemma4UnifiedImageProcessor,
)
from tensorrt_llm._torch.models.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration
from tensorrt_llm._torch.models.modeling_utils import (
    _GEMMA4_ARCHITECTURES,
    MODEL_CLASS_MAPPER_MAPPING,
    MODEL_CLASS_MAPPING,
    get_model_architecture,
)

_UNIFIED_ARCH = "Gemma4UnifiedForConditionalGeneration"

# Mirrors the structure (not the full contents) of the 12B checkpoint's
# config.json; the values below are the real 12B geometry.
_UNIFIED_CONFIG_DICT = {
    "architectures": [_UNIFIED_ARCH],
    "model_type": "gemma4_unified",
    "image_token_id": 258880,
    "audio_token_id": 258881,
    "video_token_id": 258884,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "gemma4_unified_text",
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "vocab_size": 262400,
    },
    "vision_config": {
        "model_type": "gemma4_unified_vision",
        "mm_embed_dim": 3840,
        "mm_posemb_size": 1120,
        "output_proj_dims": 3840,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "rms_norm_eps": 1e-6,
    },
    "audio_config": {
        "model_type": "gemma4_unified_audio",
        "audio_embed_dim": 640,
        "rms_norm_eps": 1e-6,
    },
}


def test_config_registered_with_transformers():
    """All four gemma4_unified model_types resolve through AutoConfig's mapping."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for model_type in (
        "gemma4_unified",
        "gemma4_unified_text",
        "gemma4_unified_vision",
        "gemma4_unified_audio",
    ):
        assert model_type in CONFIG_MAPPING


def test_config_shim_parses_checkpoint_dict():
    """The shim reconstructs nested sub-configs from a checkpoint-style dict
    with the exact fields the model implementation reads."""
    config = Gemma4UnifiedConfig(**_UNIFIED_CONFIG_DICT)

    assert config.model_type == "gemma4_unified"
    assert config.image_token_id == 258880
    assert config.audio_token_id == 258881
    assert config.video_token_id == 258884

    # Text backbone parses via the native Gemma4 text config (dense Gemma 4).
    assert config.text_config.hidden_size == 3840
    assert config.text_config.num_hidden_layers == 48

    # Vision: encoder-free projector geometry.
    assert config.vision_config.mm_embed_dim == 3840
    assert config.vision_config.output_proj_dims == 3840
    assert config.vision_config.patch_size == 16
    assert config.vision_config.pooling_kernel_size == 3

    # Audio: output_proj_dims / hidden_size alias audio_embed_dim (matches HF).
    assert config.audio_config.audio_embed_dim == 640
    assert config.audio_config.output_proj_dims == 640
    assert config.audio_config.hidden_size == 640


def test_config_shim_handles_absent_modalities():
    """HF sets a sub-config to None when the modality is absent; the shim must
    preserve that so the model skips building the corresponding embedder."""
    config = Gemma4UnifiedConfig(
        text_config=_UNIFIED_CONFIG_DICT["text_config"],
        vision_config=None,
        audio_config=None,
    )
    assert config.vision_config is None
    assert config.audio_config is None


def test_sub_config_shims_standalone():
    vision = Gemma4UnifiedVisionConfig()
    assert vision.mm_embed_dim == 3840
    assert vision.model_type == "gemma4_unified_vision"

    audio = Gemma4UnifiedAudioConfig()
    assert audio.audio_embed_dim == 640
    assert audio.output_proj_dims == 640
    assert audio.model_type == "gemma4_unified_audio"


def test_vendored_image_processor_output_geometry():
    """The vendored encoder-free image processor produces (batch, max_soft_tokens,
    patch_dim) pixel patches with -1-padded position ids."""
    image_processor = Gemma4UnifiedImageProcessor()
    image = torch.randint(0, 256, (3, 240, 320), dtype=torch.uint8)

    outputs = image_processor(image, return_tensors="pt")

    max_soft_tokens = image_processor.max_soft_tokens  # 280 for the 12B
    patch_dim = (image_processor.patch_size * image_processor.pooling_kernel_size) ** 2 * 3
    assert outputs["pixel_values"].shape == (1, max_soft_tokens, patch_dim)
    assert outputs["image_position_ids"].shape == (1, max_soft_tokens, 2)

    # Real patches carry non-negative grid positions; padded rows are -1 on both axes.
    position_ids = outputs["image_position_ids"][0]
    num_real = outputs["num_soft_tokens_per_image"][0]
    assert (position_ids[:num_real] >= 0).all()
    if num_real < max_soft_tokens:
        assert (position_ids[num_real:] == -1).all()
        assert (outputs["pixel_values"][0, num_real:] == 0).all()


def test_vendored_audio_feature_extractor_framing():
    """The vendored audio feature extractor chunks a raw 16 kHz waveform into
    (num_tokens, 640) frames with an all-valid boolean mask."""
    feature_extractor = Gemma4UnifiedAudioFeatureExtractor()
    one_second = np.random.randn(16_000).astype(np.float32)

    outputs = feature_extractor(one_second, return_tensors="pt")

    # 16000 samples / 640 samples-per-token = 25 tokens
    assert outputs["input_features"].shape == (1, 25, 640)
    assert outputs["input_features_mask"].shape == (1, 25)
    assert outputs["input_features_mask"].dtype == torch.bool
    assert outputs["input_features_mask"].all()

    # Frame content is the raw waveform, unchanged.
    torch.testing.assert_close(
        outputs["input_features"][0].reshape(-1),
        torch.from_numpy(one_second),
    )


def test_auto_model_registered():
    """The new arch is discoverable in the auto-model + Gemma4 arch registries
    (importing the package above ran the @register_auto_model decorator)."""
    assert _UNIFIED_ARCH in MODEL_CLASS_MAPPING
    assert _UNIFIED_ARCH in _GEMMA4_ARCHITECTURES


def test_get_model_architecture_resolves_wrapper():
    config = SimpleNamespace(architectures=[_UNIFIED_ARCH])
    cls, arch = get_model_architecture(config)
    assert cls is Gemma4UnifiedForConditionalGeneration
    assert arch == _UNIFIED_ARCH


def test_weight_mapper_registered():
    # The Gemma4 HF weight mapper must claim the unified arch too, otherwise the
    # generic mapper would mishandle the per-layer head_dim / k_eq_v / layer_scalar
    # logic. register_mapper("HF", name) stores the mapper under f"{name}_{format}".
    mapper_cls = MODEL_CLASS_MAPPER_MAPPING.get(f"{_UNIFIED_ARCH}_HF")
    assert mapper_cls is Gemma4HfWeightMapper


def test_load_weights_filters_and_remaps():
    """The wrapper hands the text core only the `model.language_model.*` weights
    (remapped to `model.*`) and drops the encoder-free multimodal projection
    tensors."""
    captured = {}

    class _FakeLLM:
        def load_weights(self, weights, weight_mapper=None):
            captured.update(weights)

    # __new__ bypasses __init__ (which builds the real text core on GPU); this
    # exercises only the pure-Python key routing in load_weights. The encoder-free
    # MM embedders are consulted only when present, so None = text-only routing.
    wrapper = Gemma4UnifiedForConditionalGeneration.__new__(Gemma4UnifiedForConditionalGeneration)
    wrapper.llm = _FakeLLM()
    wrapper.embed_vision = None
    wrapper.embed_audio = None

    raw_weights = {
        "model.language_model.embed_tokens.weight": 0,
        "model.language_model.layers.0.self_attn.q_proj.weight": 1,
        "model.language_model.layers.0.layer_scalar": 2,
        "model.language_model.norm.weight": 3,
        # Encoder-free MM projection tensors -- must be dropped by text-only routing:
        "model.vision_embedder.patch_dense.weight": 90,
        "model.embed_vision.embedding_projection.weight": 91,
        "model.embed_audio.embedding_projection.weight": 92,
    }
    wrapper.load_weights(raw_weights, weight_mapper=None)

    # Exactly the four text-backbone tensors reach the LLM.
    assert len(captured) == 4
    # "model.language_model." was remapped to "model.".
    assert "model.embed_tokens.weight" in captured
    assert "model.layers.0.self_attn.q_proj.weight" in captured
    assert "model.layers.0.layer_scalar" in captured
    assert "model.norm.weight" in captured
    # None of the encoder-free MM projection tensors leaked through.
    for key in captured:
        assert "vision_embedder" not in key
        assert "embed_vision" not in key
        assert "embed_audio" not in key
