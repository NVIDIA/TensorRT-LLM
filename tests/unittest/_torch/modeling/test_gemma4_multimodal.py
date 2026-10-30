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
"""Unit + E2E tests for the Gemma4 multimodal model components.

Tier 0/1 (no LLM_MODELS_ROOT needed, GPU only):
  - ``TestGemma4VisionTower`` — refactored TRT-LLM ``Gemma4VisionModel``
    (Option B: ``Gemma4VisionAttention`` inherits ``Attention`` base, vision
    runs through the TRTLLM attention backend, weights load via fused
    ``qkv_proj`` + clamp-buffer remap).
  - ``TestGemma4MultimodalEmbedder`` — RMSNorm + projection parity vs HF.
  - ``TestGemma4ForConditionalGeneration`` — wrapper instantiation; verifies
    that ``vision_tower`` is the refactored class (not HF ``AutoModel``).

Tier 2 (LLM_MODELS_ROOT gated, real tokenizer/processor):
  - ``TestGemma4InputProcessor`` — image/audio/video pre-processing.
  - ``TestGemma4MultimodalE2E`` — full LLM+vision context/decode execution via the
    shared ``TestModelingMultimodal`` skeleton (image / multi-image modalities).

Audio-side tests (``TestGemma4InputProcessor.test_audio_*``) are intentionally
left unchanged — audio tower refactor (Conformer migration) is a follow-up.
"""

from __future__ import annotations

import os
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Type
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

# Gemma4 requires transformers>=5.5.0 (native Gemma4 config/model classes).
pytest.importorskip(
    "transformers", minversion="5.5.0", reason="Gemma4 requires transformers>=5.5.0"
)

from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig  # noqa: E402
from transformers import Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4MultimodalEmbedder as HFGemma4MultimodalEmbedder,
)
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4VisionModel as HFGemma4VisionModel,
)
from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor  # noqa: E402

from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_gemma4_unified import (  # noqa: E402
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedImageProcessor,
    Gemma4UnifiedInputProcessor,
    Gemma4UnifiedProcessorShim,
)
from tensorrt_llm._torch.models.modeling_gemma4_vision import Gemma4VisionModel  # noqa: E402
from tensorrt_llm._torch.models.modeling_gemma4mm import (  # noqa: E402
    Gemma4ForConditionalGeneration,
    Gemma4InputProcessor,
    Gemma4MultimodalEmbedder,
    Gemma4MultimodalModelBase,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin  # noqa: E402
from tensorrt_llm._torch.models.modeling_multimodal_utils import (  # noqa: E402
    VisionNormalization,
    find_input_mm_embeds,
    get_multimodal_embeddings,
    vision_normalization_from_processor,
)
from tensorrt_llm.inputs.multimodal import (  # noqa: E402
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalInput,
    MultimodalItemMetadata,
    MultimodalParams,
    MultimodalRuntimeData,
)
from tensorrt_llm.llmapi.llm_args import MultimodalConfig  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

# ---------------------------------------------------------------------------
# Small configs for unit-level tests
# ---------------------------------------------------------------------------

# Real Gemma4 vision dims scaled down for speed. ``head_dim`` is the floor
# imposed by MMHA (kernel rejects head_dim < 32). Keep hidden_size = heads *
# head_dim so the fused ``qkv_proj`` shape stays consistent.
SMALL_VISION_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 1024,
    "use_clipped_linears": False,
    "standardize": True,
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


class _Gemma4EncoderCacheHarness(Gemma4MultimodalModelBase):
    """Lightweight Gemma4 encoder-cache harness without model weights."""

    supports_encoder_cache = True
    encode_multimodal_inputs = Gemma4ForConditionalGeneration.encode_multimodal_inputs

    def __init__(self, embedding_dim: int = 12) -> None:
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=4096)
        )
        self._embedding_dim = embedding_dim
        self.encoder_calls = 0
        self.audio_tower = None
        self.embed_audio = object()
        self._initialize_multimodal_encoder_cache(4096)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def embedding_dtype(self) -> torch.dtype:
        return torch.float32

    def _get_image_features(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        self.encoder_calls += 1
        return torch.full(
            (pixel_values.shape[0] * 2, self.embedding_dim),
            float(self.encoder_calls),
            dtype=self.embedding_dtype,
        )

    def _get_audio_features(
        self, audio_features: torch.Tensor, audio_features_mask: torch.Tensor | None
    ) -> torch.Tensor:
        self.encoder_calls += 1
        return torch.full(
            (audio_features.shape[0] * 2, self.embedding_dim),
            float(self.encoder_calls),
            dtype=self.embedding_dtype,
        )


def _make_keyed_image_param(
    item_hashes: list[list[int]] | None = None,
) -> MultimodalParams:
    if item_hashes is None:
        item_hashes = [[1, 2, 3, 4, 5, 6, 7, 8]]
    item_count = len(item_hashes)
    embedding_lengths = [2] * item_count
    pixel_values = torch.arange(item_count, dtype=torch.float32).reshape(item_count, 1, 1)
    image_position_ids = torch.arange(item_count * 2).reshape(item_count, 1, 2)
    return MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=item_hashes,
            multimodal_positions=[0] * item_count,
            multimodal_lengths=embedding_lengths,
        ),
        multimodal_data={
            "image": {
                "pixel_values": pixel_values,
                "image_position_ids": image_position_ids,
                "image_seq_lens": [1] * item_count,
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("image", i) for i in range(len(embedding_lengths))],
                output_embedding_lengths=embedding_lengths,
            ),
            "mm_processor_kwargs_hash": "kwargs-a",
        },
        multimodal_runtime=MultimodalRuntimeData(
            embed_mask_cumsum=torch.arange(1, sum(embedding_lengths) + 1, dtype=torch.int64),
            past_seen_token_num=0,
            chunk_end_pos=sum(embedding_lengths),
        ),
    )


def _make_keyed_video_param(item_hashes: list[list[int]]) -> MultimodalParams:
    item_count = len(item_hashes)
    embedding_lengths = [2] * item_count
    return MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=item_hashes,
            multimodal_positions=[0] * item_count,
            multimodal_lengths=embedding_lengths,
        ),
        multimodal_data={
            "video": {"pixel_values": torch.arange(item_count).reshape(item_count, 1, 1)},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("video", i) for i in range(len(embedding_lengths))],
                output_embedding_lengths=embedding_lengths,
            ),
            "mm_processor_kwargs_hash": "kwargs-a",
        },
        multimodal_runtime=MultimodalRuntimeData(
            embed_mask_cumsum=torch.arange(1, sum(embedding_lengths) + 1, dtype=torch.int64),
            past_seen_token_num=0,
            chunk_end_pos=sum(embedding_lengths),
        ),
    )


def _make_keyed_audio_param(item_hashes: list[list[int]]) -> MultimodalParams:
    item_count = len(item_hashes)
    embedding_lengths = [2] * item_count
    return MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=item_hashes,
            multimodal_positions=[0] * item_count,
            multimodal_lengths=embedding_lengths,
        ),
        multimodal_data={
            "audio": {
                "audio_features": torch.arange(item_count * 2).reshape(item_count, 1, 2),
                "audio_features_mask": torch.ones(item_count, 1),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("audio", i) for i in range(len(embedding_lengths))],
                output_embedding_lengths=embedding_lengths,
            ),
            "mm_processor_kwargs_hash": "kwargs-a",
        },
        multimodal_runtime=MultimodalRuntimeData(
            embed_mask_cumsum=torch.arange(1, sum(embedding_lengths) + 1, dtype=torch.int64),
            past_seen_token_num=0,
            chunk_end_pos=sum(embedding_lengths),
        ),
    )


def _make_item_input_processor(
    *,
    image_soft_tokens: int = 280,
    video_soft_tokens: int = 70,
    num_frames: int = 32,
    audio_seq_length: int | None = None,
) -> Gemma4InputProcessor:
    """Build a Gemma4 input processor from HF processor configs alone, without model files."""
    processor = Gemma4InputProcessor.__new__(Gemma4InputProcessor)
    processor._config = SimpleNamespace(
        vision_config=SimpleNamespace(pooling_kernel_size=3),
        audio_config=None if audio_seq_length is None else SimpleNamespace(),
    )
    processor._image_processor = Gemma4ImageProcessor(max_soft_tokens=image_soft_tokens)
    processor._processor = SimpleNamespace(
        image_processor=processor._image_processor,
        video_processor=Gemma4VideoProcessor(
            max_soft_tokens=video_soft_tokens, num_frames=num_frames
        ),
        audio_seq_length=audio_seq_length,
        feature_extractor=SimpleNamespace(feature_size=128),
    )
    processor._dtype = torch.bfloat16
    return processor


def _with_item_metadata(multimodal_data: dict, processor: Gemma4InputProcessor) -> MultimodalParams:
    """Attach encoder item metadata the way the input-processor wrapper does."""
    metadata = processor.get_mm_item_metadata([], multimodal_data)
    multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    return MultimodalParams(multimodal_data=multimodal_data)


def _make_normalization_processor(image_processor, dtype=torch.bfloat16, *, manual=False):
    processor = Gemma4InputProcessor.__new__(Gemma4InputProcessor)
    processor._image_processor = image_processor
    processor._processor = None
    processor._dtype = dtype
    processor._vision_gpu_normalize = True
    processor._tokenizer = lambda *args, **kwargs: {"input_ids": torch.tensor([[1]])}
    if image_processor is not None and not manual:

        def process(*, text, images, return_tensors, **kwargs):
            output = image_processor(images=images, return_tensors=return_tensors, **kwargs)
            output["input_ids"] = torch.tensor([[1]])
            return output

        processor._processor = process
    return processor


def _encode_normalization_buckets(buckets, modality="image"):
    model = SimpleNamespace(
        _get_image_features=lambda **kwargs: kwargs["pixel_values"].flatten(0, 1),
        embed_audio=None,
    )
    return Gemma4MultimodalModelBase.encode_multimodal_inputs(
        model, [MultimodalParams(multimodal_data={modality: bucket}) for bucket in buckets]
    )


def _make_vision_normalization() -> VisionNormalization:
    normalization = vision_normalization_from_processor(
        [Image.new("RGB", (1, 1))], Gemma4ImageProcessor(), torch.float32
    )
    assert normalization is not None
    return normalization


@pytest.mark.parametrize("device", [pytest.param("cpu", marks=pytest.mark.cpu_only), "cuda"])
@pytest.mark.parametrize("producer_dtype", [torch.float32, torch.float64])
def test_gemma4_vision_gpu_normalize_unified_shim(device, producer_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for device normalization exactness")
    image_processor = Gemma4UnifiedImageProcessor(
        max_soft_tokens=70,
        do_resize=False,
        do_normalize=True,
        image_mean=[0.2, 0.4, 0.6],
        image_std=[0.1, 0.3, 0.7],
    )
    tokenizer = mock.Mock(return_value={"input_ids": torch.tensor([[1]])})
    tokenizer.convert_tokens_to_ids.return_value = 1
    processor = _make_normalization_processor(image_processor)
    processor._processor = Gemma4UnifiedProcessorShim(
        feature_extractor=None,
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=None,
    )
    values = np.arange(48 * 48 * 3, dtype=np.uint16).reshape(48, 48, 3) % 256
    inputs = {
        "prompt": "<|image|>",
        "multi_modal_data": {"image": [Image.fromarray(values.astype(np.uint8))]},
    }
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(producer_dtype)
        with mock.patch.object(
            image_processor, "preprocess", wraps=image_processor.preprocess
        ) as preprocess:
            _, output = processor.call_with_text_prompt(inputs, None)
        assert preprocess.call_count == 1
        processor._vision_gpu_normalize = False
        _, baseline = processor.call_with_text_prompt(inputs, None)
    finally:
        torch.set_default_dtype(previous_dtype)
    bucket = output["multimodal_data"]["image"]
    assert bucket["pixel_values"].dtype == torch.uint8
    assert "vision_normalization" in bucket
    expected = baseline["multimodal_data"]["image"]["pixel_values"]
    bucket = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in bucket.items()
    }
    actual = _encode_normalization_buckets([bucket]).reshape(expected.shape).cpu()
    assert torch.equal(actual, expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("processor_class", [Gemma4ImageProcessor, Gemma4UnifiedImageProcessor])
@pytest.mark.parametrize("case", ["off", "gray", "rgba", "float", "tensor", "kwargs", "noop"])
def test_gemma4_vision_gpu_normalize_fallback(processor_class, case):
    image_processor = processor_class(max_soft_tokens=70, do_resize=False)
    processor = _make_normalization_processor(image_processor)
    image = Image.new("RGB", (48, 48))
    inputs = {"prompt": "image", "multi_modal_data": {"image": [image]}}
    if case == "off":
        processor._vision_gpu_normalize = False
    elif case in ("gray", "rgba"):
        inputs["multi_modal_data"]["image"] = [image.convert("L" if case == "gray" else "RGBA")]
    elif case == "float":
        inputs["multi_modal_data"]["image"] = [np.zeros((48, 48, 3), dtype=np.float32)]
    elif case == "tensor":
        inputs["multi_modal_data"]["image"] = [torch.zeros(3, 48, 48, dtype=torch.uint8)]
    elif case == "kwargs":
        inputs["mm_processor_kwargs"] = {"image_mean": [0.5] * 3}
    elif case == "noop":
        image_processor.do_rescale = image_processor.do_normalize = False
    _, output = processor.call_with_text_prompt(inputs, None)
    bucket = output["multimodal_data"]["image"]
    assert "vision_normalization" not in bucket
    processor._vision_gpu_normalize = False
    _, baseline = processor.call_with_text_prompt(inputs, None)
    assert torch.equal(bucket["pixel_values"], baseline["multimodal_data"]["image"]["pixel_values"])


@pytest.mark.cpu_only
@pytest.mark.parametrize("processor_class", [Gemma4ImageProcessor, Gemma4UnifiedImageProcessor])
def test_gemma4_vision_gpu_normalize_mixed_fallback(processor_class):
    image_processor = processor_class(max_soft_tokens=70, do_resize=False)
    processor = _make_normalization_processor(image_processor)
    # An empty video list exercises the existing mixed preprocessing path without frame decoding.
    inputs = {
        "prompt": "image and video",
        "multi_modal_data": {"image": [Image.new("RGB", (48, 48))], "video": []},
    }
    _, output = processor.call_with_text_prompt(inputs, None)
    assert "vision_normalization" not in output["multimodal_data"]["image"]
    assert Gemma4ForConditionalGeneration.supports_vision_gpu_normalize
    assert Gemma4UnifiedForConditionalGeneration.supports_vision_gpu_normalize


@pytest.mark.cpu_only
@pytest.mark.parametrize("processor_class", [Gemma4ImageProcessor, Gemma4UnifiedImageProcessor])
def test_gemma4_vision_gpu_normalize_manual_fallback(processor_class):
    processor = _make_normalization_processor(
        processor_class(max_soft_tokens=70, do_resize=False), manual=True
    )
    inputs = {"prompt": "image", "multi_modal_data": {"image": [Image.new("RGB", (48, 48))]}}
    _, output = processor.call_with_text_prompt(inputs, None)
    bucket = output["multimodal_data"]["image"]
    assert bucket["pixel_values"].dtype == torch.uint8
    processor._vision_gpu_normalize = False
    _, baseline = processor.call_with_text_prompt(inputs, None)
    expected = baseline["multimodal_data"]["image"]["pixel_values"]
    assert torch.equal(_encode_normalization_buckets([bucket]).reshape(expected.shape), expected)


@pytest.mark.parametrize("device", [pytest.param("cpu", marks=pytest.mark.cpu_only), "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("producer_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("processor_class", [Gemma4ImageProcessor, Gemma4UnifiedImageProcessor])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "do_normalize": True,
            "image_mean": [0.2, 0.4, 0.6],
            "image_std": [0.1, 0.3, 0.7],
            "rescale_factor": 1 / 127,
        },
        {
            "do_rescale": False,
            "do_normalize": True,
            "image_mean": [1.0, 2.0, 3.0],
            "image_std": [2.0, 3.0, 4.0],
        },
    ],
)
def test_gemma4_vision_gpu_normalize_all_uint8(
    processor_class, settings, dtype, device, producer_dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for device normalization exactness")
    image_processor = processor_class(max_soft_tokens=70, do_resize=False, **settings)
    processor = _make_normalization_processor(image_processor, dtype)
    values = np.arange(48 * 96, dtype=np.uint16).reshape(48, 96) % 256
    image = Image.fromarray(
        np.stack([values, (values + 85) % 256, (values + 171) % 256], -1).astype(np.uint8)
    )
    inputs = {"prompt": "image", "multi_modal_data": {"image": [image]}}
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(producer_dtype)
        _, output = processor.call_with_text_prompt(inputs, None)
        processor._vision_gpu_normalize = False
        _, baseline = processor.call_with_text_prompt(inputs, None)
    finally:
        torch.set_default_dtype(previous_dtype)
    bucket = output["multimodal_data"]["image"]
    assert bucket["pixel_values"].dtype == torch.uint8
    assert "vision_normalization" in bucket
    expected = baseline["multimodal_data"]["image"]["pixel_values"]
    bucket = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in bucket.items()
    }
    actual = _encode_normalization_buckets([bucket]).reshape(expected.shape).cpu()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    # Raw and already-normalized requests can share an encoder call.
    cpu_bucket = {"pixel_values": expected.to(device)}
    mixed = _encode_normalization_buckets([bucket, cpu_bucket]).cpu()
    assert torch.equal(mixed, torch.cat([expected, expected]).flatten(0, 1))


@pytest.mark.parametrize("device", [pytest.param("cpu", marks=pytest.mark.cpu_only), "cuda"])
@pytest.mark.parametrize("processor_class", [Gemma4ImageProcessor, Gemma4UnifiedImageProcessor])
def test_gemma4_vision_gpu_normalize_real_image(processor_class, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for device normalization exactness")
    path = Path(
        os.environ.get(
            "TRTLLM_TEST_IMAGE",
            Path(__file__).parents[3] / "integration/test_input_files/merlion.png",
        )
    )
    if not path.is_file() or path.read_bytes().startswith(
        b"version https://git-lfs.github.com/spec/"
    ):
        pytest.skip("Real-image fixture is not materialized")
    processor = _make_normalization_processor(processor_class(max_soft_tokens=70))
    with Image.open(path) as source:
        image = source.convert("RGB")
    inputs = {"prompt": "image", "multi_modal_data": {"image": [image]}}
    _, output = processor.call_with_text_prompt(inputs, None)
    bucket = output["multimodal_data"]["image"]
    processor._vision_gpu_normalize = False
    _, baseline = processor.call_with_text_prompt(inputs, None)
    expected = baseline["multimodal_data"]["image"]["pixel_values"]
    bucket = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in bucket.items()
    }
    actual = _encode_normalization_buckets([bucket]).reshape(expected.shape).cpu()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", [pytest.param("cpu", marks=pytest.mark.cpu_only), "cuda"])
@pytest.mark.parametrize("producer_dtype", [torch.float32, torch.float64])
def test_gemma4_vision_gpu_normalize_video_uint8(device, producer_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for device normalization exactness")
    from tensorrt_llm._torch.models.modeling_gemma4_unified import Gemma4UnifiedVideoProcessor

    values = torch.arange(3 * 48 * 96, dtype=torch.int64).reshape(3, 48, 96).remainder(256).byte()
    for video_class in (Gemma4VideoProcessor, Gemma4UnifiedVideoProcessor):
        video_processor = video_class(
            max_soft_tokens=70,
            do_resize=False,
            do_normalize=True,
            image_mean=[0.2, 0.4, 0.6],
            image_std=[0.1, 0.3, 0.7],
        )
        frames = [Image.fromarray(values.permute(1, 2, 0).numpy())] * 2
        processor = _make_normalization_processor(None)
        processor._processor = SimpleNamespace(video_processor=video_processor)
        inputs = {"prompt": "<|video|>", "multi_modal_data": {"video": [frames]}}
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(producer_dtype)
            _, raw = processor.call_with_text_prompt(inputs, None)
            processor._vision_gpu_normalize = False
            _, baseline = processor.call_with_text_prompt(inputs, None)
        finally:
            torch.set_default_dtype(previous_dtype)
        bucket = raw["multimodal_data"]["video"]
        assert bucket["pixel_values"].dtype == torch.uint8
        assert "vision_normalization" in bucket
        expected = baseline["multimodal_data"]["video"]["pixel_values"]
        bucket = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in bucket.items()
        }
        actual = (
            _encode_normalization_buckets([bucket], modality="video").reshape(expected.shape).cpu()
        )
        assert torch.equal(actual, expected)


# Mirror the engine's ``encoder_max_num_tokens`` runtime budget. The encoder
# ``AttentionMetadata`` is sized once at load to this maximum; each forward
# re-preps it with the real per-image sequence lengths.
_ENCODER_TEST_MAX_NUM_TOKENS = 8192


def _make_dummy_pixel_input(vision_cfg, B=1, side=6, dtype=torch.float32, device="cuda"):
    """Match the smoke ``gemma4_vision_smoke.py:make_dummy_input`` contract.

    Returns ``(pixel_values, pixel_position_ids, output_length)`` where
    ``output_length = side**2 // pooling_kernel_size**2``.
    """
    N = side * side
    C = vision_cfg.patch_size**2 * 3
    pixel_values = torch.randn(B, N, C, device=device, dtype=dtype)
    pos = (
        torch.stack(
            torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(1, -1, 2)
        .expand(B, -1, -1)
        .contiguous()
    )
    output_length = N // (vision_cfg.pooling_kernel_size**2)
    return pixel_values, pos, output_length


def _build_trt_vision_tower(vision_cfg, dtype=torch.float32, device="cuda"):
    """Build the refactored TRT-LLM ``Gemma4VisionModel`` with the TRTLLM attn backend.

    ``get_sub_model_config`` in ``modeling_gemma4mm.py`` always picks
    ``attn_backend="TRTLLM"`` for the vision tower, so unit tests mirror that.
    """
    mc = ModelConfig(
        pretrained_config=vision_cfg,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        attn_backend="TRTLLM",
    )
    tower = Gemma4VisionModel(mc).to(device).to(dtype).eval()
    # The engine builds the encoder AttentionMetadata after model load via
    # `setup_mm_encoder_attn_metadata`; standalone tests must mirror
    # that before the encoder forward.
    tower.setup_attn_metadata(max_num_tokens=_ENCODER_TEST_MAX_NUM_TOKENS)
    return tower


# ---------------------------------------------------------------------------
# Tier 0/1 — refactored vision tower
# ---------------------------------------------------------------------------


class TestGemma4VisionTower(unittest.TestCase):
    """Refactored ``Gemma4VisionModel`` (Option B) — Attention base class path.

    Covers:
      - build via ``ModelConfig`` with TRTLLM backend
      - random-weight forward (shape + NaN check)
      - HF state_dict → ``load_weights`` weight fusion (q/k/v → qkv_proj)
      - HF ↔ TRT forward parity (no clamp)
    """

    def test_build_from_model_config(self):
        """``Gemma4VisionModel(ModelConfig)`` exposes the expected submodules."""
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        tower = _build_trt_vision_tower(vision_cfg, dtype=torch.bfloat16)

        self.assertIsNotNone(tower.patch_embedder)
        self.assertIsNotNone(tower.encoder)
        self.assertIsNotNone(tower.pooler)
        self.assertEqual(len(tower.encoder.layers), vision_cfg.num_hidden_layers)
        # Standardize buffers must register when standardize=True (HF parity).
        self.assertTrue(hasattr(tower, "std_bias"))
        self.assertTrue(hasattr(tower, "std_scale"))

    @torch.no_grad()
    def test_forward_random_weights(self):
        """Forward smoke test with HF-default-init weights.

        Production loads weights via ``load_weights(hf_state_dict)``, so
        seeding through an HF tower's ``state_dict`` is the right sanity
        check. PyTorch-default init on TRT-LLM ``Linear`` / fused ``qkv_proj``
        can saturate the TRTLLM attention kernel at this small head_dim and
        produce NaN — that is a kernel-input contract, not a model bug, and
        is already covered indirectly by the parity tests below.
        """
        torch.manual_seed(0)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        tower = _build_trt_vision_tower(vision_cfg, dtype=torch.float32)
        tower.load_weights(dict(hf_tower.state_dict()))

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg, dtype=torch.float32)
        with torch.inference_mode():
            out = tower(pv, pos, output_length=output_length).last_hidden_state

        self.assertEqual(out.shape, torch.Size([output_length, vision_cfg.hidden_size]))
        self.assertFalse(out.isnan().any(), "Vision tower output contains NaN")

    @torch.no_grad()
    def test_load_weights_fuses_qkv(self):
        """HF ``q_proj / k_proj / v_proj`` collapse into the fused ``qkv_proj``.

        Validates the ``_load_weights_impl`` ``params_map`` regex
        (``self_attn.q_proj.linear.weight`` → ``self_attn.q_proj.weight``) plus
        the built-in qkv fusion (``qkv_proj`` ← stack(q,k,v)).

        When ``head_dim`` is padded to an FMHA-supported size (e.g. 32 → 64,
        72 → 80), the fused ``qkv_proj.weight`` has zero-padded tail channels
        per head; comparison must apply the same zero-pad to the HF reference.
        """
        torch.manual_seed(0)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)

        trt_tower.load_weights(dict(hf_tower.state_dict()))

        # Spot-check: fused qkv_proj.weight == vstack(zero-pad(q,k,v)) from HF.
        hf_sd = hf_tower.state_dict()
        for i in range(vision_cfg.num_hidden_layers):
            attn = trt_tower.encoder.layers[i].self_attn
            nh = attn.num_heads
            nkv = attn.num_key_value_heads
            hf_hd = attn.hf_head_dim
            padded_hd = attn.head_dim

            def _pad_head_dim(t: torch.Tensor, heads: int) -> torch.Tensor:
                # t: (heads * hf_hd, hidden) → (heads * padded_hd, hidden)
                if padded_hd == hf_hd:
                    return t
                t = t.view(heads, hf_hd, -1)
                zeros = t.new_zeros(heads, padded_hd - hf_hd, t.shape[-1])
                return torch.cat([t, zeros], dim=1).reshape(heads * padded_hd, -1)

            q = _pad_head_dim(hf_sd[f"encoder.layers.{i}.self_attn.q_proj.linear.weight"], nh)
            k = _pad_head_dim(hf_sd[f"encoder.layers.{i}.self_attn.k_proj.linear.weight"], nkv)
            v = _pad_head_dim(hf_sd[f"encoder.layers.{i}.self_attn.v_proj.linear.weight"], nkv)
            expected = torch.cat([q, k, v], dim=0)
            got = trt_tower.encoder.layers[i].self_attn.qkv_proj.weight.data
            torch.testing.assert_close(got.float(), expected.float(), atol=0, rtol=0)

    @torch.no_grad()
    def test_forward_parity_no_clamp(self):
        """HF ↔ TRT forward parity with ``use_clipped_linears=False``.

        Same shape/tolerance contract as ``gemma4_vision_smoke.py``
        Tier 1 (atol/rtol=5e-2) — the smoke at real dims hit
        diff.mean≈0.009; small dims should stay well inside.
        """
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = False
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg)
        with torch.inference_mode():
            hf_out = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            trt_out = trt_tower(pv, pos, output_length=output_length).last_hidden_state

        hf_flat = hf_out.reshape(-1, vision_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        atol, rtol = 5e-2, 5e-2
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=atol, rtol=rtol),
            f"max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )

    @torch.no_grad()
    def test_forward_batched_matches_per_image_loop(self):
        """B>1 single-call equivalence to per-image looped calls.

        The caller ``_get_image_features`` was rewritten to feed all images
        in one batched ``vision_tower`` call (varlen ``cu_seqlens`` attention).
        This regression test guards that the batched path is numerically
        equivalent to looping ``B=1`` calls and concatenating. If a future
        change introduces cross-image leakage (e.g. shared attn_metadata
        state not properly reset, or pooler bleeding across images), the
        loop-vs-batched diff catches it.
        """
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = False
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        B = 3
        pv_batched, pos_batched, output_length = _make_dummy_pixel_input(vision_cfg, B=B)

        # Path 1 (legacy): per-image loop, B=1 each, then concat.
        per_image = []
        with torch.inference_mode():
            for i in range(B):
                out_i = trt_tower(
                    pv_batched[i : i + 1],
                    pos_batched[i : i + 1],
                    output_length=output_length,
                ).last_hidden_state
                per_image.append(out_i)
        looped = torch.cat(per_image, dim=0)

        # Path 2 (new): single batched call.
        with torch.inference_mode():
            batched = trt_tower(
                pv_batched, pos_batched, output_length=output_length
            ).last_hidden_state

        # Shape must match (both flat, N_valid_total = B * output_length when
        # all images have full validity, as in this dummy input).
        self.assertEqual(looped.shape, batched.shape)
        # Values should be byte-equivalent (deterministic FULL attention on
        # both paths, no cross-image bleed). Allow a small atol for tiny
        # kernel-order tile reshuffles, but no rtol slack.
        diff = (looped - batched).abs()
        self.assertTrue(
            torch.allclose(looped, batched, atol=1e-4, rtol=0),
            f"Cross-image batched call diverges from per-image loop: "
            f"max diff={diff.max().item():.6f} mean diff={diff.mean().item():.6f}",
        )

    @torch.no_grad()
    def test_forward_image_seq_lens_kwarg_matches_default(self):
        """``image_seq_lens`` kwarg path is byte-equivalent to the default.

        The tower derives per-image seq_lens entirely on CPU — no GPU
        reduction, no D2H sync. Two CPU branches:
          - ``image_seq_lens`` is provided (production: ``Gemma4InputProcessor``
            emits it alongside ``image_position_ids``)
          - ``image_seq_lens is None`` and the dummy/test input has no ``-1``
            padding → fall back to ``[N] * B`` (every patch valid)

        For a full-grid synthetic input the two branches must produce
        identical output, since the explicit list ``[N] * B`` is exactly
        what the default fallback computes.
        """
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = False
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        B = 3
        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg, B=B)
        full_seq_lens = [pos.shape[1]] * B

        with torch.inference_mode():
            default_path = trt_tower(pv, pos, output_length=output_length).last_hidden_state
            explicit_path = trt_tower(
                pv, pos, output_length=output_length, image_seq_lens=full_seq_lens
            ).last_hidden_state

        self.assertEqual(default_path.shape, explicit_path.shape)
        diff = (default_path - explicit_path).abs()
        self.assertTrue(
            torch.allclose(default_path, explicit_path, atol=1e-4, rtol=0),
            f"image_seq_lens kwarg path diverges from default-all-valid path: "
            f"max diff={diff.max().item():.6f}",
        )


class TestGemma4VisionTowerClamp(unittest.TestCase):
    """``use_clipped_linears=True`` — exercises the clamp-buffer remap path.

    HF lays out clamps as:
      ``encoder.layers.{i}.self_attn.{q,k,v}_proj.{input,output}_{min,max}``
      ``encoder.layers.{i}.self_attn.o_proj.{input,output}_{min,max}``
    TRT-LLM collapses to:
      ``encoder.layers.{i}.self_attn.qkv_input_{min,max}`` (shared, q/k/v must agree)
      ``encoder.layers.{i}.self_attn.{q,k,v}_output_{min,max}``
      ``encoder.layers.{i}.self_attn.o_{input,output}_{min,max}``
    """

    @torch.no_grad()
    def test_clamp_buffers_remap_and_parity(self):
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = True
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)

        # Inject finite, non-default clamps into HF state_dict so the remap
        # path has something concrete to copy (default is ±inf — a no-op).
        # Same value on q/k/v so the fusion assertion in
        # ``_remap_clamp_buffers`` is satisfied.
        hf_sd = dict(hf_tower.state_dict())
        QKV_LO, QKV_HI = -1.5, 1.5
        Q_OUT_LO, Q_OUT_HI = -2.0, 2.0
        K_OUT_LO, K_OUT_HI = -2.5, 2.5
        V_OUT_LO, V_OUT_HI = -3.0, 3.0
        O_IN_LO, O_IN_HI = -4.0, 4.0
        O_OUT_LO, O_OUT_HI = -5.0, 5.0
        for i in range(vision_cfg.num_hidden_layers):
            prefix = f"encoder.layers.{i}.self_attn"
            for sec in ("q", "k", "v"):
                hf_sd[f"{prefix}.{sec}_proj.input_min"] = torch.tensor(QKV_LO)
                hf_sd[f"{prefix}.{sec}_proj.input_max"] = torch.tensor(QKV_HI)
            hf_sd[f"{prefix}.q_proj.output_min"] = torch.tensor(Q_OUT_LO)
            hf_sd[f"{prefix}.q_proj.output_max"] = torch.tensor(Q_OUT_HI)
            hf_sd[f"{prefix}.k_proj.output_min"] = torch.tensor(K_OUT_LO)
            hf_sd[f"{prefix}.k_proj.output_max"] = torch.tensor(K_OUT_HI)
            hf_sd[f"{prefix}.v_proj.output_min"] = torch.tensor(V_OUT_LO)
            hf_sd[f"{prefix}.v_proj.output_max"] = torch.tensor(V_OUT_HI)
            hf_sd[f"{prefix}.o_proj.input_min"] = torch.tensor(O_IN_LO)
            hf_sd[f"{prefix}.o_proj.input_max"] = torch.tensor(O_IN_HI)
            hf_sd[f"{prefix}.o_proj.output_min"] = torch.tensor(O_OUT_LO)
            hf_sd[f"{prefix}.o_proj.output_max"] = torch.tensor(O_OUT_HI)

        # Mirror the injected values into HF buffers so HF and TRT both clamp.
        hf_tower.load_state_dict(hf_sd, strict=False)

        trt_tower.load_weights(hf_sd)

        # Buffers must now be live (non-inf) on TRT side.
        for i in range(vision_cfg.num_hidden_layers):
            attn = trt_tower.encoder.layers[i].self_attn
            self.assertAlmostEqual(attn.qkv_input_min.item(), QKV_LO, places=5)
            self.assertAlmostEqual(attn.qkv_input_max.item(), QKV_HI, places=5)
            self.assertAlmostEqual(attn.q_output_min.item(), Q_OUT_LO, places=5)
            self.assertAlmostEqual(attn.q_output_max.item(), Q_OUT_HI, places=5)
            self.assertAlmostEqual(attn.k_output_min.item(), K_OUT_LO, places=5)
            self.assertAlmostEqual(attn.k_output_max.item(), K_OUT_HI, places=5)
            self.assertAlmostEqual(attn.v_output_min.item(), V_OUT_LO, places=5)
            self.assertAlmostEqual(attn.v_output_max.item(), V_OUT_HI, places=5)
            self.assertAlmostEqual(attn.o_input_min.item(), O_IN_LO, places=5)
            self.assertAlmostEqual(attn.o_input_max.item(), O_IN_HI, places=5)
            self.assertAlmostEqual(attn.o_output_min.item(), O_OUT_LO, places=5)
            self.assertAlmostEqual(attn.o_output_max.item(), O_OUT_HI, places=5)

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg)
        with torch.inference_mode():
            hf_out = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            trt_out = trt_tower(pv, pos, output_length=output_length).last_hidden_state

        hf_flat = hf_out.reshape(-1, vision_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        atol, rtol = 5e-2, 5e-2
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=atol, rtol=rtol),
            f"clamped parity max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )

    def test_clamp_qkv_input_mismatch_raises(self):
        """q/k/v input clamps must agree — collapse must not silently lose info."""
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = True
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)

        hf_sd = dict(hf_tower.state_dict())
        # Set q/k/v input_min to *different* values for layer 0 — load must fail.
        hf_sd["encoder.layers.0.self_attn.q_proj.input_min"] = torch.tensor(-1.0)
        hf_sd["encoder.layers.0.self_attn.k_proj.input_min"] = torch.tensor(-1.1)
        hf_sd["encoder.layers.0.self_attn.v_proj.input_min"] = torch.tensor(-1.0)
        hf_sd["encoder.layers.0.self_attn.q_proj.input_max"] = torch.tensor(1.0)
        hf_sd["encoder.layers.0.self_attn.k_proj.input_max"] = torch.tensor(1.0)
        hf_sd["encoder.layers.0.self_attn.v_proj.input_max"] = torch.tensor(1.0)

        with self.assertRaises(ValueError):
            trt_tower.load_weights(hf_sd)


# ---------------------------------------------------------------------------
# Multimodal embedder (unchanged from skeleton — already covers the
# refactored ``Gemma4MultimodalEmbedder``).
# ---------------------------------------------------------------------------


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
        torch.nn.init.xavier_uniform_(embedder.embedding_projection.weight.data)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = embedder(x)
        self.assertEqual(out.shape, torch.Size([4, 128]))
        self.assertFalse(out.isnan().any())

    @torch.no_grad()
    def test_embedder_matches_hf(self):
        """Compare embedder output with HF Gemma4MultimodalEmbedder."""
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        dtype = torch.bfloat16
        device = "cuda"

        hf_embedder = HFGemma4MultimodalEmbedder(vision_cfg, text_cfg).to(dtype).to(device).eval()

        trt_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_cfg.hidden_size,
            eps=vision_cfg.rms_norm_eps,
            dtype=dtype,
        ).to(device)

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


# ---------------------------------------------------------------------------
# Vision + embedder end-to-end (no LLM, no real model required)
# ---------------------------------------------------------------------------


class TestGemma4VisionPipeline(unittest.TestCase):
    """Full vision-pipeline parity: pixel_values → tower → embedder → text_hidden.

    Replaces the pre-refactor ``test_vision_pipeline_matches_hf`` which built
    the tower via ``AutoModel.from_config`` and the projection via
    ``load_state_dict``. Both sides now use the refactored TRT-LLM tower +
    embedder; weights flow through the production ``load_weights`` path.
    """

    @torch.no_grad()
    def test_pipeline_matches_hf(self):
        torch.manual_seed(0)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        dtype = torch.float32
        device = "cuda"

        # --- HF pipeline ---
        hf_tower = HFGemma4VisionModel(vision_cfg).to(device).to(dtype).eval()
        hf_embedder = HFGemma4MultimodalEmbedder(vision_cfg, text_cfg).to(device).to(dtype).eval()

        # --- TRT-LLM pipeline ---
        trt_tower = _build_trt_vision_tower(vision_cfg, dtype=dtype, device=device)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        trt_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_cfg.hidden_size,
            eps=vision_cfg.rms_norm_eps,
            dtype=dtype,
        ).to(device)
        trt_embedder.embedding_projection.weight.data.copy_(
            hf_embedder.embedding_projection.weight.data
        )

        pv, pos, output_length = _make_dummy_pixel_input(
            vision_cfg, B=1, side=6, dtype=dtype, device=device
        )

        with torch.inference_mode():
            hf_hidden = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            hf_out = hf_embedder(hf_hidden)
            trt_hidden = trt_tower(pv, pos, output_length=output_length).last_hidden_state
            # Production ``_get_image_features`` adds a batch dim around the
            # embedder call. Mirror that exactly so the projection's input
            # shape contract is what production sees.
            trt_out = trt_embedder(trt_hidden.unsqueeze(0)).squeeze(0)

        # HF returns (B, output_length, H); TRT returns (N_valid, H). For B=1
        # with no padding, N_valid == output_length, so the flattened shapes
        # must match.
        hf_flat = hf_out.reshape(-1, text_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=5e-2, rtol=5e-2),
            f"Pipeline max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )


# ---------------------------------------------------------------------------
# Multimodal wrapper (LLM + vision tower + embedder)
# ---------------------------------------------------------------------------


class TestGemma4ForConditionalGeneration(unittest.TestCase):
    """Test the multimodal VLM wrapper."""

    @staticmethod
    def _make_model() -> Gemma4ForConditionalGeneration:
        text_config = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        vision_config = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        config = Gemma4Config(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=None,
        )
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend="FLASHINFER",
        )
        return Gemma4ForConditionalGeneration(model_config)

    def test_instantiation_with_vision(self):
        """VLM wrapper creates LLM + vision tower + embedder.

        Asserts ``vision_tower`` is the refactored TRT-LLM ``Gemma4VisionModel``
        (Option B), *not* the HF ``AutoModel`` placeholder used pre-refactor.
        """
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
            attn_backend="TRTLLM",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIs(model.model_config.extra_attrs, mc.extra_attrs)
        self.assertIs(model.llm.model.model_config.extra_attrs, mc.extra_attrs)
        self.assertIsNotNone(model.llm)
        self.assertIsNotNone(model.vision_tower)
        self.assertIsNotNone(model.embed_vision)
        self.assertIsNone(model.audio_tower)
        self.assertIsNone(model.embed_audio)
        # Regression guard for Option B: the vision tower must be the native
        # TRT-LLM class, not ``transformers.AutoModel`` output.
        self.assertIsInstance(model.vision_tower, Gemma4VisionModel)

    def test_text_submodel_preserves_explicit_attention_backend(self):
        """The text submodel honors explicit backend overrides."""
        text_config = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        vision_config = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        config = Gemma4Config(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=None,
        )

        for backend in ("FLASHINFER", "TRTLLM"):
            model_config = ModelConfig(
                pretrained_config=config,
                mapping=Mapping(world_size=1, tp_size=1, rank=0),
                attn_backend=backend,
            )
            text_model_config = Gemma4ForConditionalGeneration.get_sub_model_config(
                model_config, "text_config"
            )
            vision_model_config = Gemma4ForConditionalGeneration.get_sub_model_config(
                model_config, "vision_config"
            )

            self.assertEqual(text_model_config.attn_backend, backend)
            self.assertEqual(vision_model_config.attn_backend, "TRTLLM")

    def test_encoder_cache_reuses_image_embedding_across_requests(self):
        """Persistent cache reuse applies to the shared dense/MoE Gemma4 wrapper."""
        self.assertTrue(issubclass(Gemma4ForConditionalGeneration, MultimodalModelMixin))
        self.assertTrue(Gemma4ForConditionalGeneration.supports_encoder_cache)

        model = _Gemma4EncoderCacheHarness()
        first = model._get_or_encode_multimodal_embeddings([_make_keyed_image_param()])
        second = model._get_or_encode_multimodal_embeddings([_make_keyed_image_param()])

        self.assertEqual(model.encoder_calls, 1)
        torch.testing.assert_close(second, first)
        self.assertEqual(len(model._multimodal_encoder_cache), 1)

    @pytest.mark.cpu_only
    def test_encoder_cache_partial_hit_slices_gemma4_image_input(self):
        """A partial hit encodes only the missing Gemma4 image."""
        model = _Gemma4EncoderCacheHarness()
        shared_hash = [1] * 8
        normalization = _make_vision_normalization()
        for hashes in ([shared_hash], [shared_hash, [2] * 8]):
            param = _make_keyed_image_param(item_hashes=hashes)
            image = param.multimodal_data["image"]
            image["vision_normalization"] = normalization
            image["pixel_values"] = torch.arange(len(hashes) * 3, dtype=torch.uint8).reshape(
                len(hashes), 1, 3
            )
            with mock.patch.object(
                type(model), "encode_multimodal_inputs", wraps=model.encode_multimodal_inputs
            ) as encode:
                embeddings = model._get_or_encode_multimodal_embeddings([param])
            encoded_image = encode.call_args.args[0][0].multimodal_data["image"]
            self.assertIs(encoded_image["vision_normalization"], normalization)
            torch.testing.assert_close(encoded_image["pixel_values"], image["pixel_values"][-1:])

        self.assertEqual(model.encoder_calls, 2)
        torch.testing.assert_close(embeddings[:2], torch.ones(2, model.embedding_dim))
        torch.testing.assert_close(embeddings[2:], torch.full((2, model.embedding_dim), 2.0))
        self.assertEqual(len(model._multimodal_encoder_cache), 2)

    @pytest.mark.cpu_only
    def test_build_multimodal_encoder_input_slices_gemma4_image_metadata(self):
        """Gemma4 image position metadata follows the selected images."""
        param = _make_keyed_image_param(item_hashes=[[0] * 8, [1] * 8, [2] * 8])
        source_image = param.multimodal_data["image"]
        source_image["pixel_values"] = torch.arange(9, dtype=torch.uint8).reshape(3, 1, 3)
        source_image["vision_normalization"] = _make_vision_normalization()

        residual = _Gemma4EncoderCacheHarness().build_multimodal_encoder_input(param, [2, 0])

        residual_image = residual.multimodal_data["image"]
        torch.testing.assert_close(
            residual_image["pixel_values"], source_image["pixel_values"][[2, 0]]
        )
        torch.testing.assert_close(
            residual_image["image_position_ids"], source_image["image_position_ids"][[2, 0]]
        )
        self.assertEqual(residual_image["image_seq_lens"], [1, 1])
        self.assertIs(residual_image["vision_normalization"], source_image["vision_normalization"])

    def test_build_multimodal_encoder_input_slices_gemma4_audio_input(self):
        """Gemma4 audio features and their mask remain item-aligned."""
        audio_features = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        audio_mask = torch.arange(12).reshape(3, 4)
        shared_stats = torch.arange(3)
        param = MultimodalParams(
            multimodal_data={
                "audio": {
                    "audio_features": audio_features,
                    "audio_features_mask": audio_mask,
                    "shared_stats": shared_stats,
                },
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("audio", 0), ("audio", 1), ("audio", 2)], output_embedding_lengths=[1, 1, 1]
                ),
            }
        )

        residual = _Gemma4EncoderCacheHarness().build_multimodal_encoder_input(param, [2, 0])

        residual_audio = residual.multimodal_data["audio"]
        torch.testing.assert_close(residual_audio["audio_features"], audio_features[[2, 0]])
        torch.testing.assert_close(residual_audio["audio_features_mask"], audio_mask[[2, 0]])
        assert residual_audio["shared_stats"] is shared_stats

    @pytest.mark.cpu_only
    def test_build_multimodal_encoder_input_slices_whole_videos(self):
        """A video item is the run of flattened frames its `num_frames` entry declares."""
        param = MultimodalParams(
            multimodal_data={
                "video": {
                    "pixel_values": torch.arange(18, dtype=torch.uint8).reshape(6, 1, 3),
                    "image_position_ids": torch.arange(6).reshape(6, 1, 1).expand(6, 1, 2),
                    "image_seq_lens": [10, 11, 12, 13, 14, 15],
                    "num_frames": [2, 1, 3],
                    "vision_normalization": _make_vision_normalization(),
                },
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("video", 0), ("video", 1), ("video", 2)],
                    output_embedding_lengths=[1, 1, 1],
                ),
            }
        )

        residual = _Gemma4EncoderCacheHarness().build_multimodal_encoder_input(
            param, [2, 0], modality="video"
        )

        video = residual.multimodal_data["video"]
        torch.testing.assert_close(
            video["pixel_values"], param.multimodal_data["video"]["pixel_values"][[3, 4, 5, 0, 1]]
        )
        self.assertEqual(video["image_position_ids"][:, 0, 0].tolist(), [3, 4, 5, 0, 1])
        self.assertEqual(video["image_seq_lens"], [13, 14, 15, 10, 11])
        self.assertEqual(video["num_frames"], [3, 2])
        self.assertIs(
            video["vision_normalization"], param.multimodal_data["video"]["vision_normalization"]
        )

    def test_item_scheduling_encodes_selected_items_in_scheduler_order(self):
        """Selected items are sliced, encoded once per modality run, and split back per item."""

        class _ContentHarness(_Gemma4EncoderCacheHarness):
            def _get_image_features(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
                del kwargs
                self.encoder_calls += 1
                # Two rows per image or video frame, each holding its first pixel value.
                return pixel_values[:, 0, :1].repeat_interleave(2, dim=0)

        processor = _make_item_input_processor()
        # 18 valid patches pool into 2 soft tokens per image or frame.
        images = _with_item_metadata(
            {
                "image": {
                    "pixel_values": torch.arange(3.0).reshape(3, 1, 1).expand(3, 18, 1),
                    "image_seq_lens": [18] * 3,
                }
            },
            processor,
        )
        videos = _with_item_metadata(
            {
                "video": {
                    "pixel_values": torch.arange(10.0, 15.0).reshape(5, 1, 1).expand(5, 18, 1),
                    "image_seq_lens": [18] * 5,
                    "num_frames": [2, 3],
                }
            },
            processor,
        )
        model = _ContentHarness()

        outputs = model.forward_multimodal_encoder_items(
            model.prepare_multimodal_encoder_inputs([(images, 2), (images, 0), (videos, 1)])
        )

        self.assertEqual(
            [output.flatten().tolist() for output in outputs],
            [[2.0] * 2, [0.0] * 2, [12.0] * 2 + [13.0] * 2 + [14.0] * 2],
        )
        self.assertEqual(model.encoder_calls, 2)

    @torch.no_grad()
    def test_item_encoding_matches_full_request_encoding(self):
        """Encoding selected items reproduces their rows of a whole-request encode."""
        torch.manual_seed(0)
        model = self._make_model()
        vision_cfg = model._top_config.vision_config
        model.vision_tower.load_weights(dict(HFGemma4VisionModel(vision_cfg).state_dict()))
        torch.nn.init.normal_(model.embed_vision.embedding_projection.weight, std=0.02)
        model.vision_tower.setup_attn_metadata(max_num_tokens=_ENCODER_TEST_MAX_NUM_TOKENS)
        processor = _make_item_input_processor()

        pixel_values, position_ids, _ = _make_dummy_pixel_input(
            vision_cfg, B=5, side=6, dtype=torch.bfloat16
        )
        # Image or frame 1 keeps only the top half of its patches.
        position_ids = position_ids.clone()
        position_ids[1, 18:] = -1
        seq_lens = (position_ids != -1).all(dim=-1).sum(dim=-1).tolist()
        for modality, extra_fields in (("image", {}), ("video", {"num_frames": [2, 3]})):
            with self.subTest(modality=modality):
                param = _with_item_metadata(
                    {
                        modality: {
                            "pixel_values": pixel_values,
                            "image_position_ids": position_ids,
                            "image_seq_lens": seq_lens,
                            **extra_fields,
                        }
                    },
                    processor,
                )
                lengths = param.get_item_layout()[1]
                expected = torch.split(model.encode_multimodal_inputs([param]), lengths)
                selected = [len(lengths) - 1, 0]

                outputs = model.forward_multimodal_encoder_items(
                    model.prepare_multimodal_encoder_inputs(
                        [(param, item_idx) for item_idx in selected]
                    )
                )

                for item_idx, output in zip(selected, outputs, strict=True):
                    torch.testing.assert_close(output, expected[item_idx], atol=1e-2, rtol=1e-2)

    def test_encoder_cache_partial_hit_slices_gemma4_audio_input(self):
        """A partial hit encodes only the missing Gemma4 audio item."""
        model = _Gemma4EncoderCacheHarness()
        shared_hash = [1] * 8
        model._get_or_encode_multimodal_embeddings([_make_keyed_audio_param([shared_hash])])

        embeddings = model._get_or_encode_multimodal_embeddings(
            [_make_keyed_audio_param([shared_hash, [2] * 8])]
        )

        self.assertEqual(model.encoder_calls, 2)
        torch.testing.assert_close(embeddings[:2], torch.ones(2, model.embedding_dim))
        torch.testing.assert_close(embeddings[2:], torch.full((2, model.embedding_dim), 2.0))
        self.assertEqual(len(model._multimodal_encoder_cache), 2)

    @pytest.mark.cpu_only
    def test_build_multimodal_encoder_input_uses_typed_audio_item_count(self):
        features = torch.arange(24).reshape(3, 4, 2)
        param = MultimodalParams(
            multimodal_data={
                "audio": {"audio_features": features},
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("audio", 0), ("audio", 1), ("audio", 2)], [8, 8, 8], [1, 1, 1]
                ),
            }
        )

        residual = _Gemma4EncoderCacheHarness().build_multimodal_encoder_input(param, [2, 0])

        torch.testing.assert_close(
            residual.multimodal_data["audio"]["audio_features"], features[[2, 0]]
        )

    def test_encoder_cache_partial_hit_with_unsupported_item_axis_becomes_full_miss(self):
        """Unexpected Gemma4 image and audio layouts bypass partial cache reuse."""
        model = _Gemma4EncoderCacheHarness()
        shared_hash = [1] * 8
        cases = (
            ("image", "pixel_values", _make_keyed_image_param),
            ("audio", "audio_features", _make_keyed_audio_param),
        )

        for modality, input_key, make_param in cases:
            with self.subTest(modality=modality):
                model._get_or_encode_multimodal_embeddings([make_param([shared_hash])])
                param = make_param([shared_hash, [2] * 8])
                input_tensor = param.multimodal_data[modality][input_key]
                param.multimodal_data[modality][input_key] = torch.cat(
                    (input_tensor, input_tensor[:1]), dim=0
                )

                with unittest.mock.patch(
                    "tensorrt_llm._torch.models.modeling_gemma4mm.logger.warning_once"
                ) as warning_once:
                    partition = model.partition_encoder_cache(
                        param, model._multimodal_encoder_cache
                    )

                self.assertTrue(partition.is_full_miss)
                self.assertEqual(partition.hits, {})
                self.assertEqual(partition.miss_indices, [0, 1])
                warning_once.assert_called_once()

    def test_encoder_cache_reuses_full_video_hits_and_reencodes_partial_hits(self):
        """Gemma4 video reuses full hits and re-encodes complete partial-hit requests."""
        model = _Gemma4EncoderCacheHarness()
        shared_hash = [1] * 8
        first = model._get_or_encode_multimodal_embeddings([_make_keyed_video_param([shared_hash])])
        full_hit = model._get_or_encode_multimodal_embeddings(
            [_make_keyed_video_param([shared_hash])]
        )
        with unittest.mock.patch(
            "tensorrt_llm._torch.models.modeling_gemma4mm.logger.warning_once"
        ) as warning_once:
            partial_hit = model._get_or_encode_multimodal_embeddings(
                [_make_keyed_video_param([shared_hash, [2] * 8])]
            )
        repeated = model._get_or_encode_multimodal_embeddings(
            [_make_keyed_video_param([shared_hash, [2] * 8])]
        )

        warning_once.assert_called_once()
        self.assertEqual(model.encoder_calls, 2)
        torch.testing.assert_close(full_hit, first)
        torch.testing.assert_close(first, torch.ones(2, model.embedding_dim))
        torch.testing.assert_close(partial_hit, torch.full((4, model.embedding_dim), 2.0))
        torch.testing.assert_close(repeated, partial_hit)
        self.assertEqual(len(model._multimodal_encoder_cache), 2)

    def test_chunked_prefill_reuses_cached_vision_embeddings(self):
        """Later active chunks slice cached features without rerunning vision."""
        model = self._make_model()
        embed_mask_cumsum = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        multimodal_param = MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": torch.zeros(1, 1, 1, device="cuda"),
                }
            },
            multimodal_runtime=MultimodalRuntimeData(
                past_seen_token_num=0,
                chunk_end_pos=2,
                embed_mask_cumsum=embed_mask_cumsum,
            ),
        )
        expected_embeddings = torch.randn(4, SMALL_TEXT_CONFIG["hidden_size"], device="cuda")

        with unittest.mock.patch.object(
            model, "_get_image_features", return_value=expected_embeddings
        ) as image_encoder:
            all_embeddings = get_multimodal_embeddings(
                model.encode_multimodal_inputs, [multimodal_param]
            )
            first_chunk = find_input_mm_embeds(all_embeddings, [multimodal_param])

            multimodal_param.multimodal_runtime = MultimodalRuntimeData(
                past_seen_token_num=2,
                chunk_end_pos=4,
                embed_mask_cumsum=embed_mask_cumsum,
            )
            all_embeddings = get_multimodal_embeddings(
                model.encode_multimodal_inputs, [multimodal_param]
            )
            second_chunk = find_input_mm_embeds(all_embeddings, [multimodal_param])

        image_encoder.assert_called_once()
        torch.testing.assert_close(first_chunk[0], expected_embeddings[:2])
        torch.testing.assert_close(second_chunk[0], expected_embeddings[2:])

    def test_chunk_without_multimodal_tokens_is_inactive(self):
        """A text-only chunk does not schedule the multimodal encoder."""
        multimodal_param = MultimodalParams(
            multimodal_data={"image": {"pixel_values": torch.zeros(1, 1, 1)}},
            multimodal_runtime=MultimodalRuntimeData(
                past_seen_token_num=0,
                chunk_end_pos=2,
                embed_mask_cumsum=torch.tensor([0, 0, 1, 2], dtype=torch.int64),
            ),
        )
        self.assertEqual(
            Gemma4ForConditionalGeneration.select_multimodal_params(None, [multimodal_param], 1),
            [],
        )

    def test_mixed_modality_batch_preserves_request_order(self):
        """Cached mixed-modality embeddings retain their request order."""
        model = self._make_model()
        params = [
            MultimodalParams(
                multimodal_data={"video": {"pixel_values": torch.tensor([[[2.0]]])}},
                multimodal_runtime=MultimodalRuntimeData(
                    past_seen_token_num=0,
                    chunk_end_pos=1,
                    embed_mask_cumsum=torch.tensor([1], dtype=torch.int64),
                ),
            ),
            MultimodalParams(
                multimodal_data={"image": {"pixel_values": torch.tensor([[[1.0]]])}},
                multimodal_runtime=MultimodalRuntimeData(
                    past_seen_token_num=0,
                    chunk_end_pos=1,
                    embed_mask_cumsum=torch.tensor([1], dtype=torch.int64),
                ),
            ),
            MultimodalParams(
                multimodal_data={"video": {"pixel_values": torch.tensor([[[3.0]]])}},
                multimodal_runtime=MultimodalRuntimeData(
                    past_seen_token_num=0,
                    chunk_end_pos=1,
                    embed_mask_cumsum=torch.tensor([1], dtype=torch.int64),
                ),
            ),
        ]

        with unittest.mock.patch.object(
            model,
            "_get_image_features",
            side_effect=lambda pixel_values, **_: pixel_values[:, 0],
        ):
            embeddings = get_multimodal_embeddings(model.encode_multimodal_inputs, params)

        expected = torch.tensor([[2.0], [1.0], [3.0]])
        torch.testing.assert_close(embeddings[0], expected)
        for param, expected_embedding in zip(params, expected, strict=True):
            torch.testing.assert_close(
                param.multimodal_data["multimodal_embedding"], expected_embedding.unsqueeze(0)
            )

    def test_single_request_with_multiple_modalities_raises(self):
        """Mixed-modality requests are rejected until prompt-order handling is supported."""
        model = self._make_model()
        multimodal_param = MultimodalParams(
            multimodal_data={
                "image": {"pixel_values": torch.tensor([[[1.0]]])},
                "video": {"pixel_values": torch.tensor([[[2.0]]])},
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "requests containing multiple multimodal input types are not supported",
        ):
            model.encode_multimodal_inputs([multimodal_param])

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
            attn_backend="TRTLLM",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIsNotNone(model.llm)
        self.assertIsNone(model.vision_tower)
        self.assertIsNone(model.embed_vision)


# ---------------------------------------------------------------------------
# Encoder items for item scheduling and memory profiling (processor configs only)
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestGemma4EncoderItems(unittest.TestCase):
    """Item metadata, encoder budgets, and profiling dummies of the Gemma4 input processor."""

    def test_image_items_cost_padded_patches_and_yield_valid_soft_tokens(self):
        processor = _make_item_input_processor()
        position_ids = torch.zeros(2, 2520, 2, dtype=torch.long)
        position_ids[1, 1260:] = -1  # The second image fills half of its patch slots.

        metadata = processor.get_mm_item_metadata(
            [],
            {
                "image": {
                    "pixel_values": torch.zeros(2, 2520, 1),
                    "image_position_ids": position_ids,
                }
            },
        )

        self.assertEqual(list(metadata.item_refs), [("image", 0), ("image", 1)])
        self.assertEqual(list(metadata.encoder_token_lengths), [2520, 2520])
        self.assertEqual(list(metadata.output_embedding_lengths), [280, 140])

    def test_video_items_span_their_frames(self):
        processor = _make_item_input_processor()
        data = {
            "video": {
                "pixel_values": torch.zeros(5, 630, 1),
                "image_seq_lens": [630, 630, 315, 315, 315],
                "num_frames": [2, 3],
            }
        }
        metadata = processor.get_mm_item_metadata([], data)

        self.assertEqual(list(metadata.item_refs), [("video", 0), ("video", 1)])
        self.assertEqual(list(metadata.encoder_token_lengths), [1260, 1890])
        self.assertEqual(list(metadata.output_embedding_lengths), [140, 105])
        self.assertEqual(metadata.input_partitions["video"]["image_seq_lens"].item_lengths, (2, 3))
        self.assertEqual(
            metadata.input_partitions["video"]["image_seq_lens"].select(
                data["video"]["image_seq_lens"], [1, 0]
            ),
            [315, 315, 315, 630, 630],
        )
        reordered = processor.get_mm_item_metadata(
            [], data, [{"modality": "video", "index": 1}, {"modality": "video", "index": 0}]
        )
        self.assertEqual(reordered.item_refs, (("video", 1), ("video", 0)))
        self.assertEqual(reordered.encoder_token_lengths, (1890, 1260))
        self.assertEqual(reordered.output_embedding_lengths, (105, 140))

    def test_audio_items_count_subsampled_frames(self):
        """Each clip yields ceil(valid mel frames / 4) rows, the count HF's processor inserts."""
        processor = _make_item_input_processor(audio_seq_length=750)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0] = True
        mask[1, :5] = True

        metadata = processor.get_mm_item_metadata(
            [],
            {"audio": {"audio_features": torch.zeros(2, 10, 128), "audio_features_mask": mask}},
        )

        self.assertEqual(list(metadata.item_refs), [("audio", 0), ("audio", 1)])
        self.assertEqual(list(metadata.encoder_token_lengths), [3, 3])
        self.assertEqual(list(metadata.output_embedding_lengths), [3, 2])

    def test_mixed_modality_request_is_rejected(self):
        processor = _make_item_input_processor(audio_seq_length=750)
        with self.assertRaisesRegex(ValueError, "multiple multimodal input types"):
            processor.get_mm_item_metadata(
                [],
                {
                    "image": {"pixel_values": torch.zeros(1, 2520, 1)},
                    "audio": {"audio_features": torch.zeros(1, 10, 128)},
                },
            )

    def test_max_tokens_per_item(self):
        processor = _make_item_input_processor(image_soft_tokens=1120)
        self.assertEqual(processor.get_mm_max_tokens_per_item(), {"image": 10080, "video": 20160})
        # Under a budget, images that do not fit are left out and videos keep the frames that fit.
        self.assertEqual(
            processor.get_mm_max_tokens_per_item(max_num_encoder_tokens=8192), {"video": 13 * 630}
        )

        audio_processor = _make_item_input_processor(audio_seq_length=750)
        self.assertEqual(
            audio_processor.get_mm_max_tokens_per_item(),
            {"image": 2520, "video": 20160, "audio": 750},
        )

    def test_video_beyond_the_default_frame_count_fills_the_budget(self):
        """A budget above the default video admits longer videos, so the profiling dummy grows."""
        processor = _make_item_input_processor(num_frames=32)
        self.assertEqual(
            processor.get_mm_max_tokens_per_item(max_num_encoder_tokens=30000)["video"], 47 * 630
        )

        budget = 64 * 630
        self.assertEqual(
            processor.get_mm_max_tokens_per_item(max_num_encoder_tokens=budget)["video"], budget
        )
        dummy = processor.get_dummy_mm_data(max_num_encoder_tokens=budget, mm_counts={"video": 1})
        self.assertEqual(dummy["video"]["num_frames"], [64])
        metadata = processor.get_mm_item_metadata([], dummy)
        self.assertEqual(list(metadata.encoder_token_lengths), [budget])

    def test_encoder_iteration_bounds(self):
        processor = _make_item_input_processor()
        self.assertEqual(processor.get_max_mm_encoder_output_embeddings(20160), 2240)
        # Each image or video frame is one attention sequence of at least 630 padded patches.
        self.assertEqual(
            processor.get_mm_encoder_attention_metadata_capacity(20160), {"attention": 32}
        )

        audio_processor = _make_item_input_processor(audio_seq_length=750)
        self.assertEqual(audio_processor.get_max_mm_encoder_output_embeddings(20160), 20160)

    def test_dummy_items_have_the_largest_legal_size(self):
        processor = _make_item_input_processor(audio_seq_length=750)
        budget = 8192
        max_tokens_per_item = processor.get_mm_max_tokens_per_item(max_num_encoder_tokens=budget)
        self.assertEqual(set(max_tokens_per_item), {"image", "video", "audio"})

        for modality, item_tokens in max_tokens_per_item.items():
            with self.subTest(modality=modality):
                num_items = budget // item_tokens
                dummy = processor.get_dummy_mm_data(
                    max_num_encoder_tokens=budget, mm_counts={modality: num_items}
                )

                metadata = processor.get_mm_item_metadata([], dummy)

                self.assertEqual(list(metadata.encoder_token_lengths), [item_tokens] * num_items)
                # Every patch slot or mel frame holds valid input.
                rows_per_item = item_tokens if modality == "audio" else item_tokens // 9
                self.assertEqual(
                    list(metadata.output_embedding_lengths), [rows_per_item] * num_items
                )

    def test_dummy_rejects_items_beyond_the_budget(self):
        processor = _make_item_input_processor()
        with self.assertRaisesRegex(ValueError, "exceed"):
            processor.get_dummy_mm_data(max_num_encoder_tokens=5000, mm_counts={"image": 2})

    def test_dummy_vision_layout_matches_hf_processors(self):
        """A dummy image or frame has the patch layout HF gives a full-size image or frame."""
        from PIL import Image

        processor = _make_item_input_processor(image_soft_tokens=70)
        # 70 soft tokens lay out on a 7 x 10 grid: 21 x 30 patches of 16 px.
        hf_image = Gemma4ImageProcessor(max_soft_tokens=70)(
            images=[Image.new("RGB", (480, 336))], return_tensors="pt"
        )
        image = processor.get_dummy_mm_data(max_num_encoder_tokens=630, mm_counts={"image": 1})[
            "image"
        ]
        torch.testing.assert_close(image["image_position_ids"], hf_image["image_position_ids"])
        self.assertEqual(image["pixel_values"].shape, hf_image["pixel_values"].shape)

        hf_video = processor._video_processor(
            videos=[torch.zeros(2, 3, 336, 480, dtype=torch.uint8)],
            do_sample_frames=False,
            return_tensors="pt",
        )
        video = processor.get_dummy_mm_data(max_num_encoder_tokens=1260, mm_counts={"video": 1})[
            "video"
        ]
        torch.testing.assert_close(
            video["image_position_ids"], hf_video["video_position_ids"].flatten(0, 1)
        )
        self.assertEqual(video["num_frames"], [2])

    def test_only_the_encoder_model_opts_into_item_scheduling(self):
        """Gemma4 Unified projects merged patches without a tower, so it keeps both paths off."""
        self.assertTrue(Gemma4ForConditionalGeneration.supports_mm_encoder_item_scheduling)
        self.assertFalse(Gemma4UnifiedForConditionalGeneration.supports_mm_encoder_item_scheduling)

        unified = Gemma4UnifiedInputProcessor.__new__(Gemma4UnifiedInputProcessor)
        self.assertEqual(unified.get_mm_max_tokens_per_item(), {})
        with self.assertRaises(NotImplementedError):
            unified.get_mm_max_tokens_per_item(max_num_encoder_tokens=8192)
        self.assertIsNone(
            unified.get_mm_item_metadata([], {"image": {"pixel_values": torch.zeros(1)}})
        )


# ---------------------------------------------------------------------------
# Input processor (real tokenizer/processor — gated on LLM_MODELS_ROOT)
# ---------------------------------------------------------------------------


def _get_model_path():
    llm_models_root = os.environ.get("LLM_MODELS_ROOT")
    if not llm_models_root:
        return None

    model_subdirs = (
        "gemma/gemma-4-26B-A4B-it",
        "gemma4/gemma-4-26B-A4B-it",
    )
    for model_subdir in model_subdirs:
        model_path = os.path.join(llm_models_root, model_subdir)
        if os.path.isfile(os.path.join(model_path, "config.json")):
            return model_path
    return os.path.join(llm_models_root, model_subdirs[0])


MODEL_26B_PATH = _get_model_path()


def _model_available():
    if MODEL_26B_PATH is None:
        return False
    return os.path.isfile(os.path.join(MODEL_26B_PATH, "config.json"))


@pytest.mark.cpu_only
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
        self.assertEqual(img_data["pixel_values"].dim(), 3)

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
        self.assertEqual(pv.shape[0], 2)

    def test_item_metadata_matches_prompt_soft_tokens(self):
        """Each image's encoder item yields exactly the soft tokens of its prompt placeholder."""
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        images = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8)),
        ]
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
        input_ids, mm_data = proc(
            {"prompt": prompt, "multi_modal_data": {"image": images}},
            SamplingParams(max_tokens=10),
        )

        metadata = proc.get_mm_item_metadata(input_ids, mm_data["multimodal_data"])

        self.assertEqual(
            list(metadata.output_embedding_lengths),
            [proc.get_num_tokens_per_image(image=image) for image in images],
        )
        self.assertEqual(
            sum(metadata.output_embedding_lengths),
            sum(1 for token in input_ids if token == self.config.image_token_id),
        )

    def test_torch_tensor_image(self):
        """Torch tensor images disable rescaling."""
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
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

        sr_src = 48000
        dur_sec = 0.5
        stereo = np.random.randn(int(sr_src * dur_sec), 2).astype(np.float32)
        normalized = _normalize_audio_inputs([(stereo, sr_src)], target_sr=16000)
        self.assertEqual(len(normalized), 1)
        arr = normalized[0]
        self.assertEqual(arr.ndim, 1)
        self.assertEqual(arr.dtype, np.float32)
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
        """
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        if not hasattr(proc._processor, "video_processor"):
            self.skipTest("Processor has no video_processor")

        frames = [torch.randint(0, 256, (3, 180, 320), dtype=torch.uint8) for _ in range(40)]
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
        predicted_video_tokens = proc.get_num_tokens_per_video(video=frames)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsNotNone(mm_data)
        self.assertIn("video", mm_data["multimodal_data"])
        video_data = mm_data["multimodal_data"]["video"]
        self.assertIn("pixel_values", video_data)
        self.assertEqual(video_data["pixel_values"].dim(), 3)
        self.assertEqual(video_data["pixel_values"].shape[0], len(frames))
        self.assertEqual(video_data["num_frames"], [len(frames)])
        metadata = proc.get_mm_item_metadata(input_ids, mm_data["multimodal_data"])
        self.assertEqual(list(metadata.output_embedding_lengths), [predicted_video_tokens])

        video_token_id = self._video_token_id()
        if video_token_id is not None:
            actual_video_tokens = sum(1 for t in input_ids if t == video_token_id)
            self.assertGreater(
                actual_video_tokens,
                0,
                "Expected video soft tokens in expanded input_ids",
            )
            self.assertEqual(
                predicted_video_tokens,
                actual_video_tokens,
                "Expected one timestamp and soft-token block per supplied frame",
            )

    def test_video_token_prediction_for_non_square_pil_frames(self):
        """Video token prediction pools non-square PIL patch grids per dimension."""
        from PIL import Image

        proc = self._make_processor()
        if not hasattr(proc._processor, "video_processor"):
            self.skipTest("Processor has no video_processor")

        video_processor = proc._processor.video_processor
        patch_size = int(video_processor.patch_size)
        pooling_kernel_size = int(video_processor.pooling_kernel_size)
        patch_grid_height = 2 * pooling_kernel_size + 1
        patch_grid_width = 3 * pooling_kernel_size + 1
        height = patch_grid_height * patch_size
        width = patch_grid_width * patch_size
        frames = [Image.new("RGB", (width, height)) for _ in range(2)]

        expected_tokens_per_frame = (patch_grid_height // pooling_kernel_size) * (
            patch_grid_width // pooling_kernel_size
        )
        # Use the real video processor, but disable resizing temporarily: its resize path produces
        # pooling-aligned dimensions, while the regression requires a non-divisible patch grid.
        with mock.patch.object(video_processor, "do_resize", False):
            predicted_video_tokens = proc.get_num_tokens_per_video(video=frames)

        self.assertEqual(predicted_video_tokens, len(frames) * expected_tokens_per_frame)

    def _video_token_id(self):
        return getattr(self.config, "video_token_id", None)


# ---------------------------------------------------------------------------
# Tier 2 — Full LLM+vision E2E via shared TestModelingMultimodal skeleton.
# ---------------------------------------------------------------------------
#
# Reuses ``test_modeling_multimodal.TestModelingMultimodal`` (the abstract
# base used by qwen3vl / nemotron_h_multimodal / etc.). Mirrors the qwen3vl
# pattern: provide config + class hooks, gate on LLM_MODELS_ROOT.
#
# Audio scenarios are intentionally not included — audio tower refactor is
# follow-up work.


_GEMMA4_E2E_PATH = _get_model_path()


def _gemma4_e2e_model_available() -> bool:
    if _GEMMA4_E2E_PATH is None or not os.path.isfile(
        os.path.join(_GEMMA4_E2E_PATH, "config.json")
    ):
        return False
    # The shared multimodal skeleton's default `get_raw_test_inputs` reads
    # ``{LLM_MODELS_ROOT}/multimodals/test_data/seashore.png``; gate on that
    # too so the test reports skipped instead of failing in CI environments
    # where multimodal test fixtures aren't mirrored.
    test_img = os.path.join(
        os.environ["LLM_MODELS_ROOT"],
        "multimodals",
        "test_data",
        "seashore.png",
    )
    return os.path.isfile(test_img)


# Reduced-layer text+vision config for E2E. Real ``_name_or_path`` so the
# input processor + tokenizer pick up the genuine Gemma4 processor files;
# attention head_dim follows the existing dummy-config recipe in
# ``test_gemma4_e2e_dummy.py`` (128 / 256 for TRTLLM FMHA compatibility).
GEMMA4_E2E_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    # HF Gemma4 (transformers 5.5.3) does not advertise SDPA support; pin
    # to eager so PreTrainedModel.__init__ does not raise on dispatch check.
    "_attn_implementation": "eager",
    # Match the real processor output used by this E2E.
    "image_token_id": 258880,
    "audio_token_id": 262273,
    "video_token_id": 262146,
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
    "boa_token_index": 256001,
    "eoa_token_index": 256002,
    "_name_or_path": _GEMMA4_E2E_PATH or "",
    "text_config": {
        "model_type": "gemma4_text",
        "vocab_size": 262400,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "global_head_dim": 256,
        "num_global_key_value_heads": 4,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "sliding_window": 1024,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "use_double_wide_mlp": False,
        "final_logit_softcapping": None,
        # Exercise the custom mask for image-token blocks.
        "use_bidirectional_attention": "vision",
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    },
    "vision_config": {
        "model_type": "gemma4_vision",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 72,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 10240,
        "use_clipped_linears": False,
        "standardize": True,
        "torch_dtype": "bfloat16",
    },
}


@pytest.mark.skipif(
    not _gemma4_e2e_model_available(),
    reason=(
        "LLM_MODELS_ROOT not set or Gemma4 model not available — "
        "E2E generation requires real tokenizer/processor files."
    ),
)
class TestGemma4MultimodalE2E(unittest.TestCase):
    """Full LLM + vision execution via the shared multimodal skeleton.

    Implemented as a thin wrapper around ``TestModelingMultimodal`` so we
    reuse the standardized context+generation phases and KV cache setup.
    Component tests above cover HF numerical parity; this test targets the
    fused TRTLLM runtime path because full-vocabulary random-weight BF16
    parity is numerically unstable across the fused and eager implementations.
    """

    # The actual abstract base + skeleton live in
    # ``test_modeling_multimodal``. Importing it pulls in helpers that depend
    # on ``utils.llm_data.llm_models_root`` — they only load when the gate
    # above passes.

    def test_image_scenarios(self):
        from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal

        from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import (
            Gemma4HfWeightMapper,
        )

        class _Gemma4E2EImpl(TestModelingMultimodal):
            def get_model_config(self) -> Dict:
                return deepcopy(GEMMA4_E2E_CONFIG)

            def get_trtllm_model_class(self) -> Type:
                return Gemma4ForConditionalGeneration

            def get_hf_model_class(self) -> Type:
                return HFGemma4ForConditionalGeneration

            def get_weight_mapper_class(self) -> Type:
                return Gemma4HfWeightMapper

            def get_model_type(self) -> str:
                return "gemma4"

            def get_model_config_class(self) -> Type:
                return Gemma4Config

            @property
            def skip_hf_inference(self) -> bool:
                return True

            def get_kv_cache_manager(
                self,
                dtype,
                config,
                tokens_per_block,
                max_seq_len,
                batch_size,
                num_blocks,
            ):
                del dtype, max_seq_len
                from test_modeling_gemma4 import _build_gemma4_kv_cache_manager

                return _build_gemma4_kv_cache_manager(
                    config.text_config,
                    num_blocks=num_blocks,
                    tokens_per_block=tokens_per_block,
                    batch_size=batch_size,
                )

            def get_scenarios(self) -> List[MultimodalScenario]:
                # Single sanity-image scenario for the in-PR test list.
                # CUDA-graph / chunked-prefill / kv-cache-reuse permutations
                # are tracked separately in the integration test list.
                return [
                    MultimodalScenario(
                        modality="image",
                        use_cuda_graph=False,
                        chunked_prefill=False,
                        kv_cache_reuse=False,
                    ),
                ]

        # TestModelingMultimodal is abstract — instantiate a dynamic
        # subclass and run only ``test_all`` to keep the surface minimal.
        runner = _Gemma4E2EImpl("test_all")
        runner.setUp()
        try:
            runner.test_all()
        finally:
            runner.tearDown()


class TestGemma4VisionHeadDimPadding(unittest.TestCase):
    """Head_dim 72 (26B/31B vision) gets zero-padded to 80 so the trtllm-gen
    FMHA dispatcher finds a matching cubin (sm100a ships H64/H80/H128/H256/H512;
    no H72). qkv_proj/o_proj weights pad along the head dim; norm + RoPE run
    on the unpadded HF channels; the padded last channels stay zero through
    the kernel and don't contribute to the output (zero × anything = 0 in
    both QK^T and softmax(QK^T)V).

    Without this padding, vision attention falls into the unfused MHA path
    (``Fall back to unfused MHA``) with O(B × T²) workspace, OOMing 26B/31B
    MMMU at LLM-scale ``max_num_requests``.
    """

    HEAD_DIM_72_CONFIG = {
        # nh=2, head_dim=72 → hidden_size=144. Small enough to run on CPU/GPU
        # quickly while exercising the 72→80 padding path.
        "hidden_size": 144,
        "intermediate_size": 288,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 72,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 1024,
        "use_clipped_linears": False,
        "standardize": True,
        "model_type": "gemma4_vision",
    }

    def test_attention_overrides_head_dim_to_80(self):
        """``Gemma4VisionAttention`` should pad HF head_dim=72 to kernel
        head_dim=80 at init; qkv_proj weight dim should reflect padded size."""
        vision_cfg = Gemma4VisionConfig(**self.HEAD_DIM_72_CONFIG)
        trt_tower = _build_trt_vision_tower(vision_cfg)
        attn = trt_tower.encoder.layers[0].self_attn

        self.assertEqual(attn.hf_head_dim, 72)
        self.assertEqual(attn.head_dim, 80)
        self.assertEqual(attn.head_dim_pad, 8)
        # qkv_proj fused: q_size + 2*kv_size = nh*80 + 2*nkv*80 = (2 + 2*2)*80 = 480.
        expected_qkv_out = (
            vision_cfg.num_attention_heads + 2 * vision_cfg.num_key_value_heads
        ) * attn.head_dim
        self.assertEqual(attn.qkv_proj.weight.shape[0], expected_qkv_out)
        # q_norm operates on unpadded HF head_dim (72).
        self.assertEqual(attn.q_norm.weight.shape[0], 72)

    def test_attention_no_pad_when_head_dim_supported(self):
        """Head_dim already in FMHA cubin set (64) → padding is a no-op."""
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["head_dim"] = 64
        cfg_dict["hidden_size"] = 64 * cfg_dict["num_attention_heads"]
        vision_cfg = Gemma4VisionConfig(**cfg_dict)
        trt_tower = _build_trt_vision_tower(vision_cfg)
        attn = trt_tower.encoder.layers[0].self_attn

        self.assertEqual(attn.hf_head_dim, 64)
        self.assertEqual(attn.head_dim, 64)
        self.assertEqual(attn.head_dim_pad, 0)

    @torch.no_grad()
    def test_load_weights_zero_pads_qkv_and_o_proj(self):
        """Weight load should pad q/k/v_proj (rows) and o_proj (cols) head dim
        72→80 with zeros; the appended slices must be exactly zero."""
        vision_cfg = Gemma4VisionConfig(**self.HEAD_DIM_72_CONFIG)
        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        nh = vision_cfg.num_attention_heads
        for layer in trt_tower.encoder.layers:
            qkv_w = layer.self_attn.qkv_proj.weight.data
            # qkv_proj.weight shape: ((nh + 2*nkv) * 80, hidden_size).
            # Reshape to (heads_total, 80, hidden); last 8 channels of each
            # head must be exactly zero (zero-pad from load_weights).
            heads_total = nh + 2 * vision_cfg.num_key_value_heads
            qkv_w_3d = qkv_w.view(heads_total, 80, -1)
            self.assertTrue(
                torch.all(qkv_w_3d[:, 72:, :] == 0),
                "qkv_proj.weight last 8 channels per head must be zero (padding)",
            )

            o_w = layer.self_attn.o_proj.weight.data
            # o_proj.weight shape: (hidden_size, nh * 80) — pad last 8 cols per head.
            o_w_3d = o_w.view(-1, nh, 80)
            self.assertTrue(
                torch.all(o_w_3d[:, :, 72:] == 0),
                "o_proj.weight last 8 cols per head must be zero (padding)",
            )

    @torch.no_grad()
    def test_forward_parity_head_dim_72(self):
        """HF ↔ TRT forward parity at head_dim=72 with padding-to-80 dance.

        The zero-pad math is mathematically equivalent to unpadded HF
        computation: zeros in q/k don't contribute to QK^T, zeros in v
        produce zero output channels, zeros in o_proj's last columns ignore
        those zero channels. Same fp32 tolerance as the head_dim=32 parity
        test (``test_forward_parity_no_clamp``).
        """
        vision_cfg = Gemma4VisionConfig(**self.HEAD_DIM_72_CONFIG)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg)
        with torch.inference_mode():
            hf_out = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            trt_out = trt_tower(pv, pos, output_length=output_length).last_hidden_state

        hf_flat = hf_out.reshape(-1, vision_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        atol, rtol = 5e-2, 5e-2
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=atol, rtol=rtol),
            f"head_dim=72 padded-to-80 diverges from HF: "
            f"max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )


if __name__ == "__main__":
    unittest.main()
