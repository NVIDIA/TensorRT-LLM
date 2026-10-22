# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch
from pydantic import ValidationError
from transformers import Gemma4Config, PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.cpu_only

_DEFAULTS = {"vision": "TRTLLM", "audio": "SDPA"}
_SUPPORTED = {"vision": ("TRTLLM", "FLASHINFER", "VANILLA"), "audio": ("SDPA",)}


def test_omitted_encoder_backends_preserve_model_defaults():
    config = ModelConfig(attn_backend="VANILLA")
    defaults = {"vision": "FLASHINFER", "audio": "SDPA"}
    result = config.get_multimodal_encoder_attention_backends(
        defaults=defaults, supported_backends=_SUPPORTED
    )
    assert result == defaults
    assert result is not defaults
    assert config.attn_backend == "VANILLA"


def test_partial_encoder_override_preserves_other_towers_and_decoder():
    mm_config = MultimodalConfig(encoder_attention_backend={"vision": "vanilla"})
    config = ModelConfig(attn_backend="FLASHINFER", multimodal_config=mm_config)
    result = config.get_multimodal_encoder_attention_backends(
        defaults=_DEFAULTS, supported_backends=_SUPPORTED
    )
    assert result == {"vision": "VANILLA", "audio": "SDPA"}
    assert config.attn_backend == "FLASHINFER"
    assert mm_config.encoder_attention_backend == {"vision": "VANILLA"}
    assert MultimodalConfig.model_validate_json(mm_config.model_dump_json()) == mm_config


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"video": "TRTLLM"}, ValueError, "unknown encoders.*video"),
        ({"audio": "TRTLLM"}, ValueError, "audio encoder does not support"),
        ({"vision": "unknown"}, ValidationError, "encoder_attention_backend.vision"),
    ],
)
def test_invalid_encoder_selection_fails_at_initialization(overrides, error, message):
    with pytest.raises(error, match=message):
        config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_attention_backend=overrides)
        )
        config.get_multimodal_encoder_attention_backends(
            defaults=_DEFAULTS, supported_backends=_SUPPORTED
        )


def test_explicit_flashinfer_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr("tensorrt_llm._torch.flashinfer_utils.IS_FLASHINFER_AVAILABLE", False)
    config = ModelConfig(
        multimodal_config=MultimodalConfig(encoder_attention_backend={"vision": "FLASHINFER"})
    )
    with pytest.raises(ValueError, match="FlashInfer is unavailable"):
        config.get_multimodal_encoder_attention_backends(
            defaults=_DEFAULTS, supported_backends=_SUPPORTED
        )


class _RecordingVisionEncoder(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.weight = torch.nn.Parameter(torch.ones(1))


@pytest.mark.parametrize(
    ("decoder_backend", "override"),
    [("TRTLLM", None), ("TRTLLM", "VANILLA"), ("FLASHINFER", "VANILLA")],
)
def test_qwen_vision_override_reaches_encoder_without_changing_decoder(decoder_backend, override):
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VisionModelBase

    config = ModelConfig(
        pretrained_config=PretrainedConfig(
            text_config=PretrainedConfig(dtype=torch.float32),
            vision_config=PretrainedConfig(),
        ),
        attn_backend=decoder_backend,
        multimodal_config=MultimodalConfig(
            encoder_attention_backend={"vision": override} if override else None
        ),
    )
    config._frozen = True
    encoder = Qwen3VisionModelBase(config, _RecordingVisionEncoder)
    assert encoder.visual.model_config.attn_backend == (override or decoder_backend)
    assert encoder.model_config.extra_attrs is config.extra_attrs
    assert encoder.model_config._frozen
    if override is None:
        assert encoder.model_config is config
    assert config.attn_backend == decoder_backend


def test_hf_qwen_sdpa_matches_eager_with_the_same_weights():
    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

    from tensorrt_llm._torch.models.modeling_qwen2vl import (
        Qwen2VisionModelBase,
        Qwen2VisionTransformerPretrainedModel,
    )
    from tensorrt_llm.models.modeling_utils import QuantConfig

    vision_config = Qwen2VLVisionConfig(
        depth=1,
        embed_dim=16,
        hidden_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
    )
    encoders = []
    for backend in ("VANILLA", "SDPA"):
        config = ModelConfig(
            pretrained_config=PretrainedConfig(
                vision_config=deepcopy(vision_config),
                dtype=torch.float32,
            ),
            quant_config=QuantConfig(),
            multimodal_config=MultimodalConfig(encoder_attention_backend={"vision": backend}),
        )
        encoders.append(Qwen2VisionModelBase(config, Qwen2VisionTransformerPretrainedModel))
    eager, sdpa = encoders
    assert eager.visual.config._attn_implementation == "eager"
    assert sdpa.visual.config._attn_implementation == "sdpa"
    sdpa.visual.load_state_dict(eager.visual.state_dict())
    pixels = torch.randn(4, 12)
    grid = torch.tensor([[1, 2, 2]])
    with torch.inference_mode():
        torch.testing.assert_close(
            sdpa.visual(pixels, grid_thw=grid),
            eager.visual(pixels, grid_thw=grid),
            rtol=1e-5,
            atol=1e-6,
        )


def test_gemma4_towers_use_independent_backends_and_share_registry():
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration

    config = ModelConfig(
        pretrained_config=Gemma4Config(text_config={}, vision_config={}, audio_config={}),
        attn_backend="FLASHINFER",
        multimodal_config=MultimodalConfig(
            encoder_attention_backend={"vision": "VANILLA", "audio": "VANILLA"}
        ),
    )
    text = Gemma4ForConditionalGeneration.get_sub_model_config(config, "text_config")
    vision = Gemma4ForConditionalGeneration.get_sub_model_config(config, "vision_config")
    audio = Gemma4ForConditionalGeneration.get_sub_model_config(config, "audio_config")
    assert text.attn_backend == "FLASHINFER"
    assert vision.attn_backend == "VANILLA"
    assert audio.attn_backend == "VANILLA"
    assert text.extra_attrs is vision.extra_attrs is audio.extra_attrs is config.extra_attrs


def test_encoder_free_gemma4_rejects_encoder_attention_settings():
    from tensorrt_llm._torch.models.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )

    config = ModelConfig(
        multimodal_config=MultimodalConfig(encoder_attention_backend={"vision": "TRTLLM"})
    )
    with pytest.raises(ValueError, match="enabled encoders: \\[\\]"):
        Gemma4UnifiedForConditionalGeneration(config)


def test_minimax_fixed_sdpa_rejects_unsupported_backend_before_loading_text():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3VLForConditionalGeneration

    config = ModelConfig(
        pretrained_config=PretrainedConfig(vision_config=PretrainedConfig()),
        multimodal_config=MultimodalConfig(encoder_attention_backend={"vision": "TRTLLM"}),
    )
    with pytest.raises(ValueError, match="vision encoder does not support.*TRTLLM"):
        MiniMaxM3VLForConditionalGeneration(config)


def test_backend_contract_distinguishes_unsupported_from_disabled_encoders():
    from tensorrt_llm._torch.models.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration
    from tensorrt_llm._torch.pyexecutor.model_loader import (
        _validate_encoder_attention_backend_support,
    )

    config = ModelConfig(
        pretrained_config=Gemma4Config(text_config={}, vision_config=None, audio_config={})
    )
    assert Gemma4ForConditionalGeneration.supports_encoder_attention_backend
    assert Gemma4ForConditionalGeneration.get_supported_encoder_attention_backends(config) == {
        "audio": frozenset({"VANILLA"})
    }
    config.pretrained_config.audio_config = None
    assert Gemma4ForConditionalGeneration.get_supported_encoder_attention_backends(config) == {}
    assert not Gemma4UnifiedForConditionalGeneration.supports_encoder_attention_backend

    overrides = MultimodalConfig(encoder_attention_backend={"vision": "TRTLLM"})
    with pytest.raises(ValueError, match="does not support.*encoder_attention_backend"):
        _validate_encoder_attention_backend_support(
            Gemma4UnifiedForConditionalGeneration, overrides
        )
    _validate_encoder_attention_backend_support(Gemma4UnifiedForConditionalGeneration, None)
    _validate_encoder_attention_backend_support(Gemma4ForConditionalGeneration, overrides)
    config.multimodal_config = overrides
    with pytest.raises(ValueError, match="unknown encoders.*vision"):
        config.get_multimodal_encoder_attention_backends(
            defaults={"vision": "TRTLLM"},
            supported_backends=Gemma4ForConditionalGeneration.get_supported_encoder_attention_backends(
                config
            ),
        )


def test_qwen_backend_contract_preserves_hf_and_native_distinction():
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5_VLModel, Qwen2VLModel
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLModel
    from tensorrt_llm._torch.models.modeling_qwen3vl_moe import Qwen3MoeVLModel

    config = ModelConfig(pretrained_config=PretrainedConfig(vision_config=PretrainedConfig()))
    assert Qwen2VLModel.get_supported_encoder_attention_backends(config)["vision"] == frozenset(
        {"FLASH_ATTENTION_2", "SDPA", "VANILLA"}
    )
    native = frozenset({"TRTLLM", "FLASHINFER", "VANILLA"})
    for model_cls in (Qwen2_5_VLModel, Qwen3VLModel, Qwen3MoeVLModel):
        assert model_cls.supports_encoder_attention_backend
        assert model_cls.get_supported_encoder_attention_backends(config) == {"vision": native}
