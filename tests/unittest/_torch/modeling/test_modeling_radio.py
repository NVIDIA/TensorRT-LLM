# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.models import modeling_radio
from tensorrt_llm._torch.models.modeling_radio import RADIOVisionModel
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_TINY_VIT = modeling_radio.VITTIMMConfig(
    embed_dim=64,
    depth=2,
    num_attention_heads=2,
    intermediate_size=128,
    img_size=32,
)


def _make_vision_config():
    """Minimal PretrainedConfig mimicking Nemotron-Nano-V3's `vision_config`.

    Patterned on the `vision_config` block of config.json shipped with newer nemotron nano
    multimodal models; fields pared down to what RADIOVisionModel reads.
    """
    config = PretrainedConfig()
    config.patch_size = 16
    config.adaptor_names = None
    config.feature_normalizer_config = None
    config.inter_feature_normalizer_config = None
    config.max_resolution = 64
    config.vitdet_window_size = None
    config.preferred_resolution = (32, 32)
    config.video_temporal_patch_size = 1
    config.separate_video_embedder = True
    config.torch_dtype = torch.bfloat16
    config.args = {
        "model": "vit_tiny_test",
        "in_chans": None,
        "input_size": None,
        "drop": 0.0,
        "cpe_max_size": 64,
        "cls_token_per_teacher": False,
        "teachers": [{"name": "dummy"}],
        "register_multiple": None,
        "cpe_num_registers": None,
    }
    return config


@pytest.fixture
def tiny_vit_config():
    with mock.patch.dict(
        modeling_radio.VIT_TIMM_CONFIG_BY_NAME,
        {"vit_tiny_test": _TINY_VIT},
    ):
        yield


def _make_fp8_model_config():
    vision_config = _make_vision_config()
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8,
        kv_cache_quant_algo=QuantAlgo.FP8,
    )
    return model_config_lib.ModelConfig(
        pretrained_config=vision_config,
        quant_config=quant_config,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_radio_fp8_parent_kv_cache_does_not_leak_into_vit(tiny_vit_config):
    """When the parent LLM uses FP8 KV cache, the RADIO vision encoder must not inherit it.

    A ViT has no KV cache (kv_cache_manager=None). If `kv_cache_quant_algo=FP8` leaks into the
    vision tower, FlashInfer raises at forward time about it not being supported.
    """
    vision_model = RADIOVisionModel(_make_fp8_model_config(), disable_quantization=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    vision_model = vision_model.to(device).to(dtype)

    # 32x32 image: multiple of `patch_size=16`, so `min_resolution_step` is satisfied.
    pixel_values = torch.randn(1, 3, 32, 32, device=device, dtype=dtype)

    with torch.inference_mode():
        features = vision_model.forward(pixel_values)

    assert features.shape[0] == 1
    assert features.shape[-1] == _TINY_VIT.embed_dim
