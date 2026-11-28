# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image input on the DeepSeek-V4.1 targets (text only), on CPU: each target refuses a context request that carries
image input with a per-request error, admits a generation-only request of an image prompt and every text request,
and reads the image token of a vision checkpoint for its Engram hash."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.common import text_model_config
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.image_input import (
    refuse_image_request,
    refuse_image_step,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams

pytestmark = pytest.mark.cpu_only

_IMG = 129264
_PKG = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
_TARGETS = {
    "dep2": ("flash_nvfp4__sm_103__dep2", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep2"),
    "dep": ("flash_nvfp4__sm_103__dep", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep"),
    "tep8": (
        "flash_nvfp4__sm_103__tep8_moe_tp4_ep2",
        "ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2",
    ),
}


def _module(name):
    return importlib.import_module(f"{_PKG}.{_TARGETS[name][0]}.modeling")


def _request(data, generation_only=False):
    return SimpleNamespace(py_multimodal_data=data, is_generation_only_request=generation_only)


_IMAGE = {"image": {"pixel_values": [torch.zeros(1)]}}


def test_refuse_image_input():
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_request(_request(_IMAGE))
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_request(_request({"multimodal_embedding": torch.zeros(4, 8)}))
    refuse_image_request(
        _request(_IMAGE, generation_only=True)
    )  # its prompt ran on the context server
    refuse_image_request(_request(None))
    refuse_image_request(_request({}))
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_step([MultimodalParams(multimodal_data=_IMAGE)])
    refuse_image_step(None)
    refuse_image_step([])
    refuse_image_step([MultimodalParams()])


@pytest.mark.parametrize("name", list(_TARGETS))
def test_each_target_refuses_image_context_requests_at_admission(name):
    """The engine calls `validate_request` on every new request (a ValueError fails that request only)."""
    target = getattr(_module(name), _TARGETS[name][1])
    with pytest.raises(ValueError, match="serves text only"):
        target.validate_request(None, _request(_IMAGE))
    target.validate_request(None, _request(_IMAGE, generation_only=True))
    target.validate_request(None, _request(None))


@pytest.mark.parametrize("composite", [False, True], ids=["flat", "composite"])
@pytest.mark.parametrize(
    "layers, image", [(32, _IMG), (0, None)], ids=["vision", "vision-0-layers"]
)
def test_engram_reads_the_image_token_of_a_vision_checkpoint(composite, layers, image):
    """`text_model_config` (every target's shell) hands the Engram hash the image token (hashed as the pad id: the
    n-gram break stock applies) only for a checkpoint with a vision tower, the composite and the flat form alike."""
    vision = PretrainedConfig(num_hidden_layers=layers)
    text = PretrainedConfig(architectures=["DeepseekV41ForCausalLM"], hidden_size=8)
    if composite:
        cfg = PretrainedConfig(text_config=text, vision_config=vision, image_token_id=_IMG)
    else:
        cfg = text
        cfg.vision_config, cfg.image_token_id = vision, _IMG
    out = text_model_config(ModelConfig(pretrained_config=cfg))
    assert out.pretrained_config is text
    assert text._v41_image_token_id == image


def test_text_checkpoint_has_no_image_token():
    text = PretrainedConfig(architectures=["DeepseekV41ForCausalLM"], hidden_size=8)
    text_model_config(ModelConfig(pretrained_config=text))
    assert text._v41_image_token_id is None
