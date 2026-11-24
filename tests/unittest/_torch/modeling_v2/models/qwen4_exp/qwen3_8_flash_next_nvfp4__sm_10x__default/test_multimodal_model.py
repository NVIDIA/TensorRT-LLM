# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's multimodal model declaration."""

import pytest

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (
    modeling as M,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (
    vision as V,
)
from tensorrt_llm._torch.modeling_v2.multimodal import MultimodalModelForCausalLM


@pytest.mark.cpu_only
def test_the_class_declares_what_the_shared_base_needs():
    model = M.ModelingV2Qwen38FlashNextNvfp4Sm10xDefaultForConditionalGeneration
    assert issubclass(model, MultimodalModelForCausalLM)
    assert model.llm_class is M.ModelingV2Qwen38FlashNextNvfp4Sm10xDefault
    spec = model.mm_encoders["vision"]
    assert spec.cls is V.VisionTower
    assert spec.weight_prefix == "model.visual." and spec.modalities == ("image", "video")
    assert model.llm_requires_input_ids
    assert model.supports_encoder_cache
