# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's multimodal model declarations and vision tower contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.configs import Qwen4ExpVisionConfig
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_10x__default import (
    modeling as fp8,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_10x__default import (
    vision as fp8_vision,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (
    modeling as nvfp4,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (
    vision as nvfp4_vision,
)
from tensorrt_llm._torch.modeling_v2.multimodal import MultimodalModelForCausalLM
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase
from tensorrt_llm._utils import get_sm_version


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "model, llm, tower",
    [
        (
            fp8.ModelingV2Qwen38FlashNextFp8Sm10xDefaultForConditionalGeneration,
            fp8.ModelingV2Qwen38FlashNextFp8Sm10xDefault,
            fp8_vision.VisionTower,
        ),
        (
            nvfp4.ModelingV2Qwen38FlashNextNvfp4Sm10xDefaultForConditionalGeneration,
            nvfp4.ModelingV2Qwen38FlashNextNvfp4Sm10xDefault,
            nvfp4_vision.VisionTower,
        ),
    ],
)
def test_the_class_declares_what_the_shared_base_needs(model, llm, tower):
    assert issubclass(model, MultimodalModelForCausalLM)
    assert model.llm_class is llm
    spec = model.mm_encoders["vision"]
    assert spec.cls is tower
    assert spec.weight_prefix == "model.visual." and spec.modalities == ("image", "video")
    assert model.llm_requires_input_ids
    assert model.supports_encoder_cache and model.supports_mm_encoder_item_scheduling
    assert model.supports_vision_gpu_normalize
    assert model.supported_encoder_attention_backends == {"vision": frozenset()}


@pytest.mark.cpu_only
def test_profiling_inputs_fill_the_item_budget_with_legal_frames():
    processor = Qwen3VLInputProcessorBase.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=Qwen4ExpVisionConfig())
    processor._processor = SimpleNamespace(
        image_processor=SimpleNamespace(size={"shortest_edge": 65_536, "longest_edge": 16_777_216}),
        video_processor=SimpleNamespace(
            size={"shortest_edge": 4_096, "longest_edge": 25_165_824}, min_frames=4, max_frames=768
        ),
    )
    with torch.device("meta"):
        tower = nvfp4_vision.VisionTower(processor.config.vision_config, torch.bfloat16)
    budgets = tower.get_mm_max_tokens_per_item(processor)
    assert budgets == {"image": 32_768, "video": 49_152}
    # Keep the large pixel allocation on meta while the processor builds the grid on the CPU.
    with patch.object(
        torch, "zeros", side_effect=lambda shape, **kw: torch.empty(shape, device="meta", **kw)
    ):
        for modality, budget in budgets.items():
            data = tower.get_dummy_mm_data(
                processor, max_num_encoder_tokens=budget, mm_counts={modality: 1}, dtype=tower.dtype
            )
            ((frames, height, width),) = data[modality][f"{modality}_grid_thw"].tolist()
            assert frames * height * width == budget
            assert height * width <= 32_768


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="The Qwen4-Exp vision component requires a supported Blackwell GPU.",
)
@pytest.mark.parametrize("vision", [fp8_vision, nvfp4_vision], ids=["fp8", "nvfp4"])
@torch.inference_mode()
def test_qwen4_exp_grouped_tower_matches_independent_items(vision):
    from .test_vision_warm_up import _tower

    tower = _tower(vision)
    grid = [(2, 4, 8), (1, 8, 8), (1, 12, 8)]
    rng = torch.Generator(device="cuda").manual_seed(17)
    pixels = [
        torch.randn(t * h * w, tower.pixel_width, dtype=tower.dtype, device="cuda", generator=rng)
        for t, h, w in grid
    ]
    independent = torch.cat(
        [tower(item, torch.tensor([shape])) for item, shape in zip(pixels, grid, strict=True)]
    )
    grouped = tower(torch.cat(pixels), torch.tensor(grid))
    assert torch.isfinite(independent).all() and torch.equal(grouped, independent)
