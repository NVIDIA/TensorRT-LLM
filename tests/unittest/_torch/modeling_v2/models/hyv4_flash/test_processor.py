# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash processor on CPU, read from the checkpoint (``HYV4_FLASH_CKPT`` or
``LLM_MODELS_ROOT/Hy4-flash-FP8``)."""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PATCH_SIZE = 16
_MERGE_SIZE = 2


@pytest.fixture(scope="module")
def checkpoint_dir() -> str:
    override = os.environ.get("HYV4_FLASH_CKPT")
    path = Path(override) if override else None
    if path is None:
        from utils.llm_data import llm_models_root

        root = llm_models_root()
        path = None if root is None else Path(root) / "Hy4-flash-FP8"
    if path is None or not (path / "processor_config.json").is_file():
        pytest.skip("set HYV4_FLASH_CKPT or place the checkpoint at LLM_MODELS_ROOT/Hy4-flash-FP8")
    return str(path)


@pytest.fixture(scope="module")
def processor(checkpoint_dir):
    from transformers import AutoProcessor

    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    # The model-type lookup imports the module, whose import registers the processor.
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid("hy_v4_flash", "image")
    return AutoProcessor.from_pretrained(checkpoint_dir)


def test_auto_processor_resolves_the_checkpoint_processor(processor):
    from tensorrt_llm._torch.models.modeling_hyv4_flash import (
        HYV4FlashProcessor,
        HYV4VLImageProcessor,
        HYV4VLVideoProcessor,
    )

    assert isinstance(processor, HYV4FlashProcessor)
    assert isinstance(processor.image_processor, HYV4VLImageProcessor)
    assert isinstance(processor.video_processor, HYV4VLVideoProcessor)
    assert processor.image_token_id == 120120


def test_image_patch_count_and_order(processor):
    """Patches are row-major over the merged grid: block row, merge row, block col, merge col."""
    image_processor = processor.image_processor
    # 64x96 is aligned and inside the pixel budget, so the resize is a no-op.
    height, width = 64, 96
    raw = (
        (torch.arange(3 * height * width, dtype=torch.int64) % 251)
        .to(torch.uint8)
        .reshape(3, height, width)
    )

    out = image_processor(images=raw, return_tensors="pt")
    grid_h, grid_w = height // _PATCH_SIZE, width // _PATCH_SIZE
    assert out["image_grid_thw"].tolist() == [[1, grid_h, grid_w]]

    mean = torch.tensor(list(image_processor.image_mean)).view(3, 1, 1)
    std = torch.tensor(list(image_processor.image_std)).view(3, 1, 1)
    expected = (raw.float() * image_processor.rescale_factor - mean) / std
    patches = []
    for block_h in range(grid_h // _MERGE_SIZE):
        for merge_h in range(_MERGE_SIZE):
            for block_w in range(grid_w // _MERGE_SIZE):
                for merge_w in range(_MERGE_SIZE):
                    top = (block_h * _MERGE_SIZE + merge_h) * _PATCH_SIZE
                    left = (block_w * _MERGE_SIZE + merge_w) * _PATCH_SIZE
                    patches.append(
                        expected[:, top : top + _PATCH_SIZE, left : left + _PATCH_SIZE].reshape(-1)
                    )
    torch.testing.assert_close(out["pixel_values"], torch.stack(patches), rtol=1e-5, atol=1e-5)


def test_placeholders(checkpoint_dir, processor):
    """One start+pad+end run per image; the pad expands to merged rows x (merged cols + 1)."""
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    with open(Path(checkpoint_dir) / "tokenizer_config.json", encoding="utf-8") as f:
        tokens = json.load(f)
    image_placeholder = "".join(
        tokens[key] for key in ("image_start_token", "image_token", "image_end_token")
    )
    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder("hy_v4_flash", "image") == image_placeholder
    )

    image_out = processor.image_processor(
        images=torch.zeros(3, 64, 96, dtype=torch.uint8), return_tensors="pt"
    )
    assert processor.replace_image_token(image_out, 0) == processor.image_token * 8
    video_inputs = {"video_grid_thw": torch.tensor([[4, 4, 6]])}
    assert processor.replace_video_token(video_inputs, 0) == processor.video_token * (4 * 8)
