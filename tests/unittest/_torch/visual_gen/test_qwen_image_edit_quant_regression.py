# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Qwen-Image-Edit quantized-output regression checks.

These tests validate saved BF16/FP8/NVFP4 edit outputs from the same
input image, prompt, seed, resolution, and denoising settings. They are
skipped by default because they require pre-generated image artifacts.

Set:
  QWEN_IMAGE_EDIT_QUANT_COMPARE_DIR=/path/to/quant_compare_run
  QWEN_IMAGE_EDIT_INPUT_IMAGE=/path/to/source_image.png

The expected invariant is that a quantized edit output should track the
BF16 edited output more closely than the unedited input image. If it is
closer to the input, the quantized path may be under-editing.
"""

from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


_COMPARE_DIR_ENV = "QWEN_IMAGE_EDIT_QUANT_COMPARE_DIR"
_INPUT_IMAGE_ENV = "QWEN_IMAGE_EDIT_INPUT_IMAGE"


def _require_compare_dir() -> Path:
    value = os.environ.get(_COMPARE_DIR_ENV)
    if not value:
        pytest.skip(f"Set {_COMPARE_DIR_ENV} to run Qwen-Image-Edit quant regression checks.")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{_COMPARE_DIR_ENV} does not point to a directory: {path}")
    return path


def _require_input_image() -> Path:
    value = os.environ.get(_INPUT_IMAGE_ENV)
    if not value:
        pytest.skip(f"Set {_INPUT_IMAGE_ENV} to run Qwen-Image-Edit quant regression checks.")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"{_INPUT_IMAGE_ENV} does not point to an image file: {path}")
    return path


def _image_path(name: str) -> Path:
    path = _require_compare_dir() / name
    assert path.is_file(), f"Missing generated image: {path}"
    return path


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _to_lpips_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


@lru_cache(maxsize=1)
def _lpips_metric():
    lpips = pytest.importorskip("lpips")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return lpips.LPIPS(net="alex").to(device).eval(), device


def _lpips_distance(left: Image.Image, right: Image.Image) -> float:
    if left.size != right.size:
        right = right.resize(left.size, Image.Resampling.BICUBIC)
    metric, device = _lpips_metric()
    with torch.no_grad():
        return float(
            metric(
                _to_lpips_tensor(left).to(device),
                _to_lpips_tensor(right).to(device),
            ).item()
        )


@pytest.fixture(scope="module")
def quant_images() -> dict[str, Image.Image]:
    paths = {
        "input": _require_input_image(),
        "bf16": _image_path("bf16.png"),
        "fp8": _image_path("fp8_block_scales.png"),
        "nvfp4": _image_path("nvfp4.png"),
    }
    return {name: _load_rgb(path) for name, path in paths.items()}


def test_qwen_image_edit_quant_outputs_exist_and_match_shape(
    quant_images: dict[str, Image.Image],
) -> None:
    """All compared images should be RGB and have the same resolution."""
    expected_size = quant_images["bf16"].size
    for name, image in quant_images.items():
        assert image.mode == "RGB", name
        assert image.size == expected_size, (name, image.size, expected_size)


def test_qwen_image_edit_quant_metrics_file_matches_expected_outputs() -> None:
    """The saved quant metrics must reference both FP8 and NVFP4 outputs."""
    metrics_path = _require_compare_dir() / "quant_compare_metrics.json"
    assert metrics_path.is_file(), f"Missing metrics file: {metrics_path}"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["candidates"]["fp8_block_scales"]["lpips_alex"] is not None
    assert metrics["candidates"]["nvfp4"]["lpips_alex"] is not None
    assert Path(metrics["candidates"]["fp8_block_scales"]["image"]).name == "fp8_block_scales.png"
    assert Path(metrics["candidates"]["nvfp4"]["image"]).name == "nvfp4.png"


@pytest.mark.parametrize(
    "quant_name",
    ["fp8", "nvfp4"],
    ids=["fp8_block_scales", "nvfp4"],
)
def test_qwen_image_edit_quant_output_tracks_bf16_more_than_input(
    quant_images: dict[str, Image.Image],
    quant_name: str,
) -> None:
    """A quantized edit should be closer to BF16 edit than to the original image."""
    quant_to_bf16 = _lpips_distance(quant_images[quant_name], quant_images["bf16"])
    quant_to_input = _lpips_distance(quant_images[quant_name], quant_images["input"])
    assert quant_to_bf16 < quant_to_input, (
        f"{quant_name} appears closer to the unedited input than to the BF16 edit: "
        f"LPIPS(quant,bf16)={quant_to_bf16:.6f}, "
        f"LPIPS(quant,input)={quant_to_input:.6f}"
    )


def test_qwen_image_edit_nvfp4_edit_magnitude_is_not_collapsed_to_input(
    quant_images: dict[str, Image.Image],
) -> None:
    """NVFP4 should preserve a meaningful amount of the BF16 edit movement."""
    bf16_to_input = _lpips_distance(quant_images["bf16"], quant_images["input"])
    nvfp4_to_input = _lpips_distance(quant_images["nvfp4"], quant_images["input"])
    assert nvfp4_to_input >= 0.75 * bf16_to_input, (
        "NVFP4 output appears under-edited relative to BF16: "
        f"LPIPS(nvfp4,input)={nvfp4_to_input:.6f}, "
        f"LPIPS(bf16,input)={bf16_to_input:.6f}"
    )
