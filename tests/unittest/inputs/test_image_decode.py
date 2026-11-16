# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-parity tests for the `ImageMediaIO` JPEG decode fast path and the image formats."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torchvision.io
from PIL import Image

from tensorrt_llm.inputs import utils as input_utils
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.media_io import ImageMediaIO, _decode_jpeg_image, convert_image_mode
from tensorrt_llm.inputs.registry import (
    INPUT_PROCESSOR_REGISTRY,
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
)
from tensorrt_llm.inputs.utils import (
    async_build_multimodal_prompt,
    async_load_image,
    default_multimodal_input_loader,
    load_image,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("mode", "image_format", "size", "progressive"),
    [
        ("RGB", "JPEG", (8, 7), False),
        ("L", "JPEG", (8, 7), False),
        ("CMYK", "JPEG", (8, 7), False),
        ("RGBA", "PNG", (8, 7), False),
        ("RGB", "JPEG", (64, 48), False),
        ("RGB", "JPEG", (64, 48), True),
    ],
)
def test_image_loading_preserves_rgb_pixels(
    mode: str, image_format: str, size: tuple[int, int], progressive: bool, tmp_path: Path
) -> None:
    width, height = size
    shape = (height, width) if mode == "L" else (height, width, len(mode))
    if size == (64, 48):
        pixels = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        # Span multiple 4:2:0 MCUs to catch IDCT/upsampling rounding changes.
        save_options = {"quality": 90, "subsampling": 2, "progressive": progressive}
    else:
        pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        save_options = {}
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotate on display; neither decode path applies it.
    buffer = BytesIO()
    Image.frombytes(mode, size, pixels.tobytes()).save(
        buffer, format=image_format, exif=exif, **save_options
    )
    encoded = buffer.getvalue()
    image_path = tmp_path / f"input.{image_format.lower()}"
    image_path.write_bytes(encoded)
    expected = np.asarray(convert_image_mode(Image.open(BytesIO(encoded)), "RGB"))
    expected_pixels = torch.from_numpy(np.array(expected, copy=True)).permute(2, 0, 1)
    expected_tensor = expected_pixels.to(dtype=torch.get_default_dtype()).div_(255)
    # Only 8-bit RGB and grayscale JPEGs skip the Pillow decode.
    uses_fast_path = image_format == "JPEG" and mode in ("RGB", "L")
    decoded = _decode_jpeg_image(encoded)
    assert (decoded is not None) == uses_fast_path
    if uses_fast_path:
        torch.testing.assert_close(decoded, expected_pixels, rtol=0, atol=0)

    for output_format in ("np", "pt"):
        media_io = ImageMediaIO(format=output_format)
        outputs = (
            media_io.load_bytes(encoded),
            media_io.load_base64(
                f"image/{image_format.lower()}", base64.b64encode(encoded).decode()
            ),
            media_io.load_file(str(image_path)),
        )
        for output in outputs:
            if output_format == "np":
                np.testing.assert_array_equal(output, expected)
                assert output.flags.c_contiguous
            else:
                torch.testing.assert_close(output, expected_tensor, rtol=0, atol=0)
                assert output.is_contiguous()


def test_jpeg_fast_path_keeps_decompression_bomb_check(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
    # Pillow raises above twice the limit; the torchvision decode must not bypass it.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    with pytest.raises(Image.DecompressionBombError):
        ImageMediaIO(format="np").load_bytes(buffer.getvalue())


def test_jpeg_fast_path_keeps_decompression_bomb_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 40)
    with pytest.warns(Image.DecompressionBombWarning):
        ImageMediaIO(format="np").load_bytes(buffer.getvalue())


@pytest.mark.parametrize("output_format", ["np", "pt"])
def test_jpeg_decoder_error_falls_back_to_pillow(
    output_format: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (17, 83, 191)).save(buffer, format="JPEG")
    encoded = buffer.getvalue()
    expected = np.asarray(convert_image_mode(Image.open(BytesIO(encoded)), "RGB"))
    decoder = Mock(side_effect=RuntimeError("JPEG decoder failed"))
    monkeypatch.setattr(torchvision.io, "decode_image", decoder)

    output = ImageMediaIO(format=output_format).load_bytes(encoded)

    decoder.assert_called_once()
    if output_format == "np":
        np.testing.assert_array_equal(output, expected)
        assert output.flags.c_contiguous
    else:
        expected_tensor = (
            torch.from_numpy(np.array(expected, copy=True))
            .permute(2, 0, 1)
            .to(dtype=torch.get_default_dtype())
            .div_(255)
        )
        torch.testing.assert_close(output, expected_tensor, rtol=0, atol=0)
        assert output.is_contiguous()


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    path = tmp_path / "sample image.jpg"
    pixels = np.random.default_rng(0).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, quality=90)
    return str(path)


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_image_loaders_default_to_uint8_and_keep_pt(image_format: str, tmp_path: Path) -> None:
    """Image loaders return uint8 (H, W, 3) arrays by default; "pt" gives the same pixels as floats."""
    from tensorrt_llm.inputs.media_io import uint8_image_to_tensor

    path = tmp_path / f"input.{image_format.lower()}"
    pixels = np.random.default_rng(0).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format=image_format)
    expected = np.array(convert_image_mode(Image.open(path), "RGB"))
    encoded = path.read_bytes()
    defaults = [
        ImageMediaIO().load_bytes(encoded),
        ImageMediaIO().load_file(str(path)),
        load_image(str(path)),
        asyncio.run(async_load_image(str(path))),
    ]
    for image in defaults:
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image, expected)
    floats = [
        ImageMediaIO(format="pt").load_bytes(encoded),
        ImageMediaIO(format="pt").load_file(str(path)),
        load_image(str(path), format="pt"),
        asyncio.run(async_load_image(str(path), format="pt")),
    ]
    for image in floats:
        assert image.dtype == torch.float32
        assert image.shape == (3, 48, 64)
        # The helper models use to turn uint8 arrays into float images gives the same tensor.
        torch.testing.assert_close(image, uint8_image_to_tensor(expected), rtol=0, atol=0)
    assert isinstance(load_image(str(path), format="pil"), Image.Image)


@pytest.mark.parametrize(
    ("image_data_format", "media_io_kwargs", "expected_format"),
    [
        (None, None, "np"),
        (None, {"image": {"format": "pt"}}, "pt"),
        ("pil", {"image": {"format": "pt"}}, "pil"),
    ],
    ids=["default", "model_preference", "explicit_format"],
)
def test_default_loader_image_format(
    sample_image_path: str,
    image_data_format: str | None,
    media_io_kwargs: dict | None,
    expected_format: str,
) -> None:
    """`default_multimodal_input_loader` uses an explicit format, else the given preference."""
    model_type = "test_image_format_model"
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        model_type,
        MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<image>"},
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            content_format=ContentFormat.PASSTHROUGH,
        ),
    )
    try:
        [inputs] = default_multimodal_input_loader(
            tokenizer=None,
            model_dir="unused",
            model_type=model_type,
            modality="image",
            prompts=["Describe the image."],
            media=[sample_image_path],
            image_data_format=image_data_format,
            media_io_kwargs=media_io_kwargs,
        )
    finally:
        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata(model_type)
    [image] = inputs["multi_modal_data"]["image"]
    expected_type = {"np": np.ndarray, "pt": torch.Tensor, "pil": Image.Image}[expected_format]
    assert isinstance(image, expected_type)


# The image format each registered model type declares. `None` means the loader default, uint8 "np".
_DECLARED_IMAGE_FORMATS = {
    # The checkpoint's own processor reads the images; keep the float tensors it always got.
    "hyperclovax_vlm": "pt",
    "step3p7": "pt",
    # Declared "np" before images defaulted to it.
    "cosmos3": "np",
    "cosmos3_edge": "np",
    "cosmos3_omni": "np",
    "minimax_m3_vl": "np",
    "qwen3_5": "np",
    "qwen3_5_moe": "np",
    "qwen3_vl": "np",
    "qwen3_vl_moe": "np",
    "qwen4_exp": "np",
    # PIL is the native input of its processor.
    "glm5_next": "pil",
    # HF processors that rescale uint8 images themselves.
    "gemma3": None,
    "gemma4": None,
    "gemma4_unified": None,
    "llama4": None,
    "llava_next": None,
    "minicpmv4_6": None,
    "mistral3": None,
    "mistral_large_3": None,
    "qwen2_5_vl": None,
    "qwen2_vl": None,
    # Processors that turn uint8 arrays into the PIL images or float tensors they need.
    "deepseek_v41": None,
    "kimi_k25": None,
    "kimi_k3": None,
    "llava_llama": None,
    "mistral_common": None,
    "NemotronH_Nano_Omni_Reasoning_V3": None,
    "NemotronH_Nano_VL_V2": None,
    "nemotron_h_omni": None,
    "phi4mm": None,
    # Declares "pil", but reads it from the checkpoint, so helpers without it use the default.
    "muse_spark": None,
    # No images.
    "whisper": None,
}


@pytest.mark.parametrize(("model_type", "image_format"), sorted(_DECLARED_IMAGE_FORMATS.items()))
def test_registered_model_image_format(model_type: str, image_format: str | None) -> None:
    preferences = INPUT_PROCESSOR_REGISTRY.get_preferred_media_io_kwargs(model_type)
    assert preferences.get("image", {}).get("format") == image_format


@pytest.mark.parametrize(
    ("model_type", "media_io_kwargs", "image_type"),
    [
        # Nothing declared: the loader default.
        ("qwen2_5_vl", None, np.ndarray),
        ("gemma4", None, np.ndarray),
        # Declared formats.
        ("step3p7", None, torch.Tensor),
        ("glm5_next", None, Image.Image),
        # The caller's kwargs win, as per-request kwargs do in trtllm-serve.
        ("qwen2_5_vl", {"image": {"format": "pt"}}, torch.Tensor),
        ("step3p7", {"image": {"format": "np"}}, np.ndarray),
    ],
)
@pytest.mark.parametrize("helper", ["async_build_multimodal_prompt", "default_loader"])
def test_prompt_helpers_follow_model_image_format(
    monkeypatch: pytest.MonkeyPatch,
    sample_image_path: str,
    helper: str,
    model_type: str,
    media_io_kwargs: dict | None,
    image_type: type,
) -> None:
    """Helpers that name the model decode images in the format it declares."""

    # Only the media path is under test: skip loading and rendering the template.
    async def render(**kwargs) -> str:
        return "rendered"

    monkeypatch.setattr(input_utils, "async_apply_chat_template", render)
    monkeypatch.setattr(input_utils, "apply_chat_template", lambda **kwargs: "rendered")
    monkeypatch.setattr(input_utils, "resolve_hf_chat_template", lambda *args: None)
    monkeypatch.setattr(input_utils, "AutoProcessor", Mock())
    if helper == "async_build_multimodal_prompt":
        inputs = asyncio.run(
            async_build_multimodal_prompt(
                model_type=model_type,
                tokenizer=Mock(),
                processor=None,
                prompt="Describe the image.",
                media=[sample_image_path],
                modality="image",
                media_io_kwargs=media_io_kwargs,
            )
        )
    else:
        [inputs] = default_multimodal_input_loader(
            tokenizer=Mock(),
            model_dir="unused",
            model_type=model_type,
            modality="image",
            prompts=["Describe the image."],
            media=[sample_image_path],
            media_io_kwargs=media_io_kwargs,
        )
    [image] = inputs["multi_modal_data"]["image"]
    assert isinstance(image, image_type)
