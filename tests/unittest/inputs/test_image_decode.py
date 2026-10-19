# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-parity tests for the `ImageMediaIO` JPEG decode fast path."""

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torchvision.io
from PIL import Image

from tensorrt_llm.inputs.media_io import ImageMediaIO, _decode_jpeg_image, convert_image_mode

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
