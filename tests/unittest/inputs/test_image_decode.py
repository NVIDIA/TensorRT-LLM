# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-parity tests for the `ImageMediaIO` JPEG decode fast path."""

import base64
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm.inputs.media_io import ImageMediaIO, _decode_jpeg_image, convert_image_mode

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("mode", "image_format"),
    [("RGB", "JPEG"), ("L", "JPEG"), ("CMYK", "JPEG"), ("RGBA", "PNG")],
)
def test_image_loading_preserves_rgb_pixels(mode, image_format, tmp_path):
    shape = (7, 8) if mode == "L" else (7, 8, len(mode))
    pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotate on display; neither decode path applies it.
    buffer = BytesIO()
    Image.frombytes(mode, (8, 7), pixels.tobytes()).save(buffer, format=image_format, exif=exif)
    encoded = buffer.getvalue()
    image_path = tmp_path / f"input.{image_format.lower()}"
    image_path.write_bytes(encoded)
    expected = np.asarray(convert_image_mode(Image.open(BytesIO(encoded)), "RGB"))
    expected_tensor = (
        torch.from_numpy(np.array(expected, copy=True))
        .permute(2, 0, 1)
        .to(dtype=torch.get_default_dtype())
        .div_(255)
    )
    # Only 8-bit RGB and grayscale JPEGs skip the Pillow decode.
    uses_fast_path = image_format == "JPEG" and mode in ("RGB", "L")
    assert (_decode_jpeg_image(encoded) is not None) == uses_fast_path

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


def test_jpeg_fast_path_keeps_decompression_bomb_check(monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="JPEG")
    # Pillow raises above twice the limit; the torchvision decode must not bypass it.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16)
    with pytest.raises(Image.DecompressionBombError):
        ImageMediaIO(format="np").load_bytes(buffer.getvalue())
