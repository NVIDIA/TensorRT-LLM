# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transparent pixels become white in RGB whatever mode the transparency is stored in."""

import asyncio
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from tensorrt_llm.inputs.media_io import _load_and_convert_image, convert_image_mode
from tensorrt_llm.inputs.utils import async_load_image, load_image

pytestmark = pytest.mark.cpu_only

_OPAQUE = (200, 0, 0)
_HIDDEN = (10, 20, 30)
_WHITE = [255, 255, 255]


def _rgba() -> Image.Image:
    image = Image.new("RGBA", (4, 4), (*_OPAQUE, 255))
    image.putpixel((0, 0), (*_HIDDEN, 0))
    return image


def _la() -> Image.Image:
    image = Image.new("LA", (4, 4), (100, 255))
    image.putpixel((0, 0), (10, 0))
    return image


def _pa() -> Image.Image:
    image = Image.new("PA", (4, 4), (1, 255))
    image.putpalette([0, 0, 0, *_OPAQUE] + [0] * (256 * 3 - 6))
    image.putpixel((0, 0), (0, 0))
    return image


def _palette_with_transparent_index() -> Image.Image:
    image = Image.new("P", (4, 4), 1)
    image.putpalette([0, 0, 0, *_OPAQUE] + [0] * (256 * 3 - 6))
    image.info["transparency"] = 0
    image.putpixel((0, 0), 0)
    return image


def _rgb_with_color_key() -> Image.Image:
    image = Image.new("RGB", (4, 4), _OPAQUE)
    image.info["transparency"] = _HIDDEN
    image.putpixel((0, 0), _HIDDEN)
    return image


def _gray_with_color_key() -> Image.Image:
    image = Image.new("L", (4, 4), 100)
    image.info["transparency"] = 10
    image.putpixel((0, 0), 10)
    return image


@pytest.mark.parametrize(
    "make_image",
    [_rgba, _la, _pa, _palette_with_transparent_index, _gray_with_color_key, _rgb_with_color_key],
    ids=["RGBA", "LA", "PA", "P_transparent_index", "L_color_key", "RGB_color_key"],
)
def test_transparent_pixel_is_white_in_every_mode(make_image) -> None:
    converted = convert_image_mode(make_image(), "RGB")

    assert converted.mode == "RGB"
    assert np.asarray(converted)[0, 0].tolist() == _WHITE


def test_opaque_pixels_are_unchanged() -> None:
    assert np.asarray(convert_image_mode(_rgba(), "RGB"))[2, 2].tolist() == list(_OPAQUE)
    assert np.asarray(convert_image_mode(_la(), "RGB"))[2, 2].tolist() == [100, 100, 100]
    assert np.asarray(convert_image_mode(_palette_with_transparent_index(), "RGB"))[
        2, 2
    ].tolist() == list(_OPAQUE)


def test_image_without_transparency_still_uses_plain_conversion() -> None:
    gray = Image.new("L", (2, 2), 77)

    assert np.asarray(convert_image_mode(gray, "RGB")).tolist() == (
        np.asarray(gray.convert("RGB")).tolist()
    )


def test_loaded_png_with_a_transparent_palette_index_is_white() -> None:
    buffer = BytesIO()
    _palette_with_transparent_index().save(buffer, format="PNG", transparency=0)
    buffer.seek(0)

    assert np.asarray(_load_and_convert_image(buffer))[0, 0].tolist() == _WHITE


def test_plain_rgb_image_is_returned_unchanged() -> None:
    image = Image.new("RGB", (2, 2), _OPAQUE)

    assert convert_image_mode(image, "RGB") is image


def test_loaded_gif_with_a_transparent_index_is_white() -> None:
    buffer = BytesIO()
    _palette_with_transparent_index().save(buffer, format="GIF", transparency=0)
    buffer.seek(0)

    assert np.asarray(_load_and_convert_image(buffer))[0, 0].tolist() == _WHITE


@pytest.mark.parametrize("make_image", [_la, _palette_with_transparent_index], ids=["LA", "P"])
def test_pil_input_to_load_image_is_flattened_like_a_decoded_file(make_image) -> None:
    assert np.asarray(load_image(make_image(), format="pil"))[0, 0].tolist() == _WHITE
    assert (
        np.asarray(asyncio.run(async_load_image(make_image(), format="pil")))[0, 0].tolist()
        == _WHITE
    )


def test_palette_transparency_moved_into_an_rgba_palette_is_white() -> None:
    image = _palette_with_transparent_index()
    image.apply_transparency()
    assert "transparency" not in image.info and image.palette.mode == "RGBA"

    converted = np.asarray(convert_image_mode(image, "RGB"))

    assert converted[0, 0].tolist() == _WHITE
    assert converted[2, 2].tolist() == list(_OPAQUE)


def test_partial_alpha_is_blended_not_thresholded_for_la() -> None:
    image = Image.new("LA", (2, 2), (100, 255))
    image.putpixel((0, 0), (100, 128))

    converted = np.asarray(convert_image_mode(image, "RGB"))

    # (100 * 128 + 255 * 127) / 255 = 177.2
    assert converted[0, 0].tolist() == [177, 177, 177]
    assert converted[1, 1].tolist() == [100, 100, 100]


def test_partial_alpha_is_blended_for_a_palette_with_byte_transparency() -> None:
    image = Image.new("P", (2, 2), 1)
    image.putpalette([100, 100, 100, *_OPAQUE] + [0] * (256 * 3 - 6))
    image.info["transparency"] = bytes([128])
    image.putpixel((0, 0), 0)

    converted = np.asarray(convert_image_mode(image, "RGB"))

    assert converted[0, 0].tolist() == [177, 177, 177]
    assert converted[1, 1].tolist() == list(_OPAQUE)
