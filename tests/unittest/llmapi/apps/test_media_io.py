# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from unittest.mock import AsyncMock, patch

import pytest

from tensorrt_llm.inputs import MultimodalDataTracker
from tensorrt_llm.inputs.media_io import (
    AudioMediaIO,
    BaseMediaIO,
    ImageMediaIO,
    VideoMediaIO,
    convert_image_mode,
)
from tensorrt_llm.serve.chat_utils import parse_chat_message_content_part

pytestmark = pytest.mark.cpu_only


class CustomError(Exception):
    pass


class TestMultimodalLoadErrorPropagation:
    """Verify that errors from multimodal loading propagate."""

    @pytest.fixture
    def mm_tracker(self):
        return MultimodalDataTracker(model_type="dummy")

    @pytest.mark.parametrize(
        "part, patch_target",
        [
            (
                {"type": "image_url", "image_url": {"url": "http://bad-url/img.png"}},
                "tensorrt_llm.inputs.media_io.ImageMediaIO.async_load",
            ),
            (
                {"type": "video_url", "video_url": {"url": "http://bad-url/vid.mp4"}},
                "tensorrt_llm.inputs.media_io.VideoMediaIO.async_load",
            ),
            (
                {"type": "audio_url", "audio_url": {"url": "http://bad-url/aud.wav"}},
                "tensorrt_llm.inputs.media_io.AudioMediaIO.async_load",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_loader_exception_propagates(self, mm_tracker, part, patch_target):
        """Exceptions from async loaders must propagate, not be swallowed."""
        with patch(patch_target, new_callable=AsyncMock, side_effect=CustomError):
            result = parse_chat_message_content_part(part, mm_tracker)
            assert result is not None
            with pytest.raises(CustomError):
                await result["data"]

    @pytest.mark.asyncio
    async def test_image_embeds_exception_propagates(self, mm_tracker):
        """Exceptions from image embed decoding must propagate."""
        part = {"type": "image_embeds", "image_embeds": {"data": "notbase64"}}
        with patch(
            "tensorrt_llm.serve.chat_utils.load_base64_image_embeds",
            side_effect=CustomError,
        ):
            result = parse_chat_message_content_part(part, mm_tracker)
            assert result is not None
            with pytest.raises(CustomError):
                await result["data"]


class TestVideoMediaIOMergeInteraction:
    """`VideoMediaIO.merge_kwargs` couples `fps` and `num_frames`."""

    @pytest.mark.parametrize(
        "runtime, expected",
        [
            ({"num_frames": 32}, {"num_frames": 32}),
            ({"fps": 4}, {"fps": 4}),
            ({"num_frames": 32, "fps": 4}, {"num_frames": 32, "fps": 4}),
        ],
    )
    def test_overriding_one_drops_partner_unless_both_given(self, runtime, expected):
        server = {"num_frames": 8, "fps": 1}
        assert VideoMediaIO.merge_kwargs(server, runtime) == expected

    def test_unrelated_request_key_does_not_trigger_drop(self):
        merged = VideoMediaIO.merge_kwargs(
            {"num_frames": 8, "fps": 1},
            {"format": "pt"},
        )
        assert merged == {"num_frames": 8, "fps": 1, "format": "pt"}

    @pytest.mark.parametrize("media_io_cls", [BaseMediaIO, ImageMediaIO, AudioMediaIO])
    def test_non_video_classes_use_plain_shallow_merge(self, media_io_cls):
        merged = media_io_cls.merge_kwargs(
            {"num_frames": 8, "fps": 1},
            {"num_frames": 32},
        )
        assert merged == {"num_frames": 32, "fps": 1}


class TestImageAlphaHandling:
    """RGBA -> RGB has two defensible semantics.

    The caller picks, and the default must stay what every existing caller
    already gets.
    """

    @staticmethod
    def _rgba_png() -> bytes:
        """Build a 3-pixel RGBA fixture.

        One opaque, one half-transparent and one fully transparent pixel, all
        sharing the same stored RGB so the two semantics are separable.
        """
        from io import BytesIO

        from PIL import Image

        im = Image.new("RGBA", (3, 1))
        im.putpixel((0, 0), (200, 30, 30, 255))
        im.putpixel((1, 0), (200, 30, 30, 128))
        im.putpixel((2, 0), (200, 30, 30, 0))
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def test_drop_alpha_matches_pil_and_diffusers(self):
        """Match diffusers.

        Its load_image defaults to image.convert("RGB"), so pipelines ported
        from diffusers need that exact behavior to stay aligned.
        """
        from io import BytesIO

        from PIL import Image

        png = self._rgba_png()
        reference = list(Image.open(BytesIO(png)).convert("RGB").getdata())
        got = ImageMediaIO(format="pil", drop_alpha=True).load_bytes(png)
        assert list(got.getdata()) == reference

    def test_default_composites_and_is_unchanged(self):
        """Keep compositing onto white by default.

        No existing LLM or VLM caller may shift.
        """
        got = ImageMediaIO(format="pil").load_bytes(self._rgba_png())
        assert list(got.getdata()) == [(200, 30, 30), (227, 142, 142), (255, 255, 255)]

    def test_semantics_coincide_for_opaque_images(self):
        """Coincide on opaque media.

        Every committed golden uses opaque media, so the two semantics must be
        bit-identical there.
        """
        from io import BytesIO

        from PIL import Image

        buf = BytesIO()
        Image.new("RGBA", (2, 1), (10, 20, 30, 255)).save(buf, format="PNG")
        png = buf.getvalue()
        composited = ImageMediaIO(format="pil").load_bytes(png)
        dropped = ImageMediaIO(format="pil", drop_alpha=True).load_bytes(png)
        assert list(composited.getdata()) == list(dropped.getdata())

    def test_mode_rgba_preserves_alpha(self):
        got = ImageMediaIO(format="pil", mode="RGBA").load_bytes(self._rgba_png())
        assert got.mode == "RGBA"
        assert list(got.getdata())[2] == (200, 30, 30, 0)

    def test_convert_image_mode_default_is_unchanged(self):
        """The shared helper is publicly exported; its default must not move."""
        from PIL import Image

        im = Image.new("RGBA", (1, 1), (200, 30, 30, 0))
        assert convert_image_mode(im, "RGB").getpixel((0, 0)) == (255, 255, 255)
        assert convert_image_mode(im, "RGB", drop_alpha=True).getpixel((0, 0)) == (200, 30, 30)
