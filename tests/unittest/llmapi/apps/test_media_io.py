# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensorrt_llm.inputs import MultimodalDataTracker
from tensorrt_llm.inputs.media_io import AudioMediaIO, BaseMediaIO, ImageMediaIO, VideoMediaIO
from tensorrt_llm.metrics.perf_utils import _input_processing_observer
from tensorrt_llm.serve.chat_utils import parse_chat_message_content_part

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media_cls, mime",
    [(ImageMediaIO, "image/png"), (AudioMediaIO, "audio/wav"), (VideoMediaIO, "video/mp4")],
)
@pytest.mark.parametrize("scheme", ["http", "data", "file"])
async def test_media_metrics_separate_download_decode_and_queue(media_cls, mime, scheme):
    media = media_cls()
    observer = MagicMock()
    token = _input_processing_observer.set(observer)
    url, method = {
        "http": ("http://example.test/media", "load_bytes"),
        "data": (f"data:{mime};base64,AA==", "load_base64"),
        "file": ("file:///unused", "load_file"),
    }[scheme]
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with (
                patch.object(BaseMediaIO, "_executor", pool),
                patch.object(media_cls, method, return_value="decoded"),
                patch("tensorrt_llm.inputs.media_io._get_aiohttp_session", new_callable=AsyncMock),
                patch(
                    "tensorrt_llm.inputs.media_io._safe_aiohttp_get",
                    new_callable=AsyncMock,
                    return_value=b"data",
                ),
            ):
                assert await media.async_load(url) == "decoded"
    finally:
        _input_processing_observer.reset(token)
    samples = [args.args for args in observer.call_args_list]
    assert [stage for stage, _, _ in samples] == (["download"] if scheme == "http" else []) + [
        "decode_queue",
        "decode",
    ]
    assert all(seconds >= 0 and modality == mime.split("/")[0] for _, seconds, modality in samples)


@pytest.mark.asyncio
async def test_media_decode_failure_is_recorded():
    observer = MagicMock()
    token = _input_processing_observer.set(observer)
    try:
        with patch.object(ImageMediaIO, "load_base64", side_effect=ValueError("bad media")):
            with pytest.raises(ValueError, match="bad media"):
                await ImageMediaIO().async_load("data:image/png;base64,AA==")
    finally:
        _input_processing_observer.reset(token)
    assert [args.args[0] for args in observer.call_args_list] == ["decode_queue", "decode"]


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
