# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile

import tensorrt_llm.inputs.media_io as media_io_module
from tensorrt_llm.inputs import MultimodalDataTracker
from tensorrt_llm.inputs.media_io import AudioMediaIO, BaseMediaIO, ImageMediaIO, VideoMediaIO
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


def _flac_bytes(frames: int, channels: int = 2) -> bytes:
    buffer = BytesIO()
    soundfile.write(
        buffer,
        np.zeros((frames, channels), dtype=np.int16),
        16000,
        format="FLAC",
        subtype="PCM_16",
    )
    return buffer.getvalue()


class TestAudioDecodedSizeLimit:
    """Audio decoding stops at `_MAX_AUDIO_SAMPLES` decoded samples."""

    LIMIT = 4000

    @pytest.fixture(autouse=True)
    def small_limit(self, monkeypatch):
        monkeypatch.setattr(media_io_module, "_MAX_AUDIO_SAMPLES", self.LIMIT)

    def test_audio_within_limit_matches_soundfile(self, tmp_path):
        encoded = _flac_bytes(frames=self.LIMIT // 2, channels=2)
        path = tmp_path / "ok.flac"
        path.write_bytes(encoded)
        expected, expected_rate = soundfile.read(BytesIO(encoded))
        loader = AudioMediaIO()
        for data, rate in (
            loader.load_bytes(encoded),
            loader.load_base64("audio/flac", base64.b64encode(encoded).decode()),
            loader.load_file(str(path)),
            loader.load_file(path.as_uri()),
        ):
            assert rate == expected_rate
            np.testing.assert_array_equal(data, expected, strict=True)

    def test_audio_over_limit_is_rejected(self, tmp_path):
        # Two channels, so LIMIT // 2 + 1 frames is just over the sample limit.
        encoded = _flac_bytes(frames=self.LIMIT // 2 + 1, channels=2)
        path = tmp_path / "big.flac"
        path.write_bytes(encoded)
        loader = AudioMediaIO()
        with pytest.raises(ValueError, match="more than 4000 samples"):
            loader.load_bytes(encoded)
        with pytest.raises(ValueError, match="more than 4000 samples"):
            loader.load_base64("audio/flac", base64.b64encode(encoded).decode())
        with pytest.raises(ValueError, match="more than 4000 samples"):
            loader.load_file(str(path))

    @pytest.mark.asyncio
    async def test_chat_audio_part_over_limit_is_rejected(self):
        encoded = _flac_bytes(frames=self.LIMIT + 1, channels=1)
        part = {
            "type": "input_audio",
            "input_audio": {"data": base64.b64encode(encoded).decode(), "format": "flac"},
        }
        result = parse_chat_message_content_part(part, MultimodalDataTracker(model_type="dummy"))
        with pytest.raises(ValueError, match="more than 4000 samples"):
            await result["data"]
