# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import base64
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

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
    # Nonzero samples that differ per channel, so a loader that silences or
    # reorders channels fails the comparison with soundfile.read.
    t = np.arange(frames)[:, None]
    c = np.arange(channels)[None, :]
    pcm = ((t * (2 * c + 3)) % 20000 + 1) * (1 - 2 * (c % 2))
    buffer = BytesIO()
    soundfile.write(buffer, pcm.astype(np.int16), 16000, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def _without_frame_count(flac: bytes) -> bytes:
    # Zero the 36-bit total-samples field of STREAMINFO, which starts in the
    # low nibble of byte 21, so the stream no longer states its length.
    data = bytearray(flac)
    data[21] &= 0xF0
    data[22:26] = bytes(4)
    return bytes(data)


class TestAudioDecodedSizeLimit:
    """Audio decoding stops at `_MAX_AUDIO_SAMPLES` decoded samples."""

    LIMIT = 4000

    @pytest.fixture(autouse=True)
    def small_limit(self, monkeypatch):
        monkeypatch.setattr(media_io_module, "_MAX_AUDIO_SAMPLES", self.LIMIT)

    @pytest.fixture
    def audio_reads(self, monkeypatch):
        """Frame counts passed to each ``SoundFile.read`` call."""
        reads = []
        original_read = soundfile.SoundFile.read

        def recording_read(audio_file, *args, **kwargs):
            reads.append(kwargs.get("frames"))
            return original_read(audio_file, *args, **kwargs)

        monkeypatch.setattr(soundfile.SoundFile, "read", recording_read)
        return reads

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

    def test_audio_over_limit_in_header_is_rejected_before_reading(self, audio_reads):
        encoded = _flac_bytes(frames=self.LIMIT // 2 + 1, channels=2)
        with pytest.raises(ValueError, match="more than 4000 samples"):
            AudioMediaIO().load_bytes(encoded)
        assert audio_reads == []

    def test_audio_with_memory_reservation_obeys_limit(self, audio_reads):
        reservation = Mock()
        reserver = Mock(return_value=reservation)
        loader = AudioMediaIO(decoded_memory_reserver=reserver)
        encoded = _flac_bytes(frames=self.LIMIT // 2, channels=2)
        data, rate = loader.load_base64("audio/flac", base64.b64encode(encoded).decode())
        assert data.shape == (self.LIMIT // 2, 2)
        assert rate == 16000
        reserver.assert_called_once_with(self.LIMIT * 8, self.LIMIT * 8)
        reservation.commit.assert_called_once_with(data.nbytes)
        reservation.release.assert_called_once()
        assert audio_reads == [self.LIMIT // 2]

        oversized = _flac_bytes(frames=self.LIMIT // 2 + 1, channels=2)
        with pytest.raises(ValueError, match="more than 4000 samples"):
            loader.load_bytes(oversized)
        assert audio_reads == [self.LIMIT // 2]
        reserver.assert_called_once()

    def test_audio_of_unknown_length_over_limit_is_rejected(self, audio_reads):
        encoded = _without_frame_count(_flac_bytes(frames=self.LIMIT, channels=2))
        with soundfile.SoundFile(BytesIO(encoded)) as audio_file:
            assert audio_file.frames == media_io_module._SF_UNKNOWN_FRAMES
        with pytest.raises(ValueError, match="more than 4000 samples"):
            AudioMediaIO().load_bytes(encoded)
        assert audio_reads == [self.LIMIT // 2 + 1]

    def test_audio_base64_over_encoded_limit_is_rejected(self, monkeypatch):
        encoded = _flac_bytes(frames=self.LIMIT // 2, channels=2)
        monkeypatch.setattr(media_io_module, "_MAX_RESPONSE_BYTES", len(encoded) // 2)
        with pytest.raises(ValueError, match="Base64 media input exceeds"):
            AudioMediaIO().load_base64("audio/flac", base64.b64encode(encoded).decode())

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
