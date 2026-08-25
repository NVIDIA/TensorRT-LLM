# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for async media loading in tensorrt_llm.inputs.media_io and utils.

Covers:
- parse_data_uri: the shared RFC 2397 parser behind every `data:` URL path
- async_load_image / async_load_audio with data URLs and file paths
- CPU-bound work is offloaded to a thread pool (event loop not blocked)
- aiohttp session is reused across calls (_get_aiohttp_session)
- MultimodalDataTracker.retrieve_all_async gathers all modalities concurrently
"""

import base64
import tempfile
import threading
from io import BytesIO
from unittest.mock import patch
from urllib.parse import urlparse

import numpy as np
import pytest
import pytest_asyncio
import soundfile
import torch
from PIL import Image

import tensorrt_llm.inputs.media_io as media_io_module
import tensorrt_llm.inputs.utils as utils_module
from tensorrt_llm.inputs.media_io import (
    AudioMediaIO,
    ImageMediaIO,
    NotADataURIError,
    UnsupportedDataURIEncodingError,
    _get_aiohttp_session,
    parse_data_uri,
)
from tensorrt_llm.inputs.utils import (
    MultimodalDataTracker,
    async_load_audio,
    async_load_image,
    load_base64_image,
    load_base64_video,
)

pytestmark = pytest.mark.cpu_only


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_image_data_url() -> str:
    img = Image.new("RGB", (8, 8), color=(100, 150, 200))
    image_buf = BytesIO()
    img.save(image_buf, format="JPEG")
    b64_str = base64.b64encode(image_buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64_str}"


def _make_jpeg_bytes() -> bytes:
    """Build a small in-memory JPEG and return its raw bytes."""
    img = Image.new("RGB", (8, 8), color=(100, 150, 200))
    image_buf = BytesIO()
    img.save(image_buf, format="JPEG")
    return image_buf.getvalue()


def _make_wav_bytes(sample_rate: int = 16000) -> bytes:
    """Build a short in-memory WAV clip and return its raw bytes."""
    time_axis = np.linspace(0, 0.05, int(sample_rate * 0.05), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * 440 * time_axis) * 0.5).astype(np.float32)
    audio_buf = BytesIO()
    soundfile.write(audio_buf, audio_samples, sample_rate, format="WAV")
    return audio_buf.getvalue()


def _data_uri(header: str, payload: bytes) -> str:
    """Assemble a `data:` URI from a header and a to-be-base64-encoded payload."""
    return f"data:{header},{base64.b64encode(payload).decode()}"


def _make_audio_file(tmp_path: str) -> str:
    """Write a short WAV file and return its path."""
    sample_rate = 16000
    time_axis = np.linspace(0, 0.05, int(sample_rate * 0.05), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * 440 * time_axis) * 0.5).astype(np.float32)
    soundfile.write(tmp_path, audio_samples, sample_rate)
    return tmp_path


# ──────────────────────────────────────────────────────────────
# parse_data_uri
# ──────────────────────────────────────────────────────────────


class TestParseDataUri:
    """RFC 2397 parsing, shared by every `data:` URL code path."""

    @pytest.mark.parametrize(
        "url, expected_media_type",
        [
            ("data:image/png;base64,QUJD", "image/png"),
            # `base64` is not required to be the only parameter, nor the first.
            ("data:audio/wav;codecs=opus;base64,QUJD", "audio/wav"),
            ("data:image/jpeg;charset=utf-8;base64,QUJD", "image/jpeg"),
            # RFC 2397 permits omitting the media type entirely.
            ("data:;base64,QUJD", ""),
            # Parameter names are case-insensitive per RFC 2045.
            ("data:text/plain;BASE64,QUJD", "text/plain"),
            ("data:text/plain;Base64,QUJD", "text/plain"),
        ],
    )
    def test_returns_media_type_and_undecoded_payload(self, url, expected_media_type):
        media_type, payload = parse_data_uri(url)
        assert media_type == expected_media_type
        assert payload == "QUJD"  # returned still encoded; callers decode

    def test_base64_must_be_a_whole_parameter_not_a_substring(self):
        """`;name=base64` is a parameter *value*, so the URI is not base64."""
        with pytest.raises(NotImplementedError, match="Only base64 data URLs"):
            parse_data_uri("data:image/png;name=base64,hello")

    def test_empty_payload_parses_rather_than_raising(self):
        """A trailing comma with no data is well-formed; decoding fails later."""
        assert parse_data_uri("data:image/jpeg;base64,") == ("image/jpeg", "")

    def test_empty_spec_is_not_base64(self):
        """`data:,hello` has no parameters at all, so it is not base64."""
        with pytest.raises(NotImplementedError, match="Only base64 data URLs"):
            parse_data_uri("data:,hello")

    def test_non_base64_raises_not_implemented(self):
        """A well-formed but unencoded data URI is unsupported, not malformed."""
        with pytest.raises(NotImplementedError, match="Only base64 data URLs") as excinfo:
            parse_data_uri("data:image/png,hello")
        # NotImplementedError so callers already catching it keep working;
        # ValueError so a bad client payload maps to a client error.
        assert isinstance(excinfo.value, NotImplementedError)
        assert isinstance(excinfo.value, ValueError)

    def test_missing_comma_raises_readable_value_error(self):
        with pytest.raises(ValueError, match="Malformed data URI") as excinfo:
            parse_data_uri("data:image/png;base64")
        message = str(excinfo.value)
        assert "unpack" not in message, "should not leak a tuple-unpacking error"
        assert "data:image/png;base64" in message

    def test_malformed_and_unsupported_are_distinguishable_by_type(self):
        """The two failure modes must not collapse into one bare ValueError.

        Both are ValueErrors so either maps to a client error, but only the
        unsupported-encoding case is an UnsupportedDataURIEncodingError. Callers can
        therefore tell them apart without matching on message text.
        """
        with pytest.raises(ValueError) as malformed:
            parse_data_uri("data:image/png;base64")  # no comma
        with pytest.raises(ValueError) as unsupported:
            parse_data_uri("data:image/png,hello")  # comma, but not base64

        assert type(malformed.value) is ValueError
        assert not isinstance(malformed.value, UnsupportedDataURIEncodingError)
        assert isinstance(unsupported.value, UnsupportedDataURIEncodingError)

    @pytest.mark.parametrize(
        "path",
        [
            "/tmp/holiday,2026.mp4",  # absolute path, comma in the filename
            "./clips/a,b.mp4",  # relative path
            "holiday,2026.mp4",  # bare filename
            "/tmp/plain.mp4",  # no comma at all
        ],
    )
    def test_non_data_scheme_is_rejected_as_such(self, path):
        """A filesystem path must not be diagnosed as bad base64.

        Without a scheme check, a path containing a "," is split like a data
        URI and reported as an encoding problem, which is misleading.
        """
        with pytest.raises(NotADataURIError, match="Expected a 'data:' URI") as excinfo:
            parse_data_uri(path)
        assert not isinstance(excinfo.value, UnsupportedDataURIEncodingError)
        assert path in str(excinfo.value)

    @pytest.mark.parametrize(
        "url",
        [
            "data:image/jpeg;charset=utf-8;base64,QUJD",
            "data:;base64,QUJD",
            "data:audio/wav;codecs=opus;base64,QUJD",
            "data:image/png,hello",
            "data:image/png;base64",
            "/tmp/holiday,2026.mp4",
        ],
    )
    def test_accepts_str_and_parseresult_identically(self, url):
        """Callers holding a parsed URL pass it through, avoiding a re-serialize."""

        def outcome(value):
            try:
                return ("ok", parse_data_uri(value))
            except ValueError as exc:
                return (type(exc), str(exc))

        assert outcome(url) == outcome(urlparse(url))


class TestUnsupportedDataURIEncodingError:
    def test_subclasses_both_not_implemented_error_and_value_error(self):
        assert issubclass(UnsupportedDataURIEncodingError, NotImplementedError)
        assert issubclass(UnsupportedDataURIEncodingError, ValueError)

    def test_is_caught_by_either_except_clause(self):
        for exception_type in (NotImplementedError, ValueError):
            with pytest.raises(exception_type):
                raise UnsupportedDataURIEncodingError("boom")


# ──────────────────────────────────────────────────────────────
# load_base64_image / load_base64_video — the synchronous helpers
# ──────────────────────────────────────────────────────────────


class TestLoadBase64Image:
    """`load_base64_image` takes an already-parsed URL, not a string."""

    def test_parameterized_data_url_loads(self):
        url = _data_uri("image/jpeg;charset=utf-8;base64", _make_jpeg_bytes())
        assert isinstance(load_base64_image(urlparse(url)), Image.Image)

    def test_plain_data_url_still_loads(self):
        url = _data_uri("image/jpeg;base64", _make_jpeg_bytes())
        assert isinstance(load_base64_image(urlparse(url)), Image.Image)

    def test_empty_media_type_loads(self):
        url = _data_uri(";base64", _make_jpeg_bytes())
        assert isinstance(load_base64_image(urlparse(url)), Image.Image)

    @pytest.mark.parametrize("parameter", ["BASE64", "Base64"])
    def test_case_variants_load(self, parameter):
        url = _data_uri(f"image/jpeg;{parameter}", _make_jpeg_bytes())
        assert isinstance(load_base64_image(urlparse(url)), Image.Image)

    def test_non_base64_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs") as excinfo:
            load_base64_image(urlparse("data:image/png,hello"))
        # Both, so callers catching either builtin behave sensibly.
        assert isinstance(excinfo.value, NotImplementedError)
        assert isinstance(excinfo.value, ValueError)

    def test_base64_as_parameter_value_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs"):
            load_base64_image(urlparse("data:image/png;name=base64,hello"))

    def test_empty_spec_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs") as excinfo:
            load_base64_image(urlparse("data:,hello"))
        assert isinstance(excinfo.value, NotImplementedError)
        assert isinstance(excinfo.value, ValueError)

    def test_missing_comma_raises_value_error(self):
        with pytest.raises(ValueError, match="Malformed data URI"):
            load_base64_image(urlparse("data:image/png;base64"))

    def test_parsed_url_is_not_re_serialized(self, monkeypatch):
        """The ParseResult is handed to the parser as-is, not urlunparse'd back."""
        monkeypatch.setattr(
            media_io_module,
            "urlparse",
            lambda _: pytest.fail("parse_data_uri re-parsed an already-parsed URL"),
        )
        url = _data_uri("image/jpeg;base64", _make_jpeg_bytes())
        assert isinstance(load_base64_image(urlparse(url)), Image.Image)


class TestLoadBase64Video:
    """`load_base64_video` takes the URL string and returns raw bytes."""

    PAYLOAD = b"\x00\x01\x02fake-mp4-bytes"

    def test_parameterized_data_url_loads(self):
        url = _data_uri("video/mp4;codecs=avc1;base64", self.PAYLOAD)
        assert load_base64_video(url) == self.PAYLOAD

    def test_plain_data_url_still_loads(self):
        url = _data_uri("video/mp4;base64", self.PAYLOAD)
        assert load_base64_video(url) == self.PAYLOAD

    def test_empty_media_type_loads(self):
        url = _data_uri(";base64", self.PAYLOAD)
        assert load_base64_video(url) == self.PAYLOAD

    @pytest.mark.parametrize("parameter", ["BASE64", "Base64"])
    def test_case_variants_load(self, parameter):
        url = _data_uri(f"video/mp4;{parameter}", self.PAYLOAD)
        assert load_base64_video(url) == self.PAYLOAD

    def test_empty_payload_decodes_to_empty_bytes(self):
        """Matches pre-existing behaviour: parsing succeeds, payload is empty."""
        assert load_base64_video("data:video/mp4;base64,") == b""

    def test_non_base64_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs") as excinfo:
            load_base64_video("data:video/mp4,hello")
        assert isinstance(excinfo.value, NotImplementedError)
        assert isinstance(excinfo.value, ValueError)

    def test_base64_as_parameter_value_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs"):
            load_base64_video("data:video/mp4;name=base64,hello")

    def test_empty_spec_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs") as excinfo:
            load_base64_video("data:,hello")
        assert isinstance(excinfo.value, NotImplementedError)
        assert isinstance(excinfo.value, ValueError)

    def test_missing_comma_raises_value_error(self):
        with pytest.raises(ValueError, match="Malformed data URI"):
            load_base64_video("data:video/mp4;base64")


# ──────────────────────────────────────────────────────────────
# async_load_image
# ──────────────────────────────────────────────────────────────


class TestAsyncLoadImage:
    @pytest.mark.asyncio
    async def test_load_image_from_parameterized_data_url(self):
        """Extra RFC 2397 parameters ahead of `base64` must not break parsing."""
        url = _data_uri("image/jpeg;charset=utf-8;base64", _make_jpeg_bytes())
        image = await async_load_image(url, format="pil")
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    @pytest.mark.asyncio
    async def test_load_image_from_data_url_without_media_type(self):
        url = _data_uri(";base64", _make_jpeg_bytes())
        image = await async_load_image(url, format="pil")
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    @pytest.mark.asyncio
    async def test_load_image_from_non_base64_data_url_raises(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs"):
            await async_load_image("data:image/png,hello", format="pil")

    @pytest.mark.asyncio
    async def test_load_image_from_data_url_pil(self):
        url = _make_image_data_url()
        image = await async_load_image(url, format="pil")
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    @pytest.mark.asyncio
    async def test_load_image_from_data_url_pt(self):
        url = _make_image_data_url()
        image_tensor = await async_load_image(url, format="pt")
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape[0] == 3  # C x H x W

    @pytest.mark.asyncio
    async def test_load_image_pil_input_passthrough(self):
        """Already-decoded PIL image should be returned as RGB."""
        grayscale_img = Image.new("L", (4, 4))
        rgb_image = await async_load_image(grayscale_img, format="pil")
        assert isinstance(rgb_image, Image.Image)
        assert rgb_image.mode == "RGB"

    @pytest.mark.asyncio
    async def test_cpu_work_runs_in_executor(self):
        """PIL decoding must not run on the event loop thread."""
        event_loop_thread_id = threading.current_thread().ident
        worker_thread_ids = []

        original_load = utils_module._load_and_convert_image

        def tracking_load(*args, **kwargs):
            worker_thread_ids.append(threading.current_thread().ident)
            return original_load(*args, **kwargs)

        url = _make_image_data_url()
        with patch.object(utils_module, "_load_and_convert_image", tracking_load):
            await async_load_image(url, format="pil")

        assert len(worker_thread_ids) == 1
        assert worker_thread_ids[0] != event_loop_thread_id, (
            "PIL decoding ran on the event loop thread — event loop is being blocked"
        )


# ──────────────────────────────────────────────────────────────
# async_load_audio
# ──────────────────────────────────────────────────────────────


class TestAsyncLoadAudio:
    @pytest.mark.asyncio
    async def test_load_audio_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = _make_audio_file(f.name)
        audio_array, sample_rate = await async_load_audio(wav_path)
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000  # matches the sr=16000 used in _make_audio_file

    @pytest.mark.asyncio
    async def test_cpu_work_runs_in_executor(self):
        """soundfile.read must not run on the event loop thread."""
        event_loop_thread_id = threading.current_thread().ident
        worker_thread_ids = []

        original_read = soundfile.read

        def tracking_read(*args, **kwargs):
            worker_thread_ids.append(threading.current_thread().ident)
            return original_read(*args, **kwargs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = _make_audio_file(f.name)

        with patch("tensorrt_llm.inputs.utils.soundfile.read", tracking_read):
            await async_load_audio(wav_path)

        assert len(worker_thread_ids) == 1
        assert worker_thread_ids[0] != event_loop_thread_id, (
            "soundfile.read ran on the event loop thread — event loop is being blocked"
        )


# ──────────────────────────────────────────────────────────────
# BaseMediaIO.async_load — data URLs
# ──────────────────────────────────────────────────────────────


class TestMediaIODataUrl:
    """`BaseMediaIO.async_load` reaches `data:` URLs through the shared parser."""

    @pytest.mark.asyncio
    async def test_load_audio_from_parameterized_data_url(self):
        # The payload is PCM WAV; `codecs=opus` is present only to exercise a
        # second parameter ahead of `base64`, which used to be misread as the
        # encoding and rejected.
        url = _data_uri("audio/wav;codecs=opus;base64", _make_wav_bytes())
        audio_array, sample_rate = await AudioMediaIO().async_load(url)
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000  # matches the default in _make_wav_bytes

    @pytest.mark.asyncio
    async def test_load_audio_from_data_url_without_media_type(self):
        url = _data_uri(";base64", _make_wav_bytes())
        audio_array, sample_rate = await AudioMediaIO().async_load(url)
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000

    @pytest.mark.asyncio
    async def test_load_audio_from_non_base64_data_url_raises(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs"):
            await AudioMediaIO().async_load("data:audio/wav,hello")

    @pytest.mark.asyncio
    async def test_load_audio_from_data_url_without_comma_raises(self):
        with pytest.raises(ValueError, match="Malformed data URI"):
            await AudioMediaIO().async_load("data:audio/wav;base64")

    @pytest.mark.asyncio
    async def test_base64_as_parameter_value_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Only base64 data URLs"):
            await AudioMediaIO().async_load("data:audio/wav;name=base64,hello")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("parameter", ["BASE64", "Base64"])
    async def test_case_variants_load(self, parameter):
        url = _data_uri(f"audio/wav;{parameter}", _make_wav_bytes())
        audio_array, _ = await AudioMediaIO().async_load(url)
        assert isinstance(audio_array, np.ndarray)

    @pytest.mark.asyncio
    async def test_empty_media_type_reaches_every_subclass(self):
        """No `load_base64` override dispatches on media_type, so "" is safe."""
        image = await ImageMediaIO(format="pil").async_load(
            _data_uri(";base64", _make_jpeg_bytes())
        )
        assert isinstance(image, Image.Image)

        audio_array, _ = await AudioMediaIO().async_load(_data_uri(";base64", _make_wav_bytes()))
        assert isinstance(audio_array, np.ndarray)


# ──────────────────────────────────────────────────────────────
# Session reuse
# ──────────────────────────────────────────────────────────────


class TestSessionReuse:
    @pytest_asyncio.fixture(autouse=True)
    async def reset_global_session(self):
        media_io_module._global_aiohttp_session = None
        yield
        if media_io_module._global_aiohttp_session is not None:
            await media_io_module._global_aiohttp_session.close()
        media_io_module._global_aiohttp_session = None

    @pytest.mark.asyncio
    async def test_same_session_returned_on_repeated_calls(self):
        first_session = await _get_aiohttp_session()
        second_session = await _get_aiohttp_session()
        assert first_session is second_session, (
            "Expected the same ClientSession object to be returned"
        )

    @pytest.mark.asyncio
    async def test_new_session_created_after_close(self):
        first_session = await _get_aiohttp_session()
        await first_session.close()
        second_session = await _get_aiohttp_session()
        assert first_session is not second_session, (
            "Expected a fresh session after the old one was closed"
        )
        assert not second_session.closed


# ──────────────────────────────────────────────────────────────
# MultimodalDataTracker.retrieve_all_async — concurrency
#
# We bypass add_data (which requires a registry-registered model type)
# and insert coroutines directly into _data / _embeddings to isolate
# the gather logic being tested.
# ──────────────────────────────────────────────────────────────


class TestRetrieveAllAsync:
    def _make_tracker(self) -> MultimodalDataTracker:
        return MultimodalDataTracker(model_type="test_model")

    def _inject(
        self,
        tracker: MultimodalDataTracker,
        modality: str,
        values: list,
        *,
        is_embedding: bool = False,
    ):
        """Directly insert coroutines into the tracker without registry checks."""
        target = tracker._embeddings if is_embedding else tracker._data
        for val in values:

            async def _coro(x=val):
                return x

            target[modality].append(_coro())

    @pytest.mark.asyncio
    async def test_single_modality_returns_correct_data(self):
        image_values = [1, 2, 3]
        tracker = self._make_tracker()
        self._inject(tracker, "image", image_values)

        data, embeddings = await tracker.retrieve_all_async()
        assert data == {"image": image_values}
        assert embeddings is None

    @pytest.mark.asyncio
    async def test_multiple_modalities_returns_correct_data(self):
        image_values = ["img1", "img2"]
        video_values = ["vid1"]
        tracker = self._make_tracker()
        self._inject(tracker, "image", image_values)
        self._inject(tracker, "video", video_values)

        data, _ = await tracker.retrieve_all_async()
        assert data["image"] == image_values
        assert data["video"] == video_values

    @pytest.mark.asyncio
    async def test_embeddings_returned_separately(self):
        image_data = ["img"]
        embedding_data = ["emb"]
        tracker = self._make_tracker()
        self._inject(tracker, "image", image_data, is_embedding=False)
        self._inject(tracker, "image", embedding_data, is_embedding=True)

        data, embeddings = await tracker.retrieve_all_async()
        assert data == {"image": image_data}
        assert embeddings == {"image": embedding_data}

    @pytest.mark.asyncio
    async def test_empty_tracker_returns_none(self):
        tracker = self._make_tracker()
        data, embeddings = await tracker.retrieve_all_async()
        assert data is None
        assert embeddings is None
