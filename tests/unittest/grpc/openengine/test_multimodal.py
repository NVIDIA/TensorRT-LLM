# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenEngine multimodal input mapping and discovery."""

import asyncio
import sys
import threading
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

grpc = pytest.importorskip(
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.67.1,<2")'
)
from conftest import AbortError, FakeServicerContext  # noqa: E402

import tensorrt_llm.grpc.openengine.request_mapping as request_mapping  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import generation_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.capabilities import MediaCapabilities  # noqa: E402
from tensorrt_llm.grpc.openengine.errors import UnsupportedFeatureError  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402
from tensorrt_llm.inputs import media_io  # noqa: E402
from tensorrt_llm.inputs.media_io import MediaFetchError  # noqa: E402
from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData  # noqa: E402
from tensorrt_llm.inputs.registry import (  # noqa: E402
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    BaseMultimodalInputProcessor,
)

pytestmark = pytest.mark.cpu_only


def _ignore_synthetic_decoded_size(monkeypatch) -> None:
    handler = request_mapping._MEDIA_HANDLERS[generation_pb2.MODALITY_IMAGE]
    monkeypatch.setitem(
        request_mapping._MEDIA_HANDLERS,
        generation_pb2.MODALITY_IMAGE,
        request_mapping._MediaHandler(
            handler.name,
            handler.loader_factory,
            lambda _decoded: 0,
            handler.runtime_available,
            handler.reserves_decoded_bytes,
        ),
    )


def _set_video_runtime_available(monkeypatch, available: bool) -> None:
    handler = request_mapping._MEDIA_HANDLERS[generation_pb2.MODALITY_VIDEO]
    monkeypatch.setitem(
        request_mapping._MEDIA_HANDLERS,
        generation_pb2.MODALITY_VIDEO,
        request_mapping._MediaHandler(
            handler.name,
            handler.loader_factory,
            handler.decoded_bytes,
            lambda: available,
            handler.reserves_decoded_bytes,
        ),
    )


class _ImageConfig:
    # Composite configs can expose their text model type on the instance; the
    # top-level class attribute is the key used by the multimodal registry.
    model_type = "openengine_test_image"

    def __init__(self) -> None:
        self.model_type = "openengine_test_text"


class _ImageInputProcessor(BaseMultimodalInputProcessor):
    """Minimal concrete processor used only for capability detection."""

    def __init__(self, config: _ImageConfig) -> None:
        self._config = config

    @property
    def processor(self):
        return None

    @property
    def tokenizer(self):
        return None

    @property
    def config(self):
        return self._config

    @property
    def dtype(self):
        return None

    def call_with_text_prompt(self, inputs, sampling_params):
        raise NotImplementedError

    def get_preferred_media_io_kwargs(self):
        return {
            "image": {"format": "np"},
            "video": {"format": "np", "num_frames": 4},
        }


class _AudioInputProcessor:
    """Registered processor that does not use BaseMultimodalInputProcessor."""

    _registered_model_type = _ImageConfig.model_type

    def __init__(self, config: _ImageConfig) -> None:
        self.config = config

    def __call__(self, inputs, sampling_params):
        raise NotImplementedError


class _NonCallableRegisteredInputProcessor:
    """Malformed registration used to verify capability discovery fails closed."""

    _registered_model_type = _ImageConfig.model_type

    def __init__(self, config: _ImageConfig) -> None:
        self.config = config


class _Tokenizer:
    def convert_ids_to_tokens(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str]:
        del skip_special_tokens
        return [f"token-{token_id}" for token_id in token_ids]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(f"token-{token_id}" for token_id in token_ids)


class _ResultHandle:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.finished = False
        self.prompt_token_ids = []
        self.outputs = []
        self.cached_tokens = 0
        self.error = None
        self.aborted = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        self.finished = self._result.finished
        self.prompt_token_ids = self._result.prompt_token_ids
        self.outputs = self._result.outputs
        self.cached_tokens = self._result.cached_tokens
        self.error = self._result.error
        yield self._result

    def abort(self) -> None:
        self.aborted = True


class _ImageLlm:
    def __init__(self, input_processor, config) -> None:
        self.input_processor = input_processor
        self._hf_model_config = config
        self.args = SimpleNamespace(guided_decoding_backend=None)
        self.tokenizer = _Tokenizer()
        output = SimpleNamespace(
            index=0,
            token_ids=[99],
            text="caption",
            logprobs=[],
            prompt_logprobs=[],
            finish_reason="length",
            stop_reason=None,
        )
        self.result_handle = _ResultHandle(
            SimpleNamespace(
                # Represents expansion of one placeholder into five image tokens.
                prompt_token_ids=[1, 2, 3, 4, 5, 6],
                outputs=[output],
                cached_tokens=0,
                error=None,
                finished=True,
            )
        )
        self.generate_kwargs = None

    def generate_async(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.result_handle


@pytest.fixture
def image_llm(monkeypatch):
    _set_video_runtime_available(monkeypatch, True)
    config = _ImageConfig()
    input_processor = _ImageInputProcessor(config)
    seen_model_types = []

    def is_valid(model_type, modality):
        seen_model_types.append(model_type)
        return model_type == _ImageConfig.model_type and modality == "image"

    monkeypatch.setattr(MULTIMODAL_PLACEHOLDER_REGISTRY, "is_valid", is_valid)
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda model_type, modality: "<image_{}>",
    )
    return (
        SimpleNamespace(
            _hf_model_config=config,
            input_processor=input_processor,
        ),
        seen_model_types,
    )


@pytest.fixture
def image_video_llm(monkeypatch):
    _set_video_runtime_available(monkeypatch, True)
    config = _ImageConfig()
    input_processor = _ImageInputProcessor(config)
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "is_valid",
        lambda model_type, modality: (
            model_type == _ImageConfig.model_type and modality in {"image", "video"}
        ),
    )
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda model_type, modality: f"<{modality}_{{}}>",
    )
    return SimpleNamespace(
        _hf_model_config=config,
        input_processor=input_processor,
    )


@pytest.fixture
def image_video_audio_llm(monkeypatch):
    _set_video_runtime_available(monkeypatch, True)
    config = _ImageConfig()
    input_processor = _ImageInputProcessor(config)
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "is_valid",
        lambda model_type, modality: (
            model_type == _ImageConfig.model_type and modality in {"image", "video", "audio"}
        ),
    )
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda model_type, modality: f"<{modality}_{{}}>",
    )
    return SimpleNamespace(
        _hf_model_config=config,
        input_processor=input_processor,
    )


def test_media_capability_requires_the_loaded_multimodal_processor(image_llm):
    llm, seen_model_types = image_llm

    assert request_mapping.media_capabilities_for_model(llm).modalities == {"image"}
    assert seen_model_types == [_ImageConfig.model_type] * 3
    assert (
        request_mapping.media_capabilities_for_model(
            SimpleNamespace(_hf_model_config=llm._hf_model_config, input_processor=object())
        ).modalities
        == set()
    )


def test_media_capability_accepts_a_registered_non_base_processor(monkeypatch):
    config = _ImageConfig()
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "is_valid",
        lambda model_type, modality: model_type == _ImageConfig.model_type and modality == "audio",
    )
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda model_type, modality: "",
    )
    llm = SimpleNamespace(
        _hf_model_config=config,
        input_processor=_AudioInputProcessor(config),
    )

    assert request_mapping.media_capabilities_for_model(llm).modalities == {"audio"}


def test_media_capability_rejects_a_registered_non_callable_processor(monkeypatch):
    config = _ImageConfig()
    monkeypatch.setattr(MULTIMODAL_PLACEHOLDER_REGISTRY, "is_valid", lambda *_args: True)
    llm = SimpleNamespace(
        _hf_model_config=config,
        input_processor=_NonCallableRegisteredInputProcessor(config),
    )

    assert request_mapping.media_capabilities_for_model(llm).modalities == set()


def test_media_capabilities_are_hashable():
    first = MediaCapabilities({"image": "<image_{}>"})
    second = MediaCapabilities({"image": "<image_{}>"})

    assert first == second
    assert hash(first) == hash(second)


def test_media_capability_requires_the_runtime_decoder(image_video_llm, monkeypatch):
    _set_video_runtime_available(monkeypatch, False)

    assert request_mapping.media_capabilities_for_model(image_video_llm).modalities == {"image"}


def test_video_runtime_availability_requires_importable_opencv(monkeypatch):
    monkeypatch.setattr(request_mapping.importlib, "import_module", lambda _name: object())
    assert request_mapping._video_runtime_available() is True


@pytest.mark.parametrize("error", [ImportError(), OSError()])
def test_video_runtime_unavailable_when_opencv_fails_to_load(monkeypatch, error):
    def missing(_name):
        raise error

    monkeypatch.setattr(request_mapping.importlib, "import_module", missing)
    assert request_mapping._video_runtime_available() is False


def test_unknown_decoded_image_type_is_rejected():
    with pytest.raises(TypeError, match="unsupported decoded image type"):
        request_mapping._decoded_image_bytes(object())


def test_decoded_video_bytes_include_frames_and_audio():
    video = VideoData(
        frames=np.zeros((2, 3, 4, 3), dtype=np.uint8),
        metadata={},
        audio=AudioData(samples=np.zeros(8, dtype=np.float32), sample_rate=16000),
    )

    assert request_mapping._decoded_video_bytes(video) == 2 * 3 * 4 * 3 + 8 * 4
    with pytest.raises(TypeError, match="unsupported decoded video type"):
        request_mapping._decoded_video_bytes(object())


def test_decoded_audio_bytes_count_samples_and_reject_unknown_types():
    assert request_mapping._decoded_audio_bytes((np.zeros(8, dtype=np.float32), 16000)) == 8 * 4
    with pytest.raises(TypeError, match="unsupported decoded audio type"):
        request_mapping._decoded_audio_bytes(object())


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("source", ["bytes", "base64", "file"])
def test_audio_reserves_before_decoding_and_preserves_samples(
    channels, source, tmp_path, monkeypatch
):
    import base64

    import soundfile

    samples = np.arange(32 * channels, dtype=np.float64).reshape(32, channels) / 128
    encoded = BytesIO()
    soundfile.write(encoded, samples, 16000, format="FLAC")
    data = encoded.getvalue()
    expected, _ = soundfile.read(BytesIO(data))
    budget = request_mapping._DecodedMediaBudget(samples.nbytes)
    original_read = soundfile.SoundFile.read

    def checked_read(audio, *args, **kwargs):
        assert (budget._used, budget._reserved_retained) == (samples.nbytes, samples.nbytes)
        assert kwargs["out"].nbytes == samples.nbytes
        return original_read(audio, *args, **kwargs)

    monkeypatch.setattr(soundfile.SoundFile, "read", checked_read)
    loader = media_io.AudioMediaIO(decoded_memory_reserver=budget.reserve)
    if source == "bytes":
        decoded, rate = loader.load_bytes(data)
    elif source == "base64":
        decoded, rate = loader.load_base64("audio/flac", base64.b64encode(data).decode())
    else:
        path = tmp_path / "audio.flac"
        path.write_bytes(data)
        decoded, rate = loader.load_file(path.as_uri())
    np.testing.assert_array_equal(decoded, expected)
    assert rate == 16000
    assert (budget._used, budget._retained, budget._reserved_retained) == (
        samples.nbytes,
        samples.nbytes,
        0,
    )


@pytest.mark.parametrize("outcome", ["oversized", "failure", "truncated", "cancelled"])
def test_audio_budget_rejects_before_allocation_and_cleans_up(outcome, monkeypatch):
    import soundfile

    encoded = BytesIO()
    soundfile.write(encoded, np.zeros((32, 2)), 16000, format="FLAC")
    budget = request_mapping._DecodedMediaBudget(511 if outcome == "oversized" else 512)
    loader = media_io.AudioMediaIO(decoded_memory_reserver=budget.reserve)

    def read(audio, *args, **kwargs):
        if outcome == "oversized":
            pytest.fail("oversized audio must be rejected before decoding")
        if outcome == "failure":
            raise RuntimeError("decode failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        return kwargs["out"][:0]

    monkeypatch.setattr(soundfile.SoundFile, "read", read)
    if outcome == "oversized":
        monkeypatch.setattr(
            np, "empty", lambda *args, **kwargs: pytest.fail("allocated oversized audio")
        )
    error = {
        "oversized": ValueError,
        "failure": RuntimeError,
        "truncated": ValueError,
        "cancelled": asyncio.CancelledError,
    }[outcome]
    with pytest.raises(error):
        loader.load_bytes(encoded.getvalue())
    assert (budget._used, budget._retained, budget._reserved_retained) == (0, 0, 0)


@pytest.mark.asyncio
async def test_audio_request_uses_one_shared_budget_without_double_charging(
    image_video_audio_llm, monkeypatch
):
    import soundfile

    encoded = BytesIO()
    soundfile.write(encoded, np.zeros(32), 16000, format="FLAC")
    monkeypatch.setattr(request_mapping, "_MAX_TOTAL_DECODED_MEDIA_BYTES", 256)
    item = generation_pb2.MediaItem(
        modality=generation_pb2.MODALITY_AUDIO, raw_bytes=encoded.getvalue(), mime_type="audio/flac"
    )
    request = generation_pb2.GenerateRequest(prompt="transcribe", media=[item])
    mapped = await request_mapping._input_from_request(request, image_video_audio_llm)
    assert mapped["multi_modal_data"]["audio"][0][0].nbytes == 256
    request.media.append(item)
    with pytest.raises(ValueError, match="decoded media exceeds"):
        await request_mapping._input_from_request(request, image_video_audio_llm)


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_encoder", [False, True])
async def test_gemma4_audio_capability_requires_checkpoint_encoder(audio_encoder, monkeypatch):
    class GemmaConfig:
        model_type = "gemma4"
        audio_config = object() if audio_encoder else None

    llm = SimpleNamespace(input_processor=_ImageInputProcessor(GemmaConfig()))
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "is_valid",
        lambda model_type, modality: modality in {"image", "audio"},
    )
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda model_type, modality: f"<{modality}_{{}}>",
    )
    capabilities = request_mapping.media_capabilities_for_model(llm)
    assert "image" in capabilities.modalities
    assert ("audio" in capabilities.modalities) == audio_encoder
    if not audio_encoder:
        request = generation_pb2.GenerateRequest(
            prompt="transcribe",
            media=[
                generation_pb2.MediaItem(
                    modality=generation_pb2.MODALITY_AUDIO,
                    raw_bytes=b"unused",
                    mime_type="audio/wav",
                )
            ],
        )
        with pytest.raises(UnsupportedFeatureError, match="audio"):
            await request_mapping._input_from_request(request, llm, capabilities)


def test_decoded_media_budget_settles_peak_to_retained_bytes():
    budget = request_mapping._DecodedMediaBudget(20, peak_limit=20)

    first = budget.reserve(15, 5)
    first.commit(5)
    assert (budget._used, budget._retained, budget._reserved_retained) == (5, 5, 0)
    second = budget.reserve(15, 5)
    second.release()
    assert (budget._used, budget._retained, budget._reserved_retained) == (5, 5, 0)


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "resampled",
        "oversized",
        "decode_failure",
        "concat_failure",
        "no_audio",
        "cancelled",
    ],
)
def test_embedded_audio_reserves_chunks_and_concatenation(outcome, monkeypatch):
    budget = request_mapping._DecodedMediaBudget(27 if outcome == "oversized" else 35)
    converted = []
    concatenated = []
    grabs = iter([True])
    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=lambda _key: 1,
        grab=lambda: next(grabs, False),
        retrieve=lambda _scratch: (True, np.zeros((1, 1, 3), dtype=np.uint8)),
        release=lambda: None,
    )
    monkeypatch.setattr(
        media_io,
        "_get_cv2",
        lambda: SimpleNamespace(
            VideoCapture=lambda _source: capture,
            CAP_PROP_FRAME_COUNT=0,
            CAP_PROP_FPS=1,
            COLOR_BGR2RGB=2,
            cvtColor=lambda _source, _code, dst: dst.fill(0),
        ),
    )

    def audio_frame(index):
        def to_ndarray():
            converted.append(index)
            if outcome == "decode_failure" and index == 1:
                raise RuntimeError("audio decode failed")
            if outcome == "cancelled" and index == 0:
                budget.cancel()
            return np.full((1, 4), index, dtype=np.float32)

        return SimpleNamespace(
            samples=4,
            layout=SimpleNamespace(channels=("mono",)),
            to_ndarray=to_ndarray,
        )

    stream = SimpleNamespace(
        rate=16000,
        format=SimpleNamespace(name="s16" if outcome == "resampled" else "fltp"),
        layout=SimpleNamespace(name="mono"),
    )
    container = SimpleNamespace(
        streams=SimpleNamespace(audio=[] if outcome == "no_audio" else [stream]),
        decode=lambda _stream: [audio_frame(0), audio_frame(1)],
    )
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(
            open=lambda _source: nullcontext(container),
            AudioResampler=lambda **_kwargs: SimpleNamespace(resample=lambda frame: [frame]),
        ),
    )
    monkeypatch.setenv("TRTLLM_ENABLE_PYAV", "1")
    concatenate = np.concatenate

    def checked_concatenate(chunks, **kwargs):
        concatenated.append(True)
        assert (budget._used, budget._retained, budget._reserved_retained) == (67, 0, 35)
        if outcome == "concat_failure":
            raise MemoryError("audio concatenation failed")
        return concatenate(chunks, **kwargs)

    monkeypatch.setattr(media_io.np, "concatenate", checked_concatenate)

    def load():
        return media_io._load_video_by_cv2(
            "video.mp4",
            fps=-1,
            format="np",
            extract_audio=True,
            decoded_memory_reserver=budget.reserve,
        )

    if outcome in ("success", "resampled", "no_audio"):
        video = load()
        retained = media_io._decoded_video_bytes(video)
        assert retained == (3 if outcome == "no_audio" else 35)
        assert (budget._used, budget._retained, budget._reserved_retained) == (
            retained,
            retained,
            0,
        )
        if outcome != "no_audio":
            np.testing.assert_array_equal(video.audio.samples, [0] * 4 + [1] * 4)
            assert video.audio.sample_rate == 16000
    else:
        with pytest.raises(MemoryError if outcome == "concat_failure" else ValueError):
            load()
        assert (budget._used, budget._retained, budget._reserved_retained) == (0, 0, 0)
    if outcome in ("oversized", "cancelled"):
        assert converted == [0]
        assert not concatenated


def test_video_retained_limit_does_not_reject_1080p_conversion_peak():
    budget = request_mapping._DecodedMediaBudget(512 * 1024 * 1024)
    values = 10 * 1080 * 1920 * 3
    reservation = budget.reserve(values * 9, values * 4)
    reservation.commit(values * 4)


@pytest.mark.parametrize("workers", [1, 8])
def test_three_720p_videos_fit_independently_of_load_concurrency(workers):
    budget = request_mapping._DecodedMediaBudget(512 * 1024 * 1024)
    values = 10 * 720 * 1280 * 3

    def decode(_index):
        reservation = budget.reserve(values * 9, values * 4)
        reservation.commit(values * 4)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(decode, range(3)))


@pytest.mark.parametrize("cancel", [False, True])
def test_video_peak_reservation_waits_for_settlement_or_cancellation(cancel, monkeypatch):
    budget = request_mapping._DecodedMediaBudget(20, peak_limit=20)
    first = budget.reserve(15, 5)
    waiting = threading.Event()
    wait = budget._condition.wait

    def observed_wait():
        waiting.set()
        return wait()

    monkeypatch.setattr(budget._condition, "wait", observed_wait)
    with ThreadPoolExecutor(max_workers=1) as pool:
        second = pool.submit(budget.reserve, 15, 5)
        try:
            assert waiting.wait(timeout=5)
            assert not second.done()
            if cancel:
                budget.cancel()
                with pytest.raises(ValueError, match="cancelled"):
                    second.result(timeout=5)
            else:
                first.commit(5)
                second.result(timeout=5).commit(5)
        finally:
            budget.cancel()
            first.release()


def test_media_capability_rejects_a_malformed_placeholder_at_startup(image_llm, monkeypatch):
    llm, _ = image_llm
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda model_type, modality: "<image_{missing}>",
    )

    with pytest.raises(RuntimeError, match="invalid multimodal placeholder"):
        request_mapping.media_capabilities_for_model(llm)


@pytest.mark.asyncio
async def test_handler_table_drives_capability_and_mapping(image_llm, monkeypatch):
    llm, _ = image_llm

    def video_loader_factory(_llm, _budget):
        async def load(item, index):
            return f"video-{index}:{bytes(item.raw_bytes).decode()}"

        return load

    monkeypatch.setitem(
        request_mapping._MEDIA_HANDLERS,
        generation_pb2.MODALITY_VIDEO,
        request_mapping._MediaHandler("video", video_loader_factory, lambda decoded: len(decoded)),
    )
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "is_valid",
        lambda model_type, modality: (
            model_type == _ImageConfig.model_type and modality in {"image", "video"}
        ),
    )
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda model_type, modality: f"<{modality}_{{}}>",
    )

    capabilities = request_mapping.media_capabilities_for_model(llm)
    assert capabilities.modalities == {"image", "video"}

    request = generation_pb2.GenerateRequest(
        prompt="video",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                raw_bytes=b"clip",
            )
        ],
    )
    mapped = await request_mapping._input_from_request(request, llm, capabilities)

    assert mapped["multi_modal_data"] == {"video": ["video-0:clip"]}
    assert mapped["mm_item_order"] == [
        {"modality": "video", "index": 0, "placeholder": "<video_1>"}
    ]


@pytest.mark.asyncio
async def test_text_only_input_mapping_is_unchanged():
    llm = SimpleNamespace()

    prompt = generation_pb2.GenerateRequest(prompt="hello")
    token_ids = generation_pb2.GenerateRequest(token_ids=generation_pb2.TokenIds(ids=[1, 2, 3]))

    assert await request_mapping._input_from_request(prompt, llm) == "hello"
    assert await request_mapping._input_from_request(token_ids, llm) == {
        "prompt_token_ids": [1, 2, 3]
    }


@pytest.mark.asyncio
async def test_token_ids_are_rejected_when_the_processor_does_not_consume_them():
    llm = SimpleNamespace(input_processor=SimpleNamespace(supports_token_ids_input=False))
    request = generation_pb2.GenerateRequest(token_ids=generation_pb2.TokenIds(ids=[1, 2, 3]))

    with pytest.raises(UnsupportedFeatureError, match="send a text prompt"):
        await request_mapping._input_from_request(request, llm)


@pytest.mark.asyncio
async def test_image_sources_preserve_order_uuid_and_model_decode_defaults(image_llm, monkeypatch):
    llm, _ = image_llm
    created_with = []

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            created_with.append((defaults, runtime))
            return cls()

        async def async_load(self, source):
            # Finish in the opposite order to prove gather preserves request order.
            if source.startswith("https"):
                await asyncio.sleep(0.01)
            return f"decoded:{source}"

        async def async_load_bytes(self, data):
            return f"decoded-bytes:{data.decode()}"

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    _ignore_synthetic_decoded_size(monkeypatch)
    request = generation_pb2.GenerateRequest(
        prompt="<image_1><image_2><image_3> describe",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                url="https://example.com/first.png",
                uuid="first",
            ),
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                data_uri="data:image/png;base64,c2Vjb25k",
            ),
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"third",
                mime_type="image/png",
                uuid="third",
            ),
        ],
    )

    mapped = await request_mapping._input_from_request(request, llm)

    assert created_with == [({"format": "np"}, None)]
    assert mapped["prompt"] == request.prompt
    assert mapped["multi_modal_data"] == {
        "image": [
            "decoded:https://example.com/first.png",
            "decoded:data:image/png;base64,c2Vjb25k",
            "decoded-bytes:third",
        ]
    }
    assert mapped["multi_modal_uuids"] == {"image": ["first", None, "third"]}
    assert mapped["mm_item_order"] == [
        {"modality": "image", "index": 0, "placeholder": "<image_1>"},
        {"modality": "image", "index": 1, "placeholder": "<image_2>"},
        {"modality": "image", "index": 2, "placeholder": "<image_3>"},
    ]


@pytest.mark.asyncio
async def test_media_sources_preserve_global_order(image_video_audio_llm, monkeypatch):
    created_with = []
    audio_sources = []

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            created_with.append(("image", defaults, runtime))
            return cls()

        async def async_load_bytes(self, data):
            return np.full((1, 1, 3), data[0], dtype=np.uint8)

    class _VideoIo:
        @classmethod
        def create(cls, defaults, runtime):
            created_with.append(("video", defaults, runtime))
            return cls()

        async def async_load(self, source):
            return VideoData(
                frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                metadata={"source": source},
            )

    class _AudioIo:
        @classmethod
        def create(cls, defaults, runtime):
            created_with.append(("audio", defaults, runtime))
            return cls()

        async def async_load(self, source):
            audio_sources.append(source)
            return np.zeros(8, dtype=np.float32), 16000

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    monkeypatch.setattr(request_mapping, "VideoMediaIO", _VideoIo)
    monkeypatch.setattr(request_mapping, "AudioMediaIO", _AudioIo)
    request = generation_pb2.GenerateRequest(
        prompt="mixed media",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                url="https://example.com/first.mp4",
                uuid="video-id",
            ),
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"i",
                mime_type="image/png",
            ),
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_AUDIO,
                url="https://example.com/audio.wav",
                uuid="audio-id",
            ),
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                data_uri="data:video/mp4;base64,dmlkZW8=",
            ),
        ],
    )

    mapped = await request_mapping._input_from_request(request, image_video_audio_llm)

    assert len(created_with) == 3
    video_modality, video_defaults, video_runtime = created_with[0]
    assert video_modality == "video"
    assert video_defaults == {"format": "np", "num_frames": 4}
    assert set(video_runtime) == {"decoded_memory_reserver"}
    assert callable(video_runtime["decoded_memory_reserver"])
    assert created_with[1] == ("image", {"format": "np"}, None)
    audio_modality, audio_defaults, audio_runtime = created_with[2]
    assert audio_modality == "audio"
    assert audio_defaults is None
    assert set(audio_runtime) == {"decoded_memory_reserver"}
    assert callable(audio_runtime["decoded_memory_reserver"])
    assert audio_sources == ["https://example.com/audio.wav"]
    assert mapped["multi_modal_data"]["audio"][0][1] == 16000
    assert [video.metadata["source"] for video in mapped["multi_modal_data"]["video"]] == [
        "https://example.com/first.mp4",
        "data:video/mp4;base64,dmlkZW8=",
    ]
    assert mapped["multi_modal_data"]["image"][0].shape == (1, 1, 3)
    assert mapped["multi_modal_uuids"] == {
        "video": ["video-id", None],
        "image": [None],
        "audio": ["audio-id"],
    }
    assert mapped["mm_item_order"] == [
        {"modality": "video", "index": 0, "placeholder": "<video_1>"},
        {"modality": "image", "index": 0, "placeholder": "<image_1>"},
        {"modality": "audio", "index": 0, "placeholder": "<audio_1>"},
        {"modality": "video", "index": 1, "placeholder": "<video_2>"},
    ]


@pytest.mark.asyncio
async def test_image_loading_has_a_bounded_concurrency(image_llm, monkeypatch):
    llm, _ = image_llm

    class _ImageIo:
        active = 0
        maximum_active = 0

        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            type(self).active += 1
            type(self).maximum_active = max(type(self).maximum_active, type(self).active)
            try:
                await asyncio.sleep(0.01)
                return data
            finally:
                type(self).active -= 1

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    _ignore_synthetic_decoded_size(monkeypatch)
    request = generation_pb2.GenerateRequest(
        prompt="images",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=str(index).encode(),
            )
            for index in range(request_mapping._MAX_CONCURRENT_MEDIA_LOADS * 2 + 1)
        ],
    )

    mapped = await request_mapping._input_from_request(request, llm)

    assert _ImageIo.maximum_active == request_mapping._MAX_CONCURRENT_MEDIA_LOADS
    assert mapped["multi_modal_data"]["image"] == [
        str(index).encode() for index in range(request_mapping._MAX_CONCURRENT_MEDIA_LOADS * 2 + 1)
    ]


@pytest.mark.asyncio
async def test_image_loading_cancels_siblings_after_a_failure(image_llm, monkeypatch):
    llm, _ = image_llm
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            if data == b"slow":
                slow_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    slow_cancelled.set()
                    raise
            await slow_started.wait()
            raise ValueError("decode failed")

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    request = generation_pb2.GenerateRequest(
        prompt="images",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=value,
            )
            for value in (b"slow", b"broken")
        ],
    )

    with pytest.raises(ValueError, match="decode failed"):
        await request_mapping._input_from_request(request, llm)
    assert slow_cancelled.is_set()


@pytest.mark.asyncio
async def test_image_count_is_bounded_before_decoding(image_llm, monkeypatch):
    llm, _ = image_llm

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            raise AssertionError("the media-count guard must run before decoder creation")

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    request = generation_pb2.GenerateRequest(
        prompt="too many images",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
            for _ in range(request_mapping._MAX_MEDIA_ITEMS + 1)
        ],
    )

    with pytest.raises(ValueError, match="at most 64 media items"):
        await request_mapping._input_from_request(request, llm)


@pytest.mark.asyncio
async def test_model_capability_is_checked_before_media_count_limit():
    request = generation_pb2.GenerateRequest(
        prompt="too many images",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
            for _ in range(request_mapping._MAX_MEDIA_ITEMS + 1)
        ],
    )

    with pytest.raises(UnsupportedFeatureError, match="media input is not supported"):
        await request_mapping._input_from_request(request, SimpleNamespace())


@pytest.mark.asyncio
async def test_total_decoded_image_bytes_are_bounded(image_llm, monkeypatch):
    llm, _ = image_llm

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    monkeypatch.setattr(request_mapping, "_MAX_TOTAL_DECODED_MEDIA_BYTES", 20)
    request = generation_pb2.GenerateRequest(
        prompt="images",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
            for _ in range(2)
        ],
    )

    with pytest.raises(ValueError, match="decoded media exceeds"):
        await request_mapping._input_from_request(request, llm)


@pytest.mark.asyncio
async def test_concurrent_video_loads_share_budget_before_allocation(image_video_llm, monkeypatch):
    allocations = []

    class _VideoIo:
        def __init__(self, reserver):
            self._reserver = reserver

        @classmethod
        def create(cls, defaults, runtime):
            return cls(runtime["decoded_memory_reserver"])

        async def async_load_bytes(self, data):
            reservation = self._reserver(12, 12)
            allocations.append(bytes(data))
            try:
                await asyncio.Future()
            finally:
                reservation.release()

    monkeypatch.setattr(request_mapping, "VideoMediaIO", _VideoIo)
    monkeypatch.setattr(request_mapping, "_MAX_TOTAL_DECODED_MEDIA_BYTES", 20)
    request = generation_pb2.GenerateRequest(
        prompt="videos",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                raw_bytes=value,
            )
            for value in (b"first", b"second")
        ],
    )

    with pytest.raises(ValueError, match="decoded media exceeds"):
        await request_mapping._input_from_request(request, image_video_llm)

    assert allocations == [b"first"]


@pytest.mark.asyncio
async def test_image_mapping_supports_token_ids(image_llm, monkeypatch):
    llm, _ = image_llm

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            return "decoded"

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    _ignore_synthetic_decoded_size(monkeypatch)
    request = generation_pb2.GenerateRequest(
        token_ids=generation_pb2.TokenIds(ids=[10, 20]),
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
        ],
    )

    mapped = await request_mapping._input_from_request(request, llm)

    assert mapped["prompt_token_ids"] == [10, 20]
    assert mapped["multi_modal_data"] == {"image": ["decoded"]}
    assert "multi_modal_uuids" not in mapped


@pytest.mark.asyncio
async def test_generate_submits_image_and_reports_expanded_prompt_usage(image_llm, monkeypatch):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            return "decoded"

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    _ignore_synthetic_decoded_size(monkeypatch)
    servicer = OpenEngineInferenceServicer(llm, model="image-model")
    request = generation_pb2.GenerateRequest(
        request_id="image-request",
        model="image-model",
        token_ids=generation_pb2.TokenIds(ids=[1, 2]),
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
        ],
    )

    responses = [response async for response in servicer.Generate(request, FakeServicerContext())]

    assert llm.generate_kwargs["inputs"]["multi_modal_data"] == {"image": ["decoded"]}
    assert responses[-1].usage.prompt_tokens == 6
    assert responses[-1].usage.total_tokens == 7


@pytest.mark.asyncio
async def test_duplicate_request_id_is_reserved_while_media_loads(image_llm, monkeypatch):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    load_count = 0

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            nonlocal load_count
            load_count += 1
            load_started.set()
            await release_load.wait()
            return np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    servicer = OpenEngineInferenceServicer(llm, model="image-model")
    request = generation_pb2.GenerateRequest(
        request_id="duplicate-image-request",
        model="image-model",
        prompt="<image_1>",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
        ],
    )

    async def collect(context):
        return [response async for response in servicer.Generate(request, context)]

    first_context = FakeServicerContext()
    first = asyncio.create_task(collect(first_context))
    await load_started.wait()
    duplicate_context = FakeServicerContext()
    try:
        with pytest.raises(AbortError):
            await asyncio.wait_for(collect(duplicate_context), timeout=1)
        assert duplicate_context.abort_code == grpc.StatusCode.ALREADY_EXISTS
        assert load_count == 1
        assert servicer.active_request_count() == 1
        assert servicer.abort_request_by_id(request.request_id)
    finally:
        release_load.set()
        first_outcome = await asyncio.gather(first, return_exceptions=True)

    assert isinstance(first_outcome[0], AbortError)
    assert first_context.abort_code == grpc.StatusCode.CANCELLED
    assert llm.generate_kwargs is None
    assert servicer.active_request_count() == 0


@pytest.mark.asyncio
async def test_generate_maps_media_errors_to_grpc_statuses(image_llm):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )
    servicer = OpenEngineInferenceServicer(llm, model="image-model")

    unsupported_context = FakeServicerContext()
    video_request = generation_pb2.GenerateRequest(
        request_id="video-request",
        model="image-model",
        prompt="video",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                raw_bytes=b"video",
            )
        ],
    )
    with pytest.raises(AbortError):
        async for _ in servicer.Generate(video_request, unsupported_context):
            pass
    assert unsupported_context.abort_code == grpc.StatusCode.UNIMPLEMENTED

    malformed_context = FakeServicerContext()
    file_request = generation_pb2.GenerateRequest(
        request_id="file-request",
        model="image-model",
        prompt="image",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                url="file:///tmp/private.png",
            )
        ],
    )
    with pytest.raises(AbortError):
        async for _ in servicer.Generate(file_request, malformed_context):
            pass
    assert malformed_context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert servicer.active_request_count() == 0


@pytest.mark.asyncio
async def test_generate_maps_transient_media_fetch_to_unavailable(image_llm, monkeypatch):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load(self, source):
            raise MediaFetchError("connection timed out")

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    servicer = OpenEngineInferenceServicer(llm, model="image-model")
    context = FakeServicerContext()
    request = generation_pb2.GenerateRequest(
        request_id="remote-image",
        model="image-model",
        prompt="image",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                url="https://example.com/image.png",
            )
        ],
    )

    with pytest.raises(AbortError):
        async for _ in servicer.Generate(request, context):
            pass
    assert context.abort_code == grpc.StatusCode.UNAVAILABLE


@pytest.mark.asyncio
async def test_audio_is_explicitly_unsupported(image_llm):
    llm, _ = image_llm
    request = generation_pb2.GenerateRequest(
        prompt="media",
        media=[generation_pb2.MediaItem(modality=generation_pb2.MODALITY_AUDIO, raw_bytes=b"data")],
    )

    with pytest.raises(UnsupportedFeatureError, match="audio"):
        await request_mapping._input_from_request(request, llm)


@pytest.mark.asyncio
async def test_image_is_rejected_for_a_text_only_loaded_model():
    request = generation_pb2.GenerateRequest(
        prompt="image",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
        ],
    )

    with pytest.raises(UnsupportedFeatureError, match="loaded model"):
        await request_mapping._input_from_request(
            request, SimpleNamespace(input_processor=object())
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "message"),
    [
        (generation_pb2.MediaItem(raw_bytes=b"image"), "modality must be set"),
        (
            generation_pb2.MediaItem(modality=generation_pb2.MODALITY_IMAGE),
            "must set exactly one source",
        ),
        (
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                url="file:///tmp/private.png",
            ),
            "must use http or https",
        ),
        (
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
                mime_type="video/mp4",
            ),
            "must start with image/",
        ),
    ],
)
async def test_malformed_image_items_are_rejected(image_llm, item, message):
    llm, _ = image_llm
    request = generation_pb2.GenerateRequest(prompt="image", media=[item])

    with pytest.raises(ValueError, match=message):
        await request_mapping._input_from_request(request, llm)


@pytest.mark.asyncio
async def test_raw_video_mime_type_is_validated(image_video_llm):
    request = generation_pb2.GenerateRequest(
        prompt="video",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                raw_bytes=b"video",
                mime_type="image/png",
            )
        ],
    )

    with pytest.raises(ValueError, match="must start with video/"):
        await request_mapping._input_from_request(request, image_video_llm)


@pytest.mark.asyncio
async def test_raw_audio_mime_type_is_validated(image_video_audio_llm):
    request = generation_pb2.GenerateRequest(
        prompt="audio",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_AUDIO,
                raw_bytes=b"audio",
                mime_type="video/mp4",
            )
        ],
    )

    with pytest.raises(ValueError, match="must start with audio/"):
        await request_mapping._input_from_request(request, image_video_audio_llm)
