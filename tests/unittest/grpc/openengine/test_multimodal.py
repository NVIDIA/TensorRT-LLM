# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenEngine multimodal input mapping and discovery."""

import asyncio
import base64
import gc
import re
import sys
import threading
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

grpc = pytest.importorskip(
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.67.1,<2")'
)
from conftest import AbortError, FakeServicerContext  # noqa: E402

import tensorrt_llm.grpc.openengine.request_mapping as request_mapping  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import (  # noqa: E402
    generation_pb2,
    model_pb2,
    server_pb2,
)
from tensorrt_llm.grpc.openengine.capabilities import MediaCapabilities  # noqa: E402
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.errors import UnsupportedFeatureError  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402
from tensorrt_llm.inputs import media_io  # noqa: E402
from tensorrt_llm.inputs.media_io import MediaFetchError  # noqa: E402
from tensorrt_llm.inputs.multimodal import MultimodalParams  # noqa: E402
from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData  # noqa: E402
from tensorrt_llm.inputs.registry import (  # noqa: E402
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    BaseMultimodalInputProcessor,
    DefaultInputProcessor,
)
from tensorrt_llm.llmapi.llm import PreprocessedInputs  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _ignore_synthetic_decoded_size(monkeypatch) -> None:
    handler = request_mapping._MEDIA_HANDLERS[generation_pb2.MODALITY_IMAGE]
    monkeypatch.setitem(
        request_mapping._MEDIA_HANDLERS,
        generation_pb2.MODALITY_IMAGE,
        request_mapping._MediaHandler(
            handler.name,
            handler.validate,
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
            handler.validate,
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

    @property
    def supported_modalities(self):
        return frozenset({"image", "video", "audio"})

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
    supported_modalities = frozenset({"image", "video", "audio"})

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
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [
            1000 + int(token[7:-1]) if token.startswith("<image_") else ord(token)
            for token in re.findall(r"<image_\d+>|.", text, re.DOTALL)
        ]

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

    def preprocess(self, inputs, sampling_params, disaggregated_params):
        return inputs


class _Reservation:
    def __init__(self) -> None:
        self.handle: Any = None
        self.aborted = False
        self.released = False
        self._release_task: asyncio.Task[None] | None = None

    def bind(self, handle: Any) -> None:
        self.handle = handle
        if self.aborted:
            handle.abort()

    def abort(self) -> bool:
        if self.handle is not None:
            self.handle.abort()
        self.aborted = True
        return True

    def release(self) -> asyncio.Task[None]:
        async def mark_released() -> None:
            self.released = True

        if self._release_task is None:
            self._release_task = asyncio.create_task(mark_released())
        return self._release_task


class _Frontend:
    def __init__(self, abort_before_bind: bool = False) -> None:
        self.abort_before_bind = abort_before_bind
        self.reservations: list[_Reservation] = []

    async def reserve(self, _request_id: str) -> _Reservation:
        reservation = _Reservation()
        self.reservations.append(reservation)
        if self.abort_before_bind:
            self.abort_before_bind = False
            reservation.abort()
        return reservation


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


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["", "<|startoftranscript|><|fr|><|translate|><|notimestamps|>"])
async def test_audio_text_prompt_survives_preparation_without_placeholders(monkeypatch, prompt):
    import soundfile

    config = _ImageConfig()
    processor = _AudioInputProcessor(config)
    processor.supports_token_ids_input = False
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY, "is_valid", lambda _, modality: modality == "audio"
    )
    monkeypatch.setattr(MULTIMODAL_PLACEHOLDER_REGISTRY, "get_placeholder", lambda *_: "")
    llm = _ImageLlm(processor, config)
    llm.args.reasoning_parser = None
    llm.args.max_seq_len = 2048
    inference = OpenEngineInferenceServicer(llm, "audio-model")
    control = OpenEngineControlServicer(llm, "audio-model", inference)
    info = await control.GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext())
    assert info.supports_text_input is True
    assert info.supports_token_ids_input is False
    encoded = BytesIO()
    soundfile.write(encoded, np.zeros(32), 16000, format="WAV")
    request = generation_pb2.GenerateRequest(
        request_id="audio-task-prompt",
        model="audio-model",
        prompt=prompt,
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_AUDIO,
                raw_bytes=encoded.getvalue(),
                mime_type="audio/wav",
            )
        ],
    )
    responses = [response async for response in inference.Generate(request, FakeServicerContext())]
    assert responses[-1].HasField("finished")
    inputs = llm.generate_kwargs["inputs"]
    assert inputs["prompt"] == prompt
    samples, rate = inputs["multi_modal_data"]["audio"][0]
    assert samples.shape == (32,) and rate == 16000
    assert inference.active_request_count() == 0


def test_media_capability_rejects_a_registered_non_callable_processor(monkeypatch):
    config = _ImageConfig()
    monkeypatch.setattr(MULTIMODAL_PLACEHOLDER_REGISTRY, "is_valid", lambda *_args: True)
    llm = SimpleNamespace(
        _hf_model_config=config,
        input_processor=_NonCallableRegisteredInputProcessor(config),
    )

    assert request_mapping.media_capabilities_for_model(llm).modalities == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [False, True])
async def test_disabled_mm_encoder_agrees_between_discovery_and_generate(image_llm, disabled):
    capability_llm, _ = image_llm
    llm = _ImageLlm(capability_llm.input_processor, capability_llm._hf_model_config)
    llm.args.disable_mm_encoder = disabled
    llm.args.reasoning_parser = None
    llm.args.max_seq_len = 2048
    inference = OpenEngineInferenceServicer(llm, model="image-model")
    control = OpenEngineControlServicer(llm, "image-model", inference)

    info = await control.GetModelInfo(
        model_pb2.GetModelInfoRequest(model="image-model"), FakeServicerContext()
    )

    assert info.supports_multimodal is (not disabled)
    assert list(info.extra["trtllm_supported_modalities"]) == ([] if disabled else ["image"])
    if disabled:
        context = FakeServicerContext()
        request = generation_pb2.GenerateRequest(
            request_id="disabled-encoder",
            model="image-model",
            prompt="<image_1>",
            media=[
                generation_pb2.MediaItem(modality=generation_pb2.MODALITY_IMAGE, raw_bytes=b"image")
            ],
        )
        with pytest.raises(AbortError):
            async for _ in inference.Generate(request, context):
                pass
        assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED
        assert llm.generate_kwargs is None
        assert inference.active_request_count() == 0


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


@pytest.mark.parametrize("audio_format", ["FLAC", "MP3"])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("source", ["bytes", "base64", "file"])
def test_audio_reserves_before_decoding_and_preserves_samples(
    audio_format, channels, source, tmp_path, monkeypatch
):
    import base64

    import soundfile

    samples = np.arange(32 * channels, dtype=np.float64).reshape(32, channels) / 128
    encoded = BytesIO()
    soundfile.write(encoded, samples, 16000, format=audio_format)
    data = encoded.getvalue()
    expected, _ = soundfile.read(BytesIO(data))
    with soundfile.SoundFile(BytesIO(data)) as audio:
        reserved_bytes = audio.frames * audio.channels * np.dtype(np.float64).itemsize
    budget = request_mapping._DecodedMediaBudget(reserved_bytes)
    original_read = soundfile.SoundFile.read

    def checked_read(audio, *args, **kwargs):
        assert (budget._used, budget._reserved_retained) == (reserved_bytes, reserved_bytes)
        assert kwargs["out"].nbytes == reserved_bytes
        return original_read(audio, *args, **kwargs)

    monkeypatch.setattr(soundfile.SoundFile, "read", checked_read)
    loader = media_io.AudioMediaIO(decoded_memory_reserver=budget.reserve)
    if source == "bytes":
        decoded, rate = loader.load_bytes(data)
    elif source == "base64":
        decoded, rate = loader.load_base64(
            "audio/mpeg" if audio_format == "MP3" else "audio/flac",
            base64.b64encode(data).decode(),
        )
    else:
        path = tmp_path / f"audio.{audio_format.lower()}"
        path.write_bytes(data)
        decoded, rate = loader.load_file(path.as_uri())
    if audio_format == "MP3":
        # Buffered and ordinary MP3 reads can differ by float32 rounding.
        np.testing.assert_allclose(decoded, expected, rtol=0, atol=np.finfo(np.float32).eps)
    else:
        np.testing.assert_array_equal(decoded, expected)
    assert rate == 16000
    assert (budget._used, budget._retained, budget._reserved_retained) == (
        expected.nbytes,
        expected.nbytes,
        0,
    )


@pytest.mark.parametrize("outcome", ["oversized", "failure", "cancelled"])
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

    monkeypatch.setattr(soundfile.SoundFile, "read", read)
    if outcome == "oversized":
        monkeypatch.setattr(
            np, "empty", lambda *args, **kwargs: pytest.fail("allocated oversized audio")
        )
    error = {
        "oversized": ValueError,
        "failure": RuntimeError,
        "cancelled": asyncio.CancelledError,
    }[outcome]
    with pytest.raises(error):
        loader.load_bytes(encoded.getvalue())
    assert (budget._used, budget._retained, budget._reserved_retained) == (0, 0, 0)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("decoded_frames", [0, 17])
def test_audio_short_read_preserves_samples_and_accounts_for_owning_storage(
    channels, decoded_frames, monkeypatch
):
    import soundfile

    encoded = BytesIO()
    samples = np.arange(32 * channels, dtype=np.float64).reshape(32, channels) / 128
    soundfile.write(encoded, samples, 16000, format="FLAC")
    expected, _ = soundfile.read(BytesIO(encoded.getvalue()))
    budget = request_mapping._DecodedMediaBudget(samples.nbytes)
    original_read = soundfile.SoundFile.read

    def short_read(audio, *args, **kwargs):
        assert budget._used == samples.nbytes
        return original_read(audio, *args, **kwargs)[:decoded_frames]

    monkeypatch.setattr(soundfile.SoundFile, "read", short_read)
    decoded, rate = media_io.AudioMediaIO(decoded_memory_reserver=budget.reserve).load_bytes(
        encoded.getvalue()
    )
    np.testing.assert_array_equal(decoded, expected[:decoded_frames])
    assert rate == 16000
    assert decoded.flags.owndata
    assert decoded.base is None
    assert (budget._used, budget._retained, budget._reserved_retained) == (
        decoded.nbytes,
        decoded.nbytes,
        0,
    )


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
@pytest.mark.parametrize("processor_kind", ["gemma4", "nemotron"])
@pytest.mark.parametrize("audio_encoder", [False, True])
async def test_audio_capability_follows_loaded_processor(processor_kind, audio_encoder):
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4InputProcessor
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import (
        NemotronHMultimodalInputProcessor,
    )

    if processor_kind == "gemma4":
        processor = Gemma4InputProcessor.__new__(Gemma4InputProcessor)
        config = SimpleNamespace(
            model_type="gemma4",
            vision_config=object(),
            audio_config=object() if audio_encoder else None,
        )
    else:
        processor = NemotronHMultimodalInputProcessor.__new__(NemotronHMultimodalInputProcessor)
        config = SimpleNamespace(model_type="nemotron_h_omni")
        processor._audio_extractor = object() if audio_encoder else None
    processor._config = config
    llm = _ImageLlm(processor, config)
    llm.args.reasoning_parser = None
    llm.args.max_seq_len = 2048
    inference = OpenEngineInferenceServicer(llm, "audio-model")
    control = OpenEngineControlServicer(llm, "audio-model", inference)
    info = await control.GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext())
    assert ("audio" in info.extra["trtllm_supported_modalities"]) is audio_encoder
    assert "image" in info.extra["trtllm_supported_modalities"]
    assert {"image", "video"} <= processor.supported_modalities
    assert ("audio" in processor.supported_modalities) is audio_encoder
    if not audio_encoder:
        request = generation_pb2.GenerateRequest(
            request_id="unsupported-audio",
            model="audio-model",
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
            await request_mapping._input_from_request(request, llm)
        context = FakeServicerContext()
        with pytest.raises(AbortError):
            async for _ in inference.Generate(request, context):
                pass
        assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.parametrize("audio_encoder", [False, True])
def test_gemma4_without_vision_encoder_omits_images_and_video(audio_encoder):
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4InputProcessor

    processor = Gemma4InputProcessor.__new__(Gemma4InputProcessor)
    processor._config = SimpleNamespace(
        model_type="gemma4", vision_config=None, audio_config=object() if audio_encoder else None
    )
    expected = {"audio"} if audio_encoder else frozenset()
    assert processor.supported_modalities == expected
    assert (
        request_mapping.media_capabilities_for_model(
            SimpleNamespace(input_processor=processor)
        ).modalities
        == expected
    )


@pytest.mark.parametrize("config_state", ["absent", "none"])
def test_gemma4_missing_processor_config_fails_closed(config_state):
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4InputProcessor

    config = SimpleNamespace(model_type="gemma4", audio_config=object())
    processor = Gemma4InputProcessor.__new__(Gemma4InputProcessor)
    if config_state == "none":
        processor._config = None
    llm = SimpleNamespace(input_processor=processor, _hf_model_config=config)
    assert processor.supported_modalities == frozenset()
    assert request_mapping.media_capabilities_for_model(llm).modalities == frozenset()


def test_whisper_declares_audio_without_multimodal_base():
    from tensorrt_llm._torch.models.modeling_whisper import WhisperInputProcessor

    processor = WhisperInputProcessor.__new__(WhisperInputProcessor)
    processor.config = SimpleNamespace(model_type="whisper")
    assert processor.supported_modalities == {"audio"}
    llm = SimpleNamespace(input_processor=processor)
    assert request_mapping.media_capabilities_for_model(llm).modalities == {"audio"}


def test_multimodal_processor_defaults_to_registered_modalities(monkeypatch):
    from tensorrt_llm.inputs.registry import MultimodalPlaceholderMetadata

    class RegisteredProcessor(_ImageInputProcessor):
        supported_modalities = BaseMultimodalInputProcessor.supported_modalities

    monkeypatch.setitem(
        MULTIMODAL_PLACEHOLDER_REGISTRY._multimodal_placeholder_by_model_type,
        _ImageConfig.model_type,
        MultimodalPlaceholderMetadata(placeholder_map={"image": "<image>", "audio": "<audio>"}),
    )
    processor = RegisteredProcessor(_ImageConfig())
    assert processor.supported_modalities == {"image", "audio"}
    processor._config = SimpleNamespace(model_type="openengine_unknown_processor")
    assert processor.supported_modalities == frozenset()


def test_registered_processor_without_modality_contract_fails_closed(monkeypatch):
    monkeypatch.delattr(_AudioInputProcessor, "supported_modalities")
    monkeypatch.setattr(MULTIMODAL_PLACEHOLDER_REGISTRY, "is_valid", lambda *_: True)
    llm = SimpleNamespace(input_processor=_AudioInputProcessor(_ImageConfig()))
    assert request_mapping.media_capabilities_for_model(llm).modalities == frozenset()


def test_default_input_processor_declares_token_ids_supported():
    processor = DefaultInputProcessor(model_path="unused", config=None, tokenizer=None)
    assert processor.supports_token_ids_input is True
    assert processor.supported_modalities == frozenset()


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


@pytest.mark.parametrize("cancel_during", ["grab", "retrieve", "convert"])
def test_video_cancellation_stops_between_frame_operations(cancel_during, monkeypatch):
    budget = request_mapping._DecodedMediaBudget(1024)
    operations = []

    def grab():
        operations.append("grab")
        if cancel_during == "grab" and operations.count("grab") == 2:
            budget.cancel()
        return True

    def retrieve(_scratch):
        operations.append("retrieve")
        if cancel_during == "retrieve" and operations.count("retrieve") == 2:
            budget.cancel()
        return True, np.zeros((1, 1, 3), dtype=np.uint8)

    def convert(_source, _code, dst):
        operations.append("convert")
        dst.fill(0)
        if cancel_during == "convert":
            budget.cancel()

    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=lambda key: 3 if key == 0 else 1,
        grab=grab,
        retrieve=retrieve,
        release=lambda: operations.append("release"),
    )
    monkeypatch.setattr(
        media_io,
        "_get_cv2",
        lambda: SimpleNamespace(
            VideoCapture=lambda _source: capture,
            CAP_PROP_FRAME_COUNT=0,
            CAP_PROP_FPS=1,
            COLOR_BGR2RGB=2,
            cvtColor=convert,
        ),
    )
    with pytest.raises(ValueError, match="cancelled"):
        media_io._load_video_by_cv2(
            "video.mp4",
            num_frames=3,
            fps=-1,
            format="np",
            decoded_memory_reserver=budget.reserve,
        )
    expected = ["grab", "retrieve", "convert"]
    if cancel_during in ("grab", "retrieve"):
        expected.append("grab")
    if cancel_during == "retrieve":
        expected.append("retrieve")
    assert operations == expected + ["release"]
    assert (budget._used, budget._retained, budget._reserved_retained) == (0, 0, 0)


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


@pytest.mark.parametrize("abandon_first", [False, True])
def test_abandoned_preprocessing_reclaims_exports_regardless_of_completion_order(abandon_first):
    import torch

    from tensorrt_llm.grpc.openengine.servicer import _PreprocessingWork

    params = MultimodalParams(multimodal_data={"image": [torch.ones(1024 * 1024)]})
    params.to_handle("multimodal_data")
    handle = params.multimodal_data["image"][0]
    path = Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
    result = PreprocessedInputs(prompt_token_ids=[1], multimodal_params=params)
    work = _PreprocessingWork()
    assert path.exists()
    if abandon_first:
        work.abandon()
    work.run(lambda: result)
    work.abandon()
    work.abandon()
    gc.collect()
    assert not path.exists()


@pytest.mark.parametrize("tokenized", [False, True])
@pytest.mark.parametrize("fast_path", [False, True])
def test_passthrough_placeholder_policy_follows_processor_input_path(
    image_llm, monkeypatch, tokenized, fast_path
):
    from tensorrt_llm.inputs.content_format import ContentFormat

    capability_llm, _ = image_llm
    llm = _ImageLlm(capability_llm.input_processor, capability_llm._hf_model_config)
    monkeypatch.setattr(llm.input_processor, "supports_token_id_mm_expansion", fast_path)
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_content_format",
        lambda model_type: ContentFormat.PASSTHROUGH,
    )
    request = generation_pb2.GenerateRequest(
        media=[generation_pb2.MediaItem(modality=generation_pb2.MODALITY_IMAGE, raw_bytes=b"image")]
    )
    if tokenized:
        request.token_ids.ids.extend(llm.tokenizer.encode("Describe this image"))
    else:
        request.prompt = "Describe this image"
    prepared = request_mapping._prepare_input_from_request(request, llm)
    if tokenized and fast_path:
        with pytest.raises(ValueError, match="placeholder count"):
            request_mapping._validate_media_placeholders(prepared, llm)
    else:
        request_mapping._validate_media_placeholders(prepared, llm)
        assert not llm.input_processor.requires_input_placeholders(tokenized=tokenized)


def test_submitted_preprocessing_preserves_exports_for_a_delayed_consumer():
    import torch

    from tensorrt_llm.grpc.openengine.servicer import _PreprocessingWork

    params = MultimodalParams(multimodal_data={"image": torch.ones(8)})
    params.to_handle("multimodal_data")
    handle = params.multimodal_data["image"]
    path = Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
    work = _PreprocessingWork()
    work.run(lambda: PreprocessedInputs(prompt_token_ids=[1], multimodal_params=params))
    work.transfer()
    work.abandon()
    assert path.exists()
    assert params.multimodal_data["image"] is handle
    params.to_tensor("multimodal_data")
    torch.testing.assert_close(params.multimodal_data["image"], torch.ones(8))
    params.multimodal_data = {}
    gc.collect()
    assert not path.exists()


def test_native_processor_registration_overrides_hf_config_policy(monkeypatch):
    from tensorrt_llm.inputs.content_format import ContentFormat

    class NativeProcessor(_ImageInputProcessor):
        _registered_model_type = "mistral_common"

    def content_format(model_type):
        assert model_type == "mistral_common"
        return ContentFormat.PASSTHROUGH

    monkeypatch.setattr(MULTIMODAL_PLACEHOLDER_REGISTRY, "get_content_format", content_format)
    assert not NativeProcessor(_ImageConfig()).requires_input_placeholders()


@pytest.mark.parametrize(
    "tokens,sequences",
    [
        ([], [(1,)]),
        ([1, 1, 1, 2], [(1,), (1, 1), (1, 1, 2), (2,)]),
        ([0, 1, 2, 0, 1, 2], [(1, 2), (2,), (9,)]),
    ],
)
def test_marker_scan_matches_reference_including_overlaps(tokens, sequences):
    counts = request_mapping._count_token_sequences(tokens, sequences)
    for sequence in sequences:
        expected = sum(
            tuple(tokens[index : index + len(sequence)]) == sequence
            for index in range(len(tokens) - len(sequence) + 1)
        )
        assert counts[sequence] == expected


@pytest.mark.asyncio
async def test_media_and_preprocessing_do_not_block_control_executor(image_llm, monkeypatch):
    from tensorrt_llm.inputs.media_io import BaseMediaIO

    loop = asyncio.get_running_loop()
    capability_llm, _ = image_llm
    llm = _ImageLlm(capability_llm.input_processor, capability_llm._hf_model_config)
    llm.args = SimpleNamespace(
        guided_decoding_backend=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        max_batch_size=8,
        max_num_tokens=2048,
        kv_cache_config=SimpleNamespace(tokens_per_block=32),
    )
    llm._executor = SimpleNamespace(
        get_kv_cache_capacity=lambda: {}, get_openengine_discovery=lambda: {}
    )
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_input") as input_pool,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_media") as media_pool,
    ):
        release = threading.Event()
        entered = [asyncio.Event(), asyncio.Event()]

        def block(index):
            loop.call_soon_threadsafe(entered[index].set)
            assert release.wait(timeout=5)
            return threading.current_thread().name

        # Saturate both media pools, then require a real Control GetServerInfo RPC
        # (which uses asyncio's default executor) to remain responsive.
        tasks = [loop.run_in_executor(input_pool, block, 0)]
        with BaseMediaIO.use_executor(media_pool):
            tasks.append(asyncio.create_task(BaseMediaIO._run_in_executor(block, 1)))
        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), timeout=2)
            inference = OpenEngineInferenceServicer(
                llm, "image-model", input_executor=input_pool, media_executor=media_pool
            )
            control = OpenEngineControlServicer(llm, "image-model", inference)
            info = await asyncio.wait_for(
                control.GetServerInfo(server_pb2.GetServerInfoRequest(), FakeServicerContext()),
                timeout=1,
            )
            assert info.engine_name == "tensorrt_llm"
        finally:
            release.set()
            names = await asyncio.gather(*tasks)
        assert names[0].startswith("test_input")
        assert names[1].startswith("test_media")


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
        request_mapping._MediaHandler(
            "video",
            lambda _item, _index: None,
            video_loader_factory,
            lambda decoded: len(decoded),
        ),
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
@pytest.mark.parametrize("model_defaults", [None, {"format": "np"}])
async def test_image_sources_preserve_order_uuid_and_model_decode_defaults(
    image_llm, monkeypatch, model_defaults
):
    llm, _ = image_llm
    if model_defaults is None:
        monkeypatch.delattr(_ImageInputProcessor, "get_preferred_media_io_kwargs")
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

    assert created_with == [(model_defaults, None)]
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
@pytest.mark.parametrize(
    ("invalid_case", "metadata", "expected_code"),
    [
        ("sampling", (), grpc.StatusCode.INVALID_ARGUMENT),
        ("metadata", (("openengine-priority", "1"),), grpc.StatusCode.UNIMPLEMENTED),
        (
            "affinity",
            (("x-trtllm-subagent-affinity-id", "parent"),),
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        ("rank", (("openengine-target-dp-rank", "1"),), grpc.StatusCode.INVALID_ARGUMENT),
    ],
)
async def test_request_validation_precedes_media_loading(
    image_llm, monkeypatch, invalid_case, metadata, expected_code
):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            raise AssertionError("media loading must not start for an invalid request")

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    request = generation_pb2.GenerateRequest(
        request_id=f"invalid-{invalid_case}",
        model="image-model",
        prompt="<image_1>",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"image",
            )
        ],
    )
    if invalid_case == "sampling":
        request.sampling.num_sequences = 0
    context = FakeServicerContext(metadata=metadata)
    servicer = OpenEngineInferenceServicer(llm, model="image-model")

    with pytest.raises(AbortError):
        async for _ in servicer.Generate(request, context):
            pass

    assert context.abort_code == expected_code
    assert llm.generate_kwargs is None


@pytest.mark.asyncio
async def test_capability_error_precedes_other_request_validation(image_llm):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )
    request = generation_pb2.GenerateRequest(
        request_id="unsupported-video",
        model="image-model",
        prompt="video",
        sampling=generation_pb2.SamplingParams(num_sequences=0),
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                raw_bytes=b"video",
            )
        ],
    )
    context = FakeServicerContext()
    servicer = OpenEngineInferenceServicer(llm, model="image-model")

    with pytest.raises(AbortError):
        async for _ in servicer.Generate(request, context):
            pass

    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED
    assert "video" in context.abort_details


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        {},
        {"url": "https://"},
        {"url": "https://example.com:bad/image"},
        {"url": "https://example.com:65536/image"},
        {"url": "https://[invalid/image"},
        {"data_uri": "data:image/png;base64"},
        {"data_uri": "data:image/png;base64,"},
        {"data_uri": "data:image/png,AAAA"},
        {"data_uri": "data:audio/wav;base64,AAAA"},
    ],
)
async def test_all_media_descriptions_are_validated_before_loading(image_llm, monkeypatch, source):
    llm, _ = image_llm

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            raise AssertionError("media loading must not start before every item is valid")

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    request = generation_pb2.GenerateRequest(
        prompt="images",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                raw_bytes=b"valid",
            ),
            generation_pb2.MediaItem(modality=generation_pb2.MODALITY_IMAGE, **source),
        ],
    )

    with pytest.raises(ValueError, match=r"media\[1\]"):
        await request_mapping._input_from_request(request, llm)


@pytest.mark.parametrize(
    "source",
    [
        {"url": "https://example.com/image.png"},
        {"url": "https://example.com:8443/image.png"},
        {"data_uri": "data:image/png;base64,AAAA"},
        {"data_uri": "data:;base64,AAAA"},
        {"raw_bytes": b"image", "mime_type": "image/png"},
    ],
)
def test_media_description_validation_accepts_supported_sources(image_llm, source):
    llm, _ = image_llm
    request = generation_pb2.GenerateRequest(
        prompt="image",
        media=[generation_pb2.MediaItem(modality=generation_pb2.MODALITY_IMAGE, **source)],
    )

    prepared = request_mapping._prepare_input_from_request(request, llm)

    assert len(prepared.media) == 1
    assert prepared.media[0][0] == request.media[0]


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
        token_ids=generation_pb2.TokenIds(ids=[1001, 2]),
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
@pytest.mark.parametrize("input_kind", ["prompt", "token_ids"])
@pytest.mark.parametrize(
    "shape", ["missing", "duplicate", "extra_bare_pad", "unexpected_video_pad"]
)
async def test_placeholder_mismatch_does_not_poison_hashing(
    image_llm, monkeypatch, input_kind, shape
):
    import torch

    capability_llm, _ = image_llm
    llm = _ImageLlm(capability_llm.input_processor, capability_llm._hf_model_config)
    processor = llm.input_processor
    processor.multimodal_hashing_supported = None
    monkeypatch.setattr(processor, "get_mm_token_ids", lambda: torch.tensor([21, 23]))
    monkeypatch.setattr(
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        "get_placeholder",
        lambda _model_type, _modality: "<|vision_start|><|image_pad|><|vision_end|>",
    )

    class _QwenTokenizer(_Tokenizer):
        def encode(self, text, add_special_tokens=True):
            ids = {
                "<|vision_start|>": 20,
                "<|image_pad|>": 21,
                "<|vision_end|>": 22,
                "<|video_pad|>": 23,
            }
            return [ids.get(token, 0) for token in re.findall(r"<\|[^>]+\|>|.", text)]

    llm.tokenizer = _QwenTokenizer()
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    prompts = {
        "missing": "describe",
        "duplicate": placeholder * 2,
        "extra_bare_pad": placeholder + "<|image_pad|>",
        "unexpected_video_pad": placeholder + "<|video_pad|>",
    }
    decoder_creations = 0

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            nonlocal decoder_creations
            decoder_creations += 1
            return cls()

        async def async_load_bytes(self, data):
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def preprocess(inputs, sampling_params, disaggregated_params):
        processor.multimodal_hashing_supported = True
        return inputs

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    monkeypatch.setattr(llm, "preprocess", preprocess)
    servicer = OpenEngineInferenceServicer(llm, model="image-model")
    request = generation_pb2.GenerateRequest(
        request_id="placeholder-validation",
        model="image-model",
        media=[
            generation_pb2.MediaItem(modality=generation_pb2.MODALITY_IMAGE, raw_bytes=b"image")
        ],
    )

    def set_input(prompt):
        if input_kind == "prompt":
            request.prompt = prompt
        else:
            request.token_ids.CopyFrom(generation_pb2.TokenIds(ids=llm.tokenizer.encode(prompt)))

    set_input(prompts[shape])
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        async for _ in servicer.Generate(request, context):
            pass

    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "placeholder" in context.abort_details
    assert processor.multimodal_hashing_supported is None
    assert decoder_creations == 0
    assert llm.generate_kwargs is None
    assert servicer.active_request_count() == 0

    set_input(placeholder)
    responses = [response async for response in servicer.Generate(request, FakeServicerContext())]
    assert responses
    assert decoder_creations == 1
    assert processor.multimodal_hashing_supported is True
    assert servicer.active_request_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "local_abort", "coordinator_abort", "rpc_cancel"])
@pytest.mark.parametrize("modality", ["image", "video"])
async def test_media_preprocessing_is_offloaded_and_abort_aware(
    image_video_llm, monkeypatch, outcome, modality
):
    capability_llm = image_video_llm
    llm = _ImageLlm(capability_llm.input_processor, capability_llm._hf_model_config)
    loop = asyncio.get_running_loop()
    serving_thread = threading.get_ident()
    request_context = ContextVar("preprocess_request_context", default=None)
    started = asyncio.Event()
    finished = asyncio.Event()
    reclaimed = asyncio.Event()
    release = threading.Event()
    params = MultimodalParams()
    preprocessed = PreprocessedInputs(prompt_token_ids=[1, 2, 3], multimodal_params=params)
    export_paths = []
    release_handles = params.release_handles

    def reclaim():
        release_handles()
        loop.call_soon_threadsafe(reclaimed.set)

    monkeypatch.setattr(params, "release_handles", reclaim)
    preprocess_calls = []
    submitted = []

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            from tensorrt_llm.inputs.media_io import BaseMediaIO

            def decode():
                assert threading.current_thread().name.startswith("test_media")
                frames = np.zeros((1, 1, 3), dtype=np.uint8)
                return frames if modality == "image" else VideoData(frames=[frames], metadata={})

            return await BaseMediaIO._run_in_executor(decode)

    def preprocess(inputs, sampling_params, disaggregated_params):
        assert threading.get_ident() != serving_thread
        assert threading.current_thread().name.startswith("test_input")
        assert request_context.get() == "image-request"
        preprocess_calls.append(inputs)
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(timeout=5), "test must release the preprocessing thread"
            import torch

            params.multimodal_data = {modality: {"pixel_values": torch.ones(1024 * 1024)}}
            params.to_handle("multimodal_data")
            handle = params.multimodal_data[modality]["pixel_values"]
            export_paths.append(
                Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
            )
            return preprocessed
        finally:
            loop.call_soon_threadsafe(finished.set)

    def generate_async(**kwargs):
        assert threading.get_ident() == serving_thread
        if isinstance(kwargs["inputs"], PreprocessedInputs):
            kwargs["inputs"].multimodal_params.to_tensor("multimodal_data")
        submitted.append(kwargs["inputs"])
        return llm.result_handle

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    monkeypatch.setattr(request_mapping, "VideoMediaIO", _ImageIo)
    monkeypatch.setattr(llm, "preprocess", preprocess)
    monkeypatch.setattr(llm, "generate_async", generate_async)
    # Pools must close during the call phase, before pytest-threadleak's snapshot.
    with (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_input") as input_pool,
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_media") as media_pool,
    ):
        frontend = _Frontend()
        servicer = OpenEngineInferenceServicer(
            llm,
            model="image-model",
            frontend=frontend,
            input_executor=input_pool,
            media_executor=media_pool,
        )
        request = generation_pb2.GenerateRequest(
            request_id="preprocessing",
            model="image-model",
            prompt=f"<{modality}_1>",
            media=[
                generation_pb2.MediaItem(
                    modality=(
                        generation_pb2.MODALITY_IMAGE
                        if modality == "image"
                        else generation_pb2.MODALITY_VIDEO
                    ),
                    raw_bytes=b"media",
                )
            ],
        )
        context = FakeServicerContext()

        async def collect():
            token = request_context.set("image-request")
            try:
                return [response async for response in servicer.Generate(request, context)]
            finally:
                request_context.reset(token)

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            text_request = generation_pb2.GenerateRequest(
                request_id="concurrent-text", model="image-model", prompt="hello"
            )

            async def collect_text():
                return [
                    response
                    async for response in servicer.Generate(text_request, FakeServicerContext())
                ]

            assert await asyncio.wait_for(collect_text(), timeout=1)
            assert submitted == ["hello"]
            if outcome == "complete":
                release.set()
                assert await asyncio.wait_for(task, timeout=2)
                assert submitted[1] is preprocessed
            elif outcome == "rpc_cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
                assert context.abort_code is None
            else:
                if outcome == "local_abort":
                    assert servicer.abort_request_by_id(request.request_id)
                else:
                    assert frontend.reservations[0].abort()
                with pytest.raises(AbortError):
                    await asyncio.wait_for(task, timeout=1)
                assert context.abort_code == grpc.StatusCode.CANCELLED
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), timeout=2)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        if outcome != "complete":
            await asyncio.wait_for(reclaimed.wait(), timeout=2)
            gc.collect()
            assert export_paths and not any(path.exists() for path in export_paths)
        assert len(preprocess_calls) == 1
        assert len(submitted) == (2 if outcome == "complete" else 1)
        assert servicer.active_request_count() == 0
        if outcome != "complete":
            monkeypatch.setattr(llm, "preprocess", lambda *args: preprocessed)
            responses = [
                response async for response in servicer.Generate(request, FakeServicerContext())
            ]
            assert responses
            assert submitted[-1] is preprocessed
            assert len(submitted) == 2
            assert servicer.active_request_count() == 0
        assert all(reservation.released for reservation in frontend.reservations)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ValueError("invalid processor input"), grpc.StatusCode.INVALID_ARGUMENT),
        (TypeError("processor contract changed"), grpc.StatusCode.INTERNAL),
        (RuntimeError("processor failed"), grpc.StatusCode.INTERNAL),
    ],
)
async def test_preprocessing_errors_are_classified(image_llm, monkeypatch, error, expected_code):
    capability_llm, _ = image_llm
    llm = _ImageLlm(capability_llm.input_processor, capability_llm._hf_model_config)

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def preprocess(*args):
        raise error

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    monkeypatch.setattr(llm, "preprocess", preprocess)
    outcomes = []
    metrics = SimpleNamespace(record_outcome=lambda *args: outcomes.append(args))
    servicer = OpenEngineInferenceServicer(llm, model="image-model", metrics=metrics)
    request = generation_pb2.GenerateRequest(
        request_id="preprocess-error",
        model="image-model",
        prompt="<image_1>",
        media=[
            generation_pb2.MediaItem(modality=generation_pb2.MODALITY_IMAGE, raw_bytes=b"image")
        ],
    )
    context = FakeServicerContext()

    with pytest.raises(AbortError):
        async for _ in servicer.Generate(request, context):
            pass

    assert context.abort_code == expected_code
    expected_outcome = (
        "rejected" if expected_code == grpc.StatusCode.INVALID_ARGUMENT else "engine_error"
    )
    assert outcomes == [("aggregated", expected_outcome)]
    assert llm.generate_kwargs is None
    assert servicer.active_request_count() == 0


@pytest.mark.asyncio
async def test_duplicate_request_id_is_reserved_while_media_loads(image_llm, monkeypatch):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )
    load_started = asyncio.Event()
    load_cancelled = asyncio.Event()
    load_count = 0
    block_load = True

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            nonlocal block_load, load_count
            load_count += 1
            if not block_load:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            load_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                load_cancelled.set()
                raise

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
    with pytest.raises(AbortError):
        await asyncio.wait_for(collect(duplicate_context), timeout=1)
    assert duplicate_context.abort_code == grpc.StatusCode.ALREADY_EXISTS
    assert load_count == 1
    assert servicer.active_request_count() == 1
    assert servicer.abort_request_by_id(request.request_id)
    first_outcome = await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), timeout=1)

    assert isinstance(first_outcome[0], AbortError)
    assert first_context.abort_code == grpc.StatusCode.CANCELLED
    assert load_cancelled.is_set()
    assert llm.generate_kwargs is None
    assert servicer.active_request_count() == 0

    block_load = False
    responses = await collect(FakeServicerContext())
    assert responses
    assert servicer.active_request_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_path", ["coordinator", "rpc", "coordinator_and_rpc"])
async def test_media_loading_stops_for_coordinator_abort_or_rpc_cancellation(
    image_llm, monkeypatch, cancel_path
):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )
    load_started = asyncio.Event()
    load_cancelled = asyncio.Event()
    block_load = True

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            return cls()

        async def async_load_bytes(self, data):
            if not block_load:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            load_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                load_cancelled.set()
                raise

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    frontend = _Frontend()
    servicer = OpenEngineInferenceServicer(llm, model="image-model", frontend=frontend)
    request = generation_pb2.GenerateRequest(
        request_id=f"{cancel_path}-during-media",
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

    context = FakeServicerContext()
    task = asyncio.create_task(collect(context))
    await load_started.wait()
    if cancel_path == "coordinator":
        assert frontend.reservations[0].abort()
        with pytest.raises(AbortError):
            await asyncio.wait_for(task, timeout=1)
        assert context.abort_code == grpc.StatusCode.CANCELLED
    else:
        if cancel_path == "coordinator_and_rpc":
            assert frontend.reservations[0].abort()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert context.abort_code is None

    assert load_cancelled.is_set()
    assert llm.generate_kwargs is None
    assert servicer.active_request_count() == 0
    assert frontend.reservations[0].released

    block_load = False
    responses = await collect(FakeServicerContext())
    assert responses
    assert servicer.active_request_count() == 0
    assert len(frontend.reservations) == 2
    assert all(reservation.released for reservation in frontend.reservations)


@pytest.mark.asyncio
async def test_coordinator_abort_before_reservation_binding_prevents_media_loading(
    image_llm, monkeypatch
):
    capability_llm, _ = image_llm
    llm = _ImageLlm(
        capability_llm.input_processor,
        capability_llm._hf_model_config,
    )
    decoder_creations = 0

    class _ImageIo:
        @classmethod
        def create(cls, defaults, runtime):
            nonlocal decoder_creations
            decoder_creations += 1
            return cls()

        async def async_load_bytes(self, data):
            return np.zeros((1, 1, 3), dtype=np.uint8)

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    frontend = _Frontend(abort_before_bind=True)
    servicer = OpenEngineInferenceServicer(llm, model="image-model", frontend=frontend)
    request = generation_pb2.GenerateRequest(
        request_id="abort-before-bind",
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

    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await collect(context)

    assert context.abort_code == grpc.StatusCode.CANCELLED
    assert decoder_creations == 0
    assert llm.generate_kwargs is None
    assert servicer.active_request_count() == 0
    assert frontend.reservations[0].released

    responses = await collect(FakeServicerContext())
    assert responses
    assert decoder_creations == 1
    assert servicer.active_request_count() == 0
    assert len(frontend.reservations) == 2
    assert all(reservation.released for reservation in frontend.reservations)


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
@pytest.mark.parametrize(
    "message", ["connection timed out", "remote media server returned HTTP 503"]
)
async def test_generate_maps_transient_media_fetch_to_failed_precondition(
    image_llm, monkeypatch, message
):
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
            raise MediaFetchError(message)

    monkeypatch.setattr(request_mapping, "ImageMediaIO", _ImageIo)
    servicer = OpenEngineInferenceServicer(llm, model="image-model")
    context = FakeServicerContext()
    request = generation_pb2.GenerateRequest(
        request_id="remote-image",
        model="image-model",
        prompt="<image_1>",
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
    assert context.abort_code == grpc.StatusCode.FAILED_PRECONDITION
    assert message in context.abort_details
    assert llm.generate_kwargs is None
    assert servicer.active_request_count() == 0


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
