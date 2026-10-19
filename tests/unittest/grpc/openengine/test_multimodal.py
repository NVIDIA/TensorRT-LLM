# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenEngine image input mapping and discovery."""

import asyncio
from collections.abc import AsyncIterator, Sequence
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
from tensorrt_llm.inputs.media_io import MediaFetchError  # noqa: E402
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
        request_mapping._MediaHandler(handler.name, handler.loader_factory, lambda _decoded: 0),
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
        return {"image": {"format": "np"}}


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


def test_media_capability_requires_the_loaded_multimodal_processor(image_llm):
    llm, seen_model_types = image_llm

    assert request_mapping.media_capabilities_for_model(llm).modalities == {"image"}
    assert seen_model_types == [_ImageConfig.model_type]
    assert (
        request_mapping.media_capabilities_for_model(
            SimpleNamespace(_hf_model_config=llm._hf_model_config, input_processor=object())
        ).modalities
        == set()
    )


def test_media_capabilities_are_hashable():
    first = MediaCapabilities({"image": "<image_{}>"})
    second = MediaCapabilities({"image": "<image_{}>"})

    assert first == second
    assert hash(first) == hash(second)


def test_unknown_decoded_image_type_is_rejected():
    with pytest.raises(TypeError, match="unsupported decoded image type"):
        request_mapping._decoded_image_bytes(object())


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

    def video_loader_factory(_llm):
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
@pytest.mark.parametrize(
    ("modality", "name"),
    [
        (generation_pb2.MODALITY_VIDEO, "video"),
        (generation_pb2.MODALITY_AUDIO, "audio"),
    ],
)
async def test_video_and_audio_are_explicitly_unsupported(image_llm, modality, name):
    llm, _ = image_llm
    request = generation_pb2.GenerateRequest(
        prompt="media",
        media=[generation_pb2.MediaItem(modality=modality, raw_bytes=b"data")],
    )

    with pytest.raises(UnsupportedFeatureError, match=name):
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
            "must be an image type",
        ),
    ],
)
async def test_malformed_image_items_are_rejected(image_llm, item, message):
    llm, _ = image_llm
    request = generation_pb2.GenerateRequest(prompt="image", media=[item])

    with pytest.raises(ValueError, match=message):
        await request_mapping._input_from_request(request, llm)
