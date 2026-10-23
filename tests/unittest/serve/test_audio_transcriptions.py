# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for multipart Whisper transcription requests."""

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

# Load this runtime-independent module without initializing TensorRT/CUDA.
_SOURCE = Path(__file__).resolve().parents[3] / "tensorrt_llm/serve/audio_transcriptions.py"
_SPEC = importlib.util.spec_from_file_location("audio_transcriptions_cpu_test", _SOURCE)
audio = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audio
_SPEC.loader.exec_module(audio)

pytestmark = pytest.mark.cpu_only


def _wav(samples: np.ndarray, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        try:
            options, waveform = await audio.parse_transcription_request(
                request, served_model="whisper-test", sampling_rate=16000, max_samples=16000
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "options": options.model_dump(),
            "samples": len(waveform),
            "dtype": str(waveform.dtype),
            "peak": float(np.max(np.abs(waveform))),
        }

    return TestClient(app)


def _post(client: TestClient, *, data=None, content=None):
    return client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-test", **(data or {})},
        files={
            "file": (
                "clip.wav",
                content if content is not None else _wav(np.zeros(1600, dtype=np.float32)),
                "audio/wav",
            )
        },
    )


def test_valid_upload_and_defaults(client: TestClient) -> None:
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["samples"] == 1600
    assert body["dtype"] == "float32"
    assert body["options"]["response_format"] == "json"
    assert body["options"]["temperature"] == 0
    assert body["options"]["language"] is None


def test_explicit_options(client: TestClient) -> None:
    response = _post(
        client,
        data={"language": "de", "temperature": "0.4", "response_format": "text", "stream": "false"},
    )
    assert response.status_code == 200
    assert response.json()["options"]["language"] == "de"
    assert response.json()["options"]["temperature"] == 0.4


@pytest.mark.parametrize(
    "fields",
    [
        {"temperature": "nan"},
        {"temperature": "inf"},
        {"temperature": "-0.1"},
        {"temperature": "1.1"},
        {"response_format": "verbose_json"},
        {"stream": "true"},
        {"prompt": "previous transcript"},
        {"timestamp_granularities[]": "word"},
        {"typo": "value"},
    ],
)
def test_unsupported_or_invalid_options(client: TestClient, fields: dict) -> None:
    assert _post(client, data=fields).status_code == 400


def test_unknown_model(client: TestClient) -> None:
    assert _post(client, data={"model": "wrong"}).status_code == 404


@pytest.mark.parametrize("content", [b"", b"not an audio file"])
def test_invalid_audio(client: TestClient, content: bytes) -> None:
    assert _post(client, content=content).status_code == 400


def test_empty_waveform(client: TestClient) -> None:
    assert _post(client, content=_wav(np.zeros(0))).status_code == 400


def test_nonfinite_waveform(client: TestClient) -> None:
    assert _post(client, content=_wav(np.array([np.nan]))).status_code == 400


def test_resamples_and_downmixes(client: TestClient) -> None:
    stereo = np.column_stack((np.ones(800), -np.ones(800)))
    response = _post(client, content=_wav(stereo, rate=8000))
    assert response.status_code == 200
    assert response.json()["samples"] == 1600
    assert response.json()["peak"] == 0


def test_duration_limit_and_exact_boundary(client: TestClient) -> None:
    assert _post(client, content=_wav(np.zeros(16000))).status_code == 200
    response = _post(client, content=_wav(np.zeros(16001)))
    assert response.status_code == 400
    assert "duration" in response.json()["detail"].lower()


def test_missing_file(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-test"},
        files={"other": ("clip.wav", b"audio")},
    )
    assert response.status_code == 400


def test_missing_model(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/transcriptions", files={"file": ("clip.wav", _wav(np.zeros(10)))}
    )
    assert response.status_code == 400


def test_duplicate_fields(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        files=[
            ("model", (None, "whisper-test")),
            ("model", (None, "wrong")),
            ("file", ("clip.wav", _wav(np.zeros(10)), "audio/wav")),
        ],
    )
    assert response.status_code == 400


def test_file_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "MAX_AUDIO_UPLOAD_BYTES", 20)
    assert _post(client).status_code == 413


def test_text_and_json_responses() -> None:
    assert audio.transcription_response("hello", "json").body == b'{"text":"hello"}'
    response = audio.transcription_response("hello", "text")
    assert response.body == b"hello"
    assert response.media_type == "text/plain"


class _Processor:
    def __init__(self) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=16000, n_samples=16000)
        self.tokenizer = SimpleNamespace(decode=self.decode)
        self.english_only = False
        self.prefix_calls = []

    def get_decoder_prompt_ids(self, **kwargs) -> list:
        self.prefix_calls.append(kwargs)
        if self.english_only:
            if "language" in kwargs:
                raise ValueError("English-only tokenizer does not accept a language.")
            return []
        if kwargs["language"] not in ("de", "en"):
            raise ValueError("Unsupported language.")
        return [(3, 13), (1, 11), (2, 12)]

    @staticmethod
    def decode(tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return " ".join(map(str, tokens))


@pytest.fixture
def transcription_client() -> TestClient:
    app = FastAPI()
    app.state.processor = _Processor()
    app.state.model_config = SimpleNamespace(model_type="whisper", decoder_start_token_id=10)
    app.state.calls = []
    app.state.fail_generation = False

    async def generate(inputs: dict, temperature: float, request: Request) -> str:
        app.state.calls.append((inputs, temperature, request))
        if app.state.fail_generation:
            raise RuntimeError("engine failed")
        return "Hello world."

    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        try:
            return await audio.create_transcription_response(
                request,
                served_model="whisper-test",
                model_config=app.state.model_config,
                processor=app.state.processor,
                generate=generate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TestClient(app, raise_server_exceptions=False)


def test_transcription_handoff(transcription_client: TestClient) -> None:
    response = _post(transcription_client, data={"temperature": "0.4"})
    assert response.status_code == 200
    assert response.json() == {"text": "Hello world."}
    inputs, temperature, _ = transcription_client.app.state.calls[0]
    assert inputs["prompt"] == ""
    waveform, sampling_rate = inputs["multi_modal_data"]["audio"][0]
    assert waveform.shape == (1600,)
    assert waveform.dtype == np.float32
    assert sampling_rate == 16000
    assert temperature == 0.4
    assert transcription_client.app.state.processor.prefix_calls == []


def test_transcription_text_output(transcription_client: TestClient) -> None:
    response = _post(transcription_client, data={"response_format": "text"})
    assert response.status_code == 200
    assert response.text == "Hello world."
    assert response.headers["content-type"].startswith("text/plain")


def test_language_prefix(transcription_client: TestClient) -> None:
    assert _post(transcription_client, data={"language": "de"}).status_code == 200
    inputs, _, _ = transcription_client.app.state.calls[0]
    assert inputs["prompt"] == "10 11 12 13"
    assert transcription_client.app.state.processor.prefix_calls == [
        {"language": "de", "task": "transcribe", "no_timestamps": True}
    ]


def test_unsupported_language_does_not_generate(transcription_client: TestClient) -> None:
    assert _post(transcription_client, data={"language": "xx"}).status_code == 400
    assert transcription_client.app.state.calls == []


def test_english_only_checkpoint(transcription_client: TestClient) -> None:
    transcription_client.app.state.processor.english_only = True
    assert _post(transcription_client, data={"language": "en"}).status_code == 200
    assert transcription_client.app.state.calls[0][0]["prompt"] == "10"
    assert _post(transcription_client, data={"language": "de"}).status_code == 400
    assert len(transcription_client.app.state.calls) == 1


def test_wrong_model_type_does_not_generate(transcription_client: TestClient) -> None:
    transcription_client.app.state.model_config.model_type = "llama"
    assert _post(transcription_client).status_code == 400
    assert transcription_client.app.state.calls == []


def test_missing_processor_does_not_generate(transcription_client: TestClient) -> None:
    transcription_client.app.state.processor = None
    assert _post(transcription_client).status_code == 500
    assert transcription_client.app.state.calls == []


def test_engine_failure_does_not_return_success(transcription_client: TestClient) -> None:
    transcription_client.app.state.fail_generation = True
    assert _post(transcription_client).status_code == 500
