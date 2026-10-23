# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving integration checks using a fake generator, without GPU inference."""

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


class _Result:
    def __init__(self, *, fail: bool = False, cancel: bool = False) -> None:
        self.outputs = [SimpleNamespace(text="A short transcript.")]
        self.fail = fail
        self.cancel = cancel
        self.aborted = False

    async def aresult(self):
        await asyncio.sleep(0)
        if self.cancel:
            raise asyncio.CancelledError
        if self.fail:
            raise RuntimeError("backend failed")
        return self

    def abort(self) -> None:
        self.aborted = True


@pytest.fixture
def server():
    instance = object.__new__(OpenAIServer)
    instance.app = FastAPI()
    instance.model = "whisper-test"
    instance.model_config = SimpleNamespace(
        model_type="whisper", decoder_start_token_id=1, max_target_positions=448
    )
    instance.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000, n_samples=16000)
    )
    instance.use_harmony = False
    instance._register_runtime_control_routes = lambda: None
    instance._register_rl_control_routes = lambda: None
    instance.result = _Result()
    instance.calls = []
    instance.preprocessed = []
    instance.watch_stopped = False

    def preprocess(inputs, sampling_params):
        instance.preprocessed.append((inputs, sampling_params))
        return inputs

    def generate(**kwargs):
        instance.calls.append(kwargs)
        return instance.result

    async def watch(request, result):
        try:
            await asyncio.Event().wait()
        finally:
            instance.watch_stopped = True

    instance.await_disconnected = watch
    instance.generator = SimpleNamespace(
        preprocess=preprocess,
        generate_async=generate,
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        instance._input_proc_executor = pool
        yield instance


def _upload(client: TestClient, **fields):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600), 16000, format="WAV")
    return client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-test", **fields},
        files={"file": ("short.wav", buffer.getvalue(), "audio/wav")},
    )


def _client(server) -> TestClient:
    server.app.add_api_route(
        "/v1/audio/transcriptions", server.openai_transcription, methods=["POST"]
    )
    return TestClient(server.app)


def test_generation_and_disconnect_cleanup(server) -> None:
    response = _upload(_client(server))
    assert response.status_code == 200
    assert response.json() == {"text": "A short transcript."}
    assert len(server.preprocessed) == 1
    params = server.calls[0]["sampling_params"]
    assert params.max_tokens == 447
    assert params.temperature == 0
    assert params.top_k == 1
    assert server.calls[0]["streaming"] is False
    assert server.watch_stopped
    assert not server.result.aborted


def test_sampling_temperature_and_text_response(server) -> None:
    response = _upload(_client(server), temperature="0.4", response_format="text")
    assert response.status_code == 200
    assert response.text == "A short transcript."
    assert server.calls[0]["sampling_params"].temperature == 0.4
    assert server.calls[0]["sampling_params"].top_k == 0


def test_backend_error_is_500_and_cleans_up(server) -> None:
    server.result.fail = True
    response = _upload(_client(server))
    assert response.status_code == 500
    assert response.json()["type"] == "InternalServerError"
    assert server.watch_stopped


def test_unknown_model_error_envelope(server) -> None:
    response = _upload(_client(server), model="other")
    assert response.status_code == 404
    assert response.json()["object"] == "error"
    assert response.json()["code"] == 404
    assert server.calls == []


def test_validation_error_envelope(server) -> None:
    response = _upload(_client(server), prompt="context")
    assert response.status_code == 400
    assert response.json()["type"] == "BadRequestError"
    assert server.calls == []


def test_cancelled_generation_aborts_backend(server) -> None:
    server.result.cancel = True

    async def run() -> None:
        with pytest.raises(asyncio.CancelledError):
            await server._generate_audio_transcription(
                {"prompt": "", "multi_modal_data": {}},
                0.0,
                SimpleNamespace(headers={}),
            )

    asyncio.run(run())
    assert server.result.aborted
    assert server.watch_stopped


def test_route_is_registered(server) -> None:
    server.register_routes()
    assert "/v1/audio/transcriptions" in {route.path for route in server.app.routes}
