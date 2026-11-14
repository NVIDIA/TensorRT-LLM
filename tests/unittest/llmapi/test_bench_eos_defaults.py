# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EOS defaults of trtllm-bench and benchmark_serving under speculative decoding."""

import asyncio
import gc
import json
import random
import socket
from argparse import Namespace
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import numpy as np
import pytest
import torch
from aiohttp import web
from click.testing import CliRunner
from transformers import LlamaConfig, LlamaForCausalLM

from tensorrt_llm.bench.benchmark import low_latency, throughput
from tensorrt_llm.bench.benchmark.utils import general
from tensorrt_llm.bench.benchmark.utils.general import resolve_end_id
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.llmapi.disagg_utils import CtxGenServerConfig, DisaggServerConfig, ServerRole
from tensorrt_llm.metrics.collector import speculative_config_info
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import ModelCard, ModelList
from tensorrt_llm.serve.scripts import benchmark_serving
from tensorrt_llm.serve.scripts.benchmark_serving import (
    server_uses_speculative_decoding,
    speculative_decoding_from_models,
)

pytestmark = pytest.mark.cpu_only

EAGLE3_ARGS = SimpleNamespace(
    speculative_config=SimpleNamespace(
        decoding_type="Eagle3", max_draft_len=3, speculative_model="draft"
    )
)
NO_SPEC_ARGS = SimpleNamespace(speculative_config=None, decoding_config=None)


@pytest.mark.parametrize(
    "llm_args, end_id, expected",
    [
        (EAGLE3_ARGS, None, 2),
        (NO_SPEC_ARGS, None, -1),
        (EAGLE3_ARGS, -1, -1),
        (NO_SPEC_ARGS, 7, 7),
    ],
)
def test_resolve_end_id(llm_args, end_id, expected):
    assert resolve_end_id(llm_args, 2, end_id) == expected


def test_speculative_config_info():
    assert speculative_config_info(NO_SPEC_ARGS) is None
    assert speculative_config_info(EAGLE3_ARGS) == {
        "spec_enabled": "true",
        "spec_method": "Eagle3",
        "spec_num_tokens": "3",
        "spec_draft_model": "draft",
    }


def _models_payload(args):
    card = ModelCard(id="m", metadata={"speculative_config": speculative_config_info(args)})
    return ModelList(data=[card]).model_dump()


@pytest.mark.parametrize(
    "payload, expected",
    [
        (_models_payload(EAGLE3_ARGS), True),
        (_models_payload(NO_SPEC_ARGS), False),
        (ModelList(data=[ModelCard(id="m")]).model_dump(), None),
        ({"object": "list", "data": []}, None),
        ({"data": [{"id": "m", "metadata": {}}]}, None),
    ],
)
def test_speculative_decoding_from_models(payload, expected):
    assert speculative_decoding_from_models(payload) is expected


@pytest.mark.parametrize("args, expected", [(EAGLE3_ARGS, True), (NO_SPEC_ARGS, False)])
def test_server_uses_speculative_decoding(args, expected, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def run():
        app = web.Application()

        async def models(_):
            return web.json_response(_models_payload(args))

        app.router.add_get("/v1/models", models)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        try:
            port = runner.addresses[0][1]
            return await server_uses_speculative_decoding(f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) is expected


def test_server_uses_speculative_decoding_unreachable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        assert asyncio.run(server_uses_speculative_decoding(url)) is None


@pytest.mark.parametrize("args, expected", [(EAGLE3_ARGS, True), (NO_SPEC_ARGS, False)])
def test_server_uses_speculative_decoding_authenticated(args, expected, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-models-access")

    async def run():
        app = web.Application()

        async def models(request):
            if request.headers.get("Authorization") != "Bearer test-models-access":
                raise web.HTTPUnauthorized()
            return web.json_response(_models_payload(args))

        app.router.add_get("/v1/models", models)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        try:
            port = runner.addresses[0][1]
            return await server_uses_speculative_decoding(f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) is expected


def test_server_uses_speculative_decoding_short_timeout(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class TimedOutSession:
        def __init__(self, *, trust_env, timeout):
            assert 0 < timeout.total <= 5

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def get(self, url, **kwargs):
            raise asyncio.TimeoutError

    monkeypatch.setattr(aiohttp, "ClientSession", TimedOutSession)
    assert asyncio.run(server_uses_speculative_decoding("http://localhost")) is None


@pytest.mark.parametrize("generation_args, expected", [(EAGLE3_ARGS, True), (NO_SPEC_ARGS, False)])
def test_disagg_models_use_generation_decoding_config(generation_args, expected):
    config = DisaggServerConfig(
        server_configs=[
            CtxGenServerConfig(type="ctx", hostname="context", port=8001),
            CtxGenServerConfig(type="gen", hostname="generation", port=8002),
        ]
    )

    worker_args = {"context:8001": NO_SPEC_ARGS, "generation:8002": generation_args}

    def client_factory(router, role: ServerRole, max_retries):
        async def get_json(endpoint, response_type, server):
            return response_type.model_validate(_models_payload(worker_args[server]))

        return SimpleNamespace(get_json=get_json, shutdown=AsyncMock())

    coordinator = SimpleNamespace(
        ctx_router=SimpleNamespace(servers=["context:8001"]),
        gen_router=SimpleNamespace(servers=["generation:8002"]),
        is_ready=AsyncMock(return_value=True),
        start=AsyncMock(),
        stop=AsyncMock(),
    )
    service = OpenAIDisaggregatedService(config, coordinator, client_factory=client_factory)

    async def run():
        await service.setup()
        try:
            return await service.get_model()
        finally:
            await service.teardown()

    models = asyncio.run(run())
    assert speculative_decoding_from_models(models.model_dump()) is expected


class _BenchmarkStarted(BaseException):
    pass


@pytest.mark.parametrize(
    "command_module, speculative, override, expected",
    [
        (throughput, True, None, (2, 2)),
        (throughput, False, None, (-1, -1)),
        (throughput, True, -1, (-1, -1)),
        (throughput, False, 7, (7, 7)),
        (low_latency, True, None, (2, 0)),
        (low_latency, False, None, (-1, -1)),
    ],
    ids=[
        "throughput-spec",
        "throughput-plain",
        "throughput-ignore-override",
        "throughput-eos-override",
        "latency-spec",
        "latency-plain",
    ],
)
def test_bench_command_sampling_ids(
    command_module, speculative, override, expected, tmp_path, monkeypatch
):
    monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
    model_dir = tmp_path / "model"
    model_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        max_position_embeddings=64,
    )
    with torch.random.fork_rng(devices=[]):
        LlamaForCausalLM(model_config).save_pretrained(model_dir)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps({"task_id": 0, "input_ids": [1, 3], "output_tokens": 4}) + "\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        "speculative_config:\n  decoding_type: Eagle3\n  max_draft_len: 3\n  speculative_model: draft\n"
        if speculative
        else "{}\n"
    )
    tokenizer = Mock(eos_token_id=2, pad_token_id=0)
    monkeypatch.setattr(command_module, "initialize_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(general, "get_benchmark_engine_settings", lambda *args, **kwargs: (1, 16))

    def get_llm(runtime_config, kwargs):
        return Mock(
            args=SimpleNamespace(speculative_config=kwargs.get("speculative_config")),
            startup_metrics={},
        )

    monkeypatch.setattr(command_module, "get_llm", get_llm)
    sampled = {}

    async def capture_sampling(llm, sampling_params, *args, **kwargs):
        sampled["ids"] = (sampling_params.end_id, sampling_params.pad_id)
        raise _BenchmarkStarted

    monkeypatch.setattr(command_module, "async_benchmark", capture_sampling)
    arguments = [
        "--dataset",
        str(dataset),
        "--config",
        str(config),
        "--num_requests",
        "1",
        "--warmup",
        "0",
    ]
    if override is not None:
        arguments += ["--eos_id", str(override)]
    command = (
        throughput.throughput_command
        if command_module is throughput
        else low_latency.latency_command
    )
    environment = BenchmarkEnvironment(model="model", checkpoint_path=model_dir, workspace=tmp_path)
    with pytest.raises(_BenchmarkStarted):
        CliRunner().invoke(command, arguments, obj=environment, catch_exceptions=False)
    assert sampled["ids"] == expected


@contextmanager
def _models_endpoint(payload):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            return None

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    "payload, explicit, expected",
    [
        (_models_payload(EAGLE3_ARGS), None, False),
        (_models_payload(NO_SPEC_ARGS), None, True),
        (ModelList(data=[ModelCard(id="m")]).model_dump(), None, False),
        (_models_payload(NO_SPEC_ARGS), False, False),
    ],
    ids=["speculative", "plain", "unknown", "explicit-no-ignore-eos"],
)
def test_serving_main_passes_automatic_eos_to_benchmark(payload, explicit, expected, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(benchmark_serving, "get_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        benchmark_serving.RandomDataset,
        "sample",
        lambda *args, **kwargs: [benchmark_serving.SampleRequest("hello", 1, 4)],
    )
    sampled = {}

    async def capture_benchmark(**kwargs):
        sampled["ignore_eos"] = kwargs["ignore_eos"]
        return {}

    monkeypatch.setattr(benchmark_serving, "benchmark", capture_benchmark)
    with _models_endpoint(payload) as (base_url, requests):
        args = Namespace(
            seed=0,
            backend="openai",
            model="m",
            served_model_name=None,
            tokenizer=None,
            tokenizer_mode="auto",
            base_url=base_url,
            endpoint="/v1/completions",
            trust_remote_code=False,
            dataset_name="random",
            dataset_path=None,
            random_ids=True,
            download_path=None,
            download_timeout=180,
            num_prompts=1,
            random_prefix_len=0,
            random_input_len=1,
            random_output_len=4,
            random_range_ratio=0.0,
            tokenize_on_client=False,
            goodput=None,
            top_p=None,
            top_k=None,
            min_p=None,
            temperature=None,
            ignore_eos=explicit,
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            profile=False,
            percentile_metrics="ttft",
            metric_percentiles="99",
            max_concurrency=1,
            lora_modules=None,
            non_streaming=False,
            no_test_input=True,
            save_result=False,
        )
        gc_enabled = gc.isenabled()
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            benchmark_serving.main(args)
        finally:
            if gc_enabled:
                gc.enable()
            random.setstate(random_state)
            np.random.set_state(numpy_state)
        assert bool(requests) is (explicit is None)
    assert sampled["ignore_eos"] is expected
