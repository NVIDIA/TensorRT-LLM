# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace

import pytest

from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.generator_ipc import GeneratorIpcServer
from tensorrt_llm.serve.generator_proxy import GeneratorProxy, GeneratorService


class _Generator:
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            skip_tokenizer_init=True, encode_only=False, model="/tmp/model", trust_remote_code=False
        )
        self._hf_model_dir = "/tmp/model"
        self.llm_id = "test-generator"
        self.model = "test-model"
        self.disaggregated_params = None
        self._executor = SimpleNamespace(resource_governor_queue=None, _fatal_error=None)
        self.kv_cache_events = []
        self.iteration_stats = []

    def _check_health(self) -> bool:
        return self._executor._fatal_error is None

    async def get_kv_cache_events_async(self, timeout):
        del timeout
        while self.kv_cache_events:
            yield self.kv_cache_events.pop(0)

    async def get_stats_async(self, timeout):
        del timeout
        while self.iteration_stats:
            yield self.iteration_stats.pop(0)

    def shutdown(self) -> None:
        pass


@pytest.fixture
def proxy_gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm.serve.generator_proxy.ModelLoader.load_hf_model_config",
        lambda *args, **kwargs: SimpleNamespace(is_encoder_decoder=False),
    )
    monkeypatch.setattr(
        "tensorrt_llm.serve.generator_proxy.ModelLoader.load_hf_generation_config",
        lambda *args, **kwargs: None,
    )
    generator = _Generator()
    endpoint = f"ipc://{tmp_path / 'generator.sock'}"
    server = GeneratorIpcServer(GeneratorService(generator), endpoint)
    server.start()
    proxies = []

    def create_proxy(*, owns_lifecycle=False):
        proxy = GeneratorProxy(server.address, owns_lifecycle=owns_lifecycle)
        proxies.append(proxy)
        return proxy

    try:
        yield generator, create_proxy
    finally:
        for proxy in proxies:
            proxy.shutdown()
        server.close()


def test_preprocess_runs_in_frontend_without_ipc():
    proxy = GeneratorProxy.__new__(GeneratorProxy)
    proxy.args = SimpleNamespace(
        backend="pytorch",
        reasoning_parser=None,
        return_perf_metrics=False,
        stream_interval=1,
    )
    proxy.tokenizer = None
    proxy.input_processor = None
    proxy._hf_model_config = SimpleNamespace(is_encoder_decoder=False)
    proxy._generation_config = None

    result = proxy.preprocess(
        {"prompt_token_ids": [1, 2, 3]},
        SamplingParams(end_id=2, max_tokens=1),
    )

    assert result.prompt_token_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_remote_stores_are_shared_across_frontends(proxy_gateway):
    _, create_proxy = proxy_gateway
    first = create_proxy()
    second = create_proxy()
    try:
        response = SimpleNamespace(id="resp_test", output=[])
        await first.conversation_store.store_response(response)
        loaded = await second.conversation_store.load_response("resp_test")
        assert loaded == response

        job = SimpleNamespace(id="video_test", status="queued")
        await first.video_store.upsert(job.id, job)
        assert await second.video_store.get(job.id) == job
    finally:
        first.shutdown()
        second.shutdown()


@pytest.mark.asyncio
async def test_kv_cache_events_are_drained_once_across_frontends(proxy_gateway):
    generator, create_proxy = proxy_gateway
    generator.kv_cache_events.extend([{"event_id": 1}, {"event_id": 2}])
    first = create_proxy(owns_lifecycle=True)
    second = create_proxy()
    try:
        first_events = [event async for event in first.get_kv_cache_events_async(0)]
        second_events = [event async for event in second.get_kv_cache_events_async(0)]
        assert first_events == [{"event_id": 1}, {"event_id": 2}]
        assert second_events == []

        generator.kv_cache_events.append({"event_id": 3})
        second_events = [event async for event in second.get_kv_cache_events_async(0)]
        assert second_events == [{"event_id": 3}]
    finally:
        first.shutdown()
        second.shutdown()


@pytest.mark.asyncio
async def test_iteration_stats_buffer_is_shared_and_destructive(proxy_gateway):
    generator, create_proxy = proxy_gateway
    generator.iteration_stats.extend([{"iteration": 1}, {"iteration": 2}])
    first = create_proxy(owns_lifecycle=True)
    second = create_proxy()
    try:
        drained = [stat async for stat in first.get_stats_async(0)]
        assert drained == [{"iteration": 1}, {"iteration": 2}]
        assert await second.take_iteration_stats(0) == drained
        assert await first.take_iteration_stats(0) == []

        generator.iteration_stats.append({"iteration": 3})
        assert await second.take_iteration_stats(0) == [{"iteration": 3}]
    finally:
        first.shutdown()
        second.shutdown()


def test_fatal_health_status_is_propagated_to_frontend(proxy_gateway):
    generator, create_proxy = proxy_gateway
    proxy = create_proxy()
    try:
        generator._executor._fatal_error = RuntimeError("engine failed")
        assert not proxy._check_health()
        assert proxy._executor._fatal_error == "engine failed"
    finally:
        proxy.shutdown()
