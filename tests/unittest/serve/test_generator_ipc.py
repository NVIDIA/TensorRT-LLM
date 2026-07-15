# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import itertools
from dataclasses import dataclass, field

import pytest

from tensorrt_llm.serve.generator_ipc import (
    GeneratorIpcClient,
    GeneratorIpcClosedError,
    GeneratorIpcRemoteError,
    GeneratorIpcServer,
)


@dataclass
class _FakeCompletion:
    index: int = 0
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def text_diff(self) -> str:
        return self.text[-1:]

    @property
    def token_ids_diff(self) -> list[int]:
        return self.token_ids[-1:]


class _FakePromise:
    def __init__(
        self,
        request_id: int,
        prompt_token_ids: list[int],
        label: str,
        *,
        fail: bool = False,
    ) -> None:
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [_FakeCompletion()]
        self.finished = False
        self.error = None
        self.prompt = None
        self.context_logits = None
        self.disaggregated_params = None
        self.metrics_dict = {}
        self.time_breakdown_metrics = None
        self.cached_tokens = 2
        self.avg_decoded_tokens_per_iter = 1.5
        self._label = label
        self._fail = fail
        self._step = 0
        self._aborted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        delay = 0.1 if self._label == "alias" else (0.005 if self._label == "fast" else 0.01)
        await asyncio.sleep(delay)
        if self._aborted:
            self.finished = True
            self.outputs[0].finish_reason = "cancelled"
            raise StopAsyncIteration
        if self._fail and self._step == 1:
            raise ValueError("fake generation failed")
        if self._step == 3:
            self.finished = True
            self.outputs[0].finish_reason = "stop"
            raise StopAsyncIteration
        self._step += 1
        self.outputs[0].text += f"{self._label}{self._step}"
        self.outputs[0].token_ids.append(self._step)
        return self

    def abort(self) -> None:
        self._aborted = True


class _FakeUnaryPromise:
    def __init__(self, value):
        self._value = value
        self._aborted = False

    async def aresult(self):
        await asyncio.sleep(0.01)
        return self._value

    def __await__(self):
        return self.aresult().__await__()

    def abort(self):
        self._aborted = True


class _FakeGenerator:
    def __init__(self) -> None:
        self._request_ids = itertools.count(100)
        self.promises: dict[int, _FakePromise] = {}

    def _check_health(self) -> bool:
        return True

    def add(self, left: int, right: int) -> int:
        return left + right

    async def async_echo(self, value: str) -> str:
        await asyncio.sleep(0.005)
        return value

    def fail_unary(self) -> None:
        raise RuntimeError("fake unary failed")

    def generate_async(
        self,
        inputs: list[int],
        *,
        label: str,
        fail: bool = False,
        unary: bool = False,
        **kwargs,
    ):
        del kwargs
        if unary:
            if label == "visual":
                import torch

                from tensorrt_llm.visual_gen.output import VisualGenOutput

                return _FakeUnaryPromise(VisualGenOutput(image=torch.ones((1, 3, 2, 2))))
            return _FakeUnaryPromise({"label": label})
        request_id = next(self._request_ids)
        promise = _FakePromise(request_id, inputs, label, fail=fail)
        self.promises[request_id] = promise
        return promise


@pytest.fixture
def gateway(tmp_path):
    endpoint = f"ipc://{tmp_path / 'generator.sock'}"
    generator = _FakeGenerator()
    server = GeneratorIpcServer(generator, endpoint)
    server.start()
    try:
        yield generator, server
    finally:
        server.close()


@pytest.mark.asyncio
async def test_concurrent_interleaved_streams_and_unary_calls(gateway):
    _, server = gateway
    with GeneratorIpcClient(server.address, queue_size=8) as client:
        slow = client.generate_async([1, 2], label="slow", streaming=True)
        fast = client.generate_async([3], label="fast", streaming=True)

        async def collect(promise):
            return [
                (output.request_id, output.prompt_token_ids, output.outputs[0].text)
                async for output in promise
            ]

        slow_results, fast_results, add_result, echo_result = await asyncio.gather(
            collect(slow),
            collect(fast),
            client.call_async("add", 20, 22),
            client.call_async("async_echo", "hello"),
        )

        assert [item[2] for item in slow_results] == ["slow1", "slow1slow2", "slow1slow2slow3"]
        assert [item[2] for item in fast_results] == ["fast1", "fast1fast2", "fast1fast2fast3"]
        assert {item[0] for item in slow_results} == {slow.request_id}
        assert {item[0] for item in fast_results} == {fast.request_id}
        assert slow_results[0][1] == [1, 2]
        assert fast_results[0][1] == [3]
        assert slow.cached_tokens == 2
        assert slow.avg_decoded_tokens_per_iter == 1.5
        assert slow.id == slow.request_id
        assert slow._done
        assert add_result == 42
        assert echo_result == "hello"


@pytest.mark.asyncio
async def test_multiple_client_identities_are_routed_independently(gateway):
    _, server = gateway
    with (
        GeneratorIpcClient(server.address) as first_client,
        GeneratorIpcClient(server.address) as second_client,
    ):
        first = first_client.generate_async([1], label="first")
        second = second_client.generate_async([2], label="second")
        await asyncio.gather(first.aresult(), second.aresult())

        assert first.outputs[0].text == "first1first2first3"
        assert second.outputs[0].text == "second1second2second3"


@pytest.mark.asyncio
async def test_non_llm_generation_uses_unary_awaitable(gateway):
    _, server = gateway
    with GeneratorIpcClient(server.address) as client:
        output = client.generate_async([], label="image", unary=True)
        assert await output == {"label": "image"}


@pytest.mark.asyncio
async def test_visual_output_uses_shared_file_payload(gateway, tmp_path, monkeypatch):
    monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path))
    _, server = gateway
    with GeneratorIpcClient(server.address) as client:
        future = client.generate_async([], label="visual", unary=True)
        output = await future
        assert output.image.shape == (1, 3, 2, 2)
        assert not list(tmp_path.glob("ipc_*.pt"))


@pytest.mark.asyncio
async def test_abort_reaches_real_promise(gateway):
    generator, server = gateway
    with GeneratorIpcClient(server.address) as client:
        output = client.generate_async([1], label="abort")
        await output.__anext__()
        output.abort()
        await output.aresult()

        assert generator.promises[output.request_id]._aborted
        assert output.aborted()


@pytest.mark.asyncio
async def test_client_disconnect_aborts_outstanding_generation(gateway):
    generator, server = gateway
    client = GeneratorIpcClient(server.address)
    output = client.generate_async([1], label="disconnect")
    await output.__anext__()
    client.close()
    await asyncio.sleep(0.1)
    assert generator.promises[output.request_id]._aborted


@pytest.mark.asyncio
async def test_cross_client_alias_cancels_generation(gateway):
    generator, server = gateway
    with (
        GeneratorIpcClient(server.address) as owner,
        GeneratorIpcClient(server.address) as peer,
    ):
        output = owner.generate_async([1], label="alias")
        owner.register_alias("video_test", output)
        assert await peer.cancel_alias("video_test")
        await output.aresult()
        assert generator.promises[output.request_id]._aborted


@pytest.mark.asyncio
async def test_slow_stream_applies_backpressure_without_blocking_other_requests(gateway):
    _, server = gateway
    with GeneratorIpcClient(server.address, queue_size=1) as client:
        output = client.generate_async([1], label="slow")
        await asyncio.sleep(0.1)
        assert client.call("add", 1, 2) == 3
        await output.aresult()
        assert output.outputs[0].text == "slow1slow2slow3"


@pytest.mark.asyncio
async def test_remote_stream_and_unary_errors(gateway):
    _, server = gateway
    with GeneratorIpcClient(server.address) as client:
        output = client.generate_async([1], label="error", fail=True)
        await output.__anext__()
        with pytest.raises(GeneratorIpcRemoteError, match="fake generation failed"):
            await output.__anext__()
        with pytest.raises(GeneratorIpcRemoteError, match="fake unary failed"):
            await client.call_async("fail_unary")


@pytest.mark.asyncio
async def test_server_close_propagates_to_active_stream(gateway):
    _, server = gateway
    client = GeneratorIpcClient(server.address)
    output = client.generate_async([1], label="close")
    server.close()
    try:
        with pytest.raises(GeneratorIpcClosedError):
            while True:
                await output.__anext__()
    finally:
        client.close()
