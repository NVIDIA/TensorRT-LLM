# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable exporter contracts in fresh processes with real Prometheus registries."""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _run_worker(mode: str) -> None:
    result = subprocess.run(
        [sys.executable, __file__, mode], capture_output=True, text=True, timeout=90, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mode",
    [
        "serving",
        "multiprocess",
        "lifecycle",
        "collection_requests",
        "collection_iterations",
        "http_enablement",
        "rpc_shutdown",
    ],
)
def test_prometheus_contract(mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect counts, group isolation, and real listener cleanup across process boundaries."""
    # Existing operator-owned samples must neither be scraped nor deleted.
    sentinel = tmp_path / "counter_999.db"
    sentinel.write_bytes(b"operator-owned metrics")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _run_worker(mode)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"operator-owned metrics"


def _sample(text: str, name: str, **labels: str) -> float:
    from prometheus_client.parser import text_string_to_metric_families

    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(key) == value for key, value in labels.items())
    )


async def _scrape(port: int) -> str:
    def read() -> str:
        with urlopen(f"http://127.0.0.1:{port}/prometheus/metrics", timeout=5) as response:
            assert response.status == 200
            assert "text/plain" in response.headers["Content-Type"]
            return response.read().decode()

    return await asyncio.to_thread(read)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        dtype="auto",
        backend="pytorch",
        return_perf_metrics=True,
        enable_iter_perf_stats=True,
        guided_decoding_backend=None,
        reasoning_parser=None,
        max_seq_len=100,
        max_batch_size=8,
        max_num_tokens=100,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        kv_cache_config=None,
        prometheus_metrics_config=SimpleNamespace(e2e_request_latency_buckets=[0.05, 0.15]),
    )


def _result_metrics(tokens: int, prompt_tokens: int = 0) -> dict:
    from tensorrt_llm.metrics.enums import MetricNames

    return {
        "finished_reason": "length",
        MetricNames.PROMPT_TOKENS: prompt_tokens,
        MetricNames.GENERATION_TOKENS: tokens,
        MetricNames.E2E: 0.1,
        MetricNames.TTFT: 0.02,
    }


async def _serving() -> None:
    import grpc
    from conftest import FakeServicerContext
    from test_servicer import _FakeLlm, _FakeResultHandle

    from tensorrt_llm.grpc.openengine.bindings import generation_pb2, openengine_pb2_grpc
    from tensorrt_llm.grpc.openengine.metrics import OpenEngineMetrics
    from tensorrt_llm.grpc.openengine.server import OpenEngineServer
    from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer

    gate = asyncio.Event()

    def output(index: int, tokens: list[int], reason: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            index=index,
            token_ids=tokens,
            text="A" * len(tokens),
            logprobs=[],
            prompt_logprobs=[],
            finish_reason=reason,
            stop_reason=None,
        )

    partial = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[output(0, [10], None), output(1, [11], None)],
        cached_tokens=0,
        error=None,
        finished=False,
    )
    final = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[output(0, [10, 12], "length"), output(1, [11], "length")],
        cached_tokens=0,
        error=None,
        finished=True,
        candidate_metrics=[_result_metrics(2, 3), _result_metrics(1)],
    )

    class Handle(_FakeResultHandle):
        async def __aiter__(self):
            async for result in super().__aiter__():
                yield result
                await gate.wait()

    llm = _FakeLlm([])
    llm.args = _args()
    llm.llm_id = "test-engine"
    llm._check_health = lambda: True
    reads = []

    def get_stats(timeout):
        del timeout
        reads.append(threading.get_ident())
        return [{"numActiveRequests": 2, "iterLatencyMS": 3}]

    import threading

    llm.get_stats = get_stats
    llm.result_handle = Handle([partial, final])
    metrics = OpenEngineMetrics(llm, "test-model", "127.0.0.1", 0, owner=True)
    server = OpenEngineServer("127.0.0.1", 0, llm, "test-model", metrics=metrics)
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
            stub = openengine_pb2_grpc.InferenceStub(channel)
            request = generation_pb2.GenerateRequest(
                request_id="two-outputs",
                model="alias",
                prompt="hello",
                sampling=generation_pb2.SamplingParams(num_sequences=2),
            )
            stream = stub.Generate(request, timeout=10)
            assert (await stream.read()).HasField("token")
            deadline = time.monotonic() + 5
            while True:
                text = await _scrape(metrics.port)
                if _sample(text, "trtllm_num_requests_running") == 2:
                    break
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
            assert _sample(text, "trtllm_openengine_rpc_outcomes_total") == 0
            assert reads and all(tid != threading.get_ident() for tid in reads)
            gate.set()
            while await stream.read() is not grpc.aio.EOF:
                pass
            text = await _scrape(metrics.port)
            assert _sample(text, "trtllm_request_success_total") == 2
            assert _sample(text, "trtllm_prompt_tokens_total") == 3
            assert _sample(text, "trtllm_generation_tokens_total") == 3
            assert _sample(text, "trtllm_e2e_request_latency_seconds_count") == 2
            assert _sample(text, "trtllm_e2e_request_latency_seconds_bucket", le="0.15") == 2
            assert _sample(text, "trtllm_openengine_rpc_outcomes_total", outcome="completed") == 1
            assert _sample(text, "trtllm_request_success_total", model_name="alias") == 0

            # An in-band engine error may finish with OK gRPC transport status.
            llm.result_handle = _FakeResultHandle(
                [SimpleNamespace(error="engine failed", outputs=[], finished=True)]
            )
            events = [event async for event in stub.Generate(request, timeout=5)]
            assert events[-1].HasField("error")
            text = await _scrape(metrics.port)
            assert (
                _sample(text, "trtllm_openengine_rpc_outcomes_total", outcome="engine_error") == 1
            )

            with pytest.raises(grpc.aio.AioRpcError):
                async for _ in stub.Generate(generation_pb2.GenerateRequest(), timeout=5):
                    pass
            text = await _scrape(metrics.port)
            assert _sample(text, "trtllm_openengine_rpc_outcomes_total", outcome="rejected") == 1

            # An exception inside engine submission remains an INTERNAL engine
            # failure, even if its type is also used by frontend coordination.
            from tensorrt_llm.grpc.openengine.coordinator import CoordinationError

            with pytest.MonkeyPatch.context() as patch:

                def fail_submit(**kwargs):
                    raise CoordinationError("submission failed")

                patch.setattr(llm, "generate_async", fail_submit)
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    async for _ in stub.Generate(request, timeout=5):
                        pass
                assert error.value.code() == grpc.StatusCode.INTERNAL
            text = await _scrape(metrics.port)
            assert (
                _sample(text, "trtllm_openengine_rpc_outcomes_total", outcome="engine_error") == 2
            )
            assert _sample(text, "trtllm_openengine_rpc_outcomes_total", outcome="unavailable") == 0

            # A recording failure must not change a successful inference result.
            def fail_record(_):
                raise OSError("metrics storage full")

            metrics.collector.log_request_metrics_dict = fail_record
            llm.result_handle = _FakeResultHandle([final])
            events = [event async for event in stub.Generate(request, timeout=5)]
            assert events[-1].HasField("finished")
            text = await _scrape(metrics.port)
            assert _sample(text, "trtllm_openengine_rpc_outcomes_total", outcome="completed") == 2
            assert _sample(text, "trtllm_openengine_metrics_errors_total", source="request") == 1

        # The watchdog must finalize even if its suspended generator never closes,
        # and closing it later must not record a second outcome.
        import tensorrt_llm.grpc.openengine.servicer as servicer_module

        servicer_module.RESPONSE_STALL_TIMEOUT_SECONDS = 0
        llm.result_handle = _FakeResultHandle([partial])
        servicer = OpenEngineInferenceServicer(llm, "test-model", metrics=metrics)
        responses = servicer.Generate(request, FakeServicerContext())
        await responses.__anext__()
        for _ in range(3):
            await asyncio.sleep(0)
        assert (
            _sample(
                await _scrape(metrics.port),
                "trtllm_openengine_rpc_outcomes_total",
                outcome="stalled",
            )
            == 1
        )
        await responses.aclose()
        assert (
            _sample(
                await _scrape(metrics.port),
                "trtllm_openengine_rpc_outcomes_total",
                outcome="stalled",
            )
            == 1
        )
    finally:
        await server.stop(0)


async def _multiprocess(attached: bool) -> None:
    from tensorrt_llm.grpc.openengine.metrics import OpenEngineMetrics

    def get_stats(timeout):
        del timeout
        assert not attached, "attached frontend consumed shared engine stats"
        return [
            {
                "attentionDpRank": 0,
                "numActiveRequests": 7,
                "numCompletedRequests": 2,
                "kvCacheStats": {"maxNumBlocks": 10, "usedNumBlocks": 3, "reusedBlocks": 2},
            },
            {
                "attentionDpRank": 1,
                "numActiveRequests": 7,
                "numCompletedRequests": 0,
                "kvCacheStats": {"maxNumBlocks": 10, "usedNumBlocks": 7, "reusedBlocks": 5},
            },
        ]

    llm = SimpleNamespace(args=_args(), get_stats=get_stats)
    metrics = OpenEngineMetrics(llm, "test-model", "127.0.0.1", 0, owner=not attached)
    metrics.record_result(SimpleNamespace(candidate_metrics=[_result_metrics(1, 2)]))
    await metrics.start()
    try:
        if attached:
            return
        _run_worker("attached")
        expected_kv = (("0", 3, 2), ("1", 7, 5))
        deadline = time.monotonic() + 5
        while True:
            text = await _scrape(metrics.port)
            if all(
                _sample(text, "trtllm_kv_cache_used_blocks", dp_rank=rank) == used
                and _sample(text, "trtllm_kv_cache_reused_blocks_total", dp_rank=rank) == reused
                for rank, used, reused in expected_kv
            ):
                break
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        assert _sample(text, "trtllm_request_success_total") == 2
        assert _sample(text, "trtllm_prompt_tokens_total") == 4
        assert _sample(text, "trtllm_num_requests_running") == 7
        assert _sample(text, "trtllm_num_requests_completed_total") == 2
        for rank, used, reused in expected_kv:
            assert _sample(text, "trtllm_kv_cache_used_blocks", dp_rank=rank) == used
            assert _sample(text, "trtllm_kv_cache_reused_blocks_total", dp_rank=rank) == reused
        assert "pid=" not in text
        # Another live group using the same parent cannot see these counters.
        _run_worker("isolated")
        assert _sample(await _scrape(metrics.port), "trtllm_request_success_total") == 2
    finally:
        await metrics.stop()


async def _lifecycle() -> None:
    import threading

    from tensorrt_llm.grpc.openengine.metrics import OpenEngineMetrics

    llm = SimpleNamespace(args=_args(), get_stats=lambda timeout: [])
    resolve = socket.getaddrinfo

    def ipv6_first(host, *args, **kwargs):
        addresses = resolve(host, *args, **kwargs)
        if host == "localhost":
            addresses.sort(key=lambda address: address[0] != socket.AF_INET6)
        return addresses

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket, "getaddrinfo", ipv6_first)
        localhost = OpenEngineMetrics(llm, "test-model", "localhost", 0, True)
        try:
            await localhost.start()
            await _scrape(localhost.port)  # Scrape over IPv4 even with IPv6-first DNS.
        finally:
            await localhost.stop()
            await localhost.wait_for_collection()
    if socket.has_ipv6:
        try:
            ipv6_socket = socket.create_server(("::1", 0), family=socket.AF_INET6)
        except OSError:
            pass  # IPv6 can be disabled even when the Python build supports it.
        else:
            ipv6_socket.close()

            def ipv6_only(host, *args, **kwargs):
                return resolve("::1" if host == "metrics-v6.test" else host, *args, **kwargs)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(socket, "getaddrinfo", ipv6_only)
                for host in ("[::1]", "metrics-v6.test"):
                    ipv6 = OpenEngineMetrics(llm, "test-model", host, 0, True)
                    try:
                        await ipv6.start()
                        await asyncio.to_thread(
                            lambda: socket.create_connection(("::1", ipv6.port), timeout=2).close()
                        )
                    finally:
                        await ipv6.stop()
                        await ipv6.wait_for_collection()
    with socket.create_server(("127.0.0.1", 0)) as occupied:
        service = OpenEngineMetrics(llm, "test-model", "127.0.0.1", occupied.getsockname()[1], True)
        try:
            with pytest.raises(OSError):
                await service.start()
        finally:
            await service.stop()
    service = OpenEngineMetrics(llm, "test-model", "127.0.0.1", 0, True)
    await service.start()
    port = service.port
    try:
        service.check_health()
        await _scrape(port)
        with pytest.raises(HTTPError) as error:
            await asyncio.to_thread(urlopen, f"http://127.0.0.1:{port}/health", timeout=5)
        assert error.value.code == 404
        service._http.should_exit = True
        await asyncio.to_thread(service._thread.join, 5)
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            service.check_health()
    finally:
        await service.stop()
    with socket.create_server(("127.0.0.1", port)):
        pass

    entered = threading.Event()
    release = threading.Event()

    def blocked_stats(timeout):
        del timeout
        entered.set()
        assert release.wait(10)
        return [{"numActiveRequests": 999}]

    llm.get_stats = blocked_stats
    service = OpenEngineMetrics(llm, "test-model", "127.0.0.1", 0, True)
    await service.start()
    try:
        while not entered.is_set():
            await asyncio.sleep(0.01)
        # The engine owner must reach engine shutdown before joining this RPC.
        await asyncio.wait_for(service.stop(), timeout=3)
    finally:
        release.set()
        await service.wait_for_collection()
        await service.stop()
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    assert _sample(generate_latest(registry).decode(), "trtllm_num_requests_running") == 0


async def _collection_mode(requests: bool) -> None:
    from tensorrt_llm.grpc.openengine.metrics import OpenEngineMetrics

    args = _args()
    args.return_perf_metrics = requests
    args.enable_iter_perf_stats = not requests
    reads = []

    def get_stats(timeout):
        del timeout
        reads.append(True)
        return [{"numActiveRequests": 7}]

    llm = SimpleNamespace(args=args, get_stats=get_stats)
    metrics = OpenEngineMetrics(llm, "test-model", "127.0.0.1", 0, True)
    metrics.record_result(SimpleNamespace(candidate_metrics=[_result_metrics(3, 2)]))
    metrics.record_outcome("aggregated", "completed")
    await metrics.start()
    try:
        if requests:
            await asyncio.sleep(1.2)
        else:
            deadline = time.monotonic() + 5
            while _sample(await _scrape(metrics.port), "trtllm_num_requests_running") != 7:
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
        metrics.check_health()
        text = await _scrape(metrics.port)
        assert bool(reads) is not requests
        assert _sample(text, "trtllm_request_success_total") == int(requests)
        assert _sample(text, "trtllm_generation_tokens_total") == (3 if requests else 0)
        assert _sample(text, "trtllm_num_requests_running") == (0 if requests else 7)
        assert _sample(text, "trtllm_openengine_rpc_outcomes_total", outcome="completed") == 1
    finally:
        await metrics.stop()
        await metrics.wait_for_collection()


async def _rpc_shutdown() -> None:
    """An unanswered stats RPC must time out or release its caller on close."""
    import threading

    from tensorrt_llm.executor.proxy import GenerationExecutorProxy
    from tensorrt_llm.executor.rpc import RPCCancelled, RPCClient, RPCServer, RPCTimeout
    from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr
    from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy

    for mode in (
        "deadline",
        "rpc_deadline",
        "sync",
        "future",
        "async",
        "queued",
        "send_sync",
        "send_future",
        "send_stream",
        "stream_admission",
    ):
        entered, release = threading.Event(), threading.Event()

        class App:
            async def fetch_stats_wait_async(self, timeout):
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.01)
                return []

        server = RPCServer(App(), async_run_task=True)
        server.bind(get_unique_ipc_addr())
        server.start()
        client = RPCClient(server.address, hmac_key=server.hmac_key, num_workers=1)
        proxy = SimpleNamespace(rpc_client=client)
        try:
            if mode == "stream_admission":
                from tensorrt_llm.executor.rpc import rpc_client

                class ClosingQueue(rpc_client.AsyncQueue):
                    def __init__(self):
                        super().__init__()
                        client.close()

                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(rpc_client, "AsyncQueue", ClosingQueue)
                    stream = client.never_stream().remote_streaming()
                    with pytest.raises(RPCCancelled):
                        await asyncio.wait_for(stream.__anext__(), 3)
                assert not client._streaming_queues
                continue
            if mode.startswith("send_"):

                async def blocked_send(request):
                    entered.set()
                    while not release.is_set():
                        await asyncio.sleep(0.01)

                client._client_socket.put_async = blocked_send
            if mode == "queued":
                # Hold the sole worker until close has stopped the RPC loop.
                blocker = client._executor.submit(release.wait)
                read = asyncio.wrap_future(
                    client.fetch_stats_wait_async(timeout=0.1).remote_future()
                )
                closing = asyncio.create_task(asyncio.to_thread(client.close))
                await asyncio.to_thread(client._loop_thread.join, 3)
                assert not client._loop_thread.is_alive()
                release.set()
                await asyncio.wait_for(closing, 3)
                assert blocker.result(timeout=1)
                with pytest.raises(RPCCancelled):
                    await asyncio.wait_for(read, 3)
                with pytest.raises(RPCCancelled):
                    GenerationExecutorProxy.get_stats(proxy, timeout=0.1)
                continue
            if mode == "send_stream":
                stream = client.never_stream().remote_streaming()
                read = asyncio.create_task(stream.__anext__())
            elif mode == "async":
                read = asyncio.create_task(
                    client.fetch_stats_wait_async(timeout=0.1).remote_async()
                )
            elif mode in ("future", "send_future"):
                read = asyncio.wrap_future(
                    client.fetch_stats_wait_async(timeout=0.1).remote_future()
                )
            else:
                proxy_type = (
                    GenerationExecutorRpcProxy
                    if mode == "rpc_deadline"
                    else GenerationExecutorProxy
                )
                read = asyncio.create_task(
                    asyncio.to_thread(proxy_type.get_stats, proxy, timeout=0.1)
                )
            assert await asyncio.to_thread(entered.wait, 3)
            if mode == "deadline":
                with pytest.raises(RPCTimeout):
                    await asyncio.wait_for(read, 3)
            elif mode == "rpc_deadline":
                assert await asyncio.wait_for(read, 3) == []
            else:
                await asyncio.wait_for(asyncio.to_thread(client.close), 3)
                with pytest.raises(RPCCancelled):
                    await asyncio.wait_for(read, 3)
                assert not client._streaming_queues
        finally:
            release.set()
            await asyncio.to_thread(client.close)
            if mode.startswith("send_"):
                # Drain cancellation callbacks queued after the deliberately stopped loop.
                async def drain_cancelled_sends():
                    pending = asyncio.all_tasks() - {asyncio.current_task()}
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                drain_thread = threading.Thread(target=client._loop.run_forever)
                drain_thread.start()
                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(drain_cancelled_sends(), client._loop)
                )
                client._loop.call_soon_threadsafe(client._loop.stop)
                await asyncio.to_thread(drain_thread.join, 3)
                client._loop.close()
            await asyncio.to_thread(server.shutdown)


def _http_enablement() -> None:
    """File collection must expose Prometheus without opting HTTP headers in."""
    from tensorrt_llm.serve.openai_server import OpenAIServer

    with pytest.MonkeyPatch.context() as patch:
        # Exercise the real constructor and route registration without loading a model.
        patch.setattr(
            OpenAIServer, "_init_llm", lambda self, _: setattr(self, "use_harmony", False)
        )
        for requested, output_dir in [(False, None), (True, None), (False, "unused-jsonl")]:
            args = _args()
            args.return_perf_metrics = requested
            args.perf_metrics_output_dir = output_dir
            generator = SimpleNamespace(
                args=args, _executor=SimpleNamespace(resource_governor_queue=None)
            )
            server = OpenAIServer(generator, "test-model", None, None, None)
            try:
                enabled = requested or output_dir is not None
                assert (
                    any(route.path == "/prometheus/metrics" for route in server.app.routes)
                    == enabled
                )
                assert server._expose_perf_metrics is requested
                assert server._collect_perf_metrics is enabled
                assert args.return_perf_metrics is enabled
            finally:
                server._input_proc_executor.shutdown(wait=True)
                server._media_load_executor.shutdown(wait=True)
                server._metrics_stats_executor.shutdown(wait=True)


if __name__ == "__main__":
    from tensorrt_llm.grpc.openengine.metrics import MetricsDirectory

    mode = sys.argv[1]
    previous = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    directory = MetricsDirectory(attached=mode == "attached")
    owned_path = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"])
    try:
        if mode == "isolated":
            assert not list(owned_path.glob("*.db"))
        elif mode in ("attached", "multiprocess"):
            asyncio.run(_multiprocess(attached=mode == "attached"))
        elif mode == "serving":
            asyncio.run(_serving())
        elif mode.startswith("collection_"):
            asyncio.run(_collection_mode(requests=mode == "collection_requests"))
        elif mode == "http_enablement":
            _http_enablement()
        elif mode == "rpc_shutdown":
            asyncio.run(_rpc_shutdown())
        else:
            asyncio.run(_lifecycle())
    finally:
        directory.close()
    if mode != "attached":
        assert not owned_path.exists()
        assert os.environ.get("PROMETHEUS_MULTIPROC_DIR") == previous
