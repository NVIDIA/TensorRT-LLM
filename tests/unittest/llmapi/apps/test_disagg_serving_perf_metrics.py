import os
from typing import Tuple

import openai
import pytest
import requests
from test_common.http_utils import wait_for_endpoint_ready
from test_common.perf_metrics_utils import (
    get_prometheus_metrics,
    get_timing_metrics,
    validate_timing_metrics,
    wait_for_perf_metrics_jsonl,
)
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.serve.perf_metrics import (
    CTX_CHUNK_METRICS_HEADER,
    RETURN_METRICS_HEADER,
    SERVER_TIMING_HEADER,
    SSE_METRICS_EVENT,
    START_END_TIME_HEADER,
    STEP_METRICS_HEADER,
)

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer


@pytest.fixture
def test_ports():
    return get_free_ports(3)


@pytest.fixture
def disagg_port(test_ports: list[int]):
    return test_ports[0]


@pytest.fixture
def ctx_port(test_ports: list[int]):
    return test_ports[1]


@pytest.fixture
def gen_port(test_ports: list[int]):
    return test_ports[2]


@pytest.fixture
def model_name():
    model_path = os.path.join(llm_models_root(), "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    return model_path


@pytest.fixture
def perf_metrics_output_dir(tmp_path):
    return tmp_path / "perf_metrics"


@pytest.fixture
def disagg_cluster_config(disagg_port: int):
    return {
        "cluster_uri": f"http://localhost:{disagg_port}",
        "cluster_name": "",
    }


def worker_config(model_name: str, disagg_cluster_config: dict, perf_metrics_output_dir):
    return {
        "model": model_name,
        "disagg_cluster": disagg_cluster_config,
        "cache_transceiver_config": {
            "backend": "DEFAULT",
        },
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.2,
            "enable_block_reuse": False,
        },
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
        "return_perf_metrics": True,
        "perf_metrics_output_dir": str(perf_metrics_output_dir),
        "perf_metrics_max_requests": 1000,
    }


@pytest.fixture
def workers(
    model_name: str,
    disagg_cluster_config: dict,
    ctx_port: int,
    gen_port: int,
    perf_metrics_output_dir,
):
    model_path = get_model_path(model_name)
    extra_config = worker_config(model_name, disagg_cluster_config, perf_metrics_output_dir)

    def worker(server_role: str, port: int):
        return RemoteOpenAIServer(
            model_path,
            port=port,
            env=os.environ.copy(),
            cli_args=["--server_role", server_role],
            llmapi_launch=False,
            extra_config=extra_config,
            log_path=f"output_{server_role}.log",
            wait=False,
        )

    with worker("context", ctx_port) as ctx_worker, worker("generation", gen_port) as gen_worker:
        yield ctx_worker, gen_worker


@pytest.fixture
def disagg_server(disagg_cluster_config: dict, workers, disagg_port: int, perf_metrics_output_dir):
    disagg_config = {
        "hostname": "localhost",
        "port": disagg_port,
        "disagg_cluster": disagg_cluster_config,
        "perf_metrics_max_requests": 1000,
        "return_perf_metrics": True,
        "perf_metrics_output_dir": str(perf_metrics_output_dir),
    }
    with RemoteDisaggOpenAIServer(
        ctx_servers=[],
        gen_servers=[],
        port=disagg_config["port"],
        llmapi_launch=False,
        disagg_config=disagg_config,
    ) as server:
        yield server


@pytest.fixture
def client(disagg_server: RemoteDisaggOpenAIServer):
    return disagg_server.get_client()


@pytest.fixture
def async_client(disagg_server: RemoteDisaggOpenAIServer):
    return disagg_server.get_async_client()


async def send_request(
    client: openai.AsyncOpenAI, stream: bool, repeat: int, max_token: int, model_name: str
):
    for _ in range(repeat):
        prompt = "What is the result of 1+1? Answer in one word: "
        completion = await client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_token,
            temperature=0.0,
            stream=stream,
        )
        if stream:
            output = []
            async for chunk in completion:
                output.append(chunk.choices[0].text)
            assert len(output) > 0
            message = "".join(output)
        else:
            assert completion.id is not None
            message = completion.choices[0].text
        assert message.startswith("2.")


def check_historgram(metrics_dict: dict, count: int, range: tuple[float, float]):
    assert metrics_dict["count"] == count
    mean = metrics_dict["sum"] / metrics_dict["count"]
    assert mean > range[0] and mean < range[1]


@pytest.mark.timeout(300)
def test_return_perf_metrics_and_jsonl_dump(
    workers: Tuple[RemoteOpenAIServer, RemoteOpenAIServer],
    disagg_server: RemoteDisaggOpenAIServer,
    model_name: str,
    perf_metrics_output_dir,
):
    assert len(workers) == 2
    for worker in workers:
        worker.wait_for_server(timeout=120)
    wait_for_endpoint_ready(disagg_server.url_root + "/health")

    response = requests.post(
        f"{disagg_server.url_root}/v1/completions",
        headers={RETURN_METRICS_HEADER: "1"},
        json={
            "model": model_name,
            "prompt": "Reply with one token.",
            "max_tokens": 1,
            "temperature": 0.0,
        },
        timeout=120,
    )
    assert response.status_code == 200
    assert response.json()["id"] is not None
    assert response.headers.get(SERVER_TIMING_HEADER)
    assert response.headers.get(START_END_TIME_HEADER)
    assert response.headers.get(STEP_METRICS_HEADER)
    assert response.headers.get(CTX_CHUNK_METRICS_HEADER)

    timing_metrics = get_timing_metrics(perf_metrics_output_dir)
    validate_timing_metrics(timing_metrics, "test_return_perf_metrics_and_jsonl_dump")
    worker_metric_keys = {
        "ctx_request_id",
        "perf_metrics",
        "request_id",
        "time_breakdown_metrics",
    }
    perf_metric_keys = {
        "first_iter",
        "kv_cache_metrics",
        "last_iter",
        "timing_metrics",
    }
    for worker_metrics in (
        timing_metrics["ctx_perf_metrics"],
        timing_metrics["gen_perf_metrics"],
    ):
        assert set(worker_metrics) == worker_metric_keys
        assert set(worker_metrics["perf_metrics"]) == perf_metric_keys
        assert "kv_cache_hit_rate" not in worker_metrics["perf_metrics"]["kv_cache_metrics"]
    ctx_timing_metrics = timing_metrics["ctx_perf_metrics"]["perf_metrics"]["timing_metrics"]
    assert "kv_cache_transfer_start" not in ctx_timing_metrics
    assert "kv_cache_transfer_end" not in ctx_timing_metrics
    assert "kv_cache_size" not in ctx_timing_metrics
    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir, expected_count=3)
    disagg_record = next(record for record in records if "ctx_server" in record)
    disagg_request_id = disagg_record["disagg_request_id"]
    for record in records:
        assert record["status"] == "complete"
        assert record["disagg_request_id"] == disagg_request_id

    assert "ctx_perf_metrics" in disagg_record
    assert "gen_perf_metrics" in disagg_record
    assert (
        disagg_record["disagg_server_arrival_time"]
        <= disagg_record["disagg_server_first_token_time"]
    )

    payload = {
        "model": model_name,
        "prompt": "Reply with one token.",
        "max_tokens": 1,
        "temperature": 0.0,
    }
    response = requests.post(
        f"{disagg_server.url_root}/v1/completions",
        json=payload,
        timeout=120,
    )
    assert response.status_code == 200
    assert not response.headers.get(SERVER_TIMING_HEADER)
    assert not response.headers.get(START_END_TIME_HEADER)
    assert not response.headers.get(STEP_METRICS_HEADER)
    assert not response.headers.get(CTX_CHUNK_METRICS_HEADER)

    response = requests.post(
        f"{disagg_server.url_root}/v1/completions",
        json={**payload, "stream": True},
        timeout=120,
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert f"event: {SSE_METRICS_EVENT}" not in response.text


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_completion_metrics(
    async_client: openai.AsyncOpenAI,
    workers: Tuple[RemoteOpenAIServer, RemoteOpenAIServer],
    disagg_server: RemoteDisaggOpenAIServer,
    model_name: str,
):
    assert len(workers) == 2
    for worker in workers:
        worker.wait_for_server(timeout=120)
    wait_for_endpoint_ready(disagg_server.url_root + "/health")

    max_token = 10
    total_requests = 10
    await send_request(
        client=async_client,
        stream=True,
        repeat=total_requests,
        max_token=max_token,
        model_name=model_name,
    )
    metrics = get_prometheus_metrics(disagg_server.url_root)
    print(metrics)

    for role in ["ctx", "gen"]:
        assert metrics[f"{role}_total_requests"] == total_requests
        assert metrics[f"{role}_completed_requests"] == total_requests
        assert metrics[f"{role}_error_requests"] == 0
        assert f"{role}_retry_requests" in metrics

    check_historgram(
        metrics["gen_first_token_latency_seconds"],
        total_requests,
        (0.0, 0.3),
    )
    check_historgram(metrics["gen_complete_latency_seconds"], total_requests, (0.0, 0.6))

    assert metrics["total_requests"] == total_requests
    assert metrics["stream_requests"] == total_requests
    assert metrics["nonstream_requests"] == 0
    assert metrics["total_responses"] == total_requests
    assert metrics["validation_exceptions"] == 0
    assert metrics["http_exceptions"] == 0
    assert metrics["internal_errors"] == 0
    check_historgram(metrics["queue_latency_seconds"], total_requests, (0.0, 0.03))

    # test non streaming part
    await send_request(
        client=async_client,
        stream=False,
        repeat=total_requests,
        max_token=max_token,
        model_name=model_name,
    )

    metrics = get_prometheus_metrics(disagg_server.url_root)
    for role in ["ctx", "gen"]:
        assert metrics[f"{role}_total_requests"] == total_requests * 2
        assert metrics[f"{role}_completed_requests"] == total_requests * 2
        assert metrics[f"{role}_error_requests"] == 0
        assert f"{role}_retry_requests" in metrics

    assert metrics["total_requests"] == total_requests * 2
    assert metrics["stream_requests"] == total_requests
    assert metrics["nonstream_requests"] == total_requests
    assert metrics["total_responses"] == total_requests * 2
    assert metrics["validation_exceptions"] == 0
    assert metrics["http_exceptions"] == 0
    assert metrics["internal_errors"] == 0

    check_historgram(
        metrics["gen_complete_latency_seconds"],
        total_requests * 2,
        (0.0, 0.6),
    )
    check_historgram(metrics["queue_latency_seconds"], total_requests * 2, (0.0, 0.03))
