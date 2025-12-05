import os
from typing import Tuple

import openai
import pytest
import requests

from tensorrt_llm._utils import get_free_ports

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from .utils import wait_for_endpoint_ready


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
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture
def disagg_cluster_config(disagg_port: int):
    return {
        "cluster_uri": f"http://localhost:{disagg_port}",
        "cluster_name": "",
    }


def worker_config(model_name: str, disagg_cluster_config: dict):
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
    }


@pytest.fixture
def workers(model_name: str, disagg_cluster_config: dict, ctx_port: int, gen_port: int):
    model_path = get_model_path(model_name)
    extra_config = worker_config(model_name, disagg_cluster_config)

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
def disagg_server(disagg_cluster_config: dict, workers, disagg_port: int):
    disagg_config = {
        "port": disagg_port,
        "disagg_cluster": disagg_cluster_config,
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


def get_metrics(disagg_server: RemoteDisaggOpenAIServer):
    response = requests.get(disagg_server.url_root + "/prometheus/metrics")
    assert response.status_code == 200
    # Parse Prometheus metrics lines into a dictionary of {metric_name: value}
    metrics = {}
    print(response.text)
    for line in response.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric = parts[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        import re

        if bucket_match := re.match(r'(.+)_bucket\{le="([^"]+)"\}', metric):
            # Try to parse bucket boundaries out of metrics like ..._bucket{le="0.005"}
            base_metric, le_value = bucket_match.groups()
            if base_metric not in metrics:
                metrics[base_metric] = {}
            try:
                metrics[base_metric][float(le_value)] = value
            except ValueError:
                continue
        elif sum_match := re.match(r"(.+)_sum$", metric):
            base_metric = sum_match.groups()[0]
            if base_metric not in metrics:
                metrics[base_metric] = {}
            metrics[base_metric]["sum"] = value
        elif count_match := re.match(r"(.+)_count$", metric):
            base_metric = count_match.groups()[0]
            if base_metric not in metrics:
                metrics[base_metric] = {}
            metrics[base_metric]["count"] = value
        elif total_match := re.match(r"(.+)_total$", metric):
            base_metric = total_match.groups()[0]
            print(f"Total metric {metric}: {base_metric} = {value}")
            metrics[base_metric] = value
        else:
            # ignore prometheus built-in metrics
            pass
    return metrics


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
    metrics = get_metrics(disagg_server)
    print(metrics)

    for role in ["ctx", "gen"]:
        assert metrics[f"{role}_total_requests"] == total_requests
        assert metrics[f"{role}_completed_requests"] == total_requests
        assert metrics[f"{role}_error_requests"] == 0
        assert f"{role}_retry_requests" in metrics

    check_historgram(metrics["gen_first_token_latency_seconds"], total_requests, (0.0, 0.1))
    check_historgram(metrics["gen_complete_latency_seconds"], total_requests, (0.0, 0.5))

    assert metrics["total_requests"] == total_requests
    assert metrics["stream_requests"] == total_requests
    assert metrics["nonstream_requests"] == 0
    assert metrics["total_responses"] == total_requests
    assert metrics["validation_exceptions"] == 0
    assert metrics["http_exceptions"] == 0
    assert metrics["internal_errors"] == 0
    check_historgram(metrics["queue_latency_seconds"], total_requests, (0.0, 0.01))

    # test non streaming part
    await send_request(
        client=async_client,
        stream=False,
        repeat=total_requests,
        max_token=max_token,
        model_name=model_name,
    )

    metrics = get_metrics(disagg_server)
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

    check_historgram(metrics["gen_complete_latency_seconds"], total_requests * 2, (0.0, 0.5))
    check_historgram(metrics["queue_latency_seconds"], total_requests * 2, (0.0, 0.01))
