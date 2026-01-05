import os
from typing import Tuple

import openai
import pytest
from test_common.http_utils import wait_for_endpoint_ready
from test_common.perf_metrics_utils import (
    get_prometheus_metrics,
    get_timing_metrics,
    validate_timing_metrics,
)

from tensorrt_llm._utils import get_free_ports
from tests.unittest.utils.llm_data import llm_models_root

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
        "return_perf_metrics": True,
        "perf_metrics_max_requests": 1000,
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
        "perf_metrics_max_requests": 1000,
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
    timing_metrics = get_timing_metrics(disagg_server.url_root)
    validate_timing_metrics(timing_metrics, "test_completion_metrics")

    metrics = get_prometheus_metrics(disagg_server.url_root)
    print(metrics)

    for role in ["ctx", "gen"]:
        assert metrics[f"{role}_total_requests"] == total_requests
        assert metrics[f"{role}_completed_requests"] == total_requests
        assert metrics[f"{role}_error_requests"] == 0
        assert f"{role}_retry_requests" in metrics

    check_historgram(metrics["gen_first_token_latency_seconds"], total_requests, (0.0, 0.3))
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

    check_historgram(metrics["gen_complete_latency_seconds"], total_requests * 2, (0.0, 0.6))
    check_historgram(metrics["queue_latency_seconds"], total_requests * 2, (0.0, 0.03))
