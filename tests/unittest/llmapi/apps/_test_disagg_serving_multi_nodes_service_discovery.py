import os
import shutil
import subprocess
import tempfile
import uuid

import openai
import pytest
from test_common.perf_metrics_utils import get_timing_metrics, validate_timing_metrics

from tensorrt_llm._utils import get_free_port
from tensorrt_llm.llmapi.disagg_utils import ServerRole

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from .utils import expand_slurm_nodelist, wait_for_endpoint_down, wait_for_endpoint_ready

RANK = int(os.environ.get("SLURM_PROCID", 0))
NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
NODE_LIST = expand_slurm_nodelist(os.environ.get("SLURM_NODELIST", ""))
SLURM_NTASKS_PER_NODE = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))

# This a multi-node QA test, use a fixed port instead of finding a free port
# so that all nodes can have the same disagg server config
DISAGG_SERVER_PORT = 8000


# This test is supposed to run with 2 nodes or more
def is_ctx_node():
    assert len(NODE_LIST) == 2
    return NODE_RANK == 0


def is_gen_node():
    assert len(NODE_LIST) == 2
    return NODE_RANK == 1


def is_disagg_node():
    return NODE_RANK == 0


# The test is run on multinodes but only the first node's output is used for assertion
def is_pytest_node():
    return NODE_RANK == 0


def env():
    # Remove MPI related environment variables to isolate the ctx/gen processes
    # so that they will not be in the same MPI communicator, otherwise the rank and world_size may mismatch
    return {
        k: v
        for k, v in os.environ.items()
        if not ("PMI_" in k or "OMPI_" in k or "PMIX_" in k or "SLURM_" in k)
        and k not in ["UCX_TLS", "UCX_NET_DEVICES"]
    }


@pytest.fixture
def model_name():
    return "llama-3.1-model/Llama-3.1-8B-Instruct"


@pytest.fixture
def disagg_host():
    return NODE_LIST[0]


@pytest.fixture(params=["etcd", "http"])
def service_discovery(request, disagg_host: str):
    if request.param == "etcd":
        work_dir = tempfile.mkdtemp()
        data_dir = f"{work_dir}/disagg_test-etcd-{uuid.uuid4()}"
        etcd = subprocess.Popen(["etcd", "--data-dir", data_dir])
        yield etcd, f"etcd://{disagg_host}:2379"
        try:
            etcd.kill()
            etcd.wait(timeout=10)
            shutil.rmtree(data_dir)
        except Exception:
            pass
    else:
        yield None, f"http://{disagg_host}:{DISAGG_SERVER_PORT}"


@pytest.fixture
def disagg_cluster_config(service_discovery: tuple):
    _, uri = service_discovery
    return {
        "cluster_uri": uri,
        "cluster_name": "",
    }


@pytest.fixture
def worker(model_name: str, disagg_cluster_config: dict):
    extra_config = {
        "disagg_cluster": disagg_cluster_config,
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False,
        },
        "disable_overlap_scheduler": True,
        "return_perf_metrics": True,
        "perf_metrics_max_requests": 1000,
    }
    # start workers on 0.0.0.0:<free_port>, then the workers should be able to
    # report their correct hostname:port to the disagg server
    port = get_free_port()
    if is_ctx_node():
        print(f"starting ctx_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = 1, 1
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(
            model_path,
            port=port,
            cli_args=args,
            host="0.0.0.0",
            env=env(),
            llmapi_launch=False,
            rank=RANK % SLURM_NTASKS_PER_NODE,
            extra_config=extra_config,
            role=ServerRole.CONTEXT,
        ) as server:
            yield server
    elif is_gen_node():
        print(f"starting gen_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = 1, 1
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(
            model_path,
            port=port,
            cli_args=args,
            host="0.0.0.0",
            env=env(),
            llmapi_launch=False,
            rank=RANK % SLURM_NTASKS_PER_NODE,
            extra_config=extra_config,
            role=ServerRole.GENERATION,
        ) as server:
            yield server
    else:
        yield None


# different from non-service-discovery version, disagg server doesn't have to
# wait for ctx/gen servers to get ready
@pytest.fixture
def disagg_server(disagg_cluster_config: dict):
    if is_disagg_node():
        disagg_config = {
            "disagg_cluster": disagg_cluster_config,
            "port": DISAGG_SERVER_PORT,
            "hostname": "0.0.0.0",
            "perf_metrics_max_requests": 1000,
        }
        print(f"starting disagg_server for rank {RANK} node rank {NODE_RANK}")
        # ctx/gen servers are unnecessary for service discovery test
        with RemoteDisaggOpenAIServer(
            ctx_servers=[],
            gen_servers=[],
            port=DISAGG_SERVER_PORT,
            disagg_config=disagg_config,
            llmapi_launch=False,
            env=env(),
            wait_ready=False,  # wait it to be ready in test body
        ) as server:
            yield server
    else:
        print(f"skipping disagg_server for rank {RANK} node rank {NODE_RANK}")
        yield None


@pytest.fixture
def client(disagg_server: RemoteDisaggOpenAIServer):
    if is_pytest_node():
        return disagg_server.get_client()
    else:
        print(f"skipping client for rank {RANK} node rank {NODE_RANK}")
        return None


def test_completion(
    disagg_server: RemoteDisaggOpenAIServer,
    worker: RemoteOpenAIServer,
    client: openai.OpenAI,
    disagg_host: str,
    model_name: str,
):
    disagg_health_url = f"http://{disagg_host}:{DISAGG_SERVER_PORT}/health/"
    wait_for_endpoint_ready(disagg_health_url)
    if is_pytest_node():
        print(f"running test_completion on rank {RANK} node rank {NODE_RANK}")
        prompt = "What is the result of 1+1? Answer in one word: "
        for _ in range(10):
            completion = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
            )
            print(f"Output: {completion.choices[0].text}")
            assert completion.id is not None
            message = completion.choices[0].text
            assert message.startswith("2.")

        perf_metrics = get_timing_metrics(disagg_server.url_root)
        validate_timing_metrics(perf_metrics, "multinode test_completion")

        disagg_server.terminate()

    elif is_gen_node():
        # keep gen workers alive until the test ends
        wait_for_endpoint_down(disagg_health_url)
        assert True
    else:
        assert True
