# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
import uuid

import openai
import pytest
from test_common import cluster_env
from test_common.http_utils import wait_for_endpoint_down, wait_for_endpoint_ready

from tensorrt_llm._utils import get_free_port, get_sm_version
from tensorrt_llm.llmapi.disagg_utils import ServerRole

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from .utils import expand_slurm_nodelist

RANK = int(os.environ.get("SLURM_PROCID", 0))
NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
NODE_LIST = expand_slurm_nodelist(os.environ.get("SLURM_NODELIST", ""))
SLURM_NTASKS_PER_NODE = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", RANK % SLURM_NTASKS_PER_NODE))

# This a multi-node QA test, use a fixed port instead of finding a free port
# so that all nodes can have the same disagg server config
DISAGG_SERVER_PORT = 8000
DISAGG_SERVER_START_TIMEOUT_S = 1200
WORKER_READY_PORT = 8003
WORKER_READY_TIMEOUT_S = 300


@pytest.fixture(autouse=True)
def require_supported_slurm_topology():
    required_variables = (
        "SLURM_PROCID",
        "SLURM_NODEID",
        "SLURM_LOCALID",
        "SLURM_NODELIST",
        "SLURM_NTASKS_PER_NODE",
    )
    missing_variables = [variable for variable in required_variables if variable not in os.environ]
    if missing_variables:
        pytest.fail(
            "This test must run directly under a two-node Slurm allocation; "
            "trtllm-llmapi-launch removes the required topology variables. "
            f"Missing: {', '.join(missing_variables)}"
        )
    if len(NODE_LIST) != 2:
        pytest.skip("This test requires exactly two Slurm nodes")


# This test uses exactly two nodes.
def wait_for_worker_ready():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", WORKER_READY_PORT))
        listener.listen(1)
        listener.settimeout(WORKER_READY_TIMEOUT_S)
        connection, _ = listener.accept()
        with connection:
            if connection.recv(5) != b"ready":
                raise RuntimeError("Generation worker sent an invalid ready signal")


def signal_worker_ready(host: str):
    deadline = time.monotonic() + WORKER_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, WORKER_READY_PORT), timeout=5) as connection:
                connection.sendall(b"ready")
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Generation worker could not signal readiness")


def is_ctx_node():
    return NODE_RANK == 0


def is_gen_node():
    return NODE_RANK == 1


def is_worker_task():
    return LOCAL_RANK == 0


def is_disagg_node():
    return RANK == 0


# The test is run on multinodes but only the first node's output is used for assertion
def is_pytest_node():
    return RANK == 0


def env():
    # Remove MPI related environment variables to isolate the ctx/gen processes
    # so that they will not be in the same MPI communicator, otherwise the rank and world_size may mismatch
    e = {
        k: v
        for k, v in os.environ.items()
        if not ("PMI_" in k or "OMPI_" in k or "PMIX_" in k or "SLURM_" in k)
        and k not in ["UCX_TLS", "UCX_NET_DEVICES"]
    }
    # GB300 (sm_103 + aarch64) needs UCX_TLS set explicitly. A cluster rule, if
    # one matches, overrides this.
    if get_sm_version() == 103 and platform.machine().lower() == "aarch64":
        e["UCX_TLS"] = "cuda_copy,cuda_ipc,sm,self,tcp"
    cluster_env.apply_ucx_env(e, cluster_name=os.environ.get("SLURM_CLUSTER_NAME"))
    return e


@pytest.fixture
def model_name():
    return "Qwen3.5-4B"


@pytest.fixture
def disagg_host():
    return NODE_LIST[0]


@pytest.fixture(params=["etcd", "http"])
def service_discovery(request, disagg_host: str):
    if request.param == "etcd" and is_disagg_node():
        work_dir = tempfile.mkdtemp()
        data_dir = f"{work_dir}/disagg_test-etcd-{uuid.uuid4()}"
        etcd = subprocess.Popen(
            [
                "etcd",
                "--data-dir",
                data_dir,
                "--listen-client-urls",
                "http://0.0.0.0:2379",
                "--advertise-client-urls",
                f"http://{disagg_host}:2379",
            ]
        )
        yield etcd, f"etcd://{disagg_host}:2379"
        try:
            etcd.kill()
            etcd.wait(timeout=10)
            shutil.rmtree(data_dir)
        except Exception:
            pass
    elif request.param == "http":
        yield None, f"http://{disagg_host}:{DISAGG_SERVER_PORT}"
    else:
        yield None, f"etcd://{disagg_host}:2379"


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
            "free_gpu_memory_fraction": 0.8,
            "enable_block_reuse": False,
        },
        "disable_overlap_scheduler": True,
        # Qwen3.5 reserves recurrent state for max_batch_size * pp_size
        # sequences up front; the defaults do not fit on 80 GB GPUs.
        "max_batch_size": 128,
        "max_seq_len": 8192,
    }
    # start workers on 0.0.0.0:<free_port>, then the workers should be able to
    # report their correct hostname:port to the disagg server
    port = get_free_port()
    if is_ctx_node() and is_worker_task():
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
            rank=0,
            extra_config=extra_config,
            role=ServerRole.CONTEXT,
        ) as server:
            yield server
    elif is_gen_node() and is_worker_task():
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
            rank=0,
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
    if is_worker_task():
        wait_for_endpoint_ready(disagg_health_url, timeout=DISAGG_SERVER_START_TIMEOUT_S)
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

        wait_for_worker_ready()
        disagg_server.terminate()

    elif is_gen_node() and is_worker_task():
        # Keep the generation worker alive until the test ends.
        signal_worker_ready(disagg_host)
        wait_for_endpoint_down(disagg_health_url)
        assert True
    else:
        assert True
