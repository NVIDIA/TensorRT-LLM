# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-node service-discovery test run directly under Slurm.

Run one selected parametrization with one task per node, for example:
``srun --nodes=2 --ntasks-per-node=1 python -m pytest <this module>::test_completion[etcd]``.
Rank zero owns the router and etcd; each task owns its node's worker.
"""

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
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))

# This a multi-node QA test, use a fixed port instead of finding a free port
# so that all nodes can have the same disagg server config
DISAGG_SERVER_PORT = 8000
ETCD_CLIENT_PORT = 2379
DISAGG_SERVER_START_TIMEOUT_S = 1200
GEN_TASK_HANDSHAKE_PORT = 8003
GEN_TASK_HANDSHAKE_TIMEOUT_S = 300
ROUTER_READY_TIMEOUT_S = RemoteOpenAIServer.MAX_SERVER_START_WAIT_S + DISAGG_SERVER_START_TIMEOUT_S
ROUTER_STOP_TIMEOUT_S = RemoteOpenAIServer.MAX_SERVER_START_WAIT_S
READY_MESSAGE_PREFIX = "ready "


@pytest.fixture(autouse=True)
def require_supported_slurm_topology() -> None:
    """Require the direct two-node Slurm topology used by this test."""
    required_variables = (
        "SLURM_PROCID",
        "SLURM_NODEID",
        "SLURM_LOCALID",
        "SLURM_NODELIST",
        "SLURM_STEP_NUM_NODES",
    )
    missing_variables = [variable for variable in required_variables if variable not in os.environ]
    if missing_variables:
        launcher_hint = (
            " trtllm-llmapi-launch removes these variables."
            if "SLURM_PROCID" in missing_variables
            else ""
        )
        pytest.fail(
            "This test must run as a two-node srun step; "
            f"missing: {', '.join(missing_variables)}.{launcher_hint}"
        )
    if os.environ["SLURM_STEP_NUM_NODES"] != "2" or len(NODE_LIST) != 2:
        pytest.fail(
            "Launch one pytest task per node with srun, for example "
            "srun --nodes=2 --ntasks-per-node=1 python -m pytest ...; "
            f"got SLURM_STEP_NUM_NODES={os.environ['SLURM_STEP_NUM_NODES']}, "
            f"SLURM_NODELIST={NODE_LIST}"
        )
    if LOCAL_RANK != 0 or RANK != NODE_RANK:
        pytest.fail(
            "This test supports exactly one pytest task per Slurm node; "
            f"got SLURM_PROCID={RANK}, SLURM_NODEID={NODE_RANK}, "
            f"SLURM_LOCALID={LOCAL_RANK}"
        )


def wait_for_gen_task_handshake(test_id: str) -> None:
    """Wait until the generation task is ready for router shutdown."""
    expected_signal = f"{READY_MESSAGE_PREFIX}{test_id}".encode()
    deadline = time.monotonic() + GEN_TASK_HANDSHAKE_TIMEOUT_S
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", GEN_TASK_HANDSHAKE_PORT))
        listener.listen()
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            listener.settimeout(remaining)
            try:
                connection, _ = listener.accept()
            except TimeoutError as error:
                raise RuntimeError(
                    "Generation task did not connect to rank 0 on port "
                    f"{GEN_TASK_HANDSHAKE_PORT} within "
                    f"{GEN_TASK_HANDSHAKE_TIMEOUT_S}s"
                ) from error
            with connection:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                connection.settimeout(remaining)
                chunks = []
                while True:
                    try:
                        chunk = connection.recv(4096)
                    except TimeoutError as error:
                        raise RuntimeError(
                            "Generation task did not complete its handshake within "
                            f"{GEN_TASK_HANDSHAKE_TIMEOUT_S}s"
                        ) from error
                    if not chunk:
                        break
                    chunks.append(chunk)
                ready_signal = b"".join(chunks)
            if not ready_signal:
                continue
            if ready_signal != expected_signal:
                raise RuntimeError(
                    "Generation task handshake is for a different test: "
                    f"expected {expected_signal!r}, got {ready_signal!r}"
                )
            return
    raise RuntimeError(f"Generation task did not connect within {GEN_TASK_HANDSHAKE_TIMEOUT_S}s")


def send_gen_task_handshake(host: str, test_id: str) -> None:
    """Tell rank zero that the generation task is awaiting router shutdown."""
    deadline = time.monotonic() + GEN_TASK_HANDSHAKE_TIMEOUT_S
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, GEN_TASK_HANDSHAKE_PORT), timeout=5) as connection:
                connection.sendall(f"{READY_MESSAGE_PREFIX}{test_id}".encode())
                return
        except OSError as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(
        f"Could not reach rank 0 at {host}:{GEN_TASK_HANDSHAKE_PORT} within "
        f"{GEN_TASK_HANDSHAKE_TIMEOUT_S}s"
    ) from last_error


def is_ctx_node() -> bool:
    return NODE_RANK == 0


def is_gen_node() -> bool:
    return NODE_RANK == 1


def is_worker_task() -> bool:
    return LOCAL_RANK == 0


def is_rank_zero() -> bool:
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
    """Provide the requested service-discovery endpoint."""
    etcd_uri = f"etcd://{disagg_host}:{ETCD_CLIENT_PORT}"
    if request.param == "http":
        yield None, f"http://{disagg_host}:{DISAGG_SERVER_PORT}"
    elif not is_rank_zero():
        yield None, etcd_uri
    else:
        work_dir = tempfile.mkdtemp()
        data_dir = f"{work_dir}/disagg_test-etcd-{uuid.uuid4()}"
        etcd = subprocess.Popen(
            [
                "etcd",
                "--data-dir",
                data_dir,
                "--listen-client-urls",
                f"http://0.0.0.0:{ETCD_CLIENT_PORT}",
                "--advertise-client-urls",
                f"http://{disagg_host}:{ETCD_CLIENT_PORT}",
            ]
        )
        yield etcd, etcd_uri
        try:
            etcd.kill()
            etcd.wait(timeout=10)
            shutil.rmtree(data_dir)
        except Exception:
            pass


@pytest.fixture
def disagg_cluster_config(service_discovery: tuple):
    _, uri = service_discovery
    return {
        "cluster_uri": uri,
        "cluster_name": "",
    }


@pytest.fixture
def worker(model_name: str, disagg_cluster_config: dict):
    """Start one context or generation worker on each Slurm node."""
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
    """Start the service-discovery router on global rank zero; other tasks yield None."""
    if is_rank_zero():
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
    if is_rank_zero():
        return disagg_server.get_client()
    else:
        print(f"skipping client for rank {RANK} node rank {NODE_RANK}")
        return None


def test_completion(
    request,
    disagg_server: RemoteDisaggOpenAIServer,
    worker: RemoteOpenAIServer,
    client: openai.OpenAI,
    disagg_host: str,
    model_name: str,
):
    """Verify repeated completions while coordinating worker shutdown."""
    disagg_health_url = f"http://{disagg_host}:{DISAGG_SERVER_PORT}/health/"
    if is_worker_task():
        wait_for_endpoint_ready(
            disagg_health_url,
            timeout=ROUTER_READY_TIMEOUT_S,
            server_proc=disagg_server.proc if is_rank_zero() else worker.proc,
        )
    if is_rank_zero():
        print(f"running test_completion on rank {RANK} node rank {NODE_RANK}")
        wait_for_gen_task_handshake(request.node.callspec.id)
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

        disagg_server.terminate()

    elif is_gen_node() and is_worker_task():
        # Keep the generation worker alive until the disaggregated server exits.
        send_gen_task_handshake(disagg_host, request.node.callspec.id)
        wait_for_endpoint_down(disagg_health_url, timeout=ROUTER_STOP_TIMEOUT_S)
        assert worker.proc.poll() is None, "generation worker exited during the test"
