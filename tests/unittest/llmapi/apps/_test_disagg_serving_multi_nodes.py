# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-node disaggregated serving test run directly under Slurm.

Run one selected parametrization with one task per node, for example::

    srun --kill-on-bad-exit=1 --nodes=2 --ntasks-per-node=1 \
        python -m pytest <this module>::test_completion[ctx_tp2pp1-gen_tp2pp1]
"""

import os
import platform
import socket
import time

import openai
import pytest
from test_common import cluster_env
from test_common.http_utils import (wait_for_endpoint_down,
                                    wait_for_endpoint_ready)
from test_common.socket_utils import recv_exact

from tensorrt_llm._utils import get_sm_version

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from .utils import expand_slurm_nodelist

RANK = int(os.environ.get("SLURM_PROCID", 0))
NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
NODE_LIST = expand_slurm_nodelist(os.environ.get("SLURM_STEP_NODELIST", ""))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))

pytestmark = pytest.mark.threadleak(enabled=False)

# This test uses exactly two nodes: ctx/disagg-server/client on the first node
# and gen on the second node.
# This is a multi-node test, and will not be scheduled to the same node running other tests
# using fixed ports should be safe.
CTX_SERVER_PORT = 8001
GEN_SERVER_PORT = 8002
DISAGG_SERVER_PORT = 8000
# The disagg server starts once the local ctx server is ready, while the gen
# server on the other node starts independently and may need longer, so the
# default 180-second readiness window of the disagg server is too short.
DISAGG_SERVER_START_TIMEOUT_S = 1200
GEN_TASK_HANDSHAKE_PORT = 8003
# Worker startup can take minutes; bound coordination separately from model loading.
GEN_TASK_HANDSHAKE_TIMEOUT_S = 300
ROUTER_READY_TIMEOUT_S = (RemoteOpenAIServer.MAX_SERVER_START_WAIT_S +
                          DISAGG_SERVER_START_TIMEOUT_S)
# Completion work can span multiple client requests; keep it separate from shutdown.
COMPLETION_WORK_TIMEOUT_S = RemoteOpenAIServer.MAX_SERVER_START_WAIT_S
# Router shutdown should be quick once rank zero explicitly signals completion.
ROUTER_STOP_TIMEOUT_S = 300
READY_MESSAGE_PREFIX = "ready "
SHUTDOWN_MESSAGE = b"shutdown"


@pytest.fixture(scope="module", autouse=True)
def require_supported_slurm_topology() -> None:
    """Require the direct two-node Slurm topology used by this test."""
    required_variables = (
        "SLURM_PROCID",
        "SLURM_NODEID",
        "SLURM_LOCALID",
        "SLURM_STEP_NODELIST",
        "SLURM_STEP_NUM_NODES",
        "SLURM_STEP_NUM_TASKS",
    )
    missing_variables = [
        variable for variable in required_variables
        if variable not in os.environ
    ]
    if missing_variables:
        launcher_hint = (" trtllm-llmapi-launch removes these variables."
                         if "SLURM_PROCID" in missing_variables else "")
        pytest.fail("This test must run as a two-node srun step; "
                    f"missing: {', '.join(missing_variables)}.{launcher_hint}")
    if (os.environ["SLURM_STEP_NUM_NODES"] != "2"
            or os.environ["SLURM_STEP_NUM_TASKS"] != "2"
            or len(NODE_LIST) != 2):
        pytest.fail(
            "Launch one pytest task per node with srun, for example "
            "srun --kill-on-bad-exit=1 --nodes=2 --ntasks-per-node=1 "
            "python -m pytest ...; "
            f"got SLURM_STEP_NUM_NODES={os.environ['SLURM_STEP_NUM_NODES']}, "
            f"SLURM_STEP_NUM_TASKS={os.environ['SLURM_STEP_NUM_TASKS']}, "
            f"SLURM_STEP_NODELIST={NODE_LIST}")
    if LOCAL_RANK != 0 or RANK != NODE_RANK:
        pytest.fail(
            "This test supports exactly one pytest task per Slurm node; "
            f"got SLURM_PROCID={RANK}, SLURM_NODEID={NODE_RANK}, "
            f"SLURM_LOCALID={LOCAL_RANK}")


# Exclude the current node from the node list, then return other nodes by idx
def get_the_other_host(idx=0):
    assert len(NODE_LIST) >= 2
    node_list = NODE_LIST.copy()
    curr_host = socket.gethostname()
    if curr_host in NODE_LIST:
        # gethostname returns the exact node name in node list
        node_list.remove(curr_host)
    else:
        # gethostname returns the full domain
        curr_host = curr_host.split('.')[0]
        assert curr_host in node_list
        node_list.remove(curr_host)
    return node_list[idx]


def wait_for_gen_task_handshake(test_id: str) -> socket.socket:
    """Wait for readiness and return the live coordination connection."""
    expected_signal = f"{READY_MESSAGE_PREFIX}{test_id}".encode()
    deadline = time.monotonic() + GEN_TASK_HANDSHAKE_TIMEOUT_S
    empty_connections = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", GEN_TASK_HANDSHAKE_PORT))
        listener.listen()
        while (remaining := deadline - time.monotonic()) > 0:
            listener.settimeout(remaining)
            try:
                connection, peer = listener.accept()
            except TimeoutError:
                break
            connection.settimeout(max(deadline - time.monotonic(), 0.001))
            chunks = []
            while True:
                try:
                    chunk = connection.recv(4096)
                except TimeoutError as error:
                    connection.close()
                    raise RuntimeError(
                        "Generation task did not complete its handshake on port "
                        f"{GEN_TASK_HANDSHAKE_PORT} within "
                        f"{GEN_TASK_HANDSHAKE_TIMEOUT_S}s") from error
                if not chunk:
                    break
                chunks.append(chunk)
            ready_signal = b"".join(chunks)
            if not ready_signal:
                connection.close()
                empty_connections += 1
                print(f"Ignoring empty generation handshake from {peer}")
                continue
            if ready_signal != expected_signal:
                connection.close()
                raise RuntimeError(
                    "Generation task handshake is for a different test: "
                    f"expected {expected_signal!r}, got {ready_signal!r}")
            return connection
    raise RuntimeError(
        "Generation task did not complete a valid handshake on port "
        f"{GEN_TASK_HANDSHAKE_PORT} within {GEN_TASK_HANDSHAKE_TIMEOUT_S}s; "
        f"ignored {empty_connections} empty connection(s)")


def wait_for_gen_task_shutdown(host: str, test_id: str) -> None:
    """Announce readiness, then wait until rank zero finishes requests."""
    ready_signal = f"{READY_MESSAGE_PREFIX}{test_id}".encode()
    deadline = time.monotonic() + GEN_TASK_HANDSHAKE_TIMEOUT_S
    connection = None
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            connection = socket.create_connection(
                (host, GEN_TASK_HANDSHAKE_PORT), timeout=min(5, remaining))
            break
        except (ConnectionRefusedError, TimeoutError, OSError) as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Generation task could not coordinate with rank zero at "
                    f"{host}:{GEN_TASK_HANDSHAKE_PORT} within "
                    f"{GEN_TASK_HANDSHAKE_TIMEOUT_S}s") from error
            time.sleep(1)
    if connection is None:
        raise RuntimeError(
            "Generation task could not coordinate with rank zero at "
            f"{host}:{GEN_TASK_HANDSHAKE_PORT} within "
            f"{GEN_TASK_HANDSHAKE_TIMEOUT_S}s")
    with connection:
        connection.sendall(ready_signal)
        connection.shutdown(socket.SHUT_WR)
        try:
            shutdown_signal = recv_exact(connection, len(SHUTDOWN_MESSAGE),
                                         COMPLETION_WORK_TIMEOUT_S)
        except TimeoutError as error:
            raise RuntimeError(
                "Rank zero did not finish completion work within "
                f"{COMPLETION_WORK_TIMEOUT_S}s") from error
        except EOFError as error:
            shutdown_signal = error.args[0]
        if shutdown_signal != SHUTDOWN_MESSAGE:
            raise RuntimeError(
                "Rank zero closed the generation coordination channel "
                f"before signaling shutdown; got {shutdown_signal!r}")


def signal_gen_task_shutdown(connection: socket.socket) -> None:
    """Tell the generation task to start its bounded router-down wait."""
    connection.sendall(SHUTDOWN_MESSAGE)


def is_ctx_node() -> bool:
    return NODE_RANK == 0


def is_gen_node() -> bool:
    return NODE_RANK == 1


def is_rank_zero() -> bool:
    return RANK == 0


def env():
    # Remove MPI related environment variables to isolate the ctx/gen processes
    # so that they will not be in the same MPI communicator, otherwise the rank and world_size may mismatch
    e = {
        k: v
        for k, v in os.environ.items()
        if not ('PMI_' in k or 'OMPI_' in k or 'PMIX_' in k or 'SLURM_' in k)
        and k not in ["UCX_TLS", "UCX_NET_DEVICES"]  # avoid UCX failure on oci
    }
    # Some GB300 machines have NICs that misbehave with UCX's default transport
    # auto-selection, so UCX_TLS must be set explicitly there. Identify GB300 via
    # sm_103 (Blackwell Ultra) + aarch64 (Grace) -- this excludes HGX B300
    # (also sm_103 but x86_64) and avoids depending on the GPU device name string.
    # A cluster rule, if one matches, overrides this.
    if get_sm_version() == 103 and platform.machine().lower() == "aarch64":
        e["UCX_TLS"] = "cuda_copy,cuda_ipc,sm,self,tcp"
    cluster_env.apply_ucx_env(e,
                              cluster_name=os.environ.get("SLURM_CLUSTER_NAME"))
    return e


@pytest.fixture(scope="module")
def model_name():
    return "Qwen3.5-4B"


@pytest.fixture(scope="module", params=['pytorch'], ids=["pytorch"])
def backend(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[(1, 1), (2, 1), (1, 2)],
    ids=lambda tp_pp_size: f'ctx_tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
def ctx_tp_pp_size(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[(1, 1), (2, 1), (1, 2)],
    ids=lambda tp_pp_size: f'gen_tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
def gen_tp_pp_size(request):
    return request.param


@pytest.fixture(scope="module")
def worker(model_name: str, ctx_tp_pp_size: tuple, gen_tp_pp_size: tuple):
    """Start one context or generation worker on each Slurm node."""
    extra_config = {
        "cache_transceiver_config": {
            "backend": "DEFAULT"
        },
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
    if is_ctx_node():
        print(f"starting ctx_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = ctx_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(
                model_path,
                port=CTX_SERVER_PORT,
                cli_args=args,
                host="0.0.0.0",
                env=env(),
                llmapi_launch=False,
                # Poll /health; the default Slurm rank on node 1
                # would sleep for the full startup budget.
                rank=0,
                extra_config=extra_config) as server:
            yield server
    elif is_gen_node():
        print(f"starting gen_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = gen_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(
                model_path,
                port=GEN_SERVER_PORT,
                cli_args=args,
                host="0.0.0.0",
                env=env(),
                llmapi_launch=False,
                # Poll /health; the default Slurm rank on node 1
                # would sleep for the full startup budget.
                rank=0,
                extra_config=extra_config) as server:
            yield server
    else:
        raise AssertionError(f"Unexpected Slurm node rank: {NODE_RANK}")


@pytest.fixture(scope="module")
def disagg_server(worker: RemoteOpenAIServer):
    """Start the disaggregated router on global rank zero; other tasks yield None."""
    if is_rank_zero():
        print(f"starting disagg_server for rank {RANK} node rank {NODE_RANK}")
        ctx_url = f"localhost:{CTX_SERVER_PORT}"
        gen_url = f"{get_the_other_host(0)}:{GEN_SERVER_PORT}"
        with RemoteDisaggOpenAIServer(
                ctx_servers=[ctx_url],
                gen_servers=[gen_url],
                port=DISAGG_SERVER_PORT,
                llmapi_launch=False,
                env=env(),
                server_start_timeout=DISAGG_SERVER_START_TIMEOUT_S) as server:
            yield server
    else:
        print(f"skipping disagg_server for rank {RANK} node rank {NODE_RANK}")
        yield None


@pytest.fixture(scope="module")
def client(disagg_server: RemoteDisaggOpenAIServer):
    if is_rank_zero():
        return disagg_server.get_client()
    else:
        print(f"skipping client for rank {RANK} node rank {NODE_RANK}")
        return None


def test_completion(request: pytest.FixtureRequest, client: openai.OpenAI,
                    disagg_server: RemoteDisaggOpenAIServer,
                    worker: RemoteOpenAIServer, model_name: str) -> None:
    """Verify one completion while coordinating worker shutdown."""
    if is_rank_zero():
        print(f"running test_completion on rank {RANK} node rank {NODE_RANK}")
        with wait_for_gen_task_handshake(
                request.node.callspec.id) as gen_connection:
            prompt = "What is the result of 1+1? Answer in one word: "
            completion = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
            )
            print(f"Output: {completion.choices[0].text}")
            assert completion.id is not None
            message = completion.choices[0].text
            assert message.startswith('2.')
            signal_gen_task_shutdown(gen_connection)
            disagg_server.terminate()

    elif is_gen_node():
        # Keep the generation worker alive until the disaggregated server exits.
        router_host = get_the_other_host(0)
        url = f"http://{router_host}:{DISAGG_SERVER_PORT}/health/"
        wait_for_endpoint_ready(url,
                                timeout=ROUTER_READY_TIMEOUT_S,
                                server_proc=worker.proc)
        wait_for_gen_task_shutdown(router_host, request.node.callspec.id)
        wait_for_endpoint_down(url, timeout=ROUTER_STOP_TIMEOUT_S)
        exit_code = worker.proc.poll()
        assert exit_code is None, (
            f"generation worker exited with code {exit_code} before the router stopped"
        )
