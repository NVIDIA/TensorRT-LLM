import os
import socket
import time

import openai
import pytest
import requests

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from .utils import expand_slurm_nodelist

RANK = int(os.environ.get("SLURM_PROCID", 0))
NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
NODE_LIST = expand_slurm_nodelist(os.environ.get("SLURM_NODELIST", ""))
SLURM_NTASKS_PER_NODE = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))

pytestmark = pytest.mark.threadleak(enabled=False)

# This test assumes that there are >2 nodes, we run ctx/disagg-server/client on the first node,
# and run gen the second node.

CTX_SERVER_PORT = 8001
GEN_SERVER_PORT = 8002
DISAGG_SERVER_PORT = 8000


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


def is_ctx_node():
    return NODE_RANK == 0


def is_gen_node():
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
        if not ('PMI_' in k or 'OMPI_' in k or 'PMIX_' in k or 'SLURM_' in k)
    }


@pytest.fixture(scope="module")
def model_name():
    return "llama-3.1-model/Llama-3.1-8B-Instruct"


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
    extra_config = {
        "cache_transceiver_config": {
            "backend": "UCX"
        },
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.5,
            "enable_block_reuse": False,
        },
        "disable_overlap_scheduler": True,
    }
    if is_ctx_node():
        print(f"starting ctx_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = ctx_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(model_path,
                                port=CTX_SERVER_PORT,
                                cli_args=args,
                                host="0.0.0.0",
                                env=env(),
                                llmapi_launch=False,
                                rank=RANK % SLURM_NTASKS_PER_NODE,
                                extra_config=extra_config) as server:
            yield server
    elif is_gen_node():
        print(f"starting gen_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = gen_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(model_path,
                                port=GEN_SERVER_PORT,
                                cli_args=args,
                                host="0.0.0.0",
                                env=env(),
                                llmapi_launch=False,
                                rank=RANK % SLURM_NTASKS_PER_NODE,
                                extra_config=extra_config) as server:
            yield server
    else:
        yield None


def wait_for_endpoint_ready(url: str, timeout: int = 300):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            time.sleep(1)
            if requests.get(url).status_code == 200:
                print(f"endpoint {url} is ready")
                return
        except Exception as err:
            print(f"endpoint {url} is not ready, with exception: {err}")


def wait_for_endpoint_down(url: str, timeout: int = 300):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if requests.get(url).status_code >= 100:
                print(
                    f"endpoint {url} returned status code {requests.get(url).status_code}"
                )
                time.sleep(1)
        except Exception as err:
            print(f"endpoint {url} is down, with exception: {err}")
            return


@pytest.fixture(scope="module")
def disagg_server(worker: RemoteOpenAIServer):
    if is_disagg_node():
        print(f"starting disagg_server for rank {RANK} node rank {NODE_RANK}")
        ctx_url = f"localhost:8001"  # Use localhost since the ctx server is on the same node
        # TODO: Hopefully the NODE_LIST is ordered by NODE_RANK, this test is only expected to run with 2 nodes now
        # We need to test with 4 nodes or more in the future, which should be easier with service discovery
        gen_url = f"{get_the_other_host(0)}:8002"
        with RemoteDisaggOpenAIServer(ctx_servers=[ctx_url],
                                      gen_servers=[gen_url],
                                      port=DISAGG_SERVER_PORT,
                                      llmapi_launch=False,
                                      env=env()) as server:
            yield server
    else:
        print(f"skipping disagg_server for rank {RANK} node rank {NODE_RANK}")
        yield None


@pytest.fixture(scope="module")
def client(disagg_server: RemoteDisaggOpenAIServer):
    if is_pytest_node():
        return disagg_server.get_client()
    else:
        print(f"skipping client for rank {RANK} node rank {NODE_RANK}")
        return None


def test_completion(client: openai.OpenAI,
                    disagg_server: RemoteDisaggOpenAIServer, model_name: str):
    if len(NODE_LIST) != 2:
        pytest.skip("This test is only expected to run with 2 nodes")
        return
    if is_pytest_node():
        print(f"running test_completion on rank {RANK} node rank {NODE_RANK}")
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
        disagg_server.terminate()

    elif is_gen_node():
        # keep gen workers alive until the test ends, again we hope the NODE_LIST is ordered by NODE_RANK
        url = f"http://{get_the_other_host(0)}:{DISAGG_SERVER_PORT}/health/"
        wait_for_endpoint_ready(url)
        wait_for_endpoint_down(url)
        assert True
    else:
        assert True
