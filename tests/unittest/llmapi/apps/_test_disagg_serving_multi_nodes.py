import os
import socket
import subprocess
import time

import openai
import pytest
import requests

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer
from .utils import expand_slurm_nodelist, get_local_interfaces, get_local_ip

RANK = int(os.environ.get("SLURM_PROCID", 0))
NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
NODE_LIST = expand_slurm_nodelist(os.environ.get("SLURM_NODELIST", ""))
SLURM_NTASKS_PER_NODE = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))

pytestmark = pytest.mark.threadleak(enabled=False)

# This test assumes that there are 2 nodes, one for ctx and one for gen.
# So we can use the node rank to determine which node the ctx/gen server is on.
# Then run the disagg-server and test_chat on the ctx node.

CTX_SERVER_PORT = 8001
GEN_SERVER_PORT = 8002
DISAGG_SERVER_PORT = 8000


# We run this test on two nodes, so by removing the current node from the node list, we can get the other node
def get_the_other_host():
    assert len(NODE_LIST) == 2
    assert socket.gethostname() in NODE_LIST
    node_list = NODE_LIST.copy()
    node_list.remove(socket.gethostname())
    return node_list[0]


def find_nic():
    test_ip = socket.gethostbyname(get_the_other_host())
    print(f"test_ip: {test_ip} for the other host {get_the_other_host()}")
    try:
        # iproute2 may not be installed
        result = subprocess.check_output(
            f"ip route get {test_ip} | sed -E 's/.*?dev (\\S+) .*/\\1/;t;d'")
        nic_name = result.decode('utf-8').strip()
        print(f"get NIC name from ip route, result: {nic_name}")
        return nic_name
    except Exception as e:
        print(f"Failed to find NIC from ip route: {e}")
        try:
            # Establish a socket to the test ip, then get the local ip from the socket,
            # enumerate the local interfaces and find the one with the local ip
            local_ip = get_local_ip(test_ip)
            for nic_name, ip in get_local_interfaces().items():
                if ip == local_ip:
                    return nic_name
        except OSError as e:
            print(f"Failed to find NIC from local interfaces: {e}")
        return None


def env():
    # Remove MPI related environment variables to isolate the ctx/gen processes
    # so that they will not be in the same MPI communicator, otherwise the rank and world_size may mismatch
    new_env = {
        k: v
        for k, v in os.environ.items()
        if not ('PMI_' in k or 'OMPI_' in k or 'PMIX_' in k or 'SLURM_' in k)
    }
    nic = find_nic()
    if nic:
        # TODO: integrate this into disagg-serving
        # setting TRTLLM_UCX_INTERFACE manually if possible because the interfaces found automatically by TRTLLM can have the same ip across nodes, then cache transceiver may fail to send/receive kv cache
        new_env["TRTLLM_UCX_INTERFACE"] = nic
    return new_env


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
    host = socket.gethostname()
    assert len(NODE_LIST) == 2
    assert host in NODE_LIST
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
    if NODE_RANK == 0:
        print(f"starting ctx_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = ctx_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(model_path,
                                port=CTX_SERVER_PORT,
                                cli_args=args,
                                host="0.0.0.0",
                                env=env(),
                                llmapi_launch=True,
                                rank=RANK % SLURM_NTASKS_PER_NODE,
                                extra_config=extra_config) as server:
            yield server
    elif NODE_RANK == 1:
        print(f"starting gen_server for rank {RANK} node rank {NODE_RANK}")
        model_path = get_model_path(model_name)
        tp_size, pp_size = gen_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(model_path,
                                port=GEN_SERVER_PORT,
                                cli_args=args,
                                host=host,
                                env=env(),
                                rank=RANK % SLURM_NTASKS_PER_NODE,
                                extra_config=extra_config) as server:
            yield server


@pytest.fixture(scope="module")
def disagg_server(worker: RemoteOpenAIServer):
    if RANK == 0:
        print(f"starting disagg_server for rank {RANK} node rank {NODE_RANK}")
        ctx_url = f"localhost:8001"  # Use localhost since the ctx server is on the same node
        gen_url = f"{get_the_other_host()}:8002"
        print(f"ctx_url: {ctx_url} gen_url: {gen_url}")
        with RemoteDisaggOpenAIServer(ctx_servers=[ctx_url],
                                      gen_servers=[gen_url],
                                      port=DISAGG_SERVER_PORT,
                                      llmapi_launch=True,
                                      env=env()) as server:
            yield server
    else:
        print(f"skipping disagg_server for rank {RANK} node rank {NODE_RANK}")
        yield None


@pytest.fixture(scope="module")
def client(disagg_server: RemoteDisaggOpenAIServer):
    if RANK == 0:
        return disagg_server.get_client()
    else:
        print(f"skipping client for rank {RANK} node rank {NODE_RANK}")
        return None


def wait_for_endpoint_ready(url: str, timeout: int = 300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            time.sleep(1)
            if requests.get(url).status_code == 200:
                print(f"endpoint {url} is ready")
                return
        except Exception:
            pass


def wait_for_endpoint_down(url: str, timeout: int = 300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url).status_code >= 100:
                time.sleep(1)
        except Exception as err:
            print(f"endpoint {url} is down, with exception: {err}")
            return


def test_completion(client: openai.OpenAI, model_name: str):
    if RANK == 0:
        print(f"running test_completion on rank {RANK} node rank {NODE_RANK}")
        prompt = "What is the result of 1+1? Answer in one word: "
        completion = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
        )
        print(f"Completion: {completion}")
        print(f"Output: {completion.choices[0].text}")
        assert completion.id is not None
        message = completion.choices[0].text
        assert message.startswith('2.')
    else:
        url = f"http://{get_the_other_host()}:{DISAGG_SERVER_PORT}/health/"
        # keep gen workers alive until the test ends
        wait_for_endpoint_ready(url)
        wait_for_endpoint_down(url)
        assert True
