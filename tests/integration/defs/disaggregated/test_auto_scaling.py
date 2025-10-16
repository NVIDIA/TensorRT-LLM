import asyncio
import os
import subprocess
import tempfile

import openai
import pytest
import requests
import yaml

from tensorrt_llm.logger import logger

TEST_PORT = 18000
HEARTBEAT_INTERVAL = 1
INACTIVE_TIMEOUT = 2

ROUTER_TYPES = ["round_robin",
                "load_balancing"]  # kv_cache_aware doesn't support auto-scaling


@pytest.fixture
def model_name():
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture
def disagg_cluster_config():
    # same cluster config for workers and proxy server
    return {
        "cluster_uri": f"http://localhost:{TEST_PORT}",
        "cluster_name": "test_cluster",
        "heartbeat_interval_sec": HEARTBEAT_INTERVAL,
        "inactive_timeout_sec": INACTIVE_TIMEOUT,
    }


@pytest.fixture
def router(request):
    return request.param


@pytest.fixture
def disagg_server_config(disagg_cluster_config, router):
    return {
        "hostname": "localhost",
        "port": TEST_PORT,
        "disagg_cluster": disagg_cluster_config,
        "context_servers": {
            "router": {
                "type": router
            }
        },
        "generation_servers": {
            "router": {
                "type": router
            }
        },
    }


@pytest.fixture
def worker_config(disagg_cluster_config):
    return {
        "disagg_cluster": disagg_cluster_config,
        "disable_overlap_scheduler": True,
        "cache_transceiver_config": {
            "backend": "DEFAULT"
        },
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.2,
            "enable_partial_reuse": False,
        },
        "cuda_graph_config": {},
    }


def _run_worker(model_name, worker_config, role, port=8000, device=-1):
    worker_config_path = tempfile.NamedTemporaryFile(delete=False)
    with open(worker_config_path.name, "w+") as f:
        yaml.dump(worker_config, f)
        f.flush()
        cmd = [
            "trtllm-serve",
            "serve",
            model_name,
            "--host",
            "localhost",
            "--port",
            str(port),
            "--extra_llm_api_options",
            worker_config_path.name,
            "--server_role",
            "context" if role.startswith("ctx") else "generation",
        ]
        env = os.environ.copy()
        if device != -1:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        return subprocess.Popen(cmd, env=env)


def run_ctx_worker(model_name,
                   ctx_worker_config,
                   port=TEST_PORT + 100,
                   device=0):
    return _run_worker(model_name, ctx_worker_config, "ctx", port, device)


def run_gen_worker(model_name,
                   gen_worker_config,
                   port=TEST_PORT + 200,
                   device=1):
    return _run_worker(model_name, gen_worker_config, "gen", port, device)


def run_disagg_server(disagg_cluster_config, port=TEST_PORT):
    disagg_server_config_path = f"/tmp/disagg_server_{port}_config.yaml"
    disagg_cluster_config["port"] = port
    with open(disagg_server_config_path, "w+") as f:
        yaml.dump(disagg_cluster_config, f)
    cmds = ["trtllm-serve", "disaggregated", "-c", disagg_server_config_path]
    f = open("disagg_server.log", "w+")
    p = subprocess.Popen(cmds, stdout=f, stderr=f)
    return p


async def wait_for_disagg_server_ready(port):
    while True:
        await asyncio.sleep(3)
        logger.info(f"Waiting for disagg server to be ready")
        try:
            info_resp = requests.get(f"http://localhost:{port}/cluster_info")
            if info_resp.status_code == 200:
                info = info_resp.json()
                if info["is_ready"]:
                    break
                logger.info(
                    f"Waiting for disagg server to be ready: {info_resp.json()}"
                )
            else:
                logger.info(
                    f"Failed to get cluster info: {info_resp.status_code}")
            await asyncio.sleep(3)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get cluster info: {e}")


async def wait_for_worker_ready(port):
    while True:
        await asyncio.sleep(3)
        logger.info(f"Waiting for worker {port} to be ready")
        try:
            info_resp = requests.get(f"http://localhost:{port}/health")
            if info_resp.status_code == 200:
                break
        except requests.exceptions.RequestException as e:
            logger.info(f"Failed to get worker info: {e}")


def verify_cluster_info(ready,
                        ctx_workers=-1,
                        gen_workers=-1,
                        port=TEST_PORT,
                        expected_code=200):
    info_resp = requests.get(f"http://localhost:{port}/cluster_info")
    assert info_resp.status_code == expected_code
    info = info_resp.json()
    print("verify_cluster_info", info, ready, ctx_workers, gen_workers)
    assert info["is_ready"] == ready
    if ctx_workers != -1:
        assert len(info["current_workers"]["context_servers"]) == ctx_workers
    if gen_workers != -1:
        assert len(info["current_workers"]["generation_servers"]) == gen_workers


def terminate(*args):
    try:
        for arg in args:
            if arg and isinstance(arg, subprocess.Popen):
                arg.terminate()
                arg.wait(timeout=10)
    except Exception:
        pass


def request_completion(model_name, prompt, port=TEST_PORT):
    client = openai.OpenAI(api_key="tensorrt_llm",
                           base_url=f"http://localhost:{port}/v1")
    return client.completions.create(model=model_name,
                                     prompt=prompt,
                                     max_tokens=10,
                                     temperature=0.0)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("router", ROUTER_TYPES, indirect=True)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(600)
async def test_service_discovery(model_name, disagg_server_config,
                                 worker_config, router):
    ctx_worker1 = None
    gen_worker1 = None
    disagg_server = None
    try:
        # initial cluster, 1 ctx, 1 gen, request should succeed
        ctx_worker1 = run_ctx_worker(model_name, worker_config, TEST_PORT + 100)
        gen_worker1 = run_gen_worker(model_name, worker_config, TEST_PORT + 200)
        disagg_server = run_disagg_server(disagg_server_config, TEST_PORT)
        await wait_for_disagg_server_ready(TEST_PORT)
        verify_cluster_info(True, 1, 1)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=TEST_PORT)
        print(response)
    finally:
        terminate(ctx_worker1, gen_worker1, disagg_server)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize(
    "router", ["round_robin"], indirect=True
)  # use only round_robin to reduce the test time, this router type doesn't matter for this test
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(600)
async def test_minimal_instances(model_name, disagg_server_config,
                                 worker_config, router):
    # the cluster should have at least 2 ctx and 2 gen workers
    minimal_instances = {
        "context_servers": 2,
        "generation_servers": 2,
    }
    disagg_server_config["disagg_cluster"][
        "minimal_instances"] = minimal_instances
    worker_config["disagg_cluster"]["minimal_instances"] = minimal_instances

    processes = []

    try:
        processes.append(
            run_ctx_worker(model_name, worker_config, TEST_PORT + 100))
        processes.append(
            run_gen_worker(model_name, worker_config, TEST_PORT + 200))
        processes.append(run_disagg_server(disagg_server_config, TEST_PORT))
        await wait_for_worker_ready(TEST_PORT + 100)
        await wait_for_worker_ready(TEST_PORT + 200)
        verify_cluster_info(False, 1, 1)
        # with only 1 ctx and 1 gen worker, the request should fail
        with pytest.raises(Exception):
            response = request_completion(model_name,
                                          "Hello, my name is",
                                          port=TEST_PORT)
            print(response)

        processes.append(
            run_ctx_worker(model_name, worker_config, TEST_PORT + 101))
        processes.append(
            run_gen_worker(model_name, worker_config, TEST_PORT + 201))
        await wait_for_disagg_server_ready(TEST_PORT)
        verify_cluster_info(True, 2, 2)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=TEST_PORT)
        print(response)
    finally:
        terminate(*processes)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("router", ROUTER_TYPES, indirect=True)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(600)
async def test_worker_restart(model_name, disagg_server_config, worker_config,
                              router):
    ctx_worker1 = None
    ctx_worker2 = None
    gen_worker1 = None
    gen_worker2 = None
    disagg_server = None

    try:
        # initial cluster, 1 ctx, 1 gen, request should succeed
        ctx_worker1 = run_ctx_worker(model_name,
                                     worker_config,
                                     TEST_PORT + 100,
                                     device=0)
        gen_worker1 = run_gen_worker(model_name,
                                     worker_config,
                                     TEST_PORT + 200,
                                     device=1)
        disagg_server = run_disagg_server(disagg_server_config, TEST_PORT)
        await wait_for_disagg_server_ready(TEST_PORT)
        verify_cluster_info(True, 1, 1)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=TEST_PORT)
        print(response)
        # kill gen1, the request should fail
        terminate(gen_worker1)
        await asyncio.sleep(INACTIVE_TIMEOUT)
        verify_cluster_info(False, 1, 0)
        with pytest.raises(Exception):
            request_completion(model_name, "Hello, my name is", port=TEST_PORT)

        test_prompt = "The capital of France is"

        # add gen2, the request should succeed
        gen_worker2 = run_gen_worker(model_name,
                                     worker_config,
                                     TEST_PORT + 201,
                                     device=2)
        await wait_for_worker_ready(TEST_PORT + 201)
        await asyncio.sleep(INACTIVE_TIMEOUT)
        verify_cluster_info(True, 1, 1)

        response = request_completion(model_name, test_prompt, port=TEST_PORT)
        print(response)
        response_text = response.choices[0].text
        assert len(response.choices[0].text) >= 1

        # kill ctx1, the request should fail
        terminate(ctx_worker1)
        await asyncio.sleep(INACTIVE_TIMEOUT)
        verify_cluster_info(False, 0, 1)
        with pytest.raises(Exception):
            request_completion(model_name, test_prompt, port=TEST_PORT)

        # add ctx2, the request should succeed
        ctx_worker2 = run_ctx_worker(model_name,
                                     worker_config,
                                     TEST_PORT + 101,
                                     device=3)
        await wait_for_worker_ready(TEST_PORT + 101)
        verify_cluster_info(True, 1, 1)

        response = request_completion(model_name, test_prompt, port=TEST_PORT)
        response_text = response.choices[0].text
        assert len(response.choices[0].text) >= 1

        # restart ctx1 and gen1 with the same ports, we have 2 ctxs and 2 gens now
        ctx_worker1 = run_ctx_worker(model_name, worker_config, TEST_PORT + 100)
        gen_worker1 = run_gen_worker(model_name, worker_config, TEST_PORT + 200)
        await wait_for_worker_ready(TEST_PORT + 100)
        await wait_for_worker_ready(TEST_PORT + 200)
        await asyncio.sleep(INACTIVE_TIMEOUT)
        verify_cluster_info(True, 2, 2)

        # send 10 requests, the responses will be generated by the different ctx/gen workers (but we can't verify it now)
        for _ in range(10):
            response = request_completion(model_name,
                                          test_prompt,
                                          port=TEST_PORT)
            assert response.choices[0].text == response_text
        print(response)
    finally:
        terminate(ctx_worker1, ctx_worker2, gen_worker1, gen_worker2,
                  disagg_server)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("router", ["round_robin"], indirect=True)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(300)
async def test_disagg_server_restart(model_name, disagg_server_config,
                                     worker_config, router):
    ctx_worker1 = None
    gen_worker1 = None
    disagg_server = None
    try:
        # initial cluster, 1 ctx, 1 gen, request should succeed
        ctx_worker1 = run_ctx_worker(model_name, worker_config, TEST_PORT + 100)
        gen_worker1 = run_gen_worker(model_name, worker_config, TEST_PORT + 200)
        disagg_server = run_disagg_server(disagg_server_config, TEST_PORT)
        await wait_for_disagg_server_ready(TEST_PORT)
        verify_cluster_info(True, 1, 1)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=TEST_PORT)
        print(response)
        response_text = response.choices[0].text

        # kill disagg server, the request should fail
        terminate(disagg_server)
        await asyncio.sleep(INACTIVE_TIMEOUT)
        with pytest.raises(Exception):
            verify_cluster_info(False, 1, 1, expected_code=500)

        # restart disagg server, the request should succeed
        disagg_server = run_disagg_server(disagg_server_config, TEST_PORT)
        await wait_for_disagg_server_ready(TEST_PORT)
        verify_cluster_info(True, 1, 1)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=TEST_PORT)
        print(response)
        assert response.choices[0].text == response_text

    finally:
        terminate(disagg_server, ctx_worker1, gen_worker1)
