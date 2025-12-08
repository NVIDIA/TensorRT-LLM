import asyncio
import os
import shutil
import subprocess
import tempfile
import traceback
import uuid

import openai
import pytest
import requests
import yaml

from tensorrt_llm._utils import get_free_port
from tensorrt_llm.logger import logger

HEARTBEAT_INTERVAL = 1
INACTIVE_TIMEOUT = 2
# check cluster status with a larger interval than inactive timeout to avoid flaky tests
CHECK_STATUS_INTERVAL = 3

ROUTER_TYPES = ["round_robin", "load_balancing", "kv_cache_aware"]
USED_PORTS = set()


# get_free_port doesn't guarantee that consecutive calls will return different ports
# if no server is bound to the port immediately after the call
def get_free_unused_port():
    global USED_PORTS
    max_attempts = 100
    for _ in range(max_attempts):
        port = get_free_port()
        if port not in USED_PORTS:
            USED_PORTS.add(port)
            return port
        else:
            logger.info(f"Port {port} is already used, trying another one")
    raise Exception(
        f"Failed to find a free unused port after {max_attempts} attempts")


@pytest.fixture
def model_name():
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture
def disagg_port():
    return get_free_unused_port()


@pytest.fixture
def work_dir():
    return tempfile.mkdtemp()


@pytest.fixture
def service_discovery(request, disagg_port, work_dir):
    if request.param == "etcd":
        data_dir = f"{work_dir}/disagg_test-etcd-{uuid.uuid4()}"
        etcd = subprocess.Popen(["etcd", "--data-dir", data_dir])
        yield etcd, f"etcd://localhost:2379"
        try:
            etcd.kill()
            etcd.wait(timeout=10)
            shutil.rmtree(data_dir)
        except Exception:
            pass
    else:
        yield None, f"http://localhost:{disagg_port}"


@pytest.fixture
def disagg_cluster_config(service_discovery):
    # same cluster config for workers and proxy server
    _, uri = service_discovery
    return {
        "cluster_uri": uri,
        "cluster_name": "test_cluster",
        "heartbeat_interval_sec": HEARTBEAT_INTERVAL,
        "inactive_timeout_sec": INACTIVE_TIMEOUT,
    }


@pytest.fixture
def router(request):
    return request.param


@pytest.fixture
def disagg_server_config(disagg_cluster_config, router, disagg_port):
    return {
        "hostname": "localhost",
        "port": disagg_port,
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


class ProcessWrapper:

    def __init__(self, process, log_file=None, log_path=None, port=0):
        self.process = process
        self.log_file = log_file
        self.log_path = log_path
        self.port = port


def _run_worker(model_name, worker_config, role, port, work_dir, device=-1):
    if port == 0:
        port = get_free_unused_port()
    worker_config_path = os.path.join(work_dir, f"{role}_{port}_config.yaml")
    with open(worker_config_path, "w+") as f:
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
            worker_config_path,
            "--server_role",
            "context" if role.startswith("ctx") else "generation",
        ]
        env = os.environ.copy()
        if device != -1:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        log_path = os.path.join(work_dir, f"output_{role}.log")
        log_file = open(log_path, "w+")
        print(f"Running {role} on port {port}")
        return ProcessWrapper(subprocess.Popen(cmd,
                                               env=env,
                                               stdout=log_file,
                                               stderr=log_file),
                              log_file=log_file,
                              log_path=log_path,
                              port=port)


def run_ctx_worker(model_name, ctx_worker_config, work_dir, port=0, device=0):
    return _run_worker(model_name, ctx_worker_config, "ctx", port, work_dir,
                       device)


def run_gen_worker(model_name, gen_worker_config, work_dir, port=0, device=1):
    return _run_worker(model_name, gen_worker_config, "gen", port, work_dir,
                       device)


def run_disagg_server(disagg_cluster_config, work_dir, port=0):
    disagg_server_config_path = os.path.join(work_dir,
                                             "disagg_server_config.yaml")
    disagg_cluster_config["port"] = port
    with open(disagg_server_config_path, "w+") as f:
        yaml.dump(disagg_cluster_config, f)
    cmds = ["trtllm-serve", "disaggregated", "-c", disagg_server_config_path]
    log_path = os.path.join(work_dir, "disagg_server.log")
    log_file = open(log_path, "w+")
    p = subprocess.Popen(cmds, stdout=log_file, stderr=log_file)
    return ProcessWrapper(p, log_file=log_file, log_path=log_path, port=port)


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
                        port=0,
                        expected_code=200):
    assert port > 0, "port must be positive"
    info_resp = requests.get(f"http://localhost:{port}/cluster_info")
    assert info_resp.status_code == expected_code
    info = info_resp.json()
    print("verify_cluster_info", info, ready, ctx_workers, gen_workers)
    assert info["is_ready"] == ready
    if ctx_workers != -1:
        assert len(info["current_workers"]["context_servers"]) == ctx_workers
    if gen_workers != -1:
        assert len(info["current_workers"]["generation_servers"]) == gen_workers


def tail(f, n):
    try:
        proc = subprocess.Popen(['tail', '-n', str(n), f],
                                stdout=subprocess.PIPE)
        return proc.stdout.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to tail {f}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return ""


def terminate(*args, show_log_lines=30, release_port=True):
    for arg in args:
        if arg and isinstance(arg, ProcessWrapper):
            try:
                # tail the log file for better debugging on CI
                if os.path.exists(arg.log_path):
                    print(f"-------------{arg.log_path}---------------")
                    print(tail(arg.log_path, show_log_lines))
            except Exception as e:
                print(f"Failed to tail {arg.log_path}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
            if arg.process:
                print(f"Killing process {arg.process.pid}")
                try:
                    arg.process.kill()
                    arg.process.wait(timeout=10)
                    arg.process = None
                    if arg.log_file:
                        arg.log_file.close()
                        arg.log_file = None
                    if release_port:
                        global USED_PORTS
                        USED_PORTS.discard(arg.port)
                except Exception:
                    print(f"Failed to terminate process {arg.process.pid}")
            else:
                print(f"Process is None on port {arg.port}")


def request_completion(model_name, prompt, port):
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
@pytest.mark.parametrize("service_discovery", ["etcd", "http"], indirect=True)
async def test_service_discovery(model_name, disagg_server_config,
                                 worker_config, router, service_discovery,
                                 disagg_port, work_dir):
    ctx_worker1 = None
    gen_worker1 = None
    disagg_server = None
    try:
        # initial cluster, 1 ctx, 1 gen, request should succeed
        ctx_worker1 = run_ctx_worker(model_name, worker_config, work_dir)
        gen_worker1 = run_gen_worker(model_name, worker_config, work_dir)
        disagg_server = run_disagg_server(disagg_server_config, work_dir,
                                          disagg_port)
        await wait_for_disagg_server_ready(disagg_port)
        verify_cluster_info(True, 1, 1, port=disagg_port)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=disagg_port)
        print(response)
    finally:
        terminate(ctx_worker1, gen_worker1, disagg_server)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize(
    "router", ["round_robin"], indirect=True
)  # use only round_robin to reduce the test time, this router type doesn't matter for this test
@pytest.mark.parametrize("service_discovery", ["etcd", "http"], indirect=True)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(600)
async def test_minimal_instances(model_name, disagg_server_config,
                                 worker_config, router, service_discovery,
                                 disagg_port, work_dir):
    # the cluster should have at least 2 ctx and 2 gen workers
    minimal_instances = {
        "context_servers": 2,
        "generation_servers": 2,
    }
    disagg_server_config["disagg_cluster"][
        "minimal_instances"] = minimal_instances
    worker_config["disagg_cluster"]["minimal_instances"] = minimal_instances

    ctx_worker1 = None
    gen_worker1 = None
    ctx_worker2 = None
    gen_worker2 = None
    disagg_server = None
    try:
        ctx_worker1 = run_ctx_worker(model_name, worker_config, work_dir)
        gen_worker1 = run_gen_worker(model_name, worker_config, work_dir)
        disagg_server = run_disagg_server(disagg_server_config, work_dir,
                                          disagg_port)
        await wait_for_worker_ready(ctx_worker1.port)
        await wait_for_worker_ready(gen_worker1.port)
        verify_cluster_info(False, 1, 1, port=disagg_port)
        # with only 1 ctx and 1 gen worker, the request should fail
        with pytest.raises(Exception):
            response = request_completion(model_name,
                                          "Hello, my name is",
                                          port=disagg_port)
            print(response)

        ctx_worker2 = run_ctx_worker(model_name, worker_config, work_dir)
        gen_worker2 = run_gen_worker(model_name, worker_config, work_dir)
        await wait_for_disagg_server_ready(disagg_port)
        verify_cluster_info(True, 2, 2, port=disagg_port)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=disagg_port)
        print(response)
    finally:
        terminate(ctx_worker1, ctx_worker2, gen_worker1, gen_worker2,
                  disagg_server)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("router", ROUTER_TYPES, indirect=True)
@pytest.mark.parametrize("service_discovery", ["etcd", "http"], indirect=True)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(600)
async def test_worker_restart(model_name, disagg_server_config, worker_config,
                              router, service_discovery, disagg_port, work_dir):
    ctx_worker1 = None
    ctx_worker2 = None
    gen_worker1 = None
    gen_worker2 = None
    disagg_server = None

    try:
        # initial cluster, 1 ctx, 1 gen, request should succeed
        ctx_worker1 = run_ctx_worker(model_name,
                                     worker_config,
                                     work_dir,
                                     device=0)
        gen_worker1 = run_gen_worker(model_name,
                                     worker_config,
                                     work_dir,
                                     device=1)
        disagg_server = run_disagg_server(disagg_server_config, work_dir,
                                          disagg_port)
        await wait_for_disagg_server_ready(disagg_port)
        verify_cluster_info(True, 1, 1, port=disagg_port)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=disagg_port)
        print(response)
        # kill gen1, the request should fail
        terminate(gen_worker1)
        await asyncio.sleep(CHECK_STATUS_INTERVAL)
        verify_cluster_info(False, 1, 0, port=disagg_port)
        with pytest.raises(Exception):
            request_completion(model_name,
                               "Hello, my name is",
                               port=disagg_port)

        test_prompt = "The capital of France is"

        # add gen2, the request should succeed
        gen_worker2 = run_gen_worker(model_name,
                                     worker_config,
                                     work_dir,
                                     port=0,
                                     device=2)
        await wait_for_worker_ready(gen_worker2.port)
        await asyncio.sleep(CHECK_STATUS_INTERVAL)
        verify_cluster_info(True, 1, 1, port=disagg_port)

        response = request_completion(model_name, test_prompt, port=disagg_port)
        print(response)
        response_text = response.choices[0].text
        assert len(response.choices[0].text) >= 1

        # kill ctx1, the request should fail
        terminate(ctx_worker1)
        await asyncio.sleep(CHECK_STATUS_INTERVAL)
        verify_cluster_info(False, 0, 1, port=disagg_port)
        with pytest.raises(Exception):
            request_completion(model_name, test_prompt, port=disagg_port)

        # add ctx2, the request should succeed
        ctx_worker2 = run_ctx_worker(model_name,
                                     worker_config,
                                     work_dir,
                                     port=0,
                                     device=3)
        await wait_for_worker_ready(ctx_worker2.port)
        verify_cluster_info(True, 1, 1, port=disagg_port)

        response = request_completion(model_name, test_prompt, port=disagg_port)
        response_text = response.choices[0].text
        assert len(response.choices[0].text) >= 1

        # start ctx1 and gen1 again, we have 2 ctxs and 2 gens now
        # Note: Do NOT start them with the same ports as the previous ones, the ports may be not released immediately after terminate,
        # causing a port conflict and test timeout.
        ctx_worker1 = run_ctx_worker(model_name, worker_config, work_dir)
        gen_worker1 = run_gen_worker(model_name, worker_config, work_dir)
        await wait_for_worker_ready(ctx_worker1.port)
        await wait_for_worker_ready(gen_worker1.port)
        await asyncio.sleep(CHECK_STATUS_INTERVAL)
        verify_cluster_info(True, 2, 2, port=disagg_port)

        # send 10 requests, the responses will be generated by the different ctx/gen workers (but we can't verify it now)
        for _ in range(10):
            response = request_completion(model_name,
                                          test_prompt,
                                          port=disagg_port)
            assert response.choices[0].text == response_text
        print(response)
    finally:
        terminate(ctx_worker1, ctx_worker2, gen_worker1, gen_worker2,
                  disagg_server)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("router", ["round_robin"], indirect=True)
@pytest.mark.parametrize("service_discovery", ["etcd", "http"], indirect=True)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(300)
async def test_disagg_server_restart(model_name, disagg_server_config,
                                     worker_config, router, service_discovery,
                                     disagg_port, work_dir):
    ctx_worker1 = None
    gen_worker1 = None
    disagg_server = None
    try:
        # initial cluster, 1 ctx, 1 gen, request should succeed
        ctx_worker1 = run_ctx_worker(model_name, worker_config, work_dir)
        gen_worker1 = run_gen_worker(model_name, worker_config, work_dir)
        disagg_server = run_disagg_server(disagg_server_config, work_dir,
                                          disagg_port)
        await wait_for_disagg_server_ready(disagg_port)
        verify_cluster_info(True, 1, 1, port=disagg_port)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=disagg_port)
        print(response)
        response_text = response.choices[0].text

        # kill disagg server, the request should fail
        terminate(disagg_server)
        await asyncio.sleep(CHECK_STATUS_INTERVAL)
        with pytest.raises(Exception):
            verify_cluster_info(False, 1, 1, expected_code=500)

        # restart disagg server, the request should succeed
        disagg_server = run_disagg_server(disagg_server_config, work_dir,
                                          disagg_port)
        await wait_for_disagg_server_ready(disagg_port)
        verify_cluster_info(True, 1, 1, port=disagg_port)
        response = request_completion(model_name,
                                      "Hello, my name is",
                                      port=disagg_port)
        print(response)
        assert response.choices[0].text == response_text

    finally:
        terminate(disagg_server, ctx_worker1, gen_worker1)
