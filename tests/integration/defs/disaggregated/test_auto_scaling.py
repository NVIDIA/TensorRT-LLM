import asyncio
import os

import pytest
import requests
from defs.conftest import llm_models_root
from disagg_test_utils import (CHECK_STATUS_INTERVAL, request_completion,
                               run_ctx_worker, run_disagg_server,
                               run_gen_worker, terminate, verify_cluster_info,
                               wait_for_disagg_server_ready,
                               wait_for_disagg_server_status,
                               wait_for_port_released)

ROUTER_TYPES = ["round_robin", "load_balancing", "kv_cache_aware"]


@pytest.fixture
def model_name():
    model_path = os.path.join(llm_models_root(),
                              "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    return model_path


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("router", ROUTER_TYPES, indirect=True)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.timeout(900)
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
@pytest.mark.timeout(900)
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
        await wait_for_disagg_server_status(disagg_port, 1, 1)
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
@pytest.mark.timeout(900)
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
                                     device=0)
        await wait_for_disagg_server_status(disagg_port, 1, 1)
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
                                     device=1)
        await wait_for_disagg_server_status(disagg_port, 1, 1)
        await asyncio.sleep(CHECK_STATUS_INTERVAL)
        verify_cluster_info(True, 1, 1, port=disagg_port)

        response = request_completion(model_name, test_prompt, port=disagg_port)
        response_text = response.choices[0].text
        assert len(response.choices[0].text) >= 1

        # start ctx1 and gen1 again, we have 2 ctxs and 2 gens now
        ctx_worker1 = run_ctx_worker(model_name,
                                     worker_config,
                                     work_dir,
                                     port=0,
                                     device=0)
        gen_worker1 = run_gen_worker(model_name,
                                     worker_config,
                                     work_dir,
                                     port=0,
                                     device=1)
        await wait_for_disagg_server_status(disagg_port, 2, 2)
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
@pytest.mark.timeout(900)
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
        # wait for the port to be released, so we can rebind the new process to the same port
        await wait_for_port_released(disagg_port)
        await asyncio.sleep(CHECK_STATUS_INTERVAL)

        with pytest.raises(requests.exceptions.RequestException):
            verify_cluster_info(False,
                                1,
                                1,
                                port=disagg_port,
                                expected_code=500)

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
