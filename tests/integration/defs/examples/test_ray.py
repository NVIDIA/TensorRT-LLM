# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import socket
import subprocess
import time

import aiohttp
import psutil

try:
    import ray
except ImportError:
    import tensorrt_llm.executor.ray.stub as ray

import pytest
from defs.common import venv_check_call, wait_for_server
from defs.conftest import get_device_count, llm_models_root
from defs.trt_test_alternative import popen


@pytest.fixture(scope="module")
def ray_example_root(llm_root):
    example_root = os.path.join(llm_root, "examples", "ray_orchestrator")
    return example_root


def test_llm_inference_async_ray(ray_example_root, llm_venv):
    script_path = os.path.join(ray_example_root, "llm_inference_async_ray.py")
    model_path = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
    venv_check_call(llm_venv, [script_path, "--model", model_path])


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("tp_size,pp_size,ep_size", [
    (2, 1, -1),
    (1, 2, -1),
    (2, 2, -1),
    (2, 1, 2),
],
                         ids=["tp2", "pp2", "tp2pp2", "tep2"])
def test_llm_inference_distributed_ray(ray_example_root, llm_venv, tp_size,
                                       pp_size, ep_size):
    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs.")

    script_path = os.path.join(ray_example_root,
                               "llm_inference_distributed_ray.py")

    cmd = [
        script_path, "--tp_size",
        str(tp_size), "--pp_size",
        str(pp_size), "--moe_ep_size",
        str(ep_size)
    ]

    if ep_size != -1:
        model_dir = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
        cmd.extend(["--model_dir", model_dir])
    else:
        model_dir = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
        cmd.extend(["--model_dir", model_dir])

    venv_check_call(llm_venv, cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2"])
def test_ray_disaggregated_serving(ray_example_root, llm_venv, tp_size):
    _run_ray_disaggregated_serving(ray_example_root, tp_size, "NIXL", "CPP")


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2"])
def test_ray_disaggregated_serving_python(ray_example_root, llm_venv, tp_size):
    _run_ray_disaggregated_serving(ray_example_root, tp_size, "NIXL", "PYTHON")


DISAGG_SERVING_PORTS = (8000, 8001, 8002)

IGNORED_PSUTIL_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied,
                         psutil.ZombieProcess)


def _cleanup_leftover_servers(ports: tuple = DISAGG_SERVING_PORTS,
                              timeout: float = 120) -> None:
    """Kill leaked disagg server processes and wait for their ports to free.

    A passing test can still leave live trtllm-serve/orted subprocesses behind
    (see nvbugs 6601574): the next test's servers then fail to bind the ports
    and the proxy never comes up. SO_REUSEADDR only covers sockets left in
    TIME_WAIT, not ports held by processes that are still alive.
    """
    patterns = ("trtllm-serve", "orted")
    # Never kill our own process tree: importing tensorrt_llm initializes MPI,
    # which attaches a singleton orted daemon to this very pytest process;
    # killing it makes Open MPI abort the whole test session.
    me = psutil.Process()
    protected = {me.pid}
    try:
        protected.update(parent.pid for parent in me.parents())
        protected.update(child.pid for child in me.children(recursive=True))
    except IGNORED_PSUTIL_ERRORS:
        pass

    # Restrict the kill to stale processes: listeners on the disagg ports and
    # orphans left behind after a previous test's process tree was torn down.
    # This spares unrelated serving/MPI jobs on shared development machines.
    try:
        port_owners = {
            conn.pid
            for conn in psutil.net_connections(kind="tcp")
            if conn.pid and conn.status == psutil.CONN_LISTEN and conn.laddr
            and conn.laddr.port in ports
        }
    except IGNORED_PSUTIL_ERRORS:
        port_owners = set()

    victims = []
    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            if proc.info["pid"] in protected:
                continue
            cmdline = " ".join(proc.info["cmdline"] or [])
            if not any(pattern in cmdline for pattern in patterns):
                continue
            if proc.info["pid"] in port_owners or proc.info["ppid"] == 1:
                victims.append(proc)
        except IGNORED_PSUTIL_ERRORS:
            pass

    for proc in victims:
        try:
            children = proc.children(recursive=True)
        except IGNORED_PSUTIL_ERRORS:
            children = []
        for target in [proc] + children:
            try:
                if target.pid not in protected:
                    target.kill()
            except IGNORED_PSUTIL_ERRORS:
                pass

    deadline = time.time() + timeout
    for port in ports:
        while time.time() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    pass
            except OSError:
                break  # nothing is listening on this port anymore
            time.sleep(2)
        else:
            raise RuntimeError(
                f"Port {port} is still in use after {timeout}s of cleanup")


def _run_ray_disaggregated_serving(ray_example_root, tp_size,
                                   transceiver_backend, transceiver_runtime):

    if get_device_count() < tp_size * 2:
        pytest.skip(f"Need {tp_size * 2} GPUs.")

    _cleanup_leftover_servers()

    disagg_dir = os.path.join(ray_example_root, "disaggregated")
    script_path = os.path.join(disagg_dir, "disagg_serving_local.sh")
    model_dir = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"

    try:
        runtime_env = {
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"
            }
        }
        ray.init(address="local",
                 include_dashboard=False,
                 ignore_reinit_error=True,
                 runtime_env=runtime_env)
        gcs_addr = ray.get_runtime_context().gcs_address
        ray_port = str(gcs_addr.split(":")[1])

        env_copy = os.environ.copy()
        env_copy.update({
            "RAY_ADDRESS": f"localhost:{ray_port}",
            "TLLM_RAY_FORCE_LOCAL_CLUSTER": "0"
        })
        with popen(
            [
                "bash", script_path, "--executor", "ray", "--attach", "--model",
                model_dir, "--tp_size",
                str(tp_size), "--transceiver_backend", transceiver_backend,
                "--transceiver_runtime", transceiver_runtime
            ],
                cwd=disagg_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_copy,
        ):
            assert wait_for_server("localhost", 8000, timeout_seconds=300), \
                "Disaggregated server failed to start within 5 minutes"

            _run_completion_requests()
    finally:
        ray.shutdown()
        _cleanup_leftover_servers()


def _run_completion_requests():
    prompts = [
        "What is the capital of Germany?",
        "Explain the theory of relativity.",
        "What are the benefits of using asyncio in Python?",
        "Describe the process of photosynthesis.",
        "How does a blockchain work?",
    ]
    max_tokens = 32

    async def send_request(session, prompt):
        payload = {
            "model": "TinyLlama-1.1B-Chat-v1.0",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }
        async with session.post("http://localhost:8000/v1/completions",
                                json=payload) as response:
            response_text = await response.text()
            assert response.status == 200, (
                f"Completion request failed with HTTP {response.status}: "
                f"{response_text}")
            return json.loads(response_text)

    async def run_requests():
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await asyncio.gather(
                *[send_request(session, prompt) for prompt in prompts])

    responses = asyncio.run(run_requests())
    response_ids = set()
    generated_texts = []
    for index, response in enumerate(responses):
        choices = response.get("choices") or []
        assert choices, f"Request {index} has no choices: {response}"

        choice = choices[0]
        text = choice.get("text") or ""
        assert text.strip(), f"Request {index} returned empty text: {response}"
        assert choice.get("finish_reason") == "length", (
            f"Request {index} has unexpected finish reason: {response}")
        assert choice.get("disaggregated_params") is not None, (
            f"Request {index} is missing disaggregated metadata: {response}")

        usage = response.get("usage") or {}
        assert usage.get("completion_tokens") == max_tokens, (
            f"Request {index} completion_tokens mismatch: "
            f"got={usage.get('completion_tokens')} expected={max_tokens}")
        assert usage.get("total_tokens") == (
            usage.get("prompt_tokens", 0) +
            max_tokens), (f"Request {index} has inconsistent usage: {usage}")

        response_id = response.get("id")
        assert response_id, f"Request {index} is missing response id: {response}"
        assert response_id not in response_ids, (
            f"Request {index} reused response id {response_id}")
        response_ids.add(response_id)
        generated_texts.append(text)
        print(f"Request {index} response: {text}")

    content = "\n".join(generated_texts)
    for expected_string in [
            "The capital of Germany is Berlin",
            "Asyncio is a Python library",
    ]:
        assert expected_string in content, (
            f"Expected string {expected_string!r} not found in responses")
    assert "Berlin Berlin" not in content, (
        "Unexpected string 'Berlin Berlin' found in responses")
