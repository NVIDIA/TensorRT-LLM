# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import subprocess

import aiohttp

try:
    import ray
except ImportError:
    import tensorrt_llm.executor.ray.stub as ray

import pytest
from defs.common import (get_free_port_in_ci, single_host_ucx_env,
                         venv_check_call, wait_for_server)
from defs.conftest import get_device_count, llm_models_root
from defs.trt_test_alternative import popen


@pytest.fixture(scope="module")
def ray_example_root(llm_root):
    example_root = os.path.join(llm_root, "examples", "ray_orchestrator")
    return example_root


def test_llm_inference_async_ray(ray_example_root, llm_venv):
    script_path = os.path.join(ray_example_root, "llm_inference_async_ray.py")
    model_path = f"{llm_models_root()}/Qwen3/Qwen3-0.6B"
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
        model_dir = f"{llm_models_root()}/Qwen3/Qwen3-0.6B"
        cmd.extend(["--model_dir", model_dir])

    venv_check_call(llm_venv, cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2"])
def test_ray_disaggregated_serving(ray_example_root, llm_venv, tp_size):
    _run_ray_disaggregated_serving(ray_example_root, tp_size)


def _run_ray_disaggregated_serving(ray_example_root, tp_size):

    if get_device_count() < tp_size * 2:
        pytest.skip(f"Need {tp_size * 2} GPUs.")

    # CI nodes can be shared with other jobs, so a fixed port may already be
    # taken. Ask for free ports; get_free_port_in_ci() takes them from the
    # stage's port range when CI sets one, and otherwise probes for an unused
    # port.
    server_port = get_free_port_in_ci()
    ctx_port = get_free_port_in_ci()
    gen_port = get_free_port_in_ci()

    disagg_dir = os.path.join(ray_example_root, "disaggregated")
    script_path = os.path.join(disagg_dir, "disagg_serving_local.sh")
    model_dir = f"{llm_models_root()}/Qwen3/Qwen3-0.6B"

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
            "TLLM_RAY_FORCE_LOCAL_CLUSTER": "0",
            # With UCX auto-selection the NIXL agent AM channel can pick an IB
            # loopback path that times out on some DGX nodes, turning into a
            # permanent request hang (https://nvbugs/6670614).
            **single_host_ucx_env(),
        })
        cmd = [
            "bash",
            script_path,
            "--attach",
            "--executor",
            "ray",
            "--model",
            model_dir,
            "--tp_size",
            str(tp_size),
            "--transceiver_backend",
            "NIXL",
            "--transceiver_runtime",
            "PYTHON",
            "--port",
            str(server_port),
            "--ctx_port",
            str(ctx_port),
            "--gen_port",
            str(gen_port),
        ]
        with popen(
                cmd,
                cwd=disagg_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_copy,
        ):
            assert wait_for_server(
                "localhost", server_port, timeout_seconds=300), \
                "Disaggregated server failed to start within 5 minutes"

            _run_completion_requests(server_port)
    finally:
        ray.shutdown()


def _run_completion_requests(server_port):
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
            "model": "Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }
        async with session.post(
                f"http://localhost:{server_port}/v1/completions",
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
    assert "The capital of Germany is Berlin" in content, (
        "Expected string 'The capital of Germany is Berlin' not found in responses"
    )
    # Different models (and sampling runs) phrase the continuation of the raw
    # asyncio prompt differently, so match case-insensitively on the prompt's
    # own keyword instead of an exact phrase (see test_disaggregated.py's
    # run_client_tests for the same rationale).
    assert "asyncio" in content.lower(), (
        "Expected 'asyncio' not found in responses")
    assert "Berlin Berlin" not in content, (
        "Unexpected string 'Berlin Berlin' found in responses")
