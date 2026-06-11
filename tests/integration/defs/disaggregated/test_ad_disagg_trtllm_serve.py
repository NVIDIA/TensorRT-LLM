# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from pathlib import Path

import pytest
import requests
from defs.common import get_free_port_in_ci as get_free_port
from defs.conftest import llm_models_root
from disagg_test_utils import (
    CHECK_STATUS_INTERVAL,
    HEARTBEAT_INTERVAL,
    INACTIVE_TIMEOUT,
    run_ctx_worker,
    run_disagg_server,
    run_gen_worker,
    terminate,
)
from openai import OpenAI

pytest_plugins = ["disagg_test_utils"]

SERVER_START_TIMEOUT_S = 300
SERVER_READY_REQUEST_TIMEOUT_S = 5
OPENAI_REQUEST_TIMEOUT_S = 60
PROXY_PORT_MAX_RETRIES = 5
TINYLLAMA_MODEL_DIR = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
AUTODEPLOY_BACKEND = "_autodeploy"
EXPECTED_COMPLETION_SUBSTRING = "Berlin"


def tinyllama_model_path():
    return str(Path(llm_models_root()) / TINYLLAMA_MODEL_DIR)


def worker_cuda_devices(num_workers):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        if len(devices) < num_workers:
            pytest.skip(
                f"AutoDeploy trtllm-serve disagg smoke requires {num_workers} "
                f"visible GPUs, got {len(devices)}"
            )
        return devices[:num_workers]

    return [str(device) for device in range(num_workers)]


def autodeploy_worker_config(disagg_cluster, disable_overlap_scheduler=False):
    config = {
        "backend": AUTODEPLOY_BACKEND,
        "max_batch_size": 1,
        "cuda_graph_config": {"batch_sizes": [1]},
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "disagg_cluster": disagg_cluster,
    }
    if disable_overlap_scheduler:
        config["disable_overlap_scheduler"] = True

    return config


def disagg_cluster_config(port):
    """Create the service-discovery config shared by workers and proxy."""
    return {
        "cluster_uri": f"http://localhost:{port}",
        "cluster_name": "autodeploy_disagg_smoke",
        "heartbeat_interval_sec": HEARTBEAT_INTERVAL,
        "inactive_timeout_sec": INACTIVE_TIMEOUT,
        "minimal_instances": {
            "context_servers": 1,
            "generation_servers": 1,
        },
    }


def proxy_config(port, disagg_cluster):
    """Create a disaggregated proxy config that discovers workers dynamically."""
    return {
        "hostname": "localhost",
        "port": port,
        "backend": AUTODEPLOY_BACKEND,
        "disagg_cluster": disagg_cluster,
        "context_servers": {"router": {"type": "round_robin"}},
        "generation_servers": {"router": {"type": "round_robin"}},
    }


def _process_log(process_wrapper):
    """Read captured subprocess output when the utility saved it to a file."""
    if process_wrapper is None or process_wrapper.log_path is None:
        return "No process log was captured."
    try:
        with open(process_wrapper.log_path) as log_file:
            return log_file.read()
    except OSError as exc:
        return f"Failed to read process log {process_wrapper.log_path}: {exc}"


async def wait_for_disagg_server_ready_or_exit(port, processes, timeout, request_timeout):
    """Wait for proxy readiness, but fail fast if any subprocess exits."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_readiness_error = "no readiness check was attempted"
    while loop.time() < deadline:
        for name, process_wrapper in processes.items():
            if (
                process_wrapper
                and process_wrapper.process
                and process_wrapper.process.poll() is not None
            ):
                # Process exited before the server became ready.
                log = _process_log(process_wrapper)
                startup_error = RuntimeError(
                    f"{name} process exited before disaggregated server became ready "
                    f"(returncode={process_wrapper.process.returncode}).\n{log}"
                )
                raise startup_error

        try:
            response = requests.get(
                f"http://localhost:{port}/cluster_info", timeout=request_timeout
            )
            if response.status_code == 200 and response.json().get("is_ready", False):
                # Server is ready.
                return
            last_readiness_error = (
                f"last /cluster_info response: status={response.status_code}, body={response.text}"
            )
        except requests.RequestException as exc:
            last_readiness_error = f"last /cluster_info request failed: {exc}"

        await asyncio.sleep(CHECK_STATUS_INTERVAL)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for disaggregated server on port {port}; "
        f"{last_readiness_error}"
    )


@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.skip_less_device(2)
@pytest.mark.timeout(900)
@pytest.mark.asyncio(loop_scope="module")
async def test_openai_completion(work_dir):
    """Smoke test AutoDeploy disagg through trtllm-serve and the OpenAI API.

    The lower-level tests in ``test_ad_disagg.py`` drive AutoDeploy workers
    directly and inspect context/generation handoff metadata. This test instead
    verifies the trtllm-serve deployment shape: context worker, generation
    worker, disaggregated proxy, and an OpenAI-compatible completion request.
    """
    model = tinyllama_model_path()
    ctx_device, gen_device = worker_cuda_devices(2)

    last_port_conflict = None
    response = None
    for attempt in range(PROXY_PORT_MAX_RETRIES):
        disagg_port = get_free_port()
        disagg_cluster = disagg_cluster_config(disagg_port)
        ctx_worker = None
        gen_worker = None
        disagg_server = None

        try:
            # Use the same service-discovery path as the broader PyTorch disagg
            # tests for worker ports. Passing port=0 lets each trtllm-serve worker
            # bind an OS-selected port in the child process and register that port
            # with the disaggregated proxy.
            ctx_worker = run_ctx_worker(
                model,
                autodeploy_worker_config(disagg_cluster, disable_overlap_scheduler=True),
                work_dir,
                port=0,
                device=ctx_device,
            )
            gen_worker = run_gen_worker(
                model,
                autodeploy_worker_config(disagg_cluster),
                work_dir,
                port=0,
                device=gen_device,
            )
            disagg_server = run_disagg_server(
                proxy_config(disagg_port, disagg_cluster),
                work_dir,
                disagg_port,
                save_log=True,
            )
            try:
                await wait_for_disagg_server_ready_or_exit(
                    disagg_port,
                    {
                        "context worker": ctx_worker,
                        "generation worker": gen_worker,
                        "disaggregated proxy": disagg_server,
                    },
                    SERVER_START_TIMEOUT_S,
                    SERVER_READY_REQUEST_TIMEOUT_S,
                )
            except RuntimeError as exc:
                last_port_conflict = exc
                if "disaggregated proxy" not in str(exc) or (
                    "EADDRINUSE" not in str(exc)
                    and "address already in use" not in str(exc).lower()
                ):
                    raise
                print(
                    f"AutoDeploy disagg serve attempt {attempt + 1} of {PROXY_PORT_MAX_RETRIES} "
                    f"failed with proxy port conflict, retrying: {exc}"
                )
                continue

            client = OpenAI(
                api_key="tensorrt_llm",
                base_url=f"http://localhost:{disagg_port}/v1",
                timeout=OPENAI_REQUEST_TIMEOUT_S,
                max_retries=0,
            )
            response = client.completions.create(
                model=model,
                prompt="What is the capital of Germany?",
                max_tokens=32,
                temperature=0,
                extra_body={"ignore_eos": True},
            )
            break
        finally:
            terminate(ctx_worker, gen_worker, disagg_server)

    if response is None:
        raise RuntimeError(
            f"Failed to start AutoDeploy disagg serve smoke after {PROXY_PORT_MAX_RETRIES} "
            "proxy port attempts"
        ) from last_port_conflict

    assert response.choices
    response_text = response.choices[0].text
    assert EXPECTED_COMPLETION_SUBSTRING in response_text, (
        f"expected {EXPECTED_COMPLETION_SUBSTRING!r} in response, got {response_text!r}"
    )
