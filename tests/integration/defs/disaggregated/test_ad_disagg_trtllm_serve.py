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

import pytest
import requests
from defs.conftest import check_device_contain
from disagg_test_utils import CHECK_STATUS_INTERVAL, HEARTBEAT_INTERVAL, INACTIVE_TIMEOUT

pytest_plugins = ["disagg_test_utils"]


@pytest.fixture(autouse=True)
def skip_b300():
    if check_device_contain(["B300"]):
        pytest.skip(
            "AutoDeploy disagg tests are disabled on B300/GB300 until capacity is available: "
            "https://nvbugs/6301621"
        )


SERVER_START_TIMEOUT_S = 300
SERVER_READY_REQUEST_TIMEOUT_S = 5
OPENAI_REQUEST_TIMEOUT_S = 60
PROXY_PORT_MAX_RETRIES = 5
AUTODEPLOY_BACKEND = "_autodeploy"
EXPECTED_COMPLETION_SUBSTRING = "Berlin"


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
