# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared utilities for disaggregated tests."""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from functools import wraps

import openai
import pytest
import requests
import yaml
from defs.common import get_free_port_in_ci as get_free_port

from tensorrt_llm.logger import logger

# Service discovery constants
HEARTBEAT_INTERVAL = 1
INACTIVE_TIMEOUT = 2
# Check cluster status with a larger interval than inactive timeout to avoid flaky tests
CHECK_STATUS_INTERVAL = 3


class ProcessWrapper:
    """Wrapper for subprocess with log file and port information."""

    def __init__(self, process, log_file=None, log_path=None, port=0):
        self.process = process
        self.log_file = log_file
        self.log_path = log_path
        self.port = port


def periodic_check(timeout=300, interval=3):
    """Decorator for periodic checking with timeout.

    Retries the decorated async function until it returns True or timeout is reached.
    Sleeps for interval seconds between retries.

    Args:
        timeout: Maximum time to wait in seconds
        interval: Time to sleep between checks in seconds

    Raises:
        TimeoutError: If timeout is reached without success
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    result = await func(*args, **kwargs)
                    if result:
                        return result
                except Exception as e:
                    logger.debug(f"Check failed: {e}")
                await asyncio.sleep(interval)
            raise TimeoutError(f"Timeout after {timeout}s waiting for {func.__name__}")

        return wrapper

    return decorator


def _run_worker(
    model_name, worker_config, role, port, work_dir, device=-1, save_log=False, env=None
):
    """Run a worker process (context or generation).

    Args:
        model_name: Path to the model
        worker_config: Worker configuration dict
        role: Role name (ctx/gen)
        port: Port number
        work_dir: Working directory for config files
        device: CUDA device ID (-1 for default)
        save_log: Whether to save logs to file
        env: Environment variables for the subprocess

    Returns:
        ProcessWrapper: Wrapped subprocess
    """
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
            "--config",
            worker_config_path,
            "--server_role",
            "context" if role.startswith("ctx") else "generation",
        ]
        if env is None:
            env = os.environ.copy()
        else:
            env = env.copy()
        log_file = None
        log_path = None
        if save_log:
            log_path = os.path.join(work_dir, f"worker_{role}_{port}.log")
            log_file = open(log_path, "w+")
            stdout = log_file
            stderr = log_file
        else:
            stdout = sys.stdout
            stderr = sys.stderr
        if device != -1:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        print(f"Running {role} on port {port}")
        return ProcessWrapper(
            subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr),
            log_file=log_file,
            log_path=log_path,
            port=port,
        )


def run_ctx_worker(model_name, ctx_worker_config, work_dir, port=0, device=0, env=None):
    """Launch a context worker with service discovery.

    Use port=0 to let the worker choose a free port.
    """
    return _run_worker(model_name, ctx_worker_config, "ctx", port, work_dir, device, env=env)


def run_gen_worker(model_name, gen_worker_config, work_dir, port=0, device=1, env=None):
    """Launch a generation worker with service discovery.

    Use port=0 to let the worker choose a free port.
    """
    return _run_worker(model_name, gen_worker_config, "gen", port, work_dir, device, env=env)


def run_disagg_server(disagg_cluster_config, work_dir, port=0, save_log=False, env=None):
    """Launch the disaggregated server.

    Args:
        disagg_cluster_config: Server configuration dict
        work_dir: Working directory for config files
        port: Port number
        save_log: Whether to save logs to file
        env: Environment variables for the subprocess

    Returns:
        ProcessWrapper: Wrapped subprocess
    """
    disagg_server_config_path = os.path.join(work_dir, "disagg_server_config.yaml")
    disagg_cluster_config["port"] = port
    with open(disagg_server_config_path, "w+") as f:
        yaml.dump(disagg_cluster_config, f)
    cmds = ["trtllm-serve", "disaggregated", "-c", disagg_server_config_path]
    log_file = None
    log_path = None
    if save_log:
        log_path = os.path.join(work_dir, "disagg_server.log")
        log_file = open(log_path, "w+")
        stdout = log_file
        stderr = log_file
    else:
        stdout = sys.stdout
        stderr = sys.stderr
    p = subprocess.Popen(cmds, env=env, stdout=stdout, stderr=stderr)
    return ProcessWrapper(p, log_file=log_file, log_path=log_path, port=port)


async def _wait_for_disagg_server_status(port, ready, min_ctx_workers=-1, min_gen_workers=-1):
    """Check disagg server status via /cluster_info endpoint.

    Args:
        port: Server port
        ready: Whether to check is_ready flag
        min_ctx_workers: Minimum context workers (-1 to skip check)
        min_gen_workers: Minimum generation workers (-1 to skip check)

    Returns:
        bool: True if all conditions are met
    """
    try:
        info_resp = requests.get(f"http://localhost:{port}/cluster_info", timeout=5)
        if info_resp.status_code != 200:
            return False
        info = info_resp.json()

        if ready and not info.get("is_ready", False):
            return False

        if min_ctx_workers != -1:
            ctx_count = len(info.get("current_workers", {}).get("context_servers", []))
            if ctx_count < min_ctx_workers:
                return False

        if min_gen_workers != -1:
            gen_count = len(info.get("current_workers", {}).get("generation_servers", []))
            if gen_count < min_gen_workers:
                return False

        return True
    except Exception as e:
        logger.debug(f"Failed to check server status: {e}")
        return False


@periodic_check(timeout=300, interval=3)
async def wait_for_disagg_server_ready(port):
    """Wait for disagg server to be ready."""
    return await _wait_for_disagg_server_status(port, True)


@periodic_check(timeout=300, interval=3)
async def wait_for_disagg_server_status(port, min_ctx_workers=-1, min_gen_workers=-1):
    """Wait for disagg server to have minimum number of workers."""
    return await _wait_for_disagg_server_status(port, False, min_ctx_workers, min_gen_workers)


@periodic_check(timeout=300, interval=3)
async def wait_for_worker_ready(port):
    """Wait for worker to be ready via /health endpoint."""
    logger.info(f"Waiting for worker {port} to be ready")
    try:
        info_resp = requests.get(f"http://localhost:{port}/health", timeout=5)
        return info_resp.status_code == 200
    except Exception:
        return False


@periodic_check(timeout=300, interval=3)
async def wait_for_port_released(port):
    """Wait for port to be released after killing a process.

    When we kill a server, the port is not released immediately.
    If the port is not released, bind will fail with OSError: [Errno 98] Address already in use.
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            print(f"Port {port} is released")
            return True
    except OSError:
        return False


def verify_cluster_info(ready, ctx_workers=-1, gen_workers=-1, port=0, expected_code=200):
    """Verify cluster info from /cluster_info endpoint.

    Args:
        ready: Expected is_ready status
        ctx_workers: Expected number of context workers (-1 to skip check)
        gen_workers: Expected number of generation workers (-1 to skip check)
        port: Server port
        expected_code: Expected HTTP status code
    """
    assert port > 0, "port must be positive"
    info_resp = requests.get(f"http://localhost:{port}/cluster_info")
    assert info_resp.status_code == expected_code
    info = info_resp.json()
    logger.info(f"verify_cluster_info: {info}, ready={ready}, ctx={ctx_workers}, gen={gen_workers}")
    assert info["is_ready"] == ready
    if ctx_workers != -1:
        assert len(info["current_workers"]["context_servers"]) == ctx_workers
    if gen_workers != -1:
        assert len(info["current_workers"]["generation_servers"]) == gen_workers


def tail(file_path, n):
    """Read last n lines from a file.

    Args:
        file_path: Path to file
        n: Number of lines to read

    Returns:
        str: Last n lines of the file
    """
    try:
        proc = subprocess.Popen(["tail", "-n", str(n), file_path], stdout=subprocess.PIPE)
        return proc.stdout.read().decode("utf-8")
    except Exception as e:
        print(f"Failed to tail {file_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return ""


def terminate(*args, show_log_lines=30):
    """Terminate processes and show their logs.

    Args:
        *args: ProcessWrapper instances to terminate
        show_log_lines: Number of log lines to show for debugging
    """
    for arg in args:
        if arg and isinstance(arg, ProcessWrapper):
            try:
                # Print log tail for debugging
                if arg.log_path and os.path.exists(arg.log_path):
                    print(f"-------------{arg.log_path}---------------")
                    try:
                        print(tail(arg.log_path, show_log_lines))
                    except Exception as e:
                        print(f"Failed to read log: {e}")
            except Exception as e:
                print(f"Failed to tail {arg.log_path}: {e}")

            if arg.process:
                print(f"Killing process {arg.process.pid}")
                try:
                    arg.process.kill()
                    arg.process.wait(timeout=10)
                    arg.process = None
                    if arg.log_file:
                        arg.log_file.close()
                        arg.log_file = None
                except Exception as e:
                    print(f"Failed to terminate process {arg.process.pid}: {e}")
            else:
                print(f"Process is None on port {arg.port}")


def request_completion(model_name, prompt, port):
    """Make a completion request to the disagg server.

    Args:
        model_name: Model name for the request
        prompt: Prompt text
        port: Server port

    Returns:
        Completion response from OpenAI client
    """
    client = openai.OpenAI(api_key="tensorrt_llm", base_url=f"http://localhost:{port}/v1")
    return client.completions.create(
        model=model_name, prompt=prompt, max_tokens=10, temperature=0.0
    )


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def disagg_port():
    """Get a free port for disaggregated server."""
    return get_free_port()


@pytest.fixture
def work_dir():
    """Create a temporary working directory."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def router(request):
    """Parameterized router fixture."""
    return request.param


@pytest.fixture
def service_discovery(request, disagg_port, work_dir):
    """Setup service discovery (etcd or http).

    Args:
        request.param: "etcd" or "http"

    Yields:
        tuple: (process or None, uri string)
    """
    if request.param == "etcd":
        data_dir = f"{work_dir}/disagg_test-etcd-{uuid.uuid4()}"
        etcd = subprocess.Popen(["etcd", "--data-dir", data_dir])
        yield etcd, "etcd://localhost:2379"
        try:
            etcd.kill()
            etcd.wait(timeout=10)
            shutil.rmtree(data_dir)
        except Exception:
            print(f"Failed to kill etcd: {traceback.format_exc()}")
    else:
        yield None, f"http://localhost:{disagg_port}"


@pytest.fixture
def disagg_cluster_config(service_discovery):
    """Create cluster config for workers and proxy server."""
    _, uri = service_discovery
    return {
        "cluster_uri": uri,
        "cluster_name": "test_cluster",
        "heartbeat_interval_sec": HEARTBEAT_INTERVAL,
        "inactive_timeout_sec": INACTIVE_TIMEOUT,
    }


@pytest.fixture
def disagg_server_config(disagg_cluster_config, router, disagg_port):
    """Create disaggregated server configuration."""
    return {
        "hostname": "localhost",
        "port": disagg_port,
        "disagg_cluster": disagg_cluster_config,
        "context_servers": {"router": {"type": router}},
        "generation_servers": {"router": {"type": router}},
    }


@pytest.fixture
def worker_config(disagg_cluster_config):
    """Create worker configuration."""
    return {
        "disagg_cluster": disagg_cluster_config,
        "disable_overlap_scheduler": True,
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.2,
            "enable_partial_reuse": False,
        },
        "cuda_graph_config": {},
    }
