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

import asyncio
import os
import signal
import subprocess
import tempfile
import time

import pytest
import requests
import yaml
from defs.common import get_free_port_in_ci as get_free_port
from disagg_test_utils import (CHECK_STATUS_INTERVAL, HEARTBEAT_INTERVAL,
                               INACTIVE_TIMEOUT, verify_cluster_info,
                               wait_for_disagg_server_ready)

from tensorrt_llm.logger import logger

# Configuration file paths
EXAMPLES_DIR = "examples/disaggregated"
TEST_CONFIGS_DIR = "tests/integration/defs/disaggregated/test_configs"
CLIENTS_DIR = f"{EXAMPLES_DIR}/clients"
CONTEXT_CONFIG_FILE = f"{TEST_CONFIGS_DIR}/context_extra-llm-api-config.yml"
GENERATION_CONFIG_FILE = f"{TEST_CONFIGS_DIR}/gen_extra-llm-api-config.yml"
ETCD_CONFIG_FILE = f"{TEST_CONFIGS_DIR}/etcd_config.yaml"
DISAGG_CONFIG_FILE = f"{EXAMPLES_DIR}/disagg_config.yaml"
CLIENT_SCRIPT_FILE = f"{CLIENTS_DIR}/disagg_client.py"
PROMPTS_FILE = f"{CLIENTS_DIR}/prompts.json"


def kill_automated_disaggregated_processes():
    """Kill any existing automated disaggregated processes."""
    try:
        subprocess.run(['pkill', '-9', '-f', 'trtllm-serve'], check=False)
    except Exception:
        pass


def cleanup_automated_output_files():
    """Clean up output files from previous runs."""
    for file in [
            'output.json', 'output_streaming.json', 'output_workers.log',
            'output_disagg.log'
    ]:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass


def start_context_server(config,
                         gpu_id: int,
                         port: int,
                         disagg_cluster_config: dict,
                         env=None) -> subprocess.Popen:
    """Start a context server with service discovery on specified GPU and port."""
    # Create worker config file with cluster settings
    worker_config = {
        "disagg_cluster": disagg_cluster_config,
        "disable_overlap_scheduler": True,
        "cache_transceiver_config": {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 2048
        },
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.2,
        },
    }

    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.yaml',
                                     delete=False,
                                     dir='.') as f:
        yaml.dump(worker_config, f)
        config_file = f.name

    cmd = [
        "trtllm-serve", "serve", config['model_path'], "--host", "localhost",
        "--port",
        str(port), "--config", config_file, "--server_role", "context"
    ]

    # FIX: Merge env with os.environ to preserve system variables
    server_env = os.environ.copy()
    if env:
        server_env.update(env)
    server_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    server_env["TRTLLM_USE_UCX_KVCACHE"] = "1"

    logger.info(f"Starting CONTEXT server on GPU {gpu_id} (port {port})...")
    # Write logs to files so we can debug issues
    stdout_log = open(f'context_server_{gpu_id}_stdout.log', 'w')
    stderr_log = open(f'context_server_{gpu_id}_stderr.log', 'w')
    process = subprocess.Popen(cmd,
                               env=server_env,
                               stdout=stdout_log,
                               stderr=stderr_log,
                               text=True,
                               bufsize=1)
    return process


def start_generation_server(config,
                            gpu_id: int,
                            port: int,
                            disagg_cluster_config: dict,
                            env=None) -> subprocess.Popen:
    """Start a generation server with service discovery on specified GPU and port."""
    # Create worker config file with cluster settings
    worker_config = {
        "disagg_cluster": disagg_cluster_config,
        "cache_transceiver_config": {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": 2048
        },
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.2,
        },
    }

    with tempfile.NamedTemporaryFile(mode='w',
                                     suffix='.yaml',
                                     delete=False,
                                     dir='.') as f:
        yaml.dump(worker_config, f)
        config_file = f.name

    cmd = [
        "trtllm-serve", "serve", config['model_path'], "--host", "localhost",
        "--port",
        str(port), "--config", config_file, "--server_role", "generation"
    ]

    # FIX: Merge env with os.environ to preserve system variables
    server_env = os.environ.copy()
    if env:
        server_env.update(env)
    server_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    server_env["TRTLLM_USE_UCX_KVCACHE"] = "1"

    logger.info(f"Starting GENERATION server on GPU {gpu_id} (port {port})...")
    # Write logs to files so we can debug issues
    stdout_log = open(f'generation_server_{gpu_id}_stdout.log', 'w')
    stderr_log = open(f'generation_server_{gpu_id}_stderr.log', 'w')
    process = subprocess.Popen(cmd,
                               env=server_env,
                               stdout=stdout_log,
                               stderr=stderr_log,
                               text=True,
                               bufsize=1)
    return process


def start_disaggregated_service(config, env=None) -> subprocess.Popen:
    """Launch the disaggregated service with service discovery."""
    cmd = ["trtllm-serve", "disaggregated", "-c", DISAGG_CONFIG_FILE]

    logger.info("Launching disaggregated service...")
    # Write logs to files so we can debug issues
    stdout_log = open('disagg_service_stdout.log', 'w')
    stderr_log = open('disagg_service_stderr.log', 'w')
    process = subprocess.Popen(cmd,
                               env=env,
                               stdout=stdout_log,
                               stderr=stderr_log,
                               text=True,
                               bufsize=1)
    return process


def wait_for_server_health(port: int, timeout: int = 120) -> bool:
    """Wait for server to be healthy by checking /health endpoint."""
    url = f"http://localhost:{port}/health"
    start_time = time.time()
    logger.info(f"Waiting for server on port {port} to be healthy...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Server on port {port} is healthy")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)

    logger.error(f"Timed out waiting for server on port {port}")
    return False


def run_client_test(config, env=None) -> bool:
    """Run the disaggregated client test."""
    cmd = [
        "python3", f"./{CLIENT_SCRIPT_FILE}", "-c", DISAGG_CONFIG_FILE, "-p",
        f"./{PROMPTS_FILE}"
    ]

    logger.info("Running disaggregated client test...")
    result = subprocess.run(cmd,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    if result.returncode == 0:
        logger.info("Client test succeeded")
        logger.info(f"Client output: {result.stdout}")
        return True
    else:
        logger.error(f"Client test failed with return code {result.returncode}")
        logger.error(f"Error output: {result.stderr}")
        logger.error(f"Standard output: {result.stdout}")
        return False


def kill_server_by_port(port: int) -> bool:
    """Find and kill a process by port using lsof."""
    try:
        # Find PID using port
        cmd = ["lsof", "-t", f"-i:{port}"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)

        if result.stdout.strip():
            pid = int(result.stdout.strip())
            os.kill(pid, signal.SIGKILL)
            logger.info(f"Killed process {pid} on port {port}")
            return True
        else:
            logger.warning(f"No process found on port {port}")
            return False
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {e}")
        return False


def cleanup_processes(processes):
    """Kill all started processes."""
    logger.info("Cleaning up all processes...")

    for name, process in processes.items():
        if process.poll() is None:  # Still running
            logger.info(f"Terminating {name} (PID: {process.pid})")
            try:
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                logger.warning(f"Force killing {name} (PID: {process.pid})")
                try:
                    process.kill()
                except ProcessLookupError:
                    pass


def start_etcd_server(working_dir, env=None) -> subprocess.Popen:
    """Start etcd server."""
    cmd = ["etcd"]

    logger.info("Starting etcd server...")
    # Write logs to files so we can debug issues
    stdout_log = open('etcd_stdout.log', 'w')
    stderr_log = open('etcd_stderr.log', 'w')
    process = subprocess.Popen(cmd,
                               env=env,
                               stdout=stdout_log,
                               stderr=stderr_log,
                               text=True,
                               bufsize=1,
                               cwd=working_dir)
    return process


def cleanup_etcd_data(env=None):
    """Clean up etcd data using etcdctl."""
    cmd = ["etcdctl", "del", "--prefix", "trtllm/"]

    logger.info("Cleaning etcd data...")
    result = subprocess.run(cmd,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    if result.returncode == 0:
        logger.info("Successfully cleaned etcd data")
    else:
        logger.warning(f"Failed to clean etcd data: {result.stderr}")


def create_config_files(config, disagg_cluster_config, disagg_port):
    """Create necessary configuration files with service discovery."""
    # Create context config file
    context_config_content = """pytorch_backend_config:
  disable_overlap_scheduler: True
cache_transceiver_config:
  backend: "DEFAULT"
  max_tokens_in_buffer: 2048"""

    with open(CONTEXT_CONFIG_FILE, 'w') as file:
        file.write(context_config_content)

    # Create generation config file
    generation_config_content = """cache_transceiver_config:
  backend: "DEFAULT"
  max_tokens_in_buffer: 2048"""

    with open(GENERATION_CONFIG_FILE, 'w') as file:
        file.write(generation_config_content)

    # Create etcd config file
    etcd_config_content = """server_type: "etcd"
hostname: "localhost"
port: 2379
health_check_timeout: 5.0"""

    with open(ETCD_CONFIG_FILE, 'w') as file:
        file.write(etcd_config_content)

    # Create disagg config file with service discovery - no hardcoded URLs or ports
    disagg_config = {
        "hostname": "localhost",
        "port": disagg_port,
        "backend": "pytorch",
        "disagg_cluster": disagg_cluster_config,
        "context_servers": {
            "router": {
                "type": "round_robin"
            }
        },
        "generation_servers": {
            "router": {
                "type": "round_robin"
            }
        }
    }
    with open(DISAGG_CONFIG_FILE, 'w') as file:
        yaml.dump(disagg_config, file)

    return True


def run_automated_disaggregated_test(example_dir, env=None, cwd=None):
    """Run automated disaggregated test with service discovery."""
    kill_automated_disaggregated_processes()
    cleanup_automated_output_files()

    # Use absolute path for model to avoid path resolution issues
    # The symlink is created in the cwd directory, so resolve the path from there
    if cwd:
        model_path = os.path.join(cwd, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    else:
        model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    config = {"model_path": model_path}

    # DEBUG: Log the model path and verify it exists (using print so it shows in pytest output)
    print(f"\n=== DEBUG: Model Configuration ===")
    print(f"Model path configured: {model_path}")
    print(f"Model path exists: {os.path.exists(model_path)}")
    print(f"Working directory (cwd): {cwd}")
    print(f"==================================\n")

    # Get a free port for the disaggregated proxy server
    disagg_port = get_free_port()

    # Define cluster config for service discovery
    disagg_cluster_config = {
        "cluster_uri": "etcd://localhost:2379",
        "cluster_name": "test_cluster",
        "heartbeat_interval_sec": HEARTBEAT_INTERVAL,
        "inactive_timeout_sec": INACTIVE_TIMEOUT,
    }

    # Create configuration files with service discovery
    create_config_files(config, disagg_cluster_config, disagg_port)
    processes = {}

    try:
        # Step 1: Start etcd server
        logger.info("Step 1: Starting etcd server...")
        processes["etcd"] = start_etcd_server(".", env=env)
        time.sleep(5)  # Give etcd time to start

        # Step 2: Clean etcd data
        logger.info("Step 2: Cleaning etcd data...")
        cleanup_etcd_data(env=env)

        # Step 3: Start context server on GPU 0 with auto port
        logger.info(
            "Step 3: Starting context server on GPU 0 with auto port...")
        processes["context_1"] = start_context_server(
            config,
            gpu_id=0,
            port=0,
            disagg_cluster_config=disagg_cluster_config,
            env=env)

        # Step 4: Start generation server on GPU 1 with auto port
        logger.info(
            "Step 4: Starting generation server on GPU 1 with auto port...")
        processes["generation_1"] = start_generation_server(
            config,
            gpu_id=1,
            port=0,
            disagg_cluster_config=disagg_cluster_config,
            env=env)

        # Step 5: Start disaggregated proxy service
        logger.info("Step 5: Starting disaggregated proxy service...")
        processes["disagg_service"] = start_disaggregated_service(config,
                                                                  env=env)

        # Step 6: Wait for service discovery to complete
        logger.info("Step 6: Waiting for workers to be discovered...")
        logger.info(f"Disagg server port: {disagg_port}")

        # Give processes a moment to fail if they're going to fail
        print("\n=== Waiting 3 seconds for processes to initialize... ===")
        time.sleep(3)

        # Check if processes are still running and FAIL immediately if any died
        for name, proc in processes.items():
            if proc.poll() is not None:
                print(
                    f"\n!!! Process {name} exited with code {proc.returncode} !!!"
                )
                if proc.stderr:
                    stderr_output = proc.stderr.read()
                    print(f"{name} stderr: {stderr_output[:2000]}")
                if proc.stdout:
                    stdout_output = proc.stdout.read()
                    print(f"{name} stdout: {stdout_output[:2000]}")
                raise RuntimeError(
                    f"Process {name} died during startup with exit code {proc.returncode}. "
                    f"Check logs above for stderr/stdout output.")

        # DEBUG: Log process status if they're still running
        print("\n=== All processes are still running. Process Status: ===")
        for name, proc in processes.items():
            print(f"  {name}: PID={proc.pid}, running={proc.poll() is None}")
        print("========================================================\n")

        asyncio.run(wait_for_disagg_server_ready(disagg_port))
        verify_cluster_info(ready=True,
                            ctx_workers=1,
                            gen_workers=1,
                            port=disagg_port)

        # Step 7: Run first client test
        logger.info("Step 7: Running first client test...")
        first_test_success = run_client_test(config, env=env)
        if not first_test_success:
            logger.error("First client test failed")
            return False

        # Step 8: Add second context server dynamically on GPU 2
        logger.info(
            "Step 8: Adding second context server on GPU 2 with auto port...")
        processes["context_2"] = start_context_server(
            config,
            gpu_id=2,
            port=0,
            disagg_cluster_config=disagg_cluster_config,
            env=env)

        # Step 9: Wait for new worker discovery and verify
        logger.info("Step 9: Waiting for new worker discovery...")
        time.sleep(CHECK_STATUS_INTERVAL + 2)
        verify_cluster_info(ready=True,
                            ctx_workers=2,
                            gen_workers=1,
                            port=disagg_port)

        second_test_success = run_client_test(config, env=env)
        if not second_test_success:
            logger.error("Second client test failed")
            return False

        # Step 10: Kill first context server
        logger.info("Step 10: Killing first context server...")
        if "context_1" in processes:
            process = processes["context_1"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

        # Step 11: Wait for inactive timeout and verify worker removal
        logger.info(
            "Step 11: Waiting for inactive timeout and worker removal...")
        time.sleep(INACTIVE_TIMEOUT + CHECK_STATUS_INTERVAL)
        verify_cluster_info(ready=True,
                            ctx_workers=1,
                            gen_workers=1,
                            port=disagg_port)

        # Step 12: Run final client test
        logger.info("Step 12: Running final client test...")
        final_test_success = run_client_test(config, env=env)
        if not final_test_success:
            logger.error("Final client test failed")
            return False

        logger.info("✅ All service discovery tests passed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during automated test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        cleanup_processes(processes)
        kill_automated_disaggregated_processes()


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_automated_disaggregated_complete(disaggregated_test_root,
                                          disaggregated_example_root, llm_venv,
                                          llama_model_root):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    success = run_automated_disaggregated_test(
        disaggregated_example_root,
        env=llm_venv._new_env,
        cwd=llm_venv.get_working_directory())
    assert success, "Automated disaggregated test failed"
