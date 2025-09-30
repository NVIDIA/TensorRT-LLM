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

import os
import signal
import subprocess
import time

import pytest
import requests

from tensorrt_llm.logger import logger

# Configuration file paths
EXAMPLES_DIR = "examples/disaggregated"
CLIENTS_DIR = f"{EXAMPLES_DIR}/clients"
CONTEXT_CONFIG_FILE = f"{EXAMPLES_DIR}/context_extra-llm-api-config.yml"
GENERATION_CONFIG_FILE = f"{EXAMPLES_DIR}/gen_extra-llm-api-config.yml"
ETCD_CONFIG_FILE = f"{EXAMPLES_DIR}/etcd_config.yaml"
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
                         env=None) -> subprocess.Popen:
    """Start a context server on specified GPU and port."""
    cmd = [
        "trtllm-serve", config['model_path'], "--host", "localhost", "--port",
        str(port), "--extra_llm_api_options", f"./{CONTEXT_CONFIG_FILE}",
        "--metadata_server_config_file", ETCD_CONFIG_FILE, "--server_role",
        "CONTEXT"
    ]

    server_env = env.copy() if env else os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    server_env["TRTLLM_USE_UCX_KVCACHE"] = "1"

    logger.info(f"Starting CONTEXT server on GPU {gpu_id} (port {port})...")
    process = subprocess.Popen(cmd,
                               env=server_env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               bufsize=1)
    return process


def start_generation_server(config,
                            gpu_id: int,
                            port: int,
                            env=None) -> subprocess.Popen:
    """Start a generation server on specified GPU and port."""
    cmd = [
        "trtllm-serve", config['model_path'], "--host", "localhost", "--port",
        str(port), "--extra_llm_api_options", f"./{GENERATION_CONFIG_FILE}",
        "--metadata_server_config_file", ETCD_CONFIG_FILE, "--server_role",
        "GENERATION"
    ]

    server_env = env.copy() if env else os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    server_env["TRTLLM_USE_UCX_KVCACHE"] = "1"

    logger.info(f"Starting GENERATION server on GPU {gpu_id} (port {port})...")
    process = subprocess.Popen(cmd,
                               env=server_env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               bufsize=1)
    return process


def start_disaggregated_service(config, env=None) -> subprocess.Popen:
    """Launch the disaggregated service."""
    cmd = [
        "trtllm-serve", "disaggregated", "-c", DISAGG_CONFIG_FILE, "-m",
        ETCD_CONFIG_FILE
    ]

    logger.info("Launching disaggregated service...")
    process = subprocess.Popen(cmd,
                               env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
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
    process = subprocess.Popen(cmd,
                               env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
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


def create_config_files(config):
    """Create necessary configuration files"""
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

    disagg_config_content = """hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8002"
"""

    with open(DISAGG_CONFIG_FILE, 'w') as file:
        file.write(disagg_config_content)

    return True


def run_automated_disaggregated_test(example_dir, env=None, cwd=None):
    """Run automated disaggregated test with given configuration."""
    kill_automated_disaggregated_processes()
    cleanup_automated_output_files()

    config = {"model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

    # Create configuration files
    create_config_files(config)
    processes = {}

    try:
        # Step 1: Start etcd server
        logger.info("Step 1: Starting etcd server...")
        processes["etcd"] = start_etcd_server(".", env=env)
        time.sleep(5)  # Give etcd time to start

        # Step 2: Clean etcd data
        logger.info("Step 2: Cleaning etcd data...")
        cleanup_etcd_data(env=env)

        # Step 3: Start context server on GPU 0 (port 8001)
        logger.info("Step 3: Starting context server on GPU 0 (port 8001)...")
        processes["context_8001"] = start_context_server(config,
                                                         gpu_id=0,
                                                         port=8001,
                                                         env=env)

        # Step 4: Start generation server on GPU 1 (port 8002)
        logger.info(
            "Step 4: Starting generation server on GPU 1 (port 8002)...")
        processes["generation_8002"] = start_generation_server(config,
                                                               gpu_id=1,
                                                               port=8002,
                                                               env=env)

        # Step 5: Wait till gen and context ready
        logger.info(
            "Step 5: Waiting for context and generation servers to be ready...")
        if not wait_for_server_health(port=8001):
            logger.error("Context server on port 8001 failed to start")
            return False
        if not wait_for_server_health(port=8002):
            logger.error("Generation server on port 8002 failed to start")
            return False

        # Step 6: Start disaggregated service
        logger.info("Step 6: Starting disaggregated service...")
        processes["disagg_service"] = start_disaggregated_service(config,
                                                                  env=env)

        # Step 7: Wait for disaggregated service and run first client test
        logger.info(
            "Step 7: Waiting for disaggregated service and running first client test..."
        )
        if not wait_for_server_health(port=8000):
            logger.error("Disaggregated service failed to start")
            return False

        first_test_success = run_client_test(config, env=env)
        if not first_test_success:
            logger.error("First client test failed")
            return False

        # Step 8: Start second context server on GPU 2 (port 8003)
        logger.info(
            "Step 8: Starting second context server on GPU 2 (port 8003)...")
        processes["context_8003"] = start_context_server(config,
                                                         gpu_id=2,
                                                         port=8003,
                                                         env=env)

        # Step 9: Wait till ready and then 10 seconds, run second client test
        logger.info(
            "Step 9: Waiting for second context server and running second client test..."
        )
        if not wait_for_server_health(port=8003):
            logger.error("Second context server on port 8003 failed to start")
            return False

        logger.info("Waiting additional 10 seconds for system stabilization...")
        time.sleep(10)

        second_test_success = run_client_test(config, env=env)
        if not second_test_success:
            logger.error("Second client test failed")
            return False

        # Step 10: Kill 8001 process (first context server)
        logger.info("Step 10: Killing first context server (port 8001)...")
        if "context_8001" in processes:
            process = processes["context_8001"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        kill_server_by_port(8001)

        # Step 11: Wait a few seconds and run final client test
        logger.info(
            "Step 11: Waiting a few seconds and running final client test...")
        time.sleep(5)

        final_test_success = run_client_test(config, env=env)
        if not final_test_success:
            logger.error("Final client test failed")
            return False

        logger.info("✅ All automated disaggregated tests passed successfully!")
        return True

    except Exception as e:
        logger.exception(f"Error during automated test: {e}")
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
