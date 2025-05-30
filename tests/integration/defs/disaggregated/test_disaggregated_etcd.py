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
        str(port), "--backend", "pytorch", "--extra_llm_api_options",
        config['extra_llm_api_path'], "--metadata_server_config_file",
        config['etcd_config_path'], "--server_role", "CONTEXT"
    ]

    server_env = env.copy() if env else os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
        str(port), "--backend", "pytorch", "--extra_llm_api_options",
        config['extra_llm_api_path'], "--metadata_server_config_file",
        config['etcd_config_path'], "--server_role", "GENERATION"
    ]

    server_env = env.copy() if env else os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
        "trtllm-serve", "disaggregated", "-c", config['disagg_config_path'],
        "-m", config['etcd_config_path']
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
        "python3", config['client_script_path'], "-c",
        config['disagg_config_path'], "-p", config['prompts_path']
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


def run_automated_disaggregated_test(example_dir, env=None, cwd=None):
    """Run automated disaggregated test with given configuration."""
    kill_automated_disaggregated_processes()
    cleanup_automated_output_files()

    config = {
        "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "extra_llm_api_path": f"{example_dir}/extra-llm-api-config.yml",
        "etcd_config_path": f"{example_dir}/etcd_config.yaml",
        "disagg_config_path": f"{example_dir}/disagg_config.yaml",
        "client_script_path": f"{example_dir}/clients/disagg_client.py",
        "prompts_path": f"{example_dir}/clients/prompts.json"
    }

    processes = {}

    try:
        # Start initial servers
        processes["context_8001"] = start_context_server(config,
                                                         gpu_id=0,
                                                         port=8001,
                                                         env=env)
        processes["generation_8002"] = start_generation_server(config,
                                                               gpu_id=1,
                                                               port=8002,
                                                               env=env)

        # Wait for initial servers to be healthy
        if not wait_for_server_health(port=8001):
            return False
        if not wait_for_server_health(port=8002):
            return False

        # Start disaggregated service
        processes["disagg_service"] = start_disaggregated_service(config,
                                                                  env=env)

        # Wait for disaggregated service
        if not wait_for_server_health(port=8000):
            return False

        # Start second context server
        processes["context_8003"] = start_context_server(config,
                                                         gpu_id=2,
                                                         port=8003,
                                                         env=env)

        # Wait for second context server
        if not wait_for_server_health(port=8003):
            return False

        # Run the first client test
        first_test_success = run_client_test(config, env=env)
        if not first_test_success:
            logger.error("First client test failed")
            return False

        # Kill the first context server
        if "context_8001" in processes:
            process = processes["context_8001"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        kill_server_by_port(8001)

        # Wait a moment for the service to recognize the server is gone
        logger.info(
            "Waiting for the service to recognize the server removal...")
        time.sleep(20)

        # Run the second client test
        second_test_success = run_client_test(config, env=env)
        if not second_test_success:
            logger.error("Second client test failed")
            return False

        logger.info("Both client tests passed successfully!")
        return True

    except Exception as e:
        logger.exception(f"Error during test: {e}")
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
