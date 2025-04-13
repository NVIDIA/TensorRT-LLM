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
"""
Stress test script for inference of model using TensorRT-LLM with PyTorch/TRT backend.
This script is used for stress testing inference performance using trtllm-serve and genai-perf.
"""
import contextlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from glob import glob
from typing import List, Optional, Tuple

import pandas as pd
import pytest
import requests
import yaml
from defs.conftest import get_device_count, get_device_memory, llm_models_root
from defs.trt_test_alternative import (Popen, cleanup_process_tree, print_info,
                                       print_warning)

# Install genai-perf in requirements-dev.txt will affect triton and pytorch version mismatch
# def genai_perf_install():
#     """Ensures genai-perf is installed without affecting the global environment"""

#     import os
#     import subprocess
#     import sys

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     requirements_file = os.path.join(current_dir,
#                                      "requirements-stress-test.txt")

#     if not os.path.exists(requirements_file):
#         with open(requirements_file, "w") as f:
#             f.write("genai-perf\n")

#     subprocess.check_call(
#         [sys.executable, "-m", "pip", "install", "-r", requirements_file])

# Define a constant for process termination timeouts
GRACEFUL_TERMINATION_TIMEOUT = 10  # seconds - set longer when stress large model


@dataclass(frozen=True)
class ServerConfig:
    """Dataclass to store server configuration for trtllm-serve"""
    port: int = 8000
    host: str = "localhost"
    pp_size: int = 1
    ep_size: Optional[int] = 1
    max_batch_size: Optional[int] = 1024  # 2048 is default value in BuildConfig
    max_num_tokens: Optional[int] = 8192  # 8192 is default value in BuildConfig
    kv_cache_free_gpu_memory_fraction: Optional[
        float] = 0.9  # 0.9 is default value in BuildConfig
    capacity_scheduler_policy: str = "GUARANTEED_NO_EVICT"
    wait_interval: int = 10  # seconds
    max_wait_seconds: int = 600  # 10 mins <- Larger model need longer model loading time
    health_check_timeout: float = 8  # seconds <- Make it smaller than wait_interval

    @property
    def url(self) -> str:
        """Get the server URL"""
        return f"http://localhost:{self.port}"


@dataclass(frozen=True)
class ModelConfig:
    """Dataclass to store model configuration for stress tests"""
    model_dir: str
    tp_size: int
    memory_requirement: int
    backend: Optional[str] = None

    def __str__(self) -> str:
        model_name = os.path.basename(self.model_dir)
        backend_str = f"_{self.backend}" if self.backend else ""
        return f"{model_name}_tp{self.tp_size}{backend_str}"

    @property
    def model_name(self) -> str:
        """Extract model name from model_dir for genai-perf"""
        return os.path.basename(self.model_dir)


@dataclass(frozen=True)
class StressTestConfig:
    """Dataclass to store stress test configuration"""
    model_config: ModelConfig
    server_config: ServerConfig
    # Stress test parameters for stress-test mode
    # stress_time:
    # Used as control parameter to get request count for stress test in stage3
    stress_time: int = 300  # 5 mins
    # stress_timeout:
    # Maximum time allowed for stress test to run; to prevent hanging tests
    # Must be greater than stress_time to account for initialization, warmup, etc.
    stress_timeout: int = 480  # 8 mins

    # Customized stress test parameters for stress-stage-alone mode
    customized_stress_test: bool = True
    # customized_stress_time:
    # Used as control parameter to get request count for customized stress test in stage3 alone
    customized_stress_time: int = 60  # 1 mins
    # customized_stress_timeout:
    # Maximum time allowed for customized stress test to complete
    # Must be greater than customized_stress_time to account for initialization, warmup, etc prevent run indefinitely
    customized_stress_timeout: int = 180  # 3 mins
    customized_stress_concurrency: int = 128
    customized_stress_request_rate: int = 20

    @property
    def request_count_stress_test(self) -> int:
        """Calculate request count for stress test"""
        # Cannot set exact stress time in genai-perf test, WR is set the stress_time as customized value to get request count
        stress_request_count = self.customized_stress_request_rate * self.customized_stress_time
        return stress_request_count


@dataclass(frozen=True)
class PerformanceParams:
    """Dataclass to store test parameters for genai-perf"""
    input_len_mean: int = 64  # customized for tinyllama and llama-v3-8b-instruct-hf
    input_len_std: int = 16
    output_len_mean: int = 128  # customized for tinyllama and llama-v3-8b-instruct-hf
    output_len_std: int = 32
    # test_timeout:
    # Maximum time allowed for the entire performance test to complete
    # Ensure indefinite runs specially for different concurrency values
    test_timeout: int = 3600  # 1 hours for tinyllama and llama-v3-8b-instruct-hf
    concurrency_list: List[int] = field(
        default_factory=lambda:
        [8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024])

    @property
    def request_count_list(self) -> List[int]:
        """Calculate request count based on concurrency"""
        # Keep fair amount of request count even when concurrency is low
        result = []
        for concurrency in self.concurrency_list:
            if concurrency <= 128:
                result.append(128)
            else:
                result.append(concurrency * 2)
        return result


def filter_server_output(
        pipe,
        pattern_to_exclude=r'INFO: .+ - "POST /v1/completions HTTP/1.1" 200 OK'
):
    """
    Filter function that reads from pipe and writes to stdout,
    excluding lines that match the given pattern.
    """
    pattern = re.compile(pattern_to_exclude)
    try:
        for line in iter(pipe.readline, ''):
            # Print lines that don't match the pattern
            if not pattern.search(line):
                print(line, end='', flush=True)
    except (BrokenPipeError, IOError, ValueError) as e:
        print_warning(f"Pipe error in filter_server_output: {str(e)}")


@contextlib.contextmanager
def launch_process(cmd, start_new_session=True, filter_pattern=None):
    """
    Context manager to handle process execution and filter output.

    Args:
        cmd: Command list to execute
        start_new_session: Whether to start the process in a new session
        filter_pattern: Optional regex pattern to exclude from output

    Yields:
        The process object
    """
    process = None
    stdout_reader = None
    stderr_reader = None

    try:
        # Setup pipes for stdout and stderr
        process = Popen(
            cmd,
            start_new_session=start_new_session,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # Line buffered
            universal_newlines=True)  # Text mode

        print_info(f"Process started with PID: {process.pid}")

        # Start threads to filter and process output
        if filter_pattern:
            stdout_reader = threading.Thread(
                target=filter_server_output,
                args=(process.stdout, filter_pattern),
                daemon=True  # Make sure thread doesn't block program exit
            )
            stdout_reader.start()

            stderr_reader = threading.Thread(
                target=filter_server_output,
                args=(process.stderr, filter_pattern),
                daemon=True  # Make sure thread doesn't block program exit
            )
            stderr_reader.start()

        yield process
    finally:
        if process:
            print_info(f"Stopping process with PID: {process.pid}")
            if process.poll() is None:
                # Send termination signal
                process.terminate()

                try:
                    process.wait(timeout=GRACEFUL_TERMINATION_TIMEOUT)
                    print_info("Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Process didn't exit within timeout, force kill
                    print_warning(
                        "Process did not terminate gracefully, force killing..."
                    )
                    cleanup_process_tree(process, has_session=True)
                    print_info("Process killed forcefully")

            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


def get_model_path(model_dir: str) -> str:
    """Get the full path to a model using llm_models_root"""
    return os.path.join(llm_models_root(), model_dir)


def check_server_health(server_url: str,
                        timeout: float = 10.0) -> Tuple[bool, Optional[str]]:
    """
    Check if the server is healthy by making a request to its health endpoint.

    Args:
        server_url: The base URL of the server
        timeout: Timeout in seconds for the health check request

    Returns:
        A tuple of (is_healthy, error_message)
        - is_healthy: True if server is healthy, False otherwise
        - error_message: None if healthy, error message string if not
    """
    try:
        # Increase timeout if needed
        response = requests.get(f"{server_url}/health", timeout=timeout)

        if response.status_code == 200:
            return True, None
        else:
            return False, f"Server health check failed with status code: {response.status_code}"
    except requests.RequestException as e:
        return False, f"Server health check failed: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error during health check: {str(e)}"


@pytest.mark.parametrize("test_mode", ["stress-test", "stress-stage-alone"],
                         ids=lambda x: x)
@pytest.mark.parametrize("backend", ["trt", "pytorch"], ids=lambda x: x)
@pytest.mark.parametrize("capacity_scheduler_policy",
                         ["GUARANTEED_NO_EVICT", "MAX_UTILIZATION"],
                         ids=lambda x: x)
@pytest.mark.parametrize(
    "config",
    [
        # Configuration for TinyLlama model
        ModelConfig(model_dir="llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
                    tp_size=1,
                    memory_requirement=12),
        # Configuration for Llama-v3 model
        ModelConfig(model_dir="llama-models-v3/llama-v3-8b-instruct-hf",
                    tp_size=1,
                    memory_requirement=12),
        # Configuration for DeepSeek-V3 model
        ModelConfig(model_dir="DeepSeek-V3", tp_size=8, memory_requirement=96),
    ],
    ids=lambda x: f"{os.path.basename(x.model_dir)}_tp{x.tp_size}")
def test_run_stress_test(config, backend, capacity_scheduler_policy, test_mode):
    """Run the stress test with the provided configuration, backend, and test mode.

    This test function calls the stress_test function with the given parameters.
    The function should start with test_ prefix to be recognized as a test function by pytest.

    Args:
        config: Model configuration for the test (injected by pytest.mark.parametrize)
        backend: Backend to use ("trt" or "pytorch")
        capacity_scheduler_policy: Scheduler policy ("GUARANTEED_NO_EVICT", "MAX_UTILIZATION")
        test_mode: Test mode ("stress-test" or "stress-stage-alone")
    """
    # Create a new ModelConfig with the backend parameter
    # Convert 'trt' to None as expected by the ModelConfig
    backend_param = None if backend == "trt" else backend

    new_config = ModelConfig(model_dir=config.model_dir,
                             tp_size=config.tp_size,
                             memory_requirement=config.memory_requirement,
                             backend=backend_param)

    # Initialize server config with specified capacity scheduler policy
    server_config = ServerConfig(
        capacity_scheduler_policy=capacity_scheduler_policy)

    # Call the existing stress_test function with the new config and test mode
    stress_test(new_config, test_mode, server_config)


def stress_test(config, test_mode, server_config=None):
    """Test LLM model performance using trtllm-serve and genai-perf.

    This function supports multiple testing modes controlled by the --test-mode option:
    - "stress-test": Runs the measure capacity stage first, then the stress stage,
      using the same server instance.
    - "stress-stage-alone": Performs only the stress stage with customized
      stress_concurrency and calculated request count.

    Args:
        config: Model configuration for the test (injected by pytest.mark.parametrize)
        test_mode: Test mode from the --test-mode option
            ("stress-test" or "stress-stage-alone")
        server_config: Optional server configuration to use, if None a default
            will be created
    """
    # Ensure genai-perf is installed
    # genai_perf_install()
    # Import genai-perf - needed after installation to make sure it's available
    # import genai_perf  # noqa: F401

    # Test mode handling - determine which tests to run
    if test_mode == "stress-test":
        run_performance = True
        run_stress = True
    elif test_mode == "stress-stage-alone":
        run_performance = False
        run_stress = True
    else:
        pytest.skip(f"Skipping test for unsupported mode: {test_mode}. "
                    f"Supported modes: stress-test, stress-stage-alone")
        return

    # Skip if not enough GPU memory
    if get_device_memory() < config.memory_requirement:
        pytest.skip(
            f"Not enough GPU memory. Required: {config.memory_requirement}GB")

    # Skip if not enough GPUs for tensor parallelism
    if get_device_count() < config.tp_size:
        pytest.skip(f"Not enough GPUs. Required: {config.tp_size}")

    # Get full model path
    model_path = get_model_path(config.model_dir)
    model_name = config.model_name

    # Initialize server config that will be used for all tests if not provided
    test_server_config = server_config if server_config is not None else ServerConfig(
    )

    # Define test configurations
    performance_config = PerformanceParams() if run_performance else None
    stress_config = StressTestConfig(
        model_config=config,
        server_config=test_server_config) if run_stress else None

    # Check if server is already running
    is_healthy, _ = check_server_health(test_server_config.url,
                                        test_server_config.health_check_timeout)
    if is_healthy:
        raise RuntimeError(
            f"Server is already running at {test_server_config.url}. Please stop it manually before running the stress test."
        )

    # Start server
    print_info("Starting trtllm-serve server...")
    print_info(f"Model path: {model_path}")

    # Verify that model path exists
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path does not exist: {model_path}")

    # Create a temporary YAML file for 'capacity_scheduler_policy'
    extra_llm_options = {
        "scheduler_config": {
            "capacity_scheduler_policy":
            test_server_config.capacity_scheduler_policy
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False) as temp_file:
        yaml.dump(extra_llm_options, temp_file)
        extra_llm_options_path = temp_file.name

    # Build server command
    server_cmd = [
        "trtllm-serve",
        model_path,
        "--port",
        str(test_server_config.port),
        "--host",
        test_server_config.host,
        "--tp_size",
        str(config.tp_size),
        "--pp_size",
        str(test_server_config.pp_size),
    ]

    # Only add ep_size parameter if it's not None
    if test_server_config.ep_size is not None:
        server_cmd.extend(["--ep_size", str(test_server_config.ep_size)])

    # Add remaining parameters
    server_cmd.extend([
        "--max_batch_size",
        str(test_server_config.max_batch_size),
        "--max_num_tokens",
        str(test_server_config.max_num_tokens),
        "--kv_cache_free_gpu_memory_fraction",
        str(test_server_config.kv_cache_free_gpu_memory_fraction),
        "--extra_llm_api_options",
        extra_llm_options_path,
    ])

    # Add backend option only if specified
    # backend = None means trt backend
    # backend = pytorch means pytorch backend
    if config.backend:
        server_cmd.extend(["--backend", config.backend])

    # Log the command we're about to run
    print_info(f"Running command: {' '.join(server_cmd)}")

    try:
        # Start server with the launch_process context manager and filtered output
        # HTTP access log pattern to filter out
        http_log_pattern = r'INFO: .+ - "POST /v1/completions HTTP/1.1" 200 OK'
        with launch_process(server_cmd,
                            start_new_session=True,
                            filter_pattern=http_log_pattern) as server_process:
            server_pid = server_process.pid
            print_info(f"Server started with PID: {server_pid}")

            # Wait for server to initialize
            print_info("Waiting for server to initialize...")
            server_ready = False
            for wait_sec in range(0, test_server_config.max_wait_seconds,
                                  test_server_config.wait_interval):
                deadline = time.time() + test_server_config.wait_interval
                is_healthy, error_msg = check_server_health(
                    test_server_config.url,
                    test_server_config.health_check_timeout)

                if is_healthy:
                    print_info(f"Server is ready after {wait_sec} seconds!")
                    server_ready = True
                    break
                else:
                    if wait_sec >= test_server_config.max_wait_seconds - test_server_config.wait_interval:
                        print_warning(error_msg)

                # Check if process is still running
                if server_process.poll() is not None:
                    print_warning(
                        f"ERROR: Server process died. Exit code: {server_process.returncode}"
                    )
                    try:
                        # Try to get process stderr if available
                        stderr_output = server_process.stderr.read(
                        ) if server_process.stderr else "No stderr available"
                        print_warning(f"Server stderr output: {stderr_output}")
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Server process died. Exit code: {server_process.returncode}"
                    )

                print_info(
                    f"Still waiting for server... ({wait_sec} seconds elapsed)")

                time.sleep(max(0, deadline - time.time()))

            # Final check if we didn't already confirm server is ready
            if not server_ready:
                is_healthy, error_msg = check_server_health(
                    test_server_config.url,
                    test_server_config.health_check_timeout)
                if not is_healthy:
                    print_warning(
                        f"ERROR: Server failed to start properly after {test_server_config.max_wait_seconds} seconds."
                    )
                    raise RuntimeError(f"Server failed to start: {error_msg}")

            # Run performance tests only if server is healthy
            print_info(
                f"Server is running with model {model_name}. Starting tests...")

            # Run performance test first if enabled
            stage2_output = None  # Initialize stage2_output to None
            if run_performance:
                print_info("=== Running STAGE 1 PERFORMANCE TEST ===")
                measure_capacity_stage(model_name, model_path,
                                       test_server_config, performance_config)
                print_info("=== Running STAGE 2 ANALYSIS ===")
                stage2_output = extract_stress_test_metrics(
                    current_model=model_name)
                print_info(f"Stage 2 output: {stage2_output}")
                print_info("=== Running STAGE 3 STRESS TEST ===")
                stress_stage(model_name, model_path, test_server_config,
                             stress_config, stage2_output)

            # Then run stress test if enabled (will run after performance test if both are enabled)
            if run_stress and not run_performance:  # Only run here if not already run above
                print_info(
                    "=== Running STAGE 3 STRESS TEST WITH CUSTOMIZED PARAMETERS ==="
                )
                stress_stage(model_name, model_path, test_server_config,
                             stress_config, None)
    finally:
        # Clean up temp yaml file
        if os.path.exists(extra_llm_options_path):
            os.unlink(extra_llm_options_path)


def create_genai_perf_command(model_name,
                              model_path,
                              request_count,
                              concurrency,
                              input_len_mean=PerformanceParams.input_len_mean,
                              input_len_std=PerformanceParams.input_len_std,
                              output_len_mean=PerformanceParams.output_len_mean,
                              output_len_std=PerformanceParams.output_len_std,
                              warmup_request_count=10):
    """
    Create a command list for genai-perf with standardized parameters.

    Args:
        model_name: Name of the model
        model_path: Path to the model
        request_count: Number of requests to send
        concurrency: Number of concurrent requests
        input_len_mean: Mean input length
        input_len_std: Standard deviation of input length
        output_len_mean: Mean output length
        output_len_std: Standard deviation of output length
        warmup_request_count: Number of warmup requests

    Returns:
        List of command-line arguments for genai-perf
    """
    return [
        "genai-perf",
        "profile",
        "-m",
        model_name,
        "--tokenizer",
        model_path,
        "--service-kind",
        "openai",
        "--endpoint-type",
        "completions",
        "--random-seed",
        "123",
        "--synthetic-input-tokens-mean",
        str(input_len_mean),
        "--synthetic-input-tokens-stddev",
        str(input_len_std),
        "--output-tokens-mean",
        str(output_len_mean),
        "--output-tokens-stddev",
        str(output_len_std),
        "--request-count",
        str(request_count),
        "--concurrency",
        str(concurrency),
        "--warmup-request-count",
        str(warmup_request_count),
        "--verbose",
    ]


def run_genai_perf_process(cmd, test_start_time, test_timeout, server_config):
    """
    Run a genai-perf process and monitor both the process and server health.

    Args:
        cmd: Command list to execute genai-perf
        test_start_time: Start time of the test
        test_timeout: Timeout for the test in seconds
        server_config: Server configuration object

    Returns:
        Boolean indicating whether the process completed successfully
    """
    # Start genai-perf process with our context manager
    with launch_process(cmd, start_new_session=True) as process:
        # Set monitoring parameters
        last_health_check = time.time()
        process_completed = False

        # Monitor both the server and genai-perf process
        while process.poll() is None:
            current_time = time.time()

            # Check if genai-perf is still running but exceeded timeout
            elapsed_time = current_time - test_start_time
            if elapsed_time > test_timeout:
                cleanup_process_tree(process, has_session=True)
                raise RuntimeError(
                    f"genai-perf test timed out after {test_timeout} seconds")

            # Check server health periodically
            if current_time - last_health_check > server_config.health_check_timeout:
                is_healthy, error_msg = check_server_health(
                    server_config.url, server_config.health_check_timeout)

                if is_healthy:
                    print_info(
                        f"Server health check passed after {elapsed_time:.1f} seconds of test"
                    )
                else:
                    # Raise an exception to stop the test
                    cleanup_process_tree(process, has_session=True)
                    raise RuntimeError(
                        f"Server health check failed during test: {error_msg}")

                # Update last health check time
                last_health_check = current_time

            # Short sleep to prevent CPU spinning
            time.sleep(0.5)

        # Check final status of genai-perf process
        retcode = process.poll()
        if retcode is not None:
            if retcode != 0:
                cleanup_process_tree(process, has_session=True)
                raise RuntimeError(
                    f"genai-perf exited with non-zero code: {retcode}")
            else:
                print_info("genai-perf completed successfully")
                process_completed = True
        else:
            cleanup_process_tree(process, has_session=True)
            raise RuntimeError(
                "genai-perf did not complete normally, will terminate")

    return process_completed


def measure_capacity_stage(model_name, model_path, server_config,
                           performance_params):
    """Run performance test with multiple concurrency levels"""
    total_start_time = time.time()
    total_tests = len(performance_params.concurrency_list)
    completed_tests = 0
    test_times = []

    print("Test Parameters (constant for all runs):")
    print("----------------------------------------")
    print(f"Input Length Mean: {performance_params.input_len_mean}")
    print(f"Input Length Std: {performance_params.input_len_std}")
    print(f"Output Length Mean: {performance_params.output_len_mean}")
    print(f"Output Length Std: {performance_params.output_len_std}")
    print(f"Test Timeout: {performance_params.test_timeout} seconds")
    print("----------------------------------------")

    # Iterate through concurrency levels and corresponding request counts
    for test_index, (concurrency, request_count) in enumerate(
            zip(performance_params.concurrency_list,
                performance_params.request_count_list)):
        test_start_time = time.time()

        print_info(
            f"Running test {test_index+1}/{total_tests}: concurrency={concurrency}, request_count={request_count}"
        )

        # Prepare genai-perf command
        cmd = create_genai_perf_command(
            model_name=model_name,
            model_path=model_path,
            request_count=request_count,
            concurrency=concurrency,
            input_len_mean=performance_params.input_len_mean,
            input_len_std=performance_params.input_len_std,
            output_len_mean=performance_params.output_len_mean,
            output_len_std=performance_params.output_len_std,
            warmup_request_count=10)

        # Run genai-perf process
        process_completed = run_genai_perf_process(
            cmd, test_start_time, performance_params.test_timeout,
            server_config)

        # Increment completed tests counter if the process completed successfully
        if process_completed:
            completed_tests += 1

        test_end_time = time.time()
        duration = int(test_end_time - test_start_time)
        print_info(
            f"Test {test_index+1}/{total_tests} completed in {duration} seconds"
        )
        test_times.append((concurrency, request_count, duration))

    total_time = time.time() - total_start_time

    # Print summary
    print("\n========== Performance Test Summary ==========")
    print(f"Total tests run: {total_tests}")
    print(f"Successfully completed tests: {completed_tests}")
    print(f"Total time spent: {format_time(int(total_time))}")
    print("\nDetailed test times:")
    print("Concurrency  Request Count  Time Spent")
    print("----------------------------------------")
    for concurrency, request_count, duration in test_times:
        print(
            f"{concurrency:10d}  {request_count:12d}  {format_time(duration)}")


def stress_stage(model_name,
                 model_path,
                 server_config,
                 stress_config,
                 stage2_output=None):
    """Run a single stress test with the configured parameters"""
    # Validate inputs
    if not model_name or not model_path:
        raise ValueError("model_name and model_path must be provided")

    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    # Determine stress test parameters
    if stage2_output is None:
        if stress_config.customized_stress_test:
            # Use customized parameters when stage2_output is None but customized test is enabled
            stress_concurrency = stress_config.customized_stress_concurrency
            request_count = stress_config.request_count_stress_test
            test_timeout = stress_config.customized_stress_timeout
        else:
            raise ValueError(
                "stage2_output is required when not using customized stress test"
            )
    else:
        if model_name not in stage2_output:
            raise ValueError(f"No data for model {model_name} in stage2_output")

        model_results = stage2_output[model_name]
        stress_concurrency = model_results["concurrency"]
        stress_request_rate = model_results["request_rate"]
        stress_time = stress_config.stress_time
        # Ensure request_count is an integer by using int() conversion
        request_count = int(stress_request_rate * stress_time)
        test_timeout = stress_config.stress_timeout

    print_info(
        f"Running stress test with concurrency={stress_concurrency}, request_count={request_count}"
    )

    test_start_time = time.time()

    # Prepare genai-perf command
    cmd = create_genai_perf_command(
        model_name=model_name,
        model_path=model_path,
        request_count=request_count,
        concurrency=stress_concurrency,
        input_len_mean=PerformanceParams.input_len_mean,
        input_len_std=PerformanceParams.input_len_std,
        output_len_mean=PerformanceParams.output_len_mean,
        output_len_std=PerformanceParams.output_len_std,
        warmup_request_count=10)

    # Start genai-perf process
    process_completed = run_genai_perf_process(cmd, test_start_time,
                                               test_timeout, server_config)

    test_end_time = time.time()
    duration = int(test_end_time - test_start_time)
    print_info(
        f"Stress test completed in {duration} seconds. Success: {process_completed}"
    )

    # Display summary for stress test
    if process_completed:
        print("\n========== Stress Test Summary ==========")
        print(f"Model: {model_name}")
        print(f"Concurrency: {stress_concurrency}")
        print(f"Request Count: {request_count}")
        print(f"Time Spent: {format_time(duration)}")


# Helper function to format time
def format_time(seconds: int) -> str:
    """Format time in seconds to a human-readable string"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def extract_stress_test_metrics(artifacts_dir="./artifacts",
                                current_model=None):
    """
    Extract stress test metrics from the artifacts directory

    Args:
        artifacts_dir (str): Path to the artifacts directory
        current_model (str, optional): If provided, only analyze artifacts for this model
    """
    # Find all profile_export_genai_perf.json files in the artifacts directory
    json_files = glob(os.path.join(artifacts_dir,
                                   "**/profile_export_genai_perf.json"),
                      recursive=True)

    if not json_files:
        raise RuntimeError(
            "No profile_export_genai_perf.json files found in the artifacts directory"
        )

    # Get a list of directory names in the artifacts directory
    directories = [
        d for d in os.listdir(artifacts_dir)
        if os.path.isdir(os.path.join(artifacts_dir, d))
    ]

    # Extract model names from directory names (before "-openai-completions-")
    model_name_map = {}
    for directory in directories:
        if "-openai-completions-" in directory:
            model_name = directory.split("-openai-completions-")[0]
            model_name_map[directory] = model_name
            print(f"Found model: {model_name} in directory: {directory}")

    if not model_name_map and current_model:
        raise RuntimeError(
            f"No model directories found with the expected naming pattern for model: {current_model}"
        )

    # Initialize a list to store metrics
    output_token_throughput = []
    concurrency = []
    request_throughput = []
    model_name = []

    # Process each JSON file
    for json_file in json_files:
        try:
            # Extract the directory containing the JSON file
            # Get the first directory name after artifacts_dir
            rel_path = os.path.relpath(json_file, artifacts_dir)
            first_dir = rel_path.split(os.sep)[0]

            # Skip this file if it's not from the current model we're analyzing
            if current_model and first_dir in model_name_map:
                if model_name_map[first_dir] != current_model:
                    continue

            print(f"Processing {json_file}")

            with open(json_file, "r") as f:
                results = json.load(f)

                reqThroughput = results.get("request_throughput",
                                            {}).get("avg", 0)
                tokThroughput = results.get("output_token_throughput",
                                            {}).get("avg", 0)
                conCurrency = results.get("input_config",
                                          {}).get("concurrency", 0)

                # Try to determine model name from directory structure first
                if first_dir in model_name_map:
                    modelName = model_name_map[first_dir]
                else:
                    # Fall back to model name from JSON if we can't extract from directory
                    modelName = results.get("input_config",
                                            {}).get("model", ["unknown"])
                    modelName = modelName[0] if isinstance(modelName,
                                                           list) else modelName

            # Check that values are valid before appending
            if reqThroughput and tokThroughput and conCurrency and modelName:
                request_throughput.append(reqThroughput)
                output_token_throughput.append(tokThroughput)
                concurrency.append(conCurrency)
                model_name.append(modelName)
            else:
                raise ValueError(
                    f"Please check {json_file} due to missing or invalid metrics"
                )

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Check if we have any valid data
    if not model_name:
        if current_model:
            raise RuntimeError(
                f"No valid data extracted for model: {current_model}")
        else:
            raise RuntimeError("No valid data extracted from the JSON files")

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        "Model": model_name,
        "Concurrency": concurrency,
        "RequestThroughput": request_throughput,
        "OutputTokenThroughput": output_token_throughput
    })

    # Sort by Model, Concurrency, and RequestThroughput
    metrics_df = metrics_df.sort_values(
        by=["Model", "Concurrency", "RequestThroughput"])

    print("\n========== Stress Test Metrics Summary ==========")
    print(metrics_df.to_string(index=False))

    # Define the high performance threshold
    throughput_threshold = 0.5  # value range [0,1], 0.95 maybe too high, suggest use 0.5
    concurrency_no_gain_count_threshold = 5  # change this value per different model and throughput threshold
    high_perf_results = {}

    # Calculate normalized throughput for each model
    normalized_df = metrics_df.copy()

    for model_name in normalized_df["Model"].unique():
        # Get min and max values from model
        model_data = normalized_df[normalized_df["Model"] == model_name]
        min_val = model_data["OutputTokenThroughput"].min()
        max_val = model_data["OutputTokenThroughput"].max()

        range_val = max_val - min_val
        if range_val == 0:
            raise ValueError(
                "Please check OutputTokenThroughput from genai-perf")
        else:
            normalized_df.loc[
                normalized_df["Model"] == model_name,
                "NormalizedThroughput"] = (
                    (normalized_df.loc[normalized_df["Model"] == model_name,
                                       "OutputTokenThroughput"] - min_val) /
                    range_val)

        # Find rows where normalized throughput exceeds threshold
        high_perf_rows = normalized_df[(normalized_df["Model"] == model_name)
                                       & (normalized_df["NormalizedThroughput"]
                                          > throughput_threshold)].sort_values(
                                              by="Concurrency")

        high_perf_indices = high_perf_rows.index.tolist()

        # Rule setup to get the highest throughput point
        if len(high_perf_rows) >= concurrency_no_gain_count_threshold:
            optimized_idx = high_perf_indices[
                -concurrency_no_gain_count_threshold]
            optimized_row = normalized_df.loc[optimized_idx]

            high_perf_results[model_name] = {
                "concurrency":
                int(optimized_row["Concurrency"]),
                "normalized_throughput":
                float(optimized_row["NormalizedThroughput"]),
                "throughput":
                float(optimized_row["OutputTokenThroughput"]),
                "request_rate":
                float(optimized_row["RequestThroughput"])
            }
        elif len(high_perf_rows) > 0:
            optimized_idx = high_perf_indices[0]
            optimized_row = normalized_df.loc[optimized_idx]

            high_perf_results[model_name] = {
                "concurrency":
                int(optimized_row["Concurrency"]),
                "normalized_throughput":
                float(optimized_row["NormalizedThroughput"]),
                "throughput":
                float(optimized_row["OutputTokenThroughput"]),
                "request_rate":
                float(optimized_row["RequestThroughput"]),
                "note":
                f"Only {len(high_perf_indices)} values above threshold {throughput_threshold}, using the first one"
            }
        else:
            raise ValueError(
                f"No high performance point found for {model_name}")

    # Print the normalized results
    print(
        "\n========== Normalized Token Throughput by Model and Concurrency =========="
    )
    print(normalized_df[[
        "Model", "Concurrency", "OutputTokenThroughput", "NormalizedThroughput"
    ]].to_string(index=False))
    # Print the high performance results
    print(
        f"\n========== High Performance Results (Threshold = {throughput_threshold}) =========="
    )
    for model, results in high_perf_results.items():
        print(f"\nModel: {model}")
        for key, value in results.items():
            print(f"{key}: {value}")

    # Return the high performance concurrency values to potentially use for stress testing
    return high_perf_results
