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
Stress test script for inference of model using TensorRT LLM with PyTorch/TRT backend.
This script is used for stress testing inference performance using trtllm-serve and genai-perf.

The script supports three test modes:
1. "stress-test": Runs performance test followed by stress test
2. "stress-stage-alone": Runs only stress test with customized parameters
3. "stress-test-with-accuracy": Runs performance test, stress test, and accuracy tests (GSM8K)

Accuracy testing is performed using lm_eval with GSM8K dataset:
- Baseline accuracy test: Run before stress test to establish baseline
- Post-stress accuracy test: Run after stress test to verify accuracy stability

Usage example for accuracy testing:
    pytest tests/integration/defs/stress_test/stress_test.py::test_run_stress_test[stress-test-with-accuracy]
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
GRACEFUL_TERMINATION_TIMEOUT = 300  # seconds - set longer when stress large model


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
    stress_time: int = 180  # 3 mins default, can be overridden
    # stress_timeout:
    # Maximum time allowed for stress test to run; to prevent hanging tests
    # Must be greater than stress_time to account for initialization, warmup, etc.
    stress_timeout: int = 300  # 5 mins default, can be overridden

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

    # Accuracy test parameters
    enable_accuracy_test: bool = False  # Enable accuracy testing with GSM8K
    accuracy_test_timeout: int = 1200  # 20 minutes timeout for accuracy tests
    accuracy_test_concurrency: int = 512  # Concurrency for accuracy tests
    accuracy_test_max_retries: int = 3  # Max retries for accuracy tests
    accuracy_test_max_gen_toks: int = 256  # Max generation tokens for accuracy tests
    accuracy_test_max_length: int = 4096  # Max input length for accuracy tests

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


class RequestCounter:
    """Thread-safe counter for tracking completion requests"""

    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.count += 1

    def get_count(self):
        with self.lock:
            return self.count

    def reset(self):
        with self.lock:
            self.count = 0


def filter_server_output(
        pipe,
        pattern_to_exclude=r'INFO: .+ - "POST /v1/completions HTTP/1.1" 200 OK',
        counter=None):
    """
    Filter function that reads from pipe and writes to stdout,
    excluding lines that match the given pattern.

    If a counter is provided, counts occurrences of the pattern.
    """
    pattern = re.compile(pattern_to_exclude)
    try:
        for line in iter(pipe.readline, ''):
            # Count matches if counter is provided
            if counter is not None and pattern.search(line):
                counter.increment()

            # Print lines that don't match the pattern
            if not pattern.search(line):
                print(line, end='', flush=True)
    except (BrokenPipeError, IOError, ValueError) as e:
        print_warning(f"Pipe error in filter_server_output: {str(e)}")


@contextlib.contextmanager
def launch_process(cmd,
                   start_new_session=True,
                   filter_pattern=None,
                   request_counter=None):
    """
    Context manager to handle process execution and filter output.

    Args:
        cmd: Command list to execute
        start_new_session: Whether to start the process in a new session
        filter_pattern: Optional regex pattern to exclude from output
        request_counter: Optional counter to track requests

    Yields:
        The process object
    """
    process = None
    stdout_reader = None
    stderr_reader = None

    try:
        # Only create pipes if we plan to filter output
        if filter_pattern:
            process = Popen(
                cmd,
                start_new_session=start_new_session,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line buffered
                universal_newlines=True)  # Text mode

            print_info(f"Process started with PID: {process.pid}")

            # Start threads to filter and process output
            stdout_reader = threading.Thread(
                target=filter_server_output,
                args=(process.stdout, filter_pattern, request_counter),
                daemon=True  # Make sure thread doesn't block program exit
            )
            stdout_reader.start()

            stderr_reader = threading.Thread(
                target=filter_server_output,
                args=(process.stderr, filter_pattern, request_counter),
                daemon=True  # Make sure thread doesn't block program exit
            )
            stderr_reader.start()
        else:
            process = Popen(cmd,
                            start_new_session=start_new_session,
                            stdout=None,
                            stderr=None)

            print_info(f"Process started with PID: {process.pid}")

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


@pytest.mark.parametrize(
    "test_mode",
    ["stress-test", "stress-stage-alone", "stress-test-with-accuracy"],
    ids=lambda x: x)
@pytest.mark.parametrize("backend", ["trt", "pytorch"], ids=lambda x: x)
@pytest.mark.parametrize("capacity_scheduler_policy",
                         ["GUARANTEED_NO_EVICT", "MAX_UTILIZATION"],
                         ids=lambda x: x)
@pytest.mark.parametrize("stress_time_timeout", [(180, 300), (300, 450),
                                                 (600, 900), (3600, 5400)],
                         ids=lambda x: f"stress_time_{x[0]}s_timeout_{x[1]}s")
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
        # Configuration for DeepSeek-R1 model
        ModelConfig(model_dir="DeepSeek-R1/DeepSeek-R1",
                    tp_size=8,
                    memory_requirement=96),
    ],
    ids=lambda x: f"{os.path.basename(x.model_dir)}_tp{x.tp_size}")
def test_run_stress_test(config, stress_time_timeout, backend,
                         capacity_scheduler_policy, test_mode):
    """Run the stress test with the provided configuration, backend, and test mode.

    This test function calls the stress_test function with the given parameters.
    The function should start with test_ prefix to be recognized as a test function by pytest.

    Args:
        config: Model configuration for the test (injected by pytest.mark.parametrize)
        stress_time_timeout: Tuple of (stress_time, stress_timeout) in seconds
        backend: Backend to use ("trt" or "pytorch")
        capacity_scheduler_policy: Scheduler policy ("GUARANTEED_NO_EVICT", "MAX_UTILIZATION")
        test_mode: Test mode ("stress-test" or "stress-stage-alone")
    """
    # Create a new ModelConfig with the backend parameter
    # Convert 'trt' to None as expected by the ModelConfig

    new_config = ModelConfig(model_dir=config.model_dir,
                             tp_size=config.tp_size,
                             memory_requirement=config.memory_requirement,
                             backend=backend)

    # Extract stress_time and stress_timeout from the tuple
    stress_time, stress_timeout = stress_time_timeout

    # Initialize server config with specified capacity scheduler policy
    server_config = ServerConfig(
        capacity_scheduler_policy=capacity_scheduler_policy)

    # Call the existing stress_test function with the new config and test mode
    stress_test(new_config, test_mode, server_config, stress_time,
                stress_timeout)


def stress_test(config,
                test_mode,
                server_config=None,
                stress_time=None,
                stress_timeout=None):
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
        stress_time: Optional stress time in seconds, overrides the default in StressTestConfig
        stress_timeout: Optional stress timeout in seconds, overrides the default in StressTestConfig
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
    elif test_mode == "stress-test-with-accuracy":
        run_performance = True
        run_stress = True
    else:
        pytest.skip(
            f"Skipping test for unsupported mode: {test_mode}. "
            f"Supported modes: stress-test, stress-stage-alone, stress-test-with-accuracy"
        )
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
    performance_config = None
    if run_performance:
        performance_config = PerformanceParams()

        # For DeepSeek-V3 or DeepSeek-R1 specific parameters
        if "DeepSeek-V3" in config.model_dir or "DeepSeek-R1" in config.model_dir:
            performance_config = PerformanceParams(
                test_timeout=
                36000  # 10 hours for DeepSeek-V3 or DeepSeek-R1, change this value if needed
            )

    # For DeepSeek-V3 specific server parameters
    if "DeepSeek-V3" in config.model_dir or "DeepSeek-R1" in config.model_dir:
        test_server_config = ServerConfig(
            port=test_server_config.port,
            host=test_server_config.host,
            pp_size=test_server_config.pp_size,
            ep_size=8,  # DeepSeek-V3 or DeepSeek-R1 specific ep_size
            max_batch_size=
            2048,  # DeepSeek-V3 or DeepSeek-R1 specific max_batch_size
            max_num_tokens=
            2048,  # DeepSeek-V3 or DeepSeek-R1 specific max_num_tokens
            kv_cache_free_gpu_memory_fraction=
            0.7,  # DeepSeek-V3 or DeepSeek-R1 specific kv_cache fraction
            capacity_scheduler_policy=test_server_config.
            capacity_scheduler_policy,
            wait_interval=test_server_config.wait_interval,
            max_wait_seconds=
            28800,  # DeepSeek-V3 or DeepSeek-R1 specific wait time (8 hours)
            health_check_timeout=test_server_config.health_check_timeout)

    # Create a StressTestConfig with customized time parameters if provided
    if run_stress:
        # Enable accuracy test for stress-test-with-accuracy mode
        enable_accuracy = (test_mode == "stress-test-with-accuracy")

        stress_config = StressTestConfig(model_config=config,
                                         server_config=test_server_config,
                                         enable_accuracy_test=enable_accuracy)

        # Override stress_time and stress_timeout if provided
        if stress_time is not None:
            stress_config = StressTestConfig(
                model_config=config,
                server_config=test_server_config,
                stress_time=stress_time,
                stress_timeout=stress_timeout
                if stress_timeout is not None else stress_time * 2,
                enable_accuracy_test=enable_accuracy)
    else:
        stress_config = None

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

    # Create a temporary YAML file for extra_llm_options
    extra_llm_options = {
        "scheduler_config": {
            "capacity_scheduler_policy":
            test_server_config.capacity_scheduler_policy
        },
    }

    # Add DeepSeek-V3 or DeepSeek-R1 specific configuration
    if "DeepSeek-V3" in config.model_dir or "DeepSeek-R1" in config.model_dir:

        extra_llm_options["enable_attention_dp"] = True

        if config.backend == "pytorch":
            extra_llm_options.update({
                "cuda_graph_config": {
                    "enable_padding": True,
                    "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 384],
                },
                "print_iter_log": True,
            })

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
        "--backend",
        config.backend,
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

    # Log the command we're about to run
    print_info(f"Running command: {' '.join(server_cmd)}")

    try:
        # Create a request counter to track completions
        request_counter = RequestCounter()

        # Start server with the launch_process context manager and filtered output
        # HTTP access log pattern to filter out
        http_log_pattern = r'INFO: .+ - "POST /v1/completions HTTP/1.1" 200 OK'
        with launch_process(server_cmd,
                            start_new_session=True,
                            filter_pattern=http_log_pattern,
                            request_counter=request_counter) as server_process:
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

            # Run baseline accuracy test first if enabled
            baseline_accuracy_success = True
            if stress_config and stress_config.enable_accuracy_test:
                baseline_accuracy_success, baseline_accuracy_value = run_accuracy_test(
                    model_path, test_server_config, stress_config, "baseline")

            # Run performance test first if enabled
            stage2_output = None  # Initialize stage2_output to None
            if run_performance:
                print_info("=== Running STAGE 1 PERFORMANCE TEST ===")
                measure_capacity_stage(model_name,
                                       model_path,
                                       test_server_config,
                                       performance_config,
                                       request_counter=request_counter)
                print_info("=== Running STAGE 2 ANALYSIS ===")
                stage2_output = extract_stress_test_metrics(
                    current_model=model_name)
                print_info(f"Stage 2 output: {stage2_output}")
                print_info("=== Running STAGE 3 STRESS TEST ===")
                stress_stage(model_name,
                             model_path,
                             test_server_config,
                             stress_config,
                             stage2_output,
                             request_counter=request_counter)

            # Then run stress test if enabled (will run after performance test if both are enabled)
            if run_stress and not run_performance:  # Only run here if not already run above
                print_info(
                    "=== Running STAGE 3 STRESS TEST WITH CUSTOMIZED PARAMETERS ==="
                )
                stress_stage(model_name,
                             model_path,
                             test_server_config,
                             stress_config,
                             None,
                             request_counter=request_counter)

            # Run post-stress accuracy test if enabled
            post_stress_accuracy_success = True
            if stress_config and stress_config.enable_accuracy_test:
                post_stress_accuracy_success, post_stress_accuracy_value = run_accuracy_test(
                    model_path, test_server_config, stress_config,
                    "post_stress")

                # Report accuracy test results
                if baseline_accuracy_success and post_stress_accuracy_success:
                    print_info("=== ACCURACY TEST SUMMARY ===")
                    print_info("✓ Baseline accuracy test: PASSED")
                    print_info("✓ Post-stress accuracy test: PASSED")

                    # Compare accuracy values if both are available
                    if baseline_accuracy_value is not None and post_stress_accuracy_value is not None:
                        accuracy_drop = baseline_accuracy_value - post_stress_accuracy_value
                        accuracy_drop_percentage = (
                            accuracy_drop / baseline_accuracy_value) * 100

                        print_info(
                            f"Baseline accuracy: {baseline_accuracy_value:.4f}")
                        print_info(
                            f"Post-stress accuracy: {post_stress_accuracy_value:.4f}"
                        )
                        print_info(
                            f"Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop_percentage:.2f}%)"
                        )

                        # Define threshold for significant accuracy drop (e.g., 5%)
                        accuracy_drop_threshold = 0.05  # 5%
                        # Assert that accuracy drop is within acceptable threshold
                        assert accuracy_drop_percentage <= (
                            accuracy_drop_threshold * 100
                        ), f"Accuracy drop {accuracy_drop_percentage:.2f}% exceeds threshold {accuracy_drop_threshold * 100}%"
                        print_info(
                            "✓ Model accuracy appears stable under stress conditions"
                        )
                else:
                    print_warning("=== ACCURACY TEST SUMMARY ===")
                    if not baseline_accuracy_success:
                        print_warning("✗ Baseline accuracy test: FAILED")
                    if not post_stress_accuracy_success:
                        print_warning("✗ Post-stress accuracy test: FAILED")
                    print_warning(
                        "Model accuracy may be affected by stress conditions")
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


def run_genai_perf_process(cmd,
                           test_start_time,
                           test_timeout,
                           server_config,
                           request_counter=None):
    """
    Run a genai-perf process and monitor both the process and server health.

    Args:
        cmd: Command list to execute genai-perf
        test_start_time: Start time of the test
        test_timeout: Timeout for the test in seconds
        server_config: Server configuration object
        request_counter: Optional counter to track requests

    Returns:
        Boolean indicating whether the process completed successfully
    """
    # Start genai-perf process with our context manager
    with launch_process(cmd,
                        start_new_session=True,
                        filter_pattern=None,
                        request_counter=request_counter) as process:
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
                    print_warning(f"Server health check failed: {error_msg}")
                    cleanup_process_tree(process, has_session=True)
                    raise RuntimeError(
                        f"Server health check failed during test: {error_msg}")

                # Update last health check time
                last_health_check = current_time

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


def measure_capacity_stage(model_name,
                           model_path,
                           server_config,
                           performance_params,
                           request_counter=None):
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

    # Reset the counter before starting tests
    if request_counter:
        request_counter.reset()

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
            server_config, request_counter)

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

    if request_counter:
        print(
            f"Total successful completion requests: {request_counter.get_count()}"
        )


def stress_stage(model_name,
                 model_path,
                 server_config,
                 stress_config,
                 stage2_output=None,
                 request_counter=None):
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

    # Reset the counter before starting the stress test
    if request_counter:
        request_counter.reset()

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
                                               test_timeout, server_config,
                                               request_counter)

    test_end_time = time.time()
    duration = int(test_end_time - test_start_time)

    # Now print the counter results after the test has completed
    if request_counter:
        print(
            f"Total successful completion requests: {request_counter.get_count()}"
        )

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


def parse_accuracy_from_lm_eval_output(output_text: str) -> float:
    """
    Parse accuracy value from lm_eval output for GSM8K flexible-extract exact_match

    Args:
        output_text: The output text from lm_eval command

    Returns:
        float: The accuracy value (0.7582 in the example)
    """
    import re

    # Look for the specific pattern: |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7559|±  |0.0118|
    patterns = [
        r'flexible-extract\|\s+\d+\|exact_match\|\↑\s+\|(\d+\.\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            accuracy_value = float(match.group(1))
            print_info(f"Extracted accuracy value: {accuracy_value}")
            return accuracy_value

    print_warning("Could not find accuracy value in lm_eval output")
    print_warning(f"Output text: {output_text}")
    return None


def run_accuracy_test(model_path: str,
                      server_config: ServerConfig,
                      stress_config: StressTestConfig,
                      test_phase: str = "baseline") -> tuple[bool, float]:
    """
    Run accuracy test using lm_eval with GSM8K dataset

    Args:
        model_path: Path of the model being tested
        server_config: Server configuration containing URL and port
        stress_config: Stress test configuration containing accuracy test parameters
        test_phase: Phase of the test ("baseline" or "post_stress")

    Returns:
        tuple: (Boolean indicating whether the accuracy test completed successfully, accuracy value)
    """
    if not stress_config.enable_accuracy_test:
        print_info(f"Skipping accuracy test for {test_phase} phase (disabled)")
        return True, None

    print_info(f"=== Running {test_phase.upper()} ACCURACY TEST (GSM8K) ===")

    # Create lm_eval command
    lm_eval_cmd = [
        "lm_eval", "--model", "local-completions", "--tasks", "gsm8k",
        "--model_args",
        f"model={model_path},base_url={server_config.url}/v1/completions,"
        f"num_concurrent={stress_config.accuracy_test_concurrency},"
        f"max_retries={stress_config.accuracy_test_max_retries},"
        f"tokenized_requests=False,"
        f"timeout={stress_config.accuracy_test_timeout},"
        f"max_gen_toks={stress_config.accuracy_test_max_gen_toks},"
        f"max_length={stress_config.accuracy_test_max_length}",
        "--trust_remote_code"
    ]

    test_start_time = time.time()
    accuracy_value = None

    try:
        # Run lm_eval process with timeout monitoring
        print_info(f"Running lm_eval command: {' '.join(lm_eval_cmd)}")

        # Use subprocess.run to capture output directly
        result = subprocess.run(lm_eval_cmd,
                                capture_output=True,
                                text=True,
                                timeout=stress_config.accuracy_test_timeout)

        # Check if process completed successfully
        if result.returncode == 0:
            test_end_time = time.time()
            duration = int(test_end_time - test_start_time)
            print_info(
                f"{test_phase.capitalize()} accuracy test completed successfully in {format_time(duration)}"
            )

            # Parse accuracy value from output
            output_text = result.stdout
            accuracy_value = parse_accuracy_from_lm_eval_output(output_text)
            return True, accuracy_value
        else:
            print_warning(
                f"lm_eval exited with non-zero code: {result.returncode}")
            print_warning(f"stderr: {result.stderr}")
            return False, None

    except subprocess.TimeoutExpired:
        print_warning(
            f"Accuracy test timed out after {stress_config.accuracy_test_timeout} seconds"
        )
        return False, None
    except Exception as e:
        print_warning(f"Error during {test_phase} accuracy test: {str(e)}")
        return False, None


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
                conCurrency = results.get("input_config", {}).get(
                    "perf_analyzer", {}).get("stimulus",
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
