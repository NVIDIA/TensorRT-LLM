import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import yaml
from click.testing import CliRunner
from utils.cpp_paths import llm_root  # noqa: F401

from tensorrt_llm.commands.bench import main


class TimeoutError(Exception):
    """Exception raised when a test times out."""

    pass


@contextmanager
def timeout(seconds):
    """Context manager that raises TimeoutError if code block exceeds time limit.

    Args:
        seconds: Maximum time in seconds to allow the code block to run

    Raises:
        TimeoutError: If the code block execution exceeds the time limit
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Test execution exceeded {seconds} seconds timeout")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.fixture(scope="module")
def shared_dataset(llm_root):  # noqa: F811
    """Prepare dataset once for all tests in this module."""
    model_name = "meta-llama/Llama-3.1-8B"
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = _prepare_dataset(llm_root, temp_dir, model_name, num_requests=10)
        # Read dataset content to return it (temp_dir will be deleted)
        with open(dataset_path, "r") as f:
            dataset_content = f.read()
        yield dataset_content


def _prepare_dataset(root_dir: str, temp_dir: str, model_path_or_name: str, num_requests: int = 10):
    """Prepare a synthetic dataset for benchmarking."""
    _DATASET_NAME = "synthetic_128_128.txt"
    dataset_path = Path(temp_dir, _DATASET_NAME)
    dataset_tool = Path(root_dir, "benchmarks", "cpp", "prepare_dataset.py")
    script_dir = Path(root_dir, "benchmarks", "cpp")

    # Generate a small dataset to run a test - matching workload configuration
    command = [
        "python3",
        f"{dataset_tool}",
        "--stdout",
        "--tokenizer",
        model_path_or_name,
        "token-norm-dist",
        "--input-mean",
        "128",
        "--output-mean",
        "128",
        "--input-stdev",
        "0",
        "--output-stdev",
        "0",
        "--num-requests",
        str(num_requests),
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(
        command, cwd=str(script_dir), capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare dataset: {result.stderr}")
    # Grab the stdout and write it to a dataset file for passing to suite.
    with open(dataset_path, "w") as dataset:
        dataset.write(result.stdout)
    return dataset_path


@pytest.mark.parametrize(
    "allreduce_strategy",
    [
        "AUTO",
        "NCCL",
        "ONESHOT",
        pytest.param(
            "TWOSHOT",
            marks=pytest.mark.skip(
                reason="TWOSHOT requires C++ fix for seq_len < tp_size fallback"
            ),
        ),
    ],
)
def test_allreduce_strategies(llm_root, shared_dataset, allreduce_strategy):  # noqa: F811
    """Test all AllReduceStrategy enum values with multi-GPU configuration.

    This test validates that all allreduce strategies defined in the AllReduceStrategy
    enum work correctly with TP=2. The strategy is configured via the detect_sharding
    transform config and automatically applied as a global variable.


    Test has a 300 second timeout to prevent indefinite hangs.
    Test will be skipped if fewer than 2 GPUs are available.

    Args:
        llm_root: Root directory fixture
        shared_dataset: Shared dataset fixture (prepared once for all test runs)
        allreduce_strategy: Strategy to test (AllReduceStrategy enum values)
    """
    # Fixed timeout for all strategies (5 minutes should be enough)
    TEST_TIMEOUT_SECONDS = 300

    model_name = "meta-llama/Llama-3.1-8B"
    tp_size = 2
    max_batch_size = 256
    max_num_tokens = 8192

    if not torch.cuda.is_available() or torch.cuda.device_count() < tp_size:
        pytest.skip(f"Allreduce strategy test requires at least {tp_size} GPUs, skipping")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write shared dataset to temp location
        dataset_path = Path(temp_dir, "synthetic_128_128.txt")
        with open(dataset_path, "w") as f:
            f.write(shared_dataset)

        # Create configuration with allreduce strategy in transform config
        extra_llm_api_options_path = f"{temp_dir}/extra_llm_api_options.yaml"
        with open(extra_llm_api_options_path, "w") as f:
            yaml.dump(
                {
                    "model": model_name,
                    "max_batch_size": max_batch_size,
                    "max_num_tokens": max_num_tokens,
                    "max_seq_len": 256,
                    "transforms": {
                        "detect_sharding": {
                            "stage": "sharding",
                            "allreduce_strategy": allreduce_strategy,
                        },
                        "compile_model": {
                            "stage": "compile",
                            "backend": "torch-cudagraph",
                            "cuda_graph_batch_sizes": [1, 16, 256],
                        },
                    },
                },
                f,
            )

        # Run benchmark with specified allreduce strategy with timeout protection
        runner = CliRunner()
        args = [
            "--model",
            model_name,
            "throughput",
            "--backend",
            "_autodeploy",
            "--dataset",
            str(dataset_path),
            "--extra_llm_api_options",
            extra_llm_api_options_path,
            "--tp",
            str(tp_size),
            "--max_batch_size",
            str(max_batch_size),
            "--max_num_tokens",
            str(max_num_tokens),
        ]

        try:
            with timeout(TEST_TIMEOUT_SECONDS):
                result = runner.invoke(main, args, catch_exceptions=False)
                assert result.exit_code == 0, f"Benchmark failed with output: {result.output}"
        except TimeoutError as e:
            pytest.fail(
                f"Test timed out after {TEST_TIMEOUT_SECONDS}s for strategy {allreduce_strategy}. "
                f"This might indicate a hang (e.g., TWOSHOT without C++ fix). Error: {e}"
            )
