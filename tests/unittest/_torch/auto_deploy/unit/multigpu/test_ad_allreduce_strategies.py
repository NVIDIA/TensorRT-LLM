import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

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
def shared_dataset(llm_root):
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
        "ONESHOT",
        "TWOSHOT",
        "MIN_LATENCY",
        "NCCL",
    ],
)
def test_allreduce_strategies(llm_root, shared_dataset, allreduce_strategy):
    """Test different allreduce strategies with multi-GPU configuration.

    This test validates that all allreduce strategies work correctly with TP=4.
    Note: TWOSHOT strategy will automatically fall back to ONESHOT when sequence
    length is smaller than TP size during initialization.

    Test has a 300 second timeout to prevent indefinite hangs.

    Args:
        llm_root: Root directory fixture
        shared_dataset: Shared dataset fixture (prepared once for all test runs)
        allreduce_strategy: Strategy to test (AUTO, ONESHOT, TWOSHOT, MIN_LATENCY, NCCL)
    """
    # Fixed timeout for all strategies (5 minutes should be enough)
    TEST_TIMEOUT_SECONDS = 300

    model_name = "meta-llama/Llama-3.1-8B"
    tp_size = 4
    max_batch_size = 256
    max_num_tokens = 8192

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write shared dataset to temp location
        dataset_path = Path(temp_dir, "synthetic_128_128.txt")
        with open(dataset_path, "w") as f:
            f.write(shared_dataset)

        # Create configuration with specified allreduce strategy
        extra_llm_api_options_path = f"{temp_dir}/extra_llm_api_options.yaml"
        with open(extra_llm_api_options_path, "w") as f:
            yaml.dump(
                {
                    "model": model_name,
                    "allreduce_strategy": allreduce_strategy,
                    "max_batch_size": max_batch_size,
                    "max_num_tokens": max_num_tokens,
                    "max_seq_len": 256,
                    "transforms": {
                        "compile_model": {
                            "stage": "compile",
                            "backend": "torch-cudagraph",
                            "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256],
                        }
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
