import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from _model_test_utils import get_small_model_config
from click.testing import CliRunner
from utils.cpp_paths import llm_root  # noqa: F401

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    ShardingTransformConfig,
    ShardingTransformContainer,
    SplitDimension,
    WeightShardingInfo,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import recompile
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm.commands.bench import main
from tensorrt_llm.functional import AllReduceStrategy

# needed since LLM API uses MPI executor pool internally for TP>1, which leaks a thread on shutdown
pytestmark = pytest.mark.threadleak(enabled=False)


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


@pytest.fixture(scope="module", autouse=True)
def prewarm_flashinfer_jit():
    """Pre-warm FlashInfer JIT kernels before multi-GPU tests.

    This prevents a race condition where multiple MPI ranks try to JIT-compile
    FlashInfer kernels simultaneously to the same cache directory, causing
    Ninja build failures like: "ninja: error: opening build log: No such file or directory"

    By triggering the compilation in the main process first, the kernels are
    cached and available for all worker ranks.
    """
    try:
        import flashinfer
        import flashinfer.page
        import flashinfer.sampling

        if torch.cuda.is_available():
            # Prevent concurrent JIT warmup across multiple pytest processes (e.g., xdist).
            try:
                import fcntl  # Linux-only
            except ImportError:
                fcntl = None

            lock_f = None
            if fcntl is not None:
                import pathlib
                import tempfile

                lock_path = pathlib.Path(tempfile.gettempdir()) / "flashinfer_jit_prewarm.lock"
                lock_f = open(lock_path, "w")
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            # Create dummy tensors to trigger kernel JIT compilation
            with torch.no_grad():
                device = torch.device("cuda:0")

                # Trigger page kernel compilation
                try:
                    # Force module loading (this triggers JIT compilation)
                    _ = flashinfer.page.gen_page_module()
                except Exception as exc:  # noqa: BLE001
                    import warnings

                    warnings.warn(f"FlashInfer page-kernel prewarm failed: {exc!r}", RuntimeWarning)

                # Trigger sampling kernel compilation
                try:
                    dummy_probs = torch.softmax(torch.randn(1, 100, device=device), dim=-1)
                    _ = flashinfer.sampling.sampling_from_probs(dummy_probs, deterministic=True)
                except Exception as exc:  # noqa: BLE001
                    import warnings

                    warnings.warn(
                        f"FlashInfer sampling-kernel prewarm failed: {exc!r}", RuntimeWarning
                    )

                torch.cuda.empty_cache()
            if lock_f is not None:
                lock_f.close()

    except ImportError:
        pass  # FlashInfer not available

    yield


@pytest.fixture(scope="module")
def shared_dataset(llm_root):  # noqa: F811
    """Prepare dataset once for all tests in this module."""
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    config = get_small_model_config(model_name)
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = _prepare_dataset(
            llm_root, temp_dir, config["args"]["model"], num_requests=10
        )
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
        "SYMM_MEM",
    ],
)
def test_allreduce_strategies(llm_root, shared_dataset, allreduce_strategy):  # noqa: F811
    """Test different allreduce strategies with multi-GPU configuration making sure that there are no crashes or hangs.

    Configuration:
        The allreduce_strategy is set in the transforms config:
        ```yaml
        transforms:
          detect_sharding:
            allreduce_strategy: "ONESHOT"  # or AUTO, NCCL, TWOSHOT, etc.
        ```

    Test configuration:
        - Model: Llama-3.1-8B with TP=2
        - Dataset: 10 synthetic requests (128 input, 128 output tokens)
        - Timeout: 300 seconds to catch hangs
        - Skipped if fewer than 2 GPUs available

    Args:
        llm_root: Root directory fixture
        shared_dataset: Shared dataset fixture (prepared once for all test runs)
        allreduce_strategy: Strategy to test (AUTO, ONESHOT, TWOSHOT, MIN_LATENCY, NCCL)
    """
    # Fixed timeout for all strategies (5 minutes should be enough)
    TEST_TIMEOUT_SECONDS = 300

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    config = get_small_model_config(model_name)
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

        # Create configuration with specified allreduce strategy in transforms
        extra_llm_api_options_path = f"{temp_dir}/extra_llm_api_options.yaml"
        with open(extra_llm_api_options_path, "w") as f:
            yaml.dump(
                {
                    **config["args"],
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
                            "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256],
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
        ]

        # Only pass --model_path if it's a local filesystem path
        # Note: --model_path must come BEFORE the subcommand (throughput)
        if str(config["args"]["model"]).startswith("/"):
            args.extend(["--model_path", str(config["args"]["model"])])

        # Add the subcommand and its options
        args.extend(
            [
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
        )

        try:
            with timeout(TEST_TIMEOUT_SECONDS):
                result = runner.invoke(main, args, catch_exceptions=False)
                assert result.exit_code == 0, f"Benchmark failed with output: {result.output}"
        except TimeoutError as e:
            pytest.fail(
                f"Test timed out after {TEST_TIMEOUT_SECONDS}s for strategy {allreduce_strategy}. "
                f"This might indicate a hang (e.g., TWOSHOT without C++ fix). Error: {e}"
            )


@pytest.mark.parametrize(
    "strategy",
    [
        "AUTO",
        "NCCL",
        "TWOSHOT",
        "MIN_LATENCY",
        "SYMM_MEM",
    ],
)
def test_allreduce_strategy_propagation(strategy):
    """Test that allreduce_strategy is correctly propagated to graph nodes.

    This test verifies that when we set an allreduce_strategy on the ShardingConfig,
    it gets properly injected into the transforms and passed to the torch_dist_all_reduce
    nodes in the compiled graph.
    """

    # Create a simple MLP model
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 256, bias=False)
            self.linear2 = nn.Linear(256, 128, bias=False)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    model = SimpleMLP()
    dummy_input = torch.randn(2, 128)

    # Export to graph
    gm = torch_export_to_gm(model, (dummy_input,))

    # Find linear nodes in the graph
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op

    linear_nodes = [node for node in gm.graph.nodes if is_linear_op(node)]
    assert len(linear_nodes) == 2, f"Expected 2 linear nodes, found {len(linear_nodes)}"

    linear1_node, linear2_node = linear_nodes[0], linear_nodes[1]

    # Create sharding config with specified strategy
    rank, world_size = 0, 4

    config = ShardingTransformConfig(
        rank=rank,
        world_size=world_size,
        stage="sharding",
        allreduce_strategy=AllReduceStrategy[strategy],
    )
    sharding_container = ShardingTransformContainer(config=config)

    # Add transforms: column shard linear1, row shard linear2 (triggers allreduce)
    sharding_container.add(
        WeightShardingInfo(
            target_node=linear1_node.name,
            config=config,
            split_dim=SplitDimension.COLUMN,
            dist_op=None,
        )
    )
    sharding_container.add(
        WeightShardingInfo(
            target_node=linear2_node.name,
            config=config,
            split_dim=SplitDimension.ROW,
            dist_op="all_reduce",
        )
    )

    # Verify transforms have the strategy injected
    assert len(sharding_container.weight_sharding_transforms) == 2
    for transform in sharding_container.weight_sharding_transforms:
        assert transform.config.allreduce_strategy == AllReduceStrategy[strategy], (
            f"Transform {transform.target_node} should have strategy {strategy}, "
            f"got {transform.config.allreduce_strategy}"
        )

    # Apply transforms
    for transform in sharding_container.weight_sharding_transforms:
        node = next((n for n in gm.graph.nodes if n.name == transform.target_node), None)
        if node:
            transform.check_and_apply(gm, node)

    recompile(gm)

    # Verify the graph contains torch_dist_all_reduce nodes with correct strategy
    allreduce_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.torch_dist_all_reduce)
    ]

    # Should have exactly one allreduce node (from linear2 row sharding)
    assert len(allreduce_nodes) == 1, f"Expected 1 allreduce node, found {len(allreduce_nodes)}"

    # Verify the allreduce node has the correct strategy argument
    allreduce_node = allreduce_nodes[0]
    # torch_dist_all_reduce signature: (input, strategy_name)
    assert len(allreduce_node.args) == 2, (
        f"Expected 2 args for allreduce node, got {len(allreduce_node.args)}"
    )

    strategy_arg = allreduce_node.args[1]
    assert strategy_arg == strategy, (
        f"Expected allreduce strategy '{strategy}', got '{strategy_arg}'"
    )

    print(f"✓ Test passed: allreduce_strategy '{strategy}' correctly propagated to graph node")
