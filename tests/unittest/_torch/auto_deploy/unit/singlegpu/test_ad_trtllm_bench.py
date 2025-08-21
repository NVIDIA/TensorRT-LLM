import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml
from _model_test_utils import _hf_model_dir_or_hub_id
from utils.cpp_paths import llm_root  # noqa: F401
from utils.llm_data import llm_models_root


def parse_kv_cache_metrics(log_output: str, free_mem_ratio: float = 0.8):
    """Parse KV cache metrics from the benchmark log output."""
    metrics = {}

    # Simple patterns based on actual log format
    patterns = {
        "current_cache_size": r"Current cache size:\s*(\d+)",
        "free_mem_pre_mb": r"Free memory before forward pass \(MB\):\s*(\d+)",
        "free_mem_post_mb": r"Free memory after forward pass \(MB\):\s*(\d+)",
    }

    # Extract metrics using simple regex patterns
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, log_output, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            metrics[metric_name] = value
            print(f"  ✅ Found {metric_name}: {value}")
        else:
            print(f"  ❌ Could not find {metric_name}")

    # Calculate new_cache_size using the same formula as in resize_kv_cache
    # new_cache_size = free_mem_post * 1024 * 1024 * free_mem_ratio + current_cache_size
    if "free_mem_post_mb" in metrics and "current_cache_size" in metrics:
        metrics["new_cache_size"] = int(
            metrics["free_mem_post_mb"] * 1024 * 1024 * free_mem_ratio
            + metrics["current_cache_size"]
        )
        print(
            f"  ✅ Calculated new_cache_size: {metrics['new_cache_size']} (using free_mem_ratio={free_mem_ratio})"
        )
    else:
        print("  ❌ Cannot calculate new_cache_size - missing required metrics")

    return metrics


def run_benchmark(
    model_name: str,
    dataset_path: str,
    temp_dir: str,
    backend: str = "_autodeploy",
    report_json_path: str = None,
    max_batch_size: int = 32,
    num_hidden_layers: int = 2,
    free_mem_ratio: float = 0.1,
):
    """Run benchmark and capture KV cache metrics from log output."""

    # Read the test config to get free_mem_ratio
    config_path = f"{temp_dir}/extra_llm_api_options.yaml"

    # Build the command to run the benchmark
    cmd = [
        "python",
        "-m",
        "tensorrt_llm.commands.bench",
        "--model",
        model_name,
        "throughput",
        "--backend",
        backend,
        "--dataset",
        str(dataset_path),
        "--max_batch_size",
        str(max_batch_size),
    ]

    # Add report_json argument if path is provided
    if report_json_path:
        cmd.extend(["--report_json", report_json_path])

    if backend == "_autodeploy":
        # Add extra_llm_api_options only for autodeploy backend
        cmd.extend(["--extra_llm_api_options", config_path])

    # Run benchmark as subprocess to capture ALL output
    import os

    env = os.environ.copy()
    if backend == "pytorch":
        env["TLLM_OVERRIDE_LAYER_NUM"] = str(num_hidden_layers)
        print(f"📋 Using TLLM_OVERRIDE_LAYER_NUM from env: {env['TLLM_OVERRIDE_LAYER_NUM']}")
        cmd.extend(["--kv_cache_free_gpu_mem_fraction", str(free_mem_ratio)])
    print(f"🚀 Running benchmark command ({backend} backend): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    # Check if the command succeeded
    assert result.returncode == 0, (
        f"Benchmark failed with return code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Combine stdout and stderr for parsing
    full_log_output = f"{result.stdout}\n{result.stderr}"

    # Parse KV cache metrics from the combined log output (only for autodeploy backend)
    kv_cache_metrics = {}
    if backend == "_autodeploy":
        kv_cache_metrics = parse_kv_cache_metrics(full_log_output, free_mem_ratio)
        print("📊 KV Cache Metrics parsed from logs:")
        if kv_cache_metrics:
            for key, value in kv_cache_metrics.items():
                if "mb" in key.lower():
                    print(f"  {key}: {value}MB")
                else:
                    print(f"  {key}: {value} bytes")
        else:
            print("  ⚠️ No KV cache metrics were parsed successfully")
    else:
        print(f"📊 KV Cache Metrics: Skipped for {backend} backend")

    # Return parsed JSON report with KV cache metrics if requested
    if report_json_path and Path(report_json_path).exists():
        with open(report_json_path, "r") as f:
            report_data = json.load(f)

        # Add KV cache metrics to the report (only for autodeploy backend)
        if backend == "_autodeploy":
            report_data["kv_cache_metrics"] = kv_cache_metrics
        report_data["backend"] = backend
        return report_data
    return None


def compare_backends_performance(
    autodeploy_tokens_per_sec: float,
    pytorch_tokens_per_sec: float,
    relative_tolerance: float = 0.20,
    absolute_tolerance: float = 10.0,
):
    """
    Compare performance between autodeploy and pytorch backends.
    Fails if autodeploy is significantly worse than pytorch.

    Args:
        autodeploy_tokens_per_sec: Performance of autodeploy backend
        pytorch_tokens_per_sec: Performance of pytorch backend
        relative_tolerance: Relative tolerance (20% by default for backend comparison)
        absolute_tolerance: Absolute tolerance (10 tokens/sec by default)
    """
    # Calculate performance difference
    performance_diff = pytorch_tokens_per_sec - autodeploy_tokens_per_sec
    relative_diff = performance_diff / pytorch_tokens_per_sec if pytorch_tokens_per_sec > 0 else 0

    print("=== BACKEND PERFORMANCE COMPARISON ===")
    print(f"PyTorch backend: {pytorch_tokens_per_sec:.2f} tokens/sec/user")
    print(f"Autodeploy backend: {autodeploy_tokens_per_sec:.2f} tokens/sec/user")
    print(f"Performance difference: {performance_diff:.2f} tokens/sec ({relative_diff:.2%})")

    # If autodeploy is better than or equal to pytorch, always pass
    if autodeploy_tokens_per_sec >= pytorch_tokens_per_sec:
        print("✅ Autodeploy backend matches or exceeds PyTorch backend performance")
        return

    # Autodeploy is slower - check if it's within acceptable tolerance
    within_relative_tolerance = relative_diff <= relative_tolerance
    within_absolute_tolerance = performance_diff <= absolute_tolerance

    if within_relative_tolerance or within_absolute_tolerance:
        print("✅ Autodeploy backend performance within acceptable tolerance")
        print(
            f"   Tolerance: {relative_tolerance:.2%} relative OR {absolute_tolerance:.2f} tokens/sec absolute"
        )
    else:
        assert False, (
            f"Autodeploy backend significantly underperforms compared to PyTorch! "
            f"Autodeploy: {autodeploy_tokens_per_sec:.2f} tokens/sec/user, "
            f"PyTorch: {pytorch_tokens_per_sec:.2f} tokens/sec/user, "
            f"Performance gap: {performance_diff:.2f} tokens/sec ({relative_diff:.2%}), "
            f"Tolerance: {relative_tolerance:.2%} relative OR {absolute_tolerance:.2f} tokens/sec absolute"
        )


def assert_performance_within_tolerance(
    actual_tokens_per_sec: float,
    golden_tokens_per_sec: float,
    relative_tolerance: float = 0.15,
    absolute_tolerance: float = 10.0,
):
    """
    Assert that actual performance is within tolerance of golden result.
    Only fails if performance is WORSE than golden - improvements always pass.

    Args:
        actual_tokens_per_sec: Measured performance metric
        golden_tokens_per_sec: Expected performance metric
        relative_tolerance: Relative tolerance (15% by default)
        absolute_tolerance: Absolute tolerance (10 tokens/sec by default)
    """
    # If actual performance is better than or equal to golden, always pass
    if actual_tokens_per_sec >= golden_tokens_per_sec:
        print(
            f"✅ Performance improvement detected:"
            f" {actual_tokens_per_sec:.2f} >= {golden_tokens_per_sec:.2f} tokens/sec/user"
        )
        return

    # Performance is worse than golden - check if it's within acceptable tolerance
    performance_drop = golden_tokens_per_sec - actual_tokens_per_sec
    relative_drop = (
        performance_drop / golden_tokens_per_sec if golden_tokens_per_sec > 0 else float("inf")
    )

    # Performance should be within relative tolerance OR absolute tolerance
    within_relative_tolerance = relative_drop <= relative_tolerance
    within_absolute_tolerance = performance_drop <= absolute_tolerance

    assert within_relative_tolerance or within_absolute_tolerance, (
        f"Performance regression detected! "
        f"Actual: {actual_tokens_per_sec:.2f} tokens/sec/user, "
        f"Golden: {golden_tokens_per_sec:.2f} tokens/sec/user, "
        f"Performance drop: {performance_drop:.2f} tokens/sec ({relative_drop:.2%}), "
        f"Tolerance: {relative_tolerance:.2%} relative OR {absolute_tolerance:.2f} tokens/sec absolute"
    )


def prepare_dataset(root_dir: str, temp_dir: str, model_name: str):
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
        model_name,
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
        "10",
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


def calculate_expected_kv_cache_metrics(free_mem_ratio: float):
    """Calculate expected KV cache metrics based on actual GPU memory."""
    try:
        import torch

        if torch.cuda.is_available():
            # Get total GPU memory in MB
            _, total_mem_bytes = torch.cuda.mem_get_info(0)
            total_mem_mb = total_mem_bytes // (1024 * 1024)

            # Estimate expected values based on model size
            # For TinyLlama-1.1B, model should be 2.2GB
            estimated_model_size_mb = 2200  # Conservative estimate
            # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/6335 check why there is extra consumption
            extra_consumption_mb = 2700
            expected_free_mem_range = (
                total_mem_mb - estimated_model_size_mb - extra_consumption_mb,
                total_mem_mb - estimated_model_size_mb,
            )

            # Current cache size is typically small initially (16MB range)
            expected_current_cache_size = 16777216

            # Free memory values should be in reasonable range
            expected_free_mem_pre_range = expected_free_mem_range
            expected_free_mem_post_range = expected_free_mem_range

            print("📊 GPU Memory Analysis:")
            print(f"  Total GPU memory: {total_mem_mb}MB")
            print(
                f"  Expected free memory range: {expected_free_mem_range[0]}-{expected_free_mem_range[1]}MB"
            )

            return {
                "total_mem_mb": total_mem_mb,
                "expected_current_cache_size": expected_current_cache_size,
                "expected_free_mem_pre_range": expected_free_mem_pre_range,
                "expected_free_mem_post_range": expected_free_mem_post_range,
                "free_mem_ratio": free_mem_ratio,
            }
        else:
            return None
    except ImportError:
        return None


def validate_kv_cache_metrics_dynamic(kv_cache_metrics: dict, expected_metrics: dict):
    """Validate KV cache metrics using dynamic expected values."""

    # Validate current_cache_size (should be relatively stable)
    current_cache_size = kv_cache_metrics.get("current_cache_size")
    expected_cache_size = expected_metrics["expected_current_cache_size"]
    if current_cache_size:
        cache_diff = abs(current_cache_size - expected_cache_size) / expected_cache_size
        assert cache_diff <= 0.5, (  # 50% tolerance for cache size
            f"Current cache size outside expected range: {current_cache_size} vs expected ~{expected_cache_size}"
        )
        print(f"  ✅ current_cache_size: {current_cache_size} bytes (within range)")

    # Validate free memory values are in reasonable ranges
    free_mem_pre = kv_cache_metrics.get("free_mem_pre_mb")
    free_mem_post = kv_cache_metrics.get("free_mem_post_mb")

    if free_mem_pre:
        pre_range = expected_metrics["expected_free_mem_pre_range"]
        assert pre_range[0] <= free_mem_pre <= pre_range[1], (
            f"Free memory before forward pass outside expected range: "
            f"{free_mem_pre}MB not in range {pre_range[0]}-{pre_range[1]}MB"
        )
        print(f"  ✅ free_mem_pre_mb: {free_mem_pre}MB (within range)")

    if free_mem_post:
        post_range = expected_metrics["expected_free_mem_post_range"]
        assert post_range[0] <= free_mem_post <= post_range[1], (
            f"Free memory after forward pass outside expected range: "
            f"{free_mem_post}MB not in range {post_range[0]}-{post_range[1]}MB"
        )
        print(f"  ✅ free_mem_post_mb: {free_mem_post}MB (within range)")

    # Validate memory reduction (pre should be > post)
    if free_mem_pre and free_mem_post:
        memory_reduction = free_mem_pre - free_mem_post
        assert memory_reduction > 0, (
            f"Expected memory reduction during forward pass, got {memory_reduction}MB"
        )
        print(f"  ✅ Memory reduction during forward pass: {memory_reduction}MB")

    # Validate calculated new_cache_size
    new_cache_size = kv_cache_metrics.get("new_cache_size")
    if new_cache_size and free_mem_post and current_cache_size:
        expected_new_cache = int(
            free_mem_post * 1024 * 1024 * expected_metrics["free_mem_ratio"] + current_cache_size
        )
        cache_size_diff = abs(new_cache_size - expected_new_cache) / expected_new_cache
        assert cache_size_diff <= 0.01, (  # 1% tolerance for calculated value
            f"Calculated new_cache_size mismatch: {new_cache_size} vs expected {expected_new_cache}"
        )
        print(f"  ✅ new_cache_size: {new_cache_size} bytes (calculation correct)")


def extract_performance_metric(report_data, report_name="benchmark"):
    """Extract performance metric from a benchmark report with validation."""
    assert report_data is not None, f"Failed to capture {report_name} report"
    assert "performance" in report_data, f"Performance metrics not found in {report_name} report"

    tokens_per_sec = report_data["performance"].get("output_throughput_per_user_tok_s")
    assert tokens_per_sec is not None, (
        f"output_throughput_per_user_tok_s not found in {report_name} performance metrics"
    )

    return tokens_per_sec


def validate_and_extract_kv_cache_metrics(report_data, free_mem_ratio, require_metrics=True):
    """
    Validate and extract KV cache metrics from report.

    Args:
        report_data: The benchmark report data
        free_mem_ratio: Free memory ratio for calculating expected metrics
        require_metrics: If True, fail when metrics are missing. If False, just warn.

    Returns:
        Tuple of (kv_cache_metrics, expected_metrics) or (None, None) if validation fails
    """
    required_metrics = [
        "current_cache_size",
        "free_mem_pre_mb",
        "free_mem_post_mb",
        "new_cache_size",
    ]

    # Extract KV cache metrics
    kv_cache_metrics = report_data.get("kv_cache_metrics", {})

    if not kv_cache_metrics:
        message = (
            "KV cache metrics not found! "
            "The autodeploy backend must log memory statistics for this test to pass. "
            f"Expected metrics: {', '.join(required_metrics)}"
        )
        if require_metrics:
            assert False, f"REQUIRED {message}"
        else:
            print(f"ℹ️ {message}")
            assert False, "KV cache metrics are missing"

    # Check for missing metrics
    missing_metrics = [metric for metric in required_metrics if metric not in kv_cache_metrics]

    if missing_metrics:
        message = (
            f"Missing required KV cache metrics: {missing_metrics}. "
            f"Found metrics: {list(kv_cache_metrics.keys())}. "
            f"All of {required_metrics} are required for the test to pass."
        )
        if require_metrics:
            assert False, message
        else:
            print(f"ℹ️ KV cache validation skipped - {message}")
            assert False, "KV cache metrics are missing"

    # Calculate expected metrics
    expected_metrics = calculate_expected_kv_cache_metrics(free_mem_ratio)
    assert expected_metrics, "Could not determine expected metrics for this GPU"

    return kv_cache_metrics, expected_metrics


def print_kv_cache_metrics(kv_cache_metrics):
    """Print KV cache metrics in a formatted way."""
    print("=== KV CACHE METRICS (DYNAMIC VALIDATION) ===")
    for metric_name, actual_value in kv_cache_metrics.items():
        if "mb" in metric_name.lower():
            print(f"{metric_name}: {actual_value}MB")
        else:
            print(f"{metric_name}: {actual_value} bytes")


def trtllm_bench_unified_comparison(
    llm_root,  # noqa: F811
    comparison_mode="backend",
    free_mem_ratio=0.1,
    num_hidden_layers=2,
    max_batch_size=32,  # below this value the kv cache resizing is skipped
    golden_tokens_per_sec=1400,
    backend_relative_tolerance=0.3,
    backend_absolute_tolerance=250.0,
    golden_relative_tolerance=0.1,
    golden_absolute_tolerance=5.0,
):
    """
    Unified test that compares autodeploy backend performance in two modes:
    - "backend": compares against pytorch backend performance
    - "golden": compares against predefined golden performance values

    Args:
        llm_root: Root directory for LLM models (pytest fixture)
        comparison_mode: Either "backend" or "golden" to determine comparison type
        free_mem_ratio: Ratio of free memory to use for KV cache
        num_hidden_layers: Number of hidden layers for the model
        max_batch_size: Maximum batch size for benchmarking
        golden_tokens_per_sec: Golden performance value in tokens/sec/user
        backend_relative_tolerance: Relative tolerance for backend comparison
        backend_absolute_tolerance: Absolute tolerance for backend comparison
        golden_relative_tolerance: Relative tolerance for golden comparison
        golden_absolute_tolerance: Absolute tolerance for golden comparison
    """
    model_name = _hf_model_dir_or_hub_id(
        f"{llm_models_root()}/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/extra_llm_api_options.yaml", "w") as f:
            yaml.dump(
                {
                    "model_kwargs": {"num_hidden_layers": num_hidden_layers},
                    "cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32],
                    "compile_backend": "torch-opt",
                    "free_mem_ratio": free_mem_ratio,
                    "runtime": "trtllm",
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, model_name)

        # Always run autodeploy backend
        autodeploy_report_path = f"{temp_dir}/autodeploy_report.json"
        print("=== RUNNING AUTODEPLOY BACKEND ===")
        autodeploy_report = run_benchmark(
            model_name,
            dataset_path,
            temp_dir,
            "_autodeploy",
            autodeploy_report_path,
            max_batch_size,
            num_hidden_layers,
            free_mem_ratio,
        )

        # Extract autodeploy performance metrics
        autodeploy_tokens_per_sec = extract_performance_metric(autodeploy_report, "autodeploy")

        # Validate and extract KV cache metrics (now required for both modes after user's changes)
        kv_cache_metrics, expected_metrics = validate_and_extract_kv_cache_metrics(
            autodeploy_report, free_mem_ratio, require_metrics=True
        )

        if comparison_mode == "backend":
            # Backend comparison mode: also run pytorch backend
            pytorch_report_path = f"{temp_dir}/pytorch_report.json"
            print("=== RUNNING PYTORCH BACKEND ===")
            pytorch_report = run_benchmark(
                model_name,
                dataset_path,
                temp_dir,
                "pytorch",
                pytorch_report_path,
                max_batch_size,
                num_hidden_layers,
                free_mem_ratio,
            )

            # Extract pytorch performance metrics
            pytorch_tokens_per_sec = extract_performance_metric(pytorch_report, "pytorch")

            # Compare backend performance
            compare_backends_performance(
                autodeploy_tokens_per_sec,
                pytorch_tokens_per_sec,
                relative_tolerance=backend_relative_tolerance,
                absolute_tolerance=backend_absolute_tolerance,
            )

            # Validate KV cache metrics
            validate_kv_cache_metrics_dynamic(kv_cache_metrics, expected_metrics)
            print("✅ KV Cache Metrics validation passed")

            print("=== BACKEND COMPARISON TEST PASSED ===")
            print(f"Autodeploy: {autodeploy_tokens_per_sec:.2f} tokens/sec/user")
            print(f"PyTorch: {pytorch_tokens_per_sec:.2f} tokens/sec/user")

        elif comparison_mode == "golden":
            # Golden comparison mode: compare against golden values
            print("=== PERFORMANCE METRICS ===")
            print(f"Measured performance: {autodeploy_tokens_per_sec:.2f} tokens/sec/user")
            print(f"Golden performance: {golden_tokens_per_sec:.2f} tokens/sec/user")

            # Print KV cache metrics
            print_kv_cache_metrics(kv_cache_metrics)

            # Performance validation
            assert_performance_within_tolerance(
                autodeploy_tokens_per_sec,
                golden_tokens_per_sec,
                relative_tolerance=golden_relative_tolerance,
                absolute_tolerance=golden_absolute_tolerance,
            )

            # KV cache metrics validation
            print(
                f"Validating {len(kv_cache_metrics)} KV cache metrics against GPU-specific ranges..."
            )
            validate_kv_cache_metrics_dynamic(kv_cache_metrics, expected_metrics)

            print("=== ALL TESTS PASSED ===")
            print(f"Performance: ✅ {autodeploy_tokens_per_sec:.2f} tokens/sec/user within bounds")
            print("KV Cache Metrics: ✅ All metrics within GPU-specific expected ranges")

        else:
            raise ValueError(
                f"Invalid comparison_mode: {comparison_mode}. Must be 'backend' or 'golden'"
            )


def test_trtllm_bench(llm_root):  # noqa: F811
    model_name = _hf_model_dir_or_hub_id(
        f"{llm_models_root()}/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/extra_llm_api_options.yaml", "w") as f:
            yaml.dump(
                {
                    "model_kwargs": {"num_hidden_layers": 2},
                    "cuda_graph_batch_sizes": [1, 2],
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, model_name)
        run_benchmark(model_name, dataset_path, temp_dir)


@pytest.mark.skip(reason="https://nvbugs/5458798")
@pytest.mark.no_xdist
def test_trtllm_bench_backend_comparison(llm_root):  # noqa: F811
    """Test that compares autodeploy backend performance against pytorch backend
    with given relative and absolute thresholds.

    It also checks the memory footprint of the autodeploy backend by parsing the
    log output from the resize_kv_cache function and extracting the following metrics:
    current_cache_size - the cache size before resize
    free_mem_pre_mb - the free memory before forward pass
    free_mem_post_mb - the free memory after forward pass
    new_cache_size - the cache size after resize

    The following checks are performed:
    1. free_mem_pre_fw_pass and free_mem_post_fw_pass are in:
       [Total mem - expected_model_size - extra_consumption, Total mem - expected_model_size]
    2. memory_reduction = free_mem_pre_fw_pass - free_mem_post_fw_pass > 0
    3. expected_new_cache = free_mem_post * free_mem_ratio + current_cache_size
       cache_size_diff = abs(new_cache_size - expected_new_cache) / expected_new_cache
       assert cache_size_diff <= 0.01

    extra_consumption_mb = 2700 - this is unexplained memory consumption to be investigated.
    """
    trtllm_bench_unified_comparison(llm_root, comparison_mode="backend")
