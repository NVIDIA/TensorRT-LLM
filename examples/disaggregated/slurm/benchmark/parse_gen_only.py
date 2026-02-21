#!/usr/bin/env python3
"""
Parse gen-only benchmark job directories and extract completion status,
saturation metrics, and configuration into a CSV file.

Usage:
    source /lustre/fsw/coreai_comparch_trtllm/bbuddharaju/venvs/pareto/bin/activate
    python parse_gen_only.py --dir_list <job_dirs> --output_dir <output_dir>

Or run directly with the venv Python:
    /lustre/fsw/coreai_comparch_trtllm/bbuddharaju/venvs/pareto/bin/python parse_gen_only.py ...

Optional nsys profile analysis:
    python parse_gen_only.py --dir_list <job_dirs> --output_dir <output_dir> --analyze-nsys
"""

import argparse
import csv
import glob
import os
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml

# =============================================================================
# NSYS Profile Analysis Constants (DeepSeek-R1 / DeepSeekV3 Architecture)
# =============================================================================
NUM_LAYERS = 61
NUM_DENSE_LAYERS = 3  # first_k_dense_replace
NUM_MOE_LAYERS = NUM_LAYERS - NUM_DENSE_LAYERS  # 58

# Kernel categorization based on DeepSeekV3 components
KERNEL_CATEGORIES = {
    # ============ ATTENTION ============
    "attention": {
        "fmha": ["fmhaSm100", "fmha"],
        "qkv_projection": ["applyMLARopeAndAssignQKVKernel"],
        "output_projection": ["nvjet_sm103_tst_176x128"],
        "allreduce": ["twoshotAllreduceKernel"],
        "helix_cp": ["helixAllToAllKernel", "helix_postprocess"],
    },
    # ============ MOE ============
    "moe": {
        "gate_routing": ["deepseek_v3_topk_kernel", "routingIndices"],
        "dispatch_prepare": ["moeA2APrepareDispatchKernel", "moeA2ASanitizeExpertIdsKernel",
                            "computeStridesTmaWarpSpecializedKernel", "expandInputRowsKernel",
                            "blockExpertPrefixSumKernel", "globalExpertPrefixSumKernel",
                            "mergeExpertPrefixSumKernel"],
        "dispatch_comm": ["moeA2ADispatchKernel"],
        "expert_compute": ["doActivationKernel", "device_kernel",
                          "kernel_cutlass_kernel_tensorrt_llm_torch"],
        "combine_prepare": ["moeA2APrepareCombineKernel", "finalizeMoeRoutingNoFillingKernel"],
        "combine_comm": ["moeA2ACombineKernel"],
        "memset": ["moeOutputMemsetKernel"],
    },
    # ============ DENSE MLP ============
    "dense_mlp": {
        "gemm": ["nvjet_sm103_tst_256x224", "nvjet_sm103_tst_128x224",
                 "nvjet_sm103_tst_64x64_64x16", "nvjet_sm103_tss_32x64"],
        "activation": ["silu_and_mul_kernel"],
        "splitk": ["splitKreduce_kernel"],
    },
    # ============ NORMALIZATION ============
    "layernorm": {
        "rmsnorm": ["RMSNormKernel", "FusedAddRMSNormKernel"],
    },
    # ============ QUANTIZATION ============
    "quantization": {
        "quantize": ["quantize_with_block_size"],
    },
    # ============ MEMORY/MISC ============
    "memory_ops": {
        "copy": ["CatArrayBatchedCopy"],
        "elementwise": ["elementwise_kernel", "unrolled_elementwise_kernel",
                       "vectorized_elementwise_kernel", "index_elementwise_kernel"],
        "gather_scatter": ["vectorized_gather_kernel", "_scatter_gather_elementwise_kernel"],
        "reduce": ["reduce_kernel"],
        "scan": ["DeviceScanKernel", "DeviceScanInitKernel"],
    },
}


@dataclass
class KernelStats:
    """Statistics for a single kernel type."""
    name: str
    count: int
    total_ns: int
    avg_ns: float
    min_ns: int
    max_ns: int
    category: str
    subcategory: str

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1e6

    @property
    def avg_us(self) -> float:
        return self.avg_ns / 1e3


# =============================================================================
# NSYS Profile Analysis Functions
# =============================================================================


def categorize_kernel(kernel_name: str) -> Tuple[str, str]:
    """Categorize a kernel based on its name."""
    for category, subcats in KERNEL_CATEGORIES.items():
        for subcat, patterns in subcats.items():
            for pattern in patterns:
                if pattern.lower() in kernel_name.lower():
                    return category, subcat
    return "unknown", "unknown"


def find_sqlite_files(job_dir: str, gpu_filter: str = "gpu0") -> List[str]:
    """
    Find SQLite files in job directory.

    Args:
        job_dir: Path to job directory
        gpu_filter: "gpu0" for GPU 0 only, "all" for all GPUs

    Returns:
        List of SQLite file paths
    """
    if gpu_filter == "gpu0":
        # Only GPU 0 files (ending with _0.sqlite)
        pattern = os.path.join(job_dir, "nsys_worker_proc_GEN_*_0.sqlite")
    else:
        # All GPU files
        pattern = os.path.join(job_dir, "nsys_worker_proc_GEN_*.sqlite")

    sqlite_files = glob.glob(pattern)
    return sorted(sqlite_files)


def get_kernel_stats_from_sqlite(sqlite_path: str) -> List[KernelStats]:
    """Get kernel statistics from SQLite database."""
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()

        query = """
            SELECT
                s.value as kernel_name,
                COUNT(*) as count,
                SUM(k.end - k.start) as total_ns,
                AVG(k.end - k.start) as avg_ns,
                MIN(k.end - k.start) as min_ns,
                MAX(k.end - k.start) as max_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            GROUP BY s.value
            ORDER BY total_ns DESC
        """

        cursor.execute(query)
        results = []
        for row in cursor.fetchall():
            name, count, total_ns, avg_ns, min_ns, max_ns = row
            cat, subcat = categorize_kernel(name)
            results.append(KernelStats(
                name=name, count=count, total_ns=total_ns,
                avg_ns=avg_ns, min_ns=min_ns, max_ns=max_ns,
                category=cat, subcategory=subcat
            ))

        conn.close()
        return results
    except Exception as e:
        print(f"Warning: Failed to analyze SQLite {sqlite_path}: {e}")
        return []


def detect_iterations(kernel_stats: List[KernelStats]) -> int:
    """Detect number of iterations from FMHA kernel count."""
    for s in kernel_stats:
        if "fmha" in s.name.lower():
            # FMHA runs once per layer per iteration
            return s.count // NUM_LAYERS
    return 1  # fallback


def analyze_sqlite(sqlite_path: str) -> Dict:
    """
    Extract component-level timing from SQLite.

    Returns dict with:
        - total_gpu_time_ms: Total GPU time
        - iterations: Detected number of iterations
        - gpu_time_per_iter_ms: GPU time per iteration
        - Category breakdowns (attention, moe, etc.)
        - MoE subcategory breakdowns
    """
    result = {
        "total_gpu_time_ms": None,
        "iterations": None,
        "gpu_time_per_iter_ms": None,
        # Category totals
        "attn_time_ms": None,
        "moe_time_ms": None,
        "dense_mlp_time_ms": None,
        "layernorm_time_ms": None,
        "quant_time_ms": None,
        "memory_ops_time_ms": None,
        "unknown_time_ms": None,
        # MoE subcategories
        "moe_gate_routing_ms": None,
        "moe_dispatch_prep_ms": None,
        "moe_dispatch_comm_ms": None,
        "moe_expert_compute_ms": None,
        "moe_combine_prep_ms": None,
        "moe_combine_comm_ms": None,
        "moe_memset_ms": None,
        # Communication breakdown
        "helix_cp_time_ms": None,
        "tp_allreduce_time_ms": None,
        "moe_comm_total_ms": None,
        # Derived metrics
        "moe_comm_pct": None,
        "moe_compute_pct": None,
        "comm_total_pct": None,
    }

    kernel_stats = get_kernel_stats_from_sqlite(sqlite_path)
    if not kernel_stats:
        return result

    # Detect iterations
    iterations = detect_iterations(kernel_stats)
    result["iterations"] = iterations

    # Calculate total GPU time
    total_gpu_time_ms = sum(s.total_ms for s in kernel_stats)
    result["total_gpu_time_ms"] = round(total_gpu_time_ms, 3)
    result["gpu_time_per_iter_ms"] = round(total_gpu_time_ms / iterations, 3) if iterations > 0 else None

    # Category breakdown
    category_time = defaultdict(float)
    for s in kernel_stats:
        category_time[s.category] += s.total_ms

    result["attn_time_ms"] = round(category_time.get("attention", 0), 3)
    result["moe_time_ms"] = round(category_time.get("moe", 0), 3)
    result["dense_mlp_time_ms"] = round(category_time.get("dense_mlp", 0), 3)
    result["layernorm_time_ms"] = round(category_time.get("layernorm", 0), 3)
    result["quant_time_ms"] = round(category_time.get("quantization", 0), 3)
    result["memory_ops_time_ms"] = round(category_time.get("memory_ops", 0), 3)
    result["unknown_time_ms"] = round(category_time.get("unknown", 0), 3)

    # MoE subcategory breakdown
    moe_subcat_time = defaultdict(float)
    for s in kernel_stats:
        if s.category == "moe":
            moe_subcat_time[s.subcategory] += s.total_ms

    result["moe_gate_routing_ms"] = round(moe_subcat_time.get("gate_routing", 0), 3)
    result["moe_dispatch_prep_ms"] = round(moe_subcat_time.get("dispatch_prepare", 0), 3)
    result["moe_dispatch_comm_ms"] = round(moe_subcat_time.get("dispatch_comm", 0), 3)
    result["moe_expert_compute_ms"] = round(moe_subcat_time.get("expert_compute", 0), 3)
    result["moe_combine_prep_ms"] = round(moe_subcat_time.get("combine_prepare", 0), 3)
    result["moe_combine_comm_ms"] = round(moe_subcat_time.get("combine_comm", 0), 3)
    result["moe_memset_ms"] = round(moe_subcat_time.get("memset", 0), 3)

    # Communication breakdown
    helix_time = 0
    allreduce_time = 0
    for s in kernel_stats:
        if s.category == "attention":
            if s.subcategory == "helix_cp":
                helix_time += s.total_ms
            elif s.subcategory == "allreduce":
                allreduce_time += s.total_ms

    result["helix_cp_time_ms"] = round(helix_time, 3)
    result["tp_allreduce_time_ms"] = round(allreduce_time, 3)

    # MoE communication total (dispatch + combine)
    moe_comm_total = moe_subcat_time.get("dispatch_comm", 0) + moe_subcat_time.get("combine_comm", 0)
    result["moe_comm_total_ms"] = round(moe_comm_total, 3)

    # Derived percentages
    moe_time = category_time.get("moe", 0)
    if moe_time > 0:
        result["moe_comm_pct"] = round(100.0 * moe_comm_total / moe_time, 2)
        moe_compute = moe_subcat_time.get("expert_compute", 0)
        result["moe_compute_pct"] = round(100.0 * moe_compute / moe_time, 2)

    if total_gpu_time_ms > 0:
        total_comm = moe_comm_total + helix_time + allreduce_time
        result["comm_total_pct"] = round(100.0 * total_comm / total_gpu_time_ms, 2)
        # Unknown percentage
        unknown_time = category_time.get("unknown", 0)
        result["unknown_pct"] = round(100.0 * unknown_time / total_gpu_time_ms, 2)

    return result


def analyze_job_nsys(job_dir: str, gpu_filter: str = "gpu0") -> Dict:
    """
    Analyze nsys profiles for a job directory.

    Returns aggregated profile metrics (averaged across available SQLite files).
    """
    sqlite_files = find_sqlite_files(job_dir, gpu_filter)

    if not sqlite_files:
        return {}

    # Analyze each SQLite file and aggregate
    all_results = []
    for sqlite_path in sqlite_files:
        result = analyze_sqlite(sqlite_path)
        if result.get("total_gpu_time_ms") is not None:
            all_results.append(result)

    if not all_results:
        return {}

    # For GPU 0 only, just return the single result
    if len(all_results) == 1:
        return all_results[0]

    # Average across multiple files (for all GPUs case)
    aggregated = {}
    numeric_keys = [k for k in all_results[0].keys() if all_results[0][k] is not None]

    for key in numeric_keys:
        values = [r[key] for r in all_results if r.get(key) is not None]
        if values:
            aggregated[key] = round(sum(values) / len(values), 3)

    return aggregated


def is_job_directory(path: str) -> bool:
    """Check if a path is a job directory by looking for gen_config.yaml."""
    return os.path.isfile(os.path.join(path, "gen_config.yaml"))


def find_job_directories(dir_list: List[str]) -> List[str]:
    """
    Given a list of paths, find all job directories.
    Each path can be either a job directory or a parent containing job directories.
    """
    job_dirs = []
    for path in dir_list:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            print(f"Warning: {path} is not a directory, skipping")
            continue

        if is_job_directory(path):
            job_dirs.append(path)
        else:
            # Scan subdirectories for job directories.
            for entry in os.listdir(path):
                subdir = os.path.join(path, entry)
                if os.path.isdir(subdir) and is_job_directory(subdir):
                    job_dirs.append(subdir)

    return job_dirs


def parse_gen_config(job_dir: str) -> Dict:
    """Parse gen_config.yaml and extract configuration values."""
    config_path = os.path.join(job_dir, "gen_config.yaml")
    config = {}

    try:
        with open(config_path, "r") as f:
            gen_config = yaml.safe_load(f)

        config["gen_tp"] = gen_config.get("tensor_parallel_size", None)
        config["gen_cp"] = gen_config.get("context_parallel_size", None)
        config["gen_pp"] = gen_config.get("pipeline_parallel_size", None)
        config["gen_ep"] = gen_config.get("moe_expert_parallel_size", None)
        config["is_attn_dp"] = gen_config.get("enable_attention_dp", False)

        # Get moe_config.backend.
        moe_config = gen_config.get("moe_config", {})
        config["moe_backend"] = moe_config.get("backend", None) if moe_config else None

        # Get cuda_graph_config.max_batch_size.
        cuda_graph_config = gen_config.get("cuda_graph_config", {})
        config["max_batch_size"] = (
            cuda_graph_config.get("max_batch_size", None)
            if cuda_graph_config
            else None
        )

    except Exception as e:
        print(f"Warning: Failed to parse gen_config.yaml in {job_dir}: {e}")

    return config


def parse_global_batch_size(job_dir: str) -> Optional[int]:
    """Extract global batch size from directory name (e.g., batch32 -> 32)."""
    dir_name = os.path.basename(job_dir)
    match = re.search(r"batch(\d+)", dir_name)
    if match:
        return int(match.group(1))
    return None


def get_slurm_id(job_dir: str) -> Optional[str]:
    """Get Slurm ID from 8_done_<slurm_id>.txt filename."""
    pattern = os.path.join(job_dir, "8_done_*.txt")
    done_files = glob.glob(pattern)
    if done_files:
        # Extract slurm_id from filename like 8_done_818972.txt.
        filename = os.path.basename(done_files[0])
        match = re.match(r"8_done_(\d+)\.txt", filename)
        if match:
            return match.group(1)
    return None


def parse_bench_log(job_dir: str) -> Tuple[bool, Optional[float]]:
    """
    Parse 6_bench.log for median TPOT.
    Returns (is_complete, median_tpot_ms).
    """
    bench_log_path = os.path.join(job_dir, "6_bench.log")

    if not os.path.isfile(bench_log_path):
        return False, None

    try:
        with open(bench_log_path, "r") as f:
            content = f.read()

        # Look for "Median TPOT (ms):" line.
        match = re.search(r"Median TPOT \(ms\):\s+([\d.]+)", content)
        if match:
            median_tpot = float(match.group(1))
            return True, median_tpot
    except Exception as e:
        print(f"Warning: Failed to parse 6_bench.log in {job_dir}: {e}")

    return False, None


def parse_iteration_logs(
    job_dir: str, max_batch_size: Optional[int]
) -> Dict:
    """
    Parse 3_output_GEN_0.log for saturation analysis and iteration TPOT.

    Extracts:
    - Saturation metrics (how many iterations ran at full batch)
    - Median TPOT from prev_device_step_time of saturated iterations

    This provides a more accurate TPOT than the benchmark client reports,
    especially for CP>1 configurations where there's significant overhead
    between device step completion and client-observed latency.

    Returns dict with saturation metrics and iter_median_tpot_ms.
    """
    log_path = os.path.join(job_dir, "3_output_GEN_0.log")
    result = {
        "saturation_pct": None,
        "saturated_iters": 0,
        "total_iters": 0,
        "first_saturated_iter": None,
        "last_saturated_iter": None,
        "iter_median_tpot_ms": None,
    }

    if not os.path.isfile(log_path):
        return result

    if max_batch_size is None:
        return result

    try:
        # Pattern to match iteration logs with num_scheduled_requests and prev_device_step_time.
        # Example: iter = 4, ... prev_device_step_time = 11.969311714172363ms, ... num_scheduled_requests: 16, ...
        pattern = re.compile(
            r"iter = (\d+),.*prev_device_step_time\s*=\s*([\d.]+)ms,.*num_scheduled_requests:\s*(\d+)"
        )

        iterations = []
        saturated_device_step_times = []

        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                # Skip lines with N/A for prev_device_step_time
                if "prev_device_step_time = N/A" in line:
                    continue

                match = pattern.search(line)
                if match:
                    iter_num = int(match.group(1))
                    prev_device_step_time = float(match.group(2))
                    num_scheduled = int(match.group(3))
                    iterations.append((iter_num, num_scheduled, prev_device_step_time))

                    # Collect device step times for saturated iterations
                    if num_scheduled == max_batch_size:
                        saturated_device_step_times.append(prev_device_step_time)

        if not iterations:
            return result

        result["total_iters"] = len(iterations)

        # Find saturated iterations (where num_scheduled_requests == max_batch_size).
        saturated_iters = [
            (iter_num, num_scheduled)
            for iter_num, num_scheduled, _ in iterations
            if num_scheduled == max_batch_size
        ]

        result["saturated_iters"] = len(saturated_iters)

        if saturated_iters:
            result["first_saturated_iter"] = saturated_iters[0][0]
            result["last_saturated_iter"] = saturated_iters[-1][0]

        if result["total_iters"] > 0:
            result["saturation_pct"] = round(
                100.0 * result["saturated_iters"] / result["total_iters"], 2
            )

        # Calculate median TPOT from saturated iterations
        # TPOT = prev_device_step_time (time per iteration = time per token per request)
        if saturated_device_step_times:
            result["iter_median_tpot_ms"] = round(
                statistics.median(saturated_device_step_times), 4
            )

    except Exception as e:
        print(f"Warning: Failed to parse 3_output_GEN_0.log in {job_dir}: {e}")

    return result


def analyze_job(job_dir: str, analyze_nsys: bool = False, gpu_filter: str = "gpu0") -> Dict:
    """Analyze a single job directory and return results."""
    result = {
        "job_dir": job_dir,
        "slurm_id": None,
        "num_gpus": None,
        "global_batch_size": None,
        "is_attn_dp": None,
        "gen_tp": None,
        "gen_cp": None,
        "gen_pp": None,
        "gen_ep": None,
        "moe_backend": None,
        "is_complete": False,
        "bench_median_tpot_ms": None,  # TPOT from benchmark client (6_bench.log)
        "iter_median_tpot_ms": None,   # TPOT from iteration logs (prev_device_step_time)
        "tpot_discrepancy_pct": None,  # % difference: (bench - iter) / iter * 100
        "done_file_present": False,
        "saturation_pct": None,
        "saturated_iters": 0,
        "total_iters": 0,
        "first_saturated_iter": None,
        "last_saturated_iter": None,
        "gen_output_tput_per_user": None,
        "gen_output_tput_per_gpu": None,
        # NSYS profile columns (populated if analyze_nsys=True)
        "nsys_gpu_time_per_iter_ms": None,
        "nsys_attn_time_ms": None,
        "nsys_moe_time_ms": None,
        "nsys_moe_comm_pct": None,
        "nsys_moe_compute_pct": None,
        "nsys_helix_cp_time_ms": None,
        "nsys_comm_total_pct": None,
        "nsys_unknown_time_ms": None,
        "nsys_unknown_pct": None,
    }

    # Get Slurm ID and check done file.
    slurm_id = get_slurm_id(job_dir)
    result["slurm_id"] = slurm_id
    result["done_file_present"] = slurm_id is not None

    # Parse global batch size from directory name.
    result["global_batch_size"] = parse_global_batch_size(job_dir)

    # Parse gen_config.yaml.
    config = parse_gen_config(job_dir)
    result["is_attn_dp"] = config.get("is_attn_dp")
    result["gen_tp"] = config.get("gen_tp")
    result["gen_cp"] = config.get("gen_cp")
    result["gen_pp"] = config.get("gen_pp")
    result["gen_ep"] = config.get("gen_ep")
    result["moe_backend"] = config.get("moe_backend")
    max_batch_size = config.get("max_batch_size")

    # Calculate num_gpus = tp * cp * pp.
    tp = result["gen_tp"]
    cp = result["gen_cp"]
    pp = result["gen_pp"]
    if all(v is not None for v in [tp, cp, pp]):
        result["num_gpus"] = tp * cp * pp

    # Verify global_batch_size from directory name matches expected value from config.
    # global_batch_size = cuda_graph_config.max_batch_size * pp * dp_size
    # where dp_size = tp if attention_dp is enabled, else 1
    global_batch_size = result["global_batch_size"]
    is_attn_dp = result["is_attn_dp"]
    assert all(v is not None for v in [global_batch_size, max_batch_size, pp, tp, is_attn_dp]), (
        f"Missing required config values in {job_dir}: "
        f"global_batch_size={global_batch_size}, max_batch_size={max_batch_size}, "
        f"pp={pp}, tp={tp}, is_attn_dp={is_attn_dp}"
    )
    dp_size = tp if is_attn_dp else 1
    expected_global_batch_size = max_batch_size * pp * dp_size
    assert global_batch_size == expected_global_batch_size, (
        f"global_batch_size mismatch in {job_dir}: "
        f"directory name says {global_batch_size}, but expected "
        f"{expected_global_batch_size} (cuda_graph_config.max_batch_size={max_batch_size} * pp={pp} * dp_size={dp_size})"
    )

    # Parse bench log for completion status only.
    # NOTE: bench_median_tpot_ms is kept for reference but NOT used for throughput.
    # The bench client measures higher latency than actual device step time,
    # especially for CP>1 configurations (~35-45% higher vs ~7.5% for CP=1).
    is_complete, bench_median_tpot = parse_bench_log(job_dir)
    result["is_complete"] = is_complete
    result["bench_median_tpot_ms"] = bench_median_tpot

    # Parse iteration logs for saturation analysis AND iteration-based TPOT.
    # iter_median_tpot_ms is computed from prev_device_step_time of saturated
    # iterations (where num_scheduled_requests == max_batch_size).
    # This is more accurate than bench TPOT as it measures actual GPU execution time.
    saturation_info = parse_iteration_logs(job_dir, max_batch_size)
    result["saturation_pct"] = saturation_info["saturation_pct"]
    result["saturated_iters"] = saturation_info["saturated_iters"]
    result["total_iters"] = saturation_info["total_iters"]
    result["first_saturated_iter"] = saturation_info["first_saturated_iter"]
    result["last_saturated_iter"] = saturation_info["last_saturated_iter"]
    result["iter_median_tpot_ms"] = saturation_info["iter_median_tpot_ms"]

    # Calculate TPOT discrepancy between bench and iteration logs.
    iter_tpot = result["iter_median_tpot_ms"]
    if bench_median_tpot is not None and iter_tpot is not None and iter_tpot > 0:
        result["tpot_discrepancy_pct"] = round(
            100.0 * (bench_median_tpot - iter_tpot) / iter_tpot, 2
        )

    # Calculate performance stats using ITERATION TPOT (not bench TPOT).
    # This provides more accurate throughput metrics, especially for CP>1.
    result["gen_output_tput_per_user"] = None
    result["gen_output_tput_per_gpu"] = None

    if iter_tpot is not None and iter_tpot > 0:
        # gen_output_tput_per_user = 1000 / iter_tpot (tokens/sec per user).
        result["gen_output_tput_per_user"] = round(1000.0 / iter_tpot, 2)

        # gen_output_tput_per_gpu = 1000 * global_bs / (iter_tpot * num_gpus),
        # where num_gpus = tp_size * cp_size * pp_size.
        global_bs = result["global_batch_size"]
        tp = result["gen_tp"]
        cp = result["gen_cp"]
        pp = result["gen_pp"]

        if all(v is not None for v in [global_bs, tp, cp, pp]):
            num_gpus = tp * cp * pp
            if num_gpus > 0:
                result["gen_output_tput_per_gpu"] = round(
                    1000.0 * global_bs / (iter_tpot * num_gpus), 2
                )

    # Analyze nsys profiles if requested
    if analyze_nsys:
        nsys_result = analyze_job_nsys(job_dir, gpu_filter)
        if nsys_result:
            result["nsys_gpu_time_per_iter_ms"] = nsys_result.get("gpu_time_per_iter_ms")
            result["nsys_attn_time_ms"] = nsys_result.get("attn_time_ms")
            result["nsys_moe_time_ms"] = nsys_result.get("moe_time_ms")
            result["nsys_moe_comm_pct"] = nsys_result.get("moe_comm_pct")
            result["nsys_moe_compute_pct"] = nsys_result.get("moe_compute_pct")
            result["nsys_helix_cp_time_ms"] = nsys_result.get("helix_cp_time_ms")
            result["nsys_comm_total_pct"] = nsys_result.get("comm_total_pct")
            result["nsys_unknown_time_ms"] = nsys_result.get("unknown_time_ms")
            result["nsys_unknown_pct"] = nsys_result.get("unknown_pct")

    return result


def compute_pareto_frontier_with_results(
    results: List[Dict]
) -> List[Dict]:
    """
    Compute Pareto frontier from result dicts, maximizing both X and Y.
    Returns list of result dicts that are on the Pareto frontier.
    """
    if not results:
        return []

    # Sort by X (descending) to find Pareto frontier.
    sorted_results = sorted(
        results, key=lambda r: -r["gen_output_tput_per_user"]
    )

    pareto = []
    max_y = float("-inf")

    for r in sorted_results:
        if r["gen_output_tput_per_gpu"] > max_y:
            pareto.append(r)
            max_y = r["gen_output_tput_per_gpu"]

    # Sort by X ascending for plotting.
    pareto.sort(key=lambda r: r["gen_output_tput_per_user"])
    return pareto


def dedupe_pareto_by_batch_size(pareto_results: List[Dict]) -> List[Dict]:
    """
    For each global_batch_size, keep only the point with highest output_tput_per_gpu.
    """
    if not pareto_results:
        return []

    # Group by global_batch_size.
    by_bs: Dict[int, List[Dict]] = {}
    for r in pareto_results:
        bs = r["global_batch_size"]
        if bs not in by_bs:
            by_bs[bs] = []
        by_bs[bs].append(r)

    # Keep only the one with highest tput_per_gpu for each batch size.
    deduped = []
    for bs, group in by_bs.items():
        best = max(group, key=lambda r: r["gen_output_tput_per_gpu"])
        deduped.append(best)

    # Sort by X ascending.
    deduped.sort(key=lambda r: r["gen_output_tput_per_user"])
    return deduped


def plot_pareto(results: List[Dict], output_dir: str) -> Optional[str]:
    """
    Plot Pareto frontier for CP jobs (CP>1) and TP-only jobs (CP=1).
    Creates two versions:
    1) Full plot with all points (dimmer) + Pareto frontier
    2) Denoised plot with only Pareto frontier, one point per global BS
    X-axis: gen_output_tput_per_user
    Y-axis: gen_output_tput_per_gpu
    """
    # Filter completed jobs with valid throughput data.
    valid_results = [
        r for r in results
        if r["is_complete"]
        and r["gen_output_tput_per_user"] is not None
        and r["gen_output_tput_per_gpu"] is not None
        and r["gen_cp"] is not None
    ]

    if not valid_results:
        print("No valid results for Pareto plot")
        return None

    # Separate into CP jobs (CP > 1) and TP-only jobs (CP = 1).
    cp_jobs = [r for r in valid_results if r["gen_cp"] > 1]
    tp_only_jobs = [r for r in valid_results if r["gen_cp"] == 1]

    # Compute Pareto frontiers (with full result dicts).
    cp_pareto = compute_pareto_frontier_with_results(cp_jobs)
    tp_pareto = compute_pareto_frontier_with_results(tp_only_jobs)

    # Deduped Pareto (one per global BS with highest tput_per_gpu).
    cp_pareto_deduped = dedupe_pareto_by_batch_size(cp_pareto)
    tp_pareto_deduped = dedupe_pareto_by_batch_size(tp_pareto)

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Plot 1: Full plot with all points (dimmer) + Pareto frontier
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all points (dimmer).
    if cp_jobs:
        cp_x = [r["gen_output_tput_per_user"] for r in cp_jobs]
        cp_y = [r["gen_output_tput_per_gpu"] for r in cp_jobs]
        ax.scatter(cp_x, cp_y, alpha=0.25, color="blue", marker="o", s=40,
                   label="KVP>1")

    if tp_only_jobs:
        tp_x = [r["gen_output_tput_per_user"] for r in tp_only_jobs]
        tp_y = [r["gen_output_tput_per_gpu"] for r in tp_only_jobs]
        ax.scatter(tp_x, tp_y, alpha=0.25, color="orange", marker="s", s=40,
                   label="KVP=1")

    # Plot Pareto frontiers (brighter).
    if cp_pareto:
        pareto_x = [r["gen_output_tput_per_user"] for r in cp_pareto]
        pareto_y = [r["gen_output_tput_per_gpu"] for r in cp_pareto]
        ax.plot(pareto_x, pareto_y, "b-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="blue", s=80, zorder=5,
                   edgecolors="black", linewidths=1)
        # Add batch size labels.
        for r in cp_pareto:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r["gen_output_tput_per_gpu"]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    if tp_pareto:
        pareto_x = [r["gen_output_tput_per_user"] for r in tp_pareto]
        pareto_y = [r["gen_output_tput_per_gpu"] for r in tp_pareto]
        ax.plot(pareto_x, pareto_y, color="orange", linestyle="-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="orange", s=80, zorder=5,
                   edgecolors="black", linewidths=1)
        # Add batch size labels.
        for r in tp_pareto:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r["gen_output_tput_per_gpu"]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    ax.set_xlabel("Output Throughput per User (tok/s)", fontsize=12)
    ax.set_ylabel("Output Throughput per GPU (tok/s/gpu)", fontsize=12)
    ax.set_title("Kimi K2 FP4 250k/8k TRTLLM Pareto, GB300 NVL72, Gen-Only SOL, #Gen GPUs Per Instance<=64", fontsize=10)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plot_path_full = os.path.join(output_dir, "pareto_plot_full.png")
    plt.savefig(plot_path_full, dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 2: Denoised - only Pareto frontier, one point per global BS
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    if cp_pareto_deduped:
        pareto_x = [r["gen_output_tput_per_user"] for r in cp_pareto_deduped]
        pareto_y = [r["gen_output_tput_per_gpu"] for r in cp_pareto_deduped]
        ax.plot(pareto_x, pareto_y, "b-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="blue", s=100, zorder=5,
                   edgecolors="black", linewidths=1.5,
                   label="KVP>1")
        # Add batch size labels.
        for r in cp_pareto_deduped:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r["gen_output_tput_per_gpu"]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    if tp_pareto_deduped:
        pareto_x = [r["gen_output_tput_per_user"] for r in tp_pareto_deduped]
        pareto_y = [r["gen_output_tput_per_gpu"] for r in tp_pareto_deduped]
        ax.plot(pareto_x, pareto_y, color="orange", linestyle="-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="orange", s=100, zorder=5,
                   edgecolors="black", linewidths=1.5,
                   label="KVP=1")
        # Add batch size labels.
        for r in tp_pareto_deduped:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r["gen_output_tput_per_gpu"]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    ax.set_xlabel("Output Throughput per User (tok/s)", fontsize=12)
    ax.set_ylabel("Output Throughput per GPU (tok/s/gpu)", fontsize=12)
    ax.set_title("Kimi K2 FP4 250k/8k TRTLLM Pareto, GB300 NVL72, Gen-Only SOL, #Gen GPUs Per Instance<=64 (Denoised)", fontsize=10)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plot_path_denoised = os.path.join(output_dir, "pareto_plot_denoised.png")
    plt.savefig(plot_path_denoised, dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Save Pareto frontier to CSV
    # =========================================================================
    pareto_csv_path = os.path.join(output_dir, "pareto_frontier.csv")

    # Build set of deduped job_dirs for quick lookup.
    cp_deduped_dirs = {r["job_dir"] for r in cp_pareto_deduped}
    tp_deduped_dirs = {r["job_dir"] for r in tp_pareto_deduped}

    pareto_rows = []

    # Add KVP>1 Pareto points.
    for r in cp_pareto:
        row = {
            "category": "KVP>1",
            "deduped": r["job_dir"] not in cp_deduped_dirs,
            "slurm_id": r["slurm_id"],
            "global_batch_size": r["global_batch_size"],
            "gen_tp": r["gen_tp"],
            "gen_cp": r["gen_cp"],
            "gen_pp": r["gen_pp"],
            "gen_ep": r["gen_ep"],
            "is_attn_dp": r["is_attn_dp"],
            "moe_backend": r["moe_backend"],
            "iter_median_tpot_ms": r["iter_median_tpot_ms"],
            "bench_median_tpot_ms": r["bench_median_tpot_ms"],
            "tpot_discrepancy_pct": r["tpot_discrepancy_pct"],
            "gen_output_tput_per_user": r["gen_output_tput_per_user"],
            "gen_output_tput_per_gpu": r["gen_output_tput_per_gpu"],
            "job_dir": r["job_dir"],
        }
        pareto_rows.append(row)

    # Add KVP=1 Pareto points.
    for r in tp_pareto:
        row = {
            "category": "KVP=1",
            "deduped": r["job_dir"] not in tp_deduped_dirs,
            "slurm_id": r["slurm_id"],
            "global_batch_size": r["global_batch_size"],
            "gen_tp": r["gen_tp"],
            "gen_cp": r["gen_cp"],
            "gen_pp": r["gen_pp"],
            "gen_ep": r["gen_ep"],
            "is_attn_dp": r["is_attn_dp"],
            "moe_backend": r["moe_backend"],
            "iter_median_tpot_ms": r["iter_median_tpot_ms"],
            "bench_median_tpot_ms": r["bench_median_tpot_ms"],
            "tpot_discrepancy_pct": r["tpot_discrepancy_pct"],
            "gen_output_tput_per_user": r["gen_output_tput_per_user"],
            "gen_output_tput_per_gpu": r["gen_output_tput_per_gpu"],
            "job_dir": r["job_dir"],
        }
        pareto_rows.append(row)

    # Write Pareto CSV.
    if pareto_rows:
        pareto_fieldnames = [
            "category",
            "deduped",
            "slurm_id",
            "global_batch_size",
            "gen_tp",
            "gen_cp",
            "gen_pp",
            "gen_ep",
            "is_attn_dp",
            "moe_backend",
            "iter_median_tpot_ms",
            "bench_median_tpot_ms",
            "tpot_discrepancy_pct",
            "gen_output_tput_per_user",
            "gen_output_tput_per_gpu",
            "job_dir",
        ]
        with open(pareto_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pareto_fieldnames)
            writer.writeheader()
            writer.writerows(pareto_rows)

    print(f"\nPareto plots saved:")
    print(f"  Full plot: {plot_path_full}")
    print(f"  Denoised plot: {plot_path_denoised}")
    print(f"  Pareto CSV: {pareto_csv_path}")
    print(f"  KVP>1 jobs: {len(cp_jobs)} total, {len(cp_pareto)} Pareto, {len(cp_pareto_deduped)} kept after dedupe")
    print(f"  KVP=1 jobs: {len(tp_only_jobs)} total, {len(tp_pareto)} Pareto, {len(tp_pareto_deduped)} kept after dedupe")

    return plot_path_full


def plot_profile_breakdown(results: List[Dict], output_dir: str) -> Optional[str]:
    """
    Plot stacked bar chart of component timing breakdown.
    Shows Attention, MoE, and other components for each configuration.
    """
    # Filter results with nsys data
    valid_results = [
        r for r in results
        if r.get("nsys_gpu_time_per_iter_ms") is not None
        and r.get("is_complete")
    ]

    if not valid_results:
        print("No valid results with nsys data for profile breakdown plot")
        return None

    # Sort by num_gpus and batch size for better visualization
    valid_results = sorted(valid_results, key=lambda r: (
        r.get("num_gpus") or 0,
        r.get("global_batch_size") or 0
    ))

    # Create labels
    labels = []
    for r in valid_results:
        tp = r.get("gen_tp", "?")
        cp = r.get("gen_cp", "?")
        bs = r.get("global_batch_size", "?")
        labels.append(f"TP{tp}_CP{cp}_BS{bs}")

    # Extract timing data
    attn_times = [r.get("nsys_attn_time_ms") or 0 for r in valid_results]
    moe_times = [r.get("nsys_moe_time_ms") or 0 for r in valid_results]

    # Calculate "other" as total - attn - moe
    other_times = []
    for r in valid_results:
        total = r.get("nsys_gpu_time_per_iter_ms") or 0
        attn = r.get("nsys_attn_time_ms") or 0
        moe = r.get("nsys_moe_time_ms") or 0
        other_times.append(max(0, total - attn - moe))

    fig, ax = plt.subplots(figsize=(max(10, len(valid_results) * 0.8), 7))

    x = range(len(labels))
    width = 0.6

    # Stacked bars
    ax.bar(x, attn_times, width, label="Attention", color="#2196F3")
    ax.bar(x, moe_times, width, bottom=attn_times, label="MoE", color="#FF9800")
    ax.bar(x, other_times, width,
           bottom=[a + m for a, m in zip(attn_times, moe_times)],
           label="Other", color="#9E9E9E")

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("GPU Time per Iteration (ms)", fontsize=12)
    ax.set_title("GPU Time Breakdown by Component", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "profile_breakdown.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Profile breakdown plot saved: {plot_path}")
    return plot_path


def plot_moe_breakdown(results: List[Dict], output_dir: str) -> Optional[str]:
    """
    Plot MoE component breakdown: communication vs compute percentage.
    """
    # Filter results with MoE data
    valid_results = [
        r for r in results
        if r.get("nsys_moe_comm_pct") is not None
        and r.get("nsys_moe_compute_pct") is not None
        and r.get("is_complete")
    ]

    if not valid_results:
        print("No valid results with MoE breakdown data")
        return None

    # Sort by num_gpus and batch size
    valid_results = sorted(valid_results, key=lambda r: (
        r.get("num_gpus") or 0,
        r.get("global_batch_size") or 0
    ))

    # Create labels
    labels = []
    for r in valid_results:
        tp = r.get("gen_tp", "?")
        cp = r.get("gen_cp", "?")
        bs = r.get("global_batch_size", "?")
        labels.append(f"TP{tp}_CP{cp}_BS{bs}")

    comm_pct = [r.get("nsys_moe_comm_pct") or 0 for r in valid_results]
    compute_pct = [r.get("nsys_moe_compute_pct") or 0 for r in valid_results]
    other_pct = [max(0, 100 - c - p) for c, p in zip(comm_pct, compute_pct)]

    fig, ax = plt.subplots(figsize=(max(10, len(valid_results) * 0.8), 7))

    x = range(len(labels))
    width = 0.6

    ax.bar(x, compute_pct, width, label="Expert Compute", color="#4CAF50")
    ax.bar(x, comm_pct, width, bottom=compute_pct, label="Communication (A2A)", color="#F44336")
    ax.bar(x, other_pct, width,
           bottom=[c + p for c, p in zip(compute_pct, comm_pct)],
           label="Other (routing, memset, etc.)", color="#9E9E9E")

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Percentage of MoE Time (%)", fontsize=12)
    ax.set_title("MoE Time Breakdown: Compute vs Communication", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "moe_breakdown.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"MoE breakdown plot saved: {plot_path}")
    return plot_path


def plot_comm_vs_throughput(results: List[Dict], output_dir: str) -> Optional[str]:
    """
    Plot communication overhead vs throughput.
    X-axis: Total communication percentage
    Y-axis: Output throughput per GPU
    """
    # Filter results with both metrics
    valid_results = [
        r for r in results
        if r.get("nsys_comm_total_pct") is not None
        and r.get("gen_output_tput_per_gpu") is not None
        and r.get("is_complete")
    ]

    if not valid_results:
        print("No valid results for comm vs throughput plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate by CP configuration
    cp_jobs = [r for r in valid_results if (r.get("gen_cp") or 1) > 1]
    tp_jobs = [r for r in valid_results if (r.get("gen_cp") or 1) == 1]

    if cp_jobs:
        x = [r["nsys_comm_total_pct"] for r in cp_jobs]
        y = [r["gen_output_tput_per_gpu"] for r in cp_jobs]
        ax.scatter(x, y, s=80, color="blue", alpha=0.7, label=f"CP jobs (CP>1) [{len(cp_jobs)}]",
                   edgecolors="black", linewidths=0.5)
        # Add labels
        for r in cp_jobs:
            ax.annotate(
                f"BS{r.get('global_batch_size', '?')}",
                (r["nsys_comm_total_pct"], r["gen_output_tput_per_gpu"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8
            )

    if tp_jobs:
        x = [r["nsys_comm_total_pct"] for r in tp_jobs]
        y = [r["gen_output_tput_per_gpu"] for r in tp_jobs]
        ax.scatter(x, y, s=80, color="orange", alpha=0.7, label=f"TP-only jobs (CP=1) [{len(tp_jobs)}]",
                   edgecolors="black", linewidths=0.5)
        for r in tp_jobs:
            ax.annotate(
                f"BS{r.get('global_batch_size', '?')}",
                (r["nsys_comm_total_pct"], r["gen_output_tput_per_gpu"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8
            )

    ax.set_xlabel("Total Communication Overhead (%)", fontsize=12)
    ax.set_ylabel("Output Throughput per GPU (tok/s/gpu)", fontsize=12)
    ax.set_title("Communication Overhead vs Throughput", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comm_vs_throughput.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comm vs throughput plot saved: {plot_path}")
    return plot_path


def plot_nsys_profiles(results: List[Dict], output_dir: str):
    """Generate all nsys profile comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating NSYS Profile Plots")
    print("=" * 60)

    plot_profile_breakdown(results, output_dir)
    plot_moe_breakdown(results, output_dir)
    plot_comm_vs_throughput(results, output_dir)


def write_csv(results: List[Dict], output_dir: str) -> str:
    """Write results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gen_only_results.csv")

    # Sort results by: num_gpus, global_batch_size, is_attn_dp, gen_tp, gen_cp.
    def sort_key(r):
        tp = r.get("gen_tp") or 0
        cp = r.get("gen_cp") or 0
        pp = r.get("gen_pp") or 0
        num_gpus = tp * cp * pp
        return (
            num_gpus,
            r.get("global_batch_size") or 0,
            1 if r.get("is_attn_dp") else 0,  # attnDP enabled first.
            tp,
            cp,
        )

    results = sorted(results, key=sort_key)

    fieldnames = [
        "slurm_id",
        "num_gpus",
        "global_batch_size",
        "is_attn_dp",
        "gen_tp",
        "gen_cp",
        "gen_pp",
        "gen_ep",
        "moe_backend",
        "is_complete",
        "iter_median_tpot_ms",      # TPOT from iteration logs (used for throughput)
        "bench_median_tpot_ms",     # TPOT from benchmark client (reference only)
        "tpot_discrepancy_pct",     # % difference: (bench - iter) / iter * 100
        "done_file_present",
        "saturation_pct",
        "saturated_iters",
        "total_iters",
        "first_saturated_iter",
        "last_saturated_iter",
        "gen_output_tput_per_user",
        "gen_output_tput_per_gpu",
        # NSYS profile columns
        "nsys_gpu_time_per_iter_ms",
        "nsys_attn_time_ms",
        "nsys_moe_time_ms",
        "nsys_moe_comm_pct",
        "nsys_moe_compute_pct",
        "nsys_helix_cp_time_ms",
        "nsys_comm_total_pct",
        "nsys_unknown_time_ms",
        "nsys_unknown_pct",
        "job_dir",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Parse gen-only benchmark job directories and extract metrics to CSV."
    )
    parser.add_argument(
        "--dir_list",
        nargs="+",
        required=True,
        help="List of paths; each can be a job directory or a parent containing job directories.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the CSV results file will be saved.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate Pareto frontier plot for CP vs TP-only jobs (default: True).",
    )
    parser.add_argument(
        "--analyze-nsys",
        action="store_true",
        default=False,
        help="Analyze nsys SQLite profiles and add timing columns to results.",
    )
    parser.add_argument(
        "--nsys-gpu-filter",
        choices=["gpu0", "all"],
        default="gpu0",
        help="Which GPUs to analyze: 'gpu0' (default) or 'all'.",
    )
    args = parser.parse_args()

    # Find all job directories.
    job_dirs = find_job_directories(args.dir_list)
    print(f"Found {len(job_dirs)} job directories to analyze")

    if args.analyze_nsys:
        print(f"NSYS analysis enabled (GPU filter: {args.nsys_gpu_filter})")

    if not job_dirs:
        print("No job directories found. Exiting.")
        return

    # Analyze each job.
    results = []
    for job_dir in job_dirs:
        print(f"Analyzing: {job_dir}")
        result = analyze_job(job_dir, analyze_nsys=args.analyze_nsys, gpu_filter=args.nsys_gpu_filter)
        results.append(result)

    # Write results to CSV.
    output_path = write_csv(results, args.output_dir)
    print(f"\nResults written to: {output_path}")
    print(f"Total jobs analyzed: {len(results)}")

    # Summary.
    complete_count = sum(1 for r in results if r["is_complete"])
    print(f"Completed jobs: {complete_count}/{len(results)}")

    # Generate Pareto plot if requested.
    if args.plot:
        plot_pareto(results, args.output_dir)

    # Generate nsys profile plots if requested.
    if args.analyze_nsys:
        nsys_count = sum(1 for r in results if r.get("nsys_gpu_time_per_iter_ms") is not None)
        print(f"\nJobs with nsys profile data: {nsys_count}/{len(results)}")
        if nsys_count > 0:
            plot_nsys_profiles(results, args.output_dir)


if __name__ == "__main__":
    main()
