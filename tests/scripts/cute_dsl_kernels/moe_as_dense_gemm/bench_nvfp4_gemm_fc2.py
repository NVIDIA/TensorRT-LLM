# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark script for nvfp4_gemm (FC2-like shapes)
# Uses torch.ops.trtllm.nvfp4_gemm with autotune and CUPTI profiling
#
# Usage:
#   # Run all FC2 configs with default settings (logs to bench_nvfp4_gemm_fc2.log):
#   python bench_nvfp4_gemm_fc2.py
#
#   # Run single config:
#   python bench_nvfp4_gemm_fc2.py --mnk 64,7168,65536
#
#   # Disable logging to file:
#   python bench_nvfp4_gemm_fc2.py --no_log_file

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path to import testing module
sys.path.insert(0, str(Path(__file__).parent.parent))
from testing import CuptiProfiler

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.autotuner import autotune

# Script directory for default paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Default file paths (similar to the original shell script)
DEFAULT_LOG_FILE = SCRIPT_DIR / "bench_nvfp4_gemm_fc2.log"
DEFAULT_CACHE_FILE = SCRIPT_DIR / "autotune_cache_nvfp4_gemm_fc2.json"


class TeeOutput:
    """Tee stdout to both console and a file (like shell's tee command)."""

    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def create_nvfp4_tensors(m: int, n: int, k: int, device: str = "cuda"):
    """Create FP4 quantized tensors for nvfp4_gemm benchmark.

    Args:
        m: M dimension (batch/tokens)
        n: N dimension (output features)
        k: K dimension (input features)
        device: Device to create tensors on

    Returns:
        Tuple of (act_fp4, weight_fp4, act_sf, weight_sf, alpha, global_scale)
    """
    sf_vec_size = 16

    # Create random activation tensor and quantize it
    activation = torch.randn(m, k, dtype=torch.bfloat16, device=device)

    # Calculate global scale for activation
    global_scale = torch.ops.trtllm.calculate_nvfp4_global_scale(activation, None)

    # Quantize activation to FP4
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
        activation, global_scale, sf_vec_size, False, True
    )

    # Create pre-quantized weight tensor (simulating loaded weights)
    # Weight shape: [n, k//2] (packed FP4)
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device)

    # Create weight scale factors
    # Weight scale shape depends on the swizzled layout
    num_weight_sf_blocks = n * (k // sf_vec_size)
    # Pad to multiple of 128*4 as required by nvfp4_gemm
    num_weight_sf_blocks = ((num_weight_sf_blocks + 128 * 4 - 1) // (128 * 4)) * (128 * 4)
    weight_sf = torch.randint(0, 256, (num_weight_sf_blocks,), dtype=torch.uint8, device=device)

    # Alpha scaling factor
    alpha = torch.tensor([1.0], dtype=torch.float32, device=device)

    return act_fp4, weight_fp4, act_sf, weight_sf, alpha, global_scale


def benchmark_nvfp4_gemm(
    m: int,
    n: int,
    k: int,
    warmup_iterations: int = 10,
    iterations: int = 50,
    allowed_backends: str = "cutlass,cublaslt,cutedsl,cuda_core",
    output_dtype: torch.dtype = torch.bfloat16,
    cache_path: str = None,
):
    """Benchmark nvfp4_gemm with CUPTI profiling.

    Args:
        m: M dimension
        n: N dimension
        k: K dimension
        warmup_iterations: Number of warmup iterations
        iterations: Number of benchmark iterations
        allowed_backends: Comma-separated list of backends for autotune
        output_dtype: Output data type
        cache_path: Path to save/load autotune cache (optional)

    Returns:
        Average execution time in microseconds
    """
    print(f"Running nvfp4_gemm benchmark with M={m}, N={n}, K={k}")
    print(f"Allowed backends: {allowed_backends}")
    print(f"Output dtype: {output_dtype}")

    # Create tensors
    act_fp4, weight_fp4, act_sf, weight_sf, alpha, _ = create_nvfp4_tensors(m, n, k)

    print(f"act_fp4 shape: {act_fp4.shape}, weight_fp4 shape: {weight_fp4.shape}")
    print(f"act_sf shape: {act_sf.shape}, weight_sf shape: {weight_sf.shape}")

    # Warmup runs with autotune enabled to find the best backend/tactic
    print(f"Running {warmup_iterations} warmup iterations with autotune enabled...")
    if cache_path:
        print(f"Autotune cache path: {cache_path}")
    with autotune(tune_mode=True, cache_path=cache_path):
        for _ in range(warmup_iterations):
            output = torch.ops.trtllm.nvfp4_gemm(
                act_fp4,
                weight_fp4,
                act_sf,
                weight_sf,
                alpha,
                output_dtype,
                False,  # to_userbuffers
                allowed_backends,
            )
    torch.cuda.synchronize()

    # Benchmark with CUPTI
    print(f"Running {iterations} benchmark iterations with CUPTI...")
    profiler = CuptiProfiler()
    profiler.start()

    for _ in range(iterations):
        output = torch.ops.trtllm.nvfp4_gemm(
            act_fp4,
            weight_fp4,
            act_sf,
            weight_sf,
            alpha,
            output_dtype,
            False,  # to_userbuffers
            allowed_backends,
        )

    torch.cuda.synchronize()
    profiler.stop()

    total_time_ms = profiler.get_duration()
    avg_time_us = (total_time_ms / iterations) * 1000

    print(f"Output shape: {output.shape}")
    print(f"Total time: {total_time_ms:.3f} ms")
    print(f"Average time per iteration: {avg_time_us:.2f} us")

    return avg_time_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark nvfp4_gemm for FC2-like shapes")

    parser.add_argument(
        "--mnk",
        type=str,
        default=None,
        help="M,N,K dimensions (comma-separated). If not specified, runs all FC2 configs.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--allowed_backends",
        type=str,
        default="cutlass,cublaslt,cutedsl,cuda_core",
        help="Comma-separated list of backends for autotune (cutlass,cublaslt,cutedsl,cuda_core)",
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Output data type",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=str(DEFAULT_CACHE_FILE),
        help=f"Path to save/load autotune cache (default: {DEFAULT_CACHE_FILE})",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=str(DEFAULT_LOG_FILE),
        help=f"Path to save benchmark log (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--no_log_file",
        action="store_true",
        help="Disable logging to file (only print to console)",
    )

    args = parser.parse_args()

    # Setup logging to file if enabled
    tee = None
    if not args.no_log_file:
        tee = TeeOutput(args.log_file)
        sys.stdout = tee
        print(f"Logging to: {args.log_file}")

    output_dtype = torch.float16 if args.output_dtype == "float16" else torch.bfloat16

    if args.mnk:
        # Single run with specified dimensions
        m, n, k = [int(x.strip()) for x in args.mnk.split(",")]
        benchmark_nvfp4_gemm(
            m,
            n,
            k,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            allowed_backends=args.allowed_backends,
            output_dtype=output_dtype,
            cache_path=args.cache_path,
        )
    else:
        # Run all FC2 configurations
        # FC2 parameters from bench_moe_as_dense_gemm.sh
        FC2_K = 65536
        FC2_N = 7168

        # Generate M values: 1, 2, 4, 8, 16, 32, 48, 64, 80, ..., 512
        M_VALUES = [1, 2, 4, 8, 16] + list(range(32, 513, 16))

        print("=" * 60)
        print("===== nvfp4_gemm FC2 Benchmark =====")
        print("=" * 60)
        print(f"N={FC2_N}, K={FC2_K}")
        print(f"M values: {M_VALUES}")
        print(f"Allowed backends: {args.allowed_backends}")
        print(f"Autotune cache: {args.cache_path}")
        print("=" * 60)

        results = []
        for m in M_VALUES:
            print(f"\n--- M={m} ---")
            try:
                avg_time = benchmark_nvfp4_gemm(
                    m,
                    FC2_N,
                    FC2_K,
                    warmup_iterations=args.warmup_iterations,
                    iterations=args.iterations,
                    allowed_backends=args.allowed_backends,
                    output_dtype=output_dtype,
                    cache_path=args.cache_path,
                )
                results.append((m, avg_time))
            except Exception as e:
                print(f"Error for M={m}: {e}")
                results.append((m, float("nan")))

        # Print summary
        print("\n" + "=" * 60)
        print("Summary (nvfp4_gemm FC2)")
        print("=" * 60)
        print(f"{'M':>6} | {'Time (us)':>12}")
        print("-" * 22)
        for m, time_us in results:
            print(f"{m:>6} | {time_us:>12.2f}")

        print("\nBenchmark complete!")

    # Cleanup logging
    if tee is not None:
        sys.stdout = tee.terminal
        tee.close()
        print(f"Results saved to: {args.log_file}")


if __name__ == "__main__":
    main()
