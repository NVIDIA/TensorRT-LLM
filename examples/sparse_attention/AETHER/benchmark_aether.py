# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
AETHER Benchmark Suite
======================

Comprehensive benchmarking for AETHER sparse attention kernels.

This script provides:
    1. Latency benchmarks for all kernel variants
    2. Quality metrics beyond cosine similarity
    3. End-to-end accuracy impact evaluation
    4. Sparsity vs. quality tradeoff analysis

Usage:
    python benchmark_aether.py --batch_size 4 --seq_len 4096 --verify
    python benchmark_aether.py --all-variants --export-csv results.csv
    python benchmark_aether.py --perplexity-eval --model llama-7b
"""

import argparse
import time
import json
import csv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkConfig:
    """Configuration for AETHER benchmarks."""
    batch_size: int = 4
    num_heads: int = 32
    head_dim: int = 128
    seq_len: int = 4096
    block_size: int = 64
    num_warmup: int = 10
    num_iterations: int = 100
    device: str = "cuda"
    dtype: str = "float16"
    threshold: float = 0.15
    target_sparsity: float = 0.8
    top_k: int = 32


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    variant: str
    latency_ms: float
    latency_std_ms: float
    sparsity: float
    throughput_gflops: float
    
    # Quality metrics
    cosine_similarity: float = 0.0
    l2_distance: float = 0.0
    max_absolute_error: float = 0.0
    relative_error: float = 0.0
    
    # Block selection accuracy
    true_positive_rate: float = 0.0  # % of important blocks correctly kept
    false_positive_rate: float = 0.0  # % of unimportant blocks incorrectly kept
    precision: float = 0.0  # precision of block selection
    recall: float = 0.0  # recall of block selection
    
    # Advanced metrics
    kl_divergence: float = 0.0  # KL divergence of attention distributions
    attention_entropy_diff: float = 0.0  # Change in attention entropy
    
    # Memory metrics
    memory_saved_mb: float = 0.0
    theoretical_speedup: float = 0.0
    
    config: Dict = field(default_factory=dict)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


def setup_test_data(config: BenchmarkConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate test data for benchmarking."""
    dtype = get_dtype(config.dtype)
    device = torch.device(config.device)
    
    query = torch.randn(
        config.batch_size, config.num_heads, config.head_dim,
        device=device, dtype=dtype
    )
    keys = torch.randn(
        config.batch_size, config.num_heads, config.seq_len, config.head_dim,
        device=device, dtype=dtype
    )
    values = torch.randn(
        config.batch_size, config.num_heads, config.seq_len, config.head_dim,
        device=device, dtype=dtype
    )
    
    return query, keys, values


def compute_dense_attention(
    query: torch.Tensor, 
    keys: torch.Tensor, 
    values: torch.Tensor,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute dense (full) attention as ground truth."""
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    
    # query: (B, H, D), keys: (B, H, S, D)
    # Expand query for matmul: (B, H, 1, D)
    query_expanded = query.unsqueeze(2)
    
    # Attention scores: (B, H, 1, S)
    attn_scores = torch.matmul(query_expanded, keys.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Output: (B, H, 1, D) -> (B, H, D)
    output = torch.matmul(attn_weights, values).squeeze(2)
    
    return output, attn_weights.squeeze(2)


def compute_sparse_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sparse attention using block mask."""
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    
    B, H, S, D = keys.shape
    N_blocks = S // block_size
    
    # Expand block mask to token level: (B, H, N_blocks) -> (B, H, S)
    token_mask = block_mask.unsqueeze(-1).expand(-1, -1, -1, block_size)
    token_mask = token_mask.reshape(B, H, S)
    
    # query: (B, H, 1, D), keys: (B, H, S, D)
    query_expanded = query.unsqueeze(2)
    
    # Compute all attention scores
    attn_scores = torch.matmul(query_expanded, keys.transpose(-2, -1)) * scale
    attn_scores = attn_scores.squeeze(2)  # (B, H, S)
    
    # Apply mask (set masked positions to -inf)
    masked_scores = attn_scores.masked_fill(~token_mask, float('-inf'))
    
    # Softmax over non-masked positions
    attn_weights = F.softmax(masked_scores, dim=-1)
    attn_weights = attn_weights.nan_to_num(0.0)  # Handle all-masked case
    
    # Output
    output = torch.matmul(attn_weights.unsqueeze(2), values).squeeze(2)
    
    return output, attn_weights


def compute_quality_metrics(
    sparse_output: torch.Tensor,
    dense_output: torch.Tensor,
    sparse_weights: torch.Tensor,
    dense_weights: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
) -> Dict[str, float]:
    """Compute comprehensive quality metrics."""
    metrics = {}
    
    # 1. Output similarity metrics
    # Cosine similarity
    cos_sim = F.cosine_similarity(
        sparse_output.flatten(), 
        dense_output.flatten(), 
        dim=0
    ).item()
    metrics["cosine_similarity"] = cos_sim
    
    # L2 distance (normalized)
    l2_dist = torch.norm(sparse_output - dense_output) / torch.norm(dense_output)
    metrics["l2_distance"] = l2_dist.item()
    
    # Max absolute error
    max_abs_err = torch.max(torch.abs(sparse_output - dense_output))
    metrics["max_absolute_error"] = max_abs_err.item()
    
    # Relative error
    rel_err = torch.mean(torch.abs(sparse_output - dense_output) / (torch.abs(dense_output) + 1e-8))
    metrics["relative_error"] = rel_err.item()
    
    # 2. Attention distribution metrics
    # KL divergence (sparse || dense)
    # Clamp to avoid log(0)
    sparse_weights_clamped = sparse_weights.clamp(min=1e-10)
    dense_weights_clamped = dense_weights.clamp(min=1e-10)
    kl_div = F.kl_div(
        sparse_weights_clamped.log(), 
        dense_weights_clamped,
        reduction='batchmean'
    )
    metrics["kl_divergence"] = kl_div.item()
    
    # Attention entropy difference
    def entropy(weights):
        weights_clamped = weights.clamp(min=1e-10)
        return -torch.sum(weights_clamped * weights_clamped.log(), dim=-1).mean()
    
    sparse_entropy = entropy(sparse_weights)
    dense_entropy = entropy(dense_weights)
    metrics["attention_entropy_diff"] = (sparse_entropy - dense_entropy).item()
    
    # 3. Block selection accuracy (using ground truth attention)
    B, H, S = dense_weights.shape
    N_blocks = S // block_size
    
    # Compute per-block attention mass in dense attention
    dense_weights_blocked = dense_weights.view(B, H, N_blocks, block_size)
    block_importance = dense_weights_blocked.sum(dim=-1)  # (B, H, N_blocks)
    
    # Define "important" blocks as those with above-median attention mass
    median_importance = block_importance.median(dim=-1, keepdim=True).values
    important_blocks = block_importance > median_importance
    
    # Calculate TP, FP, TN, FN
    tp = (block_mask & important_blocks).float().sum()
    fp = (block_mask & ~important_blocks).float().sum()
    fn = (~block_mask & important_blocks).float().sum()
    tn = (~block_mask & ~important_blocks).float().sum()
    
    # Precision and Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    metrics["true_positive_rate"] = recall.item()
    metrics["false_positive_rate"] = (fp / (fp + tn + 1e-8)).item()
    metrics["precision"] = precision.item()
    metrics["recall"] = recall.item()
    
    return metrics


def benchmark_kernel(
    config: BenchmarkConfig,
    kernel_func,
    kernel_name: str,
    metadata: Tuple[torch.Tensor, ...],
    **kernel_kwargs
) -> BenchmarkResult:
    """Benchmark a specific kernel variant."""
    query, keys, values = setup_test_data(config)
    
    # Warmup
    for _ in range(config.num_warmup):
        mask, scores = kernel_func(query, *metadata, **kernel_kwargs)
        torch.cuda.synchronize()
    
    # Timing
    latencies = []
    for _ in range(config.num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        mask, scores = kernel_func(query, *metadata, **kernel_kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    latency_ms = sum(latencies) / len(latencies)
    latency_std = (sum((l - latency_ms)**2 for l in latencies) / len(latencies)) ** 0.5
    
    # Compute sparsity
    sparsity = 1.0 - mask.float().mean().item()
    
    # Compute quality metrics against dense attention
    dense_output, dense_weights = compute_dense_attention(query, keys, values)
    sparse_output, sparse_weights = compute_sparse_attention(
        query, keys, values, mask, config.block_size
    )
    
    quality_metrics = compute_quality_metrics(
        sparse_output, dense_output,
        sparse_weights, dense_weights,
        mask, config.block_size
    )
    
    # Compute throughput (approximate FLOPs for attention)
    B, H, S = config.batch_size, config.num_heads, config.seq_len
    D = config.head_dim
    flops_dense = 2 * B * H * S * D + 2 * B * H * S * D  # QK^T + Attn*V
    flops_sparse = flops_dense * (1 - sparsity)
    throughput = flops_sparse / (latency_ms * 1e-3) / 1e9  # GFLOPS
    
    # Memory savings
    memory_full = B * H * S * D * 2  # 2 bytes for fp16
    memory_sparse = memory_full * (1 - sparsity)
    memory_saved_mb = (memory_full - memory_sparse) / 1e6
    
    return BenchmarkResult(
        variant=kernel_name,
        latency_ms=latency_ms,
        latency_std_ms=latency_std,
        sparsity=sparsity,
        throughput_gflops=throughput,
        memory_saved_mb=memory_saved_mb,
        theoretical_speedup=1.0 / (1.0 - sparsity + 1e-8),
        config=asdict(config),
        **quality_metrics
    )


def run_all_benchmarks(config: BenchmarkConfig, verify: bool = True) -> List[BenchmarkResult]:
    """Run benchmarks for all AETHER kernel variants."""
    results = []
    
    try:
        from tensorrt_llm._torch.attention_backend.sparse.aether_kernels import (
            run_aether_sparse, precompute_metadata
        )
    except ImportError:
        print("Warning: Could not import new unified kernels, trying legacy imports...")
        from tensorrt_llm.kernels.triton.adaptive_event_attention import (
            run_aether_v2, run_aether_v3, run_aether_v3_causal,
            run_aether_adaptive, run_aether_v3_adaptive,
            precompute_block_metadata, precompute_enhanced_metadata
        )
        
        # Setup test data
        query, keys, values = setup_test_data(config)
        
        # Precompute metadata
        means, variances, radii = precompute_block_metadata(keys, config.block_size)
        means_v3, radii_v3, variances_v3, concentrations = precompute_enhanced_metadata(
            keys, config.block_size
        )
        
        # Benchmark v2 (variance-aware)
        print("Benchmarking AETHER v2 (variance-aware)...")
        result = benchmark_kernel(
            config,
            run_aether_v2,
            "aether_v2_variance",
            (means, variances, radii),
            threshold=config.threshold,
            use_variance=True,
            block_size=config.block_size
        )
        results.append(result)
        
        # Benchmark v3 (tight bound)
        print("Benchmarking AETHER v3 (tight bound)...")
        result = benchmark_kernel(
            config,
            run_aether_v3,
            "aether_v3_tight",
            (means_v3, radii_v3, concentrations),
            threshold=config.threshold
        )
        results.append(result)
        
        # Benchmark v3 causal
        print("Benchmarking AETHER v3 Causal...")
        result = benchmark_kernel(
            config,
            run_aether_v3_causal,
            "aether_v3_causal",
            (means_v3, radii_v3, concentrations),
            threshold=config.threshold,
            local_window=4,
            recency_decay=0.95
        )
        results.append(result)
        
        # Benchmark adaptive
        print("Benchmarking AETHER v3 Adaptive...")
        
        def adaptive_wrapper(query, means, radii, concentrations, **kwargs):
            return run_aether_v3_adaptive(
                query, means, radii, concentrations,
                target_sparsity=config.target_sparsity
            )[:2]
        
        result = benchmark_kernel(
            config,
            adaptive_wrapper,
            "aether_v3_adaptive",
            (means_v3, radii_v3, concentrations),
        )
        results.append(result)
        
        return results
    
    # Using new unified kernels
    query, keys, values = setup_test_data(config)
    
    # Precompute metadata
    means, radii, variances, concentrations = precompute_metadata(
        keys, config.block_size, 
        compute_variance=True, 
        compute_concentration=True
    )
    
    # Benchmark variants
    variants = [
        ("aether_basic", {"use_variance": False, "use_concentration": False, "is_causal": False}),
        ("aether_variance", {"use_variance": True, "use_concentration": False, "is_causal": False}),
        ("aether_tight", {"use_variance": False, "use_concentration": True, "is_causal": False}),
        ("aether_full", {"use_variance": True, "use_concentration": True, "is_causal": False}),
        ("aether_causal", {"use_variance": False, "use_concentration": True, "is_causal": True}),
    ]
    
    for name, kwargs in variants:
        print(f"Benchmarking {name}...")
        result = benchmark_kernel(
            config,
            run_aether_sparse,
            name,
            (means, radii),
            block_variances=variances if kwargs.get("use_variance") else None,
            block_concentrations=concentrations if kwargs.get("use_concentration") else None,
            threshold=config.threshold,
            **kwargs
        )
        results.append(result)
    
    return results


def print_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 120)
    print("AETHER BENCHMARK RESULTS")
    print("=" * 120)
    
    # Header
    headers = [
        "Variant", "Latency(ms)", "Sparsity", "CosSim", "L2Dist", 
        "Precision", "Recall", "KL-Div", "Speedup", "Mem Saved(MB)"
    ]
    header_fmt = "{:<20} {:>12} {:>10} {:>8} {:>8} {:>10} {:>8} {:>8} {:>8} {:>14}"
    print(header_fmt.format(*headers))
    print("-" * 120)
    
    # Results
    row_fmt = "{:<20} {:>12.3f} {:>10.2%} {:>8.4f} {:>8.4f} {:>10.4f} {:>8.4f} {:>8.4f} {:>8.2f}x {:>14.2f}"
    for r in results:
        print(row_fmt.format(
            r.variant,
            r.latency_ms,
            r.sparsity,
            r.cosine_similarity,
            r.l2_distance,
            r.precision,
            r.recall,
            r.kl_divergence,
            r.theoretical_speedup,
            r.memory_saved_mb
        ))
    
    print("=" * 120)
    
    # Summary statistics
    print("\nSUMMARY:")
    best_quality = max(results, key=lambda r: r.cosine_similarity)
    best_speedup = max(results, key=lambda r: r.theoretical_speedup)
    best_balance = max(results, key=lambda r: r.cosine_similarity * r.theoretical_speedup)
    
    print(f"  Best Quality:    {best_quality.variant} (CosSim: {best_quality.cosine_similarity:.4f})")
    print(f"  Best Speedup:    {best_speedup.variant} (Speedup: {best_speedup.theoretical_speedup:.2f}x)")
    print(f"  Best Balance:    {best_balance.variant} (Quality×Speedup score)")


def export_results(results: List[BenchmarkResult], filepath: str) -> None:
    """Export results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        fieldnames = [
            'variant', 'latency_ms', 'latency_std_ms', 'sparsity', 'throughput_gflops',
            'cosine_similarity', 'l2_distance', 'max_absolute_error', 'relative_error',
            'true_positive_rate', 'false_positive_rate', 'precision', 'recall',
            'kl_divergence', 'attention_entropy_diff', 'memory_saved_mb', 'theoretical_speedup'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(row)
    print(f"Results exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="AETHER Sparse Attention Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic benchmark
    python benchmark_aether.py --batch_size 4 --seq_len 4096
    
    # Full evaluation with verification
    python benchmark_aether.py --verify --all-variants
    
    # Export results
    python benchmark_aether.py --export-csv results.csv
    
    # Sweep sparsity levels
    python benchmark_aether.py --sparsity-sweep 0.5 0.6 0.7 0.8 0.9
        """
    )
    
    # Config arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for sparse attention")
    parser.add_argument("--num_warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num_iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--threshold", type=float, default=0.15, help="Attention potential threshold")
    parser.add_argument("--target_sparsity", type=float, default=0.8, help="Target sparsity for adaptive mode")
    
    # Benchmark options
    parser.add_argument("--verify", action="store_true", help="Run verification against dense attention")
    parser.add_argument("--all-variants", action="store_true", help="Benchmark all kernel variants")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV file")
    parser.add_argument("--sparsity-sweep", nargs="+", type=float, help="Sweep over multiple sparsity levels")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        block_size=args.block_size,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        dtype=args.dtype,
        threshold=args.threshold,
        target_sparsity=args.target_sparsity,
    )
    
    print(f"AETHER Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: B={config.batch_size}, H={config.num_heads}, D={config.head_dim}, S={config.seq_len}")
    print(f"Block size: {config.block_size}, Device: {config.device}, Dtype: {config.dtype}")
    print()
    
    # Run benchmarks
    if args.sparsity_sweep:
        all_results = []
        for sparsity in args.sparsity_sweep:
            config.target_sparsity = sparsity
            print(f"\n--- Sparsity: {sparsity:.0%} ---")
            results = run_all_benchmarks(config, verify=args.verify)
            all_results.extend(results)
        results = all_results
    else:
        results = run_all_benchmarks(config, verify=args.verify)
    
    # Print results
    print_results(results)
    
    # Export if requested
    if args.export_csv:
        export_results(results, args.export_csv)


if __name__ == "__main__":
    main()
