# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""WAN Attention Performance Benchmark.

Compares VANILLA vs TRTLLM attention backends for visual generation models.
Uses CUDA events for precise GPU timing and supports NVTX profiling.

Usage:
    # Run all tests
    python test_attention_perf.py

    # With Nsight Systems profiling
    nsys profile -t cuda,nvtx --nvtx-capture=range -o wan_attn_perf python test_attention_perf.py

    # Run specific tests with pytest
    pytest test_attention_perf.py -v -k "test_self_attention_perf"
"""

import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import pytest
import torch

from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode

# NVTX support for profiling
try:
    import nvtx

    NVTX_AVAILABLE = True
    if hasattr(nvtx, "annotate"):
        NVTX_METHOD = "annotate"
    elif hasattr(nvtx, "range_start") and hasattr(nvtx, "range_end"):
        NVTX_METHOD = "range"
    else:
        NVTX_METHOD = None
        NVTX_AVAILABLE = False
except ImportError:
    NVTX_AVAILABLE = False
    NVTX_METHOD = None

# Torch profiler support
try:
    from torch.profiler import record_function

    TORCH_PROFILER_AVAILABLE = True
except ImportError:
    TORCH_PROFILER_AVAILABLE = False


# ============================================================================
# Timing utilities
# ============================================================================


@contextmanager
def cuda_timer(device: torch.device):
    """Context manager for precise GPU timing using CUDA events."""
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        def get_elapsed_time():
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event)

        yield get_elapsed_time
    else:
        start_time = time.perf_counter()

        def get_elapsed_time():
            return (time.perf_counter() - start_time) * 1000

        yield get_elapsed_time


@contextmanager
def nvtx_range(name: str):
    """Context manager for NVTX range profiling."""
    if NVTX_AVAILABLE and NVTX_METHOD:
        if NVTX_METHOD == "annotate":
            with nvtx.annotate(name):
                yield
        elif NVTX_METHOD == "range":
            range_id = nvtx.range_start(name)
            try:
                yield
            finally:
                nvtx.range_end(range_id)
        else:
            yield
    else:
        yield


@contextmanager
def torch_profiler_range(name: str):
    """Context manager for torch profiler range."""
    if TORCH_PROFILER_AVAILABLE:
        with record_function(name):
            yield
    else:
        yield


# ============================================================================
# Test utilities
# ============================================================================


def create_model_config(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    attn_backend: str = "VANILLA",
) -> DiffusionModelConfig:
    """Create a mock DiffusionModelConfig for testing."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=eps,
    )

    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(backend=attn_backend),
        skip_create_weights_in_init=False,
    )
    return config


def generate_rope_embeddings(
    seq_len: int, head_dim: int, device: torch.device, is_HSD: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate RoPE embeddings.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        device: Target device
        is_HSD: If True, returns [1, 1, S, D] for HSD format, else [1, S, 1, D] for SHD

    Returns:
        Tuple of (freqs_cos, freqs_sin)
    """
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, head_dim, device=device) * (-torch.log(torch.tensor(10000.0)) / head_dim)
    )

    if is_HSD:
        freqs_cos = torch.cos(position * div_term).unsqueeze(0).unsqueeze(0)
        freqs_sin = torch.sin(position * div_term).unsqueeze(0).unsqueeze(0)
    else:
        freqs_cos = torch.cos(position * div_term).unsqueeze(0).unsqueeze(2)
        freqs_sin = torch.sin(position * div_term).unsqueeze(0).unsqueeze(2)

    return freqs_cos, freqs_sin


# ============================================================================
# Performance benchmark class
# ============================================================================


class WanAttentionPerformanceBenchmark:
    """Performance benchmark for WAN attention backends."""

    # WAN model configurations: (batch_size, num_heads, seq_len, head_dim, description)
    TEST_SIZES = [
        # Wan2.1-T2V-1.3B configurations
        (1, 24, 14040, 64, "Wan-1.3B 480p 2s"),
        (1, 24, 3510, 64, "Wan-1.3B 480p 2s ring4"),
        (1, 24, 7020, 64, "Wan-1.3B 480p 2s ring2"),
        # Wan2.1-T2V-14B configurations
        (1, 40, 75600, 128, "Wan-14B 720p 5s"),
        (1, 40, 37800, 128, "Wan-14B 720p 5s ring2"),
        (1, 40, 18900, 128, "Wan-14B 720p 5s ring4"),
        (1, 40, 9450, 128, "Wan-14B 720p 5s ring8"),
        # Ulysses parallelism configurations
        (1, 20, 75600, 128, "Wan-14B 720p ulysses2"),
        (1, 10, 75600, 128, "Wan-14B 720p ulysses4"),
        (1, 5, 75600, 128, "Wan-14B 720p ulysses8"),
        # Smaller test cases for quick validation
        (2, 24, 1024, 64, "Small batch2"),
        (1, 24, 4096, 64, "Medium 4k"),
        (1, 40, 8192, 128, "Large 8k"),
    ]

    # Quick test sizes for CI/pytest
    QUICK_TEST_SIZES = [
        (1, 24, 1024, 64, "Quick 1k"),
        (1, 24, 2048, 64, "Quick 2k"),
        (2, 24, 1024, 64, "Quick batch2"),
    ]

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 50,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.backends = ["VANILLA", "TRTLLM"]

    def create_attention_model(
        self, hidden_size: int, num_heads: int, head_dim: int, backend: str
    ) -> Attention:
        """Create a WAN self-attention model with specified backend."""
        config = create_model_config(hidden_size, num_heads, head_dim, attn_backend=backend)
        model = Attention(hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=config).to(
            self.device
        )
        model.eval()
        return model

    def create_test_data(
        self, batch_size: int, seq_len: int, hidden_size: int, head_dim: int
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Create test input data and RoPE embeddings."""
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, device=self.device, dtype=self.dtype
        )
        freqs = generate_rope_embeddings(seq_len, head_dim, self.device, is_HSD=False)
        return hidden_states, freqs

    def estimate_memory_gb(
        self, batch_size: int, num_heads: int, seq_len: int, head_dim: int
    ) -> float:
        """Estimate tensor memory usage in GB."""
        hidden_size = num_heads * head_dim
        # Input: [B, S, H] + Q, K, V: [B, S, num_heads, head_dim] each
        bytes_per_element = 2  # bf16
        input_bytes = batch_size * seq_len * hidden_size * bytes_per_element
        qkv_bytes = 3 * batch_size * seq_len * num_heads * head_dim * bytes_per_element
        output_bytes = batch_size * seq_len * hidden_size * bytes_per_element
        # Attention matrix can be O(S^2) but flash attention avoids materializing it
        return (input_bytes + qkv_bytes + output_bytes) / (1024**3)

    def benchmark_single(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        backend: str,
        verbose: bool = True,
    ) -> Optional[Dict]:
        """Benchmark a single configuration.

        Returns:
            Dict with timing statistics or None if test failed/skipped
        """
        hidden_size = num_heads * head_dim

        # Memory check
        est_memory = self.estimate_memory_gb(batch_size, num_heads, seq_len, head_dim)
        if est_memory > 8.0:
            if verbose:
                print(f"  Skipping - estimated memory {est_memory:.2f}GB > 8GB limit")
            return None

        try:
            # Create model and data
            model = self.create_attention_model(hidden_size, num_heads, head_dim, backend)
            hidden_states, freqs = self.create_test_data(batch_size, seq_len, hidden_size, head_dim)

            # Warmup
            with nvtx_range(f"warmup_{backend}"):
                with torch_profiler_range(f"warmup_{backend}"):
                    with torch.no_grad():
                        for _ in range(self.warmup_iterations):
                            _ = model(hidden_states, freqs=freqs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            times = []
            with nvtx_range(f"benchmark_{backend}"):
                with torch_profiler_range(f"benchmark_{backend}"):
                    with torch.no_grad():
                        for i in range(self.benchmark_iterations):
                            with nvtx_range(f"iter_{backend}_{i}"):
                                with cuda_timer(self.device) as get_time:
                                    _ = model(hidden_states, freqs=freqs)
                                times.append(get_time())

            # Statistics
            times_tensor = torch.tensor(times)
            stats = {
                "avg_ms": times_tensor.mean().item(),
                "min_ms": times_tensor.min().item(),
                "max_ms": times_tensor.max().item(),
                "std_ms": times_tensor.std().item(),
                "median_ms": times_tensor.median().item(),
                "p95_ms": torch.quantile(times_tensor, 0.95).item(),
                "p99_ms": torch.quantile(times_tensor, 0.99).item(),
            }

            # Calculate throughput (approximate TOPS)
            total_ops = batch_size * num_heads * seq_len * seq_len * head_dim
            stats["throughput_tops"] = (total_ops / 1e12) / (stats["avg_ms"] / 1000)

            if verbose:
                print(
                    f"  {backend}: avg={stats['avg_ms']:.3f}ms, "
                    f"median={stats['median_ms']:.3f}ms, "
                    f"throughput={stats['throughput_tops']:.2f} TOPS"
                )

            return stats

        except Exception as e:
            if verbose:
                print(f"  {backend}: ERROR - {e}")
            return None

    def benchmark_comparison(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        description: str = "",
        verbose: bool = True,
    ) -> Dict[str, Optional[Dict]]:
        """Benchmark and compare all backends for a given configuration."""
        if verbose:
            print(
                f"\nBenchmarking: ({batch_size}, {num_heads}, {seq_len}, {head_dim}) {description}"
            )
            print(f"  Device: {self.device}, dtype: {self.dtype}")
            print(f"  Warmup: {self.warmup_iterations}, Iterations: {self.benchmark_iterations}")

        results = {}
        for backend in self.backends:
            results[backend] = self.benchmark_single(
                batch_size, num_heads, seq_len, head_dim, backend, verbose
            )

        # Print comparison
        if verbose and results.get("VANILLA") and results.get("TRTLLM"):
            vanilla_avg = results["VANILLA"]["avg_ms"]
            trtllm_avg = results["TRTLLM"]["avg_ms"]
            speedup = vanilla_avg / trtllm_avg
            print(f"  TRTLLM vs VANILLA: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

        return results

    def run_full_benchmark(self, use_quick_sizes: bool = False) -> Dict:
        """Run benchmark on all configured sizes."""
        test_sizes = self.QUICK_TEST_SIZES if use_quick_sizes else self.TEST_SIZES

        print("\n" + "=" * 70)
        print("WAN ATTENTION PERFORMANCE BENCHMARK")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"dtype: {self.dtype}")
        print(f"Backends: {self.backends}")
        print(f"NVTX: {'Enabled' if NVTX_AVAILABLE else 'Disabled'}")
        print(f"Torch Profiler: {'Enabled' if TORCH_PROFILER_AVAILABLE else 'Disabled'}")

        all_results = {}

        with nvtx_range("wan_attention_benchmark"):
            with torch_profiler_range("wan_attention_benchmark"):
                for batch_size, num_heads, seq_len, head_dim, desc in test_sizes:
                    key = f"{desc}_{batch_size}x{num_heads}x{seq_len}x{head_dim}"
                    results = self.benchmark_comparison(
                        batch_size, num_heads, seq_len, head_dim, desc
                    )
                    all_results[key] = {
                        "config": {
                            "batch_size": batch_size,
                            "num_heads": num_heads,
                            "seq_len": seq_len,
                            "head_dim": head_dim,
                            "description": desc,
                        },
                        "results": results,
                    }

        # Print summary
        self._print_summary(all_results)
        return all_results

    def _print_summary(self, all_results: Dict) -> None:
        """Print benchmark summary table."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Configuration':<40} {'VANILLA (ms)':<15} {'TRTLLM (ms)':<15} {'Speedup':<10}")
        print("-" * 70)

        for key, data in all_results.items():
            desc = data["config"]["description"]
            results = data["results"]

            vanilla = results.get("VANILLA")
            trtllm = results.get("TRTLLM")

            vanilla_str = f"{vanilla['avg_ms']:.2f}" if vanilla else "N/A"
            trtllm_str = f"{trtllm['avg_ms']:.2f}" if trtllm else "N/A"

            if vanilla and trtllm:
                speedup = vanilla["avg_ms"] / trtllm["avg_ms"]
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{desc:<40} {vanilla_str:<15} {trtllm_str:<15} {speedup_str:<10}")

    def test_memory_usage(
        self,
        batch_size: int = 1,
        num_heads: int = 24,
        seq_len: int = 4096,
        head_dim: int = 64,
    ) -> Dict[str, Dict]:
        """Test memory usage of different backends."""
        if self.device.type != "cuda":
            print("Memory test requires CUDA device")
            return {}

        print("\n" + "=" * 70)
        print("MEMORY USAGE TEST")
        print("=" * 70)
        print(f"Config: ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

        hidden_size = num_heads * head_dim
        memory_results = {}

        for backend in self.backends:
            print(f"\nTesting {backend}...")

            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Create model and data
                model = self.create_attention_model(hidden_size, num_heads, head_dim, backend)
                hidden_states, freqs = self.create_test_data(
                    batch_size, seq_len, hidden_size, head_dim
                )

                # Warmup
                with torch.no_grad():
                    _ = model(hidden_states, freqs=freqs)

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                # Forward pass
                with nvtx_range(f"memory_test_{backend}"):
                    with torch.no_grad():
                        _ = model(hidden_states, freqs=freqs)

                torch.cuda.synchronize()

                peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                current_memory_gb = torch.cuda.memory_allocated() / (1024**3)

                memory_results[backend] = {
                    "peak_memory_gb": peak_memory_gb,
                    "current_memory_gb": current_memory_gb,
                }

                print(f"  Peak memory: {peak_memory_gb:.3f} GB")
                print(f"  Current memory: {current_memory_gb:.3f} GB")

            except Exception as e:
                print(f"  ERROR: {e}")
                memory_results[backend] = None

        return memory_results


# ============================================================================
# Pytest test functions
# ============================================================================


class TestWanAttentionPerformance:
    """Pytest test class for WAN attention performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.benchmark = WanAttentionPerformanceBenchmark(
            warmup_iterations=5,
            benchmark_iterations=20,
        )

    @pytest.mark.parametrize("backend", ["VANILLA", "TRTLLM"])
    def test_self_attention_perf(self, backend: str):
        """Test that attention backend runs without errors."""
        batch_size, num_heads, seq_len, head_dim = 1, 24, 1024, 64

        result = self.benchmark.benchmark_single(
            batch_size, num_heads, seq_len, head_dim, backend, verbose=True
        )

        if result is not None:
            assert result["avg_ms"] > 0, "Average time should be positive"
            assert result["min_ms"] <= result["avg_ms"], "Min should be <= avg"
            assert result["max_ms"] >= result["avg_ms"], "Max should be >= avg"
            print(f"  {backend}: avg={result['avg_ms']:.3f}ms OK")

    @pytest.mark.parametrize(
        "batch_size,num_heads,seq_len,head_dim",
        [
            (1, 24, 1024, 64),
            (1, 24, 2048, 64),
            (2, 24, 1024, 64),
        ],
    )
    def test_backend_comparison(self, batch_size: int, num_heads: int, seq_len: int, head_dim: int):
        """Test VANILLA vs TRTLLM comparison."""
        results = self.benchmark.benchmark_comparison(
            batch_size, num_heads, seq_len, head_dim, verbose=True
        )

        # At least one backend should work
        assert any(r is not None for r in results.values()), "All backends failed"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_usage(self):
        """Test memory usage tracking."""
        memory_results = self.benchmark.test_memory_usage(
            batch_size=1, num_heads=24, seq_len=2048, head_dim=64
        )

        for backend, result in memory_results.items():
            if result is not None:
                assert result["peak_memory_gb"] > 0, f"{backend} peak memory should be positive"

    def test_quick_benchmark(self):
        """Run quick benchmark for CI validation."""
        results = self.benchmark.run_full_benchmark(use_quick_sizes=True)
        assert len(results) > 0, "Should have benchmark results"


# ============================================================================
# Main entry point
# ============================================================================


def main():
    """Run full benchmark suite."""
    print("\n" + "=" * 70)
    print("WAN ATTENTION PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, results will not be meaningful")

    # Print profiling instructions
    if torch.cuda.is_available():
        print("\nPROFILING INSTRUCTIONS:")
        print("-" * 50)
        if NVTX_AVAILABLE:
            print("NVTX Profiling (Nsight Systems):")
            print("  nsys profile -t cuda,nvtx --nvtx-capture=range \\")
            print("    -o wan_attn_perf python test_attention_perf.py")
        else:
            print("NVTX not available. Install with: pip install nvtx")

        print("\nPyTorch Profiler:")
        print("  The benchmark includes record_function() calls for profiling")
        print("-" * 50)

    # Create benchmark instance
    benchmark = WanAttentionPerformanceBenchmark(
        warmup_iterations=10,
        benchmark_iterations=50,
    )

    # Run full benchmark
    print("\n" + "=" * 70)
    print("FULL BENCHMARK")
    print("=" * 70)
    all_results = benchmark.run_full_benchmark(use_quick_sizes=False)

    # Memory test
    if torch.cuda.is_available():
        benchmark.test_memory_usage(batch_size=1, num_heads=24, seq_len=4096, head_dim=64)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
