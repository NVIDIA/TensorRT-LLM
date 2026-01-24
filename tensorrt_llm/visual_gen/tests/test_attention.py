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

import time
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from visual_gen.layers.attention import ditAttnProcessor

# Add NVTX support for performance profiling
try:
    import nvtx

    NVTX_AVAILABLE = True
    # Test which NVTX API is available
    if hasattr(nvtx, "annotate"):
        NVTX_METHOD = "annotate"
    elif hasattr(nvtx, "range_start") and hasattr(nvtx, "range_end"):
        NVTX_METHOD = "range"
    elif hasattr(nvtx, "mark"):
        NVTX_METHOD = "mark"
    else:
        NVTX_METHOD = None
        NVTX_AVAILABLE = False
        print("NVTX imported but no compatible API found")
except ImportError:
    NVTX_AVAILABLE = False
    NVTX_METHOD = None
    print("NVTX not available, skipping NVTX profiling")

try:
    from torch.profiler import record_function

    TORCH_PROFILER_AVAILABLE = True
except ImportError:
    TORCH_PROFILER_AVAILABLE = False
    print("torch.profiler not available, skipping torch profiler")

try:
    from visual_gen.configs.op_manager import AttentionOpManager
    from visual_gen.configs.parallel import DiTParallelConfig
    from visual_gen.configs.pipeline import PipelineConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    raise


@contextmanager
def cuda_timer(device):
    """Context manager for precise GPU timing using CUDA events"""
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Record start event
        start_event.record()

        def get_elapsed_time():
            # Record end event and wait for completion
            end_event.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            return start_event.elapsed_time(end_event)

        yield get_elapsed_time

    else:
        # Fallback to CPU timing for non-CUDA devices
        start_time = time.perf_counter()

        def get_elapsed_time():
            return (time.perf_counter() - start_time) * 1000  # Convert to milliseconds

        yield get_elapsed_time


@contextmanager
def nvtx_range(name):
    """Context manager for NVTX range profiling"""
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
        elif NVTX_METHOD == "mark":
            nvtx.mark(name)
            yield
        else:
            yield
    else:
        yield


@contextmanager
def torch_profiler_range(name):
    """Context manager for torch profiler range"""
    if TORCH_PROFILER_AVAILABLE:
        with record_function(name):
            yield
    else:
        yield


class TestditAttentionPerformance:
    """Performance tests for ditAttention with different attention implementations"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment for pytest"""
        self._init_test_environment()

    def _init_test_environment(self):
        """Initialize test environment - can be called directly or via pytest fixture"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

        # Initialize pipeline and parallel configs
        PipelineConfig.current_dit_block_id = 0
        PipelineConfig.num_dit_layers = 28

        # Initialize DiTParallelConfig for non-distributed testing
        try:
            DiTParallelConfig.set_config(
                tp_size=1, ulysses_size=1, ring_size=1, dp_size=1, cfg_size=1, fsdp_size=1
            )
        except Exception as e:
            print(f"Warning: Could not initialize DiTParallelConfig: {e}")
            print("Tests will run without parallel configuration")

        # Test attention sizes: (batch_size, num_heads, seq_len, head_dim)
        self.test_sizes = [
            (1, 24, 14040, 64),  # Wan2.1-T2V-1.3B, 480p, 2s
            (1, 24, 3510, 64),  # Wan2.1-T2V-1.3B, 480p, 2s with ring 4
            (1, 40, 75600, 128),  # Wan2.1-T2V-14B, 720p, 5s
            (1, 40, 37800, 128),  # Wan2.1-T2V-14B, 720p, 5s with ring 2
            (1, 40, 18900, 128),  # Wan2.1-T2V-14B, 720p, 5s with ring 4
            (1, 40, 9450, 128),  # Wan2.1-T2V-14B, 720p, 5s with ring8
            (1, 20, 75600, 128),  # Wan2.1-T2V-14B, 720p, 5s with ulysses 2
            (1, 10, 75600, 128),  # Wan2.1-T2V-14B, 720p, 5s with ulysses 4
            (1, 5, 75600, 128),  # Wan2.1-T2V-14B, 720p, 5s with ulysses 8
            (2, 5, 75600, 128),  # Batch size 2
        ]

        # Available attention implementations
        self.attn_impls = ["default"]

        # Add sage-attn if available
        try:
            import sageattention  # noqa: F401

            self.attn_impls.append("sage-attn")
            print("SageAttention available, adding sage-attn implementation")
        except ImportError:
            print("SageAttention not available, skipping sage-attn tests")

        # Add trtllm-attn if available
        try:
            import tensorrt_llm  # noqa: F401

            self.attn_impls.append("trtllm-attn")
            print("TensorRT-LLM available, adding trtllm-attn implementation")
        except ImportError:
            print("TensorRT-LLM not available, skipping trtllm-attn tests")

        # Add flash-attn3 if available
        try:
            import flash_attn_interface  # noqa: F401

            self.attn_impls.extend(["flash-attn3", "flash-attn3-fp8"])
            print("FlashAttn3 available, adding flash-attn3, flash-attn3-fp8 implementation")
        except ImportError:
            print("FlashAttn3 not available, skipping flash-attn3 tests")

        # Add te if available
        try:
            import transformer_engine.pytorch  # noqa: F401

            self.attn_impls.extend(["te", "te-fp8"])
            print("TransformerEngine available, adding te, te-fp8 implementation")
        except ImportError:
            print("TransformerEngine not available, skipping te tests")

        # Add fivx if available
        try:
            import flashinfer_vx  # noqa: F401

            self.attn_impls.extend(["fivx"])
            print("FlashInfer-VX (SageAttn for Blackwell) available, adding fivx implementation")
        except ImportError:
            print("FlashInfer-VX (SageAttn for Blackwell) not available, skipping fivx tests")

    def create_test_data(self, batch_size, num_heads, seq_len, head_dim, dtype):
        """Create test query, key, value tensors"""
        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=self.device
        )
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=self.device)
        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=self.device
        )
        return query, key, value

    def configure_attention(self, impl, num_heads, head_dim):
        """Configure attention settings for testing"""
        AttentionOpManager.attn_type = impl

    def test_attention_impl_correctness(self):
        """Test correctness of different attention implementations"""
        # Use a smaller test size for correctness test to avoid memory issues
        batch_size, num_heads, seq_len, head_dim = 1, 12, 1024, 128
        dtype = torch.bfloat16

        query, key, value = self.create_test_data(batch_size, num_heads, seq_len, head_dim, dtype)
        scale = 1.0 / (head_dim**0.5)

        # Reference computation with PyTorch's scaled_dot_product_attention
        expected = F.scaled_dot_product_attention(query, key, value, scale=scale)

        results = {}
        for impl in self.attn_impls:
            print(f"Testing {impl} implementation...")

            # Configure attention
            self.configure_attention(impl, num_heads, head_dim)

            # Get attention function
            try:
                AttentionOpManager.set_attn_config(attn_type=impl)
                attn_fn = ditAttnProcessor().visual_gen_attn
            except Exception as e:
                print(f"Skipping {impl} - failed to get attention function: {e}")
                continue

            # Compute result
            with torch.no_grad():
                try:
                    result = attn_fn(query, key, value, scale=scale, tensor_layout="HND")
                except Exception as e:
                    print(f"Skipping {impl} due to runtime error: {e}")
                    raise e

            results[impl] = result

            # Check correctness against reference
            if impl == "default":
                # Default should be exactly equal
                assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5), (
                    "Default implementation not matching reference"
                )
                print(f"✓ {impl} correctness verified")
            else:
                # Other implementations may have some numerical differences
                max_diff = torch.max(torch.abs(result - expected)).item()
                print(f"{impl} max difference from reference: {max_diff}")

                cos_sim = torch.nn.functional.cosine_similarity(
                    result.flatten(), expected.flatten(), dim=0
                )
                print(f"{impl} cosine similarity: {cos_sim.item():.6f}")

                # Check if result is reasonable (not NaN, not too different)
                if torch.isnan(result).any():
                    print(f"x {impl} produced NaN values")
                elif max_diff > 1e-1:  # Very large difference indicates a problem
                    print(f"x {impl} has very large difference from reference: {max_diff}")
                else:
                    print(f"✓ {impl} correctness verified (within reasonable bounds)")

    @pytest.mark.parametrize(
        "test_size_idx", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )  # Covers all test_sizes indices
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_attention_impl_performance(self, test_size_idx, dtype):
        """Benchmark performance of different attention implementations using test_sizes"""
        if self.device.type == "cpu" and dtype in [torch.float16, torch.bfloat16]:
            pytest.skip("CPU doesn't support float16/bfloat16")

        # Skip if test_size_idx is beyond available test sizes
        if test_size_idx >= len(self.test_sizes):
            pytest.skip(f"Test size index {test_size_idx} not available")

        # Get test size from test_sizes
        batch_size, num_heads, seq_len, head_dim = self.test_sizes[test_size_idx]

        # Skip very large tensors in individual pytest runs to avoid OOM
        tensor_size_gb = (
            batch_size * num_heads * seq_len * head_dim * 3 * 2
        ) / 1024**3  # 3 tensors (q,k,v), 2 bytes per element
        if tensor_size_gb > 4:  # Skip if larger than 4GB for individual tests
            pytest.skip(f"Tensor too large for individual test (~{tensor_size_gb:.1f}GB)")

        query, key, value = self.create_test_data(batch_size, num_heads, seq_len, head_dim, dtype)
        scale = 1.0 / (head_dim**0.5)

        performance_results = {}
        warmup_iterations = 10
        benchmark_iterations = 50

        print(
            f"\nBenchmarking shape: ({batch_size}, {num_heads}, {seq_len}, {head_dim}), dtype: {dtype}"
        )
        print(
            f"Device: {self.device}, Using {'CUDA Events' if self.device.type == 'cuda' else 'CPU Timer'}"
        )

        for impl in self.attn_impls:
            print(f"\nTesting {impl} implementation...")

            # Configure attention
            self.configure_attention(impl, num_heads, head_dim)

            # Get attention function
            try:
                AttentionOpManager.set_attn_config(attn_type=impl)
                attn_fn = ditAttnProcessor().visual_gen_attn
            except Exception as e:
                print(f"Skipping {impl} - failed to get attention function: {e}")
                continue

            # Warmup - important for GPU kernels
            print(f"  Warming up with {warmup_iterations} iterations...")
            with nvtx_range(f"attention_warmup_{impl}"):
                with torch_profiler_range(f"attention_warmup_{impl}"):
                    with torch.no_grad():
                        for _ in range(warmup_iterations):
                            try:
                                _ = attn_fn(query, key, value, scale=scale, tensor_layout="HND")
                            except Exception as e:
                                print(f"Skipping {impl} due to warmup error: {e}")
                                break
                        else:
                            # Only proceed if warmup succeeded

                            # Ensure all warmup operations are complete
                            if self.device.type == "cuda":
                                torch.cuda.synchronize()

                            # Benchmark with CUDA events
                            print(f"  Benchmarking with {benchmark_iterations} iterations...")
                            times = []

                            with nvtx_range(f"attention_benchmark_{impl}"):
                                with torch_profiler_range(f"attention_benchmark_{impl}"):
                                    with torch.no_grad():
                                        for i in range(benchmark_iterations):
                                            with nvtx_range(f"attention_iter_{impl}_{i}"):
                                                with cuda_timer(self.device) as get_time:
                                                    attn_fn(
                                                        query,
                                                        key,
                                                        value,
                                                        scale=scale,
                                                        tensor_layout="HND",
                                                    )

                                                elapsed_time = get_time()
                                                times.append(elapsed_time)

                                            # Print progress every 10 iterations
                                            if (i + 1) % 10 == 0:
                                                print(
                                                    f"    Progress: {i + 1}/{benchmark_iterations}"
                                                )

                    # Calculate detailed statistics
                    times = torch.tensor(times)
                    avg_time = times.mean().item()
                    min_time = times.min().item()
                    max_time = times.max().item()
                    std_time = times.std().item()
                    median_time = times.median().item()

                    # Calculate percentiles
                    p95_time = torch.quantile(times, 0.95).item()
                    p99_time = torch.quantile(times, 0.99).item()

                    performance_results[impl] = {
                        "avg_time_ms": avg_time,
                        "min_time_ms": min_time,
                        "max_time_ms": max_time,
                        "std_time_ms": std_time,
                        "median_time_ms": median_time,
                        "p95_time_ms": p95_time,
                        "p99_time_ms": p99_time,
                        "all_times": times.tolist(),
                    }

                    print(f"\n  {impl} Performance Results:")
                    print(f"    Average time: {avg_time:.3f} ms")
                    print(f"    Median time:  {median_time:.3f} ms")
                    print(f"    Min time:     {min_time:.3f} ms")
                    print(f"    Max time:     {max_time:.3f} ms")
                    print(f"    Std dev:      {std_time:.3f} ms")
                    print(f"    95th percentile: {p95_time:.3f} ms")
                    print(f"    99th percentile: {p99_time:.3f} ms")

                    # Calculate theoretical throughput (attention operations per second)
                    # Attention complexity: O(seq_len^2 * head_dim) for each head
                    total_ops = (
                        batch_size * num_heads * seq_len * seq_len * head_dim
                    )  # Approximate attention ops
                    throughput_tops = (total_ops / 1e12) / (avg_time / 1000)  # TOPS
                    print(f"    Throughput:   {throughput_tops:.2f} TOPS (approx)")

        # Print performance comparison
        if len(performance_results) > 1:
            baseline_avg = performance_results["default"]["avg_time_ms"]
            baseline_median = performance_results["default"]["median_time_ms"]

            print(f"\n{'=' * 60}")
            print("PERFORMANCE COMPARISON")
            print(f"{'=' * 60}")
            print(
                f"Baseline (default): {baseline_avg:.3f} ms (avg), {baseline_median:.3f} ms (median)"
            )
            print("-" * 60)

            for impl, stats in performance_results.items():
                if impl != "default":
                    speedup_avg = baseline_avg / stats["avg_time_ms"]
                    speedup_median = baseline_median / stats["median_time_ms"]
                    print(f"{impl}:")
                    print(
                        f"  Speedup (avg):    {speedup_avg:.2f}x {'faster' if speedup_avg > 1 else 'slower'}"
                    )
                    print(
                        f"  Speedup (median): {speedup_median:.2f}x {'faster' if speedup_median > 1 else 'slower'}"
                    )

    def test_all_sizes_performance(self):
        """Test performance for all sizes in test_sizes"""
        if self.device.type == "cpu":
            pytest.skip("Performance tests only meaningful on GPU")

        dtype = torch.bfloat16

        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE PERFORMANCE TEST - ALL SIZES")
        print(f"{'=' * 80}")
        print(f"Device: {self.device}, dtype: {dtype}")
        print(f"Testing {len(self.test_sizes)} different attention sizes")

        all_results = {}

        for size_idx, (batch_size, num_heads, seq_len, head_dim) in enumerate(self.test_sizes):
            print(f"\n{'-' * 60}")
            print(
                f"Test Size {size_idx + 1}/{len(self.test_sizes)}: ({batch_size}, {num_heads}, {seq_len}, {head_dim})"
            )
            print(f"{'-' * 60}")

            # Skip if tensor too large to avoid OOM
            tensor_size_gb = (
                batch_size * num_heads * seq_len * head_dim * 3 * 2
            ) / 1024**3  # rough estimate in GB
            if tensor_size_gb > 8:  # Skip if larger than 8GB
                print(f"  Skipping - tensor too large (~{tensor_size_gb:.1f}GB)")
                continue

            try:
                query, key, value = self.create_test_data(
                    batch_size, num_heads, seq_len, head_dim, dtype
                )
                scale = 1.0 / (head_dim**0.5)

                size_results = {}
                warmup_iterations = 5
                benchmark_iterations = 20

                for impl in self.attn_impls:
                    print(f"\n  Testing {impl}...")

                    # Configure attention
                    self.configure_attention(impl, num_heads, head_dim)

                    # Get attention function
                    try:
                        AttentionOpManager.set_attn_config(attn_type=impl)
                        attn_fn = ditAttnProcessor().visual_gen_attn
                    except Exception as e:
                        print(f"    Skipping {impl} - failed to get attention function: {e}")
                        continue

                    # Warmup
                    try:
                        with nvtx_range(
                            f"attention_warmup_{impl}_size_{self.test_sizes[size_idx]}"
                        ):
                            with torch_profiler_range(
                                f"attention_warmup_{impl}_size_{self.test_sizes[size_idx]}"
                            ):
                                with torch.no_grad():
                                    for _ in range(warmup_iterations):
                                        _ = attn_fn(
                                            query, key, value, scale=scale, tensor_layout="HND"
                                        )

                                if self.device.type == "cuda":
                                    torch.cuda.synchronize()

                        # Benchmark
                        times = []
                        with nvtx_range(
                            f"attention_benchmark_{impl}_size_{self.test_sizes[size_idx]}"
                        ):
                            with torch_profiler_range(
                                f"attention_benchmark_{impl}_size_{self.test_sizes[size_idx]}"
                            ):
                                with torch.no_grad():
                                    for i in range(benchmark_iterations):
                                        with cuda_timer(self.device) as get_time:
                                            attn_fn(
                                                query, key, value, scale=scale, tensor_layout="HND"
                                            )
                                        times.append(get_time())

                        # Calculate statistics
                        times = torch.tensor(times)
                        avg_time = times.mean().item()
                        median_time = times.median().item()

                        # Calculate theoretical throughput
                        total_ops = (
                            batch_size * num_heads * seq_len * seq_len * head_dim
                        )  # Approximate attention ops
                        throughput_tops = (total_ops / 1e12) / (avg_time / 1000)  # TOPS

                        size_results[impl] = {
                            "avg_time_ms": avg_time,
                            "median_time_ms": median_time,
                            "throughput_tops": throughput_tops,
                        }

                        print(
                            f"    Avg time: {avg_time:.3f} ms, Throughput: {throughput_tops:.2f} TOPS"
                        )

                    except Exception as e:
                        print(f"    Error testing {impl}: {e}")
                        continue

                all_results[f"size_{size_idx}"] = {
                    "shape": (batch_size, num_heads, seq_len, head_dim),
                    "results": size_results,
                }

                # Print comparison for this size
                if len(size_results) > 1 and "default" in size_results:
                    baseline_time = size_results["default"]["avg_time_ms"]
                    baseline_throughput = size_results["default"]["throughput_tops"]

                    print("\n  Performance vs baseline (default):")
                    for impl, stats in size_results.items():
                        if impl != "default":
                            speedup = baseline_time / stats["avg_time_ms"]
                            throughput_ratio = stats["throughput_tops"] / baseline_throughput
                            print(
                                f"    {impl}: {stats['avg_time_ms']:.3f} ms, {speedup:.2f}x faster, {throughput_ratio:.2f}x throughput"
                            )

            except Exception as e:
                print(f"  Error testing size {size_idx}: {e}")
                continue

        # Print summary
        print(f"\n{'=' * 80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'=' * 80}")

        for size_name, size_data in all_results.items():
            shape = size_data["shape"]
            results = size_data["results"]
            print(f"\nShape: {shape}")

            # Show basic performance metrics
            for impl, stats in results.items():
                print(
                    f"  {impl}: {stats['avg_time_ms']:.3f} ms, {stats['throughput_tops']:.2f} TOPS"
                )

            # Show speedup comparison if multiple implementations exist
            if len(results) > 1 and "default" in results:
                default_time = results["default"]["avg_time_ms"]
                default_throughput = results["default"]["throughput_tops"]

                print("  Speedup vs default:")
                for impl, stats in results.items():
                    if impl != "default":
                        speedup = default_time / stats["avg_time_ms"]
                        throughput_ratio = stats["throughput_tops"] / default_throughput
                        print(
                            f"    {impl}: {stats['avg_time_ms']:.3f} ms, {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} ({throughput_ratio:.2f}x throughput)"
                        )

    def test_memory_usage(self):
        """Test memory usage of different attention implementations"""
        if self.device.type != "cuda":
            print("Memory usage test only available on CUDA devices")
            return

        # Use a medium-sized test case
        batch_size, num_heads, seq_len, head_dim = self.test_sizes[1]  # Use second test size
        dtype = torch.bfloat16

        print(f"Memory usage test: ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

        for impl in self.attn_impls:
            print(f"\nTesting {impl} memory usage...")

            # Configure attention
            self.configure_attention(impl, num_heads, head_dim)

            # Get attention function
            try:
                AttentionOpManager.set_attn_config(attn_type=impl)
                attn_fn = ditAttnProcessor().visual_gen_attn
            except Exception as e:
                print(f"  Skipping {impl} - failed to get attention function: {e}")
                continue

            # Create test data
            query, key, value = self.create_test_data(
                batch_size, num_heads, seq_len, head_dim, dtype
            )
            scale = 1.0 / (head_dim**0.5)

            # Warmup
            try:
                with nvtx_range(f"attention_memory_warmup_{impl}"):
                    with torch_profiler_range(f"attention_memory_warmup_{impl}"):
                        with torch.no_grad():
                            attn_fn(query, key, value, scale=scale, tensor_layout="HND")

                # Clear cache before test
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Run forward pass
                with nvtx_range(f"attention_memory_test_{impl}"):
                    with torch_profiler_range(f"attention_memory_test_{impl}"):
                        with torch.no_grad():
                            result = attn_fn(query, key, value, scale=scale, tensor_layout="HND")

                # Get memory stats
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB

                print(f"  Peak memory: {peak_memory:.3f} GB")
                print(f"  Current memory: {current_memory:.3f} GB")

                # Calculate theoretical memory usage
                input_memory = (
                    batch_size * num_heads * seq_len * head_dim * 3 * 2
                ) / 1024**3  # q,k,v tensors in GB
                output_memory = (
                    batch_size * num_heads * seq_len * head_dim * 2
                ) / 1024**3  # output tensor in GB
                theoretical_min = input_memory + output_memory

                print(f"  Theoretical minimum: {theoretical_min:.3f} GB (input + output)")
                print(
                    f"  Memory overhead: {(peak_memory - theoretical_min):.3f} GB ({((peak_memory / theoretical_min - 1) * 100):.1f}%)"
                )

            except Exception as e:
                print(f"  Error testing {impl}: {e}")
                continue


if __name__ == "__main__":
    # Run basic correctness and performance tests
    test_class = TestditAttentionPerformance()
    test_class._init_test_environment()  # Use the direct initialization method

    print("Running ditAttention performance tests...")
    print(f"Device: {test_class.device}")
    print(f"Available implementations: {test_class.attn_impls}")
    print(
        f"Timing method: {'CUDA Events' if test_class.device.type == 'cuda' else 'CPU perf_counter'}"
    )
    print(f"Test sizes: {len(test_class.test_sizes)} configurations")
    print(
        f"NVTX profiling: {'Enabled' if NVTX_AVAILABLE else 'Disabled'}"
        + (f" (method: {NVTX_METHOD})" if NVTX_AVAILABLE and NVTX_METHOD else "")
    )
    print(f"Torch profiler: {'Enabled' if TORCH_PROFILER_AVAILABLE else 'Disabled'}")

    if test_class.device.type == "cuda":
        print("\n" + "=" * 60)
        print("PROFILING INSTRUCTIONS")
        print("=" * 60)

        if NVTX_AVAILABLE:
            print("NVTX PROFILING (Nsight Systems):")
            print(
                "  nsys profile -t cuda,nvtx --nvtx-capture=range --output=attention_profile python tests/test_attention.py"
            )
            print("  nsys-ui attention_profile.nsys-rep")
            print(f"  NVTX method: {NVTX_METHOD}")
            print()

        if TORCH_PROFILER_AVAILABLE:
            print("PYTORCH PROFILER:")
            print("  The test includes torch.profiler.record_function() calls")
            print("  You can also add torch.profiler.profile() wrapper around the main test")
            print("  Example:")
            print("    with torch.profiler.profile(")
            print(
                "        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],"
            )
            print("        record_shapes=True,")
            print("        with_stack=True")
            print("    ) as prof:")
            print("        # run tests")
            print("    prof.export_chrome_trace('attention_trace.json')")
            print()

        print("PROFILE RANGES:")
        print("  - visual_gen_attention_tests: Complete test suite")
        print("  - attention_warmup_<impl>: Warmup phase for each implementation")
        print("  - attention_benchmark_<impl>: Benchmark phase for each implementation")
        print("  - attention_memory_test_<impl>: Memory usage test for each implementation")

        if not NVTX_AVAILABLE and not TORCH_PROFILER_AVAILABLE:
            print("\nTo enable profiling, install:")
            print("  pip install nvtx-plugins  # for NVTX support")
            print("  # PyTorch profiler is usually included with PyTorch")

        print("=" * 60)

    with nvtx_range("visual_gen_attention_tests"):
        with torch_profiler_range("visual_gen_attention_tests"):
            # Run correctness test
            print("\n" + "=" * 50)
            print("CORRECTNESS TEST")
            print("=" * 50)
            with nvtx_range("correctness_tests"):
                with torch_profiler_range("correctness_tests"):
                    test_class.test_attention_impl_correctness()
            print("✓ All correctness tests completed!")

            # Run comprehensive performance test for all sizes
            print("\n" + "=" * 50)
            print("COMPREHENSIVE PERFORMANCE TEST")
            print("=" * 50)
            with nvtx_range("comprehensive_performance_tests"):
                with torch_profiler_range("comprehensive_performance_tests"):
                    test_class.test_all_sizes_performance()

            # Run memory test if on CUDA
            if test_class.device.type == "cuda":
                print("\n" + "=" * 50)
                print("MEMORY USAGE TEST")
                print("=" * 50)
                with nvtx_range("memory_usage_tests"):
                    with torch_profiler_range("memory_usage_tests"):
                        test_class.test_memory_usage()

    print("\n✓ All tests completed!")
