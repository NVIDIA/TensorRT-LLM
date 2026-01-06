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

import pynvml
import pytest
import torch
import torch.nn.functional as F

from visual_gen.configs.op_manager import LinearOpManager
from visual_gen.layers.linear import ditLinear


def get_compute_capability():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    pynvml.nvmlShutdown()
    return str(cc[0]) + str(cc[1])


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


class TestditLinearPerformance:
    """Performance tests for ditLinear with different linear_impl implementations"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment for pytest"""
        self._init_test_environment()

    def _init_test_environment(self):
        """Initialize test environment - can be called directly or via pytest fixture"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

        # Test matrix sizes: (batch_size, seq_len, in_features, out_features)
        self.test_sizes = [
            (1, 14040, 1536, 1536),  # Wan2.1-T2V-1.3B, 480p, 2s, to_q, to_k, to_v, to_out.0
            (1, 14040, 1536, 8960),  # Wan2.1-T2V-1.3B, 480p, 2s, ffn.net.0.proj
            (1, 14040, 8960, 1536),  # Wan2.1-T2V-1.3B, 480p, 2s, ffn.net.2
            (1, 75600, 5120, 5120),  # Wan2.1-T2V-14B, 720p, 5s, to_q, to_k, to_v, to_out.0
            (1, 75600, 5120, 13824),  # Wan2.1-T2V-14B, 720p, 5s, ffn.net.0.proj
            (1, 75600, 13824, 5120),  # Wan2.1-T2V-14B, 720p, 5s, ffn.net.2
            # cp2
            (1, 75600 // 2, 5120, 5120),  # Wan2.1-T2V-14B, 720p, 5s, to_q, to_k, to_v, to_out.0
            (1, 75600 // 2, 5120, 13824),  # Wan2.1-T2V-14B, 720p, 5s, ffn.net.0.proj
            (1, 75600 // 2, 13824, 5120),  # Wan2.1-T2V-14B, 720p, 5s, ffn.net.2
            # cp4
            (1, 75600 // 4, 5120, 5120),  # Wan2.1-T2V-14B, 720p, 5s, to_q, to_k, to_v, to_out.0
            (1, 75600 // 4, 5120, 13824),  # Wan2.1-T2V-14B, 720p, 5s, ffn.net.0.proj
            (1, 75600 // 4, 13824, 5120),  # Wan2.1-T2V-14B, 720p, 5s, ffn.net.2
        ]

        # Available linear implementations
        self.linear_impls = ["default"]

        # Add trtllm Linear layers if TensorRT-LLM is available
        try:
            import tensorrt_llm  # noqa: F401

            sm_version = get_compute_capability()
            self.linear_impls.append("trtllm-fp8-per-tensor")
            if sm_version == "90":
                self.linear_impls.append("trtllm-fp8-blockwise")
            if sm_version == "120":
                self.linear_impls.append("trtllm-nvfp4")
            print("TensorRT-LLM available, adding trtllm Linear layers")
        except ImportError:
            print("TensorRT-LLM not available, skipping trtllm tests")

        # Add transformer_engine layers
        try:
            import transformer_engine  # noqa: F401

            sm_version = get_compute_capability()
            self.linear_impls.append("te-fp8-per-tensor")
            if sm_version == "90":
                self.linear_impls.append("te-fp8-blockwise")
            if sm_version == "100":
                self.linear_impls.append("te-MXFP8-blockwise-32")
            print("TransformerEngine available, adding TE fp8 implementation")
        except ImportError:
            print("TransformerEngine not available, skipping TE fp8 tests")

        # Add torchao layers
        try:
            import torch_ao  # noqa: F401

            if sm_version != "120":
                self.linear_impls.append("torch-ao-fp8")
        except ImportError:
            print("TorchAO not available, skipping its tests")

        # Add flashinfer layers
        try:
            import flashinfer  # noqa: F401

            self.linear_impls.append("flashinfer-nvfp4-trtllm")
            self.linear_impls.append("flashinfer-nvfp4-cudnn")
            self.linear_impls.append("flashinfer-nvfp4-cutlass")
        except ImportError:
            print("Flashinfer not available, skipping its tests")

        try:
            import deep_gemm  # noqa: F401
            self.linear_impls.append("deepgemm-MXFP8")
        except ImportError:
            print("DeepGEMM not available, skipping its tests")

    def create_test_data(self, batch_size, seq_len, in_features, out_features, dtype):
        """Create test input and weight tensors"""
        input_tensor = torch.randn(batch_size, seq_len, in_features, dtype=dtype, device=self.device)
        weight = torch.randn(out_features, in_features, dtype=dtype, device=self.device)
        bias = torch.randn(out_features, dtype=dtype, device=self.device)
        return input_tensor, weight, bias

    def test_linear_impl_correctness(self):
        """Test correctness of different linear implementations"""
        # Use the first test size for correctness test
        batch_size, seq_len, in_features, out_features = self.test_sizes[0]
        dtype = torch.bfloat16

        input_tensor, weight, bias = self.create_test_data(batch_size, seq_len, in_features, out_features, dtype)

        # Reference computation with standard torch linear
        expected = F.linear(input_tensor, weight, bias)

        results = {}
        for impl in self.linear_impls:
            # Set linear implementation type
            LinearOpManager.set_linear_type(impl)

            # Create ditLinear instance
            visual_gen_linear = ditLinear(in_features, out_features, device=self.device, dtype=dtype)
            visual_gen_linear.weight.data = weight.clone()
            visual_gen_linear.bias.data = bias.clone()

            # Compute result
            with torch.no_grad():
                result = visual_gen_linear(input_tensor)

            results[impl] = result

            # Check correctness against reference
            if impl == "default":
                # Default should be exactly equal
                assert torch.allclose(
                    result, expected, rtol=1e-5, atol=1e-6
                ), "Default implementation not matching reference"
            else:
                # Other implementations may have some numerical differences
                max_diff = torch.max(torch.abs(result - expected)).item()
                print(f"{impl} max difference from reference: {max_diff}")

                cos_sim = torch.nn.functional.cosine_similarity(
                    result.flatten().float(), expected.flatten().float(), dim=0
                )
                print(f"{impl} cosine similarity: {cos_sim.item():.6f}")

                mse = torch.nn.functional.mse_loss(result.flatten().float(), expected.flatten().float())
                print(f"{impl} mse_loss: {mse.item():.6f}")

    @pytest.mark.parametrize("test_size_idx", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # Covers all test_sizes indices
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_linear_impl_performance(self, test_size_idx, dtype):
        """Benchmark performance of different linear implementations using test_sizes"""
        if self.device.type == "cpu" and dtype in [torch.float16, torch.bfloat16]:
            pytest.skip("CPU doesn't support float16/bfloat16")

        # Skip if test_size_idx is beyond available test sizes
        if test_size_idx >= len(self.test_sizes):
            pytest.skip(f"Test size index {test_size_idx} not available")

        # Get test size from test_sizes
        batch_size, seq_len, in_features, out_features = self.test_sizes[test_size_idx]

        # Skip very large tensors in individual pytest runs to avoid OOM
        tensor_size_gb = (batch_size * seq_len * in_features * 2) / 1024**3
        if tensor_size_gb > 4:  # Skip if larger than 4GB for individual tests
            pytest.skip(f"Tensor too large for individual test (~{tensor_size_gb:.1f}GB)")

        input_tensor, weight, bias = self.create_test_data(batch_size, seq_len, in_features, out_features, dtype)

        performance_results = {}
        warmup_iterations = 10
        benchmark_iterations = 50

        print(f"\nBenchmarking shape: ({batch_size}, {seq_len}, {in_features}) -> {out_features}, dtype: {dtype}")
        print(f"Device: {self.device}, Using {'CUDA Events' if self.device.type == 'cuda' else 'CPU Timer'}")

        for impl in self.linear_impls:
            print(f"\nTesting {impl} implementation...")

            # Set linear implementation type
            LinearOpManager.set_linear_type(impl)

            # Create ditLinear instance
            visual_gen_linear = ditLinear(in_features, out_features, device=self.device, dtype=dtype)
            visual_gen_linear.weight.data = weight.clone()
            visual_gen_linear.bias.data = bias.clone()

            # Warmup - important for GPU kernels
            print(f"  Warming up with {warmup_iterations} iterations...")
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = visual_gen_linear(input_tensor)

            # Ensure all warmup operations are complete
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark with CUDA events
            print(f"  Benchmarking with {benchmark_iterations} iterations...")
            times = []

            with torch.no_grad():
                for i in range(benchmark_iterations):
                    with cuda_timer(self.device) as get_time:
                        visual_gen_linear(input_tensor)

                    elapsed_time = get_time()
                    times.append(elapsed_time)

                    # Print progress every 10 iterations
                    if (i + 1) % 10 == 0:
                        print(f"    Progress: {i + 1}/{benchmark_iterations}")

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

            # Calculate throughput
            total_ops = batch_size * seq_len * in_features * out_features * 2  # multiply-add ops
            throughput_tops = (total_ops / 1e12) / (avg_time / 1000)  # TOPS
            print(f"    Throughput:   {throughput_tops:.2f} TOPS")

        # Print performance comparison
        if len(performance_results) > 1:
            baseline_avg = performance_results["default"]["avg_time_ms"]
            baseline_median = performance_results["default"]["median_time_ms"]

            print(f"\n{'='*60}")
            print("PERFORMANCE COMPARISON")
            print(f"{'='*60}")
            print(f"Baseline (default): {baseline_avg:.3f} ms (avg), {baseline_median:.3f} ms (median)")
            print("-" * 60)

            for impl, stats in performance_results.items():
                if impl != "default":
                    speedup_avg = baseline_avg / stats["avg_time_ms"]
                    speedup_median = baseline_median / stats["median_time_ms"]
                    print(f"{impl}:")
                    print(f"  Speedup (avg):    {speedup_avg:.2f}x {'faster' if speedup_avg > 1 else 'slower'}")
                    print(f"  Speedup (median): {speedup_median:.2f}x {'faster' if speedup_median > 1 else 'slower'}")

    def test_all_sizes_performance(self):
        """Test performance for all sizes in test_sizes"""
        if self.device.type == "cpu":
            pytest.skip("Performance tests only meaningful on GPU")

        dtype = torch.bfloat16

        print(f"\n{'='*80}")
        print("COMPREHENSIVE PERFORMANCE TEST - ALL SIZES")
        print(f"{'='*80}")
        print(f"Device: {self.device}, dtype: {dtype}")
        print(f"Testing {len(self.test_sizes)} different matrix sizes")

        all_results = {}

        for size_idx, (batch_size, seq_len, in_features, out_features) in enumerate(self.test_sizes):
            print(f"\n{'-'*60}")
            print(
                f"Test Size {size_idx + 1}/{len(self.test_sizes)}: ({batch_size}, {seq_len}, {in_features}) -> {out_features}"
            )
            print(f"{'-'*60}")

            # Skip if tensor too large to avoid OOM
            tensor_size_gb = (batch_size * seq_len * in_features * 2) / 1024**3  # rough estimate in GB
            if tensor_size_gb > 8:  # Skip if larger than 8GB
                print(f"  Skipping - tensor too large (~{tensor_size_gb:.1f}GB)")
                continue

            try:
                input_tensor, weight, bias = self.create_test_data(
                    batch_size, seq_len, in_features, out_features, dtype
                )

                size_results = {}
                warmup_iterations = 5
                benchmark_iterations = 20

                for impl in self.linear_impls:
                    print(f"\n  Testing {impl}...")

                    # Set linear implementation type
                    LinearOpManager.set_linear_type(impl)

                    # Create ditLinear instance
                    visual_gen_linear = ditLinear(in_features, out_features, device=self.device, dtype=dtype)
                    visual_gen_linear.weight.data = weight.clone()
                    visual_gen_linear.bias.data = bias.clone()

                    # Warmup
                    with torch.no_grad():
                        for _ in range(warmup_iterations):
                            _ = visual_gen_linear(input_tensor)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    # Benchmark
                    times = []
                    with torch.no_grad():
                        for i in range(benchmark_iterations):
                            with cuda_timer(self.device) as get_time:
                                visual_gen_linear(input_tensor)
                            times.append(get_time())

                    # Calculate statistics
                    times = torch.tensor(times)
                    avg_time = times.mean().item()
                    median_time = times.median().item()

                    # Calculate throughput
                    total_ops = batch_size * seq_len * in_features * out_features * 2  # multiply-add ops
                    throughput_tops = (total_ops / 1e12) / (avg_time / 1000)  # TOPS

                    size_results[impl] = {
                        "avg_time_ms": avg_time,
                        "median_time_ms": median_time,
                        "throughput_tops": throughput_tops,
                    }

                    print(f"    Avg time: {avg_time:.3f} ms, Throughput: {throughput_tops:.2f} TOPS")

                all_results[f"size_{size_idx}"] = {
                    "shape": (batch_size, seq_len, in_features, out_features),
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
                            print(f"    {impl}: {speedup:.2f}x faster, {throughput_ratio:.2f}x throughput")

            except Exception as e:
                print(f"  Error testing size {size_idx}: {e}")
                continue

        # Print summary
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")

        for size_name, size_data in all_results.items():
            shape = size_data["shape"]
            results = size_data["results"]
            print(f"\nShape: {shape}")

            # Show basic performance metrics
            for impl, stats in results.items():
                print(f"  {impl}: {stats['avg_time_ms']:.3f} ms, {stats['throughput_tops']:.2f} TOPS")

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
                            f"    {impl}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} ({throughput_ratio:.2f}x throughput)"
                        )

    def test_memory_usage(self):
        """Test memory usage of different linear implementations"""
        if self.device.type != "cuda":
            print("Memory usage test only available on CUDA devices")
            return

        # Use a medium-sized test case
        batch_size, seq_len, in_features, out_features = self.test_sizes[1]  # Use second test size
        dtype = torch.bfloat16

        print(f"Memory usage test: ({batch_size}, {seq_len}, {in_features}) -> {out_features}")

        for impl in self.linear_impls:
            print(f"\nTesting {impl} memory usage...")

            # Set linear implementation type
            LinearOpManager.set_linear_type(impl)

            # Create test data and model
            input_tensor, weight, bias = self.create_test_data(batch_size, seq_len, in_features, out_features, dtype)

            visual_gen_linear = ditLinear(in_features, out_features, device=self.device, dtype=dtype)
            visual_gen_linear.weight.data = weight.clone()
            visual_gen_linear.bias.data = bias.clone()

            # Warmup
            with torch.no_grad():
                visual_gen_linear(input_tensor)

            # Clear cache before test
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run forward pass
            with torch.no_grad():
                visual_gen_linear(input_tensor)

            # Get memory stats
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB

            print(f"  Peak memory: {peak_memory:.3f} GB")
            print(f"  Current memory: {current_memory:.3f} GB")


if __name__ == "__main__":
    # Run basic correctness test
    test_class = TestditLinearPerformance()
    test_class._init_test_environment()  # Use the direct initialization method

    print("Running ditLinear performance tests...")
    print(f"Device: {test_class.device}")
    print(f"Available implementations: {test_class.linear_impls}")
    print(f"Timing method: {'CUDA Events' if test_class.device.type == 'cuda' else 'CPU perf_counter'}")
    print(f"Test sizes: {len(test_class.test_sizes)} configurations")

    # Run correctness test
    print("\n" + "=" * 50)
    print("CORRECTNESS TEST")
    print("=" * 50)
    test_class.test_linear_impl_correctness()
    print("✓ All correctness tests passed!")

    # Run comprehensive performance test for all sizes
    print("\n" + "=" * 50)
    print("COMPREHENSIVE PERFORMANCE TEST")
    print("=" * 50)
    test_class.test_all_sizes_performance()

    # Run memory test if on CUDA
    if test_class.device.type == "cuda":
        print("\n" + "=" * 50)
        print("MEMORY USAGE TEST")
        print("=" * 50)
        test_class.test_memory_usage()

    print("\n✓ All tests completed!")
