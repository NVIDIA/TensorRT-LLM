# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend import (
    VSAMetadataBuilder,
    set_vsa_forward_context,
)

# ============================================================================
# Flash Attention 4 availability
# ============================================================================
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import _cute_dsl_import_error
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import _flash_attn_fwd as _fa4_fwd
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
    _flash_attn_fwd_import_error as _fa4_import_error,
)
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    QuantAttentionConfig,
    VideoSparseAttentionConfig,
)

_flash_attn4_available = _fa4_fwd is not None
_cute_dsl_available = _cute_dsl_import_error is None


def _require_attention_backend(backend: str, head_dim: Optional[int] = None) -> None:
    if backend == "FA4" and not _flash_attn4_available:
        pytest.fail(
            "FlashAttention 4 backend is required for FA4 attention perf test"
            + (f": {_fa4_import_error}" if _fa4_import_error else "")
        )
    if backend == "CUTEDSL" and not _cute_dsl_available:
        pytest.fail(
            "CuTe DSL backend is required for CUTEDSL attention perf test"
            + (f": {_cute_dsl_import_error}" if _cute_dsl_import_error else "")
        )
    if backend == "CUTEDSL":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for CUTEDSL attention perf test")
        compute_capability = torch.cuda.get_device_capability()
        gpu_arch = f"sm_{compute_capability[0]}{compute_capability[1]}a"
        if gpu_arch not in ("sm_100a", "sm_103a"):
            pytest.skip("CUTEDSL attention perf test requires a supported Blackwell-class GPU")
        if head_dim is not None and head_dim != 128:
            pytest.skip("CUTEDSL attention perf test requires head_dim=128")


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


def bench_fn(device: torch.device, fn, n_iters: int) -> torch.Tensor:
    """Time `fn` over `n_iters` GPU-timed iterations; returns per-iteration ms."""
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            with cuda_timer(device) as get_t:
                fn()
            times.append(get_t())
    return torch.tensor(times)


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
    quant_attention_config: "QuantAttentionConfig | None" = None,
    vsa_sparsity: "float | None" = None,
) -> DiffusionModelConfig:
    """Create a mock DiffusionModelConfig for testing."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=eps,
    )

    sparse_attention_config = (
        VideoSparseAttentionConfig(vsa_sparsity=vsa_sparsity) if vsa_sparsity is not None else None
    )

    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(
            backend=attn_backend,
            quant_attention_config=quant_attention_config,
            sparse_attention_config=sparse_attention_config,
        ),
        skip_create_weights_in_init=False,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if attn_backend == "TRTLLM" else None
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


def _is_sage_attention_enabled(model: Attention) -> bool:
    """Return whether the created attention model is using SageAttention."""
    attention_backend = getattr(model, "attn", None)
    inner_backend = getattr(attention_backend, "inner_backend", attention_backend)
    return getattr(inner_backend, "quant_attention_config", None) is not None


def _init_attention_weights(attn: Attention, std: float = 0.02) -> None:
    """Initialize TRT-LLM Linear weights to N(0, std) to avoid RMSNorm NaN."""
    with torch.no_grad():
        if getattr(attn, "qkv_proj", None) is not None:
            attn.qkv_proj.weight.normal_(0.0, std)
            if attn.qkv_proj.bias is not None:
                attn.qkv_proj.bias.zero_()
        if getattr(attn, "qk_norm", False):
            attn.norm_q.weight.fill_(1.0)
            attn.norm_k.weight.fill_(1.0)
        attn.to_out[0].weight.normal_(0.0, std)
        if attn.to_out[0].bias is not None:
            attn.to_out[0].bias.zero_()


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
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        backend: str,
        quant_attention_config: "QuantAttentionConfig | None" = None,
        vsa_sparsity: "float | None" = None,
        init_std: "float | None" = None,
    ) -> Attention:
        """Create a WAN self-attention model with specified backend."""
        config = create_model_config(
            hidden_size,
            num_heads,
            head_dim,
            attn_backend=backend,
            quant_attention_config=quant_attention_config,
            vsa_sparsity=vsa_sparsity,
        )
        model = Attention(hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, config=config).to(
            self.device
        )
        model.eval()
        if init_std is not None:
            _init_attention_weights(model, init_std)
        return model

    def create_cross_attention_model(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        backend: str,
        quant_attention_config: "QuantAttentionConfig | None" = None,
    ) -> Attention:
        """Create a WAN cross-attention model with specified backend."""
        config = create_model_config(
            hidden_size,
            num_heads,
            head_dim,
            attn_backend=backend,
            quant_attention_config=quant_attention_config,
        )
        model = Attention(hidden_size, num_heads, qkv_mode=QKVMode.SEPARATE_QKV, config=config).to(
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

    def benchmark_cross_attn_single(
        self,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        head_dim: int,
        backend: str,
        quant_attention_config: "QuantAttentionConfig | None" = None,
        verbose: bool = True,
    ) -> Optional[Dict]:
        """Benchmark a single cross-attention configuration.

        Returns:
            Dict with timing statistics or None if test failed/skipped
        """
        hidden_size = num_heads * head_dim

        try:
            model = self.create_cross_attention_model(
                hidden_size,
                num_heads,
                head_dim,
                backend,
                quant_attention_config=quant_attention_config,
            )

            hidden_states = torch.randn(
                batch_size, seq_len_q, hidden_size, device=self.device, dtype=self.dtype
            )
            encoder_hidden_states = torch.randn(
                batch_size, seq_len_kv, hidden_size, device=self.device, dtype=self.dtype
            )

            # Warmup
            with torch.no_grad():
                for _ in range(self.warmup_iterations):
                    _ = model(hidden_states, encoder_hidden_states=encoder_hidden_states)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            times = []
            with torch.no_grad():
                for i in range(self.benchmark_iterations):
                    with cuda_timer(self.device) as get_time:
                        _ = model(hidden_states, encoder_hidden_states=encoder_hidden_states)
                    times.append(get_time())

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

            total_ops = batch_size * num_heads * seq_len_q * seq_len_kv * head_dim
            stats["throughput_tops"] = (total_ops / 1e12) / (stats["avg_ms"] / 1000)

            if verbose:
                print(
                    f"  {backend} (cross S_q={seq_len_q}, S_kv={seq_len_kv}): "
                    f"avg={stats['avg_ms']:.3f}ms, "
                    f"median={stats['median_ms']:.3f}ms, "
                    f"throughput={stats['throughput_tops']:.2f} TOPS"
                )

            return stats

        except (RuntimeError, torch.cuda.CudaError) as e:
            if verbose:
                print(f"  {backend} (cross): ERROR - {e}")
            return None

    def benchmark_single(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        backend: str,
        verbose: bool = True,
        quant_attention_config: "QuantAttentionConfig | None" = None,
        vsa_sparsity: "float | None" = None,
        latent_shape: "Tuple[int, int, int] | None" = None,
        init_std: "float | None" = None,
    ) -> Optional[Dict]:
        """Benchmark a single configuration.

        For the VSA path, pass ``vsa_sparsity`` and the 3-D ``latent_shape``
        (T, H, W) with ``seq_len == T * H * W``; the forward is run inside a
        VSA forward context with a zero ``gate_compress``.

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
            model = self.create_attention_model(
                hidden_size,
                num_heads,
                head_dim,
                backend,
                quant_attention_config,
                vsa_sparsity=vsa_sparsity,
                init_std=init_std,
            )
            hidden_states, freqs = self.create_test_data(batch_size, seq_len, hidden_size, head_dim)

            # Fail fast if the requested backend silently resolved to something
            # else (e.g. CUTEDSL/VSA downgraded to VANILLA dense for an
            # unsupported config).
            resolved_backend = getattr(model, "attn_backend", backend)
            if vsa_sparsity is not None:
                assert resolved_backend == "CUTEDSL", (
                    f"VSA run requested CUTEDSL but the attention module resolved to "
                    f"{resolved_backend!r}; refusing to report VSA timings for a dense fallback."
                )

            # VSA needs an active forward context + a zero gate_compress; other
            # backends use neither.
            vsa_metadata = None
            gate_compress = None
            if vsa_sparsity is not None:
                vsa_metadata = VSAMetadataBuilder().build(
                    current_timestep=0,
                    raw_latent_shape=latent_shape,
                    patch_size=(1, 1, 1),
                    vsa_sparsity=vsa_sparsity,
                    device=self.device,
                )
                gate_compress = torch.zeros_like(hidden_states)

            def _forward():
                ctx = (
                    set_vsa_forward_context(vsa_metadata)
                    if vsa_metadata is not None
                    else nullcontext()
                )
                kwargs = {"gate_compress": gate_compress} if gate_compress is not None else {}
                with ctx:
                    return model(hidden_states, freqs=freqs, **kwargs)

            # Warmup
            with nvtx_range(f"warmup_{backend}"):
                with torch_profiler_range(f"warmup_{backend}"):
                    with torch.no_grad():
                        for _ in range(self.warmup_iterations):
                            _ = _forward()

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
                                    _ = _forward()
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
                "times_ms": times,
                "uses_sage": _is_sage_attention_enabled(model),
                "resolved_backend": resolved_backend,
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

        except AssertionError:
            # Backend-resolution / fail-fast checks are genuine test failures.
            raise
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
        quant_attention_config: "QuantAttentionConfig | None" = None,
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
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                backend,
                quant_attention_config=quant_attention_config,
                verbose=verbose,
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

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize(
        ("backend", "quant_attention_config"),
        [
            ("VANILLA", None),
            ("TRTLLM", None),
            ("FA4", None),
            ("CUTEDSL", None),
            ("CUTEDSL", QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8")),
        ],
    )
    def test_self_attention_perf(
        self,
        head_dim: int,
        backend: str,
        quant_attention_config: "QuantAttentionConfig | None",
    ):
        """Test that attention backend runs without errors."""
        _require_attention_backend(backend, head_dim)

        batch_size, num_heads, seq_len = 1, 24, 1024

        result = self.benchmark.benchmark_single(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            backend,
            quant_attention_config=quant_attention_config,
            verbose=True,
        )

        assert result is not None, f"{backend} benchmark failed to produce results"
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
# Flash Attention 4 performance tests
# ============================================================================


class TestFlashAttn4Performance:
    """Performance benchmarks comparing FA4 and VANILLA attention backends."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup benchmark with FA4 and VANILLA backends."""
        if not _flash_attn4_available:
            pytest.fail("FlashAttention 4 backend is required for TestFlashAttn4Performance")
        self.benchmark = WanAttentionPerformanceBenchmark(
            warmup_iterations=5,
            benchmark_iterations=20,
        )
        # Override backends so benchmark_comparison covers VANILLA and FA4
        self.benchmark.backends = ["VANILLA", "FA4"]

    @pytest.mark.parametrize(
        "model_name,batch,seq_len,num_heads,head_dim",
        [
            # WAN 1.3B (num_attention_heads=12, attention_head_dim=128)
            ("wan_1.3b_480p_33f", 1, 14040, 12, 128),  # 480x832x33f
            ("wan_1.3b_480p_81f", 1, 32760, 12, 128),  # 480x832x81f
            # WAN 14B (num_attention_heads=40, attention_head_dim=128)
            ("wan_14b_480p_33f", 1, 14040, 40, 128),  # 480x832x33f
            ("wan_14b_480p_81f", 1, 32760, 40, 128),  # 480x832x81f
            ("wan_14b_720p_81f", 1, 75600, 40, 128),  # 720x1280x81f
        ],
    )
    def test_fa4_vs_vanilla_wan_shapes(
        self,
        model_name: str,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
    ):
        """Compare FA4 vs VANILLA timing on real WAN model shapes.

        Seq lens computed from VAE scale factors (spatial=8, temporal=4) and
        patch_size=[1,2,2]:
          480x832x33f  -> seq_len  14,040
          480x832x81f  -> seq_len  32,760
          720x1280x81f -> seq_len  75,600
        """
        results = self.benchmark.benchmark_comparison(
            batch,
            num_heads,
            seq_len,
            head_dim,
            description=f"FA4 vs VANILLA {model_name}",
            verbose=True,
        )

        vanilla = results.get("VANILLA")
        fa4 = results.get("FA4")

        assert vanilla is not None, f"VANILLA benchmark failed for {model_name}"
        assert fa4 is not None, f"FA4 benchmark failed for {model_name}"

        speedup = vanilla["avg_ms"] / fa4["avg_ms"]
        print(
            f"\n  {model_name}: FA4 speedup={speedup:.2f}x "
            f"({'faster' if speedup > 1 else 'slower'})"
        )
        print(f"    VANILLA: avg={vanilla['avg_ms']:.3f}ms  p95={vanilla['p95_ms']:.3f}ms")
        print(f"    FA4:     avg={fa4['avg_ms']:.3f}ms  p95={fa4['p95_ms']:.3f}ms")


# ============================================================================
# Main entry point
# ============================================================================


class TestFlashAttn4CrossAttnPerformance:
    """Performance benchmarks for FA4 cross-attention vs VANILLA."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup benchmark with FA4 and VANILLA backends."""
        if not _flash_attn4_available:
            pytest.fail("FlashAttention 4 backend is required for TestFlashAttn4Performance")
        self.benchmark = WanAttentionPerformanceBenchmark(
            warmup_iterations=5,
            benchmark_iterations=20,
        )
        self.device = self.benchmark.device
        self.dtype = self.benchmark.dtype

    @pytest.mark.parametrize(
        "model_name,batch,seq_len_q,seq_len_kv,num_heads,head_dim",
        [
            # WAN 1.3B cross-attention: visual tokens attend to 512 text tokens
            ("wan_1.3b_480p_33f_cross", 1, 14040, 512, 12, 128),
            ("wan_1.3b_480p_81f_cross", 1, 32760, 512, 12, 128),
            # WAN 14B cross-attention
            ("wan_14b_480p_33f_cross", 1, 14040, 512, 40, 128),
            ("wan_14b_720p_81f_cross", 1, 75600, 512, 40, 128),
        ],
    )
    def test_fa4_vs_vanilla_cross_attn_wan_shapes(
        self,
        model_name: str,
        batch: int,
        seq_len_q: int,
        seq_len_kv: int,
        num_heads: int,
        head_dim: int,
    ):
        """Compare FA4 vs VANILLA cross-attention on real WAN model shapes."""
        results = {}
        for backend in ["VANILLA", "FA4"]:
            results[backend] = self.benchmark.benchmark_cross_attn_single(
                batch,
                num_heads,
                seq_len_q,
                seq_len_kv,
                head_dim,
                backend,
                verbose=True,
            )

        vanilla = results.get("VANILLA")
        fa4 = results.get("FA4")

        assert vanilla is not None, f"VANILLA cross-attn benchmark failed for {model_name}"
        assert fa4 is not None, f"FA4 cross-attn benchmark failed for {model_name}"

        speedup = vanilla["avg_ms"] / fa4["avg_ms"]
        print(
            f"\n  {model_name}: FA4 cross-attn speedup={speedup:.2f}x "
            f"({'faster' if speedup > 1 else 'slower'})"
        )
        print(f"    VANILLA: avg={vanilla['avg_ms']:.3f}ms  p95={vanilla['p95_ms']:.3f}ms")
        print(f"    FA4:     avg={fa4['avg_ms']:.3f}ms  p95={fa4['p95_ms']:.3f}ms")

    @pytest.mark.parametrize(
        "batch,seq_len_q,seq_len_kv,num_heads,head_dim",
        [
            (1, 1024, 512, 12, 128),
            (1, 2048, 512, 12, 128),
            (2, 1024, 512, 12, 128),
        ],
    )
    def test_fa4_cross_attn_quick(
        self,
        batch: int,
        seq_len_q: int,
        seq_len_kv: int,
        num_heads: int,
        head_dim: int,
    ):
        """Quick FA4 cross-attention correctness and timing check."""
        for backend in ["VANILLA", "FA4"]:
            result = self.benchmark.benchmark_cross_attn_single(
                batch,
                num_heads,
                seq_len_q,
                seq_len_kv,
                head_dim,
                backend,
                verbose=True,
            )
            assert result is not None, f"{backend} cross-attn failed"
            assert result["avg_ms"] > 0


# ============================================================================
# VSA fine-stage kernel vs FA4 (kernel-level)
# ============================================================================


class TestVsaVsFa4KernelPerformance:
    """Kernel-level VSA fine-stage vs FA4 benchmark.

    Times the two kernels directly (no QKV proj / norm / gate / tile-untile) so
    the comparison reflects only the attention math.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        if not _flash_attn4_available:
            pytest.skip(
                "FlashAttention 4 not available"
                + (f": {_fa4_import_error}" if _fa4_import_error else "")
            )
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
            CUTE_AVAILABLE,
            block_sparse_attn_from_indices_cute,
            is_cute_supported,
        )

        if not CUTE_AVAILABLE:
            pytest.skip("cuda-bindings or cutlass-dsl not importable (VSA CuTe path)")
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.warmup = 10
        self.iters = 50
        self._cute_fn = block_sparse_attn_from_indices_cute
        self._is_cute_supported = is_cute_supported

    @pytest.mark.parametrize(
        "batch,num_heads,seq_len,head_dim,block_size,sparsity",
        [
            (1, 40, 131_072, 128, 64, 0.5),
        ],
        ids=["wan_BS1_H40_S131k_D128_blk64_s0.5"],
    )
    def test_vsa_kernel_vs_fa4(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        block_size: int,
        sparsity: float,
    ):
        from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import VSA_KERNEL_MAX_CUBES

        assert seq_len % block_size == 0, "seq_len must be a multiple of block_size"
        num_cubes = seq_len // block_size
        if num_cubes > VSA_KERNEL_MAX_CUBES:
            pytest.skip(
                f"num_cubes={num_cubes} exceeds VSA_KERNEL_MAX_CUBES={VSA_KERNEL_MAX_CUBES}"
            )
        topk = max(1, int(round((1.0 - sparsity) * num_cubes)))

        # FA4 expects [B, S, H, D]; VSA CuTe expects [B, H, S, D].
        q_nhd = torch.randn(
            batch, seq_len, num_heads, head_dim, device=self.device, dtype=self.dtype
        )
        k_nhd = torch.randn_like(q_nhd)
        v_nhd = torch.randn_like(q_nhd)
        q_bhsd = q_nhd.transpose(1, 2).contiguous()
        k_bhsd = k_nhd.transpose(1, 2).contiguous()
        v_bhsd = v_nhd.transpose(1, 2).contiguous()

        if not self._is_cute_supported(q_bhsd):
            pytest.skip("VSA CuTe path requires sm_100+ Blackwell + bf16/fp16 + head_dim=128")

        # Group2QInterleaveKV requires paired Q-blocks (2i, 2i+1) to share KV
        # selections. A constant index set across all Q-blocks trivially satisfies that.
        q2k_idx = (
            torch.arange(topk, device=self.device, dtype=torch.int32)
            .view(1, 1, 1, topk)
            .expand(batch, num_heads, num_cubes, topk)
            .contiguous()
        )
        q2k_num = torch.full(
            (batch, num_heads, num_cubes), topk, dtype=torch.int32, device=self.device
        )
        variable_block_sizes = torch.full(
            (num_cubes,), block_size, dtype=torch.int32, device=self.device
        )

        softmax_scale = head_dim**-0.5

        def fa4_call():
            _fa4_fwd(
                q_nhd,
                k_nhd,
                v_nhd,
                softmax_scale=softmax_scale,
                causal=False,
                window_size_left=None,
                window_size_right=None,
                learnable_sink=None,
                softcap=0.0,
                pack_gqa=None,
                mask_mod=None,
                block_sparse_tensors=None,
                return_lse=True,
            )

        def vsa_call():
            self._cute_fn(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                q2k_idx,
                q2k_num,
                variable_block_sizes,
            )

        with torch.no_grad():
            for _ in range(self.warmup):
                fa4_call()
                vsa_call()
        torch.cuda.synchronize()

        with nvtx_range("bench_fa4"):
            fa4_times = bench_fn(self.device, fa4_call, self.iters)
        with nvtx_range("bench_vsa_fine"):
            vsa_times = bench_fn(self.device, vsa_call, self.iters)

        fa4_avg = fa4_times.mean().item()
        vsa_avg = vsa_times.mean().item()
        speedup = fa4_avg / vsa_avg

        print(
            f"\n  VSA vs FA4 (BS={batch}, H={num_heads}, S={seq_len}, "
            f"D={head_dim}, blk={block_size}, sparsity={sparsity:.2f}, "
            f"topk={topk}/{num_cubes}):"
        )
        print(f"    FA4 (dense):      avg={fa4_avg:.3f}ms")
        print(f"    VSA (fine-stage): avg={vsa_avg:.3f}ms")
        print(f"    VSA speedup vs FA4: {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")
        assert fa4_avg > 0 and vsa_avg > 0


# ============================================================================
# VSA vs VANILLA at Wan 2.2 T2V 14B production shape — module-level
# ============================================================================


class TestVsaVsVanillaWan22T2v14bModulePerformance:
    """Module-level VSA vs VANILLA at Wan 2.2 T2V 14B 720p / 81-frame shape.

    Drives the shared WanAttentionPerformanceBenchmark engine so VSA is timed
    and reported exactly like the VANILLA/TRTLLM/FA4 backends.
    """

    _BATCH = 1
    _NUM_HEADS = 40
    _HEAD_DIM = 128
    _LATENT_SHAPE = (21, 45, 80)  # DiT seq shape after VAE + patchify
    _SEQ_LEN = 21 * 45 * 80  # 75_600

    @pytest.fixture(autouse=True)
    def setup(self):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention import (
            CUTE_AVAILABLE,
            is_cute_supported,
        )

        if not CUTE_AVAILABLE:
            pytest.skip("cuda-bindings or cutlass-dsl not importable (VSA CuTe path)")
        self._is_cute_supported = is_cute_supported
        self.benchmark = WanAttentionPerformanceBenchmark(
            warmup_iterations=5, benchmark_iterations=20
        )

    @pytest.mark.parametrize(
        "sparsity",
        [0.5, 0.75, 0.85, 0.875],
        ids=["s0.5", "s0.75", "s0.85", "s0.875"],
    )
    def test_vsa_module_vs_vanilla_wan22_t2v_14b(self, sparsity: float):
        bench = self.benchmark
        probe_q = torch.empty(
            self._BATCH,
            self._NUM_HEADS,
            64,
            self._HEAD_DIM,
            device=bench.device,
            dtype=bench.dtype,
        )
        if not self._is_cute_supported(probe_q):
            pytest.skip("VSA CuTe path requires sm_100+ Blackwell + bf16/fp16 + head_dim=128")

        common = dict(
            batch_size=self._BATCH,
            num_heads=self._NUM_HEADS,
            seq_len=self._SEQ_LEN,
            head_dim=self._HEAD_DIM,
            init_std=0.02,
            verbose=False,
        )
        vanilla = bench.benchmark_single(backend="VANILLA", **common)
        vsa = bench.benchmark_single(
            backend="CUTEDSL", vsa_sparsity=sparsity, latent_shape=self._LATENT_SHAPE, **common
        )
        assert vanilla is not None and vsa is not None, "benchmark returned None (skipped/failed)"
        assert vsa["resolved_backend"] == "CUTEDSL", (
            f"VSA timings came from {vsa['resolved_backend']!r}, not the CUTEDSL/VSA path"
        )
        assert vanilla["avg_ms"] > 0 and vsa["avg_ms"] > 0

        speedup = vanilla["avg_ms"] / vsa["avg_ms"]
        print(
            f"\n  VSA vs VANILLA (Wan 2.2 T2V 14B 720p/81f, latent={self._LATENT_SHAPE}, "
            f"S={self._SEQ_LEN}, sparsity={sparsity:.3f}): "
            f"VANILLA={vanilla['avg_ms']:.3f}ms, VSA={vsa['avg_ms']:.3f}ms, "
            f"speedup={speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})"
        )


# ============================================================================
# Main entry point
# ============================================================================


# ============================================================================
# SageAttention performance tests
# ============================================================================

_sm100_only = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SageAttention cubins require SM100 (Blackwell).",
)

# All five supported (qk_dtype, v_dtype, (q_block_size, k_block_size, v_block_size)) recipes.
_QUANT_CONFIGS = [
    QuantAttentionConfig(qk_dtype="fp8", q_block_size=1, k_block_size=1, v_block_size=1),
    QuantAttentionConfig(qk_dtype="fp8", q_block_size=1, k_block_size=4, v_block_size=1),
    QuantAttentionConfig(qk_dtype="int8", q_block_size=1, k_block_size=1, v_block_size=1),
    QuantAttentionConfig(qk_dtype="int8", q_block_size=1, k_block_size=4, v_block_size=1),
    QuantAttentionConfig(qk_dtype="int8", q_block_size=1, k_block_size=16, v_block_size=1),
]

# Human-readable labels for the five configs.
_QUANT_CONFIG_IDS = ["fp8_k1", "fp8_k4", "int8_k1", "int8_k4", "int8_k16"]


@_sm100_only
class TestSageAttentionPerformance:
    """Performance benchmarks comparing VANILLA, TRTLLM, and TRTLLM+SageAttention.

    All tests use WAN model dimensions (head_dim=128, num_heads=12) because the
    SM100 SageAttention cubins are generated for head_dim=128 only.
    """

    # WAN-realistic seq_lens (VAE 8x spatial, 4x temporal, patch [1,2,2])
    WAN_SEQ_LENS = [
        (1, 12, 14040, 128, "wan_1.3b_480p_33f"),
        (1, 12, 32760, 128, "wan_1.3b_480p_81f"),
        (1, 40, 14040, 128, "wan_14b_480p_33f"),
        (1, 40, 32760, 128, "wan_14b_480p_81f"),
        (1, 40, 75600, 128, "wan_14b_720p_81f"),
    ]

    # Smaller sizes used by quick/CI tests.
    QUICK_SEQ_LENS = [
        (1, 12, 1024, 128, "quick_1k"),
        (1, 12, 4096, 128, "quick_4k"),
        (2, 12, 1024, 128, "quick_batch2"),
    ]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.benchmark = WanAttentionPerformanceBenchmark(
            warmup_iterations=5,
            benchmark_iterations=20,
        )

    def _bench(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        quant_cfg: QuantAttentionConfig,
        verbose: bool = True,
    ) -> Optional[Dict]:
        return self.benchmark.benchmark_single(
            batch,
            num_heads,
            seq_len,
            head_dim,
            backend="TRTLLM",
            verbose=verbose,
            quant_attention_config=quant_cfg,
        )

    # ------------------------------------------------------------------
    # Quick / CI tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "quant_cfg,cfg_id", zip(_QUANT_CONFIGS, _QUANT_CONFIG_IDS, strict=True)
    )
    @pytest.mark.parametrize(
        "batch,num_heads,seq_len,head_dim,desc",
        QUICK_SEQ_LENS,
    )
    def test_sage_runs_and_times(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        desc: str,
        quant_cfg: QuantAttentionConfig,
        cfg_id: str,
    ):
        """Verify every sage config produces a valid timing result at quick sizes."""
        if quant_cfg.qk_dtype == "int8" and torch.cuda.get_device_capability()[1] == 3:
            pytest.skip("SM103 does not have Int8 Tensor Cores.")

        result = self._bench(batch, num_heads, seq_len, head_dim, quant_cfg)
        assert result is not None, f"sage {cfg_id} failed for {desc}"
        assert result["uses_sage"], f"sage {cfg_id} fell back unexpected for {desc}"
        assert result["avg_ms"] > 0
        print(
            f"  sage {cfg_id} ({desc}): avg={result['avg_ms']:.3f}ms  p95={result['p95_ms']:.3f}ms"
        )

    @pytest.mark.parametrize(
        "quant_cfg,cfg_id", zip(_QUANT_CONFIGS, _QUANT_CONFIG_IDS, strict=True)
    )
    def test_sage_vs_vanilla_quick(self, quant_cfg: QuantAttentionConfig, cfg_id: str):
        """Compare SageAttention timing against VANILLA at a quick size.

        Does not assert a minimum speedup — the goal is to catch regressions
        where sage unexpectedly becomes much slower than plain SDPA.
        """
        if quant_cfg.qk_dtype == "int8" and torch.cuda.get_device_capability()[1] == 3:
            pytest.skip("SM103 does not have Int8 Tensor Cores.")

        batch, num_heads, seq_len, head_dim = 1, 12, 4096, 128

        vanilla = self.benchmark.benchmark_single(
            batch, num_heads, seq_len, head_dim, backend="VANILLA", verbose=False
        )
        trtllm = self.benchmark.benchmark_single(
            batch, num_heads, seq_len, head_dim, backend="TRTLLM", verbose=False
        )
        sage = self._bench(batch, num_heads, seq_len, head_dim, quant_cfg)

        assert vanilla is not None, "VANILLA benchmark failed"
        assert trtllm is not None, "TRTLLM benchmark failed"
        assert sage is not None, f"SageAttention {cfg_id} benchmark failed"
        assert sage["uses_sage"], f"SageAttention {cfg_id} fell back unexpected at S={seq_len}"

        speedup_vs_vanilla = vanilla["avg_ms"] / sage["avg_ms"]
        speedup_vs_trtllm = trtllm["avg_ms"] / sage["avg_ms"]
        print(
            f"\n  sage {cfg_id} at S={seq_len}: "
            f"avg={sage['avg_ms']:.3f}ms  "
            f"vs_vanilla={speedup_vs_vanilla:.2f}x  "
            f"vs_trtllm={speedup_vs_trtllm:.2f}x"
        )

    # ------------------------------------------------------------------
    # Full WAN-shape benchmarks
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "quant_cfg,cfg_id", zip(_QUANT_CONFIGS, _QUANT_CONFIG_IDS, strict=True)
    )
    @pytest.mark.parametrize(
        "batch,num_heads,seq_len,head_dim,model_name",
        WAN_SEQ_LENS,
    )
    def test_sage_vs_trtllm_wan_shapes(
        self,
        batch: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        model_name: str,
        quant_cfg: QuantAttentionConfig,
        cfg_id: str,
    ):
        """SageAttention vs TRTLLM on real WAN model sequence lengths."""
        if quant_cfg.qk_dtype == "int8" and torch.cuda.get_device_capability()[1] == 3:
            pytest.skip("SM103 does not have Int8 Tensor Cores.")

        trtllm = self.benchmark.benchmark_single(
            batch, num_heads, seq_len, head_dim, backend="TRTLLM", verbose=False
        )
        sage = self._bench(batch, num_heads, seq_len, head_dim, quant_cfg)

        assert trtllm is not None, f"TRTLLM failed for {model_name}"
        assert sage is not None, f"sage {cfg_id} failed for {model_name}"
        assert sage["uses_sage"], f"sage {cfg_id} fell back unexpected for {model_name}"

        speedup = trtllm["avg_ms"] / sage["avg_ms"]
        print(
            f"\n  {model_name} sage={cfg_id}: "
            f"trtllm={trtllm['avg_ms']:.3f}ms  "
            f"sage={sage['avg_ms']:.3f}ms  "
            f"speedup={speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})"
        )


def _run_fa4_benchmark(benchmark: WanAttentionPerformanceBenchmark) -> None:
    """Run FA4 vs VANILLA comparison benchmark and print summary table."""
    benchmark.backends = ["VANILLA", "FA4"]

    print("\n" + "=" * 70)
    print("FLASH ATTENTION 4 vs VANILLA BENCHMARK")
    print("=" * 70)
    print(f"{'Configuration':<40} {'VANILLA (ms)':<15} {'FA4 (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, num_heads, seq_len, head_dim, desc in benchmark.TEST_SIZES:
        results = benchmark.benchmark_comparison(
            batch_size, num_heads, seq_len, head_dim, desc, verbose=False
        )
        vanilla = results.get("VANILLA")
        fa4 = results.get("FA4")

        vanilla_str = f"{vanilla['avg_ms']:.2f}" if vanilla else "N/A"
        fa4_str = f"{fa4['avg_ms']:.2f}" if fa4 else "N/A"
        if vanilla and fa4:
            speedup = vanilla["avg_ms"] / fa4["avg_ms"]
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{desc:<40} {vanilla_str:<15} {fa4_str:<15} {speedup_str:<10}")


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

    # Run full benchmark (VANILLA vs TRTLLM)
    print("\n" + "=" * 70)
    print("FULL BENCHMARK (VANILLA vs TRTLLM)")
    print("=" * 70)
    all_results = benchmark.run_full_benchmark(use_quick_sizes=False)

    # Memory test
    if torch.cuda.is_available():
        benchmark.test_memory_usage(batch_size=1, num_heads=24, seq_len=4096, head_dim=64)

    # FA4 benchmark (if available)
    if _flash_attn4_available:
        fa4_benchmark = WanAttentionPerformanceBenchmark(
            warmup_iterations=10,
            benchmark_iterations=50,
        )
        _run_fa4_benchmark(fa4_benchmark)
    else:
        print("\nSkipping FA4 benchmark (FlashAttention 4 not installed)")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
