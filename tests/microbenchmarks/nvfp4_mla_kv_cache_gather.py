# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the GLM-5.2 and DeepSeek-V4-Pro NVFP4 MLA gather kernels.

The model shapes and Top-K values intentionally match the checkpoints:

* GLM-5.2: ``nvfp4_mla_kv_cache_gather``, Top-K 2048, 576-wide KV
  with a 64-wide residual-quantized suffix.
* DeepSeek-V4-Pro: ``nvfp4_mla_kv_cache_gather_direct``, Top-K 1024,
  512-wide compressed KV with a 64-wide residual-quantized RoPE suffix.

Run on one Blackwell GPU; no model weights are needed::

    python tests/microbenchmarks/nvfp4_mla_kv_cache_gather.py

By default the benchmark sweeps batch sizes 1, 2, 4, 8, 16, 32, 64, and 128.
Use ``--csv results.csv`` to also save machine-readable results.
"""

import argparse
import csv
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401


@dataclass(frozen=True)
class ModelSpec:
    name: str
    operator: str
    topk: int
    head_dim: int
    residual_dim: int
    direct: bool

    @property
    def data_bytes_per_token(self) -> int:
        return (self.head_dim + self.residual_dim) // 2

    @property
    def scales_per_token(self) -> int:
        return (self.head_dim + self.residual_dim) // 16

    @property
    def bytes_per_pair(self) -> int:
        # Packed values + scales + FP8 output + index read/write.
        return (
            self.data_bytes_per_token
            + self.scales_per_token
            + self.head_dim
            + 2 * torch.int32.itemsize
        )


MODEL_SPECS = {
    "glm52": ModelSpec(
        name="GLM-5.2",
        operator="trtllm::nvfp4_mla_kv_cache_gather",
        topk=2048,
        head_dim=576,
        residual_dim=64,
        direct=False,
    ),
    "dsv4-pro": ModelSpec(
        name="DeepSeek-V4-Pro",
        operator="trtllm::nvfp4_mla_kv_cache_gather_direct",
        topk=1024,
        head_dim=512,
        residual_dim=64,
        direct=True,
    ),
}


class GatherBenchmark:
    def __init__(
        self,
        spec: ModelSpec,
        max_batch_size: int,
        num_pool_tokens: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.num_pool_tokens = num_pool_tokens
        self.data_pool = torch.full(
            (num_pool_tokens, spec.data_bytes_per_token),
            0x22,
            dtype=torch.uint8,
            device=device,
        )
        self.scale_pool = torch.ones(
            (num_pool_tokens, spec.scales_per_token),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self.indices_template = torch.randint(
            num_pool_tokens,
            (max_batch_size, spec.topk),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        self.indices = self.indices_template.clone()
        self.output = torch.empty(
            (max_batch_size, spec.topk, spec.head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.global_dequant_scale = torch.ones(1, dtype=torch.float32, device=device)

        if spec.direct:
            self.compact_indices = None
            self.host_pool_pointers = None
            self.host_pool_mapping = None
        else:
            self.compact_indices = torch.empty_like(self.indices)
            self.host_pool_pointers = torch.zeros((1, 2, 2), dtype=torch.int64)
            self.host_pool_pointers[0, 0, 0] = self.data_pool.data_ptr()
            self.host_pool_pointers[0, 0, 1] = self.scale_pool.data_ptr()
            self.host_pool_mapping = torch.tensor([[0, 0]], dtype=torch.int32)

    def call_for_batch(self, batch_size: int) -> Callable[[], None]:
        spec = self.spec
        indices = self.indices[:batch_size]
        output = self.output[:batch_size]
        if spec.direct:

            def run() -> None:
                torch.ops.trtllm.nvfp4_mla_kv_cache_gather_direct(
                    self.data_pool,
                    self.scale_pool,
                    indices,
                    output,
                    self.global_dequant_scale,
                    spec.residual_dim,
                    self.num_pool_tokens,
                )

            return run

        compact_indices = self.compact_indices
        host_pool_pointers = self.host_pool_pointers
        host_pool_mapping = self.host_pool_mapping
        assert compact_indices is not None
        assert host_pool_pointers is not None
        assert host_pool_mapping is not None
        compact_indices = compact_indices[:batch_size]

        def run() -> None:
            torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
                host_pool_pointers,
                host_pool_mapping,
                indices,
                output,
                compact_indices,
                self.global_dequant_scale,
                0,
                spec.residual_dim,
                self.num_pool_tokens,
            )

        return run

    def reset_for_batch(self, batch_size: int) -> Callable[[], None] | None:
        if not self.spec.direct:
            return None

        def reset() -> None:
            self.indices[:batch_size].copy_(self.indices_template[:batch_size])

        return reset


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _time_cuda_us(
    fn: Callable[[], None],
    reset: Callable[[], None] | None,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        if reset is not None:
            reset()
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        if reset is not None:
            # The direct DSV4 op compacts indices in place. Restore the
            # indexer output before each sample, outside the timed interval.
            reset()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _run_model(
    spec: ModelSpec,
    batch_sizes: list[int],
    num_pool_tokens: int,
    seed: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[dict[str, int | float | str]]:
    max_batch_size = max(batch_sizes)
    benchmark = GatherBenchmark(spec, max_batch_size, num_pool_tokens, seed, device)
    print(
        f"\n{spec.name}: op={spec.operator} topk={spec.topk} "
        f"head_dim={spec.head_dim} residual_dim={spec.residual_dim}"
    )
    print(" batch   topk   p50_us   p90_us  mean_us  effective_GB/s")

    results = []
    for batch_size in batch_sizes:
        samples = _time_cuda_us(
            benchmark.call_for_batch(batch_size),
            benchmark.reset_for_batch(batch_size),
            warmup,
            iters,
        )
        p50_us = statistics.median(samples)
        p90_us = _percentile(samples, 0.90)
        mean_us = statistics.fmean(samples)
        num_pairs = batch_size * spec.topk
        effective_gb_s = num_pairs * spec.bytes_per_pair / p50_us / 1000.0
        print(
            f"{batch_size:6d} {spec.topk:6d} {p50_us:8.2f} {p90_us:8.2f} "
            f"{mean_us:8.2f} {effective_gb_s:15.2f}"
        )
        results.append(
            {
                "model": spec.name,
                "operator": spec.operator,
                "batch_size": batch_size,
                "topk": spec.topk,
                "head_dim": spec.head_dim,
                "residual_dim": spec.residual_dim,
                "num_pairs": num_pairs,
                "p50_us": p50_us,
                "p90_us": p90_us,
                "mean_us": mean_us,
                "effective_gb_s": effective_gb_s,
            }
        )
    return results


def _write_csv(path: Path, results: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SPECS,
        default=list(MODEL_SPECS),
        help="Models to benchmark (default: glm52 dsv4-pro).",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Batch sizes to sweep (default: 1 2 4 8 16 32 64 128).",
    )
    parser.add_argument(
        "--pool-tokens",
        type=int,
        default=None,
        help="KV pool rows; default is max(batch_sizes) * model Top-K.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path.")
    args = parser.parse_args()

    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("--batch-sizes values must be positive")
    if args.pool_tokens is not None and args.pool_tokens <= 0:
        parser.error("--pool-tokens must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 10:
        raise RuntimeError(
            f"NVFP4 MLA gather requires a Blackwell GPU, got compute capability {capability}"
        )

    batch_sizes = args.batch_sizes
    max_batch_size = max(batch_sizes)
    all_results = []
    for model in args.models:
        spec = MODEL_SPECS[model]
        num_pool_tokens = args.pool_tokens or max_batch_size * spec.topk
        all_results.extend(
            _run_model(
                spec,
                batch_sizes,
                num_pool_tokens,
                args.seed,
                args.warmup,
                args.iters,
                device,
            )
        )
        torch.cuda.empty_cache()

    if args.csv is not None:
        _write_csv(args.csv, all_results)
        print(f"\nSaved {len(all_results)} rows to {args.csv}")


if __name__ == "__main__":
    main()
