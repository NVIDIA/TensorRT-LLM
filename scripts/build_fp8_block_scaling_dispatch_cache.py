#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a correctness-gated FP8 block-scaling GEMM dispatch cache."""

import argparse
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch import (
    BackendMeasurement,
    CacheIdentity,
    DispatchBackend,
    DispatchCache,
    DispatchEntry,
    DispatchKey,
    DispatchPolicy,
    choose_fastest_correct,
    select_static_backend,
)
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_cache import write_dispatch_cache
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_runtime import (
    is_deep_gemm_compatible,
    make_dispatch_key,
    make_runtime_identity,
)


@dataclass(frozen=True, slots=True)
class GemmShape:
    m: int
    n: int
    k: int
    count: int = 1


def _parse_shape(value: str) -> GemmShape:
    shape_value, separator, count_value = value.partition(":")
    dimensions = shape_value.lower().split("x")
    if len(dimensions) != 3:
        raise argparse.ArgumentTypeError(f"shape must be formatted as MxNxK, got {value!r}")
    m, n, k = (int(dimension) for dimension in dimensions)
    if min(m, n, k) <= 0:
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {value}")
    count = int(count_value) if separator else 1
    if count <= 0:
        raise argparse.ArgumentTypeError(f"shape count must be positive: {value}")
    return GemmShape(m=m, n=n, k=k, count=count)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        type=_parse_shape,
        help="MxNxK or MxNxK:frequency",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--correctness-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Skip shapes where no correctness-gated backend is available.",
    )
    return parser.parse_args()


def _per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    padded_m = ((m + 127) // 128) * 128
    padded_n = ((n + 127) // 128) * 128
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    blocks = x_padded.view(-1, 128, padded_n // 128, 128)
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    quantized = (blocks * (448.0 / amax)).to(torch.float8_e4m3fn)
    return (
        quantized.view_as(x_padded)[:m, :n].contiguous(),
        (amax / 448.0).view(blocks.size(0), blocks.size(2)),
    )


def _relative_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_double = actual.double()
    expected_double = expected.double()
    denominator = (actual_double.square() + expected_double.square()).sum()
    similarity = 2 * (actual_double * expected_double).sum() / denominator
    return float(1 - similarity)


def _measure(
    backend: DispatchBackend,
    operation: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    correctness_threshold: float,
) -> BackendMeasurement:
    output = operation()
    torch.cuda.synchronize()
    correct = (
        bool(torch.isfinite(output).all())
        and _relative_diff(output, reference) < correctness_threshold
    )

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return BackendMeasurement(
        backend=backend,
        correct=correct,
        median_ms=statistics.median(timings),
    )


def _safe_measure(
    shape: GemmShape,
    backend: DispatchBackend,
    operation: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    correctness_threshold: float,
) -> BackendMeasurement:
    try:
        return _measure(
            backend,
            operation,
            reference,
            warmup=warmup,
            iterations=iterations,
            correctness_threshold=correctness_threshold,
        )
    except RuntimeError as error:
        print(
            f"warning={shape.m}x{shape.n}x{shape.k} backend={backend.value} failed: {error}",
            file=sys.stderr,
        )
        return BackendMeasurement(
            backend=backend,
            correct=False,
            median_ms=float("inf"),
        )


def _unsupported_measurement(
    shape: GemmShape,
    backend: DispatchBackend,
    reason: str,
) -> BackendMeasurement:
    print(
        f"warning={shape.m}x{shape.n}x{shape.k} backend={backend.value} skipped: {reason}",
        file=sys.stderr,
    )
    return BackendMeasurement(
        backend=backend,
        correct=False,
        median_ms=float("inf"),
    )


def _is_trtllm_measurement_supported(key: DispatchKey, identity: CacheIdentity) -> bool:
    return identity.sm != 90 or key.k % 128 == 0


def _build_entry(
    shape: GemmShape,
    args: argparse.Namespace,
) -> tuple[DispatchEntry | None, tuple[BackendMeasurement, ...]]:
    torch.manual_seed(0)
    a_bf16 = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16) / shape.k
    b_bf16 = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16) / shape.k
    a, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a_bf16)
    b, b_scale = _per_block_cast_to_fp8(b_bf16)
    reference = a_bf16 @ b_bf16.t()
    identity = make_runtime_identity(a)
    key = make_dispatch_key(a, b, a_scale, b_scale)

    if _is_trtllm_measurement_supported(key, identity):
        trt = _safe_measure(
            shape,
            DispatchBackend.TRTLLM,
            lambda: torch.ops.trtllm.fp8_block_scaling_gemm_trtllm(a, b, a_scale, b_scale),
            reference,
            warmup=args.warmup,
            iterations=args.iters,
            correctness_threshold=args.correctness_threshold,
        )
    else:
        trt = _unsupported_measurement(
            shape,
            DispatchBackend.TRTLLM,
            "Hopper FP8 block scaling GEMM requires K to be a multiple of 128",
        )
    measurements = [trt]
    if identity.deep_gemm_available and is_deep_gemm_compatible(
        key, (a, b, a_scale, b_scale)
    ):
        measurements.append(
            _safe_measure(
                shape,
                DispatchBackend.DEEP_GEMM,
                lambda: torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm(a, b, a_scale, b_scale),
                reference,
                warmup=args.warmup,
                iterations=args.iters,
                correctness_threshold=args.correctness_threshold,
            )
        )

    static_decision = select_static_backend(DispatchPolicy(), key, identity.sm)
    selected = (
        static_decision.backend
        if static_decision is not None
        else choose_fastest_correct(tuple(measurements))
    )
    if selected is None or not any(
        measurement.backend is selected and measurement.correct for measurement in measurements
    ):
        return None, tuple(measurements)
    return DispatchEntry(key=key, backend=selected), tuple(measurements)


def main() -> None:
    args = _parse_args()
    entries = []
    identity = make_runtime_identity(
        torch.empty(0, device="cuda", dtype=torch.float8_e4m3fn)
    )
    weighted_trt_ms = 0.0
    weighted_selected_ms = 0.0
    print("shape,backend,correct,median_ms,selected")
    for shape in args.shape:
        entry, measurements = _build_entry(shape, args)
        for measurement in measurements:
            selected = entry is not None and entry.backend is measurement.backend
            print(
                f"{shape.m}x{shape.n}x{shape.k},{measurement.backend.value},"
                f"{str(measurement.correct).lower()},{measurement.median_ms:.6f},"
                f"{str(selected).lower()}"
            )
        if entry is None:
            message = f"No correctness-gated backend for {shape.m}x{shape.n}x{shape.k}"
            if args.skip_unsupported:
                print(f"unsupported={shape.m}x{shape.n}x{shape.k}")
                continue
            raise RuntimeError(message)
        entries.append(entry)
        trt_measurement = next(
            measurement
            for measurement in measurements
            if measurement.backend is DispatchBackend.TRTLLM
        )
        selected_measurement = next(
            measurement for measurement in measurements if measurement.backend is entry.backend
        )
        weighted_trt_ms += trt_measurement.median_ms * shape.count
        weighted_selected_ms += selected_measurement.median_ms * shape.count
    write_dispatch_cache(args.output, DispatchCache(identity, tuple(entries)))
    print(f"cache={args.output}")
    print(f"synthetic_replay_trt_ms={weighted_trt_ms:.6f}")
    print(f"synthetic_replay_dispatched_ms={weighted_selected_ms:.6f}")
    if weighted_selected_ms > 0.0:
        print(f"synthetic_replay_speedup={weighted_trt_ms / weighted_selected_ms:.6f}")
    else:
        print("synthetic_replay_speedup=nan")


if __name__ == "__main__":
    main()
