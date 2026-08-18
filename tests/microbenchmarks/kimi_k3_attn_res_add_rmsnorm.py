# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the incremental Kimi K3 attention-output add fusion.

The production baseline after the trailing-RMSNorm fusion is two kernels:

    updated_prefix = prefix_sum + attention_output
    trtllm::attn_res_rmsnorm_fwd(updated_prefix, ...)

The new path is one kernel and still materializes ``updated_prefix`` for the
MLP residual that follows:

    updated_prefix, output =
        trtllm::attn_res_add_rmsnorm_fwd(prefix_sum, attention_output, ...)

CUDA-graph replay timings remove Python and dispatcher overhead. ``--profile``
emits eager calls inside NVTX ranges for Nsys attribution.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm

HIDDEN_SIZE = 7168
RMS_EPS = 1e-6


@dataclass
class CaseInputs:
    prefix_sum: torch.Tensor
    attention_output: torch.Tensor
    block_residual: torch.Tensor
    res_weight: torch.Tensor
    score_rms_weight: torch.Tensor
    output_rms_weight: torch.Tensor


def _parse_candidates(value: str) -> list[int]:
    candidates = [int(item) for item in value.split(",")]
    if any(candidate not in {*range(1, 10), 12} for candidate in candidates):
        raise argparse.ArgumentTypeError("candidate counts must be in [1, 9] or equal to 12")
    return candidates


def _make_inputs(num_candidates: int) -> CaseInputs:
    device = torch.device("cuda")
    shape = (1, 1, HIDDEN_SIZE)
    prefix_sum = torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(-0.05, 0.05)
    attention_output = torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(-0.05, 0.05)
    block_residual = torch.empty(
        (num_candidates - 1, *shape),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.05, 0.05)
    res_weight = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device).uniform_(-0.02, 0.02)
    score_rms_weight = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device).uniform_(
        0.98, 1.02
    )
    output_rms_weight = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device=device).uniform_(
        0.98, 1.02
    )
    return CaseInputs(
        prefix_sum=prefix_sum,
        attention_output=attention_output,
        block_residual=block_residual,
        res_weight=res_weight,
        score_rms_weight=score_rms_weight,
        output_rms_weight=output_rms_weight,
    )


def _attn_res(inputs: CaseInputs, updated_prefix: torch.Tensor) -> torch.Tensor:
    output, _rsigma, _probs, _logits = torch.ops.trtllm.attn_res_fwd(
        updated_prefix,
        inputs.block_residual,
        inputs.res_weight,
        inputs.score_rms_weight,
        RMS_EPS,
    )
    return output


def _three_kernel(inputs: CaseInputs) -> tuple[torch.Tensor, torch.Tensor]:
    updated_prefix = inputs.prefix_sum + inputs.attention_output
    mixed = _attn_res(inputs, updated_prefix)
    output = flashinfer_rmsnorm(mixed, inputs.output_rms_weight, RMS_EPS)
    return updated_prefix, output


def _two_kernel(inputs: CaseInputs) -> tuple[torch.Tensor, torch.Tensor]:
    updated_prefix = inputs.prefix_sum + inputs.attention_output
    output = torch.ops.trtllm.attn_res_rmsnorm_fwd(
        updated_prefix,
        inputs.block_residual,
        inputs.res_weight,
        inputs.score_rms_weight,
        inputs.output_rms_weight,
        RMS_EPS,
        RMS_EPS,
    )
    return updated_prefix, output


def _fused(inputs: CaseInputs) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        inputs.prefix_sum,
        inputs.attention_output,
        inputs.block_residual,
        inputs.res_weight,
        inputs.score_rms_weight,
        inputs.output_rms_weight,
        RMS_EPS,
        RMS_EPS,
    )


def _capture(
    fn: Callable[[], Any],
) -> tuple[torch.cuda.CUDAGraph, Any]:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _time_graph(
    fn: Callable[[], Any],
    iterations: int,
    samples: int,
    chain_length: int,
) -> tuple[float, float, float]:
    def chained_fn() -> Any:
        output = None
        for _ in range(chain_length):
            output = fn()
        return output

    graph, output = _capture(chained_fn)
    del output
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations / chain_length)
    return statistics.median(timings), min(timings), max(timings)


def _similarity(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    return cosine, relative_l2


def _benchmark_case(
    num_candidates: int,
    iterations: int,
    samples: int,
    chain_length: int,
) -> dict[str, float | int]:
    inputs = _make_inputs(num_candidates)
    expected_prefix, expected_output = _two_kernel(inputs)
    actual_prefix, actual_output = _fused(inputs)
    _, three_kernel_output = _three_kernel(inputs)
    torch.cuda.synchronize()
    if not torch.equal(actual_prefix, expected_prefix):
        raise AssertionError(f"N={num_candidates}: fused updated prefix is not exact")
    cosine, relative_l2 = _similarity(actual_output, expected_output)
    if cosine <= 0.9999 or relative_l2 >= 5e-3:
        raise AssertionError(f"N={num_candidates}: cosine={cosine}, relative_l2={relative_l2}")
    # attn_res_fwd + the production RMSNorm is what shipped before the trailing
    # norm was folded into the kernel, so this -- not two_kernel, which is also
    # a norm-fusing op -- is the baseline the fusion has to be judged against.
    three_cosine, three_relative_l2 = _similarity(actual_output, three_kernel_output)

    add_us, add_min_us, add_max_us = _time_graph(
        lambda: inputs.prefix_sum + inputs.attention_output,
        iterations,
        samples,
        chain_length,
    )
    three_us, three_min_us, three_max_us = _time_graph(
        lambda: _three_kernel(inputs), iterations, samples, chain_length
    )
    two_us, two_min_us, two_max_us = _time_graph(
        lambda: _two_kernel(inputs), iterations, samples, chain_length
    )
    fused_us, fused_min_us, fused_max_us = _time_graph(
        lambda: _fused(inputs), iterations, samples, chain_length
    )

    return {
        "num_tokens": 1,
        "num_candidates": num_candidates,
        "iterations": iterations,
        "samples": samples,
        "chain_length": chain_length,
        "cosine": cosine,
        "relative_l2": relative_l2,
        "cosine_vs_three_kernel": three_cosine,
        "relative_l2_vs_three_kernel": three_relative_l2,
        "add_us": add_us,
        "add_min_us": add_min_us,
        "add_max_us": add_max_us,
        "three_kernel_us": three_us,
        "three_kernel_min_us": three_min_us,
        "three_kernel_max_us": three_max_us,
        "two_kernel_us": two_us,
        "two_kernel_min_us": two_min_us,
        "two_kernel_max_us": two_max_us,
        "fused_us": fused_us,
        "fused_min_us": fused_min_us,
        "fused_max_us": fused_max_us,
        "fused_vs_two_kernel_pct": (fused_us / two_us - 1.0) * 100.0,
        "saved_vs_two_kernel_us": two_us - fused_us,
        "fused_vs_three_kernel_pct": (fused_us / three_us - 1.0) * 100.0,
        "saved_vs_three_kernel_us": three_us - fused_us,
    }


def _profile_case(
    num_candidates: int,
    iterations: int,
) -> None:
    inputs = _make_inputs(num_candidates)
    modes: Sequence[tuple[str, Callable[[], Any]]] = (
        ("add", lambda: inputs.prefix_sum + inputs.attention_output),
        ("three_kernel", lambda: _three_kernel(inputs)),
        ("two_kernel", lambda: _two_kernel(inputs)),
        ("fused", lambda: _fused(inputs)),
    )
    for _name, fn in modes:
        for _ in range(10):
            fn()
    torch.cuda.synchronize()

    for name, fn in modes:
        range_name = f"attn_res_add|T=1|N={num_candidates}|mode={name}"
        torch.cuda.nvtx.range_push(range_name)
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(
            json.dumps(
                {"profile_range": range_name, "iterations": iterations},
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=_parse_candidates,
        default=_parse_candidates("1,2,3,4,5,6,7,8,9,12"),
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--chain-length",
        type=int,
        default=1,
        help="Capture this many copies of each mode in one CUDA graph.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit eager kernels in NVTX ranges for Nsys instead of timing.",
    )
    args = parser.parse_args()

    if args.chain_length < 1:
        parser.error("--chain-length must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError(f"SM100/SM103 is required, got {capability}")
    torch.manual_seed(0)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": capability,
                "profile": args.profile,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for num_candidates in args.candidates:
        if args.profile:
            _profile_case(num_candidates, min(args.iterations, 100))
        else:
            print(
                json.dumps(
                    _benchmark_case(
                        num_candidates, args.iterations, args.samples, args.chain_length
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
