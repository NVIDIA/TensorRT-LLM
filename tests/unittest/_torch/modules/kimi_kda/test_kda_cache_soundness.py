# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill must use the caller's stream and current packed sequence metadata.

Delayed producer operations exercise stream ordering. Repeated calls with
changed sequence boundaries exercise metadata regeneration.
"""

import gc

import pytest
import torch

pytest.importorskip("fla")

from kda_prefill_test_utils import (  # noqa: E402
    assert_kda_close,
    run_fla_prefill,
    run_indexed_prefill,
)

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch  # noqa: E402

NUM_HEADS = 96
HEAD_DIM = 128


def _has_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in {(10, 0), (10, 3), (10, 7)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="FlashInfer KDA prefill requires SM100/SM103/SM107",
)


@pytest.fixture(scope="module")
def dispatch_pair() -> tuple[KDAKernelDispatch, KDAKernelDispatch]:
    optimized = KDAKernelDispatch(use_optimized_prefill=True, use_optimized_decode=False)
    reference = KDAKernelDispatch(use_optimized_prefill=False, use_optimized_decode=False)
    assert optimized.prefill_kernel_path == "optimized"
    assert reference.prefill_kernel_path == "fla"
    return optimized, reference


@pytest.fixture(scope="module")
def gate_params() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    a_log = torch.randn(NUM_HEADS, generator=generator, dtype=torch.float32, device="cuda") * 0.5
    dt_bias = (
        torch.randn(
            NUM_HEADS * HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1
    )
    return a_log, dt_bias


def _make_inputs(total_tokens: int, seed: int, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def random_tensor(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(
            *shape,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        ).to(dtype)

    q = random_tensor(batch_size, total_tokens, NUM_HEADS, HEAD_DIM)
    k = random_tensor(batch_size, total_tokens, NUM_HEADS, HEAD_DIM)
    v = random_tensor(batch_size, total_tokens, NUM_HEADS, HEAD_DIM)
    g = random_tensor(batch_size, total_tokens, NUM_HEADS, HEAD_DIM)
    beta = random_tensor(batch_size, total_tokens, NUM_HEADS, dtype=torch.float32)
    return q, k, v, g, beta


def _make_cu_seqlens(sequence_lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )


@pytest.mark.parametrize("regime", ["eqlen", "varlen"])
@torch.no_grad()
def test_indexed_prefill_uses_current_stream(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
    regime: str,
) -> None:
    """Inputs, indexed state updates, and consumers stay on the current stream."""
    optimized, reference = dispatch_pair
    if regime == "eqlen":
        inputs = _make_inputs(256, seed=31337, batch_size=2)
        cu_seqlens = None
    else:
        sequence_lengths = [100, 257, 300]
        inputs = _make_inputs(sum(sequence_lengths), seed=31337)
        cu_seqlens = _make_cu_seqlens(sequence_lengths)

    reference_output, reference_state = run_fla_prefill(
        reference,
        gate_params,
        inputs,
        cu_seqlens,
    )
    run_indexed_prefill(optimized, gate_params, inputs, cu_seqlens)
    torch.cuda.synchronize()

    execution_stream = torch.cuda.Stream()
    with torch.cuda.stream(execution_stream):
        torch.cuda._sleep(1 << 28)
        streamed_inputs = tuple(tensor * 1.0 for tensor in inputs)
        streamed_cu_seqlens = _make_cu_seqlens(sequence_lengths) if regime == "varlen" else None
        actual_output, actual_state, _ = run_indexed_prefill(
            optimized,
            gate_params,
            streamed_inputs,
            streamed_cu_seqlens,
        )
        actual_output = actual_output * 1.0
        actual_state = actual_state * 1.0
    torch.cuda.synchronize()

    assert_kda_close(f"stream_{regime}/output", actual_output, reference_output)
    assert_kda_close(f"stream_{regime}/state", actual_state, reference_state)


@pytest.mark.parametrize(
    "aligned_lengths,unaligned_lengths",
    [([128, 256, 192], [100, 257, 219]), ([256], [300])],
    ids=["varlen", "single_sequence"],
)
@torch.no_grad()
def test_replaced_packed_metadata_matches_fla(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
    aligned_lengths: list[int],
    unaligned_lengths: list[int],
) -> None:
    """Each invocation must use its current packed sequence boundaries."""
    optimized, reference = dispatch_pair
    for iteration, lengths in enumerate((aligned_lengths, unaligned_lengths, aligned_lengths)):
        inputs = _make_inputs(sum(lengths), seed=11 + iteration)
        cu = _make_cu_seqlens(lengths)
        actual_output, actual_state, _ = run_indexed_prefill(
            optimized,
            gate_params,
            inputs,
            cu,
        )
        expected_output, expected_state = run_fla_prefill(
            reference,
            gate_params,
            inputs,
            cu.clone(),
        )
        assert_kda_close("metadata/output", actual_output, expected_output)
        assert_kda_close("metadata/state", actual_state, expected_state)


@torch.no_grad()
def test_prefill_scratch_plateaus_over_distinct_token_counts(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Prefill must not retain GPU scratch for every distinct packed token count."""
    optimized, _ = dispatch_pair

    def run(first_length: int, seed: int) -> None:
        lengths = [first_length, 200]
        inputs = _make_inputs(sum(lengths), seed=seed)
        run_indexed_prefill(optimized, gate_params, inputs, _make_cu_seqlens(lengths))

    def settle() -> None:
        torch.cuda.synchronize()
        gc.collect()

    # Warm compilation before measuring retained tensor storage. The measured
    # calls each use a new token count, so a cache retaining per-shape scratch
    # cannot hide behind reuse of the warmup allocations.
    for i in range(2):
        run(100 + 64 * i, seed=700 + i)
    settle()
    baseline = torch.cuda.memory_allocated()

    num_calls = 24
    for i in range(num_calls):
        run(300 + 64 * i, seed=800 + i)
    settle()
    grown = torch.cuda.memory_allocated() - baseline
    # Allow fixed runtime overhead, but not one large scratch set per shape.
    budget = 128 << 20
    assert grown <= budget, (
        f"prefill retains {grown / (1 << 20):.0f} MiB after {num_calls} new token counts; "
        f"budget is {budget / (1 << 20):.0f} MiB"
    )
