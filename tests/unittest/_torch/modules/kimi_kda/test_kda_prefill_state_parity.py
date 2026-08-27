# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State parity tests: indexed optimized KDA prefill vs FLA, kernel level.

Complements test_kda_prefill_op.py (module outputs, no recurrent-state pool)
with the cases the executor runtime actually exercises through
KDAKernelDispatch:

  * selected state-pool row parity in the pool's V-first [N, H, V, K] layout
    — K == V == 128 for Kimi K3, so a layout mix-up is invisible to shape
    checks and only caught numerically;
  * a LARGE carried initial state — the K4 sign error fixed by dev-tech commit
    e45ae259 (NV = U - W@S, was U + W@S) is masked by tiny/zero initial states
    and random inputs, and only blows up when a real-magnitude state is
    carried in;
  * non-64-aligned varlen (single-sequence pad path and multi-sequence masked
    path);
  * fresh input tensors on every call — the runtime never reuses tensor
    objects, so wrapper and compile-cache behavior sees runtime-style tensor
    lifetimes instead of benchmark-style stable-object reuse.
"""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch  # noqa: E402
from tests.unittest._torch.modules.kimi_kda.kda_prefill_test_utils import (  # noqa: E402
    assert_kda_close,
    run_fla_prefill,
    run_indexed_prefill,
)

NUM_HEADS = 96
HEAD_DIM = 128


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
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


def _make_inputs(
    batch_size: int,
    total_tokens: int,
    seed: int,
    *,
    num_heads: int = NUM_HEADS,
    initial_state_scale: float | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def random_tensor(
        *shape: int,
        dtype: torch.dtype = torch.bfloat16,
        scale: float = 1.0,
    ) -> torch.Tensor:
        return (
            torch.randn(
                *shape,
                generator=generator,
                dtype=torch.float32,
                device="cuda",
            ).to(dtype)
            * scale
        )

    inputs = (
        random_tensor(batch_size, total_tokens, num_heads, HEAD_DIM),
        random_tensor(batch_size, total_tokens, num_heads, HEAD_DIM),
        random_tensor(batch_size, total_tokens, num_heads, HEAD_DIM),
        random_tensor(batch_size, total_tokens, num_heads, HEAD_DIM),
        random_tensor(batch_size, total_tokens, num_heads, dtype=torch.float32),
    )
    initial_state = None
    if initial_state_scale is not None:
        initial_state = random_tensor(
            batch_size,
            num_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            scale=initial_state_scale,
        )
        value_axis = torch.linspace(0.5, 1.5, HEAD_DIM, device="cuda").view(1, 1, -1, 1)
        initial_state = initial_state * value_axis
    return inputs, initial_state


def _make_cu_seqlens(sequence_lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )


def _expand_initial_state(initial_state: torch.Tensor | None, num_sequences: int):
    if initial_state is None or initial_state.shape[0] == num_sequences:
        return initial_state
    sequence_scale = torch.linspace(0.5, 1.5, num_sequences, device="cuda").view(
        num_sequences, 1, 1, 1
    )
    return initial_state[:1].expand(num_sequences, -1, -1, -1).contiguous() * sequence_scale


@pytest.mark.parametrize(
    "batch_size,sequence_length,sequence_lengths,initial_state_scale",
    [
        (2, 256, None, None),
        (1, 300, None, 1.0),
        (1, None, [128, 256, 192], 1.0),
        (1, None, [100, 257, 64], 1.0),
        (1, None, [300], 1.0),
    ],
    ids=[
        "eqlen_fresh",
        "eqlen_padded_carried",
        "varlen_aligned_carried",
        "varlen_unaligned_carried",
        "single_unaligned_carried",
    ],
)
@torch.no_grad()
def test_indexed_state_pool_matches_fla(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    sequence_length: int | None,
    sequence_lengths: list[int] | None,
    initial_state_scale: float | None,
) -> None:
    """Cover equal-length and packed state layouts, padding, and carried states."""
    optimized, reference = dispatch_pair
    total_tokens = sequence_length if sequence_lengths is None else sum(sequence_lengths)
    inputs, initial_state = _make_inputs(
        batch_size,
        total_tokens,
        seed=100 + batch_size + total_tokens,
        initial_state_scale=initial_state_scale,
    )
    cu_seqlens = None if sequence_lengths is None else _make_cu_seqlens(sequence_lengths)
    num_sequences = batch_size if sequence_lengths is None else len(sequence_lengths)
    initial_state = _expand_initial_state(initial_state, num_sequences)

    actual_output, actual_state, _ = run_indexed_prefill(
        optimized,
        gate_params,
        inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    expected_output, expected_state = run_fla_prefill(
        reference,
        gate_params,
        inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    assert_kda_close("state_pool/output", actual_output, expected_output)
    assert_kda_close("state_pool/state", actual_state, expected_state)


@pytest.mark.parametrize(
    "sequence_lengths,gate_scale,initial_state_scale",
    [
        ([(97 * (index + 3)) % 911 + 45 for index in range(24)], 1.0, None),
        ([1150, 1200, 980, 1100, 1279, 1024, 1216, 1090], 0.05, 1.0),
    ],
    ids=["packed_eval", "long_weak_gate_carried"],
)
@torch.no_grad()
def test_runtime_scale_state_pool_matches_fla(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
    sequence_lengths: list[int],
    gate_scale: float,
    initial_state_scale: float | None,
) -> None:
    """Cover packed evaluation traffic and long-lived recurrent state."""
    optimized, reference = dispatch_pair
    inputs, initial_state = _make_inputs(
        1,
        sum(sequence_lengths),
        seed=4242 + len(sequence_lengths),
        initial_state_scale=initial_state_scale,
    )
    q, k, v, g, beta = inputs
    inputs = q, k, v, (g.float() * gate_scale).to(g.dtype), beta
    initial_state = _expand_initial_state(initial_state, len(sequence_lengths))
    cu_seqlens = _make_cu_seqlens(sequence_lengths)

    actual_output, actual_state, _ = run_indexed_prefill(
        optimized,
        gate_params,
        inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    expected_output, expected_state = run_fla_prefill(
        reference,
        gate_params,
        inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    assert_kda_close("runtime_scale/output", actual_output, expected_output)
    assert_kda_close("runtime_scale/state", actual_state, expected_state)


def _num_chunks(sequence_lengths: list[int]) -> int:
    return sum((length + 63) // 64 for length in sequence_lengths)


@pytest.mark.parametrize(
    "victim_lengths,poison_lengths",
    [([8191, 1], [8127, 65]), ([8000, 150, 42], [7999, 129, 64])],
    ids=["autotuner_boundary", "mixed_partial_chunks"],
)
@torch.no_grad()
def test_final_state_ignores_poisoned_scratch(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
    victim_lengths: list[int],
    poison_lengths: list[int],
) -> None:
    """Partial final chunks cannot consume stale rows from a cached scratch buffer."""
    optimized, reference = dispatch_pair
    assert sum(victim_lengths) == sum(poison_lengths)
    assert _num_chunks(victim_lengths) == _num_chunks(poison_lengths)
    total_tokens = sum(victim_lengths)

    poison_inputs, _ = _make_inputs(1, total_tokens, seed=31)
    q, k, v, g, beta = poison_inputs
    poison_inputs = q * 100, k * 100, v * 1000, g, beta * 10
    run_indexed_prefill(
        optimized,
        gate_params,
        poison_inputs,
        _make_cu_seqlens(poison_lengths),
    )
    torch.cuda.synchronize()

    victim_inputs, initial_state = _make_inputs(
        1,
        total_tokens,
        seed=32,
        initial_state_scale=1.0,
    )
    initial_state = _expand_initial_state(initial_state, len(victim_lengths))
    cu_seqlens = _make_cu_seqlens(victim_lengths)
    actual_output, actual_state, _ = run_indexed_prefill(
        optimized,
        gate_params,
        victim_inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    expected_output, expected_state = run_fla_prefill(
        reference,
        gate_params,
        victim_inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    assert_kda_close("poisoned_scratch/output", actual_output, expected_output)
    assert_kda_close("poisoned_scratch/state", actual_state, expected_state)


def _run_head_count_case(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    num_heads: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(4711 + num_heads)
    gate_params = (
        torch.randn(num_heads, generator=generator, dtype=torch.float32, device="cuda") * 0.5,
        torch.randn(
            num_heads * HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1,
    )
    sequence_lengths = [1150, 731, 1024, 987]
    inputs, initial_state = _make_inputs(
        1,
        sum(sequence_lengths),
        seed=4711 + num_heads,
        num_heads=num_heads,
        initial_state_scale=1.0,
    )
    initial_state = _expand_initial_state(initial_state, len(sequence_lengths))
    cu_seqlens = _make_cu_seqlens(sequence_lengths)
    optimized, reference = dispatch_pair
    actual_output, actual_state, _ = run_indexed_prefill(
        optimized,
        gate_params,
        inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    expected_output, expected_state = run_fla_prefill(
        reference,
        gate_params,
        inputs,
        cu_seqlens,
        initial_state=initial_state,
    )
    assert_kda_close(f"heads_{num_heads}/output", actual_output, expected_output)
    assert_kda_close(f"heads_{num_heads}/state", actual_state, expected_state)


@pytest.mark.parametrize("num_heads", [6, 12], ids=["tp16", "tp8"])
@torch.no_grad()
def test_tensor_parallel_head_counts_match_fla(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    num_heads: int,
) -> None:
    _run_head_count_case(dispatch_pair, num_heads)


@torch.no_grad()
def test_head_count_compile_cache_isolation(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
) -> None:
    """Back-to-back head counts cannot reuse an incompatible compiled kernel."""
    _run_head_count_case(dispatch_pair, 8)
    _run_head_count_case(dispatch_pair, 4)


@torch.no_grad()
def test_chunked_continuation_matches_fla(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A state-pool row carried across chunks matches one full FLA prefill."""
    optimized, reference = dispatch_pair
    total_tokens = 3000
    split = 2048
    inputs, _ = _make_inputs(1, total_tokens, seed=777)
    full_output, full_state = run_fla_prefill(
        reference,
        gate_params,
        inputs,
        _make_cu_seqlens([total_tokens]),
    )

    first_inputs = tuple(tensor[:, :split] for tensor in inputs)
    second_inputs = tuple(tensor[:, split:] for tensor in inputs)
    first_output, _, state_pool = run_indexed_prefill(
        optimized,
        gate_params,
        first_inputs,
        _make_cu_seqlens([split]),
    )
    second_output, second_state, _ = run_indexed_prefill(
        optimized,
        gate_params,
        second_inputs,
        _make_cu_seqlens([total_tokens - split]),
        state_pool=state_pool,
    )

    assert_kda_close("chunked/first_output", first_output, full_output[:, :split])
    assert_kda_close("chunked/second_output", second_output, full_output[:, split:])
    assert_kda_close("chunked/state", second_state, full_state)
