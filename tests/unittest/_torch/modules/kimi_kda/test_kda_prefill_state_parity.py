# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexed FlashInfer state parity against the independent FLA path.

Cover V-first state layout, large carried states, partial packed sequences,
changing head counts, chunked continuation, and repeated eager/graph calls.
Inputs are raw Q/K/V; both recurrent kernels normalize Q/K internally.
"""

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
        "eqlen_partial_carried",
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


@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "cuda_graph"])
@torch.no_grad()
def test_prefill_is_repeatable(use_cuda_graph: bool) -> None:
    """Repeated prefill calls must agree in eager execution and graph replay."""
    generator = torch.Generator(device="cuda").manual_seed(20260921)
    length = 1186
    shape = (1, length, NUM_HEADS, HEAD_DIM)

    def random_tensor(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device="cuda", generator=generator)

    q = random_tensor(shape).bfloat16()
    k = random_tensor(shape).bfloat16()
    v = random_tensor(shape).bfloat16()
    g = random_tensor(shape).bfloat16()
    beta = random_tensor(shape[:-1])
    a_log = torch.zeros(NUM_HEADS, device="cuda")
    dt_bias = torch.zeros(NUM_HEADS * HEAD_DIM, device="cuda")
    state = torch.empty(1, NUM_HEADS, HEAD_DIM, HEAD_DIM, device="cuda")
    indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_seqlens = _make_cu_seqlens([length])
    dispatch = KDAKernelDispatch(use_optimized_prefill=True, use_optimized_decode=False)

    def run() -> torch.Tensor:
        state.zero_()
        output, _ = dispatch.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=a_log,
            dt_bias=dt_bias,
            scale=HEAD_DIM**-0.5,
            initial_state=None,
            safe_gate=True,
            lower_bound=-5.0,
            cu_seqlens=cu_seqlens,
            state_pool=state,
            state_indices=indices,
        )
        return output

    for _ in range(3):
        output = run()
    expected = output.clone()
    expected_state = state.clone()
    graph = None
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
    for _ in range(128):
        if graph is None:
            output = run()
        else:
            graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
