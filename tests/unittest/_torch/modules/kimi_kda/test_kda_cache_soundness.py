# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-integration soundness tests for indexed optimized KDA prefill.

Covers the two classes of bug that ordinary op-math parity tests cannot see:

1. STREAM soundness. The executor runs the model on a dedicated non-blocking
   ``torch.cuda.Stream`` (``py_executor.execution_stream``); the CuTe DSL
   kernels must launch on torch's current stream, not the DSL default stream.
   A default-stream launch races with the projections producing q/k/v/g/beta
   and with consumers of the output and mutated state-pool rows — silent,
   intermittent corruption e2e (GSM8K 68.99 vs 97.01 on the FLA control)
   while single-stream unit tests all pass.
   ``test_indexed_prefill_uses_current_stream`` reproduces this
   deterministically by delaying the execution stream with ``_sleep`` so any
   kernel launched on the default stream reads not-yet-written inputs.

2. id()-keyed cache soundness under the runtime's fresh-tensor-per-call
   pattern. CPython recycles an object's address (= its id) as soon as it is
   freed; the ``_prune_on_gc`` weakref guard must pop the cache entry at
   dealloc, before a new tensor can alias the key. The recycled-ID test
   attempts to engineer exactly that aliasing: cache a verdict under id(t1),
   free t1, re-allocate until a new tensor lands on the same id, refill it
   with data for which the stale verdict would be WRONG, and parity-check
   against FLA. ``test_repeated_single_sequence_metadata_matches_fla``
   additionally covers the Phase 2.1 poisoning case (same cu_seqlens object,
   two calls).
"""

import gc

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


def _op_module():
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops

    return cute_dsl_kimi_k3_custom_ops


def _flush_tensor_cache_pins() -> None:
    from fla.ops.utils.index import prepare_chunk_indices as fla_prepare_chunk_indices
    from fla.ops.utils.index import prepare_chunk_offsets

    from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices

    for _ in range(5):
        dummy = torch.tensor([0, 64], dtype=torch.long, device="cuda")
        fla_prepare_chunk_indices(dummy, 64)
        prepare_chunk_offsets(dummy, 64)
        prepare_chunk_indices(dummy, 64)


def _release_cu_seqlens(module, holder: list[torch.Tensor]) -> tuple[int, bool]:
    old_id = id(holder[0])
    holder.clear()
    _flush_tensor_cache_pins()
    gc.collect()
    return old_id, old_id not in module._varlen_pure_cache


def _allocate_with_recycled_id(target_id: int, make, attempts: int = 512):
    retained = []
    for _ in range(attempts):
        candidate = make()
        if id(candidate) == target_id:
            return candidate
        retained.append(candidate)
    return None


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
def test_recycled_cu_seqlens_id_matches_fla(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
    aligned_lengths: list[int],
    unaligned_lengths: list[int],
) -> None:
    """A recycled tensor ID cannot reuse stale alignment or length metadata."""
    optimized, reference = dispatch_pair
    module = _op_module()
    cu_holder = [_make_cu_seqlens(aligned_lengths)]
    inputs = _make_inputs(sum(aligned_lengths), seed=11)
    run_indexed_prefill(optimized, gate_params, inputs, cu_holder[0])
    torch.cuda.synchronize()
    assert module._varlen_pure_cache.get(id(cu_holder[0])) is True

    old_id, pruned = _release_cu_seqlens(module, cu_holder)
    if not pruned:
        collision = _allocate_with_recycled_id(
            old_id,
            lambda: _make_cu_seqlens(unaligned_lengths),
            attempts=64,
        )
        assert collision is None, "a live cache entry allowed its tensor ID to be recycled"
        pytest.skip("cu_seqlens remains pinned, so its ID cannot be recycled")

    cu_seqlens = _allocate_with_recycled_id(
        old_id,
        lambda: _make_cu_seqlens(unaligned_lengths),
    )
    if cu_seqlens is None:
        pytest.skip("could not obtain a recycled cu_seqlens tensor ID")

    inputs = _make_inputs(sum(unaligned_lengths), seed=12)
    actual_output, actual_state, _ = run_indexed_prefill(
        optimized,
        gate_params,
        inputs,
        cu_seqlens,
    )
    expected_output, expected_state = run_fla_prefill(
        reference,
        gate_params,
        inputs,
        _make_cu_seqlens(unaligned_lengths),
    )
    assert_kda_close("recycled_id/output", actual_output, expected_output)
    assert_kda_close("recycled_id/state", actual_state, expected_state)
    assert module._varlen_pure_cache.get(id(cu_seqlens)) is False


@torch.no_grad()
def test_repeated_single_sequence_metadata_matches_fla(
    dispatch_pair: tuple[KDAKernelDispatch, KDAKernelDispatch],
    gate_params: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Repeated unaligned calls cannot poison cached single-sequence metadata."""
    optimized, reference = dispatch_pair
    cu_seqlens = _make_cu_seqlens([300])
    for iteration in range(2):
        inputs = _make_inputs(300, seed=40 + iteration)
        actual_output, actual_state, _ = run_indexed_prefill(
            optimized,
            gate_params,
            inputs,
            cu_seqlens,
        )
        expected_output, expected_state = run_fla_prefill(
            reference,
            gate_params,
            inputs,
            _make_cu_seqlens([300]),
        )
        assert_kda_close(f"iteration_{iteration}/output", actual_output, expected_output)
        assert_kda_close(f"iteration_{iteration}/state", actual_state, expected_state)
