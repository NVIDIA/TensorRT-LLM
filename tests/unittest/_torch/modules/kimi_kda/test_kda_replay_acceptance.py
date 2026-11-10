# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-launch KDA replay acceptance record matches the torch ops it replaces."""

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (
    can_record_kda_replay_acceptance,
    record_kda_replay_acceptance,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KDAReplayState
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaAcceptanceBatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_SLOTS = 700
POOL_COLUMNS = 4


def _reference(prev, state_indices, num_accepted, is_dummy_request, num_bonus_tokens):
    """The torch ops KDAReplayState used before the fused kernel."""
    slots = state_indices.to(torch.long)
    accepted = (num_accepted - num_bonus_tokens).to(torch.int32).clamp(min=0)
    if is_dummy_request is not None:
        accepted = torch.where(is_dummy_request, prev[slots], accepted)
    prev[slots] = accepted


def _pool():
    """A per-slot state pool whose column 0 is the replay count, as in the V2 manager."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    return torch.randint(
        -(2**31),
        2**31 - 1,
        (NUM_SLOTS, POOL_COLUMNS),
        dtype=torch.int32,
        device="cuda",
        generator=gen,
    )


def _inputs(batch, index_dtype, accepted_dtype, with_dummy, max_accepted=4, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    real = torch.randperm(NUM_SLOTS - 1, device="cuda", generator=gen)[:batch]
    num_accepted = torch.randint(-1, max_accepted + 1, (batch,), device="cuda", generator=gen).to(
        accepted_dtype
    )
    if batch:
        # Always cover "none accepted" (1) and "all accepted".
        num_accepted[0] = 1
        num_accepted[-1] = max_accepted
    is_dummy = None
    state_indices = real
    if with_dummy:
        is_dummy = torch.rand(batch, device="cuda", generator=gen) < 0.3
        # Padding dummies share the reserved last slot.
        state_indices = torch.where(is_dummy, NUM_SLOTS - 1, real)
    return state_indices.to(index_dtype), num_accepted, is_dummy


@pytest.mark.parametrize("batch", [0, 1, 3, 64, 600])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("accepted_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("with_dummy", [False, True])
@pytest.mark.parametrize("num_bonus_tokens", [0, 1])
def test_matches_torch_on_strided_column(
    batch, index_dtype, accepted_dtype, with_dummy, num_bonus_tokens
):
    state_indices, num_accepted, is_dummy = _inputs(batch, index_dtype, accepted_dtype, with_dummy)
    pool = _pool()
    expected = pool.clone()
    prev = pool[:, 0]
    assert prev.stride(0) == POOL_COLUMNS
    assert can_record_kda_replay_acceptance(prev, state_indices, num_accepted, is_dummy)

    record_kda_replay_acceptance(prev, state_indices, num_accepted, is_dummy, num_bonus_tokens)
    _reference(expected[:, 0], state_indices, num_accepted, is_dummy, num_bonus_tokens)

    # Column 0 matches bit for bit and the neighbouring columns are untouched.
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)


def test_contiguous_count_vector():
    state_indices, num_accepted, is_dummy = _inputs(32, torch.int32, torch.int32, True)
    prev = _pool()[:, 0].contiguous()
    expected = prev.clone()
    record_kda_replay_acceptance(prev, state_indices, num_accepted, is_dummy, 1)
    _reference(expected, state_indices, num_accepted, is_dummy, 1)
    torch.testing.assert_close(prev, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    ["cpu", "int64_counts", "strided_indices", "strided_accepted", "uint8_dummy", "short_dummy"],
)
def test_unsupported_layouts_are_rejected(case):
    state_indices, num_accepted, is_dummy = _inputs(8, torch.int32, torch.int32, True)
    prev = _pool()[:, 0]
    if case == "cpu":
        prev, state_indices, num_accepted, is_dummy = (
            t.cpu() for t in (prev, state_indices, num_accepted, is_dummy)
        )
    elif case == "int64_counts":
        prev = prev.to(torch.int64)
    elif case == "strided_indices":
        state_indices = torch.stack([state_indices, state_indices], dim=1)[:, 0]
    elif case == "strided_accepted":
        num_accepted = torch.stack([num_accepted, num_accepted], dim=1)[:, 0]
    elif case == "uint8_dummy":
        is_dummy = is_dummy.to(torch.uint8)
    elif case == "short_dummy":
        is_dummy = is_dummy[:-1]
    assert not can_record_kda_replay_acceptance(prev, state_indices, num_accepted, is_dummy)
    with pytest.raises(ValueError):
        record_kda_replay_acceptance(prev, state_indices, num_accepted, is_dummy, 1)


def _bound_state(prev):
    state = KDAReplayState(3)
    state.prev_num_accepted_tokens = prev
    return state


def _batch(state_indices, num_accepted, is_dummy):
    batch = num_accepted.numel()
    return MambaAcceptanceBatch(
        attention_metadata=None,
        num_contexts=0,
        num_generations=batch,
        num_accepted_tokens=num_accepted,
        accepted_positions=num_accepted - 1,
        source_state_indices=torch.arange(batch, device=num_accepted.device),
        destination_state_indices=state_indices,
        is_dummy_request=is_dummy,
    )


@pytest.mark.parametrize("strided_indices", [False, True])
def test_state_update_and_record_acceptance(monkeypatch, strided_indices):
    launches = []
    monkeypatch.setattr(
        _kda_kernels,
        "record_kda_replay_acceptance",
        lambda *args: launches.append(args) or record_kda_replay_acceptance(*args),
    )
    state_indices, num_accepted, is_dummy = _inputs(48, torch.int32, torch.int32, True)
    if strided_indices:
        # Takes the torch path, with the same result.
        state_indices = torch.stack([state_indices, state_indices], dim=1)[:, 0]

    pool = _pool()
    expected = pool.clone()
    _bound_state(pool[:, 0]).update(_batch(state_indices, num_accepted, is_dummy))
    _reference(expected[:, 0], state_indices, num_accepted, is_dummy, 1)
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)

    _bound_state(pool[:, 0]).record_acceptance(state_indices, num_accepted - 1, is_dummy)
    _reference(expected[:, 0], state_indices, num_accepted - 1, is_dummy, 0)
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)

    assert len(launches) == (0 if strided_indices else 2)


def test_cuda_graph_replay():
    batch = 16
    state_indices, num_accepted, is_dummy = _inputs(batch, torch.int32, torch.int32, True)
    pool = _pool()
    state = _bound_state(pool[:, 0])
    acceptance = _batch(state_indices, num_accepted, is_dummy)
    state.update(acceptance)  # Compile outside the capture.

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.update(acceptance)

    for seed in range(2, 5):
        new_indices, new_accepted, new_dummy = _inputs(
            batch, torch.int32, torch.int32, True, seed=seed
        )
        state_indices.copy_(new_indices)
        num_accepted.copy_(new_accepted)
        is_dummy.copy_(new_dummy)
        expected = pool.clone()
        _reference(expected[:, 0], state_indices, num_accepted, is_dummy, 1)
        graph.replay()
        torch.testing.assert_close(pool, expected, rtol=0, atol=0)
