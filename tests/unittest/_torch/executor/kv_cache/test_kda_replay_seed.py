# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Batched seeding of the KDA replay caches of transferred requests.

``KDAReplayState.seed_transferred_slots`` seeds every layer with the K3
package's ``seed_kda_replay_layers`` (TRTLLM_KDA_BATCHED_REPLAY_SEED). The
replay buffers here are per-layer views into slot pages that hold every
layer's buffers, as the V2 manager binds them, and the result must match the
per-layer loop bit for bit, including slots and bytes it must not touch.

Seeding and resetting slots runs when requests join a decode batch, while the
previous step is still in flight, so neither may synchronize with the device.
"""

import contextlib

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda import cache_manager
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KDAReplayState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS = 4
NUM_SLOTS = 12
NUM_HEADS = 4
HEAD_DIM = 16
SECTION_DIM = NUM_HEADS * HEAD_DIM
COMMITTED = 3
NUM_SPEC = 3
EXTENDED = COMMITTED + NUM_SPEC
ALIGNED_HEADS = 8
# Bytes between one layer's buffer and the next inside a slot page.
LAYER_PAD = 40


def _kernels():
    return pytest.importorskip(f"{cache_manager._K3_KDA_PACKAGE}._kda_kernels")


def _page_views(physical_shape, dtype, generator):
    """Per-layer ``[slots, *physical_shape]`` views into one padded slot-page pool."""
    elements = 1
    for size in physical_shape:
        elements *= size
    layer_stride = elements + LAYER_PAD
    pool = torch.randn(NUM_SLOTS, NUM_LAYERS * layer_stride, generator=generator, device="cuda").to(
        dtype
    )
    inner = []
    stride = 1
    for size in reversed(physical_shape):
        inner.insert(0, stride)
        stride *= size
    views = [
        pool.as_strided(
            (NUM_SLOTS, *physical_shape),
            (pool.stride(0), *inner),
            storage_offset=layer * layer_stride,
        )
        for layer in range(NUM_LAYERS)
    ]
    return pool, views


def _make_state(conv_dtype, draft_records):
    generator = torch.Generator(device="cuda").manual_seed(0)
    pools = []
    conv_pool, conv_states = _page_views((3 * SECTION_DIM, COMMITTED), conv_dtype, generator)
    pools.append(conv_pool)
    layers = [{} for _ in range(NUM_LAYERS)]
    for name in cache_manager._KDA_REPLAY_CONV_ROLES:
        pool, views = _page_views((EXTENDED, SECTION_DIM), torch.bfloat16, generator)
        pools.append(pool)
        for layer, view in zip(layers, views):
            layer[name] = view.transpose(-1, -2)
    if draft_records:
        pool, views = _page_views((3, NUM_SPEC, NUM_HEADS, HEAD_DIM), torch.float32, generator)
        pools.append(pool)
        for layer, view in zip(layers, views):
            layer[cache_manager._KDA_DRAFT_RECORDS_ROLE] = view
    else:
        for name, dtype in (
            ("kda_k_cache", torch.bfloat16),
            ("kda_g_cache", torch.float32),
            ("kda_v_cache", torch.bfloat16),
        ):
            pool, views = _page_views((NUM_SPEC, SECTION_DIM), dtype, generator)
            pools.append(pool)
            for layer, view in zip(layers, views):
                layer[name] = view
        pool, views = _page_views((NUM_SPEC, ALIGNED_HEADS), torch.bfloat16, generator)
        pools.append(pool)
        for layer, view in zip(layers, views):
            layer["kda_beta_cache"] = view[..., :NUM_HEADS]

    state = KDAReplayState(NUM_SPEC, draft_records=draft_records)
    state._conv_states = tuple(conv_states)
    state._conv_section_dims = (SECTION_DIM, SECTION_DIM, SECTION_DIM)
    state._committed_window = COMMITTED
    state.pool_layers = layers
    state.prev_num_accepted_tokens = torch.full((NUM_SLOTS,), 2, dtype=torch.int32, device="cuda")
    pools.append(state.prev_num_accepted_tokens)
    return state, pools


def _seed(monkeypatch, batched, conv_dtype, draft_records, slots):
    monkeypatch.setattr(cache_manager, "_KDA_BATCHED_REPLAY_SEED", batched)
    state, pools = _make_state(conv_dtype, draft_records)
    state.seed_transferred_slots(torch.tensor(slots, dtype=torch.long, device="cuda"))
    torch.cuda.synchronize()
    return state, pools


@pytest.mark.parametrize("draft_records", [False, True], ids=["draft_caches", "draft_records"])
@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("slots", [[3], [0, 5, 11], list(range(0, NUM_SLOTS, 2))])
def test_batched_seed_matches_per_layer_loop(monkeypatch, conv_dtype, draft_records, slots):
    _kernels()
    batched, batched_pools = _seed(monkeypatch, True, conv_dtype, draft_records, slots)
    assert batched._get_batched_seed_plan() is not None
    looped, looped_pools = _seed(monkeypatch, False, conv_dtype, draft_records, slots)
    assert looped._batched_seed_plan is False
    for got, want in zip(batched_pools, looped_pools):
        assert torch.equal(got, want)
    assert batched.prev_num_accepted_tokens[slots].eq(0).all()


def test_mismatched_layouts_fall_back_to_per_layer_loop(monkeypatch):
    kernels = _kernels()
    monkeypatch.setattr(cache_manager, "_KDA_BATCHED_REPLAY_SEED", True)
    state, pools = _make_state(torch.bfloat16, draft_records=False)
    reference, reference_pools = _make_state(torch.bfloat16, draft_records=False)
    # One layer's live conv state in its own contiguous tensor: same values, other strides.
    for target in (state, reference):
        target._conv_states = (target._conv_states[0].contiguous(), *target._conv_states[1:])
    layers = [state._layer_cache_fields(i) for i in range(NUM_LAYERS)]
    assert (
        kernels.plan_kda_replay_seed(
            state._conv_states,
            [
                tuple(layer[name] for name in cache_manager._KDA_REPLAY_CONV_ROLES)
                for layer in layers
            ],
            [],
        )
        is None
    )
    slots = torch.tensor([1, 7], dtype=torch.long, device="cuda")
    state.seed_transferred_slots(slots)
    assert state._batched_seed_plan is False
    monkeypatch.setattr(cache_manager, "_KDA_BATCHED_REPLAY_SEED", False)
    reference.seed_transferred_slots(slots)
    torch.cuda.synchronize()
    for got, want in zip(pools, reference_pools):
        assert torch.equal(got, want)


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("batched", [True, False], ids=["batched", "per_layer"])
def test_seed_transferred_slots_does_not_wait_for_the_step_in_flight(monkeypatch, batched):
    if batched:
        _kernels()
    slots = [0, 5, 11]
    monkeypatch.setattr(cache_manager, "_KDA_BATCHED_REPLAY_SEED", batched)
    # Compile the kernels on a throwaway state, and plan this state's seed, outside the check.
    _seed(monkeypatch, batched, torch.bfloat16, False, [1])
    state, pools = _make_state(torch.bfloat16, draft_records=False)
    assert (state._get_batched_seed_plan() is not None) == batched

    with _no_host_sync():
        state.seed_transferred_slots(
            cache_manager._host_slots_to_device(slots, torch.device("cuda"))
        )

    torch.cuda.synchronize()
    _, reference_pools = _seed(monkeypatch, False, torch.bfloat16, False, slots)
    for got, want in zip(pools, reference_pools):
        assert torch.equal(got, want)


def test_fresh_slot_reset_does_not_wait_for_the_step_in_flight():
    state = KDAReplayState(NUM_SPEC, stochastic_rounding=True)
    state._seed_rank_offset = 7
    state.prev_num_accepted_tokens = torch.full((NUM_SLOTS,), 2, dtype=torch.int32, device="cuda")
    state.rand_seed = torch.zeros(NUM_SLOTS, dtype=torch.int64, device="cuda")
    host_slots = [1, 4, 9]
    slots = cache_manager._host_slots_to_device(host_slots, torch.device("cuda"))

    with _no_host_sync():
        state.reset_slots(slots, host_slots)

    torch.cuda.synchronize()
    expected_prev = [0 if slot in host_slots else 2 for slot in range(NUM_SLOTS)]
    assert state.prev_num_accepted_tokens.tolist() == expected_prev
    expected_seeds = [
        cache_manager.compute_deterministic_mamba_seed(1, slot, 7) if slot in host_slots else 0
        for slot in range(NUM_SLOTS)
    ]
    assert state.rand_seed.tolist() == expected_seeds


def test_commit_pending_drafts_does_not_wait_for_the_step_in_flight(monkeypatch):
    from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels

    state, _ = _make_state(torch.bfloat16, draft_records=False)
    # Stand-ins: the stubbed commit below never reads the recurrent states.
    state._ssm_states = state._conv_states
    commits = []
    # Only the slot bookkeeping is under test; the commit kernel is stubbed out.
    monkeypatch.setattr(
        _kda_kernels,
        "commit_kda_pending_drafts",
        lambda *args, **kwargs: commits.append(args[9]),
    )

    with _no_host_sync():
        state.commit_pending_drafts(5)

    torch.cuda.synchronize()
    assert len(commits) == NUM_LAYERS
    assert all(slots.tolist() == [5] for slots in commits)
    assert state.prev_num_accepted_tokens.tolist() == [
        0 if slot == 5 else 2 for slot in range(NUM_SLOTS)
    ]
