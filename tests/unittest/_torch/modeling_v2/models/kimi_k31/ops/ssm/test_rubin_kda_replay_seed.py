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
"""Kimi K3.1 KDA replay state: seeding the replay caches of transferred requests in all layers.

``RubinKdaReplayState`` stacks each kind of per-layer view (conv states and the replay buffers a
transferred slot is seeded or cleared in) into one ``[layers, slots, ...]`` view and seeds every
local KDA layer with one copy per buffer kind and conv section. The views here are per-layer
views into slot pages that hold every layer's buffers, as the V2 manager binds them. The stacked
seeding must write what the shared per-layer loop writes, bit for bit, including slots and bytes
it must not touch, and it runs when requests join a decode batch, while the previous step is
still in flight, so it may not synchronize with the device.
"""

import contextlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_dsa_cache_manager as dsa_cache_manager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaReplayState,
)
from tensorrt_llm._torch.modules.kimi_kda import cache_manager as shared_cache_manager

# sm_107 only: l0_gb300 collects ops/ as a directory.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 7),
    reason="Kimi K3.1 Rubin: sm_107",
)

NUM_LAYERS = 4
NUM_SLOTS = 12
NUM_HEADS = 4
HEAD_DIM = 16
SECTION_DIM = NUM_HEADS * HEAD_DIM
COMMITTED = 3
NUM_SPEC = 3
EXTENDED = COMMITTED + NUM_SPEC
ALIGNED_HEADS = 8
# Elements between one layer's buffer and the next inside a slot page.
LAYER_PAD = 40
_CONV_ROLES = shared_cache_manager._KDA_REPLAY_CONV_ROLES
_DRAFT_ROLES = shared_cache_manager._KDA_REPLAY_DRAFT_ROLES
# One seed of the stacked views: the accepted-token counts, the seven replay buffer kinds, the
# conv rows of the seeded slots and the three conv sections.
_STACKED_KERNELS = 12
# The shared per-layer loop: the accepted-token counts, then eleven per layer.
_PER_LAYER_KERNELS = 1 + 11 * NUM_LAYERS


class _Log:
    """Stands in for the module's logger and keeps the once-per-key lines it is given."""

    def __init__(self):
        self.info: list[str] = []
        self.warning: list[str] = []

    def info_once(self, *msg, key):
        self.info.append(" ".join(map(str, msg)))

    def warning_once(self, *msg, key):
        self.warning.append(" ".join(map(str, msg)))


@pytest.fixture
def log(monkeypatch):
    log = _Log()
    monkeypatch.setattr(dsa_cache_manager, "logger", log)
    return log


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


def _make_state(conv_dtype=torch.bfloat16, draft_records=False, per_layer=False):
    """A K3.1 replay state on per-layer views of padded slot-page pools, and the pools it writes.

    With ``per_layer``, the state plans nothing and seeds with the shared per-layer loop.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    pools = []
    conv_pool, conv_states = _page_views((3 * SECTION_DIM, COMMITTED), conv_dtype, generator)
    pools.append(conv_pool)
    layers = [{} for _ in range(NUM_LAYERS)]
    for name in _CONV_ROLES:
        pool, views = _page_views((EXTENDED, SECTION_DIM), torch.bfloat16, generator)
        pools.append(pool)
        for layer, view in zip(layers, views):
            layer[name] = view.transpose(-1, -2)
    if draft_records:
        pool, views = _page_views((3, NUM_SPEC, NUM_HEADS, HEAD_DIM), torch.float32, generator)
        pools.append(pool)
        for layer, view in zip(layers, views):
            layer[shared_cache_manager._KDA_DRAFT_RECORDS_ROLE] = view
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

    state = RubinKdaReplayState(NUM_SPEC, draft_records=draft_records)
    state._conv_states = tuple(conv_states)
    state._conv_section_dims = (SECTION_DIM,) * 3
    state._committed_window = COMMITTED
    state.pool_layers = layers
    state.prev_num_accepted_tokens = torch.full((NUM_SLOTS,), 2, dtype=torch.int32, device="cuda")
    pools.append(state.prev_num_accepted_tokens)
    if per_layer:
        state._plan_batched_seed = lambda: None
    return state, pools


def _seed(state, slots):
    state.seed_transferred_slots(torch.tensor(slots, dtype=torch.long, device="cuda"))
    torch.cuda.synchronize()


def _assert_seeded(state, slots, draft_records):
    """The seeded slots hold each layer's committed conv window, and nothing else of their replay
    caches; their accepted-token counts are cleared."""
    assert state.prev_num_accepted_tokens[slots].eq(0).all()
    for layer, conv_state in enumerate(state._conv_states):
        fields = state._layer_cache_fields(layer)
        committed = conv_state[slots]
        for section, name in enumerate(_CONV_ROLES):
            buffer = fields[name][slots]
            window = committed[:, section * SECTION_DIM : (section + 1) * SECTION_DIM]
            assert torch.equal(buffer[..., :COMMITTED], window.to(buffer.dtype))
            assert not buffer[..., COMMITTED:].any()
        for name in _DRAFT_ROLES:
            buffer = fields.get(name)
            assert (buffer is None) == draft_records
            if buffer is not None:
                assert not buffer[slots].any()


@pytest.mark.parametrize("draft_records", [False, True], ids=["draft_caches", "draft_records"])
@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "slots", [[3], [0, 5, 11], list(range(0, NUM_SLOTS, 2))], ids=["one", "spread", "every_other"]
)
def test_stacked_seed_is_bit_identical_to_the_layer_loop(log, conv_dtype, draft_records, slots):
    """Every byte of every pool, in the seeded slots, the other slots and the padding between the
    layers' buffers, ends up as the shared per-layer loop leaves it."""
    stacked, stacked_pools = _make_state(conv_dtype, draft_records)
    looped, looped_pools = _make_state(conv_dtype, draft_records, per_layer=True)

    _seed(stacked, slots)
    _seed(looped, slots)

    plan = stacked._get_batched_seed_plan()
    assert plan is not None and plan[0] is dsa_cache_manager._STACKED_SEEDER
    assert looped._batched_seed_plan is False
    for got, want in zip(stacked_pools, looped_pools, strict=True):
        assert torch.equal(got, want)
    _assert_seeded(stacked, slots, draft_records)
    assert log.info == [
        f"KDA replay caches of transferred requests are seeded in all {NUM_LAYERS} local KDA "
        "layers at once"
    ]
    assert log.warning == []


def _give_layer_its_own_tensor(state, kind):
    """Moves one layer's view of ``kind`` into a tensor of its own, so that the layers' views of
    that kind are not evenly spaced; returns that tensor."""
    if kind == "conv_state":
        own = state._conv_states[1].contiguous()
        state._conv_states = (state._conv_states[0], own, *state._conv_states[2:])
    else:
        own = state.pool_layers[1][kind].clone()
        state.pool_layers[1][kind] = own
    return own


@pytest.mark.parametrize("kind", ["conv_state", "kda_conv_k", "kda_g_cache"])
def test_uneven_views_seed_layer_by_layer(log, kind):
    """When one kind's per-layer views do not stack, the state plans nothing and warns once that
    it seeds layer by layer; the per-layer loop then writes what it writes for even views."""
    state, pools = _make_state()
    reference, reference_pools = _make_state(per_layer=True)
    pools.append(_give_layer_its_own_tensor(state, kind))
    reference_pools.append(_give_layer_its_own_tensor(reference, kind))
    slots = [1, 7]

    assert state._get_batched_seed_plan() is None
    _seed(state, slots)
    _seed(reference, slots)

    assert state._batched_seed_plan is False
    assert len(log.warning) == 1
    assert log.warning[0].endswith("transferred requests are seeded layer by layer")
    assert log.info == []
    for got, want in zip(pools, reference_pools, strict=True):
        assert torch.equal(got, want)
    _assert_seeded(state, slots, draft_records=False)


def _cuda_kernels(seed) -> list[str]:
    """Names of the CUDA kernels one call of ``seed`` launches, after a warm-up call; copies and
    fills of memory are not counted."""
    seed()
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        seed()
        torch.cuda.synchronize()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.lower().startswith(("memcpy", "memset"))
    ]


def test_stacked_seed_launches_once_per_buffer_kind():
    """Twelve launches whatever the layer count, where the shared per-layer loop launches one plus
    eleven per layer. Both write the same bytes, so only the launch count tells a seeding that
    silently fell back to the loop."""
    slots = torch.tensor([0, 5, 11], dtype=torch.long, device="cuda")
    stacked, _ = _make_state()
    looped, _ = _make_state(per_layer=True)

    stacked_kernels = _cuda_kernels(lambda: stacked.seed_transferred_slots(slots))
    looped_kernels = _cuda_kernels(lambda: looped.seed_transferred_slots(slots))

    assert len(stacked_kernels) == _STACKED_KERNELS, stacked_kernels
    assert len(looped_kernels) == _PER_LAYER_KERNELS, looped_kernels


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_stacked_seed_does_not_wait_for_the_step_in_flight():
    """Neither planning on freshly bound views nor seeding, with the host slot IDs copied the way
    the manager copies them, synchronizes with the device."""
    host_slots = [0, 5, 11]
    # Launch the seeding's kernels once outside the check.
    _seed(_make_state()[0], [1])
    state, pools = _make_state()
    reference, reference_pools = _make_state(per_layer=True)

    with _no_host_sync():
        assert state._get_batched_seed_plan() is not None
        state.seed_transferred_slots(
            shared_cache_manager._host_slots_to_device(host_slots, torch.device("cuda"))
        )

    torch.cuda.synchronize()
    _seed(reference, host_slots)
    for got, want in zip(pools, reference_pools, strict=True):
        assert torch.equal(got, want)
