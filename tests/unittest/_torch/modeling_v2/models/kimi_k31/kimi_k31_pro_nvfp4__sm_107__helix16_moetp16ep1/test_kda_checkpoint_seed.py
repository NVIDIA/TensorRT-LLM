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
"""The served KDA replay state seeding transferred slots in checkpoint mode, in all layers at once.

A transferred slot's checkpoint 0 and the committed window of its conv page are seeded from the
live pools. ``KimiK31KDAReplayState`` stacks the per-layer views of the live pools, checkpoint 0
and the conv pages' windows into ``[layers, slots, ...]`` views and seeds every local KDA layer
with one gather and one scatter per pool kind. The views here sit in one slot page that holds
every layer's live state and checkpoint payload, as the V2 manager binds them. The stacked seed
must write what the per-layer ``seed_kda_checkpoints`` loop writes, bit for bit, including bytes
it must not touch (other slots, later checkpoints, the conv page past the window, headers and
padding); views that do not stack keep that loop; the replay mode's stacked seed is unchanged;
and the seed runs while the previous step is in flight, so it may not synchronize with the device.
"""

import contextlib
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_ull_kda_dsa_cache_manager as cache_manager,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_SLOTS = 12
CONV_WINDOW = 3
# Bytes between one layer's region of a slot page and the next.
LAYER_PAD = 512


def _align(nbytes: int, granule: int = 256) -> int:
    return -(-nbytes // granule) * granule


def _make_state(
    num_layers,
    num_spec,
    num_heads,
    head_dim,
    *,
    checkpoints=True,
    per_layer=False,
    uneven=None,
    seed=0,
):
    """A served replay state on per-layer views of one slot-page pool, and that pool.

    Each slot page holds, per layer, the live recurrent state, the live conv state and the
    ``kda_ckpt`` payload. With ``per_layer``, the state plans nothing and seeds layer by layer;
    ``uneven`` moves one layer's view of that kind (``ssm``, ``conv`` or ``ckpt``) into a tensor of
    its own, which is returned with the pool.
    """
    layout = cache_manager._kda_checkpoint_layout(
        num_spec, num_heads, head_dim, CONV_WINDOW, torch.float32, torch.bfloat16
    )
    ssm_bytes = layout.state_bytes
    conv_bytes = layout.conv_dim * CONV_WINDOW * 2
    ssm_offset, conv_offset = 0, _align(ssm_bytes)
    ckpt_offset = conv_offset + _align(conv_bytes)
    layer_bytes = ckpt_offset + layout.nbytes + LAYER_PAD
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pool = torch.randint(
        0,
        256,
        (NUM_SLOTS, num_layers * layer_bytes),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    ssm_states, conv_states, layers = [], [], []
    for layer in range(num_layers):
        base = layer * layer_bytes
        ssm = pool[:, base + ssm_offset : base + ssm_offset + ssm_bytes].view(torch.float32)
        ssm = ssm.unflatten(-1, layout.state_shape)
        ssm.copy_(torch.randn(ssm.shape, generator=generator, device="cuda"))
        conv = pool[:, base + conv_offset : base + conv_offset + conv_bytes].view(torch.bfloat16)
        conv = conv.unflatten(-1, (layout.conv_dim, CONV_WINDOW))
        conv.copy_(torch.randn(conv.shape, generator=generator, device="cuda"))
        raw = pool[:, base + ckpt_offset : base + ckpt_offset + layout.nbytes]
        views = cache_manager.kda_checkpoint_views(raw, layout)
        ssm_states.append(ssm)
        conv_states.append(conv)
        layers.append({"kda_ckpt": views.ckpt, "kda_vconv": views.vconv})
    own = None
    if uneven == "ssm":
        own = ssm_states[1] = ssm_states[1].clone()
    elif uneven == "conv":
        own = conv_states[1] = conv_states[1].clone()
    elif uneven == "ckpt":
        own = torch.randint(
            0,
            256,
            (NUM_SLOTS, layout.nbytes),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        views = cache_manager.kda_checkpoint_views(own, layout)
        layers[1] = {"kda_ckpt": views.ckpt, "kda_vconv": views.vconv}

    state = cache_manager.KimiK31KDAReplayState(num_spec, checkpoints=checkpoints)
    state._ssm_states = tuple(ssm_states)
    state._conv_states = tuple(conv_states)
    state._conv_section_dims = (num_heads * head_dim,) * 3
    state._committed_window = CONV_WINDOW
    state.pool_layers = layers
    state.prev_num_accepted_tokens = torch.full(
        (NUM_SLOTS,), num_spec, dtype=torch.int32, device="cuda"
    )
    if per_layer:
        state._plan_batched_seed = lambda: None
    pools = [pool, state.prev_num_accepted_tokens]
    if own is not None:
        pools.append(own)
    return state, pools


def _seed(state, slots):
    state.seed_transferred_slots(torch.tensor(slots, dtype=torch.long, device="cuda"))
    torch.cuda.synchronize()


def _assert_pools_equal(got, want):
    for got_pool, want_pool in zip(got, want, strict=True):
        assert torch.equal(got_pool, want_pool)


def _assert_seeded(state, slots):
    """The seeded slots' checkpoint 0 and conv windows hold the live pools; their accepted-token
    counts are cleared."""
    assert state.prev_num_accepted_tokens[slots].eq(0).all()
    for layer, (ssm, conv) in enumerate(zip(state._ssm_states, state._conv_states)):
        views = state.pool_layers[layer]
        assert torch.equal(views["kda_ckpt"][slots, 0], ssm[slots])
        assert torch.equal(views["kda_vconv"][slots, :, :CONV_WINDOW], conv[slots])


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
    monkeypatch.setattr(cache_manager, "logger", log)
    return log


_CONFIGS = [
    # (num_layers, num_spec, num_heads, head_dim)
    (1, 7, 1, 32),
    (3, 1, 2, 16),
    (5, 3, 1, 32),
    (8, 7, 2, 16),
]
_SLOT_SETS = {
    "one": [3],
    "first_and_last": [0, NUM_SLOTS - 1],
    "spread": [1, 5, 6, 10],
    "all": list(range(NUM_SLOTS)),
}


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda c: "L{}_spec{}_h{}_d{}".format(*c))
@pytest.mark.parametrize("slots", list(_SLOT_SETS.values()), ids=list(_SLOT_SETS))
def test_stacked_checkpoint_seed_is_bit_identical_to_the_layer_loop(log, config, slots):
    """Every byte of the slot pages, in the seeded slots, the other slots, the later checkpoints,
    the conv page past the window, the headers and the padding, ends up as the per-layer loop
    leaves it."""
    stacked, stacked_pools = _make_state(*config)
    looped, looped_pools = _make_state(*config, per_layer=True)
    _assert_pools_equal(stacked_pools, looped_pools)

    _seed(stacked, slots)
    _seed(looped, slots)

    plan = stacked._get_batched_seed_plan()
    assert plan is not None and plan[0] is cache_manager._STACKED_CHECKPOINT_SEEDER
    assert looped._batched_seed_plan is False
    _assert_pools_equal(stacked_pools, looped_pools)
    _assert_seeded(stacked, slots)
    assert log.info == [
        f"KDA checkpoints of transferred requests are seeded in all {config[0]} local KDA "
        "layers at once (one launch)"
    ]
    assert log.warning == []


@pytest.mark.parametrize("kind", ["ssm", "conv", "ckpt"])
def test_uneven_views_seed_layer_by_layer(log, kind):
    """When one kind's per-layer views do not stack, the state plans nothing and warns once that
    it seeds layer by layer; the per-layer loop then writes what it writes for even views."""
    config = (3, 3, 1, 32)
    state, pools = _make_state(*config, uneven=kind)
    reference, reference_pools = _make_state(*config, per_layer=True, uneven=kind)
    slots = [2, 7]

    assert state._get_batched_seed_plan() is None
    _seed(state, slots)
    _seed(reference, slots)

    assert state._batched_seed_plan is False
    assert len(log.warning) == 1
    assert log.warning[0].endswith("transferred requests are seeded layer by layer")
    assert log.info == []
    _assert_pools_equal(pools, reference_pools)
    _assert_seeded(state, slots)


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


@pytest.mark.parametrize("num_layers", [2, 8])
def test_stacked_seed_launches_a_fixed_number_of_kernels(num_layers):
    """Two launches whatever the layer count (the accepted-token counts, then the one-launch seed
    of every layer's checkpoint 0 and conv window); five on the torch form of the stacked seed (a
    gather and a scatter per pool kind), where the per-layer loop launches one plus four per
    layer. All write the same bytes, so only the launch count tells a seed that silently fell back
    to the loop."""
    slots = torch.tensor([0, 5, 11], dtype=torch.long, device="cuda")
    stacked, _ = _make_state(num_layers, 7, 1, 32)
    two_kernel, _ = _make_state(num_layers, 7, 1, 32)
    _torch_form(two_kernel)
    looped, _ = _make_state(num_layers, 7, 1, 32, per_layer=True)

    stacked_kernels = _cuda_kernels(lambda: stacked.seed_transferred_slots(slots))
    torch_kernels = _cuda_kernels(lambda: two_kernel.seed_transferred_slots(slots))
    looped_kernels = _cuda_kernels(lambda: looped.seed_transferred_slots(slots))

    assert len(stacked_kernels) == 2, stacked_kernels
    assert any("_seed_checkpoints" in name for name in stacked_kernels), stacked_kernels
    assert len(torch_kernels) == 5, torch_kernels
    assert len(looped_kernels) == 1 + 4 * num_layers, looped_kernels


def _torch_form(state) -> None:
    """Plan ``state``'s stacked seed, then keep its torch form (a gather and a scatter per kind)."""
    seeder, plan = state._get_batched_seed_plan()
    assert plan.one_launch
    state._batched_seed_plan = (seeder, dataclasses.replace(plan, one_launch=False))


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda c: "L{}_spec{}_h{}_d{}".format(*c))
def test_one_launch_seed_is_bit_identical_to_its_torch_form(config):
    """The one-launch seed and the torch form of the stacked seed leave every byte of the slot
    pages the same, including the slots, checkpoints, conv columns and padding they must not
    touch."""
    slots = [1, 5, 6, NUM_SLOTS - 1]
    one, one_pools = _make_state(*config)
    two, two_pools = _make_state(*config)
    _torch_form(two)
    _seed(one, slots)
    _seed(two, slots)
    assert one._get_batched_seed_plan()[1].one_launch
    _assert_pools_equal(one_pools, two_pools)
    _assert_seeded(one, slots)


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
    config = (4, 7, 1, 32)
    # Launch the seed's kernels once outside the check.
    _seed(_make_state(*config)[0], [1])
    state, pools = _make_state(*config)
    reference, reference_pools = _make_state(*config, per_layer=True)

    with _no_host_sync():
        assert state._get_batched_seed_plan() is not None
        state.seed_transferred_slots(
            cache_manager._host_slots_to_device(host_slots, torch.device("cuda"))
        )

    torch.cuda.synchronize()
    _seed(reference, host_slots)
    _assert_pools_equal(pools, reference_pools)


# ---- the served manager on real V2 pools -------------------------------------------------------
_KDA_MASK = [True, False, True, False]
_MLA_MASK = [False, True, False, True]
_KDA_HEADS = 8
_KDA_HEAD_DIM = 128
_NUM_SPEC = 7


def _text_config():
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=4,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3],
            "full_attn_layers": [2, 4],
        },
        dsa_config={"dsa_layers": [2, 4], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _make_manager(checkpoints: bool):
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekSparseAttentionConfig,
        KvCacheConfig,
        MTPDecodingConfig,
    )
    from tensorrt_llm.mapping import Mapping

    return cache_manager.KimiK3DSAHybridCacheManager(
        mamba_d_state=_KDA_HEAD_DIM,
        mamba_d_conv=CONV_WINDOW + 1,
        mamba_num_heads=_KDA_HEADS,
        mamba_n_groups=_KDA_HEADS,
        mamba_head_dim=_KDA_HEAD_DIM,
        mamba_num_layers=2,
        mamba_layer_mask=_KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=512, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        spec_config=MTPDecodingConfig(max_draft_len=_NUM_SPEC),
        kda_replay_num_spec=_NUM_SPEC,
        kda_checkpoints=checkpoints,
        layer_mask=_MLA_MASK,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp8"
        ),
        pretrained_config=_text_config(),
    )


def _served_views(replay) -> list[torch.Tensor]:
    views = [*replay._ssm_states, *replay._conv_states, replay.prev_num_accepted_tokens]
    for layer in replay.pool_layers:
        views += [view for view in layer.values() if isinstance(view, torch.Tensor)]
    return views


@pytest.mark.parametrize("checkpoints", [True, False], ids=["checkpoints", "replay"])
def test_served_manager_seeds_its_pools_in_one_stacked_pass(checkpoints):
    """The served manager's own V2 pools stack, and seeding transferred slots through them writes
    what the per-layer loop writes; without checkpoints the replay caches keep their own stacked
    seed."""
    manager = _make_manager(checkpoints)
    try:
        replay = manager._kda_replay
        assert type(replay) is cache_manager.KimiK31KDAReplayState
        assert replay.checkpoints == checkpoints
        plan = replay._get_batched_seed_plan()
        seeder = (
            cache_manager._STACKED_CHECKPOINT_SEEDER
            if checkpoints
            else cache_manager._STACKED_SEEDER
        )
        assert plan is not None and plan[0] is seeder

        generator = torch.Generator(device="cuda").manual_seed(5)
        views = _served_views(replay)
        for view in views:
            if view.is_floating_point():
                view.copy_(torch.randn(view.shape, generator=generator, device=view.device))
            else:
                view.fill_(3)
        before = [view.clone() for view in views]
        num_slots = replay._ssm_states[0].shape[0]
        slots = torch.tensor(
            sorted({0, num_slots // 2, num_slots - 1}), dtype=torch.long, device="cuda"
        )

        replay.seed_transferred_slots(slots)
        torch.cuda.synchronize()
        stacked = [view.clone() for view in views]

        for view, value in zip(views, before):
            view.copy_(value)
        replay._batched_seed_plan = False
        replay.seed_transferred_slots(slots)
        torch.cuda.synchronize()

        for got, want in zip(stacked, views, strict=True):
            assert torch.equal(got, want)
        if not checkpoints:
            return
        host_slots = slots.tolist()
        for layer, (ssm, conv) in enumerate(zip(replay._ssm_states, replay._conv_states)):
            ckpt = replay.pool_layers[layer]["kda_ckpt"]
            assert torch.equal(ckpt[host_slots, 0], ssm[host_slots])
            vconv = replay.pool_layers[layer]["kda_vconv"]
            assert torch.equal(vconv[host_slots, :, : conv.shape[-1]], conv[host_slots])
    finally:
        manager.shutdown()
