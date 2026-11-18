# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA checkpoint state and K3UllCacheManager on a small CPU pool fixture.

The fixture lays out the slots as the KV cache manager V2 coalesces them (all layers' SSM
states, then conv states, then checkpoint payloads, then the count) and hands out the same
strided views as ``_get_state_buffer``. It does not model the CUDA allocator or streams, so
the suspend / release commits that run on the manager's cache stream are not covered here.
On a GPU the pool can also hand out each layer's buffer as its own tensor over the shared
memory, as the V2 manager's ``_get_state_buffer`` does.
"""

import dataclasses
import math
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.cache import (
    k3_ull_cache_manager as cache,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayState,
    KimiK3HybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor._util import model_declared_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaRole,
    MambaStateLayout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

pytestmark = pytest.mark.cpu_only

HEADS, DIM, NUM_SPEC, WINDOW = 2, 8, 3, 3
SLOTS = 6
LAYERS = (1, 2)  # global ids of the local KDA layers
CHECKPOINT, COUNT = "kda_ckpt", "prev_num_accepted_tokens"
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _context(state_dtype: torch.dtype, slot_capacity: int | None = SLOTS) -> MambaStateLayout:
    return MambaStateLayout(
        layer_mask=(False, True, True, False),
        pp_layers=(0, 1, 2, 3),
        mamba_pp_layers=LAYERS,
        mapping=Mapping(),
        conv_state_shape=(3 * HEADS * DIM, WINDOW),
        ssm_state_shape=(HEADS, DIM, DIM),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=state_dtype,
        n_groups_per_rank=HEADS,
        max_batch_size=4,
        state_index_capacity=5,
        slot_capacity=slot_capacity,
        spec_config=NS(tokens_per_gen_step=1 + NUM_SPEC),
        conv_section_dims=(HEADS * DIM,) * 3,
        conv_state_layout="q_k_v",
    )


class Pool:
    """Coalesced slots with the V2 manager's per-buffer views.

    ``wrapped`` (CUDA): every buffer view is its own tensor over the pool memory, as the V2
    manager builds them, instead of a view of one storage. ``reversed_roles``: those roles lay
    the layers out in reverse order, so their per-layer views are not evenly spaced upwards.
    """

    def __init__(
        self,
        context: MambaStateLayout,
        nbytes: int,
        misalign: int = 0,
        device: str = "cpu",
        wrapped: bool = False,
        reversed_roles: tuple = (),
    ):
        self.context = context
        self.wrapped = wrapped
        ssm_bytes = math.prod(context.ssm_state_shape) * context.ssm_state_dtype.itemsize
        conv_bytes = math.prod(context.conv_state_shape) * context.conv_state_dtype.itemsize
        sizes = {
            MambaRole.SSM_STATE: ssm_bytes,
            MambaRole.CONV_STATE: conv_bytes,
            CHECKPOINT: nbytes,
        }
        self.offsets, offset = {}, 0
        for role in (MambaRole.SSM_STATE, MambaRole.CONV_STATE, CHECKPOINT):
            order = range(len(LAYERS))
            for local in reversed(order) if role in reversed_roles else order:
                self.offsets[local, str(role)] = offset
                offset += sizes[role]
        self.offsets[0, COUNT] = offset
        self.slot_bytes = offset + 256
        self.backing = torch.full(
            (SLOTS * self.slot_bytes + 512 + misalign,), 0xCD, dtype=torch.uint8, device=device
        )
        start = (-self.backing.data_ptr()) % 256 + misalign
        self.base = start
        self.slots = self.backing[start : start + SLOTS * self.slot_bytes].view(
            SLOTS, self.slot_bytes
        )
        self.slots[:, self.offsets[0, COUNT] :].zero_()

    def get_state_buffer(self, local_layer_idx, role, dtype, shape):
        offset = self.offsets[local_layer_idx, str(role)]
        typed = self.backing[self.base + offset :].view(dtype)
        inner = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
        if self.wrapped:
            typed = convert_to_torch_tensor(TensorWrapper(typed.data_ptr(), dtype, [typed.numel()]))
        return typed.as_strided([SLOTS, *shape], [self.slot_bytes // dtype.itemsize, *inner])

    def live_states(self):
        ssm = [
            self.get_state_buffer(
                i,
                MambaRole.SSM_STATE,
                self.context.ssm_state_dtype,
                list(self.context.ssm_state_shape),
            )
            for i in range(len(LAYERS))
        ]
        conv = [
            self.get_state_buffer(
                i, MambaRole.CONV_STATE, torch.bfloat16, list(self.context.conv_state_shape)
            )
            for i in range(len(LAYERS))
        ]
        return ssm, conv

    def pool_group_descs(self, roles=None):
        roles = roles or (str(MambaRole.SSM_STATE), str(MambaRole.CONV_STATE), CHECKPOINT)
        ids = [NS(layer_id=i, role=r) for i in range(len(LAYERS)) for r in roles]
        ids.append(NS(layer_id=0, role=COUNT))
        variant = NS(coalesced_buffers=[NS(buffer_ids=ids)])
        return [NS(slot_desc=NS(variants=[variant]))]


def _bound_state(state_dtype: torch.dtype, misalign: int = 0, **pool_kwargs):
    context = _context(state_dtype)
    state = cache.K3UllKdaCheckpointState(NUM_SPEC)
    state.validate(context)
    pool = Pool(context, state.layout(context).nbytes, misalign, **pool_kwargs)
    state.bind_replay_pool(
        [
            {
                role: pool.get_state_buffer(i, role, dtype, shape)
                for role, (dtype, shape) in state.replay_layout(context, layer).items()
            }
            for i, layer in enumerate(LAYERS)
        ]
    )
    ssm, conv = pool.live_states()
    generator = torch.Generator().manual_seed(1)
    for tensor in (*ssm, *conv):
        tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype))
    state.bind(context, ssm, conv)
    return context, state, pool, ssm, conv


def _manager(state, pool=None) -> cache.K3UllCacheManager:
    manager = object.__new__(cache.K3UllCacheManager)
    manager._kda_replay = state
    manager._speculative_state = state
    manager.mamba_layer_offsets = {layer: i for i, layer in enumerate(LAYERS)}
    manager.mamba_pp_layers = list(LAYERS)
    manager.layer_offsets = {layer: i for i, layer in enumerate(LAYERS)}
    manager.local_num_mamba_layers = len(LAYERS)
    manager._request_id_to_state_index = {}
    if pool is not None:
        manager.impl = NS(pool_group_descs=pool.pool_group_descs())
    return manager


def test_classes_keep_the_runtime_contracts() -> None:
    assert issubclass(cache.K3UllCacheManager, KimiK3HybridCacheManager)
    assert issubclass(cache.K3UllCacheManager, MambaHybridCacheManagerV2)
    assert cache.K3UllCacheManager.accepts_appended_draft_layers
    # The fused DSpark accept tail records the acceptance only for the base implementation.
    assert (
        cache.K3UllCacheManager.update_mamba_states is KimiK3HybridCacheManager.update_mamba_states
    )
    assert issubclass(cache.K3UllKdaCheckpointState, KDAReplayState)
    assert cache.K3UllKdaCheckpointState.record_acceptance is KDAReplayState.record_acceptance
    assert cache.K3UllKdaCheckpointState.update is KDAReplayState.update


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_pool_declarations(state_dtype: torch.dtype) -> None:
    context = _context(state_dtype)
    state = cache.K3UllKdaCheckpointState(NUM_SPEC)
    state.validate(context)
    layout = state.layout(context)
    assert layout.state_dtype == state_dtype and layout.conv_width == WINDOW + NUM_SPEC
    assert state.replay_shares_state_slot
    checkpoint = {DataRole(CHECKPOINT): (torch.uint8, [layout.nbytes])}
    assert state.replay_layout(context, LAYERS[0]) == {
        **checkpoint,
        DataRole(COUNT): (torch.int32, [64]),
    }
    assert state.replay_layout(context, LAYERS[1]) == checkpoint
    empty = dataclasses.replace(context, mamba_pp_layers=())
    assert state.replay_layout(empty, 0) == {}


def test_validate_rejects_unsupported_state() -> None:
    state = cache.K3UllKdaCheckpointState(NUM_SPEC)
    with pytest.raises(TypeError):
        state.validate(_context(torch.float16))
    with pytest.raises(ValueError):
        cache.K3UllKdaCheckpointState(NUM_SPEC + 1).validate(_context(torch.float32))
    wide = dataclasses.replace(_context(torch.float32), ssm_state_shape=(HEADS, DIM, 2 * DIM))
    with pytest.raises(ValueError):
        state.validate(wide)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("misalign", [0, 16])
def test_lifecycle(state_dtype: torch.dtype, misalign: int) -> None:
    context, state, pool, ssm, conv = _bound_state(state_dtype, misalign)
    assert len(state.checkpoint_views) == len(LAYERS)
    assert state.checkpoint_unit_bytes in (16, 32, 64, 128, 256)
    assert state.checkpoint_slot_units * state.checkpoint_unit_bytes == pool.slot_bytes
    assert state.prev_num_accepted_tokens.dtype == torch.int32
    assert state.prev_num_accepted_tokens.shape == (SLOTS,)

    # A transfer completes: checkpoint 0 and the conv window come from the live pools.
    transferred = torch.tensor([4, 1], dtype=torch.int64)
    state.prev_num_accepted_tokens[transferred] = 2
    state.seed_transferred_slots(transferred)
    assert state.prev_num_accepted_tokens[transferred].tolist() == [0, 0]
    for i in range(len(LAYERS)):
        layer = state.make_layer_cache(i, conv[i], ssm[i])
        assert isinstance(layer, cache.K3UllKdaLayerCache)
        assert layer.has_kda_checkpoints and not layer.has_kda_replay_caches
        assert layer.prev_num_accepted_tokens is state.prev_num_accepted_tokens
        conv_flat, state_flat = layer.kda_ckpt_kernel_caches
        assert state_flat.dtype == state_dtype and conv_flat.dtype == torch.bfloat16
        for s in transferred.tolist():
            assert torch.equal(layer.kda_ckpt[s, 0], ssm[i][s].to(state_dtype))
            assert torch.equal(layer.kda_vconv[s, :, :WINDOW], conv[i][s])

    # A verify of the batch [4, 1] writes all positions of both slots through the flat views
    # (values exact in BF16), and nothing else of the slots.
    ssm_live = [x.clone() for x in ssm]
    conv_live = [x.clone() for x in conv]
    batch = transferred.to(torch.int32)
    slots = torch.zeros(2, 1 + NUM_SPEC, dtype=torch.int32)
    accepted = torch.zeros(2, dtype=torch.int32)
    state.checkpoint_indices(batch, slots, accepted)
    assert accepted.tolist() == [1, 1]
    for i in range(len(LAYERS)):
        views = state.checkpoint_views[i]
        for b, s in enumerate(transferred.tolist()):
            for t in range(1 + NUM_SPEC):
                views.state_flat[slots[b, t]].fill_(50 * i + 10 * s + t)
            columns = torch.arange(WINDOW + NUM_SPEC, dtype=torch.float32)
            views.conv_flat[slots[b, 0]].copy_((columns + 100 * i + s).expand_as(views.vconv[s]))
    for i in range(len(LAYERS)):
        for s in transferred.tolist():
            for t in range(1 + NUM_SPEC):
                assert torch.all(state.checkpoint_views[i].ckpt[s, t] == 50 * i + 10 * s + t)
    for i in range(len(LAYERS)):
        assert torch.equal(ssm[i], ssm_live[i]) and torch.equal(conv[i], conv_live[i])
    assert state.prev_num_accepted_tokens[transferred].tolist() == [0, 0]

    # The speculative tail records 2 and 3 accepted drafts; a dummy row keeps its count.
    state.update(
        MambaAcceptanceBatch(
            attention_metadata=None,
            num_contexts=0,
            num_generations=3,
            num_accepted_tokens=torch.tensor([3, 4, 1], dtype=torch.int32),
            accepted_positions=torch.tensor([2, 3, 0], dtype=torch.int32),
            source_state_indices=torch.arange(3, dtype=torch.int32),
            destination_state_indices=torch.tensor([4, 1, 5], dtype=torch.int32),
            is_dummy_request=torch.tensor([False, False, True]),
        )
    )
    assert state.prev_num_accepted_tokens[[4, 1, 5]].tolist() == [2, 3, 0]
    slots2, accepted2 = state.checkpoint_indices(batch)
    assert accepted2.tolist() == [3, 4]
    assert torch.equal(slots2, slots)
    for i in range(len(LAYERS)):
        assert torch.equal(ssm[i], ssm_live[i]) and torch.equal(conv[i], conv_live[i])

    # Suspension of slot 1 commits checkpoint 3 and conv columns [3, 6) to its live state.
    state.commit_pending_drafts(1)
    assert state.prev_num_accepted_tokens[1] == 0
    assert state.prev_num_accepted_tokens[4] == 2
    for i in range(len(LAYERS)):
        assert torch.all(ssm[i][1] == float(50 * i + 10 + 3))
        expected = torch.arange(3, 3 + WINDOW, dtype=torch.float32) + 100 * i + 1
        assert torch.equal(conv[i][1].float(), expected.expand_as(conv[i][1]))
        assert torch.equal(ssm[i][4], ssm_live[i][4])

    # A layer that advanced its pools reseeds itself; another loads its checkpoint first.
    layer = state.make_layer_cache(0, conv[0], ssm[0])
    layer.load_checkpoints(torch.tensor([4]))
    assert torch.all(ssm[0][4] == float(10 * 4 + 2))
    layer.commit_conv_window(torch.tensor([4], dtype=torch.int32), conv[0])
    assert torch.equal(layer.kda_ckpt[4, 0], ssm[0][4].to(state_dtype))

    # A fresh slot (reset by the base class) reads checkpoint 0.
    state.reset_slots(torch.tensor([4]), [4])
    assert state.checkpoint_indices(torch.tensor([4]))[1].tolist() == [1]

    # The count buffer is the slot's last 256 bytes; the padding past the slots is untouched.
    assert torch.all(pool.backing[pool.base + SLOTS * pool.slot_bytes :] == 0xCD)
    state.shutdown()
    assert state.pool_layers is None and state.prev_num_accepted_tokens is None


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("device", "wrapped"),
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=_CUDA),
        pytest.param("cuda", True, marks=_CUDA),
    ],
)
def test_transfer_seed_runs_on_all_layers_at_once(state_dtype, device, wrapped) -> None:
    """On the V2 manager's coalesced slots every seeded kind stacks over the layers, and seeding
    transferred slots that way writes the same pool bytes as seeding layer by layer."""
    pools = []
    for stacked in (True, False):
        _, state, pool, _, _ = _bound_state(state_dtype, 16, device=device, wrapped=wrapped)
        assert state._stacked_seed is not None
        if not stacked:
            state._stacked_seed = None
        state.prev_num_accepted_tokens[[4, 1, 2]] = 2
        before = pool.slots.clone()
        state.seed_transferred_slots(torch.tensor([4, 1], dtype=torch.int32, device=device))
        assert state.prev_num_accepted_tokens[[4, 1, 2]].tolist() == [0, 0, 2]
        assert not torch.equal(pool.slots, before)
        pools.append(pool.slots.clone())
    assert torch.equal(pools[0], pools[1])


@pytest.mark.parametrize("reversed_roles", [(MambaRole.SSM_STATE,), (CHECKPOINT,)])
def test_transfer_seed_falls_back_to_the_layer_loop(reversed_roles) -> None:
    """A kind whose per-layer views are not evenly spaced upwards keeps the layer loop."""
    _, state, _, ssm, conv = _bound_state(torch.float32, reversed_roles=reversed_roles)
    assert state._stacked_seed is None
    state.seed_transferred_slots(torch.tensor([3, 0]))
    for i in range(len(LAYERS)):
        views = state.checkpoint_views[i]
        for s in (3, 0):
            assert torch.equal(views.ckpt[s, 0], ssm[i][s])
            assert torch.equal(views.vconv[s, :, :WINDOW], conv[i][s])


def test_bind_requires_checkpoint_views() -> None:
    context = _context(torch.float32)
    state = cache.K3UllKdaCheckpointState(NUM_SPEC)
    ssm = [torch.zeros(SLOTS, HEADS, DIM, DIM)]
    with pytest.raises(RuntimeError):
        state.bind(context, ssm, [torch.zeros(SLOTS, 3 * HEADS * DIM, WINDOW)])
    with pytest.raises(RuntimeError):
        state.checkpoint_indices(torch.tensor([0]))


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_manager_api(state_dtype: torch.dtype) -> None:
    context, state, pool, ssm, conv = _bound_state(state_dtype)
    manager = _manager(state, pool)
    assert manager.uses_kda_checkpoints
    for i, layer in enumerate(LAYERS):
        views = manager.kda_checkpoint_views(layer)
        state_flat, conv_flat, unit = views
        assert state_flat is state.checkpoint_views[i].state_flat
        assert conv_flat is state.checkpoint_views[i].conv_flat
        assert unit == state.checkpoint_unit_bytes
        caches = manager.kda_kernel_caches(layer)
        assert caches[0] is conv_flat and caches[1] is state_flat

    adapter = cache.K3UllKdaState(manager)
    assert adapter.caches(LAYERS[1])[1] is state.checkpoint_views[1].state_flat
    assert adapter.kda_checkpoint_views(LAYERS[0]).unit_bytes == state.checkpoint_unit_bytes
    generation_state_indices = torch.tensor([3, 0, 5, 2], dtype=torch.int32)
    state.prev_num_accepted_tokens[[3, 0, 5, 2]] = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    slots = torch.zeros(3, 1 + NUM_SPEC, dtype=torch.int32)
    accepted = torch.zeros(3, dtype=torch.int32)
    adapter.indices(generation_state_indices, slots, accepted)
    assert accepted.tolist() == [2, 1, 4]
    expected, _ = manager.checkpoint_indices(generation_state_indices[:3])
    assert torch.equal(slots, expected)
    assert "K3UllCacheManager" in adapter.describe()

    # Manager-level seed / commit run over every local layer.
    manager.commit_kda_checkpoints(torch.tensor([5]))
    for i in range(len(LAYERS)):
        assert torch.equal(ssm[i][5], state.checkpoint_views[i].ckpt[5, 3].to(ssm[i].dtype))
    assert state.prev_num_accepted_tokens[5] == 3
    manager.seed_kda_checkpoints(torch.tensor([5]))
    assert state.prev_num_accepted_tokens[5] == 0
    for i in range(len(LAYERS)):
        assert torch.equal(state.checkpoint_views[i].ckpt[5, 0], ssm[i][5].to(state_dtype))

    # A completed disaggregated transfer seeds the requests' slots (the base class's hook).
    for i in range(len(LAYERS)):
        ssm[i][2].fill_(7.0)
    state.prev_num_accepted_tokens[2] = 3
    manager._request_id_to_state_index = {11: 2}
    state.unseeded_requests.add(11)
    manager.on_state_transfer_complete([11, 12])
    assert 11 not in state.unseeded_requests
    assert state.prev_num_accepted_tokens[2] == 0
    for i in range(len(LAYERS)):
        assert torch.all(state.checkpoint_views[i].ckpt[2, 0] == 7.0)


def test_manager_binds_the_pool() -> None:
    context = _context(torch.bfloat16)
    state = cache.K3UllKdaCheckpointState(NUM_SPEC)
    state.validate(context)
    pool = Pool(context, state.layout(context).nbytes)
    manager = _manager(state, pool)
    manager._get_state_buffer = pool.get_state_buffer
    manager._state_layout = context
    manager._replay_layouts = {}
    manager._index_mapper_capacity = SLOTS
    manager.kv_cache_config = NS(max_util_for_resume=0.8)
    manager._host_state_indices = torch.empty(context.max_batch_size, dtype=torch.int32)
    manager._state_indices_buffer = NS(tensor=torch.empty_like(manager._host_state_indices))
    manager.get_region_group_ids = lambda _region: ()
    manager.all_ssm_states, manager.all_conv_states = pool.live_states()
    manager._region_providers()
    manager._setup_model_state()
    assert manager.uses_kda_checkpoints
    assert state._ssm_states[1] is manager.all_ssm_states[1]

    split = _manager(cache.K3UllKdaCheckpointState(NUM_SPEC))
    split.impl = NS(pool_group_descs=pool.pool_group_descs((str(MambaRole.SSM_STATE), CHECKPOINT)))
    with pytest.raises(RuntimeError):
        split._check_checkpoint_pool_group()


def test_manager_selects_its_state() -> None:
    manager = object.__new__(cache.K3UllCacheManager)
    manager.spec_config = NS(tokens_per_gen_step=1 + NUM_SPEC)
    manager.kda_context_only = False
    manager._requested_num_spec = None
    manager._state_layout = _context(torch.bfloat16, slot_capacity=None)
    state = manager._initialize_spec_state()
    assert isinstance(state, cache.K3UllKdaCheckpointState)
    assert manager._kda_replay is state and state.num_speculative_tokens == NUM_SPEC
    manager.kda_context_only = True
    assert manager._initialize_spec_state() is None
    manager.spec_config = None
    with pytest.raises(ValueError):
        manager._initialize_spec_state()
    with pytest.raises(ValueError):
        cache.K3UllCacheManager(kda_draft_records=True)
    with pytest.raises(ValueError):
        cache.K3UllCacheManager(mamba_ssm_stochastic_rounding=True)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_estimate_bytes(monkeypatch, state_dtype: torch.dtype) -> None:
    params = NS(
        num_heads=96,
        head_dim=128,
        conv_kernel=4,
        mamba_ssm_cache_dtype=state_dtype,
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(cache, "_local_kda_layers", lambda *args, **kwargs: (params, 69))
    mapping = Mapping(world_size=16, rank=3, tp_size=16)
    spec = NS(tokens_per_gen_step=8)
    nbytes = {torch.bfloat16: 1_619_200, torch.float32: 3_192_064}[state_dtype]
    expected = 69 * nbytes + 256
    model_config = NS()
    assert (
        cache.estimate_kda_checkpoint_bytes_per_slot(model_config, mapping, spec_config=spec)
        == expected
    )
    assert (
        cache.K3UllCacheManager._extra_state_bytes_per_rank(model_config, mapping, spec_config=spec)
        == expected
    )
    assert (
        cache.K3UllKdaCheckpointState.estimate_bytes_per_slot(
            model_config, mapping, spec_config=spec
        )
        == expected
    )
    assert cache.estimate_kda_checkpoint_bytes_per_slot(model_config, mapping) == 0
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    assert (
        cache.estimate_kda_checkpoint_bytes_per_slot(
            model_config, mapping, spec_config=spec, is_disagg=True
        )
        == 0
    )
    monkeypatch.setattr(cache, "_local_kda_layers", lambda *args, **kwargs: (params, 0))
    monkeypatch.delenv("TRTLLM_DISAGG_ROLE")
    assert (
        cache.estimate_kda_checkpoint_bytes_per_slot(model_config, mapping, spec_config=spec) == 0
    )


def test_engine_builds_the_declared_manager() -> None:
    """RuntimeHooks.kv_cache_manager_cls: the engine resolves KimiK3HybridCacheManager for this
    model and builds the declared subclass instead."""
    hooks = RuntimeHooks(kv_cache_manager_cls=cache.K3UllCacheManager)
    model = NS(runtime_hooks=hooks)
    assert (
        model_declared_kv_cache_manager_cls(model, KimiK3HybridCacheManager)
        is cache.K3UllCacheManager
    )
    assert (
        model_declared_kv_cache_manager_cls(model, MambaHybridCacheManagerV2)
        is cache.K3UllCacheManager
    )


def test_constructor_rejects_pool_rebalancing() -> None:
    config = NS(use_kv_cache_manager_v2=True, enable_kv_pool_rebalance=True)
    with pytest.raises(ValueError, match="enable_kv_pool_rebalance"):
        cache.K3UllCacheManager(kv_cache_config=config)
    with pytest.raises(ValueError, match="enable_kv_pool_rebalance"):
        cache.K3UllCacheManager(*([None] * 9), config)


@pytest.mark.parametrize("folded", [True, False])
def test_block_offset_copy_goes_to_the_open_commit(monkeypatch, folded: bool) -> None:
    calls = []

    def fold(*args):
        calls.append(("fold", args))
        return folded

    def base_copy(self, *args):
        calls.append(("base", args))

    monkeypatch.setattr(cache, "fold_block_offsets_copy", fold)
    monkeypatch.setattr(MambaHybridCacheManagerV2, "copy_batch_block_offsets", base_copy)
    manager = object.__new__(cache.K3UllCacheManager)
    copy_idx = torch.tensor([3, 5])
    manager.index_mapper = NS(get_copy_index=lambda ids, num_ctx, beam, cross: copy_idx)
    manager._use_per_layer_page_tables = False
    manager._page_table_materializer = NS(uses_device_expansion=False)
    manager.kv_cache_type = None
    manager.host_kv_cache_block_offsets = "host"
    manager.index_scales = "scales"
    manager.kv_offset = "offset"
    manager._stream = "stream"
    dst = torch.zeros(1)

    manager.copy_batch_block_offsets(dst, [7, 9], 1, 0, 2)

    assert calls[0] == ("fold", ("host", dst, copy_idx, "scales", "offset", "stream"))
    assert [c[0] for c in calls] == (["fold"] if folded else ["fold", "base"])

    # Beam search, per-layer page tables and device expansion keep the base class's copy.
    calls.clear()
    manager.copy_batch_block_offsets(dst, [7, 9], 2, 0, 4)
    manager._page_table_materializer = NS(uses_device_expansion=True)
    manager.copy_batch_block_offsets(dst, [7, 9], 1, 0, 2)
    assert [c[0] for c in calls] == ["base", "base"]
