# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for model-owned hybrid managers and shared recurrent state."""

from dataclasses import replace
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules.fla.cache_manager import GDNReplayState, Qwen35HybridCacheManagerV2
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayLayerCache,
    KDAReplayState,
    KimiK3HybridCacheManagerV2,
)
from tensorrt_llm._torch.modules.mamba.cache_manager import (
    Mamba2State,
    NemotronHybridCacheManagerV2,
)
from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import (
    PLE_CONV_STATE,
    PLE_NGRAM_CONTEXT,
    Qwen4ExpHybridCacheManagerV2,
    Qwen4ExpPLECacheParams,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    IntermediateLayerCache,
    IntermediateState,
    MambaAcceptanceBatch,
    MambaHybridCacheManager,
    MambaRole,
    MambaStateLayout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.replay import (
    ReplayHistory,
    ReplayLayerCache,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    KVCacheManagerConfig,
    LayerId,
)


@pytest.mark.parametrize(
    "manager_cls",
    [
        NemotronHybridCacheManagerV2,
        Qwen35HybridCacheManagerV2,
        Qwen4ExpHybridCacheManagerV2,
        KimiK3HybridCacheManagerV2,
    ],
)
def test_model_managers_inherit_the_common_lifecycle(manager_cls):
    assert manager_cls.__bases__ == (MambaHybridCacheManagerV2,)
    for method in (
        "try_commit_blocks",
        "prepare_resources",
        "_setup_state_indices",
        "free_resources",
    ):
        assert getattr(manager_cls, method) is getattr(MambaHybridCacheManagerV2, method)


@pytest.mark.parametrize(
    "manager_cls",
    [
        MambaHybridCacheManagerV2,
        NemotronHybridCacheManagerV2,
        Qwen35HybridCacheManagerV2,
        Qwen4ExpHybridCacheManagerV2,
        KimiK3HybridCacheManagerV2,
    ],
)
def test_v2_managers_do_not_expose_legacy_accessors(manager_cls):
    for name in (
        "get_intermediate_ssm_states",
        "get_intermediate_conv_states",
        "get_mamba_ssm_cache_dtype",
        "use_replay_state_update",
        "_init_speculative_state",
        "_validate_model_layout",
    ):
        assert not hasattr(manager_cls, name)


def test_legacy_managers_keep_compatibility_accessors():
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        CppMambaHybridCacheManager,
        MambaCacheManager,
        MixedMambaHybridCacheManager,
        PythonMambaCacheManager,
    )

    for manager_cls in (
        PythonMambaCacheManager,
        MambaCacheManager,
        MixedMambaHybridCacheManager,
        CppMambaHybridCacheManager,
    ):
        for name in (
            "get_intermediate_ssm_states",
            "get_intermediate_conv_states",
            "get_mamba_ssm_cache_dtype",
            "use_replay_state_update",
        ):
            assert hasattr(manager_cls, name)


@pytest.mark.parametrize(
    "cls",
    [
        MambaHybridCacheManager,
        MambaHybridCacheManagerV2,
        NemotronHybridCacheManagerV2,
        Qwen35HybridCacheManagerV2,
        Qwen4ExpHybridCacheManagerV2,
        KimiK3HybridCacheManagerV2,
        Mamba2State,
        ReplayHistory,
        GDNReplayState,
    ],
)
def test_inherited_method_overrides_are_annotated(cls):
    for name, attribute in vars(cls).items():
        if isinstance(attribute, (classmethod, staticmethod)):
            function = attribute.__func__
        elif isinstance(attribute, property):
            function = attribute.fget
        else:
            function = attribute
        if not isinstance(function, FunctionType):
            continue
        # A standalone concrete class does not implement a project override
        # contract merely because object defines __init__.
        inherited = any(name in vars(parent) for parent in cls.__mro__[1:] if parent is not object)
        assert getattr(function, "__override__", False) == inherited, f"{cls.__name__}.{name}"


def _layout(layers=(0, 1), slot_capacity=5):
    return MambaStateLayout(
        layer_mask=(True, True),
        pp_layers=layers,
        mamba_pp_layers=layers,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        conv_state_shape=(24, 3),
        ssm_state_shape=(2, 4, 4),
        conv_state_dtype=torch.float32,
        ssm_state_dtype=torch.float32,
        n_groups_per_rank=2,
        max_batch_size=3,
        state_index_capacity=4,
        slot_capacity=slot_capacity,
        spec_config=SimpleNamespace(tokens_per_gen_step=3),
        conv_section_dims=(8, 8, 8),
        conv_state_layout="q_k_v",
    )


def _state_views(layers=2, slots=5):
    states = torch.zeros(layers, slots, 2, 4, 4)
    return SimpleNamespace(
        all_ssm_states=list(states.unbind()),
        all_conv_states=[torch.zeros(slots, 24, 3) for _ in range(layers)],
        ssm_state_shape=[2, 4, 4],
    )


def _prepare_model_layout(manager):
    """Supply constructor-resolved inputs when testing the pre-pool hook alone."""
    manager._state_layout = replace(_layout(), spec_config=manager.spec_config)
    manager.local_num_mamba_layers = 2
    manager._global_n_groups = 2
    if isinstance(manager, Qwen4ExpHybridCacheManagerV2):
        manager._ple_params = None
        manager._is_ple_draft = False
        manager._pretrained_config = None
        manager._mamba_layer_mask = [True, True]
        manager.conv_state_dtype = torch.float32


def test_qwen_managers_share_gdn_algorithm_not_model_inheritance():
    states = []
    for cls in (Qwen35HybridCacheManagerV2, Qwen4ExpHybridCacheManagerV2):
        manager = object.__new__(cls)
        manager._requested_replay = True
        manager.spec_config = SimpleNamespace(tokens_per_gen_step=3)
        _prepare_model_layout(manager)
        manager._speculative_state = manager._initialize_model_state()
        states.append(manager._speculative_state)
    assert all(type(state) is GDNReplayState for state in states)
    assert states[0] is not states[1]


@pytest.mark.parametrize(
    "manager_cls",
    [
        NemotronHybridCacheManagerV2,
        Qwen35HybridCacheManagerV2,
        Qwen4ExpHybridCacheManagerV2,
        KimiK3HybridCacheManagerV2,
    ],
)
def test_model_initialization_validates_selected_replay(manager_cls):
    manager = object.__new__(manager_cls)
    manager.spec_config = SimpleNamespace(tokens_per_gen_step=3)
    manager._requested_replay = True
    manager._requested_num_spec = 2
    manager._kda_replay = None
    _prepare_model_layout(manager)
    manager._state_layout = replace(
        manager._state_layout, n_groups_per_rank=0, conv_state_layout="x_b_c"
    )
    with pytest.raises(ValueError, match="requires"):
        manager._initialize_model_state()


@pytest.mark.parametrize(
    "requested_num_spec, selected_num_spec, decoding_type",
    [
        (2, None, "MTP"),
        (None, 2, "MTP"),
        (None, 2, "NGram"),
        (None, None, "MTP"),
        (None, None, None),
    ],
)
def test_k3_initializes_only_the_selected_state(
    monkeypatch, requested_num_spec, selected_num_spec, decoding_type
):
    manager = object.__new__(KimiK3HybridCacheManagerV2)
    manager._requested_num_spec = requested_num_spec
    manager._kda_replay = None
    manager.spec_config = (
        SimpleNamespace(decoding_type=decoding_type, tokens_per_gen_step=3)
        if decoding_type is not None
        else None
    )
    _prepare_model_layout(manager)
    intermediate_factory = Mock(wraps=IntermediateState)
    replay_factory = Mock(wraps=KDAReplayState)
    selector = Mock(return_value=selected_num_spec)
    monkeypatch.setitem(
        KimiK3HybridCacheManagerV2._initialize_model_state.__globals__,
        "IntermediateState",
        intermediate_factory,
    )
    monkeypatch.setitem(
        KimiK3HybridCacheManagerV2._initialize_model_state.__globals__,
        "KDAReplayState",
        replay_factory,
    )
    monkeypatch.setitem(
        KimiK3HybridCacheManagerV2._initialize_model_state.__globals__,
        "get_kda_replay_num_spec",
        selector,
    )

    manager._speculative_state = manager._initialize_model_state()
    num_spec = requested_num_spec if requested_num_spec is not None else selected_num_spec
    if num_spec is not None:
        intermediate_factory.assert_not_called()
        replay_factory.assert_called_once_with(num_spec)
        assert manager._speculative_state is manager._kda_replay
        state = manager._kda_replay
    else:
        replay_factory.assert_not_called()
        intermediate_factory.assert_called_once_with()
        assert manager._kda_replay is None
        state = manager._speculative_state
    if requested_num_spec is None:
        selector.assert_called_once_with(manager.spec_config, manager_supports_replay=True)
    else:
        selector.assert_not_called()

    manager._state_layout = object()
    manager.all_ssm_states, manager.all_conv_states = [], []
    bind, update, reset = Mock(), Mock(), Mock()
    monkeypatch.setattr(state, "bind", bind)
    monkeypatch.setattr(state, "update", update)
    monkeypatch.setattr(state, "reset_slots", reset)
    manager._setup_model_state()
    bind.assert_called_once_with(
        manager._state_layout, manager.all_ssm_states, manager.all_conv_states
    )
    batch, slots, host_slots = object(), object(), [0]
    manager._update_speculative_state(batch)
    update.assert_called_once_with(batch)
    manager._reset_model_slots(slots, host_slots)
    reset.assert_called_once_with(slots, host_slots)

    shutdown = Mock(wraps=state.shutdown)
    monkeypatch.setattr(state, "shutdown", shutdown)
    manager._shutdown_model_state()
    shutdown.assert_called_once_with()
    manager._shutdown_model_state()
    assert shutdown.call_count == 2
    if num_spec is not None:
        assert manager._speculative_state is manager._kda_replay


def test_base_manager_without_model_state():
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.spec_config = None
    manager._speculative_state = manager._initialize_model_state()
    assert manager._speculative_state is None
    manager._setup_model_state()
    manager._reset_model_slots(object(), [0])
    manager._shutdown_model_state()
    assert manager.intermediate_state_indices is None
    assert manager.intermediate_ssm_states is None
    assert manager.intermediate_conv_states is None
    assert manager.get_replay_state_update_metadata() is None


@pytest.mark.parametrize(
    "manager_cls",
    [
        MambaHybridCacheManager,
        MambaHybridCacheManagerV2,
        Qwen35HybridCacheManagerV2,
        Qwen4ExpHybridCacheManagerV2,
        KimiK3HybridCacheManagerV2,
    ],
)
def test_seed_accessor_is_not_a_common_manager_capability(manager_cls):
    assert not hasattr(manager_cls, "get_mamba_ssm_rand_seed")


def test_mamba2_seed_lifecycle_without_speculative_decoding():
    manager = object.__new__(NemotronHybridCacheManagerV2)
    manager.spec_config = None
    manager._requested_replay = False
    manager._state_layout = replace(_layout(), spec_config=None, stochastic_rounding=True)
    views = _state_views()
    manager.all_ssm_states = views.all_ssm_states
    manager.all_conv_states = views.all_conv_states
    manager._speculative_state = manager._initialize_model_state()
    manager._setup_model_state()
    seeds = manager.get_mamba_ssm_rand_seed()
    before = seeds.clone()
    manager._reset_model_slots(torch.tensor([1]), [1])
    assert manager.get_mamba_ssm_rand_seed() is seeds
    assert seeds[1] != before[1]
    torch.testing.assert_close(seeds[[0, 2, 3, 4]], before[[0, 2, 3, 4]])
    assert manager.intermediate_ssm_states is None
    manager._shutdown_model_state()
    assert manager.get_mamba_ssm_rand_seed() is None


@pytest.mark.parametrize(
    "state", [IntermediateState(), ReplayHistory(3), GDNReplayState(3), KDAReplayState(2)]
)
def test_common_manager_consumes_model_state_contract(state):
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.spec_config = SimpleNamespace(tokens_per_gen_step=3)
    manager._speculative_state = state
    manager._state_layout = _layout()
    views = _state_views()
    manager.all_ssm_states = views.all_ssm_states
    manager.all_conv_states = views.all_conv_states
    manager.mamba_layer_offsets = {42: 0}
    manager._setup_model_state()
    assert manager.intermediate_state_indices is state.intermediate_indices
    assert manager.intermediate_ssm_states is state.intermediate_ssm
    assert manager.intermediate_conv_states is state.intermediate_conv
    payload = manager.mamba_layer_cache(42)
    assert payload.conv is manager.all_conv_states[0]
    assert payload.temporal is manager.all_ssm_states[0]
    if isinstance(state, KDAReplayState):
        assert payload.kda_qkg_cache is not None
    elif isinstance(state, ReplayHistory):
        assert payload.old_x is not None
    else:
        assert payload.intermediate_ssm is not None
    assert (manager.get_replay_state_update_metadata() is not None) == isinstance(
        state, ReplayHistory
    )
    manager._shutdown_model_state()
    assert manager.intermediate_state_indices is None
    assert manager.intermediate_ssm_states is None
    assert manager.intermediate_conv_states is None


def test_base_manager_requires_model_for_speculative_decoding():
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.spec_config = SimpleNamespace(tokens_per_gen_step=3)
    with pytest.raises(NotImplementedError, match="model-specific"):
        manager._initialize_model_state()
    manager._speculative_state = None
    with pytest.raises(NotImplementedError, match="model manager"):
        manager._update_speculative_state(object())


@pytest.mark.parametrize(
    "manager_cls",
    [NemotronHybridCacheManagerV2, Qwen35HybridCacheManagerV2, Qwen4ExpHybridCacheManagerV2],
)
@pytest.mark.parametrize("mode", ["plain", "intermediate", "replay"])
def test_model_owns_speculative_state_lifecycle(monkeypatch, manager_cls, mode):
    manager = object.__new__(manager_cls)
    manager.spec_config = SimpleNamespace(tokens_per_gen_step=3) if mode != "plain" else None
    manager._requested_replay = mode == "replay"
    _prepare_model_layout(manager)
    manager._speculative_state = manager._initialize_model_state()
    state = manager._speculative_state
    assert isinstance(state, IntermediateState) == (mode != "replay")
    assert isinstance(state, ReplayHistory) == (mode == "replay")

    manager._state_layout = object()
    manager.local_num_mamba_layers = 0
    if manager_cls is Qwen4ExpHybridCacheManagerV2:
        manager._ple_conv_states = {}
        manager._ple_ngram_contexts = {}
    manager.mamba_layer_offsets = {42: 0}
    conv, ssm, payload = object(), object(), object()
    manager.all_conv_states, manager.all_ssm_states = [conv], [ssm]
    bind, reset, update, layer_cache = Mock(), Mock(), Mock(), Mock(return_value=payload)
    monkeypatch.setattr(state, "bind", bind)
    monkeypatch.setattr(state, "reset_slots", reset)
    monkeypatch.setattr(state, "update", update)
    monkeypatch.setattr(state, "make_layer_cache", layer_cache)
    manager._setup_model_state()
    bind.assert_called_once_with(
        manager._state_layout, manager.all_ssm_states, manager.all_conv_states
    )
    cache = manager.mamba_layer_cache(42)
    if mode == "plain":
        assert cache.conv is conv and cache.temporal is ssm
        layer_cache.assert_not_called()
    else:
        assert cache is payload
        layer_cache.assert_called_once_with(0, conv, ssm)

    if mode != "replay":
        state.intermediate_ssm = [ssm] if mode == "intermediate" else None
    state.intermediate_conv = [conv] if mode != "plain" else None
    assert manager.intermediate_ssm_states == ([ssm] if mode == "intermediate" else None)
    assert manager.intermediate_conv_states == ([conv] if mode != "plain" else None)
    batch, slots, host_slots = object(), object(), [0]
    manager._reset_model_slots(slots, host_slots)
    reset.assert_called_once_with(slots, host_slots)
    manager._update_speculative_state(batch)
    update.assert_called_once_with(batch)

    shutdown = Mock(wraps=state.shutdown)
    monkeypatch.setattr(state, "shutdown", shutdown)
    manager._shutdown_model_state()
    shutdown.assert_called_once_with()
    assert manager.intermediate_ssm_states is None and state.intermediate_conv is None
    if mode == "replay":
        assert state.intermediate_ssm is None


def test_ple_adds_buffers_without_overriding_snapshot_or_budget_solver():
    manager = object.__new__(Qwen4ExpHybridCacheManagerV2)
    manager._mamba_layer_mask = [True, True]
    manager.mamba_pp_layers = [0, 1]
    manager.ssm_bytes, manager.conv_bytes = 128, 288
    params = Qwen4ExpPLECacheParams(
        ple_layer_mask=[False, True],
        num_ple_layers=1,
        short_conv_channels=16,
        short_conv_state_len=6,
        ngram_context_len=2,
        conv_state_dtype=torch.bfloat16,
    )
    manager._init_qwen4_exp_ple_geometry(params, 2, torch.bfloat16)
    core = manager._get_recurrent_buffer_configs(0)
    ple = manager._get_recurrent_buffer_configs(1)
    assert [buffer.role for buffer in core] == [MambaRole.SSM_STATE, MambaRole.CONV_STATE]
    assert [buffer.role for buffer in ple[2:]] == [PLE_NGRAM_CONTEXT, PLE_CONV_STATE]
    assert manager._mamba_state_bytes_per_slot() == 2 * (128 + 288) + 16 * 6 * 2 + 2 * 8


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("page_index_scale", [1, 2])
def test_warmup_cleanup_clears_registered_persistent_roles_only(
    monkeypatch, replay, page_index_scale
):
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        mamba_cache_manager_v2 as module,
    )

    manager = object.__new__(Qwen4ExpHybridCacheManagerV2)
    manager._mamba_layer_mask = [False, True]
    manager.pp_layers = manager.mamba_pp_layers = [1]
    manager.layer_offsets = manager.mamba_layer_offsets = {1: 0}
    manager.local_num_mamba_layers = 1
    manager._state_layout = _layout(layers=(1,))
    manager.max_batch_size = manager._state_layout.max_batch_size
    manager.mapping = manager._state_layout.mapping
    manager.ssm_state_shape = list(manager._state_layout.ssm_state_shape)
    manager.conv_state_shape = list(manager._state_layout.conv_state_shape)
    manager.ssm_state_dtype = manager.conv_state_dtype = torch.float32
    manager.ssm_bytes, manager.conv_bytes = 128, 288
    manager._num_reserved_dummy_slots = 0
    manager._guard_page_by_layer = {}
    manager.kv_cache_config = SimpleNamespace()
    manager._minimum_live_gpu_quota = lambda: 0
    manager._init_qwen4_exp_ple_geometry(
        Qwen4ExpPLECacheParams(
            ple_layer_mask=[False, True],
            num_ple_layers=1,
            short_conv_channels=16,
            short_conv_state_len=6,
            ngram_context_len=2,
            conv_state_dtype=torch.bfloat16,
        ),
        2,
        torch.float32,
    )
    manager._build_cache_config(
        KVCacheManagerConfig(
            tokens_per_block=32,
            cache_tiers=[GpuCacheTierConfig(quota=1 << 20)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0), buffers=[BufferConfig(role=Role.KEY, size=256)]
                )
            ],
            initial_pool_ratio=[1.0],
        )
    )
    raw_buffers = {}

    def wrap_buffer(role, dtype, shape):
        raw_buffers[role] = torch.full(shape, 7, dtype=dtype)
        return raw_buffers[role]

    manager.impl = SimpleNamespace(
        get_mem_pool_base_address=lambda layer, role, mode: role,
        get_page_index_upper_bound=lambda layer, role: 5 * page_index_scale,
        get_page_index_scale=lambda layer, role: page_index_scale,
    )
    monkeypatch.setattr(module, "TensorWrapper", wrap_buffer)
    monkeypatch.setattr(module, "convert_to_torch_tensor", lambda tensor: tensor)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    manager._setup_states()
    manager._speculative_state = GDNReplayState(3) if replay else IntermediateState()
    manager._setup_model_state()
    state = manager._speculative_state
    scratch_before = {}
    for name, tensor in vars(state).items():
        if isinstance(tensor, torch.Tensor) and not name.startswith("_stacked_"):
            if tensor.is_floating_point():
                tensor.fill_(torch.nan)
            scratch_before[name] = tensor.clone()

    # Invalid scratch values must neither trigger the persistent-cache check
    # nor be modified by warmup cleanup (especially the arange index buffer).
    assert not manager.check_invalid_values_in_kv_cache()
    manager._ple_conv_states[1][0, 0, 0] = torch.nan
    assert manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
    for tensor in (
        manager.all_ssm_states[0],
        manager.all_conv_states[0],
        manager._ple_conv_states[1],
        manager._ple_ngram_contexts[1],
    ):
        assert torch.count_nonzero(tensor) == 0
    for name, expected in scratch_before.items():
        torch.testing.assert_close(getattr(state, name), expected, equal_nan=True)
    if page_index_scale > 1:
        # The view must not clear a different layer's interleaved sub-pages.
        for raw in raw_buffers.values():
            assert torch.all(raw[1::page_index_scale] == 7)


@pytest.mark.parametrize("affine", [False, True])
def test_gdn_bind_accepts_affine_and_indirect_views(affine):
    manager = _state_views(layers=3)
    if not affine:
        backing = torch.zeros(4, 5, 2, 4, 4)
        manager.all_ssm_states = [backing[0], backing[1], backing[3]]
    state = GDNReplayState(3)
    state.bind(_layout(layers=(0, 1, 2)), manager.all_ssm_states, manager.all_conv_states)
    assert (state._state_strides is not None) == affine
    assert (state._state_descriptors is not None) != affine
    assert state.old_x.shape == (3, 5, 2, 16, 2, 4)
    owners = []
    for cls in (Qwen35HybridCacheManagerV2, Qwen4ExpHybridCacheManagerV2):
        owner = object.__new__(cls)
        owner._speculative_state = state
        owners.append(owner)
        assert owner.use_gdn_cached_replay_all_layer_commit
    state.shutdown()
    assert state._state_descriptors is None
    assert all(not owner.use_gdn_cached_replay_all_layer_commit for owner in owners)


@pytest.mark.parametrize(
    "state", [IntermediateState(), ReplayHistory(3), GDNReplayState(3), KDAReplayState(2)]
)
def test_no_local_recurrent_layers_allocate_no_algorithm_buffers(state):
    state.bind(_layout(layers=(), slot_capacity=0), [], [])
    state.reset_slots(torch.tensor([0]), [0])
    assert not any(isinstance(value, torch.Tensor) for value in vars(state).values())


def test_kda_acceptance_reset_relocation_and_transfer():
    manager = _state_views()
    state = KDAReplayState(2)
    state.bind(_layout(), manager.all_ssm_states, manager.all_conv_states)
    state.prev_num_accepted_tokens.copy_(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))
    for buffer in state._layer_replay_buffers():
        for slot in range(5):
            buffer[:, slot].fill_(slot + 10)
    before = [buffer.clone() for buffer in state._layer_replay_buffers()]
    state.relocate_slots([0, 1], [1, 2])
    assert state.prev_num_accepted_tokens.tolist() == [1, 1, 2, 4, 5]
    for old, new in zip(before, state._layer_replay_buffers()):
        torch.testing.assert_close(new[:, 1], old[:, 0])
        torch.testing.assert_close(new[:, 2], old[:, 1])
    batch = MambaAcceptanceBatch(
        attention_metadata=None,
        num_contexts=0,
        num_generations=2,
        num_accepted_tokens=torch.tensor([3, 2]),
        accepted_positions=torch.tensor([2, 1]),
        source_state_indices=torch.arange(2),
        destination_state_indices=torch.tensor([1, 2]),
        is_dummy_request=torch.tensor([False, True]),
    )
    state.update(batch)
    assert state.prev_num_accepted_tokens.tolist() == [1, 2, 2, 4, 5]
    state.reset_slots(torch.tensor([1]), [1])
    assert state.prev_num_accepted_tokens[1] == 0
    for conv in manager.all_conv_states:
        conv[2].copy_(torch.arange(72).reshape(24, 3))
    state.seed_transferred_slots(torch.tensor([2]))
    for layer, conv in enumerate(manager.all_conv_states):
        for section, replay in enumerate((state.kda_conv_q, state.kda_conv_k, state.kda_conv_v)):
            torch.testing.assert_close(
                replay[layer, 2, :, :3], conv[2, section * 8 : (section + 1) * 8]
            )
            assert torch.count_nonzero(replay[layer, 2, :, 3:]) == 0
    assert state.prev_num_accepted_tokens[2] == 0
    assert torch.count_nonzero(state.kda_qkg_cache[:, 2]) == 0


def test_kda_scratch_cost_matches_allocated_slot_buffers():
    layout = _layout()
    state = KDAReplayState(2)
    state.bind(layout, _state_views().all_ssm_states, _state_views().all_conv_states)
    buffers = (*state._layer_replay_buffers(), state.prev_num_accepted_tokens)
    # Beta cache rows retain their logical head count but include alignment padding.
    actual = sum(buffer.untyped_storage().nbytes() for buffer in buffers)
    estimated = sum(state.bytes_per_slot(layout, layer_id) for layer_id in layout.mamba_pp_layers)
    assert actual == estimated * layout.slot_capacity


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("kind", ["intermediate", "mamba2", "replay", "gdn", "kda"])
def test_layer_payload_preserves_storage_and_strides(kind, layer):
    states = {
        "intermediate": IntermediateState,
        "mamba2": Mamba2State,
        "replay": lambda: ReplayHistory(3),
        "gdn": lambda: GDNReplayState(3),
        "kda": lambda: KDAReplayState(2),
    }
    state = states[kind]()
    layout = replace(_layout(), stochastic_rounding=True)
    # Interleave persistent slots to exercise borrowed, non-contiguous pool views.
    conv = torch.zeros(2, 10, 24, 3)[:, ::2]
    ssm = torch.zeros(2, 10, 2, 4, 4)[:, ::2]
    state.bind(layout, list(ssm.unbind()), list(conv.unbind()))
    payload = state.make_layer_cache(layer, conv[layer], ssm[layer])
    expected_type = {
        "intermediate": IntermediateLayerCache,
        "mamba2": IntermediateLayerCache,
        "replay": ReplayLayerCache,
        "gdn": ReplayLayerCache,
        "kda": KDAReplayLayerCache,
    }[kind]
    assert type(payload) is expected_type
    pairs = [(payload.conv, conv[layer]), (payload.temporal, ssm[layer])]
    if kind == "kda":
        assert payload.has_kda_replay_caches
        assert not hasattr(payload, "intermediate_ssm")
        assert not hasattr(payload, "intermediate_conv_window")
        assert not hasattr(payload, "mamba_ssm_rand_seed")
        for name in (
            "kda_conv_q",
            "kda_conv_k",
            "kda_conv_v",
            "kda_qkg_cache",
            "kda_v_cache",
            "kda_beta_cache",
        ):
            pairs.append((getattr(payload, name), getattr(state, name)[layer]))
        assert payload.prev_num_accepted_tokens is state.prev_num_accepted_tokens
    else:
        pairs.append((payload.intermediate_conv_window, state.intermediate_conv[layer]))
        if kind in ("replay", "gdn"):
            assert payload.intermediate_ssm is None
            assert payload.cache_buf_idx is state.cache_buf_idx
            assert payload.prev_num_accepted_tokens is state.prev_num_accepted_tokens
            for name in ("old_x", "old_B", "old_dt", "old_dA_cumsum"):
                pairs.append((getattr(payload, name), getattr(state, name)[layer]))
        else:
            pairs.append((payload.intermediate_ssm, state.intermediate_ssm[layer]))
        if kind == "intermediate":
            assert payload.mamba_ssm_rand_seed is None
        else:
            assert payload.mamba_ssm_rand_seed is state.rand_seed
    for view, source in pairs:
        assert view.data_ptr() == source.data_ptr()
        assert view.shape == source.shape
        assert view.stride() == source.stride()
        view.fill_(7)
        assert torch.all(source == 7)
    state.shutdown()


def test_kda_seed_lifecycle_is_independent_of_layer_payload():
    state = KDAReplayState(2)
    layout = replace(_layout(), stochastic_rounding=True, seed_rank_offset=17)
    views = _state_views()
    state.bind(layout, views.all_ssm_states, views.all_conv_states)
    original = state.rand_seed.clone()
    payload = state.make_layer_cache(0, views.all_conv_states[0], views.all_ssm_states[0])
    assert not hasattr(payload, "mamba_ssm_rand_seed")
    state.reset_slots(torch.tensor([1, 3]), [1, 3])
    assert torch.all(state.rand_seed[[1, 3]] != original[[1, 3]])
    torch.testing.assert_close(state.rand_seed[[0, 2, 4]], original[[0, 2, 4]])
    state.shutdown()
    assert state.rand_seed is None


def test_mamba2_seed_lifecycle_does_not_require_speculative_decoding():
    layout = replace(
        _layout(), spec_config=None, stochastic_rounding=True, ssm_state_dtype=torch.float16
    )
    first, second = Mamba2State(), Mamba2State()
    first.bind(layout, _state_views().all_ssm_states, _state_views().all_conv_states)
    second.bind(layout, _state_views().all_ssm_states, _state_views().all_conv_states)
    torch.testing.assert_close(first.rand_seed, second.rand_seed)
    assert torch.all(first.rand_seed > 0)
    assert first.intermediate_ssm is None
    original = first.rand_seed.clone()
    first.reset_slots(torch.tensor([1]), [1])
    assert first.rand_seed[1] != original[1]
    torch.testing.assert_close(first.rand_seed[[0, 2, 3, 4]], original[[0, 2, 3, 4]])


def test_replay_and_intermediate_are_independent_algorithms():
    assert ReplayHistory.__bases__ == (object,)
    assert GDNReplayState.__bases__ == (ReplayHistory,)
    assert ReplayHistory.__module__.endswith("mamba_cache_manager.replay")
    assert not any("modules.mamba" in cls.__module__ for cls in GDNReplayState.__mro__)
    assert Mamba2State.__bases__ == (IntermediateState,)
    assert not issubclass(ReplayHistory, IntermediateState)
    assert not issubclass(GDNReplayState, Mamba2State)
    assert ReplayHistory(3).intermediate_ssm is None
    assert GDNReplayState(3).intermediate_ssm is None
    state = IntermediateState()
    assert not {"rand_seed", "_seed_request_counter", "_seed_rank_offset"}.intersection(vars(state))
    assert not hasattr(state, "_promote_conv")


@pytest.mark.parametrize("replay_cls", [ReplayHistory, GDNReplayState])
def test_mamba_seed_reset_matches_between_intermediate_and_replay(replay_cls):
    layout = replace(_layout(), stochastic_rounding=True, seed_rank_offset=17)
    views = _state_views()
    intermediate, replay = Mamba2State(), replay_cls(3)
    for state in (intermediate, replay):
        state.bind(layout, views.all_ssm_states, views.all_conv_states)
    torch.testing.assert_close(intermediate.rand_seed, replay.rand_seed)
    original = intermediate.rand_seed.clone()
    replay.prev_num_accepted_tokens.fill_(7)
    replay.cache_buf_idx.fill_(1)
    for buffer in (replay.old_x, replay.old_B, replay.old_dt, replay.old_dA_cumsum):
        buffer.fill_(4)
    slots = torch.tensor([3, 1])
    for state in (intermediate, replay):
        state.reset_slots(slots, [3, 1])
    torch.testing.assert_close(intermediate.rand_seed, replay.rand_seed)
    assert torch.all(intermediate.rand_seed[slots] != original[slots])
    torch.testing.assert_close(intermediate.rand_seed[[0, 2, 4]], original[[0, 2, 4]])
    assert replay.prev_num_accepted_tokens.tolist() == [7, 0, 7, 0, 7]
    assert replay.cache_buf_idx.tolist() == [1, 0, 1, 0, 1]
    for buffer in (replay.old_x, replay.old_B, replay.old_dt, replay.old_dA_cumsum):
        assert torch.count_nonzero(buffer[:, slots]) == 0
        assert torch.all(buffer[:, [0, 2, 4]] == 4)
    for state in (intermediate, replay):
        cache = state.make_layer_cache(0, views.all_conv_states[0], views.all_ssm_states[0])
        assert cache.mamba_ssm_rand_seed is state.rand_seed
        state.shutdown()
        assert state.rand_seed is None


@pytest.mark.parametrize(
    "state_cls", [IntermediateState, Mamba2State, ReplayHistory, GDNReplayState]
)
def test_accepted_state_promotion_preserves_algorithm_boundary(monkeypatch, state_cls):
    from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager

    replay = state_cls in (ReplayHistory, GDNReplayState)
    state = state_cls(3) if replay else state_cls()
    views = _state_views()
    state.bind(_layout(), views.all_ssm_states, views.all_conv_states)
    state.intermediate_conv.copy_(
        torch.arange(state.intermediate_conv.numel()).reshape(state.intermediate_conv.shape)
    )
    if not replay:
        state.intermediate_ssm.fill_(7)
    batch = MambaAcceptanceBatch(
        attention_metadata=None,
        num_contexts=0,
        num_generations=2,
        num_accepted_tokens=torch.tensor([1, 3]),
        accepted_positions=torch.tensor([0, 2]),
        source_state_indices=torch.tensor([1, 0]),
        destination_state_indices=torch.tensor([4, 2]),
        is_dummy_request=None,
    )
    calls = []

    def promote(destination, source, source_indices, positions, destination_indices):
        calls.append(destination.data_ptr())
        for src, pos, dst in zip(
            source_indices.tolist(), positions.tolist(), destination_indices.tolist()
        ):
            destination[:, dst].copy_(source[:, src, pos])

    monkeypatch.setattr(mamba_cache_manager, "_promote_mamba_state_triton", promote)
    state.update(batch)
    assert len(calls) == (2 if replay else 4)
    for layer, conv in enumerate(views.all_conv_states):
        torch.testing.assert_close(conv[4], state.intermediate_conv[layer, 1, 0])
        torch.testing.assert_close(conv[2], state.intermediate_conv[layer, 0, 2])
        assert torch.count_nonzero(conv[[0, 1, 3]]) == 0
    for ssm in views.all_ssm_states:
        if replay:
            assert torch.count_nonzero(ssm) == 0
        else:
            assert torch.all(ssm[[4, 2]] == 7)
            assert torch.count_nonzero(ssm[[0, 1, 3]]) == 0
    cache = state.make_layer_cache(0, views.all_conv_states[0], views.all_ssm_states[0])
    assert (cache.intermediate_ssm is None) == replay
    if replay:
        assert state.prev_num_accepted_tokens.tolist() == [0, 0, 3, 0, 1]
        assert state.intermediate_ssm is None
    state.shutdown()


@pytest.mark.parametrize(
    "state",
    [IntermediateState(), Mamba2State(), ReplayHistory(3), GDNReplayState(3), KDAReplayState(2)],
)
def test_algorithm_borrows_only_tensor_views_and_releases_them(state):
    views = _state_views()
    state.bind(_layout(), views.all_ssm_states, views.all_conv_states)
    assert state._conv_states[0] is views.all_conv_states[0]
    assert not hasattr(state, "manager")
    if isinstance(state, (IntermediateState, ReplayHistory)):
        assert state._ssm_states[0] is views.all_ssm_states[0]
        assert not hasattr(state, "_publish_compatibility_views")
    state.shutdown()
    state.shutdown()
    assert state._conv_states == ()
    if isinstance(state, (IntermediateState, ReplayHistory)):
        assert state._ssm_states == ()
    assert not any(isinstance(value, torch.Tensor) for value in vars(state).values())
    # Releasing borrowed references must not mutate the persistent pool.
    assert views.all_conv_states[0].shape == (5, 24, 3)


@pytest.mark.parametrize(
    "cls", [IntermediateState, Mamba2State, ReplayHistory, GDNReplayState, KDAReplayState]
)
def test_algorithm_methods_do_not_depend_on_manager(cls):
    for function in vars(cls).values():
        if not isinstance(function, FunctionType):
            continue
        code = function.__code__
        assert "manager" not in code.co_varnames[: code.co_argcount]
        assert not {
            "_request_id_to_state_index",
            "_request_id_to_is_dummy",
            "_publish_compatibility_views",
            "_publish_replay_views",
        }.intersection(code.co_names)


@pytest.mark.parametrize(
    "manager_cls",
    [
        NemotronHybridCacheManagerV2,
        Qwen35HybridCacheManagerV2,
        Qwen4ExpHybridCacheManagerV2,
        KimiK3HybridCacheManagerV2,
    ],
)
def test_model_compatibility_properties_follow_owned_state(manager_cls):
    manager = object.__new__(manager_cls)
    manager._speculative_state = IntermediateState()
    if manager_cls is KimiK3HybridCacheManagerV2:
        manager._kda_replay = None
    fields = {
        "intermediate_state_indices": "intermediate_indices",
        "intermediate_ssm_states": "intermediate_ssm",
        "intermediate_conv_states": "intermediate_conv",
    }
    for property_name, field in fields.items():
        value = torch.zeros(1)
        setattr(manager._speculative_state, field, value)
        assert getattr(manager, property_name) is value
        assert property_name not in vars(manager)
        with pytest.raises(AttributeError):
            setattr(manager, property_name, value)
    manager._speculative_state.shutdown()
    assert all(getattr(manager, name) is None for name in fields)


def test_k3_resolves_transfer_requests_before_calling_algorithm():
    manager = object.__new__(KimiK3HybridCacheManagerV2)
    manager._request_id_to_state_index = {101: 3, 202: 1}
    replay = KDAReplayState(2)
    replay.prev_num_accepted_tokens = torch.zeros(5, dtype=torch.int32)
    replay.seed_transferred_slots = Mock()
    manager._kda_replay = replay
    manager.on_state_transfer_complete([101, 999, 202, 101])
    slots = replay.seed_transferred_slots.call_args.args[0]
    assert slots.tolist() == [1, 3]
    assert slots.dtype == torch.long
    replay.seed_transferred_slots.reset_mock()
    manager.on_state_transfer_complete([999])
    replay.seed_transferred_slots.assert_not_called()


@pytest.mark.parametrize("draft_lengths", [[], [0, 0], [1, 0], [1, 1]])
def test_k3_host_acceptance_handles_empty_mixed_and_completed_requests(draft_lengths):
    manager = object.__new__(KimiK3HybridCacheManagerV2)
    replay = KDAReplayState(2)
    replay.prev_num_accepted_tokens = torch.tensor([5, 6], dtype=torch.int32)
    manager._kda_replay = replay
    manager._request_id_to_state_index = {101: 0}
    manager._request_id_to_is_dummy = {}
    requests = [
        SimpleNamespace(
            py_request_id=101 + i, py_draft_tokens=[1] * length, py_num_accepted_draft_tokens=1
        )
        for i, length in enumerate(draft_lengths)
    ]
    batch = SimpleNamespace(generation_requests=requests)
    if draft_lengths == [1, 0]:
        with pytest.raises(RuntimeError, match="Mixed drafted/undrafted"):
            manager._record_replay_request_acceptance(batch)
    else:
        manager._record_replay_request_acceptance(batch)
    assert replay.prev_num_accepted_tokens.tolist() == (
        [1, 6] if draft_lengths == [1, 1] else [5, 6]
    )


def _slot_view_manager(
    monkeypatch, *, scale=6, role_count=2, page_shape=(4, 2, 3), dtype=torch.float32
):
    manager = object.__new__(Qwen4ExpHybridCacheManagerV2)
    manager.layer_offsets = {42: 0}
    page_elements = 1
    for dim in page_shape:
        page_elements *= dim
    page_bytes = page_elements * dtype.itemsize
    # First role belongs to a later coalesced layer, not the pool's first page.
    layer_offset = 2 if scale > 2 else 0
    backing = torch.arange((3 * scale + layer_offset) * page_elements, dtype=torch.float32).to(
        dtype
    )
    addresses = {
        Role.KEY: 1000 + layer_offset * page_bytes,
        Role.VALUE: 1000 + (layer_offset + 1) * page_bytes,
    }
    converter = SimpleNamespace(scale=scale, expansion=1, layer_offset=layer_offset)
    manager.impl = SimpleNamespace(
        get_mem_pool_base_address=lambda layer, role, mode: addresses[role],
        get_page_stride=lambda layer, role: page_bytes,
        get_page_index_converter=lambda layer, role: converter,
        get_page_index_upper_bound=lambda layer, role: 3 * scale - layer_offset,
    )
    globals_ = Qwen4ExpHybridCacheManagerV2._get_view_by_role_and_layer.__globals__
    monkeypatch.setitem(globals_, "TensorWrapper", lambda address, dtype, shape: (address, shape))

    def convert(wrapper):
        address, shape = wrapper
        count = 1
        for dim in shape:
            count *= dim
        offset = (address - 1000) // dtype.itemsize
        return backing[offset : offset + count].reshape(shape)

    monkeypatch.setitem(globals_, "convert_to_torch_tensor", convert)
    roles = (Role.KEY, Role.VALUE)[:role_count]
    return manager, roles, list(page_shape), converter, backing


@pytest.mark.parametrize("role_count,scale", [(1, 1), (2, 2), (1, 6), (2, 6)])
@pytest.mark.parametrize("page_shape", [(4, 2, 3), (2, 4, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slot_role_view_preserves_coalesced_stride(
    monkeypatch, role_count, scale, page_shape, dtype
):
    manager, roles, shape, converter, backing = _slot_view_manager(
        monkeypatch, scale=scale, role_count=role_count, page_shape=page_shape, dtype=dtype
    )
    view = manager._get_view_by_role_and_layer(42, roles, dtype=dtype, page_shape=shape)
    assert view.shape == (3, role_count, *page_shape)
    assert view.stride(0) == scale * 24
    assert view.stride(1) == 24
    assert view.data_ptr() == backing.data_ptr() + converter.layer_offset * 24 * dtype.itemsize
    view[1, 0].fill_(7)
    start = (converter.layer_offset + scale) * 24
    assert torch.all(backing[start : start + 24] == 7)


@pytest.mark.parametrize("invalid", ["stride", "adjacency", "scale", "expansion", "page_count"])
def test_slot_role_view_rejects_incompatible_physical_layout(monkeypatch, invalid):
    manager, roles, shape, converter, _ = _slot_view_manager(monkeypatch)
    if invalid == "stride":
        manager.impl.get_page_stride = lambda layer, role: 1
    elif invalid == "adjacency":
        manager.impl.get_mem_pool_base_address = lambda layer, role, mode: 1000
    elif invalid == "scale":
        converter.scale = 1
    elif invalid == "expansion":
        converter.expansion = 2
    else:
        manager.impl.get_page_index_upper_bound = lambda layer, role: 1
    with pytest.raises(RuntimeError):
        manager._get_view_by_role_and_layer(42, roles, dtype=torch.float32, page_shape=shape)


def test_qwen4_position_uses_model_owned_slot_view(monkeypatch):
    manager = object.__new__(Qwen4ExpHybridCacheManagerV2)
    manager.qsa_position_layer_id = 42
    manager.tokens_per_block = 4
    manager._get_view_by_role_and_layer = Mock(
        return_value=torch.zeros(3, 1, 4, 3, dtype=torch.int32)
    )
    assert manager.get_qsa_position_buffer().shape == (3, 4, 3)
    manager._get_view_by_role_and_layer.assert_called_once()
    manager.qsa_position_layer_id = None
    manager._get_view_by_role_and_layer.reset_mock()
    assert manager.get_qsa_position_buffer() is None
    manager._get_view_by_role_and_layer.assert_not_called()


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_qwen4_main_kv_view_supplies_only_logical_geometry(monkeypatch, kv_layout):
    manager = object.__new__(Qwen4ExpHybridCacheManagerV2)
    namespace = Qwen4ExpHybridCacheManagerV2.get_buffers.__globals__
    manager._qsa_enabled = True
    manager.dtype = next(iter(namespace["QSA_SPARSE_KV_CACHE_DTYPES"]))
    manager.kv_cache_type = namespace["CacheTypeCpp"].SELF
    manager.layer_offsets = {42: 0}
    manager.head_dim_per_layer = [3]
    manager.num_kv_heads_per_layer = [2]
    manager.tokens_per_block = 4
    monkeypatch.setitem(namespace, "binding_to_torch_dtype", lambda dtype: torch.bfloat16)
    sentinel = object()
    manager._get_view_by_role_and_layer = Mock(return_value=sentinel)
    assert manager.get_buffers(42, kv_layout) is sentinel
    manager._get_view_by_role_and_layer.assert_called_once_with(
        42,
        (Role.KEY, Role.VALUE),
        dtype=torch.bfloat16,
        page_shape=[4, 2, 3] if kv_layout == "NHD" else [2, 4, 3],
    )
    assert manager.get_buffers(99, kv_layout) is None
    manager.kv_cache_type = namespace["CacheTypeCpp"].SELFKONLY
    with pytest.raises(NotImplementedError, match="both K and V"):
        manager.get_buffers(42, kv_layout)


@pytest.mark.parametrize("invalid", [None, "pool", "scale", "expansion"])
def test_qwen4_attention_pool_layout_validates_backend_contract(invalid):
    manager = object.__new__(Qwen4ExpHybridCacheManagerV2)
    manager.layer_offsets = {42: 0}
    manager.qsa_position_layer_id = 42
    manager.qsa_sparse_layer_ids = [42]
    manager.layer_to_pool_mapping_dict = {0: 0}
    manager.num_attention_op_pools = 1
    converter = SimpleNamespace(scale=6, expansion=1)
    manager.impl = SimpleNamespace(get_page_index_converter=lambda layer, role: converter)
    if invalid == "pool":
        manager.layer_to_pool_mapping_dict[0] = 1
    elif invalid == "scale":
        converter.scale = 0
    elif invalid == "expansion":
        converter.expansion = 2
    if invalid is None:
        assert manager.get_qsa_attention_pool_layout() == (0, 6)
    else:
        with pytest.raises(RuntimeError):
            manager.get_qsa_attention_pool_layout()


def test_qwen4_requires_one_mapping_and_skips_nonlocal_layers():
    manager = object.__new__(Qwen4ExpHybridCacheManagerV2)
    manager.qsa_position_layer_id = 42
    manager.qsa_sparse_layer_ids = [42, 43, 99]
    manager.layer_offsets = {42: 0, 43: 1}
    manager.layer_to_pool_mapping_dict = {0: 0, 1: 0}
    manager.num_attention_op_pools = 1
    converters = {
        0: SimpleNamespace(scale=6, expansion=1),
        1: SimpleNamespace(scale=6, expansion=1),
    }
    manager.impl = SimpleNamespace(
        get_page_index_converter=Mock(side_effect=lambda layer, role: converters[layer])
    )
    assert manager.get_qsa_attention_pool_layout() == (0, 6)
    assert {call.args[0] for call in manager.impl.get_page_index_converter.call_args_list} == {0, 1}
    converters[1].scale = 4
    with pytest.raises(RuntimeError, match="do not share"):
        manager.get_qsa_attention_pool_layout()


def test_constructor_validates_before_pool_and_binds_after_slot_capacity(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        mamba_cache_manager_v2 as module,
    )

    events = []
    zeros, arange = torch.zeros, torch.arange

    def cpu_zeros(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return zeros(*args, **kwargs)

    def cpu_arange(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return arange(*args, **kwargs)

    def create_pool(self, *args, **kwargs):
        assert events == ["validate"]
        assert self._state_layout.slot_capacity is None
        assert len(self._get_recurrent_buffer_configs(0)) == 4
        events.append("pool")
        self.max_batch_size = kwargs["max_batch_size"]
        self.mapping = kwargs["mapping"]
        self.layer_offsets = {0: 0, 1: 1}
        self.impl = SimpleNamespace(
            get_layer_group_id=lambda layer: 0,
            get_page_index_scale=lambda layer, role: 1,
            get_page_index_upper_bound=lambda layer, role: 5,
        )

    class ObservedQwen4(Qwen4ExpHybridCacheManagerV2):
        def _initialize_model_state(self):
            state = super()._initialize_model_state()
            events.append("validate")
            return state

        def _get_state_buffer(self, local_layer_idx, role, dtype, state_shape):
            assert events[:2] == ["validate", "pool"]
            assert self._state_layout.slot_capacity == 5
            events.append(role)
            return zeros([5, *state_shape], dtype=dtype)

    monkeypatch.setattr(module.KVCacheManagerV2, "__init__", create_pool)
    monkeypatch.setattr(module, "get_pp_layers", lambda *args, **kwargs: ([0, 1], None))
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "arange", cpu_arange)
    config = SimpleNamespace(enable_block_reuse=False)
    config.model_copy = lambda **kwargs: config
    params = Qwen4ExpPLECacheParams(
        ple_layer_mask=[True, False],
        num_ple_layers=1,
        short_conv_channels=16,
        short_conv_state_len=6,
        ngram_context_len=2,
        conv_state_dtype=torch.bfloat16,
    )
    manager = ObservedQwen4(
        4,
        4,
        2,
        2,
        4,
        1,
        [True, False],
        torch.float32,
        torch.float32,
        config,
        module.CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        layer_mask=[False, True],
        use_replay_state_update=False,
        qwen4_exp_ple_cache_params=params,
    )
    assert events == [
        "validate",
        "pool",
        MambaRole.SSM_STATE,
        MambaRole.CONV_STATE,
        PLE_CONV_STATE,
        PLE_NGRAM_CONTEXT,
    ]
    assert manager.ple_layer_cache(0)[0].shape == (5, 16, 6)


@pytest.mark.parametrize("model_type", ["nemotron_hybrid", "qwen3_next", "kimi_linear"])
@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_constructor_resolves_model_replay(monkeypatch, model_type, enabled):
    from tensorrt_llm._torch.modules.fla import cache_manager as gdn
    from tensorrt_llm._torch.modules.kimi_kda import cache_manager as kda
    from tensorrt_llm._torch.modules.mamba import cache_manager as mamba
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import legacy

    spec = SimpleNamespace(tokens_per_gen_step=4)
    selectors = [
        Mock(return_value=SimpleNamespace(uses_replay=enabled)),
        Mock(return_value=object() if enabled else None),
        Mock(return_value=SimpleNamespace(num_speculative_tokens=3) if enabled else None),
    ]
    monkeypatch.setattr(mamba, "select_mamba2_state", selectors[0])
    monkeypatch.setattr(gdn, "select_gdn_replay_state", selectors[1])
    monkeypatch.setattr(kda, "select_kda_replay_state", selectors[2])
    mamba_init, kv_init = Mock(), Mock()
    monkeypatch.setattr(legacy.MambaCacheManager, "__init__", mamba_init)
    monkeypatch.setattr(legacy.KVCacheManager, "__init__", kv_init)
    legacy.MixedMambaHybridCacheManager(
        4,
        4,
        2,
        2,
        4,
        1,
        [True, False],
        torch.float32,
        torch.float32,
        SimpleNamespace(enable_block_reuse=False),
        legacy.CacheTypeCpp.SELF,
        num_layers=1,
        layer_mask=[False, True],
        num_kv_heads=2,
        head_dim=4,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        spec_config=spec,
        model_type=model_type,
        use_replay_state_update=None,
    )
    expected_selector = ["nemotron_hybrid", "qwen3_next", "kimi_linear"].index(model_type)
    assert [selector.call_count for selector in selectors] == [
        int(i == expected_selector) for i in range(3)
    ]
    kwargs = mamba_init.call_args.kwargs
    assert kwargs["model_type"] == (
        "nemotron_hybrid" if model_type == "nemotron_hybrid" else "qwen3_next"
    )
    assert kwargs["use_replay_state_update"] == (enabled and model_type != "kimi_linear")
    assert kwargs["kda_replay_num_spec"] == (3 if enabled and model_type == "kimi_linear" else None)
    kv_init.assert_called_once()


@pytest.mark.parametrize("requested", [False, True])
def test_legacy_replay_explicit_override_bypasses_auto_selection(monkeypatch, requested):
    from tensorrt_llm._torch.modules.mamba import cache_manager as mamba
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import legacy

    selector = Mock(side_effect=AssertionError("Explicit replay flag must bypass selection"))
    monkeypatch.setattr(mamba, "select_mamba2_state", selector)
    assert legacy._resolve_legacy_replay_options(
        legacy.CppMambaHybridCacheManager,
        "nemotron_hybrid",
        object(),
        torch.float32,
        False,
        requested,
    ) == ("nemotron_hybrid", requested, None)
    selector.assert_not_called()


def test_legacy_kda_replay_retains_cpp_manager_gate(monkeypatch):
    from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import legacy

    monkeypatch.setattr(_kda_kernels, "is_kda_mtp_verify_available", lambda: True)
    spec = SimpleNamespace(tokens_per_gen_step=4)
    for manager_cls, expected in (
        (legacy.MixedMambaHybridCacheManager, 3),
        (legacy.CppMambaHybridCacheManager, None),
    ):
        assert legacy._resolve_legacy_replay_options(
            manager_cls,
            "kimi_linear",
            spec,
            torch.float32,
            False,
            False,
        ) == ("qwen3_next", False, expected)
