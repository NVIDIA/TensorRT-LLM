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
"""Disaggregated completion reservation in KVCacheManagerV2.

With no eviction tier and a KV connector whose worker declares
``requires_completion_reservation``, or ``kv_cache_config.enable_disagg_completion_reservation``
set, a full-attention disaggregated server reserves the pages an admitted request still needs to
finish: a decode request its output growth, a connector prefill its remaining prompt. A new
request is admitted only if the reservation of every admitted request still fits. Without either
opt-in the server keeps its existing admission.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)


class _ConnectorDouble:
    """Connector surface used here: offers no prefix and holds no load."""

    def __init__(self, declares: object = True, capacity_only: bool = False) -> None:
        self.prefix_reservations_enabled = False
        self.capacity_only = capacity_only
        self.admission_query_capable = False
        self.worker = SimpleNamespace(requires_completion_reservation=declares)

    def release_unstarted_prefix_loads(self, request: LlmRequest) -> None:
        pass

    def has_pending_load(self, request: LlmRequest) -> bool:
        return False

    def release_prefix_reservation(self, request: LlmRequest) -> None:
        pass

    def reset_request_state(self, request: LlmRequest) -> None:
        pass


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "available, capacities, limits, candidate_limit, expected",
    [
        ([7], [8, 0], {1: 6}, 3, True),
        ([6], [8, 0], {1: 6}, 3, False),
        ([4], [20, 0], {1: 6}, 3, True),
        ([100, 6], [8, 0], {1: 6}, 3, False),
    ],
)
def test_completion_reservation_counts_unallocated_pages_in_every_pool(
    available: list[int],
    capacities: list[int],
    limits: dict[int, int],
    candidate_limit: int,
    expected: bool,
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = dict(limits)
    manager._disagg_completion_pool_counts = {i: 1 for i in range(len(available))}
    manager.kv_cache_map = {
        i + 1: Mock(capacity=capacity, num_blocks=(capacity + 3) // 4)
        for i, capacity in enumerate(capacities)
    }
    manager._get_storage_statistics = Mock(
        return_value=[SimpleNamespace(available=count) for count in available]
    )

    assert manager._can_reserve_disagg_completion(2, candidate_limit) is expected
    # Checking admission neither consumes a reservation nor touches a cache.
    assert manager._disagg_completion_limits == limits
    for cache in manager.kv_cache_map.values():
        cache.resize.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_block_reuse", [False, True])
@pytest.mark.parametrize("ledger_tokens_per_block, expected_blocks", [(4, 10), (16, 3)])
def test_generation_limit_counts_pool_variants_and_speculative_padding(
    enable_block_reuse: bool, ledger_tokens_per_block: int, expected_blocks: int
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._ledger_tokens_per_block = ledger_tokens_per_block
    manager.max_seq_len = 32
    manager._generation_kv_capacity_headroom = 7  # 1 extra KV + 5 draft + 1 base token
    manager.enable_block_reuse = enable_block_reuse
    request = Mock(max_new_tokens=20)

    # Clamp to the model sequence length, then add the maximum generation lead:
    # 32 + 7 = 39 tokens.
    limit = manager._disagg_generation_limit(request, 24)
    assert limit == expected_blocks + int(enable_block_reuse)

    manager._disagg_completion_limits = {}
    manager._disagg_completion_pool_counts = {0: 2}
    manager.kv_cache_map = {1: Mock(capacity=0, num_blocks=0)}
    stats = SimpleNamespace(available=2 * limit - 1)
    manager._get_storage_statistics = Mock(return_value=[stats])
    assert not manager._can_reserve_disagg_completion(1, limit)
    stats.available = 2 * limit
    assert manager._can_reserve_disagg_completion(1, limit)


def _clamp_manager(available: int, max_seq_len: int, enable_block_reuse: bool = False):
    manager = object.__new__(KVCacheManagerV2)
    manager._ledger_tokens_per_block = 4
    manager._generation_kv_capacity_headroom = 3
    manager.enable_block_reuse = enable_block_reuse
    manager.max_seq_len = max_seq_len
    manager._disagg_completion_pool_counts = {0: 2}
    manager._get_storage_statistics = Mock(return_value=[SimpleNamespace(available=available)])
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_block_reuse", [False, True])
def test_max_seq_len_is_clamped_to_the_empty_pool(enable_block_reuse: bool) -> None:
    """A bound larger than the empty pool would be refused on every pass."""
    manager = _clamp_manager(available=20, max_seq_len=1000, enable_block_reuse=enable_block_reuse)
    manager._clamp_max_seq_len_to_completion_pool()
    cap_blocks = 10
    assert manager.max_seq_len == (cap_blocks - int(enable_block_reuse)) * 4 - 3
    assert manager._disagg_completion_limit(manager.max_seq_len) <= cap_blocks
    assert manager._disagg_completion_limit(manager.max_seq_len + 1) > cap_blocks
    # A request of exactly the clamped length fits beside nothing else.
    manager._disagg_completion_limits = {}
    manager.kv_cache_map = {1: Mock(num_blocks=0)}
    assert manager._can_reserve_disagg_completion(
        1, manager._disagg_completion_limit(manager.max_seq_len)
    )


@pytest.mark.cpu_only
def test_max_seq_len_within_the_pool_is_unchanged() -> None:
    manager = _clamp_manager(available=200, max_seq_len=64)
    manager._clamp_max_seq_len_to_completion_pool()
    assert manager.max_seq_len == 64


@pytest.mark.cpu_only
def test_a_pool_without_room_for_one_request_fails_at_startup() -> None:
    manager = _clamp_manager(available=1, max_seq_len=64)
    with pytest.raises(ValueError, match="at least one request"):
        manager._clamp_max_seq_len_to_completion_pool()


def _arrival(prompt: int, output: int, generation_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        orig_prompt_len=prompt, max_new_tokens=output, is_generation_only_request=generation_only
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_block_reuse", [False, True])
def test_arrival_budget_rejects_a_prompt_whose_bound_exceeds_the_empty_pool(
    enable_block_reuse: bool,
) -> None:
    """On a pool smaller than one request's bound, a request that would be refused on every pass
    is reported through the arrival budget, so the executor rejects it instead of deadlocking."""
    manager = _clamp_manager(available=20, max_seq_len=1000, enable_block_reuse=enable_block_reuse)
    manager._host_history_capacity_blocks = None
    manager._disagg_context_completion_reservation = True
    assert not manager.kv_block_budget_applies()
    manager._clamp_max_seq_len_to_completion_pool()
    assert manager.kv_block_budget_applies()
    cap_blocks = 10
    clamped = manager.max_seq_len

    # A decode request that may run to max_seq_len (no max_tokens) fits the clamped length.
    required, capacity = manager.get_request_kv_block_budget(
        _arrival(prompt=8, output=1000, generation_only=True)
    )
    assert capacity == cap_blocks and required <= capacity
    # A prompt longer than the clamped length is refused by every pass: rejected on arrival.
    for generation_only in (False, True):
        required, capacity = manager.get_request_kv_block_budget(
            _arrival(prompt=clamped + 1, output=1, generation_only=generation_only)
        )
        assert required > capacity
    # The connector-prefill bound uses the prompt alone and fits up to the clamped length.
    required, capacity = manager.get_request_kv_block_budget(
        _arrival(prompt=clamped, output=1000, generation_only=False)
    )
    assert required <= capacity


@pytest.mark.cpu_only
def test_arrival_budget_ignores_context_requests_without_the_prefill_reservation() -> None:
    manager = _clamp_manager(available=20, max_seq_len=1000)
    manager._host_history_capacity_blocks = None
    manager._disagg_context_completion_reservation = False
    manager._clamp_max_seq_len_to_completion_pool()
    assert manager.get_request_kv_block_budget(_arrival(10_000, 1, generation_only=False)) is None
    required, capacity = manager.get_request_kv_block_budget(_arrival(10_000, 1, True))
    assert required > capacity


@pytest.mark.cpu_only
def test_executor_rejects_a_request_the_reservation_can_never_admit() -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType

    manager = _clamp_manager(available=20, max_seq_len=1000)
    manager._host_history_capacity_blocks = None
    manager._disagg_context_completion_reservation = True
    manager._clamp_max_seq_len_to_completion_pool()
    executor = object.__new__(PyExecutor)
    executor.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})
    executor._validate_encoder_output_len = Mock()
    request = SimpleNamespace(
        orig_prompt_len=manager.max_seq_len + 1,
        max_new_tokens=1,
        is_generation_only_request=False,
        py_beam_width=1,
    )
    with pytest.raises(ValueError, match="exceeds its capacity of 10 blocks"):
        executor._validate_request_budget(request)


@pytest.mark.cpu_only
@pytest.mark.parametrize("available, fits", [(4, False), (40, True)])
def test_a_refused_gen_init_creates_no_cache_without_block_reuse(
    available: int, fits: bool
) -> None:
    """Without block reuse a fresh cache holds no block: the bound is checked before anything is
    created, so a refused request costs no create / free cycle per pass."""
    manager = object.__new__(KVCacheManagerV2)
    manager._ledger_tokens_per_block = 4
    manager._generation_kv_capacity_headroom = 1
    manager.enable_block_reuse = False
    manager.max_seq_len = 1024
    manager._has_cp_helix = False
    manager._disagg_completion_limits = {}
    manager._disagg_completion_pool_counts = {0: 1}
    manager.kv_cache_map = {}
    manager._get_storage_statistics = Mock(return_value=[SimpleNamespace(available=available)])
    manager.prepare_context_cache = Mock(return_value=None)
    manager._release_failed_disagg_init = Mock(return_value=True)
    req = SimpleNamespace(py_request_id=9, prompt_len=24, max_new_tokens=8)

    assert manager._prepare_disagg_gen_init_cache(req) is False
    # A bound that fits goes on to create the cache (which this stub refuses).
    assert manager.prepare_context_cache.called is fits


def _context_reservation_manager(available: int, **connector_overrides) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager._ledger_tokens_per_block = 4
    manager._generation_kv_capacity_headroom = 1
    manager.enable_block_reuse = False
    manager.is_draft = False
    manager._disagg_completion_limits = {}
    manager._disagg_completion_pool_counts = {0: 1}
    # The connector declares the reservation, which turns the connector prefill one on.
    manager._disagg_context_completion_reservation = True
    manager.kv_cache_map = {7: Mock(num_blocks=0)}
    manager._get_storage_statistics = Mock(return_value=[SimpleNamespace(available=available)])
    connector = dict(
        prefix_reservations_enabled=False,
        capacity_only=False,
        worker=SimpleNamespace(requires_completion_reservation=True),
    )
    connector.update(connector_overrides)
    manager.kv_connector_manager = SimpleNamespace(**connector)
    return manager


def _first_chunk(**overrides) -> SimpleNamespace:
    fields = dict(
        py_request_id=7,
        prompt_len=16,
        is_first_context_chunk=True,
        is_dummy=False,
        is_generation_only_request=False,
        is_disagg_generation_init_state=False,
        py_connector_allocation_reported=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.cpu_only
@pytest.mark.parametrize("available, fits", [(5, True), (4, False)])
def test_context_reservation_records_a_bound_only_when_it_fits(available: int, fits: bool) -> None:
    # 16 prompt tokens + 1 base token in 4-token blocks: 5 blocks.
    manager = _context_reservation_manager(available=available)
    request = _first_chunk()

    assert manager.reserve_context_completion(request) is fits
    assert manager._disagg_completion_limits == ({7: 5} if fits else {})
    if fits:
        # A recorded bound is not checked again on later passes.
        manager._get_storage_statistics.return_value = [SimpleNamespace(available=0)]
        assert manager.reserve_context_completion(request)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "request_overrides, connector_overrides, has_pool_counts",
    [
        (dict(is_first_context_chunk=False), {}, True),
        ({}, {}, False),
        ({}, dict(prefix_reservations_enabled=True), True),
        (dict(is_disagg_generation_init_state=True), {}, True),
    ],
    ids=["later_chunk", "eviction_tier", "source_reservations", "gen_init"],
)
def test_context_reservation_is_a_noop_outside_conservative_connector_prefill(
    request_overrides: dict, connector_overrides: dict, has_pool_counts: bool
) -> None:
    manager = _context_reservation_manager(available=0, **connector_overrides)
    if not has_pool_counts:
        manager._disagg_completion_pool_counts = {}

    assert manager.reserve_context_completion(_first_chunk(**request_overrides))
    assert manager._disagg_completion_limits == {}
    manager._get_storage_statistics.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "request_overrides, released",
    [
        ({}, True),
        (dict(py_connector_allocation_reported=True), False),
        (dict(is_first_context_chunk=False), False),
        # A reverted disaggregated generation-init admission, before its transfer.
        (dict(is_disagg_generation_init_state=True), True),
    ],
    ids=["unconsumed", "reported", "later_chunk", "gen_init"],
)
def test_unconsumed_context_reservation_is_returned(
    request_overrides: dict, released: bool
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = {7: 5}
    manager._disagg_completion_pool_counts = {0: 1}

    manager._release_unconsumed_context_reservation(_first_chunk(**request_overrides))

    assert (7 not in manager._disagg_completion_limits) is released


@pytest.mark.cpu_only
@pytest.mark.parametrize("first_chunk", [True, False])
def test_failed_first_chunk_resize_releases_its_bound(first_chunk: bool) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = {7: 5}
    manager._disagg_completion_pool_counts = {0: 1}
    kv_cache = Mock(capacity=0, resize=Mock(return_value=False))
    manager.kv_cache_map = {7: kv_cache}
    manager._has_cp_helix = False
    manager.is_disagg = True
    manager.num_extra_kv_tokens = 0
    manager.is_draft = False
    manager.kv_connector_manager = SimpleNamespace(
        prefix_reservations_enabled=False,
        capacity_only=False,
        worker=SimpleNamespace(requires_completion_reservation=True),
    )
    request = SimpleNamespace(
        py_request_id=7,
        is_disagg_generation_init_state=False,
        is_dummy_request=False,
        context_current_position=0,
        is_first_context_chunk=first_chunk,
        py_connector_allocation_reported=False,
    )

    assert not KVCacheManagerV2.resize_context(manager, request, 4)

    kv_cache.resize.assert_called_once_with(4)
    if first_chunk:
        kv_cache.suspend.assert_called_once_with()
        assert 7 not in manager._disagg_completion_limits
    else:
        kv_cache.suspend.assert_not_called()
        assert manager._disagg_completion_limits == {7: 5}


@pytest.mark.cpu_only
@pytest.mark.parametrize("reported", [False, True])
def test_revert_without_capacity_growth_returns_an_unconsumed_bound(reported: bool) -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = {7: 5}
    manager._disagg_completion_pool_counts = {0: 1}
    manager.is_draft = False
    manager.kv_connector_manager = _ConnectorDouble()
    request = _first_chunk(py_connector_allocation_reported=reported, py_ctx_pre_resize_cap=None)

    assert manager.revert_allocate_context(request)

    assert (7 in manager._disagg_completion_limits) is reported


def _attention_layer(sliding_window_size: int | None) -> AttentionLayerConfig:
    return AttentionLayerConfig(
        layer_id=0,
        buffers=[BufferConfig(role="key", size=256)],
        sliding_window_size=sliding_window_size,
    )


def _gate_manager(**overrides) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    fields = dict(
        is_disagg=True,
        can_evict=False,
        is_draft=False,
        max_beam_width=1,
        mapping=SimpleNamespace(pp_size=1),
        max_seq_len=64,
        kv_cache_manager_py_config=SimpleNamespace(layers=[_attention_layer(None)]),
        kv_connector_manager=_ConnectorDouble(),
        enable_disagg_completion_reservation=False,
    )
    fields.update(overrides)
    # can_evict is a property over the configured tiers; set the tier flag behind it.
    fields["_can_evict"] = fields.pop("can_evict")
    for name, value in fields.items():
        setattr(manager, name, value)
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, applies",
    [
        pytest.param({}, True, id="full_attention"),
        # A window that covers max_seq_len behaves as full attention.
        pytest.param(
            dict(kv_cache_manager_py_config=SimpleNamespace(layers=[_attention_layer(64)])),
            True,
            id="window_covers_max_seq_len",
        ),
        pytest.param(dict(is_disagg=False), False, id="not_disagg"),
        pytest.param(dict(can_evict=True), False, id="eviction_tier"),
        pytest.param(dict(kv_connector_manager=None), False, id="no_connector"),
        pytest.param(
            dict(kv_connector_manager=_ConnectorDouble(declares=False)),
            False,
            id="undeclared_connector",
        ),
        # Only an explicit True counts, not a truthy value such as a mock's attribute.
        pytest.param(
            dict(kv_connector_manager=_ConnectorDouble(declares="yes")),
            False,
            id="truthy_non_bool_declaration",
        ),
        pytest.param(
            dict(kv_connector_manager=_ConnectorDouble(capacity_only=True)),
            False,
            id="capacity_only_connector",
        ),
        pytest.param(dict(mapping=SimpleNamespace(pp_size=2)), False, id="pipeline_parallel"),
        pytest.param(
            dict(
                kv_cache_manager_py_config=SimpleNamespace(
                    layers=[
                        _attention_layer(None),
                        SsmLayerConfig(
                            layer_id=1, buffers=[BufferConfig(role="ssm_state", size=256)]
                        ),
                    ]
                )
            ),
            False,
            id="recurrent_layer",
        ),
        # The configuration opts in without a declaring connector, under the same conditions.
        pytest.param(
            dict(kv_connector_manager=None, enable_disagg_completion_reservation=True),
            True,
            id="config_without_connector",
        ),
        pytest.param(
            dict(
                kv_connector_manager=None,
                enable_disagg_completion_reservation=True,
                kv_cache_manager_py_config=SimpleNamespace(layers=[_attention_layer(16)]),
            ),
            False,
            id="config_sliding_window",
        ),
    ],
)
def test_completion_reservation_gate(overrides: dict, applies: bool) -> None:
    assert _gate_manager(**overrides)._disagg_completion_reservation_applies() is applies


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "model_default, env, killed, expected",
    [
        (False, None, False, False),
        (True, None, False, True),
        (True, "0", False, False),
        (False, "1", False, True),
        (True, None, True, False),
        (True, "1", True, True),
        (True, "off", False, False),
        (False, "Yes", False, True),
        (True, None, "yes", False),
    ],
)
def test_model_default_reserves_decode_headroom_without_an_opt_in(
    monkeypatch, model_default: bool, env, killed: bool, expected: bool
) -> None:
    """A model's opt-in (RuntimeHooks.disagg_gen_completion_reservation) turns the decode reservation on for a
    qualifying manager without the configuration or a connector; the variable decides either way, the older kill
    switch turns the model's default off. The connector prefill reservation still needs the opt-in."""
    import tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 as kvcm

    monkeypatch.setattr(kvcm, "_gen_completion_reservation_model_default", False)
    kvcm.set_gen_completion_reservation_default(model_default)
    if env is None:
        monkeypatch.delenv(kvcm.GEN_COMPLETION_RESERVATION_ENV, raising=False)
    else:
        monkeypatch.setenv(kvcm.GEN_COMPLETION_RESERVATION_ENV, env)
    if killed:
        monkeypatch.setenv(
            "TRTLLM_M31_DISABLE_GEN_COMPLETION_RESV", killed if isinstance(killed, str) else "1"
        )
    else:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_GEN_COMPLETION_RESV", raising=False)
    manager = _gate_manager(kv_connector_manager=None)
    assert manager._disagg_completion_reservation_applies() is expected
    assert manager._disagg_completion_reservation_opted_in() is False
    # The qualifying conditions still hold: an eviction tier keeps the existing admission.
    assert (
        _gate_manager(
            kv_connector_manager=None, can_evict=True
        )._disagg_completion_reservation_applies()
        is False
    )


def _real_manager(
    kv_cache_config: KvCacheConfig | None = None, **overrides: object
) -> KVCacheManagerV2:
    """A one-layer manager whose tiny GPU pool makes page pressure reachable.

    Its connector declares the reservation unless ``kv_connector_manager`` is overridden.
    """
    if kv_cache_config is None:
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            host_cache_size=0,
            max_gpu_total_bytes=128 << 10,
            max_util_for_resume=1.0,
        )
    kwargs = dict(
        num_layers=1,
        num_kv_heads=8,
        head_dim=64,
        tokens_per_block=4,
        max_seq_len=65536,
        max_batch_size=4,
        max_num_tokens=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=4096,
        is_disagg=True,
        kv_connector_manager=_ConnectorDouble(),
    )
    kwargs.update(overrides)
    torch.cuda.init()
    return KVCacheManagerV2(kv_cache_config, CacheType.SELF, **kwargs)


def _gen_init_request(request_id: int, prompt: int, output: int) -> LlmRequest:
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=output,
        input_tokens=[request_id] * prompt,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    return request


@pytest.mark.parametrize("reserve_growth", [False, True])
def test_disagg_generation_progress_under_real_gpu_page_pressure(reserve_growth: bool) -> None:
    """The guarded path defers a second long prompt; the unguarded control exhausts the pool."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    manager = _real_manager(
        KvCacheConfig(
            enable_block_reuse=True,
            host_cache_size=0,
            max_gpu_total_bytes=128 << 10,
            max_util_for_resume=1.0,
        )
    )
    try:
        assert manager._disagg_completion_pool_counts == {0: 1}
        if not reserve_growth:
            manager._disagg_completion_pool_counts.clear()
        total_blocks = manager.get_kv_cache_stats().max_num_blocks
        prompt_len = total_blocks // 3 * manager.tokens_per_block
        assert prompt_len >= 8
        # Allocation granularity can make the physical pool larger than the tiny
        # requested byte quota. Keep this pressure probe within the actual table.
        assert 2 * prompt_len < manager.max_seq_len

        first = _gen_init_request(1, prompt_len, prompt_len)
        second = _gen_init_request(2, prompt_len, prompt_len)
        short = _gen_init_request(3, 4, 1)

        assert manager.prepare_disagg_gen_init(first)
        assert manager.kv_cache_map[1].capacity == prompt_len
        assert manager.prepare_disagg_gen_init(second) is not reserve_growth
        first.state = LlmRequestState.GENERATION_IN_PROGRESS
        if reserve_growth:
            assert 2 not in manager.kv_cache_map
            assert 2 not in manager._disagg_completion_limits
            assert manager.prepare_disagg_gen_init(short)
            for _ in range(prompt_len):
                assert manager.try_allocate_generation(first)
            manager.free_resources(first)
            assert 1 not in manager._disagg_completion_limits
            assert manager.prepare_disagg_gen_init(second)
            # Cancellation must return both the physical pages and the logical bound.
            manager.free_resources(second)
            manager.free_resources(short)
            assert not manager._disagg_completion_limits
            assert manager.get_kv_cache_stats().free_num_blocks == total_blocks
        else:
            second.state = LlmRequestState.GENERATION_IN_PROGRESS
            for _ in range(prompt_len):
                first_fits = manager.try_allocate_generation(first)
                second_fits = manager.try_allocate_generation(second)
                if not first_fits and not second_fits:
                    break
            else:
                pytest.fail("The unguarded control did not exhaust its GPU KV pool")
    finally:
        manager.shutdown()


def test_config_opt_in_reserves_decode_headroom_without_a_connector() -> None:
    """A generation server with no connector reserves decode headroom when configured to.

    The same pair of generation-init requests that main admits together is split: the second
    waits until the first frees its bound. A context request is never bounded, since only a
    connector prefill reserves its prompt.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    manager = _real_manager(
        KvCacheConfig(
            enable_block_reuse=False,
            host_cache_size=0,
            max_gpu_total_bytes=128 << 10,
            max_util_for_resume=1.0,
            enable_disagg_completion_reservation=True,
        ),
        kv_connector_manager=None,
    )
    try:
        assert manager._disagg_completion_pool_counts == {0: 1}
        total_blocks = manager.get_kv_cache_stats().max_num_blocks
        prompt_len = total_blocks // 3 * manager.tokens_per_block
        assert prompt_len >= 8
        first = _gen_init_request(1, prompt_len, prompt_len)
        second = _gen_init_request(2, prompt_len, prompt_len)

        assert manager.prepare_disagg_gen_init(first)
        assert 1 in manager._disagg_completion_limits
        assert not manager.prepare_disagg_gen_init(second)
        assert 2 not in manager.kv_cache_map
        assert 2 not in manager._disagg_completion_limits

        context = LlmRequest(
            request_id=3,
            max_new_tokens=1,
            input_tokens=[3] * prompt_len,
            sampling_config=SamplingConfig(1),
            is_streaming=False,
        )
        assert manager.reserve_context_completion(context)
        assert 3 not in manager._disagg_completion_limits

        manager.free_resources(first)
        assert manager.prepare_disagg_gen_init(second)
        manager.free_resources(second)
        assert not manager._disagg_completion_limits
        assert manager.get_kv_cache_stats().free_num_blocks == total_blocks
    finally:
        manager.shutdown()


def test_reverted_gen_init_releases_its_completion_bound() -> None:
    """A reverted gen-init frees its cache and bound, and is admitted again afterwards."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    manager = _real_manager()
    try:
        assert manager._disagg_completion_pool_counts == {0: 1}
        total_blocks = manager.get_kv_cache_stats().max_num_blocks
        prompt_len = total_blocks // 3 * manager.tokens_per_block
        assert prompt_len >= 8
        request = _gen_init_request(1, prompt_len, prompt_len)

        assert manager.prepare_disagg_gen_init(request)
        assert 1 in manager._disagg_completion_limits

        assert not manager.revert_allocate_context(request)
        assert 1 not in manager._disagg_completion_limits
        assert 1 not in manager.kv_cache_map
        assert manager.get_kv_cache_stats().free_num_blocks == total_blocks

        assert manager.prepare_disagg_gen_init(request)
        assert 1 in manager._disagg_completion_limits
        manager.free_resources(request)
        assert not manager._disagg_completion_limits
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "pool_counts, in_map, freed",
    [({0: 1}, True, True), ({}, True, False), ({0: 1}, False, False)],
    ids=["reservation_on", "reservation_off", "no_cache"],
)
def test_failed_init_release_follows_the_reservation(
    pool_counts: dict, in_map: bool, freed: bool
) -> None:
    manager = _gate_manager()
    manager._disagg_completion_pool_counts = pool_counts
    manager._disagg_completion_limits = {}
    manager.kv_cache_map = {5: Mock()} if in_map else {}
    manager.free_resources = Mock()
    request = SimpleNamespace(py_request_id=5)

    assert manager._release_failed_disagg_init(request) is freed
    assert manager.free_resources.called is freed
