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

Tests that build a manager without ``__init__`` assign both reservation dicts whenever they
assign either, so the class-level defaults are never written.
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
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)


class _ConnectorDouble:
    """The KV connector surface the manager touches on the paths these tests take.

    It never offers a prefix and holds no load, so freeing a request only resets state.
    ``declares`` is what the worker sets ``requires_completion_reservation`` to.
    """

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
        ([3], [8, 8], {1: 6, 2: 3}, 3, False),
        ([5], [8, 8], {1: 6, 2: 3}, 3, True),
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
def test_completion_reservation_skips_a_bound_whose_cache_is_gone() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager._disagg_completion_limits = {1: 100, 2: 3}
    manager._disagg_completion_pool_counts = {0: 1}
    manager.kv_cache_map = {2: Mock(num_blocks=0)}
    manager._get_storage_statistics = Mock(return_value=[SimpleNamespace(available=3)])

    assert manager._can_reserve_disagg_completion(2, 3)


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


def _context_reservation_manager(available: int, **connector_overrides) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager._ledger_tokens_per_block = 4
    manager._generation_kv_capacity_headroom = 1
    manager.enable_block_reuse = False
    manager.is_draft = False
    manager._disagg_completion_limits = {}
    manager._disagg_completion_pool_counts = {0: 1}
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
def test_context_reservation_records_a_bound_once_it_fits() -> None:
    # 16 prompt tokens + 1 base token in 4-token blocks: 5 blocks.
    manager = _context_reservation_manager(available=5)
    request = _first_chunk()

    assert manager.reserve_context_completion(request)
    assert manager._disagg_completion_limits == {7: 5}
    # A recorded bound is not checked again on later passes.
    manager._get_storage_statistics.return_value = [SimpleNamespace(available=0)]
    assert manager.reserve_context_completion(request)


@pytest.mark.cpu_only
def test_context_reservation_defers_without_recording() -> None:
    manager = _context_reservation_manager(available=4)

    assert not manager.reserve_context_completion(_first_chunk())
    assert manager._disagg_completion_limits == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "request_overrides, connector_overrides, has_pool_counts",
    [
        (dict(is_first_context_chunk=False), {}, True),
        ({}, {}, False),
        ({}, dict(prefix_reservations_enabled=True), True),
        ({}, dict(capacity_only=True), True),
        (dict(is_dummy=True), {}, True),
        (dict(is_generation_only_request=True), {}, True),
        (dict(is_disagg_generation_init_state=True), {}, True),
    ],
    ids=[
        "later_chunk",
        "eviction_tier",
        "source_reservations",
        "capacity_only",
        "dummy",
        "generation_only",
        "gen_init",
    ],
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
    """A first chunk that cannot grow is suspended and returns its bound with it."""
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
    """The release comes before every early return of ``revert_allocate_context``.

    A first chunk whose resize grew no capacity keeps its cache, but an unconsumed bound must
    still go back so that the next admission checks it again.
    """
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
    for name, value in fields.items():
        setattr(manager, name, value)
    return manager


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "overrides, applies",
    [
        ({}, True),
        # A window that covers max_seq_len behaves as full attention.
        (
            dict(kv_cache_manager_py_config=SimpleNamespace(layers=[_attention_layer(64)])),
            True,
        ),
        (dict(is_disagg=False), False),
        (dict(can_evict=True), False),
        # With the configuration off, only a serving connector that declares it turns it on.
        (dict(kv_connector_manager=None), False),
        (dict(kv_connector_manager=_ConnectorDouble(declares=False)), False),
        # Only an explicit True counts, not a truthy value such as a mock's attribute.
        (dict(kv_connector_manager=_ConnectorDouble(declares="yes")), False),
        (dict(kv_connector_manager=_ConnectorDouble(capacity_only=True)), False),
        (dict(is_draft=True), False),
        (dict(max_beam_width=2), False),
        (dict(mapping=SimpleNamespace(pp_size=2)), False),
        (
            dict(kv_cache_manager_py_config=SimpleNamespace(layers=[_attention_layer(16)])),
            False,
        ),
        (
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
        ),
        # The configuration opts in without a declaring connector, under the same conditions.
        (dict(kv_connector_manager=None, enable_disagg_completion_reservation=True), True),
        (
            dict(
                kv_connector_manager=_ConnectorDouble(declares=False),
                enable_disagg_completion_reservation=True,
            ),
            True,
        ),
        (
            dict(
                kv_connector_manager=None,
                enable_disagg_completion_reservation=True,
                is_disagg=False,
            ),
            False,
        ),
        (
            dict(
                kv_connector_manager=None,
                enable_disagg_completion_reservation=True,
                can_evict=True,
            ),
            False,
        ),
        (
            dict(
                kv_connector_manager=None,
                enable_disagg_completion_reservation=True,
                mapping=SimpleNamespace(pp_size=2),
            ),
            False,
        ),
        (
            dict(
                kv_connector_manager=None,
                enable_disagg_completion_reservation=True,
                kv_cache_manager_py_config=SimpleNamespace(layers=[_attention_layer(16)]),
            ),
            False,
        ),
    ],
    ids=[
        "full_attention",
        "window_covers_max_seq_len",
        "not_disagg",
        "eviction_tier",
        "no_connector",
        "undeclared_connector",
        "truthy_non_bool_declaration",
        "capacity_only_connector",
        "draft",
        "beam_search",
        "pipeline_parallel",
        "sliding_window",
        "recurrent_layer",
        "config_without_connector",
        "config_with_undeclared_connector",
        "config_not_disagg",
        "config_eviction_tier",
        "config_pipeline_parallel",
        "config_sliding_window",
    ],
)
def test_completion_reservation_gate(overrides: dict, applies: bool) -> None:
    assert _gate_manager(**overrides)._disagg_completion_reservation_applies() is applies


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
@pytest.mark.parametrize("enable_block_reuse", [False, True])
def test_disagg_generation_progress_under_real_gpu_page_pressure(
    reserve_growth: bool, enable_block_reuse: bool
) -> None:
    """Prompt-only admission can fill the pool before either request finishes.

    Use the real allocator and admission path; the control disables only the reservation.
    The guarded path must defer the second long prompt, admit a short request alongside the
    first, and release reservations on free.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    manager = _real_manager(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
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


_GATE_DEFAULT_CONFIG = dict(
    enable_block_reuse=False,
    host_cache_size=0,
    max_gpu_total_bytes=128 << 10,
    max_util_for_resume=1.0,
)


@pytest.mark.parametrize(
    "config_overrides, manager_overrides, expected",
    [
        ({}, {}, {0: 1}),
        ({}, dict(is_disagg=False), {}),
        # An explicit host tier; the automatic one depends on the host's memory.
        (dict(host_cache_size=16 << 20), {}, {}),
        # Without either opt-in, a server with no tier below GPU keeps main's admission.
        ({}, dict(kv_connector_manager=None), {}),
        ({}, dict(kv_connector_manager=_ConnectorDouble(declares=False)), {}),
        ({}, dict(kv_connector_manager=_ConnectorDouble(capacity_only=True)), {}),
        ({}, dict(max_beam_width=2), {}),
        ({}, dict(is_draft=True), {}),
        (dict(max_attention_window=[16]), {}, {}),
        # The configuration opts a server with no connector in, under the same conditions.
        (dict(enable_disagg_completion_reservation=True), dict(kv_connector_manager=None), {0: 1}),
        (
            dict(enable_disagg_completion_reservation=True),
            dict(kv_connector_manager=None, is_disagg=False),
            {},
        ),
        (
            dict(enable_disagg_completion_reservation=True, host_cache_size=16 << 20),
            dict(kv_connector_manager=None),
            {},
        ),
        (
            dict(enable_disagg_completion_reservation=True),
            dict(kv_connector_manager=None, is_draft=True),
            {},
        ),
        (
            dict(enable_disagg_completion_reservation=True, host_cache_size=16 << 20),
            dict(kv_connector_manager=None, is_estimating_kv_cache=True),
            {},
        ),
    ],
    ids=[
        "baseline",
        "not_disagg",
        "host_tier",
        "no_connector",
        "undeclared_connector",
        "capacity_only_connector",
        "beam_search",
        "draft",
        "sliding_window",
        "config_without_connector",
        "config_not_disagg",
        "config_host_tier",
        "config_draft",
        "config_estimating",
    ],
)
def test_completion_reservation_gate_on_a_real_manager(
    config_overrides: dict,
    manager_overrides: dict,
    expected: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor turns the reservation on only where the gate allows it.

    A target manager whose configuration asks for the reservation where it cannot apply says so;
    a draft manager, which never reserves, stays quiet, and so does the estimation manager, which
    is built without the connector. Pipeline parallelism needs one process per rank;
    ``test_completion_reservation_gate`` covers that clause.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    warned: list[str] = []
    monkeypatch.setattr(logger, "warning", lambda *msg: warned.append(" ".join(map(str, msg))))
    config = dict(_GATE_DEFAULT_CONFIG)
    config.update(config_overrides)
    manager = _real_manager(KvCacheConfig(**config), **manager_overrides)
    try:
        assert manager._disagg_completion_pool_counts == expected
        assert manager._disagg_completion_limits == {}
        ignored = (
            config.get("enable_disagg_completion_reservation", False)
            and not expected
            and not manager_overrides.get("is_draft", False)
            and not manager_overrides.get("is_estimating_kv_cache", False)
        )
        assert any("enable_disagg_completion_reservation" in msg for msg in warned) is ignored
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "connector",
    [None, _ConnectorDouble(declares=False)],
    ids=["no_connector", "undeclared_connector"],
)
def test_admission_is_unchanged_without_the_opt_in(connector: _ConnectorDouble | None) -> None:
    """A disaggregated server with no tier below GPU keeps main's admission unless asked.

    Two generation-init requests whose completion bounds together exceed the pool are both
    admitted, and a first context chunk is neither deferred nor bounded.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    manager = _real_manager(kv_connector_manager=connector)
    try:
        assert not manager.can_evict
        assert manager._disagg_completion_pool_counts == {}
        total_blocks = manager.get_kv_cache_stats().max_num_blocks
        prompt_len = total_blocks // 3 * manager.tokens_per_block
        assert prompt_len >= 8
        first = _gen_init_request(1, prompt_len, prompt_len)
        second = _gen_init_request(2, prompt_len, prompt_len)
        # Each bound covers twice the prompt, so the two cannot both fit the pool.
        assert 2 * manager._disagg_generation_limit(first, prompt_len) > total_blocks

        assert manager.prepare_disagg_gen_init(first)
        assert manager.prepare_disagg_gen_init(second)
        assert manager._disagg_completion_limits == {}

        context = LlmRequest(
            request_id=3,
            max_new_tokens=1,
            input_tokens=[3] * prompt_len,
            sampling_config=SamplingConfig(1),
            is_streaming=False,
        )
        assert manager.reserve_context_completion(context)
        assert manager._disagg_completion_limits == {}

        manager.free_resources(first)
        manager.free_resources(second)
        assert manager.get_kv_cache_stats().free_num_blocks == total_blocks
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
        KvCacheConfig(**_GATE_DEFAULT_CONFIG, enable_disagg_completion_reservation=True),
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
    """A generation-init admission that the coordinator reverts leaves no bound behind.

    This mirrors DisaggCoordinator.revert_deferred_gen_init, which reaches
    revert_allocate_context through PyExecutor._revert_ctx_alloc. A fresh gen-init holds
    history past its pre-resize capacity of zero, so revert frees its cache outright; the
    test pins that no bound leaks and that the request is admitted again afterwards.
    """
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
