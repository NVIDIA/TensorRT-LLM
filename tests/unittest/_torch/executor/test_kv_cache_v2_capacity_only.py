# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor._util import _get_num_reserved_index_slots
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


@pytest.mark.parametrize(
    ("spec_config", "expected"),
    [
        (None, 1),
        (SimpleNamespace(enable_confidence_scheduling=False), 1),
        (SimpleNamespace(enable_confidence_scheduling=True), 2),
    ],
    ids=["no-speculation", "static-dspark", "confidence-dspark"],
)
def test_index_mapper_reservation_tracks_confidence_scheduling(
    spec_config: SimpleNamespace | None, expected: int
) -> None:
    assert _get_num_reserved_index_slots(spec_config) == expected


def _manager(
    *,
    is_draft: bool,
    kv_compression_manages_history: bool = False,
    kv_reserve_draft_tokens: int = 0,
) -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = is_draft
    manager._has_cp_helix = False
    manager.kv_compression_manages_history = kv_compression_manages_history
    manager._kv_reserve_draft_tokens = kv_reserve_draft_tokens
    manager.kv_cache_map = {}
    return manager


def _request(
    request_id: int,
    *,
    rewind: int = 0,
    accepted_draft_tokens: int = 0,
    draft_tokens: list[int] | None = None,
    verify_len: int | None = None,
    complete: bool = False,
) -> SimpleNamespace:
    request = SimpleNamespace(
        py_request_id=request_id,
        py_rewind_len=rewind,
        py_num_accepted_draft_tokens=accepted_draft_tokens,
        py_draft_tokens=draft_tokens,
        max_beam_num_tokens=201,
        state=LlmRequestState.GENERATION_COMPLETE
        if complete
        else LlmRequestState.GENERATION_IN_PROGRESS,
    )
    if verify_len is not None:
        request.py_verify_len = verify_len
    return request


def _cache(*, capacity: int = 256, active: bool = True) -> MagicMock:
    cache = MagicMock()
    cache.capacity = capacity
    cache.is_active = active
    cache.resize.return_value = True
    return cache


@pytest.fixture(autouse=True)
def _disable_draft_token_relocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kv_cache_v2_module, "_update_kv_cache_draft_token_location", MagicMock())


def test_manager_initializes_capacity_only_policy_to_false() -> None:
    class StopInitialization(RuntimeError):
        pass

    class StopAfterPolicyConfig:
        @property
        def enable_swa_scratch_reuse(self):
            raise StopInitialization

    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    mapping = SimpleNamespace(cp_config={})

    with (
        patch.object(kv_cache_v2_module, "get_pp_layers", return_value=([0], 1)),
        pytest.raises(StopInitialization),
    ):
        manager.__init__(
            StopAfterPolicyConfig(),
            kv_cache_v2_module.CacheTypeCpp.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=128,
            tokens_per_block=64,
            max_seq_len=256,
            max_batch_size=1,
            mapping=mapping,
        )

    assert manager.kv_compression_manages_history is False


def test_default_generation_resize_updates_capacity_and_history() -> None:
    manager = _manager(is_draft=False)
    request = _request(1, rewind=3)
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(253, 200)


def test_capacity_only_is_scoped_to_target_manager() -> None:
    request = _request(1, rewind=3)
    batch = SimpleNamespace(generation_requests=[request])
    target = _manager(is_draft=False, kv_compression_manages_history=True)
    draft = _manager(is_draft=True)
    target_cache = _cache()
    draft_cache = _cache()
    target.kv_cache_map[request.py_request_id] = target_cache
    draft.kv_cache_map[request.py_request_id] = draft_cache

    draft.update_resources(batch)
    target.update_resources(batch)

    draft_cache.resize.assert_called_once_with(253, 200)
    target_cache.resize.assert_called_once_with(253, None)


@pytest.mark.parametrize(
    ("is_draft", "expected_capacity"),
    [(True, 201), (False, 230)],
    ids=["draft-reclaims-reserve", "target-has-no-reserve"],
)
def test_dynamic_tree_reserved_capacity(is_draft: bool, expected_capacity: int) -> None:
    manager = _manager(is_draft=is_draft, kv_reserve_draft_tokens=60)
    # The runtime tree used 31 draft positions: 26 rejected and 5 accepted.
    request = _request(1, rewind=26, accepted_draft_tokens=5)
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(expected_capacity, 200)


@pytest.mark.parametrize(
    ("verify_len", "rewind", "accepted", "expected_capacity"),
    [(2, 1, 1, 252), (5, 2, 3, 254)],
    ids=["reclaims-unverified-suffix", "full-window-keeps-uniform-accounting"],
)
def test_target_reclaims_only_ragged_unverified_capacity(
    verify_len: int, rewind: int, accepted: int, expected_capacity: int
) -> None:
    manager = _manager(is_draft=False, kv_reserve_draft_tokens=5)
    request = _request(
        1,
        rewind=rewind,
        accepted_draft_tokens=accepted,
        draft_tokens=[1] * 5,
        verify_len=verify_len,
    )
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(expected_capacity, 200)


def test_capacity_only_completion_preserves_history() -> None:
    manager = _manager(is_draft=False, kv_compression_manages_history=True)
    request = _request(1, complete=True)
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(None, None)


def test_capacity_only_skips_suspended_cache() -> None:
    manager = _manager(is_draft=False, kv_compression_manages_history=True)
    request = _request(1, rewind=3)
    cache = _cache(active=False)
    manager.kv_cache_map[request.py_request_id] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_not_called()


def test_generation_update_has_no_request_compaction_marker() -> None:
    manager = _manager(is_draft=False, kv_compression_manages_history=True)
    request = _request(1, rewind=3)
    cache = _cache()
    manager.kv_cache_map[request.py_request_id] = cache

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    assert "py_kv_cache_kv_compression_manages_history" not in vars(request)
    assert "py_kv_cache_compaction" not in vars(request)


def test_llm_request_has_no_compression_consumer_marker() -> None:
    request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=[1],
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )

    assert "py_kv_cache_kv_compression_manages_history" not in vars(request)
    assert "py_kv_cache_compaction" not in vars(request)


def test_disagg_gen_transition_reserves_target_drafts_without_context_drafts():
    manager = _manager(is_draft=False)
    manager.max_total_draft_tokens = 4
    request = SimpleNamespace(
        py_draft_tokens=[],
        is_disagg_generation_transmission_complete=True,
        context_phase_params=SimpleNamespace(draft_tokens=None),
        py_disable_speculative_decoding=False,
    )

    assert manager._effective_draft_len(request) == 4
    assert manager._required_gen_capacity(request, 128) == 133


def test_disagg_gen_transition_does_not_reserve_disabled_speculation():
    manager = _manager(is_draft=False)
    manager.max_total_draft_tokens = 4
    request = SimpleNamespace(
        py_draft_tokens=[],
        is_disagg_generation_transmission_complete=True,
        context_phase_params=SimpleNamespace(draft_tokens=None),
        py_disable_speculative_decoding=True,
    )

    assert manager._effective_draft_len(request) == 0


def test_disagg_gen_transition_prefers_context_drafts():
    manager = _manager(is_draft=False)
    manager.max_total_draft_tokens = 4
    request = SimpleNamespace(
        py_draft_tokens=[],
        is_disagg_generation_transmission_complete=True,
        context_phase_params=SimpleNamespace(draft_tokens=[1, 2]),
        py_disable_speculative_decoding=False,
    )

    assert manager._effective_draft_len(request) == 2
