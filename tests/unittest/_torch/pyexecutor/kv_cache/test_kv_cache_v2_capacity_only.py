# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


def _manager(
    *,
    is_draft: bool,
    kv_compression_manages_history: bool = False,
    kv_reserve_draft_tokens: int = 0,
) -> KVCacheManagerV2:
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = False
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager.kv_cache_type = CacheType.SELF
    manager.is_draft = is_draft
    manager.max_beam_width = 1
    manager._has_cp_helix = False
    manager.kv_compression_manages_history = kv_compression_manages_history
    manager._kv_reserve_draft_tokens = kv_reserve_draft_tokens
    manager._allocated_draft_lens = {}
    manager._inflight_draft_lens = {}
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


@pytest.mark.parametrize("complete", [False, True], ids=["in_progress", "finished"])
def test_generation_commits_stable_tokens_after_relocation_and_resize(
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    calls = []
    manager = _manager(is_draft=False)
    manager.enable_block_reuse = True
    manager.kv_connector_manager = None
    request = _request(1, rewind=2, accepted_draft_tokens=2, complete=complete)
    request.py_num_accepted_draft_tokens_indices = [0, 3]
    request.py_rewind_draft_token_separate_adjustment = 0
    request.is_dummy_request = False
    request.is_context_only_request = False
    request.prompt_len = 100
    request.get_tokens_view = lambda beam_id: list(range(201))
    batch = SimpleNamespace(generation_requests=[request])
    attn_metadata = object()
    cache = _cache()
    cache.num_committed_tokens = 197
    cache.history_length = 197

    def resize(capacity: int | None, history_length: int) -> bool:
        calls.append(("resize", (capacity, history_length)))
        cache.history_length = history_length
        return True

    def commit(tokens: list[int], is_end: bool = False) -> None:
        assert cache.history_length == 200
        calls.append(("commit", list(tokens), is_end))

    cache.resize.side_effect = resize
    cache.commit.side_effect = commit
    manager.kv_cache_map[request.py_request_id] = cache

    relocate = MagicMock(
        side_effect=lambda *args, **kwargs: calls.append(
            ("relocate", kwargs["include_finished_requests"])
        )
    )
    monkeypatch.setattr(kv_cache_v2_module, "_update_kv_cache_draft_token_location", relocate)

    manager.update_resources(batch, attn_metadata, 2.0)

    assert calls == [
        ("relocate", True),
        ("resize", (None if complete else 254, 200)),
        ("commit", [197, 198, 199], complete),
    ]


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


@pytest.mark.cpu_only
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

    # The reservation belongs to the completed step; live request fields may
    # already describe the next overlap iteration.
    manager._allocated_draft_lens[request.py_request_id] = 5
    request.py_verify_len = 1
    request.py_draft_tokens = [9]
    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(expected_capacity, 200)
    assert request.py_request_id not in manager._allocated_draft_lens


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


@pytest.mark.parametrize("ctx_draft_len, expected", [(2, 4), (4, 4), (5, 5)])
def test_disagg_gen_transition_reserves_padded_context_drafts(ctx_draft_len, expected):
    # The GEN pads a CTX draft shorter than its own width (e.g. a CTX with
    # max_draft_len 1) up to max_total_draft_tokens before its first forward.
    manager = _manager(is_draft=False)
    manager.max_total_draft_tokens = 4
    request = SimpleNamespace(
        py_draft_tokens=[],
        is_disagg_generation_transmission_complete=True,
        context_phase_params=SimpleNamespace(draft_tokens=list(range(ctx_draft_len))),
        py_disable_speculative_decoding=False,
    )

    assert manager._effective_draft_len(request) == expected


def _generation_manager(draft_len: int, capacity: int = 100) -> tuple[KVCacheManagerV2, MagicMock]:
    manager = _manager(is_draft=False)
    manager.kv_cache_type = CacheType.SELF
    manager.max_beam_width = 1
    manager._effective_draft_len = MagicMock(return_value=draft_len)
    manager._fresh_page_fill = None
    cache = _cache(capacity=capacity)
    cache.beam_width = 1

    def resize(new_capacity, history_length=None):
        if new_capacity is not None:
            cache.capacity = new_capacity
        return True

    cache.resize.side_effect = resize
    return manager, cache


def _generation_request(request_id: int, *, committed: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        py_beam_width=1,
        is_dummy_request=False,
        py_rewind_len=0,
        py_num_accepted_draft_tokens=0,
        max_beam_num_tokens=committed,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
    )


def _finish_step(request: SimpleNamespace, *, draft_len: int, accepted: int) -> None:
    """Record a verification step's outcome the way the sampler does."""
    request.py_num_accepted_draft_tokens = accepted
    request.py_rewind_len = draft_len - accepted
    request.max_beam_num_tokens += 1 + accepted


def _drive(widths: list[int], accepted: list[int], *, overlap: bool) -> None:
    """Allocate and update in executor order and check every in-flight step.

    ``tokens`` is the device-side token count (golden token included, so KV
    exists for ``tokens - 1`` positions). A step with draft width ``d`` writes
    KV for the golden token and ``d`` drafts, so it needs capacity
    ``tokens + d`` from its launch until it is reconciled. Under the overlap
    scheduler step N+1 is allocated and launched before step N's update, so
    step N's rewind must leave step N+1's pages in place.
    """
    manager, cache = _generation_manager(draft_len=widths[0])
    request = _generation_request(3, committed=cache.capacity + 1)
    manager.kv_cache_map[request.py_request_id] = cache
    batch = SimpleNamespace(generation_requests=[request])
    tokens = request.max_beam_num_tokens

    def allocate(step: int) -> None:
        manager._effective_draft_len.return_value = widths[step]
        assert manager.try_allocate_generation(request)

    if overlap:
        allocate(0)
    for step, (width, step_accepted) in enumerate(zip(widths, accepted)):
        if not overlap:
            allocate(step)
        assert cache.capacity >= tokens + width, f"step {step} launched short"
        tokens += 1 + step_accepted
        has_next = overlap and step + 1 < len(widths)
        if has_next:
            # The scheduler allocates step N+1 before step N's update runs.
            allocate(step + 1)
            assert cache.capacity >= tokens + widths[step + 1], f"step {step + 1} launched short"
        _finish_step(request, draft_len=width, accepted=step_accepted)
        manager.update_resources(batch)
        if has_next:
            assert cache.capacity >= tokens + widths[step + 1], (
                f"step {step} update freed pages step {step + 1} is writing"
            )
        else:
            assert cache.capacity == tokens - 1
    # Every allocation was reconciled exactly once: nothing leaks.
    assert cache.capacity == tokens - 1
    assert manager._inflight_draft_lens == {}


_WIDTH_SCHEDULES = {
    "constant": [3, 3, 3, 3, 3, 3],
    "growing": [1, 1, 3, 3, 3, 3],
    "shrinking": [3, 3, 1, 1, 0, 0],
    "mixed": [0, 3, 1, 3, 2, 3],
}


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("schedule", list(_WIDTH_SCHEDULES))
@pytest.mark.parametrize("acceptance", ["none", "all", "alternating"])
def test_generation_rewind_uses_each_steps_own_draft_width(
    overlap: bool, schedule: str, acceptance: str
) -> None:
    widths = _WIDTH_SCHEDULES[schedule]
    accepted = {
        "none": [0] * len(widths),
        "all": list(widths),
        "alternating": [width if step % 2 else 0 for step, width in enumerate(widths)],
    }[acceptance]
    _drive(widths, accepted, overlap=overlap)


def test_overlap_update_reconciles_the_older_allocation() -> None:
    manager, cache = _generation_manager(draft_len=1)
    request = _generation_request(4, committed=101)
    manager.kv_cache_map[request.py_request_id] = cache

    assert manager.try_allocate_generation(request)
    manager._effective_draft_len.return_value = 3
    assert manager.try_allocate_generation(request)
    # Step N: 1 base + 1 draft; step N+1: 1 base + 3 drafts.
    assert cache.capacity == 106
    assert manager._inflight_draft_lens == {4: [1, 3]}

    # Step N accepted its draft: nothing of step N is rewound, and step N+1's
    # four slots stay allocated.
    _finish_step(request, draft_len=1, accepted=1)
    manager.update_resources(SimpleNamespace(generation_requests=[request]))
    assert cache.capacity == 106
    assert manager._inflight_draft_lens == {4: [3]}

    # Step N+1 accepted none of its three drafts.
    _finish_step(request, draft_len=3, accepted=0)
    manager.update_resources(SimpleNamespace(generation_requests=[request]))
    assert cache.capacity == 103
    assert manager._inflight_draft_lens == {}


def test_overlap_revert_drops_only_the_newest_allocation() -> None:
    manager, cache = _generation_manager(draft_len=2)
    request = _generation_request(5, committed=101)
    manager.kv_cache_map[request.py_request_id] = cache

    assert manager.try_allocate_generation(request)
    manager._effective_draft_len.return_value = 4
    assert manager.try_allocate_generation(request)
    assert cache.capacity == 108

    # Step N+1 is skipped (can_queue=False) before step N's update.
    manager.revert_allocate_generation(request)
    assert cache.capacity == 103
    assert manager._inflight_draft_lens == {5: [2]}

    _finish_step(request, draft_len=2, accepted=0)
    manager.update_resources(SimpleNamespace(generation_requests=[request]))
    assert cache.capacity == 101
    assert manager._inflight_draft_lens == {}


def test_overlap_cuda_graph_padding_widens_only_the_newest_allocation() -> None:
    manager, cache = _generation_manager(draft_len=1)
    request = _generation_request(6, committed=101)
    manager.kv_cache_map[request.py_request_id] = cache

    assert manager.try_allocate_generation(request)
    assert manager.try_allocate_generation(request)
    assert cache.capacity == 104

    # CUDA-graph padding restores step N+1's draft length to the static 3.
    request.py_draft_tokens = [0, 0, 0]
    manager.extend_capacity_for_tokens(request)
    assert cache.capacity == 106
    assert manager._inflight_draft_lens == {6: [1, 3]}

    # Step N rejected its draft: rewind its one slot, keep step N+1's four.
    _finish_step(request, draft_len=1, accepted=0)
    manager.update_resources(SimpleNamespace(generation_requests=[request]))
    assert cache.capacity == 105
    assert manager._inflight_draft_lens == {6: [3]}


def test_update_of_a_suspended_cache_still_consumes_its_allocation() -> None:
    manager, cache = _generation_manager(draft_len=2)
    request = _generation_request(7, committed=101)
    manager.kv_cache_map[request.py_request_id] = cache

    assert manager.try_allocate_generation(request)
    # The next iteration's scheduler suspended the cache before this update.
    cache.is_active = False
    _finish_step(request, draft_len=2, accepted=1)
    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    assert cache.capacity == 103
    assert manager._inflight_draft_lens == {}
