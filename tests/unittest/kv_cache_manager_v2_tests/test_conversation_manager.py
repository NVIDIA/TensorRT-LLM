# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversation drop-plan ownership and completion ordering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import ConversationManager

pytestmark = pytest.mark.cpu_only


def _request(
    request_id: int,
    conversation_id: str = "DYNAMO_SESSION_ID",
    *,
    cache_salt: str | None = None,
    lora_task_id: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        is_dummy_request=False,
        py_conversation_params=SimpleNamespace(conversation_id=conversation_id),
        cache_salt=cache_salt,
        lora_task_id=lora_task_id,
    )


def _cache(tokens: tuple[int, ...] = (1, 2, 3, 4)) -> MagicMock:
    cache = MagicMock()
    cache.committed_tokens = list(tokens)
    cache.tokens_per_block = 4
    cache.reuse_scope = SimpleNamespace(lora_id=None, salt=None)
    return cache


@pytest.mark.parametrize("completion", ["finish", "empty_plan", "clear"])
def test_overlapping_completion_after_owner_cleanup(completion: str) -> None:
    manager = ConversationManager(max_num_turns=1)
    owner, overlapping = _request(1), _request(2)
    manager.prepare_request(owner)
    manager.prepare_request(overlapping)
    cache = _cache()
    cache.plan_committed_block_drop.return_value = None

    if completion == "finish":
        manager.finish_request(owner)
    elif completion == "empty_plan":
        manager.save_drop_plan(owner, cache)
    else:
        manager.clear()

    cache.reset_mock()
    manager.save_drop_plan(overlapping, cache)
    cache.plan_committed_block_drop.assert_not_called()
    assert not manager._conversation_states


def test_overlapping_completion_preserves_new_owner() -> None:
    manager = ConversationManager(max_num_turns=1)
    owner, overlapping, successor = _request(1), _request(2), _request(3)
    manager.prepare_request(owner)
    manager.prepare_request(overlapping)
    manager.finish_request(owner)
    manager.prepare_request(successor)
    cache = _cache()

    manager.save_drop_plan(overlapping, cache)
    manager.finish_request(overlapping)
    cache.plan_committed_block_drop.assert_not_called()

    manager.save_drop_plan(successor, cache)
    cache.plan_committed_block_drop.assert_called_once_with()
    state = manager._conversation_states["DYNAMO_SESSION_ID", None, None]
    assert state.current_request_id is None
    assert list(state.planned_drop_handles) == [cache.plan_committed_block_drop.return_value]


def test_repeated_completion_does_not_plan_again() -> None:
    manager = ConversationManager(max_num_turns=1)
    request = _request(1)
    manager.prepare_request(request)
    cache = _cache()
    cache.plan_committed_block_drop.return_value = None

    manager.save_drop_plan(request, cache)
    manager.save_drop_plan(request, cache)

    cache.plan_committed_block_drop.assert_called_once_with()
    assert not manager._conversation_states


def test_overlapping_completion_preserves_saved_plan() -> None:
    manager = ConversationManager(max_num_turns=1)
    owner, overlapping, successor = _request(1), _request(2), _request(3)
    manager.prepare_request(owner)
    manager.prepare_request(overlapping)
    cache = _cache()
    first_plan, next_plan = MagicMock(), MagicMock()
    cache.plan_committed_block_drop.side_effect = [first_plan, next_plan]

    manager.save_drop_plan(owner, cache)
    manager.save_drop_plan(overlapping, cache)
    cache.plan_committed_block_drop.assert_called_once_with()
    first_plan.drop.assert_not_called()

    manager.prepare_request(successor)
    manager.save_drop_plan(successor, cache)
    first_plan.drop.assert_called_once_with()
    next_plan.drop.assert_not_called()
    state = manager._conversation_states["DYNAMO_SESSION_ID", None, None]
    assert list(state.planned_drop_handles) == [next_plan]


def test_overlapping_completion_cannot_plan_for_active_owner() -> None:
    manager = ConversationManager(max_num_turns=1)
    owner, overlapping = _request(1), _request(2)
    manager.prepare_request(owner)
    manager.prepare_request(overlapping)
    cache = _cache()

    manager.save_drop_plan(overlapping, cache)
    manager.finish_request(overlapping)

    cache.plan_committed_block_drop.assert_not_called()
    assert manager._conversation_states["DYNAMO_SESSION_ID", None, None].current_request_id == 1


@pytest.mark.parametrize(
    "first_scope, second_scope",
    [
        ((None, "tenant-A"), (None, "tenant-B")),
        ((None, None), (None, "")),
        ((None, "tenant-中文"), (None, "tenant-中文\x00other")),
        ((1, "same-salt"), (2, "same-salt")),
        ((None, "same-salt"), (0, "same-salt")),
    ],
)
def test_same_session_isolates_active_requests_and_plans_by_reuse_scope(
    first_scope: tuple[int | None, str | None], second_scope: tuple[int | None, str | None]
) -> None:
    manager = ConversationManager(max_num_turns=1)
    first = _request(1, lora_task_id=first_scope[0], cache_salt=first_scope[1])
    second = _request(2, lora_task_id=second_scope[0], cache_salt=second_scope[1])
    successor = _request(3, lora_task_id=second_scope[0], cache_salt=second_scope[1])
    first_cache, second_cache, successor_cache = _cache(), _cache(), _cache()
    manager.prepare_request(first)
    manager.prepare_request(second)

    manager.save_drop_plan(first, first_cache)
    manager.save_drop_plan(second, second_cache)

    first_cache.plan_committed_block_drop.assert_called_once_with()
    second_cache.plan_committed_block_drop.assert_called_once_with()
    first_cache.plan_committed_block_drop.return_value.drop.assert_not_called()
    second_cache.plan_committed_block_drop.return_value.drop.assert_not_called()

    manager.prepare_request(successor)
    manager.save_drop_plan(successor, successor_cache)
    second_cache.plan_committed_block_drop.return_value.drop.assert_called_once_with()
    first_cache.plan_committed_block_drop.return_value.drop.assert_not_called()


@pytest.mark.parametrize(
    "next_tokens",
    [
        (10, 11, 12, 13, 14, 15, 16, 17),
        (1, 2, 3, 4, 50, 60, 70, 80),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5, 60, 70, 80),
    ],
)
def test_unrelated_prefix_cannot_advance_conversation_retention(
    next_tokens: tuple[int, ...],
) -> None:
    manager = ConversationManager(max_num_turns=1)
    first, unrelated = _request(1), _request(2)
    first_cache = _cache((1, 2, 3, 4, 5, 6))
    unrelated_cache = _cache(next_tokens)
    manager.prepare_request(first)
    manager.save_drop_plan(first, first_cache)
    manager.prepare_request(unrelated)
    manager.save_drop_plan(unrelated, unrelated_cache)

    unrelated_cache.plan_committed_block_drop.assert_not_called()
    first_cache.plan_committed_block_drop.return_value.drop.assert_not_called()
    state = manager._conversation_states["DYNAMO_SESSION_ID", None, None]
    assert state.current_request_id is None
    assert list(state.planned_drop_handles) == [first_cache.plan_committed_block_drop.return_value]


def test_extending_partial_block_prefix_advances_retention() -> None:
    manager = ConversationManager(max_num_turns=1)
    first, continuation = _request(1), _request(2)
    first_cache = _cache((1, 2, 3, 4, 5, 6))
    continuation_cache = _cache((1, 2, 3, 4, 5, 6, 7, 8, 9))
    manager.prepare_request(first)
    manager.save_drop_plan(first, first_cache)
    manager.prepare_request(continuation)
    manager.save_drop_plan(continuation, continuation_cache)

    continuation_cache.plan_committed_block_drop.assert_called_once_with()
    first_cache.plan_committed_block_drop.return_value.drop.assert_called_once_with()
