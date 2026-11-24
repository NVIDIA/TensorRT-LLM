# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA cache manager's speculative state (``K31HybridCacheManager._initialize_spec_state``). CPU only.

The fused verify kernel replays at most ``MAX_NUM_SPEC`` draft tokens per step, so a speculative config with more
drafts is refused when the cache manager is built, before any pool is allocated.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import kda_cache_manager
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def max_num_spec() -> int:
    """The verify kernel's draft capacity; its module imports the CuTe DSL."""
    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("the KDA verify kernel module imports the CuTe DSL")
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.kda_mtp_decode_v2 import (
        MAX_NUM_SPEC,
    )

    return MAX_NUM_SPEC


@pytest.fixture
def replay_states(monkeypatch) -> list:
    """Record the draft count of every replay state built, without allocating one."""
    built = []

    class _ReplayState:
        def __init__(self, num_spec):
            built.append(num_spec)

        def validate(self, layout):
            pass

    monkeypatch.setattr(kda_cache_manager, "K31KDAReplayState", _ReplayState)
    return built


def _manager(tokens_per_gen_step=None, requested_num_spec=None, context_kv_only=False):
    """A manager with only the fields ``_initialize_spec_state`` reads."""
    manager = kda_cache_manager.K31HybridCacheManager.__new__(
        kda_cache_manager.K31HybridCacheManager
    )
    manager._requested_num_spec = requested_num_spec
    manager.spec_config = (
        None
        if tokens_per_gen_step is None
        else SimpleNamespace(tokens_per_gen_step=tokens_per_gen_step)
    )
    manager.kda_context_only = False
    manager._context_kv_only = context_kv_only
    manager._state_layout = None
    return manager


def test_verify_kernel_capacity(max_num_spec) -> None:
    assert max_num_spec == 7


@pytest.mark.parametrize("source", ["spec_config", "requested"])
@pytest.mark.parametrize("context_kv_only", [False, True], ids=["server", "context_kv_only_server"])
def test_more_drafts_than_the_verify_replays_are_refused(
    max_num_spec, replay_states, source, context_kv_only
) -> None:
    num_spec = max_num_spec + 1
    if source == "spec_config":
        manager = _manager(tokens_per_gen_step=num_spec + 1, context_kv_only=context_kv_only)
    else:
        manager = _manager(requested_num_spec=num_spec, context_kv_only=context_kv_only)
    with pytest.raises(
        ValueError, match=f"at most {max_num_spec} draft tokens per step, got {num_spec}"
    ):
        manager._initialize_spec_state()
    assert replay_states == []


def test_verify_capacity_builds_the_replay_state(max_num_spec, replay_states) -> None:
    manager = _manager(tokens_per_gen_step=max_num_spec + 1)
    assert manager._initialize_spec_state() is manager._kda_replay
    assert replay_states == [max_num_spec]
    assert manager.kda_context_only is False


def test_context_kv_only_server_has_no_replay_state(max_num_spec, replay_states) -> None:
    manager = _manager(tokens_per_gen_step=max_num_spec + 1, context_kv_only=True)
    assert manager._initialize_spec_state() is None
    assert manager.kda_context_only is True
    assert replay_states == []


def test_no_speculative_decoding_has_no_replay_state(replay_states) -> None:
    manager = _manager()
    manager.kda_context_only = True
    assert manager._initialize_spec_state() is None
    assert manager.kda_context_only is False
    assert replay_states == []
