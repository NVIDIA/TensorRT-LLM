# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 keeps its KDA replay caches in the recurrent-state slot (``ops/ssm/kda_cache_manager``). CPU only.

The fused verify / decode kernels address the recurrent state and the replay caches with one state index, so the
replay caches are buffers of each KDA layer's recurrent layer config, laid out as the replay layout lists them with the
accepted-draft count on the first local KDA layer. No replay region is declared.

The manager's speculative state (``K31HybridCacheManager._initialize_spec_state``): a generation server builds the
replay state for the verify kernel's ``MAX_NUM_SPEC`` drafts; a disaggregated context server keeps none.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm import kda_cache_manager
from tensorrt_llm._torch.modules.kimi_kda.replay_region import KDAReplayRegion
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaStateLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole

pytestmark = pytest.mark.cpu_only

NUM_SPEC = 4
HEADS, HEAD_DIM, CONV_WINDOW = 2, 128, 3
COUNT_ROLE = DataRole("prev_num_accepted_tokens")
#: Two local KDA layers (global ids 0 and 2) at manager layers 0 and 1.
KDA_LAYERS = (0, 2)
LAYER_OFFSETS = {0: 0, 1: 1, 2: 2}


def _context() -> MambaStateLayout:
    dim = HEADS * HEAD_DIM
    return MambaStateLayout(
        layer_mask=(True, False, True),
        pp_layers=(0, 1, 2),
        mamba_pp_layers=KDA_LAYERS,
        mapping=Mapping(),
        conv_state_shape=(3 * dim, CONV_WINDOW),
        ssm_state_shape=(HEADS, HEAD_DIM, HEAD_DIM),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        n_groups_per_rank=1,
        max_batch_size=4,
        state_index_capacity=8,
        slot_capacity=None,
        spec_config=SimpleNamespace(tokens_per_gen_step=NUM_SPEC + 1),
        conv_section_dims=(dim, dim, dim),
        conv_state_layout="q_k_v",
    )


def _region(state) -> KDAReplayRegion:
    context = _context()
    state.validate(context)
    return KDAReplayRegion(
        state, context, LAYER_OFFSETS, max_living_caches=8, max_util_for_resume=0.8
    )


def _expected_buffers(state, first_layer: bool) -> list[tuple[str, int]]:
    """The recurrent-config buffers K3.1 was validated with: the replay layout in its order, then the count."""
    buffers = [
        (name, dtype.itemsize * math.prod(shape))
        for name, (dtype, shape) in state.pool_layout(_context()).items()
    ]
    if first_layer:
        buffers.append((str(COUNT_ROLE), 16))
    return buffers


def _as_pairs(buffers: list[BufferConfig]) -> list[tuple[str, int]]:
    return [(str(buffer.role), buffer.size) for buffer in buffers]


def test_the_replay_caches_share_the_state_slot() -> None:
    state = kda_cache_manager.K31KDAReplayState(NUM_SPEC)
    assert not state.draft_records
    region = _region(state)
    assert region.region is None
    assert tuple(region.declare_regions()) == ()
    assert region.bytes_per_slot == 0
    first = dict(_as_pairs(region.shared_buffers(0)))
    later = dict(_as_pairs(region.shared_buffers(2)))
    assert {"kda_conv_q", "kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache"} <= set(
        later
    )
    # The first KDA layer also holds the 16-byte accepted-token count; every layer holds the same replay caches.
    assert first.pop(str(COUNT_ROLE)) == 16
    assert str(COUNT_ROLE) not in later
    assert first == later and all(size > 0 for size in later.values())


def test_the_recurrent_layer_config_holds_the_replay_caches(monkeypatch) -> None:
    state_buffers = [BufferConfig(role=DataRole("ssm_state"), size=64)]
    monkeypatch.setattr(
        MambaHybridCacheManagerV2,
        "_get_recurrent_buffer_configs",
        lambda self, global_layer_id: list(state_buffers),
    )
    state = kda_cache_manager.K31KDAReplayState(NUM_SPEC)
    manager = kda_cache_manager.K31HybridCacheManager.__new__(
        kda_cache_manager.K31HybridCacheManager
    )
    manager._kda_replay = state
    manager._kda_replay_region = _region(state)
    assert _as_pairs(manager._get_recurrent_buffer_configs(0)) == [("ssm_state", 64)] + (
        _expected_buffers(state, first_layer=True)
    )
    assert _as_pairs(manager._get_recurrent_buffer_configs(2)) == [("ssm_state", 64)] + (
        _expected_buffers(state, first_layer=False)
    )


# --- The speculative state ---------------------------------------------------------------------


@pytest.fixture
def max_num_spec() -> int:
    """The verify kernel's draft capacity; its module imports the CuTe DSL."""
    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("the KDA verify kernel module imports the CuTe DSL")
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm.kda_mtp_decode_v2 import (
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


def _manager(tokens_per_gen_step=None, requested_num_spec=None, context_server=False):
    """A manager with only the fields ``_initialize_spec_state`` reads (``kda_context_only``: the
    engine's flag of a disaggregated context server, ``TRTLLM_DISAGG_ROLE=context``)."""
    manager = kda_cache_manager.K31HybridCacheManager.__new__(
        kda_cache_manager.K31HybridCacheManager
    )
    manager._requested_num_spec = requested_num_spec
    manager.spec_config = (
        None
        if tokens_per_gen_step is None
        else SimpleNamespace(tokens_per_gen_step=tokens_per_gen_step)
    )
    manager.kda_context_only = context_server
    manager._state_layout = None
    return manager


def test_verify_capacity_builds_the_replay_state(max_num_spec, replay_states) -> None:
    manager = _manager(tokens_per_gen_step=max_num_spec + 1)
    assert manager._initialize_spec_state() is manager._kda_replay
    assert replay_states == [max_num_spec]
    assert manager.kda_context_only is False


def test_a_context_server_has_no_replay_state(max_num_spec, replay_states) -> None:
    manager = _manager(tokens_per_gen_step=max_num_spec + 1, context_server=True)
    assert manager._initialize_spec_state() is None
    assert manager.kda_context_only is True
    assert replay_states == []
