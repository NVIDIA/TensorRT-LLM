# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-based scheduler capacity on KVCacheManagerV2.

``get_max_resource_count`` and ``get_needed_resource_to_completion`` report
bytes so that pool groups with different slot sizes stay commensurable, and so
that the context term tracks SWA scratch reuse: with scratch on, only the first
layer of each (window, slot size) group grows per token, and the rest cost one
retained window per request.
"""

import types
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

_BYTES_PER_LAYER_TOKEN = 100
_TOKENS_PER_BLOCK = 16
_WINDOW = 64
# Two windowed layers sharing one (window, size) key, then two full-attention
# layers.  The shared key is what scratch reuse collapses.
_WINDOWS = [_WINDOW, _WINDOW, None, None]
_QUOTA_BYTES = 8 * 1024**3


def _make_manager(*, scratch: bool) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.num_local_layers = len(_WINDOWS)
    manager.max_attention_window_vec = list(_WINDOWS)
    manager.tokens_per_block = _TOKENS_PER_BLOCK
    manager.num_extra_kv_tokens = 0
    manager.enable_swa_scratch_reuse = scratch
    manager.impl = SimpleNamespace(get_quota=lambda level: _QUOTA_BYTES)
    manager.get_layer_bytes_per_token = types.MethodType(
        lambda self, local_layer_idx, data_role: _BYTES_PER_LAYER_TOKEN, manager
    )
    return manager


def _context_request(prompt_len: int) -> SimpleNamespace:
    return SimpleNamespace(
        is_context_init_state=True,
        is_generation_in_progress_state=False,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.CONTEXT_INIT,
        prompt_len=prompt_len,
        max_new_tokens=100,
    )


def _generation_request(prompt_len: int, max_new_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        is_context_init_state=False,
        is_generation_in_progress_state=True,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
    )


def test_max_resource_count_reports_gpu_quota_bytes():
    assert _make_manager(scratch=True).get_max_resource_count() == _QUOTA_BYTES


def test_context_cost_drops_with_swa_scratch_reuse():
    request = _context_request(prompt_len=1000)

    with_scratch = _make_manager(scratch=True).get_needed_resource_to_completion(request)
    without_scratch = _make_manager(scratch=False).get_needed_resource_to_completion(request)

    # Both windowed layers grow per token without scratch; with scratch only the
    # first does, and the second costs one retained window.
    full_attn_per_token = 2 * _BYTES_PER_LAYER_TOKEN
    window_bytes = _WINDOW * _BYTES_PER_LAYER_TOKEN
    assert without_scratch == 1000 * (full_attn_per_token + 2 * _BYTES_PER_LAYER_TOKEN)
    assert with_scratch == 1000 * (full_attn_per_token + _BYTES_PER_LAYER_TOKEN) + window_bytes
    assert with_scratch < without_scratch


def test_generation_cost_ignores_swa_scratch_reuse():
    request = _generation_request(prompt_len=10, max_new_tokens=20)

    # A generation request holds no stale SWA block, so the flag must not move it.
    assert _make_manager(scratch=True).get_needed_resource_to_completion(request) == _make_manager(
        scratch=False
    ).get_needed_resource_to_completion(request)


def test_generation_slope_is_full_attention_only():
    manager = _make_manager(scratch=True)
    base = manager.get_needed_resource_to_completion(_generation_request(10, 20))
    longer = manager.get_needed_resource_to_completion(_generation_request(10, 21))

    assert longer - base == 2 * _BYTES_PER_LAYER_TOKEN


def test_extra_kv_tokens_are_charged():
    manager = _make_manager(scratch=True)
    base = manager.get_needed_resource_to_completion(_context_request(1000))
    manager.num_extra_kv_tokens = 4
    assert manager.get_needed_resource_to_completion(_context_request(1000)) > base


def test_unclassified_request_state_is_rejected():
    manager = _make_manager(scratch=True)
    request = _context_request(10)
    request.is_context_init_state = False
    request.state = LlmRequestState.GENERATION_COMPLETE

    with pytest.raises(ValueError, match="Unsupported request state"):
        manager.get_needed_resource_to_completion(request)
