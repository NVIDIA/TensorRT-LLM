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
"""Dispatch regression tests for one-model draft KV resolution."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.speculative.utils import get_draft_kv_cache_manager

ONE_MODEL = SimpleNamespace(spec_dec_mode=SimpleNamespace(use_one_engine=lambda: True))


class _FakeResources:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_resource_manager(self, key):
        return self._mapping.get(key)


class _SharedTargetWithView:
    """Target manager carrying appended draft layers (unified KV cache)."""

    _view = object()

    def get_draft_kv_cache_view(self):
        return self._view


class _PlainSharedTarget:
    """Same-geometry shared drafter: no view, drafter uses the manager."""


def test_registered_separate_manager_wins():
    separate = object()
    resources = _FakeResources(
        {
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: separate,
            ResourceManagerType.KV_CACHE_MANAGER: _SharedTargetWithView(),
        }
    )
    assert get_draft_kv_cache_manager(ONE_MODEL, resources) is separate


def test_shared_manager_falls_back_to_draft_view():
    target = _SharedTargetWithView()
    resources = _FakeResources({ResourceManagerType.KV_CACHE_MANAGER: target})
    assert get_draft_kv_cache_manager(ONE_MODEL, resources) is target._view


def test_plain_shared_manager_resolves_to_none():
    resources = _FakeResources({ResourceManagerType.KV_CACHE_MANAGER: _PlainSharedTarget()})
    assert get_draft_kv_cache_manager(ONE_MODEL, resources) is None


def test_no_target_manager_resolves_to_none():
    assert get_draft_kv_cache_manager(ONE_MODEL, _FakeResources({})) is None


def test_missing_inputs_resolve_to_none():
    assert get_draft_kv_cache_manager(None, _FakeResources({})) is None
    assert get_draft_kv_cache_manager(ONE_MODEL, None) is None


class _BrokenViewTarget:
    def get_draft_kv_cache_view(self):
        raise AttributeError("max_blocks_per_seq")


def test_broken_view_construction_propagates():
    # A failure inside view construction must surface, not silently
    # downgrade to "no view": getattr fetches the bound method without
    # executing it, so the exception escapes from the call itself.
    resources = _FakeResources({ResourceManagerType.KV_CACHE_MANAGER: _BrokenViewTarget()})
    with pytest.raises(AttributeError, match="max_blocks_per_seq"):
        get_draft_kv_cache_manager(ONE_MODEL, resources)
