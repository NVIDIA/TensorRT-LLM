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
"""Locality domain plumbing that needs no GPU: request propagation and fork-join gating."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    resolve_fork_join_attn,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def _request(request_id: int, locality_domain_id):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingParams(n=2, best_of=2, top_k=4)._get_sampling_config(),
        is_streaming=False,
        locality_domain_id=locality_domain_id,
    )


@pytest.mark.parametrize("domain", [None, 0, 1])
def test_child_request_inherits_locality_domain(domain):
    parent = _request(7, domain)
    assert parent.py_locality_domain_id == domain
    parent.create_child_request(8)
    (child,) = parent.child_requests
    assert child.py_locality_domain_id == domain


def test_fork_join_requires_two_domains_and_the_runtime():
    assert resolve_fork_join_attn(2, True, runtime_enabled=True, mock_localization=False)
    assert not resolve_fork_join_attn(1, True, runtime_enabled=True, mock_localization=False)
    assert not resolve_fork_join_attn(2, False, runtime_enabled=True, mock_localization=False)
    # Mock localization keeps the split KV cache on one stream.
    assert not resolve_fork_join_attn(2, True, runtime_enabled=False, mock_localization=True)


def test_fork_join_rejects_bad_configurations():
    with pytest.raises(ValueError, match="exactly 2"):
        resolve_fork_join_attn(3, True, runtime_enabled=True, mock_localization=False)
    with pytest.raises(RuntimeError, match="disabled on this device"):
        resolve_fork_join_attn(2, True, runtime_enabled=False, mock_localization=False)


def test_locality_domains_reject_sparse_attention_at_manager_creation():
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    creator = KvCacheCreator.__new__(KvCacheCreator)
    creator._llm_args = SimpleNamespace(enable_locality_domains=True)
    creator._sparse_attention_config = SimpleNamespace(algorithm="dsa")
    with pytest.raises(ValueError, match="sparse attention"):
        creator._create_kv_cache_manager(Mock())


def test_placement_round_robins_over_arrival_order():
    manager = SimpleNamespace(num_locality_domains=2, _next_locality_domain=0)
    picks = [KVCacheManagerV2.pick_locality_domain(manager) for _ in range(4)]
    assert picks == [0, 1, 0, 1]


def _placer(matches, load=(0, 0)):
    return SimpleNamespace(
        num_locality_domains=2,
        enable_block_reuse=True,
        tokens_per_block=32,
        _next_locality_domain=0,
        _locality_domain_load=list(load),
        _prefix_match_per_domain=lambda *args: list(matches),
    )


def test_placement_follows_the_domain_holding_the_prompt_prefix():
    manager = _placer(matches=(0, 96))
    tokens = list(range(128))
    assert KVCacheManagerV2.pick_locality_domain(manager, tokens) == 1
    assert KVCacheManagerV2.pick_locality_domain(manager, tokens) == 1


def test_placement_round_robins_without_a_full_block_match():
    manager = _placer(matches=(0, 31))
    picks = [KVCacheManagerV2.pick_locality_domain(manager, list(range(64))) for _ in range(4)]
    assert picks == [0, 1, 0, 1]


def test_placement_does_not_pile_a_hot_prefix_onto_a_loaded_domain():
    # Domain 1 already holds 30 more live requests than domain 0, past max(8, total // 4).
    manager = _placer(matches=(0, 96), load=(10, 40))
    assert KVCacheManagerV2.pick_locality_domain(manager, list(range(128))) == 0


def _manager(num_locality_domains, is_draft):
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = SimpleNamespace(num_locality_domains=num_locality_domains)
    manager.is_draft = is_draft
    manager.enable_block_reuse = False
    manager._next_locality_domain = 0
    return manager


@pytest.mark.parametrize("target_domains, draft_domains", [(2, 1), (1, 2)])
def test_draft_manager_with_a_different_domain_count_gets_a_valid_domain(
    target_domains, draft_domains
):
    target = _manager(target_domains, is_draft=False)
    draft = _manager(draft_domains, is_draft=True)
    for request_id in range(4):
        request = _request(request_id, target.pick_locality_domain())
        assert target._request_locality_domain_id(request) == request.py_locality_domain_id
        draft_domain = draft._request_locality_domain_id(request)
        if draft_domains == 1:
            assert draft_domain is None
        else:
            assert 0 <= draft_domain < draft_domains
            assert draft._request_locality_domain_id(request) == draft_domain
