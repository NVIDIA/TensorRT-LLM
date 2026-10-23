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


def test_placement_alternates_domains_when_ids_advance_in_even_steps():
    # n=2 sampling gives each request a parent and a child id, so new requests
    # arrive with ids 0, 2, 4, ...; parity placement would put them all on domain 0.
    manager = SimpleNamespace(num_locality_domains=2, _next_locality_domain=0)
    picks = [
        KVCacheManagerV2.pick_locality_domain(manager, request_id) for request_id in (0, 2, 4, 6)
    ]
    assert picks == [0, 1, 0, 1]
