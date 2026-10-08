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
"""Tests for the V2 one-model draft max_tokens backstop in the GPU budget split."""

import pytest

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

pytestmark = pytest.mark.cpu_only

GB = 1 << 30


def _make_creator(
    *,
    costs_available: bool = False,
    is_v2: bool = True,
    separate_draft: bool = True,
    two_model: bool = False,
    user_max_tokens=None,
    slope: int = 100,
) -> KvCacheCreator:
    c = object.__new__(KvCacheCreator)
    c._kv_cache_config = KvCacheConfig(max_gpu_total_bytes=GB, max_tokens=user_max_tokens)
    c._is_kv_cache_manager_v2 = is_v2
    c._max_batch_size = 8
    c._draft_model_engine = object() if two_model else None
    c._should_create_separate_draft_kv_cache = lambda: separate_draft
    c._get_kv_size_per_token = lambda *args, **kwargs: CacheCost(slope=slope)
    costs = (CacheCost(slope=80), CacheCost(slope=20)) if costs_available else None
    c._get_target_and_draft_cache_costs = lambda *args, **kwargs: costs
    return c


def _split(c, budget_attr="max_gpu_total_bytes"):
    return c._split_kv_cache_budget_for_draft(budget_attr, c._kv_cache_config, None)


def test_successful_split_sets_no_max_tokens_on_either_manager():
    # The split already sizes both managers; a max_tokens carried into the
    # split configs would clamp the target's quota below its share.
    target, draft = _split(_make_creator(costs_available=True))
    assert target.max_tokens is None
    assert draft.max_tokens is None
    assert target.max_gpu_total_bytes + draft.max_gpu_total_bytes == GB


def test_unsplittable_budget_bounds_only_the_draft():
    c = _make_creator(slope=100)
    target, draft = _split(c)
    assert target is c._kv_cache_config
    assert target.max_tokens is None
    assert target.max_gpu_total_bytes == GB
    assert draft is not target
    assert draft.max_tokens == GB // 100


def test_zero_slope_leaves_draft_unbounded():
    # Every attention layer windowed: tokens_for_budget returns 0, which V2
    # must not read as a cap of zero tokens.
    _, draft = _split(_make_creator(slope=0))
    assert draft is None


def test_user_max_tokens_is_left_alone():
    _, draft = _split(_make_creator(user_max_tokens=1234))
    assert draft is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"separate_draft": False},
        {"two_model": True},
        {"is_v2": False},
    ],
    ids=["target_only", "two_model", "v1"],
)
def test_backstop_is_scoped_to_one_model_v2_drafts(kwargs):
    _, draft = _split(_make_creator(**kwargs))
    assert draft is None


def test_offload_budget_is_not_bounded():
    c = _make_creator()
    c._kv_cache_config.host_cache_size = GB
    _, draft = _split(c, "host_cache_size")
    assert draft is None
