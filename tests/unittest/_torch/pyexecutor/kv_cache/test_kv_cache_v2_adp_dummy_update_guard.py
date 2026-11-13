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
"""A terminated attention-DP dummy never resizes a newer dummy's draft mirror.

Under attention DP every rank pads an idle iteration with a generation dummy that reuses the
singleton ATTENTION_DP_DUMMY_REQUEST_ID. A terminated dummy freed its caches in
``_handle_responses``; whatever ``update_resources`` still finds mapped under that id belongs to a
newer dummy, so the deferred update of the old batch must leave it, and its in-flight step widths,
alone.
"""

from types import SimpleNamespace

import pytest

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.cpu_only

CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

DUMMY_ID = 2**64 - 2
MAX_DRAFT = 7


class _KvCache:
    """Records resizes; the guard must not issue any."""

    def __init__(self) -> None:
        self.capacity = 0
        self.history_length = 0
        self.is_active = True
        self.resizes = []

    def resize(self, capacity=None, history_length=None) -> bool:
        self.resizes.append((capacity, history_length))
        return True


def test_terminated_adp_dummy_never_resizes_a_newer_dummys_cache():
    draft = KVCacheManagerV2.__new__(KVCacheManagerV2)
    draft.kv_cache_type = CacheType.SELF
    draft.is_draft = True
    mirror = _KvCache()
    draft.kv_cache_map = {DUMMY_ID: mirror}
    draft._inflight_draft_lens = {DUMMY_ID: [MAX_DRAFT]}
    draft._allocated_draft_lens = {DUMMY_ID: MAX_DRAFT}

    terminated = SimpleNamespace(
        py_request_id=DUMMY_ID,
        is_attention_dp_dummy=True,
        state=LlmRequestState.GENERATION_COMPLETE,
    )
    draft.update_resources(SimpleNamespace(generation_requests=[terminated], context_requests=[]))

    assert mirror.resizes == []
    # The newer dummy's own step bookkeeping is left for its own update.
    assert draft._inflight_draft_lens == {DUMMY_ID: [MAX_DRAFT]}
    assert draft._allocated_draft_lens == {DUMMY_ID: MAX_DRAFT}
