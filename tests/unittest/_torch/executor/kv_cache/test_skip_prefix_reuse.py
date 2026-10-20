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
"""Requests returning context logits skip KV cache prefix reuse.

Context logits, and the prompt logprobs derived from them, exist only for
tokens that pass through the forward pass. A reused prefix would leave those
positions unscored, so these requests set ``skip_prefix_reuse``: the V1 block
manager neither credits nor claims their cached prefix, and the V2 manager caps
their reuse claim at zero. Both still store the blocks the requests compute.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _skip_prefix_reuse_claim_limit,
    _tighter_claim_limit,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import SamplingConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("return_context_logits", [False, True])
def test_requests_returning_context_logits_skip_prefix_reuse(return_context_logits):
    req = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(),
        is_streaming=False,
        return_context_logits=return_context_logits,
    )
    assert req.skip_prefix_reuse is return_context_logits


class TestSkipPrefixReuseClaimLimit:
    def test_a_skipping_request_claims_nothing(self):
        assert _skip_prefix_reuse_claim_limit(SimpleNamespace(skip_prefix_reuse=True)) == 0

    def test_a_plain_request_is_not_capped(self):
        assert _skip_prefix_reuse_claim_limit(SimpleNamespace(skip_prefix_reuse=False)) is None


class TestTighterClaimLimit:
    @pytest.mark.parametrize(
        "first, second, expected",
        [(None, None, None), (None, 8, 8), (8, None, 8), (0, 8, 0), (12, 8, 8)],
    )
    def test_the_stricter_cap_wins_and_none_means_uncapped(self, first, second, expected):
        assert _tighter_claim_limit(first, second) == expected
