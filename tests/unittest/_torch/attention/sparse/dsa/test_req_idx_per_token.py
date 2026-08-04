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
"""
Tests for the token->request map used by DSAtrtllmAttentionMetadata.

Two invariants, one test each:

1. build_req_idx_per_token (device searchsorted) must equal
   prepare_for_indices_conversion()'s host repeat_interleave for every batch
   layout, so the two builders cannot drift.
2. on_update_kv_lens() must rebuild the map for the CURRENT seq_lens. The MTP
   draft loop rewrites seq_lens to one token per request without re-running
   prepare(); reusing prepare()'s map misattributes every draft token to
   request 0, corrupting indexer K-writes and top-k reads through the wrong
   block table (https://nvbugs/6513132, https://nvbugs/6513093).
"""

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSAtrtllmAttentionMetadata,
    build_req_idx_per_token,
)


def _host_reference(seq_lens: torch.Tensor) -> torch.Tensor:
    """The host build from prepare_for_indices_conversion()."""
    return torch.repeat_interleave(
        torch.arange(len(seq_lens), dtype=torch.int32, device=seq_lens.device),
        seq_lens,
        dim=0,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([4, 4, 4], id="target_forward_mtp3"),
        pytest.param([1, 1, 1], id="draft_loop"),
        pytest.param([37, 5, 1, 1], id="mixed_ctx_gen"),
        pytest.param([2, 0, 3], id="zero_length_row"),
        pytest.param([0, 4], id="leading_zero_row"),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matches_host_repeat_interleave(seq_lens, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    num_tokens = int(seq_lens.sum())

    result = build_req_idx_per_token(seq_lens, num_tokens)

    assert result.to(torch.int32).tolist() == _host_reference(seq_lens).tolist()


def test_on_update_kv_lens_rebuilds_stale_map():
    """The draft-loop transition: the regression this PR fixes.

    Bare-instance construction (object.__new__ + backing fields) mirrors
    test_dsa_indexer.py; kv_cache_manager=None and num_generations=0 confine
    on_update_kv_lens() to the map rebuild under test.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    max_draft_len = 3
    num_requests = 3

    md = object.__new__(DSAtrtllmAttentionMetadata)
    md.kv_cache_manager = None
    md._num_generations = 0
    # Collaborators invoked at the end of on_update_kv_lens(); unrelated to
    # the map rebuild under test (same stubbing style as test_dsa_indexer.py).
    md.kv_lens_cuda = torch.tensor([100, 200, 300], dtype=torch.int32, device=device)
    md._compute_kv_lens_row_reorder = Mock()
    md.prepare_dense_topk_indices = Mock()

    # prepare() state for the target forward: 1 + max_draft_len tokens per
    # request, map built host-side via repeat_interleave.
    target_seq_lens = torch.full((num_requests,), 1 + max_draft_len, dtype=torch.int32)
    md._seq_lens = target_seq_lens
    md._seq_lens_cuda = target_seq_lens.to(device)
    md._num_tokens = int(target_seq_lens.sum())
    md.req_idx_per_token = torch.empty(md._num_tokens, dtype=torch.int32, device=device)
    md.req_idx_per_token[:] = _host_reference(md._seq_lens_cuda)

    # The draft loop rewrites seq_lens to one token per request (what
    # _preprocess_inputs does between draft iterations) without re-running
    # prepare(). The stale prefix misattributes every token to request 0.
    draft_seq_lens = torch.ones(num_requests, dtype=torch.int32)
    md._seq_lens = draft_seq_lens
    md._seq_lens_cuda = draft_seq_lens.to(device)
    md._num_tokens = num_requests
    assert md.req_idx_per_token[:num_requests].tolist() == [0, 0, 0]

    md.on_update_kv_lens()

    assert md.req_idx_per_token[:num_requests].tolist() == [0, 1, 2]
