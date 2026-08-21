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

1. build_req_idx_per_token must match the host repeat_interleave build for
   every layout, so the device and host builders cannot drift.
2. on_update_kv_lens() must rebuild the map after the MTP draft loop rewrites
   seq_lens, or every draft token is misattributed to request 0
   (https://nvbugs/6513132, https://nvbugs/6513093).
"""

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import (
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
    """on_update_kv_lens() must replace prepare()'s stale map (fails pre-fix)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    max_draft_len = 3
    num_requests = 3

    md = object.__new__(DSAtrtllmAttentionMetadata)
    md.kv_cache_manager = None
    md._num_generations = 0
    # __init__ (bypassed by object.__new__) defaults this to False;
    # on_update_kv_lens() reads it since #16925.
    md.in_mtp_draft_loop = False
    # Stub collaborators unrelated to the map rebuild (test_dsa_indexer.py style).
    md.kv_lens_cuda = torch.tensor([100, 200, 300], dtype=torch.int32, device=device)
    md._compute_kv_lens_row_reorder = Mock()
    md.prepare_dense_topk_indices = Mock()

    # prepare()-time state: target forward, 1 + max_draft_len tokens/request.
    target_seq_lens = torch.full((num_requests,), 1 + max_draft_len, dtype=torch.int32)
    md._seq_lens = target_seq_lens
    md._seq_lens_cuda = target_seq_lens.to(device)
    md._num_tokens = int(target_seq_lens.sum())
    md.req_idx_per_token = torch.empty(md._num_tokens, dtype=torch.int32, device=device)
    md.req_idx_per_token[:] = _host_reference(md._seq_lens_cuda)

    # Draft loop: one token per request; the stale prefix reads [0, 0, 0].
    draft_seq_lens = torch.ones(num_requests, dtype=torch.int32)
    md._seq_lens = draft_seq_lens
    md._seq_lens_cuda = draft_seq_lens.to(device)
    md._num_tokens = num_requests
    assert md.req_idx_per_token[:num_requests].tolist() == [0, 0, 0]

    md.on_update_kv_lens()

    assert md.req_idx_per_token[:num_requests].tolist() == [0, 1, 2]
