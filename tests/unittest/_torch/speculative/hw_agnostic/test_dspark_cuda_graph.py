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
"""CUDA-graph-safety unit tests for the DSpark batched draft-attention primitives.

The batched, sync-free primitives must be **numerically identical, per request**,
to the validated scalar path — only the host-int ``start_pos`` and the per-request
window indexing are tensorized. These tests assert that equivalence on CPU. The
fused batched attention itself is covered on GPU by
``test_dspark_cute_dsl_attention.py``.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dspark import (
    apply_dspark_rotary,
    apply_dspark_rotary_batched,
    get_dspark_topk_idxs,
    get_dspark_topk_idxs_batched,
    precompute_dspark_freqs_cis,
)


@pytest.mark.parametrize("start_positions", [(1, 3, 20), (5, 5, 5), (2, 7, 200)])
def test_batched_topk_matches_scalar(start_positions):
    """Fixed-size masked batched topk == scalar topk per request (valid slots)."""
    window, block = 8, 3
    start_pos = torch.tensor(start_positions, dtype=torch.long)
    batched = get_dspark_topk_idxs_batched(window, block, start_pos)
    assert tuple(batched.shape) == (len(start_positions), block, window + block)
    for i, sp in enumerate(start_positions):
        scalar = get_dspark_topk_idxs(window, 1, block, int(sp))[0]  # [block, topk_i]
        # The batched row drops the -1-masked context slots to recover the scalar
        # (variable-width) index list; both must then be identical.
        for m in range(block):
            valid = batched[i, m][batched[i, m] >= 0]
            torch.testing.assert_close(valid, scalar[m].to(valid.dtype))


def test_batched_topk_respects_partial_window_valid_len():
    """Only actually written circular slots are visible for partial seeds."""
    window, block = 8, 3
    start_pos = torch.tensor([3, 10, 20], dtype=torch.long)
    valid_len = torch.tensor([0, 3, 8], dtype=torch.long)
    batched = get_dspark_topk_idxs_batched(window, block, start_pos, valid_len)
    assert tuple(batched.shape) == (3, block, window + block)

    for i, (pos, count) in enumerate(zip(start_pos.tolist(), valid_len.tolist())):
        expected_context = sorted((pos - age) % window for age in range(count))
        row = batched[i, 0]
        context = row[(row >= 0) & (row < window)].tolist()
        assert context == expected_context
        block_indices = row[row >= window].tolist()
        assert block_indices == list(range(window, window + block))


@pytest.mark.parametrize("ndim", [3, 4])
def test_batched_rotary_matches_scalar_per_row(ndim):
    """apply_dspark_rotary_batched (per-row freqs) == scalar applied row by row."""
    torch.manual_seed(0)
    G, s, h, rd = 3, 4, 2, 8
    x = torch.randn(G, s, h, rd) if ndim == 4 else torch.randn(G, s, rd)
    table = precompute_dspark_freqs_cis(rd, 64)
    # Per-row absolute start positions -> per-row freq windows.
    starts = [1, 9, 30]
    per_row = torch.stack([table[sp : sp + s] for sp in starts], dim=0)  # [G, s, rd/2]
    got = apply_dspark_rotary_batched(x, per_row)
    for i, sp in enumerate(starts):
        ref_i = apply_dspark_rotary(x[i : i + 1], table[sp : sp + s])
        torch.testing.assert_close(got[i : i + 1], ref_i)
