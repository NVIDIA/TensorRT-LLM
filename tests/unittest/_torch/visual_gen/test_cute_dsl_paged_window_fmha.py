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
"""Fixed window attention with a prefix sequence over a paged K/V pool, against its reference."""

import pytest
import torch
from utils.kv_causal_reference import paged_window_reference

from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl.fmha import (
    _get_gpu_arch,
    cute_dsl_paged_window_fmha_fwd,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the CuTe DSL FMHA kernel needs Blackwell",
)


def make_pool(num_logical_pages, page_size, heads_kv, head_dim, dtype, num_layers=3):
    """One layer's view of a pool of ``num_layers`` interleaved layers, with the logical pages
    placed at random physical pages; also returns the logical-to-view-index table."""
    num_physical = num_logical_pages + 5
    pool = torch.randn(
        num_physical * num_layers, 2, heads_kv, page_size, head_dim, dtype=dtype, device="cuda"
    )
    layer = 1
    view = pool[layer:]
    order = torch.randperm(num_physical)[:num_logical_pages]
    table = (order * num_layers).to(torch.int32).cuda()
    return view, table


def run_case(*, prefix, starts, lengths, block_size, heads, heads_kv, head_dim, page_size, dtype):
    num_blocks = len(starts)
    num_positions = max(max(s + n for s, n in zip(starts, lengths)), prefix)
    num_logical = -(-num_positions // page_size) + 1
    kv, table = make_pool(num_logical, page_size, heads_kv, head_dim, dtype)
    q = torch.randn(num_blocks * block_size, heads, head_dim, dtype=dtype, device="cuda")
    out = torch.full_like(q, float("nan"))
    cum_k = torch.tensor(starts, dtype=torch.int32, device="cuda")
    kv_len = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    scale = head_dim**-0.5
    cute_dsl_paged_window_fmha_fwd(
        q,
        kv,
        table,
        cum_k,
        kv_len,
        out,
        prefix_tokens=prefix,
        block_size=block_size,
        sm_scale=scale,
    )
    want = torch.empty_like(q)
    paged_window_reference(
        q,
        kv,
        table,
        cum_k,
        kv_len,
        want,
        prefix_tokens=prefix,
        block_size=block_size,
        sm_scale=scale,
    )
    torch.testing.assert_close(out, want, rtol=2e-2, atol=2e-2)


CASES = [
    # prefix, starts, lengths, block_size
    pytest.param(48, [0, 0, 0, 48], [144, 240, 336, 384], 96, id="chunk-start-clean-pass"),
    pytest.param(48, [400, 496, 592], [384, 384, 384], 96, id="window-after-prefix"),
    pytest.param(64, [64, 160], [200, 200], 100, id="window-starts-at-prefix-end"),
    pytest.param(0, [0], [384], 384, id="no-prefix-one-block"),
    pytest.param(0, [37, 37], [300, 300], 128, id="no-prefix-unaligned-shared-range"),
    pytest.param(300, [10, 20], [150, 600], 64, id="window-inside-prefix"),
    pytest.param(257, [1000, 1000], [5, 1], 96, id="tiny-windows-partial-tiles"),
    pytest.param(100, [100], [0], 96, id="empty-window"),
    pytest.param(128, [128, 300], [128, 512], 192, id="aligned-prefix-and-window"),
]


@pytest.mark.parametrize("prefix,starts,lengths,block_size", CASES)
def test_matches_reference(prefix, starts, lengths, block_size):
    run_case(
        prefix=prefix,
        starts=starts,
        lengths=lengths,
        block_size=block_size,
        heads=4,
        heads_kv=2,
        head_dim=128,
        page_size=32,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_head_dims_and_page_sizes(head_dim, page_size):
    run_case(
        prefix=70,
        starts=[200, 296],
        lengths=[310, 310],
        block_size=96,
        heads=8,
        heads_kv=2,
        head_dim=head_dim,
        page_size=page_size,
        dtype=torch.float16,
    )


def test_arch_is_supported():
    assert _get_gpu_arch()
