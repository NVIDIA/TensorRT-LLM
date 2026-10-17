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
"""trtllm::copy_kv_slots_: pool-row copies across every layer, K and V, every head."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  # loads the custom ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")

LAYERS, PAGES, HEADS, TPB = 3, 5, 2, 8


def make_pool(dtype, head_dim):
    """Layers interleaved per page, as the K/V pool stores them: view page
    ``page * LAYERS + layer``; a layer-0 pool row is ``view_page * rows_per_page + slot``."""
    return torch.randn(LAYERS * PAGES, 2, HEADS, TPB, head_dim, device=DEV).to(dtype)


def reference(pool, src, dst):
    rows = pool.clone().view(-1, pool.shape[-1])
    rpp = 2 * HEADS * TPB
    for s, d in zip(src.tolist(), dst.tolist()):
        if s < 0:
            continue
        for layer in range(LAYERS):
            for r in range(2 * HEADS):
                off = layer * rpp + r * TPB
                rows[d + off] = rows[s + off]
    return rows.view_as(pool)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("head_dim", [64, 72])  # 72 fp16 is 144 bytes: 16-byte vectors; 72 bf16 too
def test_copies_every_layer_and_head_and_skips_sentinels(dtype, head_dim):
    pool = make_pool(dtype, head_dim)
    rpp = 2 * HEADS * TPB
    view = lambda page: page * LAYERS  # noqa: E731
    src = torch.tensor(
        [view(0) * rpp + 3, view(1) * rpp + 0, -1, view(2) * rpp + 7, -1], device=DEV
    )
    dst = torch.tensor(
        [view(3) * rpp + 0, view(3) * rpp + 1, 12345, view(4) * rpp + 5, -1], device=DEV
    )
    want = reference(pool, src, dst)
    torch.ops.trtllm.copy_kv_slots_(pool, src, dst, LAYERS, HEADS, TPB)
    torch.testing.assert_close(pool, want, rtol=0, atol=0)


def test_graph_replay_reads_rows_at_launch():
    pool = make_pool(torch.bfloat16, 64)
    rpp = 2 * HEADS * TPB
    src = torch.tensor([0 * LAYERS * rpp + 1], device=DEV)
    dst = torch.tensor([2 * LAYERS * rpp + 2], device=DEV)
    graph = torch.cuda.CUDAGraph()
    torch.ops.trtllm.copy_kv_slots_(pool, src, dst, LAYERS, HEADS, TPB)
    with torch.cuda.graph(graph):
        torch.ops.trtllm.copy_kv_slots_(pool, src, dst, LAYERS, HEADS, TPB)
    pool.normal_()
    src.fill_(1 * LAYERS * rpp + 4)
    dst.fill_(4 * LAYERS * rpp + 6)
    want = reference(pool, src, dst)
    graph.replay()
    torch.testing.assert_close(pool, want, rtol=0, atol=0)


def test_bad_arguments():
    pool = make_pool(torch.bfloat16, 64)
    src = torch.zeros(2, dtype=torch.int64, device=DEV)
    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.trtllm.copy_kv_slots_(pool.transpose(0, 1), src, src, LAYERS, HEADS, TPB)
    with pytest.raises(RuntimeError, match="int64"):
        torch.ops.trtllm.copy_kv_slots_(pool, src.int(), src, LAYERS, HEADS, TPB)
    with pytest.raises(RuntimeError, match="match src"):
        torch.ops.trtllm.copy_kv_slots_(pool, src, src[:1], LAYERS, HEADS, TPB)
    with pytest.raises(RuntimeError, match="positive"):
        torch.ops.trtllm.copy_kv_slots_(pool, src, src, 0, HEADS, TPB)
    torch.ops.trtllm.copy_kv_slots_(pool, src[:0], src[:0], LAYERS, HEADS, TPB)  # nothing to do
