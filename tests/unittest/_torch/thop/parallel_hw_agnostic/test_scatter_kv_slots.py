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
"""trtllm::scatter_kv_slots_ against an index_put reference."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  # registers the trtllm:: ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")


def reference(pool, k, v, dst, dst2=None, src=None):
    tpb = pool.shape[3]
    token = torch.arange(dst.numel(), device=DEV) if src is None else src
    out = pool.clone()
    for slots in (dst, dst2):
        if slots is None:
            continue
        page, slot = slots // tpb, slots % tpb
        out[page, 0, :, slot] = k[token]
        out[page, 1, :, slot] = v[token]
    return out


def fused_kv(tokens, heads, head_dim, dtype, q_heads=3):
    """K and V as strided slices of one fused QKV projection, like a model produces them."""
    qkv = torch.randn(tokens, (q_heads + 2 * heads) * head_dim, device=DEV).to(dtype)
    k = qkv[:, q_heads * head_dim : (q_heads + heads) * head_dim].view(tokens, heads, head_dim)
    v = qkv[:, (q_heads + heads) * head_dim :].view(tokens, heads, head_dim)
    return k, v


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize("head_dim", [128, 64, 6, 3])  # 6 and 3: narrower than 16-byte copies
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("second", [False, True])
def test_scatter_matches_reference(dtype, head_dim, strided, second):
    torch.manual_seed(0)
    pages, heads, tpb, tokens = 37, 4, 32, 90

    def values(*shape):  # every byte pattern matters for a byte copy, int8 included
        if dtype == torch.int8:
            return torch.randint(-128, 128, shape, device=DEV, dtype=torch.int8)
        return torch.randn(*shape, device=DEV).to(dtype)

    pool = values(pages, 2, heads, tpb, head_dim)
    if strided:
        k, v = fused_kv(tokens, heads, head_dim, dtype)
        if dtype == torch.int8:
            k.copy_(values(*k.shape)), v.copy_(values(*v.shape))
        assert not k.is_contiguous()
    else:
        k, v = values(tokens, heads, head_dim), values(tokens, heads, head_dim)
    all_slots = torch.randperm(pages * tpb, device=DEV)
    dst = all_slots[:tokens].contiguous()
    dst2 = all_slots[tokens : 2 * tokens].contiguous() if second else None
    expected = reference(pool, k, v, dst, dst2)
    torch.ops.trtllm.scatter_kv_slots_(pool, k, v, dst, dst2)
    assert torch.equal(pool, expected)


def test_src_gathers_a_subset_of_tokens():
    torch.manual_seed(1)
    pool = torch.zeros(9, 2, 2, 32, 128, device=DEV, dtype=torch.bfloat16)
    k, v = fused_kv(50, 2, 128, torch.bfloat16)
    src = torch.tensor([49, 0, 7, 7, 13], device=DEV)
    dst = torch.tensor([0, 31, 32, 100, 287], device=DEV)
    expected = reference(pool, k, v, dst, src=src)
    torch.ops.trtllm.scatter_kv_slots_(pool, k, v, dst, src=src)
    assert torch.equal(pool, expected)


def test_strided_pool_view():
    """A pool that is a strided view, e.g. one layer of a layer-interleaved buffer."""
    torch.manual_seed(2)
    whole = torch.zeros(20, 2, 2, 4, 16, 64, device=DEV, dtype=torch.float16)
    pool = whole[:, 1]  # [20, 2, 4, 16, 64], page stride skips the other layer
    k = torch.randn(30, 4, 64, device=DEV, dtype=torch.float16)
    v = torch.randn(30, 4, 64, device=DEV, dtype=torch.float16)
    dst = torch.randperm(20 * 16, device=DEV)[:30]
    expected = reference(pool, k, v, dst)
    torch.ops.trtllm.scatter_kv_slots_(pool, k, v, dst)
    assert torch.equal(pool, expected)
    assert whole[:, 0].abs().max().item() == 0, "the other layer must stay untouched"


def test_graph_replay_reads_indices_at_launch():
    torch.manual_seed(3)
    pool = torch.zeros(8, 2, 2, 32, 128, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(16, 2, 128, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dst = torch.arange(16, device=DEV)
    torch.ops.trtllm.scatter_kv_slots_(pool, k, v, dst)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.scatter_kv_slots_(pool, k, v, dst)
    pool.zero_()
    dst.copy_(torch.arange(100, 116, device=DEV))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(pool, reference(torch.zeros_like(pool), k, v, dst))


def test_bad_arguments():
    pool = torch.zeros(4, 2, 2, 32, 128, device=DEV, dtype=torch.bfloat16)
    k = torch.zeros(8, 2, 128, device=DEV, dtype=torch.bfloat16)
    dst = torch.arange(8, device=DEV)
    with pytest.raises(RuntimeError, match="heads of"):
        torch.ops.trtllm.scatter_kv_slots_(pool, k[:, :1], k[:, :1], dst)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.ops.trtllm.scatter_kv_slots_(pool, k.half(), k.half(), dst)
    with pytest.raises(RuntimeError, match="int64"):
        torch.ops.trtllm.scatter_kv_slots_(pool, k, k, dst.int())
    with pytest.raises(RuntimeError, match="source tokens"):
        torch.ops.trtllm.scatter_kv_slots_(pool, k, k, torch.arange(9, device=DEV))
    with pytest.raises(RuntimeError, match="entries"):
        torch.ops.trtllm.scatter_kv_slots_(pool, k, k, dst, dst2=dst[:4])


def test_traces_under_torch_compile():
    """The fake registration lets dynamo trace the op without a graph break."""
    torch.manual_seed(4)
    pool = torch.zeros(6, 2, 2, 32, 64, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(10, 2, 64, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dst = torch.randperm(6 * 32, device=DEV)[:10]
    expected = reference(pool, k, v, dst)

    @torch.compile(fullgraph=True, backend="aot_eager")
    def scatter(pool, k, v, dst):
        torch.ops.trtllm.scatter_kv_slots_(pool, k, v, dst)
        return pool

    assert torch.equal(scatter(pool, k, v, dst), expected)
