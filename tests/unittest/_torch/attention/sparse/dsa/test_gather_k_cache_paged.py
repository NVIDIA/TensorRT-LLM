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
"""gather_k_cache_paged: the same bytes as torch.ops.trtllm.indexer_k_cache_gather_op on page-strided views."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch.attention.backends.sparse.dsa.kernels import gather_k_cache_paged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("head_dim", [128, 64, 256])
@pytest.mark.parametrize("layer_stride", [1, 3])
def test_matches_cpp_gather(head_dim, layer_stride):
    g = torch.Generator(device="cuda").manual_seed(head_dim + layer_stride)
    tpb, pages, per_token = 64, 40, head_dim + 4
    page_bytes = tpb * per_token
    # Pages of one layer interleaved with other layers' pages: page stride = layer_stride * page_bytes.
    pool = torch.randint(
        0, 256, (pages * layer_stride, page_bytes), dtype=torch.uint8, device="cuda", generator=g
    )
    k_cache_4d = pool[::layer_stride].view(pages, tpb, 1, per_token)
    num_tokens = 1500
    page = torch.randint(0, pages, (num_tokens,), device="cuda", generator=g)
    tok = torch.randint(0, tpb, (num_tokens,), device="cuda", generator=g)
    slot_fp8 = (page * page_bytes + tok * head_dim).to(torch.int64)
    slot_scale = (page * page_bytes + tpb * head_dim + tok * 4).to(torch.int64)
    start, n = 37, 1400
    ref_k, ref_s = torch.ops.trtllm.indexer_k_cache_gather_op(
        k_cache_4d, slot_fp8, slot_scale, start, n, head_dim
    )
    out = gather_k_cache_paged(k_cache_4d, slot_fp8, slot_scale, start, n, head_dim)
    assert out is not None
    k, s = out
    assert k.dtype == ref_k.dtype and k.shape == ref_k.shape and s.shape == ref_s.shape
    assert torch.equal(k.view(torch.uint8), ref_k.view(torch.uint8))
    assert torch.equal(s.view(torch.uint8), ref_s.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_negative_slots_are_skipped_and_other_layouts_fall_back():
    tpb, per_token = 64, 132
    pool = torch.zeros(4, tpb * per_token, dtype=torch.uint8, device="cuda")
    k_cache_4d = pool.view(4, tpb, 1, per_token)
    slot = torch.tensor([-1, 0, 128], dtype=torch.int64, device="cuda")
    out = gather_k_cache_paged(k_cache_4d, slot, slot + tpb * 128, 0, 3, 128)
    assert out is not None and out[0].shape == (3, 128)
    assert gather_k_cache_paged(k_cache_4d.transpose(1, 3), slot, slot, 0, 3, 128) is None
