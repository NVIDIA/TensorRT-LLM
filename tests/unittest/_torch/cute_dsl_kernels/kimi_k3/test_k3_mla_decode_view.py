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
"""k3_mla_decode_view: the Kimi K3 MLA decode kernels' per-layer inputs from the attention metadata for R generation
requests of T tokens (views of the metadata buffers, no copies), and a reason string for every step it cannot take."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_mla.decode_view import k3_mla_decode_view

MAX_SEQS, MAX_PAGES, POOL_PAGES = 16, 32, 40


def _attn(num_heads=6):
    return SimpleNamespace(
        num_heads=num_heads, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, q_scaling=1.0,
        layer_idx=3, get_local_layer_idx=lambda meta: 1,
    )  # fmt: skip


def _meta(num_generations, num_contexts=0, pool=None, **overrides):
    if pool is None:
        pool = torch.zeros(POOL_PAGES, 1, 64, 1, 576, dtype=torch.bfloat16)
    meta = SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        beam_width=1,
        tokens_per_block=64,
        helix_position_offsets=None,
        kv_cache_manager=SimpleNamespace(get_buffers=lambda layer_idx: pool),
        kv_cache_block_offsets=torch.arange(2 * MAX_SEQS * 2 * MAX_PAGES, dtype=torch.int32).view(
            2, MAX_SEQS, 2, MAX_PAGES
        ),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        kv_lens_cuda_runtime=torch.arange(100, 100 + MAX_SEQS, dtype=torch.int32),
    )
    for key, value in overrides.items():
        setattr(meta, key, value)
    return meta


SPLITS = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (8, 1), (1, 8), (2, 4), (4, 2)] + [
    (r, 8) for r in range(2, 9)
]


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
def test_view_requests(num_requests, tokens):
    """R x T: the R page-table rows of the layer's pool as a strided view of kv_cache_block_offsets, the R lengths as
    a view of kv_lens_cuda_runtime, R and T."""
    meta = _meta(num_requests)
    view = k3_mla_decode_view(_attn(), meta, num_requests * tokens)
    assert isinstance(view, dict), view
    table = view["page_table"]
    assert tuple(table.shape) == (num_requests, MAX_PAGES) and table.stride() == (2 * MAX_PAGES, 1)
    assert table.data_ptr() == meta.kv_cache_block_offsets[1, 0, 0].data_ptr()
    assert torch.equal(table, meta.kv_cache_block_offsets[1, :num_requests, 0])
    assert view["seq_len"].data_ptr() == meta.kv_lens_cuda_runtime.data_ptr()
    assert view["seq_len"].shape == (num_requests,)
    assert (view["num_requests"], view["tokens_per_request"]) == (num_requests, tokens)
    assert view["row_stride"] == 576 and view["page_offset"] == 0
    assert view["pool"].numel() == POOL_PAGES * 64 * 576
    assert view["softmax_scale"] == pytest.approx(192**-0.5)


@pytest.mark.cpu_only
def test_view_interleaved_pool():
    """A layer's view of a layer-interleaved pool: the whole pool with the layer's slot as the page offset."""
    layers = 3
    pools = torch.zeros(POOL_PAGES, layers, 1, 64, 1, 640, dtype=torch.bfloat16)
    view = k3_mla_decode_view(_attn(), _meta(2, pool=pools[:, 2]), 16)
    assert isinstance(view, dict), view
    assert view["page_offset"] == 2 and view["row_stride"] == 640
    assert view["pool"].numel() == POOL_PAGES * layers * 64 * 640


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_generations,num_tokens,overrides,attn_heads",
    [
        (2, 16, dict(num_contexts=1), 6),  # a context request
        (0, 8, {}, 6),  # no generation request
        (9, 9, {}, 6),  # more than 8 requests
        (1, 16, {}, 6),  # more than 8 tokens per request
        (2, 7, {}, 6),  # tokens not uniform over the requests
        (2, 8, dict(beam_width=2), 6),
        (2, 8, dict(is_spec_dec_tree=True), 6),
        (2, 8, dict(tokens_per_block=32), 6),
        (2, 8, dict(helix_position_offsets=torch.zeros(1)), 6),
        (2, 8, dict(kv_cache_manager=None), 6),
        (2, 8, {}, 8),  # heads not a multiple of 6
    ],
)
def test_view_reasons(num_generations, num_tokens, overrides, attn_heads):
    """Every step the kernels cannot take returns a reason string (and does not raise)."""
    overrides = dict(overrides)
    num_contexts = overrides.pop("num_contexts", 0)
    reason = k3_mla_decode_view(
        _attn(attn_heads), _meta(num_generations, num_contexts, **overrides), num_tokens
    )
    assert isinstance(reason, str) and reason


@pytest.mark.cpu_only
def test_view_pool_reasons():
    """A quantized or differently laid out pool returns a reason string."""
    fp8 = torch.zeros(POOL_PAGES, 1, 64, 1, 576, dtype=torch.float8_e4m3fn)
    assert isinstance(k3_mla_decode_view(_attn(), _meta(1, pool=fp8), 1), str)
    narrow = torch.zeros(POOL_PAGES, 1, 64, 1, 512, dtype=torch.bfloat16)
    assert isinstance(k3_mla_decode_view(_attn(), _meta(1, pool=narrow), 1), str)
