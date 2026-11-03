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
"""DSA FlashMLA prefill over an FP8 cache: the compact context KV gather, the top-k rebase and the rows."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import flash_mla_prefill as fmp
from tensorrt_llm._utils import get_sm_version

_TPB = 64
_D = 576


def _pool_metadata(kv_lens, *, num_layers=3, layer=1, fp8=True, seed=0):
    """A paged pool [blocks, layers, 1, TPB, D] with the requests' pages at random physical blocks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks_per_req = [(n + _TPB - 1) // _TPB for n in kv_lens]
    num_blocks = sum(blocks_per_req) + 5
    raw = torch.randn(num_blocks, num_layers, 1, _TPB, _D, generator=g, device="cuda")
    pool = raw.to(torch.float8_e4m3fn) if fp8 else raw.to(torch.bfloat16)
    perm = torch.randperm(num_blocks, generator=g, device="cuda")
    table = torch.full((len(kv_lens), max(blocks_per_req)), -1, dtype=torch.int32, device="cuda")
    o = 0
    for i, nb in enumerate(blocks_per_req):
        table[i, :nb] = perm[o : o + nb].to(torch.int32)
        o += nb
    indptr = torch.zeros(len(kv_lens) + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(torch.tensor(kv_lens), 0)
    meta = SimpleNamespace(
        _ensure_pool_view_cached=lambda: None,
        _cached_pool_view=pool.squeeze(2).view(-1, 1, _D),
        _cached_tokens_per_block=_TPB,
        _cached_block_table_ctx=table,
        num_ctx_mla_kv_tokens=int(indptr[-1]),
        num_contexts=len(kv_lens),
        ctx_kv_indptr=indptr.cuda(),
        kv_cache_manager=SimpleNamespace(
            get_primary_pool_page_index_params=lambda local: (num_layers, local)
        ),
    )
    mla = SimpleNamespace(
        mqa=SimpleNamespace(
            get_local_layer_idx=lambda m: layer,
            kv_scale_quant_orig=torch.full((1,), 0.5, device="cuda"),
        )
    )
    return pool, table, meta, mla


def _reference_rows(pool, table, kv_lens, layer, scale):
    rows = []
    for i, n in enumerate(kv_lens):
        pos = torch.arange(n, device="cuda")
        blk = table[i, pos // _TPB].long()
        rows.append(pool[blk, layer, 0, pos % _TPB].float() * scale)
    return torch.cat(rows).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("fp8", [True, False])
def test_gather_context_kv_matches_paged_rows(fp8):
    kv_lens = [130, 1, 64, 1000]
    pool, table, meta, mla = _pool_metadata(kv_lens, fp8=fp8)
    out = fmp.gather_context_kv_bf16(mla, meta)
    ref = _reference_rows(pool, table, kv_lens, 1, 0.5 if fp8 else 1.0)
    assert out.shape == (sum(kv_lens), 1, _D)
    torch.testing.assert_close(out.view(-1, _D), ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gather_context_kv_row_range():
    kv_lens = [130, 70, 1000]
    pool, table, meta, mla = _pool_metadata(kv_lens, seed=3)
    ref = _reference_rows(pool, table, kv_lens, 1, 0.5)
    out = fmp.gather_context_kv_bf16(mla, meta, 130, 1200)
    torch.testing.assert_close(out.view(-1, _D), ref[130:1200], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_rebase_topk_masks_out_of_request_positions():
    kv_lens = [100, 7, 300]
    _, _, meta, _ = _pool_metadata(kv_lens)
    topk = torch.tensor(
        [[0, 99, 100, -1], [3, 6, 7, 2], [299, 0, -5, 150]], dtype=torch.int32, device="cuda"
    )
    req = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    out = fmp.rebase_topk_to_compact_kv(topk, req, meta).view(3, 4).cpu()
    assert out.tolist() == [[0, 99, -1, -1], [103, 106, -1, 102], [406, 107, -1, 257]]
    shifted = fmp.rebase_topk_to_compact_kv(topk[1:], req[1:], meta, row_lo=100).view(2, 4).cpu()
    assert shifted.tolist() == [[3, 6, -1, 2], [306, 7, -1, 157]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("tokens,heads,dim", [(1, 1, 64), (77, 96, 64), (300, 12, 64), (5, 3, 48)])
def test_copy_head_rows_matches_strided_copy(tokens, heads, dim):
    """copy_head_rows writes the rope part of the padded FlashMLA q exactly as Tensor.copy_ (48 takes the
    copy_ fallback: not a power of two)."""
    g = torch.Generator(device="cuda").manual_seed(tokens)
    q = torch.randn(tokens, heads, 128 + dim, generator=g, device="cuda").to(torch.bfloat16)
    ref = torch.randn(tokens, 128, 512 + dim, generator=g, device="cuda").to(torch.bfloat16)
    out = ref.clone()
    ref[:, :heads, 512:].copy_(q[:, :, 128:])
    fmp.copy_head_rows(q[:, :, 128:], out[:, :heads, 512:])
    assert torch.equal(out, ref)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or fmp.flash_mla_sparse_fwd is None
    or get_sm_version() not in (100, 103),
    reason="needs FlashMLA on SM100 / SM103",
)
@pytest.mark.parametrize("num_heads", [96, 12])
def test_flash_mla_rows_match_reference(num_heads):
    torch.manual_seed(0)
    n, rows, k, nope, rope, lora, v = 70, 3000, 2048, 128, 64, 512, 128
    mla = SimpleNamespace(
        kv_lora_rank=lora,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        softmax_scale=(nope + rope) ** -0.5,
        _bmm_bf16_out=lambda a, b, bt, out: torch.bmm(a, bt, out=out),
    )
    q = (torch.randn(n, num_heads, nope + rope, device="cuda") * 0.5).to(torch.bfloat16)
    w_uk = (torch.randn(num_heads, lora, nope, device="cuda") * 0.05).to(torch.bfloat16)
    w_uv = (torch.randn(num_heads, v, lora, device="cuda") * 0.05).to(torch.bfloat16)
    kv = (torch.randn(rows, 1, lora + rope, device="cuda") * 0.5).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(rows, device="cuda")[:k] for _ in range(n)]).to(torch.int32)
    idx[:, k - 100 :] = -1
    out = torch.empty(n, num_heads, v, dtype=torch.bfloat16, device="cuda")
    fmp.flash_mla_sparse_prefill_rows(mla, q, w_uk, w_uv, kv, idx.view(n, 1, k), out)

    q_abs = torch.einsum("thd,hcd->thc", q[..., :nope].float(), w_uk.float())
    q_full = torch.cat([q_abs, q[..., nope:].float()], dim=-1)
    sel = kv[idx.clamp(min=0).long(), 0].float()  # [n, k, 576]
    s = torch.einsum("thc,tkc->thk", q_full, sel) * mla.softmax_scale
    s = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    p = s.softmax(-1)
    lat = torch.einsum("thk,tkc->thc", p, sel[..., :lora])
    ref = torch.einsum("thc,hvc->thv", lat, w_uv.float())
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
