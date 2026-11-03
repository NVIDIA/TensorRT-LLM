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
"""DSA sparse MLA prefill on the opt-in PrimTS D576 kernel over the FP8 latent pool (Kimi K3.1)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
    get_prims_ts_toolchain_unsupported_reason,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import flash_mla_prefill as fmp
from tensorrt_llm._torch.attention.backends.sparse.dsa import prims_ts_prefill as ptp
from tensorrt_llm._utils import get_sm_version

from .test_flash_mla_prefill import _pool_metadata, _reference_rows

NOPE, ROPE, LORA, V, K = 128, 64, 512, 128, 2048


def _skip_unless_kernels():
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("PrimTS sparse MLA prefill requires SM100 or SM103")
    reason = get_prims_ts_toolchain_unsupported_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("tokens,heads", [(1, 1), (37, 96), (300, 12)])
def test_quantize_query_matches_torch(tokens, heads):
    """The fused query quantization equals saturate(x * s) cast to E4M3 for both the latent and RoPE parts."""
    g = torch.Generator(device="cuda").manual_seed(tokens)
    lat = (torch.randn(tokens, heads, LORA, generator=g, device="cuda") * 200).to(torch.bfloat16)
    q = (torch.randn(tokens, heads, NOPE + ROPE, generator=g, device="cuda") * 200).to(
        torch.bfloat16
    )
    pe = q[..., NOPE:]
    scale = torch.tensor([1.5], device="cuda")
    out = torch.empty(tokens, heads, LORA + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    ptp.quantize_query_e4m3(lat, pe, scale, out)
    ref = (torch.cat([lat, pe], dim=-1).float() * 1.5).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


def _problem(kv_lens, heads, seed):
    """FP8 paged pool + per-row request-local Top-K (-1 padded) over several context requests."""
    pool, table, meta, mla = _pool_metadata(kv_lens, seed=seed)
    torch.manual_seed(seed)
    mla.kv_lora_rank = LORA
    mla.qk_nope_head_dim = NOPE
    mla.qk_rope_head_dim = ROPE
    mla.softmax_scale = (NOPE + ROPE) ** -0.5
    mla._bmm_bf16_out = lambda a, b, bt, out: torch.bmm(a, bt, out=out)
    mla.mqa.kv_scale_orig_quant = torch.full((1,), 2.0, device="cuda")
    rows_per_req = [min(n, 40) for n in kv_lens]
    req_idx = torch.cat(
        [torch.full((r,), i, dtype=torch.int32) for i, r in enumerate(rows_per_req)]
    ).cuda()
    topk = torch.full((len(req_idx), K), -1, dtype=torch.int32, device="cuda")
    r = 0
    for i, (n, nr) in enumerate(zip(kv_lens, rows_per_req)):
        for j in range(nr):
            pos = n - nr + j + 1  # causal: the row sees its own position and everything before
            sel = torch.randperm(pos, device="cuda")[: min(K, pos)].sort().values
            topk[r, : sel.numel()] = sel.to(torch.int32)
            r += 1
    t = len(req_idx)
    q = (torch.randn(t, heads, NOPE + ROPE, device="cuda") * 0.5).to(torch.bfloat16)
    w_uk = (torch.randn(heads, LORA, NOPE, device="cuda") * 0.05).to(torch.bfloat16)
    w_uv = (torch.randn(heads, V, LORA, device="cuda") * 0.05).to(torch.bfloat16)
    return pool, table, meta, mla, q, w_uk, w_uv, topk, req_idx


def _reference(q, w_uk, w_uv, kv_rows, topk, req_idx, kv_lens, scale, quant_q):
    """FP32 attention over the dequantized pool rows; the query optionally rounded to E4M3 at scale 2."""
    starts = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0)[:-1]), device="cuda")
    q_abs = torch.einsum("thd,hcd->thc", q[..., :NOPE].float(), w_uk.float())
    q_full = torch.cat([q_abs, q[..., NOPE:].float()], dim=-1)
    if quant_q:
        q_full = (q_full * 2.0).clamp(-448, 448).to(torch.float8_e4m3fn).float() / 2.0
    idx = torch.where(topk >= 0, topk + starts[req_idx.long()][:, None].int(), -1)
    sel = kv_rows[idx.clamp(min=0).long()].float()
    s = torch.einsum("thc,tkc->thk", q_full, sel) * scale
    s = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    lat = torch.einsum("thk,tkc->thc", s.softmax(-1), sel[..., :LORA])
    return torch.einsum("thc,hvc->thv", lat, w_uv.float())


@pytest.mark.parametrize("heads", [96, 12])
@pytest.mark.parametrize("kv_lens", [[130, 1, 64, 1000], [3000, 2100]], ids=["short", "long"])
def test_prefill_rows_match_reference(heads, kv_lens):
    """PrimTS prefill over the FP8 pool equals FP32 attention over the same E4M3 query / KV values (FP8 P
    tolerance) and stays within the FP8-query tolerance of the FlashMLA BF16-query path."""
    _skip_unless_kernels()
    pool, table, meta, mla, q, w_uk, w_uv, topk, req_idx = _problem(kv_lens, heads, seed=7)
    t = q.shape[0]
    out = torch.empty(t, heads, V, dtype=torch.bfloat16, device="cuda")
    ptp.prims_ts_sparse_prefill_rows(mla, q, w_uk, w_uv, topk, req_idx, meta, out)
    kv_rows = _reference_rows(pool, table, kv_lens, 1, 0.5)
    ref8 = _reference(q, w_uk, w_uv, kv_rows, topk, req_idx, kv_lens, mla.softmax_scale, True)
    rel8 = ((out.float() - ref8).norm() / ref8.norm()).item()
    assert rel8 < 3e-2, rel8

    if fmp.flash_mla_sparse_fwd is not None:
        kv = fmp.gather_context_kv_bf16(mla, meta)
        idx = fmp.rebase_topk_to_compact_kv(topk, req_idx, meta)
        flash = torch.empty_like(out)
        fmp.flash_mla_sparse_prefill_rows(mla, q, w_uk, w_uv, kv, idx, flash)
        rel = ((out.float() - flash.float()).norm() / flash.float().norm()).item()
        assert rel < 5e-2, rel


def test_prefill_unsupported_without_opt_in(monkeypatch):
    monkeypatch.delenv(ptp.SPARSE_MLA_PREFILL_ENV, raising=False)
    mla = SimpleNamespace()
    assert not ptp.prims_ts_prefill_supported(mla, 96)
