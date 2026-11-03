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
"""Inline NVFP4 latent records (kv_cache_config.dtype="nvfp4_ds_mla"): writer, decode and prefill on PrimTS."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import prims_ts_dsa_mla as dsa_lib
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
    get_prims_ts_toolchain_unsupported_reason,
)
from tensorrt_llm._torch.attention.backends.sparse import nvfp4_ds_mla_kv as nv
from tensorrt_llm._torch.attention.backends.sparse.dsa import prims_ts_prefill as ptp
from tensorrt_llm._utils import get_sm_version

TPB, LAYERS, LAYER = 64, 3, 1
NOPE, ROPE, LORA, V, K = 128, 64, 512, 128, 2048
D = LORA + ROPE


def _skip_unless_kernels():
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("PrimTS sparse MLA requires SM100 or SM103")
    reason = get_prims_ts_toolchain_unsupported_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.fixture
def record_amax(monkeypatch):
    monkeypatch.delenv(nv.NOPE_AMAX_ENV, raising=False)
    monkeypatch.delenv(nv.ROPE_AMAX_ENV, raising=False)
    nv.record_scales.cache_clear()
    yield nv.record_scales()
    nv.record_scales.cache_clear()


def test_record_scales_from_amax(monkeypatch):
    monkeypatch.delenv(nv.NOPE_AMAX_ENV, raising=False)
    monkeypatch.delenv(nv.ROPE_AMAX_ENV, raising=False)
    nv.record_scales.cache_clear()
    sc = nv.record_scales()
    assert (sc.nope, sc.rope) == (
        nv.DEFAULT_NOPE_AMAX / (6.0 * 448.0),
        nv.DEFAULT_ROPE_AMAX / 448.0,
    )
    monkeypatch.setenv(nv.NOPE_AMAX_ENV, "-1")
    nv.record_scales.cache_clear()
    with pytest.raises(ValueError, match=nv.NOPE_AMAX_ENV):
        nv.record_scales()
    nv.record_scales.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("scales", [nv.RecordScales(), nv.RecordScales(nope=1.5, rope=0.25)])
@pytest.mark.parametrize("rows", [1, 37, 300])
def test_quant_scatter_matches_reference(rows, scales):
    """The Triton writer produces the reference records byte for byte (incl. ties, tiny / large blocks)."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, D, generator=g, device="cuda") * 2
    x[:, :16] = 0.0  # all-zero block: the 2^-9 scale floor
    x[:, 16:32] = 1e-5 * torch.randn(rows, 16, generator=g, device="cuda")
    x[:, 32:48] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2, device="cuda")
    x[:, 48:64] = 300.0 * torch.randn(rows, 16, generator=g, device="cuda")
    rows_bf16 = x.to(torch.bfloat16)
    pool = torch.zeros(2 * rows + 3, nv.TOKEN_BYTES, dtype=torch.uint8, device="cuda")
    loc = torch.randperm(2 * rows + 3, device="cuda")[:rows].to(torch.int32)
    loc[0] = -1  # skipped
    nv.quant_scatter(pool, loc, rows_bf16, scales)
    ref = nv.quant_rows_reference(rows_bf16, scales)
    live = loc >= 0
    assert torch.equal(pool[loc[live].long()], ref[live])
    # Dequantized NoPE values within one scale step of the inputs: the widest E2M1 gap (4 -> 6) is two
    # scale units, and an E4M3-rounded scale saturates at most 6 * 2^-4 units below the block maximum.
    back = nv.dequant_rows(ref, scales)
    err = (back[:, :LORA] - rows_bf16[:, :LORA].float()).abs().reshape(rows, -1, 16)
    pos = torch.tensor([nv.scale_byte(b) for b in range(LORA // 16)], device="cuda")
    sf = ref[:, pos].contiguous().view(torch.float8_e4m3fn).float() * scales.nope
    assert (err <= sf[..., None] * (1 + 1e-6)).all()
    rope = rows_bf16[:, LORA:].float()
    assert torch.allclose(back[:, LORA:], rope, rtol=2**-4, atol=2**-9 * scales.rope)
    assert nv.scale_byte(0) == 320 and nv.scale_byte(1) == 328 and nv.scale_byte(4) == 321


class _Attn:
    """Stand-in for the TRTLLM attention of a K3.1 nvfp4_ds_mla layer."""

    def __init__(self, heads):
        self.num_heads = heads
        self.qk_nope_head_dim = NOPE
        self.qk_rope_head_dim = ROPE
        self.kv_lora_rank = LORA
        self.q_scaling = 1.0
        self.kv_scale_orig_quant = torch.ones(1, device="cuda")
        self.kv_scale_quant_orig = torch.ones(1, device="cuda")
        self.rope_is_identity = True
        self.use_nvfp4_ds_mla = True
        self.has_fp8_kv_cache = True

    def get_local_layer_idx(self, metadata):
        return LAYER


def _metadata(kv_lens, gen_rows_per_req, seed):
    """Paged FP8-container pool of records, block tables and token positions of one step."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks_per_req = [(n + TPB - 1) // TPB for n in kv_lens]
    num_blocks = sum(blocks_per_req) + 3
    pool = torch.zeros(num_blocks, LAYERS, 1, TPB, nv.TOKEN_BYTES, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(num_blocks, generator=g, device="cuda")
    table = torch.full((len(kv_lens), max(blocks_per_req)), -1, dtype=torch.int32, device="cuda")
    o = 0
    for i, nb in enumerate(blocks_per_req):
        table[i, :nb] = perm[o : o + nb].to(torch.int32)
        o += nb
    rows = sum(gen_rows_per_req)
    req_idx = torch.cat(
        [torch.full((r,), i, dtype=torch.int32) for i, r in enumerate(gen_rows_per_req)]
    ).cuda()
    positions = torch.cat(
        [torch.arange(n - r, n, dtype=torch.int32) for n, r in zip(kv_lens, gen_rows_per_req)]
    ).cuda()
    md = SimpleNamespace(
        _ensure_pool_view_cached=lambda: None,
        _cached_pool_view=pool.view(torch.float8_e4m3fn).view(-1, 1, nv.TOKEN_BYTES),
        _cached_tokens_per_block=TPB,
        _cached_block_table_gen=table,
        _cached_req_idx_gen=req_idx,
        _cached_block_table_ctx=table,
        _cached_req_idx_ctx=req_idx,
        token_positions_cuda=positions,
        kv_cache_manager=SimpleNamespace(
            get_primary_pool_page_index_params=lambda local: (LAYERS, local)
        ),
        fmha_plan_caches={},
        num_ctx_tokens=0,
        num_tokens=rows,
        max_num_tokens=rows,
        helix_position_offsets=None,
        shared_topk_indices=None,
    )
    return pool, table, md


def _global_rows(table, req_idx, local):
    """Pool rows of request-local positions (the convertReqIndexToGlobal mapping, -1 kept)."""
    blk = table[req_idx.long()[:, None], (local.clamp(min=0) // TPB).long()]
    rows = blk.long() * (LAYERS * TPB) + LAYER * TPB + local.long() % TPB
    return torch.where(local >= 0, rows, -1).to(torch.int32)


def _fill_history(pool_rows, table, kv_lens, gen_rows_per_req, seed):
    """Records for every cached (non-new) token, at the process's record scales."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    for i, (n, r) in enumerate(zip(kv_lens, gen_rows_per_req)):
        pos = torch.arange(n - r, device="cuda", dtype=torch.int32)
        loc = _global_rows(table, torch.full_like(pos, i), pos[:, None]).view(-1)
        x = (torch.randn(pos.numel(), D, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        nv.quant_scatter(pool_rows, loc, x, nv.record_scales())


def _query_as_read(q, sc):
    """[rows, H, 576] FP32 query values the kernel attends with: E4M3 latent part, E4M3 RoPE part at g_rope / g."""
    r = sc.rope / sc.nope
    lat = q[..., :LORA].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    pe = (q[..., LORA:].float() * r).clamp(-448, 448).to(torch.float8_e4m3fn).float() / r
    return torch.cat([lat, pe], -1)


def _reference(query, records, idx, scale):
    kv = nv.dequant_rows(records, nv.record_scales())
    out = []
    for t in range(query.shape[0]):
        sel = idx[t][idx[t] >= 0].long()
        k = kv[sel]
        s = (query[t].float() @ k.t()) * scale
        out.append(torch.softmax(s, -1) @ k[:, :LORA])
    return torch.stack(out)


@pytest.mark.parametrize(
    "kv_lens,q_len", [([3000, 5000], 4), ([700, 2045, 2050], 1)], ids=["long", "short"]
)
def test_decode_appends_and_attends(kv_lens, q_len, record_amax):
    """Generation: the library appends the step's rows as records and attends over the records in place."""
    _skip_unless_kernels()
    heads = 96  # all heads on one attention-DP rank
    gen_rows = [q_len] * len(kv_lens)
    pool, table, md = _metadata(kv_lens, gen_rows, seed=heads + q_len)
    pool_rows = pool.view(-1, nv.TOKEN_BYTES)
    _fill_history(pool_rows, table, kv_lens, gen_rows, seed=heads)
    rows = sum(gen_rows)
    g = torch.Generator(device="cuda").manual_seed(5)
    latent = (torch.randn(rows, D, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    q = (torch.randn(rows, heads * D, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    # Request-local Top-K of each row (causal, -1 padded) -> pool rows.
    local = torch.full((rows, K), -1, dtype=torch.int32, device="cuda")
    for t in range(rows):
        n = int(md.token_positions_cuda[t]) + 1
        sel = torch.randperm(n, device="cuda")[: min(K, n)].sort().values
        local[t, : sel.numel()] = sel.to(torch.int32)
    indices = _global_rows(table, md._cached_req_idx_gen, local)
    out = torch.empty(rows, heads * LORA, dtype=torch.bfloat16, device="cuda")
    fwd = SimpleNamespace(
        update_kv_cache=True,
        latent_cache=latent,
        output=out,
        sparse_runtime_params=SimpleNamespace(sparse_attn_indices=indices),
    )
    attn = _Attn(heads)
    lib = dsa_lib.PrimsTSDsaMlaFmha(attn)
    dsa_lib._forward_nvfp4_ds_mla_impl(lib, q, md, fwd)
    # The new rows were appended.
    loc = _global_rows(table, md._cached_req_idx_gen, md.token_positions_cuda[:, None]).view(-1)
    assert torch.equal(pool_rows[loc.long()], nv.quant_rows_reference(latent, record_amax))
    query = _query_as_read(q.view(rows, heads, D), record_amax)
    scale = 1.0 / math.sqrt(NOPE + ROPE)
    ref = _reference(query, pool_rows, indices, scale)
    rel = ((out.view(rows, heads, LORA).float() - ref).norm() / ref.norm()).item()
    assert rel < 3e-2, rel


def test_prefill_appends_and_attends(record_amax):
    """Context: the chunk's rows become records and the PrimTS prefill attends over them in place."""
    _skip_unless_kernels()
    heads = 96  # all heads on one attention-DP rank
    kv_lens = [130, 1, 1000]
    chunk = [40, 1, 40]  # the last rows of each request are this chunk
    pool, table, md = _metadata(kv_lens, chunk, seed=heads)
    pool_rows = pool.view(-1, nv.TOKEN_BYTES)
    _fill_history(pool_rows, table, kv_lens, chunk, seed=heads)
    t = sum(chunk)
    g = torch.Generator(device="cuda").manual_seed(9)
    latent = (torch.randn(t, D, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    q = (torch.randn(t, heads, NOPE + ROPE, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    w_uk = (torch.randn(heads, LORA, NOPE, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    w_uv = (torch.randn(heads, V, LORA, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    local = torch.full((t, K), -1, dtype=torch.int32, device="cuda")
    for r in range(t):
        n = int(md.token_positions_cuda[r]) + 1
        sel = torch.randperm(n, device="cuda")[: min(K, n)].sort().values
        local[r, : sel.numel()] = sel.to(torch.int32)
    md.num_ctx_tokens = t
    mla = SimpleNamespace(
        num_heads_tp=heads,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        kv_lora_rank=LORA,
        v_head_dim=V,
        softmax_scale=(NOPE + ROPE) ** -0.5,
        k_b_proj_trans=w_uk,
        v_b_proj=w_uv,
        _bmm_bf16_out=lambda a, b, bt, out: torch.bmm(a, bt, out=out),
        mqa=_Attn(heads),
    )
    mla.mqa.indexer = SimpleNamespace(forward_from_projected=lambda *a, **k: local)
    out = torch.empty(t, heads * V, dtype=torch.bfloat16, device="cuda")
    ptp.forward_context_sparse_attn_nvfp4_ds_mla(mla, q.view(t, -1), latent, md, out, [])
    loc = _global_rows(table, md._cached_req_idx_ctx, md.token_positions_cuda[:, None]).view(-1)
    assert torch.equal(pool_rows[loc.long()], nv.quant_rows_reference(latent, record_amax))
    q_abs = torch.einsum("thd,hcd->thc", q[..., :NOPE].float(), w_uk.float())
    query = _query_as_read(torch.cat([q_abs, q[..., NOPE:].float()], -1), record_amax)
    indices = _global_rows(table, md._cached_req_idx_ctx, local)
    lat = _reference(query, pool_rows, indices, mla.softmax_scale)
    ref = torch.einsum("thc,hvc->thv", lat, w_uv.float())
    rel = ((out.view(t, heads, V).float() - ref).norm() / ref.norm()).item()
    assert rel < 3e-2, rel
