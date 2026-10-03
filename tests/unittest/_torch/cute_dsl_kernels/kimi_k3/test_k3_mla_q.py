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
"""trtllm::k3_mla_q / k3_mla_qkv / k3_mla_qkv_out (Kimi K3 MLA decode query path: q_a RMSNorm, q_b, k_b absorb, and the
KV half into the paged latent cache) at the TP16 shape (6 heads per rank, q_lora 1536, latent 512 + rope 64), M <= 64
tokens: fused_q against the unfused chain (the model's RMSNorm, the q_b GEMM, the k_b bmm) and a reference with
the same bf16 roundings, every 8-token chunk bit-identical to the 8-token call on its rows; the KV rows of R requests
x T tokens at each request's positions in a sentinel-filled pool, nothing else written."""

import pytest
import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in (100, 103)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100 / SM103 GPU")

H, NOPE, PE, QK, LAT, QL, V = 6, 128, 64, 192, 512, 1536, 128
DQK, PAGE = LAT + PE, 64
EPS = KV_EPS = 1e-6
SENTINEL = 0x7F7F  # bf16 bits of the largest finite value: no cache row of the tests holds it

SPLITS = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (8, 1), (1, 8), (2, 4), (4, 2)] + [
    (r, 8) for r in range(2, 9)
]
SPLITS += [(3, 5), (7, 3)]  # chunks of 8 tokens that cut through requests
SPLIT_IDS = [f"{r}x{t}" for r, t in SPLITS]
# KV lengths, assigned to the requests of a step in turn: the step's rows crossing a page boundary (579, 1989), only
# the step's tokens (8), short and long contexts.
LENGTHS = (1100, 64 * 9 + 3, 8, 2049, 127, 4100, 64 * 31 + 5, 300, 64)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _weights(seed, heads=H):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w_qa = (1.0 + 0.1 * torch.randn(QL, generator=gen, device="cuda")).bfloat16()
    w_qb = (torch.randn(heads * QK, QL, generator=gen, device="cuda") * 0.03).bfloat16()
    w_kb = (torch.randn(heads, LAT, NOPE, generator=gen, device="cuda") * 0.08).bfloat16()
    w_kv = (1.0 + 0.1 * torch.randn(LAT, generator=gen, device="cuda")).bfloat16()
    return w_qa, w_qb, w_kb, w_kv


def _ag(seed, m, heads=H):
    """The fused projection's rows: [q_a 1536 | kv_a latent 512 | rope 64 | gate heads * 128]."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, QL + DQK + heads * V, generator=gen, device="cuda") * 0.7).bfloat16()


def _rms_norm(x, w, eps):
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    norm = RMSNorm(hidden_size=x.shape[1], eps=eps, dtype=torch.bfloat16).cuda()
    norm.weight.data.copy_(w)
    return norm(x.contiguous())


def _unfused_q(ag, w_qa, w_qb, w_kb):
    """The model's unfused chain: q_a_layernorm, q_b_proj (GEMM), bmm with k_b_proj_trans, q_pe copied."""
    m, heads = ag.shape[0], w_kb.shape[0]
    q = torch.matmul(_rms_norm(ag[:, :QL], w_qa, EPS), w_qb.t()).view(m, heads, QK)
    q_abs = torch.bmm(q[..., :NOPE].transpose(0, 1), w_kb.transpose(1, 2)).transpose(0, 1)
    return torch.cat([q_abs, q[..., NOPE:]], dim=-1).reshape(m, heads * DQK)


def _reference_q(ag, w_qa, w_qb, w_kb):
    """The norm in fp32, the GEMMs in float64 (cuBLAS may run fp32 GEMMs in TF32), bf16 rounding where the model
    rounds (norm output, q_b output, q_abs)."""
    m, heads = ag.shape[0], w_kb.shape[0]
    x = ag[:, :QL].float()
    qn = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w_qa.float()).bfloat16()
    q = (qn.double() @ w_qb.double().t()).bfloat16().view(m, heads, QK)
    q_abs = torch.einsum("thd,hcd->thc", q[..., :NOPE].double(), w_kb.double()).bfloat16()
    return torch.cat([q_abs, q[..., NOPE:]], dim=-1).reshape(m, heads * DQK)


def _reference_kv(ag, w_kv):
    """The cache rows in fp32, flashinfer's order ((x * rrms) * w, one bf16 rounding); the rope columns copied."""
    x = ag[:, QL : QL + LAT].float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + KV_EPS)
    return torch.cat([((x * r) * w_kv.float()).bfloat16(), ag[:, QL + LAT : QL + DQK]], dim=-1)


def _max_rel(a, b):
    return (a.float() - b.float()).abs().max().item() / b.float().abs().max().item()


def _q(ag, w_qa, w_qb, w_kb):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op  # noqa: F401

    return torch.ops.trtllm.k3_mla_q(ag, w_qa, EPS, w_qb, w_kb, True)


# Chunks of 8 tokens up to 24, of 16 up to 48 and of 32 up to 64 (op.CHUNK_TOKENS), with full and short last chunks.
# The TP4 shape (24 heads, 144 CTAs per chunk) at M 8, 40 and 64.
Q_CASES = [(m, H) for m in (1, 2, 3, 5, 8, 9, 15, 16, 24, 25, 31, 32, 33, 40, 48, 49, 57, 64)] + [
    (m, 4 * H) for m in (8, 40, 64)
]


@pytest.mark.parametrize("m,heads", Q_CASES, ids=[f"m{m}-h{h}" for m, h in Q_CASES])
def test_q(m, heads):
    """fused_q against the reference and the unfused chain (q_abs and q_pe), rerun; each 8-token chunk of rows
    bit-identical to the call on those rows alone."""
    w_qa, w_qb, w_kb, _ = _weights(20261001, heads)
    ag = _ag(m, m, heads)
    y = _q(ag, w_qa, w_qb, w_kb)
    again = _q(ag, w_qa, w_qb, w_kb)
    chunks = torch.cat([_q(ag[c : c + 8].contiguous(), w_qa, w_qb, w_kb) for c in range(0, m, 8)])
    yu = _unfused_q(ag, w_qa, w_qb, w_kb)
    yr = _reference_q(ag, w_qa, w_qb, w_kb)
    torch.cuda.synchronize()
    for part in (slice(0, LAT), slice(LAT, DQK)):
        yk = y.view(m, heads, DQK)[..., part]
        assert _max_rel(yk, yr.view(m, heads, DQK)[..., part]) <= 2e-2
        assert _max_rel(yk, yu.view(m, heads, DQK)[..., part]) <= 2e-2
    assert torch.equal(_bits(y), _bits(again))
    assert torch.equal(_bits(y), _bits(chunks))


def _kv_case(seed, num_requests, tokens, row_stride=DQK, layers=1, lens=None):
    """A sentinel-filled pool of `layers` interleaved layer slots (rows of `row_stride`), the page table as rows of an
    int32 [R, 2, W] buffer (kv_cache_block_offsets' layout; entries past a request's pages name a page no request
    owns), the lengths."""
    cpu_gen = torch.Generator().manual_seed(seed)
    if lens is None:
        lens = [max(tokens, LENGTHS[(i + seed) % len(LENGTHS)]) for i in range(num_requests)]
    pages = [(n + PAGE - 1) // PAGE for n in lens]
    total_pages = sum(pages) + 5
    width = max(pages) + 2
    pool = torch.full(
        (total_pages * layers * PAGE * row_stride,), SENTINEL, dtype=torch.int16, device="cuda"
    )
    perm = (torch.randperm(total_pages, generator=cpu_gen) * layers).to(torch.int32)
    offsets = torch.empty(num_requests, 2, width, dtype=torch.int32).fill_(int(perm[-1]))
    start = 0
    for i, n in enumerate(pages):
        offsets[i, :, :n] = perm[start : start + n]
        start += n
    return (
        pool.view(torch.bfloat16),
        offsets.cuda()[:, 0, :],
        torch.tensor(lens, dtype=torch.int32, device="cuda"),
    )


def _kv_rows(page_table, page_offset, seq_len, tokens):
    """(token, pool row) of every stored token: token u of request i at position L_i - T + u, if >= 0."""
    out = []
    for i, length in enumerate(seq_len.tolist()):
        for u in range(tokens):
            pos = length - tokens + u
            if pos >= 0:
                out.append(
                    (
                        i * tokens + u,
                        (int(page_table[i, pos // PAGE]) + page_offset) * PAGE + pos % PAGE,
                    )
                )
    return out


def _qkv(ag, weights, pool, row_stride, page_table, page_offset, seq_len):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op  # noqa: F401

    w_qa, w_qb, w_kb, w_kv = weights
    return torch.ops.trtllm.k3_mla_qkv(
        ag,
        w_qa,
        EPS,
        w_qb,
        w_kb,
        w_kv,
        KV_EPS,
        pool,
        row_stride,
        page_table,
        page_offset,
        seq_len,
        True,
    )


@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=SPLIT_IDS)
def test_qkv(num_requests, tokens):
    """The KV rows of every request at its positions (latent against the fp32 reference and the model's RMSNorm, rope
    columns bit-exact), nothing else in the pool written, fused_q bit-identical to k3_mla_q, the dense variant's rows
    bit-identical, rerun. Odd seeds: an interleaved pool (2 layers, slot 1) with rows of 640 elements."""
    seed = 13 * num_requests + tokens
    layers, slot, row_stride = (2, 1, 640) if seed % 2 else (1, 0, DQK)
    m = num_requests * tokens
    weights = _weights(seed)
    ag = _ag(seed + 1, m)
    pool, page_table, seq_len = _kv_case(seed, num_requests, tokens, row_stride, layers)
    y = _qkv(ag, weights, pool, row_stride, page_table, slot, seq_len)
    stored = _kv_rows(page_table, slot, seq_len, tokens)
    toks = torch.tensor([t for t, _ in stored], device="cuda")
    rows = torch.tensor([r for _, r in stored], device="cuda")
    pool_rows = pool.view(-1, row_stride)
    got = pool_rows[rows, :DQK].clone()
    rest = pool_rows.clone()
    rest[rows] = torch.full_like(rest[rows].view(torch.int16), SENTINEL).view(torch.bfloat16)
    ref = _reference_kv(ag, weights[3])[toks]
    model = _rms_norm(ag[:, QL : QL + LAT], weights[3], KV_EPS)[toks]
    dense = torch.empty(m, DQK, dtype=torch.bfloat16, device="cuda")
    y_dense = torch.ops.trtllm.k3_mla_qkv_out(
        ag, weights[0], EPS, weights[1], weights[2], weights[3], KV_EPS, dense, True
    )
    y_q = _q(ag, *weights[:3])
    _qkv(ag, weights, pool, row_stride, page_table, slot, seq_len)
    torch.cuda.synchronize()
    assert len(stored) == m
    assert _max_rel(got[:, :LAT], ref[:, :LAT]) <= 1e-2
    assert int((_bits(got[:, :LAT]) != _bits(model)).sum()) <= max(4, model.numel() // 1000)
    assert torch.equal(_bits(got[:, LAT:]), _bits(ag[toks, QL + LAT : QL + DQK]))
    assert bool((rest.view(torch.int16) == SENTINEL).all())
    assert torch.equal(_bits(y), _bits(y_q)) and torch.equal(_bits(y_dense), _bits(y_q))
    assert torch.equal(_bits(dense[toks]), _bits(got))
    assert torch.equal(_bits(pool_rows[rows, :DQK]), _bits(got))


def test_qkv_short_length():
    """A request whose length is below T (positions < 0) has only its tokens at positions >= 0 stored."""
    num_requests, tokens = 2, 4
    weights = _weights(5)
    ag = _ag(6, num_requests * tokens)
    pool, page_table, seq_len = _kv_case(7, num_requests, tokens, lens=[2, 100])
    _qkv(ag, weights, pool, DQK, page_table, 0, seq_len)
    stored = _kv_rows(page_table, 0, seq_len, tokens)
    rows = torch.tensor([r for _, r in stored], device="cuda")
    pool_rows = pool.view(-1, DQK)
    ref = _reference_kv(ag, weights[3])[[t for t, _ in stored]]
    rest = pool_rows.clone()
    rest[rows] = torch.full_like(rest[rows].view(torch.int16), SENTINEL).view(torch.bfloat16)
    torch.cuda.synchronize()
    assert len(stored) == 6
    assert _max_rel(pool_rows[rows, :LAT], ref[:, :LAT]) <= 1e-2
    assert bool((rest.view(torch.int16) == SENTINEL).all())


def test_q_rejects():
    """More than 64 tokens, or page-table rows / lengths that do not describe R requests of the call's tokens."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op

    w_qa, w_qb, w_kb, w_kv = _weights(1)
    assert op.supports_q(_ag(1, 64), w_qa, w_qb, w_kb)
    assert not op.supports_q(_ag(1, 65), w_qa, w_qb, w_kb)
    pool, page_table, seq_len = _kv_case(2, 2, 4)
    ag = _ag(2, 8)
    assert op.supports_kv(ag, w_kv, pool, DQK, page_table, seq_len)
    assert not op.supports_kv(
        ag[:7], w_kv, pool, DQK, page_table, seq_len
    )  # 7 tokens over 2 requests
    assert not op.supports_kv(ag, w_kv, pool, DQK, page_table[:1], seq_len)  # 1 row, 2 lengths
    assert not op.supports_kv(ag, w_kv, pool, DQK, page_table[0], seq_len)  # a flat row, 2 lengths
    assert not op.supports_kv(ag, w_kv, pool, DQK, page_table.long(), seq_len)  # int64 table
