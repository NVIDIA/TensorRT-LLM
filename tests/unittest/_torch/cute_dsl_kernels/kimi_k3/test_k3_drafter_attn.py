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
"""``trtllm::k3_drafter_attn`` / ``k3_drafter_attn_qknorm`` (the Kimi K3 DSpark drafter's block attention) for R
requests of T tokens, at the in-model TP16 group (6 query heads, 1 KV head; TP4 24 / 4 for a few splits), head dim 64,
HND pages of 64.

Every split R x T (R T <= 8, and R x 8 and R x 7 for R <= 8) with mixed per-request context lengths (blocks crossing a
page, a 128-row tile and the cluster's 16-tile round, no context, several tiles per CTA), page tables as strided row
views and as dense rows, against an fp32 reference and the model's production path (flashinfer append_paged_kv_cache
+ trtllm-gen batch_context_with_kv_cache, non-causal), plus: no NaN (the pool's unused rows are NaN), the pool left
untouched, reruns bit-identical, requests isolated (a change in one request's context or block changes only its
rows), CUDA-graph replays with rewritten inputs.

Error table: ``python3 test_k3_drafter_attn.py report``.
"""

import sys

import pytest
import torch

D = 64
PAGE = 64
EPS = 1e-5
THETA = 10000.0
TOL = 1e-2  # max |err| / max |ref|; bf16 P and output roundings are ~4e-3

# Every split the engine schedules for the drafter: R requests x T tokens with R T <= 8, R x 8, and DSpark's R x 7
# (its block under shift_label is max_draft_len tokens).
SPLITS = (
    [(1, 8), (2, 4), (4, 2), (8, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    + [(r, 8) for r in range(2, 9)]
    + [(r, 7) for r in range(1, 9)]
)
# Context lengths, cycled over the requests: blocks crossing a page (60, 121), a 128-row tile (124, 127), the
# cluster's 16-tile round (2040, 2041); starting a page / tile (0, 64, 128, 1024, 2048); several tiles per CTA.
LENGTHS = [60, 2040, 5, 124, 1000, 0, 2041, 64, 3000, 127, 1024, 121, 128, 4000, 2048, 1500]


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")


def _op():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_drafter import op

    return op


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


class Case:
    """R requests of T tokens: an HND pool [pages, 2, kv, 64, 64] (NaN except each request's context rows, optionally
    with a page stride larger than a page), distinct random pages per request, page-table rows, qkv and lengths."""

    def __init__(
        self, gen, heads, kv, num_requests, tokens, lengths, page_pad=0, strided_table=True
    ):
        self.heads, self.kv, self.r, self.t = heads, kv, num_requests, tokens
        self.lengths = list(lengths)
        need = [(c + tokens + PAGE - 1) // PAGE for c in self.lengths]
        self.width = max(need) + 2
        n_pool = sum(need) + 2 * num_requests + 8
        inner = 2 * kv * PAGE * D
        flat = torch.full(
            (n_pool * (inner + page_pad),), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        self.pool = flat.as_strided(
            (n_pool, 2, kv, PAGE, D), (inner + page_pad, kv * PAGE * D, PAGE * D, D, 1)
        )
        perm = torch.randperm(
            n_pool, generator=torch.Generator().manual_seed(int(sum(need)) + num_requests)
        )
        spare = perm[sum(need) :]
        rows, used = [], 0
        for r, n in enumerate(need):
            own = perm[used : used + n]
            used += n
            # Entries past the request's pages: spare (NaN) pages, never read by the kernel.
            rows.append(
                torch.cat([own, spare[(torch.arange(self.width - n) + 2 * r) % spare.numel()]])
            )
        dense = torch.stack(rows).to(torch.int32).cuda()
        if strided_table:
            # Rows 1 .. R of a wider table: a row stride larger than the width and a nonzero offset.
            full = torch.full(
                (num_requests + 2, self.width + 3), -1, dtype=torch.int32, device="cuda"
            )
            full[1 : num_requests + 1, : self.width] = dense
            self.table = full[1 : num_requests + 1, : self.width]
        else:
            self.table = dense
        for r, c in enumerate(self.lengths):
            for i in range((c + PAGE - 1) // PAGE):
                n_rows = min(PAGE, c - i * PAGE)
                p = int(self.table[r, i])
                self.pool[p, :, :, :n_rows] = (
                    torch.randn(2, kv, n_rows, D, generator=gen, device="cuda") * 0.5
                ).bfloat16()
        m = num_requests * tokens
        self.qkv = (
            torch.randn(m, (heads + 2 * kv) * D, generator=gen, device="cuda") * 0.5
        ).bfloat16()
        self.ctx_len = torch.tensor(self.lengths, dtype=torch.int32, device="cuda")
        self.positions = (
            self.ctx_len.view(-1, 1) + torch.arange(tokens, device="cuda", dtype=torch.int32)
        ).reshape(-1)

    def run(self, qkv=None, pool=None, table=None):
        out = torch.empty(self.r * self.t, self.heads * D, dtype=torch.bfloat16, device="cuda")
        torch.ops.trtllm.k3_drafter_attn(
            self.qkv if qkv is None else qkv, self.pool if pool is None else pool,
            self.table if table is None else table, self.ctx_len, self.heads, self.kv, out,
        )  # fmt: skip
        return out

    def rows(self, r):
        return slice(r * self.t, (r + 1) * self.t)


def production(case: Case) -> torch.Tensor:
    """The model's DFlash TRTLLM path for a non-causal layer (modeling_dflash): append the blocks' K / V into the pool
    (a copy, NaN rows zeroed: the context kernel reads the last page's rows past L), then the batched context kernel."""
    from tensorrt_llm._torch.speculative.dflash_attention import get_dflash_trtllm_gen_ops

    ops = get_dflash_trtllm_gen_ops()
    h, kv, b, t = case.heads, case.kv, case.r, case.t
    pool = torch.nan_to_num(case.pool.clone(), nan=0.0)
    table = case.table.contiguous()
    seq_after = case.ctx_len + t
    ws = ops.get_workspace_size(dtype=torch.bfloat16, num_tokens=b * t, num_gen_tokens=b * t, num_heads=h,
                                num_kv_heads=kv, head_size=D, max_num_requests=b, rotary_embedding_dim=0,
                                fp8_context_fmha=False)  # fmt: skip
    workspace = torch.empty(ws, dtype=torch.uint8, device="cuda")
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    counters = torch.zeros(
        ops.get_multi_ctas_kv_counter_size(h, b, sm), dtype=torch.uint8, device="cuda"
    )
    qkv = case.qkv
    ops.append_paged_kv_cache(
        append_key=qkv[:, h * D : (h + kv) * D].reshape(-1, kv, D),
        append_value=qkv[:, (h + kv) * D :].reshape(-1, kv, D),
        batch_indices=torch.arange(b, dtype=torch.int32, device="cuda").repeat_interleave(t),
        positions=case.positions, paged_kv_cache=pool, kv_indices=table.flatten(),
        kv_indptr=torch.arange(0, (b + 1) * case.width, case.width, dtype=torch.int32, device="cuda"),
        kv_last_page_len=((seq_after - 1) % PAGE) + 1, kv_layout="HND",
    )  # fmt: skip
    out = torch.empty(b * t, h * D, dtype=torch.bfloat16, device="cuda")
    ops.batch_context_with_kv_cache(
        query=qkv[:, : h * D].reshape(-1, h, D), kv_cache=(pool[:, 0], pool[:, 1]), workspace_buffer=workspace,
        block_tables=table, seq_lens=seq_after, max_q_len=t, max_kv_len=case.width * PAGE, bmm1_scale=D**-0.5,
        bmm2_scale=1.0, batch_size=b,
        cum_seq_lens_q=torch.arange(0, (b + 1) * t, t, dtype=torch.int32, device="cuda"),
        cum_seq_lens_kv=torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                                   seq_after.cumsum(0, dtype=torch.int32)]),
        window_left=-1, out=out.view(-1, h, D), sinks=None, enable_pdl=False, kv_layout="HND", kv_cache_sf=None,
        uses_shared_paged_kv_idx=True, causal=False, multi_ctas_kv_counter_buffer=counters,
    )  # fmt: skip
    return out


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return (a.float() - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-6)


def case_lengths(num_requests: int, offset: int):
    return [LENGTHS[(offset + 5 * r) % len(LENGTHS)] for r in range(num_requests)]


def measure_split(num_requests, tokens, offset, heads=6, kv=1):
    """One split's checks: errors per request against the fp32 reference and the production path, NaN, pool
    untouched, rerun, isolation, the device-length reference."""
    op = _op()
    gen = torch.Generator(device="cuda").manual_seed(
        1000 * num_requests + 10 * tokens + offset + heads
    )
    case = Case(gen, heads, kv, num_requests, tokens, case_lengths(num_requests, offset),
                page_pad=0 if offset % 2 == 0 else 3 * 2 * kv * PAGE * D, strided_table=offset != 3)  # fmt: skip
    before = case.pool.clone()
    out = case.run()
    torch.cuda.synchronize()
    res = dict(split=f"{num_requests}x{tokens}", heads=f"{heads}/{kv}", lengths=case.lengths,
               table="strided" if offset != 3 else "dense", page_pad=offset % 2 == 1)  # fmt: skip
    res["untouched"] = torch.equal(bits(case.pool), bits(before))
    res["nan"] = bool(torch.isnan(out.float()).any())
    ref = op.reference(case.qkv, case.pool, case.table, case.lengths, heads, kv)
    prod = production(case)
    res["err"] = [rel_err(out[case.rows(r)], ref[case.rows(r)]) for r in range(num_requests)]
    res["err_prod"] = [rel_err(prod[case.rows(r)], ref[case.rows(r)]) for r in range(num_requests)]
    res["err_vs_prod"] = [
        rel_err(out[case.rows(r)], prod[case.rows(r)]) for r in range(num_requests)
    ]
    res["abs_err"] = (out.float() - ref).abs().max().item()
    res["bits_vs_prod"] = int((bits(out) != bits(prod)).sum())
    # The device-length reference (what a CUDA-graph check would use) agrees with the host-length one.
    masked = op.reference_masked(
        case.qkv, torch.nan_to_num(case.pool, nan=0.0), case.table, case.ctx_len, heads, kv
    )
    res["masked_ref"] = rel_err(masked, ref)
    res["rerun"] = torch.equal(bits(case.run()), bits(out))
    # Isolation: request 0's context row and the last request's block V change only their own rows (a block's V
    # always reaches its outputs; its K does not when the block is a request's only key).
    iso = True
    if case.lengths[0] > 0:
        pool_c = case.pool.clone()
        row = case.lengths[0] // 2
        pool_c[int(case.table[0, row // PAGE]), 0, 0, row % PAGE] += 1.0
        out_c = case.run(pool=pool_c)
        iso &= not torch.equal(bits(out_c[case.rows(0)]), bits(out[case.rows(0)]))
        iso &= torch.equal(bits(out_c[tokens:]), bits(out[tokens:]))
    qkv_c = case.qkv.clone()
    qkv_c[num_requests * tokens - 1, (heads + kv) * D + 5] += 1.0
    out_c = case.run(qkv=qkv_c)
    last = case.rows(num_requests - 1)
    iso &= not torch.equal(bits(out_c[last]), bits(out[last]))
    iso &= torch.equal(bits(out_c[: last.start]), bits(out[: last.start]))
    res["isolated"] = iso
    res["ok"] = (res["untouched"] and not res["nan"] and max(res["err"]) <= TOL and max(res["err_vs_prod"]) <= TOL
                 and res["masked_ref"] <= 1e-5 and res["rerun"] and iso)  # fmt: skip
    return res


@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
@pytest.mark.parametrize("offset", [0, 3, 7])
def test_split(num_requests, tokens, offset):
    with torch.inference_mode():
        res = measure_split(num_requests, tokens, offset)
    assert res["ok"], res


TP4_SPLITS = [(1, 8), (2, 4), (8, 1), (4, 8), (8, 8)]


@pytest.mark.parametrize("num_requests,tokens", TP4_SPLITS, ids=[f"{r}x{t}" for r, t in TP4_SPLITS])
def test_split_tp4(num_requests, tokens):
    """TP4's group (24 query heads, 4 KV heads: four clusters per request)."""
    with torch.inference_mode():
        res = measure_split(num_requests, tokens, 1, heads=24, kv=4)
    assert res["ok"], res


def qk_norm_rope(qkv, heads, kv, q_w, k_w, positions):
    """The model's fused_qk_norm_rope, in place (plain NeoX RoPE)."""
    torch.ops.trtllm.fused_qk_norm_rope(
        qkv,
        heads,
        kv,
        kv,
        D,
        D,
        EPS,
        q_w,
        k_w,
        THETA,
        True,
        positions,
        1.0,
        0.0,
        0.0,
        1.0,
        True,
        False,
        False,
        0,
        0,
    )
    return qkv


@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
def test_split_qknorm(num_requests, tokens):
    """k3_drafter_attn_qknorm (q/k RMSNorm + RoPE in the kernel, raw qkv) against fused_qk_norm_rope + k3_drafter_attn
    and the fp32 reference; its input untouched; int64 positions give the same bits."""
    op = _op()
    gen = torch.Generator(device="cuda").manual_seed(5000 + 10 * num_requests + tokens)
    with torch.inference_mode():
        case = Case(gen, 6, 1, num_requests, tokens, case_lengths(num_requests, 2))
        raw = (case.qkv.float() * 4.0).bfloat16()
        q_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        k_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        normed = qk_norm_rope(raw.clone(), 6, 1, q_w, k_w, case.positions)
        out_a = case.run(qkv=normed)
        raw_before = raw.clone()
        out_b = torch.empty_like(out_a)
        torch.ops.trtllm.k3_drafter_attn_qknorm(raw, q_w, k_w, case.positions, EPS, THETA, case.pool, case.table,
                                                case.ctx_len, 6, 1, out_b)  # fmt: skip
        out_c = torch.empty_like(out_a)
        torch.ops.trtllm.k3_drafter_attn_qknorm(raw, q_w, k_w, case.positions.long(), EPS, THETA, case.pool,
                                                case.table, case.ctx_len, 6, 1, out_c)  # fmt: skip
        assert torch.equal(bits(raw), bits(raw_before)), "the kernel wrote its input"
        assert torch.equal(bits(out_c), bits(out_b)), "int64 positions differ from int32"
        ref = op.reference(normed, case.pool, case.table, case.lengths, 6, 1)
        for r in range(num_requests):
            rows = case.rows(r)
            assert rel_err(out_b[rows], ref[rows]) <= TOL, f"request {r}"
            assert rel_err(out_b[rows], out_a[rows]) <= TOL, (
                f"request {r} vs norm/rope + k3_drafter_attn"
            )


@pytest.mark.parametrize("num_requests,tokens", [(4, 8), (8, 1), (2, 4)])
def test_graph_replay(num_requests, tokens):
    """One captured call of each op, replayed with the lengths, page tables and qkv rewritten in place."""
    op = _op()
    gen = torch.Generator(device="cuda").manual_seed(31 + num_requests)
    with torch.inference_mode():
        lengths = [3000] * num_requests
        case = Case(gen, 6, 1, num_requests, tokens, lengths, strided_table=False)
        q_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        k_w = (1.0 + 0.2 * torch.randn(D, generator=gen, device="cuda")).bfloat16()
        out = torch.empty(num_requests * tokens, 6 * D, dtype=torch.bfloat16, device="cuda")
        out_n = torch.empty_like(out)

        def body():
            torch.ops.trtllm.k3_drafter_attn(
                case.qkv, case.pool, case.table, case.ctx_len, 6, 1, out
            )
            torch.ops.trtllm.k3_drafter_attn_qknorm(case.qkv, q_w, k_w, case.positions, EPS, THETA, case.pool,
                                                    case.table, case.ctx_len, 6, 1, out_n)  # fmt: skip

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                body()
        torch.cuda.synchronize()
        for rep in range(6):
            new = [min(c, 3000 - tokens) for c in case_lengths(num_requests, rep)]
            case.ctx_len.copy_(torch.tensor(new, dtype=torch.int32))
            case.positions.copy_(
                (case.ctx_len.view(-1, 1) + torch.arange(tokens, device="cuda")).reshape(-1)
            )
            # Rotate the page-table rows (each request reads another's pages) and redraw qkv.
            case.table.copy_(torch.roll(case.table, shifts=rep + 1, dims=0))
            case.qkv.copy_(
                (torch.randn(case.qkv.shape, generator=gen, device="cuda") * 0.5).bfloat16()
            )
            graph.replay()
            torch.cuda.synchronize()
            want = torch.empty_like(out)
            torch.ops.trtllm.k3_drafter_attn(
                case.qkv, case.pool, case.table, case.ctx_len, 6, 1, want
            )
            want_n = torch.empty_like(out)
            torch.ops.trtllm.k3_drafter_attn_qknorm(case.qkv, q_w, k_w, case.positions, EPS, THETA, case.pool,
                                                    case.table, case.ctx_len, 6, 1, want_n)  # fmt: skip
            assert torch.equal(bits(out), bits(want)), f"replay {rep}"
            assert torch.equal(bits(out_n), bits(want_n)), f"replay {rep} (qknorm)"
            # Every request has 3000 rows of context in its first pages, so any length <= 3000 reads finite rows.
            ref = op.reference(case.qkv, case.pool, case.table, new, 6, 1)
            assert rel_err(out, ref) <= TOL, f"replay {rep}"


def test_unsupported():
    op = _op()
    with torch.inference_mode():
        case = Case(torch.Generator(device="cuda").manual_seed(3), 6, 1, 2, 8, [100, 200])
        assert op.supports_attn(case.qkv, case.pool, 6, 1, 2)
        assert op.supports_attn(case.qkv[:8], case.pool, 6, 1)
        assert not op.supports_attn(case.qkv, case.pool, 6, 1)  # 16 rows of one request
        assert not op.supports_attn(case.qkv, case.pool, 6, 1, 3)  # 16 rows over 3 requests
        assert not op.supports_attn(case.qkv.repeat(5, 1), case.pool, 6, 1, 10)  # 10 requests
        out = torch.empty(16, 6 * D, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError):  # one page-table row for two requests
            torch.ops.trtllm.k3_drafter_attn(
                case.qkv, case.pool, case.table[0], case.ctx_len, 6, 1, out
            )
        with pytest.raises(ValueError):  # columns not dense
            torch.ops.trtllm.k3_drafter_attn(case.qkv, case.pool, case.table.t().contiguous().t(), case.ctx_len, 6,
                                             1, out)  # fmt: skip


def report() -> int:
    """The split checks as a markdown table (max over requests and the length offsets)."""
    print(f"{torch.cuda.get_device_name()}")
    print(
        "| heads/kv | split | lengths (offset 0) | CTM vs fp32 | prod vs fp32 | CTM vs prod | max abs | bits != prod "
        "| NaN | pool untouched | rerun | isolated | result |"
    )
    print("| :-- | :-- | :-- | --: | --: | --: | --: | --: | :-- | :-- | :-- | :-- | :-- |")
    ok_all = True
    with torch.inference_mode():
        for heads, kv, splits in ((6, 1, SPLITS), (24, 4, TP4_SPLITS)):
            for r, t in splits:
                rows = [
                    measure_split(r, t, off, heads, kv)
                    for off in ((0, 3, 7) if heads == 6 else (1,))
                ]
                ok = all(x["ok"] for x in rows)
                ok_all &= ok
                print(f"| {heads}/{kv} | {r}x{t} | {rows[0]['lengths']} | {max(max(x['err']) for x in rows):.2e} | "
                      f"{max(max(x['err_prod']) for x in rows):.2e} | {max(max(x['err_vs_prod']) for x in rows):.2e} | "
                      f"{max(x['abs_err'] for x in rows):.2e} | {sum(x['bits_vs_prod'] for x in rows)}/"
                      f"{len(rows) * r * t * heads * D} | {any(x['nan'] for x in rows)} | "
                      f"{all(x['untouched'] for x in rows)} | {all(x['rerun'] for x in rows)} | "
                      f"{all(x['isolated'] for x in rows)} | {'PASS' if ok else 'FAIL'} |", flush=True)  # fmt: skip
    print("ALL PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
