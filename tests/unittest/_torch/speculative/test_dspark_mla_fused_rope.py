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
"""The fused MLA drafter RoPE kernels write bitwise what the launches they replace write."""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="needs an SM100-family GPU"
)

KV_LORA = 512
ROPE = 64
NOPE = 128
HEAD_DIM = KV_LORA + ROPE
EPS = 1e-5


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _freqs_table(num_positions: int, gen) -> torch.Tensor:
    """A ``[P, ROPE // 2, 2]`` FP32 (cos, sin) table with YaRN-like magnitudes."""
    angle = torch.rand(num_positions, ROPE // 2, device="cuda", generator=gen) * 6.3
    scale = 1.0 + 0.3 * torch.rand(num_positions, 1, device="cuda", generator=gen)
    return torch.stack([angle.cos() * scale, angle.sin() * scale], dim=-1).contiguous()


def _rows(n, width, gen, row_stride=None):
    """BF16 rows with a spread of magnitudes; optionally a row-strided view."""
    row_stride = row_stride or width
    base = torch.randn(n, row_stride, device="cuda", generator=gen)
    base = base * torch.logspace(-2, 2, n, device="cuda").unsqueeze(1)
    return base.to(torch.bfloat16)[:, :width]


@pytest.mark.parametrize("num_tokens", [1, 7, 14, 28, 33])
def test_ctx_norm_rope_matches_per_layer_kernel(num_tokens):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        dspark_mla_ctx_norm_rope,
    )

    layers = 5
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    latent = _rows(num_tokens, layers * HEAD_DIM, gen).view(num_tokens, layers, HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(layers, KV_LORA, device="cuda", generator=gen)).to(
        torch.bfloat16
    )
    table = _freqs_table(4096, gen)
    positions = torch.randint(0, 4096, (num_tokens,), device="cuda", generator=gen)

    # Module weights are parameters that require grad, also in the server.
    out = dspark_mla_ctx_norm_rope(latent, torch.nn.Parameter(weight), positions, table, EPS)

    gathered = table[positions].contiguous()
    ref = torch.stack(
        [
            cute_dsl_dspark_rmsnorm_rope(
                latent[:, layer].contiguous(),
                weight[layer],
                gathered,
                1,
                ROPE,
                EPS,
                True,
                True,
                False,
                KV_LORA,
            )
            for layer in range(layers)
        ]
    )
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("num_tokens", [7, 14, 28, 32])
@pytest.mark.parametrize("page_size", [32, 64])
def test_block_rope_matches_separate_launches(num_tokens, page_size):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
        cute_dsl_dspark_rope_into,
        dspark_mla_block_rope,
    )

    heads = 4
    gen = torch.Generator(device="cuda").manual_seed(100 * num_tokens + page_size)
    q = _rows(num_tokens * heads, NOPE + ROPE, gen).view(num_tokens, heads, NOPE + ROPE)
    # A row-strided view, like a column slice of a wider GEMM output.
    kv = _rows(num_tokens, HEAD_DIM, gen, row_stride=1536 + HEAD_DIM)
    weight = (1.0 + 0.5 * torch.randn(KV_LORA, device="cuda", generator=gen)).to(torch.bfloat16)
    table = _freqs_table(8192, gen)
    positions = torch.randint(0, 8192, (num_tokens,), device="cuda", generator=gen)
    # An interleaved multi-layer page space, like the draft KV manager's view.
    num_pages = 40
    space = torch.randn(num_pages, 3, page_size, HEAD_DIM, device="cuda", generator=gen)
    pool = space.to(torch.bfloat16)[:, 1]
    pages = torch.randperm(num_pages, device="cuda", generator=gen)[:num_tokens].long()
    offsets = torch.randint(0, page_size, (num_tokens,), device="cuda", generator=gen)

    query = torch.full((num_tokens, heads, HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda")
    pool_new = pool.clone()
    dspark_mla_block_rope(
        q, query, kv, torch.nn.Parameter(weight), positions, table, pool_new, pages, offsets, EPS
    )

    gathered = table[positions].contiguous()
    query_ref = torch.full_like(query, 7.0)
    cute_dsl_dspark_rope_into(q, gathered, query_ref, heads, ROPE, KV_LORA)
    blk_kv = cute_dsl_dspark_rmsnorm_rope(
        kv.contiguous(), weight, gathered, 1, ROPE, EPS, True, True, False, KV_LORA
    )
    pool_ref = pool.clone()
    pool_ref[pages, offsets] = blk_kv
    torch.cuda.synchronize()
    assert torch.equal(_bits(query), _bits(query_ref))
    assert torch.equal(_bits(pool_new), _bits(pool_ref))


def test_scatter_rows_saturates_into_an_fp8_pool():
    from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8
    from tensorrt_llm._torch.speculative.spec_tail_kernels import scatter_rows_to_pool

    fp8 = torch.float8_e4m3fn
    rows, num_pages, page = 11, 6, 32
    gen = torch.Generator(device="cuda").manual_seed(3)
    src = (torch.randn(rows, HEAD_DIM, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    src[0, :4] = torch.tensor([600.0, -1000.0, 447.0, float("nan")], device="cuda").to(
        torch.bfloat16
    )
    flat = torch.randperm(num_pages * page, device="cuda", generator=gen)[:rows]
    pages, offsets = flat // page, flat % page
    pool = torch.zeros(num_pages, page, HEAD_DIM, device="cuda").to(fp8)
    scatter_rows_to_pool(src, pool, pages, offsets)
    reference = torch.zeros_like(pool.view(torch.uint8))
    reference[pages, offsets] = to_draft_fp8(src).view(torch.uint8)
    torch.cuda.synchronize()
    assert torch.equal(pool.view(torch.uint8), reference)
    assert pool[pages[0], offsets[0], :3].float().tolist() == [448.0, -448.0, 448.0]
    assert pool[pages[0], offsets[0], 3].float().isnan()


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_masked_scatter_keeps_other_ranks_slots(fp8):
    """A helix rank's store: rows it does not own leave their slot as it was, what the
    gather / where / index_put round trip wrote back."""
    from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8
    from tensorrt_llm._torch.speculative.spec_tail_kernels import scatter_rows_to_pool

    rows, num_pages, page = 14, 6, 32
    gen = torch.Generator(device="cuda").manual_seed(9)
    src = (torch.randn(rows, HEAD_DIM, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    flat = torch.randperm(num_pages * page, device="cuda", generator=gen)[:rows]
    pages, offsets = flat // page, flat % page
    owned = (torch.rand(rows, 1, device="cuda", generator=gen) < 0.5).view(-1, 1)
    pool = (torch.randn(num_pages, page, HEAD_DIM, device="cuda", generator=gen) * 2).to(
        torch.bfloat16
    )
    if fp8:
        pool = to_draft_fp8(pool)
    expected = pool.view(torch.uint8).clone()
    stored = to_draft_fp8(src) if fp8 else src
    rewritten = torch.where(owned, stored.view(torch.uint8), expected[pages, offsets])
    expected[pages, offsets] = rewritten
    scatter_rows_to_pool(src, pool, pages, offsets, rows_mask=owned)
    torch.cuda.synchronize()
    assert torch.equal(pool.view(torch.uint8), expected)


def test_draft_fp8_cast_matches_to_draft_fp8():
    from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8
    from tensorrt_llm._torch.speculative.spec_tail_kernels import draft_fp8_cast

    gen = torch.Generator(device="cuda").manual_seed(4)
    x = (torch.randn(3, 7, 4, HEAD_DIM, device="cuda", generator=gen) * 200).to(torch.bfloat16)
    x.view(-1)[:5] = torch.tensor([449.0, 1e4, -1e4, 1e-6, float("nan")], device="cuda").to(x.dtype)
    out = draft_fp8_cast(x)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), to_draft_fp8(x).view(torch.uint8))


def _fp8_neighbours(got: torch.Tensor, ref: torch.Tensor, max_frac: float) -> None:
    """Every FP8 code of ``got`` equal to or a neighbour of ``ref``'s; few differ."""
    got, ref = got.float(), ref.float()
    spacing = torch.maximum(got.abs(), ref.abs()).clamp_min(2.0**-6) * 2.0**-3
    assert bool(((got - ref).abs() <= spacing).all())
    frac = float((got != ref).float().mean())
    assert frac <= max_frac, frac


@pytest.mark.parametrize("masked", [False, True], ids=["all_rows", "owned_rows"])
@pytest.mark.parametrize("num_tokens", [1, 8, 16, 32])
def test_draft_qhead_matches_the_unfused_ops(num_tokens, masked):
    """``draft_qhead`` against q_a_layernorm + q_b + the fused block RoPE + the
    masked scatter + the absorption + the FP8 cast: the query no further from
    FP64 than theirs, the pool rows equal or one FP8 code apart."""
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import dspark_mla_block_rope
    from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8
    from tensorrt_llm._torch.speculative.dspark_mla_qhead import draft_qhead, fold_rmsnorm_weight
    from tensorrt_llm._torch.speculative.spec_tail_kernels import scatter_rows_to_pool

    torch.manual_seed(num_tokens + 100 * masked)
    heads, q_lora, eps = 64, 1536, 1e-6
    q_a = torch.randn(num_tokens, q_lora, device="cuda").bfloat16()
    kv = (torch.randn(num_tokens, HEAD_DIM, device="cuda") * 2).bfloat16()
    w_qa = (torch.rand(q_lora, device="cuda") + 0.5).bfloat16()
    w_qb = (torch.randn(heads * (NOPE + ROPE), q_lora, device="cuda") * 0.03).bfloat16()
    w_kv = (torch.rand(KV_LORA, device="cuda") + 0.5).bfloat16()
    w_uk = (torch.randn(heads, NOPE, KV_LORA, device="cuda") * 0.1).bfloat16()
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64) + 1000
    angles = torch.rand(2048, ROPE // 2, device="cuda") * 6.28
    freqs = torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()
    pages_n, page_size = 40, 16
    pool = (torch.randn(pages_n, page_size, HEAD_DIM, device="cuda") * 3).to(torch.float8_e4m3fn)
    slots = torch.randperm(pages_n * page_size, device="cuda")[:num_tokens]
    pages, offsets = slots // page_size, slots % page_size
    mask = None
    if masked:
        mask = torch.rand(num_tokens, 1, device="cuda") < 0.5
        mask[0] = True

    qf = q_a.float()
    q_a_n = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + eps) * w_qa.float()).bfloat16()
    q_ref = (q_a_n.float() @ w_qb.float().t()).bfloat16().view(num_tokens, heads, NOPE + ROPE)
    query = torch.empty(num_tokens, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty(num_tokens, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(num_tokens, device="cuda")
    dspark_mla_block_rope(
        q_ref, query, kv, w_kv, positions, freqs, scratch, rows, torch.zeros_like(rows), eps
    )
    ref_pool = pool.clone()
    scatter_rows_to_pool(scratch.view(num_tokens, -1), ref_pool, pages, offsets, rows_mask=mask)
    ref = query.clone()
    torch.ops.trtllm.bmm_out(
        q_ref[..., :NOPE].transpose(0, 1), w_uk, ref[..., :KV_LORA].transpose(0, 1)
    )
    ref8 = to_draft_fp8(ref)

    folded = fold_rmsnorm_weight(w_qb, w_qa)
    q_raw = (qf @ folded.float().t()).bfloat16().view(num_tokens, heads, NOPE + ROPE)
    got_pool = pool.clone()
    got = draft_qhead(
        q_a, kv, q_raw, eps, w_uk, w_kv, eps, positions, freqs, got_pool, pages, offsets, mask
    )
    qd = q_a.double()
    q_true = (
        (qd * torch.rsqrt(qd.pow(2).mean(-1, keepdim=True) + eps) * w_qa.double())
        @ w_qb.double().t()
    ).view(num_tokens, heads, NOPE + ROPE)
    lat_true = torch.einsum("thk,hkd->thd", q_true[..., :NOPE], w_uk.double())
    re, im = q_true[..., NOPE::2], q_true[..., NOPE + 1 :: 2]
    cos = freqs[positions][:, None, :, 0].double()
    sin = freqs[positions][:, None, :, 1].double()
    rope_true = torch.stack([re * cos - im * sin, im * cos + re * sin], dim=-1).flatten(-2)
    truth = torch.cat([lat_true, rope_true], dim=-1)

    def rel_err(x):
        return float((x.double() - truth).norm() / truth.norm())

    assert rel_err(got) <= 1.15 * rel_err(ref8) + 1e-3, (rel_err(got), rel_err(ref8))
    written = torch.zeros(pages_n, page_size, dtype=torch.bool, device="cuda")
    keep = (
        torch.ones(num_tokens, dtype=torch.bool, device="cuda") if mask is None else mask.view(-1)
    )
    written[pages[keep], offsets[keep]] = True
    assert torch.equal(got_pool[~written].view(torch.uint8), pool[~written].view(torch.uint8))
    _fp8_neighbours(got_pool[written], ref_pool[written], max_frac=0.02)
