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

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import (
    fa4_forward,
    fa4_paged_generation,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or get_sm_version() != 90 or fa4_forward() is None,
        reason="FlashAttention-4 generation path is SM90-only",
    ),
]


def _case(num_heads, num_kv_heads, head_dim, q_len, rows, seed, massive=False, tpb=64):
    """Scattered pages, cached lengths around page boundaries, page tables one page longer than needed."""
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)
    special = [tpb - q_len, tpb - 1, tpb, tpb + 1, 2 * tpb - q_len, 0, 3 * tpb + 7]
    cached = [
        special[r % len(special)] if r % 2 == 0 else int(torch.randint(0, 600, (1,), generator=gen))
        for r in range(rows)
    ]
    kv_lens = [c + q_len for c in cached]
    blocks = [(n + tpb - 1) // tpb + 1 for n in kv_lens]
    num_pages = sum(blocks) + 3
    perm = torch.randperm(num_pages, generator=gen).tolist()
    tables, nxt = [], 0
    for b in blocks:
        tables.append(perm[nxt : nxt + b])
        nxt += b
    width = max(blocks)
    page_table = torch.tensor(
        [t + [0] * (width - len(t)) for t in tables], dtype=torch.int32, device=device
    )
    dgen = torch.Generator(device=device).manual_seed(seed)
    pool = (
        0.7 * torch.randn(num_pages, 2, num_kv_heads, tpb, head_dim, generator=dgen, device=device)
    ).bfloat16()
    ntok = rows * q_len
    q = torch.randn(ntok, num_heads, head_dim, generator=dgen, device=device).bfloat16()
    k = torch.randn(ntok, num_kv_heads, head_dim, generator=dgen, device=device).bfloat16()
    v = torch.randn(ntok, num_kv_heads, head_dim, generator=dgen, device=device).bfloat16()
    if massive:
        # Qwen2.5-7B-like magnitudes: keys ~600x, queries ~80x, odd heads with every logit hugely negative.
        pool[:, 0] = (pool[:, 0].float().abs() * 600.0).bfloat16()
        k = (k.float().abs() * 600.0).bfloat16()
        sign = torch.where(torch.arange(num_heads, device=device) % 2 == 1, -1.0, 1.0)[
            None, :, None
        ]
        q = (q.float().abs() * 80.0 * sign).bfloat16()
    return dict(
        pool=pool, q=q, k=k, v=v, tables=tables, page_table=page_table, cached=cached,
        kv_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device), tpb=tpb, rows=rows,
    )  # fmt: skip


def _reference(case, num_heads, num_kv_heads, head_dim, q_len):
    """float64 causal attention over each row's cached tokens plus its own step tokens."""
    pool, q, k, v = case["pool"].double(), case["q"], case["k"], case["v"]
    gs = num_heads // num_kv_heads
    ref = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for r in range(case["rows"]):
        cached = case["cached"][r]
        pages = case["tables"][r]
        kv = (
            pool[pages].permute(1, 2, 0, 3, 4).reshape(2, num_kv_heads, -1, head_dim)[:, :, :cached]
        )
        for t in range(q_len):
            keys = torch.cat([kv[0], k[r * q_len : r * q_len + t + 1].double().permute(1, 0, 2)], 1)
            vals = torch.cat([kv[1], v[r * q_len : r * q_len + t + 1].double().permute(1, 0, 2)], 1)
            qh = q[r * q_len + t].double().view(num_kv_heads, gs, head_dim)
            probs = torch.softmax(
                torch.einsum("hgd,htd->hgt", qh, keys) / math.sqrt(head_dim), dim=-1
            )
            ref[r * q_len + t] = torch.einsum("hgt,htd->hgd", probs, vals).reshape(
                num_heads, head_dim
            )
    return ref


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,q_len,massive",
    [
        (28, 4, 128, 1, False),
        (28, 4, 128, 4, False),
        (28, 4, 128, 9, False),
        (20, 20, 128, 1, False),
        (20, 20, 128, 4, False),
        (20, 20, 128, 9, False),
        (16, 8, 64, 9, False),
        (28, 4, 128, 4, True),
    ],
)
def test_fa4_paged_generation_matches_dense_and_writes_kv(
    num_heads, num_kv_heads, head_dim, q_len, massive
):
    """The step's tokens attend the row's cache and each other causally; their K/V land at
    kv_len - q_len + j (page crossings, page tables a page longer than needed); no zero heads
    under massive activations."""
    for seed in range(2):
        case = _case(num_heads, num_kv_heads, head_dim, q_len, rows=13, seed=seed, massive=massive)
        before = case["pool"].clone()
        ref = _reference(case, num_heads, num_kv_heads, head_dim, q_len)
        out = torch.empty_like(case["q"])
        fa4_paged_generation(
            case["pool"], case["q"], case["k"], case["v"], out, case["page_table"], case["kv_lens"],
            q_len, 1.0 / math.sqrt(head_dim),
        )  # fmt: skip
        torch.cuda.synchronize()
        head_max = ref.abs().amax(-1)
        err = (out.double() - ref).abs().amax(-1)
        assert not bool(((out.abs().amax(-1) == 0) & (head_max > 1e-3)).any()), "zero head"
        assert float((err / head_max.clamp(min=1e-3)).max()) < 2e-2, f"seed {seed}"
        expect = before.clone()
        tpb = case["tpb"]
        for r in range(case["rows"]):
            for j in range(q_len):
                pos = case["cached"][r] + j
                page = case["tables"][r][pos // tpb]
                expect[page, 0, :, pos % tpb] = case["k"][r * q_len + j]
                expect[page, 1, :, pos % tpb] = case["v"][r * q_len + j]
        assert torch.equal(case["pool"], expect)


def test_fa4_paged_generation_graph_replay_matches_eager():
    """One CUDA graph, replayed on new inputs and lengths of the same shape, equals eager."""
    num_heads, num_kv_heads, head_dim, q_len = 28, 4, 128, 4
    case = _case(num_heads, num_kv_heads, head_dim, q_len, rows=16, seed=3)
    q, k, v = case["q"].clone(), case["k"].clone(), case["v"].clone()
    kv_lens, pool = case["kv_lens"].clone(), case["pool"].clone()
    out = torch.zeros_like(q)
    args = (case["page_table"], kv_lens, q_len, 1.0 / math.sqrt(head_dim))
    fa4_paged_generation(pool, q, k, v, out, *args)  # warm up / compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fa4_paged_generation(pool, q, k, v, out, *args)
    gen = torch.Generator(device="cuda").manual_seed(11)
    for step in range(3):
        q.copy_(torch.randn(q.shape, generator=gen, device="cuda").bfloat16())
        k.copy_(torch.randn(k.shape, generator=gen, device="cuda").bfloat16())
        v.copy_(torch.randn(v.shape, generator=gen, device="cuda").bfloat16())
        kv_lens.sub_(1 if step % 2 else 0)
        eager_pool, eager_out = pool.clone(), torch.zeros_like(q)
        fa4_paged_generation(eager_pool, q, k, v, eager_out, *args)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager_out)
        assert torch.equal(pool, eager_pool)


def _rope_reference(qkv, cos_sin, kv_lens, *, num_heads, num_kv_heads, head_dim, rotary_dim, q_len):
    """NeoX RoPE on the Q and K heads of packed QKV, in place, in float32: token ``j`` of row ``r`` at
    position ``kv_lens[r] - q_len + j`` (clamped at 0), ``cos_sin`` laid out [position][pair][cos, sin]."""
    tokens = qkv.shape[0]
    rows = tokens // q_len
    device = qkv.device
    pos = (
        kv_lens.long().repeat_interleave(q_len)
        - q_len
        + torch.arange(q_len, device=device).repeat(rows)
    )
    half = rotary_dim // 2
    table = cos_sin.view(-1, half, 2)[pos.clamp(min=0)]
    cos, sin = table[..., 0][:, None, :], table[..., 1][:, None, :]
    heads = qkv[:, : (num_heads + num_kv_heads) * head_dim].view(
        tokens, num_heads + num_kv_heads, head_dim
    )
    x1, x2 = heads[..., :half].float(), heads[..., half:rotary_dim].float()
    heads[..., :half] = (x1 * cos - x2 * sin).to(qkv.dtype)
    heads[..., half:rotary_dim] = (x2 * cos + x1 * sin).to(qkv.dtype)


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,rotary_dim,q_len",
    [(28, 4, 128, 128, 4), (28, 4, 128, 128, 9), (20, 20, 128, 128, 1), (16, 2, 128, 64, 4)],
)
@pytest.mark.parametrize("rope", [True, False])
def test_rope_write_kv_paged_matches_rope_then_write(
    num_heads, num_kv_heads, head_dim, rotary_dim, q_len, rope
):
    """The fused launch leaves the pool as NeoX RoPE + ``write_kv_paged`` do and the Q heads
    rotated in place the same way (partial rotary, rows whose first tokens fall before
    position 0 write nothing for them)."""
    from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import (
        rope_write_kv_paged,
        write_kv_paged,
    )

    device = torch.device("cuda")
    tpb, rows, width = 32, 13, 8
    gen = torch.Generator(device=device).manual_seed(0)
    kv_lens = torch.tensor(
        [q_len - 1 if q_len > 1 else 1]
        + [int(x) for x in torch.randint(q_len, width * tpb, (rows - 1,))],
        dtype=torch.int32,
        device=device,
    )
    page_table = torch.randperm(rows * width, device=device).view(rows, width).to(torch.int32)
    pool = torch.randn(
        rows * width, 2, num_kv_heads, tpb, head_dim, generator=gen, device=device
    ).bfloat16()
    cols = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(rows * q_len, cols + 64, generator=gen, device=device).bfloat16()[:, :cols]
    cos_sin = torch.randn(width * tpb, rotary_dim, generator=gen, device=device)
    ref_qkv, ref_pool = qkv.clone(), pool.clone()
    if rope:
        _rope_reference(ref_qkv, cos_sin, kv_lens, num_heads=num_heads, num_kv_heads=num_kv_heads,
                        head_dim=head_dim, rotary_dim=rotary_dim, q_len=q_len)  # fmt: skip
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    k = ref_qkv[:, q_size : q_size + kv_size].view(-1, num_kv_heads, head_dim)
    v = ref_qkv[:, q_size + kv_size :].view(-1, num_kv_heads, head_dim)
    write_kv_paged(ref_pool, k, v, page_table, kv_lens, q_len)
    rope_write_kv_paged(qkv, cos_sin if rope else None, pool, page_table, kv_lens, num_heads=num_heads,
                        num_kv_heads=num_kv_heads, head_dim=head_dim, rotary_dim=rotary_dim, q_len=q_len)  # fmt: skip
    torch.cuda.synchronize()
    # V bit for bit; the rotated Q and K within one bf16 rounding step (the kernel may contract the
    # rotation's multiply-adds, the reference does not), plus 1e-5 absolute for values near zero.
    assert torch.equal(pool[:, 1], ref_pool[:, 1])
    torch.testing.assert_close(pool[:, 0].float(), ref_pool[:, 0].float(), rtol=2**-7, atol=1e-5)
    torch.testing.assert_close(
        qkv[:, :q_size].float(), ref_qkv[:, :q_size].float(), rtol=2**-7, atol=1e-5
    )


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,q_len,massive",
    [
        (28, 4, 128, 1, False),
        (28, 4, 128, 4, False),
        (28, 4, 128, 9, False),
        (20, 20, 128, 9, False),
        (16, 8, 64, 4, False),
        (28, 4, 128, 9, True),
    ],
)
@pytest.mark.parametrize("splits", [2, 5, 16])
@pytest.mark.parametrize("min_split_tokens", [1, 512])
def test_fa4_paged_generation_split_matches_dense(
    num_heads, num_kv_heads, head_dim, q_len, massive, splits, min_split_tokens, monkeypatch
):
    """Split-KV generation (non-causal chunks of the cached pages + causal last part, merged) against the
    float64 reference, including rows whose cache is shorter than the split count (empty chunks) and rows
    with nothing cached before the page of their first step token; chunks of at least one page or of at
    least the default minimum length."""
    from tensorrt_llm._torch.attention.backends.fmha import flash_attn4_kernels as fa4_kernels
    from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import (
        allocate_split_workspace,
        fa4_paged_generation_split,
        split_tables,
        write_kv_paged,
    )

    monkeypatch.setattr(fa4_kernels, "MIN_SPLIT_TOKENS", min_split_tokens)
    for seed in range(2):
        case = _case(num_heads, num_kv_heads, head_dim, q_len, rows=5, seed=seed, massive=massive)
        ref = _reference(case, num_heads, num_kv_heads, head_dim, q_len)
        write_kv_paged(
            case["pool"], case["k"], case["v"], case["page_table"], case["kv_lens"], q_len
        )
        tables = split_tables(case["page_table"], case["kv_lens"], q_len, splits, case["tpb"])
        out = torch.empty_like(case["q"])
        workspace = allocate_split_workspace(
            5, splits, q_len, num_heads, head_dim, out.dtype, out.device
        )
        fa4_paged_generation_split(
            case["pool"], case["q"], out, tables, q_len, splits, 1.0 / math.sqrt(head_dim),
            workspace,
        )  # fmt: skip
        torch.cuda.synchronize()
        head_max = ref.abs().amax(-1)
        err = (out.double() - ref).abs().amax(-1)
        assert not bool(((out.abs().amax(-1) == 0) & (head_max > 1e-3)).any()), "zero head"
        assert float((err / head_max.clamp(min=1e-3)).max()) < 2e-2, f"seed {seed}"


def test_generation_splits_follow_the_grid(monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import (
        SPLITS_ENV,
        generation_splits,
    )

    monkeypatch.delenv(SPLITS_ENV, raising=False)
    assert generation_splits(120, 28, 4, 9, 132) == 1  # 480 CTAs
    assert generation_splits(60, 28, 4, 4, 132) == 1  # 240 CTAs
    assert generation_splits(4, 28, 4, 9, 132) == 9  # 16 CTAs
    assert generation_splits(1, 28, 4, 1, 132) == 16  # 4 CTAs, capped
    assert generation_splits(4, 20, 20, 9, 132) == 1  # MHA: 80 CTAs
    monkeypatch.setenv(SPLITS_ENV, "1")
    assert generation_splits(1, 28, 4, 1, 132) == 1


@pytest.mark.parametrize("q_len", [4, 9])
def test_rope_write_kv_paged_with_page_lens_uses_absolute_rope_positions(q_len):
    """A page table covering only the end of each row (``page_lens`` < ``kv_lens``, as for the n-sample
    library's private pages): RoPE at the absolute positions, K/V at slot ``page_len - q_len + j``."""
    from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import (
        rope_write_kv_paged,
        write_kv_paged,
    )

    device = torch.device("cuda")
    num_heads, num_kv_heads, head_dim, tpb, rows, width = 28, 4, 128, 32, 7, 6
    gen = torch.Generator(device=device).manual_seed(1)
    shared_tokens = 4 * tpb
    page_lens = torch.randint(q_len, width * tpb, (rows,), generator=gen, device=device).to(
        torch.int32
    )
    kv_lens = page_lens + shared_tokens
    tail_table = torch.randperm(rows * width, device=device).view(rows, width).to(torch.int32)
    pool = torch.randn(
        rows * width, 2, num_kv_heads, tpb, head_dim, generator=gen, device=device
    ).bfloat16()
    cols = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(rows * q_len, cols, generator=gen, device=device).bfloat16()
    cos_sin = torch.randn(width * tpb + shared_tokens, head_dim, generator=gen, device=device)
    ref_qkv, ref_pool = qkv.clone(), pool.clone()
    _rope_reference(ref_qkv, cos_sin, kv_lens, num_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, rotary_dim=head_dim, q_len=q_len)  # fmt: skip
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    ref_k = ref_qkv[:, q_size : q_size + kv_size].view(-1, num_kv_heads, head_dim)
    ref_v = ref_qkv[:, q_size + kv_size :].view(-1, num_kv_heads, head_dim)
    write_kv_paged(ref_pool, ref_k, ref_v, tail_table, page_lens, q_len)
    rope_write_kv_paged(qkv, cos_sin, pool, tail_table, kv_lens, num_heads=num_heads, num_kv_heads=num_kv_heads,
                        head_dim=head_dim, rotary_dim=head_dim, q_len=q_len, page_lens=page_lens)  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(pool[:, 1], ref_pool[:, 1])
    torch.testing.assert_close(pool[:, 0].float(), ref_pool[:, 0].float(), rtol=2**-7, atol=1e-5)
    torch.testing.assert_close(
        qkv[:, :q_size].float(), ref_qkv[:, :q_size].float(), rtol=2**-7, atol=1e-5
    )


@pytest.mark.parametrize(
    "per_layer_tables", [False, True], ids=["group_tables", "per_layer_tables"]
)
def test_per_layer_page_tables_are_declined(per_layer_tables):
    """A KV manager with per-layer page tables (SWA scratch reuse) indexes ``kv_cache_block_offsets``
    by local layer: layers 1 and 3 of one lifecycle group can map different pages. The library reads
    the lifecycle group's table, so it declines such a manager; layer 3 then runs on the regular
    kernel instead of reading layer 1's pages."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_generation import (
        FlashAttn4GenerationFmha,
    )
    from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionInputType,
        PredefinedAttentionMask,
    )

    manager = SimpleNamespace(
        index_mapper=object(),
        is_draft=False,
        layer_offsets={1: 1, 3: 3},
        layer_to_pool_mapping_dict={1: 1, 3: 1},  # one lifecycle group
        _use_per_layer_page_tables=per_layer_tables,
    )
    metadata = SimpleNamespace(
        num_generations=4,
        num_contexts=0,
        kv_cache_manager=manager,
        kv_cache_block_offsets=torch.zeros(4, 4, 2, 8, dtype=torch.int32),
        beam_width=1,
        is_cross=False,
        enable_helix=False,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        max_seq_len=None,
    )
    forward_args = SimpleNamespace(
        attention_input_type=AttentionInputType.mixed,
        is_fused_qkv=True,
        update_kv_cache=True,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_mask_data=None,
        attention_sinks=None,
        relative_attention_bias=None,
        out_scale=None,
        output_sf=None,
        attention_window_size=None,
    )
    fmha = SimpleNamespace(attn=SimpleNamespace(layer_idx=3, position_embedding_type=0))
    q = torch.empty(4, 8, dtype=torch.bfloat16)
    supported = FlashAttn4GenerationFmha._is_supported(
        fmha, q, None, None, metadata, forward_args, phase=FmhaPhase.GENERATION
    )
    assert supported is not per_layer_tables


def test_split_workspaces_are_owned_per_kv_cache_manager():
    """A split call writes its workspace, so concurrent executors in one process (each with its own
    KV cache manager) never share one; the layers of one executor do."""
    from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_generation import _split_workspace

    q = torch.empty((8, 4, 64), dtype=torch.bfloat16, device="cuda")
    first, second = object.__new__(type("Manager", (), {})), object.__new__(type("Manager", (), {}))
    workspace = _split_workspace(first, 2, 4, 4, 4, 64, q)
    assert _split_workspace(first, 2, 4, 4, 4, 64, q) is workspace
    other = _split_workspace(second, 2, 4, 4, 4, 64, q)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(workspace, other))
