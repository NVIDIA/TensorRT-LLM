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
"""The fused DSpark tail kernels (``spec_tail_kernels.tail_v2_enabled``) write bitwise what the launches they
replace write."""

from types import SimpleNamespace

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
        _compile_fused_dspark_rmsnorm_rope,
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

    out = dspark_mla_ctx_norm_rope(latent, torch.nn.Parameter(weight), positions, table, EPS)

    compiled = _compile_fused_dspark_rmsnorm_rope(
        HEAD_DIM, ROPE, 1, EPS, True, True, False, KV_LORA
    )
    gathered = table[positions].contiguous()
    ref = torch.empty_like(out)
    for layer in range(layers):
        compiled(latent[:, layer], weight[layer], gathered, ref[layer])
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
    # The block latents are the kv half of the fused [q_a | kv_a] GEMM output.
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
        kv, weight, gathered, 1, ROPE, EPS, True, True, False, KV_LORA
    )
    pool_ref = pool.clone()
    pool_ref[pages, offsets] = blk_kv
    torch.cuda.synchronize()
    assert torch.equal(_bits(query), _bits(query_ref))
    assert torch.equal(_bits(pool_new), _bits(pool_ref))


@pytest.mark.parametrize(
    "rows,n_blocks,holes",
    [(1, 31, False), (2, 2048, False), (3, 4097, True), (8, 30000, False), (64, 6000, True)],
)
def test_block_table_decode_parallel_matches(rows, n_blocks, holes):
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
        BLOCK_TABLE_CHUNK,
        decode_draft_block_table,
        decode_draft_block_table_parallel,
    )

    gen = torch.Generator(device="cuda").manual_seed(rows * 7 + n_blocks)
    divisor = 10
    src_full = torch.randint(0, 50000, (2, rows + 1, 2, n_blocks), device="cuda", generator=gen)
    src_full = src_full.to(torch.int32)
    lengths = torch.randint(0, n_blocks + 1, (rows + 1,), device="cuda", generator=gen)
    cols = torch.arange(n_blocks, device="cuda")
    src_full[1] = torch.where(cols < lengths.view(-1, 1, 1), src_full[1], -1)
    if holes:
        src_full[1, :, :, ::7] = -1
    src = src_full[1, :rows, 0]

    table_ref = torch.full((rows + 1, n_blocks), -5, dtype=torch.int32, device="cuda")
    counts_ref = torch.full((rows + 1,), -5, dtype=torch.long, device="cuda")
    decode_draft_block_table(src, divisor, table_ref[:rows], counts_ref[:rows])

    table = torch.full_like(table_ref, -5)
    counts = torch.full_like(counts_ref, -5)
    partial = torch.full(
        (rows * -(-n_blocks // BLOCK_TABLE_CHUNK),), -1, dtype=torch.int32, device="cuda"
    )
    decode_draft_block_table_parallel(src, divisor, table[:rows], counts[:rows], partial)
    torch.cuda.synchronize()
    assert torch.equal(table, table_ref)
    assert torch.equal(counts, counts_ref)


def _pairs(rows, n_slots, gen, ties=False, nans=False):
    """An exchanged pair buffer: ``[rows, width]`` FP32, ``(value, index)`` per slot."""
    width = triton_next_pow2(2 * n_slots)
    buf = torch.zeros(rows, width, device="cuda")
    vals = torch.randn(rows, n_slots, device="cuda", generator=gen)
    if ties:
        vals = vals.round()
    if nans:
        vals[0, 3] = float("nan")
        vals[1, 7] = float("nan")
        vals[1, 2] = float("nan")
    idx = torch.randint(0, 163840, (rows, n_slots), device="cuda", generator=gen).float()
    buf[:, 0 : 2 * n_slots : 2] = vals
    buf[:, 1 : 2 * n_slots : 2] = idx
    return buf, width


def triton_next_pow2(x):
    return 1 << (x - 1).bit_length()


@pytest.mark.parametrize("num_contexts,num_gens", [(0, 2), (1, 3), (0, 16)])
def test_next_new_tokens_rewind_matches(num_contexts, num_gens):
    from tensorrt_llm._torch.speculative.spec_tail_kernels import (
        fused_next_new_tokens,
        fused_next_new_tokens_rewind,
    )

    K = 6
    batch = num_contexts + num_gens
    gen = torch.Generator(device="cuda").manual_seed(batch)
    accepted = torch.randint(0, 1000, (batch, K + 1), device="cuda", generator=gen).to(torch.int32)
    n = torch.randint(1, K + 2, (batch,), device="cuda", generator=gen).to(torch.int32)
    drafts = torch.randint(0, 1000, (batch, K), device="cuda", generator=gen).to(torch.int32)
    kv0 = torch.randint(0, 9, (32,), device="cuda", generator=gen).to(torch.int32)
    rewind = (1 - n[num_contexts:]).contiguous()
    ref = fused_next_new_tokens(accepted, drafts, n)
    ref_kv = kv0.clone()
    ref_kv[num_contexts:batch] -= rewind
    ref_kv[num_contexts:batch].clamp_(min=0)
    kv = kv0.clone()
    out = fused_next_new_tokens_rewind(accepted, drafts, n, kv, rewind, num_contexts)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert torch.equal(kv, ref_kv)


@pytest.mark.parametrize(
    "role,n,k",
    [
        ("q_kv_a", 2112, 7168),
        ("q_b", 768, 1536),
        ("gate_up", 1792, 7168),
        ("down", 7168, 896),
        ("ctx_latent", 2880, 7168),
        ("lm_head", 10240, 7168),
        # A helix drafter's replicated q_b (64 heads).
        ("q_b", 12288, 1536),
    ],
)
@pytest.mark.parametrize("m", [7, 14, 28, 32, 40])
def test_draft_gemm_pins(monkeypatch, role, n, k, m):
    """The pinned drafter GEMMs (on Rubin the SM107 table of the helix16 drafter's
    shapes) compute x @ W^T (to BF16 rounding), and step aside above the pinned row
    range or with the switch off."""
    from tensorrt_llm._torch.speculative.dspark_draft_gemm import (
        DRAFT_GEMM_ENV_VAR,
        _pins,
        draft_gemm,
    )

    gen = torch.Generator(device="cuda").manual_seed(m + n)
    x = (torch.randn(m, k, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).to(torch.bfloat16)
    monkeypatch.setenv(DRAFT_GEMM_ENV_VAR, "0")
    assert draft_gemm(x, w, role) is None
    monkeypatch.setenv(DRAFT_GEMM_ENV_VAR, "1")
    # Module weights are parameters that require grad, also in the server.
    out = draft_gemm(x, torch.nn.Parameter(w), role)
    if not any(m <= rows for rows, _, _ in _pins(role, n, k, x.device) or ()):
        assert out is None
        return
    ref = x.float() @ w.float().t()
    assert out is not None and out.shape == (m, n) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def _exchanged_pairs(rows, n_slots, gen, ties=False, nans=False):
    """An exchanged pair buffer ``[rows, width]`` FP32 of ``(value, index)`` per slot,
    zero-padded past the slots, as the all-reduced MNNVL exchange leaves it."""
    import triton

    width = triton.next_power_of_2(2 * n_slots)
    buf = torch.zeros(rows, width, device="cuda")
    vals = torch.randn(rows, n_slots, device="cuda", generator=gen)
    if ties:
        vals = vals.round()
    if nans and rows >= 2:
        vals[0, 3] = float("nan")
        vals[1, 7] = float("nan")
        vals[1, 2] = float("nan")
    idx = torch.randint(0, 163840, (rows, n_slots), device="cuda", generator=gen).float()
    buf[:, 0 : 2 * n_slots : 2] = vals
    buf[:, 1 : 2 * n_slots : 2] = idx
    return buf


def _reduce_pairs(buf, n_slots):
    import triton

    from tensorrt_llm._torch.speculative import vocab_parallel as vp

    rows = buf.shape[0]
    out = torch.empty(rows, dtype=torch.int64, device="cuda")
    if rows:
        vp._reduce_pairs_kernel[(triton.cdiv(rows, vp._REDUCE_BLOCK_ROWS),)](
            buf,
            buf.stride(0),
            rows,
            n_slots,
            out,
            BLOCK_ROWS=vp._REDUCE_BLOCK_ROWS,
            BLOCK_SLOTS=triton.next_power_of_2(n_slots),
            num_warps=4,
        )
    return out


@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (0, 2), (0, 4), (2, 3), (3, 0), (0, 16)])
@pytest.mark.parametrize("force", [None, "const", "frac"])
@pytest.mark.parametrize("ties,nans", [(False, False), (True, True)])
@pytest.mark.parametrize("has_dummy", [True, False])
def test_fused_select_accept_matches_reduce_then_accept(
    num_contexts, num_gens, force, ties, nans, has_dummy
):
    """``fused_select_accept`` writes what the pair reduce, the strict acceptance (with
    the forced-acceptance override), the KDA replay record and the drafter's KV prep
    write, on a strided per-slot record view and an exchange with zero padding."""
    from tensorrt_llm._torch.speculative.dspark_glue_kernels import (
        kda_record_acceptance,
        kv_prepare_for_draft,
        strict_accept_forced,
    )
    from tensorrt_llm._torch.speculative.spec_tail_kernels import fused_select_accept, strict_accept

    K, n_slots = 7, 96
    batch = num_contexts + num_gens
    rows = num_contexts + num_gens * (K + 1)
    gen = torch.Generator(device="cuda").manual_seed(rows * 13 + int(ties) + len(str(force)))
    buf = _exchanged_pairs(max(rows, 2), n_slots, gen, ties=ties, nans=nans)[:rows].contiguous()
    targets = _reduce_pairs(buf, n_slots)
    # Drafts that match a random-length prefix of the targets.
    t_gen = targets[num_contexts:].view(num_gens, K + 1)
    drafts = torch.randint(0, 163840, (num_gens, K), device="cuda", generator=gen).to(torch.int32)
    for g in range(num_gens):
        keep = int(torch.randint(0, K + 1, (1,), device="cuda", generator=gen))
        drafts[g, :keep] = t_gen[g, :keep].to(torch.int32)
    pool = torch.rand(1 << 10, device="cuda", generator=gen)
    force_args = {
        None: None,
        "const": (3,),
        "frac": (2, 0.6, K + 1, pool, torch.zeros(1, dtype=torch.int64, device="cuda"), 7, 3),
    }[force]
    if force is None:
        ref_acc, ref_n = strict_accept(targets, drafts, num_contexts, batch)
    else:
        ref_force = force_args
        if force == "frac":
            ref_force = force_args[:4] + (force_args[4].clone(),) + force_args[5:]
        ref_acc, ref_n = strict_accept_forced(targets, drafts, num_contexts, batch, ref_force)

    # The KDA replay record lives one int32 per state page: a strided per-slot view.
    num_slots = 64
    pages = torch.randint(0, 7, (num_slots, 5), device="cuda", generator=gen).to(torch.int32)
    slots = torch.randperm(num_slots, device="cuda", generator=gen)[: max(batch, 1)]
    dummy = None
    if has_dummy:
        dummy = torch.zeros(num_slots, dtype=torch.bool, device="cuda")
        if num_gens > 1:
            dummy[num_contexts + num_gens - 1] = True
    ref_pages = pages.clone()
    kda_record_acceptance(ref_pages[:, 2], slots, ref_n, num_contexts, num_gens, dummy)
    kv0 = torch.randint(0, 1000, (num_slots,), device="cuda", generator=gen).to(torch.int32)
    ref_kv = kv0.clone()
    ref_rewind = kv_prepare_for_draft(ref_kv, ref_n, num_contexts, batch)

    kv = kv0.clone()
    acc, n, rewind = fused_select_accept(
        (buf, buf.stride(0), 2, n_slots),
        drafts,
        num_contexts,
        batch,
        force_args,
        kda_replay=(pages[:, 2], slots, dummy) if num_gens > 0 else None,
        kv_lens=kv,
    )
    torch.cuda.synchronize()
    gen_rows = slice(num_contexts, batch)
    assert torch.equal(acc[gen_rows], ref_acc[gen_rows])
    assert torch.equal(acc[:num_contexts, 0], ref_acc[:num_contexts, 0])
    assert torch.equal(n, ref_n)
    assert torch.equal(pages, ref_pages)
    assert torch.equal(kv, ref_kv)
    assert torch.equal(rewind, ref_rewind)
    if force == "frac":
        assert torch.equal(force_args[4], ref_force[4])


def _kda_manager_double(prev, dummy):
    """A ``KimiK3HybridCacheManager`` carrying only the state its strict-acceptance
    ``update_mamba_states`` reads (a KDA replay record over ``prev``)."""
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
        KDAReplayState,
        KimiK3HybridCacheManager,
    )

    replay = object.__new__(KDAReplayState)
    replay.prev_num_accepted_tokens = prev
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.kda_context_only = False
    mgr._kda_replay = replay
    mgr._speculative_state = replay
    mgr.local_num_mamba_layers = 1
    mgr._dummy_request_mask_buffer = None if dummy is None else SimpleNamespace(tensor=dummy)
    mgr._generation_state_indices = torch.zeros(64, dtype=torch.int32, device="cuda")
    return mgr


@pytest.mark.parametrize("glue", ["0", "1"])
@pytest.mark.parametrize("dummy_row", [None, 1])
def test_kda_replay_record_operands_match_update_mamba_states(glue, dummy_row, monkeypatch):
    """The operands ``kda_replay_record_operands`` hands the fused kernel record what
    ``update_mamba_states`` records, on its eager and its glue-fused path."""
    from tensorrt_llm._torch.pyexecutor import input_commit
    from tensorrt_llm._torch.speculative.spec_tail_kernels import fused_select_accept

    # The glue fusion runs on drafting Kimi K3.1 engines without attention DP.
    monkeypatch.setattr(input_commit, "_K31_DRAFTS", glue == "1")
    monkeypatch.setattr(input_commit, "_K31_ATTENTION_DP", False)
    K, n_slots, num_gens = 7, 32, 3
    rows = num_gens * (K + 1)
    gen = torch.Generator(device="cuda").manual_seed(11)
    buf = _exchanged_pairs(rows, n_slots, gen)
    drafts = _reduce_pairs(buf, n_slots).view(num_gens, K + 1)[:, :K].to(torch.int32).contiguous()
    drafts[0, 2:] += 1  # request 0 accepts 3 tokens, the others all K + 1
    pages = torch.randint(0, 7, (64, 4), device="cuda", generator=gen).to(torch.int32)
    slots = torch.tensor([5, 9, 40], device="cuda")
    dummy = torch.zeros(64, dtype=torch.bool, device="cuda")
    if dummy_row is not None:
        dummy[dummy_row] = True

    initial = pages.clone()
    ref_pages = pages.clone()
    ref_mgr = _kda_manager_double(ref_pages[:, 1], dummy)
    _, ref_n, _ = fused_select_accept((buf, buf.stride(0), 2, n_slots), drafts, 0, num_gens)
    md = SimpleNamespace(num_contexts=0, num_seqs=num_gens)
    type(ref_mgr).update_mamba_states(ref_mgr, md, ref_n, state_indices=slots)

    mgr = _kda_manager_double(pages[:, 1], dummy)
    operands = mgr.kda_replay_record_operands(slots)
    assert operands is not None and len(operands) == 3
    _, n, _ = fused_select_accept(
        (buf, buf.stride(0), 2, n_slots), drafts, 0, num_gens, kda_replay=operands
    )
    torch.cuda.synchronize()
    assert torch.equal(n, ref_n) and n.tolist() == [3, K + 1, K + 1]
    assert torch.equal(pages, ref_pages)
    assert pages[5, 1].item() == 2 and pages[40, 1].item() == K
    assert pages[9, 1].item() == (initial[9, 1].item() if dummy_row == 1 else K)
