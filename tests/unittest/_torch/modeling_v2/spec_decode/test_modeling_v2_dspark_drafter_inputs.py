# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""catalog/spec_decode/dspark_drafter_inputs through the M3.1 targets' S10c-C entry: the drafter's step inputs (bonus,
query / ctx positions, noise block) are bit for bit what the stock lines of DFlashWorker.prepare_1st_drafter_inputs
compute, with the target's own embedding (M3EmbedTokens), eager and in a CUDA graph."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import dflash_noise_block_embedding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")

_ROOT = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
VOCAB, HIDDEN, MASK_ID, SLOTS = 1000, 6144, 997, 20


def _target(tp):
    return importlib.import_module(f"{_ROOT}.minimax_m31_nvfp4__sm_103__{tp}.dspark")


def _stock(embed, accepted, nacc, slots, ctx_len, max_ctx, trained, block, kp1):
    """The stock lines of prepare_1st_drafter_inputs' generation branch, verbatim."""
    bonus_idx = (nacc - 1).clamp_min(0).long().unsqueeze(1)
    bonus = accepted.gather(1, bonus_idx).squeeze(1).long()
    ctx_len_gen = ctx_len[slots]
    j_block = torch.arange(0, block, dtype=torch.long, device="cuda")
    offsets_kp1 = torch.arange(0, kp1, dtype=torch.long, device="cuda")
    ctx_len_now = (ctx_len_gen + nacc.long()).clamp_(max=max_ctx)
    query_position_ids = ctx_len_now.unsqueeze(1) + j_block.unsqueeze(0)
    ctx_position_ids = ctx_len_gen.unsqueeze(1) + offsets_kp1.unsqueeze(0)
    noise = dflash_noise_block_embedding(embed, bonus, MASK_ID, block, trained)
    return bonus, query_position_ids, ctx_position_ids, noise


def _case(n, k, acc_dtype, seed, max_ctx):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kp1 = k + 1
    accepted = torch.randint(0, VOCAB, (n, kp1), generator=g, device="cuda").to(acc_dtype)
    accepted[0, :] = -1  # a padding request: its bonus id is outside the table
    nacc = torch.randint(1, kp1 + 1, (n,), generator=g, device="cuda").int()
    slots = torch.randperm(SLOTS, generator=g, device="cuda")[:n]
    ctx_len = torch.randint(0, max_ctx + 1, (SLOTS,), generator=g, device="cuda").long()
    ctx_len[slots[-1]] = max_ctx - 1  # runs into the clamp
    return accepted, nacc, slots, ctx_len


def _embed(tp):
    g = torch.Generator(device="cuda").manual_seed(7)
    table = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda").bfloat16()
    return _target(tp).M3EmbedTokens(SimpleNamespace(w={"embed": table}))


def _equal(a, b):
    return all(
        x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)
    )


@pytest.mark.parametrize("tp", ["tp4", "tp8"])
@pytest.mark.parametrize("k", [3, 7])
@pytest.mark.parametrize("acc_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("trained", [False, True])
def test_inputs_are_the_stock_inputs_bit_for_bit(tp, k, acc_dtype, trained):
    embed = _embed(tp)
    mask_row = (torch.randn(HIDDEN, device="cuda") * 3).bfloat16() if trained else None
    for n in (1, 3, 8, 16):
        for seed in range(3):
            accepted, nacc, slots, ctx_len = _case(n, k, acc_dtype, seed, max_ctx=4096)
            args = (embed, accepted, nacc, slots, ctx_len, 4096)
            ref = _stock(*args, mask_row, k, k + 1)
            got = _target(tp).s10c_c_drafter_inputs(*args, MASK_ID, mask_row, k, k + 1)
            assert got is not None
            torch.cuda.synchronize()
            assert _equal(ref, got), (n, seed)


def test_inputs_in_a_cuda_graph():
    embed = _embed("tp8")
    accepted, nacc, slots, ctx_len = _case(8, 3, torch.int32, 0, max_ctx=4096)
    fn = _target("tp8").s10c_c_drafter_inputs
    fn(embed, accepted, nacc, slots, ctx_len, 4096, MASK_ID, None, 3, 4)  # compile outside capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            got = fn(embed, accepted, nacc, slots, ctx_len, 4096, MASK_ID, None, 3, 4)
    torch.cuda.current_stream().wait_stream(s)
    a2, n2, s2, c2 = _case(8, 3, torch.int32, 1, max_ctx=4096)
    accepted.copy_(a2), nacc.copy_(n2), slots.copy_(s2), ctx_len.copy_(c2)
    g.replay()
    ref = _stock(embed, accepted, nacc, slots, ctx_len, 4096, None, 3, 4)
    torch.cuda.synchronize()
    assert _equal(ref, got)


def test_other_embeddings_keep_the_stock_path():
    accepted, nacc, slots, ctx_len = _case(2, 3, torch.int32, 0, max_ctx=4096)
    other = torch.nn.Embedding(VOCAB, HIDDEN).cuda().bfloat16()
    fn = _target("tp8").s10c_c_drafter_inputs
    assert fn(other, accepted, nacc, slots, ctx_len, 4096, MASK_ID, None, 3, 4) is None
