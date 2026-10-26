# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 prefill indexer proxy: the max-score entries the selectors read are all written by the kernel, so the
unfilled buffer (msa_indexer._unfilled_max_score) gives the same scores and the same top-16 tables as fmha_sm100's
-inf-filled one."""

import math

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    require_msa_module,
)
from tensorrt_llm._utils import get_sm_version

HEADS, DIM, PAGE, SEG = 2, 128, 128, 128
SENTINEL = 0x7FC0BEEF  # a NaN payload the kernel never writes

sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 op"
)


def _inputs(cached, q, ties):
    g = torch.Generator(device="cuda").manual_seed(cached + q + ties)
    n_pages = math.ceil((cached + q) / PAGE)
    idx_k = (torch.randn(n_pages + 8, 1, PAGE, DIM, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    idx_q = (torch.randn(q, HEADS, DIM, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    if ties:
        idx_q = (
            torch.randint(-2, 3, (q, HEADS, DIM), device="cuda", generator=g)
            .float()
            .to(torch.float8_e4m3fn)
        )
        idx_k[: n_pages // 3] = 0
    pages = torch.randperm(n_pages + 8, device="cuda", generator=g)[:n_pages].to(torch.int32)
    return idx_q, idx_k, pages


def _rows(cached, q, segmented):
    if not segmented:
        return [(0, q, cached + q)]
    return [(s, min(SEG, q - s), cached + min(s + SEG, q)) for s in range(0, q, SEG)]


def _n_valid(rows, q):
    nv = torch.empty(q, dtype=torch.int32)
    for start, n, kv in rows:
        t = torch.arange(n)
        nv[start : start + n] = ((kv - n + t + 1 + PAGE - 1) // PAGE).to(torch.int32)
    return nv.cuda()


def _select(scores, n_valid):
    out = torch.empty((HEADS, scores.shape[2], 16), dtype=torch.int32, device="cuda").permute(
        1, 0, 2
    )
    torch.ops.trtllm.minimax_m3_select_prefill_tile(scores[:HEADS].contiguous(), n_valid, out, 0, 1)
    return out


@sm107
@pytest.mark.parametrize(
    "cached,q,segmented,splits,ties",
    [
        (165000, 512, False, 1, False),
        (165000, 2000, True, -1, False),
        (165000, 2000, True, -1, True),
        (65000, 8192, True, 1, True),
        (330000, 300, True, -1, False),
    ],
)
def test_unfilled_proxy_scores_match_the_filled_ones(cached, q, segmented, splits, ties):
    fmha = require_msa_module()
    idx_q, idx_k, pages = _inputs(cached, q, ties)
    rows = _rows(cached, q, segmented)
    qo = torch.tensor([r[1] for r in rows], dtype=torch.int32)
    kv = torch.tensor([r[2] for r in rows], dtype=torch.int32)
    kv_idx = torch.cat([pages[: math.ceil(r[2] / PAGE)] for r in rows])
    plan = fmha.fmha_sm100_plan(
        qo,
        kv,
        HEADS,
        num_kv_heads=1,
        qo_offset=kv - qo,
        page_size=PAGE,
        output_maxscore=True,
        causal=True,
        num_kv_splits=splits,
    )

    def proxy(max_score):
        _, scores = fmha.fmha_sm100(
            idx_q,
            idx_k,
            idx_k,
            plan,
            kv_indices=kv_idx,
            output_o=False,
            output_maxscore=True,
            max_score=max_score,
            sm_scale=DIM**-0.5,
        )
        return scores

    filled = proxy(None)
    buffer = msa_indexer._unfilled_max_score(plan, idx_q)
    assert buffer is not None and buffer.shape == filled.shape
    buffer.view(torch.int32).fill_(SENTINEL)
    unfilled = proxy(buffer)

    n_valid = _n_valid(rows, q)
    read = (
        torch.arange(filled.shape[1], device="cuda")[None, :, None] < n_valid[None, None, :]
    ).expand(HEADS, -1, -1)
    got = unfilled[:HEADS].view(torch.int32)
    want = filled[:HEADS].view(torch.int32)
    assert int(((got == SENTINEL) & read).sum()) == 0
    assert torch.equal(got[read], want[read])
    assert torch.equal(_select(unfilled, n_valid), _select(filled, n_valid))
