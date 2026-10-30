# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the CuTe DSL MiniMax-M3 index context scorer.

The bitwise comparison with the fmha_sm100 proxy it replaces is in test_msa_proxy_split.py.
"""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

PAGE_SIZE = 128
HEAD_DIM = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="The CuTe DSL MiniMax-M3 index context scorer requires SM100/SM103.",
)


def _inputs(num_heads, q_len, kv_len, seed=0):
    """FP8 index Q as a column view and the request's pages scattered over a coalesced pool."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    q = torch.randn(q_len, num_heads + 1, HEAD_DIM, device="cuda", generator=generator)
    q[:, 0] = float("nan")  # a neighbouring projection's columns
    pool = torch.randn(num_pages + 5, 2, PAGE_SIZE, HEAD_DIM, device="cuda", generator=generator)
    pool[:, 1] = float("nan")  # the rest of a coalesced pool page
    page_table = torch.randperm(num_pages + 5, device="cuda", generator=generator)[:num_pages]
    return (
        q.to(torch.float8_e4m3fn)[:, 1:],
        pool.to(torch.float8_e4m3fn)[:, :1],
        page_table.to(torch.int32),
    )


def _reference(q, k_cache, page_table, prefix_len):
    """[heads, blocks, q_len] causal per-block max of Q.K in fp32."""
    q_len = q.shape[0]
    kv_len = prefix_len + q_len
    blocks = page_table.numel()
    keys = k_cache[page_table.long(), 0].reshape(-1, HEAD_DIM)[:kv_len].float()
    scores = torch.einsum("qhd,kd->hqk", q.float(), keys)
    positions = prefix_len + torch.arange(q_len, device=q.device)
    key_positions = torch.arange(kv_len, device=q.device)
    scores.masked_fill_(key_positions[None, None] > positions[None, :, None], float("-inf"))
    scores = torch.nn.functional.pad(scores, (0, blocks * PAGE_SIZE - kv_len), value=float("-inf"))
    return scores.view(q.shape[1], q_len, blocks, PAGE_SIZE).amax(-1).transpose(1, 2)


@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize(
    ("q_len", "prefix_len"),
    [(1, 0), (77, 5), (128, 0), (129, 383), (1000, 130), (300, 4000), (2048, 32768)],
)
def test_context_score_matches_reference(num_heads, q_len, prefix_len):
    """Raw Q.K block maxima, -inf where a block has no visible key, written only into the request's slice."""
    from tensorrt_llm._torch.attention.kernels.blackwell.minimax_m3_index_context_score import (
        minimax_m3_index_context_score,
    )

    q, k_cache, page_table = _inputs(num_heads, q_len, prefix_len + q_len)
    blocks = page_table.numel()
    token_offset = 7
    out = torch.full(
        (num_heads, blocks + 3, token_offset + q_len + 11), float("nan"), device="cuda"
    )
    minimax_m3_index_context_score(q, k_cache, page_table, prefix_len, out, token_offset)

    written = out[:, :blocks, token_offset : token_offset + q_len]
    torch.testing.assert_close(
        written, _reference(q, k_cache, page_table, prefix_len), rtol=1e-5, atol=1e-5
    )
    written.fill_(float("nan"))
    assert out.isnan().all()


def test_context_score_rejects_unsupported_head_count():
    from tensorrt_llm._torch.attention.kernels.blackwell.minimax_m3_index_context_score import (
        minimax_m3_index_context_score,
    )

    q, k_cache, page_table = _inputs(3, 16, 16)
    out = torch.empty(3, 1, 16, device="cuda")
    with pytest.raises(ValueError, match="num_heads"):
        minimax_m3_index_context_score(q, k_cache, page_table, 0, out, 0)
