# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`draft_loop_ops` must equal the composed torch ops of the one-model draft loop element for
element, eagerly and replayed from a CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch.speculative.draft_loop_ops import draft_row_select, next_new_tokens
from tensorrt_llm._torch.speculative.mtp import _select_mtp_position_ids

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DEV = "cuda"


def _row_select_reference(batch_indices, accepted, hidden, positions, p, num_ctx_tokens):
    ids = (batch_indices * p).long() + accepted - 1 + num_ctx_tokens
    return ids, hidden[ids], _select_mtp_position_ids(positions, ids) + 1


def _inputs(n, p, width, pos_shape, g, accepted=None, stride_pad=0):
    rows = n * p
    hidden = torch.randn(rows, width + stride_pad, generator=g).to(DEV, torch.bfloat16)
    hidden = hidden[:, :width]
    if accepted is None:
        accepted = torch.randint(1, p + 1, (n,), generator=g)
    accepted = accepted.to(DEV, torch.int32)
    batch_indices = torch.arange(n, dtype=torch.int32, device=DEV)
    full = torch.randint(0, 1 << 20, pos_shape[:-1] + (rows + 7,), generator=g).to(DEV, torch.int32)
    positions = full[..., :rows]
    return batch_indices, accepted, hidden, positions


@pytest.mark.parametrize("pos_shape", [(3, 1, 0), (3, 0), (0,)])
@pytest.mark.parametrize("n,p,width", [(1, 4, 10240), (7, 4, 1000), (64, 8, 2048), (256, 2, 64)])
def test_row_select_matches_composed_ops(n, p, width, pos_shape):
    g = torch.Generator().manual_seed(n * 31 + p)
    batch_indices, accepted, hidden, positions = _inputs(n, p, width, pos_shape, g, stride_pad=5)
    ref = _row_select_reference(batch_indices, accepted, hidden, positions, p, 0)
    got = draft_row_select(batch_indices, accepted, hidden, positions, p, 0)
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b)


def test_row_select_wraps_a_negative_row_like_torch():
    g = torch.Generator().manual_seed(3)
    accepted = torch.tensor([0, 2, 4], dtype=torch.int32)
    batch_indices, accepted, hidden, positions = _inputs(3, 4, 96, (3, 1, 0), g, accepted)
    ref = _row_select_reference(batch_indices, accepted, hidden, positions, 4, 0)
    got = draft_row_select(batch_indices, accepted, hidden, positions, 4, 0)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_row_select_cuda_graph_follows_accepted_counts():
    g = torch.Generator().manual_seed(4)
    batch_indices, accepted, hidden, positions = _inputs(16, 4, 512, (3, 1, 0), g)
    draft_row_select(batch_indices, accepted, hidden, positions, 4, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = draft_row_select(batch_indices, accepted, hidden, positions, 4, 0)
    for _ in range(4):
        accepted.copy_(torch.randint(1, 5, (16,), generator=g).to(DEV, torch.int32))
        hidden.copy_(torch.randn(hidden.shape, generator=g).to(DEV, torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        ref = _row_select_reference(batch_indices, accepted, hidden, positions, 4, 0)
        for a, b in zip(out, ref):
            assert torch.equal(a, b)


def _next_new_reference(accepted_tokens, drafts, batch_indices, accepted):
    golden = accepted_tokens[batch_indices, accepted - 1].unsqueeze(1)
    return torch.concat([golden, drafts], dim=1)


@pytest.mark.parametrize("n,d", [(1, 3), (5, 1), (32, 3), (256, 7)])
def test_next_new_tokens_matches_composed_ops(n, d):
    g = torch.Generator().manual_seed(n + d)
    accepted_tokens = torch.randint(0, 248320, (n, d + 1), generator=g).to(DEV, torch.int32)
    drafts = torch.randint(0, 248320, (n, d), generator=g).to(DEV, torch.int32)
    accepted = torch.randint(0, d + 2, (n,), generator=g).to(DEV, torch.int32)
    batch_indices = torch.randperm(n, generator=g).to(DEV, torch.int32)
    ref = _next_new_reference(accepted_tokens, drafts, batch_indices, accepted)
    got = next_new_tokens(accepted_tokens, drafts, batch_indices, accepted)
    assert got.dtype == ref.dtype and torch.equal(got, ref)


def test_next_new_tokens_cuda_graph():
    g = torch.Generator().manual_seed(9)
    n, d = 8, 3
    accepted_tokens = torch.randint(0, 1000, (n, d + 1), generator=g).to(DEV, torch.int32)
    drafts = torch.randint(0, 1000, (n, d), generator=g).to(DEV, torch.int32)
    accepted = torch.ones(n, dtype=torch.int32, device=DEV)
    batch_indices = torch.arange(n, dtype=torch.int32, device=DEV)
    next_new_tokens(accepted_tokens, drafts, batch_indices, accepted)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = next_new_tokens(accepted_tokens, drafts, batch_indices, accepted)
    for _ in range(3):
        accepted.copy_(torch.randint(1, d + 2, (n,), generator=g).to(DEV, torch.int32))
        drafts.copy_(torch.randint(0, 1000, (n, d), generator=g).to(DEV, torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            out, _next_new_reference(accepted_tokens, drafts, batch_indices, accepted)
        )


@pytest.mark.parametrize("n,d", [(1, 3), (32, 3), (100, 7)])
def test_row_select_moves_the_lengths_to_one_row(n, d):
    """`kv_lens[:n] -= d - accepted` and `seq_lens[:n] = 1`, entries past `n` untouched."""
    g = torch.Generator().manual_seed(n * 3 + d)
    p = d + 1
    batch_indices, accepted, hidden, positions = _inputs(n, p, 128, (3, 1, 0), g)
    kv = torch.randint(100, 9000, (n + 5,), generator=g).to(DEV, torch.int32)
    seq = torch.full((n + 5,), p, dtype=torch.int32, device=DEV)
    kv_ref, seq_ref = kv.clone(), seq.clone()
    kv_ref[:n] -= d - accepted
    seq_ref[:n].fill_(1)
    got = draft_row_select(
        batch_indices, accepted, hidden, positions, p, 0, kv_lens=kv, seq_lens=seq, draft_len=d
    )
    ref = _row_select_reference(batch_indices, accepted, hidden, positions, p, 0)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)
    assert torch.equal(kv, kv_ref)
    assert torch.equal(seq, seq_ref)
