# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash MTP layer's fused kernels against the modules they replace."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.kernels import (  # noqa: E501
    advance_after_step0,
    drafting_rows,
    embed_rms_norm_columns,
    mtp_commit,
    verify_rows,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton kernel)")


@pytest.mark.parametrize("hidden", [4096, 96])
@pytest.mark.parametrize("cols", [None, (0, 0.5), (0.5, 1.0)])
@pytest.mark.parametrize("rows", [0, 1, 7, 257])
def test_embed_rms_norm_columns_matches_modules(hidden, cols, rows):
    torch.manual_seed(0)
    vocab, eps = 1000, 1e-5
    embed = Embedding(vocab, hidden, dtype=torch.bfloat16).cuda()
    norm = RMSNorm(hidden_size=hidden, eps=eps, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        embed.weight.normal_()
        norm.weight.uniform_(0.5, 1.5)
    ids = torch.randint(0, vocab, (rows,), dtype=torch.int32, device="cuda")
    if rows > 1:
        ids[0] = -1  # rejected draft slot padding
        ids[-1] = vocab + 3
    lo, hi = (0, hidden) if cols is None else (int(cols[0] * hidden), int(cols[1] * hidden))

    got = embed_rms_norm_columns(ids, embed.weight, norm.weight, eps, lo, hi)
    if rows == 0:  # (the module path cannot launch its norm on zero rows)
        assert got.shape == (0, hi - lo) and got.dtype == embed.weight.dtype
        return
    want = norm(embed(ids))[:, lo:hi]

    assert got.shape == want.shape and got.dtype == want.dtype
    torch.testing.assert_close(got, want, atol=0, rtol=1.6e-2)
    if rows > 1:
        assert not got[0].any() and not got[-1].any()


@pytest.mark.parametrize("n", [0, 1, 5, 64, 300])
def test_drafting_rows_matches_torch(n):
    width = 4
    batch_indices = torch.arange(n, dtype=torch.int32, device="cuda")
    num_accepted = torch.randint(1, width + 1, (n,), dtype=torch.int32, device="cuda")
    got = drafting_rows(batch_indices, num_accepted, width)
    want = batch_indices.long() * width + num_accepted - 1
    assert got.dtype == torch.int64
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("n", [1, 3, 64])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "pool_column"])
def test_mtp_commit_matches_cache_manager_update(n, strided):
    """The fused drafting rows + KDA replay record against drafting_rows and the cache manager's torch update,
    also into a strided count (KDAReplayState's is column 0 of a per-slot state buffer); the other columns stay."""
    torch.manual_seed(n)
    width = 4
    batch_indices = torch.randperm(n, device="cuda").to(torch.int32)
    num_accepted = torch.randint(1, width + 1, (n,), dtype=torch.int32, device="cuda")
    state_indices = torch.randperm(2 * n, device="cuda")[:n].to(torch.int32)
    dummy = torch.rand(n, device="cuda") < 0.25
    pool = torch.randint(0, 4, (2 * n + 1, 5 if strided else 1), dtype=torch.int32, device="cuda")
    want_pool = pool.clone()
    prev = pool[:, 0]
    want_prev = want_pool[:, 0]
    current = want_prev[state_indices.long()]
    accepted = (num_accepted - 1).to(torch.int32).clamp(min=0)
    want_prev[state_indices.long()] = torch.where(dummy, current, accepted)
    rows = mtp_commit(batch_indices, num_accepted, width, state_indices, dummy, prev)
    assert rows.dtype == torch.int32
    assert torch.equal(rows.long(), drafting_rows(batch_indices, num_accepted, width))
    assert torch.equal(pool, want_pool)


@pytest.mark.parametrize("kv_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens_per_request", [2, 4])
def test_verify_rows_match_the_torch_formulas(kv_dtype, tokens_per_request):
    """Positions, visible lengths, complete-pool counts and block-table rows of the verify rows in one launch."""
    kv_lens = torch.tensor([4, 9, 17, 33, 100], dtype=kv_dtype, device="cuda")
    kpool, first_table, t = 4, 3, tokens_per_request
    positions, visible, num_candidates, request_ids = verify_rows(kv_lens, t, kpool, first_table)
    want_positions = (kv_lens - t).unsqueeze(1) + torch.arange(t, device="cuda")
    want_visible = (want_positions + 1).reshape(-1)
    assert positions.dtype == want_positions.dtype and torch.equal(positions, want_positions)
    assert visible.dtype == want_visible.dtype and torch.equal(visible, want_visible)
    assert num_candidates.dtype == torch.int32
    assert torch.equal(num_candidates, (want_visible // kpool).to(torch.int32))
    want_ids = torch.arange(
        first_table, first_table + kv_lens.shape[0], dtype=torch.int32, device="cuda"
    ).repeat_interleave(t)
    assert torch.equal(request_ids, want_ids)


@pytest.mark.parametrize("num_contexts", [0, 2])
def test_advance_after_step0_matches_the_torch_updates(num_contexts):
    """Step 1's device lengths in one launch: sequence lengths of one, context KV lengths plus one, generation KV
    lengths minus the rejected drafts."""
    n, draft_len = 7, 3
    seq_lens = torch.randint(1, 9, (n + 2,), dtype=torch.int32, device="cuda")
    kv_lens = torch.randint(10, 100, (n + 2,), dtype=torch.int32, device="cuda")
    accepted = torch.randint(1, draft_len + 2, (n,), dtype=torch.int32, device="cuda")
    want_seq, want_kv = seq_lens.clone(), kv_lens.clone()
    want_seq[:n].fill_(1)
    want_kv[num_contexts:n] -= draft_len - accepted[num_contexts:]
    want_kv[:num_contexts] += 1
    advance_after_step0(seq_lens[:n], kv_lens[:n], accepted, num_contexts, draft_len)
    assert torch.equal(seq_lens, want_seq)
    assert torch.equal(kv_lens, want_kv)
