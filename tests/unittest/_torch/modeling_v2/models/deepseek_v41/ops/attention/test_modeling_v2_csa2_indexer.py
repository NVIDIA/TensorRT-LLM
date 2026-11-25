# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_indexer entry (the fused indexers' reference): the top-k over paged index keys and the
candidate-block filter.

`csa2_indexer_topk` (dense, any GPU): run under IEEE fp32 matmuls, the op's scores and the dense
reference's differ only in summation order (~1e-6 relative); the cells' seeds are fixed and the
picks are compared exactly (a pick flips only when two keys score within that gap of the k-th
place, which random fp32 keys of these sizes do not do -- the run is the evidence).
"""

import pytest
import torch
import torch.nn.functional as F
from _v41_util import DEV, gen_for, index_pool

__extra_import_path__ = ["../../../.."]
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer import (  # noqa: E501
    csa2_candidate_blocks,
    csa2_indexer_topk,
)


def _inputs(spec, cell):
    gen = gen_for(cell)
    pool, table, deq = index_pool(spec["ratio"], spec["nkeys"], gen)
    n, h = len(spec["qpos"]), spec["heads"]
    q = torch.randn(n, h, 128, generator=gen, device=DEV).bfloat16()
    w = torch.rand(n, h, generator=gen, device=DEV)
    pos = torch.tensor(spec["qpos"], device=DEV, dtype=torch.int32)
    return pool, table, deq, q, w, pos, gen


def _ref_select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """The checkpoint's select_candidate_blocks, verbatim."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(scores.size(-1), device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, scores.size(-1)), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


@pytest.mark.parametrize(
    "cell", csa2_indexer_topk.CELLS, ids=[c.why[:44] for c in csa2_indexer_topk.CELLS]
)
def test_dense_cells(cell, certifying):  # noqa: F811
    certifying(csa2_indexer_topk)
    s = cell.spec
    pool, table, _, q, w, pos, gen = _inputs(s, cell)
    n = q.shape[0]
    segs = [(0, n, 0, s["nkeys"] * s["ratio"])]
    cb, ct = s.get("cand", (0, 0))
    cand_in = None
    with true_fp32_matmul():
        if s.get("second"):  # the consumer layer: masks come from the source layer's call
            _, cand_in = csa2_indexer_topk.reference(
                q, w, pos, segs, pool, table, s["ratio"], s["topk"], cb, ct, None, True
            )
            q = torch.randn(q.shape, generator=gen, device=DEV).bfloat16()
            kw = dict(cand_in=cand_in, want_cand=False)
        else:
            kw = dict(cand_in=None, want_cand=bool(cb))
        args = (q, w, pos, segs, pool, table, s["ratio"], s["topk"], cb, ct)
        got, got_c = csa2_indexer_topk(*args, **kw)
        ref, ref_c = csa2_indexer_topk.reference(*args, **kw)
    assert torch.equal(got, ref)
    if kw["want_cand"]:
        assert len(got_c) == len(ref_c) == 1 and torch.equal(got_c[0], ref_c[0])
    else:
        assert got_c is None and ref_c is None


def test_candidate_blocks_match_the_checkpoint_formulation():
    gen = torch.Generator(device=DEV).manual_seed(5)
    score = torch.randn(3, 200, generator=gen, device=DEV)
    c_len = torch.tensor([[151], [200], [37]], device=DEV)
    score = score.masked_fill(torch.arange(200, device=DEV) >= c_len, -torch.inf)
    assert torch.equal(
        csa2_candidate_blocks(score, c_len, 5, 8), _ref_select_candidate_blocks(score, c_len, 5, 8)
    )


def test_dense_rejections():
    s = dict(ratio=1, nkeys=40, qpos=(5, 30), heads=4, topk=8)
    cell = csa2_indexer_topk.CELLS[0]
    pool, table, _, q, w, pos, _ = _inputs(s, cell)
    segs = [(0, 2, 0, 40)]
    csa2_indexer_topk.is_valid(q, w, pos, segs, pool, table, 1, 8)
    with pytest.raises(AssertionError, match="ratio"):
        csa2_indexer_topk.is_valid(q, w, pos, segs, pool, table, 3, 8)
    with pytest.raises(AssertionError, match="span"):  # keys past the span would never be scored
        csa2_indexer_topk.is_valid(q, w, pos, [(0, 2, 0, 20)], pool, table, 1, 8)
    with pytest.raises(AssertionError, match="fp32"):
        csa2_indexer_topk.is_valid(q, w.bfloat16(), pos, segs, pool, table, 1, 8)
    with pytest.raises(AssertionError, match="uint8"):
        csa2_indexer_topk.is_valid(q, w, pos, segs, pool.float(), table, 1, 8)
    with pytest.raises(AssertionError, match="cand_block"):
        csa2_indexer_topk.is_valid(q, w, pos, segs, pool, table, 1, 8, 0, 0, None, True)


def test_consumer_pads_picks_past_its_candidates():
    """A consumer whose candidates (positions 8..15) are fewer than `topk` (12): the picks are those eight positions,
    ascending, then -1 (top-k used to fill the four spare slots with excluded positions below `c_len`)."""
    s = dict(ratio=1, nkeys=40, qpos=(30, 39), heads=4, topk=12)
    pool, table, _, q, w, pos, _ = _inputs(s, csa2_indexer_topk.CELLS[0])
    segs = [(0, 2, 0, 40)]
    mask = torch.zeros(2, 40, dtype=torch.bool, device=DEV)
    mask[:, 8:16] = True
    want = [list(range(8, 16)) + [-1] * 4] * 2
    with true_fp32_matmul():
        got, _ = csa2_indexer_topk(q, w, pos, segs, pool, table, 1, 12, 0, 0, [mask])
        ref, _ = csa2_indexer_topk.reference(q, w, pos, segs, pool, table, 1, 12, 0, 0, [mask])
    assert got.tolist() == want and ref.tolist() == want
