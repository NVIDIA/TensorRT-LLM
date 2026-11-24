# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the small DSA entries of the specialized targets: rubin_dsa_topk_remap (Triton remap),
rubin_pad_heads (pad128 buffers), rubin_helix_zero_or_empty_rows (helix combine mask) and
rubin_merge_helix_topk (the helix fallback merge).

CPU: each reference on small hand-checked inputs, the remap reference against the C++ op's
reference (``convert_req_index_to_global``), the merge's ``__call__`` (plain torch) against its
reference and against the engine's ``merge_helix_topk`` algorithm run over a list of ranks, the
guards. GPU (sm_107): every cell bit for bit against the reference (integers, copies and zeros,
compares).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_topk_remap import (  # noqa: E501
    rubin_dsa_topk_remap as remap_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_helix_zero_or_empty_rows import (  # noqa: E501
    rubin_helix_zero_or_empty_rows as mask_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_merge_helix_topk import (  # noqa: E501
    rubin_merge_helix_topk as merge_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_pad_heads import (  # noqa: E501
    rubin_pad_heads as pad_op,
)

__extra_import_path__ = ["../../../..", "."]
from _validating import certifying, validating  # noqa: E402,F401

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin family entries are certified on sm_107",
)


def _remap_inputs(spec, gen, device="cpu"):
    rows, reqs, pages, bs = spec["rows"], spec["requests"], spec["pages"], spec["block_size"]
    table = torch.randperm(reqs * pages * 2, generator=gen)[: reqs * pages].view(reqs, pages)
    table = table.to(torch.int32)
    if spec.get("holes"):
        table[:, ::5] = -1
    req = torch.randint(0, reqs, (rows,), generator=gen, dtype=torch.int32)
    hi = pages * bs + (bs if spec.get("holes") else 0)
    tok = torch.randint(-1, hi, (rows, spec["topk"]), generator=gen, dtype=torch.int32)
    return [t.to(device) for t in (req, table, tok)]


def _merge_engine(gathered, topk_rows, cp_rank):
    """The engine's merge_helix_topk after its all-gather (dsa/indexer.py), for comparison."""
    num_tokens, k = topk_rows.shape
    all_scores = gathered.transpose(0, 1).reshape(num_tokens, -1)
    top_scores, top_pos = torch.topk(all_scores, k, dim=1)
    owner = torch.div(top_pos, k, rounding_mode="floor")
    mine = (owner == cp_rank) & torch.isfinite(top_scores)
    hits = torch.zeros_like(topk_rows, dtype=torch.int32)
    hits.scatter_add_(1, torch.where(mine, top_pos - owner * k, 0), mine.to(torch.int32))
    keep = hits > 0
    order = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices
    return torch.gather(topk_rows.masked_fill(~keep, -1), 1, order)


def _merge_inputs(spec, gen):
    cp, rows, k = spec["cp"], spec["rows"], spec["k"]
    # Distinct scores across every rank: a permutation scaled into (0, 1).
    scores = (torch.randperm(cp * rows * k, generator=gen).float() + 1) / (cp * rows * k + 1)
    scores = scores.view(cp, rows, k)
    local = torch.stack([torch.randperm(10 * k, generator=gen)[:k] for _ in range(rows)]).to(
        torch.int32
    )
    drop = torch.rand(cp, rows, k, generator=gen) > spec["fill"]
    scores = scores.masked_fill(drop, float("-inf"))
    return scores, local, drop


# --- CPU -------------------------------------------------------------------------------------


def test_remap_reference_matches_the_cpp_op_reference():
    gen = torch.Generator().manual_seed(0)
    spec = dict(rows=37, requests=3, pages=5, block_size=8, topk=1024, holes=True)
    req, table, tok = _remap_inputs(spec, gen)
    args = (req, table, tok, 8, 1024, 8 * 4, 2)
    assert torch.equal(remap_op.reference(*args), convert_req_index_to_global.reference(*args))


def test_remap_guard_refuses_a_ragged_width():
    req, table, tok = _remap_inputs(
        dict(rows=4, requests=1, pages=4, block_size=8, topk=1000), torch.Generator()
    )
    with pytest.raises(AssertionError, match="multiple of 1024"):
        remap_op.is_valid(req, table, tok, 8, 1000, 8, 0)


def test_pad_reference_copies_and_zeros():
    src = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    dst = torch.full((2, 5, 4), 9.0)
    out = pad_op.reference(dst, src, 3)
    assert torch.equal(out[:, :3], src) and (out[:, 3:] == 0).all()
    zeroed = pad_op.reference(dst, None, 3)
    assert (zeroed[:, :3] == 9).all() and (zeroed[:, 3:] == 0).all()
    assert (dst == 9).all(), "the reference leaves dst untouched"


def test_mask_reference():
    bounds = torch.tensor([0, 3, 5, 0], dtype=torch.int32)
    counts = torch.tensor([4, 0, 2, 0], dtype=torch.int32)
    assert mask_op.reference(bounds, counts).tolist() == [True, True, False, True]


@pytest.mark.parametrize("cell", merge_op.CELLS, ids=[c.why[:48] for c in merge_op.CELLS])
def test_merge_call_equals_reference_and_engine_on_cpu(cell):
    """``__call__`` is plain torch: run it on CPU for every cell, against the reference and the
    engine's algorithm; across the ranks the kept entries partition the global Top-K."""
    spec = cell.spec
    gen = torch.Generator().manual_seed(spec["rows"] + spec["cp"])
    scores, local, _ = _merge_inputs(spec, gen)
    kept_total = 0
    for rank in range(spec["cp"]):
        rows = local.clone()
        rows = rows.masked_fill(~torch.isfinite(scores[rank]), -1)
        engine = _merge_engine(scores, rows.clone(), rank)
        ref = merge_op.reference(scores, rows, rank)
        got = merge_op(scores, rows, rank)
        assert torch.equal(got, ref) and torch.equal(got, engine)
        kept_total += int((got >= 0).sum())
    finite = torch.isfinite(scores).transpose(0, 1).reshape(spec["rows"], -1).sum(1)
    assert kept_total == int(finite.clamp(max=spec["k"]).sum())


def test_merge_guard():
    with pytest.raises(AssertionError, match="gathered must be"):
        merge_op.is_valid(torch.zeros(2, 3, 4), torch.zeros(3, 5, dtype=torch.int32), 0)


# --- GPU -------------------------------------------------------------------------------------


@on_sm107
@pytest.mark.parametrize("cell", remap_op.CELLS, ids=[c.why[:48] for c in remap_op.CELLS])
def test_remap_certified_cells(cell, certifying):  # noqa: F811
    certifying(remap_op)
    s = cell.spec
    req, table, tok = _remap_inputs(s, torch.Generator().manual_seed(s["rows"]), "cuda")
    stride = s["layers"] * s["block_size"]
    args = (req, table, tok, s["block_size"], s["topk"], stride, s["layers"] - 1)
    out = remap_op(*args)
    assert torch.equal(out, remap_op.reference(*args))
    assert torch.equal(out, convert_req_index_to_global(*args)), "the C++ op's integers"


@on_sm107
@pytest.mark.parametrize("cell", pad_op.CELLS, ids=[c.why[:48] for c in pad_op.CELLS])
def test_pad_certified_cells(cell, certifying):  # noqa: F811
    certifying(pad_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["n"])
    dst = torch.randn(s["n"], s["pad"], s["d"], generator=gen).to(s["dtype"]).cuda()
    src = None
    if s["src"]:
        src = torch.randn(s["n"], s["heads"], s["d"], generator=gen).to(s["dtype"]).cuda()
    ref = pad_op.reference(dst, src, s["heads"])
    assert torch.equal(pad_op(dst, src, s["heads"]), ref)


@on_sm107
@pytest.mark.parametrize("cell", mask_op.CELLS, ids=[c.why[:48] for c in mask_op.CELLS])
def test_mask_certified_cells(cell, certifying):  # noqa: F811
    certifying(mask_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["rows"])
    bounds = torch.randint(1, 100, (s["rows"],), generator=gen, dtype=torch.int32)
    counts = torch.randint(1, 100, (s["rows"],), generator=gen, dtype=torch.int32)
    bounds[torch.rand(s["rows"], generator=gen) < s["zero_frac"]] = 0
    counts[torch.rand(s["rows"], generator=gen) < s["empty_frac"]] = 0
    out = mask_op(bounds.cuda(), counts.cuda())
    assert torch.equal(out.cpu(), mask_op.reference(bounds, counts))


@on_sm107
@pytest.mark.parametrize("cell", merge_op.CELLS, ids=[c.why[:48] for c in merge_op.CELLS])
def test_merge_certified_cells(cell, certifying):  # noqa: F811
    certifying(merge_op)
    spec = cell.spec
    scores, local, _ = _merge_inputs(spec, torch.Generator().manual_seed(spec["rows"]))
    rows = local.masked_fill(~torch.isfinite(scores[0]), -1)
    ref = merge_op.reference(scores, rows, 0)
    assert torch.equal(merge_op(scores.cuda(), rows.cuda(), 0).cpu(), ref)
