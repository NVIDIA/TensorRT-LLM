# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_topk_merge`` (fast path ``draft_head``) with and without a drafter logit transform.

Gated with ``torch.equal`` on the ids and bitwise on the values against what
the built-in path computes: ``torch.topk`` over the gathered fp32 logits, or,
with the transform, over the drafter's transform of them (each shard
transformed by the drafter's own method, then gathered). Then against the
entry's own reference, and the einsum operands against ``dflash2_score_edges``.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    draft_topk_merge as merge_mod,
)
from tensorrt_llm._torch.models.modeling_dflash import dflash2_score_edges

__extra_import_path__ = ["../../../../..", "../../kernels/draft_head"]
from _draft_head_cases import (  # noqa: E402
    FAMILIES,
    MUSESPARK,
    RANK,
    TOP_K,
    VOCAB,
    assert_same_floats,
    builtin_candidates,
    draft_topk_merge,
    layout_keys,
    logits,
    merge_launch,
    seed_of,
    selector,
    split,
    transform,
)
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"

CELLS = merge_mod.DraftTopKMerge.CELLS


def _run(requests, positions, ranks, layout, kind, config, seed, canon=True):
    """(merge outputs, built-in (unary, ids), gate, anchor) for one draft step."""
    rows = requests * positions
    t = None if config is None else transform(config)
    full = logits(kind, rows, VOCAB, seed, shard=VOCAB // ranks)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    gate = torch.randn(requests, positions, RANK, generator=g, device="cuda").bfloat16()
    anchor = torch.randint(0, VOCAB, (requests,), generator=g, device="cuda")
    keys = layout_keys(layout, full, ranks, t if canon else None)
    sel = selector()
    args = (
        keys,
        VOCAB // ranks,
        requests,
        positions,
        anchor,
        gate,
        sel.predecessor_codebook,
        sel.successor_codebook,
        TOP_K,
        ranks,
    )
    kwargs = dict(merge_launch(layout, rows))
    if t is not None:
        kwargs["values"] = t.values
    out = draft_topk_merge(*args, **kwargs)
    ref = builtin_candidates(split(full, ranks), requests, positions, config)
    return out, ref, args, kwargs, gate, anchor


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_topk_merge_cells(cell, certifying):  # noqa: F811
    certifying(draft_topk_merge)
    spec = cell.spec
    config = None if spec["transform"] is None else MUSESPARK
    requests, positions, ranks = spec["requests"], spec["positions"], spec["ranks"]
    for kind in ("gemm3", "special_nan", "straddle_up", "saturated"):
        (ids, unary, preds, succs), (ref_unary, ref_ids), args, kwargs, gate, anchor = _run(
            requests, positions, ranks, spec["layout"], kind, config, seed_of(cell.why + kind)
        )
        assert torch.equal(ids, ref_ids), f"{kind}: candidate ids or their order differ"
        assert_same_floats(unary, ref_unary)
        r_ids, r_unary, r_preds, r_succs = draft_topk_merge.reference(*args, **kwargs)
        assert torch.equal(ids, r_ids)
        assert_same_floats(unary, r_unary)
        assert torch.equal(preds, r_preds) and torch.equal(succs, r_succs)
        sel = selector()
        builtin = dflash2_score_edges(
            sel.predecessor_codebook, sel.successor_codebook, ref_ids, ref_unary, gate, anchor
        )
        ours = unary.float()[:, :, None] + torch.einsum("blpr,blcr->blpc", preds, succs)
        assert_same_floats(ours, builtin)


@pytest.mark.parametrize("kind", FAMILIES)
@pytest.mark.parametrize("layout", ["int32", "halves", "rank"])
@pytest.mark.parametrize("ranks", [4, 8])
@pytest.mark.parametrize("requests", [1, 2, 8, 24])
def test_transformed_candidates_are_the_builtin_ones(requests, ranks, layout, kind):
    (ids, unary, _, _), (ref_unary, ref_ids), *_ = _run(
        requests, 7, ranks, layout, kind, MUSESPARK, seed_of(f"{requests}{ranks}{layout}{kind}")
    )
    differ = (ids != ref_ids).any(dim=-1).view(-1).nonzero().view(-1).tolist()
    assert not differ, f"rows {differ} differ from torch.topk over the transformed logits"
    assert_same_floats(unary, ref_unary)


@pytest.mark.parametrize("kind", ["gemm3", "grid", "special_nan", "edges"])
@pytest.mark.parametrize("layout", ["int32", "halves", "rank"])
@pytest.mark.parametrize("requests", [8, 24])
def test_untransformed_control(requests, layout, kind):
    """The same matrix without a transform: a failure here is the merge's, not the transform's."""
    (ids, unary, _, _), (ref_unary, ref_ids), *_ = _run(
        requests, 7, 8, layout, kind, None, seed_of(f"control{requests}{layout}{kind}")
    )
    assert torch.equal(ids, ref_ids)
    assert_same_floats(unary, ref_unary)


@pytest.mark.parametrize("kind", ["straddle_up", "straddle_down", "saturated"])
def test_without_canonical_keys_the_ties_go_wrong(kind):
    """The negative control: bf16 keys order saturated logits by their bf16 values, not by id."""
    (ids, _, _, _), (_, ref_ids), *_ = _run(1, 7, 8, "halves", kind, MUSESPARK, 3, canon=False)
    assert not torch.equal(ids, ref_ids)
