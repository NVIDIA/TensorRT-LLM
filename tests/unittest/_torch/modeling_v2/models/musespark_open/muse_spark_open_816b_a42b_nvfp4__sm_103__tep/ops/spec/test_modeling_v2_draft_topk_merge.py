# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_topk_merge`` (fast path ``draft_head``).

Gated on its cells with ``torch.equal`` against ``torch.topk`` over the full
fp32 logits (ids, values, order, sign of zero), against its own reference, and
against the built-in ``dflash2_score_edges``, in both key layouts.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    draft_topk_merge as merge_mod,
)
from tensorrt_llm._torch.models.modeling_dflash import dflash2_score_edges

__extra_import_path__ = ["../../../../..", "../../kernels/draft_head"]
from _draft_head_cases import (  # noqa: E402
    RANK,
    TOP_K,
    TP,
    VOCAB,
    assert_same_floats,
    draft_topk_merge,
    logits,
    packed_keys,
    seed_of,
    selector,
    shard_keys,
)
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"


def _merge_inputs(spec, seed):
    requests, positions, ranks = spec["requests"], spec["positions"], spec["ranks"]
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    full = logits(spec["values"], requests * positions, VOCAB, seed)
    gate = torch.randn(requests, positions, RANK, generator=g, device="cuda").bfloat16()
    anchor = torch.randint(0, VOCAB, (requests,), generator=g, device="cuda")
    return full, shard_keys(full, ranks), gate, anchor


@pytest.mark.parametrize("layout", ["allgather", "allreduce"])
@pytest.mark.parametrize(
    "cell", merge_mod.DraftTopKMerge.CELLS, ids=[c.why[:48] for c in merge_mod.DraftTopKMerge.CELLS]
)
def test_topk_merge_cells(cell, layout, certifying):  # noqa: F811
    certifying(draft_topk_merge)
    spec = cell.spec
    requests, positions, ranks = spec["requests"], spec["positions"], spec["ranks"]
    full, keys, gate, anchor = _merge_inputs(spec, seed_of(cell.why))
    if layout == "allreduce":
        keys = packed_keys(full, ranks)
    sel = selector()
    shard = VOCAB // ranks
    args = (
        keys,
        shard,
        requests,
        positions,
        anchor,
        gate,
        sel.predecessor_codebook,
        sel.successor_codebook,
        TOP_K,
        ranks,
    )
    ids, unary, preds, succs = draft_topk_merge(*args)

    # What the built-in path computes: torch.topk over the gathered fp32 logits.
    ref_unary, ref_ids = torch.topk(full.float(), TOP_K, dim=-1)
    assert torch.equal(ids.view(-1, TOP_K), ref_ids), (
        "candidate ids or their order differ from torch.topk"
    )
    assert_same_floats(unary.view(-1, TOP_K), ref_unary)

    # The entry's own reference.
    r_ids, r_unary, r_preds, r_succs = draft_topk_merge.reference(*args)
    assert torch.equal(ids, r_ids)
    assert_same_floats(unary, r_unary)
    assert torch.equal(preds, r_preds) and torch.equal(succs, r_succs)

    # The built-in edge scores, from the built-in candidates.
    builtin = dflash2_score_edges(
        sel.predecessor_codebook,
        sel.successor_codebook,
        ref_ids.view(requests, positions, TOP_K),
        ref_unary.view(requests, positions, TOP_K),
        gate,
        anchor,
    )
    ours = unary.float()[:, :, None] + torch.einsum("blpr,blcr->blpc", preds, succs)
    assert_same_floats(ours, builtin)


def test_topk_merge_nan_ranks_first_like_torch():
    spec = dict(requests=1, positions=7, ranks=8, values="special_nan")
    full, keys, gate, anchor = _merge_inputs(spec, 99)
    sel = selector()
    ids, unary, _, _ = draft_topk_merge(
        keys,
        VOCAB // 8,
        1,
        7,
        anchor,
        gate,
        sel.predecessor_codebook,
        sel.successor_codebook,
        TOP_K,
        8,
    )
    ref_unary, ref_ids = torch.topk(full.float(), TOP_K, dim=-1)
    assert torch.equal(ids.view(-1, TOP_K), ref_ids)
    assert_same_floats(unary.view(-1, TOP_K), ref_unary)


def test_topk_merge_many_tie_patterns_match_torch_topk():
    """The bitonic tie order, over many rows with heavy ties."""
    sel = selector()
    for trial in range(40):
        g = torch.Generator(device="cuda").manual_seed(1000 + trial)
        levels = (4, 8, 17, 40)[trial % 4]
        full = (
            torch.randint(0, levels, (7, VOCAB), generator=g, device="cuda").float() * 0.25 - 3.0
        ).bfloat16()
        keys = shard_keys(full, TP) if trial % 2 else packed_keys(full, TP)
        gate = torch.randn(1, 7, RANK, generator=g, device="cuda").bfloat16()
        anchor = torch.randint(0, VOCAB, (1,), generator=g, device="cuda")
        ids, unary, _, _ = draft_topk_merge(
            keys,
            VOCAB // TP,
            1,
            7,
            anchor,
            gate,
            sel.predecessor_codebook,
            sel.successor_codebook,
            TOP_K,
            TP,
        )
        ref_unary, ref_ids = torch.topk(full.float(), TOP_K, dim=-1)
        assert torch.equal(ids.view(-1, TOP_K), ref_ids), f"trial {trial}"
        assert torch.equal(unary.view(-1, TOP_K), ref_unary), f"trial {trial}"


def test_topk_merge_guards_refuse_what_the_kernel_cannot_take():
    sel = selector()
    keys = torch.zeros(7 * 8, 112, dtype=torch.int32, device="cuda")
    gate = torch.zeros(1, 7, RANK, dtype=torch.bfloat16, device="cuda")
    anchor = torch.zeros(1, dtype=torch.long, device="cuda")
    cb = (sel.predecessor_codebook, sel.successor_codebook)
    with validating(draft_topk_merge):
        with pytest.raises(ValueError):  # shard * ranks != vocab
            draft_topk_merge(keys, 1000, 1, 7, anchor, gate, *cb, 16, 8)
        with pytest.raises(ValueError):  # top_k above the 32-slot sort
            draft_topk_merge(keys, VOCAB // 8, 1, 7, anchor, gate, *cb, 64, 8)
        with pytest.raises(ValueError):  # rows do not match num_ranks
            draft_topk_merge(keys, VOCAB // 4, 1, 7, anchor, gate, *cb, 16, 4)
