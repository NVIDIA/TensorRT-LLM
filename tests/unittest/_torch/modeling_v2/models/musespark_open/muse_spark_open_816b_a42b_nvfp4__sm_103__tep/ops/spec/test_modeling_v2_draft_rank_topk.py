# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``draft_rank_topk`` (fast path ``wide_draft_topk``).

Gated on its cells with ``torch.equal`` against its torch reference, with and
without a drafter logit transform's canonical keys, and the merge over its
output against the merge over ``draft_shard_topk``'s per-chunk keys.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    draft_rank_topk as rank_mod,
)

__extra_import_path__ = ["../../../../..", "../../kernels/draft_head"]
from _draft_head_cases import (  # noqa: E402
    MUSESPARK,
    TOP_K,
    VOCAB,
    draft_head,
    logits,
    packed_keys,
    rank_keys,
    seed_of,
    selector,
    transform,
)
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "the vocab-sharded draft head requires a CUDA device"

draft_rank_topk = rank_mod.draft_rank_topk
CELLS = rank_mod.DraftRankTopK.CELLS


@pytest.mark.parametrize("kind", ["gemm3", "special_nan", "straddle_up", "saturated"])
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_rank_topk_cells(cell, kind, certifying):  # noqa: F811
    certifying(draft_rank_topk)
    spec = cell.spec
    x = logits(kind, spec["rows"], spec["shard"], seed_of(cell.why + kind))
    kwargs = {} if spec["transform"] is None else draft_head.canon_kwargs(transform(MUSESPARK))
    ranks = VOCAB // spec["shard"]
    out = draft_rank_topk(x, TOP_K, 4096, 4, 1, ranks, **kwargs)
    ref = draft_rank_topk.reference(x, TOP_K, 4096, 4, 1, ranks, **kwargs)
    assert torch.equal(out, ref), f"{(out != ref).sum().item()} key halves differ"


@pytest.mark.parametrize("kind", ["gemm400", "straddle_down", "ladder"])
def test_merge_over_rank_keys_is_the_merge_over_chunk_keys(kind):
    t = transform(MUSESPARK)
    rows, ranks = 35, 8
    full = logits(kind, rows, VOCAB, seed_of(kind))
    sel = selector()
    gate = torch.zeros(
        1, rows, sel.predecessor_codebook.shape[1], dtype=torch.bfloat16, device="cuda"
    )
    anchor = torch.zeros(1, dtype=torch.int64, device="cuda")
    outs = []
    for keys in (packed_keys(full, ranks, t), rank_keys(full, ranks, t)):
        outs.append(
            draft_head.draft_topk_merge(
                keys,
                VOCAB // ranks,
                1,
                rows,
                anchor,
                gate,
                sel.predecessor_codebook,
                sel.successor_codebook,
                TOP_K,
                ranks,
                values=t.values,
            )
        )
    for a, b in zip(*outs):
        assert torch.equal(
            a.view(torch.int32) if a.is_floating_point() else a,
            b.view(torch.int32) if b.is_floating_point() else b,
        )
