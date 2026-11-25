# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's `csa2_consumer_schedule` (the candidate consumers' sparse-logits schedule, once per step) and the
`csa2_indexer_topk_decode` options the dep target pass: `publish_stream` / `publish_done` (the candidate
source publishes on a side stream) and `cand_sched`. The picks and candidates must be bit for bit those of the default
call; the schedule buffer itself is DeepGEMM's (not every word
is written), so its own cells gate the call's geometry."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # _v41_util, the family indexer test
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # _validating
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import certifying  # noqa: E402,F401
from test_modeling_v2_csa2_indexer_decode import TOPK, _decode_cell  # noqa: E402

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_consumer_schedule import (  # noqa: E402,E501
    csa2_consumer_schedule,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (  # noqa: E402
    csa2_indexer_topk_decode,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_descriptors import (  # noqa: E402,E501
    csa2_indexer_descriptors,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


@pytest.mark.parametrize(
    "cell", csa2_consumer_schedule.CELLS, ids=[c.why[:48] for c in csa2_consumer_schedule.CELLS]
)
def test_consumer_schedule_cells(cell, certifying):  # noqa: F811
    certifying(csa2_consumer_schedule)
    s, gen = cell.spec, gen_for(cell)
    n, nb, pages = s["n"], s["blocks"], s["pages"]
    blocks = (
        torch.randperm(4 * pages * n, generator=gen, device=DEV)[: n * pages]
        .view(n, pages)
        .to(torch.int32)
        .contiguous()
    )
    counts = torch.randint(1, nb + 1, (n,), generator=gen, device=DEV, dtype=torch.int32)
    ids = torch.randint(0, pages * 8, (n, nb), generator=gen, device=DEV, dtype=torch.int32)
    got = csa2_consumer_schedule(counts, blocks, ids)
    ref = csa2_consumer_schedule.reference(counts, blocks, ids)
    assert got.shape == ref.shape and got.dtype == ref.dtype and got.device == ref.device


@pytest.mark.parametrize(
    "cell", [c for c in csa2_indexer_topk_decode.CELLS if c.spec["cand"]], ids=lambda c: c.why[:44]
)
def test_side_publication_and_shared_schedule(cell):
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    plain, pcand = csa2_indexer_topk_decode(*args, None, (8, 40), desc)
    side, done = torch.cuda.Stream(), torch.cuda.Event()
    got, cand = csa2_indexer_topk_decode(
        *args, None, (8, 40), desc, publish_stream=side, publish_done=done
    )
    torch.cuda.current_stream().wait_event(done)
    assert torch.equal(got, plain) and all(torch.equal(a, b) for a, b in zip(cand, pcand))
    q2 = q.flip(0).contiguous()
    cargs = (q2, w, pos, rid, pool, table, ratio, TOPK, mp)
    ref, _ = csa2_indexer_topk_decode(*cargs, cand, None, desc)
    sched = csa2_consumer_schedule(cand[2].reshape(-1), desc[0], cand[1])
    for _ in range(2):  # two consumer layers sharing the step's schedule
        picks, _ = csa2_indexer_topk_decode(*cargs, cand, None, desc, cand_sched=sched)
        assert torch.equal(picks, ref)
