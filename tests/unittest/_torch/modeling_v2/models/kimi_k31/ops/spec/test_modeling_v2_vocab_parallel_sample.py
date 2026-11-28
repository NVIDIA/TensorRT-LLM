# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 vocab_parallel_sample entry.

The TP group runs in one process (see the argmax entry's test). The reference reproduces the
Philox stream exactly; only the logarithms differ (the kernel's is the fast approximate one), so
a row may pick a different token only where the reference's two best scores are within that
rounding of each other.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_sample import (  # noqa: E501
    gumbel_scores,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_sample import (  # noqa: E501
    vocab_parallel_sample as op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    GREEDY_TEMPERATURE_THRESHOLD as GREEDY_THRESHOLD,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_spec_inputs import on_sm103, run_tp_group, seed_of  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103

#: A score gap below which the approximate log may legitimately swap the winner: a few fp32
#: ulps of scores that reach ~20 (logits / T up to ~5, Gumbel noise up to ~16.6).
_NEAR_TIE = 2e-5


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, vocab = spec["rows"], spec["vocab"]
    full = (torch.randn(rows, vocab, generator=gen, device="cuda") * 2.0).to(spec["dtype"])
    temperatures = torch.full((rows,), spec["temperature"], dtype=torch.float32, device="cuda")
    if spec["greedy_rows"]:
        greedy = torch.randperm(rows, generator=gen, device="cuda")[: spec["greedy_rows"]]
        temperatures[greedy] = GREEDY_THRESHOLD / 10
    seeds = torch.randint(0, 2**62, (rows,), generator=gen, device="cuda", dtype=torch.int64)
    offsets = spec.get("offset_base", 0) + 7 * torch.arange(rows, device="cuda", dtype=torch.int64)
    return full, temperatures, seeds, offsets


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    full, temperatures, seeds, offsets = _inputs(spec, seed_of(cell))
    shards = [s.contiguous() for s in full.chunk(spec["tp_size"], dim=-1)]
    results = run_tp_group(
        lambda local, all_reduce, tp, rank: op(
            local, temperatures, seeds, offsets, GREEDY_THRESHOLD, all_reduce, tp, rank
        ),
        shards,
    )
    expected = op.reference(shards, temperatures, seeds, offsets, GREEDY_THRESHOLD)
    for rank, ids in enumerate(results[1:], start=1):
        assert torch.equal(ids, results[0]), f"rank {rank} drew a different token than rank 0"
    ids = results[0]
    scores = gumbel_scores(full, temperatures, seeds, offsets, GREEDY_THRESHOLD)
    top2 = scores.topk(2, dim=-1).values
    near_tie = (top2[:, 0] - top2[:, 1]) <= _NEAR_TIE * top2[:, 0].abs().clamp_min(1.0)
    differ = ids != expected
    assert not (differ & ~near_tie).any(), (
        f"rows {differ.nonzero().flatten().tolist()} drew a token the reference does not, "
        "away from any near-tie"
    )
