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

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_sample import (  # noqa: E501
    gumbel_scores,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_sample import (  # noqa: E501
    vocab_parallel_sample as op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    GREEDY_TEMPERATURE_THRESHOLD as GREEDY_THRESHOLD,
)

__extra_import_path__ = ["../../../..", "."]
from _k31_spec_inputs import on_sm103, run_tp_group, seed_of  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

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


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
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
    assert ids.dtype == torch.int64 and ids.shape == (spec["rows"],)
    scores = gumbel_scores(full, temperatures, seeds, offsets, GREEDY_THRESHOLD)
    top2 = scores.topk(2, dim=-1).values
    near_tie = (top2[:, 0] - top2[:, 1]) <= _NEAR_TIE * top2[:, 0].abs().clamp_min(1.0)
    differ = ids != expected
    assert not (differ & ~near_tie).any(), (
        f"rows {differ.nonzero().flatten().tolist()} drew a token the reference does not, "
        "away from any near-tie"
    )
    # A different winner must still be a near-maximal score.
    picked = scores.gather(1, ids.view(-1, 1)).squeeze(1)
    assert (picked >= top2[:, 0] - _NEAR_TIE * top2[:, 0].abs().clamp_min(1.0)).all()
    greedy = temperatures <= GREEDY_THRESHOLD
    assert torch.equal(ids[greedy], full.float().argmax(-1)[greedy])
    assert differ.sum().item() <= 1, f"{differ.sum().item()} rows differ"


def test_draws_follow_the_temperature_softmax() -> None:
    """Many draws of one row land with the frequencies of softmax(logits / T).

    A few hot tokens carry almost all the mass, so 8192 draws pin each frequency to about 0.005.
    """
    tp, rows, vocab, temp = 8, 8192, 163840, 0.7
    gen = torch.Generator(device="cuda").manual_seed(11)
    row = torch.full((vocab,), -30.0, device="cuda")
    hot = torch.randperm(vocab, generator=gen, device="cuda")[:6]
    row[hot] = torch.tensor([0.0, -0.3, -0.6, -1.0, -1.5, -2.5], device="cuda")
    full = row.expand(rows, vocab).to(torch.bfloat16).contiguous()
    temperatures = torch.full((rows,), temp, dtype=torch.float32, device="cuda")
    seeds = torch.full((rows,), 1234, dtype=torch.int64, device="cuda")
    offsets = torch.arange(rows, dtype=torch.int64, device="cuda")
    shards = [s.contiguous() for s in full.chunk(tp, dim=-1)]
    ids = run_tp_group(
        lambda local, all_reduce, t, r: op(
            local, temperatures, seeds, offsets, GREEDY_THRESHOLD, all_reduce, t, r
        ),
        shards,
    )[0]
    freq = torch.bincount(ids, minlength=vocab).float() / rows
    probs = torch.softmax(full[0].float() / temp, dim=-1)
    tv = 0.5 * (freq - probs).abs().sum().item()
    assert tv < 0.03, f"total variation {tv:.4f} between the draws and softmax(logits / T)"


def test_guard_refuses_a_shard_off_the_philox_groups() -> None:
    local = torch.randn(4, 20482, device="cuda").to(torch.bfloat16)
    ones = torch.ones(4, device="cuda")
    ids = torch.zeros(4, dtype=torch.int64, device="cuda")
    with pytest.raises(AssertionError, match="multiple of 4"):
        with validating(op):
            op(local, ones, ids, ids, GREEDY_THRESHOLD, lambda b: b, 8, 1)


def test_guard_refuses_short_per_row_tensors() -> None:
    local = torch.randn(8, 20480, device="cuda").to(torch.bfloat16)
    short = torch.ones(4, device="cuda")
    ids = torch.zeros(8, dtype=torch.int64, device="cuda")
    with pytest.raises(AssertionError, match="one entry per row"):
        with validating(op):
            op(local, short, ids, ids, GREEDY_THRESHOLD, lambda b: b, 8, 1)


def test_cuda_graph_replay() -> None:
    """Captured once, replayed with new Philox offsets: each replay draws the reference's ids."""
    rows, shard = 40, 20480
    local = (torch.randn(rows, shard, device="cuda") * 2.0).to(torch.bfloat16)
    temperatures = torch.full((rows,), 0.8, device="cuda")
    seeds = torch.full((rows,), 77, dtype=torch.int64, device="cuda")
    offsets = torch.arange(rows, dtype=torch.int64, device="cuda")
    op(local, temperatures, seeds, offsets, GREEDY_THRESHOLD, None, 1, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids = op(local, temperatures, seeds, offsets, GREEDY_THRESHOLD, None, 1, 0)
    for step in range(1, 4):
        offsets.add_(rows)
        graph.replay()
        expected = op.reference([local], temperatures, seeds, offsets, GREEDY_THRESHOLD)
        assert (ids != expected).sum().item() <= 1, f"replay {step}"
