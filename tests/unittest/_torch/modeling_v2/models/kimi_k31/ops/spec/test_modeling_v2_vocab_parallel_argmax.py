# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 vocab_parallel_argmax entry.

The TP group runs in one process: every rank's local kernels on its shard, the all-reduce as the
rank-order sum of the published pair buffers, then every rank's reduction. The ids are exact, so
every rank must return the reference's ids bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_argmax import (  # noqa: E501
    vocab_parallel_argmax as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_spec_inputs import on_sm103, run_tp_group, seed_of  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103


def _logits(spec: dict, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, vocab = spec["rows"], spec["vocab"]
    full = torch.randn(rows, vocab, generator=gen, device="cuda") * 3.0
    if spec["data"] == "ties":
        # The row maximum, bit-identical, at three global columns on different ranks.
        peak = full.amax(dim=-1, keepdim=True) + 1.0
        shard = vocab // spec["tp_size"]
        cols = torch.stack(
            [
                torch.randint(r * shard, (r + 1) * shard, (rows,), generator=gen, device="cuda")
                for r in (spec["tp_size"] - 1, spec["tp_size"] // 2, 0)
            ],
            dim=-1,
        )
        full.scatter_(1, cols, peak.expand(rows, 3))
    elif spec["data"] == "nan":
        cols = torch.randint(0, vocab, (rows, 2), generator=gen, device="cuda")
        full[::2].scatter_(1, cols[::2], float("nan"))
    return full.to(spec["dtype"])


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    out_dtype = spec.get("out_dtype", torch.int64)
    full = _logits(spec, seed_of(cell))
    shards = [s.contiguous() for s in full.chunk(spec["tp_size"], dim=-1)]
    results = run_tp_group(
        lambda local, all_reduce, tp, rank: op(local, all_reduce, tp, rank, out_dtype), shards
    )
    expected = op.reference(shards, None, spec["tp_size"], None, out_dtype)
    for rank, ids in enumerate(results):
        assert torch.equal(ids, expected), f"rank {rank} disagrees with the gathered argmax"
