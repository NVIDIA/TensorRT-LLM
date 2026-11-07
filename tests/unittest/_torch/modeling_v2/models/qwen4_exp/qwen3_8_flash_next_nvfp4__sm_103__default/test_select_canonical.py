# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the default target's block selection (`_select`).

`_select` picks each row's canonical top-`block_topk` groups of its window: the exact top-k
and then the canonical rewrite, two launches, which serve a draft step's rows (the rejection
kernel's ragged windows) at every table width the engine's graph buckets use.

Both checks run over windows short of the budget, at it and past it, with a coarse score
grid so the k-th value ties: the rows equal the canonical definition
(`qsa_canonical_selection.reference`: LARGE tie-break, ascending, `-1` past the window), and
bounding the rewrite to the live columns returns the same rows.
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_canonical_selection import (  # noqa: E501
    qsa_canonical_selection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as U,
)

assert torch.cuda.is_available(), "the block selection runs on a CUDA device"

K = 512


def _case(rows: int, cols: int, seed: int) -> tuple:
    """Ragged windows over a coarse score grid: every row's window ends somewhere in
    `[1, cols]`, so rows sit below, at and past the budget, and the k-th value ties."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randint(0, 5, (rows, cols), generator=g, device="cuda").float()
    row_starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.randint(1, cols + 1, (rows,), generator=g, device="cuda").to(torch.int32)
    ends[0] = cols
    ends[1] = K
    ends[2] = 1
    indices = torch.full((rows, K), -1, dtype=torch.int32, device="cuda")
    return scores, row_starts, ends, indices


@pytest.mark.parametrize("rows,cols", [(24, 2304), (24, 4096), (8, 1024)])
def test_rows_equal_the_canonical_definition(rows: int, cols: int) -> None:
    scores, row_starts, row_ends, indices = _case(rows, cols, seed=rows * cols)
    U._select(types.SimpleNamespace(block_topk=K), scores, row_starts, row_ends, indices)

    expected = qsa_canonical_selection.reference(scores, row_starts, row_ends, indices.clone())
    assert torch.equal(indices, expected)


def test_live_cols_returns_the_same_rows() -> None:
    scores, row_starts, row_ends, indices = _case(16, 2304, seed=7)
    # Every window ends short of the table, so the bound drops columns the rewrite would read.
    row_ends.clamp_(max=1500)
    assert int(row_ends.max()) < scores.shape[1]
    bounded = indices.clone()
    U._select(types.SimpleNamespace(block_topk=K), scores, row_starts, row_ends, indices)
    U._select(
        types.SimpleNamespace(block_topk=K),
        scores,
        row_starts,
        row_ends,
        bounded,
        int(row_ends.max()),
    )
    assert torch.equal(indices, bounded)
