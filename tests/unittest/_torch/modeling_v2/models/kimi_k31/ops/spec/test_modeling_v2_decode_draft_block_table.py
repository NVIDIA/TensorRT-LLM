# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 decode_draft_block_table entry (integer ops: exact)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.decode_draft_block_table import (
    decode_draft_block_table,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_spec_inputs import on_sm103, seed_of  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103


def _table_inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, blocks, divisor = spec["rows"], spec["blocks"], spec["divisor"]
    used = torch.randint(0, blocks + 1, (rows,), generator=gen, device="cuda")
    encoded = torch.randint(0, 1 << 20, (rows, blocks), generator=gen, device="cuda") * divisor
    cols = torch.arange(blocks, device="cuda")
    # Encoded offsets as the TRTLLM metadata lays them out: a [pools, rows, 2, blocks] int32 tensor,
    # read through the plane-0 slice of one pool (strided rows), -1 past each row's used blocks.
    offsets = torch.full((2, spec["table_rows"], 2, blocks), -7, dtype=torch.int32, device="cuda")
    offsets[1, :rows, 0] = torch.where(cols[None, :] < used[:, None], encoded, -1).to(torch.int32)
    src = offsets[1, :rows, 0]
    table = torch.full((spec["table_rows"], blocks), 12345, dtype=torch.int32, device="cuda")
    counts = torch.full((spec["table_rows"],), 777, dtype=torch.long, device="cuda")
    return src, table, counts


CELLS = sample_cells(decode_draft_block_table.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells_decode_draft_block_table(cell, certifying) -> None:  # noqa: F811
    certifying(decode_draft_block_table)
    spec = cell.spec
    src, table, counts = _table_inputs(spec, seed_of(cell))
    ref_table, ref_counts = table.clone(), counts.clone()
    src_before = src.clone()
    decode_draft_block_table(src, spec["divisor"], table, counts)
    decode_draft_block_table.reference(src, spec["divisor"], ref_table, ref_counts)
    assert torch.equal(table, ref_table)
    assert torch.equal(counts, ref_counts), "rows past `rows` must stay untouched as well"
    assert torch.equal(src, src_before)
