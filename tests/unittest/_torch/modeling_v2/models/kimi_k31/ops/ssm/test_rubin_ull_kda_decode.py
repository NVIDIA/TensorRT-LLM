# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's kda_decode entry (q = 1): every cell's output rows, states and
conv windows against the FP64 recurrence; padding rows zero and unwritten."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = [".", "../../../.."]
import _kda_case as case  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_decode import (  # noqa: E402, E501
    kda_decode as op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="MR8's KDA kernels need an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    heads, rows = spec["heads"], spec["requests"]
    # The target's q1 scratch: rows + 1 dense slots, 0 the padding row.
    caches = case.checkpoint_caches(rows + 1, heads, 3, g)
    weights = case.layer_weights(heads, g)
    core = case.make_core(weights, heads, {rows: (case.pitch_of(heads), 0)}, caches)
    projected = case.packed_rows(rows, heads, g)
    out = torch.full((rows, heads * 128), 7.0, dtype=torch.bfloat16, device="cuda")
    slots = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")
    for r in spec["padded"]:
        slots[r] = 0
    args = (core, projected, out, caches, slots)
    ref = op.reference(*args)
    before = tuple(t.clone() for t in caches)
    with validating(op):
        result = op(*args)
    torch.cuda.synchronize()
    assert result is out
    case.assert_core_matches(out, caches, before, slots.view(rows, 1), None, 1, ref)


def test_is_valid_refuses_what_the_kernel_would_misplace() -> None:
    core = types.SimpleNamespace(dim=768, heads=6)
    rows = torch.zeros(2, case.pitch_of(6), dtype=torch.bfloat16)
    out = torch.zeros(2, 768, dtype=torch.bfloat16)
    caches = (torch.zeros(3, 3 * 768, 3, dtype=torch.bfloat16), torch.zeros(3, 6, 128, 128))
    op.is_valid(core, rows, out, caches, torch.tensor([1, 2], dtype=torch.int32))
    with pytest.raises(ValueError, match="share"):
        op.is_valid(core, rows, out, caches, torch.tensor([1, 1], dtype=torch.int32))
    with pytest.raises(ValueError, match="3-column"):
        wide = (torch.zeros(3, 3 * 768, 10, dtype=torch.bfloat16), caches[1])
        op.is_valid(core, rows, out, wide, torch.tensor([1, 2], dtype=torch.int32))
