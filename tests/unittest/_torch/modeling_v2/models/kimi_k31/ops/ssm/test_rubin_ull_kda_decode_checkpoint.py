# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's kda_decode_checkpoint entry: every cell's gather or scatter bit for
bit against the indexed-copy reference, caches and scratch both."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = [".", "../../../.."]
import _kda_case as case  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_decode_checkpoint import (  # noqa: E402, E501
    kda_decode_checkpoint as op,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    heads, tokens, accepted = spec["heads"], spec["tokens"], spec["accepted"]
    requests = len(accepted)
    slot_count = requests * tokens + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    scratch = case.checkpoint_caches(requests + 1, heads, 3, g)
    slots = case.verify_slots(requests, tokens, spec["padded"], slot_count, g)
    counts = torch.tensor(accepted, dtype=torch.int32, device="cuda")
    args = (caches, scratch, slots, counts, spec["scatter"])
    (conv, state), (conv_tmp, state_tmp) = op.reference(*args)
    with validating(op):
        op(*args)
    torch.cuda.synchronize()
    assert torch.equal(caches[1], state), "checkpoint states differ"
    assert torch.equal(caches[0], conv), "checkpoint conv pages differ"
    assert torch.equal(scratch[1], state_tmp), "scratch states differ"
    assert torch.equal(scratch[0], conv_tmp), "scratch conv windows differ"


def test_is_valid_refuses_what_the_copy_would_misplace() -> None:
    caches = (torch.zeros(4, 12, 10, dtype=torch.bfloat16), torch.zeros(4, 1, 128, 128))
    scratch = (torch.zeros(3, 12, 3, dtype=torch.bfloat16), torch.zeros(3, 1, 128, 128))
    slots = torch.tensor([[1] * 8, [2] * 8], dtype=torch.int32)
    counts = torch.tensor([1, 8], dtype=torch.int32)
    op.is_valid(caches, scratch, slots, counts, False)
    with pytest.raises(ValueError, match="share a page"):
        op.is_valid(caches, scratch, torch.ones(2, 8, dtype=torch.int32), counts, True)
    with pytest.raises(ValueError, match="outside 1..q"):
        op.is_valid(caches, scratch, slots, torch.tensor([0, 1], dtype=torch.int32), False)
