# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's kda_verify entry: every cell's output rows, per-position state
checkpoints and rewritten conv pages against the FP64 recurrence; no other row written."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = [".", "../../../.."]
import _kda_case as case  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_verify import (  # noqa: E402, E501
    kda_verify as op,
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
    heads, tokens, accepted = spec["heads"], spec["tokens"], spec["accepted"]
    requests = len(accepted)
    rows = requests * tokens
    slot_count = requests * tokens + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    weights = case.layer_weights(heads, g)
    pitch = case.pitch_of(heads)
    core = case.make_core(weights, heads, {(requests, tokens): (pitch, 0)}, caches)
    projected = case.packed_rows(rows, heads, g)
    out = torch.full((rows, heads * 128), 7.0, dtype=torch.bfloat16, device="cuda")
    slots = case.verify_slots(requests, tokens, spec["padded"], slot_count, g)
    counts = torch.tensor(accepted, dtype=torch.int32, device="cuda")
    args = (core, projected, out, caches, slots, counts)
    ref = op.reference(*args)
    before = tuple(t.clone() for t in caches)
    with validating(op):
        result = op(*args)
    torch.cuda.synchronize()
    assert result is out
    case.assert_core_matches(out, caches, before, slots, counts, tokens, ref)


def test_is_valid_refuses_what_the_kernel_would_misplace() -> None:
    core = types.SimpleNamespace(dim=768, heads=6)
    rows = torch.zeros(8, case.pitch_of(6), dtype=torch.bfloat16)
    out = torch.zeros(8, 768, dtype=torch.bfloat16)
    caches = (torch.zeros(4, 3 * 768, 10, dtype=torch.bfloat16), torch.zeros(4, 6, 128, 128))
    slots = torch.arange(1, 9, dtype=torch.int32).view(1, 8)
    op.is_valid(core, rows, out, caches, slots, torch.tensor([3], dtype=torch.int32))
    with pytest.raises(ValueError, match="outside 1..q"):
        op.is_valid(core, rows, out, caches, slots, torch.tensor([0], dtype=torch.int32))
    with pytest.raises(ValueError, match="q \\+ 2"):
        narrow = (caches[0][..., :9], caches[1])
        op.is_valid(core, rows, out, narrow, slots, torch.tensor([1], dtype=torch.int32))
    shared = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1]], dtype=torch.int32)
    with pytest.raises(ValueError, match="share"):
        two = torch.cat([slots, shared])
        op.is_valid(
            core,
            torch.cat([rows, rows]),
            torch.cat([out, out]),
            caches,
            two,
            torch.tensor([1, 1], dtype=torch.int32),
        )
