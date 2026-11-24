# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dense0_down_push entry on one GPU: the two-shot cells push into 16
local "peer" buffers (the owners of a TP16 all-reduce), the one-shot cells through a one-device
multicast buffer; every pushed row against the FP64 product, every other slot untouched."""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
    rubin_ull_k3_dense0 as k3_dense0,  # noqa: E501
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_dense0_down_push import (  # noqa: E501
    INTER,
    K_PADDED,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_dense0_down_push import (  # noqa: E501
    dense0_down_push as op,
)

__extra_import_path__ = ["../../../..", "."]
from _bf16_band import assert_bf16_gemm_close  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _multicast import MulticastBuffer  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the split-K CuTe DSL kernel needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

HIDDEN = k3_dense0.HIDDEN
ROW_BYTES = HIDDEN * 2
POISON = 0x7FBF  # a BF16 NaN: no product equals it
#: The rank a test rank pushes as (any slot but 0, so a rank-blind store shows).
RANK = 5


def _operands(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["rows"]
    x = torch.zeros(rows, K_PADDED, dtype=torch.bfloat16, device="cuda")
    x[:, :INTER] = torch.randn(rows, INTER, generator=g, device="cuda").to(torch.bfloat16)
    weight = torch.zeros(HIDDEN, K_PADDED, dtype=torch.bfloat16, device="cuda")
    weight[:, :INTER] = (torch.randn(HIDDEN, INTER, generator=g, device="cuda") * INTER**-0.5).to(
        torch.bfloat16
    )
    return x, weight, k3_dense0.SplitKTactic(*spec["tactic"])


def _flags(nbytes: int, current: int) -> torch.Tensor:
    """The workspace's rotation state as ``FlatAllReduce`` lays it out: [current, 2, bytes, ...]."""
    return torch.tensor([current, 2, nbytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device="cuda")


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    x, weight, tactic = _operands(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    rows, world = spec["rows"], spec["world"]
    rank = RANK % world if world > 1 else RANK * 64  # the mailbox's rank x mailbox-rows offset
    output = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    current = 1  # a non-zero current buffer, so a buffer-blind store shows
    if spec["one_shot"]:
        slots = (rows - 1) * world + rank + 1
        nbytes = slots * ROW_BYTES
        mc = MulticastBuffer(3 * nbytes)
        try:
            region = mc.uc[: 3 * nbytes].view(torch.int16)
            region.fill_(POISON)
            ws = SimpleNamespace(
                world=world,
                rank=rank,
                mc_base=mc.mc,
                peer_ptrs=torch.zeros(world, dtype=torch.int64, device="cuda"),
                flags=_flags(nbytes, current),
            )
            target = SimpleNamespace(workspace=ws, one_shot=True, output=output, pushed=False)
            with validating(op):
                op(x, weight, target, tactic, True)
            torch.cuda.synchronize()
            slab = region.view(torch.bfloat16).view(3, slots, HIDDEN).clone()
        finally:
            mc.close()
        pushed = slab[current, [t * world + rank for t in range(rows)]]
        touched = torch.zeros(3, slots, dtype=torch.bool)
        touched[current, [t * world + rank for t in range(rows)]] = True
        untouched = slab.view(torch.int16)[~touched.cuda()]
    else:
        per_owner = -(-rows // world) * world
        nbytes = 2 * per_owner * ROW_BYTES  # the two halves of a two-shot buffer
        owners = torch.full((world, 3 * nbytes // 2), POISON, dtype=torch.int16, device="cuda")
        ws = SimpleNamespace(
            world=world,
            rank=rank,
            mc_base=0,
            peer_ptrs=torch.tensor(
                [owners[o].data_ptr() for o in range(world)], dtype=torch.int64, device="cuda"
            ),
            flags=_flags(nbytes, current),
        )
        target = SimpleNamespace(workspace=ws, one_shot=False, output=output, pushed=False)
        with validating(op):
            op(x, weight, target, tactic, True)
        torch.cuda.synchronize()
        # Two-shot: row t lands in owner t % world, slot (t // world) * world + rank of the first
        # half of buffer ``current``.
        slab = owners.view(torch.bfloat16).view(world, 3, 2, per_owner, HIDDEN)
        where = [(t % world, (t // world) * world + rank) for t in range(rows)]
        pushed = torch.stack([slab[o, current, 0, s] for o, s in where])
        touched = torch.zeros(world, 3, 2, per_owner, dtype=torch.bool)
        for o, s in where:
            touched[o, current, 0, s] = True
        untouched = slab.view(torch.int16)[~touched.cuda()]
    assert target.pushed
    assert bool((untouched == POISON).all()), "stored outside this rank's slots"
    ref = op.reference(x, weight, target, tactic, True)
    assert_bf16_gemm_close(pushed, ref, x, weight)
    assert not bool((pushed.view(torch.int16) == -32768).any()), "-0 is pushed as +0"
