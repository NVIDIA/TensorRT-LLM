# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's Helix entries: helix_tail and helix_tail_gated.

The one-rank cells (P = 1, a local receive buffer) run here; the 4-rank cells of both and of
helix_alltoall_native run in ``_helix_matrix.py`` on one tray (``_k31_ull_comm_ranks``). Every cell
runs several launches in a row (the launches alternate the receive halves and re-arm what they
read), against the FP64 reference within a few BF16 ulp, and the gated rows bitwise against the
two PyTorch BF16 kernels applied to the ungated rows.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_ull_helix_tail import (  # noqa: E501
    helix_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_ull_helix_tail_gated import (  # noqa: E501
    helix_tail_gated,
)

__extra_import_path__ = [".", "../../../.."]
import _helix_inputs as inputs  # noqa: E402 — needs the path declared above
import _k31_ull_comm_ranks as ranks  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="certified on SM10x",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(helix_tail, TARGET_CELLS)
GATED_CELLS = certified_cells(helix_tail_gated, TARGET_CELLS)
ONE_RANK = [c for c in CELLS if c.spec["cp_size"] == 1]
ONE_RANK_GATED = [c for c in GATED_CELLS if c.spec["cp_size"] == 1]
LAUNCHES = 3


@pytest.mark.parametrize("cell", ONE_RANK, ids=[c.why[:44] for c in ONE_RANK])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    ws = inputs.LocalWorkspace()
    w_uv = inputs.w_uv(seed=11)
    for launch in range(LAUNCHES):
        seed = zlib.crc32(cell.why.encode()) % 2**31 + launch
        rows, stats, mask = inputs.rank_inputs(spec, seed, rank=0)
        out = torch.empty(
            spec["tokens"], spec["heads"] * inputs.VALUE, dtype=torch.bfloat16, device="cuda"
        )
        with validating(helix_tail):
            helix_tail(rows, stats, w_uv, out, mask, 0, 1, *ws.args())
        torch.cuda.synchronize()
        ref = helix_tail.reference([rows], [stats], w_uv, out, [mask], 0, 1)
        inputs.assert_within_bf16_ulp(out, ref, 3, f"{cell.why}, launch {launch}")


@pytest.mark.parametrize("cell", ONE_RANK_GATED, ids=[c.why[:44] for c in ONE_RANK_GATED])
def test_certified_cells_gated(cell) -> None:
    spec = cell.spec
    ws = inputs.LocalWorkspace()
    w_uv = inputs.w_uv(seed=13)
    for launch in range(LAUNCHES):
        seed = zlib.crc32(cell.why.encode()) % 2**31 + launch
        rows, stats, mask = inputs.rank_inputs(spec, seed, rank=0)
        gate = inputs.gate(spec, seed)
        width = spec["heads"] * inputs.VALUE
        plain = torch.empty(spec["tokens"], width, dtype=torch.bfloat16, device="cuda")
        out = torch.empty_like(plain)
        helix_tail(rows, stats, w_uv, plain, mask, 0, 1, *ws.args())
        with validating(helix_tail_gated):
            helix_tail_gated(rows, stats, w_uv, gate, out, mask, 0, 1, *ws.args())
        torch.cuda.synchronize()
        ranks.assert_bitwise(out, plain * gate.sigmoid(), f"{cell.why}: gated vs torch gate")
        ref = helix_tail_gated.reference([rows], [stats], w_uv, gate, out, [mask], 0, 1)
        inputs.assert_within_bf16_ulp(out, ref, 4, f"{cell.why}, launch {launch}")


def test_the_guard_refuses_a_receive_half_too_small() -> None:
    spec = dict(tokens=64, cp_size=1, heads=inputs.HEADS, masked=False)
    ws = inputs.LocalWorkspace()
    rows, stats, mask = inputs.rank_inputs(spec, 3, rank=0)
    out = torch.empty(64, inputs.HEADS * inputs.VALUE, dtype=torch.bfloat16, device="cuda")
    args = list(ws.args())
    args[2] = 4096  # half_bytes
    with pytest.raises(ValueError, match="receive half"):
        helix_tail.is_valid(rows, stats, inputs.w_uv(seed=1), out, mask, 0, 1, *args)


@pytest.mark.no_xdist
@pytest.mark.skipif(
    len(ranks.visible_devices()) < ranks.WORLD_SIZE,
    reason=f"the 4-rank Helix cells need {ranks.WORLD_SIZE} GPUs on one NVLink domain",
)
def test_four_rank_cells() -> None:
    """helix_tail's, helix_tail_gated's and helix_alltoall_native's 4-rank cells."""
    ranks.run("helix")
