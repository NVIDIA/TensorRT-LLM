# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 pinned-tactic W4A8 routed-expert entry: a sample of its cells bitwise against the
catalog runner entry with its autotuned tactic."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_runner import (  # noqa: E501
    HIDDEN,
    k31_w4a8_moe_runner,
    pinned_tactic,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above
from _w4a8_moe_inputs import (  # noqa: E402 — needs the path declared above
    runner_args,
    w4a8_operands,
)

op = k31_w4a8_moe_runner

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "the TRTLLM-Gen MXFP4 x MXFP8 runner is certified on SM 10.3, where the MXFP4 targets call it",
        allow_module_level=True,
    )

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _build(spec, seed):
    """The runner's arguments for one rank of ``spec`` (``w4a8_operands``), at the pin table's
    tactic for its rows with ``tactic="table"``."""
    rows, local, inter = spec["rows"], spec["local"], spec["inter"]
    tactic = spec["tactic"]
    if tactic == "table":
        tactic = pinned_tactic(local, inter, rows)
        assert tactic is not None, f"no pin for {spec}"
    return runner_args(w4a8_operands(rows, local, inter, seed), tactic)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    ref = op.reference(*args)
    assert out.shape == (cell.spec["rows"], HIDDEN) and out.dtype is torch.bfloat16
    assert torch.isfinite(ref.float()).all()
    # A tactic changes the token tiling and the CTA schedule, not the per-expert K order: exact.
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), "pinned tactic is not bitwise"
