# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_router_tc entry of the MiniMax-M3 tp4 target."""

import pytest
import torch

__extra_import_path__ = ["../../../..", "."]
from _m3_router import (  # noqa: E402, F401
    SF_SCALE,
    _build,
    _compare_fp4,
    _compare_logits,
    _seed,
    gemv,
    gemv_fp4,
    tc,
)
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "the router entries require a CUDA device"


@pytest.mark.parametrize("cell", tc.CELLS, ids=[c.why[:48] for c in tc.CELLS])
def test_tc_certified_cells(cell) -> None:
    spec = cell.spec
    x, w = _build(spec["m"], 128, spec["k"], _seed(cell))
    sf_scale = torch.tensor([SF_SCALE], dtype=torch.float32, device="cuda") if spec["fp4"] else None
    with validating(tc):
        outs = tc(x, w, sf_scale, False)
    ref = tc.reference(x, w, sf_scale, False)
    assert len(outs) == len(ref) == (3 if spec["fp4"] else 1)
    _compare_logits(outs[0], ref[0])
    if spec["fp4"]:
        _compare_fp4(outs[1], outs[2], x, sf_scale)


def test_tc_reference_is_tf32_and_the_gate_sees_it() -> None:
    """The tensor-core router rounds the weight to TF32: against a true fp32 product its logits sit far outside
    the band, so the TF32 rounding in ``reference`` is load-bearing rather than decoration."""
    x, w = _build(16, 128, 6144, 5)
    out = tc(x, w, None, False)[0]
    with pytest.raises(AssertionError):
        _compare_logits(out, gemv.reference(x, w, 1, False))


def test_guard_refuses_a_misaligned_activation() -> None:
    """A row view starting 8 bytes into the buffer keeps a legal row stride but not a legal bulk-copy source."""
    x, w = _build(2, 128, 6144, 3)
    flat = torch.empty(2 * 6144 + 4, dtype=torch.bfloat16, device="cuda")
    shifted = flat[4:].view(2, 6144)
    shifted.copy_(x)
    with pytest.raises(AssertionError, match="16-byte aligned"):
        with validating(tc):
            tc(shifted, w, None, False)
