# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_router_gemv_fp4 entry of the MiniMax-M3 tp4 target."""

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


@pytest.mark.parametrize("cell", gemv_fp4.CELLS, ids=[c.why[:48] for c in gemv_fp4.CELLS])
def test_gemv_fp4_certified_cells(cell) -> None:
    spec = cell.spec
    x, w = _build(spec["m"], spec["e"], spec["k"], _seed(cell))
    sf_scale = torch.tensor([spec["sf_scale"]], dtype=torch.float32, device="cuda")
    with validating(gemv_fp4):
        logits, q, sf = gemv_fp4(x, w, sf_scale, False)
    ref_logits, _, _ = gemv_fp4.reference(x, w, sf_scale, False)
    _compare_logits(logits, ref_logits)
    _compare_fp4(q, sf, x, sf_scale)


def test_guard_refuses_a_zero_global_scale() -> None:
    x, w = _build(1, 128, 6144, 4)
    with pytest.raises(AssertionError, match="positive and finite"):
        with validating(gemv_fp4):
            gemv_fp4(x, w, torch.zeros(1, dtype=torch.float32, device="cuda"), False)
