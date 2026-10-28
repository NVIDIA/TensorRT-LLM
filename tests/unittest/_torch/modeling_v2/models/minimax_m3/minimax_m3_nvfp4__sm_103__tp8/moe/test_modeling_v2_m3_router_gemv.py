# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_router_gemv entry of the MiniMax-M3 tp8 target."""

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


@pytest.mark.parametrize("cell", gemv.CELLS, ids=[c.why[:48] for c in gemv.CELLS])
def test_gemv_certified_cells(cell) -> None:
    spec = cell.spec
    x, w = _build(spec["m"], spec["e"], spec["k"], _seed(cell))
    with validating(gemv):
        out = gemv(x, w, 1, False)
    _compare_logits(out, gemv.reference(x, w, 1, False))


def test_the_gemv_is_a_true_fp32_product() -> None:
    """And the GEMV, by contrast, is no TF32 product: the TF32 reference rejects it."""
    x, w = _build(1, 128, 6144, 6)
    out = gemv(x, w, 1, False)
    with pytest.raises(AssertionError):
        _compare_logits(out, tc.reference(x, w, None, False)[0])
