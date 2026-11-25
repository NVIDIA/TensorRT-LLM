# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the spec/m31_capture_norm entry of the MiniMax-M3.1-Flash sm_107 targets (Eagle3 capture with
fc_norm folded in)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.m31_capture_norm import (  # noqa: E501
    m3_capture_norm as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _m3_test_inputs import assert_bf16_close, gen, randn, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402, F401

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

HIDDEN, EPS, LAYERS = 6144, 1e-6, 3
# Every target certifies the same cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Within one bf16 ulp per element on at most 0.1% of the elements; the other layer slots of the buffer
    untouched."""
    g = gen(seed_of(cell.why))
    m, slot = cell.spec["m"], cell.spec["slot"]
    h = randn((m, HIDDEN), g, 2.0)
    r = randn((m, HIDDEN), g, 2.0) if cell.spec["residual"] else None
    w = (1.0 + randn((HIDDEN,), g, 0.1).float()).to(torch.bfloat16)
    buf = torch.full((m, LAYERS * HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    args = (h, r, w, EPS, buf[:, slot * HIDDEN : (slot + 1) * HIDDEN])
    certifying(op)
    out = op(*args)
    assert_bf16_close(out, op.reference(*args))
    others = torch.cat([buf[:, : slot * HIDDEN], buf[:, (slot + 1) * HIDDEN :]], dim=1)
    assert bool((others == 7.0).all())
