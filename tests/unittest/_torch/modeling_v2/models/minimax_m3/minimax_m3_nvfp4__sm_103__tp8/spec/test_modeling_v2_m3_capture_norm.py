# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_capture_norm entry (Eagle3 capture with fc_norm folded in)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.spec.m3_capture_norm import (  # noqa: E501
    m3_capture_norm as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import (  # noqa: E402 — needs the path declared above
    EPS,
    HIDDEN,
    assert_bf16_close,
    gen,
    randn,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_capture_norm requires a CUDA device"

LAYERS = 3


def _case(spec, seed):
    g = gen(seed)
    m, slot = spec["m"], spec["slot"]
    h = randn((m, HIDDEN), g, 2.0)
    r = randn((m, HIDDEN), g, 2.0) if spec["residual"] else None
    w = (1.0 + randn((HIDDEN,), g, 0.1).float()).to(torch.bfloat16)
    buf = torch.full((m, LAYERS * HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    dst = buf[:, slot * HIDDEN : (slot + 1) * HIDDEN]
    return (h, r, w, EPS, dst), buf


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Within one bf16 ulp per element on at most 0.1% of the elements; the other layer slots of the buffer
    untouched."""
    args, buf = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    out = op(*args)
    assert_bf16_close(out, op.reference(*args))
    slot = cell.spec["slot"]
    others = torch.cat([buf[:, : slot * HIDDEN], buf[:, (slot + 1) * HIDDEN :]], dim=1)
    assert bool((others == 7.0).all())


def test_guard_refuses_a_strided_weight() -> None:
    (h, r, w, eps, dst), _ = _case(op.CELLS[0].spec, 1)
    w2 = torch.stack([w, w], dim=1)[:, 0]
    with pytest.raises(AssertionError, match="dense"):
        with validating(op):
            op(h, r, w2, eps, dst)
