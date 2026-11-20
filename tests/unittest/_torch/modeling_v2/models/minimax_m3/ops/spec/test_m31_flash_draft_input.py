# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the spec/m31_draft_input entry of the MiniMax-M3.1-Flash sm_107 targets (embedding + both RMSNorms +
concat in one launch)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.m31_draft_input import (  # noqa: E501
    m3_draft_input as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _m3_test_inputs import assert_bf16_close, gen, randn, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402, F401

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

VOCAB, EPS = 200064, 1e-6
# Every target certifies the same cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


def _case(spec, seed):
    g = gen(seed)
    m, h = spec["m"], spec["h"]
    table = randn((VOCAB, h), g)
    ids = torch.randint(0, VOCAB, (m,), generator=g, device="cuda", dtype=torch.int32)
    if spec["invalid_id"]:
        ids[0] = -1
    hidden = randn((4 * m if spec["gather"] else m, h), g)
    rows = torch.arange(m, device="cuda", dtype=torch.int64) * 4 + 2 if spec["gather"] else None
    w_e = (1.0 + randn((h,), g, 0.1).float()).to(torch.bfloat16)
    w_h = (1.0 + randn((h,), g, 0.1).float()).to(torch.bfloat16)
    out = torch.full((m, 2 * h), 7.0, dtype=torch.bfloat16, device="cuda")
    residual = torch.full((m, h), 7.0, dtype=torch.bfloat16, device="cuda")
    return (ids, table, hidden, rows, w_e, w_h, EPS, EPS, out, residual, True)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Normalized halves within one bf16 ulp per element on at most 0.1% of the elements (fp32 statistic in a
    different summation order, rsqrtf); the gathered residual and the zero embedding of an invalid id exact."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    out, residual = op(*args)
    ref_out, ref_res = op.reference(*args)
    torch.cuda.synchronize()
    assert_bf16_close(out, ref_out)
    assert torch.equal(residual, ref_res)
    if cell.spec["invalid_id"]:
        assert not bool(out[0, : cell.spec["h"]].float().any())
