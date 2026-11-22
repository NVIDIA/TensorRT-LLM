# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the spec/m31_chunk_norm entry of the MiniMax-M3.1-Flash sm_107 targets (the Eagle3 fc_norm over the
captured layers)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.m31_chunk_norm import (  # noqa: E501
    m3_chunk_norm as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _m3_test_inputs import assert_bf16_close, gen, randn, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402, F401

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

EPS = 1e-6
# Every target certifies the same cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Within one bf16 ulp per element on at most 0.1% of the elements (fp32 statistic, rsqrtf)."""
    g = gen(seed_of(cell.why))
    m, c, h = cell.spec["m"], cell.spec["c"], cell.spec["h"]
    src = randn((m, c * h + cell.spec["extra_cols"]), g, 2.0)
    w = (1.0 + randn((c, h), g, 0.1).float()).to(torch.bfloat16)
    dst = torch.full((m, c * h), 7.0, dtype=torch.bfloat16, device="cuda")
    args = (src, w, EPS, dst, True)
    certifying(op)
    out = op(*args)
    assert_bf16_close(out, op.reference(*args))
