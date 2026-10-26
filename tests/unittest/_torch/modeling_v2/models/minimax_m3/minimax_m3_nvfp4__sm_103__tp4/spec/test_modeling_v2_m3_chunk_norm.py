# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_chunk_norm entry (the Eagle3 fc_norm over the captured layers)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_chunk_norm import (  # noqa: E501
    m3_chunk_norm as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import (  # noqa: E402 — needs the path declared above
    EPS,
    assert_bf16_close,
    gen,
    randn,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_chunk_norm requires a CUDA device"


def _case(spec, seed):
    g = gen(seed)
    m, c, h = spec["m"], spec["c"], spec["h"]
    buf = randn((m, c * h + spec["extra_cols"]), g, 2.0)
    src = buf[:, : c * h + spec["extra_cols"]]
    w = (1.0 + randn((c, h), g, 0.1).float()).to(torch.bfloat16)
    dst = torch.full((m, c * h), 7.0, dtype=torch.bfloat16, device="cuda")
    return (src, w, EPS, dst, True)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Within one bf16 ulp per element on at most 0.1% of the elements (fp32 statistic, rsqrtf)."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    out = op(*args)
    assert_bf16_close(out, op.reference(*args))


def test_guard_refuses_a_strided_weight() -> None:
    src, w, eps, dst, pdl = _case(op.CELLS[0].spec, 1)
    wt = torch.empty((w.shape[1], w.shape[0]), dtype=w.dtype, device="cuda").t()
    wt.copy_(w)
    with pytest.raises(AssertionError, match="dense"):
        with validating(op):
            op(src, wt, eps, dst, pdl)
