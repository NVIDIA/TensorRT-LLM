# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_linear entry (``F.linear``, optionally with TF32).

GPU (sm_107): a sample of the cells against the entry's FP64 reference within the dtype's accumulation band, TF32's
10-bit mantissa where ``tf32`` is set; the call leaves ``allow_tf32`` as it found it.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_linear import rubin_linear as op

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import assert_within_ulp  # noqa: E402 — needs the path declared above

on_sm107 = on_sm(107)


CELLS = sample_cells(op.CELLS, size_key="m")[:2]


@on_sm107
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell):
    spec = cell.spec
    gen = torch.Generator().manual_seed(spec["m"] * 7 + spec["n"])
    x = torch.randn(spec["m"], spec["k"], generator=gen).to(spec["dtype"]).cuda()
    w = (
        (torch.randn(spec["n"], spec["k"], generator=gen) / spec["k"] ** 0.5)
        .to(spec["dtype"])
        .cuda()
    )
    prev = torch.backends.cuda.matmul.allow_tf32
    out = op(x, w, spec["tf32"])
    assert torch.backends.cuda.matmul.allow_tf32 == prev
    # Measured with the same cuBLAS calls on sm_103: BF16 1.19 ulp per element and 0.03 RMS, TF32 (ulp 2^-10)
    # 0.48 and 0.30.
    ulp = 2.0**-10 if spec["tf32"] else None
    assert_within_ulp(out, op.reference(x, w, spec["tf32"]), element_ulp=2.0, rms_ulp=0.5, ulp=ulp)
