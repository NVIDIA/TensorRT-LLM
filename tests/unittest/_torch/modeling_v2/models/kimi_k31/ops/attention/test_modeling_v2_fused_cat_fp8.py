# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_cat_fp8 entry (indexer Q / K concatenate + FP8 quantize).

The inputs are built the way the K3.1 indexer builds them: the 128-wide rows of one bf16 tensor
(Q [T, 32, 128], K [T, 128]) split into two 64-wide halves, so pe and nope are strided views.
The scales are UE8M0, and the output must equal the reference bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fused_cat_fp8 import (  # noqa: E501
    fused_cat_fp8 as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


def _rows(spec, gen) -> torch.Tensor:
    shape = (spec["tokens"], spec["heads"], 128) if spec["heads"] else (spec["tokens"], 128)
    x = torch.randn(*shape, generator=gen)
    if spec["data"] == "zeros_row":
        x.view(-1, 128)[::3] = 0
    elif spec["data"] == "wide":
        x = x * torch.pow(10.0, torch.randint(-6, 7, shape[:-1], generator=gen).float())[..., None]
    return x.to(torch.bfloat16).cuda()


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    x = _rows(spec, gen)
    pe, nope = x.split([spec["pe_dim"], 128 - spec["pe_dim"]], dim=-1)
    return dict(pe=pe, nope=nope)


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out, scale = op(**args)
    ref_out, ref_scale = op.reference(**args)
    torch.cuda.synchronize()
    assert torch.equal(scale, ref_scale), "the per-row scale differs"
    assert torch.equal(out.view(torch.uint8), ref_out.view(torch.uint8))
