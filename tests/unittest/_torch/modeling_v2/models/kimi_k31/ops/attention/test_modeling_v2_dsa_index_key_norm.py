# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 index_key_norm entry (the DSA indexer's k LayerNorm, one fixed reduction order).

The raw key is the wk column view of a packed TP16 projection row, as the target passes it. The op is gated
within one bf16 ulp of the plain torch LayerNorm (its catalog reference).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_index_key_norm import (  # noqa: E501
    index_key_norm as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


def _build(spec, seed: int):
    gen = torch.Generator().manual_seed(seed)
    n = spec["tokens"]
    x = torch.randn(n, spec["row_stride"], generator=gen)
    if spec["data"] == "offset":
        x = x + 50 * torch.randn(n, 1, generator=gen)
    elif spec["data"] == "flat":
        x[::2] = 0
        x[1::2] = 1.5
    elif spec["data"] == "wide":
        x = x * torch.pow(10.0, torch.randint(-4, 5, (n, 1), generator=gen).float())
    packed = x.to(torch.bfloat16).cuda()
    col = spec["column"]
    weight = (1 + 0.2 * torch.randn(128, generator=gen)).float().cuda()
    bias = (0.1 * torch.randn(128, generator=gen)).float().cuda()
    return dict(x=packed[:, col : col + 128], weight=weight, bias=bias, eps=spec["eps"])


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """The bf16 spacing at |x| (8 significant bits)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-126)))
    return torch.pow(2.0, e - 7)


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    got = op(**args)
    want = op.reference(**args)
    torch.cuda.synchronize()
    # The two reduce in different orders: the fp32 statistics differ in their last bits, which moves an output by
    # at most one bf16 step, plus an absolute 1e-5 where the output cancels to near zero (x near the mean, or the
    # bias cancelling the scaled value); and only a few outputs in ten thousand move at all.
    g, w = got.float(), want.float()
    excess = (g - w).abs() - _bf16_ulp(w)
    assert float(excess.max()) <= 1e-5, float(excess.max())
    moved = int((got.view(torch.int16) != want.view(torch.int16)).sum())
    assert moved <= max(2, got.numel() // 10000), moved
