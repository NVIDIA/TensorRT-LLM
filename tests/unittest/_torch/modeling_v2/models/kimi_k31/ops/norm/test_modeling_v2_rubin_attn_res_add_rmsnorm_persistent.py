# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 rubin_attn_res_add_rmsnorm_persistent entry.

A cell is (T, N, add): T tokens, N = K + 1 candidates, with or without the residual add. Each
sampled cell checks the updated prefix bit for bit (a torch bf16 add; without an add, the input itself) and
the output against the entry's reference in bf16 steps (one step).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_add_rmsnorm_persistent import (  # noqa: E501
    rubin_attn_res_add_rmsnorm_persistent as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(107)

H = 7168
# A step is one bf16 ulp (8 significant bits) at the larger of the two values, taken at no less
# than 1/32 of the row RMS: where the mixture cancels, the fp32 summation order alone moves an
# output by more than its own ulp.
FLOOR = 1 / 32
# Against the reference both sides round once: one step at most, and few elements move (fp32
# differences of about 1e-6 flip about 1e-6 / 2^-8 of the roundings). One extra rounding point
# moves about a third of them.
MAX_CHANGED = 0.01


def _inputs(tokens: int, candidates: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return (
        rn(tokens, 1, H, sc=0.05),
        rn(tokens, 1, H, sc=0.05),
        rn(candidates - 1, tokens, 1, H, sc=0.05),
        rn(H, sc=0.02),
        (1 + rn(H, sc=0.02).float()).to(torch.bfloat16),
        (1 + rn(H, sc=0.02).float()).to(torch.bfloat16),
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:56] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(op)
    spec = cell.spec
    assert spec["hidden"] == H
    layer, add, block, res_w, rms_w, out_w = _inputs(
        spec["tokens"], spec["candidates"], zlib.crc32(cell.why.encode()) % 2**31
    )
    add = add if spec["add"] else None
    args = (layer, add, block, res_w, rms_w, out_w, spec["eps"], spec["eps"])
    updated, out = op(*args)
    ref_updated, ref_out = op.reference(*args)
    if add is None:
        assert updated.data_ptr() == layer.data_ptr(), "without an add the input comes back"
    else:
        assert torch.equal(_bits(updated), _bits(ref_updated)), "the add is not a bf16 add"
    # |out - ref| per element in bf16 steps (see FLOOR).
    a, e = out.float(), ref_out.float()
    floor = e.square().mean(dim=-1, keepdim=True).sqrt() * FLOOR
    scale = torch.maximum(torch.maximum(a.abs(), e.abs()), floor).clamp_min(2.0**-126)
    worst = ((a - e).abs() / torch.exp2(torch.floor(torch.log2(scale)) - 7)).max().item()
    changed = (a != e).float().mean().item()
    assert worst <= 1 and changed <= MAX_CHANGED, (worst, changed)
