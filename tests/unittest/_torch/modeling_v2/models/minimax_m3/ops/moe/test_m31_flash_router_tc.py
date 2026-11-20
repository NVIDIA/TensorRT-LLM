# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe/m31_router_tc entry at the MiniMax-M3.1-Flash sm_107 tp2 target's cells (ROUTER48: 2..48
tokens, logits only or with the NVFP4 activation). The tp4 target's cells are driven by that target's
``moe/test_modeling_v2_m3_router_tc.py``."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.m3_router_gemv import near_tie
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.m31_router_tc import (  # noqa: E501
    m3_router_tc as op,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.m31_router_tc import (  # noqa: E501
    nvfp4_quantize,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

#: The experts' NVFP4 input scale the cells use: 448 * 6 / 8, block scales well inside E4M3.
SF_SCALE = 336.0
CELLS = op.CELLS_BY_TP[2]


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """[M, K / 2] packed e2m1 bytes -> [M, K] 4-bit codes, low nibble first."""
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(packed.shape[0], -1)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell) -> None:
    """Logits within 64 fp32 ulps of each row's largest (8 in RMS) of the TF32 product; NVFP4 scales bit-exact and
    codes bit-exact except next to an e2m1 midpoint, where the kernel's approximate reciprocals choose the side
    (and then only by one code)."""
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = (torch.randn(spec["m"], spec["k"], generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    w = torch.randn(128, spec["k"], generator=g, device="cuda") * 0.02
    sf_scale = torch.tensor([SF_SCALE], dtype=torch.float32, device="cuda") if spec["fp4"] else None
    with validating(op):
        outs = op(x, w, sf_scale, False)
    ref = op.reference(x, w, sf_scale, False)
    assert len(outs) == len(ref) == (3 if spec["fp4"] else 1)
    assert outs[0].dtype == torch.float32 and outs[0].shape == ref[0].shape
    assert_within_ulp(outs[0], ref[0], element_ulp=64.0, rms_ulp=8.0)
    if spec["fp4"]:
        _, _, scaled = nvfp4_quantize(x, sf_scale)
        assert torch.equal(outs[2], ref[2]), (
            f"{(outs[2] != ref[2]).sum().item()} block scales differ"
        )
        got, want = _codes(outs[1]), _codes(ref[1])
        differ = got != want
        assert not bool((differ & ~near_tie(scaled)).any()), "e2m1 codes differ away from a tie"
        assert bool((((got & 7).int() - (want & 7).int()).abs()[differ] <= 1).all())
        # A code that moves across a tie keeps its sign; only a zero magnitude may carry either sign.
        same_sign = ((got >> 3) == (want >> 3)) | (((got & 7) == 0) & ((want & 7) == 0))
        assert bool(same_sign[differ].all()), "a tie resolved to the opposite sign"
