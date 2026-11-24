# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the general target's fused Kimi K3 routing + MXFP8 quantization (``k31_noaux_tc_mxfp8_quant``): every
cell bitwise against the unfused chain (noaux_tc, the BF16 cast, mxfp8_quantize), and ``fused_noaux_quant`` declining
what the op does not take."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.ops.moe import (  # noqa: E501
    noaux_tc_mxfp8_quant as m,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the fused Kimi K3 route + MXFP8 quantization needs an SM10x GPU",
)


def _inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, m.NUM_EXPERTS, generator=g, device="cuda") * 2.0
    bias = torch.randn(m.NUM_EXPERTS, generator=g, device="cuda") * 0.1
    x = (torch.randn(rows, m.HIDDEN, generator=g, device="cuda") * 3).to(torch.bfloat16)
    return logits, bias, x


@pytest.mark.parametrize(
    "cell", m.k31_noaux_tc_mxfp8_quant.CELLS, ids=[c.why for c in m.k31_noaux_tc_mxfp8_quant.CELLS]
)
def test_cells_match_the_unfused_chain(cell) -> None:
    logits, bias, x = _inputs(cell.spec["rows"], seed=cell.spec["rows"])
    args = (logits, bias, x, cell.spec["scale"])
    out = m.k31_noaux_tc_mxfp8_quant(*args)
    ref = m.k31_noaux_tc_mxfp8_quant.reference(*args)
    for name, a, b in zip(("ids", "weights", "x_fp8", "x_sf"), out, ref):
        assert a.shape == b.shape, name
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), name


def test_declines_what_the_op_does_not_take() -> None:
    logits, bias, x = _inputs(65, seed=1)
    assert m.fused_noaux_quant(logits, bias, 1.0, x) is None  # above 64 rows
    logits, bias, x = _inputs(8, seed=2)
    assert m.fused_noaux_quant(logits, bias, 1.0, x.float()) is None
    assert m.fused_noaux_quant(logits.t().contiguous().t(), bias, 1.0, x) is None
    assert m.fused_noaux_quant(logits, bias, 1.0, x) is not None
