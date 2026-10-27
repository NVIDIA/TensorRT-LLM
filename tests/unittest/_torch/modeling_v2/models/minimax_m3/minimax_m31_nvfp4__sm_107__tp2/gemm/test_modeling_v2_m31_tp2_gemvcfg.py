# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2_GEMVCFG (kernels/mx_gemv/op._CONFIGS_BY_M): the TP2 per-row GEMV config of the shared down projection.

- the shared down projection (6144, 1536) at 9..16 rows takes ring 2 with the table's split 2 and contiguous k-tiles:
  bit for bit the table's (2, 12) output, on a bf16 and an MXFP8 activation;
- every other (projection, rows) keeps the table, the shared gate_up's bf16 rows stay 16 (its swept split-8 config
  slowed the 32-row step in-model), and the kill switch restores the table.
"""

import importlib
import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.mx_gemv import (  # noqa: E501
    op as mx_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.gemm.m3_mx_gemv import (  # noqa: E501
    SHARED_DOWN,
    SHARED_GATE_UP,
    m3_mx_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.gemm.m3_mx_gemv_bf16 import (  # noqa: E501
    m3_mx_gemv_bf16 as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

# The element band of test_modeling_v2_m3_mx_gemv_bf16 (the same fp32 sums against fp64).
ELEMENT_ULP = 1.5
TABLE = {SHARED_DOWN: (2, 12), SHARED_GATE_UP: (4, 12)}


def _dispatch():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
        modeling as dispatch,
    )

    return dispatch


def _x(m: int, k: int, g: torch.Generator, stress: bool) -> torch.Tensor:
    x = torch.randn(m, k, generator=g, device="cuda")
    if stress:
        block = torch.exp2(torch.randint(-12, 13, (m, k // 32), generator=g, device="cuda").float())
        x = x * block.repeat_interleave(32, dim=1)
    return x.to(torch.bfloat16)


def test_config_table() -> None:
    """The shared down at 9..16 rows takes (2, 2, 1); everything else keeps the table; the bf16 caps are unchanged."""
    assert mx_gemv.TP2_GEMVCFG
    for m in range(1, 49):
        down = mx_gemv.get_config(*SHARED_DOWN, m)
        assert down == ((2, 2, 1) if 9 <= m <= 16 else TABLE[SHARED_DOWN]), (m, down)
        assert mx_gemv.get_config(*SHARED_GATE_UP, m) == TABLE[SHARED_GATE_UP], m
    assert set(mx_gemv._CONFIGS_BY_M) == {(*SHARED_DOWN, m) for m in range(9, 17)}
    assert _dispatch().MX_GEMV_BF16_MAX_TOKENS[SHARED_GATE_UP] == 16


def test_kill_switch_restores_the_table(monkeypatch) -> None:
    """TRTLLM_M31_DISABLE_TP2_GEMVCFG=1: no per-row configs."""
    monkeypatch.setattr(m31_switches, "_DISABLED", frozenset({"TP2_GEMVCFG"}))
    try:
        m = importlib.reload(mx_gemv)
        assert not m.TP2_GEMVCFG and m._CONFIGS_BY_M == {}
        assert m.get_config(*SHARED_DOWN, 16) == TABLE[SHARED_DOWN]
    finally:
        monkeypatch.undo()
        importlib.reload(mx_gemv)


@pytest.mark.parametrize("m", [9, 12, 16])
@pytest.mark.parametrize("stress", [False, True])
def test_shared_down_ring2_is_bitwise_the_table(m, stress) -> None:
    """Ring depth only stages k-tiles: (2, 2, 1) writes the table's (2, 12) rows bit for bit, bf16 and MXFP8 in."""
    n, k = SHARED_DOWN
    g = torch.Generator(device="cuda").manual_seed(
        zlib.crc32(f"gemvcfg-down-{m}-{stress}".encode())
    )
    x = _x(m, k, g, stress)
    w, wsf = mxfp8(n, k, g, 115, 140)
    new = op(x, w, wsf, use_pdl=True)
    old = op(x, w, wsf, cfg=TABLE[SHARED_DOWN], use_pdl=True)
    assert new is not None and old is not None
    assert torch.equal(new.view(torch.int16), old.view(torch.int16))
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    pre_new = m3_mx_gemv(xq.view(m, k), xsf, w, wsf, use_pdl=True)
    pre_old = m3_mx_gemv(xq.view(m, k), xsf, w, wsf, cfg=TABLE[SHARED_DOWN], use_pdl=True)
    assert torch.equal(pre_new.view(torch.int16), pre_old.view(torch.int16))


@pytest.mark.parametrize("m", [17, 24, 32])
def test_shared_gate_up_bf16_rows_past_16_keep_the_gemm(m) -> None:
    d = _dispatch()
    assert d.linear_path(*SHARED_GATE_UP, m, False, False) == d.GENERIC_GEMM_PATH
