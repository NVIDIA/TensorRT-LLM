# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VRCFG-SWI: the sm_107 TP8 mx_gemv's swiglu-prologue calls of the dense down projection (6144, 1536) at up to 8 rows
run ring 3 instead of the table's (ring 8, clamped to the 6 k-tiles a rank owns), bit for bit the table config's
output (same split, same k-tile order); other calls keep the table; TRTLLM_M31_DISABLE_MXGEMV_VRCFG restores the table
for every call."""

import importlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    m31_switches,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.mx_gemv import (  # noqa: E501
    op as gemv,
)

N, K = 6144, 1536


def _ring(m: int, x_swiglu: bool):
    res = gemv._resolve(m, N, K, True, True, None, True, None, x_swiglu, False)
    return None if res is None else res[0][3]


def test_vrcfg_scope():
    assert gemv.MXGEMV_VRCFG and gemv._CONFIGS_SWIGLU == {(N, K): (2, 3, 0)}
    table = gemv.get_config(N, K, 8)
    assert table == (2, 8, 0)
    for m in range(1, 9):
        assert _ring(m, True) == 3
        assert _ring(m, False) == 6
    assert _ring(16, True) == 6  # past the swept rows: the table


def test_kill_switch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MXGEMV_VRCFG", "1")
    importlib.reload(m31_switches)
    try:
        g = importlib.reload(gemv)
        assert not g.MXGEMV_VRCFG and g._CONFIGS_SWIGLU == {}
        res = g._resolve(8, N, K, True, True, None, True, None, True, False)
        assert res is not None and res[0][3] == 6
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_MXGEMV_VRCFG", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(gemv)
    assert gemv.MXGEMV_VRCFG


def _mxfp8_weight(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device="cuda") / K**0.5).to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    return wq.view(N, K), wsf


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target's GEMV runs on SM107 only",
)
@pytest.mark.parametrize("m", [1, 4, 7, 8])
def test_ring3_is_bitwise_the_table(m):
    wq, wsf = _mxfp8_weight(m)
    g = torch.Generator(device="cuda").manual_seed(100 + m)
    gate_up = (
        (torch.randn(m, 2 * K, generator=g, device="cuda") * 3).to(torch.bfloat16).contiguous()
    )
    got = gemv.m3_mx_gemv_swiglu(gate_up, wq, wsf, use_pdl=True)
    table = gemv.m3_mx_gemv_swiglu(gate_up, wq, wsf, cfg=gemv.get_config(N, K, m), use_pdl=True)
    torch.cuda.synchronize()
    assert got is not None and table is not None and bool(table.any())
    assert torch.equal(got, table)
