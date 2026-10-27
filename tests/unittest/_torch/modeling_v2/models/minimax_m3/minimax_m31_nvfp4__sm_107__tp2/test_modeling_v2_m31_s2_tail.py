# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2W_S2_TAIL dispatch: default-on policy, row band, and declines that keep the stock boundary.
The fused op's numerics are its 2-rank microbench (bitwise to finalize + one-shot Lamport AR + norm + mxfp8_quantize at
1..224 rows) and, in-model, TRTLLM_M31_TP2W_S2_CHECK."""

import types

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    modeling as m,
)


def test_default_on_with_kill_switch() -> None:
    assert m.TP2W_S2_TAIL == (not m.m31_disabled("TP2W_S2_TAIL"))


def test_row_band_is_past_the_mc_finalize_and_within_the_mb() -> None:
    assert m.S2_TAIL_MIN_ROWS == m.MOE_TRITON_FINALIZE_MIN_TOKENS
    assert m.S2_TAIL_MAX_ROWS == 224


def test_boundary_declines_without_the_op(monkeypatch) -> None:
    monkeypatch.setattr(m, "_s2_op", lambda: None)
    parts = (
        torch.zeros(4, 8, dtype=torch.bfloat16),
        torch.zeros(1, 4),
        torch.zeros(1, 4, dtype=torch.int32),
    )
    assert m._s2_boundary(types.SimpleNamespace(), 1, parts, None, None, None, None) is None


def test_boundary_declines_parts_not_the_runner_layout(monkeypatch) -> None:
    monkeypatch.setattr(m, "_s2_op", lambda: object())
    assert (
        m._s2_boundary(types.SimpleNamespace(), 1, torch.zeros(1, 8), None, None, None, None)
        is None
    )


def test_decode_tail_uses_existing_moe_parts_and_stock_decline(monkeypatch) -> None:
    final_norm = object()
    norm = object()
    core = types.SimpleNamespace(w={"final_norm": final_norm}, allreduce=object())
    target = m.DecodeTarget(core)
    h = torch.zeros(88, 8, dtype=torch.bfloat16)
    residual = torch.ones_like(h)
    parts = [object(), object(), object(), object()]
    calls = []
    monkeypatch.setattr(m, "_triton_finalize_rows", lambda *args: True)
    monkeypatch.setattr(m, "is_torch_compiling", lambda: False)
    monkeypatch.setattr(m, "_s2_op", lambda: object())
    monkeypatch.setattr(target, "_moe", lambda *args, **kwargs: (parts, None, False, False))

    def boundary(c, rows, routed, shared, r, n, ar):
        assert c is core and rows == 88 and tuple(routed) == tuple(parts[1:])
        assert shared is parts[0] and r is residual and n is norm and ar is core.allreduce
        calls.append("fused")
        return None

    monkeypatch.setattr(m, "_s2_boundary", boundary)
    monkeypatch.setattr(m, "_triton_finalize", lambda *args: h)
    monkeypatch.setattr(m, "_native_allreduce_norm", lambda *args: (h, residual))
    result = target._s2_tail(3, h, residual, norm)
    assert result[0] is h and result[1] is residual
    assert calls == ["fused"]
    assert target._s2_tail(3, h, residual, final_norm) is None
    assert target._s2_tail(3, h[:48], residual[:48], norm) is None
    assert target._s2_tail(3, h.float(), residual, norm) is None
    assert calls == ["fused"]


def _decode_graph_core(monkeypatch, pinned: bool):
    monkeypatch.setattr(m, "TP2W_S2_TAIL", True)
    monkeypatch.setattr(m, "MX_TAC_TP4", pinned)
    monkeypatch.setattr(
        m,
        "_mx_mode",
        lambda state: m.MxMode(
            decode_graph=True,
            autotune=False,
            tuned_quantize_gemm=True,
            flashinfer=True,
            native_tuning=False,
        ),
    )
    monkeypatch.setattr(m, "linear_path", lambda *a: m.GENERIC_GEMM_PATH)
    monkeypatch.setattr(m.m3_prequant.SLOT, "pending", None)
    method = types.SimpleNamespace(tune_decode_graph_backends=True)
    return types.SimpleNamespace(
        w={"w": torch.zeros(16, 32), "s": torch.zeros(16, 1)},
        mxfp8=types.SimpleNamespace(quant_method=method),
    )


def test_decode_graph_qkv_takes_the_tail_mxfp8_on_the_pinned_path(monkeypatch) -> None:
    core = _decode_graph_core(monkeypatch, pinned=True)
    calls = []
    monkeypatch.setattr(
        m, "m3_mxfp8_pinned_gemm", lambda a, sf, w, s, dt, t: calls.append((a, sf)) or "gemm"
    )
    monkeypatch.setattr(
        m, "m3_mxfp8_pinned_quantize", lambda *a: (_ for _ in ()).throw(AssertionError("quantized"))
    )
    x = torch.zeros(64, 32, dtype=torch.bfloat16)
    q, sf = torch.ones(64, 32), torch.ones(64)
    m._S2_MX.publish(x, q, sf)
    out, pushed = m._linear(core, x, "w", "s", True)
    assert out == "gemm" and not pushed
    assert len(calls) == 1 and calls[0][0] is q and calls[0][1] is sf
    assert m._S2_MX.pending is None


def test_decode_graph_off_the_pinned_path_keeps_the_stock_quantize(monkeypatch) -> None:
    core = _decode_graph_core(monkeypatch, pinned=False)
    monkeypatch.setattr(
        m, "m3_mxfp8_pinned_gemm", lambda *a: (_ for _ in ()).throw(AssertionError("pinned"))
    )
    monkeypatch.setattr(m, "_mxfp8_linear", lambda core, x, w, s, mode: "stock")
    x = torch.zeros(64, 32, dtype=torch.bfloat16)
    m._S2_MX.publish(x, torch.ones(64, 32), torch.ones(64))
    out, _ = m._linear(core, x, "w", "s", True)
    assert out == "stock"
    assert m._S2_MX.pending is None


def test_tail_mxfp8_of_another_tensor_is_not_taken(monkeypatch) -> None:
    core = _decode_graph_core(monkeypatch, pinned=True)
    monkeypatch.setattr(m, "m3_mxfp8_pinned_gemm", lambda a, sf, w, s, dt, t: ("gemm", a.dtype))
    monkeypatch.setattr(
        m, "m3_mxfp8_pinned_quantize", lambda x, t: (torch.zeros(1, dtype=torch.int8), None)
    )
    m._S2_MX.publish(torch.zeros(64, 32, dtype=torch.bfloat16), torch.ones(64, 32), torch.ones(64))
    out, _ = m._linear(core, torch.zeros(64, 32, dtype=torch.bfloat16), "w", "s", True)
    assert out == ("gemm", torch.int8)
