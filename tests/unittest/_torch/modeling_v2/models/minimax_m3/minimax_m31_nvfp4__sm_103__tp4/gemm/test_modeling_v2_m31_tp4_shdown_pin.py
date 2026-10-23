# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SHDOWN pin of the MiniMax-M3.1 TP4 target: the shared down projection at the rows the decode GEMV does not
serve runs FlashInfer's CuTe DSL MXFP8 GEMM, fixed, instead of the per-start tuned backend."""

import os
import subprocess
import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4 import (
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops.gemm.m3_mxfp8_gemm import (  # noqa: E501
    m3_mxfp8_gemm,
    m3_mxfp8_gemm_cute_dsl,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import ULP, assert_within_ulp  # noqa: E402 — needs the path declared above

_N, _K = modeling.SHARED_DOWN


def test_the_pin_covers_exactly_the_rows_the_gemv_leaves_to_the_gemm():
    pinned = modeling.SHDOWN_PIN_ROWS[modeling.SHARED_DOWN]
    gemv = modeling.MX_GEMV_ROWS[modeling.SHARED_DOWN]
    assert pinned == frozenset(range(17, 33))
    assert not pinned & gemv
    assert set(modeling.SHDOWN_PIN_ROWS) == {modeling.SHARED_DOWN}


def test_the_kill_switch_restores_the_tuned_backend():
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m31_nvfp4__sm_103__tp4 import modeling; print(len(modeling.SHDOWN_PIN_ROWS))"
    )
    env = dict(os.environ, TRTLLM_M31_DISABLE_SHDOWN_PIN="1")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("m", [17, 24, 32])
def test_the_pinned_backend_is_the_mxfp8_product(m):
    """Same E4M3 operands and power-of-two scales as the reference; accumulation order then one bf16 rounding: the
    band of the target's MXFP8 GEMM test (2 ulp of the row scale, RMS 0.25 ulp). Also reports the largest difference
    from the CUTLASS backend the AutoTuner can pick here, in ulp of the row scale (report only)."""
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"shdown{m}".encode()) % 2**31)
    act, act_sf = mxfp8(m, _K, g)
    w, w_sf = mxfp8(_N, _K, g)
    out = m3_mxfp8_gemm_cute_dsl(act, act_sf, w, w_sf, torch.bfloat16)
    ref = m3_mxfp8_gemm.reference(act, act_sf, w, w_sf, torch.bfloat16, False)
    assert out.shape == (m, _N) and out.dtype == torch.bfloat16
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import FlashInferMXFP8GemmRunner as R

    cutlass = R(torch.bfloat16)([act, act_sf, w, w_sf], tactic=R.CUTLASS)
    scale = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    ulps = ((out.float() - cutlass.float()).abs() / scale / ULP[torch.bfloat16]).max().item()
    print(f"[shdown-pin] m={m}: max |cute_dsl - cutlass| = {ulps:.2f} ulp of the row scale")


class _GenericTaken(Exception):
    pass


def _decode_graph_dispatch(monkeypatch):
    """``_linear`` as inside a decode CUDA graph, with no producer-published MXFP8 activation (the shared expert's
    down projection reads the SwiGLU's bf16 output) and the tuned quantize + GEMM replaced by a sentinel."""
    mode = modeling.MxMode(
        decode_graph=True,
        autotune=False,
        tuned_quantize_gemm=True,
        flashinfer=True,
        native_tuning=False,
    )
    monkeypatch.setattr(modeling, "_mx_mode", lambda state: mode)
    monkeypatch.setattr(
        modeling.m3_prequant, "SLOT", type("Slot", (), {"take": staticmethod(lambda x: None)})()
    )

    def generic(*args, **kwargs):
        raise _GenericTaken

    monkeypatch.setattr(modeling, "_mxfp8_linear", generic)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("m", [17, 24, 32])
def test_the_bf16_shared_down_activation_takes_the_pin(monkeypatch, m):
    """The production shared down projection: a bf16 activation at a pinned row count runs the native MXFP8 quantize
    and the CuTe DSL GEMM (bitwise), never the tuned quantize + GEMM a persisted tactic cache steers."""
    _decode_graph_dispatch(monkeypatch)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"shdown-bf16-{m}".encode()) % 2**31)
    w, w_sf = mxfp8(_N, _K, g)
    core = type("Core", (), {"w": {"dn_w": w, "dn_s": w_sf}, "mxfp8": None})()
    x = torch.randn(m, _K, generator=g, device="cuda", dtype=torch.bfloat16)
    out, pushed = modeling._linear(core, x, "dn_w", "dn_s", False)
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    expect = m3_mxfp8_gemm_cute_dsl(act.view(m, _K), act_sf, w, w_sf, torch.bfloat16)
    assert not pushed
    assert torch.equal(out, expect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_unpinned_bf16_rows_keep_the_tuned_backend(monkeypatch):
    _decode_graph_dispatch(monkeypatch)
    g = torch.Generator(device="cuda").manual_seed(33)
    w, w_sf = mxfp8(_N, _K, g)
    core = type("Core", (), {"w": {"dn_w": w, "dn_s": w_sf}, "mxfp8": None})()
    x = torch.randn(33, _K, generator=g, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(modeling, "QPIN", False)  # QPIN's own tests below
    with pytest.raises(_GenericTaken):
        modeling._linear(core, x, "dn_w", "dn_s", False)


class _SwqCore:
    """The decode target's core as ``_swiglu_mlp`` reads it: weights, the MXFP8 state (unused: ``_mx_mode`` is
    patched) and the SwiGLU-OAI scalars."""

    def __init__(self, gu_out, w, w_sf):
        self.w = {"sh_gu_w": gu_out, "sh_gu_s": None, "sh_dn_w": w, "sh_dn_s": w_sf}
        self.mxfp8 = None
        self.swiglu_limit, self.swiglu_alpha, self.swiglu_beta = 7.0, 1.702, 1.0


def _swq_dispatch(monkeypatch, m, seed):
    """Run DecodeTarget._swiglu_mlp at m rows with the gate_up projection replaced by a fixed bf16 output; returns
    (output, the pinned GEMM's activation sources, the expected product)."""
    _decode_graph_dispatch(monkeypatch)
    monkeypatch.setattr(
        modeling, "QPIN", False
    )  # the gate_up stub below stands in for the generic path
    # A real slot: SWQ publishes the MXFP8 activation that the down projection takes.
    monkeypatch.setattr(modeling.m3_prequant, "SLOT", modeling.m3_prequant.PrequantSlot())
    g = torch.Generator(device="cuda").manual_seed(seed)
    gu = (torch.randn(m, 2 * _K, generator=g, device="cuda") * 4.0).to(torch.bfloat16)
    w, w_sf = mxfp8(_N, _K, g)
    core = _SwqCore(gu, w, w_sf)

    def gate_up_or_raise(core_, x, w_, s_, mode):
        if w_ is core_.w["sh_gu_w"]:
            return gu
        raise _GenericTaken

    monkeypatch.setattr(modeling, "_mxfp8_linear", gate_up_or_raise)
    sources = []
    pinned = modeling._shdown_pinned

    def recording(act, act_scale, w_, s_, source):
        sources.append(source)
        return pinned(act, act_scale, w_, s_, source)

    monkeypatch.setattr(modeling, "_shdown_pinned", recording)
    h = torch.zeros(m, 6144, device="cuda", dtype=torch.bfloat16)
    target = type("Target", (), {"core": core})()
    out, pushed = modeling.DecodeTarget._swiglu_mlp(target, h, "sh", False)
    act = torch.ops.trtllm.silu_and_mul(gu, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
    aq, asf = torch.ops.trtllm.mxfp8_quantize(act, True)
    expect = m3_mxfp8_gemm_cute_dsl(aq.view(m, _K), asf, w, w_sf, torch.bfloat16)
    return out, pushed, sources, expect


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("m", [17, 24, 32])
def test_swq_reaches_the_pinned_gemm_on_its_mxfp8_activation(monkeypatch, m):
    """SWQ at 17..32 rows: the down projection takes the prequant branch (the pin's '(mxfp8 activation)' source) and
    its product is bitwise the bf16 SwiGLU + quantize + pinned GEMM path it replaces."""
    out, pushed, sources, expect = _swq_dispatch(
        monkeypatch, m, zlib.crc32(f"swq{m}".encode()) % 2**31
    )
    assert sources == ["mxfp8"] and not pushed
    assert torch.equal(out, expect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_swq_kill_switch_keeps_the_bf16_pin_path(monkeypatch):
    """SWQ off: the same rows take the bf16 SwiGLU and reach the pin with its '(bf16 activation)' source, bitwise
    the same product."""
    monkeypatch.setattr(modeling, "SWQ_ROWS", {})
    out, pushed, sources, expect = _swq_dispatch(monkeypatch, 24, 24)
    assert sources == ["bf16"] and not pushed
    assert torch.equal(out, expect)


def test_swq_rows_are_the_pinned_rows():
    assert modeling.SWQ_ROWS[modeling.SHARED_DOWN] == frozenset(range(17, 33))
    assert modeling.SWQ_ROWS[modeling.SHARED_DOWN] <= modeling.SHDOWN_PIN_ROWS[modeling.SHARED_DOWN]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "m, k, tactic", [(16, 6144, "cute"), (32, 3072, "cute"), (33, 768, "cute"), (32, 768, "trt")]
)
def test_qpin_quantize_is_fixed_and_the_product_bitwise(monkeypatch, m, k, tactic):
    """QPIN: a bf16 projection off the decode kernels' rows quantizes on the fixed tactic (CuTe DSL; TRT-LLM at
    (32, 768)) and runs the tuned GEMM; the product is bitwise the tuned quantize + GEMM's (both quantize tactics
    give the same bytes)."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8QuantizeRunner as QR

    _decode_graph_dispatch(monkeypatch)
    monkeypatch.setattr(modeling, "SHDOWN_PIN_ROWS", {})
    seen = []
    pinned_q = modeling.m3_mxfp8_pinned_quantize

    def recording(x, t):
        seen.append(t)
        return pinned_q(x, t)

    monkeypatch.setattr(modeling, "m3_mxfp8_pinned_quantize", recording)
    g = torch.Generator(device="cuda").manual_seed(m * 7 + k)
    w, w_sf = mxfp8(6144, k, g)
    core = type("Core", (), {"w": {"dn_w": w, "dn_s": w_sf}, "mxfp8": None})()
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    out, pushed = modeling._linear(core, x, "dn_w", "dn_s", False)
    assert seen == [QR.TRTLLM if tactic == "trt" else QR.CUTE_DSL] and not pushed
    aq, asf = torch.ops.trtllm.mxfp8_quantize(x, True)
    expect = m3_mxfp8_gemm(aq.view(m, k), asf, w, w_sf, torch.bfloat16, False)
    assert torch.equal(out, expect)


def test_qpin_kill_switch():
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m31_nvfp4__sm_103__tp4 import modeling; print(modeling.QPIN)"
    )
    env = dict(os.environ, TRTLLM_M31_DISABLE_QPIN="1")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False"
