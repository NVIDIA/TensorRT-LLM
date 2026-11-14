# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_PROJ_CUTE on an SM107 GPU.

At the DEP8 projections' shapes (qkv_proj [9856, 6144], o_proj [6144, 8192]) and the decode graphs' row counts that
the pinned table covers, the pinned CuTe DSL tactic writes FlashInfer's CUTLASS MXFP8 GEMM output bit for bit, for its
heuristic and for each of its tactics that run the shape, so the backend choice changes only the time. The hand-off
GEMM with DEP_PROJ_CUTE inside DEP_PROJ_PIN's engagement returns the tuned GEMM half's output bit for bit.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    merge_mxfp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="needs an SM107 GPU")

SHAPES = {"qkv_proj": (9856, 6144), "o_proj": (6144, 8192)}
ROWS = (64, 96, 128)


def _operands(n, k, rows, seed):
    from flashinfer import mxfp8_quantize

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w_mx, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    a_mx, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout=True)
    return a_mx, a_sf, w_mx, w_sf


def _cutlass(a_mx, a_sf, w_mx, w_sf, tactic):
    from flashinfer.gemm import gemm_base as gb

    major, _ = gb.get_compute_capability(a_mx.device)
    runner = gb.get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
    out = torch.empty(a_mx.shape[0], w_mx.shape[0], device=a_mx.device, dtype=torch.bfloat16)
    ws = gb._get_cache_buf("mm_mxfp8_workspace", gb.DEFAULT_WORKSPACE_SIZE, a_mx.device)
    runner(inputs=[a_mx, w_mx.t(), a_sf, w_sf, torch.bfloat16, out, ws], tactic=tactic)
    return out


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", list(SHAPES))
def test_the_pinned_tactic_writes_every_cutlass_tactics_output(name, rows):
    from flashinfer.gemm import gemm_base as gb

    n, k = SHAPES[name]
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=rows)
    with pin.engaged_for("test DEP_PROJ_CUTE"):
        pinned = pin.run(a_mx, a_sf, w_mx, w_sf, torch.bfloat16)
    assert pinned is not None, "the table must hold this shape and bucket"
    reference = _cutlass(a_mx, a_sf, w_mx, w_sf, -1)
    assert torch.equal(pinned.view(torch.int16), reference.view(torch.int16))
    major, _ = gb.get_compute_capability(a_mx.device)
    module = gb.get_cutlass_mxfp8_gemm_module(major)
    ran = 0
    for tactic in module.cutlass_mxfp8_gemm_runner().get_valid_tactics([a_mx], None):
        try:
            out = _cutlass(a_mx, a_sf, w_mx, w_sf, tactic)
        except RuntimeError:  # a tactic that cannot implement the shape
            continue
        ran += 1
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), tactic
    assert ran > 0


def _method():
    method = MXFP8LinearMethod()
    method.__class__ = merge_mxfp8.DepMergeMxfp8LinearMethod
    method.use_cutlass = True
    method.backend = "trtllm"
    method.use_native_autotuner = False
    method._flashinfer_autotuned = False
    method.tune_decode_graph_backends = True
    return method


@pytest.mark.parametrize("rows", (96, 128))
@pytest.mark.parametrize("name", list(SHAPES))
def test_the_hand_off_gemm_is_the_tuned_gemm_half_bit_for_bit(name, rows, monkeypatch):
    n, k = SHAPES[name]
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=7 + rows)
    module = types.SimpleNamespace(weight=w_mx, weight_scale=w_sf, dtype=torch.bfloat16)
    x = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
    outs = {}
    # DEP_GEMM_PROVIDER would serve o_proj at these rows ahead of the pinned tactic (its own test covers it).
    monkeypatch.setattr(merge_mxfp8._provider, "DEP_GEMM_PROVIDER", False)
    # A spy on the pinned runner that calls through: the enabled iteration must take it, the disabled one must not.
    real_run, served = pin.run, []

    def spy(*args, **kwargs):
        out = real_run(*args, **kwargs)
        served.append(out is not None)
        return out

    monkeypatch.setattr(pin, "run", spy)
    for on in (True, False):
        served.clear()
        monkeypatch.setattr(merge_mxfp8, "DEP_PROJ_CUTE", on)
        merge_mxfp8.SLOT.publish(x, a_mx, a_sf)
        token = _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.set(True)
        try:
            with pin.engaged_for("test DEP_PROJ_CUTE"):
                outs[on] = _method().apply(module, x, None)
        finally:
            _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.reset(token)
        assert merge_mxfp8.SLOT.pending is None
        assert served == ([True] if on else []), (on, served)
    assert torch.equal(outs[True].view(torch.int16), outs[False].view(torch.int16))
