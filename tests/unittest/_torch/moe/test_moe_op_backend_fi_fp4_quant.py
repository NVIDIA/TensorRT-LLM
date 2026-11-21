# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TRTLLMOpBackend.fp4_quantize`` hands large bf16 inputs to FlashInfer's NVFP4 kernel byte for byte."""

import os
import subprocess
import sys
import types

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe import moe_op_backend


def _fp4_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.fixture(autouse=True)
def _fresh_recipe_check():
    moe_op_backend._flashinfer_fp4_recipe_is_native.cache_clear()
    yield
    moe_op_backend._flashinfer_fp4_recipe_is_native.cache_clear()


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, "True 4096"),
        ({"TRTLLM_MOE_DISABLE_FLASHINFER_FP4_QUANTIZE": "1"}, "False 4096"),
        ({"TRTLLM_MOE_FLASHINFER_FP4_QUANTIZE_MIN_ROWS": "1024"}, "True 1024"),
    ],
)
def test_the_flashinfer_route_reads_its_switches(environ, expected):
    env = {k: v for k, v in os.environ.items() if "FLASHINFER_FP4_QUANTIZE" not in k}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tensorrt_llm._torch.moe.fused_moe import moe_op_backend as m;"
            "print(m._FLASHINFER_FP4_QUANTIZE, m._FLASHINFER_FP4_QUANTIZE_MIN_ROWS)",
        ],
        env=env | environ,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == expected


@pytest.mark.parametrize("pdl_env, expected", [(None, None), ("1", None), ("0", False)])
def test_the_flashinfer_route_follows_trtllm_enable_pdl(monkeypatch, pdl_env, expected):
    """Like the TRT op, the FlashInfer kernel launches with PDL unless TRTLLM_ENABLE_PDL=0."""
    seen = {}

    def fake_fp4_quantize(input, global_scale, *args, **kwargs):
        seen.update(kwargs)
        rows, cols = input.shape
        return (
            torch.zeros(rows, cols // 2, dtype=torch.uint8),
            torch.zeros(rows * cols // 16, dtype=torch.uint8),
        )

    monkeypatch.setitem(
        sys.modules, "flashinfer", types.SimpleNamespace(fp4_quantize=fake_fp4_quantize)
    )
    monkeypatch.setattr(moe_op_backend, "_FLASHINFER_FP4_QUANTIZE", True)
    monkeypatch.setattr(moe_op_backend, "_FLASHINFER_FP4_QUANTIZE_MIN_ROWS", 1)
    if pdl_env is None:
        monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", pdl_env)
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    moe_op_backend.TRTLLMOpBackend().fp4_quantize(x, torch.ones(1), 16, False, True)
    assert seen["enable_pdl"] is expected


def test_a_flashinfer_nvfp4_recipe_knob_keeps_the_trtllm_op(monkeypatch):
    """A FlashInfer NVFP4 recipe knob changes the quantized bits, so the TRT op runs instead."""
    fp4_quantization = types.ModuleType("flashinfer.quantization.fp4_quantization")
    fp4_quantization.NVFP4_QUANT_ENV_VARS = ("FLASHINFER_NVFP4_4OVER6",)
    quantization = types.ModuleType("flashinfer.quantization")
    quantization.fp4_quantization = fp4_quantization
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.quantization = quantization
    flashinfer_calls, trtllm_calls = [], []
    flashinfer.fp4_quantize = lambda *a, **k: flashinfer_calls.append(1)
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.quantization", quantization)
    monkeypatch.setitem(sys.modules, "flashinfer.quantization.fp4_quantization", fp4_quantization)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "fp4_quantize",
        lambda *a: trtllm_calls.append(1) or (None, None),
        raising=False,
    )
    monkeypatch.setattr(moe_op_backend, "_FLASHINFER_FP4_QUANTIZE", True)
    monkeypatch.setattr(moe_op_backend, "_FLASHINFER_FP4_QUANTIZE_MIN_ROWS", 1)
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "1")
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    moe_op_backend.TRTLLMOpBackend().fp4_quantize(x, torch.ones(1), 16, False, True)
    assert flashinfer_calls == [] and trtllm_calls == [1]


@pytest.mark.skipif(not _fp4_available(), reason="needs an SM100+ GPU")
@pytest.mark.parametrize("rows, hidden", [(4095, 6144), (4096, 8192)])
@pytest.mark.parametrize("swizzled", [True, False])
def test_large_inputs_quantize_byte_equal_to_the_trtllm_op(monkeypatch, rows, hidden, swizzled):
    monkeypatch.setattr(moe_op_backend, "_FLASHINFER_FP4_QUANTIZE", True)
    monkeypatch.setattr(moe_op_backend, "_FLASHINFER_FP4_QUANTIZE_MIN_ROWS", 4096)
    flashinfer = pytest.importorskip("flashinfer")
    real = flashinfer.fp4_quantize
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(flashinfer, "fp4_quantize", spy)
    gen = torch.Generator(device="cuda").manual_seed(rows + hidden)
    x = (torch.randn(rows, hidden, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    gs = torch.tensor([448.0 * 6.0 / float(x.abs().max())], device="cuda")
    q, sf = moe_op_backend.TRTLLMOpBackend().fp4_quantize(x, gs, 16, False, swizzled)
    # The 4096-row case runs FlashInfer, the 4095-row case the TRT op.
    assert bool(calls) == (rows >= 4096)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, swizzled)
    assert (q.dtype, q.shape, sf.dtype, sf.shape) == (
        q_ref.dtype,
        q_ref.shape,
        sf_ref.dtype,
        sf_ref.shape,
    )
    assert torch.equal(q, q_ref) and torch.equal(sf, sf_ref)
