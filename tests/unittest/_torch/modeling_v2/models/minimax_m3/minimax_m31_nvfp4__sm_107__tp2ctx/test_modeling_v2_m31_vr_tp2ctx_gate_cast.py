# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_GATE_CAST (CTX's ROUTER_CAST item): the router input's bf16 -> fp32 cast through one vectorized Triton pass.

bf16 -> fp32 widening is exact, so the cast must equal ``.to(torch.float32)`` bit for bit on every input, and the gate's
logits must too: random rows plus every adversarial bf16 pattern (+-0, +-inf, quiet / signalling NaN of both signs,
subnormals, the extreme normals). Inside CUDA-graph capture the call keeps the aten cast, so eager and graph agree.
"""

import importlib

import pytest
import torch

_MOD = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx.modeling"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _special_bits():
    # bf16 bit patterns: +-0, +-inf, NaNs (quiet / signalling, both signs), subnormals, smallest / largest normals.
    return (
        torch.tensor(
            [
                0x0000,
                0x8000,
                0x7F80,
                0xFF80,
                0x7FC0,
                0xFFC0,
                0x7F81,
                0xFF81,
                0x0001,
                0x807F,
                0x0080,
                0x7F7F,
                0xFF7F,
            ],
            dtype=torch.int32,
        )
        .to(torch.int16)
        .view(torch.bfloat16)
    )  # int32 literals, wrapped to the int16 bit patterns


def _rows(rows, cols, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=gen).to(
        torch.bfloat16
    )
    flat = x.view(-1)
    specials = _special_bits().cuda()
    idx = torch.randint(0, flat.numel(), (4096,), device="cuda", generator=gen)
    flat[idx] = specials[torch.arange(4096, device="cuda") % specials.numel()]
    return x


@pytest.mark.parametrize("rows", [1024, 1160, 2049, 8192, 16384])
def test_the_cast_and_the_router_logits_are_bitwise_aten(rows):
    mod = importlib.import_module(_MOD)
    x = _rows(rows, 6144, rows)
    ref = x.to(torch.float32)
    got = mod._router_input_fp32(x)
    assert got.dtype == torch.float32 and got.shape == ref.shape
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
    w = torch.randn(128, 6144, device="cuda", dtype=torch.float32)
    assert torch.equal(
        torch.nn.functional.linear(got, w).view(torch.int32),
        torch.nn.functional.linear(ref, w).view(torch.int32),
    )


def test_small_noncontiguous_and_captured_inputs_keep_the_aten_cast():
    mod = importlib.import_module(_MOD)
    small = _rows(1023, 6144, 1)
    assert torch.equal(
        mod._router_input_fp32(small).view(torch.int32), small.to(torch.float32).view(torch.int32)
    )
    strided = _rows(4096, 12288, 2)[:, ::2]
    assert not strided.is_contiguous()
    assert torch.equal(
        mod._router_input_fp32(strided).view(torch.int32),
        strided.to(torch.float32).view(torch.int32),
    )
    x = _rows(4096, 6144, 3)
    eager = mod._router_input_fp32(x)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod._router_input_fp32(x)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        graphed = mod._router_input_fp32(x)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(graphed.view(torch.int32), eager.view(torch.int32))


def test_wiring_kill_switch_and_engagement():
    from pathlib import Path

    src = Path(importlib.import_module(_MOD).__file__).read_text()
    assert '_ROUTER_CAST = not m31_disabled("TP2CTX_GATE_CAST")' in src
    assert (
        "return torch.nn.functional.linear(_router_input_fp32(hidden_states), self.weight)" in src
    )
    assert "TP2CTX_GATE_CAST {path} router input cast taken" in src
    assert 'getattr(torch.ops.trtllm, "minimax_m31_bf16_to_fp32", None)' in src


@pytest.mark.parametrize("numel_rows", [1, 7, 1160, 4097])
def test_the_compiled_op_is_bitwise_aten(numel_rows):
    """The production op (USER 10-24 kernel rule), when the image builds it: aligned, odd and tail sizes, specials."""
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

    op = getattr(torch.ops.trtllm, "minimax_m31_bf16_to_fp32", None)
    if op is None:
        pytest.skip("this image does not build minimax_m31_bf16_to_fp32")
    x = _rows(numel_rows, 6144, 7)
    assert torch.equal(op(x).view(torch.int32), x.to(torch.float32).view(torch.int32))
    odd = _rows(1, 6151, 8).view(-1)[1:]  # misaligned base and an odd length: the scalar path
    odd = odd.contiguous() if not odd.is_contiguous() else odd
    assert torch.equal(op(odd).view(torch.int32), odd.to(torch.float32).view(torch.int32))
