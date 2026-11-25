# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kernels/vae_decoder_fp8.py``: the tiled quantize kernels of the FP8 VAE decoder.

The module's contract: the tiled SwiGLU-output and attention-output quantizes reproduce what
``torch.compile`` (Inductor) generates for the untiled path: SiLU as ``g / (1 + exp(-g))`` in fp32,
the fp32 product, the amax stored as fp16, the cast's scale from the fp16 amax, the GEMM's scale
from the fp32 amax. So the e4m3 codes, the GEMM inputs and the decoder's fp16 outputs are
bit-identical to the untiled compiled path. The reference here is that compiled path, built from
the module's own ``Fp8Linear`` (the untiled class) and the SwiGLU of the decoder's feed-forward;
the tiled modules must give the same fp16 bits. Tiles are pinned per case through ``_KN`` with
``monkeypatch``; the process scenario configuration is never re-selected.
"""

import pytest
import torch
from torch import nn
from utils.util import skip_pre_ada

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
    vae_decoder_fp8 as vd,
)

pytestmark = [
    skip_pre_ada,
    pytest.mark.skipif(not vd._TILED, reason="the scenario's vae_fp8_tiled is off: no tiled ops"),
]

C, HEADS, HEAD_DIM, HIDDEN = 1024, 16, 64, 4096
T = 2048
MLP_TILES = ((16, 1024, 8), (8, 512, 4))  # the scenario default and the module default


def _linear(fan_in: int, fan_out: int, seed: int) -> nn.Linear:
    torch.manual_seed(seed)
    lin = nn.Linear(fan_in, fan_out, bias=True).to(device="cuda", dtype=torch.float16)
    return lin


def _x(shape, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
    x.view(-1)[:7] *= 30.0  # a few outliers so the amax is not a typical value
    return x.to(torch.float16)


def _swiglu(h: torch.Tensor) -> torch.Tensor:
    # diffusers' SwiGLU after its projection: hidden * silu(gate), gate in the second half
    hidden, gate = h.chunk(2, dim=-1)
    return hidden * torch.nn.functional.silu(gate)


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.dtype == expected.dtype == torch.float16 and actual.shape == expected.shape, what
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), what


@pytest.mark.parametrize("fused", (True, False), ids=("amax2_cast2", "amax_cast"))
@pytest.mark.parametrize("tile", MLP_TILES, ids=lambda t: "x".join(map(str, t)))
def test_swiglu_down_matches_compiled_untiled(monkeypatch, tile, fused):
    monkeypatch.setattr(vd, "_FUSED", fused)
    for key, value in zip(("mlp_bt", "mlp_bc", "mlp_warps"), tile):
        monkeypatch.setitem(vd._KN, key, value)
    down = vd.Fp8Linear(_linear(HIDDEN, C, seed=2))
    # The proj GEMM's fp16 output, fed to both sides: only the SwiGLU quantize and the down GEMM differ.
    h = _x((1, T, 2 * HIDDEN), seed=3)
    tiled = vd.Fp8SwiGLUDown(nn.Identity(), down)
    reference = torch.compile(lambda t: down(_swiglu(t)), fullgraph=True, dynamic=False)
    with torch.no_grad():
        _assert_same_bits(tiled(h), reference(h), f"swiglu-down tile={tile} fused={fused}")


@pytest.mark.parametrize("layout", ("bthd_view", "flat"))
def test_to_out_matches_compiled_untiled(layout):
    """``Fp8LinearTiled`` on the attention output (the strided ``[B, T, H, D]`` view or flat)."""
    lin = vd.Fp8Linear(_linear(C, C, seed=4))
    attn = _x((1, HEADS, T, HEAD_DIM), seed=5).transpose(1, 2)  # [B, T, H, D], non-contiguous in T
    x = attn if layout == "bthd_view" else attn.reshape(1, T, C).contiguous()
    tiled = vd.Fp8LinearTiled(lin)
    reference = torch.compile(lambda t: lin(t.reshape(1, T, C)), fullgraph=True, dynamic=False)
    with torch.no_grad():
        _assert_same_bits(tiled(x), reference(x), f"to_out {layout}")
