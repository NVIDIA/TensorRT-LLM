# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_mxfp8_linear_prequantized entry: the down projection on a producer's MXFP8.

Every cell runs the entry on the BF16 rows quantized the low-M GEMM's way, in the scale layout
``mxfp8_act_layout`` names, and checks it bit for bit against ``rubin_mxfp8_linear`` on the BF16
rows, and within the GEMM's rounding against the exact reference.

``mxfp8_quantizes_in_gemm`` (where a producer's MXFP8 form saves ``rubin_mxfp8_linear`` a
quantization kernel) is checked against the kernels that entry launches on BF16 rows of the TP8 KDA
stack, on both sides of the low-M kernel's rows and of the shared GEMM's.
"""

import importlib

import pytest
import torch
from _rubin_gemm_util import activation, generator, mxfp8_weight, requires_sm107

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.rubin_situ_and_mul_mxfp8 import (  # noqa: E501
    rubin_situ_and_mul_mxfp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (
    mxfp8_scales_to_r128c4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
    mxfp8_act_layout,
    mxfp8_quantizes_in_gemm,
    rubin_mxfp8_linear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
    rubin_mxfp8_linear_prequantized as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = requires_sm107

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _quantize(x: torch.Tensor, sf_rows: int, layout: str):
    """The low-M GEMM's MXFP8 of BF16 rows, scales in ``layout`` over ``sf_rows`` rows."""
    m, k = x.shape
    bits = x.view(torch.int16).to(torch.int32) & 0x7FFF
    amax = bits.view(m, k // 32, 32).amax(-1).clamp(min=0x38D2)
    exp = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).to(torch.int32)
    inv = ((254 - exp) << 7).to(torch.int16).view(torch.bfloat16)
    q = (x.view(m, k // 32, 32) * inv.unsqueeze(-1)).to(torch.float8_e4m3fn).view(m, k)
    rows = torch.full((sf_rows, k // 32), 127, dtype=torch.uint8, device=x.device)
    rows[:m] = exp.to(torch.uint8)
    tiles = rows.view(sf_rows // 128, 4, 32, k // 128, 4)
    order = (3, 0, 2, 1, 4) if layout == "mid" else (0, 3, 2, 1, 4)
    return q.contiguous(), tiles.permute(*order).reshape(-1).contiguous()


def _args(spec, gen):
    x = activation(spec["m"], spec["k"], gen)
    weight, weight_scale = mxfp8_weight(spec["n"], spec["k"], gen)
    act, act_scale = _quantize(x, *mxfp8_act_layout(spec["m"], spec["n"], spec["k"]))
    return x, (act, act_scale, weight, weight_scale)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, args = _args(cell.spec, generator(cell.why))
    out = op(*args)
    assert_within_ulp(out, op.reference(*args), element_ulp=2.0, rms_ulp=0.5)
    # The producer-fused path is the BF16 path minus its quantization kernel.
    assert torch.equal(out, rubin_mxfp8_linear(x, args[2], args[3]))


@pytest.mark.parametrize("zero_padding", [True, False])
@pytest.mark.parametrize("m", [1, 8, 100, 128, 300])
def test_situ_operand_runs_the_down_projection(m, zero_padding) -> None:
    """The shared experts' SiTU operand (R128c4) on the down projection: finite, within the GEMM's
    rounding of the exact product. Unset padding rows' scale bytes (NaN 0xFF here) stay out of it."""
    gen = generator(f"situ {m}")
    gu = activation(m, 2 * 768, gen)
    codes, scales = rubin_situ_and_mul_mxfp8(gu, 4.0, 25.0, zero_padding=zero_padding)
    if not zero_padding:
        pad = torch.ones(-(-m // 128) * 128, 768 // 32, dtype=torch.uint8, device=gu.device)
        pad[:m] = 0
        scales[mxfp8_scales_to_r128c4(pad).bool()] = 0xFF
    weight, weight_scale = mxfp8_weight(7168, 768, gen)
    out = op(codes, scales, weight, weight_scale)
    assert torch.isfinite(out).all()
    ref = op.reference(codes, scales, weight, weight_scale)
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


def test_a_replayed_graph_matches_eager() -> None:
    _, args = _args(dict(m=8, n=7168, k=768), generator("graph"))
    eager = op(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


# The TP8 KDA stack ([q | k | v | g | f_a | b], 6284 rows padded to 6288) and the rows up to which
# the low-M kernel quantizes it inside the GEMM on sm_107: helix8's attn_res_mx_pdl stores the
# stack's MXFP8 input past them.
_KDA_STACK = (6288, 7168)
_KDA_IN_GEMM_ROWS = 32
# The modules whose quantization launches the test records.
_LINEAR_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear"
)
_LOW_M_MODULE = "tensorrt_llm._torch.kernels.blackwell.fp8_block_scale_low_m_gemm"


@pytest.fixture(scope="module")
def kda_stack_weight() -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp8_weight(*_KDA_STACK, generator("kda stack"))


def _recording(launched: list, name: str, fn):
    def call(*args):
        launched.append(name)
        return fn(*args)

    return call


@pytest.mark.parametrize("m", [1, 8, 24, 32, 33, 36, 54, 64, 96, 128, 144, 160, 161])
def test_quantizes_in_gemm_names_the_bf16_entrys_dispatch(m, kda_stack_weight, monkeypatch) -> None:
    """True exactly where ``rubin_mxfp8_linear`` on BF16 rows launches no quantization kernel
    ahead of its GEMM (neither the mid-M GEMM's nor ``rubin_mxfp8_quantize``): for the TP8 KDA
    stack, up to the low-M kernel's 32 rows."""
    n, k = _KDA_STACK
    linear = importlib.import_module(_LINEAR_MODULE)
    low_m = importlib.import_module(_LOW_M_MODULE)
    launched: list[str] = []
    quantize = linear._quantize()
    monkeypatch.setattr(
        linear, "_quantize", lambda: _recording(launched, "rubin_mxfp8_quantize", quantize)
    )
    monkeypatch.setattr(
        low_m, "_compiled_quant", _recording(launched, "mid-M quantization", low_m._compiled_quant)
    )
    x = activation(m, k, generator(f"in gemm {m}"))
    out = rubin_mxfp8_linear(x, *kda_stack_weight)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert mxfp8_quantizes_in_gemm(m, n, k) == (not launched), launched
    assert mxfp8_quantizes_in_gemm(m, n, k) == (m <= _KDA_IN_GEMM_ROWS)
