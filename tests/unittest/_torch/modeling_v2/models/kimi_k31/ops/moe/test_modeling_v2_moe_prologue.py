# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused routing + MXFP8 prologue and its routing and quantization entries: a sample of
each entry's cells bitwise against the separated routing and quantization it replaces."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    k31_moe_quant,
    k31_moe_route,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import (  # noqa: E501
    k31_moe_prologue as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("moe_prologue runs a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "moe_prologue requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "moe_prologue is certified on SM 10.3, where the MXFP4 targets call it",
        allow_module_level=True,
    )

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = spec["rows"]
    logits = torch.randn(m, NUM_EXPERTS, generator=g, device="cuda", dtype=torch.float32)
    bias = torch.randn(NUM_EXPERTS, generator=g, device="cuda", dtype=torch.float32) * 0.1
    x = (torch.randn(m, HIDDEN, generator=g, device="cuda") * 3).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero block: scale byte 0, zero codes
    if m > 1:
        x[1, 32:64] *= 1e4  # a block near the E4M3 range edge
    if spec.get("runner_weights"):
        return logits, bias, spec["scale"], x, True
    return logits, bias, spec["scale"], x


def _same(out, ref):
    for name, a, b in zip(("ids", "weights", "x_fp8", "x_sf"), out, ref):
        assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (
            f"{name} differs from the reference"
        )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    _same(out, op.reference(*args))


ROUTE_CELLS = sample_cells(k31_moe_route.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", ROUTE_CELLS, ids=[c.why[:44] for c in ROUTE_CELLS])
def test_route_cells(cell) -> None:
    spec = cell.spec
    logits, bias, scale, x = _build(
        dict(rows=spec["rows"], scale=spec["scale"]), seed=spec["rows"] * 11
    )[:4]
    args = (logits, bias, scale, spec["runner_weights"])
    _same(k31_moe_route(*args), k31_moe_route.reference(*args))


QUANT_CELLS = sample_cells(k31_moe_quant.CELLS, size_key="rows")[:1]


@pytest.mark.parametrize("cell", QUANT_CELLS, ids=[c.why[:44] for c in QUANT_CELLS])
def test_quant_cells(cell) -> None:
    x = _build(dict(rows=cell.spec["rows"], scale=1.0), seed=cell.spec["rows"] * 13)[3]
    _same(k31_moe_quant(x), k31_moe_quant.reference(x))
