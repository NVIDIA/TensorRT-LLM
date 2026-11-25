# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_hidden_add_rmsnorm entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_hidden_add as add_entry,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_hidden_add_rmsnorm as entry,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_hidden_rmsnorm as norm_entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

op = entry.cute_dsl_hidden_add_rmsnorm
HIDDEN = entry.HIDDEN
EPS = 1e-5


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(), add_entry.compile(), norm_entry.compile()


def _operands(cell, scale_b: float = 1.0):
    """`a`, `b` (bf16, column slices at the cell's pitches) and a norm weight; `b` at
    `scale_b` of `a`'s spread, so the sum is neither one operand nor flat."""
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows = cell.spec["rows"]

    def operand(pitch: int, std: float) -> torch.Tensor:
        wide = (torch.randn(rows, pitch, generator=g, device="cuda") * std).to(torch.bfloat16)
        return wide[:, pitch - HIDDEN :]

    a = operand(cell.spec["pitch_a"], 3.0)
    b = operand(cell.spec["pitch_b"], 3.0 * scale_b)
    weight = (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return a, b, weight


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """The norm kernel's bound against the fp64 reference of the norm of `torch.add(a, b)`:
    2 ulp per element against the row scale, 0.5 ulp relative RMS (its sum-of-squares order
    and approximate rsqrt). The operands are untouched."""
    certifying(op)
    fused = compiled[0]
    a, b, weight = _operands(cell)
    before = (a.clone(), b.clone())
    assert entry.takes(a, b, weight)
    out = op(a, b, weight, EPS, fused)
    assert out.is_contiguous() and out.shape == (cell.spec["rows"], HIDDEN)
    ref = op.reference(a, b, weight, EPS, fused)
    assert ref.float().abs().sum() > 0, "a zero reference proves nothing"
    assert float(ref.float().std()) > 0.1, "a flat reference proves nothing"
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)
    assert torch.equal(a, before[0]) and torch.equal(b, before[1]), "an operand was written"


@pytest.mark.parametrize("scale_b", (1.0, 1e-3, 1e3), ids=["even", "b_small", "b_large"])
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_bitwise_the_two_launches(cell, scale_b, compiled) -> None:
    """Bit for bit `cute_dsl_hidden_rmsnorm(cute_dsl_hidden_add(a, b))`, the pair the target
    ran before, at every cell and with one operand dominating either way (the sum's rounding
    then decides most bits): the fused path changes no byte of the MTP module's output."""
    fused, add, norm = compiled
    a, b, weight = _operands(cell, scale_b)
    want = norm_entry.cute_dsl_hidden_rmsnorm(
        add_entry.cute_dsl_hidden_add(a, b, add), weight, EPS, norm
    )
    assert float(want.float().std()) > 0.1, "a flat reference proves nothing"
    got = op(a, b, weight, EPS, fused)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_takes_refuses_what_the_kernel_cannot_read() -> None:
    w = torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert entry.takes(x, x, w)
    skew = torch.zeros(4, HIDDEN + 1, dtype=torch.bfloat16, device="cuda")[:, 1:]
    assert not entry.takes(skew, x, w), "a misaligned row must go to the two launches"
    assert not entry.takes(x, x[:3], w), "operands of two shapes"
    assert not entry.takes(x, x, w.float()), "an fp32 weight"
    assert not entry.takes(x, x, torch.ones(2 * HIDDEN, dtype=torch.bfloat16, device="cuda")[::2])
    assert not entry.takes(x.float(), x.float(), w), "fp32 rows"
