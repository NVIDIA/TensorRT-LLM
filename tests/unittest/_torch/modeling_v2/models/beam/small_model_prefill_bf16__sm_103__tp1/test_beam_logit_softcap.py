# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the six-block model's beam_logit_softcap entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.beam_logit_softcap import (  # noqa: E501
    beam_logit_softcap as op,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_logit_softcap requires a CUDA device"

#: Signed pre-softcap values from zero to far beyond saturation, -0 included.
SATURATION = (0.0, -0.0, 2.0**-10, -(2.0**-10), 0.5, -0.5, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0)
SATURATION += (29.0, -29.0, 30.0, -30.0, 60.0, -60.0, 150.0, -150.0, 300.0, -300.0, 1024.0, -1024.0)
SATURATION += (32768.0, -32768.0, 3.0e38, -3.0e38)


def _build(spec, seed):
    if spec.get("saturating"):
        return torch.tensor(SATURATION, dtype=torch.float32, device="cuda").to(spec["dtype"])[None]
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = 2.0 * torch.randn(spec["tokens"], spec["vocab"], generator=g, device="cuda")
    return logits.to(spec["dtype"])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Bitwise: the same three fp32 steps; finite, within the cap, fp32."""
    certifying(op)
    logits = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(logits, cell.spec["cap"])
    assert out.shape == logits.shape and out.dtype == torch.float32
    torch.testing.assert_close(out, op.reference(logits, cell.spec["cap"]), rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(out).all()) and float(out.abs().max()) <= cell.spec["cap"]


def test_saturation_and_signs() -> None:
    """Large magnitudes map to exactly +-30, signs are kept (-0 stays -0), and the result
    tracks the float64 expression within a few fp32 units of the cap."""
    logits = _build(dict(saturating=True, dtype=torch.bfloat16), 0)
    out = op(logits, 30.0)[0]
    ref = torch.tanh(logits[0].double() / 30.0) * 30.0
    assert float((out.double() - ref).abs().max()) <= 8 * 30.0 * 2.0**-24
    big = logits[0].float().abs() >= 300.0
    assert torch.equal(out[big].abs(), torch.full_like(out[big], 30.0))
    assert torch.equal(torch.signbit(out), torch.signbit(logits[0].float()))


def test_once_is_not_twice_or_none() -> None:
    """Applying the cap twice, or not at all, changes ordinary unsaturated logits."""
    logits = _build(dict(tokens=17, vocab=151936, dtype=torch.bfloat16), 5)
    once = op(logits, 30.0)
    assert not torch.equal(once, logits.float())
    assert not torch.equal(once, op(once.to(torch.bfloat16), 30.0))


@pytest.mark.parametrize(
    "case,match", [("int_cap", "positive float"), ("fp32_logits", "bf16 head output")]
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    logits = _build(dict(tokens=1, vocab=151936, dtype=torch.bfloat16), 9)
    cap = 30.0
    if case == "int_cap":
        cap = 30
    else:
        logits = logits.float()
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(logits, cap)
