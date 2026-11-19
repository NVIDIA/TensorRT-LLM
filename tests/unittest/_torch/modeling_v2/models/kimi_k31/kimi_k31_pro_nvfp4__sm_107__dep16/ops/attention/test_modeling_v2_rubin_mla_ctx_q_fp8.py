# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_mla_ctx_q_fp8, the FP8 latent columns of the padded sparse MLA prefill query.

GPU (sm_107): the 16k prefill chunk's cell bit for bit against the reference (the attention op's conversion of a
BF16 query: unit scale, nearest even, saturated to +-448), with special values spread over the real heads, written
into the latent columns of a full FP8 query buffer whose RoPE columns must stay as they were.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__dep16.ops.attention.rubin_mla_ctx_q_fp8 import (  # noqa: E501
    rubin_mla_ctx_q_fp8 as op,
)

__extra_import_path__ = ["../../../../..", "../../../ops/attention", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

on_sm107 = on_sm(107)
# The largest cell of the sample: a DEP16 prefill chunk of 16k tokens.
_CELLS = sample_cells(op.CELLS, size_key="n")[-1:]

FP8 = torch.float8_e4m3fn

# BF16 inputs whose E4M3 conversion at unit scale is a corner: signed zeros, saturation to +-448,
# infinities, subnormals, a tie to the even code.
_SPECIAL = (
    0.0,
    -0.0,
    1.0,
    -1.5,
    448.0,
    -448.0,
    480.0,  # saturated, not NaN
    1.0e4,
    -3.0e38,
    float("inf"),
    float("-inf"),
    2.0**-9,  # the smallest subnormal
    2.0**-11,  # below half of it: zero
    0.3125,
    0.328125,  # halfway between 0.3125 and 0.34375: to the even code
)


def _inputs(spec, gen):
    """The BMM's BF16 buffer [n, pad, width] (NaN beyond the real heads' latent columns) with specials spread
    over the real heads, and an FP8 query buffer [n, pad, width] holding a sentinel everywhere."""
    n, pad, heads, d, width = spec["n"], spec["pad"], spec["heads"], spec["d"], spec["width"]
    src = torch.randn(n, pad, width, generator=gen, device="cuda").mul_(96.0).to(torch.bfloat16)
    vals = torch.tensor(_SPECIAL, dtype=torch.float32).to(torch.bfloat16)
    real = src[:, :heads, :d].reshape(-1)
    pos = torch.randint(0, real.numel(), (min(4096, real.numel()),), generator=gen, device="cuda")
    real[pos] = vals.cuda()[torch.arange(pos.numel(), device="cuda") % vals.numel()]
    src[:, :heads, :d] = real.view(n, heads, d)
    src[:, heads:] = float("nan")
    src[:, :heads, d:] = float("nan")
    return src, torch.full((n, pad, width), 0x5A, dtype=torch.uint8, device="cuda").view(FP8)


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    src, quant_q = _inputs(s, torch.Generator(device="cuda").manual_seed(s["n"]))
    d = s["d"]
    ref = op.reference(quant_q[..., :d], src[..., :d], s["heads"])
    op(quant_q[..., :d], src[..., :d], s["heads"])
    assert torch.equal(quant_q[..., :d].view(torch.uint8), ref.view(torch.uint8))
    assert (quant_q[..., d:].view(torch.uint8) == 0x5A).all(), "the RoPE columns were written"
