# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_kda_mtp_verify entry (sm_107).

A sample of the certified cells runs the op (through its guard) and the entry's plain-torch reference on
identical inputs in the target's runtime layouts and compares the output (BF16, or the MXFP8 codes
dequantized with their scales) and every tensor the call writes in place.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (
    dequantize_mxfp8,
    mxfp8_scales_from_r128c4,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    rubin_kda_mtp_verify as op,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _rubin_kda_verify_inputs import (  # noqa: E402 — needs the path declared above
    BF16_ATOL,
    BF16_RTOL,
    check_bf16_output,
    check_writes,
    clone_mutated,
    verify_args,
)
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(107)

#: Per output form (BF16, MXFP8): a boundary and a typical cell of ``op.CELLS``.
_CELLS = tuple(
    cell
    for out in ("bf16", "mxfp8")
    for cell in sample_cells([c for c in op.CELLS if c.spec["out"] == out], size_key="N")[:2]
)


def check_output(out, ref, spec) -> None:
    if spec["out"] == "mxfp8":
        # One scale per 32-value group (the catalog 1x32 form).
        rows, k = out[0].shape
        got_exp = mxfp8_scales_from_r128c4(out[1], rows, k).to(torch.int32)
        want_exp = mxfp8_scales_from_r128c4(ref[1], rows, k).to(torch.int32)
        # The bf16 values agree far within a factor of two, so a scale may land one binade apart
        # where a group's amax sits at 448 * 2^k.
        assert (got_exp - want_exp).abs().max().item() <= 1, "scale exponents"
        got, want = dequantize_mxfp8(*out), dequantize_mxfp8(*ref)
        # The kernel's codes are the e4m3 rounding of its bf16 output, the reference's the same
        # rounding of its bf16 values. Those agree
        # within the BF16 band (|x| <= top + step), and each rounding moves a value by at most half
        # an e4m3 step at the larger magnitude top: 2^(floor(log2 top) - 3), or 2^-9 of the scale
        # below the normal range. So a rounding boundary between the two values puts the codes one
        # step apart: up to 2^-3 of the group amax in its top binade (2^-4 bounds one rounding).
        exp = torch.maximum(got_exp, want_exp).repeat_interleave(32, dim=1)
        scale = torch.pow(2.0, exp - 127.0)
        top = torch.maximum(got.abs(), want.abs())
        step = torch.maximum(torch.pow(2.0, torch.frexp(top).exponent - 4.0), scale * 2.0**-9)
        band = step + BF16_ATOL + BF16_RTOL * (top + step)
        diff = (got - want).abs()
        assert bool((diff <= band).all()), f"{(diff / band).max().item():.2f} of the band"
        return
    check_bf16_output(out, ref)


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:44] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    op_args = verify_args(cell.spec, seed)
    ref_args = clone_mutated(op_args)
    used = set(op_args["ssm_state_indices"].tolist())
    untouched = [s for s in range(op_args["recurrent_state"].shape[0]) if s not in used]
    before = op_args["recurrent_state"][untouched].clone()
    out = op(**op_args)
    ref = op.reference(**ref_args)
    torch.cuda.synchronize()
    check_output(out, ref, cell.spec)
    check_writes(op_args, ref_args, untouched, before)
