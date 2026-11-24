# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ihc_mapping entry: the fp32 composition against its float64 reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.mhc.ihc_mapping import (  # noqa: E501
    ihc_mapping as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "ihc_mapping requires a CUDA device"

CELLS = op.CELLS

#: `out` is bf16 rounded once from fp32.
_OUT_ELEMENT_ULP = 2.5
_OUT_RMS_ULP = 0.05
#: The gates are fp32: a 12288-term fp32 accumulation against float64, per half's row scale.
_GATE_ELEMENT_ULP = 320.0
_GATE_RMS_ULP = 16.0

# Checkpoint statistics: hc_base (mean, std) and hc_scale at layer-0 attn and the head.
CKPT_BASE_CONNECTION = (-1.60, 0.80)
CKPT_BASE_HEAD = (-1.10, 0.15)
CKPT_SCALE_CONNECTION = (1.52012, 0.0148)
CKPT_SCALE_HEAD = 0.158408


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, streams, hidden, groups = (
        spec["rows"],
        spec["streams"],
        spec["hidden"],
        spec["groups"],
    )
    gate_rows = groups * streams
    x = torch.randn(rows, streams, hidden, generator=g, device="cuda") * spec["stream_sigma"]
    x = x.to(spec["dtype"])
    fn = torch.randn(gate_rows, streams * hidden, generator=g, device="cuda") * spec["fn_rms"]
    if groups == 2:
        mean, std = CKPT_BASE_CONNECTION
        scale = CKPT_SCALE_CONNECTION
    else:
        mean, std = CKPT_BASE_HEAD
        scale = (CKPT_SCALE_HEAD,)
    base = torch.randn(gate_rows, generator=g, device="cuda") * std + mean
    return (
        x,
        fn.to(torch.float32),
        base.to(torch.float32),
        torch.tensor(scale, dtype=torch.float32, device="cuda"),
        spec["post_magnitude"],
        spec["eps"],
        spec["norm_eps"],
    )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out, gates = op(*args)
    streams = spec["streams"]
    assert out.shape == (spec["rows"], spec["hidden"]) and out.dtype == spec["dtype"]
    assert gates.shape == (spec["rows"], spec["groups"] * streams) and gates.dtype is torch.float32

    ref_out, ref_gates = op.reference(*args)
    assert_within_ulp(out, ref_out, element_ulp=_OUT_ELEMENT_ULP, rms_ulp=_OUT_RMS_ULP)
    # Pre and post halves each against their own row scale.
    for lo, hi in ((0, streams), (streams, gates.shape[-1])):
        if lo < hi:
            assert_within_ulp(
                gates[..., lo:hi],
                ref_gates[..., lo:hi],
                element_ulp=_GATE_ELEMENT_ULP,
                rms_ulp=_GATE_RMS_ULP,
            )
