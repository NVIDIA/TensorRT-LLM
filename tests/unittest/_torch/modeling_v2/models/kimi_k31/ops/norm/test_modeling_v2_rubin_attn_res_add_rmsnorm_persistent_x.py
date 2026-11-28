# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 rubin_attn_res_add_rmsnorm_persistent_x entry.

A cell is (N, form) at a generation step's rows above the PDL kernel's: DEP16's 96 (b12 k7, above
64) and helix8's 144 (b24 k5, above 128). N = K + 1 candidates and one of the flag sets the engine op
instantiates (helix8 calls all but the FP32 copies). Each sampled cell checks that each output is bit for bit
what today's calls make of the same inputs: rubin_attn_res_add_rmsnorm_persistent (after torch's bf16
add(s) of the MoE residual update), ``.float()`` and the catalog's mxfp8_quantize; without an addend,
the prefix itself comes back. Skipped where the CuTe DSL kernel does not run.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import mxfp8_quantize
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_add_rmsnorm_persistent import (  # noqa: E501
    rubin_attn_res_add_rmsnorm_persistent as today,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_add_rmsnorm_persistent_x import (  # noqa: E501
    persistent_x_available,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_add_rmsnorm_persistent_x import (  # noqa: E501
    rubin_attn_res_add_rmsnorm_persistent_x as op,
)

__extra_import_path__ = ["../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not persistent_x_available(),
    reason="certified on sm_107 (VR200), where the CuTe DSL kernel runs",
)

H = 7168


def _inputs(spec: dict, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    t, n = spec["tokens"], spec["candidates"]
    prefix = rn(t, H, sc=0.05)
    addend = rn(t, H, sc=0.05) if spec.get("addend") else None
    addend2 = rn(t, H, sc=0.05) if spec.get("addend2") else None
    block = rn(n - 1, t, H, sc=0.05)
    res_w = rn(H, sc=0.02)
    rms_w = (1 + rn(H, sc=0.02).float()).to(torch.bfloat16)
    out_w = (1 + rn(H, sc=0.02).float()).to(torch.bfloat16)
    return prefix, addend, addend2, block, res_w, rms_w, out_w


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _same(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.shape == expected.shape, (what, actual.shape, expected.shape)
    n = int((_bits(actual) != _bits(expected)).sum())
    assert n == 0, f"{what}: {n} of {actual.numel()} elements differ"


CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


@pytest.mark.parametrize(
    "cell", CELLS, ids=[f"{c.why.split()[0]} {c.why.split(', ', 2)[-1]}" for c in CELLS]
)
def test_certified_cells(cell) -> None:
    spec = cell.spec
    assert spec["hidden"] == H
    t, n, eps = spec["tokens"], spec["candidates"], spec["eps"]
    prefix, addend, addend2, block, res_w, rms_w, out_w = _inputs(
        spec, zlib.crc32(cell.why.encode()) % 2**31
    )
    flags = dict(
        f32=spec.get("f32", False), mx=spec.get("mx", False), grid_cap=spec.get("grid_cap", False)
    )
    got = op(prefix, addend, addend2, block, res_w, rms_w, out_w, eps, eps, **flags)

    # Today's calls: the MoE residual update as torch's two bf16 adds (rubin_moe_residual_add's contract), then
    # the persistent entry with the single addend in-kernel or none.
    if addend2 is not None:
        moe = (addend.float() + addend2.float()).to(torch.bfloat16)
        updated = (prefix.float() + moe.float()).to(torch.bfloat16)
        _, out = today(
            updated.view(t, 1, H), None, block.view(n - 1, t, 1, H), res_w, rms_w, out_w, eps, eps
        )
    else:
        updated, out = today(
            prefix.view(t, 1, H),
            None if addend is None else addend.view(t, 1, H),
            block.view(n - 1, t, 1, H),
            res_w,
            rms_w,
            out_w,
            eps,
            eps,
        )
    out = out.view(t, H)

    _same(got.output, out, "output vs today's persistent entry")
    if addend is None:
        assert got.updated.data_ptr() == prefix.data_ptr(), (
            "without an addend the prefix comes back"
        )
    else:
        _same(got.updated, updated.view(t, H), "updated vs today's add(s)")
    if flags["f32"]:
        _same(got.output_f32, got.output.float(), "FP32 copy vs output.float()")
    else:
        assert got.output_f32 is None
    if flags["mx"]:
        codes, scales = mxfp8_quantize(got.output, True, 32)
        _same(got.codes.view(torch.uint8), codes.view(torch.uint8), "MXFP8 codes vs mxfp8_quantize")
        _same(
            got.scales.view(torch.uint8).reshape(-1),
            scales.view(torch.uint8).reshape(-1),
            "MXFP8 scales",
        )
    else:
        assert got.codes is None and got.scales is None
