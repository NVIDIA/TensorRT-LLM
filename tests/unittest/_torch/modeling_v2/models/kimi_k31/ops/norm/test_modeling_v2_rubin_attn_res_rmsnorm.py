# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the K3.1 Rubin attn_res entries of one kernel (``kernels/norm/rubin_attn_res.py``).

rubin_attn_res_rmsnorm and rubin_attn_res_add_rmsnorm, a sample of the cells: the output against the
entry's fp32 reference (cosine / relative L2) and the updated prefix bit for bit against a torch bf16
add.

rubin_attn_res_moe_add_rmsnorm_mx, a sample of the cells: the deferred MoE residual update bit for bit against
two bf16 adds, the plain rubin_attn_res_moe_add_rmsnorm's bf16 bits, the output against the fp32
reference, and the MXFP8 form of its output bit for bit (atom bytes past the rows are not written, so
not compared): the catalog ``mxfp8_quantize``'s codes and swizzled scales of the output.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import (  # noqa: E501
    mxfp8_quantize,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm import _rubin_norm_common as common
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_moe_add_rmsnorm import (  # noqa: E501
    rubin_attn_res_moe_add_rmsnorm,
    rubin_attn_res_moe_add_rmsnorm_mx,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_rmsnorm import (  # noqa: E501
    rubin_attn_res_add_rmsnorm,
    rubin_attn_res_rmsnorm,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(107)

H = common.HIDDEN
EPS = common.EPS
# Against the fp32 reference: the kernel rounds where the reference rounds; fast-math exp2 /
# rsqrt and its reduction order remain (the C++ ops measured <= 6e-6 relative L2 there). A
# dropped candidate moves the output by tens of percent.
MIN_COSINE, MAX_RELATIVE_L2 = 0.9999, 1e-3


def _inputs(rows: int, candidates: int, seed: int, scale: float = 1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return dict(
        layer=rn(rows, H, sc=scale),
        addend=rn(rows, H, sc=0.5 * scale),
        block=rn(candidates - 1, rows, H, sc=scale),
        res_w=rn(H, sc=0.05),
        rms_w=(1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


def _assert_reference_close(out: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    a, e = out.float().flatten(), ref.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    rel = ((a - e).norm() / (e.norm() + 1e-12)).item()
    assert cosine > MIN_COSINE and rel < MAX_RELATIVE_L2, f"{what}: cos {cosine:.7f} rel {rel:.2e}"


def _eps(spec) -> tuple[float, float]:
    return spec.get("eps", EPS), spec.get("out_eps", EPS)


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


RMSNORM_CELLS = sample_cells(rubin_attn_res_rmsnorm.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", RMSNORM_CELLS, ids=[c.why[:56] for c in RMSNORM_CELLS])
def test_rmsnorm_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(rubin_attn_res_rmsnorm)
    spec = cell.spec
    t = _inputs(spec["rows"], spec["candidates"], _seed(cell))
    args = (t["layer"], t["block"], t["res_w"], t["rms_w"], t["out_w"], *_eps(spec))
    out = rubin_attn_res_rmsnorm(*args, early_trigger=spec["early_trigger"])
    ref = rubin_attn_res_rmsnorm.reference(*args, early_trigger=spec["early_trigger"])
    _assert_reference_close(out, ref, "output")


ADD_RMSNORM_CELLS = sample_cells(rubin_attn_res_add_rmsnorm.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize(
    "cell",
    ADD_RMSNORM_CELLS,
    ids=[c.why[:56] for c in ADD_RMSNORM_CELLS],
)
def test_add_rmsnorm_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(rubin_attn_res_add_rmsnorm)
    spec = cell.spec
    t = _inputs(spec["rows"], spec["candidates"], _seed(cell))
    args = (t["layer"], t["addend"], t["block"], t["res_w"], t["rms_w"], t["out_w"], *_eps(spec))
    updated, out = rubin_attn_res_add_rmsnorm(*args, early_trigger=spec["early_trigger"])
    _, ref_out = rubin_attn_res_add_rmsnorm.reference(*args, early_trigger=spec["early_trigger"])
    assert torch.equal(_bits(updated), _bits(t["layer"] + t["addend"])), "not a torch bf16 add"
    _assert_reference_close(out, ref_out, "output")


def _moe_inputs(rows: int, candidates: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    residual = rn(rows, H)
    residual[0, :64] = 0  # an all-zero 1x32 group of the update (the amax floor)
    return (
        residual,
        rn(rows, H, sc=0.5),  # routed
        rn(rows, H, sc=0.5),  # shared
        rn(candidates - 1, rows, H),  # block
        rn(H, sc=0.05),  # res_weight
        (1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),  # rms_weight
        (1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),  # out_weight
        EPS,
        EPS,
    )


MOE_ADD_MX_CELLS = sample_cells(rubin_attn_res_moe_add_rmsnorm_mx.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize(
    "cell",
    MOE_ADD_MX_CELLS,
    ids=[c.why[:56] for c in MOE_ADD_MX_CELLS],
)
def test_moe_add_mx_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(rubin_attn_res_moe_add_rmsnorm_mx)
    spec = cell.spec
    rows = spec["rows"]
    args = _moe_inputs(rows, spec["candidates"], _seed(cell))
    kw = dict(early_trigger=spec["early_trigger"])
    # Groups an amax floor would change: one all zero, one tiny.
    args[6][64:96] = 0
    args[6][96:128] = 1e-6
    got = rubin_attn_res_moe_add_rmsnorm_mx(*args, **kw)
    plain = rubin_attn_res_moe_add_rmsnorm(*args, **kw)
    ref = rubin_attn_res_moe_add_rmsnorm_mx.reference(*args, **kw)
    for i in range(2):
        assert torch.equal(_bits(got[i]), _bits(plain[i])), f"bf16 output {i}"
    residual, routed, shared = args[:3]
    assert torch.equal(_bits(got[0]), _bits(residual + (routed + shared))), "not two bf16 adds"
    _assert_reference_close(got[1], ref[1], "output")
    # The quantizer the dense MXFP8 GEMM's caller runs today: the same codes, and the same
    # swizzled scale bytes for the M rows (it also zeroes the tile's rows past M).
    cq, cs = mxfp8_quantize(got[1], True, 32)
    assert torch.equal(_bits(got[2]), _bits(cq)), "codes against mxfp8_quantize"
    # Only the atom bytes of the M rows are written.
    row = torch.arange(rows, device="cuda").view(rows, 1)
    grp = torch.arange(H // 32, device="cuda").view(1, -1)
    off = ((grp // 4) * 512 + (row % 32) * 16 + (row % 128 // 32) * 4 + grp % 4).reshape(-1)
    assert tuple(got[3].shape) == ((H // 128) * 512,)
    assert torch.equal(got[3][off], cs.view(torch.uint8)[off]), "scales against mxfp8_quantize"
