# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4 unified target's cublas_mm variant at the vision tower (ops/gemm/cublas_mm__vision)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.gemm.cublas_mm__vision import (  # noqa: E501
    cublas_mm as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "cublas_mm requires a CUDA device"


#: Band per output dtype: (element_ulp, rms_ulp), in that dtype's own ulp.
_BAND = {torch.bfloat16: (4.0, 2.0), torch.float32: (1024.0, 512.0)}


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, in the output's ulp.

    Kernel and reference consume bit-identical operands and differ only in
    the order cuBLAS and torch sum over K, and in where the bias joins the
    sum before the one rounding. Over random operands a few outputs land
    near zero, where torch's per-element rtol is meaningless, so both gates
    are against the output's scale (`assert_within_ulp`).

    bf16 outputs: 4 ulp element-wise, 2 ulp relative RMS. Measured on GB300
    over the vision cells' six shapes at 256 to 32768 rows and three seeds
    each: at most 1.98 element-wise and 0.042 RMS, 99.9th percentile 0.033,
    cosine >= 0.9999998. The bias is a bf16 [N] row of unit scale
    against products of scale sqrt(K), so it is the smallest term a wrong
    epilogue could get wrong. A product missing it, or adding it twice (one
    bias per split-K slice), still reads at least 3.66 ulp RMS
    (`test_the_band_refuses_a_misapplied_bias`). A product missing one
    16-wide K block reads at least 15 RMS and 23.6 element-wise
    (`test_the_band_refuses_a_product_missing_one_k_block`).

    fp32 outputs (deepseek's router): 1024 fp32 ulp element-wise, 512 RMS. A
    band scored in bf16 ulp would admit a kernel that rounds the fp32 output
    to bf16 (`test_the_fp32_band_refuses_a_bf16_rounded_product`).
    """
    element_ulp, rms_ulp = _BAND[out.dtype]
    assert_within_ulp(out, ref, element_ulp=element_ulp, rms_ulp=rms_ulp, ulp=ULP[out.dtype])


def _build(spec, seed):
    """Materialize one cell.

    `mat_b` is built as [N, K] and transposed, because that is the one thing a
    spec cannot say: the op reads it column-major, and a stored weight's `.t()`
    already is that, which is why no transpose kernel runs in a target.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, n, dt = spec["m"], spec["k"], spec["n"], spec["dtype"]
    mat_a = torch.randn(m, k, generator=g, device="cuda").to(dt)
    mat_b = torch.randn(n, k, generator=g, device="cuda").to(dt).t()
    out_dtype = spec["out_dtype"]
    bias = None
    if spec["bias"]:
        bias = torch.randn(n, generator=g, device="cuda").to(out_dtype or dt)
    return mat_a, mat_b, bias, out_dtype


def _vision(m, k, n):
    return dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=True, out_dtype=None)


# The qwen4 unified target's vision tower (site `vision`), the tp1 VisionTower's
# biased projections: hidden 1152, 16 heads x 72, MLP 4304, and the patch merger
# over 2 x 2 merged patches (4 x 1152 = 4608) into the language hidden 2560. The
# tower runs eager and replicated on every rank, so these shapes are rank-
# invariant. Rows: 1024 and 16384 patches through the blocks, a quarter of that
# through the merger, plus the envelope's largest tower call (32768 patches,
# 8192 merged rows).
_QWEN4_UNIFIED_VISION = (
    Cell(why="qwen4 vision qkv, 1024 patches: 1152 -> 3 x 16 x 72", spec=_vision(1024, 1152, 3456)),
    Cell(why="qwen4 vision qkv, 16384 patches", spec=_vision(16384, 1152, 3456)),
    Cell(why="qwen4 vision proj, 1024 patches: 1152 -> 1152", spec=_vision(1024, 1152, 1152)),
    Cell(why="qwen4 vision proj, 16384 patches", spec=_vision(16384, 1152, 1152)),
    Cell(why="qwen4 vision fc1, 1024 patches: 1152 -> 4304", spec=_vision(1024, 1152, 4304)),
    Cell(why="qwen4 vision fc1, 16384 patches", spec=_vision(16384, 1152, 4304)),
    Cell(
        why="qwen4 vision fc1, 32768 patches: the envelope's largest tower call",
        spec=_vision(32768, 1152, 4304),
    ),
    Cell(why="qwen4 vision fc2, 1024 patches: 4304 -> 1152", spec=_vision(1024, 4304, 1152)),
    Cell(why="qwen4 vision fc2, 16384 patches", spec=_vision(16384, 4304, 1152)),
    Cell(
        why="qwen4 vision merger fc1, 256 merged rows: 4608 -> 4608", spec=_vision(256, 4608, 4608)
    ),
    Cell(why="qwen4 vision merger fc1, 4096 merged rows", spec=_vision(4096, 4608, 4608)),
    Cell(
        why="qwen4 vision merger fc1, 8192 merged rows: the envelope's largest merger call",
        spec=_vision(8192, 4608, 4608),
    ),
    Cell(
        why="qwen4 vision merger fc2, 256 merged rows: 4608 -> 2560", spec=_vision(256, 4608, 2560)
    ),
    Cell(why="qwen4 vision merger fc2, 4096 merged rows", spec=_vision(4096, 4608, 2560)),
)


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek q_a, no bias: hidden 7168 -> q_lora 1536",
            spec=dict(m=16, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek router: bf16 in, fp32 logits out, 256 experts",
            spec=dict(
                m=16, k=7168, n=256, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32
            ),
        ),
        Cell(
            why="deepseek default q_b at attention TP4: q_lora 1536 -> 32 heads x 192",
            spec=dict(m=16, k=1536, n=6144, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek default kv_b at attention TP8: kv_lora 512 -> 16 heads x 256",
            spec=dict(m=64, k=512, n=4096, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek default o_proj at attention TP8: 16 heads x 128 -> hidden 7168",
            spec=dict(m=16, k=2048, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        # The MTP layer's own GEMMs. Its eh_proj is replicated on every rank; its
        # shared expert is bf16 (the NVFP4 export skips layer 61), so it runs
        # through cublas_mm rather than nvfp4_gemm, split over attention TP like
        # the trunk's.
        Cell(
            why="deepseek MTP eh_proj: [enorm(e) | hnorm(h)] 2 x 7168 -> hidden 7168",
            spec=dict(m=16, k=14336, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert gate_up, whole under attention DP: 7168 -> 2 x 2048",
            spec=dict(m=16, k=7168, n=4096, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert gate_up at attention TP8: 7168 -> 2 x 256",
            spec=dict(m=16, k=7168, n=512, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek MTP shared expert down at attention TP8: 256 -> hidden 7168",
            spec=dict(m=16, k=256, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
    ),
    "gpt_oss/gpt_oss_120b__sm_10x__default": (
        Cell(
            why="gpt-oss qkv with fused bias: hidden 2880 -> (64 + 8 + 8) * 64",
            spec=dict(m=16, k=2880, n=5120, dtype=torch.bfloat16, bias=True, out_dtype=None),
        ),
    ),
    "qwen4_exp_for_conditional_generation/hf_fc694b5_orig__sm_103__unified": _QWEN4_UNIFIED_VISION,
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1])
    assert out.dtype is (out_dtype or mat_a.dtype)
    _compare(out, op.reference(mat_a, mat_b, bias, out_dtype))


def _cell(prefix: str) -> Cell:
    return next(c for c in CELLS if c.why.startswith(prefix))


# The cells where each control measured closest to the band: the merger's K of
# 4608 makes the products widest against a unit-scale bias.
_CONTROL_CELLS = [
    "qwen4 vision merger fc2, 4096 merged rows",
    "qwen4 vision fc2, 1024 patches",
    "qwen4 vision qkv, 1024 patches",
]


@pytest.mark.parametrize("prefix", _CONTROL_CELLS)
@pytest.mark.parametrize("factor", [0.0, 2.0], ids=["missing", "added_twice"])
def test_the_band_refuses_a_misapplied_bias(prefix, factor) -> None:
    """A product without its bias, or with it added twice (once per split-K
    slice), fails the band: the bias is not lost in the products' scale."""
    cell = _cell(prefix)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    wrong = op.reference(mat_a, mat_b, (bias.float() * factor).to(bias.dtype), out_dtype)
    with pytest.raises(AssertionError):
        _compare(out, wrong)


@pytest.mark.parametrize("prefix", _CONTROL_CELLS)
def test_the_band_refuses_a_product_missing_one_k_block(prefix) -> None:
    """A wrong product misses the band by a margin rather than squeaking past."""
    cell = _cell(prefix)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    k = cell.spec["k"] - 16
    wrong = op.reference(mat_a[:, :k], mat_b[:k], bias, out_dtype)
    with pytest.raises(AssertionError):
        _compare(out, wrong)


def test_the_fp32_band_refuses_a_bf16_rounded_product() -> None:
    """The deepseek router asks for fp32; an output rounded to bf16 fails."""
    cell = _cell("deepseek router")
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(mat_a, mat_b, bias, out_dtype)
    assert ref.dtype is torch.float32
    with pytest.raises(AssertionError):
        _compare(ref.to(torch.bfloat16).float(), ref)


def _ok(dtype=torch.bfloat16):
    a = torch.randn(8, 64, dtype=dtype, device="cuda")
    b = torch.randn(32, 64, dtype=dtype, device="cuda").t()
    return a, b


def test_guard_refuses_a_row_major_mat_b() -> None:
    """Read column-major regardless; a row-major b is a different product."""
    a, _ = _ok()
    b = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="column-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_strided_mat_a() -> None:
    a = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")[:, :64]
    _, b = _ok()
    with pytest.raises(AssertionError, match="row-major"):
        with validating(op):
            op(a, b)


def test_guard_refuses_a_bias_in_the_wrong_dtype() -> None:
    """Accepted by the op, and the result is silently wrong."""
    a, b = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="bias dtype"):
        with validating(op):
            op(a, b, bias)


def test_guard_refuses_a_bias_on_fp32_inputs() -> None:
    """The op takes it and never adds it."""
    a, b = _ok(torch.float32)
    bias = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="silently ignored"):
        with validating(op):
            op(a, b, bias)
