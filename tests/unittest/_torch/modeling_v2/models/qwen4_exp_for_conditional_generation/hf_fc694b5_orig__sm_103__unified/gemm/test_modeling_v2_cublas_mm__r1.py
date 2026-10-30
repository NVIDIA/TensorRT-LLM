# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4 unified target's cublas_mm variant (ops/gemm/cublas_mm__r1)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.gemm.cublas_mm__r1 import (  # noqa: E501
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
    the order cuBLAS and torch sum over K. Over random operands a few
    outputs land near zero, where torch's per-element rtol is meaningless, so
    both gates are against the row scale (`assert_within_ulp`).

    bf16 outputs: 4 ulp element-wise, 2 ulp relative RMS. Measured on GB300
    over every cell here: at most 1.98 element-wise -- one rounding flip on an
    element at the bottom of the row maximum's binade, where one bf16 step is
    two units of ULP's 2^-8 -- and at most 0.048 RMS. A product missing one
    16-wide K block reads at least 11.2 element-wise and 8.5 RMS
    (`test_the_band_refuses_a_product_missing_one_k_block`).

    fp32 outputs (the qwen4 HC first projection, deepseek's router): 1024 fp32
    ulp element-wise, 512 RMS. Measured at most 192 and 97 (HC, 8192 rows).
    The same product rounded through bf16 reads at least 24653 and 13836 fp32
    ulp, so this band refuses a kernel that rounds the fp32 output to bf16 --
    which a band scored in bf16 ulp admits: that rounding is under 2 bf16 ulp
    of the row scale (`test_the_fp32_band_refuses_a_bf16_rounded_product`).
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


def _qwen4(m, k, n, out_dtype=None):
    return dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=False, out_dtype=out_dtype)


# The qwen4 unified target (tp4 / tep4 / dep4, hidden 2560) calls this entry for
# every dense bf16 projection on this rank's rows. Per rank: tp4/tep4 hold 6 q / 1
# kv attention heads and 4 key / 12 value GDN heads; dep4 holds tp1's full widths.
# Rows: decode 1..128 at tp4/tep4 and 1..32 per rank at dep4, 1024 at an MTP-7
# verify (128 x 8), 8192 for one prefill chunk at max_num_tokens, and up to 4 x
# 8192 at dep4's router, which routes the allgathered rows.
_QWEN4_UNIFIED = (
    Cell(
        why="qwen4 tp4/tep4 attn in_proj, 1 row: 2560 -> fused row 4352",
        spec=_qwen4(1, 2560, 4352),
    ),
    Cell(
        why="qwen4 tp4/tep4 attn in_proj, 128 rows: decode at max concurrency",
        spec=_qwen4(128, 2560, 4352),
    ),
    Cell(
        why="qwen4 tp4/tep4 attn in_proj, 1024 rows: an MTP-7 verify, 128 x 8",
        spec=_qwen4(1024, 2560, 4352),
    ),
    Cell(
        why="qwen4 tp4/tep4 attn in_proj, 8192 rows: one prefill chunk",
        spec=_qwen4(8192, 2560, 4352),
    ),
    Cell(
        why="qwen4 dep4 attn in_proj, 1 row: 2560 -> fused row 14080",
        spec=_qwen4(1, 2560, 14080),
    ),
    Cell(
        why="qwen4 dep4 attn in_proj, 32 rows: the per-rank decode cap",
        spec=_qwen4(32, 2560, 14080),
    ),
    Cell(
        why="qwen4 dep4 attn in_proj, 8192 rows: one prefill chunk",
        spec=_qwen4(8192, 2560, 14080),
    ),
    Cell(
        why="qwen4 tp4/tep4 o_proj | GDN out_proj, 1 row: 1536 -> 2560",
        spec=_qwen4(1, 1536, 2560),
    ),
    Cell(
        why="qwen4 tp4/tep4 o_proj | GDN out_proj, 128 rows",
        spec=_qwen4(128, 1536, 2560),
    ),
    Cell(
        why="qwen4 tp4/tep4 o_proj | GDN out_proj, 8192 rows: prefill chunk",
        spec=_qwen4(8192, 1536, 2560),
    ),
    Cell(
        why="qwen4 dep4 o_proj | GDN out_proj, 32 rows: 6144 -> 2560",
        spec=_qwen4(32, 6144, 2560),
    ),
    Cell(
        why="qwen4 dep4 o_proj | GDN out_proj, 8192 rows: prefill chunk",
        spec=_qwen4(8192, 6144, 2560),
    ),
    Cell(
        why="qwen4 tp4/tep4 GDN in_proj, 1 row: 2560 -> 4120",
        spec=_qwen4(1, 2560, 4120),
    ),
    Cell(
        why="qwen4 tp4/tep4 GDN in_proj, 128 rows",
        spec=_qwen4(128, 2560, 4120),
    ),
    Cell(
        why="qwen4 tp4/tep4 GDN in_proj, 8192 rows: prefill chunk",
        spec=_qwen4(8192, 2560, 4120),
    ),
    Cell(
        why="qwen4 dep4 GDN in_proj, 32 rows: 2560 -> 16480",
        spec=_qwen4(32, 2560, 16480),
    ),
    Cell(
        why="qwen4 dep4 GDN in_proj, 8192 rows: prefill chunk",
        spec=_qwen4(8192, 2560, 16480),
    ),
    Cell(
        why="qwen4 shared expert gate|up|logit, 1 row: 2560 -> 1288",
        spec=_qwen4(1, 2560, 1288),
    ),
    Cell(
        why="qwen4 shared expert gate|up|logit, 8192 rows",
        spec=_qwen4(8192, 2560, 1288),
    ),
    Cell(
        why="qwen4 shared expert down, 1 row: 640 -> 2560",
        spec=_qwen4(1, 640, 2560),
    ),
    Cell(
        why="qwen4 shared expert down, 8192 rows",
        spec=_qwen4(8192, 640, 2560),
    ),
    Cell(
        why="qwen4 router, 1 row: 2560 -> 512 experts",
        spec=_qwen4(1, 2560, 512),
    ),
    Cell(
        why="qwen4 router, 128 rows: tp4/tep4 decode, dep4 gathered decode",
        spec=_qwen4(128, 2560, 512),
    ),
    Cell(
        why="qwen4 router, 8192 rows: one prefill chunk",
        spec=_qwen4(8192, 2560, 512),
    ),
    Cell(
        why="qwen4 router, 32768 rows: dep4 routes 4 x 8192 gathered rows",
        spec=_qwen4(32768, 2560, 512),
    ),
    Cell(
        why="qwen4 HC first projection, 1 row: 4 x 2560 -> 352, fp32 out",
        spec=_qwen4(1, 10240, 352, torch.float32),
    ),
    Cell(
        why="qwen4 HC first projection, 128 rows, fp32 out",
        spec=_qwen4(128, 10240, 352, torch.float32),
    ),
    Cell(
        why="qwen4 HC first projection, 8192 rows, fp32 out",
        spec=_qwen4(8192, 10240, 352, torch.float32),
    ),
    Cell(
        why="qwen4 PLE key projection, 1 row: 2560 -> 10240",
        spec=_qwen4(1, 2560, 10240),
    ),
    Cell(
        why="qwen4 PLE value | MTP fc, 8192 rows: 2560 -> 2560",
        spec=_qwen4(8192, 2560, 2560),
    ),
    Cell(
        why="qwen4 lm_head, 1 row: 2560 -> vocab 248320",
        spec=_qwen4(1, 2560, 248320),
    ),
    Cell(
        why="qwen4 lm_head, 128 rows: decode at max concurrency",
        spec=_qwen4(128, 2560, 248320),
    ),
    Cell(
        why="qwen4 lm_head, 1024 rows: an MTP-7 verify, 128 x 8",
        spec=_qwen4(1024, 2560, 248320),
    ),
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
    "qwen4_exp_for_conditional_generation/hf_fc694b5_orig__sm_103__unified": _QWEN4_UNIFIED,
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


@pytest.mark.parametrize(
    "prefix",
    [
        "qwen4 dep4 o_proj | GDN out_proj, 32 rows",
        "qwen4 HC first projection, 1 row",
    ],
)
def test_the_band_refuses_a_product_missing_one_k_block(prefix) -> None:
    """A wrong product misses the band by a margin rather than squeaking past.

    The two cells are, for each output dtype, the qwen4 cell where this
    control measured closest to the band: 12.9 bf16 ulp RMS against 2, and 7.3
    bf16 ulp RMS (about 2.4e5 fp32 ulp) against 512 fp32 ulp.
    """
    cell = _cell(prefix)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    k = cell.spec["k"] - 16
    wrong = op.reference(mat_a[:, :k], mat_b[:k], bias, out_dtype)
    with pytest.raises(AssertionError):
        _compare(out, wrong)


def test_the_fp32_band_refuses_a_bf16_rounded_product() -> None:
    """The HC first projection asks for fp32; an output rounded to bf16 fails."""
    cell = _cell("qwen4 HC first projection, 128 rows")
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
