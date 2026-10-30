# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4 unified target's cublas_mm variant at the MTP layer (ops/gemm/cublas_mm__mtp)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.gemm.cublas_mm__mtp import (  # noqa: E501
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
    both gates are against the output's scale (`assert_within_ulp`).

    bf16 outputs: 4 ulp element-wise, 2 ulp relative RMS. Measured on GB300
    over the MTP cells at these seeds and two more: at most 1.95 element-wise
    (one rounding flip at the bottom of the row maximum's binade) and 0.12
    RMS. A product missing one 16-wide K block reads at least 14.1
    element-wise and 14.7 RMS on an N >= 256 MTP cell
    (`test_the_band_refuses_a_product_missing_one_k_block`).

    An output one column wide (the shared-expert gate, N = 1) is scored over
    the column, not per row: its row maximum is the element itself, which
    cancellation over K = 2560 puts arbitrarily near zero, where the same
    1-ulp difference reads as several ulp. Measured at 8192 rows on two seeds:
    4.3 and 8.7 ulp per row on an absolute difference of at most one bf16
    step, against at most 1.24 over the column (cosine 0.9999999). Wider
    outputs keep the row as the scale: the maximum of N >= 256 dot products
    is not near zero. A product missing one K block reads 17.0 to 75.1 over
    the gate's column at 1, 1024 and 8192 rows.

    fp32 outputs (deepseek's router): 1024 fp32 ulp element-wise, 512 RMS. A
    band scored in bf16 ulp would admit a kernel that rounds the fp32 output
    to bf16 (`test_the_fp32_band_refuses_a_bf16_rounded_product`).
    """
    element_ulp, rms_ulp = _BAND[out.dtype]
    if out.shape[-1] == 1:
        out, ref = out.reshape(1, -1), ref.reshape(1, -1)
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


# The qwen4 unified target's MTP layer (site `mtp`), cut like a trunk attention
# layer at tep4: 6 gated q heads and 1 KV head of 256 per rank; its fc_e / fc_h,
# index projection and shared expert replicated. fc_h runs on rows x hc_count 4.
# Rows: the drafter at 1..128, an MTP-7 verify at 128 x 8 = 1024, one 8192-row
# prefill chunk (32768 through fc_h).
_QWEN4_UNIFIED_MTP = (
    Cell(
        why="qwen4 MTP fc_e | fc_h, 1 row: 2560 -> 2560",
        spec=_qwen4(1, 2560, 2560),
    ),
    Cell(
        why="qwen4 MTP fc_e | fc_h, 128 rows: the drafter at max concurrency",
        spec=_qwen4(128, 2560, 2560),
    ),
    Cell(
        why="qwen4 MTP fc_e | fc_h, 1024 rows: fc_e over an MTP-7 verify",
        spec=_qwen4(1024, 2560, 2560),
    ),
    Cell(
        why="qwen4 MTP fc_h, 2048 rows: an MTP-3 verify (512) x hc_count 4",
        spec=_qwen4(2048, 2560, 2560),
    ),
    Cell(
        why="qwen4 MTP fc_h, 4096 rows: an MTP-7 verify (1024) x hc_count 4",
        spec=_qwen4(4096, 2560, 2560),
    ),
    Cell(
        why="qwen4 MTP fc_h, 32768 rows: one 8192-row prefill chunk x hc_count 4",
        spec=_qwen4(32768, 2560, 2560),
    ),
    Cell(
        why="qwen4 MTP q_proj at tep4, 1 row: 2560 -> 6 gated heads 3072",
        spec=_qwen4(1, 2560, 3072),
    ),
    Cell(
        why="qwen4 MTP q_proj at tep4, 1024 rows: an MTP-7 verify",
        spec=_qwen4(1024, 2560, 3072),
    ),
    Cell(
        why="qwen4 MTP q_proj at tep4, 8192 rows: one prefill chunk",
        spec=_qwen4(8192, 2560, 3072),
    ),
    Cell(
        why="qwen4 MTP k_proj | v_proj at tep4, 1 row: 2560 -> 1 head 256",
        spec=_qwen4(1, 2560, 256),
    ),
    Cell(
        why="qwen4 MTP k_proj | v_proj at tep4, 1024 rows: an MTP-7 verify",
        spec=_qwen4(1024, 2560, 256),
    ),
    Cell(
        why="qwen4 MTP k_proj | v_proj at tep4, 8192 rows: one prefill chunk",
        spec=_qwen4(8192, 2560, 256),
    ),
    Cell(
        why="qwen4 MTP index_qk_proj, 1 row: 2560 -> 4 x 128 + 128",
        spec=_qwen4(1, 2560, 640),
    ),
    Cell(
        why="qwen4 MTP shared-expert gate, 1 row: 2560 -> 1",
        spec=_qwen4(1, 2560, 1),
    ),
    Cell(
        why="qwen4 MTP shared-expert gate, 1024 rows: an MTP-7 verify",
        spec=_qwen4(1024, 2560, 1),
    ),
    Cell(
        why="qwen4 MTP shared-expert gate, 8192 rows: the envelope's largest gate row",
        spec=_qwen4(8192, 2560, 1),
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
    "qwen4_exp_for_conditional_generation/hf_fc694b5_orig__sm_103__unified": _QWEN4_UNIFIED_MTP,
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
        "qwen4 MTP q_proj at tep4, 1 row",
        "qwen4 MTP shared-expert gate, 1024 rows",
    ],
)
def test_the_band_refuses_a_product_missing_one_k_block(prefix) -> None:
    """A wrong product misses the band by a margin rather than squeaking past.

    One cell per scoring: a row-scored N >= 2 cell where this control
    measured closest to the band, and the column-scored N = 1 gate.
    """
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


def test_the_gate_sliced_from_the_fused_shared_expert_weight() -> None:
    """The target reads the gate as row 2 x 640 of its fused [gate | up | gate,
    padded to 8 rows] weight. That slice's `.t()` is the dense column-major
    [2560, 1] the cells build, and computes bitwise what a standalone copy does."""
    g = torch.Generator(device="cuda").manual_seed(2560)
    fused = torch.randn(2 * 640 + 8, 2560, generator=g, device="cuda").to(torch.bfloat16)
    x = torch.randn(1024, 2560, generator=g, device="cuda").to(torch.bfloat16)
    gate = fused[2 * 640 : 2 * 640 + 1].t()
    assert gate.data_ptr() != fused.data_ptr() and gate.stride() == (1, 2560)
    with validating(op):
        out = op(x, gate)
    alone = op(x, fused[2 * 640 : 2 * 640 + 1].clone().t())
    assert torch.equal(out, alone)
    _compare(out, op.reference(x, gate))


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
