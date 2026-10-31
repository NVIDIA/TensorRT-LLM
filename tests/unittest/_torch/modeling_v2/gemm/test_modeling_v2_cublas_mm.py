# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_mm catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as op

__extra_import_path__ = [".."]
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
    the order cuBLAS and torch sum over K, and in where a bias joins the sum
    before the one rounding. Over random operands a few outputs land near
    zero, where torch's per-element rtol is meaningless, so both gates are
    against the output's scale (`assert_within_ulp`).

    bf16 outputs: 4 ulp element-wise, 2 ulp relative RMS. Measured on GB300
    over every bf16 cell here: at most 1.98 element-wise -- one rounding flip
    on an element at the bottom of the row maximum's binade, where one bf16
    step is two units of ULP's 2^-8 -- and at most 0.12 RMS (the qwen4 MTP
    cells; 0.048 on the trunk cells, 0.042 on the vision cells, whose 99.9th
    percentile is 0.033). The deepseek cells read up to 1.47 (o_proj). A
    product missing one 16-wide K block reads at least 11.2 element-wise and
    8.5 RMS (`test_the_band_refuses_a_product_missing_one_k_block`). The
    vision cells' bias is a bf16 [N] row of unit scale against products of
    scale sqrt(K), so it is the smallest term a wrong epilogue could get
    wrong: a product missing it, or adding it twice (one bias per split-K
    slice), still reads at least 4.33 element-wise and 3.66 RMS
    (`test_the_band_refuses_a_misapplied_bias`).

    An output one column wide (the qwen4 MTP shared-expert gate, N = 1) is
    scored over the column, not per row: its row maximum is the element
    itself, which cancellation over K = 2560 puts arbitrarily near zero,
    where the same 1-ulp difference reads as several ulp. Measured at 8192
    rows on two seeds: 4.3 and 8.7 ulp per row on an absolute difference of
    at most one bf16 step, against at most 1.24 over the column (cosine
    0.9999999). Wider outputs keep the row as the scale: the maximum of
    N >= 256 dot products is not near zero. A product missing one K block
    reads 17.0 to 75.1 over the gate's column at 1, 1024 and 8192 rows.

    fp32 outputs (the qwen4 HC first projection, deepseek's router): 1024 fp32
    ulp element-wise, 512 RMS. Measured at most 192 and 97 (HC, 8192 rows);
    the deepseek router reads 22 and 15. The same product rounded through
    bf16 reads at least 24653 and 13836 fp32 ulp, so this band refuses a
    kernel that rounds the fp32 output to bf16 -- which a band scored in bf16
    ulp admits: that rounding is under 2 bf16 ulp of the row scale
    (`test_the_fp32_band_refuses_a_bf16_rounded_product`).
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


def _vision(m, k, n):
    return dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=True, out_dtype=None)


# The qwen4 unified target (tp4 / tep4 / dep4, hidden 2560) calls this entry for
# every dense bf16 projection on this rank's rows. Per rank: tp4/tep4 hold 6 q / 1
# kv attention heads and 4 key / 12 value GDN heads; dep4 holds tp1's full widths.
# Rows: decode 1..128 at tp4/tep4 and 1..32 per rank at dep4, 1024 at an MTP-7
# verify (128 x 8), 8192 for one prefill chunk at max_num_tokens, and up to 4 x
# 8192 at dep4's router, which routes the allgathered rows.
_QWEN4_UNIFIED_TRUNK = (
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

# The qwen4 unified target's MTP layer, cut like a trunk attention layer at
# tep4: 6 gated q heads and 1 KV head of 256 per rank; its fc_e / fc_h, index
# projection and shared expert replicated. fc_h runs on rows x hc_count 4.
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

# The qwen4 unified target's vision tower, the biased projections of tp1's
# VisionTower: hidden 1152, 16 heads x 72, MLP 4304, and the patch merger over
# 2 x 2 merged patches (4 x 1152 = 4608) into the language hidden 2560. The
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
    "qwen4_exp_for_conditional_generation/hf_fc694b5_orig__sm_103__unified": (
        _QWEN4_UNIFIED_TRUNK + _QWEN4_UNIFIED_MTP + _QWEN4_UNIFIED_VISION
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def test_the_entry_cells_are_all_driven() -> None:
    """`test_certified_cells` drives every cell the entry itself declares (`op.CELLS`),
    not only the targets' `TARGET_CELLS`: `certified_cells` keeps one cell per distinct spec."""
    driven = {repr(sorted(c.spec.items())) for c in CELLS}
    undriven = [c.why for c in op.CELLS if repr(sorted(c.spec.items())) not in driven]
    assert not undriven, f"entry cells test_certified_cells does not drive: {undriven}"


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


# The vision cells where the bias and K-block controls measured closest to the
# band: the merger's K of 4608 makes the products widest against a unit-scale
# bias.
_VISION_CONTROL_CELLS = [
    "qwen4 vision merger fc2, 4096 merged rows",
    "qwen4 vision fc2, 1024 patches",
    "qwen4 vision qkv, 1024 patches",
]


@pytest.mark.parametrize(
    "prefix",
    [
        "qwen4 dep4 o_proj | GDN out_proj, 32 rows",
        "qwen4 HC first projection, 1 row",
        "qwen4 MTP q_proj at tep4, 1 row",
        "qwen4 MTP shared-expert gate, 1024 rows",
        *_VISION_CONTROL_CELLS,
    ],
)
def test_the_band_refuses_a_product_missing_one_k_block(prefix) -> None:
    """A wrong product misses the band by a margin rather than squeaking past.

    The cells are, per scoring, the ones where this control measured closest
    to the band: the trunk's bf16 dep4 o_proj at 32 rows (12.9 bf16 ulp RMS
    against 2) and the fp32 HC first projection at 1 row (7.3 bf16 ulp RMS,
    about 2.4e5 fp32 ulp, against 512 fp32 ulp); a row-scored MTP cell and the
    column-scored N = 1 gate; and the biased vision cells.
    """
    cell = _cell(prefix)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, bias, out_dtype)
    k = cell.spec["k"] - 16
    wrong = op.reference(mat_a[:, :k], mat_b[:k], bias, out_dtype)
    with pytest.raises(AssertionError):
        _compare(out, wrong)


@pytest.mark.parametrize("prefix", _VISION_CONTROL_CELLS)
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


@pytest.mark.parametrize("prefix", ["qwen4 HC first projection, 128 rows", "deepseek router"])
def test_the_fp32_band_refuses_a_bf16_rounded_product(prefix) -> None:
    """The qwen4 HC first projection and the deepseek router ask for fp32; an
    output rounded to bf16 fails."""
    cell = _cell(prefix)
    mat_a, mat_b, bias, out_dtype = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(mat_a, mat_b, bias, out_dtype)
    assert ref.dtype is torch.float32
    with pytest.raises(AssertionError):
        _compare(ref.to(torch.bfloat16).float(), ref)


def test_the_gate_sliced_from_the_fused_shared_expert_weight() -> None:
    """The qwen4 target reads the MTP gate as row 2 x 640 of its fused [gate |
    up | gate, padded to 8 rows] weight. That slice's `.t()` is the dense
    column-major [2560, 1] the cells build, and computes bitwise what a
    standalone copy does."""
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
