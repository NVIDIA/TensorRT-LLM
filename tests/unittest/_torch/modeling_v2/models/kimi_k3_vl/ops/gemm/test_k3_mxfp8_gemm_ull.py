# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's MXFP8 GEMM (``ops/gemm/k3_mxfp8_gemm_ull.py`` on ``ops/gemm/_mxfp8_gemm_ull.py``)
on SM107: ``k3_mxfp8_gemm_ull.CELLS`` under ``validating`` against the entry's FP64 ``reference``,
through the autotuned entry and, for the cells up to ``K3_SPLIT_K_MAX_M`` rows, through one kept
tactic of every split-K factor its runner offers for the cell's tuning bucket.
"""

import importlib

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="the SM107 CuTe DSL MXFP8 GEMM",
)

K3 = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl"
k3_mxfp8_gemm_ull = importlib.import_module(f"{K3}.ops.gemm.k3_mxfp8_gemm_ull").k3_mxfp8_gemm


def _operands(m, n, k, seed):
    """E4M3 operands with power-of-two scales: the weight block-quantized (128 x 128), the
    activation by the 1 x 128 UE8M0 quantization the projections run."""
    linear = importlib.import_module(f"{K3}.ops.gemm.sm107_linear")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    blocks = torch.nn.functional.pad(w, (0, -k % 128, 0, -n % 128)).view(
        -(-n // 128), 128, -(-k // 128), 128
    )
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-8) / 448.0
    expanded = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    weight, weight_scale = linear.prepare_fp8_block_weight(
        (w / expanded).to(torch.float8_e4m3fn), scale
    )
    x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)
    act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
    return act, weight, act_scale, weight_scale


def _split_k_tactics(m, n, k):
    """One kept tactic per split-K factor > 1 of the runner, for the tuning bucket of ``m``."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as cdo

    mxfp8_gemm = importlib.import_module(f"{K3}.ops.gemm._mxfp8_gemm_ull")
    bucket = cdo._kimi_k3_mxfp8_tuning_bucket(m)
    shapes = [
        torch.empty(bucket, k, dtype=torch.float8_e4m3fn, device="cuda"),
        torch.empty(n, k, dtype=torch.float8_e4m3fn, device="cuda"),
        None,
        None,
        None,
    ]
    tactics = mxfp8_gemm.K3MXFP8RubinRunner(output_dtype=torch.bfloat16).get_valid_tactics(
        shapes, None
    )
    by_split = {}
    for tactic in tactics:
        if tactic[0] == "base" and tactic[8] > 1:
            by_split.setdefault(tactic[8], tactic)
    return mxfp8_gemm, by_split


@pytest.mark.parametrize("cell", k3_mxfp8_gemm_ull.CELLS, ids=lambda c: c.why[:48])
@torch.inference_mode()
def test_k3_mxfp8_gemm_ull_cells(cell):
    """Both operands are E4M3 with power-of-two scales, so every product is exact.

    Without split-K the kernel sums them in FP32 and rounds once to BF16: at most 2 units of
    ``ULP`` of the row maximum element-wise, 1 in RMS. A split-K tactic rounds each of its ``s``
    FP32 partials to BF16 and adds them with BF16 atomics, one more rounding per add; a partial
    and a running sum stay within twice the row maximum on these operands, so each split adds at
    most 2 units: ``2 + 2 s`` element-wise and ``1 + s`` in RMS.
    """
    op = k3_mxfp8_gemm_ull
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    act, weight, act_scale, weight_scale = _operands(m, n, k, seed=2)
    with validating(op):
        got = op(act, weight, act_scale, weight_scale)
    want = op.reference(act, weight, act_scale, weight_scale)
    assert_within_ulp(got, want, 2, 1)
    mxfp8_gemm, by_split = _split_k_tactics(m, n, k)
    if m > mxfp8_gemm.K3_SPLIT_K_MAX_M:
        assert not by_split, f"{cell.why}: split-K tactic kept above {mxfp8_gemm.K3_SPLIT_K_MAX_M}"
        return
    for split_k, tactic in sorted(by_split.items()):
        got = mxfp8_gemm.run_with_tactic(act, weight, act_scale, weight_scale, tactic)
        assert_within_ulp(got, want, 2 + 2 * split_k, 1 + split_k)
