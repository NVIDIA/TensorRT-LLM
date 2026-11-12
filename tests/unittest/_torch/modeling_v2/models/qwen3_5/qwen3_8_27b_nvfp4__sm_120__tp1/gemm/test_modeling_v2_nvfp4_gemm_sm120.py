# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_gemm_sm120 entry.

Each cell twice: the entry's call as a served step makes it (the cached tactic, or
the CUTLASS default on a miss), and every tactic the AutoTuner could pick for that
row count -- each against the reference of the dequantized operands, in the band
`_compare` derives. The candidate set itself is pinned: FlashInfer alone exactly
where its fill rule holds, the CUDA-core GEMM only up to 8 rows. Beyond the cells:
a tuned row slice takes the tactic and the bytes of the unsliced call, the tuning
buckets and their near-tie wiring, and the guards.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm import (  # noqa: E501
    _nvfp4_gemm_runner as nvfp4_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.nvfp4_gemm_sm120 import (  # noqa: E501
    nvfp4_gemm_sm120 as op,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Every candidate and the reference multiply the same E2M1 x E4M3-scaled values
    exactly and accumulate in fp32; they differ in summation order (tile shape, split-K
    reduced in fp32), then alpha and one rounding.

    BF16 out: the order moves an output only at a BF16 rounding midpoint, a one-ulp flip
    = 2 units of the row maximum's 2**-8; 4 units element-wise, 2 of relative RMS.

    FP32 out (the lm_head): the order difference itself is visible. Over K = 5120 the
    partial sums reach about sqrt(K) times the output's scale; with fp32 rounding at every
    add (2**-24 relative) in two different orders the expected difference is ~2**-20 of
    the row maximum, ~2**-17 at the maximum over 1.4e7 outputs. The band, 2**-13
    element-wise and 2**-15 relative RMS, leaves 16x on top for tensor-core accumulator
    truncation and is still 2**10 below a single misplaced 16-block.
    """
    assert out.dtype == ref.dtype and out.shape == ref.shape and out.is_contiguous()
    if out.dtype == torch.float32:
        assert_within_ulp(out, ref, element_ulp=2.0**10, rms_ulp=2.0**8, ulp=2.0**-23)
    else:
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _operands(n: int, k: int, m: int, seed: int, rows: torch.Tensor | None = None):
    """NVFP4 operands as the target produces them: `fp4_quantize` with swizzled scales,
    global scales 2688 / amax, alpha their product's inverse. ``rows`` overrides the
    activation (to quantize a row slice with the full activation's global scale)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, k, generator=g, device=DEV).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device=DEV) * k**-0.5).to(torch.bfloat16)
    act_global = (2688.0 / x.abs().amax().float()).reshape(1)
    w_global = (2688.0 / w.abs().amax().float()).reshape(1)
    a = x if rows is None else rows
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(a, act_global, 16, False, True)
    weight, weight_sf = torch.ops.trtllm.fp4_quantize(w, w_global, 16, False, True)
    alpha = (1.0 / (act_global * w_global)).reshape(1)
    return [act_fp4, weight, act_sf, weight_sf, alpha], x


def _fi_expected(n: int, k: int, m: int, out: torch.dtype) -> bool:
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return (
        out == torch.bfloat16
        and 64 <= m <= 255
        and n >= 1024
        and n % 128 == 0
        and k % 256 == 0
        and 2 * ((n + 127) // 128) * ((m + 127) // 128) <= sms
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    inputs, _ = _operands(s["n"], s["k"], s["m"], seed_of(cell.why))
    out = op(*inputs, s["out"], s["tune_rows"])
    assert out.shape == (s["m"], s["n"])
    ref = op.reference(*inputs, s["out"], s["tune_rows"])
    _compare(out, ref)

    runner = nvfp4_runner.runner_for(s["out"])
    tactics = runner.get_valid_tactics(inputs, None)
    backends = {t[0] for t in tactics}
    if _fi_expected(s["n"], s["k"], s["m"], s["out"]):
        assert backends == {"fi_cutlass"}, backends
    else:
        assert "fi_cutlass" not in backends and "cutlass" in backends, backends
        assert ("cuda_core" in backends) == (s["m"] <= nvfp4_runner.CUDA_CORE_MAX_M), backends
    ran = 0
    for tactic in tactics:
        try:
            got = runner(inputs, tactic=tactic)
        except RuntimeError:
            # The AutoTuner skips a tactic that fails to run; so does this sweep.
            torch.cuda.synchronize()
            continue
        _compare(got, ref)
        ran += 1
    assert ran > 0, "no candidate tactic ran"


def test_a_row_slice_takes_the_unsliced_tactic_and_bytes() -> None:
    """The final layer of a long prefill step runs only the rows the logits read, tuned
    as the whole step: after tuning at 2080 rows (a mixed bucket), 5 rows quantized with
    the same global scale take the 2080-row tactic and return exactly those rows of the
    2080-row call."""
    n, k, full, rows = 5120, 6144, 2080, 5
    tuner = AutoTuner.get()
    tuner.clear_cache()
    try:
        inputs, x = _operands(n, k, full, seed=21)
        with autotune():
            op(*inputs, torch.bfloat16, full)
        sliced, _ = _operands(n, k, full, seed=21, rows=x[:rows].contiguous())
        assert torch.equal(sliced[0], inputs[0][:rows])
        runner = nvfp4_runner.runner_for(torch.bfloat16)
        want = nvfp4_runner.lookup_tactic(runner, inputs, full)
        assert want != -1, "tuning left no cached tactic for the 2080-row bucket"
        assert nvfp4_runner.lookup_tactic(runner, sliced, full) == want
        out_full = op(*inputs, torch.bfloat16, full)
        out_rows = op(*sliced, torch.bfloat16, full)
        assert torch.equal(out_rows, out_full[:rows])
    finally:
        tuner.clear_cache()


def test_tuning_buckets() -> None:
    gen, mapm = nvfp4_runner._nvfp4_gen_m_buckets, nvfp4_runner._nvfp4_map_m
    buckets = gen(8192)
    assert {1, 2, 4, 8, 2048, 2080, 4095, 4096, 4128, 8191, 8192} <= set(buckets)
    assert all(b <= 8192 for b in buckets)
    for m, bucket in (
        (1, 1), (33, 32), (2048, 2048), (2049, 2080), (2080, 2080), (2081, 2048),
        (4063, 2048), (4064, 4095), (4096, 4096), (4097, 4128), (8160, 8191), (8192, 8192),
    ):  # fmt: skip
        assert mapm(m) == bucket, (m, mapm(m), bucket)
    assert nvfp4_runner.TUNING_CONFIG.use_cold_l2_cache


def test_guard_refuses_a_tune_rows_below_the_call() -> None:
    inputs, _ = _operands(5120, 6144, 8, seed=3)
    with validating(op), pytest.raises(AssertionError, match="tune_rows"):
        op(*inputs, torch.bfloat16, 4)


def test_guard_refuses_a_per_token_alpha() -> None:
    inputs, _ = _operands(5120, 6144, 8, seed=4)
    inputs[4] = inputs[4].repeat(8)
    with validating(op), pytest.raises(AssertionError, match="alpha"):
        op(*inputs, torch.bfloat16, 8)
