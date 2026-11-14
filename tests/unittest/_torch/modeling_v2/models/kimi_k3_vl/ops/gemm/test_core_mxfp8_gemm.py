# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3 MXFP8 GEMM (``ops/gemm/k3_mxfp8_gemm.py`` on ``ops/gemm/_mxfp8_gemm.py``) on SM107.

* Its runner's candidate tactics are main's ``CuteDSLKimiK3MXFP8RubinLinear`` list with K3's rules
  applied (checked here rule by rule, independently of ``keeps_tactic``), in main's order, over the
  K3 FP8 projection shapes and every K3 tuning bucket.
* With one tactic, its launch is bitwise main's runner with that tactic.
* The op asks the autotuner under its own name with that runner, and untuned it is bitwise main's
  ``cute_dsl_mxfp8_gemm_rubin(fine_grained_m=True)``.
* The runner does not share an AutoTuner cache entry with main's (split-K) runner.
* ``k3_mxfp8_gemm.CELLS`` under ``validating`` against the entry's FP64 ``reference``.
"""

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

K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
# Per-rank [N, K] of the K3 FP8 block-scale projections that reach the MXFP8 GEMM.
SHAPES = ((2112, 7168), (1152, 1536), (2304, 512), (7168, 1024), (512, 7168), (4096, 7168))
# Shapes and rows whose tactics run against main's runner (each new tactic compiles a kernel).
RUN_SHAPES = ((2112, 7168), (7168, 1024))
RUN_ROWS = (1, 48, 256, 1000)
RUN_TACTICS = 2


def _mods():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm import (  # noqa: E501
        _mxfp8_gemm as mxfp8_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm import (
        sm107_linear as linear,
    )
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as cdo

    return linear, mxfp8_gemm, cdo


def _entry():
    import importlib

    return importlib.import_module(f"{K3}.ops.gemm.k3_mxfp8_gemm").k3_mxfp8_gemm


def _tactic_inputs(m, n, k):
    """Shape-only operands for ``get_valid_tactics``."""
    a = torch.empty(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty(n, k, dtype=torch.float8_e4m3fn, device="cuda")
    return [a, b, None, None, None]


def _split_k(tactic):
    return tactic[8] if tactic[0] == "base" else 1


def _k3_rules_hold(tactic):
    if tactic[0] == "base":
        _, tiler, inst, _cluster, swap_ab, use_prefetch, _sched, _raster, split_k = tactic
    else:
        _, tiler, inst, _pref, _fallback, swap_ab, use_prefetch, *_ = tactic
        split_k = 1
    mma_m = (tiler[0], inst[0])
    return (
        mma_m != (256, 128)
        and (not use_prefetch or (not swap_ab and mma_m == (512, 256)))
        and split_k == 1
    )


def test_tactics_are_main_list_under_k3_rules():
    _, mxfp8_gemm, cdo = _mods()
    main_runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16)
    k3_runner = mxfp8_gemm.K3MXFP8RubinRunner(output_dtype=torch.bfloat16)
    buckets = cdo._get_kimi_k3_mxfp8_tuning_buckets(8192)
    dropped = 0
    for n, k in SHAPES:
        for m in buckets:
            inputs = _tactic_inputs(m, n, k)
            main = main_runner.get_valid_tactics(inputs, None)
            got = k3_runner.get_valid_tactics(inputs, None)
            want = [t for t in main if _k3_rules_hold(t)]
            assert got == want, (n, k, m)
            assert got, (n, k, m)
            assert all(len(t) == 9 for t in got if t[0] == "base"), (n, k, m)
            dropped += len(main) - len(got)
    assert dropped


def _operands(m, n, k, seed=0):
    linear, _, _ = _mods()
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


@torch.inference_mode()
def test_each_tactic_matches_main_runner():
    _, mxfp8_gemm, cdo = _mods()
    main_runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    for n, k in RUN_SHAPES:
        for m in RUN_ROWS:
            act, weight, act_scale, weight_scale = _operands(m, n, k)
            alpha = cdo._get_mxfp8_gemm_alpha(act.device)
            inputs = [act, weight, act_scale, weight_scale, alpha]
            bucket = cdo._kimi_k3_mxfp8_tuning_bucket(m)
            tactics = mxfp8_gemm.K3MXFP8RubinRunner(output_dtype=torch.bfloat16).get_valid_tactics(
                _tactic_inputs(bucket, n, k), None
            )
            assert all(_split_k(t) == 1 for t in tactics), (n, k, m)
            step = max(1, len(tactics) // RUN_TACTICS)
            for tactic in [None, *tactics[::step][:RUN_TACTICS]]:
                got = mxfp8_gemm.run_with_tactic(act, weight, act_scale, weight_scale, tactic)
                want = main_runner(inputs, tactic=-1 if tactic is None else tactic)
                assert torch.equal(got, want), (n, k, m, tactic)


@torch.inference_mode()
def test_op_tunes_k3_runner_and_matches_main_untuned(monkeypatch):
    _, mxfp8_gemm, _ = _mods()
    k3_mxfp8_gemm = _entry()
    from tensorrt_llm._torch.autotuner import AutoTuner

    for n, k in RUN_SHAPES:
        for m in RUN_ROWS:
            act, weight, act_scale, weight_scale = _operands(m, n, k, seed=1)
            got = k3_mxfp8_gemm(act, weight, act_scale, weight_scale)
            want = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                act, weight, act_scale, weight_scale, fine_grained_m=True
            )
            assert torch.equal(got, want), (n, k, m)

    seen = []
    choose_one = AutoTuner.choose_one

    def spy(self, custom_op, runners, tuning_config, inputs, **kwargs):
        seen.append((custom_op, [type(r) for r in runners], tuning_config))
        return choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs)

    monkeypatch.setattr(AutoTuner, "choose_one", spy)
    k3_mxfp8_gemm(*_operands(16, *RUN_SHAPES[0]))
    assert seen == [
        (
            mxfp8_gemm.OP_NAME,
            [mxfp8_gemm.K3MXFP8RubinRunner],
            mxfp8_gemm.K3MXFP8RubinRunner.tuning_config,
        )
    ]


def test_runner_does_not_share_main_cache_entries():
    _, mxfp8_gemm, cdo = _mods()
    got = mxfp8_gemm.K3MXFP8RubinRunner(output_dtype=torch.bfloat16).unique_id()
    want = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16).unique_id()
    assert got == want + (("split_k", (1,)),)


@pytest.mark.parametrize("cell", _entry().CELLS, ids=lambda c: c.why[:48])
@torch.inference_mode()
def test_k3_mxfp8_gemm_cells(cell):
    """Both operands are E4M3 with power-of-two scales, so every product is exact; the kernel
    sums them in FP32 (no split-K tactic) and rounds once to BF16: at most 2 units of ``ULP`` of
    the row maximum element-wise, 1 in RMS."""
    k3_mxfp8_gemm = _entry()
    n, k, m = cell.spec["n"], cell.spec["k"], cell.spec["m"]
    act, weight, act_scale, weight_scale = _operands(m, n, k, seed=2)
    with validating(k3_mxfp8_gemm):
        got = k3_mxfp8_gemm(act, weight, act_scale, weight_scale)
    want = k3_mxfp8_gemm.reference(act, weight, act_scale, weight_scale)
    assert_within_ulp(got, want, 2, 1)
