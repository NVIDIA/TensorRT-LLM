# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.visual_gen.models.wan import fp4_conv_autotuner
from tensorrt_llm._torch.visual_gen.models.wan.fp4_conv_autotuner import (
    FP4_CONV_FALLBACK_TACTIC,
    FP4_CONV_FIXED_TACTIC,
    FP4_CONV_TACTICS,
    FP4ConvTunableRunner,
    clear_fp4_conv_tactic_cache,
    run_tuned_fp4_conv,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import (
    NVFP4WanCausalConv3d,
    WanCausalConv3d,
    _fp4_compile_cache,
)


def _make_fp4_conv_case(channels, input_scale, use_residual):
    torch.manual_seed(7)
    base = WanCausalConv3d(channels, channels, 3, padding=1).cuda().to(torch.bfloat16).eval()
    conv = NVFP4WanCausalConv3d(base, input_scale=input_scale).cuda().to(torch.bfloat16).eval()
    activation = torch.randn(
        (1, channels, 1, 4, 6),
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual = torch.randn_like(activation) if use_residual else None
    expected = F.conv3d(F.pad(activation, (1, 1, 1, 1, 2, 0)), base.weight, base.bias)
    if residual is not None:
        expected = expected + residual
    return conv, activation, residual, expected


def _assert_fp4_close(actual, expected):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    relative_l2_error = (actual_f32 - expected_f32).norm() / expected_f32.norm()
    cosine_similarity = F.cosine_similarity(
        actual_f32.flatten(),
        expected_f32.flatten(),
        dim=0,
    )
    assert relative_l2_error.item() < 0.25
    assert cosine_similarity.item() > 0.96


def test_fp4_conv_tuner_uses_fixed_tactic_as_fallback():
    assert FP4ConvTunableRunner.resolve_tactic(-1) == FP4_CONV_FALLBACK_TACTIC
    assert FP4_CONV_FALLBACK_TACTIC == FP4_CONV_FIXED_TACTIC
    assert FP4ConvTunableRunner.resolve_tactic(-2) == FP4_CONV_FIXED_TACTIC
    assert FP4ConvTunableRunner.resolve_tactic("invalid") == FP4_CONV_FIXED_TACTIC
    assert FP4ConvTunableRunner.resolve_tactic(len(FP4_CONV_TACTICS)) == FP4_CONV_FIXED_TACTIC
    assert any(tactic.mma_tiler == (128, 128) for tactic in FP4_CONV_TACTICS)
    assert FP4_CONV_FIXED_TACTIC in FP4_CONV_TACTICS


def test_fp4_conv_tactic_definitions_version_persistent_key(monkeypatch):
    runner = FP4ConvTunableRunner(
        signature=("version-test",),
        problem_shape=(1, 2, 3),
        compile_tactic=lambda tactic: tactic,
        launch=lambda _compiled: None,
        output=torch.empty(1),
    )
    original_key = runner.unique_id()

    monkeypatch.setattr(
        fp4_conv_autotuner,
        "FP4_CONV_TACTICS",
        tuple(reversed(FP4_CONV_TACTICS)),
    )
    assert runner.unique_id() != original_key


def test_fp4_conv_tuner_precompiles_candidates_before_launching():
    compiled = []
    launched = []
    output = torch.empty(1)

    def compile_tactic(tactic):
        compiled.append(tactic)
        return tactic

    runner = FP4ConvTunableRunner(
        signature=("test",),
        problem_shape=(1, 2, 3),
        compile_tactic=compile_tactic,
        launch=launched.append,
        output=output,
    )

    assert runner([], do_preparation=True) is output
    assert compiled == list(FP4_CONV_TACTICS)
    assert not launched

    fixed_id = FP4_CONV_TACTICS.index(FP4_CONV_FIXED_TACTIC)
    assert runner([], tactic=fixed_id) is output
    assert launched == [FP4_CONV_FIXED_TACTIC]


def test_fp4_conv_tuner_isolates_candidate_compile_failure():
    failed_id = 2
    failed_tactic = FP4_CONV_TACTICS[failed_id]
    compiled = []

    def compile_tactic(tactic):
        compiled.append(tactic)
        if tactic == failed_tactic:
            raise RuntimeError("unsupported test tactic")
        return tactic

    runner = FP4ConvTunableRunner(
        signature=("compile-failure-test",),
        problem_shape=(1, 2, 3),
        compile_tactic=compile_tactic,
        launch=lambda _compiled: None,
        output=torch.empty(1),
    )

    runner([], do_preparation=True)
    assert compiled == list(FP4_CONV_TACTICS)
    assert failed_id not in runner.get_valid_tactics([], None)
    with pytest.raises(RuntimeError, match="failed during preparation"):
        runner([], tactic=failed_id)
    assert compiled.count(failed_tactic) == 1


def test_fp4_conv_tuner_shares_launch_failures_across_runners():
    failed_id = 3
    failed_tactic = FP4_CONV_TACTICS[failed_id]

    def launch(tactic):
        if tactic == failed_tactic:
            raise RuntimeError("unsupported launch")

    first = FP4ConvTunableRunner(
        signature=("launch-failure-test",),
        problem_shape=(1, 2, 3),
        compile_tactic=lambda tactic: tactic,
        launch=launch,
        output=torch.empty(1),
    )
    with pytest.raises(RuntimeError, match=f"tactic {failed_id} failed during launch"):
        first([], tactic=failed_id)

    second = FP4ConvTunableRunner(
        signature=("launch-failure-test",),
        problem_shape=(1, 2, 3),
        compile_tactic=lambda tactic: tactic,
        launch=launch,
        output=torch.empty(1),
    )
    assert failed_id not in second.get_valid_tactics([], None)

    different_shape = FP4ConvTunableRunner(
        signature=("launch-failure-test",),
        problem_shape=(4, 5, 6),
        compile_tactic=lambda tactic: tactic,
        launch=launch,
        output=torch.empty(1),
    )
    assert failed_id in different_shape.get_valid_tactics([], None)


def test_selected_tactic_fast_cache_obeys_tuning_and_capture(monkeypatch):
    output = torch.empty(1)
    launched = []
    fixed_id = FP4_CONV_TACTICS.index(FP4_CONV_FIXED_TACTIC)
    alternate_id = 0

    class StubTuner:
        class ProfilingCache:
            generation = 0

        is_tuning_mode = False
        is_capturing_tactics = False
        choose_count = 0
        selected_id = -1
        profiling_cache = ProfilingCache()

        def choose_one(self, _name, runners, _config, _inputs):
            self.choose_count += 1
            return runners[0], self.selected_id

    tuner = StubTuner()
    clear_fp4_conv_tactic_cache()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda _cls: tuner))
    kwargs = {
        "signature": ("test",),
        "problem_shape": (1, 2, 3),
        "tuning_inputs": (),
        "compile_tactic": lambda tactic: tactic,
        "launch": launched.append,
        "output": output,
    }
    run_tuned_fp4_conv(**kwargs)
    run_tuned_fp4_conv(**kwargs)
    assert tuner.choose_count == 1

    tuner.is_tuning_mode = True
    tuner.selected_id = alternate_id
    run_tuned_fp4_conv(**kwargs)
    assert tuner.choose_count == 2

    tuner.is_tuning_mode = False
    tuner.selected_id = fixed_id
    run_tuned_fp4_conv(**kwargs)
    run_tuned_fp4_conv(**kwargs)
    assert tuner.choose_count == 3

    tuner.profiling_cache.generation += 1
    tuner.selected_id = alternate_id
    run_tuned_fp4_conv(**kwargs)
    assert tuner.choose_count == 4

    tuner.is_capturing_tactics = True
    run_tuned_fp4_conv(**kwargs)
    assert tuner.choose_count == 5
    assert launched == [
        FP4_CONV_FIXED_TACTIC,
        FP4_CONV_FIXED_TACTIC,
        FP4_CONV_TACTICS[alternate_id],
        FP4_CONV_FIXED_TACTIC,
        FP4_CONV_FIXED_TACTIC,
        FP4_CONV_TACTICS[alternate_id],
        FP4_CONV_TACTICS[alternate_id],
    ]
    clear_fp4_conv_tactic_cache()


def test_all_valid_autotuned_fp4_conv_tactics_match_bf16_reference(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 Conv3d autotuning requires a Blackwell GPU")

    AutoTuner.get().clear_cache()
    clear_fp4_conv_tactic_cache()
    try:
        _fp4_compile_cache.clear()
        conv, activation, residual, expected = _make_fp4_conv_case(256, None, False)

        with (
            torch.inference_mode(),
            autotune(
                tune_mode=True,
                skip_dynamic_tuning_buckets=True,
            ),
        ):
            conv(activation, residual=residual)

        selected_tactics = []
        original_run_tuned = fp4_conv_autotuner.run_tuned_fp4_conv

        def record_selected_tactic(**kwargs):
            output, tactic = original_run_tuned(**kwargs)
            selected_tactics.append(tactic)
            return output, tactic

        monkeypatch.setattr(
            fp4_conv_autotuner,
            "run_tuned_fp4_conv",
            record_selected_tactic,
        )
        tuner = AutoTuner.get()
        with torch.inference_mode(), tuner.capture() as all_tactics:
            conv(activation, residual=residual)

        selected_tactics.clear()
        replayed_tactic_ids = []
        for ((runner, tactic),) in all_tactics:
            replayed_tactic_ids.append(tactic)
            with torch.inference_mode(), tuner.replay(((runner, tactic),)):
                actual = conv(activation, residual=residual)

            _assert_fp4_close(actual, expected)

        assert selected_tactics == [
            FP4ConvTunableRunner.resolve_tactic(tactic_id) for tactic_id in replayed_tactic_ids
        ]
        assert 0 < len(selected_tactics) <= len(FP4_CONV_TACTICS)
    finally:
        AutoTuner.get().clear_cache()
        clear_fp4_conv_tactic_cache()


def test_fixed_fp4_conv_residual_tactic_matches_bf16_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 Conv3d requires a Blackwell GPU")

    AutoTuner.get().clear_cache()
    clear_fp4_conv_tactic_cache()
    try:
        _fp4_compile_cache.clear()
        conv, activation, residual, expected = _make_fp4_conv_case(256, 1.0 / 50.0, True)
        with torch.inference_mode():
            actual = conv(activation, residual=residual)
        _assert_fp4_close(actual, expected)
    finally:
        AutoTuner.get().clear_cache()
        clear_fp4_conv_tactic_cache()
