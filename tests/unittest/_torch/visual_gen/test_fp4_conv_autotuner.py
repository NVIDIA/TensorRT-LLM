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
from tensorrt_llm._torch.visual_gen.models.wan.fp4_conv_autotuner import (
    FP4_CONV_FALLBACK_TACTIC,
    FP4_CONV_FIXED_TACTIC,
    FP4_CONV_TACTICS,
    FP4ConvTunableRunner,
    _selected_tactics,
    run_tuned_fp4_conv,
)
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import (
    NVFP4WanCausalConv3d,
    WanCausalConv3d,
    _fp4_compile_cache,
)


def test_fp4_conv_tuner_starts_away_from_fixed_tactic():
    assert FP4ConvTunableRunner.resolve_tactic(-1) == FP4_CONV_FALLBACK_TACTIC
    assert FP4_CONV_FALLBACK_TACTIC != FP4_CONV_FIXED_TACTIC
    assert FP4_CONV_FIXED_TACTIC in FP4_CONV_TACTICS


def test_fp4_conv_tuner_precompiles_candidates_before_launching():
    compiled = []
    launched = []
    output = torch.empty(1)

    def compile_tactic(tactic):
        compiled.append(tactic)
        return tactic

    runner = FP4ConvTunableRunner(
        signature=("test",),
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


def test_selected_tactic_fast_path_skips_autotuner_lookup(monkeypatch):
    output = torch.empty(1)
    launched = []
    fixed_id = FP4_CONV_TACTICS.index(FP4_CONV_FIXED_TACTIC)

    class StubTuner:
        @staticmethod
        def choose_one(_name, runners, _config, _inputs):
            return runners[0], fixed_id

    _selected_tactics.clear()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda _cls: StubTuner()))
    kwargs = {
        "signature": ("test",),
        "problem_shape": (1, 2, 3),
        "tuning_inputs": (),
        "compile_tactic": lambda tactic: tactic,
        "launch": launched.append,
        "output": output,
    }
    run_tuned_fp4_conv(**kwargs)

    def fail_lookup(_cls):
        pytest.fail("a cached tactic must not query the global autotuner")

    monkeypatch.setattr(AutoTuner, "get", classmethod(fail_lookup))
    run_tuned_fp4_conv(**kwargs)
    assert launched == [FP4_CONV_FIXED_TACTIC, FP4_CONV_FIXED_TACTIC]
    _selected_tactics.clear()


def test_autotuned_fp4_conv_matches_bf16_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 Conv3d autotuning requires a Blackwell GPU")

    AutoTuner.get().clear_cache()
    _selected_tactics.clear()
    try:
        _fp4_compile_cache.clear()
        torch.manual_seed(7)
        base = WanCausalConv3d(256, 256, 3, padding=1).cuda().to(torch.bfloat16).eval()
        conv = NVFP4WanCausalConv3d(base).cuda().to(torch.bfloat16).eval()
        activation = torch.randn((1, 256, 1, 4, 6), device="cuda", dtype=torch.bfloat16)

        with (
            torch.inference_mode(),
            autotune(
                tune_mode=True,
                skip_dynamic_tuning_buckets=True,
            ),
        ):
            actual = conv(activation)
        expected = F.conv3d(F.pad(activation, (1, 1, 1, 1, 2, 0)), base.weight, base.bias)

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
    finally:
        AutoTuner.get().clear_cache()
        _selected_tactics.clear()
