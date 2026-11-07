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
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import (
    NVFP4WanCausalConv3d,
    WanCausalConv3d,
    _fp4_compile_cache,
    _supports_nvfp4_device,
)
from tensorrt_llm._torch.visual_gen.modules.vae import nvfp4_conv_autotuner
from tensorrt_llm._torch.visual_gen.modules.vae.nvfp4_conv_autotuner import (
    FP4_CONV_FALLBACK_TACTIC,
    FP4_CONV_FIXED_TACTIC,
    FP4_CONV_TACTICS,
    FP4ConvTunableRunner,
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
    expected = F.conv3d(F.pad(activation, (1, 1, 1, 1, 2, 0)), conv.weight, conv.bias)
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
        nvfp4_conv_autotuner,
        "FP4_CONV_TACTICS",
        tuple(reversed(FP4_CONV_TACTICS)),
    )
    assert runner.unique_id() != original_key


def test_fp4_conv_tuner_validates_candidates_before_launching():
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

    assert runner.get_valid_tactics([], None) == list(range(len(FP4_CONV_TACTICS)))
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

    valid_tactics = runner.get_valid_tactics([], None)
    assert compiled == list(FP4_CONV_TACTICS)
    assert failed_id not in valid_tactics
    assert compiled.count(failed_tactic) == 1


def test_fp4_conv_tuner_propagates_launch_failure():
    failed_id = 3
    failed_tactic = FP4_CONV_TACTICS[failed_id]

    def launch(tactic):
        if tactic == failed_tactic:
            raise RuntimeError("unsupported launch")

    runner = FP4ConvTunableRunner(
        signature=("launch-failure-test",),
        problem_shape=(1, 2, 3),
        compile_tactic=lambda tactic: tactic,
        launch=launch,
        output=torch.empty(1),
    )
    with pytest.raises(RuntimeError, match=f"tactic {failed_id} failed during launch"):
        runner([], tactic=failed_id)


def test_all_valid_autotuned_fp4_conv_tactics_match_bf16_reference(monkeypatch):
    if not _supports_nvfp4_device(torch.device("cuda")):
        pytest.skip("NVFP4 Conv3d autotuning requires an SM100 or SM103 GPU")

    AutoTuner.get().clear_cache()
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
        original_run_tuned = nvfp4_conv_autotuner.run_tuned_fp4_conv

        def record_selected_tactic(**kwargs):
            output, tactic = original_run_tuned(**kwargs)
            selected_tactics.append(tactic)
            return output, tactic

        monkeypatch.setattr(
            nvfp4_conv_autotuner,
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


def test_fixed_fp4_conv_residual_tactic_matches_bf16_reference():
    if not _supports_nvfp4_device(torch.device("cuda")):
        pytest.skip("NVFP4 Conv3d requires an SM100 or SM103 GPU")

    AutoTuner.get().clear_cache()
    try:
        _fp4_compile_cache.clear()
        conv, activation, residual, expected = _make_fp4_conv_case(256, 1.0 / 50.0, True)
        with torch.inference_mode():
            actual = conv(activation, residual=residual)
        _assert_fp4_close(actual, expected)
    finally:
        AutoTuner.get().clear_cache()


@pytest.mark.parametrize("fuse_norm", [False, True])
def test_fixed_fp4_conv_input_fusions_match_bf16_reference(fuse_norm):
    """Exercise the fused SiLU and RMSNorm+SiLU paths through the Conv3d wrapper."""
    if not _supports_nvfp4_device(torch.device("cuda")):
        pytest.skip("NVFP4 Conv3d requires an SM100 or SM103 GPU")

    channels = 128
    torch.manual_seed(11)
    base = WanCausalConv3d(channels, channels, 3, padding=1).cuda().to(torch.bfloat16).eval()
    activation = torch.randn(
        (1, channels, 1, 4, 6),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gamma = torch.randn((channels,), device="cuda", dtype=torch.bfloat16)
    norm_scale = channels**0.5
    fused_input = activation
    if fuse_norm:
        fused_input = (
            F.normalize(fused_input.float(), dim=1).to(torch.bfloat16)
            * norm_scale
            * gamma.view(1, -1, 1, 1, 1)
        )
    fused_input = F.silu(fused_input)
    expected = F.conv3d(F.pad(fused_input, (1, 1, 1, 1, 2, 0)), base.weight, base.bias)
    conv = (
        NVFP4WanCausalConv3d(
            base,
            input_scale=1.0 / 50.0,
            absorb_silu=True,
            absorb_norm=fuse_norm,
            norm_gamma=gamma if fuse_norm else None,
            norm_scale=norm_scale if fuse_norm else None,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )

    with torch.inference_mode():
        actual = conv(activation)
    _assert_fp4_close(actual, expected)
