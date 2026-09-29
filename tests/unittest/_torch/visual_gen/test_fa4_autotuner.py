# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FA4 tuning policy, exact cache identity and per-call dispatch contracts."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile, TuningConfig
from tensorrt_llm._torch.visual_gen.attention_backend import fa4_autotuner as fa4


@pytest.fixture
def runtime(monkeypatch):
    """Mock the optional FA4 dependency, never a timing or numerical result."""
    interface = ModuleType("flash_attn.cute.interface")
    interface._flash_attn_fwd = Mock(return_value=(torch.empty(0), torch.empty(0), "diagnostics"))
    utils = ModuleType("flash_attn.cute.utils")
    utils._get_disable_2cta_default = Mock(return_value=False)
    utils._get_use_clc_scheduler_default = Mock(return_value=False)
    cute = ModuleType("flash_attn.cute")
    cute.utils = utils
    build_info = ModuleType("flash_attn.cute._trtllm_build_info")
    build_info.BUILD_ID = ("4.0.0b19", "test-revision", "test-source", "test-patch")
    package = ModuleType("flash_attn")
    package.cute = cute
    for name, module in (
        ("flash_attn", package),
        ("flash_attn.cute._trtllm_build_info", build_info),
        ("flash_attn.cute", cute),
        ("flash_attn.cute.interface", interface),
        ("flash_attn.cute.utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(fa4, "_require_tuning_api", lambda: None)
    monkeypatch.setattr(fa4, "_device_identity", lambda device: ("B200", (10, 0)))
    monkeypatch.setattr(fa4, "_cutlass_version", lambda: "4.8.0.dev0")
    return interface._flash_attn_fwd, utils


def _inputs(seq: int = 512, dtype: torch.dtype = torch.bfloat16) -> list[torch.Tensor]:
    return [torch.empty((1, seq, 2, 128), dtype=dtype) for _ in range(3)]


def test_fallback_keeps_auto_split_and_passes_no_tuning_arguments(runtime) -> None:
    kernel, _ = runtime
    inputs = _inputs()
    out, lse = fa4.Fa4Runner(inputs, 0.125)(inputs)
    assert out is kernel.return_value[0] and lse is kernel.return_value[1]
    assert kernel.call_args.kwargs == {
        "softmax_scale": 0.125,
        "causal": False,
        "softcap": 0.0,
        "return_lse": True,
        "num_splits": 0,
    }


def test_each_tactic_is_per_call_and_keeps_lse(runtime) -> None:
    kernel, utils = runtime
    inputs = _inputs()
    runner = fa4.Fa4Runner(inputs, 0.125)
    for tactic, (cta, freq) in enumerate(fa4._TACTICS):
        runner(inputs, tactic=tactic)
        assert kernel.call_args.kwargs["use_2cta"] == cta
        assert kernel.call_args.kwargs["ex2_emu_freq"] == freq
        assert kernel.call_args.kwargs["num_splits"] == 1
        assert kernel.call_args.kwargs["return_lse"] is True
    assert utils._get_disable_2cta_default() is False


@pytest.mark.parametrize("disabled,seq", [(True, 512), (False, 256)])
def test_prunes_disabled_or_ineligible_2cta(runtime, disabled: bool, seq: int) -> None:
    _, utils = runtime
    utils._get_disable_2cta_default.return_value = disabled
    inputs = _inputs(seq)
    tactics = fa4.Fa4Runner(inputs, 0.125).get_valid_tactics(inputs, OptimizationProfile())
    assert tactics[0] == -1
    assert all(not fa4._TACTICS[t][0] for t in tactics if t != -1)


def test_sm103_does_not_search_emulated_exp2(runtime, monkeypatch) -> None:
    monkeypatch.setattr(fa4, "_device_identity", lambda device: ("B300", (10, 3)))
    inputs = _inputs()
    tactics = fa4.Fa4Runner(inputs, 0.125).get_valid_tactics(inputs, OptimizationProfile())
    assert {fa4._TACTICS[t] for t in tactics if t != -1} == {
        (False, None),
        (False, 0),
        (True, None),
        (True, 0),
    }


def test_cache_identity_separates_runtime_and_layout(runtime, monkeypatch) -> None:
    _, utils = runtime
    inputs = _inputs()
    baseline = fa4.Fa4Runner(inputs, 0.125).unique_id()
    assert fa4.Fa4Runner(inputs, 0.25).unique_id() != baseline
    assert fa4.Fa4Runner(_inputs(dtype=torch.float16), 0.125).unique_id() != baseline
    strided = [torch.empty((1, 512, 4, 128), dtype=torch.bfloat16)[:, :, ::2] for _ in range(3)]
    assert fa4.Fa4Runner(strided, 0.125).unique_id() != baseline
    utils._get_disable_2cta_default.return_value = True
    assert fa4.Fa4Runner(inputs, 0.125).unique_id() != baseline
    utils._get_disable_2cta_default.return_value = False
    monkeypatch.setattr(fa4, "_device_identity", lambda device: ("B300", (10, 3)))
    assert fa4.Fa4Runner(inputs, 0.125).unique_id() != baseline


def test_real_autotuner_cache_reuse_and_unseen_shape_fallback(runtime, monkeypatch) -> None:
    # Exercise the real cache and dispatch, with GPU identity mocked for this CPU test.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "B200")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("TLLM_PROFILING_TIMER", "cuda_event")
    monkeypatch.setattr(AutoTuner, "_instance", None)
    tuner = AutoTuner.get()
    inputs = _inputs()
    runner = fa4.Fa4Runner(inputs, 0.125)
    config = TuningConfig()
    key = tuner.profiling_cache.get_cache_key(
        "visual_gen::fa4_dense", runner, tuple(t.shape for t in inputs), config, False
    )
    # A synthetic cache entry tests routing only; no performance is measured here.
    tuner.profiling_cache[key] = (0, 2, 1.0)
    serialized = tuner.profiling_cache._serialize_cache_data(tuner.profiling_cache.cache)
    tuner.profiling_cache.cache = tuner.profiling_cache._deserialize_cache_data(serialized)
    kernel, _ = runtime
    fa4.tuned_forward(*inputs, 0.125)
    assert kernel.call_args.kwargs["ex2_emu_freq"] == 4
    fa4.tuned_forward(*_inputs(768), 0.125)
    assert "ex2_emu_freq" not in kernel.call_args.kwargs
    assert kernel.call_args.kwargs["num_splits"] == 0


def test_real_autotuner_selects_and_reuses_winner(runtime, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "B200")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv("TLLM_PROFILING_TIMER", "cuda_event")
    monkeypatch.setattr(AutoTuner, "_instance", None)
    tuner = AutoTuner.get()
    # Synthetic timings exercise selection; these are not benchmark measurements.
    profile = Mock(side_effect=lambda **kw: 1.0 if kw["tactic"] == 3 else 2.0)
    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    tuner.is_tuning_mode = True
    inputs = _inputs()
    fa4.tuned_forward(*inputs, 0.125)
    assert profile.call_count == len(fa4._TACTICS) + 1
    kernel, _ = runtime
    assert kernel.call_args.kwargs["ex2_emu_freq"] == 8
    profile.reset_mock()
    tuner.is_tuning_mode = False
    fa4.tuned_forward(*inputs, 0.125)
    profile.assert_not_called()
    assert kernel.call_args.kwargs["ex2_emu_freq"] == 8


def test_never_profiles_during_outer_capture(runtime, monkeypatch) -> None:
    tuner = Mock(is_tuning_mode=True)
    monkeypatch.setattr(AutoTuner, "get", lambda: tuner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fa4.tuned_forward(*_inputs(), 0.125)
    tuner.choose_one.assert_not_called()


def test_cpu_and_causal_inputs_are_ineligible() -> None:
    inputs = _inputs()
    assert not fa4.can_tune(*inputs, causal=False)
    assert not fa4.can_tune(*inputs, causal=True)


def test_cache_identity_separates_patched_fa4_builds(runtime, monkeypatch) -> None:
    inputs = _inputs()
    runner = fa4.Fa4Runner(inputs, 0.125)
    original = runner.unique_id()
    build_info = sys.modules["flash_attn.cute._trtllm_build_info"]
    monkeypatch.setattr(build_info, "BUILD_ID", (*build_info.BUILD_ID, "new-patch"))
    assert runner.unique_id() != original
