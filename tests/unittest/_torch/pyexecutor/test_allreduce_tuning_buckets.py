# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only integration checks using the real autotuner and native bucket operators."""

from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner, DynamicTensorSpec, TuningConfig
from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def tuner(monkeypatch: pytest.MonkeyPatch) -> Iterator[AutoTuner]:
    monkeypatch.setenv("TLLM_PROFILING_TIMER", "cuda_event")
    original = AllReduceRunner.tuning_config
    original_buckets = torch.ops.trtllm.get_allreduce_tuning_buckets()
    AutoTuner._find_nearest_profile.cache_clear()
    try:
        yield AutoTuner()
    finally:
        torch.ops.trtllm.configure_allreduce_tuning_buckets(original_buckets[-1])
        AllReduceRunner.tuning_config = original
        AutoTuner._find_nearest_profile.cache_clear()


def make_runner() -> AllReduceRunner:
    return AllReduceRunner(2, [0, 1], torch.bfloat16, 0, 1e-6, False)


def test_python_lookup_uses_native_mapper(tuner: AutoTuner) -> None:
    AllReduceRunner.configure_tuning_buckets(40000)
    config = AllReduceRunner.tuning_config
    spec = config.dynamic_tensor_specs[0]
    assert spec.map_to_tuning_buckets is torch.ops.trtllm.map_to_allreduce_tuning_bucket.default
    for tokens, expected in ((0, -1), (1, 1), (32769, 32768), (40000, 40000), (40001, -1)):
        shape = AutoTuner._find_nearest_profile(
            (torch.Size((tokens, 7168)),),
            config.dynamic_tensor_specs,
            (),
            config.tune_max_num_tokens,
        )
        assert shape == ((expected, 7168),)
        assert expected == torch.ops.trtllm.map_to_allreduce_tuning_bucket(tokens)


def test_equivalent_specs_reuse_python_lookup_cache(tuner: AutoTuner) -> None:
    AllReduceRunner.configure_tuning_buckets(40000)
    config = AllReduceRunner.tuning_config
    buckets = config.dynamic_tensor_specs[0].gen_tuning_buckets
    mapper = config.dynamic_tensor_specs[0].map_to_tuning_buckets
    for _ in range(100):
        spec = DynamicTensorSpec(0, 0, buckets, mapper)
        assert AutoTuner._find_nearest_profile(
            (torch.Size((32769, 7168)),), (spec,), (), 40000
        ) == ((32768, 7168),)
    assert AutoTuner._find_nearest_profile.cache_info().currsize == 1
    assert AutoTuner._find_nearest_profile.cache_info().hits == 99


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(0, ()), (33000, (32768, 33000)), (40000, (40000,)), (40001, ())],
)
def test_skip_sweep_profiles_follow_native_buckets(
    tuner: AutoTuner, tokens: int, expected: tuple[int, ...]
) -> None:
    AllReduceRunner.configure_tuning_buckets(40000)
    tuner.skip_dynamic_tuning_buckets = True
    inputs = [torch.empty((tokens, 7168), device="meta")]
    profiles = tuner._optimization_profiles(AllReduceRunner.tuning_config, inputs)
    assert tuple(profile.get_opt_shapes()[0][0] for profile in profiles) == expected


@pytest.mark.parametrize("invalid_spec_first", [True, False])
def test_negative_required_dimension_suppresses_entire_skip_sweep(
    tuner: AutoTuner, invalid_spec_first: bool
) -> None:
    tuner.skip_dynamic_tuning_buckets = True
    invalid = DynamicTensorSpec(0, 0, (1, 2), lambda value: -1)
    valid = DynamicTensorSpec(0, 1, (1, 8), lambda value: value)
    specs = (invalid, valid) if invalid_spec_first else (valid, invalid)
    config = TuningConfig(dynamic_tensor_specs=specs)
    assert tuner._optimization_profiles(config, [torch.empty((3, 8))]) == []


def test_full_sweep_respects_ceiling_without_changing_inputs(tuner: AutoTuner) -> None:
    AllReduceRunner.configure_tuning_buckets(40000)
    tensor = torch.empty((100000, 7168), device="meta")
    profiles = tuner._optimization_profiles(AllReduceRunner.tuning_config, [tensor])
    assert tuple(profile.get_opt_shapes()[0][0] for profile in profiles) == tuple(
        torch.ops.trtllm.get_allreduce_tuning_buckets()
    )
    assert tensor.shape == (100000, 7168)


def test_runner_registers_only_measured_entries(tuner: AutoTuner) -> None:
    AllReduceRunner.configure_tuning_buckets(40000)
    runner = make_runner()
    inputs = [torch.empty((40000, 16)), None, None, None, None, None]
    op = "trtllm::tunable_allreduce::allreduce"
    shapes = tuple(tuner._get_input_sizes(inputs))
    cache = tuner.profiling_cache
    identity = runner.unique_id()
    with (
        patch.object(AutoTuner, "get", return_value=tuner),
        patch.object(torch.ops.trtllm, "register_allreduce_tactic") as register,
    ):
        runner._register_native_tactics(inputs)
        register.assert_not_called()
        for bucket, tactic in ((8192, 7), (40000, 8)):
            bucket_shapes = (torch.Size((bucket, 16)), *shapes[1:])
            key = cache.get_cache_key(
                op, runner, bucket_shapes, runner.tuning_config, apply_map_to_tuning_buckets=False
            )
            cache.cache[key] = (0, tactic, 1.0)
        runner._register_native_tactics(inputs)
        assert [(call.args[-2], call.args[-1]) for call in register.call_args_list] == [
            (8192, 7),
            (40000, 8),
        ]
        register.reset_mock()
        AllReduceRunner.configure_tuning_buckets(65536)
        assert runner.unique_id() != identity
        runner._register_native_tactics(inputs)
        register.assert_not_called()


def test_standalone_runner_initializes_configuration_once(tuner: AutoTuner) -> None:
    empty_config = replace(AllReduceRunner.tuning_config, dynamic_tensor_specs=())
    with (
        patch.object(AllReduceRunner, "tuning_config", empty_config),
        patch.object(
            torch.ops.trtllm, "get_allreduce_tuning_buckets", return_value=[1, 2, 4]
        ) as get_buckets,
    ):
        make_runner()
        make_runner()
        get_buckets.assert_called_once_with()
        assert AllReduceRunner.tuning_config.tune_max_num_tokens == 4
