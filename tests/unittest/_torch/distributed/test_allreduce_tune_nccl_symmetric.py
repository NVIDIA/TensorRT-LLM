# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_AR_TUNE_NCCL_SYMMETRIC: the AllReduce tuner's candidate list with and without NCCL_SYMMETRIC."""

import pytest
import torch

from tensorrt_llm._torch.autotuner import OptimizationProfile
from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm.functional import AllReduceStrategy


def _runner(exclude: bool = False):
    return torch_custom_ops.AllReduceRunner(
        2, [0, 1], torch.bfloat16, 0, 1e-6, False, exclude_nccl_symmetric=exclude
    )


@pytest.mark.parametrize("rows", [8, 4096, 16384])
@pytest.mark.parametrize("tune_symmetric", [True, False])
def test_nccl_symmetric_candidate_follows_knob(
    monkeypatch, rows: int, tune_symmetric: bool
) -> None:
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", tune_symmetric)
    inputs = [torch.empty(rows, 6144, dtype=torch.bfloat16)]
    tactics = _runner().get_valid_tactics(inputs, OptimizationProfile())
    assert AllReduceStrategy.NCCL.value in tactics
    assert (AllReduceStrategy.NCCL_SYMMETRIC.value in tactics) is tune_symmetric
    # Off removes only NCCL_SYMMETRIC: the other candidates are unchanged.
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", True)
    stock = _runner().get_valid_tactics(inputs, OptimizationProfile())
    assert [t for t in stock if t != AllReduceStrategy.NCCL_SYMMETRIC.value] == [
        t for t in tactics if t != AllReduceStrategy.NCCL_SYMMETRIC.value
    ]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("exclude", [False, True])
@pytest.mark.parametrize(
    "cached",
    [
        AllReduceStrategy.NCCL_SYMMETRIC.value,
        AllReduceStrategy.NCCL.value,
        AllReduceStrategy.ONESHOT.value,
    ],
)
def test_cached_tactic_obeys_symmetric_policy(monkeypatch, enabled, exclude, cached):
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", enabled)
    seen = []

    def raw(
        input_tensor,
        residual,
        norm_weight,
        scale,
        bias,
        workspace,
        group,
        tactic,
        op,
        eps,
        trigger_completion_at_end,
    ):
        seen.append(tactic)
        return input_tensor

    monkeypatch.setattr(torch.ops.trtllm, "allreduce", raw)
    x = torch.empty(2, 16, dtype=torch.bfloat16)
    assert _runner(exclude).forward([x, None, None, None, None, None], tactic=cached) is x
    forbidden = cached == AllReduceStrategy.NCCL_SYMMETRIC.value and (exclude or not enabled)
    assert seen == [AllReduceStrategy.NCCL.value if forbidden else cached]


def test_environment_policy_separates_cache_identity(monkeypatch):
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", True)
    enabled = _runner().unique_id()
    explicit_exclusion = _runner(True).unique_id()
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", False)
    disabled = _runner().unique_id()
    assert disabled != enabled
    assert disabled == explicit_exclusion


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("exclude", [False, True])
def test_native_registration_rejects_cached_symmetric_tactic(monkeypatch, enabled, exclude):
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", enabled)
    runner = _runner(exclude)
    tuner = torch_custom_ops.AutoTuner.get()
    cache = tuner.profiling_cache
    monkeypatch.setattr(cache, "cache", {})
    inputs = [torch.empty(2, 16, dtype=torch.bfloat16), None, None, None, None, None]
    shapes = tuple(tuner._get_input_sizes(inputs))
    buckets = runner.tuning_config.dynamic_tensor_specs[0].gen_tuning_buckets
    for bucket in buckets:
        bucket_shapes = (torch.Size((bucket, *shapes[0][1:])), *shapes[1:])
        key = cache.get_cache_key(
            "trtllm::tunable_allreduce::allreduce",
            runner,
            bucket_shapes,
            runner.tuning_config,
            apply_map_to_tuning_buckets=False,
        )
        cache.cache[key] = (0, AllReduceStrategy.NCCL_SYMMETRIC.value, 0.0)
    registrations = []

    def register(
        input_tensor, residual, norm_weight, scale, bias, workspace, group, op, bucket, tactic
    ):
        registrations.append((bucket, tactic))

    monkeypatch.setattr(torch.ops.trtllm, "register_allreduce_tactic", register)
    runner._register_native_tactics(inputs)
    expected = (
        AllReduceStrategy.NCCL_SYMMETRIC.value
        if enabled and not exclude
        else AllReduceStrategy.NCCL.value
    )
    assert registrations == [(bucket, expected) for bucket in buckets]
