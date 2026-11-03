# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TLLM_MOE_TUNE_UP_TO_MAX_NUM_TOKENS: the trtllm-gen MoE runners tune only the
token buckets a forward of the engine can look up."""

import pytest

from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops
from tensorrt_llm._torch.utils import get_last_power_of_2_num_tokens_buckets

ENV = "TLLM_MOE_TUNE_UP_TO_MAX_NUM_TOKENS"
RUNNERS = [
    ops.FP4BlockScaleMoERunner,
    ops.FP8BlockScaleMoERunner,
    ops.MxE4m3MxE2m1BlockScaleMoERunner,
    ops.E4m3MxE2m1BlockScaleMoERunner,
    ops.Bf16MxE2m1BlockScaleMoERunner,
    ops.FP8FP4BlockScaleMoERunner,
]
FULL_LADDER = tuple(get_last_power_of_2_num_tokens_buckets(8192))


def _spec(runner, ep_size, max_num_tokens, use_dp):
    (spec,) = runner.get_dynamic_tensor_specs(ep_size, max_num_tokens, use_dp)
    return spec


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
def test_full_ladder_by_default(monkeypatch, runner):
    monkeypatch.delenv(ENV, raising=False)
    assert tuple(_spec(runner, 8, 32, False).gen_tuning_buckets) == FULL_LADDER


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda r: r.__name__)
@pytest.mark.parametrize("max_num_tokens", [32, 40, 64, 8192])
@pytest.mark.parametrize("use_dp,ep_size", [(False, 8), (True, 16)])
def test_capped_buckets_cover_every_runtime_m(monkeypatch, runner, max_num_tokens, use_dp, ep_size):
    monkeypatch.setenv(ENV, "1")
    spec = _spec(runner, ep_size, max_num_tokens, use_dp)
    buckets = set(spec.gen_tuning_buckets)
    assert max(buckets) <= max_num_tokens
    for tokens in range(1, max_num_tokens + 1):
        # Under attention DP the runner sees the A2A pool, ep_size x the per-rank tokens.
        m = tokens * ep_size if use_dp else tokens
        assert spec.map_to_tuning_buckets(m) in buckets, (tokens, m)


def test_cap_keeps_a_larger_engine_bucket(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert ops._moe_tuning_m_values(16384, 8192) == FULL_LADDER + (16384,)
    assert ops._moe_tuning_m_values(32, 8192) == (1, 2, 4, 8, 16, 32)
