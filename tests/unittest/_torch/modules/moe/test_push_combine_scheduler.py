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
"""Host-only tests for the Push-Combine scheduler and factory integration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm._torch.modules.fused_moe.moe_scheduler as moe_scheduler
from tensorrt_llm._torch.modules.fused_moe.communication import (
    CommunicationFactory,
    NVLinkOneSided,
    NVLinkOneSidedTilePushCombine,
)


def _scheduler_with_comm(comm):
    scheduler = moe_scheduler.ExternalCommMoEScheduler.__new__(
        moe_scheduler.ExternalCommMoEScheduler
    )
    scheduler.moe = SimpleNamespace(comm=comm)
    return scheduler


@pytest.mark.parametrize(
    ("comm_cls", "is_tuning_mode", "has_dist", "expect_barrier"),
    [
        pytest.param(NVLinkOneSidedTilePushCombine, True, True, True, id="push-distributed-tuning"),
        pytest.param(NVLinkOneSidedTilePushCombine, False, True, False, id="push-inference"),
        pytest.param(NVLinkOneSidedTilePushCombine, True, False, False, id="push-local-tuning"),
        pytest.param(NVLinkOneSided, True, True, False, id="onesided-distributed-tuning"),
    ],
)
def test_push_combine_peers_are_aligned_only_while_tuning(
    monkeypatch, comm_cls, is_tuning_mode, has_dist, expect_barrier
):
    dist = Mock()
    tuner = SimpleNamespace(is_tuning_mode=is_tuning_mode, _dist=dist if has_dist else None)
    monkeypatch.setattr(moe_scheduler.AutoTuner, "get", lambda: tuner)

    _scheduler_with_comm(Mock(spec=comm_cls))._align_ep_peers_while_tuning()

    if expect_barrier:
        dist.barrier.assert_called_once_with()
    else:
        dist.barrier.assert_not_called()


def _model_config(**overrides):
    mapping = SimpleNamespace(
        enable_attention_dp=True,
        dp_size=4,
        moe_tp_size=1,
        moe_ep_size=4,
        world_size=4,
    )
    config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(hidden_size=7168),
        torch_dtype=None,
        quant_config=None,
        max_num_tokens=64,
        moe_max_num_tokens=None,
        use_cuda_graph=False,
        use_low_precision_moe_combine=False,
        moe_load_balancer=None,
        extra_attrs={},
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"use_low_precision_moe_combine": True}, "use_low_precision", id="fp8"),
        pytest.param({"moe_load_balancer": object()}, "load balancing", id="eplb"),
    ],
)
def test_forced_push_combine_rejects_unsupported_features(monkeypatch, overrides, match):
    monkeypatch.setenv("TRTLLM_FORCE_COMM_METHOD", "PUSH_COMBINE")
    with pytest.raises(ValueError, match=match):
        CommunicationFactory.create_strategy(
            _model_config(**overrides),
            num_experts=256,
            num_slots=256,
            top_k=8,
            expert_size_per_partition=64,
        )
