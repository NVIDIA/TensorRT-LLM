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
"""KV cache estimation without a host tier (TRTLLM_KV_ESTIMATION_NO_HOST_TIER).

The estimation pass builds throwaway probe managers. Without an explicit host
budget they size a host tier on their own, so the switch hands them a host
budget of 0 instead. The real build that follows is untouched.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

pytestmark = pytest.mark.cpu_only

GB = 1 << 30
SWITCH = "TRTLLM_KV_ESTIMATION_NO_HOST_TIER"


@pytest.fixture(autouse=True)
def clean_switch(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)


def make_creator(*, host_cache_size, with_draft: bool) -> KvCacheCreator:
    """A creator whose manager constructors are recorders, with or without a one-model draft pool."""
    c = object.__new__(KvCacheCreator)
    c._kv_cache_config = KvCacheConfig(
        max_gpu_total_bytes=10 * GB,
        host_cache_size=host_cache_size,
    )
    c._tokens_per_block = 64
    c._max_seq_len = 1024
    c._max_num_tokens = 0
    c._max_batch_size = 1
    c._max_beam_width = 1
    c._is_disagg = False
    c._cache_transceiver_config = None
    c._speculative_config = None
    c._mapping = Mock()
    c._draft_config = None
    c._draft_model_engine = None
    c._model_engine = Mock(cuda_graph_batch_sizes=[], cuda_graph_padding_enabled=False)
    c._model_engine.model.model_config.is_encoder_decoder = False
    c._llm_args = SimpleNamespace(kv_cache_compression_config=None, sleep_config=None)
    c._disable_overlap_scheduler = False
    c._skip_est = False
    c._kv_connector_manager = None
    c._is_kv_cache_manager_v2 = True
    c._fp8_ctx_mla_kv_len_cap = None

    c._kv_cache_manager_cls = Mock()
    c._kv_cache_manager_cls.get_cache_size_per_token = Mock(return_value=(80, 0))
    c._get_kv_size_per_token = Mock(return_value=CacheCost(slope=100, intercept=0))
    c._get_draft_cache_cost = Mock(return_value=CacheCost(slope=20, intercept=0))
    c._should_create_separate_draft_kv_cache = Mock(return_value=with_draft)
    c._needs_gpu_kv_cache_budget_split = Mock(return_value=True)
    c._joint_reuse_supported = Mock(return_value=False)
    c._warn_if_admitted_concurrency_outruns_graph_capture = Mock()
    c._create_kv_cache_manager = Mock(return_value=Mock())
    c._create_one_model_draft_kv_cache_manager = Mock(return_value=Mock())
    return c


def target_config(c):
    return c._create_kv_cache_manager.call_args.kwargs["kv_cache_config_override"]


def draft_config(c):
    return c._create_one_model_draft_kv_cache_manager.call_args.kwargs["kv_cache_config_override"]


@pytest.mark.parametrize("host_cache_size", [None, 20 * GB])
def test_switch_gives_the_estimation_managers_no_host_tier(monkeypatch, host_cache_size):
    monkeypatch.setenv(SWITCH, "1")
    c = make_creator(host_cache_size=host_cache_size, with_draft=True)

    c.build_managers({}, estimating_kv_cache=True)

    assert target_config(c).host_cache_size == 0
    assert draft_config(c).host_cache_size == 0
    # The GPU side of the probe is unchanged.
    assert target_config(c).max_gpu_total_bytes == 10 * GB


def test_switch_without_a_draft_pool(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    c = make_creator(host_cache_size=None, with_draft=False)

    c.build_managers({}, estimating_kv_cache=True)

    assert target_config(c).host_cache_size == 0
    c._create_one_model_draft_kv_cache_manager.assert_not_called()


@pytest.mark.parametrize("env", [{}, {SWITCH: "0"}])
def test_without_the_switch_the_probes_size_their_own_host_tier(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    c = make_creator(host_cache_size=20 * GB, with_draft=True)

    c.build_managers({}, estimating_kv_cache=True)

    # Estimation drops the explicit budget; an unset size asks for the auto host tier.
    assert target_config(c).host_cache_size is None
    assert draft_config(c).host_cache_size is None


def test_the_real_build_keeps_its_host_tier(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    c = make_creator(host_cache_size=20 * GB, with_draft=True)

    c.build_managers({}, estimating_kv_cache=False)

    target_share = target_config(c).host_cache_size
    draft_share = draft_config(c).host_cache_size
    assert target_share > 0 and draft_share > 0
    assert target_share + draft_share == 20 * GB


def test_the_switch_does_not_change_the_users_config(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    c = make_creator(host_cache_size=20 * GB, with_draft=True)

    c.build_managers({}, estimating_kv_cache=True)

    assert c._kv_cache_config.host_cache_size == 20 * GB
