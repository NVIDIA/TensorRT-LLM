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
"""The side-effect-free prefix probe that ranks queued context requests."""

from types import SimpleNamespace

import pytest
from test_mooncake_store_connector import TOKENS_PER_BLOCK as TPB
from test_mooncake_store_connector import make_request, request_data, store_config  # noqa: F401
from test_mooncake_store_connector import make_scheduler as make_leader

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import SchedulerOutput
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import scheduler as mcs

pytestmark = pytest.mark.cpu_only


# ---- Mooncake leader ----


@pytest.mark.parametrize("local", [0, TPB])
def test_probe_answers_what_the_lookup_offers(store_config, local):  # noqa: F811
    tokens = list(range(5 * TPB))
    probed_leader = make_leader(store_config, hit_blocks=3)
    looked_leader = make_leader(store_config, hit_blocks=3)
    probed = probed_leader.probe_num_matched_tokens(make_request(1, tokens), local)
    offered, _ = looked_leader.get_num_new_matched_tokens(make_request(1, tokens), local)
    assert probed == offered > 0
    assert probed_leader._worker.queries == looked_leader._worker.queries
    assert probed_leader._requests == {}


def test_probe_declines_an_unaligned_local_match(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=3)
    assert leader.probe_num_matched_tokens(make_request(1, list(range(5 * TPB))), TPB + 1) == 0
    assert leader._worker.queries == []


def test_probe_never_offers_the_whole_prompt(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=99)
    assert leader.probe_num_matched_tokens(make_request(1, list(range(3 * TPB))), 0) == 2 * TPB


def test_probe_with_nothing_past_the_local_match_skips_the_store(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=99)
    assert leader.probe_num_matched_tokens(make_request(1, list(range(5 * TPB))), 4 * TPB) == 0
    assert leader._worker.queries == []


def test_lookup_adopts_the_probe_chain_and_behaves_as_without_it(store_config):  # noqa: F811
    tokens = list(range(5 * TPB))
    leader = make_leader(store_config, hit_blocks=2)
    request = make_request(1, tokens)
    assert leader.probe_num_matched_tokens(request, 0) == 2 * TPB
    chain = leader._probe_chains[1]
    assert leader.get_num_new_matched_tokens(request, 0)[0] == 2 * TPB
    assert leader._requests[1].chain is chain
    assert 1 not in leader._probe_chains
    output = SchedulerOutput(new_requests=[request_data(1, tokens, [10, 11, 12, 13, 14])])
    metadata = leader.build_connector_meta(output)
    assert [page.page_index for page in metadata.loads[0].pages] == [10, 11]
    assert [page.page_index for page in metadata.saves[0].pages] == [12, 13, 14]


def test_probe_after_a_lookup_leaves_the_offer_alone(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=2)
    request = make_request(1, list(range(5 * TPB)))
    leader.get_num_new_matched_tokens(request, 0)
    state = leader._requests[1]
    assert leader.probe_num_matched_tokens(request, 0) == 2 * TPB
    assert (state.load_first_block, state.load_blocks) == (0, 2)
    assert leader._probe_chains == {}


@pytest.mark.parametrize("forget", ["request_finished", "request_reset"])
def test_finished_or_reset_request_drops_its_probe_chain(store_config, forget):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=2)
    request = make_request(1, list(range(5 * TPB)))
    leader.probe_num_matched_tokens(request, 0)
    if forget == "request_finished":
        assert leader.request_finished(request, []) is False
    else:
        leader.request_reset(request)
    assert leader._probe_chains == {}


def test_probe_chain_cache_is_bounded(store_config, monkeypatch):  # noqa: F811
    monkeypatch.setattr(mcs, "PROBE_CHAIN_CACHE_SIZE", 2)
    leader = make_leader(store_config, hit_blocks=2)
    for request_id in (1, 2, 3):
        leader.probe_num_matched_tokens(make_request(request_id, list(range(5 * TPB))), 0)
    assert list(leader._probe_chains) == [2, 3]


def test_probe_skips_a_request_no_key_may_name(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=2)
    request = make_request(1, list(range(5 * TPB)), lora_task_id=7)
    assert leader.probe_num_matched_tokens(request, 0) == 0
    assert leader._worker.queries == [] and leader._probe_chains == {}


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(capacity_only=True, role=SimpleNamespace(loads=True)),
        SimpleNamespace(capacity_only=False, role=SimpleNamespace(loads=False)),
    ],
)
def test_probe_answers_zero_for_a_role_that_cannot_load(store_config, config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=2)
    leader._config = config
    assert leader.probe_num_matched_tokens(make_request(1, list(range(5 * TPB))), 0) == 0
    assert leader._worker.queries == [] and leader._probe_chains == {}
