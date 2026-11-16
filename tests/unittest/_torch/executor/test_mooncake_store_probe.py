# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-effect-free store probe of the connector manager and the Mooncake leader."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_mooncake_store_connector import TOKENS_PER_BLOCK as TPB
from test_mooncake_store_connector import make_request, request_data, store_config  # noqa: F401
from test_mooncake_store_connector import make_scheduler as make_leader

from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector as kvc
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import SchedulerOutput
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import scheduler as mcs

pytestmark = pytest.mark.cpu_only


# ---- Mooncake leader ----


def _untouched(leader):
    return (
        leader._requests == {}
        and leader._async_boundary_offsets == {}
        and leader._stats.lookups == 0
        and leader._stats.probed == 0
    )


@pytest.mark.parametrize("local", [0, TPB, 2 * TPB - 1])
def test_probe_answers_what_the_lookup_offers(store_config, local):  # noqa: F811
    tokens = list(range(5 * TPB))
    probed_leader = make_leader(store_config, hit_blocks=3)
    looked_leader = make_leader(store_config, hit_blocks=3)
    probed = probed_leader.probe_num_matched_tokens(make_request(1, tokens), local)
    offered, _ = looked_leader.get_num_new_matched_tokens(make_request(1, tokens), local)
    assert probed == offered > 0
    assert probed_leader._worker.queries == looked_leader._worker.queries
    assert _untouched(probed_leader)


def test_probe_declines_an_unaligned_single_block_hit(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=1)
    tokens = list(range(5 * TPB))
    assert leader.probe_num_matched_tokens(make_request(1, tokens), 2 * TPB - 1) == 0
    assert _untouched(leader)


def test_probe_never_offers_the_whole_prompt(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=99)
    assert leader.probe_num_matched_tokens(make_request(1, list(range(3 * TPB))), 0) == 2 * TPB


def test_probe_with_nothing_past_the_local_match_skips_the_store(store_config):  # noqa: F811
    leader = make_leader(store_config, hit_blocks=99)
    tokens = list(range(5 * TPB))
    assert leader.probe_num_matched_tokens(make_request(1, tokens), 4 * TPB) == 0
    assert leader._worker.queries == []


def test_lookup_adopts_the_probe_chain_and_behaves_as_without_it(store_config):  # noqa: F811
    tokens = list(range(5 * TPB))
    leader = make_leader(store_config, hit_blocks=2)
    request = make_request(1, tokens)
    assert leader.probe_num_matched_tokens(request, 0) == 2 * TPB
    chain = leader._probe_chains[1]
    assert leader.get_num_new_matched_tokens(request, 0) == (2 * TPB, leader._async_load)
    assert leader._requests[1].chain is chain
    assert 1 not in leader._probe_chains
    output = SchedulerOutput(new_requests=[request_data(1, tokens, [10, 11, 12, 13, 14])])
    metadata = leader.build_connector_meta(output)
    assert [page.page_index for page in metadata.loads[0].pages] == [10, 11]
    assert [page.page_index for page in metadata.saves[0].pages] == [12, 13, 14]


def test_probe_after_a_lookup_leaves_the_offer_alone(store_config):  # noqa: F811
    tokens = list(range(5 * TPB))
    leader = make_leader(store_config, hit_blocks=2)
    request = make_request(1, tokens)
    leader.get_num_new_matched_tokens(request, 0)
    state = leader._requests[1]
    lookups = leader._stats.lookups
    assert leader.probe_num_matched_tokens(request, 0) == 2 * TPB
    assert (state.load_first_block, state.load_blocks) == (0, 2)
    assert leader._stats.lookups == lookups
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


# ---- connector manager ----


def _manager(scheduler, *, adp=True, capacity_only=False):
    """The manager's probe path, without the C++ base or its MPI setup."""
    manager = SimpleNamespace(
        scheduler=scheduler, enable_attention_dp=adp, capacity_only=capacity_only
    )
    manager._run_on_leader = lambda f: kvc.KvCacheConnectorManager._run_on_leader(manager, f)
    return manager


def _probe(manager, request, n=0):
    return kvc.KvCacheConnectorManager.probe_num_matched_tokens(manager, request, n)


def _request(dummy=False):
    return SimpleNamespace(request_id=1, is_dummy_request=dummy)


def test_manager_probe_runs_locally_under_attention_dp():
    scheduler = Mock()
    scheduler.probe_num_matched_tokens.return_value = 4096
    request = _request()
    assert _probe(_manager(scheduler), request, 128) == 4096
    scheduler.probe_num_matched_tokens.assert_called_once_with(request, 128)


def test_manager_probe_skips_dummy_and_capacity_only():
    scheduler = Mock()
    assert _probe(_manager(scheduler), _request(dummy=True)) is None
    assert _probe(_manager(scheduler, capacity_only=True), _request()) is None
    scheduler.probe_num_matched_tokens.assert_not_called()


def test_manager_probe_failure_answers_none():
    scheduler = Mock()
    scheduler.probe_num_matched_tokens.side_effect = RuntimeError("store down")
    assert _probe(_manager(scheduler), _request()) is None


def test_manager_probe_broadcasts_the_tp_leaders_answer(monkeypatch):
    sent = []
    monkeypatch.setattr(kvc, "mpi_rank", lambda: 0)
    monkeypatch.setattr(kvc, "mpi_broadcast", lambda obj, root=0: sent.append(obj) or obj)
    scheduler = Mock()
    scheduler.probe_num_matched_tokens.return_value = 512
    assert _probe(_manager(scheduler, adp=False), _request()) == 512
    assert sent == [512]


def test_manager_probe_on_a_tp_follower_takes_the_broadcast(monkeypatch):
    monkeypatch.setattr(kvc, "mpi_broadcast", lambda obj, root=0: 768 if obj is None else -1)
    assert _probe(_manager(None, adp=False), _request()) == 768


def test_manager_probe_failure_still_joins_the_broadcast(monkeypatch):
    sent = []
    monkeypatch.setattr(kvc, "mpi_rank", lambda: 0)
    monkeypatch.setattr(kvc, "mpi_broadcast", lambda obj, root=0: sent.append(obj) or obj)
    scheduler = Mock()
    scheduler.probe_num_matched_tokens.side_effect = RuntimeError("store down")
    assert _probe(_manager(scheduler, adp=False), _request()) is None
    assert sent == [None]


def test_connector_without_a_probe_answers_none():
    assert kvc.KvCacheConnectorScheduler.probe_num_matched_tokens(None, _request(), 0) is None
