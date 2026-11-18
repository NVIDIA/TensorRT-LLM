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
"""Tests for the startup record, which replays rank 0's startup collectives on a replacement rank.

``MnnvlDist``'s host collectives run over a fake host exchange.
"""

import contextlib
import importlib
import os
import pickle
import resource
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tensorrt_llm._torch.autotuner import AutoTuner, DistributedTuningStrategy
from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.distributed.host_exchange import HostExchange, MnnvlDist
from tensorrt_llm._torch.distributed.membership import REPLACEMENT_CANNOT_START, startup
from tensorrt_llm._torch.distributed.startup_record import ReplacementCannotStart, StartupRecord
from tensorrt_llm._torch.models.checkpoints.hf import rank_striped_read_ahead
from tensorrt_llm._torch.pyexecutor import _util, py_executor
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import warmup
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _compute_auto_host_tier_quota,
    _raise_on_every_rank,
    _sync_host_tier_quota,
    get_host_memory_limits,
)
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm.llmapi.llm_args import CheckpointIOPolicy
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

WORLD = 4
REPLACED = 2
GIB = 1 << 30


class FakeHostExchange:
    """Fake the ``HostExchange`` of ``rank``, where peer ``r`` sends ``peers(r, obj)``."""

    def __init__(self, rank, active_mask=(1 << WORLD) - 1, peers=lambda r, obj: obj):
        self.rank, self.active_mask, self.peers = rank, active_mask, peers
        self.nranks = WORLD
        self.calls = 0

    def _sent(self, r, obj):
        return obj if r == self.rank else self.peers(r, obj)

    def exchange(self, payload, readers=None, recipients=None, idle=False):
        self.calls += 1
        obj = pickle.loads(payload)
        read = range(WORLD) if readers is None else readers
        return [
            pickle.dumps(self._sent(r, obj)) if r in read and self.active_mask >> r & 1 else None
            for r in range(WORLD)
        ]

    def allgather_array(self, row, active_only=False):
        self.calls += 1
        return np.stack(
            [
                np.asarray(self._sent(r, row), dtype=np.int64)
                if self.active_mask >> r & 1
                else np.zeros_like(row)
                for r in range(WORLD)
                if self.active_mask >> r & 1 or not active_only
            ]
        )


def _dist(rank, host_exchange):
    """Return the ``MnnvlDist`` of ``rank``, whose host exchange is ``host_exchange``."""
    dist = MnnvlDist(
        SimpleNamespace(
            rank=rank,
            tp_rank=rank,
            tp_size=WORLD,
            cp_size=1,
            world_size=WORLD,
            dwdp_enabled=False,
            # Every rank on one node.
            node_rank=0,
            get_node_rank=lambda rank: 0,
        )
    )
    with patch.object(HostExchange, "create", return_value=host_exchange):
        assert dist.host_exchange is host_exchange
    return dist


@pytest.fixture(autouse=True)
def fresh_record(monkeypatch):
    """Give each test an empty startup record, which ``Distributed`` replays.

    A test has an autotuner cache file only if it sets one.
    """
    monkeypatch.setattr(communicator.Distributed, "startup_record", StartupRecord())
    monkeypatch.delenv("TLLM_AUTOTUNER_CACHE_PATH", raising=False)


def _record() -> StartupRecord:
    """Return this process's startup record."""
    return communicator.Distributed.startup_record


def _rank_0(peers=lambda r, obj: obj):
    """Return rank 0's ``MnnvlDist`` and start recording its startup."""
    _record().record()
    return _dist(0, FakeHostExchange(0, peers=peers))


def _publish() -> bytes:
    """End rank 0's startup and return its record as the rendezvous store carries it."""
    _record().close()
    return pickle.dumps(_record().published())


def _replacement(published: bytes):
    """Return a replacement of rank ``REPLACED`` that replays ``published`` and exchanges alone."""
    record = StartupRecord()
    record.replay(pickle.loads(published))
    communicator.Distributed.startup_record = record
    return _dist(REPLACED, FakeHostExchange(REPLACED, active_mask=1 << REPLACED))


def _startup(dist, max_tokens):
    flags = dist.tp_allgather(1)
    tactics = dist.tp_broadcast({"gemm": 3} if dist.mapping.tp_rank == 0 else None, root=0)
    max_tokens = dist.allreduce(max_tokens, op=ReduceOp.MIN)
    rows = dist.tp_allgather_int64([8, 1])
    dist.barrier()
    return flags, tactics, max_tokens, rows.tolist()


def _fewer_tokens(r, obj):
    """Give peer r 100 * r tokens fewer than this rank."""
    return obj - 100 * r if isinstance(obj, int) and obj > 1 else obj


def test_replacement_replays_rank_0_results():
    expected = ([1] * WORLD, {"gemm": 3}, 4096 - 300, [[8, 1]] * WORLD)
    assert _startup(_rank_0(_fewer_tokens), 4096) == expected
    assert not _record().replacement
    published = _publish()
    replacement = _replacement(published)
    assert _record().replacement
    # The replacement has its predecessor's capacity of 4096 - 200, which the minimum bounds.
    assert _startup(replacement, 4096 - 200) == expected
    assert replacement.host_exchange.calls == 0
    _record().close()
    # Too small a capacity may grow once the predecessor's memory is free, so a restart may succeed.
    with pytest.raises(
        RuntimeError, match=f"collective 2 .* is {4096 - 400}, beyond .* min {4096 - 300}"
    ) as failure:
        _startup(_replacement(published), 4096 - 400)
    # The message names the reduction by its call site.
    assert "in _startup" in str(failure.value)


def test_active_only_gathers_skip_absent_ranks():
    """Check that the gathers of a vote return only the active ranks' values."""
    dist = _dist(1, FakeHostExchange(1, active_mask=0b0111))
    assert dist.tp_allgather(1) == [1, 1, 1, None]
    assert dist.tp_allgather(1, active_only=True) == [1, 1, 1]
    assert dist.tp_allgather_int64([5], active_only=True).tolist() == [[5]] * 3
    assert dist.allreduce(2) == 6


def test_failed_startup_leaves_no_pause():
    """Check that a new startup records from scratch after one that failed in its estimation."""
    _rank_0()
    communicator.Distributed.pause_startup_record()
    _record().agree("max_seq_len", lambda: 8, per_process=True)
    _record().record()
    assert _record().active
    assert _record().values == {}
    with pytest.raises(RuntimeError, match="without a matching pause"):
        _record().resume()


def test_nested_agreement_refused():
    """A gather that takes another startup value would make a replacement skip that value."""
    _rank_0()
    with pytest.raises(RuntimeError, match="sentinel_vote takes startup_group_split"):
        _record().agree(
            "sentinel_vote",
            lambda: _record().agree("startup_group_split", lambda: 1, per_process=True),
            per_process=True,
        )


def test_agreement_unrecorded_without_fault_tolerance():
    assert _record().agree("rank_hosts", lambda: ["a"] * WORLD) == ["a"] * WORLD
    assert _record().published() == {"values": {}, "exchanges": []}


def test_import_loads_no_membership(monkeypatch):
    """Import the startup record, which every executor imports, without the membership package."""
    prefix = "tensorrt_llm._torch.distributed."
    package = sys.modules[prefix[:-1]]
    for name in ("membership", "startup_record"):
        monkeypatch.setattr(package, name, getattr(package, name, None), raising=False)
    for name in list(sys.modules):
        if name in (f"{prefix}membership", f"{prefix}startup_record") or name.startswith(
            f"{prefix}membership."
        ):
            monkeypatch.delitem(sys.modules, name)
    importlib.import_module(f"{prefix}startup_record")
    assert f"{prefix}membership" not in sys.modules


def _allgather(obj):
    """Return every rank's pickled payload when rank r sends ``obj + r``."""
    return [pickle.dumps(obj + r) for r in range(WORLD)]


def _unreachable(*args):
    """Fail a test whose replacement rank runs a collective instead of replaying it."""
    raise AssertionError("A replacement rank ran a recorded collective")


def _startup_collectives(record, rank, max_tokens, sample, allgather):
    """Make an agreement, a minimum reduction and a rank-varying gather on ``record``."""
    agreed = record.agree("max_seq_len", lambda: allgather(8))
    reduced = record.exchange(
        rank, pickle.dumps(max_tokens), None, lambda: allgather(max_tokens), reduce_op=min
    )
    with record.rank_varying("KV cache load sample"):
        sampled = record.exchange(rank, pickle.dumps(sample), None, lambda: allgather(sample))
    return agreed, reduced, sampled


def _replacement_record():
    """Return the record of a replacement of rank ``REPLACED`` that replays rank 0's startup."""
    original = StartupRecord()
    original.record()
    _startup_collectives(original, 0, 100, 5, _allgather)
    original.close()
    record = StartupRecord()
    record.replay(pickle.loads(pickle.dumps(original.published())))
    return record


def test_replacement_replays_recorded_results():
    """Check the replayed results, an extra agreement, and an agreement after the replay."""
    record = _replacement_record()
    results = _startup_collectives(record, REPLACED, 102, 7, _unreachable)
    assert results == (_allgather(8), _allgather(100), _allgather(5))
    with pytest.raises(ReplacementCannotStart, match="max_seq_len 2 times"):
        record.agree("max_seq_len", _unreachable)
    record.close()
    assert record.agree("max_seq_len", lambda: 16) == 16


def test_replacement_own_input_must_match_predecessor():
    """Check a replacement's own input to an agreement against its predecessor's recorded one."""
    splits = [(0, r) for r in range(WORLD)]
    original = StartupRecord()
    original.record()
    original.agree("startup_group_split", lambda: splits, own=(0, splits[0]))
    original.close()
    matching, rank_varying, differing = StartupRecord(), StartupRecord(), StartupRecord()
    for record in (matching, rank_varying, differing):
        record.replay(original.published())
    own = (REPLACED, splits[REPLACED])
    assert matching.agree("startup_group_split", _unreachable, own=own) == splits
    with rank_varying.rank_varying("process identity"):
        assert rank_varying.agree("startup_group_split", _unreachable, own=(REPLACED, 0)) == splits
    with pytest.raises(
        ReplacementCannotStart,
        match=r"input of this replacement rank to the startup value startup_group_split is "
        r"\(1, 2\), its predecessor's \(0, 2\)",
    ):
        differing.agree("startup_group_split", _unreachable, own=(REPLACED, (1, REPLACED)))


def _diverging_startup_call(record):
    """Make a collective whose payload differs from the predecessor's."""
    record.exchange(REPLACED, pickle.dumps(101), None, _unreachable)


def test_divergence_names_the_diverging_call():
    original = StartupRecord()
    original.record()
    original.exchange(0, pickle.dumps(100), None, lambda: _allgather(100))
    original.close()
    record = StartupRecord()
    record.replay(original.published())
    with pytest.raises(ReplacementCannotStart, match="payload .* differs") as failure:
        _diverging_startup_call(record)
    message = str(failure.value)
    assert "The diverging call (most recent call last)" in message
    assert "in _diverging_startup_call" in message
    assert "distributed/startup_record.py" not in message


def test_fewer_collectives_fail_at_close():
    no_agreement, no_gather = _replacement_record(), _replacement_record()
    no_agreement.exchange(REPLACED, pickle.dumps(102), None, _unreachable, reduce_op=min)
    with no_agreement.rank_varying("KV cache load sample"):
        no_agreement.exchange(REPLACED, pickle.dumps(7), None, _unreachable)
    with pytest.raises(ReplacementCannotStart, match=r"values \{'max_seq_len': 0\} times"):
        no_agreement.close()
    no_gather.agree("max_seq_len", _unreachable)
    no_gather.exchange(REPLACED, pickle.dumps(102), None, _unreachable, reduce_op=min)
    with pytest.raises(ReplacementCannotStart, match=r"next one is .* \(KV cache load sample\)"):
        no_gather.close()


def test_pauses_nest():
    """Check that nested pauses record and replay only the per-process values."""
    original = StartupRecord()
    original.record()
    original.pause()
    original.pause()
    original.agree("estimation_tactics", lambda: 1)
    original.resume()
    assert not original.active
    original.agree("steady_clock_offset", lambda: 2, per_process=True)
    with patch.object(AutoTuner, "get", return_value=_new_autotuner()):
        original.resume()
    assert original.active
    original.agree("max_seq_len", lambda: 3)
    with pytest.raises(RuntimeError, match="without a matching pause"):
        original.resume()
    original.close()
    assert original.values == {"steady_clock_offset": [2], "max_seq_len": [3]}
    record = StartupRecord()
    record.replay(original.published())
    record.pause()
    assert not record.replaying
    assert record.agree("estimation_tactics", lambda: 4) == 4
    assert record.agree("steady_clock_offset", _unreachable, per_process=True) == 2
    record.resume()
    assert record.agree("max_seq_len", _unreachable) == 3
    record.close()


def test_unmarked_rank_varying_gather_stops_replacement():
    record = _replacement_record()
    record.exchange(REPLACED, pickle.dumps(102), None, _unreachable, reduce_op=min)
    with pytest.raises(ReplacementCannotStart, match=r"ranks' .* \(KV cache load sample\)"):
        record.exchange(REPLACED, pickle.dumps(7), None, _unreachable)


def _calibrate_steady_clock(dist, now, monkeypatch):
    """Run the steady clock calibration of an executor of ``dist`` whose clock reads ``now``."""
    monkeypatch.setattr(py_executor, "get_steady_clock_now_in_seconds", lambda: now)
    executor = SimpleNamespace(global_rank=dist.mapping.rank, dist=dist)
    py_executor.PyExecutor._set_global_steady_clock_offset(executor)
    return py_executor.LlmRequest.global_steady_clock_offset


def test_replacement_replays_paused_clock_calibration(monkeypatch):
    """Check a KV cache estimation calibration on the original ranks and on a later replacement."""
    monkeypatch.setattr(py_executor, "LlmRequest", SimpleNamespace(global_steady_clock_offset=None))
    rank_0 = _rank_0(lambda r, obj: obj + r if isinstance(obj, float) else obj)
    communicator.Distributed.pause_startup_record()
    assert _calibrate_steady_clock(rank_0, 100.0, monkeypatch) == 0.0
    with patch.object(AutoTuner, "get", return_value=_new_autotuner()):
        communicator.Distributed.resume_startup_record()
    # The final executor of an original rank keeps the offset of its first one.
    assert _calibrate_steady_clock(rank_0, 200.0, monkeypatch) == 0.0
    published = _publish()
    py_executor.LlmRequest.global_steady_clock_offset = None
    replacement = _replacement(published)
    assert _calibrate_steady_clock(replacement, 500.0, monkeypatch) == -REPLACED
    assert replacement.host_exchange.calls == 0
    _record().close()


def _kv_cache_budget_vote(failed):
    return _util._any_rank_failed(failed, SimpleNamespace(world_size=WORLD))


def _final_kv_cache_token_sync(dist):
    """Make the token sync of the final KV cache build, a minimum reduction."""
    return dist.allreduce(4096, op=ReduceOp.MIN)


def test_replacement_takes_paused_kv_cache_budget_vote(distributed):
    """Check that a replacement, which skips the KV cache estimation, takes the original ranks' vote."""
    distributed.dist = _rank_0()
    communicator.Distributed.pause_startup_record()
    assert not _kv_cache_budget_vote(False)
    with patch.object(AutoTuner, "get", return_value=_new_autotuner()):
        communicator.Distributed.resume_startup_record()
    assert _final_kv_cache_token_sync(distributed.dist) == 4096
    published = _publish()
    distributed.dist = _replacement(published)
    assert not _kv_cache_budget_vote(False)
    assert _final_kv_cache_token_sync(distributed.dist) == 4096
    assert distributed.dist.host_exchange.calls == 0
    _record().close()
    distributed.dist = _replacement(published)
    assert _kv_cache_budget_vote(True)


def test_replacement_takes_kv_cache_budget_vote_without_estimation(distributed):
    """Check the vote of original ranks that skip the KV cache estimation too."""
    distributed.dist = _rank_0()
    assert not _kv_cache_budget_vote(False)
    assert _final_kv_cache_token_sync(distributed.dist) == 4096
    assert list(_record().values) == ["kv_cache_budget_failed"]
    assert len(_record().exchanges) == 1
    distributed.dist = _replacement(_publish())
    assert not _kv_cache_budget_vote(False)
    assert _final_kv_cache_token_sync(distributed.dist) == 4096
    _record().close()


def test_replacement_votes_only_where_original_ranks_did(distributed):
    """Stop a replacement that votes on a KV cache budget the original ranks never voted on."""
    distributed.dist = _rank_0()
    _final_kv_cache_token_sync(distributed.dist)
    distributed.dist = _replacement(_publish())
    with pytest.raises(
        ReplacementCannotStart, match="kv_cache_budget_failed 1 times, the original ranks 0 times"
    ):
        _kv_cache_budget_vote(False)


@pytest.mark.parametrize("fault_tolerance", [False, True], ids=["off", "rank_1"])
def test_kv_cache_budget_vote_runs_on_other_ranks(distributed, monkeypatch, fault_tolerance):
    """Check that without fault tolerance, and on an original rank other than 0, every rank votes."""
    if not fault_tolerance:
        monkeypatch.setattr(communicator.Distributed, "startup_record", None)
    failed_rank_3 = FakeHostExchange(1, peers=lambda r, obj: int(r == 3))
    distributed.dist = _dist(1, failed_rank_3)
    assert _kv_cache_budget_vote(False)
    assert failed_rank_3.calls == 1
    distributed.dist = _dist(1, FakeHostExchange(1))
    assert not _kv_cache_budget_vote(False)
    if fault_tolerance:
        assert _record().published() == {"values": {}, "exchanges": []}


def _failure_vote(dist):
    return dist.allreduce(0, op=ReduceOp.MAX)


def test_replacement_refuses_another_reduction():
    """Stop a replacement whose collective replays a recorded reduction by another operation."""
    _final_kv_cache_token_sync(_rank_0())
    published = _publish()
    with pytest.raises(
        ReplacementCannotStart,
        match="collective 0 as a max reduction, the original ranks as a min reduction",
    ) as failure:
        _failure_vote(_replacement(published))
    assert failure.value.code == REPLACEMENT_CANNOT_START
    assert "in _failure_vote" in str(failure.value)
    with pytest.raises(
        ReplacementCannotStart,
        match="collective 0 as no min or max reduction, the original ranks as a min reduction",
    ):
        _replacement(published).allgather(4096)


def _kv_load_sample(dist, sample):
    """Gather the KV cache load sample, as ``PyExecutor._initialize_execution`` does."""
    with communicator.Distributed.rank_varying("KV cache load sample"):
        return dist.tp_allgather(sample)


def test_divergent_replacement_cannot_start():
    """Stop a replacement whose collectives differ, but not for equal or rank-varying payloads."""
    rank_0 = _rank_0()
    rank_0.allgather(["payload"] * 2)
    rank_0.barrier()
    _kv_load_sample(rank_0, {"timestampUnixNanos": 1})
    replacement = _replacement(_publish())
    # A different payload stops the replacement.
    with pytest.raises(
        SystemExit, match="collective 0 of its startup differs from the original rank's"
    ) as exited:
        replacement.allgather(0)
    assert exited.value.code == REPLACEMENT_CANNOT_START
    # So does a rank-varying collective in place of another one.
    with pytest.raises(
        SystemExit,
        match=r"makes host collective 0 \(identity\), the original ranks' host collective 0\. ",
    ):
        with _record().rank_varying("identity"):
            replacement.allgather(1)
    # Two equal strings pickle differently from one string twice.
    assert replacement.allgather(["payload", "".join(["pay", "load"])]) == [["payload"] * 2] * WORLD
    # One collective fewer stops it too.
    with pytest.raises(
        SystemExit,
        match="makes 1 host collectives, the original ranks 3: the next one is host collective 1$",
    ):
        _record().close()
    replacement.barrier()
    sample = _kv_load_sample(replacement, {"timestampUnixNanos": 2})
    assert sample == [{"timestampUnixNanos": 1}] * WORLD
    with pytest.raises(SystemExit, match="one more than the original ranks made"):
        replacement.barrier()
    _record().close()


def test_bounded_status_reports_replacement_error(distributed):
    """Check that a replacement whose KV cache allocation failed raises that error."""
    mapping = SimpleNamespace(world_size=WORLD)
    distributed.dist = _rank_0()
    _raise_on_every_rank(None, mapping, "KV cache allocation")
    distributed.dist = _replacement(_publish())
    with pytest.raises(MemoryError, match="pool"):
        _raise_on_every_rank(MemoryError("pool"), mapping, "KV cache allocation")


def _quota(dist, quota):
    return communicator.Distributed.agree_rank_capacity(
        "kv_cache_device_quota", dist.mapping.rank, quota, lambda: dist.allgather(quota)
    )


def test_replacement_takes_predecessor_quota():
    """Check that a replacement of a rank with a smaller quota than rank 0 takes that quota."""
    assert _quota(_rank_0(lambda r, obj: obj - r), 100) == 100
    published = _publish()
    assert _quota(_replacement(published), 110) == 100 - REPLACED
    _record().close()
    replacement = _replacement(published)
    with pytest.raises(
        RuntimeError, match=f"is {100 - REPLACED - 1}, below its predecessor's {100 - REPLACED}"
    ):
        _quota(replacement, 100 - REPLACED - 1)


def test_quota_without_fault_tolerance_gathers_nothing(monkeypatch):
    """Check that without fault tolerance each rank keeps its own quota, with no collective."""
    monkeypatch.setattr(communicator.Distributed, "startup_record", None)
    quota = communicator.Distributed.agree_rank_capacity(
        "kv_cache_device_quota", 1, 100, lambda: pytest.fail("gathered the quotas")
    )
    assert quota == 100


def _new_autotuner():
    with (
        patch.dict(os.environ, TLLM_PROFILING_TIMER="cuda_event"),
        patch("torch.cuda.get_device_name", return_value="GPU"),
        patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        return AutoTuner()


@pytest.fixture
def distributed(monkeypatch):
    """Make ``Distributed.get`` return the ``dist`` of the rank the test runs as."""
    ranks = SimpleNamespace(dist=None)
    monkeypatch.setattr(communicator.Distributed, "get", staticmethod(lambda mapping: ranks.dist))
    return ranks


def _auto_host_tier_quota(monkeypatch, mem_available, memlock=resource.RLIM_INFINITY):
    """Return the automatic host tier quota beside an 80 GiB device quota on a node this free."""
    monkeypatch.setattr(
        rank_striped_read_ahead, "effective_available_host_memory", lambda: mem_available
    )
    monkeypatch.setattr(resource, "getrlimit", lambda limit: (memlock, resource.RLIM_INFINITY))
    mapping = SimpleNamespace(world_size=WORLD)
    limits = get_host_memory_limits(mapping)
    host_quota = _compute_auto_host_tier_quota(
        80 * GIB, limits.local_ranks, limits.mem_available, limits.memlock_limit
    )
    return _sync_host_tier_quota(host_quota, mapping)


def _failed(monkeypatch, ranks):
    """Make ``ranks`` the ranks outside the membership that a replacement started in."""
    record = SimpleNamespace(is_active=lambda rank: rank not in ranks)
    monkeypatch.setattr(startup, "membership_client", lambda: SimpleNamespace(record=record))


def test_replacement_takes_recorded_host_tier(distributed, monkeypatch):
    """Check that a replacement takes the original ranks' host tier while the node has room."""
    _failed(monkeypatch, {REPLACED})
    distributed.dist = _rank_0()
    assert _auto_host_tier_quota(monkeypatch, 720 * GIB) == 80 * GIB
    published = _publish()
    distributed.dist = _replacement(published)
    # Its own heuristic quota is 60 GiB, below the recorded 80 GiB, but its share fits.
    assert _auto_host_tier_quota(monkeypatch, 120 * GIB) == 80 * GIB
    _record().close()
    distributed.dist = _replacement(published)
    with pytest.raises(RuntimeError, match=f"{60 * GIB}-byte share of available host memory"):
        _auto_host_tier_quota(monkeypatch, 60 * GIB)


def test_replacement_host_tier_needs_memlock(distributed, monkeypatch):
    """Check that a replacement whose RLIMIT_MEMLOCK cannot hold the host tier gives up."""
    _failed(monkeypatch, {REPLACED})
    distributed.dist = _rank_0()
    assert _auto_host_tier_quota(monkeypatch, 720 * GIB) == 80 * GIB
    distributed.dist = _replacement(_publish())
    with pytest.raises(ReplacementCannotStart, match=f"RLIMIT_MEMLOCK of {50 * GIB} bytes"):
        _auto_host_tier_quota(monkeypatch, 720 * GIB, memlock=50 * GIB)


def test_node_replacements_share_free_memory(distributed, monkeypatch):
    """Check that the replacements of a whole node each need their share of its free memory."""
    _failed(monkeypatch, set(range(WORLD)))
    distributed.dist = _rank_0()
    assert _auto_host_tier_quota(monkeypatch, 720 * GIB) == 80 * GIB
    published = _publish()
    distributed.dist = _replacement(published)
    assert distributed.dist.local_world_size == WORLD
    with pytest.raises(RuntimeError, match=f"{60 * GIB}-byte share"):
        _auto_host_tier_quota(monkeypatch, 240 * GIB)
    distributed.dist = _replacement(published)
    assert _auto_host_tier_quota(monkeypatch, 400 * GIB) == 80 * GIB


def _gemm(num_tokens):
    """Return the profiling cache key of the GEMM for ``num_tokens`` tokens."""
    return ("gemm", "GemmRunner", "0", ((num_tokens,),))


class _DecoderEngine(SimpleNamespace):
    """Fake a decoder engine whose forward tunes a GEMM per token count, as ``choose_one`` does.

    A token count without a tactic is profiled at ``measured_ms``, and the tuning group merges the
    result (PARALLEL). ``tuned_forwards`` lists the token counts of the forwards in tuning mode,
    where a cached tactic is compiled. ``prime_all_runners`` lists that option of each tuning pass.
    Each of the forwards reports one step of progress to the ``report_progress`` that the tuning
    pass got.
    """

    def __init__(self, dist, measured_ms, tuner=None):
        tuner = _new_autotuner() if tuner is None else tuner
        rank = dist.mapping.rank
        tuner.setup_distributed_state(Mapping(world_size=WORLD, rank=rank, tp_size=WORLD), dist)
        profiling_cache = tuner.profiling_cache

        def forward(batch, *args, **kwargs):
            if self.tuning:
                self.tuned_forwards.append(batch.num_tokens)
            if _gemm(batch.num_tokens) not in profiling_cache.cache:
                profiling_cache.cache[_gemm(batch.num_tokens)] = (0, 3, measured_ms)
                tuner._maybe_sync_cache_data(DistributedTuningStrategy.PARALLEL, "gemm")
            if self.tuning and self.report_progress is not None:
                self.report_progress()

        super().__init__(
            _config=SimpleNamespace(
                enable_autotuner=True,
                torch_compile_enabled=False,
                max_num_tokens=16,
                max_batch_size=16,
                max_seq_len=2,
                original_max_draft_len=0,
                max_total_draft_tokens=0,
                is_spec_decode=False,
                spec_config=None,
            ),
            model=SimpleNamespace(modules=lambda: []),
            mapping=SimpleNamespace(tp_size=WORLD, has_pp=lambda: False),
            dist=dist,
            guided_decoder=None,
            _graphs=SimpleNamespace(enabled=False, no_cuda_graph=contextlib.nullcontext),
            _dummy_requests=SimpleNamespace(
                create_warmup_request=lambda resources, num_tokens, num_gen_requests: (
                    SimpleNamespace(num_tokens=num_tokens)
                ),
                release_batch_context=lambda batch, resources: contextlib.nullcontext(batch),
            ),
            _agreement=WarmupAgreement(dist, dist.mapping),
            _warmup_timer=_WarmupTimer(rank=dist.mapping.rank),
            _completed_autotuner_warmup_num_tokens=None,
            tactics_at_general_warmup=None,
            _release_megamoe_profiling_scratch=Mock(),
            _forward_warmup=Mock(side_effect=forward),
            profiling_cache=profiling_cache,
            tuner=tuner,
            tuning=False,
            report_progress=None,
            tuned_forwards=[],
            prime_all_runners=[],
        )

    def _warmup_scheduled(self, resource_manager, kv_cache_manager):
        """Run only the autotuner phase of the warmup, after the forwards."""
        self.tactics_at_general_warmup = dict(self.profiling_cache.cache)
        DecoderWarmup._run_autotuner_warmup(self, resource_manager)

    @contextlib.contextmanager
    def _autotune(self, cache_path=None, prime_all_runners=False, report_progress=None, **kwargs):
        """Run the forwards in the block in tuning mode, as ``autotune`` does, with its cache file."""
        rank = self.dist.mapping.rank
        if cache_path is not None and os.path.exists(cache_path):
            self.profiling_cache.load_cache(cache_path, rank)
        self.prime_all_runners.append(prime_all_runners)
        self.tuning, self.report_progress = True, report_progress
        try:
            yield
        finally:
            self.tuning, self.report_progress = False, None
            if cache_path is not None:
                self.profiling_cache.save_cache(cache_path, rank)

    def autotuner_warmup(self, num_tokens):
        """Run the warmup up to its autotuner phase, with KV cache room for ``num_tokens``."""
        kv_cache_manager = SimpleNamespace(get_num_available_tokens=lambda **kwargs: num_tokens)
        resource_manager = SimpleNamespace(get_resource_manager=lambda key: kv_cache_manager)
        autotuner = Mock(profiling_cache=self.profiling_cache)
        # The warmup sets the communicator that the tuner merges over, with this engine's mapping.
        autotuner.setup_distributed_state.side_effect = (
            lambda mapping, dist: self.tuner.setup_distributed_state(self.tuner.mapping, dist)
        )
        with (
            patch.object(warmup.AutoTuner, "get", return_value=autotuner),
            patch.object(warmup, "autotune", self._autotune),
            patch.object(warmup, "warmup_named_allreduce_topologies"),
            patch.object(warmup, "clear_memory_buffers"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.synchronize"),
        ):
            DecoderWarmup.run(self, resource_manager)


def _original_rank_0(estimation_num_tokens, final_num_tokens):
    """Run rank 0's autotuner warmups and return its engine, its record and its tuned tactics.

    The warmup of the KV cache estimation is not recorded, and the tactics it tunes are the ones
    that rank 0 publishes. The warmup of the final executor is recorded.
    """
    rank_0 = _DecoderEngine(_rank_0(), measured_ms=0.1)
    communicator.Distributed.pause_startup_record()
    rank_0.autotuner_warmup(estimation_num_tokens)
    with patch.object(AutoTuner, "get", return_value=rank_0.tuner):
        communicator.Distributed.resume_startup_record()
    rank_0.autotuner_warmup(final_num_tokens)
    tuned_tactics = _record().tuned_tactics
    return rank_0, _publish(), tuned_tactics


def _replacement_engine(published, tuned_tactics):
    """Return the engine of a replacement that replays ``published``.

    Its predecessor publishes ``tuned_tactics`` in the rendezvous store, and the replacement takes
    them into its autotuner, as their startups do.
    """
    values = {}
    store = SimpleNamespace(set=values.__setitem__, get=values.__getitem__)
    with patch.object(startup._state, "record", SimpleNamespace(tuned_tactics=tuned_tactics)):
        startup.publish_tuned_tactics(store, REPLACED)
    dist = _replacement(published)
    tuner = _new_autotuner()
    with (
        patch.object(startup._state, "record", _record()),
        patch.object(AutoTuner, "get", return_value=tuner),
    ):
        startup._take_tuned_tactics(store, REPLACED)
    return _DecoderEngine(dist, measured_ms=0.05, tuner=tuner)


def test_replacement_runs_warmup_originals_skip():
    """Check the final autotuner warmup, which the original ranks skip, on a replacement.

    The replacement runs it alone in tuning mode, so the cached tactics of every runner, its
    predecessor's, run once before it serves. Every shape hits them, so it profiles nothing and
    makes no host collective.
    """
    rank_0, published, tuned_tactics = _original_rank_0(16, 16)
    # Only the estimation's warmup ran.
    assert rank_0.tuned_forwards == [16, 1]
    assert rank_0.prime_all_runners == [False]
    assert tuned_tactics == rank_0.profiling_cache.cache
    replacement = _replacement_engine(published, tuned_tactics)
    replacement.autotuner_warmup(16)
    assert replacement.tuned_forwards == [16, 1]
    assert replacement.prime_all_runners == [True]
    assert replacement.profiling_cache.cache == tuned_tactics
    # Its general warmup forwards and CUDA graph captures already run its predecessor's tactics.
    assert replacement.tactics_at_general_warmup == tuned_tactics
    assert replacement.dist.host_exchange.calls == 0
    assert replacement.tuner._dist is replacement.dist
    _record().close()


def test_replacement_tunes_a_missing_shape_alone():
    """Check a shape that its predecessor's tactics lack in the warmup a replacement runs alone.

    The replacement tunes every tactic of it on its own, with no host collective.
    """
    _, published, tuned_tactics = _original_rank_0(16, 16)
    del tuned_tactics[_gemm(1)]
    replacement = _replacement_engine(published, tuned_tactics)
    replacement.autotuner_warmup(16)
    assert replacement.profiling_cache.cache == {**tuned_tactics, _gemm(1): (0, 3, 0.05)}
    assert replacement.dist.host_exchange.calls == 0
    _record().close()


def test_replacement_warmup_reports_startup_progress():
    """Check that each step of the warmup a replacement runs alone counts as startup progress.

    The steps go through the progress report of its replayed collectives, so a long warmup that
    keeps making progress is not taken for a hung startup. An original rank reports none.
    """
    rank_0_progress = _record().report_progress = Mock()
    _, published, tuned_tactics = _original_rank_0(16, 16)
    rank_0_progress.assert_not_called()
    replacement = _replacement_engine(published, tuned_tactics)
    in_tuning_pass = []
    _record().report_progress = lambda: in_tuning_pass.append(replacement.tuning)
    replacement.autotuner_warmup(16)
    assert replacement.tuned_forwards == [16, 1]
    # The replayed skip vote reports outside the pass, and each shape of the pass inside it.
    assert False in in_tuning_pass
    assert in_tuning_pass.count(True) == len(replacement.tuned_forwards)
    _record().close()


def test_original_rank_reports_startup_progress(monkeypatch):
    """Report each startup step of an original rank to the authority."""
    client = Mock()
    monkeypatch.setattr(startup._state, "client", client)
    monkeypatch.setattr(startup._state, "steps", 0)
    startup._state.record.report_progress()
    client.report.assert_called_once_with(waiting_for=0, progress=1)


@pytest.mark.parametrize("cache_file", [False, True])
def test_replacement_replays_originals_final_warmup(monkeypatch, tmp_path, cache_file):
    """Check a final autotuner warmup that the original ranks run, for a larger token count.

    The replacement starts from the tactics its predecessor had before that warmup, so it tunes the
    same shapes and replays the original ranks' merges of them. Unlike them, it runs the cached
    tactics of every runner once. The autotuner cache file that the original ranks save holds the
    shapes of that warmup, so the replacement starts without it.
    """
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "0")
    cache_path = tmp_path / "autotuner.json"
    if cache_file:
        monkeypatch.setenv("TLLM_AUTOTUNER_CACHE_PATH", str(cache_path))
    rank_0, published, tuned_tactics = _original_rank_0(8, 16)
    assert rank_0.tuned_forwards == [8, 1, 16, 1]
    assert _gemm(16) not in tuned_tactics
    if cache_file:
        saved = _new_autotuner().profiling_cache
        saved.load_cache(cache_path, 0)
        assert _gemm(16) in saved.cache
    llm_args = Mock(checkpoint_io_policy=CheckpointIOPolicy.NATIVE, env_overrides=None)
    startup._start_alone({"llm_args": llm_args})
    replacement = _replacement_engine(published, tuned_tactics)
    replacement.autotuner_warmup(16)
    assert replacement.tuned_forwards == [16, 1]
    assert rank_0.prime_all_runners == [False, False]
    assert replacement.prime_all_runners == [True]
    # The merges give it the original ranks' tactic.
    assert replacement.profiling_cache.cache == rank_0.profiling_cache.cache
    assert replacement.dist.host_exchange.calls == 0
    _record().close()


def _cache_hit_votes(dist, hit):
    """Vote on one cache hit and on the hits of two profiles, as ``choose_one`` does."""
    tuner = _new_autotuner()
    tuner.setup_distributed_state(
        Mapping(world_size=WORLD, rank=dist.mapping.rank, tp_size=WORLD), dist
    )
    profiles = [SimpleNamespace(get_opt_shapes=lambda shape=shape: shape) for shape in (1, 2)]
    return tuner._hit_on_all_ranks(hit), tuner._profile_hits_on_all_ranks(profiles, [hit, True])


def test_replacement_replays_cache_hit_votes():
    """Check that a replacement whose autotuner cache hits differ from its predecessor's starts.

    It takes its predecessor's tactics, so its own hits may differ, and it replays the original
    ranks' votes.
    """
    assert _cache_hit_votes(_rank_0(), hit=True) == (True, [True, True])
    replacement = _replacement(_publish())
    assert _cache_hit_votes(replacement, hit=False) == (True, [True, True])
    assert replacement.host_exchange.calls == 0
    _record().close()


class _WarmupDecided(Exception):
    """Stop a warmup at its first phase after the general and the context warmup decisions."""


def _decide_warmup(dist, guided_decoder):
    """Run the warmup of an engine of ``dist`` up to its first phase after the warmup decisions."""
    engine = SimpleNamespace(
        _config=SimpleNamespace(max_beam_width=1),
        mapping=SimpleNamespace(cp_size=1, has_cp_helix=lambda: False),
        sparse_attention_config=None,
        guided_decoder=guided_decoder,
        _context_warmups_supported=True,
        _context_logits_heads=(),
        _agreement=WarmupAgreement(dist, dist.mapping),
        _warmup_timer=_WarmupTimer(rank=dist.mapping.rank),
        _metrics={},
        _graphs=Mock(),
        _prewarm_engram_kernels=Mock(),
        _prewarm_csa2_sparse_indexer=Mock(),
        _prewarm_minimax_m3_msa_indexer=Mock(),
        _prewarm_fmha_libraries=Mock(),
        _prewarm_cute_dsl_indexer_q=Mock(side_effect=_WarmupDecided),
    )
    with (
        patch.object(warmup, "warmup_sampling_module"),
        patch.object(warmup.AutoTuner, "get"),
        pytest.raises(_WarmupDecided),
    ):
        DecoderWarmup._warmup_scheduled(engine, Mock(), Mock())


def test_replacement_replays_guided_decoding_warmup_decisions():
    """Check that a replacement replays the warmup decisions of a capturable guided decoder."""
    _decide_warmup(_rank_0(), guided_decoder=Mock())
    assert [[pickle.loads(p) for p in e.payloads] for e in _record().exchanges] == [
        [0] * WORLD,
        [1] * WORLD,
    ]
    published = _publish()
    replacement = _replacement(published)
    _decide_warmup(replacement, guided_decoder=Mock())
    assert replacement.host_exchange.calls == 0
    _record().close()
    with pytest.raises(ReplacementCannotStart, match="collective 0 of its startup differs"):
        _decide_warmup(_replacement(published), guided_decoder=None)
