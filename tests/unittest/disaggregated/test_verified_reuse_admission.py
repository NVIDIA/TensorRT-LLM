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
"""Verified-range KV reuse admission for disaggregated generation.

The gen worker must not admit received KV block ranges to the reuse tree on
transfer SUCCESS alone: a false-success (write skipped, partial, or misrouted
while SUCCESS is notified) would put never-written pages in the radix tree
under the prompt's token hashes, and reuse would re-serve the poisoned prefix.

These tests cover the admission contract:
- the sender attests destination bytes only for completed submissions;
- the receiver tracks attested bytes against the published byte total;
- reuse commit is fail-closed on an unverified (or missing) verdict;
- the verified verdict requires agreement from every rank.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.bounce as bounce_mod
import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base import CacheKind, Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    RxSession,
    TaskStatus,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# Helpers (mirroring test_chunked_transfer.py)
# ---------------------------------------------------------------------------


def _make_params(rid: int = 42) -> DisaggregatedParams:
    return DisaggregatedParams(disagg_request_id=rid)


def _make_chunk(prompt_len: int = 8) -> Chunk:
    return Chunk(
        block_ids_per_layer_groups=[np.array([0], dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=prompt_len),
        is_last=True,
    )


def _stub_receiver():
    receiver = MagicMock()
    receiver._enforce_physical_ownership = False
    receiver.setup_session = MagicMock()
    receiver.dispatch_task = MagicMock()
    return receiver


def _make_rx_session(
    rid: int = 42,
    prompt_len: int = 8,
    expected_write_bytes=None,
) -> RxSession:
    session = RxSession(
        request_id=rid,
        params=_make_params(rid),
        receiver=_stub_receiver(),
        prompt_len=prompt_len,
    )
    session.receive(_make_chunk(prompt_len), expected_write_bytes=expected_write_bytes)
    session._receiver._bounce.is_bounced.return_value = False
    return session


def _make_recv_task(expected_write_bytes=None) -> KVRecvTask:
    return KVRecvTask(
        unique_rid=42,
        chunk=_make_chunk(),
        slice_id=0,
        params=_make_params(),
        aux_slot=None,
        expected_write_bytes=expected_write_bytes,
    )


def _success(session, transfer_size: int, peer_rank: int = 0) -> None:
    session.process_kv_agent_result(
        peer_rank=peer_rank,
        receiver_slice_id=0,
        is_last_slice=True,
        status=AgentResult.SUCCESS,
        transfer_size=transfer_size,
    )


# ---------------------------------------------------------------------------
# KVRecvTask.write_verified
# ---------------------------------------------------------------------------


class TestRecvTaskWriteVerified:
    def test_exact_coverage_verifies(self):
        task = _make_recv_task(expected_write_bytes=4096)
        task.verified_write_bytes = 4096
        task.status = TaskStatus.TRANSFERRED
        assert task.write_verified

    def test_short_coverage_is_unverified(self):
        task = _make_recv_task(expected_write_bytes=4096)
        task.verified_write_bytes = 2048
        task.status = TaskStatus.TRANSFERRED
        assert not task.write_verified

    def test_over_coverage_is_unverified(self):
        """Duplicated/misrouted writes are as disqualifying as missing ones."""
        task = _make_recv_task(expected_write_bytes=4096)
        task.verified_write_bytes = 8192
        task.status = TaskStatus.TRANSFERRED
        assert not task.write_verified

    def test_unknown_expected_bytes_is_unverified(self):
        task = _make_recv_task(expected_write_bytes=None)
        task.verified_write_bytes = 4096
        task.status = TaskStatus.TRANSFERRED
        assert not task.write_verified

    def test_untransferred_task_is_unverified(self):
        task = _make_recv_task(expected_write_bytes=0)
        assert not task.write_verified

    def test_empty_transfer_verifies_trivially(self):
        """Fully cached prefix: nothing published, nothing to write."""
        task = _make_recv_task(expected_write_bytes=0)
        task.status = TaskStatus.TRANSFERRED
        assert task.write_verified


# ---------------------------------------------------------------------------
# RxSession accounting
# ---------------------------------------------------------------------------


class TestRxSessionVerification:
    def test_receive_plumbs_expected_bytes_to_task(self):
        session = _make_rx_session(expected_write_bytes=4096)
        assert session._kv_tasks[0].expected_write_bytes == 4096

    def test_success_results_accumulate_verified_bytes(self):
        session = _make_rx_session(expected_write_bytes=4096)
        session._kv_tasks[0].expected_transfers = 2
        _success(session, 1024, peer_rank=0)
        _success(session, 3072, peer_rank=1)
        assert session._kv_tasks[0].verified_write_bytes == 4096
        assert session._kv_tasks[0].status == TaskStatus.TRANSFERRED
        assert session.kv_write_verified()

    def test_false_success_without_attested_bytes_is_unverified(self):
        """The proven defect: SUCCESS notified, write never submitted."""
        session = _make_rx_session(expected_write_bytes=4096)
        session._kv_tasks[0].expected_transfers = 1
        _success(session, 0)
        # Transfer looks complete (this is what the old code trusted) ...
        assert session._kv_tasks[0].status == TaskStatus.TRANSFERRED
        # ... but the published range has no attested write coverage.
        assert not session.kv_write_verified()

    def test_partial_write_is_unverified(self):
        session = _make_rx_session(expected_write_bytes=4096)
        session._kv_tasks[0].expected_transfers = 2
        _success(session, 1024, peer_rank=0)
        _success(session, 1024, peer_rank=1)
        assert session._kv_tasks[0].status == TaskStatus.TRANSFERRED
        assert not session.kv_write_verified()

    def test_failed_result_bytes_do_not_count(self):
        session = _make_rx_session(expected_write_bytes=4096)
        session._kv_tasks[0].expected_transfers = 1
        session.process_kv_agent_result(
            peer_rank=0,
            receiver_slice_id=0,
            is_last_slice=True,
            status=AgentResult.FAILED,
            transfer_size=4096,
        )
        assert session._kv_tasks[0].verified_write_bytes == 0
        assert not session.kv_write_verified()

    def test_session_without_tasks_is_unverified(self):
        session = RxSession(
            request_id=7,
            params=_make_params(7),
            receiver=_stub_receiver(),
            prompt_len=8,
        )
        assert not session.kv_write_verified()


# ---------------------------------------------------------------------------
# Reuse-commit gate (fail-closed)
# ---------------------------------------------------------------------------


def _commit(req) -> MagicMock:
    adapter = MagicMock()
    stub = SimpleNamespace(_reuse_adapter=adapter)
    KvCacheTransceiverV2.commit_blocks_for_reuse(stub, req)
    return adapter


class TestCommitGate:
    def test_verified_request_commits(self):
        req = SimpleNamespace(py_request_id=1, py_kv_transfer_verified=True)
        adapter = _commit(req)
        adapter.commit_blocks_for_reuse.assert_called_once_with(req)

    def test_unverified_request_is_not_admitted(self):
        req = SimpleNamespace(py_request_id=1, py_kv_transfer_verified=False)
        adapter = _commit(req)
        adapter.commit_blocks_for_reuse.assert_not_called()

    def test_missing_verdict_is_fail_closed(self):
        req = SimpleNamespace(py_request_id=1)
        adapter = _commit(req)
        adapter.commit_blocks_for_reuse.assert_not_called()


# ---------------------------------------------------------------------------
# Cross-rank verdict consensus
# ---------------------------------------------------------------------------


def _consensus_stub(gathered):
    """Transceiver stub whose allgather returns a canned per-rank outcome list."""
    return SimpleNamespace(
        _union=KvCacheTransceiverV2._union,
        _intersection=KvCacheTransceiverV2._intersection,
    ), (lambda local: gathered)


class TestVerifiedConsensus:
    def test_verified_requires_every_rank(self):
        # rid 1: all ranks verified; rid 2: rank 1 missing coverage.
        gathered = [
            [[], [], [1, 2], [1, 2], [1, 2]],
            [[], [], [1, 2], [1, 2], [1]],
        ]
        stub, allgather = _consensus_stub(gathered)
        cancelled, failed, completed, quiesced, verified = KvCacheTransceiverV2._consensus_outcome(
            stub,
            [1, 2],
            [],
            [],
            [1, 2],
            allgather,
            True,
            locally_quiesced=[1, 2],
            locally_verified=[1, 2],
        )
        assert completed == [1, 2]
        assert verified == {1}

    def test_no_sync_uses_local_verdict(self):
        stub, _ = _consensus_stub(None)
        result = KvCacheTransceiverV2._consensus_outcome(
            stub,
            [5],
            [],
            [],
            [5],
            lambda local: (_ for _ in ()).throw(AssertionError("must not gather")),
            False,
            locally_quiesced=[5],
            locally_verified=[5],
        )
        cancelled, failed, completed, quiesced, verified = result
        assert completed == [5]
        assert verified == {5}

    def test_quiesced_only_call_still_returns_four_values(self):
        """The ctx path passes quiesced without verified; arity unchanged."""
        stub, _ = _consensus_stub(None)
        result = KvCacheTransceiverV2._consensus_outcome(
            stub,
            [5],
            [],
            [],
            [5],
            lambda local: [local],
            False,
            locally_quiesced=[5],
        )
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Sender attestation
# ---------------------------------------------------------------------------


def _make_sender() -> transfer_mod.Sender:
    sender = object.__new__(transfer_mod.Sender)
    sender._enforce_physical_ownership = False
    sender._sessions_lock = threading.Lock()
    sender._sessions = {}
    sender._instance_rank = 5
    sender._device_id = 0
    sender._agent = Mock()
    sender._bounce = Mock()
    sender._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="ctx", instance_rank=5)
    )
    return sender


def _deliver_kv(monkeypatch, submit_result):
    """Drive _deliver_kv_to_agent with a mocked agent and return the sent frames."""
    rid = 99
    sender = _make_sender()
    dealer = Mock()
    sender._get_result_dealer = Mock(return_value=dealer)
    task = transfer_mod.KVSendTask(
        _make_chunk(),
        DisaggregatedParams(disagg_request_id=rid),
        slice_id=0,
    )
    session = SimpleNamespace(
        kv_tasks=[task],
        lock=threading.Lock(),
        status=SessionStatus.READY,
        set_exception=Mock(),
    )
    sender._sessions = {rid: session}
    write_meta = transfer_mod.WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen",
        peer_rank=2,
        peer_endpoint="receiver",
        unique_rid=rid,
        src_ptrs=np.array([0x1000, 0x3000], dtype=np.int64),
        dst_ptrs=np.array([0x2000, 0x4000], dtype=np.int64),
        sizes=np.array([0x100, 0x300], dtype=np.int64),
        slice_id=0,
        receiver_slice_id=0,
        is_last_slice=True,
    )
    monkeypatch.setattr(bounce_mod, "build_send_request", Mock(return_value=(Mock(), None)))
    monkeypatch.setattr(transfer_mod.Sender, "_submit_transfer", Mock(return_value=submit_result))
    sender._deliver_kv_to_agent(write_meta)
    dealer.send.assert_called_once()
    return transfer_mod._KV_RESULT_PREFIX.unpack(
        dealer.send.call_args.args[0][1][: transfer_mod._KV_RESULT_PREFIX.size]
    )


class TestSenderAttestation:
    def test_completed_submission_attests_actual_bytes(self, monkeypatch):
        _, _, _, _, status_code, transfer_size = _deliver_kv(monkeypatch, (True, None))
        assert status_code == transfer_mod._AGENT_RESULT_CODE[AgentResult.SUCCESS]
        assert transfer_size == 0x400

    def test_failed_submission_attests_zero_bytes(self, monkeypatch):
        _, _, _, _, status_code, transfer_size = _deliver_kv(monkeypatch, (False, -1))
        assert status_code == transfer_mod._AGENT_RESULT_CODE[AgentResult.FAILED]
        assert transfer_size == 0
