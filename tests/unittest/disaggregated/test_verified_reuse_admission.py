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
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    IntactMapper,
    ReplicatedMapper,
)
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVRecvTask,
    RxSession,
    TaskStatus,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges
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


# ---------------------------------------------------------------------------
# Receiver expectation vs summed sender bytes
# ---------------------------------------------------------------------------

_UNIT = 64  # bytes of one K, V, or index-K buffer per (layer, block)


def _make_block_chunk(num_blocks: int) -> Chunk:
    return Chunk(
        block_ids_per_layer_groups=[np.arange(num_blocks, dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(start=0, end=num_blocks * 16),
        is_last=True,
    )


def _coalesced_page_table() -> KVCachePageTable:
    """Coalesced-pool page table mirroring the MiniMax M3 TP=2 layout.

    K == V == index-K bytes per block, so V2 coalesces all three roles into
    one physical pool. The slot interleaves the sparse layer's index-K
    between K/V regions (L0K L0V L1K L1V L1IDX L2K L2V L3K L3V), yielding
    two views over the same pool: an NHD K/V view with a non-uniform layer
    stride and a REPLICATED index-K view. Mirrors
    test_minimax_m3_pool_view_scheme_coalesced_vs_separate.
    """
    u = _UNIT
    kv_entries = np.array(
        [
            (0, 0 * u, u),
            (0, 1 * u, u),
            (1, 2 * u, u),
            (1, 3 * u, u),
            (2, 5 * u, u),
            (2, 6 * u, u),
            (3, 7 * u, u),
            (3, 8 * u, u),
        ],
        dtype=BUFFER_ENTRY_DTYPE,
    )
    idx_entries = np.array([(1, 4 * u, u)], dtype=BUFFER_ENTRY_DTYPE)
    layer_group = AttentionLayerGroup(
        pool_group_idx=0,
        local_layers=[LocalLayer(i, i) for i in range(4)],
        pool_views=[
            PoolView(
                pool_idx=0,
                buffer_entries=kv_entries,
                pool_role=frozenset({"key", "value"}),
                mapper_kind=MapperKind.NHD,
                bytes_per_layer=2 * u,
            ),
            PoolView(
                pool_idx=0,
                buffer_entries=idx_entries,
                pool_role=frozenset({"index_key"}),
                mapper_kind=MapperKind.REPLICATED,
                bytes_per_layer=u,
            ),
        ],
        kv_head_num_per_rank=1,
    )
    pool = PhysicalPool(base_address=0x10000, slot_bytes=9 * u, num_slots=8)
    return KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[layer_group],
        pool_groups=[PhysicalPoolGroup(pools=[pool])],
    )


def _ignored_role_page_table() -> KVCachePageTable:
    """Page table whose slot holds regions no view transfers.

    The slot interleaves a local-only ignored-role buffer after each layer's
    K region (L0K L0LOCAL L1K L1LOCAL). Ignored roles occupy slot offsets
    but appear in no view, so no writer ever covers them.
    """
    u = _UNIT
    entries = np.array([(0, 0 * u, u), (1, 2 * u, u)], dtype=BUFFER_ENTRY_DTYPE)
    layer_group = AttentionLayerGroup(
        pool_group_idx=0,
        local_layers=[LocalLayer(0, 0), LocalLayer(1, 1)],
        pool_views=[
            PoolView(
                pool_idx=0,
                buffer_entries=entries,
                pool_role=frozenset({"key"}),
                mapper_kind=MapperKind.INDEXED,
                bytes_per_layer=u,
            ),
        ],
        kv_head_num_per_rank=1,
    )
    pool = PhysicalPool(base_address=0x20000, slot_bytes=4 * u, num_slots=8)
    return KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[layer_group],
        pool_groups=[PhysicalPoolGroup(pools=[pool])],
    )


def _expected_write_bytes(page_table: KVCachePageTable, chunk: Chunk) -> int:
    """The receiver-side expectation, exactly as the transceiver computes it."""
    return KvCacheTransceiverV2._chunk_num_bytes(SimpleNamespace(_page_table=page_table), chunk)


def _summed_sender_bytes(page_table: KVCachePageTable, chunk: Chunk) -> int:
    """Destination bytes a matched-layout, head-matched sender submits for *chunk*.

    The extract -> mapper -> WriteMeta.sizes path of
    Sender._build_kv_write_meta, run per pool view against itself as the
    peer. This is the quantity each writer attests on completion and the
    receiver sums into verified_write_bytes.
    """
    extractor = KVRegionExtractorV1(page_table)
    total = 0
    for lg_idx, block_ids in enumerate(chunk.block_ids_per_layer_groups):
        layer_group = page_table.layer_groups[lg_idx]
        for view_idx, pool_view in enumerate(layer_group.pool_views):
            # Offsets in physical slot order, as get_kv_map builds them.
            starts, bytes_per_layer = get_layer_byte_ranges(pool_view)
            offsets = np.array(sorted(starts.values()), dtype=np.int64)
            mapper_cls = (
                ReplicatedMapper if pool_view.mapper_kind == MapperKind.REPLICATED else IntactMapper
            )
            mapper = mapper_cls(offsets, offsets, bytes_per_layer, bytes_per_layer)
            region = extractor.extract(block_ids, lg_idx, view_idx)
            pairs = mapper.map(region, region)
            for pair in pairs if isinstance(pairs, list) else [pairs]:
                total += int(pair.src.memory.ptrs.size) * int(pair.src.memory.bytes_per_region)
    return total


class TestExpectedWriteBytesMatchSenderBytes:
    """The receiver expectation must equal the summed sender bytes.

    Otherwise verified-reuse admission marks every successful receive
    unverified and its blocks never enter the reuse tree. Regression:
    computing the expectation as physical slot_bytes per pool view double
    counts a coalesced slot (one slot, one view per role class) and counts
    ignored-role regions no view transfers.
    """

    def test_coalesced_pool_layout(self):
        page_table = _coalesced_page_table()
        chunk = _make_block_chunk(num_blocks=2)
        expected = _expected_write_bytes(page_table, chunk)
        assert expected == _summed_sender_bytes(page_table, chunk)
        # 2 blocks x (8 K/V units + 1 index-K unit): the 9-unit physical
        # slot is counted once, not once per view.
        assert expected == 2 * 9 * _UNIT

    def test_ignored_role_layout(self):
        page_table = _ignored_role_page_table()
        chunk = _make_block_chunk(num_blocks=2)
        expected = _expected_write_bytes(page_table, chunk)
        assert expected == _summed_sender_bytes(page_table, chunk)
        # 2 blocks x 2 K units: the two local-only units in each 4-unit
        # slot are expected from no writer.
        assert expected == 2 * 2 * _UNIT

    def test_invalid_blocks_are_not_expected(self):
        page_table = _coalesced_page_table()
        chunk = Chunk(
            block_ids_per_layer_groups=[np.array([-1, 0, 1, -1], dtype=np.int64)],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(start=0, end=64),
            is_last=True,
        )
        expected = _expected_write_bytes(page_table, chunk)
        # extract() drops -1 (BAD_PAGE_INDEX) block ids on the sender path
        # too, so the two sides agree with holes in the block table.
        assert expected == _summed_sender_bytes(page_table, chunk)
        assert expected == 2 * 9 * _UNIT
