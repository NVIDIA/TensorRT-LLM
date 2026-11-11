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
"""Actual native wire/token boundary tests; no model, transfer or GPU allocation."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgpack
import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native import transfer

pytestmark = pytest.mark.cpu_only
TOKEN = b"a" * 16
DIGEST = "b" * 64


@pytest.mark.parametrize("status", list(transfer.AgentResult))
@pytest.mark.parametrize("retained", [False, True])
def test_result_preserves_mandatory_report_token_and_adds_only_fourth_proof(status, retained):
    message = transfer._make_kv_result_msg(
        0,
        17,
        0,
        True,
        status,
        kv_result_report_token=TOKEN,
        transfer_size=256,
        retained_digest=DIGEST if retained else None,
    )
    assert len(message) == (4 if retained else 3)
    assert message[2] == TOKEN
    assert transfer._KV_RESULT_FRAME.unpack(message[1]) == (
        0,
        17,
        0,
        True,
        transfer._AGENT_RESULT_CODE[status],
        256,
    )
    if retained:
        assert message[3] == DIGEST.encode("ascii")


def recv_info():
    return transfer.RecvReqInfo(
        sender_req_id=17,
        instance_name="prefill",
        instance_rank=0,
        block_ids_per_layer_groups=[np.array([0], dtype=np.int64)],
        unique_rid=17,
        retained_proofs={0: DIGEST},
        kv_result_report_token=b"x" * 16,
    )


def test_body_cannot_supply_report_token_and_checked_request_frame_wins():
    body = msgpack.unpackb(recv_info().to_bytes(), raw=False)
    assert "kv_result_report_token" not in body
    body["kv_result_report_token"] = b"forged body token"
    packed = msgpack.packb(body)
    assert transfer.RecvReqInfo.from_bytes(packed).kv_result_report_token == b""
    sender = transfer.Sender.__new__(transfer.Sender)
    sender._sessions_lock = threading.RLock()
    sender._get_session = lambda request_id: None
    sender._retained_result_digest = lambda *args: DIGEST
    sender._instance_rank = 0
    sender._pre_cancelled_rids = set()
    sender.ownership_enabled = lambda context_first: False
    sender._save_peer_req_info = Mock()
    with patch.object(transfer.disagg_diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False):
        sender._respond_with_kv(b"", [transfer.MessageType.REQUEST_DATA, packed, TOKEN])
    actual = sender._save_peer_req_info.call_args.args[0]
    assert actual.kv_result_report_token == TOKEN
    assert actual.retained_proofs == {0: DIGEST}


@pytest.mark.parametrize("frames", [[], [b"short"], [b"x" * 17], [TOKEN, b"extra"]])
def test_request_data_requires_exact_checked_third_frame(frames):
    sender = transfer.Sender.__new__(transfer.Sender)
    with pytest.raises(ValueError):
        sender._respond_with_kv(
            b"", [transfer.MessageType.REQUEST_DATA, recv_info().to_bytes(), *frames]
        )


@pytest.mark.parametrize(
    "expected,proof,accepted",
    [
        (None, None, True),
        (None, DIGEST, False),
        (DIGEST, None, False),
        (DIGEST, DIGEST, True),
        (DIGEST, "c" * 64, False),
        (DIGEST, "a" * 16, False),
    ],
)
def test_receiver_routes_only_matching_opt_in_fourth_frame(expected, proof, accepted):
    receiver = transfer.Receiver.__new__(transfer.Receiver)
    session = SimpleNamespace(process_kv_agent_result=Mock())
    if expected is not None:
        session._retained_coverage_expected_writers = {0: SimpleNamespace(digest=expected)}
    receiver._get_session = lambda request_id: session
    message = transfer._make_kv_result_msg(
        0,
        17,
        0,
        True,
        transfer.AgentResult.SUCCESS,
        kv_result_report_token=TOKEN,
        transfer_size=256,
        retained_digest=proof,
    )
    with patch.object(transfer.disagg_diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False):
        receiver._process_kv_agent_result(b"", message)
    assert session.process_kv_agent_result.call_count == int(accepted)
    if accepted:
        assert session.process_kv_agent_result.call_args.args[4] == TOKEN


class TokenBoundarySession(transfer.RxSession):
    @property
    def request_id(self):
        return 17


@pytest.mark.parametrize("token", [b"", b"a" * 15, b"c" * 16])
def test_bad_or_replayed_token_rejects_before_retained_coverage_or_task_mutation(token):
    session = TokenBoundarySession.__new__(TokenBoundarySession)
    session._ownership_evidence_guard = lambda: nullcontext()
    session.lock = threading.Lock()
    session._retirement = None
    task = Mock()
    task._kv_result_report_tokens = {0: TOKEN}
    session._kv_tasks = [task]
    session._retained_coverage_expected_writers = {0: object()}
    with patch(
        "tensorrt_llm._torch.disaggregation.native.retained_coverage.check_completed_writer"
    ) as check:
        with pytest.raises(ValueError, match="report token"):
            session.process_kv_agent_result(
                0, 0, True, transfer.AgentResult.SUCCESS, token, transfer_size=256
            )
    check.assert_not_called()
    task.fail.assert_not_called()


@pytest.mark.parametrize("rank", ["00", "01", "8", "-1", "+0", " 0", "٠", "０", "0\n", "7 "])
def test_retained_request_rank_keys_reject_aliases_unicode_and_out_of_range(rank):
    body = msgpack.unpackb(recv_info().to_bytes(), raw=False)
    body["retained_proofs"] = {rank: DIGEST}
    with pytest.raises(ValueError, match="retained native request proof map"):
        transfer.RecvReqInfo.from_bytes(msgpack.packb(body))


def test_retained_rank_normalization_cannot_overwrite_an_admitted_writer():
    body = msgpack.unpackb(recv_info().to_bytes(), raw=False)
    body["retained_proofs"] = {"0": DIGEST, "00": "c" * 64}
    with pytest.raises(ValueError, match="retained native request proof map"):
        transfer.RecvReqInfo.from_bytes(msgpack.packb(body))


def test_retained_request_supports_complete_canonical_rank_bound():
    body = msgpack.unpackb(recv_info().to_bytes(), raw=False)
    body["retained_proofs"] = {str(rank): DIGEST for rank in range(8)}
    actual = transfer.RecvReqInfo.from_bytes(msgpack.packb(body))
    assert actual.retained_proofs == {rank: DIGEST for rank in range(8)}
