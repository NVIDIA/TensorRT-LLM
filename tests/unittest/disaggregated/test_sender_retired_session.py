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
"""A write queued for a TxSession that was retired before it ran still answers the receiver.

A cancelled context request retires its TxSession as soon as no write is in
flight, while writes queued from earlier REQUEST_DATA messages may still wait
on the sender's worker threads. Each such write finds no session. Its receiver
published the slice and counts on a terminal word from this writer; without
one the slice stays mid-write on that generation rank, which can then neither
complete nor cancel the request, while its TP peers that got their word finish
the request and free its KV.
"""

import threading
from unittest.mock import Mock

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.base import TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.extent import Chunk
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

_RID = 401
_CTX_RANK = 13


def _params(schedule_style=DisaggScheduleStyle.CONTEXT_FIRST) -> DisaggregatedParams:
    return DisaggregatedParams(disagg_request_id=_RID, schedule_style=schedule_style)


def _chunk() -> Chunk:
    return Chunk([], [], TokenRange(0, 16), True)


def _sender(owned: bool = False) -> transfer_mod.Sender:
    """A sender whose session table no longer holds the request."""
    sender = object.__new__(transfer_mod.Sender)
    sender._enforce_physical_ownership = owned
    sender._sessions_lock, sender._sessions = threading.Lock(), {}
    sender._instance_rank = _CTX_RANK
    sender.dealer = Mock()
    sender._get_result_dealer = Mock(return_value=sender.dealer)
    return sender


def _write_meta(task, meta_type=transfer_mod.WriteMetaType.KV) -> transfer_mod.WriteMeta:
    return transfer_mod.WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen9",
        peer_rank=9,
        peer_endpoint="gen9-endpoint",
        unique_rid=_RID,
        src_ptrs=np.array([0x1000], dtype=np.int64),
        dst_ptrs=np.array([0x2000], dtype=np.int64),
        sizes=np.array([0x100], dtype=np.int64),
        slice_id=0,
        receiver_slice_id=3,
        is_last_slice=False,
        meta_type=meta_type,
    )


def _receiver() -> transfer_mod.Receiver:
    receiver = object.__new__(transfer_mod.Receiver)
    receiver._enforce_physical_ownership = False
    receiver._sessions_lock, receiver._sessions = threading.Lock(), {}
    receiver._pre_cancelled_rids = {}
    receiver._shutdown = False
    receiver._bounce = Mock()
    receiver._bounce.is_bounced.return_value = False
    return receiver


def test_kv_write_for_a_retired_session_reports_failed_to_the_receiver() -> None:
    """The write fails locally and the receiver gets FAILED for the slice.

    FAILED is the writer's last word on that slice, as for a CANCELLED session.
    """
    sender = _sender()
    task = transfer_mod.KVSendTask(_chunk(), _params(), slice_id=0)

    sender._deliver_kv_to_agent(_write_meta(task))

    assert task.status is transfer_mod.TaskStatus.ERROR
    sender._get_result_dealer.assert_called_once_with("gen9-endpoint")
    sender.dealer.send.assert_called_once()
    message = sender.dealer.send.call_args.args[0]
    assert message[0] == transfer_mod.MessageType.KV_AGENT_RESULT
    peer_rank, rid, slice_id, is_last, status_code, size = transfer_mod._KV_RESULT_PREFIX.unpack(
        message[1]
    )
    assert (peer_rank, rid, slice_id, is_last, size) == (_CTX_RANK, _RID, 3, True, 0)
    assert transfer_mod._AGENT_RESULT_BY_CODE[status_code] is transfer_mod.AgentResult.FAILED


def test_the_reply_lets_a_cancelled_receiver_retire_its_slice() -> None:
    """The reply settles a cancelled slice that is still mid-write.

    A published slice whose request was cancelled stays mid-write
    (``cancel_request`` keeps refusing) until the writer's word arrives; the
    FAILED reply settles it and the session can close.
    """
    receiver = _receiver()
    session = transfer_mod.RxSession(request_id=_RID, params=_params(), receiver=receiver)
    recv_task = session.prepare_receive(_chunk())
    recv_task.expected_transfers = 1
    session.mark_transferring(recv_task.slice_id)
    session.cancel_local()
    assert session.status is SessionStatus.CANCELLED
    assert session.has_transferring_tasks()

    sender = _sender()
    write_meta = _write_meta(transfer_mod.KVSendTask(_chunk(), _params(), slice_id=0))
    write_meta.receiver_slice_id = recv_task.slice_id
    sender._deliver_kv_to_agent(write_meta)
    receiver._process_kv_agent_result(b"", sender.dealer.send.call_args.args[0])

    assert recv_task.status is transfer_mod.TaskStatus.ERROR
    assert not session.has_transferring_tasks()
    assert session.status is SessionStatus.CANCELLED
    assert session.close()


def test_aux_write_for_a_retired_session_reports_failed_to_the_receiver() -> None:
    sender = _sender()
    task = transfer_mod.AuxSendTask(_params(DisaggScheduleStyle.GENERATION_FIRST), slot=0)

    sender._deliver_aux_to_agent(_write_meta(task, transfer_mod.WriteMetaType.AUX))

    assert task.status is transfer_mod.TaskStatus.ERROR
    sender.dealer.send.assert_called_once_with(
        transfer_mod._make_aux_result_msg(_CTX_RANK, _RID, transfer_mod.AgentResult.FAILED)
    )


def test_owned_aux_write_for_a_retired_session_sends_no_unclaimed_word() -> None:
    """Under the ownership bridge the sender sends no unclaimed aux word.

    A terminal aux word needs the session's claim; with the session gone the
    sender stays silent, as before.
    """
    sender = _sender(owned=True)
    task = transfer_mod.AuxSendTask(_params(DisaggScheduleStyle.GENERATION_FIRST), slot=0)
    assert task.begin_physical_operation(9)

    sender._deliver_aux_to_agent(_write_meta(task, transfer_mod.WriteMetaType.AUX))

    assert task.status is transfer_mod.TaskStatus.ERROR
    sender.dealer.send.assert_not_called()
