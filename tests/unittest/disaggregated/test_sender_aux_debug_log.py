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
"""The sender's AUX hand-off formats its WriteMeta only when debug logging is enabled."""

import queue
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod

pytestmark = pytest.mark.cpu_only


class _UnformattableWriteMeta(transfer_mod.WriteMeta):
    def __repr__(self) -> str:
        raise AssertionError("WriteMeta was formatted although debug logging is disabled")


def _aux_write_meta(task) -> transfer_mod.WriteMeta:
    ptrs = np.arange(4, dtype=np.int64)
    return _UnformattableWriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="gen",
        peer_rank=0,
        peer_endpoint="tcp://gen:1234",
        unique_rid=401,
        src_ptrs=ptrs,
        dst_ptrs=ptrs,
        sizes=ptrs,
        meta_type=transfer_mod.WriteMetaType.AUX,
    )


def test_aux_delivery_does_not_format_write_meta_when_debug_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = object.__new__(transfer_mod.Sender)
    sender._device_id = 0
    sender._thread_local = threading.local()
    sender._pending_settlements = [{}]
    sender._send_task_queues = [queue.Queue()]
    sender._deliver_aux_to_agent = Mock()
    task = Mock()
    write_meta = _aux_write_meta(task)
    sender._send_task_queues[0].put(write_meta)
    sender._send_task_queues[0].put(None)

    monkeypatch.setattr(transfer_mod.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer_mod.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer_mod, "CUASSERT", Mock())
    monkeypatch.setattr(
        transfer_mod.logger,
        "is_severity_enabled",
        lambda severity, module="": severity != transfer_mod.logger.VERBOSE,
    )

    sender._process_task_queue(0)

    sender._deliver_aux_to_agent.assert_called_once_with(write_meta)
    task.fail.assert_not_called()
