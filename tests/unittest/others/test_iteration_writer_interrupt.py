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
"""KeyboardInterrupt handling in IterationWriter.run, with ZMQ mocked out."""

from multiprocessing import Event
from unittest import mock

import pytest

from tensorrt_llm.bench.benchmark.utils import processes
from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter

# The CPU CI stage runs pytest with -m cpu_only and skips files without this marker.
pytestmark = pytest.mark.cpu_only


def _run_with_socket(socket, tmp_path):
    context = mock.MagicMock()
    context.socket.return_value = socket
    with mock.patch.object(processes, "Context", return_value=context):
        IterationWriter.run("ipc://unused", tmp_path / "iter.log", Event())
    return context


def test_interrupt_during_recv_drains_until_end_sentinel(tmp_path):
    socket = mock.MagicMock()
    socket.recv_json.side_effect = [KeyboardInterrupt(), {"iter": 1}, {"end": True}]

    context = _run_with_socket(socket, tmp_path)

    # The drain loop keeps reading past the stats message and stops at the
    # sentinel; a fourth call would raise StopIteration from the mock.
    assert socket.recv_json.call_count == 3
    socket.close.assert_called_once()
    context.term.assert_called_once()


def test_interrupt_during_bind_does_not_drain(tmp_path):
    socket = mock.MagicMock()
    socket.bind.side_effect = KeyboardInterrupt()
    # Fail fast instead of hanging in the drain loop if the guard regresses.
    socket.recv_json.side_effect = AssertionError("receive on unbound socket")

    context = _run_with_socket(socket, tmp_path)

    # Nothing can arrive on an unbound socket, so draining would block forever.
    socket.recv_json.assert_not_called()
    socket.close.assert_called_once()
    context.term.assert_called_once()
