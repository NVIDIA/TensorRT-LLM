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
"""Tests for the native wait of the MNNVL host exchange on a GPU.

``test_host_exchange.py`` runs the host exchange over host rows with a Python stand-in for the
native wait; these tests run the native wait over device memory.
"""

import threading

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.distributed.mnnvl_host_exchange import MnnvlHostExchangeGroup
from tensorrt_llm.bindings.internal.runtime import set_wake_word, wait_for_slots

_SLOT_BYTES = 64
_JOIN_TIMEOUT_S = 10.0


class _DeviceRows:
    """Rows of zeroed device memory, in which no round's packets arrive."""

    def __init__(self, nranks, slot_bytes):
        self.stride = MnnvlHostExchangeGroup.buffer_bytes(nranks, slot_bytes)
        self._memory = torch.zeros(nranks * self.stride, dtype=torch.uint8, device="cuda")

    def row(self, rank):
        return self._memory.data_ptr() + rank * self.stride


@pytest.mark.timeout(60)
def test_wake_word_set_natively_ends_wait():
    """Check that ``set_wake_word`` from another thread ends ``wait_for_slots`` over device slots."""
    nranks, packets = 2, 1 + _SLOT_BYTES // 4
    slots = torch.zeros((nranks, packets), dtype=torch.int64, device="cuda")
    received = np.zeros((nranks, packets), dtype=np.uint64)
    active, readers = np.ones(nranks, dtype=bool), np.zeros(nranks, dtype=bool)
    waiting, wake = np.zeros(nranks, dtype=bool), np.zeros(1, dtype=np.int32)
    stream = torch.cuda.Stream()
    result = {}
    args = (stream.cuda_stream, [], slots.data_ptr(), received, active, readers, 1, 1, waiting)
    waiter = threading.Thread(
        target=lambda: result.update(arrived=wait_for_slots(*args, wake, 0.0)), daemon=True
    )
    waiter.start()
    waiter.join(0.1)
    assert waiter.is_alive(), "the wait returned before the wake word was set"
    set_wake_word(wake, 1)
    waiter.join(_JOIN_TIMEOUT_S)
    assert not waiter.is_alive(), "the wake word did not end the wait"
    assert result == {"arrived": False}


@pytest.mark.timeout(60)
def test_cancel_ends_native_wait():
    """Check that ``cancel`` from another thread ends the native wait, then the next wait too."""
    nranks = 2
    group = MnnvlHostExchangeGroup(0, nranks, _DeviceRows(nranks, _SLOT_BYTES), _SLOT_BYTES)
    active, readers = np.ones(nranks, dtype=bool), np.zeros(nranks, dtype=bool)
    for epoch in (1, 2):
        result = {}
        waiter = threading.Thread(
            target=lambda: result.update(rows=group.wait(epoch, active, readers)), daemon=True
        )
        waiter.start()
        # Past the wait's spin, so the cancel also reaches a wait that sleeps between polls.
        waiter.join(2 * MnnvlHostExchangeGroup._SPIN_S)
        assert waiter.is_alive(), "the wait returned before the cancel"
        group.cancel()
        waiter.join(_JOIN_TIMEOUT_S)
        assert not waiter.is_alive(), "the cancel did not end the wait"
        assert result == {"rows": None}
