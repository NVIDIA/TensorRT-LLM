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
"""Sequence-slot handling when a request re-enters on its first context chunk.

A request parked for an asynchronous KV-connector load computes nothing, so it
rejoins the batch still on its first context chunk and reaches
``prepare_resources`` a second time. The managers are built without their CUDA
buffers, which ``prepare_resources`` does not touch when relaxed acceptance is
off.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.resource_manager import SlotManager
from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager
from tensorrt_llm._torch.speculative.mtp import MTPHiddenStatesManager


def _manager(cls):
    manager = cls.__new__(cls)
    manager.slot_manager = SlotManager(4)
    manager.use_relaxed_acceptance_for_thinking = False
    return manager


def _batch(*requests):
    return SimpleNamespace(context_requests=list(requests))


def _request(request_id, first_chunk=True):
    return SimpleNamespace(request_id=request_id, is_first_context_chunk=first_chunk)


@pytest.mark.parametrize("cls", [Eagle3ResourceManager, MTPHiddenStatesManager])
def test_first_chunk_reentry_keeps_slot(cls):
    manager = _manager(cls)
    parked = _request(7)

    manager.prepare_resources(_batch(parked))
    slot = manager.slot_manager.get_slot(7)
    assert slot is not None

    manager.prepare_resources(_batch(parked, _request(8)))

    assert manager.slot_manager.get_slot(7) == slot
    assert manager.slot_manager.get_slot(8) not in (None, slot)
    assert len(manager.slot_manager.slot_mapping) == 2


def test_eagle3_reentry_reports_held_slot():
    manager = _manager(Eagle3ResourceManager)
    parked = _request(7)

    manager.prepare_resources(_batch(parked))
    first = list(manager.slot_ids)
    manager.prepare_resources(_batch(parked))

    assert manager.slot_ids == first
