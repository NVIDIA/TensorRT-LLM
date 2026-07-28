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
"""Aux-buffer registration must skip empty (null-pointer) buffers.

``torch.empty(n, 0)`` — which is what the draft-token buffer is when
``max_draft_len == 0`` — has ``data_ptr() == 0``. Handing that null address to
NIXL fails the whole registration on the LIBFABRIC backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker


def _fake_worker(ptrs, sizes):
    """Minimal stand-in exposing only what _register_aux_buffer touches."""
    meta = AuxBufferMeta(ptrs=np.array(ptrs, dtype=np.int64), size=np.array(sizes, dtype=np.int64))
    return SimpleNamespace(
        _aux_buffer=SimpleNamespace(meta=meta), _agent=Mock(), _registered_mem=[]
    )


def _registered_descs(worker):
    worker._agent.register_memory.assert_called_once()
    return worker._agent.register_memory.call_args[0][0].descs


def test_null_pointer_aux_buffer_is_skipped():
    # Index 1 mirrors the draft-token buffer with max_draft_len == 0.
    worker = _fake_worker([0x1000, 0, 0x3000, 0x4000], [512, 0, 128, 128])

    TransferWorker._register_aux_buffer(worker)

    descs = _registered_descs(worker)
    assert [d[0] for d in descs] == [0x1000, 0x3000, 0x4000]
    assert all(d[1] > 0 for d in descs)
    assert len(worker._registered_mem) == 1


def test_zero_size_aux_buffer_is_skipped():
    worker = _fake_worker([0x1000, 0x2000], [512, 0])

    TransferWorker._register_aux_buffer(worker)

    assert [d[0] for d in _registered_descs(worker)] == [0x1000]


def test_all_buffers_registered_when_none_are_empty():
    ptrs = [0x1000, 0x2000, 0x3000, 0x4000]
    worker = _fake_worker(ptrs, [512, 256, 128, 128])

    TransferWorker._register_aux_buffer(worker)

    descs = _registered_descs(worker)
    assert [d[0] for d in descs] == ptrs
    # Names stay tied to the original buffer index so peers still match up.
    assert [d[3] for d in descs] == [f"aux_buffer_ptr_{i}" for i in range(4)]


def test_nothing_registered_when_every_buffer_is_empty():
    worker = _fake_worker([0, 0], [0, 0])

    TransferWorker._register_aux_buffer(worker)

    worker._agent.register_memory.assert_not_called()
    assert worker._registered_mem == []
