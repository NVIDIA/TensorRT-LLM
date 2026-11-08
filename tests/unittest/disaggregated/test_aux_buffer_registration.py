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
"""Aux-buffer registration and transfer must skip empty buffers.

``torch.empty(n, 0)`` — which is what the draft-token buffer is when
``max_draft_len == 0`` — has ``data_ptr() == 0``. Handing that null address to
NIXL fails the whole registration on the LIBFABRIC backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBuffer,
    AuxBufferMeta,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.disaggregation.native.transfer import Sender, TransferWorker, WriteMeta

pytestmark = pytest.mark.cpu_only


def _fake_worker(ptrs: list[int], sizes: list[int]) -> SimpleNamespace:
    """Minimal stand-in exposing only what _register_aux_buffer touches."""
    meta = AuxBufferMeta(ptrs=np.array(ptrs, dtype=np.int64), size=np.array(sizes, dtype=np.int64))
    return SimpleNamespace(
        _aux_buffer=SimpleNamespace(meta=meta), _agent=Mock(), _registered_mem=[]
    )


def _registered_descs(worker: SimpleNamespace) -> list[tuple[int, int, int, str]]:
    worker._agent.register_memory.assert_called_once()
    return worker._agent.register_memory.call_args[0][0].descs


def _build_aux_write_meta(
    src_buffer: AuxBuffer,
    dst_buffer: AuxBuffer,
    src_slot: int = 0,
    dst_slot: int = 0,
) -> tuple[WriteMeta, Mock]:
    """Build auxiliary transfer metadata with minimal sender dependencies."""
    peer_rank_info = SimpleNamespace(
        aux_meta=dst_buffer.meta,
        instance_name="peer",
        instance_rank=1,
        self_endpoint="tcp://peer",
        dp_rank=0,
    )
    registrar = Mock()
    registrar.self_rank_info = SimpleNamespace(aux_meta=src_buffer.meta)
    registrar.get_peer_rank_info.return_value = peer_rank_info
    registrar.get_peer_overlap.return_value = SimpleNamespace(ranks=[0])
    registrar.should_send_aux.return_value = True
    registrar.get_aux_transfer_layout.return_value = None
    sender = SimpleNamespace(_registrar=registrar)
    task = SimpleNamespace(_perf_timer=None, _slot=src_slot, _unique_rid=7)
    req_info = SimpleNamespace(
        instance_name="peer", instance_rank=1, aux_slot=dst_slot, unique_rid=7
    )
    return Sender._build_aux_write_meta(sender, task, req_info), registrar


def test_null_pointer_with_non_zero_size_is_rejected() -> None:
    worker = _fake_worker([0x1000, 0, 0x3000, 0x4000], [512, 256, 128, 128])

    with pytest.raises(ValueError, match="null pointers with non-zero sizes at indices \\[1\\]"):
        TransferWorker._register_aux_buffer(worker)


def test_zero_size_aux_buffer_is_skipped() -> None:
    worker = _fake_worker([0x1000, 0x2000], [512, 0])

    TransferWorker._register_aux_buffer(worker)

    assert [d[0] for d in _registered_descs(worker)] == [0x1000]


def test_real_empty_draft_buffer_is_skipped_during_registration() -> None:
    aux_buffer = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=0, device="cpu")
    worker = SimpleNamespace(_aux_buffer=aux_buffer, _agent=Mock(), _registered_mem=[])

    TransferWorker._register_aux_buffer(worker)

    descs = _registered_descs(worker)
    assert [d[3] for d in descs] == [
        "aux_buffer_ptr_0",
        "aux_buffer_ptr_2",
        "aux_buffer_ptr_3",
    ]
    assert all(int(d[0]) != 0 and int(d[1]) > 0 for d in descs)
    assert len(worker._registered_mem) == 1


def test_all_buffers_registered_when_none_are_empty() -> None:
    ptrs = [0x1000, 0x2000, 0x3000, 0x4000]
    worker = _fake_worker(ptrs, [512, 256, 128, 128])

    TransferWorker._register_aux_buffer(worker)

    descs = _registered_descs(worker)
    assert [d[0] for d in descs] == ptrs
    # Names stay tied to the original buffer index so peers still match up.
    assert [d[3] for d in descs] == [f"aux_buffer_ptr_{i}" for i in range(4)]


def test_nothing_registered_when_every_buffer_is_empty() -> None:
    worker = _fake_worker([0, 0], [0, 0])

    TransferWorker._register_aux_buffer(worker)

    worker._agent.register_memory.assert_not_called()
    assert worker._registered_mem == []


def test_real_empty_draft_buffer_is_skipped_during_transfer() -> None:
    src_buffer = AuxBuffer(max_slot_num=3, beam_width=1, max_draft_len=0, device="cpu")
    dst_buffer = AuxBuffer(max_slot_num=3, beam_width=1, max_draft_len=0, device="cpu")

    write_meta, registrar = _build_aux_write_meta(src_buffer, dst_buffer, src_slot=1, dst_slot=2)

    expected_indices = np.array([0, 2, 3])
    np.testing.assert_array_equal(
        write_meta.src_ptrs,
        src_buffer.meta.ptrs[expected_indices] + src_buffer.meta.item_sizes[expected_indices],
    )
    np.testing.assert_array_equal(
        write_meta.dst_ptrs,
        dst_buffer.meta.ptrs[expected_indices] + dst_buffer.meta.item_sizes[expected_indices] * 2,
    )
    np.testing.assert_array_equal(write_meta.sizes, src_buffer.meta.item_sizes[expected_indices])
    assert write_meta.src_ptrs.size == 3
    assert all(int(ptr) != 0 for ptr in write_meta.src_ptrs)
    assert all(int(ptr) != 0 for ptr in write_meta.dst_ptrs)
    registrar.cache_aux_transfer_layout.assert_called_once()
    cached_layout = registrar.cache_aux_transfer_layout.call_args.args[2]
    assert all(
        not array.flags.writeable
        for array in (
            cached_layout.src_base_ptrs,
            cached_layout.dst_base_ptrs,
            cached_layout.src_item_sizes,
            cached_layout.dst_item_sizes,
        )
    )


def test_non_empty_source_requires_non_empty_destination() -> None:
    src_buffer = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=4, device="cpu")
    dst_buffer = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=0, device="cpu")

    with pytest.raises(
        ValueError,
        match="Destination auxiliary buffers are empty for non-empty source indices \\[1\\]",
    ):
        build_aux_transfer_layout(src_buffer.meta, dst_buffer.meta)


def test_aux_layout_mismatch_is_checked_lazily_during_transfer() -> None:
    src_buffer = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=4, device="cpu")
    dst_buffer = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=0, device="cpu")

    with pytest.raises(
        ValueError,
        match="Destination auxiliary buffers are empty for non-empty source indices \\[1\\]",
    ):
        _build_aux_write_meta(src_buffer, dst_buffer)


def test_destination_aux_buffer_must_be_large_enough() -> None:
    src_buffer = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=4, device="cpu")
    dst_buffer = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=2, device="cpu")

    with pytest.raises(
        ValueError,
        match="Destination auxiliary buffers are too small at indices \\[1\\]",
    ):
        build_aux_transfer_layout(src_buffer.meta, dst_buffer.meta)
