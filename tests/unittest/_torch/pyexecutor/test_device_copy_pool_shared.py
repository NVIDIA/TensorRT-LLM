# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA graph metadata buffers are pooled across graph keys, so their owners never skip uploads into them.

Every graph key's metadata takes its buffers from one pool, which hands the same block to each key that asks for the
same name, and a captured forward's device writes into a block run again at each of its replays. A record of what an
upload last wrote (``DeviceCopyBuffer``) then no longer describes the device contents, so such destinations go through
the never-skipping ``device_copy.copy`` / ``device_copy.fill_tail``; only a destination its upload alone writes (the MSA
block table) keeps a record.
"""

import pytest
import torch

from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.device_copy import CopySiteEnum, DeviceCopyBuffer, Direction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

NUM_SLOTS = 8


class _Site(CopySiteEnum):
    UPLOAD = ("test_pool_shared.upload", Direction.H2D)
    TAIL = ("test_pool_shared.tail", Direction.FILL)


@pytest.fixture(autouse=True)
def _no_check_mode(monkeypatch):
    # The skipping variants below demonstrate a stale record on purpose; check mode would raise on them.
    monkeypatch.setattr(device_copy, "CHECK_MODE", False)


def _host(values):
    return torch.tensor(values, dtype=torch.int32).pin_memory()


def _graph_key_views(name):
    """Two graph keys' metadata views from one pool, as create_cuda_graph_metadata hands them out."""
    pool = Buffers()
    large = pool.get_buffer([NUM_SLOTS], torch.int32, name, reserve_buffer=True)
    small = pool.get_buffer([NUM_SLOTS], torch.int32, name, reserve_buffer=True)
    assert small.data_ptr() == large.data_ptr()
    return large, small


def _capture_live_row_write(buf, rows, values):
    """A graph whose replay writes buf[:rows] on the device, as on_update_kv_lens writes the live slots."""
    slots = torch.tensor(values, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        buf[:rows].copy_(slots)
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def _key_b_step(buf, live, record):
    """Key B's prepare: its live slots, then -1 over the padded rows its captured producers also run. With a record,
    both writes skip what the record says the buffer holds (the stale behaviour); without, they always land."""
    if record is None:
        device_copy.copy(_Site.UPLOAD, buf[: len(live)], _host(live))
        device_copy.fill_tail(_Site.TAIL, buf, len(live), -1)
    else:
        record.update(_host(live), view=buf[: len(live)])
        record.fill_tail(_Site.TAIL, len(live), -1)


@pytest.mark.parametrize("skipping", [False, True], ids=["never_skipped", "skipping_record"])
def test_two_graph_keys_over_one_block_keep_the_padded_rows_unused(skipping):
    key_a, key_b = _graph_key_views("msa_out_cache_loc")
    replay_a = _capture_live_row_write(key_a, 6, [10, 11, 12, 13, 14, 15])
    record = DeviceCopyBuffer(key_b, site=_Site.UPLOAD) if skipping else None

    _key_b_step(key_b, [20, 21, 22], record)
    torch.cuda.synchronize()
    assert key_b.tolist() == [20, 21, 22, -1, -1, -1, -1, -1]
    # Key A's step: its replay writes six live slots into the shared block.
    replay_a.replay()
    # Key B's next step, with the same live slots.
    _key_b_step(key_b, [20, 21, 22], record)
    torch.cuda.synchronize()
    if skipping:
        # The live-row upload was skipped, so key B's live rows hold key A's live slots (the tail fill follows a
        # write of the live rows through a view, which drops its record, so it lands).
        assert key_b.tolist() == [10, 11, 12, -1, -1, -1, -1, -1]
    else:
        assert key_b.tolist() == [20, 21, 22, -1, -1, -1, -1, -1]


@pytest.mark.parametrize("skipping", [False, True], ids=["never_skipped", "skipping_record"])
def test_two_graph_keys_over_one_block_keep_their_uploads(skipping):
    key_a, key_b = _graph_key_views("msa_block_table")
    replay_a = _capture_live_row_write(key_a, 4, [7, 7, 7, 7])
    table = [1, 2, 3, 4]
    record = DeviceCopyBuffer(key_b, site=_Site.UPLOAD)

    def upload():
        if skipping:
            return record.update(_host(table), view=key_b[:4])
        return device_copy.copy(_Site.UPLOAD, key_b[:4], _host(table))

    assert upload() is True
    replay_a.replay()
    copied = upload()
    torch.cuda.synchronize()
    if skipping:
        assert copied is False
        assert key_b[:4].tolist() == [7, 7, 7, 7]
    else:
        assert copied is True
        assert key_b[:4].tolist() == table


def test_a_write_outside_the_record_must_invalidate_it():
    """Neither the upload record nor the tail record vouches for the buffer after a write outside them."""
    buf = torch.zeros(6, dtype=torch.int32, device="cuda")
    record = DeviceCopyBuffer(buf, site=_Site.UPLOAD)
    values = _host([1, 2])
    assert record.update(values, view=buf[:2]) is True
    record.fill_tail(_Site.TAIL, 2, -1)
    device_copy.copy(_Site.UPLOAD, buf[:2], _host([5, 6]))
    buf[3] = 9
    record.invalidate()
    assert record.update(values, view=buf[:2]) is True
    record.fill_tail(_Site.TAIL, 2, -1)
    torch.cuda.synchronize()
    assert buf.tolist() == [1, 2, -1, -1, -1, -1]


def test_never_skipped_writes_without_an_active_copier_are_plain_writes():
    assert device_copy.current() is None
    buf = torch.zeros(4, dtype=torch.int32, device="cuda")
    assert device_copy.copy(_Site.UPLOAD, buf[:2], _host([3, 4])) is True
    device_copy.fill_tail(_Site.TAIL, buf, 2, -1)
    torch.cuda.synchronize()
    assert buf.tolist() == [3, 4, -1, -1]
