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
"""``DFlashWorker._draft_block_hidden_states``: the block-output rows that produce the draft logits (host-side).

For every (num_gens, block, K, shift_label) shape the rows equal the stock gather of the clamped slot ids
(``dflash_draft_slot_ids``). Where those ids are one run of rows (one request, or DSpark's shift_label convention
with a block of K slots) the rows are a view of the block outputs; otherwise they are a gather. The ids are built
once per shape, outside CUDA-graph capture.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.speculative import dflash as dflash_module
from tensorrt_llm._torch.speculative import dspark as dspark_module
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_draft_slot_ids
from tensorrt_llm._torch.speculative.dspark import DSparkWorker

pytestmark = pytest.mark.cpu_only

HIDDEN = 4


@pytest.fixture(autouse=True)
def host_ids(monkeypatch):
    """The workers' slot ids on the host, and no CUDA-graph capture."""

    def on_host(num_gens, block_size, num_draft_tokens, shift_label, device="cuda"):
        return dflash_draft_slot_ids(num_gens, block_size, num_draft_tokens, shift_label, "cpu")

    monkeypatch.setattr(dflash_module, "dflash_draft_slot_ids", on_host)
    monkeypatch.setattr(dspark_module, "dflash_draft_slot_ids", on_host)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _worker(cls=DSparkWorker):
    """A worker without ``__init__`` (it needs flashinfer and a drafter)."""
    worker = cls.__new__(cls)
    nn.Module.__init__(worker)
    return worker


def _block(num_gens, block):
    """Block outputs whose rows are distinct."""
    rows = num_gens * block
    return torch.arange(rows * HIDDEN, dtype=torch.float32).reshape(rows, HIDDEN)


def _stock(out, num_gens, block, k, shift_label):
    """The gather the draft forward ran: every request's slot ids, the last one clamped to the block outputs."""
    ids = dflash_draft_slot_ids(num_gens, block, k, shift_label, "cpu").clamp(max=out.shape[0] - 1)
    return out[ids]


def _one_run(num_gens, block, k, shift_label):
    """Whether the slots are one run of rows: one request whose slots fit its block, or the shift_label
    convention with a block of K slots (each request's slots start where the previous one's end)."""
    first = 0 if shift_label else 1
    return first + k <= block and (num_gens == 1 or (shift_label and block == k))


def _is_view(rows, out):
    return rows.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()


@pytest.mark.parametrize("shift_label", [True, False], ids=["shift_label", "dflash slots"])
@pytest.mark.parametrize("block_extra", [0, 1], ids=["block K", "block K+1"])
@pytest.mark.parametrize("k", [2, 7])
@pytest.mark.parametrize("num_gens", range(1, 9))
def test_rows_equal_the_stock_gather(num_gens, k, block_extra, shift_label):
    block = k + block_extra
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=shift_label)
    out = _block(num_gens, block)

    rows = worker._draft_block_hidden_states(drafter, out, num_gens, block, k)

    assert torch.equal(rows, _stock(out, num_gens, block, k, shift_label))
    assert _is_view(rows, out) == _one_run(num_gens, block, k, shift_label)
    if _is_view(rows, out):
        first = 0 if shift_label else 1
        assert rows.data_ptr() == out[first].data_ptr()
    # The cached decision gives the same rows on the next step.
    again = worker._draft_block_hidden_states(drafter, out, num_gens, block, k)
    assert torch.equal(again, rows) and _is_view(again, out) == _is_view(rows, out)


def test_dspark_decode_shape_is_a_view():
    """DSpark's shift_label drafter at a block of K = 7: every batch of the decode path reads its rows in place."""
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=True)
    for num_gens in range(1, 9):
        out = _block(num_gens, 7)
        rows = worker._draft_block_hidden_states(drafter, out, num_gens, 7, 7)
        assert _is_view(rows, out) and rows.shape == (num_gens * 7, HIDDEN)
        assert torch.equal(rows, out)


def test_dflash_slots_of_several_requests_are_gathered():
    """Plain DFlash reads slots 1..K of each block of K + 1: a gap per request, so the rows are gathered."""
    worker = _worker(DFlashWorker)
    out = _block(3, 8)
    rows = worker._draft_block_hidden_states(object(), out, 3, 8, 7)
    assert not _is_view(rows, out)
    assert torch.equal(rows, _stock(out, 3, 8, 7, False))


def test_slot_ids_are_built_once_per_shape(monkeypatch):
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=True)
    built = []
    real = DSparkWorker._draft_slot_ids
    monkeypatch.setattr(
        DSparkWorker,
        "_draft_slot_ids",
        lambda self, *args: built.append(args[1:]) or real(self, *args),
    )
    for _ in range(3):
        worker._draft_block_hidden_states(drafter, _block(2, 7), 2, 7, 7)
        worker._draft_block_hidden_states(drafter, _block(4, 7), 4, 7, 7)
    assert built == [(2, 7, 7), (4, 7, 7)]


def test_capture_gathers_a_shape_it_has_not_seen(monkeypatch):
    """Under capture an unseen shape is gathered and not kept (building the run needs a host read); a shape seen
    before capture keeps its view."""
    worker = _worker()
    drafter = SimpleNamespace(_dspark_shift_label=True)
    seen, unseen = _block(2, 7), _block(3, 7)
    assert _is_view(worker._draft_block_hidden_states(drafter, seen, 2, 7, 7), seen)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    rows = worker._draft_block_hidden_states(drafter, unseen, 3, 7, 7)
    assert not _is_view(rows, unseen) and torch.equal(rows, unseen)
    assert _is_view(worker._draft_block_hidden_states(drafter, seen, 2, 7, 7), seen)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert _is_view(worker._draft_block_hidden_states(drafter, unseen, 3, 7, 7), unseen)
