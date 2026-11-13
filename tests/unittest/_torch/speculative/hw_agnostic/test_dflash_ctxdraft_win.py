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
"""CTXDRAFT_WIN: a disagg context worker stores only the prompt rows its windowed drafter can read.

The store is driven through ``DFlashWorker._store_prefill_context`` with a recording prefill-store hook and a
row-wise projection, over chunked prompts, a prefix-reuse prompt (first chunk past position 0) and a short prompt,
on and off. Every row within reach of the prompt end must be stored with the same value at the same slot position,
nothing below it may be stored, and the context lengths / bases must be identical.
"""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.speculative.dflash as dflash
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, ctxdraft_win_skip_rows

HID = 8
WINDOW_LEFT = 63  # window 64
MARGIN = 32
REACH = WINDOW_LEFT + 1 + MARGIN
PAGE = 16


@pytest.mark.cpu_only
def test_skip_rows():
    # prompt 300, reach 96: rows below 204 are skipped.
    assert ctxdraft_win_skip_rows(300, 0, 200, 96) == 200
    assert ctxdraft_win_skip_rows(300, 200, 100, 96) == 4
    assert ctxdraft_win_skip_rows(300, 250, 50, 96) == 0
    assert ctxdraft_win_skip_rows(50, 0, 50, 96) == 0
    # Unknown, inconsistent or degenerate inputs never skip.
    assert ctxdraft_win_skip_rows(None, 0, 200, 96) == 0
    assert ctxdraft_win_skip_rows(150, 100, 100, 96) == 0
    assert ctxdraft_win_skip_rows(300, 0, 0, 96) == 0
    assert ctxdraft_win_skip_rows(300, 0, 200, 0) == 0
    assert ctxdraft_win_skip_rows(300, -1, 200, 96) == 0


class _Recorder:
    """The fused prefill-store hook: records (batch row, slot position) -> projected row, takes every chunk."""

    def __init__(self):
        self.cache = {}

    def __call__(self, draft_model, projected, start, end, row, block_table, kv_buf):
        assert projected.shape[0] == end - start
        for j, pos in enumerate(range(start, end)):
            self.cache[(row, pos)] = projected[j].clone()
        return True


def _draft_model(left: int = WINDOW_LEFT):
    # Row-wise and exact: a row's projection never depends on which other rows are projected with it.
    return SimpleNamespace(
        fc=object(),
        hidden_norm=object(),
        project_target_hidden=lambda x: x[:, :HID] * 3.0 + x[:, HID:],
        _resolve_block_attention=lambda layer_idx: (False, (left, left)),
        model=SimpleNamespace(layers=[0, 1, 2]),
    )


def _worker(engaged: bool = True, reuse_addressable: bool = False) -> DFlashWorker:
    worker = object.__new__(DFlashWorker)
    recorder = _Recorder()
    worker._fused_prefill_store = recorder
    worker._ctx_paged = True
    worker._ctx_kv_buf = [torch.zeros(1, dtype=torch.float32)]
    worker._req_ctx_pos = {}
    worker._ctx_unified_cache = False
    worker._req_to_slot = {}
    worker._free_slots = deque(range(8))
    worker._ctx_len_host = [0] * 8
    worker._ctx_base_host = [0] * 8
    worker._max_ctx = 1 << 20
    worker._ctx_page_size = PAGE
    worker._ctx_block_tables = torch.zeros(8, 64, dtype=torch.int32)
    worker._ctx_block_counts = torch.full((8,), 64, dtype=torch.int64)
    worker._ctx_reuse_addressable = reuse_addressable
    worker._disagg_draft_kv_engaged = engaged
    worker.writes = []

    def write_ctx_len(lens, bases=None):
        worker.writes.append((dict(lens), dict(bases or {})))
        for slot, value in lens.items():
            worker._ctx_len_host[slot] = value
        for slot, value in (bases or {}).items():
            worker._ctx_base_host[slot] = value

    worker._write_ctx_len = write_ctx_len
    return worker, recorder


def _step(worker, draft_model, chunks, prompt_lens, hs_by_req, with_prompt_lens=True):
    """One context step: chunks = [(req_id, first_pos, length)]."""
    rows, positions, ids, lens = [], [], [], []
    for req_id, first, length in chunks:
        rows.append(hs_by_req[req_id][first : first + length])
        positions.extend(range(first, first + length))
        ids.append(req_id)
        lens.append(length)
    hs = torch.cat(rows, dim=0)
    spec_metadata = SimpleNamespace(
        get_hidden_states=lambda n: hs[:n],
        host_position_ids=torch.tensor(positions, dtype=torch.int32),
        request_ids=list(ids),
    )
    if with_prompt_lens:
        spec_metadata.ctx_prompt_lens = (tuple(ids), [prompt_lens[r] for r in ids])
    attn_metadata = SimpleNamespace(
        num_ctx_tokens=hs.shape[0],
        num_contexts=len(chunks),
        _seq_lens=torch.tensor(lens, dtype=torch.int32),
        draft_kv_cache_manager=None,
    )
    worker._store_prefill_context(
        draft_model, spec_metadata, attn_metadata, torch.tensor(positions), hs.shape[0]
    )


def _run(monkeypatch, *, off: bool, engaged: bool = True, with_prompt_lens: bool = True):
    monkeypatch.setattr(dflash, "_CTXDRAFT_WIN_OFF", off)
    monkeypatch.setattr(dflash, "_CTXDRAFT_WIN_MARGIN", MARGIN)
    torch.manual_seed(0)
    prompt_lens = {11: 300, 12: 250, 13: 50}
    hs_by_req = {r: torch.randn(n, 2 * HID, dtype=torch.float64) for r, n in prompt_lens.items()}
    worker, recorder = _worker(engaged=engaged)
    draft_model = _draft_model()
    # Step 1: A's first chunk, B from a reused prefix (first position 100), C whole. Step 2: A's tail.
    _step(
        worker,
        draft_model,
        [(11, 0, 200), (12, 100, 150), (13, 0, 50)],
        prompt_lens,
        hs_by_req,
        with_prompt_lens,
    )
    batch_rows_1 = {11: 0, 12: 1, 13: 2}
    cache_1 = dict(recorder.cache)
    recorder.cache.clear()
    _step(worker, draft_model, [(11, 200, 100)], prompt_lens, hs_by_req, with_prompt_lens)
    cache_2 = dict(recorder.cache)
    # Key the stores by (request, slot position).
    by_req = {}
    for (row, pos), value in cache_1.items():
        req = [r for r, b in batch_rows_1.items() if b == row][0]
        by_req[(req, pos)] = value
    for (row, pos), value in cache_2.items():
        assert row == 0
        by_req[(11, pos)] = value
    return by_req, worker, prompt_lens


@pytest.mark.cpu_only
def test_window_rows_identical_and_nothing_below(monkeypatch):
    on, worker_on, prompt_lens = _run(monkeypatch, off=False)
    off, worker_off, _ = _run(monkeypatch, off=True)
    # Same context lengths, bases and per-request positions.
    assert worker_on.writes == worker_off.writes
    assert worker_on._req_ctx_pos == worker_off._req_ctx_pos
    assert worker_on._ctx_len_host == worker_off._ctx_len_host
    # The off path stores every prompt row (B from its reused prefix on).
    assert len(off) == 300 + 150 + 50
    for (req, pos), value in off.items():
        if pos >= prompt_lens[req] - REACH:
            assert (req, pos) in on, (req, pos)
            assert torch.equal(on[(req, pos)], value)
    for req, pos in on:
        assert pos >= prompt_lens[req] - REACH
    # A: 300 - 96 = 204 -> 96 rows; B: 250 - 96 = 154 -> 96 rows; C: all 50.
    assert len(on) == 96 + 96 + 50
    assert worker_on._ctxdraft_win_skipped == 500 - len(on)


@pytest.mark.cpu_only
def test_not_taken_without_transfer_or_prompt_lens_or_with_reuse(monkeypatch):
    on, _, _ = _run(monkeypatch, off=False, engaged=False)
    assert len(on) == 500
    on, _, _ = _run(monkeypatch, off=False, with_prompt_lens=False)
    assert len(on) == 500

    monkeypatch.setattr(dflash, "_CTXDRAFT_WIN_OFF", False)
    worker, _ = _worker(reuse_addressable=True)
    assert worker._ctxdraft_win_resolve(_draft_model()) == 0
    worker, _ = _worker()
    assert worker._ctxdraft_win_resolve(_draft_model(left=-1)) == 0
    worker, _ = _worker()
    assert (
        worker._ctxdraft_win_resolve(_draft_model())
        == WINDOW_LEFT + 1 + dflash._CTXDRAFT_WIN_MARGIN
    )


@pytest.mark.cpu_only
def test_stale_prompt_lens_are_ignored(monkeypatch):
    monkeypatch.setattr(dflash, "_CTXDRAFT_WIN_OFF", False)
    monkeypatch.setattr(dflash, "_CTXDRAFT_WIN_MARGIN", MARGIN)
    worker, _ = _worker()
    spec_metadata = SimpleNamespace(
        host_position_ids=torch.arange(200, dtype=torch.int32),
        request_ids=[11],
        ctx_prompt_lens=((99,), [300]),  # another batch's list
    )
    attn_metadata = SimpleNamespace(_seq_lens=torch.tensor([200], dtype=torch.int32))
    assert worker._ctxdraft_win_skips(_draft_model(), spec_metadata, attn_metadata, 1) is None
    spec_metadata.ctx_prompt_lens = ((11,), [300])
    assert worker._ctxdraft_win_skips(_draft_model(), spec_metadata, attn_metadata, 1) == [200]
