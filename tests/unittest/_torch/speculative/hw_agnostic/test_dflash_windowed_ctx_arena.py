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
"""The DFlash worker's windowed context arena is off unless asked for, host-only.

Without ``TRTLLM_DFLASH_WINDOWED_CTX_ARENA=1`` the worker records a full-length
arena at construction, keeps no per-slot base, reads no drafter window, and its
context-length scatter writes the lengths alone, as it always has. With the
switch it records a windowed arena, and the scatter moves each slot's base with
its length.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative import dflash
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_SWITCH = "TRTLLM_DFLASH_WINDOWED_CTX_ARENA"


def _worker() -> DFlashWorker:
    return DFlashWorker(DFlashDecodingConfig(max_draft_len=7), Mapping())


@pytest.fixture
def host_only(monkeypatch):
    """Device copies stay on the host, and every ``index_copy_`` is recorded."""
    copies = []
    to = torch.Tensor.to

    def host_to(self, *args, **kwargs):
        if args and args[0] == "cuda":
            return self
        return to(self, *args, **kwargs)

    index_copy = torch.Tensor.index_copy_

    def recorded(self, dim, index, source):
        copies.append((self.data_ptr(), index.tolist(), source.tolist()))
        return index_copy(self, dim, index, source)

    monkeypatch.setattr(torch.Tensor, "to", host_to)
    monkeypatch.setattr(torch.Tensor, "index_copy_", recorded)
    monkeypatch.setattr(dflash, "prefer_pinned", lambda: False)
    return copies


@pytest.mark.parametrize("value", [None, "0", ""])
def test_without_the_switch_the_arena_is_full_length(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(_SWITCH, raising=False)
    else:
        monkeypatch.setenv(_SWITCH, value)
    worker = _worker()
    assert worker.windowed_ctx_arena is False
    assert worker._ctx_base is None and worker._ctx_keep == 0

    class _Unread:
        def __getattr__(self, name):
            raise AssertionError(f"read the drafter's {name}")

    assert worker._windowed_ctx_keep(_Unread(), 5) == 0


def test_the_switch_is_read_once_at_construction(monkeypatch):
    monkeypatch.setenv(_SWITCH, "1")
    worker = _worker()
    assert worker.windowed_ctx_arena is True
    monkeypatch.delenv(_SWITCH)
    assert worker.windowed_ctx_arena is True
    assert _worker().windowed_ctx_arena is False


def test_a_full_length_arena_scatters_the_lengths_alone(monkeypatch, host_only):
    monkeypatch.delenv(_SWITCH, raising=False)
    worker = _worker()
    worker._ctx_len = torch.zeros(4, dtype=torch.long)
    worker._ctx_len_host = [0] * 4
    worker._write_ctx_len({1: 5, 3: 0}, {1: 2})
    assert worker._ctx_len.tolist() == [0, 5, 0, 0]
    assert worker._ctx_len_host == [0, 5, 0, 0]
    assert host_only == [(worker._ctx_len.data_ptr(), [1, 3], [5, 0])]
    assert worker._ctx_base is None and worker._ctx_base_host is None


def test_a_windowed_arena_moves_each_base_with_its_length(monkeypatch, host_only):
    monkeypatch.setenv(_SWITCH, "1")
    worker = _worker()
    worker._ctx_len = torch.zeros(4, dtype=torch.long)
    worker._ctx_len_host = [0] * 4
    worker._ctx_base = torch.zeros(4, dtype=torch.long)
    worker._ctx_base_host = [0, 0, 7, 0]
    worker._write_ctx_len({1: 5, 2: 9, 3: 0}, {1: 2})
    assert worker._ctx_len.tolist() == [0, 5, 9, 0]
    # An explicit base, an unchanged one, and a reset slot's 0.
    assert worker._ctx_base.tolist() == [0, 2, 7, 0]
    assert worker._ctx_base_host == [0, 2, 7, 0]
    assert len(host_only) == 2


def test_the_windowed_keep_reads_the_drafter_windows(monkeypatch):
    monkeypatch.setenv(_SWITCH, "1")
    worker = _worker()
    worker._max_ctx = 1 << 20
    drafter = SimpleNamespace(_resolve_block_attention=lambda layer: (False, (2047, 0)))
    assert worker._windowed_ctx_keep(drafter, 5) == 2048
    unwindowed = SimpleNamespace(_resolve_block_attention=lambda layer: (False, (-1, -1)))
    assert worker._windowed_ctx_keep(unwindowed, 5) == 0
