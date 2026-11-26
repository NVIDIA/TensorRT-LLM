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
"""Workspace lifecycle regressions for token-sharded users of the shared CE transport.

Real copy/signal ordering, recovery and graph replay are covered by test_ce_transport and
by the token-sharded collective GPU tests. These CPU checks protect workspace ownership
and the allocation addresses held by captured graphs.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.parallel import token_sharded_ce_workspace as ws
from tensorrt_llm._torch.visual_gen.parallel.symm_mem_pool import PoolLayout

pytestmark = pytest.mark.cpu_only


class _Pool:
    def __init__(self):
        self.align = 256
        self.world_size = 2
        self.layout = None
        self.channels = 8
        self.released = False

    @property
    def reserved(self):
        return self.layout is not None

    @property
    def slots(self):
        return 0 if self.layout is None else self.layout.slots

    @property
    def slot_nbytes(self):
        return 0 if self.layout is None else self.layout.slot_nbytes

    def release(self):
        self.released = True

    def reserve(self, slot_nbytes, slots, min_channels):
        assert min_channels <= self.channels
        want = PoolLayout.for_request(slot_nbytes, slots, self.world_size, self.align)
        if self.layout is not None and self.layout.covers(want):
            return False
        self.layout = want
        return True


class _Transport:
    def __init__(self):
        self.pool = _Pool()
        self.events = []
        self.recoverable = True

    @contextmanager
    def boundary(self, expected):
        self.events.append(("open", tuple(expected)))
        try:
            yield
        except RuntimeError:
            self.events.append("interrupted")
            raise
        else:
            self.events.append("completion fence")

    def ready(self):
        self.events.append("recover")
        return self.recoverable

    def close(self):
        self.pool.released = True


def _workspace():
    ce = _Transport()
    return ws.CeWorkspace(ce, 2, 0, torch.device("cpu"), slots=2), ce


def test_workspace_completion_precedes_slot_reuse():
    workspace, ce = _workspace()
    workspace.open(0, [(1, 0)])
    with pytest.raises(RuntimeError, match="in flight"):
        workspace.open(1, [(1, 1)])
    with pytest.raises(RuntimeError, match="in flight"):
        workspace._reserve(512, 2, 3)
    workspace.done(0)
    workspace.open(1, [(1, 1)])
    workspace.done(1)
    assert ce.events == [
        ("open", ((1, 0),)),
        "completion fence",
        ("open", ((1, 1),)),
        "completion fence",
    ]
    workspace.close()
    assert ce.pool.released


@pytest.mark.parametrize("recoverable", [False, True])
def test_workspace_recovers_before_next_boundary(recoverable):
    workspace, ce = _workspace()
    ce.recoverable = recoverable
    workspace.open(0, [(1, 0)])
    workspace.reset()
    assert ce.events == [("open", ((1, 0),)), "interrupted"]
    if recoverable:
        workspace.open(1, [(1, 1)])
        workspace.done(1)
        assert ce.events[-3:] == ["recover", ("open", ((1, 1),)), "completion fence"]
    else:
        with pytest.raises(ws.CeUnavailableError, match="could not recover"):
            workspace.open(1, [(1, 1)])
        assert ce.events[-1] == "recover"
        assert not any(workspace._in_flight)


def test_workspace_capture_blocks_growth_and_recovery(monkeypatch):
    workspace, ce = _workspace()
    assert workspace._reserve(512, 2, 3)
    cached = workspace.view((0, "payload"), lambda: torch.ones(8))
    monkeypatch.setattr(ws, "_is_capturing", lambda: True)
    workspace.open(0, [(1, 0)])
    workspace.done(0)
    assert workspace._captured
    monkeypatch.setattr(ws, "_is_capturing", lambda: False)
    assert not workspace._reserve(512, 2, 3)
    assert workspace.view((0, "payload"), lambda: None) is cached
    with pytest.raises(RuntimeError, match="grow after a CUDA graph"):
        workspace._reserve(4096, 2, 3)
    monkeypatch.setattr(ws, "_is_capturing", lambda: True)
    workspace.open(1, [(1, 1)])
    workspace.reset()
    with pytest.raises(RuntimeError, match="Recover CE eagerly"):
        workspace.open(0, [(1, 0)])
    assert "recover" not in ce.events


def test_workspace_growth_discards_views():
    workspace, _ = _workspace()
    workspace._reserve(512, 2, 3)
    old = workspace.view((0, "payload"), lambda: torch.ones(8))
    assert workspace._reserve(4096, 2, 3)
    new = workspace.view((0, "payload"), lambda: torch.zeros(8))
    assert new is not old


def test_state_uses_collective_transport_factory(monkeypatch):
    ce = _Transport()
    calls = []

    class State(ws.CeState):
        def _make_transport(self):
            self.transport = ws.CeWorkspace(self.ce, 2, 0, torch.device("cpu"), slots=2)

    def create(group, device, min_channels):
        calls.append((group, device, min_channels))
        return ce

    monkeypatch.setattr(ws.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(ws.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(ws.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ws.CeTransport, "create", create)
    group = SimpleNamespace(group_name="test")
    state = State(group, "test")
    assert state.probe_once(min_channels=3, what="gather")
    assert state.probe_once(min_channels=3, what="gather")
    assert calls == [(group, torch.device("cpu"), 3)]
    assert state.transport.ce is ce and state.pool is ce.pool
    state.close()
    assert ce.pool.released
