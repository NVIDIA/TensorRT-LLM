# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix16 worker's managed draft-cache writers issue the pending draft-history readback first.

The last snapshot's gathers (``snapshot_managed_draft_history``) read ``_ctx_len`` and
``_ctx_position_offset``. They must be queued before any later write to that state. The target's
``_write_ctx_len`` and ``_write_slot_values`` stage their scatter into the step's input commit
when one is open and otherwise take the engine's copy. Either way they issue the pending gathers
first (``_issue_pending_history``), so every caller is covered:

* each writer, staged and copied, issues a pending readback once, before its write, and nothing
  when none is pending;
* the engine's ``_assign_slot`` with ``reset`` clears the request's old slot through
  ``_write_ctx_len`` before its own issue: the readback still comes first;
* the engine's ``_prepare_managed_history`` (the main call site, which issues first) issues once
  and makes the same writes, in the same order, as with nothing pending: no extra launches.

Read from the sources (no GPU): the target's two writers over the engine's methods they inherit,
on CPU tensors, with the commit staging and the readback replaced by recorders.
"""

from __future__ import annotations
import __future__

import ast
import importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch.overrides import TorchFunctionMode

pytestmark = pytest.mark.cpu_only

_PKG = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_TARGET = (
    _PKG
    / "_torch/_experimental/modeling_v2/models/kimi_k31"
    / "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1/modeling.py"
)
_ENGINE = _PKG / "_torch/speculative/dflash.py"

_ENGINE_METHODS = (
    "_issue_pending_history",
    "_write_ctx_len",
    "_write_slot_values",
    "_assign_slot",
    "_prepare_managed_history",
)
_TARGET_METHODS = ("_write_ctx_len", "_write_slot_values")
NUM_SLOTS = 9
_DUMMY_REQUEST_ID = -7
# The device writes the recorder logs: the engine's copies and the slot reset.
_WRITES = {"index_copy_": "copy", "zero_": "zero", "copy_": "copy"}


def _methods(path: Path, cls: str, names: tuple[str, ...]) -> list[ast.FunctionDef]:
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == cls
    )
    found = {f.name: f for f in node.body if isinstance(f, ast.FunctionDef) and f.name in names}
    assert set(found) == set(names), f"{path.name}: {sorted(set(names) - set(found))} moved"
    return [found[name] for name in names]


def _classes(events: list, staged: bool) -> type:
    """The target's writers (``_Target``) over the engine's methods (``_Engine``)."""

    def stage_scatter(target, slots, values):
        if not staged:
            return False
        events.append(("stage", _name(target), dict(zip(slots, values))))
        return True

    def cls(name: str, bases: list, body: list) -> ast.ClassDef:
        return ast.ClassDef(
            name=name, bases=bases, keywords=[], body=body, decorator_list=[], type_params=[]
        )

    module = ast.Module(
        body=[
            cls("_Engine", [], _methods(_ENGINE, "DFlashWorker", _ENGINE_METHODS)),
            cls(
                "_Target",
                [ast.Name(id="_Engine", ctx=ast.Load())],
                _methods(_TARGET, "DFlashWorker", _TARGET_METHODS),
            ),
        ],
        type_ignores=[],
    )
    ns = {
        "torch": torch,
        "Optional": Optional,
        "prefer_pinned": lambda: False,
        "ATTENTION_DP_DUMMY_REQUEST_ID": _DUMMY_REQUEST_ID,
        "stage_scatter": stage_scatter,
    }
    code = compile(
        ast.fix_missing_locations(module),
        "<helix16 writers>",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, ns)
    return ns["_Target"]


_NAMES: dict[int, str] = {}


def _name(t: torch.Tensor) -> str:
    return _NAMES.get(t.untyped_storage().data_ptr(), "other")


class _Pending:
    """A snapshot's readback: ``issue`` queues its gathers (logged)."""

    def __init__(self, events: list):
        self.events = events

    def issue(self):
        self.events.append(("issue",))


class _Writes(TorchFunctionMode):
    """Logs the device writes to the worker's state (the engine's copies, the slot reset)."""

    def __init__(self, events: list):
        super().__init__()
        self.events = events

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kind = _WRITES.get(getattr(func, "__name__", ""))
        if kind is not None and isinstance(args[0], torch.Tensor):
            self.events.append((kind, _name(args[0])))
        return func(*args, **(kwargs or {}))


class _Caches(dict):
    """The manager's ``kv_cache_map``: one cache object per request id."""

    def __missing__(self, request_id):
        self[request_id] = object()
        return self[request_id]


def _worker(staged: bool, pending: bool, histories: Optional[dict] = None):
    events: list = []
    w = object.__new__(_classes(events, staged))
    w._ctx_len = torch.full((NUM_SLOTS,), -9, dtype=torch.long)
    w._ctx_position_offset = torch.full((NUM_SLOTS,), -8, dtype=torch.long)
    w._managed_snapshot_slots = torch.full((8,), -6, dtype=torch.long)
    _NAMES.clear()
    _NAMES[w._ctx_len.untyped_storage().data_ptr()] = "ctx_len"
    _NAMES[w._ctx_position_offset.untyped_storage().data_ptr()] = "offset"
    _NAMES[w._managed_snapshot_slots.untyped_storage().data_ptr()] = "snapshot"
    w._ctx_len_host = [0] * NUM_SLOTS
    w._dummy_slot = NUM_SLOTS - 1
    w._req_to_slot = {}
    w._req_ctx_pos = {}
    w._free_slots = deque(range(NUM_SLOTS - 1))
    w._managed_cache_bindings = {}
    # The worker's default without a draft window: the whole history is restored.
    w._history_window = None
    w._has_unified_draft_cache = lambda: True
    w._rebuild_batch_to_slot = lambda rids: events.append(("batch_to_slot", tuple(rids)))
    w._pending_history = _Pending(events) if pending else None
    w._ctx_kv_manager = SimpleNamespace(
        _draft_dummy_request_ids=set(),
        kv_cache_map=_Caches(),
        get_draft_history=lambda rid: (histories or {}).get(rid),
    )
    w._graph_dummy_id_floor = 1 << 40
    return w, events


def _state(w) -> tuple:
    return (w._ctx_len.clone(), w._ctx_position_offset.clone(), list(w._ctx_len_host))


@pytest.mark.parametrize("staged", [False, True], ids=["copied", "staged"])
@pytest.mark.parametrize("writer", ["_write_ctx_len", "_write_slot_values"])
def test_each_writer_issues_the_pending_readback_first(staged: bool, writer: str) -> None:
    w, events = _worker(staged, pending=True)

    def write(values: dict) -> None:
        if writer == "_write_ctx_len":
            w._write_ctx_len(values)
        else:
            w._write_slot_values(w._ctx_position_offset, values)

    name = "ctx_len" if writer == "_write_ctx_len" else "offset"
    with _Writes(events):
        write({})  # nothing to write: nothing issued
        assert events == [] and w._pending_history is not None
        write({2: 5, 4: 6})
        first = list(events)
        write({3: 7})  # nothing pending now
    kind = "stage" if staged else "copy"
    assert first[0] == ("issue",)
    assert [e[:2] for e in first[1:]] == [(kind, name)]
    assert w._pending_history is None
    assert [e[:2] for e in events[len(first) :]] == [(kind, name)]


@pytest.mark.parametrize("staged", [False, True], ids=["copied", "staged"])
def test_reset_issues_the_pending_readback_before_clearing_the_old_slot(staged: bool) -> None:
    w, events = _worker(staged, pending=True)
    w._req_to_slot = {7: 2}
    w._free_slots = deque([3, 4])
    with _Writes(events):
        slot = w._assign_slot(7, reset=True)
    assert slot == 3 and w._req_to_slot == {7: 3}
    assert events[0] == ("issue",), events
    assert events.count(("issue",)) == 1
    kind = "stage" if staged else "copy"
    writes = [e[:2] for e in events[1:]]
    # The old slot's length is cleared, then the new slot's offset and length.
    assert writes == [(kind, "ctx_len"), ("zero", "offset"), (kind, "ctx_len")]
    assert w._ctx_len_host[2] == 0 and w._ctx_len_host[3] == 0


@pytest.mark.parametrize("staged", [False, True], ids=["copied", "staged"])
def test_main_path_issues_once_and_adds_no_launches(staged: bool) -> None:
    """``_prepare_managed_history`` over arriving, continuing and leaving requests: with a pending
    readback it issues it once, first, then makes exactly the writes (and leaves exactly the
    state) of the same steps with nothing pending."""
    histories = {
        100: SimpleNamespace(valid_length=4096, position=4100),
        101: SimpleNamespace(valid_length=77, position=80),
        102: SimpleNamespace(valid_length=900001, position=900005),
    }
    steps = [[100], [100], [100, 101], [101, 102], [102]]
    runs = {}
    for pending in (False, True):
        w, events = _worker(staged, pending=False, histories=histories)
        per_step = []
        with _Writes(events):
            for rids in steps:
                start = len(events)
                if pending:
                    w._pending_history = _Pending(events)
                w._prepare_managed_history(rids)
                per_step.append(events[start:])
        runs[pending] = (per_step, _state(w), dict(w._req_to_slot))
    plain, issued = runs[False], runs[True]
    for step, (without, with_) in enumerate(zip(plain[0], issued[0])):
        assert ("issue",) not in without, step
        assert with_[0] == ("issue",) and with_.count(("issue",)) == 1, step
        assert with_[1:] == without, step
    assert all(torch.equal(a, b) for a, b in zip(plain[1][:2], issued[1][:2]))
    assert plain[1][2] == issued[1][2] and plain[2] == issued[2]
