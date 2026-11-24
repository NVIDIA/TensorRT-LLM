# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ULL helix16 generation server numbers a windowed drafter history as a whole one.

The generation server's cache keeps the drafter's window of its history (the drafter checkpoint's
``sliding_window``, ``kimi_k31_draft_window`` on) as the context server's does, so a request
arrives with ``(valid_length, position) = (min(prompt, window), prompt)``; a whole-history cache
received ``(prompt, prompt)``. The engine's worker binds the history (``_prepare_managed_history``)
and numbers the drafter's context by position for a windowed cache and by ``valid_length`` for a
whole one: both give the slot the context length ``prompt`` and position offset 0, and so they
stay through every step's acceptance (``_advance_ctx_len``); the published history
(``snapshot_managed_draft_history``) has the same position and the windowed cache's
``valid_length`` stays within its window. The drafter's kernels read the positions
``[L - window, L)`` of that numbering through the page table, the same rows either way
(``ops/spec/test_rubin_ull_dflash2_windowed_history.py``, GPU).

Read from the sources (no GPU): the engine's methods with each ULL generation target's writers over
them, on CPU tensors, with the commit staging off.
"""

from __future__ import annotations
import __future__

import ast
import functools
import importlib.util
from collections import deque
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Optional

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG / "_torch/_experimental/modeling_v2/models/kimi_k31"
_ENGINE = _PKG / "_torch/speculative/dflash.py"
# The helix16 generation targets of this tree.
_TARGETS = tuple(
    target
    for target in (
        "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1",
        "kimi_k31_pro_nvfp4__sm_107__helix16_wide",
    )
    if (_FAMILY / target / "modeling.py").is_file()
)
_ENGINE_METHODS = (
    "_issue_pending_history",
    "_write_ctx_len",
    "_write_slot_values",
    "_assign_slot",
    "_prepare_managed_history",
    "_advance_ctx_len",
    "snapshot_managed_draft_history",
)
# The writers a target may define over the engine's.
_TARGET_WRITERS = ("_write_ctx_len", "_write_slot_values")
NUM_SLOTS = 9
_REQUEST = 100
_DUMMY_REQUEST_ID = -7
# The drafter's position limit is far above every prompt here: no context-length cap applies.
_MAX_CTX = 1 << 30


@functools.lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class_methods(path: Path, cls: str, names: tuple[str, ...]) -> list[ast.FunctionDef]:
    node = next(n for n in _parse(path).body if isinstance(n, ast.ClassDef) and n.name == cls)
    return [f for f in node.body if isinstance(f, ast.FunctionDef) and f.name in names]


def _draft_window(target: str) -> int:
    """The drafter's attention window the target decodes with (its weights.py ``DRAFT_SHAPE``)."""
    for node in _parse(_FAMILY / target / "weights.py").body:
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == [
            "DRAFT_SHAPE"
        ]:
            shape = eval(
                compile(ast.Expression(node.value), "<DRAFT_SHAPE>", "eval"),
                {"MappingProxyType": MappingProxyType},
            )
            return shape["sliding_window"]
    raise AssertionError(f"{target}/weights.py has no DRAFT_SHAPE")


class _Deferred:
    """A snapshot's readback: ``read`` gathers the published ``(valid_length, position)``."""

    def __init__(self, manager, request_ids, read):
        self.request_ids, self.read = request_ids, read

    def issue(self) -> None:
        pass


def _draft_history_values(ctx_len, position_offset, slots):
    lengths = ctx_len[slots]
    return torch.stack((lengths, lengths + position_offset[slots]), dim=1)


@functools.lru_cache(maxsize=None)
def _worker_class(target: str) -> type:
    """The target's writers (``_Target``) over the engine's methods (``_Engine``)."""
    engine = _class_methods(_ENGINE, "DFlashWorker", _ENGINE_METHODS)
    assert {f.name for f in engine} == set(_ENGINE_METHODS), "an engine method moved"
    writers = _class_methods(_FAMILY / target / "modeling.py", "DFlashWorker", _TARGET_WRITERS)

    def cls(name: str, bases: list, body: list) -> ast.ClassDef:
        return ast.ClassDef(
            name=name, bases=bases, keywords=[], body=body, decorator_list=[], type_params=[]
        )

    module = ast.Module(
        body=[
            cls("_Engine", [], engine),
            cls("_Target", [ast.Name(id="_Engine", ctx=ast.Load())], writers or [ast.Pass()]),
        ],
        type_ignores=[],
    )
    ns = {
        "torch": torch,
        "Optional": Optional,
        "prefer_pinned": lambda: False,
        "ATTENTION_DP_DUMMY_REQUEST_ID": _DUMMY_REQUEST_ID,
        "stage_scatter": lambda target, slots, values: False,
        "draft_history_values": _draft_history_values,
        "DeferredDraftHistory": _Deferred,
    }
    code = compile(
        ast.fix_missing_locations(module),
        f"<{target} draft history>",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, ns)
    return ns["_Target"]


def _worker(target: str, window: Optional[int], history: tuple[int, int]):
    """A worker bound to a unified draft cache with ``window`` as its ``draft_window_size``, whose
    manager published ``history`` ``(valid_length, position)`` for ``_REQUEST``."""
    w = object.__new__(_worker_class(target))
    w._ctx_len = torch.zeros(NUM_SLOTS, dtype=torch.long)
    w._ctx_position_offset = torch.zeros(NUM_SLOTS, dtype=torch.long)
    w._managed_snapshot_slots = torch.zeros(8, dtype=torch.long)
    w._ctx_len_host = [0] * NUM_SLOTS
    w._dummy_slot = NUM_SLOTS - 1
    w._max_ctx = _MAX_CTX
    w._req_to_slot = {}
    w._req_ctx_pos = {}
    w._free_slots = deque(range(NUM_SLOTS - 1))
    w._managed_cache_bindings = {}
    w._managed_request_ids = ()
    w._pending_history = None
    w._history_window = window
    w._history_retention = window + 1 if window is not None else None
    w._has_unified_draft_cache = lambda: True
    w._rebuild_batch_to_slot = lambda request_ids: None
    cache = object()
    histories = {_REQUEST: SimpleNamespace(valid_length=history[0], position=history[1])}
    w._ctx_kv_manager = SimpleNamespace(
        _draft_dummy_request_ids=set(),
        kv_cache_map={_REQUEST: cache},
        get_draft_history=histories.get,
    )
    w._graph_dummy_id_floor = 1 << 40
    return w


def _numbering(w) -> tuple[int, int, int, int]:
    """The slot's (context length, its host mirror, position offset, request position)."""
    slot = w._req_to_slot[_REQUEST]
    return (
        int(w._ctx_len[slot]),
        int(w._ctx_len_host[slot]),
        int(w._ctx_position_offset[slot]),
        int(w._req_ctx_pos[_REQUEST]),
    )


@pytest.mark.parametrize("prompt", [5, 700, 1023, 1024, 1025, 4133, 190000])
@pytest.mark.parametrize("target", _TARGETS)
def test_windowed_history_numbers_the_drafter_context_as_the_whole_history(target, prompt):
    window = _draft_window(target)
    whole = _worker(target, None, (prompt, prompt))
    windowed = _worker(target, window, (min(prompt, window), prompt))
    for w in (whole, windowed):
        assert w._prepare_managed_history([_REQUEST]) == [0]
    assert _numbering(whole) == _numbering(windowed) == (prompt, prompt, 0, prompt)

    # Steps: the accepted tokens of each verify (0..7 drafts plus the bonus token).
    length = prompt
    for accepted in (1, 8, 3, 8, 8, 2, 5):
        slots = torch.tensor([whole._req_to_slot[_REQUEST]])
        for w in (whole, windowed):
            w._advance_ctx_len(slots, torch.tensor([accepted]))
        length += accepted
        published = {}
        for name, w in (("whole", whole), ("windowed", windowed)):
            pending = w.snapshot_managed_draft_history()
            assert pending.request_ids == (_REQUEST,)
            published[name] = tuple(int(v) for v in pending.read()[0])
            slot = w._req_to_slot[_REQUEST]
            assert (int(w._ctx_len[slot]), int(w._ctx_position_offset[slot])) == (length, 0)
        assert published["whole"] == (length, length)
        assert published["windowed"] == (min(length, window), length)
        # The next step's binding is unchanged: nothing is renumbered.
        for w in (whole, windowed):
            assert w._prepare_managed_history([_REQUEST]) == []
        assert torch.equal(whole._ctx_len, windowed._ctx_len)
        assert torch.equal(whole._ctx_position_offset, windowed._ctx_position_offset)
