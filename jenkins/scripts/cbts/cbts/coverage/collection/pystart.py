# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Function/class-level per-test coverage tracker via sys.monitoring PY_START (Python 3.12+)."""

from __future__ import annotations

import os
import secrets
import socket
import sys
import threading
from types import CodeType
from typing import Iterable, Optional

from cbts.coverage.collection.compact_db import write_leaf_database

_MON = getattr(sys, "monitoring", None)
_DEFAULT_TOOL_ID = int(os.environ.get("CBTS_PYSTART_TOOL_ID", "4"))

# (process_uid, context, kind, reason) nobody can vouch for.
Taint = tuple[str, str, str, str]


class PyStartTracker:
    """Per-process tracker; one SQLite data file per process, merged (unioned) downstream."""

    def __init__(
        self,
        source_roots: Iterable[str],
        data_dir: str,
        stage: str = "stage",
        tool_id: int = _DEFAULT_TOOL_ID,
    ) -> None:
        self.source_roots = tuple(os.path.abspath(p).rstrip("/") + "/" for p in source_roots if p)
        self.data_dir = data_dir
        self.stage = stage
        self.tool_id = tool_id
        self._ctx: str = os.environ.get("CBTS_TEST_ID", "") or ""
        self._data: dict[str, set[tuple[str, str]]] = {}  # context -> set((filename, qualname))
        self._outcomes: dict[str, str] = {}  # context -> pytest outcome (outer pytest only)
        self._expected: dict[str, int] = {}  # context -> pool workers the coordinator spawned
        self._taints: set[Taint] = set()
        self._file_ok: dict[str, bool] = {}  # co_filename -> bool (cached source-membership)
        self._active = False
        self._save_lock = threading.Lock()  # serialize periodic-thread and atexit saves
        self._suffix = ""
        self._new_suffix()

    @property
    def process_uid(self) -> str:
        """Identity of this process's leaf database, stable for its lifetime."""
        return f"{self.stage}/{self._suffix}.pid{os.getpid()}"

    @property
    def available(self) -> bool:
        return _MON is not None and bool(self.source_roots)

    def _new_suffix(self) -> None:
        # No pid here: save() adds the live pid, so a forked child writes to a distinct file without
        # an os.register_at_fork handler (which is unsafe to run in an MPI/UCX/Ray process's fork).
        self._suffix = f"{socket.gethostname()}.X{secrets.token_urlsafe(6)}"

    def _in_source(self, filename: str) -> bool:
        if not filename or filename[0] == "<":
            return False
        return os.path.abspath(filename).startswith(self.source_roots)

    # Skip synthetic comprehension / genexpr / lambda frames; keep real functions, methods, module bodies.
    _SKIP_QUALNAMES = frozenset(("<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>", "<lambda>"))

    def _on_py_start(self, code: CodeType, offset: int):
        try:
            fn = code.co_filename
            ok = self._file_ok.get(fn)
            if ok is None:
                ok = self._file_ok[fn] = self._in_source(fn)
            if ok:
                qual = code.co_qualname
                if "<locals>" not in qual and qual not in self._SKIP_QUALNAMES:
                    self._data.setdefault(self._ctx, set()).add((fn, qual))
        except Exception:
            # A tracker fault must never propagate into monitored host code.
            pass
        # Disable this code object's PY_START (for this tool) until the next test's restart_events().
        return _MON.DISABLE

    def start(self) -> bool:
        if not self.available:
            return False
        try:
            _MON.use_tool_id(self.tool_id, "cbts-pystart")
        except ValueError:
            return False
        try:
            _MON.register_callback(self.tool_id, _MON.events.PY_START, self._on_py_start)
            _MON.set_events(self.tool_id, _MON.events.PY_START)
        except Exception:
            try:
                _MON.free_tool_id(self.tool_id)
            except Exception:
                pass
            return False
        self._active = True
        return True

    def switch_test_context(self, nodeid: Optional[str]) -> None:
        self._ctx = nodeid or ""
        if self._active:
            _MON.restart_events()

    def record_outcome(self, nodeid: Optional[str], outcome: str) -> None:
        """Record a test's pytest outcome (passed/failed/skipped) for the completeness signal."""
        self._outcomes[nodeid or ""] = outcome

    def note_expected_workers(self, nodeid: Optional[str], n: int) -> None:
        """Add to the count of subprocess pool workers the coordinator spawned for a test."""
        key = nodeid or ""
        self._expected[key] = self._expected.get(key, 0) + int(n)

    def note_taint(
        self, process_uid: Optional[str], nodeid: Optional[str], kind: str, reason: str
    ) -> None:
        """Record that a process's coverage for a test cannot be vouched for.

        ``process_uid=None`` means this process: the one that owns the context channel
        also records taints about the subscribers it lost, so a taint always names whose
        rows are in doubt, and an empty string (an unidentified subscriber) must not be
        conflated with "this process" -- only ``None`` substitutes ``self.process_uid``.
        """
        self._taints.add(
            (self.process_uid if process_uid is None else process_uid, nodeid or "", kind, reason)
        )

    def save(self) -> Optional[str]:
        # Write a compact per-process SQLite that downstream reducers can union directly.
        snap = self._data.copy()  # atomic shallow copy; each set snapshotted below
        outcomes = dict(self._outcomes)
        expected = dict(self._expected)
        taints = set(self._taints)
        if not snap and not outcomes and not expected and not taints:
            return None
        # Serialize saves so the periodic-save thread and the atexit final save never race on the
        # shared temp file (a concurrent os.remove(tmp) mid-write surfaces as a sqlite disk I/O error).
        with self._save_lock:
            os.makedirs(self.data_dir, exist_ok=True)
            path = os.path.join(
                self.data_dir, f".cbtscov.{self.stage}.{self._suffix}.pid{os.getpid()}.sqlite"
            )
            # Keep in-progress files outside the .cbtscov.* input namespace. A report or archive
            # running concurrently must only ever discover the atomically published final path.
            tmp = os.path.join(self.data_dir, f".tmp-{os.path.basename(path)}")
            if os.path.exists(tmp):
                os.remove(tmp)
            process_uid = self.process_uid
            write_leaf_database(
                tmp,
                stage=self.stage,
                process_uid=process_uid,
                touches={context: symbols.copy() for context, symbols in snap.items()},
                outcomes=outcomes,
                expected_workers=expected,
                taints=taints,
            )
            os.replace(tmp, path)
        return path

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        try:
            _MON.set_events(self.tool_id, 0)
            _MON.free_tool_id(self.tool_id)
        except Exception:
            pass
