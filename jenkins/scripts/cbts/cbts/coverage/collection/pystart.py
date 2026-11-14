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

import configparser
import contextlib
import os
import secrets
import socket
import sys
import threading
from fnmatch import fnmatchcase
from pathlib import Path
from types import CodeType
from typing import Iterable, Mapping, Optional

from cbts.coverage.collection.compact_db import canonicalize_path, write_leaf_database

_MON = getattr(sys, "monitoring", None)
_DEFAULT_TOOL_ID = int(os.environ.get("CBTS_PYSTART_TOOL_ID", "4"))
REPOSITORY_ROOT = str(Path(__file__).resolve().parents[6])

# (process_uid, context, kind, reason) nobody can vouch for.
Taint = tuple[str, str, str, str]


def read_source_config(path: str | os.PathLike[str] | None = None) -> configparser.ConfigParser:
    """Read collection settings with the checkout path resolved in this process."""
    config = configparser.ConfigParser(defaults={"repository_root": REPOSITORY_ROOT})
    with open(path or Path(__file__).with_name("coveragerc.template"), encoding="utf-8") as stream:
        config.read_file(stream)
    return config


class PyStartTracker:
    """Per-process tracker; one SQLite data file per process, merged (unioned) downstream."""

    def __init__(
        self,
        source_roots: Iterable[str],
        data_dir: str,
        stage: str = "stage",
        tool_id: int = _DEFAULT_TOOL_ID,
        *,
        repository_root: Optional[str] = None,
        include: Iterable[str] = (),
    ) -> None:
        self.source_roots = tuple(os.path.realpath(p).rstrip("/") + "/" for p in source_roots if p)
        self.repository_root = os.path.realpath(repository_root) if repository_root else None
        self.include = tuple(include)
        self.data_dir = data_dir
        self.stage = stage
        self.tool_id = tool_id
        self._ctx: str = os.environ.get("CBTS_TEST_ID", "") or ""
        self._data: dict[str, set[tuple[str, str]]] = {}  # context -> set((filename, qualname))
        self._outcomes: dict[str, str] = {}  # context -> pytest outcome (outer pytest only)
        self._taints: set[Taint] = set()
        self._diagnostics: dict[str, dict[str, str]] = {}  # process_uid -> {key: value}
        # co_filename -> absolute source path, or "" if excluded.
        self._source_files: dict[str, str] = {}
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
        if not filename or filename[0] == "<" or not filename.endswith(".py"):
            return False
        absolute = os.path.realpath(filename)
        if not absolute.startswith(self.source_roots):
            return False
        relative = canonicalize_path(absolute, self.repository_root)
        return not self.include or any(fnmatchcase(relative, pattern) for pattern in self.include)

    # Skip synthetic genexpr / lambda frames. See COLLECTION.md §5.1 for detail.
    _SKIP_QUALNAMES = frozenset(("<genexpr>", "<lambda>"))

    def _on_py_start(self, code: CodeType, offset: int):
        # A tracker fault must never propagate into monitored host code.
        with contextlib.suppress(Exception):
            filename = code.co_filename
            fn = self._source_files.get(filename)
            if fn is None:
                fn = self._source_files[filename] = (
                    os.path.realpath(filename) if self._in_source(filename) else ""
                )
            if fn:
                qual = code.co_qualname
                if qual.rpartition(".")[2] not in self._SKIP_QUALNAMES:
                    self._data.setdefault(self._ctx, set()).add((fn, qual))
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
            with contextlib.suppress(Exception):
                _MON.free_tool_id(self.tool_id)
            return False
        self._active = True
        return True

    def switch_test_context(self, nodeid: Optional[str]) -> None:
        self._ctx = nodeid or ""
        if self._active:
            # Same contract as _on_py_start: a tracker fault must never propagate into
            # the caller, which here is pytest's own per-test hookwrapper (or a
            # subscriber's context-channel callback) -- never the place to decide
            # whether a test runs.
            with contextlib.suppress(Exception):
                _MON.restart_events()

    def record_outcome(self, nodeid: Optional[str], outcome: str) -> None:
        """Record a test's pytest outcome (passed/failed/skipped) for the completeness signal."""
        self._outcomes[nodeid or ""] = outcome

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

    def note_process_diagnostic(self, process_uid: str, facts: Mapping[str, str]) -> None:
        """Record arbitrary ``{key: value}`` facts about another process.

        Diagnostic only, with no fixed fact set -- never read back by this tracker
        itself, just carried through to the leaf database for whoever reads the merged
        report. See ``ContextServer.diagnostics`` for the current source.
        """
        if process_uid and facts:
            self._diagnostics.setdefault(process_uid, {}).update(facts)

    def save(self) -> Optional[str]:
        # Write a compact per-process SQLite that downstream reducers can union directly.
        snap = self._data.copy()  # atomic shallow copy; each set snapshotted below
        outcomes = dict(self._outcomes)
        taints = set(self._taints)
        diagnostics = dict(self._diagnostics)
        if not snap and not outcomes and not taints and not diagnostics:
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
                taints=taints,
                diagnostics=diagnostics,
                repository_root=self.repository_root,
            )
            os.replace(tmp, path)
        return path

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        with contextlib.suppress(Exception):
            _MON.set_events(self.tool_id, 0)
            _MON.free_tool_id(self.tool_id)
