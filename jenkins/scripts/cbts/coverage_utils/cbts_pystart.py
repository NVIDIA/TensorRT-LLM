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

import os
import secrets
import socket
import sqlite3
import sys

_MON = getattr(sys, "monitoring", None)
_DEFAULT_TOOL_ID = int(os.environ.get("CBTS_PYSTART_TOOL_ID", "4"))


class PyStartTracker:
    """Per-process tracker; one SQLite data file per process, merged (unioned) downstream."""

    def __init__(self, source_roots, data_dir, stage="stage", tool_id=_DEFAULT_TOOL_ID):
        self.source_roots = tuple(os.path.abspath(p).rstrip("/") + "/" for p in source_roots if p)
        self.data_dir = data_dir
        self.stage = stage
        self.tool_id = tool_id
        self._ctx = os.environ.get("CBTS_TEST_ID", "") or ""
        self._data = {}  # context -> set((filename, qualname))
        self._outcomes = {}  # context -> pytest outcome (filled only in the outer pytest process)
        self._expected = {}  # context -> pool workers the coordinator spawned for it
        self._file_ok = {}  # co_filename -> bool (cached source-membership)
        self._active = False
        self._new_suffix()

    @property
    def available(self):
        return _MON is not None and bool(self.source_roots)

    def _new_suffix(self):
        self._suffix = f"{socket.gethostname()}.pid{os.getpid()}.X{secrets.token_urlsafe(6)}"

    def _in_source(self, filename):
        if not filename or filename[0] == "<":
            return False
        return os.path.abspath(filename).startswith(self.source_roots)

    # Skip synthetic comprehension / genexpr / lambda frames; keep real functions, methods, module bodies.
    _SKIP_QUALNAMES = frozenset(("<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>", "<lambda>"))

    def _on_py_start(self, code, offset):
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

    def start(self):
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
        try:
            os.register_at_fork(after_in_child=self._after_fork_child)
        except (AttributeError, ValueError):
            pass
        return True

    def _after_fork_child(self):
        # The child writes its own data file and rediscovers what it runs.
        self._new_suffix()
        self._data = {}
        if self._active:
            try:
                _MON.restart_events()
            except Exception:
                pass

    def switch_test_context(self, nodeid):
        self._ctx = nodeid or ""
        if self._active:
            _MON.restart_events()

    def record_outcome(self, nodeid, outcome):
        """Record a test's pytest outcome (passed/failed/skipped) for the completeness signal."""
        self._outcomes[nodeid or ""] = outcome

    def note_expected_workers(self, nodeid, n):
        """Add to the count of subprocess pool workers the coordinator spawned for a test."""
        key = nodeid or ""
        self._expected[key] = self._expected.get(key, 0) + int(n)

    def save(self):
        # Write a per-process SQLite the downstream merge reads directly; uploaded compressed only.
        snap = self._data.copy()  # atomic shallow copy; each set snapshotted below
        outcomes = dict(self._outcomes)
        expected = dict(self._expected)
        if not snap and not outcomes and not expected:
            return None
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, f".cbtscov.{self.stage}.{self._suffix}.sqlite")
        tmp = path + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        con = sqlite3.connect(tmp)
        try:
            con.execute("CREATE TABLE touch (test TEXT, file TEXT, qualname TEXT)")
            rows = ((ctx, f, q) for ctx, fs in snap.items() for (f, q) in fs.copy())
            con.executemany("INSERT INTO touch VALUES (?, ?, ?)", rows)
            # Stage rides in the file content so the merge attributes rows without parsing the filename.
            con.execute("CREATE TABLE proc_meta (stage TEXT)")
            con.execute("INSERT INTO proc_meta VALUES (?)", (self.stage,))
            # Per-test completeness signal; only the coordinator process fills outcome / expected.
            con.execute(
                "CREATE TABLE test_meta "
                "(test TEXT PRIMARY KEY, outcome TEXT, expected_workers INTEGER)"
            )
            con.executemany(
                "INSERT OR REPLACE INTO test_meta VALUES (?, ?, ?)",
                [(k, outcomes.get(k), expected.get(k, 0)) for k in set(outcomes) | set(expected)],
            )
            con.commit()
        finally:
            con.close()
        os.replace(tmp, path)
        return path

    def stop(self):
        if not self._active:
            return
        self._active = False
        try:
            _MON.set_events(self.tool_id, 0)
            _MON.free_tool_id(self.tool_id)
        except Exception:
            pass
