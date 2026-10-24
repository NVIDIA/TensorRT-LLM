# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The test-result cache: which tests a change can affect, and what the others already showed.

Every recorded pytest run (``testcache_driver.py``, run in the tree under
test) leaves a session record under ``<cache>/sessions/``: each test's
outcome, and every input the process ran on, each with the index of the
test it was first seen in. A recorded result **holds at commit X** when the
test ran at X itself (on a clean tree, in the same environment), or when
nothing it ran on differs at X:

- a Python module it loaded: unchanged, or changed only inside the bodies of
  functions no test up to it in that process started. A change anywhere
  else in the file (module-level code, a class body, a signature, a
  decorator, or the body of a function whose decorator compiles or traces it
  -- ``jit``, ``compile``, ``script``, ``kernel``, ``autotune``, ...) counts
  for every test that loaded the module;
- a file it read: byte for byte;
- a directory it listed: the same entries; a path it asked about: there or
  not there alike; a module name the import system searched for and found
  outside the tree: no file in the tree now answers to it;
- the tree's shared libraries it loaded: no native source changed (``cpp/``,
  ``.cu``, ``.cpp``, ``.h``, CMake), and the library on disk, when the tree
  at X is the one given, is the one it ran with;
- it did not start, before or during it, a process that runs Python or
  builds code, and nothing compiled a frame (it is not **opaque**).

So a test is **affected** by a change from A to B when no recorded result
holds at B. :func:`affected` answers that for a selection of test files:
which files to run whole (never recorded, or what their collection read
changed, so the test ids themselves may have), which test ids to run, and
which results are reused, with the commit each ran on. :func:`compare`
reads the results at B against those at A and names the regressions.
The base tree is run only for a test that fails at B and has no result
that holds at A.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .testcache_driver import RECORD_VERSION, SESSIONS_DIRNAME, file_blob_id

#: The driver, run by path with the tree's python.
DRIVER = Path(__file__).resolve().parent / "testcache_driver.py"
#: ``<workspace>/<this>`` is a run's cache.
CACHE_DIRNAME = "test_cache"
#: The cache directory when neither ``--cache`` nor ``--workspace`` is given.
CACHE_ENV = "TRTLLM_MODELING_V2_TEST_CACHE"
#: Changed files that change the tree's shared libraries once they are rebuilt.
NATIVE_SOURCES = (
    "cpp/*",
    "*.cu",
    "*.cuh",
    "*.cpp",
    "*.cc",
    "*.c",
    "*.h",
    "*.hpp",
    "*.hh",
    "*CMakeLists.txt",
    "*.cmake",
)
#: A pytest test file.
TEST_FILE = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")
#: A decorator that compiles, traces or reads the function instead of running it as Python.
COMPILING_DECORATOR = re.compile(
    r"jit|compile|script|kernel|autotune|heuristic|cute|triton|numba|dsl|trace", re.IGNORECASE
)
FAILING = ("failed", "error")
#: An index past every test's: nothing invalidates the record.
NEVER = 1 << 30
#: Exit codes of ``compare``.
EXIT_CLEAN, EXIT_REGRESSED, EXIT_ERROR, EXIT_INCOMPLETE = 0, 1, 2, 3

_SEVERITY = {
    None: 0,
    "passed": 1,
    "skipped": 2,
    "xfailed": 2,
    "xpassed": 3,
    "failed": 4,
    "error": 5,
}


class ResultCacheError(RuntimeError):
    """A question the cache cannot answer: no repository, an unknown commit, nothing selected."""


# ------------------------------------------------------------------ records
@dataclass
class Session:
    """One recorded pytest process."""

    path: Path
    commit: str | None
    clean: bool | None
    run_id: str
    ended: str
    env: dict
    env_digest: str
    python: str
    repo: str
    tests: dict[str, dict]
    collected: dict[str, list[str]]
    missing: set[str]
    deps: dict
    sys_path: list[str]
    opaque: tuple[int, str] | None

    @classmethod
    def load(cls, path: Path) -> "Session | None":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or raw.get("version") != RECORD_VERSION:
            return None
        opaque = raw.get("opaque")
        return cls(
            path=Path(path),
            commit=raw.get("commit") or None,
            clean=raw.get("clean"),
            run_id=str(raw.get("run_id") or path.stem),
            ended=str(raw.get("ended") or raw.get("started") or ""),
            env=raw.get("env") or {},
            env_digest=str(raw.get("env_digest") or ""),
            python=str(raw.get("python") or ""),
            repo=str(raw.get("repo") or ""),
            tests={k: v for k, v in (raw.get("tests") or {}).items() if isinstance(v, dict)},
            collected=raw.get("collected") or {},
            missing=set(raw.get("missing") or ()),
            deps=raw.get("deps") or {},
            sys_path=list(raw.get("sys_path") or []),
            opaque=(int(opaque["idx"]), str(opaque["reason"]))
            if isinstance(opaque, dict)
            else None,
        )

    def files(self) -> set[str]:
        """The test files this session collected or ran."""
        return set(self.collected) | {nodeid.split("::", 1)[0] for nodeid in self.tests}


def load_sessions(cache: Path) -> list[Session]:
    """Every readable session record under ``cache``, oldest first."""
    found = []
    for path in sorted((Path(cache) / SESSIONS_DIRNAME).glob("*.json")):
        session = Session.load(path)
        if session is not None:
            found.append(session)
    return sorted(found, key=lambda s: (s.ended, s.path.name))


# --------------------------------------------------------------------- git
class Git:
    """The questions about commits the cache asks, each answered once."""

    def __init__(self, repo: str | Path) -> None:
        self.repo = str(repo)
        self._trees: dict[str, dict[str, str]] = {}
        self._dirs: dict[str, dict[str, set[str]]] = {}
        self._blobs: dict[str, bytes | None] = {}
        self._diffs: dict[tuple[str, str], list[str]] = {}
        self._has: dict[str, bool] = {}

    def _run(self, *args: str, binary: bool = False):
        try:
            done = subprocess.run(
                ["git", "--no-optional-locks", "-C", self.repo, *args],
                capture_output=True,
                check=False,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        return done.stdout if binary else done.stdout.decode("utf-8", errors="replace")

    def resolve(self, rev: str) -> str | None:
        out = self._run("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
        return out.strip() if out and out.strip() else None

    def has(self, commit: str | None) -> bool:
        if not commit:
            return False
        if commit not in self._has:
            self._has[commit] = self._run("cat-file", "-e", f"{commit}^{{commit}}") is not None
        return self._has[commit]

    def is_ancestor(self, older: str, newer: str) -> bool:
        return self._run("merge-base", "--is-ancestor", older, newer) is not None

    def tree(self, commit: str) -> dict[str, str]:
        """Every tracked file at ``commit``: its path and blob id."""
        if commit not in self._trees:
            out = self._run("ls-tree", "-r", "-z", "--full-tree", commit, binary=True) or b""
            files = {}
            for entry in out.split(b"\0"):
                if b"\t" not in entry:
                    continue
                meta, _, path = entry.partition(b"\t")
                parts = meta.split()
                if len(parts) == 3 and parts[1] == b"blob":
                    files[path.decode("utf-8", errors="replace")] = parts[2].decode()
            self._trees[commit] = files
        return self._trees[commit]

    def _children(self, commit: str) -> dict[str, set[str]]:
        if commit not in self._dirs:
            children: dict[str, set[str]] = {"": set()}
            for path in self.tree(commit):
                parts = path.split("/")
                for depth in range(len(parts)):
                    parent = "/".join(parts[:depth])
                    children.setdefault(parent, set()).add(parts[depth])
            self._dirs[commit] = children
        return self._dirs[commit]

    def entries(self, commit: str, directory: str) -> frozenset[str]:
        """The names directly under ``directory`` at ``commit`` (empty when it is not one)."""
        return frozenset(self._children(commit).get(directory.strip("/"), ()))

    def exists(self, commit: str, path: str) -> bool:
        path = path.strip("/")
        return path in self.tree(commit) or path in self._children(commit)

    def blob(self, oid: str | None) -> bytes | None:
        if not oid:
            return None
        if oid not in self._blobs:
            self._blobs[oid] = self._run("cat-file", "blob", oid, binary=True)
        return self._blobs[oid]

    def changed(self, a: str, b: str) -> list[str]:
        """The paths whose content differs between ``a`` and ``b``."""
        key = (a, b) if a <= b else (b, a)
        if key not in self._diffs:
            out = self._run("diff", "--name-only", "--no-renames", "-z", a, b) or ""
            self._diffs[key] = [p for p in out.split("\0") if p]
        return self._diffs[key]


# ------------------------------------------------------------ python change
@dataclass(frozen=True)
class PyChange:
    """What changed in a Python file: outside function bodies, or which bodies."""

    skeleton: bool
    units: frozenset[str] = frozenset()


def _statements(node: ast.AST) -> list[ast.stmt]:
    out = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            out.append(child)
        elif isinstance(child, (ast.excepthandler, ast.match_case)):
            out += [c for c in ast.iter_child_nodes(child) if isinstance(c, ast.stmt)]
    return out


def _compiles(decorators: Iterable[ast.expr]) -> bool:
    return any(COMPILING_DECORATOR.search(ast.unparse(d)) for d in decorators)


def split_python(source: bytes) -> tuple[str, dict[str, str]] | None:
    """A module's skeleton and its function bodies, or None when it does not parse.

    The units are the functions at module level and in class bodies (nested
    classes included, under ``if``/``try``/``with`` blocks too), each by its
    qualified name, with the hash of its whole definition; the skeleton is
    the rest, with each unit's body elided. A function whose decorator
    compiles or traces it, or that sits in such a class, stays in the
    skeleton: its body is not run as the Python it reads as. Neither carries
    line numbers, so moved code and edited comments change neither.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    units: dict[str, list[str]] = {}

    def visit(body: list[ast.stmt], prefix: str, compiled: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not compiled and not _compiles(node.decorator_list):
                    units.setdefault(prefix + node.name, []).append(ast.dump(node))
                    node.body = [ast.Pass()]
            elif isinstance(node, ast.ClassDef):
                visit(
                    node.body, f"{prefix}{node.name}.", compiled or _compiles(node.decorator_list)
                )
            else:
                visit(_statements(node), prefix, compiled)

    visit(tree.body, "", False)
    digest = {q: hashlib.sha1("\n".join(d).encode()).hexdigest() for q, d in units.items()}
    return ast.dump(tree), digest


def python_change(old: bytes | None, new: bytes | None) -> PyChange:
    """How ``new`` differs from ``old``; a file missing on either side is a skeleton change."""
    if old is None or new is None:
        return PyChange(True)
    a, b = split_python(old), split_python(new)
    if a is None or b is None or a[0] != b[0]:
        return PyChange(True)
    units = {q for q in set(a[1]) | set(b[1]) if a[1].get(q) != b[1].get(q)}
    return PyChange(False, frozenset(units))


def unit_of(qualname: str) -> str:
    """The unit a code object's qualified name runs in: ``C.m.<locals>.<lambda>`` -> ``C.m``."""
    head = qualname.split(".<locals>.", 1)[0]
    parts = head.split(".")
    while parts and parts[-1].startswith("<"):
        parts.pop()
    return ".".join(parts)


def native_source(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in NATIVE_SOURCES)


# ---------------------------------------------------------------- validity
@dataclass
class Result:
    """A test's recorded result, and the run it came from."""

    nodeid: str
    outcome: str
    message: str
    duration: float
    commit: str | None
    session: Path


@dataclass
class Verdict:
    """Whether a session's records hold at a commit: every test with an index below ``holds_below`` does."""

    holds_below: int
    reasons: list[str] = field(default_factory=list)


class ResultCache:
    """The cache's sessions, read against one repository."""

    def __init__(
        self,
        cache: str | Path,
        repo: str | Path,
        *,
        env_digest: str = "",
        tree_dir: str | Path | None = None,
        tree_commit: str | None = None,
        sessions: list[Session] | None = None,
    ) -> None:
        self.cache = Path(cache)
        self.git = Git(repo)
        self.sessions = load_sessions(self.cache) if sessions is None else sessions
        #: Results hold only in this environment, when it is known.
        self.env_digest = env_digest
        #: The tree checked out at ``tree_commit``, whose shared libraries are compared.
        self.tree_dir = Path(tree_dir) if tree_dir else None
        self.tree_commit = tree_commit
        self._verdicts: dict[tuple[Path, str], Verdict] = {}
        self._changes: dict[tuple[str, str], PyChange] = {}
        self._native: dict[tuple[str, int, int], str | None] = {}
        self._by_test: dict[str, list[Session]] = {}
        for session in self.sessions:
            for nodeid in session.tests:
                self._by_test.setdefault(nodeid, []).append(session)

    # -------------------------------------------------------- per session
    def _change(self, old: str | None, new: str | None) -> PyChange:
        key = (old or "", new or "")
        if key not in self._changes:
            self._changes[key] = python_change(self.git.blob(old), self.git.blob(new))
        return self._changes[key]

    def _disk_blob(self, path: Path) -> str | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._native:
            self._native[key] = file_blob_id(str(path))
        return self._native[key]

    def verdict(self, session: Session, commit: str) -> Verdict:
        """Whether ``session``'s records hold at ``commit``, and from which test on they do not."""
        key = (session.path, commit)
        if key not in self._verdicts:
            self._verdicts[key] = self._judge(session, commit)
        return self._verdicts[key]

    def _judge(self, s: Session, x: str) -> Verdict:
        if not self.git.has(s.commit):
            return Verdict(-1, ["its commit is not in this repository"])
        if self.env_digest and s.env_digest != self.env_digest:
            return Verdict(-1, ["it ran in another environment"])
        if s.commit == x and s.clean:
            return Verdict(NEVER)
        found: list[tuple[int, str]] = []
        if s.opaque is not None:
            found.append((s.opaque[0], f"opaque: {s.opaque[1]}"))
        tree = self.git.tree(x)
        deps = s.deps
        loaded, executed, read = (
            deps.get("loaded", {}),
            deps.get("executed", {}),
            deps.get("read", {}),
        )
        for rel, old in (deps.get("blobs") or {}).items():
            new = tree.get(rel)
            if old == new:
                continue
            if rel in read:
                found.append((read[rel], f"{rel} changed (read as data)"))
            ran = executed.get(rel, {})
            first = min([loaded[rel]] if rel in loaded else [], default=NEVER)
            first = min([first, *ran.values()])
            if first >= NEVER:
                continue
            if old is None or new is None or not rel.endswith(".py"):
                found.append(
                    (
                        first,
                        f"{rel} {'added' if old is None else 'deleted' if new is None else 'changed'}",
                    )
                )
                continue
            change = self._change(old, new)
            if change.skeleton:
                found.append((first, f"{rel}: code outside function bodies changed"))
                continue
            hits = [
                (idx, unit_of(name))
                for name, idx in ran.items()
                if name.startswith("@") or unit_of(name) in change.units
            ]
            if hits:
                names = sorted({u for _, u in hits if u}) or sorted(change.units)
                found.append((min(i for i, _ in hits), f"{rel}: {', '.join(names[:3])} changed"))
        for directory, idx in (deps.get("listed") or {}).items():
            if self.git.entries(s.commit, directory) != self.git.entries(x, directory):
                found.append((idx, f"{directory or '.'}/ gained or lost an entry"))
        for path, idx in (deps.get("probed") or {}).items():
            if self.git.exists(s.commit, path) != self.git.exists(x, path):
                found.append(
                    (idx, f"{path} {'appeared' if self.git.exists(x, path) else 'disappeared'}")
                )
        for name, idx in (deps.get("imports") or {}).items():
            stem = name.replace(".", "/")
            for entry in s.sys_path:
                base = f"{entry}/{stem}" if entry else stem
                if any(
                    self.git.exists(s.commit, c) != self.git.exists(x, c)
                    for c in (f"{base}.py", f"{base}/__init__.py", base)
                ):
                    found.append((idx, f"`import {name}` now finds a different file ({base})"))
                    break
        native = deps.get("native") or {}
        if native:
            first = min(int(v.get("idx", -1)) for v in native.values())
            sources = [p for p in self.git.changed(s.commit, x) if native_source(p)]
            if sources:
                more = f" and {len(sources) - 3} more" if len(sources) > 3 else ""
                found.append((first, f"native sources changed: {', '.join(sources[:3])}{more}"))
            if self.tree_dir is not None and self.tree_commit == x:
                for rel, v in native.items():
                    if self._disk_blob(self.tree_dir / rel) != v.get("blob"):
                        found.append(
                            (int(v.get("idx", -1)), f"{rel} is not the library it ran with")
                        )
        if not found:
            return Verdict(NEVER)
        first = min(i for i, _ in found)
        return Verdict(first, [text for i, text in sorted(found) if i == first][:3])

    # ---------------------------------------------------------- per test
    def result(self, nodeid: str, commit: str) -> Result | None:
        """The newest recorded result of ``nodeid`` that holds at ``commit``.

        Sessions of one launch (one ``run_id``: the ranks of an MPI run, say)
        count together, the worst outcome among them.
        """
        holding = []
        for session in reversed(self._by_test.get(nodeid, [])):
            test = session.tests[nodeid]
            if int(test.get("index", -1)) < self.verdict(session, commit).holds_below:
                holding.append((session, test))
        if not holding:
            return None
        run = holding[0][0].run_id
        same = [(s, t) for s, t in holding if s.run_id == run]
        session, test = max(same, key=lambda st: _SEVERITY.get(st[1].get("outcome"), 0))
        return Result(
            nodeid=nodeid,
            outcome=str(test.get("outcome") or "error"),
            message=str(test.get("message") or ""),
            duration=float(test.get("duration") or 0.0),
            commit=session.commit,
            session=session.path,
        )

    def why_not(self, nodeid: str, commit: str) -> str:
        """Why the newest record of ``nodeid`` does not hold at ``commit``."""
        sessions = self._by_test.get(nodeid)
        if not sessions:
            return "never run"
        verdict = self.verdict(sessions[-1], commit)
        return "; ".join(verdict.reasons) or "not run since its inputs changed"

    def last_duration(self, nodeid: str) -> float:
        sessions = self._by_test.get(nodeid)
        return float(sessions[-1].tests[nodeid].get("duration") or 0.0) if sessions else 0.0

    # ---------------------------------------------------------- per file
    def known_tests(self, file: str, commit: str) -> tuple[list[str] | None, list[str]]:
        """The test ids ``file`` holds at ``commit``, or None, with why it must be collected again.

        From the newest session that collected it and whose collection holds
        there: what a collection read decides which tests exist.
        """
        reasons: list[str] = []
        for session in reversed(self.sessions):
            if file not in session.files():
                continue
            verdict = self.verdict(session, commit)
            if verdict.holds_below < 0:
                if not reasons:
                    reasons = verdict.reasons or ["its collection read something that changed"]
                continue
            names = session.collected.get(file)
            if names is None:
                return [file], []  # it failed to collect, and still would
            return [f"{file}::{n}" if n else file for n in names], []
        return None, reasons or ["never recorded"]

    def test_files(self, commit: str, paths: Iterable[str]) -> list[str]:
        """The test files at ``commit`` under ``paths``; every recorded one there without paths."""
        tree = self.git.tree(commit)
        paths = [p.strip().rstrip("/") for p in paths if p.strip()]
        if paths:
            return sorted(
                f
                for f in tree
                if TEST_FILE.search(f) and any(f == p or f.startswith(p + "/") for p in paths)
            )
        recorded = set().union(*(s.files() for s in self.sessions)) if self.sessions else set()
        return sorted(f for f in recorded if f in tree)


# ---------------------------------------------------------------- reports
@dataclass
class FilePlan:
    file: str
    whole: bool
    run: list[str]
    reused: list[Result]
    reasons: dict[str, str]
    seconds_run: float = 0.0
    seconds_reused: float = 0.0


@dataclass
class AffectedReport:
    base: str | None
    head: str
    paths: list[str]
    changed: list[str]
    files: list[FilePlan]

    def selection(self) -> list[str]:
        out = []
        for plan in self.files:
            out += [plan.file] if plan.whole else plan.run
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "head": self.head,
            "paths": self.paths,
            "changed": self.changed,
            "selection": self.selection(),
            "files": [
                {
                    "file": p.file,
                    "whole": p.whole,
                    "run": p.run,
                    "reasons": p.reasons,
                    "reused": [
                        {"nodeid": r.nodeid, "outcome": r.outcome, "commit": r.commit}
                        for r in p.reused
                    ],
                    "seconds_run": round(p.seconds_run, 1),
                    "seconds_reused": round(p.seconds_reused, 1),
                }
                for p in self.files
            ],
        }


def affected(
    cache: ResultCache,
    head: str,
    *,
    base: str | None = None,
    paths: Iterable[str] = (),
    full: bool = False,
) -> AffectedReport:
    """Which tests under ``paths`` must run at ``head``, and which recorded results hold there."""
    paths = list(paths)
    files = cache.test_files(head, paths)
    if not files:
        raise ResultCacheError(
            "no test file selected: name the tests with --paths"
            + (" (the cache is empty)" if not cache.sessions else "")
        )
    plans = []
    for file in files:
        tests, reasons = (None, ["--full"]) if full else cache.known_tests(file, head)
        if tests is None:
            seconds = sum(
                cache.last_duration(n) for n in cache._by_test if n.split("::", 1)[0] == file
            )
            plans.append(
                FilePlan(file, True, [], [], {file: "; ".join(reasons)}, seconds_run=seconds)
            )
            continue
        plan = FilePlan(file, False, [], [], {})
        for nodeid in tests:
            found = cache.result(nodeid, head)
            if found is None:
                plan.run.append(nodeid)
                plan.reasons[nodeid] = cache.why_not(nodeid, head)
                plan.seconds_run += cache.last_duration(nodeid)
            else:
                plan.reused.append(found)
                plan.seconds_reused += found.duration
        if plan.run and not plan.reused:
            plan.whole = True
            plan.reasons[file] = next(iter(plan.reasons.values()))
        plans.append(plan)
    changed = cache.git.changed(base, head) if base else []
    return AffectedReport(base, head, paths, changed, plans)


@dataclass
class Comparison:
    nodeid: str
    head: Result | None
    base: Result | None
    new: bool = False


@dataclass
class CompareReport:
    base: str
    head: str
    regressions: list[Comparison]
    pre_existing: list[Comparison]
    base_unknown: list[Comparison]
    fixed: list[Comparison]
    not_run: list[str]
    passed: int
    ran_at_head: int
    reused: int
    #: Whether the results were read in the head's environment (a session recorded there).
    env_checked: bool = True

    @property
    def verdict(self) -> str:
        if self.regressions:
            return "REGRESSED"
        if self.base_unknown or self.not_run or (self.reused and not self.env_checked):
            return "INCOMPLETE"
        return "CLEAN"

    @property
    def exit_code(self) -> int:
        return {"REGRESSED": EXIT_REGRESSED, "INCOMPLETE": EXIT_INCOMPLETE}.get(
            self.verdict, EXIT_CLEAN
        )

    def as_dict(self) -> dict[str, Any]:
        def one(c: Comparison) -> dict:
            return {
                "nodeid": c.nodeid,
                "new": c.new,
                "head": None if c.head is None else _result_dict(c.head),
                "base": None if c.base is None else _result_dict(c.base),
            }

        return {
            "verdict": self.verdict,
            "base": self.base,
            "head": self.head,
            "regressions": [one(c) for c in self.regressions],
            "pre_existing": [one(c) for c in self.pre_existing],
            "base_unknown": [one(c) for c in self.base_unknown],
            "fixed": [one(c) for c in self.fixed],
            "not_run_at_head": self.not_run,
            "passed": self.passed,
            "ran_at_head": self.ran_at_head,
            "reused": self.reused,
            "env_checked": self.env_checked,
        }


def _result_dict(r: Result) -> dict:
    return {"outcome": r.outcome, "message": r.message, "commit": r.commit}


def head_environment(sessions: Iterable[Session], head: str) -> str:
    """The environment of the newest session recorded at ``head``; empty when none ran there."""
    digests = [s.env_digest for s in sessions if s.commit == head and s.env_digest]
    return digests[-1] if digests else ""


def compare(
    cache: ResultCache, base: str, head: str, *, paths: Iterable[str] = ()
) -> CompareReport:
    """The results at ``head`` against those at ``base``, for the tests under ``paths``.

    A reused result counts once it is known to hold in the head's environment:
    ``cache`` was opened with that environment (:func:`head_environment`).
    """
    paths = list(paths)
    files = cache.test_files(head, paths)
    if not files:
        raise ResultCacheError("no test file selected: name the tests with --paths")
    regressions, pre, unknown, fixed, not_run = [], [], [], [], []
    passed = ran = reused = 0
    for file in files:
        tests, _ = cache.known_tests(file, head)
        if tests is None:
            not_run.append(file)
            continue
        at_base, _ = cache.known_tests(file, base) if cache.git.exists(base, file) else ([], [])
        for nodeid in tests:
            now = cache.result(nodeid, head)
            if now is None:
                not_run.append(nodeid)
                continue
            if now.commit == head:
                ran += 1
            else:
                reused += 1
            then = cache.result(nodeid, base)
            new = then is None and at_base is not None and nodeid not in at_base
            if now.outcome in FAILING:
                entry = Comparison(nodeid, now, then, new)
                if new or (then is not None and then.outcome not in FAILING):
                    regressions.append(entry)
                elif then is None:
                    unknown.append(entry)
                else:
                    pre.append(entry)
            else:
                passed += 1
                if then is not None and then.outcome in FAILING:
                    fixed.append(Comparison(nodeid, now, then))
    return CompareReport(
        base,
        head,
        regressions,
        pre,
        unknown,
        fixed,
        not_run,
        passed,
        ran,
        reused,
        env_checked=bool(cache.env_digest),
    )


@dataclass
class HeadResults:
    """What the recorded results say of the tests under some paths at one commit, on their own."""

    head: str
    passed: list[str]
    skipped: list[str]
    failing: list[Result]
    #: Test ids (a file whose collection is not known: the file) no result holds for.
    not_run: list[str]

    @property
    def verdict(self) -> str:
        """``FAILING``, ``INCOMPLETE`` (something is still to run, or nothing passed), or ``CLEAN``."""
        if self.failing:
            return "FAILING"
        if self.not_run or not self.passed:
            return "INCOMPLETE"
        return "CLEAN"

    def describe(self, limit: int = 5) -> str:
        """One line: the verdict and what makes it so."""
        counts = (
            f"{len(self.passed)} passed, {len(self.skipped)} skipped, {len(self.failing)} failing, "
            f"{len(self.not_run)} not run at {self.head[:10]}"
        )
        if self.failing:
            named = "; ".join(
                f"{r.nodeid}: {r.outcome}: {r.message[:120]}" for r in self.failing[:limit]
            )
            return f"FAILING ({counts}): {named}"
        if self.not_run:
            return f"INCOMPLETE ({counts}): {', '.join(self.not_run[:limit])}"
        if not self.passed:
            return f"INCOMPLETE ({counts}): no test passed"
        return f"CLEAN ({counts})"


def head_results(cache: ResultCache, head: str, paths: Iterable[str]) -> HeadResults:
    """Every test under ``paths`` at ``head``, by the newest recorded result that holds there.

    Unlike :func:`compare`, nothing is read at another commit: the question is
    whether these tests pass on this tree, as delivery and a certification
    closed without a turn ask it.
    """
    passed: list[str] = []
    skipped: list[str] = []
    failing: list[Result] = []
    not_run: list[str] = []
    for file in cache.test_files(head, list(paths)):
        tests, _ = cache.known_tests(file, head)
        if tests is None:
            not_run.append(file)
            continue
        for nodeid in tests:
            result = cache.result(nodeid, head)
            if result is None:
                not_run.append(nodeid)
            elif result.outcome in FAILING:
                failing.append(result)
            elif result.outcome == "passed":
                passed.append(nodeid)
            else:
                skipped.append(nodeid)
    return HeadResults(head, passed, skipped, failing, not_run)


def open_cache(cache_dir: str | Path, repo: str | Path, head: str) -> ResultCache:
    """The cache under ``cache_dir`` read against ``repo``, in ``head``'s environment when one ran there."""
    sessions = load_sessions(Path(cache_dir))
    return ResultCache(
        cache_dir, repo, env_digest=head_environment(sessions, head), sessions=sessions
    )


# -------------------------------------------------------------- workspace
def _yaml(path: Path) -> Any:
    import yaml

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None


def _task(workspace: Path) -> dict:
    for name in ("task.resolved.yaml", "task.yaml"):
        data = _yaml(workspace / name)
        if isinstance(data, dict):
            return data
    return {}


def default_base(workspace: Path, git: Git, head: str) -> str | None:
    """The commit a cross-model review last cleared, else the run's starting commit.

    The first is the newest ``cleared`` among the receipts in
    ``<workspace>/tasks/`` that is an ancestor of ``head``; the second, the
    parent of the oldest commit any receipt says its task integrated.
    """
    receipts = [_yaml(p) for p in sorted((workspace / "tasks").glob("*.receipt.yaml"))]
    receipts = [r for r in receipts if isinstance(r, dict)]
    cleared = []
    for receipt in receipts:
        commit = git.resolve(str(receipt.get("cleared") or "")) if receipt.get("cleared") else None
        if commit and git.is_ancestor(commit, head):
            cleared.append((str(receipt.get("written_at") or ""), commit))
    if cleared:
        return max(cleared)[1]
    integrated = []
    for receipt in receipts:
        for line in receipt.get("commits") or ():
            commit = git.resolve(str(line).split()[0]) if str(line).split() else None
            if commit and git.is_ancestor(commit, head):
                integrated.append(commit)
    oldest = [c for c in integrated if all(git.is_ancestor(c, o) for o in integrated)]
    return git.resolve(f"{oldest[0]}^") if oldest else None


# ------------------------------------------------------------------- CLI
def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 tests",
        description="Which tests a change affects, and what the unaffected ones already showed.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name, about in (
        ("affected", "the tests to run at --head: what no recorded result holds for"),
        ("compare", "the results at --head against those at --base: the regressions"),
    ):
        p = sub.add_parser(name, help=about, description=about)
        p.add_argument("--head", default="HEAD", help="The commit under review (default HEAD).")
        p.add_argument(
            "--base",
            default="",
            help="The commit it is compared with. Default, with --workspace: the commit a "
            "cross-model review last cleared (a receipt's `cleared`), else the run's start.",
        )
        p.add_argument("--paths", nargs="*", default=[], help="Test files and directories.")
        p.add_argument("--workspace", type=Path, default=None, help="The run's workspace.")
        p.add_argument(
            "--cache",
            type=Path,
            default=None,
            help=f"The cache. Default <workspace>/{CACHE_DIRNAME}, else ${CACHE_ENV}.",
        )
        p.add_argument(
            "--repo",
            type=Path,
            default=None,
            help="A checkout holding both commits; when it is at --head, its shared libraries "
            "are compared too. Default: the task's trtllm_repo_path, else here.",
        )
        p.add_argument("--python", default="", help="The tree's python, for the printed command.")
        p.add_argument(
            "--env",
            type=Path,
            default=None,
            help="`testcache_driver.py env` output: results hold only in that environment.",
        )
        p.add_argument("--out", type=Path, default=None, help="Where the selection files go.")
        p.add_argument("--json", action="store_true", help="Print the report as JSON.")
        if name == "affected":
            p.add_argument("--full", action="store_true", help="Run every selected test file.")
            p.add_argument("--verbose", action="store_true", help="List the reused results.")
    return parser.parse_args(argv)


def _python(args: argparse.Namespace, task: dict, cache: ResultCache) -> str:
    if args.python:
        return args.python
    env = task.get("environment") if isinstance(task.get("environment"), dict) else {}
    if env.get("python"):
        return str(env["python"])
    for session in reversed(cache.sessions):
        if session.python:
            return session.python
    return "python3"


def _command(python: str, cache: Path, tree: str, selection: Path | None) -> str:
    """The driver's command line; without a selection it records only the environment."""
    if selection is None:
        return f"{python} {DRIVER} run --cache {cache} --repo {tree}"
    return (
        f"{python} {DRIVER} run --cache {cache} --repo {tree} --selection {selection} "
        "-- -p no:cacheprovider -rA --timeout=180"
    )


def _write_selection(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def _short(commit: str | None) -> str:
    return (commit or "?")[:10]


def _render_affected(report: AffectedReport, verbose: bool) -> list[str]:
    whole = [p for p in report.files if p.whole]
    some = [p for p in report.files if not p.whole and p.run]
    reused = [r for p in report.files for r in p.reused]
    lines = [
        f"Tests at {_short(report.head)}"
        + (f" against {_short(report.base)}" if report.base else "")
        + f": {len(report.files)} test file(s)"
        + (f" under {', '.join(report.paths)}" if report.paths else ""),
    ]
    if report.base:
        native = [p for p in report.changed if native_source(p)]
        lines.append(
            f"  changed since the base: {len(report.changed)} file(s)"
            + (f", {len(native)} native" if native else "")
        )
    run_seconds = sum(p.seconds_run for p in report.files)
    lines.append(
        f"  run: {len(whole)} whole file(s), {sum(len(p.run) for p in some)} test(s) in "
        f"{len(some)} file(s) (last runs took {_minutes(run_seconds)})"
    )
    for plan in whole:
        lines.append(
            f"    {plan.file}: whole file: {plan.reasons.get(plan.file) or 'every test affected'}"
        )
    for plan in some:
        sample = next(iter(plan.reasons.values()), "")
        lines.append(
            f"    {plan.file}: {len(plan.run)} of {len(plan.run) + len(plan.reused)}: {sample}"
        )
    by_commit: dict[str, int] = {}
    for r in reused:
        by_commit[_short(r.commit)] = by_commit.get(_short(r.commit), 0) + 1
    failing = [r for r in reused if r.outcome in FAILING]
    lines.append(
        f"  reused: {len(reused)} result(s) ("
        + ", ".join(f"{n} from {c}" for c, n in sorted(by_commit.items()))
        + f"), {len(failing)} failing; they took {_minutes(sum(r.duration for r in reused))}"
    )
    for r in failing[:20]:
        lines.append(f"    {r.nodeid}: {r.outcome} at {_short(r.commit)}: {r.message}")
    if verbose:
        for r in reused:
            lines.append(f"    {r.nodeid}: {r.outcome} at {_short(r.commit)}")
    return lines


def _render_compare(report: CompareReport) -> list[str]:
    lines = [
        f"{report.verdict}: {_short(report.head)} against {_short(report.base)}: "
        f"{report.passed} passing ({report.ran_at_head} ran at the head, {report.reused} reused)"
    ]

    def block(title: str, items: list[Comparison]) -> None:
        if not items:
            return
        lines.append(f"  {title}: {len(items)}")
        for c in items[:30]:
            head = f"{c.head.outcome}: {c.head.message}" if c.head else "-"
            base = (
                "new at the head"
                if c.new
                else (f"{c.base.outcome} at {_short(c.base.commit)}" if c.base else "no result")
            )
            lines.append(f"    {c.nodeid}\n      head: {head}\n      base: {base}")
        if len(items) > 30:
            lines.append(f"    and {len(items) - 30} more")

    block("regressions", report.regressions)
    block("failing at the head with no result at the base", report.base_unknown)
    block("failing at both (not this change's)", report.pre_existing)
    block("fixed", report.fixed)
    if report.not_run:
        lines.append(f"  not run at the head: {len(report.not_run)}")
        lines += [f"    {n}" for n in report.not_run[:30]]
    if report.reused and not report.env_checked:
        lines.append(
            "  environment not checked: nothing was recorded at the head, so the reused results "
            "may come from another environment"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 tests {affected,compare}``."""
    args = _parse(list(sys.argv[1:] if argv is None else argv))
    task = _task(args.workspace) if args.workspace else {}
    cache_dir = args.cache or (args.workspace / CACHE_DIRNAME if args.workspace else None)
    if cache_dir is None and os.environ.get(CACHE_ENV):
        cache_dir = Path(os.environ[CACHE_ENV])
    if cache_dir is None:
        print("error: name the cache: --cache, --workspace or $" + CACHE_ENV, file=sys.stderr)
        return EXIT_ERROR
    repo = args.repo or Path(str(task.get("trtllm_repo_path") or "."))
    git = Git(repo)
    head = git.resolve(args.head)
    if head is None:
        print(f"error: {repo} has no commit {args.head!r}", file=sys.stderr)
        return EXIT_ERROR
    base = git.resolve(args.base) if args.base else None
    if args.base and base is None:
        print(f"error: {repo} has no commit {args.base!r}", file=sys.stderr)
        return EXIT_ERROR
    if base is None and args.workspace is not None:
        base = default_base(args.workspace, git, head)
    env_digest = ""
    if args.env is not None:
        try:
            env_digest = str(json.loads(args.env.read_text(encoding="utf-8"))["env_digest"])
        except (OSError, ValueError, KeyError, TypeError):
            print(f"error: {args.env} is not `testcache_driver.py env` output", file=sys.stderr)
            return EXIT_ERROR
    at = git.resolve("HEAD") if (Path(repo) / ".git").exists() else None
    sessions = load_sessions(cache_dir)
    env_digest = env_digest or head_environment(sessions, head)
    cache = ResultCache(
        cache_dir, repo, env_digest=env_digest, tree_dir=repo, tree_commit=at, sessions=sessions
    )
    out = args.out or cache_dir / "selections" / head[:12]
    python = _python(args, task, cache)
    tree = str(Path(repo).resolve()) if at == head else f"<a checkout at {head[:10]}>"
    try:
        if args.action == "affected":
            report = affected(cache, head, base=base, paths=args.paths, full=args.full)
            selection = report.selection()
            path = _write_selection(out / "head.txt", selection) if selection else None
            if args.json:
                data = report.as_dict()
                data["selection_file"] = str(path) if path else None
                print(json.dumps(data, indent=1))
                return EXIT_CLEAN
            lines = _render_affected(report, args.verbose)
            if path is not None:
                lines += [
                    f"Selection: {path} ({len(selection)} line(s)). Run it in a checkout at the head, "
                    "with the tree's python:",
                    f"  {_command(python, cache_dir, tree, path)}",
                ]
            else:
                lines += [
                    "Nothing to run: every selected test has a result that holds at the head. "
                    "Record the head's environment, which `compare` checks the reused results against:",
                    f"  {_command(python, cache_dir, tree, None)}",
                ]
            print("\n".join(lines))
            return EXIT_CLEAN
        if base is None:
            print("error: name the base with --base", file=sys.stderr)
            return EXIT_ERROR
        report = compare(cache, base, head, paths=args.paths)
        notes = []
        if report.reused and not report.env_checked:
            notes += [
                "Record the head's environment:",
                f"  {_command(python, cache_dir, tree, None)}",
            ]
        if report.not_run:
            path = _write_selection(out / "head.txt", report.not_run)
            notes += [
                f"Not run at the head: {path}",
                f"  {_command(python, cache_dir, tree, path)}",
            ]
        if report.base_unknown:
            path = _write_selection(out / "base.txt", [c.nodeid for c in report.base_unknown])
            notes += [
                f"Run at the base ({_short(base)}) in a checkout there: {path}",
                f"  {_command(python, cache_dir, f'<a checkout at {base[:10]}>', path)}",
            ]
        if args.json:
            print(json.dumps(report.as_dict(), indent=1))
        else:
            print("\n".join(_render_compare(report) + notes))
        return report.exit_code
    except ResultCacheError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


__all__ = [
    "CACHE_DIRNAME",
    "CACHE_ENV",
    "DRIVER",
    "AffectedReport",
    "CompareReport",
    "Git",
    "HeadResults",
    "PyChange",
    "Result",
    "Session",
    "ResultCache",
    "ResultCacheError",
    "affected",
    "compare",
    "default_base",
    "head_environment",
    "head_results",
    "load_sessions",
    "main",
    "open_cache",
    "python_change",
    "split_python",
    "unit_of",
]
