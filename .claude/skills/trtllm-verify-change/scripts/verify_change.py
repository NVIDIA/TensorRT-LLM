#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the tests a change declares, at the stage the change is in.

A manifest declares the tests a change must pass, grouped by the feature each
one proves, and the stage each belongs to::

    verify:
      tests:
        core:
          moe_routing:
            purpose: the router picks the experts the HF reference picks
            pytest: tests/unittest/_torch/modeling_v2/test_moe_routing.py
            paths: [tensorrt_llm/_torch/modeling_v2/**/moe*.py]
          e2e:
            purpose: GSM8K and a long-context subset on one engine
            pytest: tests/integration/defs/accuracy/.../test_accuracy.py::TestX::test_e2e[{member}]
            members: [tep4, dep4]
            stage: delivery

There are two stages:

- ``landing``: while the change is being made. ``run --stage landing`` runs
  the landing tests that the change touches. A test without ``paths`` is always
  touched. A pass is kept while the files the test covers are unchanged.
- ``delivery``: before the change is handed over. ``run --stage delivery`` runs
  every declared test of both stages. ``report --stage delivery`` is satisfied
  only by passes measured on the tree that HEAD commits, with nothing
  uncommitted.

Each test is a pytest node of the tree. It runs from its pytest root, the
nearest directory above it that holds ``pytest.ini``, the way CI runs it. Its
JUnit report decides the verdict: every selected test passed and at least one
was selected. A skipped test fails the case.

Results are appended to a ledger in the worktree's git directory, which is
never committed. Exit codes: 0 when the command succeeds, 1 when a test fails or
the stage is not satisfied, and 2 when the manifest, the tree or the arguments
are unusable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

STAGE_LANDING = "landing"
STAGE_DELIVERY = "delivery"
STAGES = (STAGE_LANDING, STAGE_DELIVERY)

VERIFY_FIELD = "verify"
TEST_KEYS = ("purpose", "pytest", "paths", "members", "devices", "stage", "timeout", "env")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
MEMBER_PLACEHOLDER = "{member}"
FEATURE_PLACEHOLDER = "{feature}"
DEFAULT_TIMEOUT_S = 3600
DEFAULT_BUDGET_MIN = 15.0
DEFAULT_BASE = "origin/main"
STATE_DIR = "verify-change"
MANIFEST_NAME = "manifest.yaml"
LEDGER_NAME = "ledger.jsonl"
PYTEST_ROOT_MARKER = "pytest.ini"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ManifestError(ValueError):
    """The manifest cannot be used; ``errors`` holds every problem found."""

    def __init__(self, path: Path, errors: Sequence[str]):
        super().__init__(f"{path}:\n  " + "\n  ".join(errors))
        self.errors = list(errors)


class GitError(RuntimeError):
    """A git command the script depends on failed."""


# ---------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class DeclaredTest:
    """One test of the manifest, as declared."""

    name: str
    feature: str
    purpose: str
    nodes: tuple[str, ...]
    stage: str = STAGE_LANDING
    paths: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    devices: int = 0
    timeout: int = DEFAULT_TIMEOUT_S
    env: tuple[tuple[str, str], ...] = ()

    def cases(self) -> list[Case]:
        """One case per member, or one case for a test without members."""
        if self.members:
            return [Case(self, member) for member in self.members]
        return [Case(self, None)]


@dataclass(frozen=True)
class Case:
    """A test at one member: what runs, and what the ledger records."""

    test: DeclaredTest
    member: str | None

    @property
    def key(self) -> str:
        """``<feature>:<test>`` or ``<feature>:<test>@<member>``."""
        base = f"{self.test.feature}:{self.test.name}"
        return f"{base}@{self.member}" if self.member else base

    def node_ids(self) -> tuple[str, ...]:
        """The test's node ids with ``{member}`` and ``{feature}`` filled in."""
        return tuple(
            node.replace(MEMBER_PLACEHOLDER, self.member or "").replace(
                FEATURE_PLACEHOLDER, self.test.feature
            )
            for node in self.test.nodes
        )


def _as_list(value: Any) -> list[Any] | None:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return None


def _relative_path_error(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "must be a nonempty string"
    if Path(value).is_absolute() or ".." in Path(value).parts:
        return "must be relative to the repository root, without '..'"
    return None


def parse_manifest(data: Any, errors: list[str]) -> list[DeclaredTest]:
    """The tests ``data`` declares under ``verify.tests``; every problem is appended to ``errors``.

    Args:
        data: The manifest as loaded from YAML. Other top-level keys are ignored,
            so the ``verify`` block can sit in a larger YAML file.
        errors: Receives one message per problem found.

    Returns:
        The tests whose entries have no problems, in declaration order.
    """
    block = data.get(VERIFY_FIELD) if isinstance(data, Mapping) else None
    by_feature = block.get("tests") if isinstance(block, Mapping) else None
    if not isinstance(by_feature, Mapping) or not by_feature:
        errors.append(f"'{VERIFY_FIELD}.tests' must map a feature to its tests")
        return []
    for key in block:
        if key != "tests":
            errors.append(f"'{VERIFY_FIELD}.{key}' is not a key of '{VERIFY_FIELD}' (['tests'])")
    tests: list[DeclaredTest] = []
    seen: dict[str, str] = {}
    for feature, entries in by_feature.items():
        where = f"{VERIFY_FIELD}.tests.{feature}"
        if not isinstance(feature, str) or not NAME_PATTERN.match(feature):
            errors.append(f"'{where}': a feature's name is letters, digits, '_', '.' and '-'")
            continue
        if not isinstance(entries, Mapping) or not entries:
            errors.append(f"'{where}' must map a test's name to its entry")
            continue
        for name, entry in entries.items():
            test = _parse_test(feature, name, entry, errors, seen)
            if test is not None:
                tests.append(test)
    return tests


def _parse_test(
    feature: str, name: Any, entry: Any, errors: list[str], seen: dict[str, str]
) -> DeclaredTest | None:
    where = f"{VERIFY_FIELD}.tests.{feature}.{name}"
    before = len(errors)
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        errors.append(f"'{where}': a test's name is letters, digits, '_', '.' and '-'")
    elif name in seen:
        errors.append(
            f"'{where}': `{name}` is also declared under `{seen[name]}`; names are unique"
        )
    else:
        seen[name] = feature
    if not isinstance(entry, Mapping):
        errors.append(f"'{where}' must be a mapping with keys {list(TEST_KEYS)}")
        return None
    for key in entry:
        if key not in TEST_KEYS:
            errors.append(f"'{where}.{key}' is not a key of a test ({list(TEST_KEYS)})")

    purpose = entry.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        errors.append(f"'{where}.purpose' must say what the test proves")

    nodes = _as_list(entry.get("pytest"))
    if not nodes:
        errors.append(f"'{where}.pytest' must be a pytest node id or a nonempty list of them")
        nodes = []
    for node in nodes:
        file = node.partition("::")[0] if isinstance(node, str) else None
        if _relative_path_error(file) or not str(file).endswith(".py"):
            errors.append(
                f"'{where}.pytest' must name repository-relative test files "
                f"(`tests/.../test_x.py[::TestY::test_z]`), got {node!r}"
            )

    paths = _as_list(entry.get("paths")) if "paths" in entry else []
    if paths is None:
        errors.append(f"'{where}.paths' must be a path or glob, or a list of them")
        paths = []
    for pattern in paths:
        problem = _relative_path_error(pattern)
        if problem:
            errors.append(f"'{where}.paths': {pattern!r} {problem}")

    members = _as_list(entry.get("members")) if "members" in entry else []
    if members is None or ("members" in entry and not members):
        errors.append(f"'{where}.members' must be a member or a nonempty list of them")
        members = []
    if members == ["all"]:
        errors.append(f"'{where}.members': 'all' is not supported here; list the members")
    for member in members:
        if not isinstance(member, str) or not NAME_PATTERN.match(member):
            errors.append(f"'{where}.members': {member!r} is not a member name")
    if not members and any(MEMBER_PLACEHOLDER in str(node) for node in nodes):
        errors.append(f"'{where}.pytest' uses {MEMBER_PLACEHOLDER}, which needs 'members'")

    devices = entry.get("devices")
    if devices is not None:
        if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
            errors.append(f"'{where}.devices' must be an integer >= 1")
        if members:
            errors.append(f"'{where}' sets both 'members' and 'devices'; set one")

    stage = entry.get("stage", STAGE_LANDING)
    if stage not in STAGES:
        errors.append(f"'{where}.stage' must be one of {list(STAGES)}, got {stage!r}")

    timeout = entry.get("timeout", DEFAULT_TIMEOUT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        errors.append(f"'{where}.timeout' must be a positive number of seconds")

    env = entry.get("env") or {}
    if not isinstance(env, Mapping):
        errors.append(f"'{where}.env' must map a variable to its value")
        env = {}
    for var, value in env.items():
        if not isinstance(var, str) or not _ENV_NAME.match(var):
            errors.append(f"'{where}.env' names {var!r}, which is not a variable name")
        if not isinstance(value, (str, int, float, bool)):
            errors.append(f"'{where}.env.{var}' must be a scalar")

    if len(errors) > before:
        return None
    return DeclaredTest(
        name=name,
        feature=feature,
        purpose=" ".join(purpose.split()),
        nodes=tuple(nodes),
        stage=stage,
        paths=tuple(paths),
        members=tuple(members),
        devices=devices or 0,
        timeout=timeout,
        env=tuple((str(k), str(v)) for k, v in env.items()),
    )


def load_manifest(path: Path) -> list[DeclaredTest]:
    """The tests the manifest at ``path`` declares.

    Raises:
        ManifestError: The file is missing, is not YAML, or declares a test badly.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(path, ["no manifest here; write one or pass --manifest"]) from None
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(path, [f"cannot be read as YAML ({exc})"]) from None
    errors: list[str] = []
    tests = parse_manifest(data, errors)
    if errors:
        raise ManifestError(path, errors)
    return tests


# ------------------------------------------------------------------------------- paths


def glob_regex(pattern: str) -> re.Pattern[str]:
    """A regex for a repository-relative path pattern.

    ``**`` matches any number of directories, ``*`` and ``?`` stay within one.
    A pattern without wildcards matches that file and everything under it.
    """
    pattern = pattern.strip().removeprefix("./")
    if "*" not in pattern and "?" not in pattern:
        return re.compile(re.escape(pattern.rstrip("/")) + r"(?:/.*)?")
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append(r"(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(r".*")
            i += 2
        elif pattern[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def matches(path: str, patterns: Sequence[str]) -> bool:
    """Whether repository-relative ``path`` matches any of ``patterns``."""
    return any(glob_regex(pattern).fullmatch(path) for pattern in patterns)


def node_file(node: str) -> str:
    """The test file of a pytest node id."""
    return node.partition("::")[0]


# --------------------------------------------------------------------------------- git


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise GitError(f"git {' '.join(args)}: {detail}")
    return result.stdout


def repo_root(start: Path) -> Path:
    """The top of the worktree that holds ``start``."""
    return Path(_git(start, "rev-parse", "--show-toplevel").strip())


def state_dir(repo: Path) -> Path:
    """Where the default manifest, the ledger and the logs live: in the worktree's git dir."""
    return Path(_git(repo, "rev-parse", "--absolute-git-dir").strip()) / STATE_DIR


@dataclass(frozen=True)
class TreeState:
    """The worktree as a test sees it: HEAD, and every file with its content id."""

    head: str
    files: Mapping[str, str]
    #: Paths whose content differs from HEAD, untracked ones included.
    dirty: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.dirty

    @cached_property
    def state_id(self) -> str:
        """A digest of every file's content; equal for equal trees, committed or not."""
        digest = hashlib.sha256()
        for path in sorted(self.files):
            digest.update(f"{path}\0{self.files[path]}\n".encode())
        return digest.hexdigest()[:20]


def read_tree_state(repo: Path) -> TreeState:
    """The worktree's state: HEAD's files, with uncommitted and untracked changes applied.

    Content ids are git blob ids (``git hash-object``, which writes nothing), so
    a file has the same id before and after it is committed.
    """
    head = _git(repo, "rev-parse", "HEAD").strip()
    files: dict[str, str] = {}
    for record in _git(repo, "ls-tree", "-r", "-z", "--full-tree", "HEAD").split("\0"):
        if record:
            meta, _, path = record.partition("\t")
            files[path] = meta.split()[2]
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames")
    dirty = sorted({entry[3:] for entry in status.split("\0") if len(entry) > 3})
    present = [path for path in dirty if (repo / path).is_file()]
    hashed = (
        _git(repo, "hash-object", "--stdin-paths", stdin="\n".join(present) + "\n").split()
        if present
        else []
    )
    for path in dirty:
        files.pop(path, None)
    files.update(zip(present, hashed))
    for path in dirty:
        if path not in files and (repo / path).is_dir():
            # A submodule or nested repository: its checked-out commit stands for its content.
            try:
                files[path] = "commit:" + _git(repo / path, "rev-parse", "HEAD").strip()
            except GitError:
                files[path] = "directory"
    return TreeState(head=head, files=files, dirty=tuple(dirty))


def changed_files(repo: Path, state: TreeState, base_ref: str) -> tuple[str, list[str]]:
    """The merge base of HEAD and ``base_ref``, and every path changed since it, uncommitted too."""
    base = _git(repo, "merge-base", "HEAD", base_ref).strip()
    committed = _git(repo, "diff", "--name-only", "-z", "--no-renames", base, "HEAD").split("\0")
    return base, sorted({path for path in committed if path} | set(state.dirty))


def inputs_digest(case: Case, state: TreeState) -> str:
    """What a pass of ``case`` depends on: its nodes, its env, and the files it covers.

    A test with ``paths`` covers its test files and the files its paths match;
    a test without ``paths`` covers the whole tree.
    """
    digest = hashlib.sha256()
    test = case.test
    head = {"nodes": case.node_ids(), "env": test.env, "member": case.member}
    digest.update(json.dumps(head, sort_keys=True).encode())
    if not test.paths:
        digest.update(state.state_id.encode())
    else:
        own = {node_file(node) for node in case.node_ids()}
        regexes = [glob_regex(pattern) for pattern in test.paths]
        for path in sorted(state.files):
            if path in own or any(regex.fullmatch(path) for regex in regexes):
                digest.update(f"{path}\0{state.files[path]}\n".encode())
    return digest.hexdigest()[:20]


# -------------------------------------------------------------------------- definition


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None


def _imported(tree: ast.Module, path: Path, repo: Path, name: str) -> tuple[Path, str] | None:
    """The file module ``path`` imports ``name`` from, and its name there; else ``None``."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            base = repo
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
            target = base.joinpath(*node.module.split("."))
            for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                if candidate.is_file():
                    return candidate, alias.name
    return None


def _binds(tree: ast.Module, name: str) -> bool:
    """Whether ``tree`` binds ``name`` at module level by an import or an assignment."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.partition(".")[0]) == name for alias in node.names):
                return True
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return True
    return False


def _has_method(path: Path, repo: Path, cls: str, method: str, depth: int = 4) -> bool | None:
    """Whether class ``cls`` of module ``path`` has ``method``, its own or a base's.

    ``None`` when a base cannot be followed through the tree's sources, so the
    method may come from it.
    """
    tree = _parse(path)
    if tree is None or depth < 0:
        return None
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None)
    if node is None:
        source = _imported(tree, path, repo, cls)
        return _has_method(source[0], repo, source[1], method, depth - 1) if source else None
    if any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method
        for n in node.body
    ):
        return True
    unknown = False
    for base in node.bases:
        if not isinstance(base, ast.Name):
            unknown = True
            continue
        found = _has_method(path, repo, base.id, method, depth - 1)
        if found:
            return True
        unknown = unknown or found is None
    return None if unknown else False


def defined(repo: Path, node: str) -> tuple[bool | None, str]:
    """Whether the tree defines pytest node ``node``, read from its sources without importing.

    The file must hold the node's class or function, and the class its method
    (its own, or one a base in the tree gives it). A parametrization such as
    ``[tep4]`` is not checked: an id that matches nothing fails when it runs.

    Returns:
        ``(True, node)``, ``(False, why)``, or ``(None, why)`` when the sources
        cannot settle it.
    """
    file, _, rest = node.partition("::")
    path = repo / file
    if not path.is_file():
        return False, f"{file} is not in the tree"
    names = [part.partition("[")[0] for part in rest.split("::") if part]
    if not names:
        return True, node
    tree = _parse(path)
    if tree is None:
        return None, f"{file} could not be parsed as Python"
    first = names[0]
    local = any(
        isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == first
        for n in tree.body
    )
    if not local and not _binds(tree, first):
        return False, f"{file} defines no `{first}`"
    if len(names) > 1:
        found = _has_method(path, repo, first, names[1])
        if found is False:
            return False, f"{file} defines no `{first}::{names[1]}`"
        if found is None:
            return None, f"{file}: cannot tell whether `{first}` has `{names[1]}`"
    return True, node


# ---------------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Selection:
    """Whether a case runs at the requested stage, and why."""

    case: Case
    selected: bool
    why: str


def select(
    tests: Sequence[DeclaredTest],
    stage: str,
    changed: Sequence[str],
    only: Sequence[str] = (),
    every: bool = False,
) -> list[Selection]:
    """Every case of ``tests``, marked with whether it runs at ``stage``.

    Args:
        tests: The declared tests.
        stage: ``landing`` or ``delivery``.
        changed: Paths changed since the base; used only at ``landing``.
        only: Test names or case keys to restrict to; empty for no restriction.
        every: At ``landing``, run every landing test, touched or not.
    """
    out = []
    for test in tests:
        for case in test.cases():
            selected, why = _why(case, stage, changed, only, every)
            out.append(Selection(case, selected, why))
    return out


def _why(
    case: Case, stage: str, changed: Sequence[str], only: Sequence[str], every: bool
) -> tuple[bool, str]:
    test = case.test
    if only and test.name not in only and case.key not in only:
        return False, "not named by --only"
    if stage == STAGE_DELIVERY:
        return True, "delivery runs every test"
    if test.stage != STAGE_LANDING:
        return False, "a delivery test"
    if only:
        return True, "named by --only"
    if every:
        return True, "--all"
    if not test.paths:
        return True, "no paths: always runs"
    own = {node_file(node) for node in case.node_ids()}
    hit = next((path for path in changed if path in own or matches(path, test.paths)), None)
    if hit:
        return True, f"changed: {hit}"
    return False, "the change touches none of its paths"


# ------------------------------------------------------------------------------ ledger


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Every record of the ledger, oldest first; unreadable lines are skipped."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def append_ledger(path: Path, record: Mapping[str, Any]) -> None:
    """Append one record to the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def standing_record(
    records: Sequence[Mapping[str, Any]],
    case: Case,
    inputs: str,
    stage: str,
    state: TreeState,
) -> Mapping[str, Any] | None:
    """The latest record that still speaks for ``case`` at ``stage``; ``None`` when none does.

    At ``landing`` that is the latest record with the same inputs. At
    ``delivery`` it must also have measured this exact tree.
    """
    for record in reversed(records):
        if record.get("key") != case.key or record.get("inputs") != inputs:
            continue
        if stage == STAGE_DELIVERY and record.get("state") != state.state_id:
            continue
        return record
    return None


def latest_record(records: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any] | None:
    """The latest record of case ``key``, whatever it measured."""
    return next((record for record in reversed(records) if record.get("key") == key), None)


# ----------------------------------------------------------------------------- running


def pytest_root(repo: Path, test_file: str) -> Path:
    """The nearest directory above ``test_file`` that holds ``pytest.ini``; else ``repo``."""
    here = (repo / test_file).parent
    while here != repo and repo in here.parents:
        if (here / PYTEST_ROOT_MARKER).is_file():
            return here
        here = here.parent
    return repo


def judge_report(report: Path) -> tuple[bool, str]:
    """Whether every test the JUnit ``report`` holds passed, at least one, and what it says."""
    try:
        cases = list(ET.parse(report).getroot().iter("testcase"))
    except (OSError, ET.ParseError) as exc:
        return False, f"no test report ({exc})"
    passed, problems = 0, []
    for case in cases:
        name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
        outcome = next(
            (child for child in case if child.tag in ("failure", "error", "skipped")), None
        )
        if outcome is None:
            passed += 1
            continue
        message = (outcome.get("message") or outcome.text or "").strip().splitlines()
        problems.append(f"{name} {outcome.tag}: {message[0][:200] if message else ''}")
    if problems:
        return False, "; ".join(problems)
    if not passed:
        return False, "the node ids selected no test"
    return True, f"{passed} passed"


def _kill(proc: subprocess.Popen[bytes]) -> None:
    for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _run(
    argv: Sequence[str], cwd: Path, env: Mapping[str, str], log_prefix: Path, timeout: float
) -> int | None:
    """Run ``argv`` in its own process group; its exit code, or ``None`` when it timed out.

    Its output goes to ``<log_prefix>.stdout.log`` and ``<log_prefix>.stderr.log``.
    """
    with (
        open(f"{log_prefix}.stdout.log", "w") as out,
        open(f"{log_prefix}.stderr.log", "w") as err,
    ):
        proc = subprocess.Popen(
            list(argv), cwd=cwd, env=dict(env), stdout=out, stderr=err, start_new_session=True
        )
        try:
            return proc.wait(timeout=max(timeout, 1.0))
        except subprocess.TimeoutExpired:
            _kill(proc)
            return None
        except KeyboardInterrupt:
            _kill(proc)
            raise


def _last_line(path: Path) -> str:
    try:
        lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
    except OSError:
        return ""
    return next((line[:200] for line in reversed(lines) if line), "")


def run_case(case: Case, repo: Path, run_dir: Path, python: str) -> tuple[bool, str]:
    """Run ``case``'s nodes from their pytest roots and judge their JUnit reports.

    Nodes that share a pytest root run in one pytest process. Logs and reports
    are written to ``run_dir``.
    """
    groups: dict[Path, list[str]] = {}
    for node in case.node_ids():
        file, sep, rest = node.partition("::")
        root = pytest_root(repo, file)
        groups.setdefault(root, []).append((repo / file).relative_to(root).as_posix() + sep + rest)
    env = {**os.environ, **dict(case.test.env)}
    slug = re.sub(r"[^A-Za-z0-9_.@-]+", "_", case.key.replace(":", "."))
    deadline = time.monotonic() + case.test.timeout
    details = []
    for index, (root, nodes) in enumerate(groups.items()):
        prefix = run_dir / (slug + (f".{index}" if len(groups) > 1 else ""))
        report = Path(f"{prefix}.junit.xml")
        argv = [python, "-m", "pytest", *nodes, "-v", "-rfEs", "-p", "no:cacheprovider"]
        argv.append(f"--junitxml={report}")
        code = _run(argv, root, env, prefix, deadline - time.monotonic())
        if code is None:
            return False, f"timed out after {case.test.timeout}s"
        if not report.is_file():
            last = _last_line(Path(f"{prefix}.stderr.log")) or _last_line(
                Path(f"{prefix}.stdout.log")
            )
            return False, f"pytest exited {code} without a test report: {last}"
        passed, detail = judge_report(report)
        if passed and code != 0:
            passed, detail = False, f"pytest exited {code} ({detail})"
        if not passed:
            return False, detail
        details.append(detail)
    return True, "; ".join(details)


# ------------------------------------------------------------------------------ output


def _duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    lines = [header, *rows]
    return "\n".join(
        "  ".join(str(cell).ljust(width) for cell, width in zip(line, widths)).rstrip()
        for line in lines
    )


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def cells(row: Sequence[str]) -> str:
        return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"

    return "\n".join([cells(header), cells(["---"] * len(header)), *(cells(r) for r in rows)])


def _member(case: Case) -> str:
    if case.member:
        return case.member
    return f"{case.test.devices} dev" if case.test.devices else "-"


def _verdict(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return "not run"
    return "passed" if record.get("passed") else "FAILED"


# ---------------------------------------------------------------------------- commands


@dataclass
class Context:
    """What every command reads: the tree, the manifest and the ledger."""

    repo: Path
    manifest: Path
    tests: list[DeclaredTest]
    state: TreeState
    ledger: Path
    records: list[dict[str, Any]]
    base: str | None
    changed: list[str]


def _context(args: argparse.Namespace) -> Context:
    repo = repo_root(Path(args.repo).resolve())
    home = state_dir(repo)
    manifest = Path(args.manifest).resolve() if args.manifest else home / MANIFEST_NAME
    tests = load_manifest(manifest)
    known = {test.name for test in tests} | {c.key for t in tests for c in t.cases()}
    unknown = [name for name in args.only if name not in known]
    if unknown:
        raise ManifestError(manifest, [f"--only names no declared test or case: {unknown}"])
    state = read_tree_state(repo)
    base, changed = None, []
    if args.stage == STAGE_LANDING and not args.only and not args.all:
        base, changed = changed_files(repo, state, args.base)
    ledger = home / LEDGER_NAME
    return Context(repo, manifest, tests, state, ledger, read_ledger(ledger), base, changed)


def _head_line(ctx: Context, stage: str) -> str:
    tree = "clean" if ctx.state.clean else f"{len(ctx.state.dirty)} uncommitted path(s)"
    line = f"stage {stage} | HEAD {ctx.state.head[:10]} ({tree})"
    if ctx.base:
        line += f" | {len(ctx.changed)} path(s) changed since {ctx.base[:10]}"
    return line + f"\nmanifest {ctx.manifest}"


def _definition_problems(ctx: Context, cases: Sequence[Case]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for node in sorted({node for case in cases for node in case.node_ids()}):
        ok, why = defined(ctx.repo, node)
        if ok is False:
            errors.append(why)
        elif ok is None:
            warnings.append(why)
    return errors, warnings


def cmd_plan(args: argparse.Namespace) -> int:
    """Print which cases the stage runs and why, check that their nodes exist, and estimate time."""
    ctx = _context(args)
    selections = select(ctx.tests, args.stage, ctx.changed, args.only, args.all)
    rows, estimate, unknown_time = [], 0.0, 0
    for sel in selections:
        case = sel.case
        inputs = inputs_digest(case, ctx.state)
        standing = standing_record(ctx.records, case, inputs, args.stage, ctx.state)
        latest = latest_record(ctx.records, case.key)
        if standing is not None:
            last = f"{_verdict(standing)}, current"
        elif latest is not None:
            last = f"{_verdict(latest)}, stale"
        else:
            last = "-"
        if sel.selected and not (standing and standing.get("passed")):
            if latest is not None and isinstance(latest.get("duration_s"), (int, float)):
                estimate += latest["duration_s"]
            else:
                unknown_time += 1
        rows.append(
            [
                case.test.feature,
                case.test.name,
                _member(case),
                case.test.stage,
                "yes" if sel.selected else "no",
                sel.why,
                last,
            ]
        )
    print(_head_line(ctx, args.stage))
    print()
    header = ["feature", "test", "member", "stage", "run", "why", "last result"]
    print(_table(header, rows))
    print()
    errors, warnings = _definition_problems(ctx, [c for t in ctx.tests for c in t.cases()])
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"undefined: {error}")
    if not errors:
        print("definitions: every node's file, class and function exist")
    to_run = sum(sel.selected for sel in selections)
    timing = f"time: about {_duration(estimate)} for the cases that would run"
    if unknown_time:
        timing += f"; {unknown_time} of {to_run} selected case(s) have no recorded duration"
    print(timing)
    if args.stage == STAGE_LANDING and estimate > args.budget_min * 60:
        print(f"warning: over the {args.budget_min:g}-minute landing budget")
    return EXIT_USAGE if errors else EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """Run the stage's cases that have no standing pass, record each result, and summarize."""
    ctx = _context(args)
    selections = select(ctx.tests, args.stage, ctx.changed, args.only, args.all)
    cases = [sel.case for sel in selections if sel.selected]
    print(_head_line(ctx, args.stage))
    if not cases:
        print("nothing to run at this stage")
        return EXIT_OK
    errors, _ = _definition_problems(ctx, cases)
    if errors:
        for error in errors:
            print(f"undefined: {error}")
        return EXIT_USAGE
    if args.stage == STAGE_DELIVERY and not ctx.state.clean:
        print("warning: uncommitted changes; delivery is satisfied only on a committed tree")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.log_dir).resolve() if args.log_dir else None
    run_dir = run_dir or state_dir(ctx.repo) / "runs" / f"{stamp}-{args.stage}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs {run_dir}\n")
    failed = reused = ran = 0
    for case in cases:
        inputs = inputs_digest(case, ctx.state)
        standing = standing_record(ctx.records, case, inputs, args.stage, ctx.state)
        if standing and standing.get("passed") and not args.rerun:
            reused += 1
            print(f"reused  {case.key}: passed at {str(standing.get('head', ''))[:10]}")
            continue
        start = time.monotonic()
        passed, detail = run_case(case, ctx.repo, run_dir, args.python)
        duration = time.monotonic() - start
        after = read_tree_state(ctx.repo)
        if after.state_id != ctx.state.state_id:
            # The tree changed under the test: the result belongs to no tree.
            detail += " (the tree changed during the run; not reusable)"
            inputs = "changed-during-run"
        ran += 1
        failed += not passed
        append_ledger(
            ctx.ledger,
            {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": socket.gethostname(),
                "stage": args.stage,
                "key": case.key,
                "feature": case.test.feature,
                "test": case.test.name,
                "member": case.member,
                "nodes": list(case.node_ids()),
                "head": ctx.state.head,
                "state": ctx.state.state_id,
                "clean": ctx.state.clean,
                "inputs": inputs,
                "passed": passed,
                "detail": detail,
                "duration_s": round(duration, 1),
                "logs": str(run_dir),
            },
        )
        print(f"{'passed' if passed else 'FAILED'}  {case.key} ({_duration(duration)}): {detail}")
        if not passed and args.fail_fast:
            break
    print(f"\nran {ran}, failed {failed}, reused {reused}, of {len(cases)} case(s)")
    return EXIT_FAILED if failed else EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    """Print each case's standing result at the stage; exit 0 only when the stage is satisfied."""
    ctx = _context(args)
    selections = select(ctx.tests, args.stage, ctx.changed, args.only, args.all)
    rows, passed = [], 0
    for sel in selections:
        if not sel.selected:
            continue
        case = sel.case
        inputs = inputs_digest(case, ctx.state)
        record = standing_record(ctx.records, case, inputs, args.stage, ctx.state)
        if record is None and latest_record(ctx.records, case.key) is not None:
            verdict = "stale"
        else:
            verdict = _verdict(record)
        passed += verdict == "passed"
        duration = _duration(record.get("duration_s")) if record else "-"
        rows.append(
            [
                case.test.feature,
                case.test.name,
                _member(case),
                case.test.stage,
                verdict,
                duration,
                case.test.purpose,
            ]
        )
    satisfied = passed == len(rows) and (args.stage != STAGE_DELIVERY or ctx.state.clean)
    tree = "clean tree" if ctx.state.clean else "uncommitted changes"
    if args.stage == STAGE_DELIVERY and not ctx.state.clean:
        tree += ", which delivery does not accept"
    summary = (
        f"{args.stage} at `{ctx.state.head[:10]}` ({tree}): "
        f"{passed} of {len(rows)} case(s) passed; "
        + ("satisfied" if satisfied else "NOT satisfied")
    )
    header = ["Feature", "Test", "Member", "Stage", "Result", "Time", "Purpose"]
    if args.format == "md":
        print(f"**Declared tests** — {summary}\n")
        print(_markdown_table(header, rows))
    else:
        print(summary)
        print()
        print(_table(header, rows))
    return EXIT_OK if satisfied else EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface: ``plan``, ``run`` and ``report``."""
    parser = argparse.ArgumentParser(prog="verify_change.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Any, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        sub.add_argument("--stage", choices=STAGES, default=STAGE_LANDING)
        sub.add_argument("--repo", default=".", help="a path inside the worktree (default: .)")
        sub.add_argument(
            "--manifest",
            help=f"the manifest (default: <git dir>/{STATE_DIR}/{MANIFEST_NAME})",
        )
        sub.add_argument(
            "--base",
            default=DEFAULT_BASE,
            help=f"what the change is measured against at landing (default: {DEFAULT_BASE})",
        )
        sub.add_argument(
            "--only",
            action="append",
            default=[],
            metavar="NAME",
            help="only this test or case (repeatable)",
        )
        sub.add_argument(
            "--all", action="store_true", help="at landing, every landing test, touched or not"
        )
        return sub

    plan = command("plan", cmd_plan, "show what the stage runs and why; run nothing")
    plan.add_argument("--budget-min", type=float, default=DEFAULT_BUDGET_MIN)
    run = command("run", cmd_run, "run the stage's cases that have no standing pass")
    run.add_argument("--rerun", action="store_true", help="run even when a pass stands")
    run.add_argument("--fail-fast", action="store_true", help="stop at the first failure")
    run.add_argument("--python", default=sys.executable, help="the interpreter that runs pytest")
    run.add_argument("--log-dir", help="where logs and JUnit reports go")
    report = command("report", cmd_report, "show each case's standing result at the stage")
    report.add_argument("--format", choices=("text", "md"), default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the command; returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ManifestError, GitError) as exc:
        print(f"verify_change: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
