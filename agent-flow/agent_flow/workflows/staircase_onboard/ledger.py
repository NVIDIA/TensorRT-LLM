# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The results ledger: every gate verdict, tied to the code it was measured on.

A gate verdict is a claim about a specific piece of code under a specific
configuration. Once the code moves, the claim describes code that no longer
exists. A record that keeps only "accuracy PASS" therefore goes stale without
anyone noticing, and a report written from such records can say "16/16 PASS"
about a tree no gate ever ran on.

This module stores each verdict together with what it depends on:

- the gate, unit and member, the measured value and the threshold it was
  judged against, and when it was measured;
- the **tree**: the commit, its tree object, and any uncommitted changes
  among the files the verdict depends on;
- the **native build**: a content fingerprint of the compiled libraries the
  engine loaded;
- the **dependency paths**: the target's own files, the family-level shared
  modules beside it, and every catalog entry the target imports, each with
  its git blob hash at measurement time;
- the **configuration**: the gate command, the explicit environment switches
  (both fingerprinted), the switch defaults the code declared at the time,
  and a fingerprint of the part of ``task.yaml`` that applies to the unit.

From that, :mod:`.freshness` decides later, against any tree, whether a
verdict still holds, and which gates delivery has to run again.

The ledger is one JSON Lines file in the workspace, and it is append-only.
Nothing in it is ever rewritten. A newer verdict for the same
``(gate, unit, member)`` key supersedes an older one by position in the file.

Gates do not import this module. ``gates.GateLog`` carries a sink, and the
workflow points that sink at :func:`record_gate_result`. That is the single
place verdicts enter the ledger. A harness outside the workflow uses the same
function or the ``record`` subcommand of this module's CLI.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .delivery import CATALOG_DIR, MODELING_V2_PACKAGE, imported_catalog_entries
from .task_schema import assembly_units

LEDGER_FILENAME = "ledger.jsonl"
SCHEMA_VERSION = 1

KIND_VERDICT = "verdict"
KIND_REGRESSION = "regression"

OUTCOME_PASS = "PASS"
OUTCOME_FAIL = "FAIL"
#: Passed its threshold but flagged for a human. A finding does not block
#: delivery, but it must never disappear from a summary.
OUTCOME_FINDING = "FINDING"
OUTCOMES: tuple[str, ...] = (OUTCOME_PASS, OUTCOME_FAIL, OUTCOME_FINDING)

#: Environment variables whose name starts with one of these are switches:
#: their explicit values are recorded, and their defaults in code are tracked.
DEFAULT_SWITCH_PREFIXES: tuple[str, ...] = ("TRTLLM_",)

#: Where the compiled libraries the engine loads live, relative to the repo.
DEFAULT_NATIVE_GLOBS: tuple[str, ...] = ("tensorrt_llm/libs/*.so", "tensorrt_llm/*.so")

#: ``task.yaml`` blocks that describe the substrate or the run's own
#: bookkeeping rather than what a gate measures. A change to them does not
#: invalidate a verdict.
#: ``agents`` is here too: which model writes the next change does not change
#: what an existing verdict measured. It is recorded in the verdict's config
#: (:attr:`VerdictContext.agents`) instead, as provenance. So is
#: ``trtllm_repo_path``: the code is compared by content, so the same code in
#: another checkout is the same evidence, and different code is caught anyway.
SPEC_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {"remediation", "source_embargo", "agents", "trtllm_repo_path"}
)
#: The ``environment`` keys that change what a gate measures. The rest
#: (partition, spool, mounts, time limits) only change where it runs.
SPEC_ENVIRONMENT_KEYS: tuple[str, ...] = ("python", "env", "container_image")

_GIT_TIMEOUT_S = 30
#: Native libraries at most this large are hashed whole; larger ones are
#: sampled. ``libtensorrt_llm.so`` is over 1 GB, and hashing it whole on a
#: networked filesystem would put tens of seconds in front of every verdict.
_FULL_HASH_LIMIT = 8 * 1024 * 1024
_SAMPLE_EDGE = 1024 * 1024
_SAMPLE_CHUNK = 64 * 1024
_SAMPLE_COUNT = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def fingerprint(value: Any) -> str:
    """A stable short digest of any JSON-able value."""
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:24]


def run_git(repo_root: str | Path, *args: str, stdin: str | None = None) -> str | None:
    """Ask git one question about ``repo_root``. ``None`` when it cannot answer.

    Every failure collapses to ``None``: no repository, no ``git`` binary, a
    hung filesystem. A ledger that raised here would turn a tooling problem
    into a failed gate.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
            input=stdin,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def is_git_repo(repo_root: str | Path) -> bool:
    out = run_git(repo_root, "rev-parse", "--is-inside-work-tree")
    return bool(out) and out.strip() == "true"


# ----------------------------------------------------------------- hashing

#: A file modified this recently is not cached: on a filesystem with coarse
#: timestamps a second write in the same tick, of the same size, would
#: otherwise read back as the first (git's "racy" entries, the same answer).
_RACY_WINDOW_NS = 2_000_000_000


def _stat_key(path: Path, stat: os.stat_result, *extra: Any) -> tuple[Any, ...] | None:
    """A cache key for ``path`` as ``stat`` describes it, or None if too fresh to trust."""
    if time.time_ns() - stat.st_mtime_ns < _RACY_WINDOW_NS:
        return None
    return (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, *extra)


_blob_cache: dict[tuple[Any, ...], str] = {}
_blob_lock = threading.Lock()


def git_blob_hash(data: bytes) -> str:
    """The id git gives ``data`` as a blob, without asking git."""
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def blob_hash(path: str | Path) -> str | None:
    """The git blob id of the file at ``path``, or ``None`` if there is no file.

    The same id ``git ls-tree`` reports for a committed file, so a working-tree
    file can be compared against ``HEAD`` without a ``git status`` over the
    whole worktree (a minute on the networked filesystem these runs use).
    Cached by ``(path, mtime, size)``.
    """
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    key = _stat_key(path, stat)
    if key is not None:
        with _blob_lock:
            cached = _blob_cache.get(key)
        if cached is not None:
            return cached
    try:
        digest = git_blob_hash(path.read_bytes())
    except OSError:
        return None
    if key is not None:
        with _blob_lock:
            _blob_cache[key] = digest
    return digest


# ------------------------------------------------------------ dependencies


def _is_dependency_file(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.startswith(".")
        and path.suffix not in (".pyc", ".pyo")
    )


def _expand(root: Path, rel: str) -> list[str]:
    """Repo-relative files at ``rel``: the file itself, or every file under it."""
    path = root / rel
    if path.is_file():
        return [Path(rel).as_posix()]
    if not path.is_dir():
        return []
    return [p.relative_to(root).as_posix() for p in path.rglob("*") if _is_dependency_file(p)]


def dependency_paths(
    repo_root: str | Path, target_rel: str, *, extra: Sequence[str] = ()
) -> list[str]:
    """The repo-relative files a verdict on ``target_rel`` depends on.

    - every file under the target directory;
    - the family-level files directly beside it (shared modules, routing);
    - the wrapper of every catalog entry any target module imports;
    - ``extra``: further files or directories the caller names.

    The catalog contracts (``.md``) are not included: a contract records a
    certification, and editing it does not change what the engine computes.
    """
    root = Path(repo_root)
    found: set[str] = set()
    if target_rel:
        target = root / target_rel
        found.update(_expand(root, target_rel))
        family = target.parent
        if family.is_dir():
            found.update(
                p.relative_to(root).as_posix() for p in family.iterdir() if _is_dependency_file(p)
            )
        for rel in sorted(found):
            if not rel.endswith(".py") or not rel.startswith(target_rel):
                continue
            try:
                entries = imported_catalog_entries(root / rel)
            except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
                continue
            for category, name in entries:
                wrapper = f"{CATALOG_DIR}/{category}/{name}.py"
                if (root / wrapper).is_file():
                    found.add(wrapper)
    for rel in extra:
        found.update(_expand(root, rel))
    return sorted(found)


def hash_paths(repo_root: str | Path, paths: Iterable[str]) -> dict[str, str | None]:
    root = Path(repo_root)
    return {rel: blob_hash(root / rel) for rel in paths}


# ------------------------------------------------------------- tree identity


@dataclass(frozen=True)
class TreeIdentity:
    """Which code a verdict was measured on.

    ``tree`` is the commit's tree object rather than the commit. Two commits
    with the same content (a re-signed or rebased commit, say) have the same
    tree, and evidence measured on one is evidence about the other.

    ``dirty`` lists the files among the verdict's dependencies whose
    working-tree content differs from the commit. ``dirty_digest`` pins that
    content, so two measurements over the same uncommitted edit compare equal
    and two over different edits do not.
    """

    commit: str = ""
    tree: str = ""
    describe: str = ""
    dirty: tuple[str, ...] = ()
    dirty_digest: str = ""

    @property
    def known(self) -> bool:
        return bool(self.tree)

    @property
    def key(self) -> str:
        """One token that compares equal exactly when the measured code does."""
        if not self.tree:
            return ""
        return f"{self.tree}+{self.dirty_digest}" if self.dirty else self.tree

    def short(self) -> str:
        if not self.known:
            return "unknown"
        label = self.describe or self.commit[:10]
        return f"{label}+dirty" if self.dirty else label

    def to_json(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "tree": self.tree,
            "describe": self.describe,
            "dirty": list(self.dirty),
            "dirty_digest": self.dirty_digest,
        }

    @classmethod
    def from_json(cls, raw: Any) -> "TreeIdentity":
        raw = raw if isinstance(raw, Mapping) else {}
        return cls(
            commit=str(raw.get("commit") or ""),
            tree=str(raw.get("tree") or ""),
            describe=str(raw.get("describe") or ""),
            dirty=tuple(str(p) for p in raw.get("dirty") or ()),
            dirty_digest=str(raw.get("dirty_digest") or ""),
        )


def committed_blobs(repo_root: str | Path, rev: str, paths: Sequence[str]) -> dict[str, str]:
    """``path -> blob id`` at ``rev`` for those of ``paths`` that exist there."""
    if not paths:
        return {}
    out = run_git(repo_root, "ls-tree", "-r", "-z", rev, "--", *paths)
    blobs: dict[str, str] = {}
    for record in (out or "").split("\0"):
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            blobs[path] = parts[2]
    return blobs


def tree_identity(
    repo_root: str | Path,
    *,
    rev: str = "HEAD",
    dependencies: Mapping[str, str | None] | None = None,
) -> TreeIdentity:
    """The identity of ``rev`` plus any uncommitted change among ``dependencies``.

    ``dependencies`` maps repo-relative paths to their working-tree blob ids,
    as :func:`hash_paths` returns them. Pass none to describe the commit alone,
    which is what the tree under merge is.
    """
    out = run_git(repo_root, "rev-parse", rev, f"{rev}^{{tree}}")
    lines = (out or "").split()
    if len(lines) < 2:
        return TreeIdentity()
    described = (run_git(repo_root, "describe", "--always", rev) or "").strip()
    dirty: list[str] = []
    if dependencies:
        committed = committed_blobs(repo_root, rev, sorted(dependencies))
        dirty = sorted(
            path
            for path, blob in dependencies.items()
            if blob != committed.get(path) and not (blob is None and path not in committed)
        )
    digest = ""
    if dirty:
        digest = fingerprint({path: dependencies[path] for path in dirty})  # type: ignore[index]
    return TreeIdentity(
        commit=lines[0], tree=lines[1], describe=described, dirty=tuple(dirty), dirty_digest=digest
    )


# ------------------------------------------------------------- native build


@dataclass(frozen=True)
class NativeBuild:
    """A content fingerprint of the compiled libraries the engine loads.

    ``fingerprint`` is empty when no library was found, which is the normal
    state of a worktree that borrows another checkout's build.
    """

    fingerprint: str = ""
    libraries: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return bool(self.fingerprint)

    def short(self) -> str:
        return self.fingerprint.split(":", 1)[-1][:12] if self.fingerprint else "none"

    def to_json(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint, "libraries": list(self.libraries)}

    @classmethod
    def from_json(cls, raw: Any) -> "NativeBuild":
        raw = raw if isinstance(raw, Mapping) else {}
        return cls(
            fingerprint=str(raw.get("fingerprint") or ""),
            libraries=tuple(str(x) for x in raw.get("libraries") or ()),
        )


_native_cache: dict[tuple[Any, ...], str] = {}
_native_lock = threading.Lock()


def _sampled_digest(path: Path, size: int) -> str:
    """Hash a library whole when small, at fixed offsets when large.

    The samples are the two ends and evenly spaced chunks between them, plus
    the size. A rebuild that changes code changes the size or the symbol and
    section tables at the ends; an identical reinstall changes nothing here,
    which is the point of hashing content rather than mtimes.
    """
    sha = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        if size <= _FULL_HASH_LIMIT:
            sha.update(handle.read())
            return sha.hexdigest()
        offsets = [0, max(0, size - _SAMPLE_EDGE)]
        step = size // (_SAMPLE_COUNT + 1)
        offsets[1:1] = [step * (i + 1) for i in range(_SAMPLE_COUNT)]
        for index, offset in enumerate(offsets):
            handle.seek(offset)
            length = _SAMPLE_EDGE if index in (0, len(offsets) - 1) else _SAMPLE_CHUNK
            sha.update(handle.read(length))
    return sha.hexdigest()


def native_build(
    repo_root: str | Path, *, globs: Sequence[str] = DEFAULT_NATIVE_GLOBS
) -> NativeBuild:
    """Fingerprint the native libraries under ``repo_root``."""
    root = Path(repo_root)
    rows: list[str] = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            try:
                stat = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            key = _stat_key(path.resolve(), stat)
            digest = None
            if key is not None:
                with _native_lock:
                    digest = _native_cache.get(key)
            if digest is None:
                try:
                    digest = _sampled_digest(path, stat.st_size)
                except OSError:
                    continue
                if key is not None:
                    with _native_lock:
                        _native_cache[key] = digest
            rows.append(f"{path.relative_to(root).as_posix()} {stat.st_size} {digest[:16]}")
    rows = sorted(set(rows))
    if not rows:
        return NativeBuild()
    return NativeBuild(fingerprint=fingerprint(rows), libraries=tuple(rows))


# ---------------------------------------------------------------- switches

_ENV_FUNCS = {"get", "getenv", "setdefault"}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


def _default_repr(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return None if node.value is None else str(node.value)
    return f"<expr {ast.unparse(node)}>"


def harvest_switch_defaults(
    source: str, *, prefixes: Sequence[str] = DEFAULT_SWITCH_PREFIXES
) -> dict[str, str | None]:
    """Every switch ``source`` reads, with the default it falls back to.

    Recognises ``os.environ.get(NAME, default)``, ``os.getenv(NAME, default)``,
    ``os.environ.setdefault(NAME, default)`` and ``os.environ[NAME]``, for
    names starting with one of ``prefixes``. A read with no default records
    ``None``; a read that has none at all (``os.environ[NAME]``) records
    ``"<required>"``. A default that is not a literal is recorded as its
    source text, so a change to the expression still reads as a change. A
    name read with two different defaults records both, joined with ``|``.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    found: dict[str, set[str | None]] = {}

    def add(name: Any, default: str | None) -> None:
        if isinstance(name, str) and name.startswith(tuple(prefixes)):
            found.setdefault(name, set()).add(default)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
            dotted = _dotted(node.func)
            last = dotted.rsplit(".", 1)[-1]
            owner = dotted.rsplit(".", 1)[0] if "." in dotted else ""
            is_env = last in _ENV_FUNCS and (owner.endswith("environ") or last == "getenv")
            if not is_env or not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            default_node = node.args[1] if len(node.args) > 1 else None
            for keyword in node.keywords:
                if keyword.arg in ("default", "value"):
                    default_node = keyword.value
            add(node.args[0].value, _default_repr(default_node))
        elif isinstance(node, ast.Subscript) and _dotted(node.value).endswith("environ"):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(node.ctx, ast.Load):
                add(key.value, "<required>")
    return {
        name: (
            next(iter(values))
            if len(values) == 1
            else " | ".join(sorted("None" if v is None else v for v in values))
        )
        for name, values in sorted(found.items())
    }


_harvest_cache: dict[tuple[Any, ...], dict[str, str | None]] = {}
_harvest_lock = threading.Lock()


def harvest_paths(
    repo_root: str | Path,
    paths: Iterable[str],
    *,
    prefixes: Sequence[str] = DEFAULT_SWITCH_PREFIXES,
) -> dict[str, str | None]:
    """Merge :func:`harvest_switch_defaults` over the Python files at ``paths``.

    ``paths`` may name directories, which are walked.
    """
    root = Path(repo_root)
    merged: dict[str, set[str | None]] = {}
    files: set[str] = set()
    for rel in paths:
        files.update(p for p in _expand(root, rel) if p.endswith(".py"))
    for rel in sorted(files):
        path = root / rel
        try:
            stat = path.stat()
        except OSError:
            continue
        key = _stat_key(path, stat, tuple(prefixes))
        harvested = None
        if key is not None:
            with _harvest_lock:
                harvested = _harvest_cache.get(key)
        if harvested is None:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            harvested = harvest_switch_defaults(text, prefixes=prefixes)
            if key is not None:
                with _harvest_lock:
                    _harvest_cache[key] = harvested
        for name, default in harvested.items():
            merged.setdefault(name, set()).add(default)
    return {
        name: (
            next(iter(values))
            if len(values) == 1
            else " | ".join(sorted("None" if v is None else v for v in values))
        )
        for name, values in sorted(merged.items())
    }


def command_env(command: str) -> dict[str, str]:
    """``NAME=value`` assignments anywhere in a shell command line."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    env: dict[str, str] = {}
    for token in tokens:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", token, flags=re.S)
        if match:
            env[match.group(1)] = match.group(2)
    return env


def switch_env(env: Mapping[str, str], prefixes: Sequence[str]) -> dict[str, str]:
    return {k: str(v) for k, v in sorted(env.items()) if str(k).startswith(tuple(prefixes))}


# -------------------------------------------------------------------- spec


def _project(value: Any, unit: str, units: frozenset[str]) -> Any:
    """Keep only what applies to ``unit`` in a per-unit mapping, recursively.

    A mapping whose keys are all unit names (or ``default``) is the per-unit
    form several ``task.yaml`` blocks accept. For one unit, only its own entry
    and ``default`` matter, so another unit's settings never invalidate it.
    """
    if isinstance(value, Mapping):
        keys = {str(k) for k in value}
        if keys and keys <= units | {"default"}:
            value = {k: v for k, v in value.items() if str(k) in (unit, "default")}
        projected = {str(k): _project(v, unit, units) for k, v in value.items()}
        return {k: v for k, v in projected.items() if v not in ({}, None)}
    if isinstance(value, (list, tuple)):
        return [_project(v, unit, units) for v in value]
    return value


def spec_projection(task_data: Mapping[str, Any], unit: str) -> dict[str, Any]:
    """The part of ``task.yaml`` that can change what ``unit``'s gates measure."""
    units = assembly_units(task_data)
    known = frozenset(units)
    projected: dict[str, Any] = {}
    for key, value in task_data.items():
        if key in SPEC_EXCLUDED_KEYS:
            continue
        if key == "features":
            # A unit's gates run with the features assembled so far; adding a
            # later feature does not change what an earlier unit measures.
            if unit in units:
                value = units[1 : units.index(unit) + 1]
        elif key == "environment" and isinstance(value, Mapping):
            value = {k: value[k] for k in SPEC_ENVIRONMENT_KEYS if k in value}
        projected[str(key)] = _project(value, unit, known)
    return projected


def spec_fingerprint(task_data: Mapping[str, Any] | None, unit: str) -> str:
    if not task_data:
        return ""
    return fingerprint(spec_projection(task_data, unit))


# ------------------------------------------------------------------ verdicts


@dataclass(frozen=True)
class LedgerPolicy:
    """What a run treats as a dependency and as a switch."""

    switch_prefixes: tuple[str, ...] = DEFAULT_SWITCH_PREFIXES
    #: Files or directories, besides each verdict's own dependencies, whose
    #: switch defaults are recorded with every verdict.
    switch_sources: tuple[str, ...] = (MODELING_V2_PACKAGE,)
    #: Files or directories every verdict depends on, on top of the target.
    extra_dependencies: tuple[str, ...] = ()
    native_globs: tuple[str, ...] = DEFAULT_NATIVE_GLOBS


@dataclass(frozen=True)
class Verdict:
    """One gate verdict, with everything needed to decide later whether it still holds."""

    gate: str
    unit: str
    member: str
    outcome: str
    value: float | None = None
    threshold: float | None = None
    reference: float | None = None
    tolerance: float | None = None
    detail: str = ""
    timestamp: str = ""
    duration_s: float = 0.0
    target_rel: str = ""
    tree: TreeIdentity = field(default_factory=TreeIdentity)
    native: NativeBuild = field(default_factory=NativeBuild)
    dependencies: Mapping[str, str | None] = field(default_factory=dict)
    #: Explicit switch values the gate ran under (task env plus command).
    env: Mapping[str, str] = field(default_factory=dict)
    #: The switch defaults the code declared at measurement time.
    switch_defaults: Mapping[str, str | None] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""
    spec_fingerprint: str = ""
    #: The gate's own one-token tree description, kept for humans.
    tree_state: str = ""
    dirty_override: bool = False
    log: str = ""
    id: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.gate, self.unit, self.member)

    @property
    def passed(self) -> bool:
        """Whether the verdict clears its threshold. A finding does."""
        return self.outcome in (OUTCOME_PASS, OUTCOME_FINDING)

    def label(self) -> str:
        member = f"@{self.member}" if self.member else ""
        return f"{self.gate}{member} [{self.unit or '-'}]"

    def value_text(self) -> str:
        if self.value is None:
            return self.detail[:120]
        if self.threshold is None:
            return f"{self.value:.4f}"
        return f"{self.value:.4f} vs {self.threshold:.4f}"

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": KIND_VERDICT,
            "schema": SCHEMA_VERSION,
            "gate": self.gate,
            "unit": self.unit,
            "member": self.member,
            "outcome": self.outcome,
            "value": self.value,
            "threshold": self.threshold,
            "reference": self.reference,
            "tolerance": self.tolerance,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "target_rel": self.target_rel,
            "tree": self.tree.to_json(),
            "native_build": self.native.to_json(),
            "dependencies": dict(sorted(self.dependencies.items())),
            "env": dict(sorted(self.env.items())),
            "switch_defaults": dict(sorted(self.switch_defaults.items())),
            "config": dict(self.config),
            "config_fingerprint": self.config_fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
            "tree_state": self.tree_state,
            "dirty_override": self.dirty_override,
            "log": self.log,
        }
        record["id"] = self.id or fingerprint(record).split(":", 1)[1][:16]
        return record

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> "Verdict":
        def number(key: str) -> float | None:
            value = raw.get(key)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        outcome = str(raw.get("outcome") or OUTCOME_FAIL).upper()
        return cls(
            gate=str(raw.get("gate") or ""),
            unit=str(raw.get("unit") or ""),
            member=str(raw.get("member") or ""),
            outcome=outcome if outcome in OUTCOMES else OUTCOME_FAIL,
            value=number("value"),
            threshold=number("threshold"),
            reference=number("reference"),
            tolerance=number("tolerance"),
            detail=str(raw.get("detail") or ""),
            timestamp=str(raw.get("timestamp") or ""),
            duration_s=float(raw.get("duration_s") or 0.0),
            target_rel=str(raw.get("target_rel") or ""),
            tree=TreeIdentity.from_json(raw.get("tree")),
            native=NativeBuild.from_json(raw.get("native_build")),
            dependencies={
                str(k): (None if v is None else str(v))
                for k, v in (raw.get("dependencies") or {}).items()
            },
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            switch_defaults={
                str(k): (None if v is None else str(v))
                for k, v in (raw.get("switch_defaults") or {}).items()
            },
            config=dict(raw.get("config") or {}),
            config_fingerprint=str(raw.get("config_fingerprint") or ""),
            spec_fingerprint=str(raw.get("spec_fingerprint") or ""),
            tree_state=str(raw.get("tree_state") or ""),
            dirty_override=bool(raw.get("dirty_override", False)),
            log=str(raw.get("log") or ""),
            id=str(raw.get("id") or ""),
        )


# --------------------------------------------------------------------- store


class Ledger:
    """The append-only JSON Lines store.

    An append is one ``write`` of one line to a file opened ``O_APPEND``,
    under an exclusive ``flock`` and a thread lock, so concurrent recorders
    in one process or in several never interleave. A torn final line (a
    process killed mid-write) is skipped on read and counted in
    :attr:`skipped`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.skipped = 0
        self._lock = threading.Lock()

    @classmethod
    def in_workspace(cls, workspace: str | Path) -> "Ledger":
        return cls(Path(workspace) / LEDGER_FILENAME)

    def append(self, record: Mapping[str, Any]) -> None:
        data = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                os.fsync(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def records(self) -> list[dict[str, Any]]:
        self.skipped = 0
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.skipped += 1
                continue
            if isinstance(record, dict):
                out.append(record)
            else:
                self.skipped += 1
        return out

    def verdicts(self) -> list[Verdict]:
        return [Verdict.from_record(r) for r in self.records() if r.get("kind") == KIND_VERDICT]

    def latest(self) -> dict[tuple[str, str, str], Verdict]:
        """The newest verdict per ``(gate, unit, member)``, in first-seen key order."""
        latest: dict[tuple[str, str, str], Verdict] = {}
        for verdict in self.verdicts():
            latest[verdict.key] = verdict
        return latest

    def digest(self) -> str:
        """A digest of the ledger's bytes, for reports to say what they were made from."""
        try:
            return "sha256:" + hashlib.sha256(self.path.read_bytes()).hexdigest()[:24]
        except FileNotFoundError:
            return ""

    def record_verdict(self, verdict: Verdict) -> Verdict:
        record = verdict.to_record()
        self.append(record)
        return replace(verdict, id=record["id"])


# ----------------------------------------------------------------- recording


@dataclass(frozen=True)
class VerdictContext:
    """What the recorder knows that a gate result does not."""

    repo_root: str = ""
    target_rel: str = ""
    unit: str = ""
    #: The explicit environment every gate command runs under (``task.yaml``'s
    #: ``environment.env``). Assignments inside the command itself are added.
    env: Mapping[str, str] = field(default_factory=dict)
    features: Sequence[str] = ()
    task_data: Mapping[str, Any] | None = None
    policy: LedgerPolicy = field(default_factory=LedgerPolicy)
    log: str = ""
    #: The backend and model each role is routed to (``role -> {backend,
    #: model, ...}``): who wrote the code a verdict measured. Part of the
    #: config fingerprint, not of freshness.
    agents: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def outcome_of(result: Any) -> str:
    """PASS, FAIL or FINDING for a gate result of any vintage.

    A result may carry an explicit ``outcome``; one that says ``finding`` is a
    finding; otherwise ``passed`` decides.
    """
    explicit = str(getattr(result, "outcome", "") or "").upper()
    if explicit in OUTCOMES:
        return explicit
    if getattr(result, "finding", False):
        return OUTCOME_FINDING
    return OUTCOME_PASS if getattr(result, "passed", False) else OUTCOME_FAIL


@dataclass(frozen=True)
class Measurement:
    """What a verdict is measured against, read once from the tree.

    Separate from the result so a batch of verdicts taken at one moment (the
    checks of one delivery report) reads the tree once.
    """

    tree: TreeIdentity
    native: NativeBuild
    dependencies: Mapping[str, str | None]
    switch_defaults: Mapping[str, str | None]


def measure(context: VerdictContext) -> Measurement:
    """Fingerprint the tree, the native build, the dependencies and the switch defaults now."""
    policy = context.policy
    root = context.repo_root
    if not root:
        return Measurement(TreeIdentity(), NativeBuild(), {}, {})
    paths = dependency_paths(root, context.target_rel, extra=policy.extra_dependencies)
    dependencies = hash_paths(root, paths)
    return Measurement(
        tree=tree_identity(root, dependencies=dependencies),
        native=native_build(root, globs=policy.native_globs),
        dependencies=dependencies,
        switch_defaults=harvest_paths(
            root, [*paths, *policy.switch_sources], prefixes=policy.switch_prefixes
        ),
    )


def verdict_from_result(
    result: Any,
    context: VerdictContext,
    *,
    gate: str | None = None,
    measurement: Measurement | None = None,
) -> Verdict:
    """Build a :class:`Verdict` from a gate result, fingerprinting the tree now.

    ``result`` is duck-typed against ``gates.GateResult`` so a gate added
    later records without this module changing: ``name``, ``passed`` and
    ``detail`` are required; ``member``, ``measured``, ``reference``,
    ``tolerance``, ``threshold``, ``command``, ``duration_s``,
    ``tree_state``, ``dirty_override``, ``outcome`` and ``finding`` are read
    when present. ``gate`` overrides the result's name as the ledger key,
    for a gate that runs at several points under one name.
    """
    policy = context.policy
    measured = measurement or measure(context)
    command = str(getattr(result, "command", "") or "")
    explicit = {str(k): str(v) for k, v in dict(context.env or {}).items()}
    explicit.update(command_env(command))
    switches = switch_env(explicit, policy.switch_prefixes)
    reference = _number(getattr(result, "reference", None))
    tolerance = _number(getattr(result, "tolerance", None))
    threshold = _number(getattr(result, "threshold", None))
    if threshold is None and reference is not None:
        threshold = reference - (tolerance or 0.0)
    gate = gate or str(getattr(result, "name", "") or "gate")
    member = str(getattr(result, "member", "") or "")
    config = {
        "gate": gate,
        "unit": context.unit,
        "member": member,
        "features": [str(f) for f in context.features],
        "env": switches,
    }
    if context.agents:
        config["agents"] = {str(role): dict(agent) for role, agent in context.agents.items()}
    return Verdict(
        gate=gate,
        unit=context.unit,
        member=member,
        outcome=outcome_of(result),
        value=_number(getattr(result, "measured", None)),
        threshold=threshold,
        reference=reference,
        tolerance=tolerance,
        detail=str(getattr(result, "detail", "") or "")[:2000],
        timestamp=utc_now(),
        duration_s=float(getattr(result, "duration_s", 0.0) or 0.0),
        target_rel=context.target_rel,
        tree=measured.tree,
        native=measured.native,
        dependencies=measured.dependencies,
        env=switches,
        switch_defaults=measured.switch_defaults,
        config=config,
        config_fingerprint=fingerprint({**config, "command": command, "env": explicit}),
        spec_fingerprint=spec_fingerprint(context.task_data, context.unit),
        tree_state=str(getattr(result, "tree_state", "") or ""),
        dirty_override=bool(getattr(result, "dirty_override", False)),
        log=context.log,
    )


def record_gate_result(
    ledger: Ledger,
    result: Any,
    context: VerdictContext,
    *,
    gate: str | None = None,
    measurement: Measurement | None = None,
) -> Verdict:
    """Record one gate result. **The** entry point for gates and harnesses.

    Stable on purpose: ``gates.GateLog`` calls a sink with each result, and
    the workflow's sink is this function with the run's context bound.
    """
    return ledger.record_verdict(
        verdict_from_result(result, context, gate=gate, measurement=measurement)
    )


def select(verdicts: Iterable[Verdict], *, gate="*", unit="*", member="*") -> list[Verdict]:
    """Verdicts whose gate, unit and member match the ``fnmatch`` patterns."""
    return [
        v
        for v in verdicts
        if fnmatch.fnmatchcase(v.gate, gate)
        and fnmatch.fnmatchcase(v.unit, unit)
        and fnmatch.fnmatchcase(v.member, member)
    ]


# ----------------------------------------------------------------------- CLI


@dataclass(frozen=True)
class _RecordedResult:
    """A gate result handed in on the command line, shaped like ``gates.GateResult``."""

    name: str
    passed: bool
    detail: str
    member: str = ""
    measured: float | None = None
    threshold: float | None = None
    command: str = ""
    outcome: str = ""
    duration_s: float = 0.0


def load_task(path: str | Path | None) -> dict[str, Any] | None:
    """Read a ``task.yaml`` for its spec fingerprint, or ``None``."""
    if not path:
        return None
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def main(argv: Sequence[str] | None = None) -> int:
    """``record``: put a verdict measured outside the workflow into a workspace's ledger.

    For a harness that runs a gate by hand -- a regression sweep, an A/B --
    so its verdict carries the same fingerprints as the workflow's own and is
    judged by the same freshness rules.
    """
    parser = argparse.ArgumentParser(prog="staircase-ledger", description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="record a verdict measured outside the workflow")
    record.add_argument("workspace")
    record.add_argument("--repo", required=True)
    record.add_argument("--target-rel", default="")
    record.add_argument("--gate", required=True)
    record.add_argument("--unit", required=True)
    record.add_argument("--member", default="")
    record.add_argument("--outcome", choices=OUTCOMES, required=True)
    record.add_argument("--value", type=float)
    record.add_argument("--threshold", type=float)
    record.add_argument("--detail", default="")
    record.add_argument("--command-line", default="", help="the command the gate ran")
    record.add_argument("--env", action="append", default=[], help="NAME=value, repeatable")
    record.add_argument("--task", help="task.yaml, for the spec fingerprint")
    record.add_argument("--log", default="", help="where the gate's log is")
    args = parser.parse_args(argv)

    env = dict(item.split("=", 1) for item in args.env if "=" in item)
    verdict = record_gate_result(
        Ledger.in_workspace(args.workspace),
        _RecordedResult(
            name=args.gate,
            passed=args.outcome != OUTCOME_FAIL,
            detail=args.detail,
            member=args.member,
            measured=args.value,
            threshold=args.threshold,
            command=args.command_line,
            outcome=args.outcome,
        ),
        VerdictContext(
            repo_root=args.repo,
            target_rel=args.target_rel,
            unit=args.unit,
            env=env,
            task_data=load_task(args.task),
            log=args.log,
        ),
    )
    sys.stdout.write(f"recorded {verdict.id}: {verdict.label()} {verdict.outcome}\n")
    return 0


__all__ = [
    "DEFAULT_NATIVE_GLOBS",
    "DEFAULT_SWITCH_PREFIXES",
    "KIND_REGRESSION",
    "KIND_VERDICT",
    "LEDGER_FILENAME",
    "Ledger",
    "LedgerPolicy",
    "Measurement",
    "NativeBuild",
    "OUTCOMES",
    "OUTCOME_FAIL",
    "OUTCOME_FINDING",
    "OUTCOME_PASS",
    "TreeIdentity",
    "Verdict",
    "VerdictContext",
    "blob_hash",
    "command_env",
    "dependency_paths",
    "fingerprint",
    "harvest_paths",
    "harvest_switch_defaults",
    "hash_paths",
    "is_git_repo",
    "load_task",
    "measure",
    "native_build",
    "outcome_of",
    "record_gate_result",
    "select",
    "spec_fingerprint",
    "spec_projection",
    "run_git",
    "tree_identity",
    "utc_now",
    "verdict_from_result",
]


if __name__ == "__main__":
    sys.exit(main())
