# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks each deployment owes, and the ledger of what they measured.

A deployment owes its boot, one accuracy check for each of its benchmarks, and its native
tests when it names any (:func:`compile_contract`). Code runs core's boot and benchmarks when `core` lands (the
onboard check) and every other check of a done deployment once, in the final pass
(``run/report.py``); a worker runs its own deployment's by hand while it builds.

Every result is appended to ``verdicts.jsonl`` with the commit it was measured on
and what it measured by content (:func:`tree_id`: the git tree, the native libraries built
into it, what the checks read of the spec, this workflow's version, the environment). A
pass on record for the same content is reused, never re-run; a verdict about any other
content says nothing about this one.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping

from ..checks import attest
from ..infra.git import git_output
from ..operators.coverage import MODELING_V2_PACKAGE
from ..spec import deployments as deployments_mod

STATUS_FILENAME = "verdicts.jsonl"
#: Where the native libraries an engine loads are built: inside the checkout, ignored by git.
NATIVE_LIBRARY_DIRS: tuple[str, ...] = ("tensorrt_llm", "tensorrt_llm/libs")
#: Where a family builds libraries of its own: anywhere under the Modeling V2 package.
FAMILY_LIBRARY_DIR = MODELING_V2_PACKAGE
NATIVE_LIBRARY_PATTERNS: tuple[str, ...] = ("*.so", "*.so.*")


@dataclass(frozen=True)
class Check:
    """One check: a gate of one deployment."""

    deployment: str
    gate: str

    @property
    def key(self) -> str:
        return f"{self.deployment}:{self.gate}"


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    #: What a reader should know of what the checks do not judge, one line each.
    notes: tuple[str, ...] = ()

    def of(self, deployment: str) -> tuple[Check, ...]:
        """The checks ``deployment`` owes."""
        return tuple(c for c in self.checks if c.deployment == deployment)


def compile_contract(task_data: Mapping[str, Any]) -> Contract:
    """Every check the spec's deployments owe, core's first."""
    checks: list[Check] = []
    notes: list[str] = []
    for deployment in deployments_mod.resolve(task_data):
        name = deployment.name
        checks.append(Check(name, "boot"))
        checks += [Check(name, f"accuracy[{b.name}]") for b in deployment.benchmarks]
        if not deployment.benchmarks:
            notes.append(f"{name}: it names no benchmark, so its boot alone judges it")
        if deployment.tests:
            checks.append(Check(name, "tests"))
        # Speculative decoding preserves the output distribution by construction, so a
        # draft path that computes the wrong thing passes boot and accuracy and shows only
        # as speed. What detects it is a native test of the drafter.
        drafts = any(
            (server.options.get("speculative_config") or {}).get("decoding_type")
            for server in deployment.servers.values()
        )
        if drafts and not deployment.tests:
            notes.append(
                f"{name}: its drafter's acceptance is judged by no check here; name a native "
                f"test for it in `tests`"
            )
    return Contract(checks=tuple(checks), notes=tuple(notes))


# ------------------------------------------------------------------ the tree a verdict is about


def native_libraries(repo: Path) -> list[Path]:
    """Each native library built into the checkout, sorted: the engine's and a family's own."""
    found: set[Path] = set()
    for directory in NATIVE_LIBRARY_DIRS:
        for pattern in NATIVE_LIBRARY_PATTERNS:
            found.update(p for p in (repo / directory).glob(pattern) if p.is_file())
    for pattern in NATIVE_LIBRARY_PATTERNS:
        found.update(p for p in (repo / FAMILY_LIBRARY_DIR).rglob(pattern) if p.is_file())
    return sorted(found)


def content_hash(path: Path) -> str:
    """The SHA-256 of a file's bytes, read every time: size and time do not say content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def library_digest(repo: str | Path) -> str:
    """One hash of the contents of the native libraries built into ``repo``.

    Content, not size and time: the same build copied to another node is the same
    libraries, and a rebuild that changes a byte is not.
    """
    digest = hashlib.sha256()
    root = Path(repo)
    for library in native_libraries(root):
        try:
            digest.update(
                f"{library.relative_to(root).as_posix()} {content_hash(library)}\n".encode()
            )
        except OSError:
            continue
    return digest.hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def flow_version() -> str:
    """A hash of this workflow's own sources: a verdict belongs to the checks that gave it."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:12]


def inputs_digest(task_data: Mapping[str, Any]) -> str:
    """A hash of what the checks read of the spec: all of it but ``notes`` and ``agents``.

    With the content of what a check reads through it: the options each deployment
    resolves to and the boot workload's prompt files, so a file edited in place counts
    as the spec edit it is.
    """
    spec = {k: v for k, v in task_data.items() if k not in ("notes", "agents")}
    boot = (task_data.get("accuracy") or {}).get("boot") or {}
    levels = [boot, *(v for v in boot.values() if isinstance(v, Mapping))]
    prompts = []
    for path in sorted(
        {str(level["prompts_file"]) for level in levels if level.get("prompts_file")}
    ):
        try:
            prompts.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
        except OSError:
            prompts.append(f"unreadable:{path}")
    try:
        resolved = [
            {role: dict(server.options) for role, server in deployment.servers.items()}
            for deployment in deployments_mod.resolve(task_data)
        ]
    except (ValueError, OSError, KeyError):
        resolved = []
    text = json.dumps([spec, resolved, prompts], sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def tree_id(
    repo: str | Path,
    commit: str,
    task_data: Mapping[str, Any] | None = None,
    environment: str = "",
    git: Callable[..., Any] = git_output,
) -> str:
    """What a verdict is about, by content.

    The git tree of ``commit`` (not the commit: the same files under another commit are
    the same thing measured) and the contents of the native libraries built into
    ``repo``, which are outside git. With ``task_data``, also what the checks read of
    the spec, this workflow's own version and ``environment`` (a digest of where the
    checks run): a verdict counts only where all five are what they were.
    ``""`` without a commit or without a git tree for it: nothing is established, and no
    verdict counts or is reused.
    """
    words = (git(str(repo), "rev-parse", f"{commit}^{{tree}}") or "").split() if commit else []
    if not words:
        return ""
    parts = [words[0], library_digest(repo)]
    if task_data is not None:
        parts += [inputs_digest(task_data), flow_version(), environment or "unknown-environment"]
    return "+".join(parts)


def libraries_of(identity: str) -> str:
    """The native libraries' digest in a verdict's identity (:func:`tree_id`); ``""`` without one."""
    parts = identity.split("+")
    return parts[1] if len(parts) > 1 else ""


# ------------------------------------------------------------------------------ the ledger


@dataclass
class Verdict:
    deployment: str
    gate: str
    passed: bool
    detail: str = ""
    #: What the check measured, by content (:func:`tree_id`), and the commit it was at.
    tree: str = ""
    commit: str = ""
    duration_s: float = 0.0
    #: The score a benchmark measured, pass or fail, and the reference it was held to.
    measured: float | None = None
    reference: float | None = None
    #: The floor the score was held to, where something judged it.
    floor: float | None = None
    #: The gate's one-line summary, with its numbers.
    summary: str = ""
    #: A score nothing judged: reported, neither a pass nor a failure.
    unjudged: bool = False
    #: Which path judged: ``trtllm-eval`` or through the endpoint.
    path: str = ""
    #: What the check attested of each server's engine (``checks/attest.py``): its options
    #: against what every rank ran, and the target class; empty for a check that attests none.
    attested: str = ""

    @property
    def key(self) -> str:
        return f"{self.deployment}:{self.gate}"

    @property
    def said(self) -> str:
        """The verdict in a word, as the report prints it."""
        return "no verdict" if self.unjudged else "PASS" if self.passed else "FAIL"


_VERDICT_FIELDS = frozenset(f.name for f in fields(Verdict))


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ContractStatus:
    """The verdicts recorded for a workspace, append-only on disk, oldest first."""

    path: Path
    verdicts: list[Verdict] = field(default_factory=list)

    @classmethod
    def load(cls, workspace: Path) -> "ContractStatus":
        path = Path(workspace) / STATUS_FILENAME
        verdicts: list[Verdict] = []
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else ():
            try:
                data = json.loads(line)
                verdicts.append(Verdict(**{k: v for k, v in data.items() if k in _VERDICT_FIELDS}))
            except (ValueError, TypeError, AttributeError):
                continue  # a torn line, or another version's: the check it was for runs again
        return cls(path=path, verdicts=verdicts)

    def record(self, check: Check, result: Any, *, tree: str, commit: str = "") -> Verdict:
        """Append one gate result for ``check``, measured on ``tree`` at ``commit``.

        A result that says the infrastructure ended it measured nothing: it is recorded
        as not passed, never as a pass.
        """
        infrastructure = str(getattr(result, "infrastructure", "") or "")
        verdict = Verdict(
            deployment=check.deployment,
            gate=check.gate,
            passed=bool(result.passed) and not infrastructure,
            detail=str(result.detail)[:500],
            tree=tree,
            commit=commit,
            duration_s=round(_number(getattr(result, "duration_s", 0.0)) or 0.0, 3),
            measured=_number(getattr(result, "measured", None)),
            reference=_number(getattr(result, "reference", None)),
            floor=_number(getattr(result, "floor", None)),
            summary=str(result.summary())[:1000] if hasattr(result, "summary") else "",
            unjudged=bool(getattr(result, "unjudged", False)) and not infrastructure,
            path=str(getattr(result, "path", "") or ""),
            attested=attest.describe(getattr(result, "engine", ())).strip(),
        )
        self.verdicts.append(verdict)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Synced: a verdict that cost a load must survive the node that measured it.
        # A line a dead process tore is closed first: the new verdict is its own line.
        torn = False
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open("rb") as written:
                written.seek(-1, os.SEEK_END)
                torn = written.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(("\n" if torn else "") + json.dumps(asdict(verdict)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return verdict

    def on(self, tree: str) -> dict[str, Verdict]:
        """The latest verdict per check key measured on ``tree``; none for an unknown tree.

        The latest outcome counts: a failure recorded after a pass replaces it.
        """
        return {v.key: v for v in self.verdicts if tree and v.tree == tree}

    def at(self, key: str, commit: str, *, passed: bool | None = None) -> Verdict | None:
        """The latest verdict of ``key`` measured at ``commit``; with ``passed``, the latest that was."""
        found = [
            v
            for v in self.verdicts
            if v.key == key and commit and v.commit == commit and passed in (None, v.passed)
        ]
        return found[-1] if found else None

    def last(self, key: str, *, passed: bool | None = None) -> Verdict | None:
        """The latest verdict of ``key`` on any content; with ``passed``, the latest that was."""
        found = [v for v in self.verdicts if v.key == key and passed in (None, v.passed)]
        return found[-1] if found else None
