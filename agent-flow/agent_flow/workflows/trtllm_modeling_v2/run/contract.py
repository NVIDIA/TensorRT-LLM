# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract: every check a run must pass, compiled from the resolved task spec.

The contract is the run's definition of "done". It names each check by its
deployment and gate: a boot, one accuracy check for each of the deployment's
benchmarks, its native tests when it names any, and the serve gate for core
when the spec turns it on. Nothing runs it while the workers build: a worker runs the
checks of its own unit by hand (``trtllm-modeling-v2 gate``) and its conclusion
is trusted. The contract is run once, whole, on the final commit
(``run/certify.py``), and a run is delivered only when every required check
passed on that one commit.

Verdicts are appended to ``contract_status.jsonl``, each with the tree it
measured (:func:`tree_id`: the git tree and the contents of the native libraries built into the
checkout). A verdict about any other tree does not count (:func:`evaluate`).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from ..checks import serve
from ..infra.git import git_output
from ..operators.coverage import MODELING_V2_PACKAGE
from ..spec import deployments as deployments_mod
from ..spec.deployments import AGGREGATED

CONTRACT_FILENAME = "contract.yaml"
STATUS_FILENAME = "contract_status.jsonl"
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
    required: bool = True

    @property
    def key(self) -> str:
        return f"{self.deployment}:{self.gate}"


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    #: What a reader should know of what the checks do not judge, one line each.
    notes: tuple[str, ...] = ()

    def required(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.required)

    def of(self, deployment: str) -> tuple[Check, ...]:
        """The checks ``deployment`` owes."""
        return tuple(c for c in self.checks if c.deployment == deployment)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"checks": [asdict(c) for c in self.checks]}
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    def render(self) -> str:
        """A markdown table, one row per check."""
        lines = ["| deployment | gate | required |", "|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.deployment} | {c.gate} | {'yes' if c.required else 'no'} |")
        if self.notes:
            lines += ["", *(f"- {note}" for note in self.notes)]
        return "\n".join(lines)


def compile_contract(task_data: Mapping[str, Any]) -> Contract:
    """Every check the spec's deployments owe, core's first.

    A deployment owes its boot, one accuracy check for each of its benchmarks, and its
    native tests when it names any. Core, when it is one server, also owes the serve
    gate where ``accuracy.serve.enabled`` asks for it.
    """
    serving = serve.ServeSpec.from_task(task_data).enabled
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
        if serving and deployment.core and deployment.kind == AGGREGATED:
            checks.append(Check(name, serve.GATE_NAME))
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
                f"test for it in `tests`, or the report says it is unverified"
            )
    return Contract(checks=tuple(checks), notes=tuple(notes))


def write_contract(workspace: Path, contract: Contract) -> Path:
    path = Path(workspace) / CONTRACT_FILENAME
    path.write_text(yaml.safe_dump(contract.as_dict(), sort_keys=False), encoding="utf-8")
    return path


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


def _content_hash(path: Path) -> str:
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
                f"{library.relative_to(root).as_posix()} {_content_hash(library)}\n".encode()
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


# ------------------------------------------------------------------------------ verdicts


@dataclass
class Verdict:
    deployment: str
    gate: str
    passed: bool
    detail: str = ""
    #: The tree the check measured (:func:`tree_id`).
    tree: str = ""
    #: The certification round that ran it.
    round: int = 0
    #: How long the check's gate ran, in seconds; 0 when it did not say.
    duration_s: float = 0.0
    #: The score a benchmark measured (``GateResult.measured``), pass or fail, the reference
    #: it was held to and its standard error.
    measured: float | None = None
    reference: float | None = None
    se: float | None = None
    #: The gate's one-line summary, with its numbers.
    summary: str = ""

    @property
    def key(self) -> str:
        return f"{self.deployment}:{self.gate}"


_VERDICT_FIELDS = frozenset(f.name for f in fields(Verdict))


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ContractStatus:
    """The verdicts recorded against a contract, append-only on disk."""

    path: Path
    verdicts: list[Verdict] = field(default_factory=list)

    @classmethod
    def load(cls, workspace: Path) -> "ContractStatus":
        path = Path(workspace) / STATUS_FILENAME
        verdicts: list[Verdict] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    data = json.loads(line)
                except ValueError:
                    continue  # a torn line: the check it was for runs again
                if not isinstance(data, dict):
                    continue
                try:
                    verdicts.append(
                        Verdict(**{k: v for k, v in data.items() if k in _VERDICT_FIELDS})
                    )
                except TypeError:
                    continue  # a line of another version of the ledger: its check runs again
        return cls(path=path, verdicts=verdicts)

    def record(self, check: Check, result: Any, *, tree: str, round: int = 0) -> Verdict:
        """Append one gate result for ``check``, measured on ``tree``.

        A result that says the infrastructure ended it measured nothing: it is
        recorded as not passed, never as a pass.
        """
        infrastructure = str(getattr(result, "infrastructure", "") or "")
        verdict = Verdict(
            deployment=check.deployment,
            gate=check.gate,
            passed=bool(result.passed) and not infrastructure,
            detail=str(result.detail)[:500],
            tree=tree,
            round=round,
            duration_s=round_seconds(getattr(result, "duration_s", 0.0)),
            measured=_number(getattr(result, "measured", None)),
            reference=_number(getattr(result, "reference", None)),
            se=_number(getattr(result, "se", None)),
            summary=str(result.summary())[:1000] if hasattr(result, "summary") else "",
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

    def on(self, tree: str, round: int | None = None) -> dict[str, Verdict]:
        """The latest verdict per check key measured on ``tree`` (in ``round``, when given).

        The latest outcome counts: a failure recorded after a pass replaces it.
        """
        last: dict[str, Verdict] = {}
        for v in self.verdicts:
            if tree and v.tree == tree and (round is None or v.round == round):
                last[v.key] = v
        return last


def round_seconds(value: Any) -> float:
    number = _number(value)
    return round(number, 3) if number is not None else 0.0


@dataclass(frozen=True)
class Evaluation:
    passed: tuple[Check, ...]
    failed: tuple[tuple[Check, Verdict], ...]
    missing: tuple[Check, ...]

    @property
    def holds(self) -> bool:
        return not self.failed and not self.missing

    def render(self) -> str:
        lines = [
            f"{len(self.passed)} passed, {len(self.failed)} failed, {len(self.missing)} missing"
        ]
        lines += [f"- FAILED {c.key}: {v.detail}" for c, v in self.failed]
        lines += [f"- MISSING {c.key}" for c in self.missing]
        return "\n".join(lines)


def evaluate(contract: Contract, verdicts: Mapping[str, Verdict]) -> Evaluation:
    """Each required check against its verdict in ``verdicts`` (:meth:`ContractStatus.on` a tree).

    A check with no verdict there is missing. Nothing is shared between checks
    and nothing is carried over from another tree.
    """
    passed, failed, missing = [], [], []
    for check in contract.required():
        verdict = verdicts.get(check.key)
        if verdict is None:
            missing.append(check)
        elif verdict.passed:
            passed.append(check)
        else:
            failed.append((check, verdict))
    return Evaluation(tuple(passed), tuple(failed), tuple(missing))


__all__ = [
    "CONTRACT_FILENAME",
    "STATUS_FILENAME",
    "Check",
    "Contract",
    "ContractStatus",
    "Evaluation",
    "Verdict",
    "compile_contract",
    "evaluate",
    "flow_version",
    "inputs_digest",
    "library_digest",
    "native_libraries",
    "tree_id",
    "write_contract",
]
