# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract: every check a run must pass, compiled from the resolved task spec.

The contract is the run's definition of "done". It names each check by unit,
gate and member: a unit is a deployment, by its name, and its member is the
deployment's shape. A deployment owes one check, its session
(``checks/session.py``, gate ``deployment``), and one more for the native
tests it names (gate ``tests``).

Results are appended to ``contract_status.jsonl`` as checks return: the ledger.
Each row is evidence of one check on one **key**: what the check measured, by
content (``run/keys.py``; the field is spelled ``marker``). A row is the check's
result wherever the key is the same, whatever the commit is called, and nowhere
else: exact equality only. The latest row under a key is that key's result. A
row never starts work: nothing is measured again because a row went out of
date. The final pass asks, once, for each requested check's row under the
frozen candidate's key (:func:`evaluate`, :func:`judged`).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ...checks import native_tests
from ...checks import session as session_mod
from ...spec import deployments as deployments_mod
from ..startup import datasets as datasets_mod

CONTRACT_FILENAME = "contract.yaml"
STATUS_FILENAME = "contract_status.jsonl"


@dataclass(frozen=True)
class Check:
    """One check: a gate, at one member, for one unit."""

    unit: str
    gate: str
    member: str
    required: bool = True
    #: What the check reads from outside the checkout, as a digest; see :func:`check_inputs`.
    inputs: str = field(default="", compare=False)
    #: The check whose result on the same candidate this one is judged against: the
    #: session of the deployment it is relative to. Its task waits for that one's.
    after: str = field(default="", compare=False)

    @property
    def key(self) -> str:
        return f"{self.unit}:{self.gate}@{self.member}"


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    #: Why a unit or variant owes less than every check at every member, one line each.
    notes: tuple[str, ...] = ()

    def required(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.required)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "checks": [
                {k: v for k, v in asdict(c).items() if k not in ("inputs", "after") or v}
                for c in self.checks
            ]
        }
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    def render(self) -> str:
        """A markdown table, one row per check."""
        lines = ["| unit | gate | member | required |", "|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c.unit} | {c.gate} | {c.member} | {'yes' if c.required else 'no'} |")
        if self.notes:
            lines += ["", *(f"- {note}" for note in self.notes)]
        return "\n".join(lines)


#: What of the spec's ``environment`` a check runs in: the interpreter, the image and how
#: host paths appear in it, and the node's shape. The rest of that block says where and
#: for how long the scheduler runs things, which changes no result.
RUNTIME_KEYS: tuple[str, ...] = ("python", "container_image", "mounts", "gpus_per_node")
#: The same of the ``slurm`` block: a Slurm job takes its image and node shape there. Not
#: its mounts: a mount added in the middle of a run (a dataset's path) changes no result,
#: and would have every deployment loaded again. What a mounted path holds is an input by
#: its own content where the spec names it.
SLURM_RUNTIME_KEYS: tuple[str, ...] = ("container_image", "gpus_per_node")

#: File digests by ``(path, size, mtime)``: an input is read once per change.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


#: A file up to this size is identified by its bytes; a larger one (an image, a weight
#: file) by its size and, when it is a link, by what it links to.
CONTENT_MAX_BYTES = 64 << 20
#: In a directory: the files read by content (configuration, tokenizer, processor, chat
#: template), each up to this size and all of them up to the total.
MANIFEST_FILE_BYTES = 32 << 20
MANIFEST_TOTAL_BYTES = 256 << 20


def _manifest(directory: Path) -> str:
    """One digest of a directory by content: a checkpoint, a dataset.

    Every file by its place and size; a link also by its target (a model hub's snapshot
    links each file to an object named for its content); and the smaller files, in
    order, by their bytes, which covers a checkpoint's configuration, tokenizer and
    processor. Weight files are not read: hashing them would cost more than the load.
    """
    digest = hashlib.sha1()
    read = 0
    for folder, names, files in os.walk(directory):
        names.sort()
        for name in sorted(files):
            path = Path(folder) / name
            stat = path.stat()
            entry = [path.relative_to(directory).as_posix(), stat.st_size]
            if path.is_symlink():
                entry.append(os.readlink(path))
            if stat.st_size <= MANIFEST_FILE_BYTES and read + stat.st_size <= MANIFEST_TOTAL_BYTES:
                read += stat.st_size
                entry.append(_file_digest(path))
            digest.update(json.dumps(entry).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _with_contents(value: Any, checkout: Path | None, directories: bool = True) -> Any:
    """``value`` with every path outside ``checkout`` paired with the identity of what is there.

    A file by its bytes, or by its size past :data:`CONTENT_MAX_BYTES`; a directory by
    its manifest (:func:`_manifest`). A file in the checkout is part of the key's source,
    its anchor tables by the entries the gates read; repeating it here would make an
    edit to another model's entry stale a result the source keeps.

    Where the content is all read, the place is left out: a small file is its bytes
    wherever it is kept, and a path in the checkout is its place in the checkout
    wherever the checkout is. A large file and a directory keep their path: their
    bytes are not all read, so two of them at two places are not known to be the same.

    Without ``directories`` a directory is its name alone (:func:`_environment`).
    """
    if isinstance(value, Mapping):
        return {str(k): _with_contents(v, checkout, directories) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_with_contents(v, checkout, directories) for v in value]
    if not isinstance(value, str) or not value or "\n" in value or len(value) > 4096:
        return value
    path = Path(value)
    try:
        if not path.is_absolute():
            return value
        resolved = path.resolve()
        if checkout is not None and resolved.is_relative_to(checkout):
            return {"in_checkout": resolved.relative_to(checkout).as_posix()}
        if resolved.is_file():
            size = resolved.stat().st_size
            if size <= CONTENT_MAX_BYTES:
                return {"sha1": _file_digest(resolved)}
            link = os.readlink(path) if path.is_symlink() else ""
            return {"path": value, "size": size, "link": link}
        if resolved.is_dir() and directories:
            return {"path": value, "manifest": _manifest(resolved)}
        return value
    except (OSError, ValueError):
        return value


def _environment(stated: Any, checkout: Path | None) -> Any:
    """The run's environment variables as a check's inputs have them.

    Each by its value; a file one names by what it holds. A directory one names is
    its name and no more: such a directory is where things are written while the run
    goes (its scratch, a cache of downloads, the run's own cache of compiled kernels
    under its workspace), so what it holds changes while the tree, the options and
    what a check is judged by stay as they are. A whole run measured both of its
    deployments and its final tests again on one candidate because its sessions had
    filled that cache. A directory a check does read is an input where the spec names
    it as one (the checkpoint, a dataset, an options file), and a benchmark's dataset
    under the models root is one by its own directory (:func:`check_inputs`).
    """
    return _with_contents(stated, checkout, directories=False)


def runtime_of(environment: Any) -> dict[str, Any]:
    """What of the spec's ``environment`` a check runs in, wherever the block states it.

    The block's own values (:data:`RUNTIME_KEYS`), and those its ``slurm`` block states
    (:data:`SLURM_RUNTIME_KEYS`), which a Slurm job reads before the block's own.
    """
    if not isinstance(environment, Mapping):
        return {}
    found = {key: environment[key] for key in RUNTIME_KEYS if environment.get(key) is not None}
    slurm = environment.get("slurm")
    if isinstance(slurm, Mapping):
        stated = {key: slurm[key] for key in SLURM_RUNTIME_KEYS if slurm.get(key) is not None}
        if stated:
            found["slurm"] = stated
    return found


def runtime_inputs(task_data: Mapping[str, Any]) -> str:
    """The run's runtime as one digest: the inputs of a check that reads nothing else of the spec.

    The interpreter, the image and mounts, the node's shape and the environment
    variables: what the final tests run in.
    """
    environment = task_data.get("environment") or {}
    repo = str(task_data.get("trtllm_repo_path") or "")
    try:
        checkout = Path(repo).resolve() if repo else None
    except OSError:
        checkout = None
    stated = environment.get("env") if isinstance(environment, Mapping) else None
    payload = _with_contents({"runtime": runtime_of(environment)}, checkout)
    payload["env"] = _environment(stated, checkout)
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _for_unit(value: Any, unit: str, others: set[str]) -> list[Any]:
    """``value`` as it applies to ``unit``: its own entry, else the value less other units' entries."""
    if isinstance(value, Mapping) and unit in value:
        return ["unit", value[unit]]
    if isinstance(value, Mapping):
        return ["run", {k: v for k, v in value.items() if k not in others}]
    return ["run", value]


def check_inputs(
    task_data: Mapping[str, Any], unit: str, gate: str, member: str, features: Sequence[str] = ()
) -> str:
    """What a check reads from outside the checkout, as one digest.

    The deployment as the spec resolves it (its options by content, its
    environment, inputs, probes and benchmarks, not how long it may take; the
    native tests it names are read by their own check alone: a session runs none
    of them), what the ``accuracy`` block says to it (the references among it), the
    checkpoint with its tokenizer and processor, and the run's runtime: its
    interpreter, its image and mounts, its environment variables. Every file or
    directory outside the checkout the spec names is taken by content
    (:func:`_with_contents`), except a directory an environment variable names, which
    is its name (:func:`_environment`). Nothing of the scheduler is in it: no allocation,
    account, queue or deadline. Conservative: a change anywhere in what applies
    to the deployment is another digest; another deployment's entry alone is not.
    A path that cannot be read where this is computed keeps its name as its identity.
    """
    accuracy = task_data.get("accuracy") or {}
    repo = str(task_data.get("trtllm_repo_path") or "")
    try:
        checkout = Path(repo).resolve() if repo else None
    except OSError:
        checkout = None
    found = deployments_mod.resolve(task_data)
    others = {d.name for d in found} - {unit}
    entries = {key: _for_unit(accuracy[key], unit, others) for key in sorted(accuracy)}
    # A block holding other deployments' entries alone says nothing to this one.
    entries = {k: v for k, v in entries.items() if v != ["run", {}]}
    environment = task_data.get("environment") or {}
    deployment = next((d for d in found if d.name == unit), None)
    stated = deployment.as_dict() if deployment is not None else {}
    stated.pop("timeout", None)
    if gate == session_mod.GATE_NAME:
        stated.pop("tests", None)
    for role, server in (deployment.servers if deployment is not None else {}).items():
        # By content, wherever the options are kept.
        key = "options" if role == deployments_mod.SERVER else role
        stated[key] = dict(server.options) | {"instances": server.instances}
    runtime = environment if isinstance(environment, Mapping) else {}
    payload = {
        "gate": gate,
        "member": member,
        "checkpoint": task_data.get("checkpoint_path"),
        "deployment": stated,
        "accuracy": entries,
        "runtime": runtime_of(runtime),
    }
    if deployment is not None and not accuracy.get("dataset_path"):
        # A dataset the evaluator finds under the models root is named by nothing in the
        # spec. Its own directory is an input here, by content; the rest of that root is
        # not (:func:`_environment`).
        found = {
            b.name: str(path)
            for b in deployment.benchmarks
            if (path := datasets_mod.under_models_root(task_data, b.name.lower())) is not None
        }
        if found:
            payload["datasets"] = found
    payload = _with_contents(payload, checkout)
    payload["env"] = _environment(runtime.get("env"), checkout)
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def compile_contract(task_data: Mapping[str, Any], world_size: int = 1) -> Contract:
    """Every check the run's deployments owe, in the order they run.

    One session per deployment (``checks/session.py``): the first deployment's
    is what ``onboarded`` means, every further one's proves that deployment. A
    deployment judged against another's
    own score (``relative_to``) needs that one's session first
    (:attr:`Check.after`). A deployment that names native tests owes them as
    one more check. Every check carries its :func:`check_inputs`; none shares
    a verdict with another.
    """
    checks: list[Check] = []
    notes: list[str] = []
    found = deployments_mod.resolve(task_data)
    by_name = {d.name: d for d in found}
    for deployment in found:
        name, member = deployment.name, deployment.shape
        base = by_name.get(deployment.relative_to or "")
        after = (
            f"{base.name}:{session_mod.GATE_NAME}@{base.shape}"
            if base is not None and deployment.benchmarks
            else ""
        )
        checks.append(
            Check(
                name,
                session_mod.GATE_NAME,
                member,
                inputs=check_inputs(task_data, name, session_mod.GATE_NAME, member),
                after=after,
            )
        )
        instances = len(list(deployment.instances()))
        notes.append(
            f"{name}: one session of {instances} model load(s) ({deployment.kind}, {member})"
            + (f", after {base.name}'s: its anchor may be that one's own score" if after else "")
        )
        if deployment.tests:
            checks.append(
                Check(
                    name,
                    native_tests.GATE_NAME,
                    member,
                    inputs=check_inputs(task_data, name, native_tests.GATE_NAME, member),
                )
            )
            notes.append(f"{name}: {len(deployment.tests)} native test(s) it names")
    return Contract(checks=tuple(checks), notes=tuple(notes))


def write_contract(workspace: Path, contract: Contract) -> Path:
    path = Path(workspace) / CONTRACT_FILENAME
    path.write_text(yaml.safe_dump(contract.as_dict(), sort_keys=False), encoding="utf-8")
    return path


@dataclass
class Verdict:
    unit: str
    gate: str
    member: str
    passed: bool
    detail: str = ""
    tree_state: str = ""
    #: The row's key: what the check measured, by content (``run/keys.py``). ``""`` is
    #: "not established": such a row is evidence for no other state.
    marker: str = ""
    #: The commit each command of the check imported the package under test
    #: from (``GateResult.imported``): one per leg or engine job that said.
    imported: list[str] = field(default_factory=list)
    #: Why the verdict says nothing about one tree (``GateResult.void``); such a
    #: verdict never passed.
    void: str = ""
    #: The check's :attr:`Check.inputs` when it ran.
    inputs: str = ""
    #: How long the check's gate ran, in seconds; 0 when it did not say.
    duration_s: float = 0.0
    #: The score a benchmark measured (``GateResult.measured``), pass or fail.
    measured: float | None = None
    #: Why what the check is for could not be observed (``GateResult.unverified``): it
    #: did not fail, and it certifies nothing (:attr:`Evaluation.unverified`).
    unverified: str = ""
    #: What the check's result holds beyond its line, as data: the final tests' lists. A
    #: row is the one record of a result, so what a later reader needs of it is in the row.
    data: dict[str, Any] = field(default_factory=dict)


#: Fields written only when they say something.
_UNSAID: dict[str, Any] = {
    "data": {},
    "imported": [],
    "void": "",
    "inputs": "",
    "duration_s": 0.0,
    "measured": None,
    "unverified": "",
}
_VERDICT_FIELDS = frozenset(f.name for f in fields(Verdict))


def _score(value: Any) -> float | None:
    """A benchmark score a gate reported; ``None`` for one it did not."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _seconds(value: Any) -> float:
    """A duration a gate reported, as seconds; 0 for one it did not."""
    try:
        return round(float(value or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


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
                if line.strip():
                    data = json.loads(line)
                    verdicts.append(
                        Verdict(**{k: v for k, v in data.items() if k in _VERDICT_FIELDS})
                    )
        return cls(path=path, verdicts=verdicts)

    def record(
        self,
        unit: str,
        result: Any,
        marker: str = "",
        inputs: str = "",
    ) -> Verdict:
        """Append one gate result, as returned by any gate, for ``unit``.

        ``marker`` is the check's key on the target as the check was started; a
        result measured on a pin names the key of that pin (``GateResult.marker``),
        and is recorded under that. ``inputs`` is the check's (:attr:`Check.inputs`).
        """
        void = str(getattr(result, "void", "") or "")
        passed = bool(result.passed) and not void
        verdict = Verdict(
            unit=unit,
            gate=str(result.name),
            member=str(result.member or ""),
            passed=passed,
            detail=str(result.detail)[:500],
            tree_state=str(getattr(result, "tree_state", "") or ""),
            marker=str(getattr(result, "marker", "") or "") or marker,
            imported=[str(i) for i in getattr(result, "imported", ()) or ()],
            void=void,
            inputs=inputs,
            duration_s=_seconds(getattr(result, "duration_s", 0.0)),
            measured=_score(getattr(result, "measured", None)),
            # Only a check that failed nothing is "not verified"; one that failed, failed.
            unverified=str(getattr(result, "unverified", "") or "")[:500] if passed else "",
            data=dict(getattr(result, "data", None) or {}),
        )
        self.verdicts.append(verdict)
        line = {k: v for k, v in asdict(verdict).items() if k not in _UNSAID or v != _UNSAID[k]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        return verdict

    def latest(self) -> dict[str, Verdict]:
        """The last verdict per check key."""
        last: dict[str, Verdict] = {}
        for v in self.verdicts:
            last[f"{v.unit}:{v.gate}@{v.member}"] = v
        return last


@dataclass(frozen=True)
class Evaluation:
    passed: tuple[Check, ...]
    failed: tuple[tuple[Check, Verdict], ...]
    missing: tuple[Check, ...]
    #: Checks whose row failed nothing and could not observe what it is for
    #: (:attr:`Verdict.unverified`): the third outcome. Such a check is not passed (it
    #: certifies nothing), not failed (there is nothing to repair in the target) and not
    #: missing (running it again on the same key says the same).
    unverified: tuple[tuple[Check, Verdict], ...] = ()

    @property
    def holds(self) -> bool:
        return not self.failed and not self.missing and not self.unverified

    @property
    def partial(self) -> bool:
        """Nothing failed and nothing is missing, and something could not be verified."""
        return bool(self.unverified) and not self.failed and not self.missing

    def render(self) -> str:
        lines = [
            f"{len(self.passed)} passed, {len(self.failed)} failed, {len(self.missing)} missing"
            + (f", {len(self.unverified)} not verified" if self.unverified else "")
        ]
        lines += [f"- {'VOID' if v.void else 'FAILED'} {c.key}: {v.detail}" for c, v in self.failed]
        lines += [f"- MISSING {c.key}" for c in self.missing]
        lines += [
            f"- NOT VERIFIED {c.key}: {v.unverified} (not a failure: nothing in the target "
            "repairs it, and it does not certify)"
            for c, v in self.unverified
        ]
        return "\n".join(lines)


def row(verdicts: Sequence[Verdict], check: Check, key: str | None) -> Verdict | None:
    """The latest row of ``check`` recorded under ``key``; ``None`` when it has none.

    Exact equality: a row under another key is about other content, and is not this
    check's result here however close the two are. ``None`` asks for the latest row
    under any key. A row recorded with other :attr:`Verdict.inputs` than the check's was
    judged by other requirements and is passed over (one without inputs counts as the
    check's).
    """
    name = check.key
    for verdict in reversed(verdicts):
        if f"{verdict.unit}:{verdict.gate}@{verdict.member}" != name:
            continue
        if key is not None and verdict.marker != key:
            continue
        if verdict.inputs and check.inputs and verdict.inputs != check.inputs:
            continue
        return verdict
    return None


def judged(contract: Contract, rows: Mapping[str, Verdict | None]) -> Evaluation:
    """Each required check of ``rows`` (by the check's name) against the row given for it.

    ``None`` for a check is "it has no row": missing. A required check that ``rows``
    does not name is left out: nothing is said of it.
    """
    passed, failed, missing, unverified = [], [], [], []
    for check in contract.required():
        if check.key not in rows:
            continue
        verdict = rows[check.key]
        if verdict is None:
            missing.append(check)
        elif verdict.passed and verdict.unverified:
            unverified.append((check, verdict))
        elif verdict.passed:
            passed.append(check)
        else:
            failed.append((check, verdict))
    return Evaluation(tuple(passed), tuple(failed), tuple(missing), tuple(unverified))


def evaluate(
    contract: Contract,
    verdicts: Iterable[Verdict] | ContractStatus,
    marker: str | None = None,
    *,
    keys: Mapping[str, str] | None = None,
) -> Evaluation:
    """Each required check against the latest row under its key.

    ``keys`` gives each check's key, by the check's name: what it measures on the
    candidate, by content. A check with no row under exactly that key is missing: it
    was measured on other content, or not at all. ``marker`` is one key for every check
    (a caller that has one identity for the whole target); with neither, each check is
    read by its latest row whatever its key.
    """
    rows = verdicts.verdicts if isinstance(verdicts, ContractStatus) else list(verdicts)
    found = {
        check.key: row(rows, check, keys.get(check.key, "") if keys is not None else marker)
        for check in contract.required()
    }
    return judged(contract, found)


def verdict_on(status: ContractStatus, check: Check, key: str) -> Verdict | None:
    """``check``'s latest row under ``key``, as :func:`evaluate` reads it."""
    return row(status.verdicts, check, key)


def measured_commit(status: ContractStatus, keyed: Iterable[tuple[Check, str]]) -> str | None:
    """The one commit every check of ``keyed`` imported when it passed under its key; ``None``.

    ``keyed`` pairs each check with its key. ``None`` when one of them has no passing row
    there, names no commit, or two name different ones: a record names a commit only
    where its checks say which.
    """
    commits: set[str] = set()
    for check, key in keyed:
        verdict = row(status.verdicts, check, key)
        if verdict is None or not verdict.passed or not verdict.imported:
            return None
        commits.update(verdict.imported)
    return commits.pop() if len(commits) == 1 else None
