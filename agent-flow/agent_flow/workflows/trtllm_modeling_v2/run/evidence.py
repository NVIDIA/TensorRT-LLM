# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence: what a card's check measured, kept in the run's ledger under the tree it measured.

A task is done on evidence, not on its worker's word. The check a card names
(``kernel:<row>``, ``cpu:<row>``, ``deployment:<name>``) is the gate's own
(``checks/gate_cli.py``); what it measured goes into the run's ledger
(``contract_status.jsonl``) under the **key** of the check on the pin it measured: what
that check consumes there, by content (``run/keys.py``), so the same content is the same
evidence whatever its commit is called. Two things write it:

- ``trtllm-modeling-v2 gate ...`` run with a run's own ``task.yaml`` (:func:`main`): a
  worker's own run of its card's check records itself. The tree it names is pinned first
  and the gate measures exactly that pin.
- the coordinator, which runs the same command once itself (:func:`run_gate_process`) for a
  task that says ``ready`` with no result for its pin.

Before a task lands, :func:`lookup` says whether its check has a result for the pin of
its tree. A session's lines are recorded as the run's own sessions record theirs (the
deployment's name, each check's line, the contract check's inputs), so a session a worker
ran on the tree that then lands unchanged is not measured a second time.

A requirement that changes takes the passes judged by the old one with it
(:func:`void_passes`): the ledger gets a line that says why, and the check is measured
again. Nothing is deleted from the ledger.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

from ..checks import gate_cli
from ..checks import verdict as check_result
from ..infra.compute import pintree
from ..infra.git import resolve_commit
from ..spec.task_schema import TaskSchemaError, load_and_validate_task_yaml
from . import graph as graph_mod
from . import keys as keys_mod
from .records.state import STATE_FILENAME
from .validation import contract as contract_mod

#: The checks a card names that the gate's front door runs: each leaves evidence.
GATED: tuple[str, ...] = tuple(gate_cli.GATE_NAMES)
#: The gate's name of a deployment's session, and of its own line.
SESSION = "deployment"
#: The ledger's ``unit`` of a row's check: a row is no deployment.
ROW_UNIT = "row"
_FIELDS = frozenset(f.name for f in dataclasses.fields(check_result.GateResult))


def gated(check_id: str) -> bool:
    """Whether ``check_id`` is one the gate's front door runs, so one that leaves evidence."""
    return check_id.partition(":")[0] in GATED


def inputs_of(workspace: str | Path, task_data: Mapping[str, Any], check_id: str) -> str | None:
    """The inputs digest of the contract's check ``check_id`` names; ``None`` when it has none.

    As the run's controller compiled and wrote them (``contract.yaml``), so that every
    process of a run keys a check the same way; before the run has written its contract,
    as the spec compiles now.
    """
    name, _, ref = check_id.partition(":")
    written = _contract_inputs(workspace, ref)
    found = next((v for (gate, _), v in written.items() if gate == name and v), None)
    if found is not None:
        return found
    try:
        contract = contract_mod.compile_contract(task_data)
    except Exception:  # a spec no contract compiles from says nothing of its inputs
        return None
    return next((c.inputs for c in contract.checks if c.unit == ref and c.gate == name), None)


def key_of(task_data: Mapping[str, Any], workspace: str | Path, commit: str, check_id: str) -> str:
    """The key ``check_id``'s result on ``commit`` is recorded under; ``""`` when not established."""
    return keys_mod.key(
        task_data,
        workspace,
        commit,
        check_id,
        inputs_of=lambda name: inputs_of(workspace, task_data, name),
    )


def verdicts(workspace: str | Path) -> list[contract_mod.Verdict]:
    """Every line of the run's ledger, oldest first; a line that cannot be read is passed over.

    Another process may be appending while this one reads: a half-written last line is
    not the ledger's, yet.
    """
    path = Path(workspace) / contract_mod.STATUS_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    known = {f.name for f in dataclasses.fields(contract_mod.Verdict)}
    found = []
    for line in text.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and {"unit", "gate", "passed"} <= set(data):
            data.setdefault("member", "")
            found.append(contract_mod.Verdict(**{k: v for k, v in data.items() if k in known}))
    return found


def _is(verdict: contract_mod.Verdict, check_id: str) -> bool:
    name, _, what = check_id.partition(":")
    if name == SESSION:
        return verdict.unit == what and verdict.gate == SESSION
    return verdict.gate == check_id


def lookup(
    workspace: str | Path, check_id: str, marker: str, *, inputs: str | None = ""
) -> contract_mod.Verdict | None:
    """The latest result of ``check_id`` recorded under the key ``marker``; ``None`` when there is none.

    An empty key is "not established": nothing is evidence for it. ``inputs`` are what
    the check is judged by now (a session's: the contract's digest of the deployment's
    options and requirements). A result recorded with other inputs was judged by other
    requirements and is not this check's result; with ``inputs`` unknown (``None``) a
    result that names its inputs cannot be matched and does not count. ``""`` asks for no
    match (a row's check has none).
    """
    if not marker:
        return None
    for verdict in reversed(verdicts(workspace)):
        if verdict.marker != marker or not _is(verdict, check_id):
            continue
        if inputs != "" and verdict.inputs != inputs:
            continue
        return verdict
    return None


def _contract_inputs(workspace: str | Path, unit: str) -> dict[tuple[str, str], str]:
    """The inputs of each contract check of ``unit``, by its gate and member.

    As the run's controller compiled and wrote them (``contract.yaml``): a line recorded
    with them is that check's own on the same inputs, however this process read the spec.
    Empty before the run has written its contract.
    """
    try:
        data = yaml.safe_load(
            (Path(workspace) / contract_mod.CONTRACT_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        return {}
    checks = data.get("checks") if isinstance(data, Mapping) else None
    return {
        (str(c.get("gate")), str(c.get("member") or "")): str(c.get("inputs") or "")
        for c in checks or ()
        if isinstance(c, Mapping) and c.get("unit") == unit
    }


def record(
    workspace: str | Path,
    task_data: Mapping[str, Any],
    check_id: str,
    lines: Sequence[Mapping[str, Any]],
    marker: str,
) -> list[contract_mod.Verdict]:
    """Write a gate's lines to the run's ledger under ``marker``; the verdicts written.

    A session's lines go under its deployment, with the inputs of the contract's check of
    that session, as the run's own sessions are recorded. A row's check is one line.
    """
    name, _, what = check_id.partition(":")
    unit = what if name == SESSION else ROW_UNIT
    inputs = _contract_inputs(workspace, unit) if name == SESSION else {}
    status = contract_mod.ContractStatus(Path(workspace) / contract_mod.STATUS_FILENAME)
    written = []
    for line in lines:
        result = SimpleNamespace(**{"member": "", "detail": "", **dict(line)})
        # Every line of a session is recorded with the session check's inputs.
        session_inputs = next((v for (gate, _), v in inputs.items() if gate == SESSION), "")
        owned = inputs.get((str(result.name), str(result.member or "")), session_inputs)
        written.append(status.record(unit, result, marker=marker, inputs=owned))
    return written


def void_passes(status: contract_mod.ContractStatus, keys: Sequence[str], why: str) -> list[str]:
    """Take the standing pass of each check of ``keys`` away, saying ``why``; the keys voided.

    For a requirement that changed: a pass judged by the old one is not a pass by the new.
    The ledger gets one more line per check, void, under the marker the pass was
    recorded on; nothing is removed, and the check is measured again when it is next asked.
    """
    latest = status.latest()
    voided = []
    for key in keys:
        held = latest.get(key)
        if held is None or not held.passed:
            continue
        result = SimpleNamespace(
            name=held.gate,
            member=held.member,
            passed=False,
            detail=why,
            void=why,
            tree_state=held.tree_state,
            imported=tuple(held.imported),
        )
        status.record(held.unit, result, marker=held.marker, inputs=held.inputs)
        voided.append(key)
    return voided


# ----------------------------------------------------------------- the front door


def _said(argv: Sequence[str]) -> argparse.Namespace:
    """What of a gate's command line the recorder needs; the gate itself parses the rest."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--task", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--json", action="store_true")
    return parser.parse_known_args(list(argv))[0]


def _measuring(argv: Sequence[str], commit: str) -> list[str]:
    """``argv`` with its ``--tree`` replaced by ``--commit <commit>``: the gate measures that pin."""
    out: list[str] = []
    skip = False
    for word in argv:
        if skip:
            skip = False
            continue
        if word == "--tree":
            skip = True
            continue
        if word.startswith("--tree=") or word == "--json":
            continue
        out.append(word)
    return [*out, "--commit", commit, "--json"]


def is_run(workspace: Path) -> bool:
    """Whether ``workspace`` is a run's: it holds the run's graph or its state."""
    return (workspace / graph_mod.GRAPH_FILENAME).is_file() or (
        workspace / STATE_FILENAME
    ).is_file()


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 gate ...``: the gate's front door, and its result in the run's ledger.

    Run with a run's own ``task.yaml`` on a tree of the run's checkout, the gate measures
    a pin of that tree and what it measured is recorded under the pin's marker: a worker's
    own run of its card's check is the evidence its task is accepted on. Anything else (a
    spec outside a run, a directory nothing pins, ``--help``) is the gate as it is,
    unrecorded. The output and the exit status are the gate's.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    said = _said(argv)
    pinned = None
    if (
        not {"-h", "--help"} & set(argv)
        and said.name in GATED
        and said.task is not None
        and is_run(said.task.resolve().parent)
    ):
        pinned = _pin(said)
    if pinned is None:
        return gate_cli.main(argv)
    workspace, task_data, commit = pinned
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        status = gate_cli.main(_measuring(argv, commit))
    text = captured.getvalue()
    try:
        payload = json.loads(text)
        lines = [dict(line) for line in payload["lines"]]
        check_id = str(payload["id"])
    except (ValueError, KeyError, TypeError):
        # A usage or spec error: the gate said why on standard error, and measured nothing.
        sys.stdout.write(text)
        return status
    marker = key_of(task_data, workspace, commit, check_id)
    if marker:
        record(workspace, task_data, check_id, lines, marker)
        outcome = str(payload.get("status") or ("pass" if payload.get("passed") else "fail"))
        said_how = {"pass": "passed", "partial": "not verified: nothing failed"}
        print(
            f"recorded in the run's ledger: {check_id} on the pin {commit[:10]} "
            f"({said_how.get(outcome, 'not passed')})",
            file=sys.stderr,
        )
    else:
        print(
            f"not recorded: `{check_id}` has no key on the pin {commit[:10]} (git could not "
            "list it, or an input of the check has no content identity)",
            file=sys.stderr,
        )
    if said.json:
        sys.stdout.write(text)
        return status
    print(f"check {check_id}")
    shown = [
        check_result.GateResult(**{k: v for k, v in line.items() if k in _FIELDS}) for line in lines
    ]
    for line in shown:
        print(line.summary())
    if shown and not shown[-1].passed and shown[-1].stdout_tail.strip():
        print(shown[-1].stdout_tail)
    if payload.get("status") == gate_cli.PARTIAL:
        # The gate's own closing line for its third outcome.
        print(
            f"partial (exit {gate_cli.EXIT_STATUS[gate_cli.PARTIAL]}): nothing failed, and this "
            f"does not certify: {payload.get('unverified')}"
        )
    return status


def _pin(said: argparse.Namespace) -> tuple[Path, dict[str, Any], str] | None:
    """The run's workspace, its spec and the commit the gate is to measure; ``None`` to leave it be."""
    workspace = said.task.resolve().parent
    try:
        task_data = load_and_validate_task_yaml(said.task)
    except (TaskSchemaError, OSError):
        return None  # the gate says what is wrong with the spec
    repo = Path(str(task_data.get("trtllm_repo_path") or ""))
    if not (repo / ".git").exists():
        return None
    if said.tree is not None:
        try:
            commit = pintree.freeze(said.tree, f"gate-{said.name}")
        except (pintree.PinError, OSError):
            return None
    else:
        commit = said.commit or "HEAD"
    commit = resolve_commit(repo, commit) if commit else None
    return (workspace, task_data, commit) if commit else None


def run_gate_process(argv: Sequence[str], cli: Sequence[str] = ()) -> tuple[int, str]:
    """Run a card's check as the worker would: the same command, in a process of its own.

    ``argv`` is what follows ``gate``. Its exit status, and the end of what it said. The
    process records its own result (:func:`main`), so the caller reads the ledger after.
    """
    command = [
        *(cli or (sys.executable, "-m", "agent_flow.workflows.trtllm_modeling_v2.cli")),
        "gate",
        *argv,
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    tail = "\n".join((done.stdout + "\n" + done.stderr).strip().splitlines()[-12:])
    return done.returncode, tail


__all__ = [
    "GATED",
    "ROW_UNIT",
    "SESSION",
    "gated",
    "inputs_of",
    "is_run",
    "lookup",
    "main",
    "key_of",
    "record",
    "run_gate_process",
    "verdicts",
    "void_passes",
]
