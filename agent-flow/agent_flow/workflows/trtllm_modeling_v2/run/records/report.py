# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The closing report: what a run delivered, under which requirements, and what it did not do.

``<workspace>/delivery_report.md``, written when a run closes and when it stops with
everything left blocked. It is read off the run's own records and says nothing they do
not: whether ``onboarded`` was pinned and on which commit; the requirements the run was
judged by (its frozen anchors) and every one that was changed during it, from what, to
what and why; what the final pass found on its frozen candidate (each requested check:
passed, failed, not verified or not run, and under which key; the final tests, with each
failing case's result at the base); every task that is not done; and what was not
verified: a check the engine gave no way to observe, and a row that landed with no check
the flow could run. A reader of the first lines knows what the result is worth.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from ...infra.files import write_text_atomic
from .journal import utc_now

REPORT_FILENAME = "delivery_report.md"
ONBOARDED_FILENAME = "onboarded.json"
#: The checks the gate's front door runs, by the first word of a task's check. A task
#: with any other, or with none, left nothing in the ledger: it landed on its receipt.
GATED = ("kernel", "deployment", "cpu")
#: The rows that name no check because the landing itself asks what proves them, with no
#: job: the delivery's rules for the task's files (``rows.DELIVERY_ROW``).
AUDITED = ("delivery-files",)
#: The task kinds whose work lands on the target.
LANDING_KINDS = ("kernel", "weights", "routing", "state", "feature", "assemble")
_UNVERIFIED = "not verified"


def _json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            found.append(data)
    return found


def onboarded(workspace: str | Path) -> dict[str, Any] | None:
    """The write-once record of the first deployment's session passing; ``None`` before it."""
    try:
        data = json.loads((Path(workspace) / ONBOARDED_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def baseline(workspace: str | Path) -> str:
    """The onboarded ref the run's spec names (``onboarded``): an earlier run's milestone."""
    try:
        spec = yaml.safe_load((Path(workspace) / "task.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return ""
    return str(spec.get("onboarded") or "").strip() if isinstance(spec, dict) else ""


def requirements(workspace: str | Path) -> list[str]:
    """What the run's benchmarks are held to now, one line each: its frozen anchors."""
    try:
        rows = yaml.safe_load((Path(workspace) / "anchors.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    lines = []
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, Mapping):
            continue
        value = "" if row.get("value") is None else f" {row['value']:g}"
        source = f" ({row['source']})" if row.get("source") else ""
        lines.append(
            f"`{row.get('deployment')}:{row.get('benchmark')}`: {row.get('kind')}{value}{source}"
        )
    return lines


def _anchor(side: Any) -> str:
    if not isinstance(side, Mapping):
        return "nothing frozen"
    value = "" if side.get("value") is None else f" {side['value']:g}"
    return f"{side.get('kind')}{value}"


def changes(workspace: str | Path) -> list[dict[str, str]]:
    """Every requirement and environment value changed during the run, oldest first.

    An anchor changed through the session step's command (``anchor_changes.jsonl``), and a
    spec key changed by a resolution (``spec_changes.jsonl``): each with what, from, to,
    why and when.
    """
    workspace = Path(workspace)
    found = []
    for row in _json_lines(workspace / "anchor_changes.jsonl"):
        found.append(
            {
                "what": f"anchor `{row.get('anchor')}`",
                "class": "requirement",
                "from": _anchor(row.get("from")),
                "to": _anchor(row.get("to")),
                "why": str(row.get("reason") or ""),
                "at": str(row.get("at") or ""),
            }
        )
    for row in _json_lines(workspace / "spec_changes.jsonl"):
        found.append(
            {
                "what": f"`{row.get('key')}`",
                "class": str(row.get("class") or "requirement"),
                "from": repr(row.get("from")),
                "to": repr(row.get("to")),
                "why": str(row.get("why") or ""),
                "at": str(row.get("at") or ""),
            }
        )
    return sorted(found, key=lambda change: change["at"])


def tasks(workspace: str | Path) -> list[dict[str, Any]]:
    """The run's tasks as its graph holds them; none when it has no graph."""
    try:
        data = yaml.safe_load((Path(workspace) / "graph.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    found = data.get("tasks") if isinstance(data, Mapping) else None
    return [task for task in found or () if isinstance(task, Mapping)]


def _asked_of_no_task(workspace: Path) -> list[str]:
    """What waits for an answer and is on no task: the run's own question."""
    try:
        graph = yaml.safe_load((workspace / "graph.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        graph = None
    asked = graph.get("run_blocked") if isinstance(graph, Mapping) else None
    lines = []
    if isinstance(asked, Mapping) and asked:
        lines.append(
            f"the run (blocked, {asked.get('kind')}): {str(asked.get('observed') or '').strip()}"
        )
    return lines


def delivery_gaps(workspace: str | Path) -> list[str]:
    """What kept a run whose checks hold from ``delivered``: the journal's last word on it.

    The deliver stage writes it under ``delivery blocked`` when it closes the run; one line
    for each thing it named.
    """
    try:
        text = (Path(workspace) / "journal.md").read_text(encoding="utf-8")
    except OSError:
        return []
    _, found, after = text.rpartition(" — delivery blocked\n")
    if not found:
        return []
    section = after.split("\n## ", 1)[0]
    return [line.strip() for line in section.splitlines() if line.strip()]


def not_done(workspace: str | Path) -> list[str]:
    """Everything that is not done, one line each.

    A task that was skipped, is blocked, or never started; and what waits for an answer
    on the run itself (a final pass that did not hold) or on the pool, which no task carries.
    """
    lines = []
    for task in tasks(workspace):
        status = str(task.get("status") or "pending")
        if status == "done":
            continue
        why = str(task.get("last") or "").strip()
        if status == "blocked":
            why = f"{why} (attempt {task.get('attempts', 1)} waits for an answer)".strip()
        if status in ("pending", "running"):
            why = why or "it did not start: what it waits for is not done"
        lines.append(f"`{task.get('id')}` ({status}): {why or 'no reason kept'}")
    return lines + _asked_of_no_task(Path(workspace))


def not_verified(workspace: str | Path) -> list[str]:
    """What the run could not verify, one line each.

    A check whose line passed nothing and failed nothing because what it is for could
    not be observed (the ledger's ``unverified`` where a line carries it, else a detail
    that says so), by its latest line. And a task that landed with a check the gate's
    front door does not run: nothing in the ledger speaks for it but its worker's receipt.
    """
    workspace = Path(workspace)
    latest: dict[str, dict[str, Any]] = {}
    for row in _json_lines(workspace / "contract_status.jsonl"):
        latest[f"{row.get('unit')}:{row.get('gate')}@{row.get('member') or ''}"] = row
    lines = []
    owned: set[str] = set()
    # A task that is done with a check not verified says so itself: the check, and why.
    for task in tasks(workspace):
        if task.get("status") != "done":
            continue
        for said in task.get("unverified") or ():
            key, _, why = str(said).partition(": ")
            owned.add(key)
            lines.append(f"`{task.get('id')}` is done with `{key}` not verified: {why}")
    for key, row in latest.items():
        said = str(row.get("unverified") or "")
        detail = str(row.get("detail") or "")
        if not said and row.get("passed") and detail.lower().startswith(_UNVERIFIED):
            said = detail
        if said and key not in owned:
            lines.append(f"`{key}`: {said}")
    for task in tasks(workspace):
        check = str(task.get("check") or "")
        if (
            task.get("status") == "done"
            and task.get("kind") in LANDING_KINDS
            and check.partition(":")[0] not in GATED
            and str(task.get("ref") or "") not in AUDITED
        ):
            lines.append(
                f"`{task.get('id')}` landed on its worker's receipt: "
                + (f"its check `{check}` is none the flow runs" if check else "it names no check")
            )
    return lines


def _epoch(at: Any) -> float | None:
    """A recorded time as epoch seconds: a number, or an ISO time; ``None`` when it is neither."""
    if isinstance(at, (int, float)):
        return float(at)
    try:
        when = datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).timestamp()


def since_onboarded(
    record: Mapping[str, Any] | None, changed: list[dict[str, str]]
) -> list[dict[str, str]] | None:
    """The changes recorded after ``onboarded`` was pinned; ``None`` when that cannot be told.

    ``onboarded`` was judged by the requirements as they were when it was pinned. A change
    made later is listed after it and is never attributed to it. A record written again
    from the ref does not know when it was pinned, and a change with no time cannot be
    placed: neither is guessed.
    """
    pinned = _epoch((record or {}).get("at"))
    if record is None or pinned is None:
        return None
    times = [_epoch(change.get("at")) for change in changed]
    if any(t is None for t in times):
        return None
    return [change for change, t in zip(changed, times) if t > pinned]


def _final_tests(record: Mapping[str, Any]) -> str:
    """The final tests of a final pass, in one line."""
    tests = record.get("tests")
    if not isinstance(tests, Mapping):
        return str(record.get("tests_said") or "not run")
    return str(tests.get("summary") or tests.get("verdict") or "ran")


def _tests_in_ledger(workspace: str | Path) -> str:
    """What the ledger holds of the final tests, where no pass was judged on them.

    A pass that was interrupted may have run them: the report says what is recorded,
    never that they did not run.
    """
    # The gate's name is ``regression.FINAL_TESTS``.
    rows = [
        row
        for row in _json_lines(Path(workspace) / "contract_status.jsonl")
        if row.get("gate") == "final-tests"
    ]
    if not rows:
        return "the ledger holds no result of the final tests"
    said = str(rows[-1].get("detail") or ("passed" if rows[-1].get("passed") else "did not pass"))
    return f"the final tests left a result in the ledger that no pass judged ({said})"


def final_lines(workspace: str | Path, record: Mapping[str, Any] | None) -> list[str]:
    """The final pass in the closing report: every requested check, then the final tests.

    ``record`` is what the loop holds of the pass it judged, or of why none stands; with
    none (this process judged no pass) the report says what the ledger holds.
    """
    if record is None:
        return [
            "No final pass was judged: no requested check is reported on a frozen candidate, "
            f"and {_tests_in_ledger(workspace)}."
        ]
    if not record.get("candidate") and record.get("said"):
        return [f"No final pass stands: {record['said']}."]
    out = [
        f"On the frozen candidate `{record.get('candidate') or '(unnamed)'}`, each requested "
        "check by its row under the candidate's key:",
        "",
        "| check | outcome | key | how |",
        "|---|---|---|---|",
    ]
    for check in record.get("checks") or ():
        said = (
            f": {check['said']}" if check.get("said") and check.get("outcome") != "passed" else ""
        )
        out.append(
            f"| `{check.get('check')}` | {check.get('outcome')} | "
            f"{'`' + str(check['key']) + '`' if check.get('key') else 'none'} | "
            f"{check.get('how')}{said} |".replace("\n", " ")
        )
    out += ["", f"**Final tests**: {_final_tests(record)}."]
    tests = record.get("tests")
    if isinstance(tests, Mapping):
        if tests.get("failures"):
            out += ["", "Fail the final tests (the candidate's result; the base's):", ""]
            out += [f"- {line}" for line in tests["failures"]]
        if tests.get("at_base"):
            out += ["", "Fail at the base too, and fail nothing:", ""]
            out += [f"- {line}" for line in tests["at_base"]]
        if tests.get("left"):
            out += [
                "",
                "Accuracy tests of the engine's harness, collected and not run by the final tests:",
                "",
            ]
            out += [f"- {line}" for line in tests["left"]]
        if tests.get("unverified"):
            out += [
                "",
                "Not verified by the final tests (nothing failed; they do not certify):",
                "",
            ]
            out += [f"- {line}" for line in tests["unverified"]]
    return out


def render(
    workspace: str | Path,
    *,
    verdict: str,
    target: str = "",
    gates: str = "",
    asked: str = "",
    final: Mapping[str, Any] | None = None,
) -> str:
    """The closing report as markdown.

    ``final`` is the final pass as the loop that judged it holds it
    (``GraphOrchestrator.final_record``): this report is its one record on disk.
    ``asked`` is what the run asked of its scheduler so far, against its bound, as the
    pool says it (``pool.Asked.render``); a run with no holder of a scheduler states none.
    """
    record = onboarded(workspace)
    held = requirements(workspace)
    changed = changes(workspace)
    left = not_done(workspace)
    unverified = not_verified(workspace)
    out = [f"# {target or 'modeling_v2 target'}: {verdict}", "", f"Written {utc_now()}.", ""]
    if asked:
        out += [f"Asked of the scheduler: {asked}.", ""]
    if record is not None:
        short = [str(u) for u in record.get("unverified") or ()]
        out.append(
            f"**Onboarded** at `{record.get('commit')}` (`{record.get('ref')}`): the first "
            f"deployment's session (`{record.get('session')}`) "
            + (
                "failed nothing on that commit, and could not verify: " + "; ".join(short) + "."
                if short
                else "passed on that commit."
            )
        )
        if record.get("recovered_at") is not None and record.get("at") is None:
            out.append(
                "Its record was written again from the ref: when it was pinned, and under "
                "which requirements, is not recorded."
            )
        elif isinstance(record.get("requirements"), list):
            then = "; ".join(str(line) for line in record["requirements"])
            out.append(f"Under the requirements in force then: {then or 'no anchor frozen'}.")
    elif baseline(workspace):
        # A run that adds to an onboarded model pins none: the milestone is not its own.
        out.append(
            f"**Onboarded before this run** (`onboarded: {baseline(workspace)}`, the "
            "spec's): `onboarded` is not pinned by this run. The milestone is the run's "
            "that pinned that ref, and its report says what it was onboarded with. This "
            "run measured its own deployments only."
        )
    else:
        out.append("**Not onboarded**: the first deployment's session has not passed.")
    out += ["", "## Requirements the run is judged by", ""]
    out += [f"- {line}" for line in held] or ["- no benchmark is held to an anchor"]
    out += ["", "## Requirements changed during the run", ""]
    if changed:
        out += ["| what | class | from | to | why |", "|---|---|---|---|---|"]
        out += [
            f"| {c['what']} | {c['class']} | {c['from']} | {c['to']} | {c['why']} |"
            for c in changed
        ]
        if record is not None:
            later = since_onboarded(record, changed)
            if later is None:
                said = (
                    "When `onboarded` was pinned against these changes is not recorded: it is "
                    "attributed to none of them."
                )
            elif later:
                said = (
                    f"{len(later)} of these came after `onboarded` was pinned "
                    f"({', '.join(c['what'] for c in later)}): it was judged by the "
                    "requirements as they were before them."
                )
            else:
                said = (
                    "`onboarded` was pinned after all of these: it holds under the "
                    "requirements as changed, not as the spec first stated them."
                )
            out += ["", said]
    else:
        out.append("None: the run was judged by the requirements it started with.")
    out += ["", "## The final pass", "", *final_lines(workspace, final)]
    out += ["", "## Not done", ""]
    out += [f"- {line}" for line in left] or ["Nothing: every task of the run is done."]
    if verdict.startswith("delivery-failed"):
        gaps = delivery_gaps(workspace)
        out += ["", "## What kept it from delivered", ""]
        out += [f"- {line}" for line in gaps] or ["- see `delivery blocked` in the journal"]
    out += ["", "## Not verified", ""]
    out += [f"- {line}" for line in unverified] or [
        "Nothing: every check that ran could observe what it is for."
    ]
    if gates.strip():
        out += ["", "## Checks", "", "```", gates.rstrip(), "```"]
    return "\n".join(out) + "\n"


def write(workspace: str | Path, **said: Any) -> Path:
    """Write the closing report whole (:func:`render`'s arguments); its path."""
    path = Path(workspace) / REPORT_FILENAME
    text = render(workspace, **said)
    write_text_atomic(path, text)
    return path


__all__ = [
    "ONBOARDED_FILENAME",
    "REPORT_FILENAME",
    "changes",
    "delivery_gaps",
    "not_done",
    "not_verified",
    "onboarded",
    "render",
    "requirements",
    "since_onboarded",
    "tasks",
    "write",
]
