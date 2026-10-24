# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What earlier turns and earlier runs learned, kept outside any one workspace.

The improver proposes lessons; this module decides what is kept. A lesson
names the symptom a worker will see, what to do about it, and the evidence it
came from. It has a scope, and a dispatch is shown only the lessons whose
scope matches it:

- ``all``: every role;
- ``role:<role>``: one role (``role:smith``, ``role:tuner``, ...);
- ``host:<site>``: runs on one site, keyed by the launch environment;
- ``family:<family>``: runs on one model family.

The store is one YAML file per scope under the lessons directory, written
under a file lock because several runs may share it. Each scope keeps at most
:data:`MAX_PER_SCOPE` lessons; seeing a lesson again refreshes it, and the
stalest one goes when the scope is full.

A lesson has a kind. A ``domain`` lesson is about the model, its kernels or
the site. A ``workflow_defect`` lesson works around this workflow's own code
(the orchestrator, a gate, the bench, the spool, integration) and files a
flow issue in the workspace's ``flow_issues.yaml``: the symptom, where the
defect probably is and how to fix it, so the operator gets a fix list and not
only a workaround. An issue may carry a patch the improver drafted: a unified
diff under ``improve/patches/`` that fixes the defect and adds the failing
scenario as a test. It is kept only when it applies to the commit this
process runs and changes nothing outside the workflow package and its tests
(:func:`patch_problems`); nothing here applies it. A lesson that names the
commit that fixed it (``fixed_by``) is retired once that commit is in the
code this process runs: it stays in the store and is shown to no one.

An ``engine_defect`` lesson is a defect of the engine under test, keyed by
what triggers it rather than by the model it was met on: the code it is in
(``affects``: a catalog entry ``<category>/<name>``, or an engine file and
symbol), the parameter regime that triggers it (``regime``: which entries,
and the dtypes, words or sizes of a vocabulary row that put a call in it),
the test or cell that detects it (``detect``), and what is done about it:
the upstream change that fixed it (``upstream_fix``: a commit, or a pull
request as ``#<n>``), or, while it is ``open``, the fix branch, draft or
workaround in ``disposition``. Kept across runs, these lessons are a registry
of known defects. :func:`known_defects` tells a run, at preflight, which of
them its tree does not have the fix for; :func:`match_defects` names the
vocabulary rows that fall in a defect's regime, so a run schedules the fix
from its first plan, and the detection is among the row's required cells. A
smith that finds a defect outside its entry reports it in the same shape
(:func:`defect_from_receipt`), and it is registered ``open``.

The driver and the workers record flow issues of their own in the
workspace's ``flow_issues.reported.jsonl`` (:func:`report_flow_issue`); each
wakes the improver, which files it as an issue of a ``workflow_defect``
lesson.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from .roles import ROLES

#: Where the store lives when neither ``--lessons-dir`` nor the variable names one.
LESSONS_ENV = "MV2FLOW_LESSONS_DIR"
DEFAULT_LESSONS_DIR = Path.home() / ".trtllm_modeling_v2" / "lessons"
#: What the improver writes in the workspace for this module to merge.
PROPOSED_FILENAME = "lessons.proposed.yaml"
#: The flow issues filed in a run, in its workspace.
FLOW_ISSUES_FILENAME = "flow_issues.yaml"
#: Serializes this process's writes of ``flow_issues.yaml``.
_ISSUES_LOCK = threading.Lock()
#: The flow issues the driver and the workers recorded, one JSON object per line.
REPORTED_ISSUES_FILENAME = "flow_issues.reported.jsonl"
#: The improver's own directory in the workspace, and where its patch drafts go.
IMPROVE_DIRNAME = "improve"
PATCHES_DIR = Path(IMPROVE_DIRNAME) / "patches"
#: The largest patch draft kept: a fix and its test, not a rewrite.
MAX_PATCH_BYTES = 256 * 1024
SCOPE_KINDS: tuple[str, ...] = ("all", "role", "host", "family")
#: What a lesson is about: this workflow's own code; the model, its kernels and the
#: site; or a defect of the engine under test that an upstream change fixed.
LESSON_KINDS: tuple[str, ...] = ("workflow_defect", "domain", "engine_defect")
#: How an ``engine_defect`` names its fix: a commit, or a pull request (``#<n>``,
#: ``<owner>/<repo>#<n>`` or its URL).
_COMMIT = re.compile(r"[0-9a-f]{7,40}")
_PULL = re.compile(r"(?:[\w.-]+/[\w.-]+)?#(\d+)|https?://[^\s/]+/[\w.-]+/[\w.-]+/pull/(\d+)/?")
#: What an engine defect's ``regime`` may say, each a condition on a vocabulary
#: row, all of which hold for the row to be in it:
#: ``targets`` (globs over the row's entry), ``mentions`` (regular expressions,
#: any found in the row's text), ``dtypes`` (any named in it) and ``at_least``
#: (``{name: number}``: a cell of the row states ``name`` at least that large).
REGIME_KEYS: tuple[str, ...] = ("targets", "mentions", "dtypes", "at_least")
#: An engine defect's states: open (no upstream fix yet), or fixed upstream.
DEFECT_OPEN = "open"
MAX_PER_SCOPE = 30
MAX_TEXT = 600
#: How many lessons one dispatch is shown, most recently seen first.
MAX_SHOWN = 12
#: The workflow package this code runs from. A flow issue's suspected file is
#: a path in it, and a lesson's fix is looked up in its checkout.
FLOW_CODE = Path(__file__).resolve().parents[1]


def default_lessons_dir() -> Path:
    env = os.environ.get(LESSONS_ENV, "").strip()
    return Path(env).expanduser() if env else DEFAULT_LESSONS_DIR


def _git_head(checkout: Path) -> str:
    """The commit ``checkout`` is at, or ``""`` without git or outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


#: The commit of the code this process runs, read once at import: a fix
#: committed to the checkout after the process started is not running yet.
RUNNING_HEAD = _git_head(FLOW_CODE)


@functools.lru_cache(maxsize=256)
def _is_ancestor(commit: str, head: str, checkout: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", checkout, "merge-base", "--is-ancestor", commit, head],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def fix_is_running(commit: str) -> bool:
    """Whether ``commit`` is in the history of the code this process runs.

    One ``git merge-base`` per commit per process. False whenever it cannot
    be told (no git, no checkout, a commit the checkout does not have), so a
    lesson whose fix cannot be confirmed stays shown.
    """
    if not commit or not RUNNING_HEAD:
        return False
    return _is_ancestor(commit, RUNNING_HEAD, str(FLOW_CODE))


@dataclass
class Lesson:
    scope: str
    #: The symptom a worker sees, verbatim where possible (an error line, a gate detail).
    trigger: str
    #: What to do when it appears.
    lesson: str
    #: Where it was learned: a path in the run that produced it.
    evidence: str = ""
    #: One of :data:`LESSON_KINDS`.
    kind: str = "domain"
    #: The commit that fixed the workflow defect this lesson works around.
    fixed_by: str = ""
    #: The upstream change that fixed the engine defect this lesson is about.
    upstream_fix: str = ""
    #: An engine defect's code: a catalog entry ``<category>/<name>``, or ``<file>[:<symbol>]``.
    affects: str = ""
    #: The parameter regime that triggers an engine defect (:data:`REGIME_KEYS`).
    regime: dict[str, Any] = field(default_factory=dict)
    #: The test node id, or the cell (a mapping), that detects an engine defect.
    detect: Any = ""
    #: What is done about an open engine defect: a fix branch or draft, or a workaround.
    disposition: str = ""
    #: ``open`` while an engine defect has no upstream fix; empty otherwise.
    status: str = ""
    seen: int = 1
    #: Workers' reports after applying it: it fixed the problem, or it did not hold here.
    helped: int = 0
    wrong: int = 0
    first_seen: str = ""
    last_seen: str = ""
    runs: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Stable across wording of the lesson: the scope and the normalized trigger."""
        trigger = re.sub(r"\s+", " ", self.trigger.strip().lower())
        return hashlib.sha1(f"{self.scope}\n{trigger}".encode()).hexdigest()[:12]

    @property
    def retired(self) -> bool:
        """Its fix is in the code this process runs, so no worker needs the workaround."""
        return fix_is_running(self.fixed_by)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Lesson":
        return cls(
            scope=str(data.get("scope", "")).strip(),
            trigger=str(data.get("trigger", "")).strip(),
            lesson=str(data.get("lesson", "")).strip(),
            evidence=str(data.get("evidence", "")).strip(),
            kind=str(data.get("kind") or "domain").strip(),
            fixed_by=str(data.get("fixed_by") or "").strip(),
            upstream_fix=str(data.get("upstream_fix") or "").strip(),
            affects=str(data.get("affects") or "").strip(),
            regime=dict(data.get("regime") or {})
            if isinstance(data.get("regime"), Mapping)
            else {},
            detect=data.get("detect")
            if isinstance(data.get("detect"), Mapping)
            else str(data.get("detect") or "").strip(),
            disposition=str(data.get("disposition") or "").strip(),
            status=str(data.get("status") or "").strip(),
            seen=int(data.get("seen", 1) or 1),
            helped=int(data.get("helped", 0) or 0),
            wrong=int(data.get("wrong", 0) or 0),
            first_seen=str(data.get("first_seen", "")),
            last_seen=str(data.get("last_seen", "")),
            runs=[str(r) for r in data.get("runs") or []],
        )


@dataclass
class FlowIssue:
    """A defect in the workflow's own code that a lesson works around: one row of the fix list."""

    #: The id of the lesson that works around it.
    lesson: str
    scope: str
    #: What a worker sees.
    symptom: str
    #: Where the defect probably is: ``<file>:<function>``, the file a path in the package.
    suspected: str
    #: What to change, and where.
    proposed_fix: str
    #: Paths in the run that show it.
    evidence: list[str] = field(default_factory=list)
    #: The commit that fixed it, once one has: written by the operator or the improver.
    fixed_by: str = ""
    #: The improver's draft of the fix, a unified diff under ``improve/patches/``; never applied.
    patch: str = ""
    #: The commit of the workflow's code the patch was checked to apply to.
    patch_base: str = ""
    #: The signatures of the improver wakes it answers: while it is open, a
    #: wake with one of them is recorded here instead of opening a turn.
    wakes: list[str] = field(default_factory=list)
    #: Those wakes, one line each: when, and what happened.
    recurrences: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FlowIssue":
        evidence = data.get("evidence") or []
        wakes = data.get("wakes") or []
        return cls(
            lesson=str(data.get("lesson") or "").strip(),
            scope=str(data.get("scope") or "").strip(),
            symptom=str(data.get("symptom") or "").strip(),
            suspected=str(data.get("suspected") or "").strip(),
            proposed_fix=str(data.get("proposed_fix") or "").strip(),
            evidence=[str(e) for e in ([evidence] if isinstance(evidence, str) else evidence)],
            fixed_by=str(data.get("fixed_by") or "").strip(),
            patch=str(data.get("patch") or "").strip(),
            patch_base=str(data.get("patch_base") or "").strip(),
            wakes=[str(w).strip() for w in ([wakes] if isinstance(wakes, str) else wakes)],
            recurrences=[str(r) for r in data.get("recurrences") or []],
            first_seen=str(data.get("first_seen") or ""),
            last_seen=str(data.get("last_seen") or ""),
        )


def scope_problem(scope: str) -> str:
    """Why ``scope`` is unusable, or ``""``."""
    kind, _, name = scope.partition(":")
    if kind not in SCOPE_KINDS:
        return f"scope {scope!r}: the kind must be one of {list(SCOPE_KINDS)}"
    if kind == "all":
        return "" if not name else f"scope {scope!r}: `all` takes no name"
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        return f"scope {scope!r}: `{kind}:` needs a name of letters, digits, `_`, `.` or `-`"
    if kind == "role" and name not in ROLES:
        return f"scope {scope!r}: no role {name!r}; roles are {list(ROLES)}"
    return ""


def _evidence_problem(root: Path, evidence: str) -> str:
    """Why ``evidence`` is not a path inside the workspace ``root``, or ``""``."""
    path = (root / evidence).resolve() if evidence else None
    if path is None or not path.exists():
        return f"evidence {evidence!r} is not a path in the workspace"
    if root != path and root not in path.parents:
        return f"evidence {evidence!r} is outside the workspace"
    return ""


def _suspected_problem(suspected: str) -> str:
    """Why ``suspected`` is not ``<file>:<function>`` in the workflow's code, or ``""``.

    The file is a path in the workflow package, or in the agent-flow tree
    that holds it.
    """
    name, _, function = (part.strip() for part in suspected.partition(":"))
    if not name or not function:
        return f"`suspected` {suspected!r} is not `<file>:<function>`"
    tree = FLOW_CODE.parents[2]
    for root in (FLOW_CODE, tree):
        path = (root / name).resolve()
        if path.is_file() and tree in path.parents:
            return ""
    return f"`suspected` file {name!r} is not a file of the workflow under {FLOW_CODE}"


# ------------------------------------------------------------------ patch drafts


@functools.lru_cache(maxsize=8)
def _toplevel(checkout: str) -> str:
    """The top level of the git checkout that holds ``checkout``, or ``""``."""
    try:
        result = subprocess.run(
            ["git", "-C", checkout, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


@dataclass(frozen=True)
class FlowLayout:
    """Where the workflow's code is in the checkout this process runs from."""

    #: The checkout's top level: a patch's paths are relative to it, as ``git diff`` writes them.
    top: Path
    #: The workflow package, and its tests, relative to :attr:`top`.
    package: str
    tests: str
    #: The commit this process runs (:data:`RUNNING_HEAD`).
    head: str


def flow_layout() -> FlowLayout | None:
    """The checkout this process runs from; None outside git, where no patch can be checked."""
    top = _toplevel(str(FLOW_CODE))
    if not top or not RUNNING_HEAD:
        return None
    root = Path(top).resolve()
    tests = FLOW_CODE.parents[2] / "tests" / FLOW_CODE.parent.name / FLOW_CODE.name
    try:
        package = FLOW_CODE.resolve().relative_to(root).as_posix()
        tests_rel = tests.resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    return FlowLayout(root, package, tests_rel, RUNNING_HEAD)


def _git_in(top: Path, *args: str, env: Mapping[str, str] | None = None, timeout: float = 120):
    """``git -C top <args>``, or None when git cannot be run."""
    try:
        return subprocess.run(
            ["git", "-C", str(top), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _patch_paths(top: Path, patch: Path) -> tuple[list[str], str]:
    """The paths ``patch`` changes, as ``git apply`` reads them, and why none could be read."""
    result = _git_in(top, "apply", "--numstat", "-z", str(patch))
    if result is None or result.returncode != 0:
        return [], (result.stderr.strip() if result is not None else "git cannot be run")
    paths: list[str] = []
    parts = iter(result.stdout.split("\0"))
    for part in parts:
        fields = part.split("\t", 2)
        if len(fields) < 3:
            continue
        # A rename or copy names no path in its own field: its two paths follow it.
        paths += [fields[2]] if fields[2] else [next(parts, ""), next(parts, "")]
    return [p for p in paths if p], ""


def _apply_problem(layout: FlowLayout, patch: Path) -> str:
    """Why ``patch`` does not apply to the commit this process runs, or ``""``.

    Checked in an index of its own read from that commit, so the checkout's
    files and its index are neither read nor touched.
    """
    with tempfile.TemporaryDirectory(prefix="mv2-patch-") as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        for args in (("read-tree", layout.head), ("apply", "--cached", "--check", str(patch))):
            result = _git_in(layout.top, *args, env=env)
            if result is None:
                return "git cannot be run"
            if result.returncode != 0:
                return result.stderr.strip() or f"`git {args[0]}` exited {result.returncode}"
    return ""


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def patch_problems(workspace: Path, patch: str) -> tuple[Path | None, list[str]]:
    """The patch draft ``patch`` names in ``workspace``, and why it cannot be kept.

    It is a ``.diff`` under the workspace's ``improve/patches/`` of at most
    :data:`MAX_PATCH_BYTES`, written as ``git diff`` writes it from the top of
    the checkout the workflow runs from. It changes the workflow package, and
    a ``test_*.py`` under the workflow's tests holds the failing scenario; it
    changes nothing else. It applies cleanly to the commit this process runs.
    """
    root = Path(workspace).resolve()
    path = (root / patch).resolve()
    if (root / PATCHES_DIR).resolve() not in path.parents or path.suffix != ".diff":
        return None, [f"patch {patch!r} is not a `.diff` under {PATCHES_DIR}/"]
    if not path.is_file():
        return None, [f"patch {patch!r} is not a file"]
    size = path.stat().st_size
    if not 0 < size <= MAX_PATCH_BYTES:
        return None, [f"patch {patch!r} is {size} bytes; a patch is 1 to {MAX_PATCH_BYTES}"]
    layout = flow_layout()
    if layout is None:
        return None, ["the workflow's code is not a git checkout here: no patch can be checked"]
    paths, why = _patch_paths(layout.top, path)
    if not paths:
        return None, [f"patch {patch!r} is not a diff git reads" + (f": {why}" if why else "")]
    package, tests = layout.package, layout.tests
    problems = []
    outside = [p for p in paths if not (_under(p, package) or _under(p, tests))]
    if outside:
        problems.append(
            f"patch {patch!r} changes files outside {package}/ and {tests}/: {', '.join(outside)}"
        )
    if not any(_under(p, package) for p in paths):
        problems.append(f"patch {patch!r} changes nothing under {package}/")
    if not any(_under(p, tests) and re.fullmatch(r"test_.*\.py", Path(p).name) for p in paths):
        problems.append(
            f"patch {patch!r} has no test: the failing scenario is a `test_*.py` under {tests}/"
        )
    if not problems:
        why = _apply_problem(layout, path)
        if why:
            problems.append(
                f"patch {patch!r} does not apply to the workflow's code at "
                f"{layout.head[:12]}: {why[:400]}"
            )
    return path, problems


def _validate_issue(
    raw: Any, lesson: Lesson, root: Path
) -> tuple[FlowIssue | None, list[str], list[str]]:
    """The flow issue a ``workflow_defect`` lesson files, what is wrong with it, and with its patch.

    The symptom defaults to the lesson's trigger and the evidence to its own.
    A patch that cannot be kept (:func:`patch_problems`) costs the issue only
    its patch.
    """
    if not isinstance(raw, Mapping):
        return None, ["`issue` is not a mapping"], []
    given = {k: v for k, v in raw.items() if v}
    issue = FlowIssue.from_dict({"symptom": lesson.trigger, "evidence": [lesson.evidence], **given})
    issue.lesson, issue.scope = lesson.id, lesson.scope
    issue.fixed_by = issue.fixed_by or lesson.fixed_by
    problems = [p for p in (_evidence_problem(root, e) for e in issue.evidence) if p]
    suspected = _suspected_problem(issue.suspected)
    problems += [suspected] if suspected else []
    for name in ("symptom", "proposed_fix"):
        text = getattr(issue, name)
        if not text:
            problems.append(f"`issue` has no `{name}`")
        elif len(text) > MAX_TEXT:
            problems.append(f"`issue.{name}` is {len(text)} characters; the limit is {MAX_TEXT}")
    refused: list[str] = []
    if issue.patch:
        path, refused = patch_problems(root, issue.patch)
        kept = path is not None and not refused
        issue.patch = str(path) if kept else ""
        issue.patch_base = RUNNING_HEAD if kept else ""
    else:
        issue.patch_base = ""
    return issue, problems, refused


def upstream_ref(fix: str) -> tuple[str, str] | None:
    """``("commit", <hash>)`` or ``("pull", <number>)`` for an ``upstream_fix``; None for neither."""
    fix = fix.strip()
    if _COMMIT.fullmatch(fix):
        return "commit", fix
    match = _PULL.fullmatch(fix)
    if match:
        return "pull", match.group(1) or match.group(2)
    return None


def validate_proposal(raw: Any, workspace: Path) -> tuple[list[Lesson], list[FlowIssue], list[str]]:
    """The proposed lessons and flow issues that are admissible, and why the rest are not.

    A lesson must have a valid scope and kind, a trigger and a lesson within
    :data:`MAX_TEXT` characters, and evidence that exists inside the
    workspace: a lesson nobody can trace is a rumor. A ``workflow_defect``
    lesson files an ``issue`` naming the suspected ``<file>:<function>`` and
    a proposed fix, unless it names the commit that fixed it already; the
    issue's ``patch`` is kept only when it passes :func:`patch_problems`, and
    a refused patch is reported without refusing its issue. An
    ``engine_defect`` lesson names the upstream change that fixed it
    (``upstream_fix``), and no other kind names one.
    """
    items = raw.get("lessons") if isinstance(raw, Mapping) else raw
    if not isinstance(items, list):
        return [], [], ["the proposal must be a mapping with a `lessons` list"]
    root = Path(workspace).resolve()
    kept: list[Lesson] = []
    issues: list[FlowIssue] = []
    errors: list[str] = []
    for index, item in enumerate(items):
        where = f"lessons[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{where}: not a mapping")
            continue
        lesson = Lesson.from_dict(item)
        problems = [scope_problem(lesson.scope)] if scope_problem(lesson.scope) else []
        if lesson.kind not in LESSON_KINDS:
            problems.append(f"kind {lesson.kind!r} is not one of {list(LESSON_KINDS)}")
        for name in ("trigger", "lesson"):
            text = getattr(lesson, name)
            if not text:
                problems.append(f"no `{name}`")
            elif len(text) > MAX_TEXT:
                problems.append(f"`{name}` is {len(text)} characters; the limit is {MAX_TEXT}")
        if lesson.fixed_by and not re.fullmatch(r"[0-9a-f]{7,40}", lesson.fixed_by):
            problems.append(f"`fixed_by` {lesson.fixed_by!r} is not a commit hash")
        problems += _upstream_problems(lesson)
        evidence = _evidence_problem(root, lesson.evidence)
        problems += [evidence] if evidence else []
        issue, refused = None, []
        if item.get("issue") is not None:
            if lesson.kind != "workflow_defect":
                problems.append("only a `workflow_defect` lesson files an `issue`")
            else:
                issue, issue_problems, refused = _validate_issue(item["issue"], lesson, root)
                problems += issue_problems
        elif lesson.kind == "workflow_defect" and not lesson.fixed_by:
            problems.append(
                "a `workflow_defect` lesson files an `issue` (suspected, proposed_fix) "
                "unless `fixed_by` names the commit that fixed it"
            )
        if problems:
            errors.append(f"{where}: " + "; ".join(problems))
            continue
        if refused:
            errors.append(f"{where}: its issue is kept without its patch: " + "; ".join(refused))
        kept.append(lesson)
        if issue is not None:
            issues.append(issue)
    return kept, issues, errors


def regime_problems(regime: Mapping[str, Any]) -> list[str]:
    """Why an engine defect's ``regime`` cannot be evaluated; empty when it can."""
    problems = [
        f"`regime` key {k!r} is not one of {list(REGIME_KEYS)}"
        for k in regime
        if k not in REGIME_KEYS
    ]
    for key in ("targets", "mentions", "dtypes"):
        value = regime.get(key)
        if value is not None and not (
            isinstance(value, list) and all(isinstance(v, str) and v for v in value)
        ):
            problems.append(f"`regime.{key}` is a list of strings")
    for pattern in regime.get("mentions") or ():
        try:
            re.compile(str(pattern))
        except re.error as exc:
            problems.append(f"`regime.mentions` {pattern!r} is not a regular expression: {exc}")
    least = regime.get("at_least")
    if least is not None and not (
        isinstance(least, Mapping)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in least.values())
    ):
        problems.append("`regime.at_least` maps a cell's field to a number")
    return problems


def _upstream_problems(lesson: Lesson) -> list[str]:
    """Why ``lesson``'s ``upstream_fix`` and engine-defect fields do not fit its kind."""
    defect_fields = ("affects", "disposition", "status")
    if lesson.kind != "engine_defect":
        problems = (
            ["only an `engine_defect` lesson names an `upstream_fix`"]
            if lesson.upstream_fix
            else []
        )
        named = [n for n in defect_fields if getattr(lesson, n)]
        named += ["regime"] if lesson.regime else []
        return problems + [f"only an `engine_defect` lesson names `{n}`" for n in named]
    problems = regime_problems(lesson.regime)
    if lesson.status not in ("", DEFECT_OPEN):
        problems.append(f"`status` {lesson.status!r} is `{DEFECT_OPEN}` or absent")
    if not lesson.upstream_fix and not lesson.disposition:
        problems.append(
            "an `engine_defect` lesson names the change that fixed it in `upstream_fix`, or, "
            "while it is open, its fix branch or workaround in `disposition`"
        )
    elif lesson.upstream_fix and upstream_ref(lesson.upstream_fix) is None:
        problems.append(
            f"`upstream_fix` {lesson.upstream_fix!r} is not a commit hash or a pull request "
            "(`#<n>`, `<owner>/<repo>#<n>` or its URL)"
        )
    if lesson.fixed_by:
        problems.append("`fixed_by` is a commit of the workflow; an engine fix is `upstream_fix`")
    return problems


def _scopes(roles: Iterable[str], host: str, family: str) -> list[str]:
    scopes = ["all", *(f"role:{role}" for role in roles)]
    scopes += [f"host:{host}"] if host else []
    scopes += [f"family:{family}"] if family else []
    return scopes


class LessonStore:
    """The lessons directory: one ``<kind>[_<name>].yaml`` per scope."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, scope: str) -> Path:
        return self.root / (scope.replace(":", "_") + ".yaml")

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def load(self, scope: str) -> list[Lesson]:
        path = self._path(scope)
        if not path.is_file():
            return []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        except (OSError, yaml.YAMLError):
            return []
        return [Lesson.from_dict(d) for d in data if isinstance(d, Mapping)]

    def _save(self, scope: str, lessons: Sequence[Lesson]) -> None:
        path = self._path(scope)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            yaml.safe_dump([asdict(lesson) for lesson in lessons], sort_keys=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    def merge(self, lessons: Iterable[Lesson], run: str, evidence_root: Path) -> list[str]:
        """Add or refresh each lesson; return one line per change.

        Evidence is stored as an absolute path, so a later run can open it. A
        refresh takes the proposal's wording and kind, and its ``fixed_by``
        and ``upstream_fix`` when it names them. A retired lesson proposed
        again without ``fixed_by`` is reopened: its symptom came back on code
        that has the fix.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        changes: list[str] = []
        by_scope: dict[str, list[Lesson]] = {}
        for lesson in lessons:
            by_scope.setdefault(lesson.scope, []).append(lesson)
        with self._locked():
            for scope, new in by_scope.items():
                current = {lesson.id: lesson for lesson in self.load(scope)}
                for lesson in new:
                    evidence = str((Path(evidence_root) / lesson.evidence).resolve())
                    old = current.get(lesson.id)
                    fixed = bool(lesson.fixed_by)
                    if old is None:
                        lesson.evidence, lesson.first_seen, lesson.last_seen = evidence, now, now
                        lesson.runs = [run]
                        current[lesson.id] = lesson
                        changes.append(f"added [{scope}] {lesson.trigger[:80]}")
                    else:
                        old.lesson, old.evidence, old.last_seen = lesson.lesson, evidence, now
                        old.kind = lesson.kind
                        old.upstream_fix = lesson.upstream_fix or old.upstream_fix
                        for name in ("affects", "regime", "detect", "disposition"):
                            setattr(old, name, getattr(lesson, name) or getattr(old, name))
                        old.status = "" if old.upstream_fix else (lesson.status or old.status)
                        if run not in old.runs:
                            old.seen += 1
                            old.runs = [*old.runs, run][-10:]
                        changes.append(f"refreshed [{scope}] {old.trigger[:80]}")
                        fixed = fixed and lesson.fixed_by != old.fixed_by
                        if fixed:
                            old.fixed_by = lesson.fixed_by
                        elif not lesson.fixed_by and old.retired:
                            changes.append(
                                f"reopened [{scope}] {old.trigger[:80]}: seen again although "
                                f"{old.fixed_by} is in the running code"
                            )
                            old.fixed_by = ""
                    if fixed:
                        kept = current[lesson.id]
                        changes.append(
                            f"fixed by {kept.fixed_by} [{scope}] {kept.trigger[:80]}"
                            + ("; retired" if kept.retired else "; not in the running code yet")
                        )
                ordered = sorted(current.values(), key=lambda x: x.last_seen, reverse=True)
                dropped = ordered[MAX_PER_SCOPE:]
                changes += [f"dropped [{scope}] {x.trigger[:80]}" for x in dropped]
                self._save(scope, ordered[:MAX_PER_SCOPE])
        return changes

    def feedback(self, marks: Mapping[str, str]) -> list[str]:
        """Count each worker report (``{lesson id: helped | wrong}``); return what changed."""
        marks = {str(k): str(v) for k, v in marks.items() if str(v) in ("helped", "wrong")}
        if not marks:
            return []
        changes: list[str] = []
        with self._locked():
            for path in sorted(self.root.glob("*.yaml")):
                scope = path.stem.replace("_", ":", 1)
                lessons = self.load(scope)
                touched = False
                for lesson in lessons:
                    mark = marks.get(lesson.id)
                    if mark:
                        setattr(lesson, mark, getattr(lesson, mark) + 1)
                        changes.append(f"{mark} [{scope}] {lesson.trigger[:80]}")
                        touched = True
                if touched:
                    self._save(scope, lessons)
        return changes

    def for_dispatch(self, role: str, *, host: str = "", family: str = "") -> list[Lesson]:
        """The lessons a turn of ``role`` is shown, most recently seen first.

        A lesson workers reported wrong at least twice, and more often than
        helpful, is no longer shown, and neither is a retired one; both stay
        in the store.
        """
        found = [
            lesson
            for scope in _scopes([role], host, family)
            for lesson in self.load(scope)
            if not (lesson.wrong >= 2 and lesson.wrong > lesson.helped) and not lesson.retired
        ]
        found.sort(key=lambda x: (x.last_seen, x.seen), reverse=True)
        return found[:MAX_SHOWN]

    def retired(self, *, host: str = "", family: str = "") -> list[Lesson]:
        """The lessons in a run's scopes, any role, whose fix is in the code this process runs."""
        return [
            lesson
            for scope in _scopes(ROLES, host, family)
            for lesson in self.load(scope)
            if lesson.retired
        ]

    def engine_defects(self, *, host: str = "", family: str = "") -> list[Lesson]:
        """The known engine defects in a run's scopes, any role: its ``engine_defect`` lessons."""
        return [
            lesson
            for scope in _scopes(ROLES, host, family)
            for lesson in self.load(scope)
            if lesson.kind == "engine_defect" and (lesson.upstream_fix or lesson.disposition)
        ]


def render_block(lessons: Sequence[Lesson]) -> str:
    """The prompt block a dispatch carries, or ``""`` with nothing to show."""
    if not lessons:
        return ""
    lines = [
        "## Lessons from earlier turns and runs",
        "",
        "Each was learned from a failure on this workflow. When you see the symptom, "
        "apply the lesson before investigating again. For each lesson you applied, add "
        "`lessons: {<id>: helped | wrong}` to your decision file: `wrong` when it did not "
        "hold here. Lessons reported wrong stop being shown.",
        "",
    ]
    for lesson in lessons:
        kind = ""
        if lesson.kind == "workflow_defect":
            kind = "a workflow defect, " + (
                f"fixed in `{lesson.fixed_by}` but not in the code this run executes, "
                if lesson.fixed_by
                else ""
            )
        elif lesson.kind == "engine_defect" and lesson.upstream_fix:
            kind = f"an engine defect fixed upstream in `{lesson.upstream_fix}`, "
        elif lesson.kind == "engine_defect":
            kind = f"an open engine defect ({lesson.disposition}), "
        lines.append(f"- `{lesson.id}` **When:** {lesson.trigger}")
        lines.append(f"  **Do:** {lesson.lesson}")
        lines.append(
            f"  (scope `{lesson.scope}`, {kind}seen in {lesson.seen} run(s); `{lesson.evidence}`)"
        )
    return "\n".join(lines)


def load_proposal(workspace: Path) -> tuple[list[Lesson], list[FlowIssue], list[str]]:
    """Read and validate ``lessons.proposed.yaml``; a missing file proposes nothing."""
    path = Path(workspace) / PROPOSED_FILENAME
    if not path.is_file():
        return [], [], []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [], [], [f"{PROPOSED_FILENAME} is not YAML: {exc}"]
    return validate_proposal(raw, workspace)


def _read_issues(path: Path) -> list[FlowIssue] | None:
    """The issues in ``path``: none when it is missing, None when it does not parse."""
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    items = data.get("issues") if isinstance(data, Mapping) else None
    if not isinstance(items, list):
        return None
    return [FlowIssue.from_dict(d) for d in items if isinstance(d, Mapping)]


def load_issues(workspace: Path) -> list[FlowIssue]:
    """The flow issues filed in ``workspace``."""
    return _read_issues(Path(workspace) / FLOW_ISSUES_FILENAME) or []


def record_issues(
    workspace: Path, issues: Iterable[FlowIssue], fixes: Mapping[str, str] | None = None
) -> list[str]:
    """File or refresh each issue in ``flow_issues.yaml``; return one line per change.

    An issue is keyed by its lesson's id. Evidence is stored as absolute
    paths, and a fix recorded earlier (by the operator or the improver) is
    kept unless the refresh names one; so is a patch. A patch is filed as
    ``improve/patches/<lesson id>.diff``. ``fixes`` maps a lesson id to the
    commit a proposal said fixed it, and marks that lesson's issue fixed. A
    refresh keeps the wake signatures filed earlier and the recurrences
    recorded (:func:`record_recurrences`). A file that does not parse is left
    alone for the operator.
    """
    with _ISSUES_LOCK:
        return _record_issues(Path(workspace), issues, fixes)


def _record_issues(
    workspace: Path, issues: Iterable[FlowIssue], fixes: Mapping[str, str] | None
) -> list[str]:
    path = workspace / FLOW_ISSUES_FILENAME
    current = _read_issues(path)
    if current is None:
        return [f"{path} does not parse as `issues:` YAML; left as it is"]
    by_lesson = {issue.lesson: issue for issue in current}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    changes: list[str] = []
    for issue in issues:
        issue.evidence = [str((Path(workspace) / e).resolve()) for e in issue.evidence]
        old = by_lesson.get(issue.lesson)
        issue.first_seen, issue.last_seen = (old.first_seen if old else now), now
        if old is not None:
            issue.fixed_by = issue.fixed_by or old.fixed_by
            issue.wakes = list(dict.fromkeys([*old.wakes, *issue.wakes]))
            issue.recurrences = old.recurrences
            if not issue.patch:
                issue.patch, issue.patch_base = old.patch, old.patch_base
        drafted = bool(issue.patch) and (old is None or issue.patch != old.patch)
        if issue.patch:
            issue.patch = _file_patch(Path(workspace), issue)
        by_lesson[issue.lesson] = issue
        verb = "refreshed" if old is not None else "filed"
        line = f"{verb} [{issue.scope}] {issue.suspected}: {issue.symptom[:80]}"
        if drafted:
            line += f"; patch {Path(issue.patch).name} applies to {issue.patch_base[:12]}"
        changes.append(line)
    for lesson_id, commit in (fixes or {}).items():
        old = by_lesson.get(lesson_id)
        if old is not None and commit and old.fixed_by != commit:
            old.fixed_by = commit
            changes.append(f"fixed by {commit} [{old.scope}] {old.suspected}")
    if changes:
        _write_issues(path, by_lesson.values())
    return changes


def _write_issues(path: Path, issues: Iterable[FlowIssue]) -> None:
    tmp = path.with_suffix(".tmp")
    data = {"issues": [asdict(issue) for issue in issues]}
    tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def record_recurrences(workspace: Path, seen: Sequence[tuple[str, str]]) -> list[str]:
    """Record each wake an open issue already answers on that issue; returns their signatures.

    ``seen`` holds a wake's signature and its line (when, and what happened)
    per wake. An issue is open until it names its ``fixed_by``: a wake that
    recurs after the fix is news, and is not recorded here.
    """
    path = Path(workspace) / FLOW_ISSUES_FILENAME
    with _ISSUES_LOCK:
        current = _read_issues(path)
        if not current:
            return []
        answered = {
            signature: issue for issue in current if not issue.fixed_by for signature in issue.wakes
        }
        recorded: list[str] = []
        for signature, line in seen:
            issue = answered.get(signature)
            if issue is not None:
                issue.recurrences.append(line)
                issue.last_seen = time.strftime("%Y-%m-%dT%H:%M:%S")
                recorded.append(signature)
        if recorded:
            _write_issues(path, current)
        return recorded


def _file_patch(workspace: Path, issue: FlowIssue) -> str:
    """``issue``'s patch moved to ``improve/patches/<lesson id>.diff``; returns where it is."""
    source = Path(issue.patch)
    target = (workspace / PATCHES_DIR / f"{issue.lesson}.diff").resolve()
    if source.is_file() and source.resolve() != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    return str(target) if target.is_file() else issue.patch


# ------------------------------------------------------------ reported flow issues


def report_flow_issue(
    workspace: Path,
    *,
    symptom: str,
    by: str = "driver",
    evidence: Sequence[str] = (),
    impact: str = "",
    suspected: str = "",
    proposed_fix: str = "",
) -> dict[str, Any]:
    """Record a flow issue the driver or a worker saw; returns the line written.

    Appended to the workspace's ``flow_issues.reported.jsonl`` under a file
    lock, since several workers may write at once. Each line wakes the
    improver once; it files the issue on a ``workflow_defect`` lesson.
    """
    if not symptom.strip():
        raise ValueError("a reported flow issue names its symptom")
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "by": by,
        "symptom": symptom.strip()[:MAX_TEXT],
        "evidence": [str(e) for e in evidence],
        "impact": impact.strip()[:MAX_TEXT],
        "suspected": suspected.strip(),
        "proposed_fix": proposed_fix.strip()[:MAX_TEXT],
    }
    path = Path(workspace) / REPORTED_ISSUES_FILENAME
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(record) + "\n")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return record


def jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Each complete line of the JSON-lines file ``path``, in order; none when it is missing.

    A line that is not a JSON object stands as its text, under ``symptom``; a
    line still being written is left for the next read.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    found: list[dict[str, Any]] = []
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        try:
            data = json.loads(line)
        except ValueError:
            data = None
        found.append(data if isinstance(data, dict) else {"symptom": line.strip()})
    return found


def reported_issues(workspace: Path) -> list[dict[str, Any]]:
    """The flow issues the driver and the workers reported in ``workspace``, in order."""
    return jsonl_records(Path(workspace) / REPORTED_ISSUES_FILENAME)


# ------------------------------------------------------------- known engine defects


@dataclass(frozen=True)
class KnownDefect:
    """An ``engine_defect`` lesson, and whether a tree has its upstream fix."""

    lesson: Lesson
    #: ``present``, ``absent``, or ``unknown`` (:func:`fix_in_tree`).
    state: str


def fix_in_tree(fix: str, repo: Path) -> str:
    """Whether the engine checkout ``repo`` has the upstream change ``fix``.

    ``present`` for a commit that is an ancestor of ``HEAD``, or that a
    commit in ``HEAD``'s history names (a cherry-pick's ``cherry picked from
    commit <hash>``); for a pull request ``#<n>`` that a commit in that
    history names, as its squash merge and its cherry-picks do. ``absent``
    otherwise, and ``unknown`` without git or a checkout.
    """
    ref = upstream_ref(fix)
    head = _git_in(Path(repo), "rev-parse", "--verify", "--quiet", "HEAD", timeout=30)
    if ref is None or head is None or head.returncode != 0:
        return "unknown"
    kind, value = ref
    if kind == "commit":
        ancestor = _git_in(Path(repo), "merge-base", "--is-ancestor", value, "HEAD")
        if ancestor is not None and ancestor.returncode == 0:
            return "present"
        grep = ["-F", f"--grep=cherry picked from commit {value}"]
    else:
        grep = ["-E", f"--grep=#{value}([^0-9]|$)"]
    named = _git_in(Path(repo), "log", "-1", "--format=%H", *grep, "HEAD")
    if named is None or named.returncode != 0:
        return "unknown"
    return "present" if named.stdout.strip() else "absent"


def known_defects(
    store: LessonStore, repo: Path, *, host: str = "", family: str = ""
) -> list[KnownDefect]:
    """The known engine defects in a run's scopes, each with whether ``repo`` has its fix.

    An open one has no upstream fix to look for: ``absent``.
    """
    return [
        KnownDefect(
            lesson, fix_in_tree(lesson.upstream_fix, repo) if lesson.upstream_fix else "absent"
        )
        for lesson in store.engine_defects(host=host, family=family)
    ]


@dataclass(frozen=True)
class DefectHit:
    """A vocabulary queue row in a known engine defect's regime."""

    defect: Lesson
    row: int
    target: str


def _row_text(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True, default=str)


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _numbers(v)]
    return []


def in_regime(defect: Lesson, row: Mapping[str, Any]) -> bool:
    """Whether vocabulary queue ``row`` falls in ``defect``'s regime.

    Its entry must match one of ``regime.targets`` (an ``affects`` that names
    a catalog entry stands in for them), and each other condition the regime
    states must hold. A defect that states neither targets nor an entry
    matches no row: a regime with no anchor would match every one.
    """
    import fnmatch

    regime = defect.regime or {}
    target = str(row.get("target") or "")
    targets = list(regime.get("targets") or ())
    if not targets and re.fullmatch(r"(?:ops/)?[a-z_]+/[a-z0-9_]+", defect.affects):
        targets = [defect.affects, f"ops/{defect.affects}"]
    if not targets or not any(fnmatch.fnmatchcase(target, t) for t in targets):
        return False
    text = _row_text(row)
    mentions = regime.get("mentions") or ()
    if mentions and not any(re.search(str(m), text, re.IGNORECASE) for m in mentions):
        return False
    dtypes = regime.get("dtypes") or ()
    if dtypes and not any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(d))}(?![A-Za-z0-9_])", text, re.IGNORECASE)
        for d in dtypes
    ):
        return False
    for name, least in (regime.get("at_least") or {}).items():
        cells = [c for c in row.get("cells") or () if isinstance(c, Mapping)]
        if not any(n >= float(least) for c in cells for n in _numbers(c.get(name))):
            return False
    return True


def match_defects(defects: Sequence[Lesson], rows: Sequence[Mapping[str, Any]]) -> list[DefectHit]:
    """Every queue row each known engine defect's regime takes in (:func:`in_regime`)."""
    return [
        DefectHit(defect, int(row.get("n", 0) or 0), str(row.get("target") or ""))
        for defect in defects
        for row in rows
        if isinstance(row, Mapping) and in_regime(defect, row)
    ]


#: The keys a smith's ``engine_defect`` report carries; ``detect`` is optional.
RECEIPT_KEYS: tuple[str, ...] = ("affects", "trigger", "detect", "proposal")


def defect_from_receipt(report: Mapping[str, Any], *, evidence: str, where: str) -> Lesson | None:
    """The open registry entry a smith's ``engine_defect`` report makes; ``None`` when incomplete.

    ``evidence`` is the decision file, relative to the workspace, and
    ``where`` names the task and its branch, where the proposal waits.
    """
    affects = str(report.get("affects") or "").strip()
    trigger = str(report.get("trigger") or "").strip()
    proposal = str(report.get("proposal") or "").strip()
    if not (affects and trigger and proposal):
        return None
    detect = report.get("detect")
    return Lesson(
        scope="all",
        trigger=f"{affects}: {trigger}"[:MAX_TEXT],
        lesson=proposal[:MAX_TEXT],
        evidence=evidence,
        kind="engine_defect",
        affects=affects,
        detect=dict(detect) if isinstance(detect, Mapping) else str(detect or ""),
        disposition=f"proposed in {where}"[:MAX_TEXT],
        status=DEFECT_OPEN,
    )


# --------------------------------------------------------------------- self-check


def check_command(workspace: Path) -> str:
    """The shell command that checks ``workspace``'s proposal with the code this process runs."""
    tree = shlex.quote(str(FLOW_CODE.parents[2]))
    code = f"import sys, {__name__} as lessons; sys.exit(lessons.main())"
    return (
        f"PYTHONPATH={tree}${{PYTHONPATH:+:$PYTHONPATH}} {shlex.quote(sys.executable)} "
        f"-c {shlex.quote(code)} check-proposal {shlex.quote(str(workspace))}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``check-proposal <workspace>``: what of the improver's proposal Python would refuse.

    The improver runs it (:func:`check_command`) before it ends its turn, so a
    patch that does not apply is fixed in the turn rather than refused after
    it. Exits 1 when anything would be refused.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "check-proposal":
        print("usage: check-proposal <workspace>", file=sys.stderr)
        return 2
    lessons, issues, errors = load_proposal(Path(args[1]))
    for error in errors:
        print(f"REFUSED {error}")
    patches = sum(1 for issue in issues if issue.patch)
    print(f"kept: {len(lessons)} lesson(s), {len(issues)} issue(s), {patches} patch(es)")
    return 1 if errors else 0


__all__ = [
    "DEFAULT_LESSONS_DIR",
    "FLOW_CODE",
    "FLOW_ISSUES_FILENAME",
    "FlowIssue",
    "FlowLayout",
    "IMPROVE_DIRNAME",
    "DEFECT_OPEN",
    "DefectHit",
    "KnownDefect",
    "LESSONS_ENV",
    "LESSON_KINDS",
    "Lesson",
    "LessonStore",
    "MAX_PATCH_BYTES",
    "MAX_PER_SCOPE",
    "PATCHES_DIR",
    "PROPOSED_FILENAME",
    "RECEIPT_KEYS",
    "REGIME_KEYS",
    "REPORTED_ISSUES_FILENAME",
    "RUNNING_HEAD",
    "SCOPE_KINDS",
    "check_command",
    "default_lessons_dir",
    "defect_from_receipt",
    "fix_in_tree",
    "fix_is_running",
    "flow_layout",
    "in_regime",
    "jsonl_records",
    "known_defects",
    "load_issues",
    "load_proposal",
    "main",
    "match_defects",
    "patch_problems",
    "record_issues",
    "record_recurrences",
    "regime_problems",
    "render_block",
    "report_flow_issue",
    "reported_issues",
    "scope_problem",
    "upstream_ref",
    "validate_proposal",
]
