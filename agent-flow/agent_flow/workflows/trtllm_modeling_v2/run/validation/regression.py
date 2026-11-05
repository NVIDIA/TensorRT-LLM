# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The regression check: what a batch of changes to shared code did to the package's tests.

No agent turn runs for it, unless the run caps what it runs (below). It runs
a declared list, every test under the modeling_v2 package's test tree
(:data:`PACKAGE_TESTS`), through the run's test results
(``checks/test_results.py``), which hold at the exact commit they ran at:

1. at the commit the checkout is at (the head), every test file with no result
   there yet, in a read-only snapshot of the head (:func:`record_at`);
2. at the commit the last check cleared, else the run's starting commit (the
   base), the tests that fail at the head and have no result there;
3. the results at the head against those at the base.

It runs only the test files the change can reach (:func:`affected_files`):
those it changed, those that import a module it changed (directly, or through
a helper or another module of the package that does), and the claims gates.
A change to code the whole package shares (outside the catalog's and the
targets' own directories, the catalog's or ``models/``'s own ``__init__.py``, a
``conftest.py``) reaches every file. A package ``__init__.py`` deeper down (a
family's, a target's) reaches the files that import any module of that package,
since importing one runs it. An import inside a function runs only when it is
called: it reaches the files that import its module by name (a routing test),
not that module's own importers. The catalog's and ``models/``'s own
``__init__.py`` import every entry and family to register them; reached, they
reach no further. A change no unit test imports or reads (integration tests and
their lists, docs, examples) reaches none. With no base to compare with, every
file runs.

A run may cap that with a curated selection (:data:`SELECTION_ENV`): a list of
test files, written by an operator or an agent from what the change reaches, that
the check runs instead (those of them present at the head), with the claims
gates; a list that names none of them leaves the check INCOMPLETE. Without one, a
run's ``regression.max_files`` caps it: when the change reaches more test files
than that, besides the claims gates, an agent picks that many of them from the
change (:func:`picked_files`), and a pick that is not a subset of at most that
many runs them all.

A cap defers what it leaves out; it never drops it. A capped check that passes
clears nothing, so the next check's base stays where the last full check left
it and the next change reaches those files again; its receipt names the head as
``deferred``, and delivery runs, with no cap, every file the change since the
last full check reaches while one is owed (:func:`owed`).

A test that passed at the base and fails now, or a new one that fails, is a
regression, and so is a test with no result at the head: the check fails,
naming each. The head a full check found clean is its receipt's ``cleared``,
the next check's base, whose results this check has already recorded.
"""

from __future__ import annotations

import ast
import contextlib
import math
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Sequence

from ...checks import test_results
from ...infra.compute.commands import CommandSpec
from ...infra.compute.executor import FEATURE, select_command_executor
from ...infra.compute.scheduler import parse_duration
from ...infra.git import git_output, read_files_at
from ...operators.coverage import CATALOG_DIR, CATALOG_TESTS_DIR, MODELING_V2_PACKAGE

#: The declared list a regression check runs: the package's source-level gates,
#: the catalog's entries and every target's own.
PACKAGE_TESTS: tuple[str, ...] = (CATALOG_TESTS_DIR,)
#: Where a check's selections go, under the workspace: ``<this>/<task id>/``.
SELECTIONS_DIRNAME = "regression"
#: How many failing tests a failure names before it counts the rest.
MAX_NAMED = 20
#: How long one test job may run at most; less where the gate sessions' usable window is
#: shorter (:func:`job_timeout`).
JOB_TIMEOUT_S = 2 * 3600
#: How long the test jobs at one commit may run in all: as many jobs as that holds at the job
#: size (:func:`max_jobs`). A job out of time keeps the files it finished, and the next runs the
#: rest.
TEST_BUDGET_S = 3 * JOB_TIMEOUT_S
#: The environment probe's wall limit: it prints one line, and a short job starts in any allocation.
PROBE_TIMEOUT_S = 600
#: What a session's usable window keeps back, at least, from its time limit: the allocation's
#: start-up and the margin its scheduler wants past a job's wall limit.
WINDOW_SLACK_S = 600

#: Where a change reaches only the files that import it: the catalog's entries, the targets.
SCOPED_SOURCE_DIRS: tuple[str, ...] = (f"{CATALOG_DIR}/", f"{MODELING_V2_PACKAGE}/models/")
#: Paths no modeling_v2 unit test imports or reads: a change there reaches no test file.
UNREACHED_PREFIXES: tuple[str, ...] = (
    "tests/integration/",
    "docs/",
    "examples/",
    "benchmarks/",
    "agent-flow/",
    ".github/",
)
UNREACHED_SUFFIXES: tuple[str, ...] = (".md", ".rst")
#: A file listing the test files (one repo-relative path a line, ``#`` comments) a regression
#: check runs instead of every file the change reaches: a curated selection for a run.
SELECTION_ENV = "MV2FLOW_REGRESSION_SELECTION"
#: Test files every check runs: the package's source-level gates read every entry.
ALWAYS_RUN = re.compile(r"(?:^|/)test_modeling_v2_[a-z_]*claims[a-z_]*\.py$")

#: Runs one command line in a tree on the run's devices; returns its exit code and output tail.
#: Called ``(command, tree)``, or with ``timeout_s=`` for a job shorter than the run's job size,
#: which a runner may say as its ``timeout_s`` (:class:`SessionJobs`; :data:`JOB_TIMEOUT_S`
#: otherwise).
RunJob = Callable[..., tuple[int, str]]
#: A read-only tree at one commit, held for the block.
TreeAt = Callable[[str], ContextManager[Path]]
#: Asks an agent for at most ``limit`` of ``files``, given what changed (``summary``):
#: ``(path, files, limit, summary)``; it writes the paths it picks to ``path``, one a line.
PickFiles = Callable[[Path, Sequence[str], int, str], None]


@dataclass
class RegressionResult:
    """What a regression check found."""

    head: str
    base: str | None
    #: ``CLEAN``, ``REGRESSED`` or ``INCOMPLETE`` (``test_results.CompareReport.verdict``).
    verdict: str
    #: One line per failing or missing test; empty when clean.
    failures: list[str] = field(default_factory=list)
    #: Tests that did not fail at the head.
    held: int = 0
    #: Test jobs this check ran.
    jobs: int = 0
    #: Which test files it ran, and why (:func:`affected_files`).
    scope: str = ""
    #: Test files the change reaches that a cap left for a later full check or delivery.
    deferred: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict == "CLEAN"

    def summary(self) -> str:
        base = (self.base or "none")[:10]
        return (
            f"{self.verdict} at {self.head[:10]} against {base}: {self.held} test(s) did not "
            f"fail at the head, {self.jobs} test job(s) run"
            + (f"; {self.scope}" if self.scope else "")
            + (
                f"; {self.deferred} reached test file(s) deferred to the next full check or "
                "delivery"
                if self.deferred
                else ""
            )
        )


def job_timeout(environment: Any) -> int:
    """How long one test job may run in ``environment``: what its gate sessions can run.

    :data:`JOB_TIMEOUT_S`, or less on a session spool: the longest usable window of a session
    gate jobs reach (those on ``environment.spool`` and the ``role: gates`` ones, the pool
    ``executor.select_command_executor`` sends them to). A session's window is its
    ``scheduler.default_wall`` and its time limit less its hand-over lead (at least
    :data:`WINDOW_SLACK_S`), the shorter of those it declares: its scheduler holds a job its
    allocation cannot finish for the successor, so a job longer than every window never starts.
    A session declaring neither is not counted.
    """
    if getattr(environment, "kind", "") != "session_spool":
        return JOB_TIMEOUT_S
    spool = Path(str(environment.spool or ""))
    windows = []
    for session in environment.sessions:
        if session.role != "gates" and Path(str(session.spool)) != spool:
            continue
        bounds = []
        if session.scheduler.get("default_wall"):
            bounds.append(parse_duration(str(session.scheduler["default_wall"])))
        if session.time_limit_s:
            lead = max(session.handover_lead_s, WINDOW_SLACK_S)
            bounds.append(session.time_limit_s - lead)
        positive = [bound for bound in bounds if bound > 0]
        if positive:
            windows.append(min(positive))
    return int(min(JOB_TIMEOUT_S, max(windows))) if windows else JOB_TIMEOUT_S


def max_jobs(timeout_s: float) -> int:
    """How many test jobs of ``timeout_s`` :data:`TEST_BUDGET_S` holds; one at least."""
    return max(1, math.ceil(TEST_BUDGET_S / max(1.0, float(timeout_s))))


class SessionJobs:
    """A :data:`RunJob` on the run's devices: each job allowed :func:`job_timeout`, or less."""

    def __init__(self, environment: Any, name: str) -> None:
        self.environment = environment
        self.name = name
        #: How long one job may run (:func:`job_timeout`).
        self.timeout_s = job_timeout(environment)

    def __call__(self, command: str, tree: Path, timeout_s: float | None = None) -> tuple[int, str]:
        executor = select_command_executor(
            self.environment, name_prefix=self.name, tree=tree, gpus="all", priority=FEATURE
        )
        wall = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        done = executor.execute(CommandSpec(argv=command, cwd=str(tree), timeout_s=wall))
        return done.returncode, (done.stdout + done.stderr)[-2000:]


def job_runner(workflow: Any, name: str) -> tuple[TreeAt, RunJob, str]:
    """``(tree_at, run_job, python)``: how ``workflow`` runs a test job, named ``name``.

    In a read-only snapshot of a commit when the run keeps them, else in the
    checkout; on the run's devices, with no agent turn, each job sized to the
    gate sessions (:class:`SessionJobs`).
    """
    repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
    snapshots = getattr(workflow, "_snapshots", None)

    def tree_at(commit: str) -> ContextManager[Path]:
        return snapshots.lease(commit) if snapshots is not None else contextlib.nullcontext(repo)

    python = str((workflow.task_data.get("environment") or {}).get("python") or "python3")
    return tree_at, SessionJobs(workflow._gate_env(), name), python


def record_at(
    commit: str,
    files: Sequence[str],
    *,
    cache_dir: Path,
    out: Path,
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
) -> tuple[test_results.Results, int, int, str]:
    """Probe the current environment, then run missing ``files`` at ``commit``.

    The probe is one short job (:data:`PROBE_TIMEOUT_S`); the test jobs, as many as
    :data:`TEST_BUDGET_S` holds at the runner's job size (:func:`_run_unrun`).
    Returns results, test jobs run, and the last job's exit code and output tail.
    A failed probe returns empty results, never historical results.
    """
    with tree_at(commit) as tree:
        command = f"{python} {shlex.quote(str(test_results.DRIVER))} --environment"
        code, tail = run_job(command, tree, timeout_s=PROBE_TIMEOUT_S)
    match = re.search(r"^MV2FLOW_TEST_ENV ([0-9a-f]{16})$", tail, re.MULTILINE)
    if code or match is None:
        return test_results.Results(commit), 0, code, f"environment probe failed: {tail}"
    return _run_unrun(
        commit,
        files,
        match.group(1),
        cache_dir=cache_dir,
        out=out,
        tree_at=tree_at,
        run_job=run_job,
        python=python,
    )


def _run_unrun(
    commit: str,
    items: Sequence[str],
    env_digest: str,
    *,
    cache_dir: Path,
    out: Path,
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
) -> tuple[test_results.Results, int, int, str]:
    """Run the ``items`` (test files, node ids) with no result at ``commit`` in ``env_digest``.

    A job at a time, each resuming from what the last recorded, while one makes progress and
    :func:`max_jobs` of the runner's job size are not spent. Returns what :func:`record_at` does.
    """
    allowed = max_jobs(getattr(run_job, "timeout_s", JOB_TIMEOUT_S))
    results = test_results.recorded(cache_dir, commit, env_digest)
    unrun = results.unrun(items)
    jobs, code, tail = 0, 0, ""
    while unrun and jobs < allowed:
        listed = test_results.write_selection(out / f"{commit[:12]}.txt", unrun)
        with tree_at(commit) as tree:
            code, tail = run_job(test_results.driver_command(python, cache_dir, tree, listed), tree)
        jobs += 1
        results = test_results.recorded(cache_dir, commit, env_digest)
        left = results.unrun(items)
        if len(left) >= len(unrun):
            break
        unrun = left
    return results, jobs, code, tail


def _module_names(path: str) -> list[str]:
    """How a test may name the module at ``path``: dotted from the root, and its last two parts."""
    stem = path[: -len(".py")] if path.endswith(".py") else path
    dotted = stem.replace("/", ".")
    parts = dotted.split(".")
    names = [dotted, stem]
    if len(parts) >= 2:
        names += [".".join(parts[-2:]), "/".join(parts[-2:])]
    return names


def _module_of(path: str) -> str:
    """The dotted module the ``.py`` file at ``path`` is, from the root (a package by its init)."""
    stem = path[: -len(".py")]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def _runs_at_import(tree: ast.Module) -> list[ast.AST]:
    """The nodes of ``tree`` that run when the module is imported.

    Everything but function bodies (and lambdas) and ``if TYPE_CHECKING:`` blocks: a class
    body, a top-level ``if`` or ``try`` runs at import.
    """
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            stack.extend(node.orelse)
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _imported(path: str, text: str, *, eager: bool = False) -> set[str]:
    """The dotted names the file at ``path`` imports: each module, and each name from one.

    ``from a import b`` gives ``a`` and ``a.b`` (``b`` may be a module); a relative
    import resolves against the file's own package. With ``eager``, only the imports that
    run when the file is imported (:func:`_runs_at_import`). A file that does not parse
    gives none.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    module = _module_of(path)
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    names: set[str] = set()
    for node in _runs_at_import(tree) if eager else ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = package.split(".") if package else []
            if node.level > len(parts) + 1:
                continue
            up = parts[: len(parts) - node.level + 1] if node.level else []
            source = ".".join([*up, *([node.module] if node.module else [])])
            if source:
                names.add(source)
                names.update(f"{source}.{a.name}" for a in node.names if a.name != "*")
    return names


def _importers(
    repo: str | Path, head: str, roots: Sequence[str], changed: Sequence[str]
) -> set[str] | None:
    """The ``.py`` files under ``roots`` at ``head`` that import one of ``changed``, directly or not.

    A file names a module by its full dotted path or by any trailing part of it (a test
    imports a helper beside it, on the path it puts first: ``from _validating import``);
    an over-match only runs a test more. ``None`` when the files cannot be read.
    """
    listing = git_output(repo, "ls-tree", "-r", "-z", "--name-only", head, "--", *roots)
    if listing is None:
        return None
    present = [p for p in listing.split("\0") if p.endswith(".py")]
    texts = read_files_at(repo, head, present)
    if texts is None:
        return None
    by_name: dict[str, set[str]] = {}
    inits: dict[str, set[str]] = {}
    for path in {*present, *(p for p in changed if p.endswith(".py"))}:
        parts = _module_of(path).split(".")
        for start in range(len(parts)):
            by_name.setdefault(".".join(parts[start:]), set()).add(path)
            if path.endswith("/__init__.py"):
                inits.setdefault(".".join(parts[start:]), set()).add(path)
    importers: dict[str, set[str]] = {}
    lazy: dict[str, set[str]] = {}
    named: dict[str, set[str]] = {}
    for path, text in texts.items():
        every = _imported(path, text)
        # A test-side helper's functions are what the tests run (a rank matrix's body): only
        # the package's own modules import lazily.
        source = path.startswith(f"{MODELING_V2_PACKAGE}/")
        eager = _imported(path, text, eager=True) if source else every
        for name in every:
            for target in by_name.get(name, ()):
                if target != path:
                    named.setdefault(target, set()).add(path)
            if name not in eager:
                for target in by_name.get(name, ()):
                    if target != path:
                        lazy.setdefault(target, set()).add(path)
                continue
            # Importing ``a.b.c`` runs the ``__init__.py`` of ``a`` and ``a.b`` as well.
            parts = name.split(".")
            targets = set(by_name.get(name, ()))
            for end in range(1, len(parts)):
                targets |= inits.get(".".join(parts[:end]), set())
            for target in targets:
                if target != path:
                    importers.setdefault(target, set()).add(path)
    registries = {f"{d}__init__.py" for d in SCOPED_SOURCE_DIRS}
    reached, frontier = set(changed), list(changed)
    called: set[str] = set()
    while frontier:
        node = frontier.pop()
        called |= lazy.get(node, set())
        if node in registries and node not in changed:
            continue
        for importer in importers.get(node, ()):
            if importer not in reached:
                reached.add(importer)
                frontier.append(importer)
    # A module that imports a reached one inside a function runs it only when called: the
    # test files that import that module by name reach it, its other importers do not.
    tests = tuple(f"{t}/" for t in roots if not t.startswith(MODELING_V2_PACKAGE))
    for module in called - reached:
        reached |= {f for f in named.get(module, ()) if f.startswith(tests)}
    return reached


def _unreached(path: str) -> bool:
    """Whether no modeling_v2 unit test imports or reads ``path`` (see :data:`UNREACHED_PREFIXES`)."""
    return path.startswith(UNREACHED_PREFIXES) or path.endswith(UNREACHED_SUFFIXES)


def _selection_lines(path: str | Path) -> list[str]:
    """The paths a selection file lists: one a line, ``#`` comments and blank lines aside."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [word for word in (line.split("#", 1)[0].strip() for line in lines) if word]


def curated_files(files: Sequence[str]) -> tuple[list[str], str] | None:
    """The curated selection of :data:`SELECTION_ENV` among ``files``, and why; ``None`` unset.

    The listed files present among ``files``, and the claims gates (:data:`ALWAYS_RUN`), which
    every check runs. Listed paths that are not among ``files`` are named in the why; a list
    that names none of them gives the claims gates alone, which :func:`run_regression_check`
    refuses to take for a check.
    """
    path = os.environ.get(SELECTION_ENV, "").strip()
    if not path:
        return None
    try:
        wanted = _selection_lines(path)
    except OSError as exc:
        raise ValueError(f"{SELECTION_ENV}={path} cannot be read: {exc}") from exc
    known = set(files)
    named = {w for w in wanted if w in known}
    picked = sorted(named | {f for f in files if ALWAYS_RUN.search(f)})
    why = f"{len(named)} curated test files ({path}) of {len(files)}, with the claims gates"
    absent = sorted({w for w in wanted if w not in known})
    if absent:
        shown = ", ".join(absent[:5]) + (f" and {len(absent) - 5} more" if len(absent) > 5 else "")
        why += f"; not test files at the head: {shown}"
    return picked, why


def picked_files(
    files: Sequence[str],
    scope: str,
    *,
    limit: int,
    pick: Callable[[Path, Sequence[str], int], None],
    path: Path,
) -> tuple[list[str], str]:
    """``files`` (and ``scope``, why) capped at ``limit`` by an agent's pick.

    When more than ``limit`` of ``files`` are reached besides :data:`ALWAYS_RUN`, ``pick``
    writes at most ``limit`` of those others to ``path``, and they run with the claims gates.
    A pick that names a file not among them, more than ``limit``, or none, or that raised,
    leaves every one of ``files`` to run.
    """
    gates = {f for f in files if ALWAYS_RUN.search(f)}
    others = [f for f in files if f not in gates]
    if len(others) <= limit:
        return list(files), scope
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    try:
        pick(path, others, limit)
        wanted = set(_selection_lines(path)) - gates
    except Exception as exc:  # an agent turn that failed picks nothing: every file runs
        problem = f"failed: {type(exc).__name__}: {exc}"
    else:
        foreign = sorted(wanted - set(others))
        if foreign:
            problem = f"names files the change does not reach: {', '.join(foreign[:5])}"
        elif len(wanted) > limit:
            problem = f"names {len(wanted)}"
        else:
            problem = "" if wanted else "names none"
    if problem:
        return list(files), f"{scope}; every one runs, as the pick of at most {limit} {problem}"
    chosen = sorted(wanted | gates)
    return chosen, f"{len(chosen)} test files an agent picked ({path}), of {scope}"


def _change_summary(
    repo: str | Path, base: str, head: str, cache_dir: Path, files: Sequence[str]
) -> str:
    """What changed from ``base`` to ``head``, for an agent picking among ``files``.

    The commits and the files they changed, and those of ``files`` whose every test failed at
    the base in the environment last recorded there: a test that failed there cannot regress.
    """
    log = (git_output(repo, "log", "--oneline", "--no-decorate", f"{base}..{head}") or "").strip()
    stat = (git_output(repo, "diff", "--no-renames", "--stat=160", base, head) or "").rstrip()
    lines = [
        f"## What changed: `git diff {base[:12]} {head[:12]}`",
        "",
        "```",
        *log.splitlines()[:40],
        "",
        *stat.splitlines()[:150],
        "```",
    ]
    at_base = test_results.recorded(cache_dir, base)
    failing = []
    for file in files:
        ids = at_base.test_ids(file) or []
        ended = [at_base.result(i) for i in ids]
        if ended and all(r is not None and r.outcome in test_results.FAILING for r in ended):
            failing.append(file)
    if failing:
        lines += [
            "",
            "## Test files whose every test failed at the base",
            "",
            "None of their tests can regress, unless the change adds one:",
            "",
            *[f"- `{f}`" for f in failing],
        ]
    return "\n".join(lines)


def affected_files(
    repo: str | Path,
    base: str | None,
    head: str,
    files: Sequence[str],
    roots: Sequence[str] = (MODELING_V2_PACKAGE, *PACKAGE_TESTS),
) -> tuple[list[str], str]:
    """The test files of ``files`` the change from ``base`` to ``head`` can reach, and why.

    Every file when there is no base, nothing changed since it, or the change
    touches code the whole package shares (:data:`SCOPED_SOURCE_DIRS`); else the test files it
    changed, those that import a module it changed, directly or through any other file under
    ``roots`` (:func:`_importers`), those beside a test helper that does (a test may start its
    helper by file path), those whose text names one (dotted, as a path, or by its last two
    parts: an over-match only runs a test more), and :data:`ALWAYS_RUN`.
    """
    if base is None:
        return list(files), "every test file (no base to compare with)"
    diff = git_output(repo, "diff", "--no-renames", "--name-only", base, head) or ""
    changed = [p for p in diff.splitlines() if p]
    if not changed:
        return list(files), "every test file (nothing changed since the base)"
    known = set(files)
    for path in changed:
        if path in known or _unreached(path):
            continue
        name = path.rsplit("/", 1)[-1]
        scoped = path.endswith(".py") and any(path.startswith(d) for d in SCOPED_SOURCE_DIRS)
        shared_init = name == "__init__.py" and f"{path.rpartition('/')[0]}/" in SCOPED_SOURCE_DIRS
        # A helper or package init of the test tree reaches its importers and the files beside it.
        test_side = path.endswith(".py") and any(path.startswith(f"{t}/") for t in PACKAGE_TESTS)
        if (not scoped and not test_side) or shared_init or name == "conftest.py":
            return list(files), f"every test file ({path} is shared by the package)"
    changed = [p for p in changed if not _unreached(p)]
    if not changed:
        chosen = sorted(f for f in files if ALWAYS_RUN.search(f))
        return (
            chosen,
            f"{len(chosen)} of {len(files)} test files (no unit test reaches what changed)",
        )
    reached = _importers(repo, head, roots, changed)
    if reached is None:
        return list(files), "every test file (the imports at the head could not be read)"
    # A helper a test starts by file path (a collective's rank matrix) reaches the tests beside it;
    # a test package's ``__init__.py`` reaches every test file under its directory.
    helpers = {p.rpartition("/")[0] for p in reached if p not in known}
    test_inits = [
        p.rpartition("/")[0] + "/"
        for p in changed
        if p.endswith("/__init__.py") and any(p.startswith(f"{t}/") for t in PACKAGE_TESTS)
    ]
    chosen = {
        f
        for f in files
        if f in reached
        or f.rpartition("/")[0] in helpers
        or f.startswith(tuple(test_inits))
        or ALWAYS_RUN.search(f)
    }
    names = sorted({n for p in changed if p.endswith(".py") for n in _module_names(p)})
    if names and files:
        args = ["grep", "-l", "-F"]
        for name in names:
            args += ["-e", name]
        found = git_output(repo, *args, head, "--", *files) or ""
        prefix = f"{head}:"
        chosen |= {
            line[len(prefix) :] if line.startswith(prefix) else line
            for line in found.splitlines()
            if line
        } & known
    picked = sorted(chosen)
    return picked, f"{len(picked)} of {len(files)} test files reach what changed"


def _named(lines: Sequence[str]) -> list[str]:
    shown = list(lines[:MAX_NAMED])
    if len(lines) > MAX_NAMED:
        shown.append(f"and {len(lines) - MAX_NAMED} more")
    return shown


def _failure(comparison: test_results.Comparison) -> str:
    head = comparison.head
    why = f": {head.message.splitlines()[0][:200]}" if head.message else ""
    was = "new, fails" if comparison.new else "passed at the base, fails now"
    return f"{head.nodeid} ({was}){why}"


def run_regression_check(
    *,
    workspace: Path,
    repo: Path,
    head: str,
    tree_at: TreeAt,
    run_job: RunJob,
    python: str,
    name: str,
    paths: Sequence[str] = PACKAGE_TESTS,
    max_files: int | None = None,
    pick: PickFiles | None = None,
    curate: bool = True,
) -> RegressionResult:
    """Run one regression check on ``head``; see the module docstring.

    ``repo`` holds both commits; ``tree_at`` gives a read-only tree at one,
    ``run_job`` runs a command line in it on the run's devices, and
    ``python`` is the tree's interpreter. ``name`` keeps this check's
    selections apart from another's. With a base, ``max_files`` caps the
    files the change reaches by ``pick``'s choice (:func:`picked_files`).
    ``curate`` off ignores :data:`SELECTION_ENV`: delivery's check runs
    every file the change reaches. The result counts what a cap left out
    (:attr:`RegressionResult.deferred`).
    """
    cache_dir = Path(workspace) / test_results.CACHE_DIRNAME
    out = Path(workspace) / SELECTIONS_DIRNAME / name
    job = dict(cache_dir=cache_dir, out=out, tree_at=tree_at, run_job=run_job, python=python)
    base = test_results.default_base(workspace, repo, head)
    listed = test_results.list_test_files(repo, head, paths)
    reached, scope = affected_files(repo, base, head, listed, roots=(MODELING_V2_PACKAGE, *paths))
    files = reached
    curated = curated_files(listed) if curate else None
    if curated is not None:
        files, why = curated
        scope = f"{why}, instead of {scope}"
        if all(ALWAYS_RUN.search(f) for f in files):
            failure = f"{SELECTION_ENV} names none of the {len(listed)} test files at the head"
            return RegressionResult(head, base, "INCOMPLETE", [failure], 0, 0, scope)
    elif max_files and pick is not None and base is not None:

        def ask(path: Path, others: Sequence[str], limit: int) -> None:
            pick(path, others, limit, _change_summary(repo, base, head, cache_dir, others))

        files, scope = picked_files(
            files, scope, limit=max_files, pick=ask, path=out / f"picked-{head[:12]}.txt"
        )
    deferred = len(set(reached) - set(files))
    selected = set(files)
    results, ran, code, tail = record_at(head, files, **job)
    if files and not results.env_digest:
        # No session at the head: nothing the jobs ran can be read.
        failure = f"the test job recorded nothing (exit {code}): {tail[-500:]}"
        return RegressionResult(head, base, "INCOMPLETE", [failure], 0, ran, scope, deferred)
    if base is None:
        # Nothing landed before the head to compare with: each failure is the head's own.
        view = test_results.head_results(results, files)
        failures = [f"{r.nodeid} (fails; no base to compare with)" for r in view.failing]
        failures += [f"{nodeid} (no result at the head)" for nodeid in view.not_run]
        verdict = "REGRESSED" if view.failing else "INCOMPLETE" if failures else "CLEAN"
        held = len(view.passed) + len(view.skipped)
        return RegressionResult(head, base, verdict, _named(failures), held, ran, scope, deferred)
    base_files = [f for f in test_results.list_test_files(repo, base, paths) if f in selected]

    def against_base() -> test_results.CompareReport:
        at_base = test_results.recorded(cache_dir, base, results.env_digest)
        return test_results.compare(results, at_base, files, base_files)

    report = against_base()
    if report.base_unknown:
        unknown = [c.head.nodeid for c in report.base_unknown]
        ran += _run_unrun(base, unknown, results.env_digest, **job)[1]
        report = against_base()
    failures = [_failure(c) for c in report.regressions]
    failures += [f"{nodeid} (no result at the head)" for nodeid in report.not_run]
    failures += [
        f"{c.head.nodeid} (fails at the head; no result at the base)" for c in report.base_unknown
    ]
    return RegressionResult(
        head, base, report.verdict, _named(failures), report.passed, ran, scope, deferred
    )


def owed(workspace: str | Path, repo: str | Path, head: str) -> list[str]:
    """The heads capped checks deferred test files at that no full check has run since.

    A capped check that passed names its head ``deferred`` in its receipt and clears nothing
    (the module docstring). A full check whose head descends from it ran what it left out: the
    change from that check's base, which the capped one did not move, reaches them again. Those
    still owed among ``head``'s ancestors, oldest first, then each receipt that does not read
    (``unreadable <file>``), which may name one; delivery runs them
    (``completion.deferred_regression_check``).
    """
    base = test_results.default_base(workspace, repo, head)
    found = []
    unreadable: list[Path] = []
    for receipt in test_results.task_receipts(workspace, unreadable):
        name = str(receipt.get("deferred") or "")
        if not name:
            continue
        commit = git_output(repo, "rev-parse", "--verify", "--quiet", f"{name}^{{commit}}") or ""
        commit = commit.strip()
        if not commit or git_output(repo, "merge-base", "--is-ancestor", commit, head) is None:
            continue
        if base and git_output(repo, "merge-base", "--is-ancestor", commit, base) is not None:
            continue
        found.append((str(receipt.get("written_at") or ""), commit))
    heads = list(dict.fromkeys(commit for _, commit in sorted(found)))
    return heads + [f"unreadable {path.name}" for path in unreadable]


__all__ = [
    "JOB_TIMEOUT_S",
    "MAX_NAMED",
    "PACKAGE_TESTS",
    "PROBE_TIMEOUT_S",
    "SELECTIONS_DIRNAME",
    "SELECTION_ENV",
    "TEST_BUDGET_S",
    "PickFiles",
    "RegressionResult",
    "SessionJobs",
    "affected_files",
    "curated_files",
    "job_runner",
    "job_timeout",
    "max_jobs",
    "owed",
    "picked_files",
    "record_at",
    "run_regression_check",
]
