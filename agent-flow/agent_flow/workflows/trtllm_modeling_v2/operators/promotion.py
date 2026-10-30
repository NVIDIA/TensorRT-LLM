# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local variants of shared catalog entries, and their promotion into the catalog.

A kernel task writes ``ops/<category>/<name>__<site>.py`` in the target, not the shared entry,
with tests beside its other ops (:func:`~.coverage.entry_paths`). Distinct declared files let
variants of one entry run concurrently; the scheduler serializes tasks declaring the same file.

A ``promote:<entry>[,<entry>...]`` kernel task merges each entry's variants into one shared
class (the union of their cells, guards and architectures) and test file. It redirects target
imports to the catalog, removes the variants, and runs the merged tests and claims gates once.

Like ``delivery.py``, this module reads files without importing the target.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..checks.delivery import Check
from .coverage import (
    MODE_PROMOTE,
    MODE_VERIFY,
    MODEL_OPS_DIRNAME,
    MODELING_V2_PACKAGE,
    VARIANT_SEPARATOR,
    QueueItem,
    entry_paths,
    imported_catalog_entries,
    model_ops_tests_dir,
    shared_entry_paths,
    split_variant,
    variant_target,
)

#: ``task.yaml``'s ``target.keep_local``: shared entries whose local variants
#: the target keeps, so no promote merges them and delivery accepts them.
KEEP_LOCAL_KEY = "keep_local"
#: The kinds besides ``kernel`` that change the target. A promote runs beside
#: them: it declares the files it writes, and a rebase that conflicts goes back
#: to the task's worker as any other does.
WRITER_KINDS: tuple[str, ...] = ("assemble", "feature", "integrate")
#: The id of the first promote task; later ones are numbered on.
PROMOTE_TASK_ID = "promote"
#: A promote's estimate: the merge, and one run of every merged test file.
PROMOTE_BASE_MINUTES, PROMOTE_MINUTES_PER_ENTRY = 30, 15

# A variant's dotted module, relative (``.ops.<category>.<name>__<site>``) or
# absolute (``...models.<family>.<target>.ops.<category>.<name>__<site>``).
_VARIANT_MODULE = re.compile(
    rf"(?:^|\.)({MODEL_OPS_DIRNAME})\.([a-z_]+)\.([a-z0-9_]+{VARIANT_SEPARATOR}[a-z0-9_]+)(?:\.|$)"
)


@dataclass(frozen=True)
class Variant:
    """One local variant on disk, ``<target>/ops/<category>/<name>__<site>.py``, and its test."""

    category: str
    name: str
    site: str
    #: The variant module, repo-relative.
    module: str
    #: Its GPU test files, repo-relative, as :func:`~.coverage.entry_paths`
    #: resolves them: the test, or a multi-rank op matrix and its rank body.
    tests: tuple[str, ...]

    @property
    def files(self) -> tuple[str, ...]:
        return (self.module, *self.tests)

    @property
    def entry(self) -> str:
        """The shared entry it is a variant of, ``<category>/<name>``."""
        return f"{self.category}/{self.name}"

    @property
    def target(self) -> str:
        """``ops/<category>/<name>__<site>``, as a dispatch names it."""
        return variant_target(self.entry, self.site)


def variant_entry(target: str) -> str:
    """``ops/<category>/<name>__<site>`` -> the shared entry ``<category>/<name>``."""
    _, category, module = target.strip("/").split("/")
    return f"{category}/{split_variant(module)[0]}"


def local_variants(repo_root: Path | str, target_rel: str) -> tuple[Variant, ...]:
    """Every local variant in the target's own ``ops/``, in path order; none without a target."""
    target = target_rel.strip().rstrip("/")
    if not target:
        return ()
    found = []
    for path in sorted((Path(repo_root) / target / MODEL_OPS_DIRNAME).glob("*/*.py")):
        name, site = split_variant(path.stem)
        if site is None:
            continue
        category = path.parent.name
        own = QueueItem(n=0, mode=MODE_VERIFY, target=f"{MODEL_OPS_DIRNAME}/{category}/{path.stem}")
        module, *tests = entry_paths(own, target, repo_root) or ()
        found.append(Variant(category, name, site, module=module, tests=tuple(tests)))
    return tuple(found)


def _imported_modules(tree: ast.AST) -> Iterable[tuple[int, str]]:
    """``(level, dotted)`` for every module an import names, and for every name imported from one."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield 0, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            yield node.level, base
            for alias in node.names:
                yield node.level, f"{base}.{alias.name}" if base else alias.name


def variant_imports(
    repo_root: Path | str, target_rel: str, *, tests: bool = True
) -> dict[str, tuple[str, ...]]:
    """Each file of the target outside its ``ops/`` that imports a local variant, and which.

    Keyed by the repo-relative file; the variants are ``ops/<category>/<name>__<site>``.
    With ``tests``, the target's tests (:func:`~.delivery.model_ops_tests_dir`)
    are read too, by their absolute imports. A variant's own test is the
    variant's, not an importer.
    """
    target = target_rel.strip().rstrip("/")
    repo = Path(repo_root)
    root = repo / target
    if not target or not root.is_dir():
        return {}
    dotted_target = target.split(f"{MODELING_V2_PACKAGE}/", 1)[-1].replace("/", ".")
    files = [
        (path, True)
        for path in sorted(root.rglob("*.py"))
        if MODEL_OPS_DIRNAME not in path.relative_to(root).parts[:1]
    ]
    if tests:
        own = {f for variant in local_variants(repo, target) for f in variant.files}
        files += [
            (path, False)
            for path in sorted((repo / model_ops_tests_dir(target)).rglob("*.py"))
            if path.relative_to(repo).as_posix() not in own
        ]
    found: dict[str, tuple[str, ...]] = {}
    for path, in_package in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        variants: set[str] = set()
        for level, dotted in _imported_modules(tree):
            match = _VARIANT_MODULE.search(dotted)
            if match is None:
                continue
            if level and not in_package:
                continue
            if level == 0 and f"{dotted_target}.{MODEL_OPS_DIRNAME}." not in dotted:
                continue
            variants.add("/".join(match.groups()))
        if variants:
            found[path.relative_to(repo).as_posix()] = tuple(sorted(variants))
    return found


def keep_local(task_data: Mapping[str, Any]) -> tuple[str, ...]:
    """The shared entries ``task.yaml`` keeps local (``target.keep_local``), ``<category>/<name>`` each."""
    block = task_data.get("target")
    raw = block.get(KEEP_LOCAL_KEY) if isinstance(block, Mapping) else None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(sorted({str(e).strip().strip("/") for e in raw if str(e).strip()}))


def check_promoted(repo_root: Path | str, target_rel: str, keep: Sequence[str] = ()) -> list[Check]:
    """No local variant waits for promotion, and the target imports none, but those kept local.

    ``keep`` names the shared entries whose variants the spec keeps.
    """
    kept = set(keep)
    problems: list[str] = []
    held = sorted({v.entry for v in local_variants(repo_root, target_rel)} & kept)
    for variant in local_variants(repo_root, target_rel):
        if variant.entry not in kept:
            problems.append(f"{variant.module} is a local variant of {variant.entry}")
    for path, variants in variant_imports(repo_root, target_rel).items():
        for target in variants:
            if variant_entry(target) not in kept:
                problems.append(f"{path} imports {target}")
    if not problems:
        detail = "no local variant waits for promotion"
        if held:
            detail += f"; kept local by target.{KEEP_LOCAL_KEY}: {', '.join(held)}"
        return [Check(name="ops:promoted", passed=True, detail=detail)]
    return [
        Check(
            name="ops:promoted",
            passed=False,
            detail="; ".join(problems)
            + ". Promote them into the shared catalog (a `promote:<category>/<name>` kernel "
            f"task) or name the entry in task.yaml `target.{KEEP_LOCAL_KEY}`",
        )
    ]


def promote_ref(entries: Iterable[str]) -> str:
    """The ``ref`` of a kernel task that promotes ``entries``."""
    return f"{MODE_PROMOTE}:{','.join(sorted(set(entries)))}"


def _foreign_copies(repo_root: Path, target_rel: str, entry: str) -> list[str]:
    """Another target's own ``ops/`` copy of ``entry`` and its test, repo-relative."""
    category, name = entry.split("/")
    models = repo_root / MODELING_V2_PACKAGE / "models"
    own = target_rel.strip().rstrip("/")
    found = []
    for path in sorted(models.glob(f"*/*/{MODEL_OPS_DIRNAME}/{category}/{name}.py")):
        target = path.parents[2].relative_to(repo_root).as_posix()
        if target == own:
            continue
        copy = QueueItem(n=0, mode=MODE_VERIFY, target=f"{MODEL_OPS_DIRNAME}/{category}/{name}")
        found.extend(entry_paths(copy, target, repo_root) or ())
    return found


def promote_touches(repo_root: Path | str, target_rel: str, entries: Sequence[str]) -> list[str]:
    """Every file a promote of ``entries`` writes, as files so no task that declares one runs beside it.

    Per entry, the shared module and its test, each local variant and its
    test, and another target's ``ops/`` copy and its test when one exists
    (that target then imports the catalog too); then the target's files that
    import a variant of them, which then import the catalog. A file that
    imports none (verify variants that hold cells only, beside a target that
    already calls the catalog entry) is not rewritten, so it is not declared:
    the target's ``modeling.py`` would queue every promote behind every other.
    """
    root = Path(repo_root)
    target = target_rel.strip().rstrip("/")
    wanted = set(entries)
    touches: list[str] = []
    for entry in sorted(wanted):
        touches.extend(shared_entry_paths(entry, root))
        for variant in local_variants(root, target):
            if variant.entry == entry:
                touches.extend(variant.files)
        touches.extend(_foreign_copies(root, target, entry))
    for path, variants in variant_imports(root, target).items():
        if any(variant_entry(v) in wanted for v in variants):
            touches.append(path)
    return list(dict.fromkeys(touches))


@dataclass(frozen=True)
class Promotion:
    """What :func:`plan` found: the entries waiting, and the task to add or why none is added."""

    #: The shared entries whose local variants wait for promotion, ``<category>/<name>`` each.
    entries: tuple[str, ...] = ()
    #: The promote task to add now, as the graph reads one (``Task.from_dict``); ``None`` when none is due.
    task: dict[str, Any] | None = None
    #: Why no task is added though entries wait; empty when one is, or when none wait.
    waiting: str = ""


def is_promote(task: Any) -> bool:
    """Whether ``task`` is a promote (a ``kernel`` task whose ``ref`` is ``promote:...``)."""
    return task.kind == "kernel" and str(task.ref).startswith(f"{MODE_PROMOTE}:")


_is_promote = is_promote


def _live(task: Any, status: Mapping[str, str]) -> bool:
    """Running, or pending with every dependency done: it is working or about to start."""
    if task.status == "running":
        return True
    return task.status == "pending" and all(status.get(dep) == "done" for dep in task.depends_on)


def _variant_entries(task: Any, target_rel: str) -> set[str]:
    """The shared entries a (non-promote) kernel task lands a local variant of, by its files."""
    if task.kind != "kernel" or _is_promote(task):
        return set()
    target = target_rel.strip().rstrip("/")
    found = set()
    for path in task.touches:
        if not (path.startswith(f"{target}/{MODEL_OPS_DIRNAME}/") and path.endswith(".py")):
            continue
        rel = path[len(target) + 1 : -len(".py")]
        if rel.count("/") == 2 and split_variant(Path(rel).name)[1] is not None:
            found.add(variant_entry(rel))
    return found


def catalog_callers(repo_root: Path | str, target_rel: str) -> set[str]:
    """The catalog entries (``<category>/<name>``) the target's package imports directly."""
    root = Path(repo_root) / target_rel.strip().rstrip("/")
    found: set[str] = set()
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*.py")):
        if MODEL_OPS_DIRNAME in path.relative_to(root).parts[:1]:
            continue
        try:
            found.update(f"{c}/{n}" for c, n in imported_catalog_entries(path))
        except (OSError, SyntaxError, ValueError):
            continue
    return found


def _promoted_entries(task: Any) -> set[str]:
    """The entries a promote task's ``ref`` (``promote:<entry>,...``) names."""
    return {e for e in str(task.ref).partition(":")[2].split(",") if e}


def plan(
    tasks: Sequence[Any], repo_root: Path | str, target_rel: str, keep: Sequence[str] = ()
) -> Promotion:
    """Which of the target's local variants wait for promotion, and the promote due now if any.

    An entry is due when no landing kernel task is pending/running, no covering promote is open
    or failed (the planner must retry or reshape failures), and the target imports every variant.
    This rewrites existing imports, not ones a task has yet to write. Once no target-changing task
    (:data:`WRITER_KINDS`) is running or ready, all waiting entries are due.

    All due entries share one promote. Its declared files (:func:`promote_touches`) exclude
    conflicting tasks but allow unrelated work alongside it. ``tasks`` are the graph's
    ``id``, ``kind``, ``status``, ``ref``, ``touches`` and ``depends_on`` records.
    """
    kept = set(keep)
    entries = tuple(sorted({v.entry for v in local_variants(repo_root, target_rel)} - kept))
    if not entries:
        return Promotion()
    covering: dict[str, Any] = {}
    for task in tasks:
        if _is_promote(task):
            for entry in _promoted_entries(task):
                covering[entry] = task
    landing = {
        entry
        for t in tasks
        if t.status in ("pending", "running")
        for entry in _variant_entries(t, target_rel)
    }
    status = {t.id: t.status for t in tasks}
    writing = any(t.kind in WRITER_KINDS and _live(t, status) for t in tasks)
    # The code that calls a variant is the package's; a test that imports one calls nothing.
    imported = {
        v for vs in variant_imports(repo_root, target_rel, tests=False).values() for v in vs
    }
    # A variant the target imports no file of is still to be wired in, unless the
    # target calls the shared entry itself: the variant then holds verified cells
    # only, and its merge rewrites nothing a writer could still be changing.
    unused = {
        v.entry
        for v in local_variants(repo_root, target_rel)
        if f"{MODEL_OPS_DIRNAME}/{v.category}/{v.name}{VARIANT_SEPARATOR}{v.site}" not in imported
    } - catalog_callers(repo_root, target_rel)
    due, held = [], []
    for entry in entries:
        promote = covering.get(entry)
        if promote is not None and promote.status in ("pending", "running"):
            held.append(f"{entry} (task {promote.id} promotes it)")
        elif promote is not None and promote.status == "failed":
            held.append(
                f"{entry} (promote {promote.id} failed; the planner retries or reshapes it)"
            )
        elif entry in landing:
            held.append(f"{entry} (a variant of it is still to land)")
        elif entry in unused and writing:
            held.append(f"{entry} (the target does not import every variant of it yet)")
        else:
            due.append(entry)
    waiting = "; ".join(held)
    if not due:
        return Promotion(entries, waiting=waiting)
    taken = {t.id for t in tasks}
    task_id, n = PROMOTE_TASK_ID, 1
    while task_id in taken:
        n += 1
        task_id = f"{PROMOTE_TASK_ID}-{n}"
    names = ", ".join(f"`{e}`" for e in due)
    return Promotion(
        entries,
        task={
            "id": task_id,
            "kind": "kernel",
            "goal": (
                f"promote the target's local variants of {names} into the shared catalog: one "
                f"entry and one test file each, the union of their cells, guards and ARCHS; the "
                f"target imports the catalog, the variants are removed; the merged tests and "
                f"the claims gates run once"
            ),
            "touches": promote_touches(repo_root, target_rel, due),
            "resources": {
                "gpus": 1,
                "est_minutes": PROMOTE_BASE_MINUTES + PROMOTE_MINUTES_PER_ENTRY * len(due),
            },
            "priority": "required",
            "ref": promote_ref(due),
        },
        waiting=waiting,
    )


__all__ = [
    "KEEP_LOCAL_KEY",
    "catalog_callers",
    "is_promote",
    "PROMOTE_TASK_ID",
    "Promotion",
    "Variant",
    "WRITER_KINDS",
    "check_promoted",
    "keep_local",
    "local_variants",
    "plan",
    "promote_ref",
    "promote_touches",
    "variant_entry",
    "variant_imports",
]
