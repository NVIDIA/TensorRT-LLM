# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence freshness: which recorded verdicts still hold on the tree as it is now.

The ledger (:mod:`.ledger`) stores each verdict with the code it depends on.
This module turns that into three answers:

1. **Which verdicts are stale.** :func:`assess` compares each verdict against
   the current tree: its dependency files, its dependency set, the native
   build, the unit's spec, and the defaults of the switches it did not set
   itself. A later commit that touches the target makes every earlier unit's
   verdict on that target stale. A flipped default makes stale every verdict
   that ran under the old one, including those of earlier units.
2. **What to run again.** :func:`rerun_queue` lists the latest verdict of
   each ``(gate, unit, member)`` that is stale, did not pass, or -- at
   delivery -- was measured on a different tree than the one being
   delivered. :func:`drain` runs that queue through a caller-supplied
   re-runner and returns what is still owed.
3. **Whether the tree under merge was regressed.** A *full regression* is a
   set of passing verdicts covering every required key, all measured on one
   tree and one native build. :func:`complete_regression` recognises one and
   the ledger records it; :func:`regression_binding` refuses a merge whose
   tree or native build differs from the last one. This is the rule the
   DeepSeek-V4.1 run lacked: its final head carried a functional change and a
   lint series that landed after the last regression, so no gate ever ran on
   the code that was delivered.

Reading the tree costs a git call or two and one hash per dependency file.
The switch-history check adds one ``git diff -G`` per distinct verdict commit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .ledger import (
    DEFAULT_SWITCH_PREFIXES,
    KIND_REGRESSION,
    SCHEMA_VERSION,
    Ledger,
    LedgerPolicy,
    NativeBuild,
    TreeIdentity,
    Verdict,
    dependency_paths,
    harvest_paths,
    harvest_switch_defaults,
    hash_paths,
    is_git_repo,
    load_task,
    native_build,
    run_git,
    spec_fingerprint,
    tree_identity,
    utc_now,
)

RERUN_QUEUE_FILENAME = "rerun_queue.json"
_REASON_PATH_CAP = 5


# -------------------------------------------------------------- regressions


@dataclass(frozen=True)
class Regression:
    """A complete set of passing verdicts, every one measured on one tree and build."""

    tree: TreeIdentity
    native: NativeBuild
    verdict_ids: tuple[str, ...]
    keys: tuple[tuple[str, str, str], ...]
    timestamp: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": KIND_REGRESSION,
            "schema": SCHEMA_VERSION,
            "timestamp": self.timestamp or utc_now(),
            "tree": self.tree.to_json(),
            "native_build": self.native.to_json(),
            "verdict_ids": list(self.verdict_ids),
            "keys": [list(k) for k in self.keys],
        }

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> "Regression":
        return cls(
            tree=TreeIdentity.from_json(raw.get("tree")),
            native=NativeBuild.from_json(raw.get("native_build")),
            verdict_ids=tuple(str(v) for v in raw.get("verdict_ids") or ()),
            keys=tuple(
                tuple(str(p) for p in k)  # type: ignore[misc]
                for k in raw.get("keys") or ()
                if isinstance(k, (list, tuple)) and len(k) == 3
            ),
            timestamp=str(raw.get("timestamp") or ""),
        )


def regressions(ledger: Ledger) -> list[Regression]:
    return [Regression.from_record(r) for r in ledger.records() if r.get("kind") == KIND_REGRESSION]


def last_regression(ledger: Ledger) -> Regression | None:
    found = regressions(ledger)
    return found[-1] if found else None


def record_regression(ledger: Ledger, regression: Regression) -> Regression:
    record = regression.to_record()
    ledger.append(record)
    return replace(regression, timestamp=record["timestamp"])


# -------------------------------------------------------------- invalidation


@dataclass(frozen=True)
class Assessment:
    """A verdict and the reasons it no longer holds (none means fresh)."""

    verdict: Verdict
    reasons: tuple[str, ...] = ()

    @property
    def stale(self) -> bool:
        return bool(self.reasons)


def _capped(prefix: str, paths: Sequence[str]) -> str:
    shown = ", ".join(paths[:_REASON_PATH_CAP])
    more = f" (+{len(paths) - _REASON_PATH_CAP} more)" if len(paths) > _REASON_PATH_CAP else ""
    return f"{prefix}: {shown}{more}"


def switch_history_flips(
    repo_root: str | Path,
    commit: str,
    *,
    prefixes: Sequence[str] = DEFAULT_SWITCH_PREFIXES,
) -> dict[str, tuple[str | None, str | None]]:
    """Switch defaults that differ between ``commit`` and the working tree.

    Only files whose diff touches a line naming a switch are read, which
    ``git diff -G`` selects without reading anything else. Returns
    ``{name: (old_default, new_default)}``; ``{}`` when git cannot answer.
    """
    if not commit or not prefixes:
        return {}
    pattern = "(" + "|".join(re.escape(p) for p in prefixes) + ")"
    out = run_git(repo_root, "diff", "--name-only", "-G", pattern, commit, "--", "*.py")
    if out is None:
        return {}
    root = Path(repo_root)
    old_all: dict[str, set[str | None]] = {}
    new_all: dict[str, set[str | None]] = {}
    for rel in (line.strip() for line in out.splitlines() if line.strip()):
        old_text = run_git(repo_root, "show", f"{commit}:{rel}") or ""
        try:
            new_text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            new_text = ""
        for name, default in harvest_switch_defaults(old_text, prefixes=prefixes).items():
            old_all.setdefault(name, set()).add(default)
        for name, default in harvest_switch_defaults(new_text, prefixes=prefixes).items():
            new_all.setdefault(name, set()).add(default)

    def merged(values: set[str | None] | None) -> str | None:
        if not values:
            return None
        if len(values) == 1:
            return next(iter(values))
        return " | ".join(sorted("None" if v is None else v for v in values))

    flips: dict[str, tuple[str | None, str | None]] = {}
    for name in sorted(set(old_all) & set(new_all)):
        old, new = merged(old_all[name]), merged(new_all[name])
        if old != new:
            flips[name] = (old, new)
    return flips


def assess(
    verdicts: Iterable[Verdict],
    repo_root: str | Path,
    *,
    task_data: Mapping[str, Any] | None = None,
    policy: LedgerPolicy = LedgerPolicy(),
    native: NativeBuild | None = None,
    check_history: bool = True,
) -> list[Assessment]:
    """Decide, against the tree at ``repo_root`` now, which verdicts still hold.

    A verdict is stale when:

    - a dependency file's content differs from what was measured;
    - a file joined the target's dependency set (a new module, a newly
      imported catalog entry);
    - the native build's fingerprint differs (both known);
    - ``task_data`` is given and the unit's spec projection differs;
    - a switch the verdict did not set explicitly has a different default
      now. Checked two ways: against the defaults recorded with the verdict,
      and, with ``check_history``, against git history from the verdict's
      commit, which also sees switches in files outside the recorded scope.
    """
    root = Path(repo_root)
    native = native if native is not None else native_build(root, globs=policy.native_globs)
    current_deps: dict[str, list[str]] = {}
    history: dict[str, dict[str, tuple[str | None, str | None]]] = {}
    out: list[Assessment] = []
    for verdict in verdicts:
        reasons: list[str] = []
        recorded = dict(verdict.dependencies)
        now = hash_paths(root, recorded)
        changed = sorted(p for p, h in recorded.items() if now.get(p) != h)
        if changed:
            reasons.append(_capped("code changed", changed))
        if verdict.target_rel:
            if verdict.target_rel not in current_deps:
                current_deps[verdict.target_rel] = dependency_paths(
                    root, verdict.target_rel, extra=policy.extra_dependencies
                )
            added = sorted(set(current_deps[verdict.target_rel]) - set(recorded))
            if added:
                reasons.append(_capped("new dependency", added))
        if (
            verdict.native.known
            and native.known
            and verdict.native.fingerprint != native.fingerprint
        ):
            reasons.append(f"native build changed: {verdict.native.short()} -> {native.short()}")
        if task_data is not None and verdict.spec_fingerprint:
            if spec_fingerprint(task_data, verdict.unit) != verdict.spec_fingerprint:
                reasons.append(f"task spec for unit {verdict.unit or '-'} changed")
        flips: dict[str, tuple[str | None, str | None]] = {}
        if verdict.switch_defaults:
            scope = [*recorded, *policy.switch_sources]
            current = harvest_paths(root, scope, prefixes=policy.switch_prefixes)
            for name, old in verdict.switch_defaults.items():
                if name in current and current[name] != old:
                    flips[name] = (old, current[name])
        if check_history and verdict.tree.commit:
            if verdict.tree.commit not in history:
                history[verdict.tree.commit] = switch_history_flips(
                    root, verdict.tree.commit, prefixes=policy.switch_prefixes
                )
            flips.update(history[verdict.tree.commit])
        for name, (old, new) in sorted(flips.items()):
            if name in verdict.env:
                continue
            reasons.append(f"default switch flipped: {name} {old!r} -> {new!r}")
        out.append(Assessment(verdict=verdict, reasons=tuple(reasons)))
    return out


# ------------------------------------------------------------- re-run queue


@dataclass(frozen=True)
class RerunEntry:
    gate: str
    unit: str
    member: str
    reasons: tuple[str, ...]
    verdict_id: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.gate, self.unit, self.member)

    def to_json(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "unit": self.unit,
            "member": self.member,
            "reasons": list(self.reasons),
            "verdict_id": self.verdict_id,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "RerunEntry":
        return cls(
            gate=str(raw.get("gate") or ""),
            unit=str(raw.get("unit") or ""),
            member=str(raw.get("member") or ""),
            reasons=tuple(str(r) for r in raw.get("reasons") or ()),
            verdict_id=str(raw.get("verdict_id") or ""),
        )


def rerun_queue(
    assessments: Iterable[Assessment],
    *,
    tree: TreeIdentity | None = None,
    native: NativeBuild | None = None,
    scope: Callable[[Verdict], bool] | None = None,
) -> list[RerunEntry]:
    """What must be measured again, in the order the verdicts were recorded.

    A latest verdict is queued when it is stale, when it did not pass, or --
    given ``tree`` -- when it was measured on a different tree or native
    build than ``tree``/``native``. The last is what binds the final
    regression to the tree under merge: a verdict carried over from an
    earlier commit is evidence about that commit. ``scope`` drops verdicts
    the run no longer requires (a unit or member the spec no longer lists).
    """
    queue: list[RerunEntry] = []
    for item in assessments:
        verdict = item.verdict
        if scope is not None and not scope(verdict):
            continue
        reasons = list(item.reasons)
        if not verdict.passed:
            reasons.append(f"last verdict {verdict.outcome}")
        if tree is not None and tree.known:
            if verdict.tree.key != tree.key:
                reasons.append(f"measured on tree {verdict.tree.short()}, not on {tree.short()}")
            elif native is not None and verdict.native.fingerprint != native.fingerprint:
                reasons.append(
                    f"measured on native build {verdict.native.short()}, not on {native.short()}"
                )
        if reasons:
            queue.append(
                RerunEntry(
                    gate=verdict.gate,
                    unit=verdict.unit,
                    member=verdict.member,
                    reasons=tuple(reasons),
                    verdict_id=verdict.id,
                )
            )
    return queue


def write_rerun_queue(workspace: str | Path, entries: Sequence[RerunEntry]) -> Path:
    """Persist the queue beside the ledger, atomically."""
    path = Path(workspace) / RERUN_QUEUE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(
        json.dumps(
            {"generated_at": utc_now(), "entries": [e.to_json() for e in entries]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def read_rerun_queue(workspace: str | Path) -> list[RerunEntry]:
    path = Path(workspace) / RERUN_QUEUE_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [RerunEntry.from_json(e) for e in raw.get("entries") or () if isinstance(e, Mapping)]


def drain(
    queue: Sequence[RerunEntry], run: Callable[[RerunEntry], bool | None]
) -> list[RerunEntry]:
    """Re-run each entry with ``run`` and return the ones still owed.

    ``run`` returns True when it re-measured the entry (the new verdict is in
    the ledger whatever it says), False when the re-run failed, and None when
    it does not know how to re-run that gate. Both of the latter stay queued.
    """
    remaining: list[RerunEntry] = []
    for entry in queue:
        outcome = run(entry)
        if outcome is not True:
            note = "re-run failed" if outcome is False else "no automatic re-run for this gate"
            remaining.append(replace(entry, reasons=(*entry.reasons, note)))
    return remaining


# ---------------------------------------------------------------- regression


def complete_regression(
    latest: Mapping[tuple[str, str, str], Verdict],
    tree: TreeIdentity,
    native: NativeBuild,
    *,
    scope: Callable[[Verdict], bool] | None = None,
) -> Regression | None:
    """A regression record if every in-scope latest verdict passed on ``tree``/``native``."""
    chosen = [v for v in latest.values() if scope is None or scope(v)]
    if not chosen or not tree.known:
        return None
    for verdict in chosen:
        if not verdict.passed or verdict.tree.key != tree.key:
            return None
        if verdict.native.fingerprint != native.fingerprint:
            return None
    return Regression(
        tree=tree,
        native=native,
        verdict_ids=tuple(v.id for v in chosen),
        keys=tuple(v.key for v in chosen),
        timestamp=utc_now(),
    )


@dataclass(frozen=True)
class Binding:
    """Whether the tree under merge is the tree of the last full regression."""

    ok: bool
    detail: str
    #: False outside a git work tree, where no claim about a tree can be made.
    claim: bool = True


def regression_binding(
    ledger: Ledger,
    repo_root: str | Path,
    *,
    rev: str = "HEAD",
    native_root: str | Path | None = None,
    policy: LedgerPolicy = LedgerPolicy(),
) -> Binding:
    """Compare the tree under merge, and its native build, with the last regression.

    The tree under merge is the commit ``rev`` -- uncommitted changes are not
    merged -- and its native build is the one under ``native_root`` (the repo
    itself by default).
    """
    if not is_git_repo(repo_root):
        return Binding(ok=True, claim=False, detail="not a git work tree; no tree to bind")
    merge_tree = tree_identity(repo_root, rev=rev)
    merge_native = native_build(native_root or repo_root, globs=policy.native_globs)
    last = last_regression(ledger)
    if last is None:
        return Binding(
            ok=False,
            detail=(
                f"no full regression is recorded; the tree under merge is "
                f"{merge_tree.short()} (native {merge_native.short()})"
            ),
        )
    problems: list[str] = []
    if last.tree.key != merge_tree.key:
        problems.append(
            f"the last full regression ran on {last.tree.short()} "
            f"({last.tree.tree[:12] or '?'}), the tree under merge is {merge_tree.short()} "
            f"({merge_tree.tree[:12]})"
        )
    if last.native.fingerprint != merge_native.fingerprint:
        problems.append(
            f"the last full regression ran on native build {last.native.short()}, "
            f"the tree under merge has {merge_native.short()}"
        )
    if problems:
        return Binding(ok=False, detail="; ".join(problems))
    return Binding(
        ok=True,
        detail=(
            f"the last full regression ({len(last.keys)} verdicts, {last.timestamp}) ran on "
            f"{merge_tree.short()} with native {merge_native.short()}"
        ),
    )


# ----------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """``status`` and ``check-merge`` over a workspace's ledger."""
    parser = argparse.ArgumentParser(prog="staircase-freshness", description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="list stale or failing verdicts and the re-run queue")
    status.add_argument("workspace")
    status.add_argument("--repo", required=True)
    status.add_argument("--task", help="task.yaml, to detect spec changes")
    status.add_argument("--write-queue", action="store_true")

    merge = sub.add_parser(
        "check-merge", help="refuse unless the tree under merge was fully regressed"
    )
    merge.add_argument("workspace")
    merge.add_argument("--repo", required=True)
    merge.add_argument("--rev", default="HEAD")
    merge.add_argument("--native-root", help="where the native build lives, if not in --repo")

    args = parser.parse_args(argv)
    ledger = Ledger.in_workspace(args.workspace)
    if args.command == "status":
        assessed = assess(ledger.latest().values(), args.repo, task_data=load_task(args.task))
        queue = rerun_queue(assessed)
        for item in assessed:
            state = "STALE" if item.stale else "fresh"
            sys.stdout.write(
                f"{state:5} {item.verdict.outcome:7} {item.verdict.label()} "
                f"tree {item.verdict.tree.short()}\n"
            )
            for reason in item.reasons:
                sys.stdout.write(f"        - {reason}\n")
        if args.write_queue:
            sys.stdout.write(f"queue: {write_rerun_queue(args.workspace, queue)}\n")
        return 1 if queue else 0
    binding = regression_binding(ledger, args.repo, rev=args.rev, native_root=args.native_root)
    sys.stdout.write(f"{'ok' if binding.ok else 'REFUSED'}: {binding.detail}\n")
    return 0 if binding.ok else 1


__all__ = [
    "Assessment",
    "Binding",
    "RERUN_QUEUE_FILENAME",
    "Regression",
    "RerunEntry",
    "assess",
    "complete_regression",
    "drain",
    "last_regression",
    "read_rerun_queue",
    "record_regression",
    "regression_binding",
    "regressions",
    "rerun_queue",
    "switch_history_flips",
    "write_rerun_queue",
]


if __name__ == "__main__":
    sys.exit(main())
