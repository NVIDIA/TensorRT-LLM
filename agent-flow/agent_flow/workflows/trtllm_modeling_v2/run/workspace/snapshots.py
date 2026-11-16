# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What one check measured: the pinned tree it ran on, and what its commands imported.

The checkout moves while checks run, one landed task at a time, so a check measures a pin
instead: the ``pin`` of the checkout's head and its ``at`` (``integration.py``), a read-only
tree of that one commit which every check of the commit shares. Making, holding and removing
those trees is done there; nothing here leases one.

What a check imported is read from what its commands print (:class:`CheckTree`): the spool's
resolve hook names the file each process imported the package from, and the engine driver
names it once per engine. A file inside a pinned tree names that tree's commit; any other
names a tree no commit pins. A check whose commands imported more than one tree, or a tree no
commit pins, is void.
"""

from __future__ import annotations

import os
import re
import shlex
import threading
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Collection, Mapping

from ...checks import check_runtime
from ...checks import verdict as check_result
from ...checks.harness import workload_io
from ...infra.compute.executor import SpoolCommandExecutor
from ...infra.compute.provenance import checkout_root

_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
#: The tree a check's processes run, as each one's origin record names it (``resolvehook``).
TREE_ENV = "MV2FLOW_TREE"
#: ``[mv2flow-resolve] pid=.. rank=.. module=<module> origin=<file>``, as the resolve hook prints it.
_RESOLVED = re.compile(r"\[mv2flow-resolve\][^\n]*?\bmodule=(\S+) origin=([^\n]*)")


@dataclass
class CheckTree:
    """One check's tree: the commit it measures, that commit's pinned tree, and what it imported.

    ``marker`` is the target's marker for ``commit``. ``uncommitted`` are the
    files a gate reads that the checkout held uncommitted when the check
    started, which no pin holds. ``snapshots`` is the run's trees: they name
    the commit of any other pinned tree a command imported from.
    """

    commit: str
    tree: Path
    marker: str = ""
    uncommitted: tuple[str, ...] = ()
    snapshots: Any = None
    #: What each command imported, in order: per command, the commit of each
    #: snapshot, or the path of each checkout, its processes imported from.
    commands: list[tuple[str, ...]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --------------------------------------------------------------- the spec

    def spec(self, task_data: Mapping[str, Any], tracked: Collection[str]) -> dict[str, Any]:
        """``task_data`` with the target in the snapshot.

        ``trtllm_repo_path`` names the snapshot, so whatever a gate resolves
        under it (the quickstart, a prompt's image, a relative anchor table) is
        the commit's. So does every word of the spec, in whatever block, that
        names a path of the checkout by its absolute path and that the commit
        tracks (in ``tracked``): a deployment's options file, a benchmark's
        dataset, a prompt file. No list of blocks says where such a word may
        stand, so a block the spec gains is covered. A path the commit does
        not track, a virtualenv built in the checkout say, stays where it is.
        """
        repo = str(task_data.get("trtllm_repo_path") or "").rstrip("/")
        if not repo:
            return {**task_data, "trtllm_repo_path": str(self.tree)}
        inside = re.compile(re.escape(repo + "/") + r"([^\s'\":;,=]+)")

        def held(rel: str) -> bool:
            rel = rel.rstrip("/")
            return rel in tracked or any(path.startswith(rel + "/") for path in tracked)

        def move(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, str) and repo + "/" in value:
                return inside.sub(
                    lambda m: f"{self.tree}/{m.group(1)}" if held(m.group(1)) else m.group(0), value
                )
            return value

        return {**move(task_data), "trtllm_repo_path": str(self.tree)}

    # ------------------------------------------------------------ the commands

    def runner(self, runner: Callable[..., Any], executor: Any) -> Callable[..., Any]:
        """``runner`` for this check's commands, noting what each imported.

        ``executor`` is what carries the commands. On the run's spool a command is a job that
        names the pin, and the node the job runs on makes the pin's tree and enters it before
        the command (``pintree.py run``): the command names no path of the tree. The path
        would be this host's, and the lead that runs a job of several nodes has no tree at
        all: each of its steps makes its own. With ``None``, or an executor of this host, the
        commands are this host's shell's and enter the tree made here
        (``check_runtime.in_tree``).
        """
        hook = shlex.quote(str(check_runtime.RANK_HOOK_DIR))
        prefix = f'PYTHONPATH={hook}"${{PYTHONPATH:+:$PYTHONPATH}}" && '
        if isinstance(executor, SpoolCommandExecutor):
            inside = runner
        else:
            inside = check_runtime.in_tree(runner, self.tree)
            prefix = f"{TREE_ENV}={shlex.quote(str(Path(self.tree).resolve()))} {prefix}"

        def run(command: str, env: Mapping[str, str] | None = None, timeout: int = 1800) -> Any:
            result = inside(f"export {prefix}{command}", env, timeout)
            self.observe(f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}")
            return result

        return run

    def observe(self, output: str) -> bool:
        """Note what one command's ``output`` says its processes imported; whether it said."""
        roots = {
            checkout_root(module, origin.strip())
            for module, origin in _RESOLVED.findall(output)
            if origin.strip() not in ("", "None")
        }
        for record in workload_io.parse_marked(output, workload_io.DRIVER_MARKER):
            if record.get("tensorrt_llm"):
                roots.add(checkout_root("tensorrt_llm", str(record["tensorrt_llm"])))
        if roots:
            with self._lock:
                self.commands.append(tuple(sorted(self._name(root) for root in roots)))
        return bool(roots)

    def _name(self, root: str) -> str:
        if os.path.realpath(root) == os.path.realpath(self.tree):
            return self.commit
        found = self.snapshots.commit_of(root) if self.snapshots is not None else None
        return found or root

    def imported(self) -> tuple[str, ...]:
        """What each command that said imported: a commit, or ``+``-joined where it was several."""
        return tuple("+".join(names) for names in self.commands)

    def void(self) -> str:
        """Why what the commands imported is not this one commit; ``""`` when it is, or none said."""
        seen = {name for names in self.commands for name in names}
        others = sorted(seen - {self.commit})
        if not others:
            return ""
        named = [
            name[:10] if _COMMIT.match(name) else f"{name} (a checkout no commit pins)"
            for name in others
        ]
        which = "some of its commands" if self.commit in seen else "its commands"
        return (
            f"{which} imported the package under test from {', '.join(named)}, not from the "
            f"snapshot of {self.commit[:10]} the check measures"
        )

    # ---------------------------------------------------------------- results

    def stamp(self, result: check_result.GateResult) -> check_result.GateResult:
        """``result`` as measured on this commit: its description, marker and imports, void or not."""
        warnings = ()
        if self.uncommitted:
            shown = ", ".join(self.uncommitted[:5])
            more = f" and {len(self.uncommitted) - 5} more" if len(self.uncommitted) > 5 else ""
            warnings = (
                f"the checkout holds {shown}{more} uncommitted; this check measured "
                f"{self.commit[:10]} without them",
            )
        return check_result.measured_on(
            result,
            tree_state=self.description,
            marker=self.marker,
            imported=self.imported(),
            void=self.void(),
            warnings=warnings,
        )

    @cached_property
    def description(self) -> str:
        """The commit as a gate line names a tree (``result.describe_tree_state``)."""
        described = check_result.describe_tree_state(self.tree)
        return self.commit[:10] if described == check_result.UNKNOWN_TREE_STATE else described

    def settle(self, measure: Callable[[], check_result.GateResult]) -> check_result.GateResult:
        """``measure()``, once, stamped with what its commands imported.

        A void result (they imported the package from somewhere else than this pinned
        tree) is returned void and is not measured again here: the tree is pinned, so the
        same command imports from the same place, and a second load would say the same.
        A job lost with its allocation is run again by the spool, and by nothing else; a
        void check fails its task, whose repair is the next measurement.
        """
        with self._lock:
            self.commands.clear()
        return self.stamp(measure())


def void_result(name: str, member: str | None, why: str) -> check_result.GateResult:
    """A check that measured nothing because it got no tree to measure."""
    return check_result.GateResult(
        name=name, member=member, passed=False, void=why, detail=why, explained=True
    )


__all__ = [
    "CheckTree",
    "void_result",
]
