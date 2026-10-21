# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A kernel task: one smith turn for exactly one catalog entry, or a promote of several.

An onboard or verify of a shared entry lands its local variant in the
target's own ``ops/`` (:meth:`SmithStage._landing`), so tasks on one entry run
side by side; a ``promote`` merges those variants into the shared catalog
(``onboard/promotion.py``).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

from agent_flow.console import print_message

from ...infra.environment import scheduler_owns_devices
from ...onboard import promotion as promotion_mod
from ...onboard.delivery import MODEL_OPS_DIRNAME, model_ops_tests_dir
from ...onboard.vocabulary import (
    HIGHEST_REUSING_TIER,
    MODE_ONBOARD,
    MODE_PROMOTE,
    MODE_VERIFY,
    TIER_DESCRIPTION,
    TIER_KERNEL_LIBRARY,
    QueueItem,
    entry_paths,
    shared_entry_paths,
    split_variant,
)
from ..covered import before_smith
from ..progress import (
    BUILD_STAGE,
    DECISIONS_DIRNAME,
    SMITH_DECISIONS,
    collect_decision_file,
    current_decision,
    decision_file_instruction,
    find_entries,
    mark_decision_dispatch,
    uncertified_report,
)

if TYPE_CHECKING:
    from ..graph import Task

#: The smith decision whose commits are integrated only when they stay inside the entry.
FAILED_RECEIPT = "FAILED_RECEIPT"


def entry_module(target: str) -> str:
    """The tail every path of ``target``'s entry module ends in.

    ``target`` is a dispatch's ``<category>/<name>`` (the shared catalog) or
    ``ops/<category>/<name>`` (the run's target's own); see the entry spec.
    """
    parts = target.strip("/").split("/")
    home = "ops" if parts[0] == "ops" and len(parts) > 2 else "catalog"
    return f"{home}/{'/'.join(parts[-2:])}.py"


def is_entry_file(path: str, target: str) -> bool:
    """Whether the repo-relative ``path`` is one of the files ``target``'s entry spans.

    The entry module, its GPU test (with a collective's rank body beside it),
    and the ``__init__.py`` of a new ``ops/<category>/``: what a smith writes
    for one entry, in both trees.
    """
    parts = target.strip("/").split("/")
    if len(parts) < 2:
        return False
    category, name = parts[-2], parts[-1]
    file = PurePosixPath(path)
    if path.startswith("tensorrt_llm/"):
        if path.endswith("/" + entry_module(target)):
            return True
        return (
            parts[0] == "ops"
            and file.name == "__init__.py"
            and path.endswith(f"/ops/{category}/__init__.py")
        )
    if path.startswith("tests/") and file.parent.name == category:
        stem = file.stem
        return stem == f"test_modeling_v2_{name}" or stem.startswith(
            (f"test_modeling_v2_{name}_", f"_{name}_")
        )
    return False


def _names_entry(entry: Any, target: str) -> bool:
    """Whether a decision's or a step's ``entry`` names ``target``'s entry, by path or by target.

    A local variant of a shared entry (``.../ops/<category>/<name>__<site>.py``)
    names the shared entry too: it is that entry, landed in a target first.
    """
    text = str(entry or "").strip().replace("\\", "/")
    module = entry_module(target)
    if not text:
        return False
    if text == target.strip("/") or text == module or text.endswith("/" + module):
        return True
    parts = target.strip("/").split("/")
    path = PurePosixPath(text)
    name, site = split_variant(path.stem)
    return (
        len(parts) == 2
        and site is not None
        and path.parent.name == parts[0]
        and path.parent.parent.name == MODEL_OPS_DIRNAME
        and name == parts[1]
    )


def _tier(value: Any) -> int | None:
    """A sourcing rung, when ``value`` is one."""
    try:
        tier = int(value)
    except (TypeError, ValueError):
        return None
    return tier if tier in TIER_DESCRIPTION else None


class SmithStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    def _smith_entry_for(self, item: QueueItem) -> dict | None:
        """The progress entry a smith wrote for this queue item.

        A concurrent batch appends several smith entries, so "the latest
        smith entry" is not this item's. Matched on the queue item's own
        `n` where the collector stamped one, and on the entry path or the
        summary text otherwise.
        """
        rows = find_entries(self.progress_path, stage=BUILD_STAGE, agent="smith")
        for row in reversed(rows):
            if str(row.get("iteration", "")) == str(item.n):
                return row
        for row in reversed(rows):
            if item.target in str(row.get("entry", "")) or item.target in str(
                row.get("summary", "")
            ):
                return row
        return None

    def _smith_decision_for(self, item: QueueItem) -> str | None:
        """The decision from the smith entry for this queue item.

        A failed receipt carries, on its second line, what became of its
        commits (:meth:`_land_failed_receipt`), and a landed one what it left
        uncertified: the planner reads it with the task's event.
        """
        row = self._smith_entry_for(item)
        if row is None:
            return None
        decision = str(row.get("decision", "")).strip().upper()
        if decision not in SMITH_DECISIONS:
            return None
        landed = self.__dict__.get("_receipt_notes", {}).get(item.n)
        if landed and decision == FAILED_RECEIPT:
            return f"{decision}\n{landed}"
        # What a landed entry still does not certify is the planner's to plan.
        said = uncertified_report(row.get("uncertified")) if decision == "LANDED" else ""
        return f"{decision}\n{said}" if said else decision

    def _run_one_smith(self, item: QueueItem, note: str = "", task: Task | None = None) -> str:
        """One catalog entry, in its own layer. Called from the batch's pool.

        **The decision file is collected here.** Unless it is, `progress.yaml`'s
        `build_stage` stays empty, `_smith_decision_for` finds no decision, and
        the whole batch scores FAILED.

        The iteration is the queue item's `n`, not a turn counter: smiths run
        concurrently, so a counter would race, and `n` is the only number
        that identifies which entry a decision is about.

        ``note`` is prepended context for a retry -- see `_retry_note`.
        ``task`` is the graph's kernel task the dispatch serves: its goal is
        the turn's instruction, and its worktree is where a failed receipt's
        commits are integrated from (:meth:`_land_failed_receipt`).

        **A resume adopts an answer the previous process never read.** The
        state is checkpointed at batch boundaries, so a process that dies
        mid-batch leaves every item `running`, though many may already have
        written a decision; dispatching them all again would pay for each
        twice. So the first time this process sees an item it asks the
        decision file first: one newer than the item's last dispatch marker
        is that dispatch's answer, and is collected instead of re-asked. A
        retry (``note``) never adopts -- the decision on disk is the failure
        being retried.
        """
        self.__dict__.setdefault("_receipt_notes", {}).pop(item.n, None)
        if not note and item.n not in self._smiths_dispatched:
            adopted = collect_decision_file(
                self._progress_ctx, self.workspace, "smith", item.n, unit=item.unit
            )
            if adopted is not None:
                print_message(
                    f"[bold cyan]→ smith {item.n} ({item.target}): adopting {adopted} written "
                    f"for a previous process's dispatch, instead of dispatching again[/bold cyan]",
                    None,
                )
                self.journal.event(
                    "smith decision adopted on resume", f"{item.n} {item.target}: {adopted}"
                )
                self._land_failed_receipt(item, task, adopted)
                return f"adopted {adopted} from a previous dispatch"
        closed = before_smith(self, item, task)
        if closed is not None:
            return closed
        self._smiths_dispatched.add(item.n)
        layer = self._make_agent(
            "smith",
            self.prompts.smith,
            session_mode="stateless",
        )
        try:
            prompt = self._smith_prompt(item, note, task=task)
            # Smiths number by queue item rather than by turn, and a retry
            # reuses that number, so the retry is reserved a path of its own.
            target = mark_decision_dispatch(self.workspace, "smith", item.n, unit=item.unit)
            prompt = f"{prompt}\n\n{decision_file_instruction(target)}"
            report = self.journal.record_dispatch("smith", f"{item.mode} {item.target}", prompt)
            result = self._run_turn(
                "smith",
                f"{item.mode} {item.target}",
                prompt,
                iteration=item.n,
                layer=layer,
                unit=item.unit,
                work=task.id if task is not None else f"{item.unit}:smith:{item.n}",
            )
            self.journal.record_result(report, result)
            decision = collect_decision_file(
                self._progress_ctx, self.workspace, "smith", item.n, unit=item.unit
            )
            if decision is None:
                print_message(
                    f"[bold yellow]! smith {item.n} ({item.target}) ended its turn "
                    f"without a valid decision file at "
                    f"{current_decision(self.workspace, 'smith', item.n, unit=item.unit)}"
                    f"[/bold yellow]",
                    None,
                )
        finally:
            layer.__exit__(None, None, None)
        self._land_failed_receipt(item, task, decision)
        return result

    def _land_failed_receipt(
        self, item: QueueItem, task: Task | None, decision: str | None
    ) -> None:
        """Integrate a FAILED_RECEIPT's commits when they change nothing but the entry's files.

        A failed receipt is not a finished task, so the task's own integration
        leaves it; what it committed is still knowledge every caller of the
        entry needs, whether or not the task runs again: the input `is_valid`
        now refuses, the guard test that proves it, the cells that hold, the
        measurement in the `note`. So its branch goes in here when every file
        it changes is one of the entry's (:func:`is_entry_file`): the files it
        was dispatched to write (its local variant's, :meth:`_landing`) or the
        shared entry's. A branch that changes anything else, the target or the
        engine, stays as it is, and the task's next attempt starts from it.
        What became of the commits is journaled and reaches the planner with
        the task's event (:meth:`_smith_decision_for`). A promote's commits are
        its task's to integrate, whole or not at all.
        """
        integration = getattr(self, "_integration", None)
        if decision != FAILED_RECEIPT or task is None or integration is None:
            return
        if item.mode == MODE_PROMOTE:
            return
        paths = integration.changed_paths(task.id)
        if not paths:
            return
        branch = integration.branch(task.id)
        landing, _ = self._landing(item, task)
        outside = [
            path
            for path in paths
            if not (is_entry_file(path, item.target) or is_entry_file(path, landing))
        ]
        if outside:
            landed = (
                f"its commits stay on {branch} for the task's next attempt: they change more "
                f"than the entry's own files ({', '.join(outside)})"
            )
        else:
            with self._hold("integration", task.id):
                outcome = integration.integrate(task.id)
                if outcome.merged:
                    integration.close(task.id)
            landed = (
                f"its {outcome.commits} commit(s), the entry's own files only, are integrated: "
                f"what it measured, refuses and tests is on the target"
                if outcome.merged
                else f"its commits stay on {branch}: {outcome.detail}"
            )
        self.journal.event("integration", f"{task.id}: {FAILED_RECEIPT}, {landed}")
        self.__dict__.setdefault("_receipt_notes", {})[item.n] = landed

    def _target_rel(self) -> str:
        """The run's target directory, repo-relative; empty before preflight names it."""
        return str(getattr(getattr(self, "_preflight", None), "target_rel", "") or "")

    def _landing(self, item: QueueItem, task: Task | None) -> tuple[str, tuple[str, ...] | None]:
        """Where ``item``'s dispatch writes: the entry it names as a target, and its files.

        An onboard or verify of a shared entry lands its local variant in the
        target's own ``ops/`` (:attr:`QueueItem.variant`), at the site the
        task's ``touches`` declare when they declare a variant of it. It lands
        in place when the task declares the shared module instead (a task the
        planner pointed at the catalog, or one written before variants), and
        when no target is known. An entry of the target's own ``ops/`` lands
        where it is; a promote spans several entries and has no one place.
        """
        if item.mode == MODE_PROMOTE:
            return item.target, None
        target_rel = self._target_rel()
        repo = self.task_data.get("trtllm_repo_path") or None
        if item.variant is None or not target_rel:
            return item.target, entry_paths(item, target_rel, repo)
        touches = list(task.touches) if task is not None else []
        category, name = item.target.strip("/").split("/")
        ops = f"{target_rel.rstrip('/')}/{MODEL_OPS_DIRNAME}/{category}/"
        placed = item
        for path in touches:
            stem, site = split_variant(PurePosixPath(path).stem)
            if path.startswith(ops) and path.endswith(".py") and site and stem == name:
                placed = QueueItem(n=item.n, mode=item.mode, target=item.target, site=site)
                break
        else:
            if shared_entry_paths(item.target)[0] in touches:
                return item.target, shared_entry_paths(item.target, repo)
        return str(placed.variant), entry_paths(placed, target_rel, repo)

    def _landing_block(self, item: QueueItem, task: Task | None) -> str:
        """Where the dispatch's files go when that is a local variant; empty otherwise."""
        landing, paths = self._landing(item, task)
        if paths is None or landing == item.target:
            return ""
        files = "\n".join(f"- `{path}`" for path in paths)
        category, name = item.target.strip("/").split("/")
        shared = " and ".join(f"`{path}`" for path in shared_entry_paths(item.target))
        start = (
            "Write the entry there"
            if item.mode == MODE_ONBOARD
            else (
                "Start the variant as a copy of the entry as it stands, file and test (the "
                "shared catalog's, or the copy another target keeps in its `ops/`), keeping its "
                f"class and its instance name (`{name}`), and add your cells there"
            )
        )
        return (
            "## Where it lands\n\n"
            f"In this target's own `ops/`, as the local variant `{landing}`:\n\n"
            f"{files}\n\n"
            f"{start}. Leave {shared} as they are: other tasks on this "
            f"entry run beside yours, each in a variant of its own. The target imports the "
            f"variant (`from .{landing.replace('/', '.')} import {name}`); before delivery one "
            f"`promote` task merges every variant of `{category}/{name}` into the catalog.\n"
        )

    def _promote_block(self, item: QueueItem) -> str:
        """What a promote merges, per entry, read off the integration checkout it was cut from."""
        repo = Path(str(self.task_data.get("trtllm_repo_path") or ""))
        target_rel = self._target_rel().rstrip("/")
        variants = promotion_mod.local_variants(repo, target_rel)
        importers = promotion_mod.variant_imports(repo, target_rel)
        models = repo / promotion_mod.MODELING_V2_PACKAGE / "models"
        blocks = ["## What to merge\n"]
        for entry in item.entries:
            shared = shared_entry_paths(entry, repo)
            lines = [
                f"### `{entry}`\n",
                f"- the shared entry: {', '.join(f'`{path}`' for path in shared)}"
                + (
                    ""
                    if (repo / shared[0]).is_file()
                    else " (not in the catalog yet: you create it)"
                ),
            ]
            own = [v for v in variants if v.entry == entry]
            lines += [
                f"- local variant `{v.target}`: {', '.join(f'`{path}`' for path in v.files)}"
                for v in own
            ]
            if not own:
                lines.append("- no local variant on the checkout: say so in your report")
            # What the variants' smiths learned and did not write into the files (a
            # helper one variant fixed, a reference two variants extended apart) is in
            # their reports; a promote's goal is the orchestrator's and carries none.
            reports = [
                path.name
                for mode in (MODE_ONBOARD, MODE_VERIFY)
                for path in self.journal.returned_reports("smith", f"{mode} {entry}")
            ]
            if reports:
                reports.sort(key=lambda report: int(report.split("-", 1)[0]))
                lines.append(
                    "- what the smiths that landed it said the merge must keep (each report's "
                    f"result, 'For the orchestrator'), in `{self.journal.reports_dir}`: "
                    + ", ".join(f"`{report}`" for report in reports)
                )
            category, name = entry.split("/")
            for copy in sorted(models.glob(f"*/*/{MODEL_OPS_DIRNAME}/{category}/{name}.py")):
                other = copy.parents[2].relative_to(repo).as_posix()
                if other != target_rel:
                    lines.append(
                        f"- another target's own copy: `{copy.relative_to(repo).as_posix()}`, "
                        f"tested under `{model_ops_tests_dir(other)}/{category}/`: it moves "
                        f"too, and that target imports the catalog"
                    )
            for path, found in importers.items():
                mine = [v for v in found if promotion_mod.variant_entry(v) == entry]
                if mine:
                    lines.append(f"- `{path}` imports {', '.join(f'`{v}`' for v in mine)}")
            blocks.append("\n".join(lines) + "\n")
        return "\n".join(blocks)

    def _promote_prompt(
        self, item: QueueItem, note: str, task: Task | None, assignment: str
    ) -> str:
        """A promote's dispatch: the entries, what each merges, and the skill with the procedure."""
        purpose = self._purpose(item, task)
        context = self._context_block()
        target_rel = self._target_rel()
        names = ", ".join(f"`{entry}`" for entry in item.entries)
        retry = f"\n{note}\n" if note else ""
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Architecture: {self.task_data['arch']}\n"
            + (f"Target: {target_rel}\n" if target_rel else "")
            + "\n"
            f"Dispatch {item.n}: **promote** this target's local variants of {names} into the "
            f"shared catalog.\n"
            f"{retry}"
            f"{assignment}"
            f"Load the `{item.skill}` skill before anything else; its `references/promote.md` "
            f"is the procedure.\n"
            + (f"\n{purpose}\n" if purpose else "")
            + f"\n{self._promote_block(item)}\n"
            f"Read `{self.vocabulary_path}` for the call sites each entry serves.\n\n"
            + self._report_instruction("smith")
            + (f"\n\n{context}\n" if context else "")
        )

    def _recorded_tier(self, item: QueueItem) -> tuple[int | None, str]:
        """The sourcing rung on record for ``item``'s entry, and where it is recorded.

        In order: the dispatch's own `vocabulary.yaml` row, when it names a
        tier; the smiths that landed the entry, newest first (their decision's
        `tier`), or a MACRO that proposed it (its constituent's `tier`); the
        vocabulary step that maps a computation onto the entry; another row
        for it. A verify row seldom names a tier and parses as tier 1 when it
        does not, so the parsed item's own tier is no record. ``(None, "")``
        when nothing records one.
        """
        vocabulary = self._raw_vocabulary()
        rows = [row for row in vocabulary.get("queue") or () if isinstance(row, dict)]
        own = next((row for row in rows if str(row.get("n")) == str(item.n)), None)
        if own is not None and _tier(own.get("tier")):
            return _tier(own.get("tier")), f"`vocabulary.yaml` queue row {item.n}"
        tier, where = self._tier_from_decisions(item)
        if tier:
            return tier, where
        for step in vocabulary.get("steps") or ():
            if isinstance(step, dict) and _tier(step.get("tier")):
                if _names_entry(step.get("entry"), item.target):
                    return _tier(step.get("tier")), "the `vocabulary.yaml` step that maps onto it"
        for row in rows:
            if str(row.get("target", "")).strip() == item.target and _tier(row.get("tier")):
                return _tier(row.get("tier")), f"`vocabulary.yaml` queue row {row.get('n')}"
        return None, ""

    def _raw_vocabulary(self) -> dict:
        """``vocabulary.yaml`` as written, unvalidated; empty when it cannot be read."""
        try:
            data = yaml.safe_load(self.vocabulary_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}

    def _tier_from_decisions(self, item: QueueItem) -> tuple[int | None, str]:
        """The rung the newest smith decision about ``item``'s entry records; ``(None, "")`` if none."""
        found: list[tuple[float, int, str]] = []
        for path in (self.workspace / DECISIONS_DIRNAME).glob("*-smith-*.yaml"):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                when = path.stat().st_mtime
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(raw, dict):
                continue
            decision = str(raw.get("decision", "")).strip()
            tier = _tier(raw.get("tier"))
            if decision in ("LANDED", FAILED_RECEIPT) and tier:
                if _names_entry(raw.get("entry"), item.target):
                    found.append((when, tier, path.name))
            if decision == "MACRO":
                for child in raw.get("constituents") or ():
                    if not isinstance(child, dict) or not _tier(child.get("tier")):
                        continue
                    if str(child.get("target", "")).strip() == item.target:
                        found.append((when, _tier(child.get("tier")), path.name))
        if not found:
            return None, ""
        _, tier, name = max(found)
        return tier, f"`{DECISIONS_DIRNAME}/{name}`"

    def _sourcing(self, item: QueueItem) -> str:
        """The dispatch's sourcing lines: the rung on record and what it means for this turn.

        Tiers 1-3 wrap something that exists; 4-5 mean an onboarding writes
        the implementation. Saying which changes what the turn is for.
        """
        tier, where = self._recorded_tier(item)
        if tier is None:
            return (
                "Sourcing: no rung is recorded for this entry. Name the one its `__call__` "
                "wraps (the kernel sources below give each source's tier) in your decision "
                "file's `tier`.\n"
            )
        line = f"Sourcing: **tier {tier} — {TIER_DESCRIPTION[tier]}** (recorded in {where}).\n"
        if item.mode == MODE_ONBOARD and tier > HIGHEST_REUSING_TIER:
            line += (
                "This entry has no implementation yet — you are writing it, not "
                "wrapping one. Correct-and-slow is acceptable; the tuning phase "
                "replaces it. Say in the contract what you rejected at the cheaper "
                "rungs.\n"
            )
        elif tier == TIER_KERNEL_LIBRARY:
            line += (
                "The entry's `__call__` calls the FlashInfer callable directly — no trtllm op "
                "wraps it. The FlashInfer API the survey found is in `surface.md` in "
                "the workspace; certify it as the kernel source `flashinfer_api` "
                "below says.\n"
            )
        return line

    @staticmethod
    def _purpose(item: QueueItem, task: Task | None) -> str:
        """What the plan asks of this dispatch: the graph task's goal, and its queue row's note.

        The row's surface says which values to certify; the goal says what
        this dispatch is for, which the planner writes per task and a row
        cannot carry: a retry's starting point, the cells a review found
        missing, the call a later member makes.
        """
        goal = (task.goal or "").strip() if task is not None else ""
        row_note = (item.note or "").strip()
        blocks = []
        if goal:
            blocks.append(f"## Your task: `{task.id}` (kernel)\n\n{goal}\n")
        if row_note and row_note != goal:
            label = "Its queue row's note" if goal else "The plan's note for this dispatch"
            blocks.append(f"{label}: {row_note}\n")
        return "\n".join(blocks)

    def _smith_prompt(self, item: QueueItem, note: str = "", task: Task | None = None) -> str:
        surface = f"The surface to certify: {item.surface}\n" if item.surface else ""
        # Directly under the dispatch line, because it changes what the turn
        # is for: finish what the last attempt started, not begin again.
        retry = f"\n{note}\n" if note else ""
        purpose = self._purpose(item, task)
        # Device assignment, because the local launch protocol hands EVERY role
        # the same line -- `CUDA_VISIBLE_DEVICES=<the claimed devices>` -- and
        # then warns that concurrent scripts on one device contend. At the
        # default ten parallel smiths that warning is a certainty, and the
        # symptom is `Executor creation failed due to insufficient GPU memory`
        # on a device with ample memory, which reads as a target that
        # allocates too much.
        #
        # Round-robin over the node's devices by dispatch number. Not a
        # scheduler: two smiths can still land on one device, but ten cannot
        # land on device 0.
        #
        # Local only. Under any other kind a scheduler hands out the devices:
        # a spool exports `CUDA_VISIBLE_DEVICES` per job and its protocol says
        # never to set it, and under Slurm the allocation is the claim, which
        # an index set here would mask.
        env_block = self.task_data.get("environment") or {}
        devices = int(env_block.get("gpus_per_node") or 1)
        assignment = ""
        if devices > 1 and not scheduler_owns_devices(str(env_block.get("kind", "local"))):
            assignment = (
                f"\n**Use `CUDA_VISIBLE_DEVICES={item.n % devices}` for this dispatch**, not the "
                f"devices in the launch protocol above. This node has {devices} devices and your "
                f"batch is running concurrently; every smith taking the same device turns a "
                f"memory error into the symptom.\n"
            )
        if item.mode == MODE_PROMOTE:
            return self._promote_prompt(item, note, task, assignment)
        tier = self._sourcing(item)
        # A smith is dispatched from `_run_one_smith` rather than from
        # `_dispatch_task`, so the shared context has to be added here too.
        context = self._context_block()
        # An `ops/<category>/<name>` target lives in this target's own `ops/`,
        # and so does a shared entry's local variant.
        target_rel = self._target_rel()
        landing = self._landing_block(item, task)
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Architecture: {self.task_data['arch']}\n"
            + (f"Target: {target_rel}\n" if target_rel else "")
            + "\n"
            f"Dispatch {item.n}: **{item.mode}** the catalog entry `{item.target}`.\n"
            f"{retry}"
            f"{assignment}"
            f"Load the `{item.skill}` skill before anything else.\n"
            + (f"\n{purpose}\n" if purpose else "")
            + (f"\n{landing}\n" if landing else "")
            + f"{surface}"
            f"{tier}\n"
            f"{self._kernel_sources_block}\n"
            f"Read `{self.vocabulary_path}` for the computation step this entry serves "
            f"and the values this checkpoint passes it.\n\n"
            + self._report_instruction("smith")
            + (f"\n\n{context}\n" if context else "")
        )
