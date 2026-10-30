# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm-modeling-v2 workflow: the run's lifecycle.

Phase ordering, check sequencing and disk verification are control flow, so
they live in code; judgement -- how to derive a forward, what to build next
-- is the agents'. This module owns init, run, resume, checkpoint, turn
dispatch and close:

1. ``preflight`` -- Python, plus one probe turn per agent configuration.
   A broken host fails here; the substrate gate, which boots a GPU, runs
   beside the first turns of the next stage and stops them when it fails.
2. ``execution`` -- the planner's task graph, driven until the contract holds
   (``run/execution/tasks.py``, ``orchestrator.py``), then delivery. With
   ``perf.enabled`` the graph holds a ``tune`` task, which starts once the
   core is correct and runs beside the remaining work. The onboarding's
   verdict never waits for it: the run closes on that verdict, and a campaign
   still running goes on past the close as the tune stage
   (:meth:`_campaign_after_onboarding`), whose outcome changes no verdict of it.
3. ``tune``      -- for ``trtllm-modeling-v2 tune``: a target onboarded
   elsewhere, tuned on its own.

Every stage checkpoints before advancing, so a crash resumes where it was.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Mapping, Sequence

import yaml
from rich.markup import escape as escape_markup

from agent_flow import (
    AgentLayer,
    AgentLayerConfig,
    BackendConfig,
    SessionConfig,
    require_tool_call_stop_hook,
)
from agent_flow.agent_runtime import AgentConfig
from agent_flow.console import print_message
from agent_flow.logger import get_logger

from ..agents.prompts import PromptBundle, build_trtllm_modeling_v2_prompts
from ..agents.roles import PACKAGED_SKILLS_DIR, resolve_role_agents
from ..agents.slots import AgentGate, Priority, turn_priority
from ..agents.turns import InProcessTurnExecutor, TurnRequest, TurnTimeout
from ..checks import boot as boot_checks
from ..checks import check_runtime
from ..checks import verdict as check_result
from ..infra.compute.commands import CommandExecutor
from ..infra.compute.environment import RunEnvironment
from ..infra.compute.rebuild import LIBRARY_PATTERNS
from ..infra.files import write_text_atomic
from ..infra.git import CommitPolicy, git_output
from ..spec.task_schema import load_and_validate_task_yaml
from ..tune import stage as tune_stage
from . import orchestrator as orchestrator_mod
from .execution.handoff import current_decision
from .execution.tasks import ExecutionStage, _start_with_the_survey
from .records import metrics as metrics_mod
from .records.journal import Journal
from .records.state import (
    STAGE_EXECUTION,
    STAGE_PREFLIGHT,
    STAGE_TUNE,
    STATE_FILENAME,
    WorkflowState,
    is_remediable,
    load_state,
    save_state,
)
from .startup import bootstrap
from .startup import primers as primers_mod
from .startup.bootstrap import HOST_FACTS_FILENAME, TROUBLESHOOTING_HINT
from .validation import contract as contract_mod
from .validation import promotions as promotions_mod

#: How many tasks run at once: the CLI's --max-parallel-tasks default.
DEFAULT_MAX_PARALLEL_TASKS = orchestrator_mod.DEFAULT_MAX_PARALLEL_TASKS
#: How many graph revisions without progress (a task done, a change landed, a check
#: turning green) end the run.
DEFAULT_REVISION_CAP = 6
#: While tasks, checks or a planner turn are in flight, how long a task that finished
#: with its checks green waits to reach the planner with the events after it. A
#: failure, a worker's report, an operator's message sent ``--now``, a failed re-check, or nothing
#: left to run or start wakes the planner at once.
DEFAULT_EVENT_WINDOW_S = 900.0
#: Where the run's skills go under the workspace, each agent session's cwd:
#: Claude Code reads `.claude/skills`, Codex `.agents/skills`.
SKILL_DIRS: tuple[str, ...] = (".claude/skills", ".agents/skills")

#: Where nothing a gate runs or reads lives, so a change there leaves every
#: verdict standing: top-level directories by prefix, documentation anywhere
#: by suffix. The files a gate does read under them are kept (`gate_inputs`).
UNGATED_PREFIXES: tuple[str, ...] = ("tests/", "docs/", ".github/", "examples/")
UNGATED_SUFFIXES: tuple[str, ...] = (".md",)
#: Where the native libraries an engine loads are built: inside the checkout,
#: and ignored by git.
NATIVE_LIBRARY_DIRS: tuple[str, ...] = ("tensorrt_llm", "tensorrt_llm/libs")
#: The spec's blocks whose values may name a file a gate reads.
GATE_INPUT_BLOCKS: tuple[str, ...] = ("accuracy", "runtime", "variants", "environment")


def _same_tree(a: Path, b: Path) -> bool:
    """Whether directory ``b`` holds exactly ``a``'s files with the same bytes."""
    import filecmp

    if not b.is_dir():
        return False
    files_a = sorted(
        p.relative_to(a) for p in a.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    files_b = sorted(
        p.relative_to(b) for p in b.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    return files_a == files_b and all(filecmp.cmp(a / f, b / f, shallow=False) for f in files_a)


class TrtllmModelingV2Workflow:
    """Drive one checkpoint to a gated modeling_v2 target."""

    survey_first = True

    def __init__(
        self,
        workspace: Path,
        *,
        max_parallel_tasks: int = DEFAULT_MAX_PARALLEL_TASKS,
        revision_cap: int = DEFAULT_REVISION_CAP,
        clean: bool = False,
        dry_run: bool = False,
        skip_substrate_gate: bool = False,
        prompts: PromptBundle | None = None,
        command_runner: Any | None = None,
        shell_runner: Any | None = None,
        git_runner: Any | None = None,
        agent_turn_timeout: float | None = None,
        entry_stage: str = "",
        event_window_s: float = DEFAULT_EVENT_WINDOW_S,
        primers: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self.event_window_s = event_window_s
        # The planner's, kernel writers' and assemblers' turns start as forks of one
        # briefing session per role (run/startup/primers.py); off, each starts afresh.
        self.use_primers = primers
        self._primers: dict[str, primers_mod.Primer] = {}
        #: Set once a role's primer is built or given up on; until then its turns start without it.
        self._primer_ready: dict[str, threading.Event] = {}
        self._primer_lock = threading.Lock()
        #: Per worker thread: the worktree block its turns carry.
        self._tls = threading.local()
        # `trtllm-modeling-v2 tune`: start (or re-enter) at the tune stage.
        self.entry_stage = entry_stage
        self.task_path = self.workspace / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self.journal_path = self.workspace / "journal.md"
        self.host_facts_path = self.workspace / HOST_FACTS_FILENAME
        self.state_path = self.workspace / STATE_FILENAME

        self.max_parallel_tasks = max(1, max_parallel_tasks)
        self.revision_cap = max(1, revision_cap)
        self.dry_run = dry_run
        self.skip_substrate_gate = skip_substrate_gate
        self._prompts_override = prompts

        # Injected so every stage is testable without a GPU, a cluster, or a
        # model. The defaults are the real ones.
        self._command_runner = command_runner
        # An explicit runner always wins; otherwise preflight resolves one from
        # the environment, and until then it is the plain shell.
        self._shell_runner_override = shell_runner
        # Questions about the checkout under test, asked on this host:
        # ``(repo, *args) -> stdout | None``. Never the gate runner; see
        # `_target_marker`.
        self._git = git_runner or git_output
        #: The cap on agents working at once, task turns and the planner's alike.
        self._agent_gate = AgentGate(self.max_parallel_tasks)
        #: How a role turn runs: one agent CLI process per turn, here.
        self._turns = InProcessTurnExecutor(
            lambda request: self._layer(request.role), timeout_s=agent_turn_timeout
        )
        #: Where this run's commands land. Resolved from the environment in
        #: `_run_preflight`, which is the first moment one exists; until
        #: then nothing has a command to run.
        self._command_executor: CommandExecutor | None = None
        self._shell_runner = shell_runner or check_runtime.default_shell_runner

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            for path in (
                self.state_path,
                self.plan_path,
                self.vocabulary_path,
                self.journal_path,
                # This run's context, not the operator's notes: the facts are
                # re-seeded by the next preflight.
                self.host_facts_path,
            ):
                path.unlink(missing_ok=True)
            shutil.rmtree(self.workspace / "reports", ignore_errors=True)
            for name in (
                "graph.yaml",
                "graph.proposed.yaml",
                contract_mod.STATUS_FILENAME,
                # A fresh survey gets its own review.
                "survey_review.md",
            ):
                (self.workspace / name).unlink(missing_ok=True)
            # Regenerated by the next preflight, but a run that fails before
            # reaching it would otherwise leave the previous checkpoint's
            # survey sitting in the workspace looking current.
            (self.workspace / "surface.md").unlink(missing_ok=True)
            (self.workspace / "prior_art.md").unlink(missing_ok=True)
            (self.workspace / "tree_map.md").unlink(missing_ok=True)

        self.resume = self.state_path.is_file()

        self.journal = Journal(path=self.journal_path)
        #: One milestone recorded at a time.
        self._milestone_lock = threading.Lock()
        self.gate_log = check_result.GateLog()
        #: Compiled at preflight; see `contract.py`.
        self._contract: contract_mod.Contract | None = None
        self._status: contract_mod.ContractStatus | None = None

        self.task_data: dict[str, Any] = {}
        self.environment: RunEnvironment | None = None
        self.prompts: PromptBundle = prompts or build_trtllm_modeling_v2_prompts()
        self._layers: dict[str, AgentLayer] = {}
        self._agent_configs: dict[str, AgentConfig] | None = None
        self._agent_config_lock = threading.Lock()
        # Set by ``_run_preflight``, which every path through ``run`` reaches
        # before anything reads it, a resume's included.
        self._preflight: Any = None
        # The surface survey, until preflight takes it.
        self._surface_block: str = (
            "## Surface survey\n\nNot taken for this run — plan on the assumption that "
            "you do not know what exists, and say so in the plan's open items.\n"
        )
        # The tree's prior art and the engine's limits for this checkpoint (preflight's
        # `_take_prior_art`): the survey's section, and what its review holds it to.
        self._prior_art_block: str = ""
        self._prior_art: Any = None
        self._engine_limits: list[Any] = []
        # Whether the survey review already sent this run's survey back once.
        self._survey_reviewed = False
        # Backend kind -> the skills its sessions load besides the run's own (the user's),
        # which the agent probe reads and every role then runs without.
        self._foreign_skills: dict[str, tuple[str, ...]] = {}

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> "TrtllmModelingV2Workflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for layer in self._layers.values():
            layer.__exit__(None, None, None)
        self._layers.clear()

    # ----------------------------------------------------------------- agents

    def _role_agents(self) -> dict[str, AgentConfig]:
        """Pin role configuration once, shared by probes, primers and every task turn."""
        with self._agent_config_lock:
            if self._agent_configs is None:
                self._agent_configs = resolve_role_agents(self.task_data)
            return self._agent_configs

    def _make_agent(
        self,
        name: str,
        system_prompt: str,
        *,
        requires_decision: bool = True,
        session_mode: str = "stateless",
        fork_from: str | None = None,
    ) -> AgentLayer:
        # Every stage uses the same resolved launcher model and explicit task overrides.
        agent = self._role_agents()[name]
        # The role reports by writing its decision file, and the Stop hook
        # refuses a turn that wrote nothing. `Write` is a weak requirement --
        # the role writes other files too -- but the enum is validated when
        # the file is read. Codex refuses per-client Claude SDK hooks and
        # writes with its own tools, so it gets no hook.
        hooks = require_tool_call_stop_hook(["Write"]) if requires_decision else None
        if agent.backend == "codex":
            hooks = None
        return AgentLayer(
            AgentLayerConfig(
                name=name,
                system_prompt=system_prompt,
                backend=BackendConfig(
                    kind=agent.backend,
                    model=agent.model,
                    reasoning_effort=agent.reasoning_effort,
                    hooks=hooks,
                    extra_mcp_servers=agent.extra_mcp_servers,
                    # The workspace is the session cwd so the run's own
                    # materialized skills are the ones the agent discovers,
                    # rather than whatever happens to be configured in the
                    # repository it is editing. The user's own skills and
                    # plugins stay out: no role is briefed on them, and an
                    # unattended session should not reach Slack or mail.
                    cwd=self.workspace,
                    writable_roots=self._writable_roots(),
                    disabled_skills=self._foreign_skills.get(agent.backend, ()),
                    plugins=False,
                ),
                session=SessionConfig(mode=session_mode, fork_from=fork_from),
            )
        )

    def _writable_roots(self) -> tuple[str, ...]:
        """What a role writes outside the workspace: the checkout and the run's spools.

        A task's worktree commits into the checkout's ``.git``, and its work is
        submitted to the spools.
        """
        roots = [self.task_data.get("trtllm_repo_path")]
        if self.environment is not None:
            env = self.environment
            roots += [env.spool, *(session.spool for session in env.sessions), *env.dev_spools]
        return tuple(dict.fromkeys(str(root) for root in roots if root and Path(root).is_dir()))

    def _run_turn(
        self,
        role: str,
        what: str,
        prompt: str,
        *,
        iteration: int | None = None,
        unit: str = "",
        work: str = "",
    ) -> str:
        """Run one role turn.

        ``work`` names the work the turn is for where ``what`` does not
        (:attr:`TurnRequest.work`). A turn forked from its role's primer that
        fails drops the primer and runs again plainly.
        """
        agent = self._role_agents()[role]
        block = getattr(self._tls, "worktree", "")
        if block:
            prompt = f"{prompt}\n\n{block}\n"
        start = self._turn_start(role, prompt)
        while True:
            request = TurnRequest(
                role=role,
                what=what,
                prompt=start.prompt,
                system_prompt=getattr(self.prompts, role, "") if self.prompts else "",
                backend=agent.backend,
                model=agent.model,
                reasoning_effort=agent.reasoning_effort,
                decision_file=(
                    str(current_decision(self.workspace, role, iteration, unit=unit))
                    if iteration is not None
                    else None
                ),
                work=work,
            )
            executor = self._turns if start.layer is None else self._turns.bound(start.layer)
            try:
                with self._agent_gate.slot():
                    return executor.run(request).text
            except TurnTimeout:
                raise
            except Exception as exc:
                if not start.forked:
                    raise
                # The fork failed, not the work: the primer goes, and the turn runs plainly.
                self._drop_primer(role, f"{type(exc).__name__}: {exc}")
            finally:
                start.close()
            start = self._turn_start(role, prompt)

    def _start_primers(self) -> None:
        """Build each primed role's primer in the background, beside the run's first turns.

        A primer recorded under the role's current prompts is used as it is.
        One being built holds nothing up: the role's turns run as they always
        did until it is ready, and it takes only the agent slots the run's own
        turns leave idle.
        """
        if not self.use_primers:
            return
        recorded = primers_mod.load(self.workspace)
        for role in primers_mod.PRIMED_ROLES:
            want = primers_mod.version(
                role,
                getattr(self.prompts, role, ""),
                agent=self._role_agents()[role],
            )
            ready = self._primer_ready.setdefault(role, threading.Event())
            held = recorded.get(role)
            if held is not None and held.version == want:
                self._primers[role] = held
                ready.set()
                continue
            threading.Thread(
                target=self._build_primer,
                args=(role, want, ready),
                name=f"mv2-primer-{role}",
                daemon=True,
            ).start()

    def _build_primer(self, role: str, want: str, ready: threading.Event) -> None:
        """One primer turn of ``role``; the session it leaves, the commit it read and its summary."""
        try:
            repo = str(self.task_data.get("trtllm_repo_path") or "")
            commit = self._head()
            summary = primers_mod.summary_path(self.workspace, role)
            summary.parent.mkdir(parents=True, exist_ok=True)
            layer = self._make_agent(role, getattr(self.prompts, role, ""), requires_decision=False)
            # Only a backend that can fork a session hands its turns this one.
            forks = primers_mod.can_fork(self._role_agents()[role])
            request = TurnRequest(
                role=role,
                what="primer",
                prompt=primers_mod.primer_prompt(
                    role, repo=repo, commit=commit, summary=summary, forks=forks
                ),
                system_prompt=getattr(self.prompts, role, ""),
            )
            with turn_priority(Priority.SIDE), self._agent_gate.slot():
                self._turns.bound(layer).run(request)
            session = str(getattr(layer, "session_id", None) or "") if forks else ""
            primer = primers_mod.Primer(role, session, commit, want, str(summary))
            with self._primer_lock:
                self._primers[role] = primer
                primers_mod.save(self.workspace, self._primers)
            self.journal.event(
                "primer",
                f"{role}: "
                + (f"session {session[:12]}" if session else "a summary only")
                + f", read at {commit[:10] or 'an unknown commit'}",
            )
            layer.__exit__(None, None, None)
        except Exception as exc:  # a run without a primer is a run as it was
            self.journal.event("primer failed", f"{role}: {type(exc).__name__}: {exc}")
        finally:
            ready.set()

    def _primed(self, role: str) -> "primers_mod.Primer | None":
        """``role``'s primer once it is built; ``None`` while it is not, or without one."""
        ready = self._primer_ready.get(role)
        if ready is None or not ready.is_set():
            return None
        with self._primer_lock:
            return self._primers.get(role)

    def _turn_start(self, role: str, prompt: str) -> "_TurnStart":
        """How a turn of ``role`` starts: as a fork of its primer, pointed at its summary, or as asked."""
        primer = self._primed(role)
        if primer is not None and primer.session_id:
            forked = self._make_agent(
                role, getattr(self.prompts, role, ""), fork_from=primer.session_id
            )
            return _TurnStart(primers_mod.primed_preamble(primer.commit) + prompt, forked, True)
        if primer is not None and primer.summary and Path(primer.summary).is_file():
            note = primers_mod.summary_note(Path(primer.summary), primer.commit)
            return _TurnStart(note + prompt)
        return _TurnStart(prompt)

    def _drop_primer(self, role: str, why: str) -> None:
        """A turn could not fork ``role``'s primer: later turns read its summary instead."""
        with self._primer_lock:
            held = self._primers.get(role)
            if held is None or not held.session_id:
                return
            self._primers[role] = primers_mod.Primer(
                held.role, "", held.commit, held.version, held.summary
            )
            primers_mod.save(self.workspace, self._primers)
        self.journal.event("primer dropped", f"{role}: a turn could not fork it ({why})")

    def _layer(self, name: str) -> AgentLayer:
        """Build a role's layer on first use and keep it for the run.

        Every role is **stateless**: what it needs from earlier turns is on disk.
        """
        if name not in self._layers:
            self._layers[name] = self._make_agent(name, getattr(self.prompts, name))
        return self._layers[name]

    def _context_block(self) -> str:
        """The host facts every dispatch carries, by reference: a fact one turn appends reaches the next."""
        if not self.host_facts_path.is_file():
            return ""
        return (
            f"## Host facts\n\n"
            f"Read `{self.host_facts_path}` before you debug anything about this "
            f"machine. It carries what this host does differently — `pytest`'s exit "
            f"code, the `TMPDIR` bound, the launch rules, which errors are noise. If "
            f"you learn a new one, append it under `## Learned during this run`."
        )

    def _materialize_skills(self) -> Path:
        """Copy the packaged skills into this run's workspace, where every backend finds them.

        Claude Code reads `.claude/skills`, Codex `.agents/skills`; both are the
        session's cwd-relative skill directories. Returns the first.
        """
        source = PACKAGED_SKILLS_DIR
        destinations = [self.workspace / name for name in SKILL_DIRS]
        if not source.is_dir():
            return destinations[0]
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)
            for skill_dir in sorted(p for p in source.iterdir() if p.is_dir()):
                target = destination / skill_dir.name
                if _same_tree(skill_dir, target):
                    # A resumed run: the copy is current, and the directory may be read-only
                    # to this process (a sandbox protects `.claude/skills`).
                    continue
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(skill_dir, target)
        return destinations[0]

    # ------------------------------------------------------------ orchestration

    def run(self, task: str | Path) -> None:
        log = get_logger().console
        state = self._init_state(task, log)
        if state is None:
            return
        self._role_agents()

        try:
            # A substrate gate that fails while the first turns run stops them.
            with orchestrator_mod.stoppable(self.workspace) as stop:
                self._run_stages(state, log, stop)
            bootstrap._await_substrate(self)
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — re-run the same command to continue "
                "from the last checkpoint, or pass --clean to start fresh"
                "[/bold yellow]",
                log,
            )
            self.journal.event("run interrupted", f"stage={state.stage}")
            raise

    def _run_stages(
        self, state: WorkflowState, log, stop: Callable[[str], None] | None = None
    ) -> None:
        """Preflight, then the stage the run is at, to its close.

        Toward the execution stage the substrate gate runs beside its first
        turns (``stop`` ends them when it fails), and an execution stage with no
        graph yet starts with the survey (:meth:`_start_with_the_survey`). A run
        the failed gate stopped is left open, as a failed preflight is:
        re-running the same command proves the substrate again and resumes.
        """
        self._state = state
        beside = stop if state.stage != STAGE_TUNE and not state.done else None
        if state.stage == STAGE_PREFLIGHT:
            if not bootstrap._run_preflight(self, state, log, stop=beside):
                return
            if self.dry_run:
                print_message(
                    "[bold cyan]→ --dry-run: preflight passed; stopping before the "
                    "first agent turn.[/bold cyan]",
                    log,
                )
                return
            state.stage = STAGE_EXECUTION
            self._checkpoint(state)
        else:
            # Resuming past preflight still needs the resolved
            # environment, since every prompt and every gate carries it.
            if not bootstrap._run_preflight(self, state, log, stop=beside):
                return
            if self.dry_run:
                print_message(
                    "[bold cyan]→ --dry-run: preflight passed; stopping before the "
                    f"{state.stage} stage.[/bold cyan]",
                    log,
                )
                return

        if state.done:
            # A closed onboarding whose campaign did not end (`campaign_owed`): only it goes on.
            self._campaign_after_onboarding(state, log)
            return

        if state.stage == STAGE_EXECUTION:
            self.journal.mark_started()
            _start_with_the_survey(self)
            execution = ExecutionStage(self)
            if execution.run_execution(state, log):
                # The onboarding's verdict; a campaign's is `state.tune["verdict"]`.
                self._close_run(state, log, verdict="delivered")
            if state.done and execution.campaign is not None:
                self._campaign_after_onboarding(state, log, execution.campaign)
            return

        if state.stage == STAGE_TUNE:
            verdict = tune_stage.run_tune_phase(self, state, log)
            if verdict is None:
                return
            self._close_run(state, log, verdict=verdict)
            return

        raise ValueError(f"unsupported workflow stage: {state.stage}")

    # --------------------------------------------------------- target identity

    def _target_marker(self) -> str:
        """A content fingerprint of the tree every check measures, or ``""``.

        Every check measures a snapshot of the commit the checkout is at
        (``run/workspace/snapshots.py``), so the fingerprint is of that commit's whole
        tree (git's blob ids), less what no gate runs or reads
        (`gate_relevant`) but for the files a gate does read there
        (`gate_inputs`), and the run's own workspace. The native libraries
        built into the checkout are outside git, so their sizes and
        modification times are part of it. Uncommitted changes are in no
        snapshot, so they leave the marker as it is.

        ``""`` means **not established**, and every caller reads it as
        "assume changed".
        """
        head = self._head()
        return self._commit_marker(head) if head else ""

    def _head(self) -> str:
        """The commit the checkout is at; ``""`` when git cannot say.

        Git is asked from this process, never through the gate runner: on a
        spool every command's output opens with a banner naming its job.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
        return words[0] if words else ""

    def _commit_marker(self, head: str, reads: Collection[str] | None = None) -> str:
        """Commit ``head``'s marker, as :meth:`_target_marker` describes it.

        The native libraries are the checkout's: every snapshot links them
        in. ``""`` when git cannot list the commit.

        With ``reads``, the marker narrowed to what a check read: every Python
        file not in ``reads`` is left out, since the Python a check's processes
        imported is all the Python that ran, while a file of any other kind (a
        header, a config, a table) may be read by code no import names, so each
        stays in.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        if not repo or not head:
            return ""
        wanted = None if reads is None else set(reads)
        policy = CommitPolicy.from_task(self.task_data, workspace=self.workspace)
        try:
            committed = self._committed_blobs(repo, head)
            if committed is None:
                return ""
            keep = gate_inputs(self.task_data, repo)
            blobs = {
                path: blob
                for path, blob in committed.items()
                if (wanted is None or not path.endswith(".py") or path in wanted)
                and gate_relevant(path, keep=keep, denied=policy.denied_paths)
            }
            libraries = native_libraries(Path(repo))
        except (OSError, ValueError):
            # Unreadable, or a name git printed that is not text: unknown, not unchanged.
            return ""
        return fingerprint(blobs, libraries)

    def _narrowed(self, result: Any) -> tuple[str, str]:
        """What ``result`` read of the tree, as a verdict records it: its list's id and marker.

        The list is kept once under ``reads/<id>.json`` in the workspace. ``("", "")``
        when the result names no one commit it measured or did not say what it read.
        """
        reads = tuple(getattr(result, "reads", ()) or ())
        commits = {str(c) for c in getattr(result, "imported", ()) or ()}
        if not reads or len(commits) != 1 or getattr(result, "void", ""):
            return "", ""
        text = "\n".join(sorted(reads))
        ident = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        path = self.workspace / READS_DIRNAME / f"{ident}.json"
        try:
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                write_text_atomic(path, json.dumps(sorted(reads)))
        except OSError:
            return "", ""
        marker = self._commit_marker(commits.pop(), reads)
        return (ident, marker) if marker else ("", "")

    def _verdict_holds(self, verdict: contract_mod.Verdict) -> bool:
        """Whether ``verdict`` holds on the target as it is.

        What it read of the tree is unchanged, or only a done promote changed it since.
        Its narrowed marker (:meth:`_commit_marker` with ``reads``) on the checkout's ``HEAD``
        is the one it recorded. Asked once per head and list.
        """
        if verdict.marker and verdict.marker in promotions_mod.carried(
            self.workspace, self._target_marker()
        ):
            # Measured before a done promote: the promote moved the kernels it ran, and its
            # boot passed on the tree it left (``validation/promotions.py``).
            return True
        head = self._head()
        if not head or not verdict.reads or not verdict.reads_marker:
            return False
        key = (head, verdict.reads)
        cache = self.__dict__.setdefault("_narrowed_now", {})
        if key not in cache:
            try:
                reads = json.loads(
                    (self.workspace / READS_DIRNAME / f"{verdict.reads}.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, ValueError):
                reads = None
            if len(cache) > 256:
                cache.clear()
            cache[key] = self._commit_marker(head, reads) if reads else ""
        return bool(cache[key]) and cache[key] == verdict.reads_marker

    def _committed_blobs(self, repo: str, head: str) -> dict[str, str] | None:
        """Every path in ``head``'s tree and its object id; the last head asked is kept."""
        cached = self.__dict__.get("_head_blobs")
        if cached is not None and cached[0] == (repo, head):
            return cached[1]
        listing = self._git(repo, "ls-tree", "-r", "-z", "--full-tree", head)
        if listing is None:
            return None
        blobs = tree_blobs(listing)
        self.__dict__["_head_blobs"] = ((repo, head), blobs)
        return blobs

    # ------------------------------------------------------------------ helpers

    def _uncommitted_gate_inputs(self, repo: str) -> list[str] | None:
        """The files a gate reads that the checkout holds uncommitted; ``None`` when git cannot say.

        No check measures them: a check measures a snapshot of a commit.
        """
        policy = CommitPolicy.from_task(self.task_data, workspace=self.workspace)
        pathspec = [*policy.allowed_paths] or ["."]
        pathspec += [f":(exclude){p}" for p in policy.denied_paths]
        dirt = self._git(
            repo, "status", "--porcelain", "-z", "--untracked-files=all", "--", *pathspec
        )
        if dirt is None:
            return None
        keep = gate_inputs(self.task_data, repo)
        return sorted(
            {
                path
                for path, _ in porcelain_paths(dirt)
                if gate_relevant(path, keep=keep, denied=policy.denied_paths)
            }
        )

    def _contract_status(self) -> contract_mod.ContractStatus:
        """Every verdict recorded against the contract, this process's and earlier ones'."""
        if self._status is None:
            self._status = contract_mod.ContractStatus.load(self.workspace)
        return self._status

    def _init_state(self, task: str | Path, log) -> WorkflowState | None:
        source = Path(task)
        if not source.is_file():
            raise FileNotFoundError(f"--task must be a path to an existing YAML file; got {source}")
        self.task_data = load_and_validate_task_yaml(source)
        check_runtime.default_triton_cache(self.task_data, self.workspace)

        if self.resume:
            state = load_state(self.state_path)
            if (
                state.done
                and campaign_owed(state)
                and (self.entry_stage == STAGE_TUNE or not is_remediable(state.verdict))
            ):
                # The onboarding closed beside a campaign that did not end: only the
                # campaign goes on. A re-run after a repair re-enters the onboarding
                # instead, where the campaign's task resumes it.
                print_message(
                    f"[bold cyan]→ the run closed '{state.verdict}'; its campaign goes on "
                    f"as the tune stage[/bold cyan]",
                    log,
                )
                self._refresh_workspace_spec(source, log)
                return state
            if state.done and is_remediable(state.verdict):
                # The run closed on something the operator fixes outside it,
                # and re-running its own checks is the only way to see the fix.
                print_message(
                    f"[bold cyan]→ re-entering stage '{state.stage}': the run closed "
                    f"'{state.verdict}', which names a repair made outside it[/bold cyan]",
                    log,
                )
                self.journal.event("re-entered a closed run", f"verdict={state.verdict}")
                state.done = False
                state.verdict = ""
            elif (
                state.done
                and self.entry_stage == STAGE_TUNE
                and (state.verdict == "delivered" or state.verdict.startswith("tune"))
            ):
                # An explicit `tune` is the operator's decision to spend more:
                # re-enter at tune, keep the campaign's history, reset its misses.
                state.done, state.verdict, state.stage = False, "", STAGE_TUNE
                state.tune = {**state.tune, "misses": 0, "entry": "re-entered"}
                state.tune.pop("verdict", None)
                if state.tune.get("phase") == "done":
                    state.tune["phase"] = "campaign"
            elif state.done:
                print_message(
                    "[bold green]✔ this run already completed; pass --clean to start "
                    "over.[/bold green]",
                    log,
                )
                return None
            print_message(f"[bold cyan]→ resuming at stage '{state.stage}'[/bold cyan]", log)
            self._refresh_workspace_spec(source, log)
            return state

        if source.resolve() != self.task_path.resolve():
            # The resolved spec, not the user's file: defaults are filled in
            # so what the agents read on disk is fully explicit.
            self.task_path.write_text(
                yaml.safe_dump(self.task_data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        state = WorkflowState(task_path=str(self.task_path))
        if self.entry_stage == STAGE_TUNE:
            # Standalone: a target onboarded elsewhere. Preflight still runs
            # (environment, prompts, skills); the stage then
            # measures and tunes the target the task names.
            state.stage = STAGE_TUNE
            state.tune = {"entry": "standalone"}
        self._checkpoint(state)
        return state

    def _refresh_workspace_spec(self, source: Path, log) -> None:
        """Keep the workspace ``task.yaml``, which the agents read, identical to the spec the gates use.

        ``_init_state`` re-reads and resolves the operator's file on **every** run.
        """
        if source.resolve() == self.task_path.resolve():
            return
        resolved = yaml.safe_dump(self.task_data, sort_keys=False, allow_unicode=True)
        if self.task_path.is_file() and self.task_path.read_text(encoding="utf-8") == resolved:
            return
        self.task_path.write_text(resolved, encoding="utf-8")
        print_message(
            f"[bold cyan]→ refreshed the workspace spec from {source}; it had drifted "
            f"from the file this run was given[/bold cyan]",
            log,
        )

    def _milestone_commit(
        self,
        marker: str,
        evaluation: contract_mod.Evaluation,
        checks: Sequence[contract_mod.Check],
    ) -> str | None:
        """The commit that holds the state ``checks`` passed on (``marker``), or ``None``.

        The commit every one of them imported (:func:`~.contract.measured_commit`)
        when its marker is ``marker``: a milestone names what was measured, and
        the checkout moving on while the checks ran does not change that. Else
        the checkout's ``HEAD``, when it is still at that state. A check whose
        verdict names no commit leaves only the second.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        if not repo or not marker:
            return None
        measured = contract_mod.measured_commit(
            contract_mod.ContractStatus.load(self.workspace), evaluation, checks, marker
        )
        if measured is not None and self._commit_marker(measured) == marker:
            return measured
        head = self._head()
        return head if head and self._commit_marker(head) == marker else None

    def _point_branch(self, suffix: str, sha: str, reason: str, unwritten: str = "") -> str:
        """Point ``<target branch>-<suffix>`` at ``sha``; the ref, or ``""`` when git refused.

        The target branch is the checkout's, or the run's own with ``HEAD``
        detached. A refusal is journaled under ``unwritten`` when it is given.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        branch = (self._git(repo, "symbolic-ref", "--quiet", "--short", "HEAD") or "").strip()
        state = getattr(self, "_state", None)
        ref = f"{branch or getattr(state, 'branch', '') or 'trtllm-modeling-v2'}-{suffix}"
        if self._git(repo, "update-ref", "-m", reason, f"refs/heads/{ref}", sha) is not None:
            return ref
        if unwritten:
            self.journal.event(unwritten, f"refs/heads/{ref} at {sha[:10]}")
        return ""

    def _mark_bringup(
        self, marker: str, evaluation: contract_mod.Evaluation
    ) -> contract_mod.Milestone | None:
        """The bring-up checks hold on the target (``marker``): name its commit, once.

        ``evaluation`` is the contract evaluated on ``marker``. The first time
        every check of :func:`~.contract.bringup_checks` passes there, the
        checkout's ``HEAD``, whose snapshot they measured, gets the branch
        ``<target branch>-bringup``, the journal says ``BRINGUP DELIVERED
        <sha>`` with the time since the run opened, and the milestone is
        appended to ``correctness.jsonl``. Nothing is recorded, and ``None``
        returned, once a bring-up or correctness milestone is, or when no
        commit is known to hold the state the checks measured.
        """
        contract = self._contract
        if contract is None or not marker or not contract_mod.bringup_holds(contract, evaluation):
            return None
        with self._milestone_lock:
            if (
                contract_mod.bringup_milestone(self.workspace) is not None
                or contract_mod.latest_milestone(self.workspace) is not None
            ):
                return None
            held = contract_mod.bringup_checks(contract)
            return self._deliver_milestone(
                contract_mod.MILESTONE_BRINGUP,
                marker,
                evaluation,
                held,
                f"{', '.join(c.key for c in held)} hold on it",
            )

    def _deliver_milestone(
        self,
        tier: str,
        marker: str,
        evaluation: contract_mod.Evaluation,
        checks: Sequence[contract_mod.Check],
        holds: str,
    ) -> contract_mod.Milestone | None:
        """Name the commit ``checks`` measured on ``marker``, point ``<branch>-<tier>`` at it, record it.

        The commit is the one the checks measured (``_milestone_commit``),
        whether or not the checkout has moved on since. The journal says
        ``<TIER> DELIVERED <sha>`` with the time since the run opened and
        ``holds``, what holds there. ``None`` when no commit is known to hold
        the state the checks measured.
        """
        sha = self._milestone_commit(marker, evaluation, checks)
        if sha is None:
            return None
        bringup = tier == contract_mod.MILESTONE_BRINGUP
        ref = self._point_branch(
            "bringup" if bringup else "correct",
            sha,
            "trtllm-modeling-v2: "
            + (
                "the first unit's boot and accuracy hold at one rank"
                if bringup
                else "every required check of the contract holds"
            ),
            unwritten=f"{'bring-up' if bringup else 'correctness'} ref not written",
        )
        opened = self.journal.opened_at()
        elapsed = time.time() - opened.timestamp() if opened is not None else None
        previous = None if bringup else contract_mod.latest_milestone(self.workspace)
        milestone = contract_mod.Milestone(
            sha=sha,
            marker=marker,
            ref=ref,
            at=time.time(),
            elapsed_s=elapsed,
            checks=len(checks),
            tier=tier,
        )
        contract_mod.record_milestone(self.workspace, milestone)
        parts = [
            f"{elapsed / 3600:.1f} h after the run started ({opened.isoformat()})"
            if opened is not None and elapsed is not None
            else "the journal does not say when the run started",
            holds,
        ]
        if ref:
            parts.append(f"branch `{ref}`")
        if previous is not None:
            parts.append(f"moved forward from {previous.sha[:10]}")
        detail = "; ".join(parts)
        label = "BRINGUP" if bringup else "CORRECTNESS"
        self.journal.event(f"{label} DELIVERED {sha}", detail)
        print_message(
            f"[bold green]✔ {label} DELIVERED {sha[:10]}: {escape_markup(detail)}[/bold green]",
            None,
        )
        return milestone

    def _mark_cells(self, marker: str, evaluation: contract_mod.Evaluation) -> list[str]:
        """Record the cells the contract holds on ``marker``; the ones that hold anew.

        A cell is a unit or a variant at one member (:func:`~.contract.cells`).
        ``cells.json`` keeps, per cell, the latest commit every one of its
        checks passed on, so a run stopped at any moment says what it proved
        and where. The commit the most cells hold on at once gets the branch
        ``<target branch>-best``, moved forward by a later commit that holds as
        many.
        """
        contract = self._contract
        if contract is None or not marker:
            return []
        holding = contract_mod.cells_holding(contract, evaluation)
        if not holding:
            return []
        owed = contract_mod.cells(contract)
        with self._milestone_lock:
            sha = self._milestone_commit(
                marker, evaluation, [c for cell in holding for c in owed[cell]]
            )
            if sha is None:
                return []
            record = contract_mod.load_cells(self.workspace)
            now = time.time()
            anew = []
            for cell in holding:
                name = contract_mod.cell_name(cell)
                if (record["cells"].get(name) or {}).get("sha") != sha:
                    anew.append(name)
                record["cells"][name] = {"sha": sha, "at": now}
            best = record["best"] or {}
            moved = len(holding) >= int(best.get("cells") or 0) and best.get("sha") != sha
            if moved:
                ref = self._point_branch(
                    "best", sha, f"trtllm-modeling-v2: {len(holding)} of {len(owed)} cells hold"
                )
                record["best"] = {"sha": sha, "cells": len(holding), "ref": ref, "at": now}
            contract_mod.save_cells(self.workspace, record)
        if anew:
            self.journal.event(
                f"CELLS {sha}",
                f"{', '.join(anew)} hold on {sha[:10]}; {len(holding)} of {len(owed)} cells hold there",
            )
        if moved:
            self.journal.event(
                f"BEST {sha}",
                f"{len(holding)} of {len(owed)} cells hold on it"
                + (f"; branch `{record['best']['ref']}`" if record["best"]["ref"] else ""),
            )
        return anew

    def _tune_requested(self) -> bool:
        perf = self.task_data.get("perf") or {}
        return bool(perf.get("enabled")) or self.entry_stage == STAGE_TUNE

    def _checkpoint(self, state: WorkflowState) -> None:
        save_state(self.state_path, state)

    def _gate_env(self) -> RunEnvironment:
        """The environment a gate runs under: the run's, as preflight resolved it."""
        assert self.environment is not None, "preflight must run before any gate"
        return self.environment

    def _campaign_after_onboarding(
        self, state: WorkflowState, log, campaign: Future | None = None
    ) -> None:
        """The tune stage past the onboarding's close: its campaign, which changes no verdict of it.

        ``campaign`` is the campaign's thread, still running when the onboarding's
        verdict came; without one, the campaign resumes from its checkpoint
        (``state.tune``) in a later process. Its outcome goes to
        ``state.tune["verdict"]`` and the journal; ``state.verdict`` stays the
        onboarding's.
        """
        state.tune["onboarded"] = state.verdict
        self._checkpoint(state)
        note = (
            f"the onboarding closed `{state.verdict}`; its campaign goes on, and its "
            f"outcome changes no verdict of the onboarding"
        )
        self.journal.event("tune stage", note)
        print_message(f"[bold cyan]→ tune stage: {escape_markup(note)}[/bold cyan]", log)
        if campaign is not None:
            try:
                decision, _ = campaign.result()
            except Exception as exc:  # the campaign's own failure: a later process resumes it
                decision = f"raised {type(exc).__name__}: {exc}"
            verdict = tune_stage.campaign_verdict(decision)
        else:
            verdict = tune_stage.resume_campaign(self, state)
        if verdict is None:
            self.journal.event(
                "tune stage", "the campaign did not end; the same command resumes it"
            )
            return
        state.tune["verdict"] = verdict
        self._checkpoint(state)
        said = f"ended `{verdict}`; the onboarding's verdict stays `{state.verdict}`"
        self.journal.event("tune stage", said)
        print_message(f"[bold]tune stage {escape_markup(said)}[/bold]", log)

    def _close_run(self, state: WorkflowState, log, *, verdict: str) -> None:
        state.done = True
        state.verdict = verdict
        self._checkpoint(state)
        metrics = metrics_mod.write(self.workspace, opened=self.journal.opened_at())
        cells = contract_mod.load_cells(self.workspace)
        self.journal.close(
            summary="\n".join(
                [
                    f"- verdict: **{verdict}**",
                    f"- stage reached: `{state.stage}`",
                    f"- cells that hold: {', '.join(sorted(cells['cells'])) or 'none'}"
                    + (
                        f"; the most at once on `{cells['best']['sha'][:10]}`"
                        + (
                            f" (branch `{cells['best']['ref']}`)"
                            if cells["best"].get("ref")
                            else ""
                        )
                        if cells["best"]
                        else ""
                    ),
                    f"- metrics: `{metrics}`",
                    "",
                    "### Gates",
                    "",
                    "```",
                    self.gate_log.render() or "(none run)",
                    "```",
                ]
            )
        )
        print_message(
            f"[bold]run closed — verdict {verdict}[/bold]",
            log,
        )


def campaign_owed(state: WorkflowState) -> bool:
    """Whether the onboarding closed beside a campaign that has not ended yet."""
    return bool(state.tune.get("onboarded")) and not state.tune.get("verdict")


@dataclass
class _TurnStart:
    """What a turn is asked with, on which layer, and whether that layer is a fork to close."""

    prompt: str
    layer: Any | None = None
    forked: bool = False

    def close(self) -> None:
        if self.forked and self.layer is not None:
            self.layer.__exit__(None, None, None)


# ----------------------------------------------------------- the target's fingerprint

#: Under the workspace: each list of files a verdict read of the tree (``_narrowed``).
READS_DIRNAME = "reads"


def gate_relevant(path: str, *, keep: Collection[str] = (), denied: Sequence[str] = ()) -> bool:
    """Whether a change to ``path``, relative to the checkout, can change what a gate measures.

    ``denied`` are the run's own paths (its workspace), never relevant;
    ``keep`` are files a gate reads where nothing else does
    (:func:`gate_inputs`), always relevant.
    """
    if path.startswith(tuple(denied)):
        return False
    if path in keep:
        return True
    return not (path.startswith(UNGATED_PREFIXES) or path.lower().endswith(UNGATED_SUFFIXES))


def gate_inputs(task_data: Mapping[str, Any], repo: str | Path) -> frozenset[str]:
    """The files a gate reads whole under the ungated paths, relative to the checkout.

    The image a multimodal boot sends
    when the spec names none, and every file inside the checkout that a
    gate-facing block of the spec names (:data:`GATE_INPUT_BLOCKS`): an
    options file, a prompts file, an image, a script in a command, and the
    accuracy anchor tables.
    """
    root = Path(repo)
    keep = {boot_checks.DEFAULT_BOOT_IMAGE}
    for block in GATE_INPUT_BLOCKS:
        for word in _path_words(task_data.get(block)):
            path = _in_checkout(word, root)
            if path is not None:
                keep.add(path)
    return frozenset(keep)


def _in_checkout(name: str, root: Path) -> str | None:
    """``name``, a path absolute or relative to ``root``, as a file in the checkout; else ``None``."""
    candidate = Path(name) if os.path.isabs(name) else root / name
    try:
        if candidate.is_file():
            return candidate.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        pass
    return None


def _path_words(value: Any) -> Iterator[str]:
    """Every word of every string in ``value`` that could name a checkout file."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _path_words(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _path_words(item)
    elif isinstance(value, str):
        for word in value.split():
            word = word.strip("'\"").rpartition("=")[2]
            if word:
                yield word


def tree_blobs(listing: str) -> dict[str, str]:
    """``{path: object id}`` from ``git ls-tree -r -z``."""
    blobs: dict[str, str] = {}
    for entry in listing.split("\0"):
        meta, tab, path = entry.partition("\t")
        fields = meta.split()
        if tab and len(fields) == 3:
            blobs[path] = fields[2]
    return blobs


def porcelain_paths(status: str) -> Iterator[tuple[str, str]]:
    """``(path, XY)`` for each entry of ``git status --porcelain -z``, a rename's source too."""
    fields = status.split("\0")
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if len(entry) < 4:
            continue
        code = entry[:2]
        yield entry[3:], code
        if "R" in code or "C" in code:
            # The source follows as a field of its own.
            if index < len(fields) and fields[index]:
                yield fields[index], code
            index += 1


def native_libraries(repo: Path) -> list[str]:
    """Each native library built into the checkout: ``<path> <size> <mtime_ns>``, sorted."""
    found: set[str] = set()
    for directory in NATIVE_LIBRARY_DIRS:
        for pattern in LIBRARY_PATTERNS:
            for path in (repo / directory).glob(pattern):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                found.add(f"{path.relative_to(repo).as_posix()} {stat.st_size} {stat.st_mtime_ns}")
    return sorted(found)


def fingerprint(blobs: Mapping[str, str], libraries: Sequence[str]) -> str:
    """The marker for a tree's ``{path: blob}`` and its native libraries."""
    digest = hashlib.sha1()
    for path in sorted(blobs):
        digest.update(os.fsencode(f"{path}\0{blobs[path]}\n"))
    for library in libraries:
        digest.update(os.fsencode(f"native\0{library}\n"))
    return f"tree:{digest.hexdigest()[:24]}"


__all__ = [
    "DEFAULT_MAX_PARALLEL_TASKS",
    "DEFAULT_REVISION_CAP",
    "GATE_INPUT_BLOCKS",
    "HOST_FACTS_FILENAME",
    "NATIVE_LIBRARY_DIRS",
    "SKILL_DIRS",
    "TROUBLESHOOTING_HINT",
    "UNGATED_PREFIXES",
    "UNGATED_SUFFIXES",
    "TrtllmModelingV2Workflow",
    "fingerprint",
    "gate_inputs",
    "gate_relevant",
    "native_libraries",
    "porcelain_paths",
    "tree_blobs",
]
