# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm-modeling-v2 workflow: the run's lifecycle.

Phase ordering, check sequencing and disk verification are control flow, so
they live in code; judgement -- how to derive a forward, what to build next
-- is the agents'. This module owns init, run, resume, checkpoint, turn
dispatch and close:

1. ``preflight`` -- Python, plus one probe turn per agent configuration
   when turns run in process. A broken host fails here.
2. ``planned``   -- the planner's task graph, driven until the contract holds
   (``run/stages/planned.py``, ``orchestrator.py``), then delivery. With
   ``perf.enabled`` the graph holds a ``tune`` task, which starts once the
   core is correct and runs beside the remaining work.
3. ``tune``      -- only for ``trtllm-modeling-v2 tune``: a target onboarded
   elsewhere, tuned on its own.

Every stage checkpoints before advancing, so a crash resumes where it was.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Mapping, Sequence

import yaml
from rich.markup import escape as escape_markup

from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig, SessionConfig
from agent_flow.console import print_message
from agent_flow.logger import get_logger

from ..agents.prompts import PromptBundle, build_trtllm_modeling_v2_prompts
from ..agents.roles import configured_agent, resolve_role_agent
from ..checks import gates as gates_mod
from ..infra.committer import CommitPolicy
from ..infra.driver import reload as reload_mod
from ..infra.environment import RunEnvironment
from ..infra.execution import CommandExecutor, TurnExecutor, TurnRequest, select_turn_executor
from ..infra.execution.agent_gate import AgentGate, Priority, turn_priority
from ..infra.execution.turns import TurnTimeout
from ..infra.substrate.managed import ManagedSessions
from ..infra.substrate.rebuild import LIBRARY_PATTERNS
from ..infra.vcs import git_output
from ..spec import variants as variants_mod
from ..spec.task_schema import assembly_units, load_and_validate_task_yaml, resolve_per_unit
from ..tune import stage as tune_stage
from . import contract as contract_mod
from . import metrics as metrics_mod
from . import orchestrator as orchestrator_mod
from . import primers as primers_mod
from .journal import Journal
from .preflight import resolve_model_key
from .progress import ProgressContext, current_decision, init_progress_file
from .stages import (
    AssembleStage,
    DeliverStage,
    KernelWriterStage,
    PlannedStage,
    PlanStage,
    PreflightStage,
)
from .stages.common import (
    ACCEPTANCE_GATED_UNITS,
    HOST_FACTS_FILENAME,
    TROUBLESHOOTING_HINT,
    WITNESS_GATED_UNITS,
    _compose_required_tools_hooks,
    _same_tree,
    _SessionLost,
    remote_turns,
)
from .state import (
    STAGE_PLANNED,
    STAGE_PREFLIGHT,
    STAGE_TUNE,
    STATE_FILENAME,
    WorkflowState,
    is_remediable,
    load_state,
    save_state,
)

#: How many tasks run at once: the CLI's --max-parallel-tasks default.
DEFAULT_MAX_PARALLEL_TASKS = orchestrator_mod.DEFAULT_MAX_PARALLEL_TASKS
#: How many graph revisions without a check turning green end the run.
DEFAULT_REVISION_CAP = 6
#: While tasks, checks or a planner turn are in flight, how long a task that finished
#: with its checks green waits to reach the planner with the events after it. A
#: failure, a worker's report, an operator's message, a failed re-check, or nothing left to run
#: or start wakes the planner at once.
DEFAULT_EVENT_WINDOW_S = 60.0
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
#: The accuracy key, of a unit or a variant, that names an anchor table: an
#: accuracy reference file, one entry per model, shared by every model in the tree.
ANCHOR_KEY = "reference_yaml"
#: What a change may move device memory through: code an engine process runs
#: or builds, and the configuration it reads (see :func:`memory_relevant`).
MEMORY_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".cu",
    ".cuh",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".so",
    ".yaml",
    ".yml",
    ".json",
)


class TrtllmModelingV2Workflow(
    PreflightStage,
    PlanStage,
    KernelWriterStage,
    AssembleStage,
    DeliverStage,
    PlannedStage,
):
    """Drive one checkpoint to a gated modeling_v2 target."""

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
        agent_transport: Any | None = None,
        agent_turn_timeout: float | None = None,
        entry_stage: str = "",
        task_worktrees: bool = True,
        event_window_s: float = DEFAULT_EVENT_WINDOW_S,
        primers: bool = True,
        manage_sessions: bool = False,
    ) -> None:
        self.workspace = Path(workspace)
        #: Replaces the process with the code now at the checkout, once a reload
        #: drained: the CLI sets it to the `os.execve` of its own command line.
        #: None, and a reload request waits for a process that can.
        self.reexec: Callable[[], None] | None = None
        # Each tree-changing task in its own worktree, integrated when it
        # finishes (run/integration.py); off, tasks share the checkout.
        self.task_worktrees = task_worktrees
        self.event_window_s = event_window_s
        # The planner's, kernel writers' and assemblers' turns start as forks of one
        # briefing session per role (run/primers.py); off, each starts afresh.
        self.use_primers = primers
        # The run submits, renews and releases its own sessions (infra/substrate/managed.py).
        self.manage_sessions = manage_sessions
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
        self.progress_path = self.workspace / "progress.yaml"
        self.journal_path = self.workspace / "journal.md"
        self.boot_spec_path = self.workspace / "boot.yaml"
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
        #: How a role turn is delivered, resolved once: spawned here, or through
        #: the filesystem to a session that can authenticate (``agent_transport``).
        self._turns: TurnExecutor = select_turn_executor(
            agent_transport,
            lambda request: self._layer(request.role),
            timeout_s=agent_turn_timeout,
        )
        #: Where this run's commands land. Resolved from the environment in
        #: `_run_preflight`, which is the first moment one exists; until
        #: then nothing has a command to run.
        self._command_executor: CommandExecutor | None = None
        #: Why each command the resolved runner issued was ended by its
        #: session rather than by itself, in order. See `_noting_session_loss`.
        self._lost_commands: list[str] = []
        # Queue items this process has itself dispatched a kernel writer for
        # (`_run_one_kernel_writer`).
        self._kernel_writers_dispatched: set[int] = set()
        self._shell_runner = shell_runner or gates_mod._default_shell_runner

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            for path in (
                self.state_path,
                self.plan_path,
                self.vocabulary_path,
                self.progress_path,
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
                # The previous run's ledger and reloads: this one has no stop before it.
                reload_mod.INFLIGHT_FILENAME,
                reload_mod.RECORD_FILENAME,
                reload_mod.REQUEST_FILENAME,
                reload_mod.RESTARTS_FILENAME,
                reload_mod.SEEN_FILENAME,
            ):
                (self.workspace / name).unlink(missing_ok=True)
            # Regenerated by the next preflight, but a run that fails before
            # reaching it would otherwise leave the previous checkpoint's
            # survey sitting in the workspace looking current.
            (self.workspace / "surface.md").unlink(missing_ok=True)

        self.resume = self.state_path.is_file()
        if not self.resume:
            init_progress_file(self.progress_path)

        self.journal = Journal(path=self.journal_path)
        #: One milestone recorded at a time.
        self._milestone_lock = threading.Lock()
        self.gate_log = gates_mod.GateLog()
        #: Compiled at preflight; see `contract.py`.
        self._contract: contract_mod.Contract | None = None
        self._status: contract_mod.ContractStatus | None = None
        self._progress_ctx = ProgressContext(path=self.progress_path)

        self.task_data: dict[str, Any] = {}
        self.environment: RunEnvironment | None = None
        self.prompts: PromptBundle = prompts or build_trtllm_modeling_v2_prompts()
        self._layers: dict[str, AgentLayer] = {}
        # Set by ``_run_preflight``, which every path through ``run`` reaches
        # before anything reads it, a resume's included.
        self._preflight: Any = None
        # The surface survey, until preflight takes it.
        self._surface_block: str = (
            "## Surface survey\n\nNot taken for this run — plan on the assumption that "
            "you do not know what exists, and say so in the plan's open items.\n"
        )
        self._runtime_block: str = ""
        self._kernel_sources_block: str = ""

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

    def _make_agent(
        self,
        name: str,
        system_prompt: str,
        *,
        requires_decision: bool = True,
        session_mode: str = "stateless",
        fork_from: str | None = None,
    ) -> AgentLayer:
        # Backend, model and reasoning effort come from `task.yaml`'s `agents`
        # block, over the defaults in `roles.py`.
        agent = resolve_role_agent(self.task_data, name)
        # The role reports by writing its decision file, and the Stop hook
        # refuses a turn that wrote nothing. `Write` is a weak requirement --
        # the role writes other files too -- but the enum is validated when
        # the file is read. Codex refuses per-client Claude SDK hooks and
        # writes with its own tools, so it gets no hook.
        hooks = _compose_required_tools_hooks(["Write"] if requires_decision else [])
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
                    # repository it is editing.
                    cwd=self.workspace,
                ),
                session=SessionConfig(mode=session_mode, fork_from=fork_from),
            )
        )

    def _run_turn(
        self,
        role: str,
        what: str,
        prompt: str,
        *,
        iteration: int | None = None,
        layer: Any | None = None,
        unit: str = "",
        work: str = "",
    ) -> str:
        """Run one role turn, wherever this run's transport delivers it.

        Both transports answer `TurnExecutor.run`, so nothing here branches on
        which one is in force. ``layer`` is the caller's own session, which a
        concurrent batch owns rather than the executor. ``work`` names the
        work the turn is for where ``what`` does not (:attr:`TurnRequest.work`).
        A turn forked from its role's primer that fails drops the primer and
        runs again plainly.
        """
        agent = configured_agent(self.task_data, role)
        block = getattr(self._tls, "worktree", "")
        if block:
            prompt = f"{prompt}\n\n{block}\n"
        start = self._turn_start(role, prompt, layer)
        while True:
            request = TurnRequest(
                role=role,
                what=what,
                prompt=start.prompt,
                system_prompt=getattr(self.prompts, role, "") if self.prompts else "",
                backend=agent.backend if agent else None,
                model=agent.model if agent else None,
                reasoning_effort=agent.reasoning_effort if agent else None,
                decision_file=(
                    str(current_decision(self.workspace, role, iteration, unit=unit))
                    if iteration is not None
                    else None
                ),
                work=work,
            )
            executor = self._turns if start.layer is None else self._turns.bound(start.layer)
            try:
                with self._agent_gate.slot(), self._turn_in_flight(role, what):
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
            start = self._turn_start(role, prompt, layer)

    #: How often a run that manages its sessions looks at them.
    MANAGED_TICK_S = 60.0

    @contextlib.contextmanager
    def _managing_sessions(self) -> Iterator[None]:
        """While the block runs, keep the run's sessions submitted and renewed, when it manages them."""
        if not self.manage_sessions:
            yield
            return
        managed = ManagedSessions(self.task_data, self.workspace)
        done = threading.Event()

        def tick() -> None:
            while not done.wait(self.MANAGED_TICK_S):
                try:
                    said = managed.tick()
                except Exception as exc:  # the next tick tries again
                    said = [f"the sessions could not be looked at: {type(exc).__name__}: {exc}"]
                if said:
                    self.journal.event("sessions", "\n".join(said))

        thread = threading.Thread(target=tick, name="mv2-sessions", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()

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
            want = primers_mod.version(role, getattr(self.prompts, role, ""))
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
            commit = (self._git(repo, "rev-parse", "HEAD") or "").strip() if repo else ""
            summary = primers_mod.summary_path(self.workspace, role)
            summary.parent.mkdir(parents=True, exist_ok=True)
            remote = remote_turns(self) is not None
            layer = None if remote else self._make_agent(
                role, getattr(self.prompts, role, ""), requires_decision=False
            )  # fmt: skip
            # Only a backend that can fork a session hands its turns this one.
            forks = not remote and primers_mod.can_fork(resolve_role_agent(self.task_data, role))
            request = TurnRequest(
                role=role,
                what="primer",
                prompt=primers_mod.primer_prompt(
                    role, repo=repo, commit=commit, summary=summary, forks=forks
                ),
                system_prompt=getattr(self.prompts, role, ""),
            )
            executor = self._turns if layer is None else self._turns.bound(layer)
            with turn_priority(Priority.SIDE), self._agent_gate.slot():
                executor.run(request)
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
            if layer is not None:
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

    def _turn_start(self, role: str, prompt: str, layer: Any | None) -> "_TurnStart":
        """How a turn of ``role`` starts: as a fork of its primer, pointed at its summary, or as asked.

        Only a transport that runs turns on this process's layers can fork one.
        """
        primer = self._primed(role)
        if primer is not None and primer.session_id and remote_turns(self) is None:
            forked = self._make_agent(
                role, getattr(self.prompts, role, ""), fork_from=primer.session_id
            )
            return _TurnStart(primers_mod.primed_preamble(primer.commit) + prompt, forked, True)
        if primer is not None and primer.summary and Path(primer.summary).is_file():
            note = primers_mod.summary_note(Path(primer.summary), primer.commit)
            return _TurnStart(note + prompt, layer)
        return _TurnStart(prompt, layer)

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

    def _turn_in_flight(self, role: str, what: str) -> contextlib.AbstractContextManager:
        """A turn a reload must wait for: one in this process, which a re-exec would end.

        A turn on the external transport outlives the process, and the next
        one adopts it, so a reload does not wait for it. One in this process
        is waited for, and starts even while a reload drains: the planner's
        thread, which the drain waits for, may be the one asking.
        """
        inflight = getattr(self, "_inflight", None)
        if inflight is None or remote_turns(self) is not None:
            return contextlib.nullcontext()
        return inflight.hold("turn", f"{role} {what}", door=False)

    def _layer(self, name: str) -> AgentLayer:
        """Build a role's layer on first use and keep it for the run.

        Every role is **stateless**: what it needs from earlier turns is on disk.
        """
        if name not in self._layers:
            self._layers[name] = self._make_agent(name, getattr(self.prompts, name))
        return self._layers[name]

    def _materialize_skills(self) -> Path:
        """Copy the packaged skills into this run's workspace, where every backend finds them.

        Claude Code reads `.claude/skills`, Codex `.agents/skills`; both are the
        session's cwd-relative skill directories. Returns the first.
        """
        source = Path(__file__).resolve().parent.parent / "agents" / "skills"
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

        try:
            # Phase 0: a substrate gate that fails while the first turns run stops them.
            with orchestrator_mod.stoppable(self.workspace) as stop:
                self._run_stages(state, log, stop)
            self._await_substrate()
        except _SessionLost as lost:
            # Raised by `_gate_until_verdict` from any gate or witness; the
            # graph and the verdicts are on disk, so re-running resumes.
            stage = state.stage
            print_message(
                f"[bold red]✘ {escape_markup(str(lost))}. No verdict was recorded for it; "
                f"re-run the same command once the session is back.[/bold red]",
                log,
            )
            self.journal.section(f"session lost during {stage}", str(lost))
            self._close_run(state, log, verdict=f"session-lost:{stage}")
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — re-run the same command to continue "
                "from the last checkpoint, or pass --clean to start fresh"
                "[/bold yellow]",
                log,
            )
            self.journal.event("run interrupted", f"stage={state.stage}")
            raise

    def _run_stages(self, state: WorkflowState, log, stop: Callable[[str], None]) -> None:
        """Preflight, then the stage the run is at, to its close.

        Toward the planned run the substrate gate runs beside its first turns
        (``stop`` ends them when it fails), and a run with no graph yet starts
        with the survey (:meth:`_start_with_the_survey`). A run the failed
        gate stopped is left open, as a failed preflight is: re-running the
        same command proves the substrate again and resumes.
        """
        beside = stop if state.stage != STAGE_TUNE else None
        if state.stage == STAGE_PREFLIGHT:
            if not self._run_preflight(state, log, stop=beside):
                return
            if self.dry_run:
                print_message(
                    "[bold cyan]→ --dry-run: preflight passed; stopping before the "
                    "first agent turn.[/bold cyan]",
                    log,
                )
                return
            state.stage = STAGE_PLANNED
            self._checkpoint(state)
        else:
            # Resuming past preflight still needs the resolved
            # environment, since every prompt and every gate carries it.
            if not self._run_preflight(state, log, stop=beside):
                return
            if self.dry_run:
                print_message(
                    "[bold cyan]→ --dry-run: preflight passed; stopping before the "
                    f"{state.stage} stage.[/bold cyan]",
                    log,
                )
                return

        if state.stage == STAGE_PLANNED:
            self.journal.mark_started()
            self._start_with_the_survey()
            if not self._run_planned(state, log):
                return
            tuned = state.tune.get("phase") == "done"
            self._close_run(state, log, verdict="tuned" if tuned else "delivered")
            return

        if state.stage == STAGE_TUNE:
            verdict = tune_stage.run_tune_phase(self, state, log)
            if verdict is None:
                return
            self._close_run(state, log, verdict=verdict)
            return

        self._close_run(state, log, verdict="delivered")

    # --------------------------------------------------------- target identity

    def _target_marker(self) -> str:
        """A content fingerprint of the tree every check measures, or ``""``.

        Every check measures a snapshot of the commit the checkout is at
        (``run/snapshots.py``), so the fingerprint is of that commit's whole
        tree (git's blob ids), less what no gate runs or reads
        (`gate_relevant`) but for the files a gate does read there
        (`gate_inputs`), and the run's own workspace. Of an anchor table only
        the entries the accuracy gates read count (`anchor_tables`). The
        native libraries built into the checkout are outside git, so their
        sizes and modification times are part of it. Uncommitted changes are
        in no snapshot, so they leave the marker as it is.

        Git is asked from this process, never through the gate runner: on a
        spool every command's output opens with a banner naming its job.

        ``""`` means **not established**, and every caller reads it as
        "assume changed".
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
        return self._commit_marker(words[0]) if words else ""

    def _commit_marker(self, head: str) -> str:
        """Commit ``head``'s marker, as :meth:`_target_marker` describes it.

        The anchor entries are read from the commit, and the native libraries
        are the checkout's: every snapshot links them in. ``""`` when git
        cannot list the commit or read a table of it.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        if not repo or not head:
            return ""
        policy = CommitPolicy.from_task(self.task_data, workspace=self.workspace)
        try:
            committed = self._committed_blobs(repo, head)
            if committed is None:
                return ""
            keep, tables = _gate_inputs(self.task_data, repo)
            blobs = {
                path: blob
                for path, blob in committed.items()
                if gate_relevant(path, keep=keep | set(tables), denied=policy.denied_paths)
                and (
                    path not in tables or gate_relevant(path, keep=keep, denied=policy.denied_paths)
                )
            }
            libraries = native_libraries(Path(repo))
            entries = self._anchor_entries(repo, head, committed, tables)
        except (OSError, ValueError):
            # Unreadable, or a name git printed that is not text: unknown, not unchanged.
            return ""
        return fingerprint(blobs, libraries, entries)

    def _narrow_marker(self, head: str, reads: Collection[str]) -> str:
        """Commit ``head``'s marker narrowed to what a check read of it; ``""`` when unknown.

        :meth:`_commit_marker` over the same files, less every Python file not
        in ``reads``: the Python a check's processes imported is all the Python
        that ran, while a file of any other kind (a header, a config, a table)
        may be read by code no import names, so each stays in. The native
        libraries and the anchor entries stay in too.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        if not repo or not head:
            return ""
        wanted = set(reads)
        policy = CommitPolicy.from_task(self.task_data, workspace=self.workspace)
        try:
            committed = self._committed_blobs(repo, head)
            if committed is None:
                return ""
            keep, tables = _gate_inputs(self.task_data, repo)
            blobs = {
                path: blob
                for path, blob in committed.items()
                if (not path.endswith(".py") or path in wanted)
                and gate_relevant(path, keep=keep | set(tables), denied=policy.denied_paths)
                and (
                    path not in tables or gate_relevant(path, keep=keep, denied=policy.denied_paths)
                )
            }
            libraries = native_libraries(Path(repo))
            entries = self._anchor_entries(repo, head, committed, tables)
        except (OSError, ValueError):
            return ""
        return fingerprint(blobs, libraries, entries)

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
                partial = path.with_suffix(f".{os.getpid()}.tmp")
                partial.write_text(json.dumps(sorted(reads)), encoding="utf-8")
                os.replace(partial, path)
        except OSError:
            return "", ""
        marker = self._narrow_marker(commits.pop(), reads)
        return (ident, marker) if marker else ("", "")

    def _verdict_holds(self, verdict: contract_mod.Verdict) -> bool:
        """Whether ``verdict`` holds on the target as it is: what it read of the tree is unchanged.

        Its narrowed marker (:meth:`_narrow_marker`) on the checkout's ``HEAD``
        is the one it recorded. Asked once per head and list.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
        if not words or not verdict.reads or not verdict.reads_marker:
            return False
        key = (words[0], verdict.reads)
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
            cache[key] = self._narrow_marker(words[0], reads) if reads else ""
        return bool(cache[key]) and cache[key] == verdict.reads_marker

    def _anchor_entries(
        self, repo: str, head: str, blobs: Mapping[str, str], tables: Mapping[str, Collection[str]]
    ) -> list[str]:
        """:func:`anchor_entries` of ``tables`` as commit ``head`` holds them; the last asked is kept."""
        key = (
            repo,
            head,
            tuple(sorted((path, tuple(sorted(keys))) for path, keys in tables.items())),
        )
        cached = self.__dict__.get("_head_anchors")
        if cached is not None and cached[0] == key:
            return cached[1]
        entries = anchor_entries(tables, self._blob_reader(repo, blobs))
        self.__dict__["_head_anchors"] = (key, entries)
        return entries

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

    def _blob_reader(self, repo: str, blobs: Mapping[str, str]) -> Callable[[str], str | None]:
        """A path's text as ``blobs`` holds it; ``None`` when they do not hold it.

        Raises ``OSError`` when git cannot read a blob they do hold.
        """

        def read(path: str) -> str | None:
            blob = blobs.get(path)
            if blob is None:
                return None
            text = self._git(repo, "cat-file", "blob", blob)
            if text is None:
                raise OSError(f"git cannot read {path} ({blob})")
            return text

        return read

    # ------------------------------------------------------------------ helpers

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
        gates_mod.default_triton_cache(self.task_data, self.workspace)

        if self.resume:
            state = load_state(self.state_path)
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
            # (environment, prompts, skills, commit helper); the stage then
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
        words = (self._git(repo, "rev-parse", "HEAD") or "").split()
        return words[0] if words and self._target_marker() == marker else None

    def _point_branch(self, suffix: str, sha: str, reason: str, unwritten: str = "") -> str:
        """Point ``<target branch>-<suffix>`` at ``sha``; the ref, or ``""`` when git refused.

        The target branch is the checkout's, or the run's own with ``HEAD``
        detached. A refusal is journaled under ``unwritten`` when it is given.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        branch = (self._git(repo, "symbolic-ref", "--quiet", "--short", "HEAD") or "").strip()
        state = getattr(self, "_planned_state", None)
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
            sha = self._milestone_commit(marker, evaluation, contract_mod.bringup_checks(contract))
            if sha is None:
                return None
            ref = self._point_branch(
                "bringup",
                sha,
                "trtllm-modeling-v2: the first unit's boot and accuracy hold at one rank",
                unwritten="bring-up ref not written",
            )
            opened = self.journal.opened_at()
            elapsed = time.time() - opened.timestamp() if opened is not None else None
            held = contract_mod.bringup_checks(contract)
            milestone = contract_mod.Milestone(
                sha=sha,
                marker=marker,
                ref=ref,
                at=time.time(),
                elapsed_s=elapsed,
                checks=len(held),
                tier=contract_mod.MILESTONE_BRINGUP,
            )
            contract_mod.record_milestone(self.workspace, milestone)
        parts = [
            f"{elapsed / 3600:.1f} h after the run started ({opened.isoformat()})"
            if opened is not None and elapsed is not None
            else "the journal does not say when the run started",
            f"{', '.join(c.key for c in held)} hold on it",
        ]
        if ref:
            parts.append(f"branch `{ref}`")
        detail = "; ".join(parts)
        self.journal.event(f"BRINGUP DELIVERED {sha}", detail)
        print_message(
            f"[bold green]✔ BRINGUP DELIVERED {sha[:10]}: {escape_markup(detail)}[/bold green]",
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

    def _report_instruction(self, role: str) -> str:
        """The one-line reminder at the end of a dispatch; the system prompt carries the protocol."""
        return "Before ending your turn, write your decision file to the path below."

    def _close_run(self, state: WorkflowState, log, *, verdict: str) -> None:
        state.done = True
        state.verdict = verdict
        self._checkpoint(state)
        if self.manage_sessions and self.task_data:
            released = ManagedSessions(self.task_data, self.workspace).release()
            if released:
                self.journal.event("sessions released", "\n".join(released))
        metrics = metrics_mod.write(
            self.workspace, opened=self.journal.opened_at(), task_data=self.task_data
        )
        cells = contract_mod.load_cells(self.workspace)
        self.journal.close(
            summary="\n".join(
                [
                    f"- verdict: **{verdict}**",
                    f"- stage reached: `{state.stage}`",
                    f"- human stops: **{state.human_stops}**",
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
            f"[bold]run closed — verdict {verdict}; human stops: {state.human_stops}[/bold]",
            log,
        )


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


def memory_relevant(path: str, *, denied: Sequence[str] = ()) -> bool:
    """Whether a change to ``path`` may move the device memory an engine leaves free.

    A gate-relevant path (:func:`gate_relevant`) that is code an engine runs or
    builds, or configuration it reads (:data:`MEMORY_SUFFIXES`), or a directory
    that may hold either, as a task's declared ``touches`` names one.
    """
    if not gate_relevant(path, denied=denied):
        return False
    name = path.rstrip("/").rsplit("/", 1)[-1]
    return path.endswith("/") or "." not in name or name.lower().endswith(MEMORY_SUFFIXES)


def gate_inputs(task_data: Mapping[str, Any], repo: str | Path) -> frozenset[str]:
    """The files a gate reads whole under the ungated paths, relative to the checkout.

    The quickstart every engine is built by, the image a multimodal boot sends
    when the spec names none, and every file inside the checkout that a
    gate-facing block of the spec names (:data:`GATE_INPUT_BLOCKS`): an
    options file, a prompts file, an image, a script in a command, and an
    anchor table the accuracy gates do not read by entry alone
    (:func:`anchor_tables`).
    """
    return _gate_inputs(task_data, repo)[0]


def anchor_tables(task_data: Mapping[str, Any], repo: str | Path) -> dict[str, frozenset[str]]:
    """The anchor tables in the checkout the gates read by entry: ``{path: model keys}``.

    An accuracy gate reads only the entry of the unit's model key
    (`gates.load_reference_score`), so a table is fingerprinted by those
    entries (:func:`anchor_entries`). A table stays whole (:func:`gate_inputs`)
    when anything else may read more of it (a block that names it under
    another key, a command line say) or a unit's model key cannot be resolved.
    """
    return _gate_inputs(task_data, repo)[1]


def _gate_inputs(
    task_data: Mapping[str, Any], repo: str | Path
) -> tuple[frozenset[str], dict[str, frozenset[str]]]:
    """:func:`gate_inputs` and :func:`anchor_tables`, found together."""
    root = Path(repo)
    named = _named_files(task_data, root)
    by_entry: dict[str, set[str]] = {}
    whole = set(_named_files(_without_anchors(task_data), root))
    for declared, model_key in _anchor_reads(task_data):
        path = _in_checkout(declared, root)
        if path is None:
            continue
        if model_key:
            by_entry.setdefault(path, set()).add(model_key)
        else:
            whole.add(path)
    tables = {path: frozenset(keys) for path, keys in by_entry.items() if path not in whole}
    return frozenset((named | whole) - set(tables)), tables


def _named_files(task_data: Mapping[str, Any], root: Path) -> frozenset[str]:
    """The quickstart, the default boot image, and every file in the checkout a gate-facing block names."""
    keep = {
        gates_mod.DEFAULT_BOOT_IMAGE,
        gates_mod.quickstart_script({"trtllm_repo_path": "."}).as_posix(),
    }
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


def _without_anchors(task_data: Mapping[str, Any]) -> dict[str, Any]:
    """The gate-facing blocks with every :data:`ANCHOR_KEY` taken out, a variant's included."""

    def strip(accuracy: Any) -> Any:
        if not isinstance(accuracy, Mapping):
            return accuracy
        return {key: value for key, value in accuracy.items() if key != ANCHOR_KEY}

    blocks = {block: task_data.get(block) for block in GATE_INPUT_BLOCKS}
    blocks["accuracy"] = strip(blocks["accuracy"])
    if isinstance(blocks["variants"], Mapping):
        blocks["variants"] = {
            name: {**entry, "accuracy": strip(entry.get("accuracy"))}
            if isinstance(entry, Mapping)
            else entry
            for name, entry in blocks["variants"].items()
        }
    return blocks


def _anchor_reads(task_data: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    """``(anchor table, model key)`` as each unit's and each variant's accuracy gate resolves them.

    The key is ``""`` when it cannot be resolved: no checkpoint path to derive it from.
    """
    specs: list[tuple[Mapping[str, Any], str]] = [
        (task_data, unit) for unit in assembly_units(task_data)
    ]
    # The members it resolves are not read here, so any world size does.
    specs += [
        (variants_mod.variant_spec(task_data, name, world_size=1), name)
        for name in variants_mod.variants(task_data)
    ]
    for spec, unit in specs:
        declared = resolve_per_unit(spec.get("accuracy") or {}, ANCHOR_KEY, unit)
        if not str(declared or "").strip():
            continue
        try:
            model_key = resolve_model_key(spec, unit)
        except KeyError:
            model_key = ""
        yield str(declared).strip(), model_key


def anchor_entries(
    tables: Mapping[str, Collection[str]], read: Callable[[str], str | None]
) -> list[str]:
    """One line per entry a gate reads of ``tables`` (:func:`anchor_tables`): table, key, digest.

    ``read`` gives a table's text by its path in the checkout, ``None`` when
    it is not there. A table that is not there, or holds no entry for the key,
    reads ``absent``; one that does not parse is identified by its whole text.
    """
    lines: list[str] = []
    for path in sorted(tables):
        text = read(path)
        try:
            data = None if text is None else _parsed_table(text)
        except yaml.YAMLError:
            unparsed = "unparsed:" + hashlib.sha1(text.encode("utf-8")).hexdigest()
            lines.extend(f"{path}\0{key}\0{unparsed}" for key in sorted(tables[path]))
            continue
        for key in sorted(tables[path]):
            digest = "absent"
            if isinstance(data, Mapping) and key in data:
                entry = json.dumps(data[key], sort_keys=True, default=str)
                digest = hashlib.sha1(entry.encode("utf-8")).hexdigest()
            lines.append(f"{path}\0{key}\0{digest}")
    return lines


@functools.lru_cache(maxsize=8)
def _parsed_table(text: str) -> Any:
    """An anchor table's text, parsed; one parse per content, since the marker is asked often."""
    return yaml.safe_load(text)


def _path_words(value: Any) -> Iterator[str]:
    """Every word of every string in ``value`` that could name a file: one with a ``/``."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _path_words(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _path_words(item)
    elif isinstance(value, str):
        for word in value.split():
            word = word.strip("'\"").rpartition("=")[2]
            if "/" in word:
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


def fingerprint(
    blobs: Mapping[str, str], libraries: Sequence[str], anchors: Sequence[str] = ()
) -> str:
    """The marker for a tree's ``{path: blob}``, its native libraries and its anchor entries."""
    digest = hashlib.sha1()
    for path in sorted(blobs):
        digest.update(os.fsencode(f"{path}\0{blobs[path]}\n"))
    for library in libraries:
        digest.update(os.fsencode(f"native\0{library}\n"))
    for entry in anchors:
        digest.update(os.fsencode(f"anchor\0{entry}\n"))
    return f"tree:{digest.hexdigest()[:24]}"


__all__ = [
    "ACCEPTANCE_GATED_UNITS",
    "ANCHOR_KEY",
    "DEFAULT_MAX_PARALLEL_TASKS",
    "DEFAULT_REVISION_CAP",
    "GATE_INPUT_BLOCKS",
    "HOST_FACTS_FILENAME",
    "MEMORY_SUFFIXES",
    "NATIVE_LIBRARY_DIRS",
    "SKILL_DIRS",
    "TROUBLESHOOTING_HINT",
    "UNGATED_PREFIXES",
    "UNGATED_SUFFIXES",
    "WITNESS_GATED_UNITS",
    "TrtllmModelingV2Workflow",
    "anchor_entries",
    "anchor_tables",
    "fingerprint",
    "gate_inputs",
    "gate_relevant",
    "memory_relevant",
    "native_libraries",
    "porcelain_paths",
    "tree_blobs",
]


if __name__ == "__main__":
    from ..cli import main

    main()
