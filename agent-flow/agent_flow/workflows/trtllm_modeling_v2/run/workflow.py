# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm-modeling-v2 workflow: the run's lifecycle.

Phase ordering, check sequencing and disk verification are control flow, so
they live in code; judgement -- how to derive a forward, what to build next
-- is the agents'. This module owns init, run, resume, checkpoint, turn
dispatch and close:

1. ``preflight`` -- Python, plus one probe turn per agent configuration.
   A broken host fails here; the allocation probe, which loads no model, runs
   beside the first turns of the next stage and stops them when it fails.
2. ``execution`` -- the task graph code makes from the survey's rows and the spec's
   deployments, driven until the contract holds (``run/execution/tasks.py``,
   ``orchestrator.py``), then delivery. A task that cannot go on is blocked and asks
   the outer agent session (``run/escalation.py``). The run ends there:
   performance work is not the flow's.

Every stage checkpoints before advancing, so a crash resumes where it was.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

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
from ..agents.turns import InProcessTurnExecutor, TurnRequest
from ..checks import check_runtime, native_tests
from ..checks import session as session_mod
from ..checks import verdict as check_result
from ..infra.compute import pool as pool_mod
from ..infra.compute.commands import CommandExecutor
from ..infra.compute.environment import RunEnvironment
from ..infra.files import publish_once, write_text_atomic
from ..infra.git import CommitPolicy, git_output
from ..spec.task_schema import load_and_validate_task_yaml
from . import escalation as escalation_mod
from . import keys as keys_mod
from . import orchestrator as orchestrator_mod
from .execution.handoff import current_decision
from .execution.tasks import ExecutionStage, _start_with_the_survey
from .records import report as report_mod
from .records.journal import Journal
from .records.state import (
    STAGE_EXECUTION,
    STAGE_PREFLIGHT,
    STATE_FILENAME,
    WorkflowState,
    is_remediable,
    load_state,
    save_state,
)
from .startup import bootstrap
from .startup.bootstrap import HOST_FACTS_FILENAME, TROUBLESHOOTING_HINT
from .validation import contract as contract_mod
from .workspace.integration import place as tree_place

#: How many tasks run at once: the CLI's --max-parallel-tasks default.
DEFAULT_MAX_PARALLEL_TASKS = orchestrator_mod.DEFAULT_MAX_PARALLEL_TASKS
#: How many attempts one task gets: the CLI's --max-attempts default.
DEFAULT_MAX_ATTEMPTS = orchestrator_mod.graph_mod.MAX_ATTEMPTS
#: The write-once record of the first deployment's session passing, beside its git ref.
ONBOARDED_FILENAME = report_mod.ONBOARDED_FILENAME
#: Where the run's skills go under the workspace, each agent session's cwd:
#: Claude Code reads `.claude/skills`, Codex `.agents/skills`.
SKILL_DIRS: tuple[str, ...] = (".claude/skills", ".agents/skills")


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
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        clean: bool = False,
        dry_run: bool = False,
        prompts: PromptBundle | None = None,
        command_runner: Any | None = None,
        shell_runner: Any | None = None,
        git_runner: Any | None = None,
        agent_turn_timeout: float | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        #: The run ended with everything left blocked: the process exits ``escalated``.
        self._escalated = False
        #: Why a start did not start (its preflight, its first holder, its agents): the
        #: process then exits with a status that is not 0, and its exit record says this.
        self._not_started = ""
        #: The shapes the run's ready and running tasks need of the pool, once it has a
        #: task list (``GraphOrchestrator.demand``); ``None`` before.
        self._task_demand: Callable[[], Mapping[int, float]] | None = None
        #: The final pass as the execution stage's loop judged it, for the closing report
        #: (``GraphOrchestrator.final_record``); ``None`` when this process judged none.
        self._final_record: Mapping[str, Any] | None = None
        #: Per worker thread: the worktree block its turns carry.
        self._tls = threading.local()
        self.task_path = self.workspace / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self.journal_path = self.workspace / "journal.md"
        self.host_facts_path = self.workspace / HOST_FACTS_FILENAME
        self.state_path = self.workspace / STATE_FILENAME

        self.max_parallel_tasks = max(1, max_parallel_tasks)
        self.max_attempts = max(1, max_attempts)
        self.dry_run = dry_run
        self._prompts_override = prompts

        # Injected so every stage is testable without a GPU, a cluster, or a
        # model. The defaults are the real ones.
        self._command_runner = command_runner
        # An explicit runner always wins; otherwise preflight resolves one from
        # the environment, and until then it is the plain shell.
        self._shell_runner_override = shell_runner
        # Questions about the checkout under test, asked on this host:
        # ``(repo, *args) -> stdout | None``. Never the gate runner; see
        # `_head`.
        self._git = git_runner or git_output
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
            # An answer left for an earlier run's task is no answer to this run's, and a
            # worker's notes of that task (its card names the file) are not this run's.
            tasks_dir = self.workspace / escalation_mod.TASKS_DIRNAME
            for pattern in (f"*{escalation_mod.ANSWER_SUFFIX}", "*.notes.md"):
                for left in tasks_dir.glob(pattern):
                    left.unlink(missing_ok=True)
            for name in (
                "graph.yaml",
                "work.yaml",
                escalation_mod.CHANGES_FILENAME,
                ONBOARDED_FILENAME,
                report_mod.REPORT_FILENAME,
                contract_mod.STATUS_FILENAME,
            ):
                (self.workspace / name).unlink(missing_ok=True)
            # Regenerated by the next preflight, but a run that fails before
            # reaching it would otherwise leave the previous checkpoint's
            # survey sitting in the workspace looking current.
            (self.workspace / "prior_art.json").unlink(missing_ok=True)
            (self.workspace / "tree_map.md").unlink(missing_ok=True)

        self.resume = self.state_path.is_file()
        self.clean_start = clean

        self.journal = Journal(path=self.journal_path)
        #: One milestone recorded at a time.
        self._onboarded_lock = threading.Lock()
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
        # The engine's own model of the architecture and its limits for this checkpoint
        # (preflight's `_take_prior_art`): a section of the survey's dispatch.
        self._prior_art_block: str = ""
        self._prior_art: Any = None
        self._engine_limits: list[Any] = []
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
        """Pin role configuration once, shared by probes and every task turn."""
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
        # Nothing wakes a worker once its turn ends, and `Monitor` is a tool to be woken
        # by: an assembler that ended its turn on one hung until the outer session ended
        # it. Codex has no such tool and refuses a name it cannot enforce.
        banned = () if agent.backend == "codex" else ("Monitor",)
        return AgentLayer(
            AgentLayerConfig(
                name=name,
                system_prompt=system_prompt,
                disallowed_tools=banned,
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
                session=SessionConfig(mode="stateless"),
            )
        )

    def _writable_roots(self) -> tuple[str, ...]:
        """What a role writes outside the workspace: the checkout, the run's spools, its trees.

        A task's tree commits into the checkout's ``.git``, its work is submitted to the
        spools, and the tree itself is on this node's own disk unless the spec states another
        place (``run/workspace/integration.py``).
        """
        roots = [self.task_data.get("trtllm_repo_path")]
        if self.environment is not None:
            env = self.environment
            roots += [env.spool]
        trees = tree_place(self.task_data, self.workspace)
        if trees is not None and roots[0] and (Path(str(roots[0])) / ".git").exists():
            trees.mkdir(parents=True, exist_ok=True)
            roots.append(trees)
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
        (:attr:`TurnRequest.work`). How many turns run at once is the orchestrator's
        one counter (``max_parallel_tasks``): a turn is only started with one in hand.
        """
        agent = self._role_agents()[role]
        block = getattr(self._tls, "worktree", "")
        if block:
            prompt = f"{prompt}\n\n{block}\n"
        request = TurnRequest(
            role=role,
            what=what,
            prompt=prompt,
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
        return self._turns.run(request).text

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
            # An allocation probe that fails while the first turns run stops them.
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
        finally:
            # No work left ends the pool; a run that stops short leaves its holders to the
            # command that resumes it, and a local watcher never outlives this process.
            bootstrap.end_pool(self, state, log)

    def _run_stages(
        self, state: WorkflowState, log, stop: Callable[[str], None] | None = None
    ) -> None:
        """Preflight, then the stage the run is at, to its close.

        Toward the execution stage the allocation probe runs beside its first
        turns (``stop`` ends them when it fails), and an execution stage with no
        graph yet starts with the survey (:meth:`_start_with_the_survey`). A run
        the failed probe stopped is left open, as a failed preflight is:
        re-running the same command probes again and resumes.
        """
        self._state = state
        beside = stop if not state.done else None
        if state.stage == STAGE_PREFLIGHT:
            if not bootstrap._run_preflight(self, state, log, stop=beside):
                self._not_started = self._not_started or "its preflight did not pass"
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
                self._not_started = self._not_started or "its preflight did not pass"
                return
            if self.dry_run:
                print_message(
                    "[bold cyan]→ --dry-run: preflight passed; stopping before the "
                    f"{state.stage} stage.[/bold cyan]",
                    log,
                )
                return

        if state.stage == STAGE_EXECUTION:
            self.journal.mark_started()
            _start_with_the_survey(self)
            execution = ExecutionStage(self)
            if execution.run_execution(state, log):
                self._close_run(state, log, verdict="delivered")
            return

        raise ValueError(f"unsupported workflow stage: {state.stage}")

    # --------------------------------------------------------- target identity

    def _head(self) -> str:
        """The commit the checkout is at; ``""`` when git cannot say.

        Git is asked from this process, never through the gate runner: on a
        spool every command's output opens with a banner naming its job.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
        return words[0] if words else ""

    def _check_key(self, check: contract_mod.Check, commit: str = "") -> str:
        """The key of ``check``'s evidence on ``commit`` of the target; ``""`` when not established.

        ``commit`` defaults to the one the checkout is at. What the check consumes there,
        by content (``run/keys.py``): the files of the commit as its class of check sees
        them, the native build, the checker, and its inputs as the run's contract has
        them. A row recorded under the same key is the check's result on any commit that
        has it. Uncommitted changes are in no pin, so they leave a key as it is.
        """
        gates = (session_mod.GATE_NAME, native_tests.GATE_NAME)
        name = f"{check.gate}:{check.unit}" if check.gate in gates else check.key
        return self._key(name, commit)

    def _key(self, check_id: str, commit: str = "", extra: Any = None) -> str:
        """The key of ``check_id``'s evidence on ``commit`` (the checkout's, by default)."""
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        commit = commit or self._head()
        if not repo or not commit:
            return ""
        try:
            entries = self._committed_entries(repo, commit)
        except (OSError, ValueError):
            return ""
        if entries is None:
            return ""
        return keys_mod.key(
            self.task_data,
            self.workspace,
            commit,
            check_id,
            inputs_of=self._inputs_of,
            entries=entries,
            extra=extra,
        )

    def _inputs_of(self, check_id: str) -> str | None:
        """The inputs digest of the contract's check ``check_id`` names; ``None`` when it has none."""
        name, _, ref = check_id.partition(":")
        contract = self._contract
        if contract is None:
            try:
                contract = contract_mod.compile_contract(self.task_data)
            except Exception:  # a spec no contract compiles from says nothing of its inputs
                return None
        return next((c.inputs for c in contract.checks if c.unit == ref and c.gate == name), None)

    def _committed_entries(self, repo: str, head: str) -> dict[str, str] | None:
        """Every path in ``head``'s tree with its mode and object id; the last head asked is kept."""
        cached = self.__dict__.get("_head_entries")
        if cached is not None and cached[0] == (repo, head):
            return cached[1]
        entries = keys_mod.committed(repo, head, self._git)
        if entries is None:
            return None
        self.__dict__["_head_entries"] = ((repo, head), entries)
        return entries

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
        keep = keys_mod.gate_inputs(self.task_data, repo)
        return sorted(
            {
                path
                for path, _ in porcelain_paths(dirt)
                if keys_mod.consumed(keys_mod.SESSION, path, keep=keep, denied=policy.denied_paths)
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
        if not self.clean_start:
            # What the outer session changed through `resolve` stands over the file the user
            # wrote, which is never rewritten; each change is in the record with its reason.
            for change in escalation_mod.reapply_changes(self.workspace, self.task_data):
                self.journal.event(
                    "spec change kept",
                    f"{change.get('key')}: {change.get('from')!r} -> {change.get('to')!r} "
                    f"({change.get('why')})",
                )
        # The resolved spec states the environment's kind and the run's one spool.
        pool_mod.resolve_spool(self.task_data, self.workspace)
        check_runtime.default_triton_cache(self.task_data, self.workspace)

        if self.resume:
            older = self._older_workspace()
            if older:
                print_message(f"[bold red]✘ {escape_markup(older)}[/bold red]", log)
                self._not_started = older
                return None
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
        self._checkpoint(state)
        return state

    def _older_workspace(self) -> str:
        """Why this workspace cannot be resumed by this flow, in one line; ``""`` when it can.

        A state file at a stage, or a graph with a task of a kind, that this flow no longer has
        was written by an older flow: nothing here runs them.
        """
        try:
            stage = json.loads(self.state_path.read_text(encoding="utf-8")).get("stage")
        except (OSError, ValueError, AttributeError):
            stage = None
        what = []
        if stage is not None and stage not in (STAGE_PREFLIGHT, STAGE_EXECUTION):
            what.append(f"its state is at stage {stage!r}")
        graph_mod = orchestrator_mod.graph_mod
        removed = graph_mod.removed_kinds(self.workspace / graph_mod.GRAPH_FILENAME)
        if removed:
            what.append(f"its graph holds {', '.join(removed)}")
        if not what:
            return ""
        return (
            f"{self.workspace} was written by an older flow ({'; '.join(what)}), which this "
            f"flow has no code for: start a new workspace, or pass --clean to start this one "
            f"afresh"
        )

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

    def _mark_onboarded(self, task: Any = None) -> dict[str, Any] | None:
        """Pin ``onboarded``, once: the first deployment's session passed on this commit.

        A git ref that is created and never moved (``refs/mv2/<run>/onboarded``) and
        ``onboarded.json`` beside the run's records, published whole and once: the commit,
        the identity of the tree at that commit (the ledger's marker of it), the content
        record of its native build, the session's checks and when; what the session failed
        nothing on and could not verify (``unverified``); and the requirements in force
        then (the frozen anchors, and how many changes had been recorded). Nothing later
        moves either: a later repair makes a later commit, and a later change of a
        requirement is listed after it, never attributed to it.

        A record that is missing or not whole while the ref stands (a publication that was
        cut) is written again from the ref: its commit, and that commit's marker and build.
        What only the moment of publication knew (when, under which requirements) is then
        said to be not recorded, not filled in from the present.
        """
        path = self.workspace / ONBOARDED_FILENAME
        with self._onboarded_lock:
            held = report_mod.onboarded(self.workspace)
            if held is not None and held.get("commit"):
                return held
            repo = str(self.task_data.get("trtllm_repo_path") or "")
            checks = list(getattr(task, "satisfies", ()) or ())
            # What the session could not verify, as its task says it: nothing failed.
            unverified = [str(u) for u in getattr(task, "unverified", ()) or ()]
            state = getattr(self, "_state", None)
            ref = f"refs/mv2/{getattr(state, 'slug', '') or 'run'}/onboarded"
            standing = (self._git(repo, "rev-parse", "--verify", "--quiet", ref) or "").strip()
            recovered = bool(standing)
            measured_key = ""
            if not standing:
                commit = ""
                if self._contract is not None and checks:
                    # The commit the session's passing rows were measured on: each row names
                    # what it imported. Not the checkout's head, and not a row looked up
                    # under the head's key: another task may have landed since the session
                    # was measured, and that commit was never measured.
                    status = self._contract_status()
                    owed = [c for c in self._contract.checks if c.key in checks]
                    commit = contract_mod.measured_commit(status, [(c, None) for c in owed]) or ""
                    rows = (contract_mod.row(status.verdicts, c, None) for c in owed)
                    measured_key = next((r.marker for r in rows if r is not None and r.passed), "")
                if not commit:
                    commit = self._head()
                    moved = (
                        measured_key
                        and commit
                        and measured_key
                        != self._key(
                            f"{session_mod.GATE_NAME}:{getattr(task, 'ref', '') or ''}", commit
                        )
                    )
                    if moved:
                        # No row says which commit was measured, and the head is not it.
                        unverified.append(
                            "the commit the session measured: its row names none, and the "
                            "checkout has moved since; pinned at the checkout's head"
                        )
                if not commit:
                    self.journal.event("onboarded", "not pinned: the checkout names no commit")
                    return None
                # The empty old value: git writes the ref only where there is none.
                self._git(repo, "update-ref", ref, commit, "")
                standing = (self._git(repo, "rev-parse", "--verify", "--quiet", ref) or "").strip()
                if not standing:
                    raise OSError(f"git did not write {ref}: `onboarded` is not pinned")
            build = ""
            pinned = getattr(getattr(self, "_snapshots", None), "pinned", None)
            if pinned is not None:
                try:
                    from ..infra.compute.pintree import record_digest

                    build = record_digest(pinned(standing).build)
                except Exception as exc:  # the record says what it could not name
                    build = f"not recorded ({type(exc).__name__}: {exc})"
            record: dict[str, Any] = {
                "ref": ref,
                "commit": standing,
                # The key of the session's evidence at that commit, not of the checkout as
                # it is now.
                "key": measured_key
                or self._key(f"{session_mod.GATE_NAME}:{getattr(task, 'ref', '') or ''}", standing),
                "build": build,
                "deployment": str(getattr(task, "ref", "") or ""),
                "session": str(getattr(task, "id", "") or ""),
                "checks": checks,
            }
            if recovered:
                # The ref stood and its record did not: when it was pinned, and under
                # which requirements, is what the lost record knew.
                record.update(at=None, recovered_at=time.time(), requirements=None, changes=None)
            else:
                record.update(
                    at=time.time(),
                    requirements=report_mod.requirements(self.workspace),
                    changes=len(report_mod.changes(self.workspace)),
                )
            if unverified:
                record["unverified"] = unverified
            text = json.dumps(record, indent=1, sort_keys=True) + "\n"
            if not publish_once(path, text):
                # What is there is no record (it was read above and is not one): replaced whole.
                write_text_atomic(path, text)
        short = (
            f"; not verified: {'; '.join(record['unverified'])}" if record.get("unverified") else ""
        )
        how = " (its record was written again from the ref)" if recovered else ""
        self.journal.event("onboarded", f"{record['commit']} ({record['ref']}){short}{how}")
        print_message(
            f"[bold green]✔ onboarded at {record['commit'][:12]}[/bold green]"
            + (f"[bold yellow]{escape_markup(short)}[/bold yellow]" if short else "")
        )
        return record

    # ------------------------------------------- what a resolution reaches
    def _respec(self, keys: Sequence[str]) -> None:
        """The spec changed under ``keys`` (``resolve``): the agents' copy and the run follow.

        The spec is changed in place, so every reader of ``task_data`` has the value; the
        workspace copy the agents read is written again, and the environment the gates run
        in is resolved again where one of its keys changed.
        """
        self.task_path.write_text(
            yaml.safe_dump(self.task_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        self.journal.event("spec changed", ", ".join(keys))
        # The obligations follow the spec: each check's inputs are what it is judged by, and
        # a result recorded with the old ones is no longer that check's result.
        if self._contract is not None:
            try:
                contract = contract_mod.compile_contract(self.task_data)
            except Exception as exc:  # the old contract stands, and the journal says so
                self.journal.event("contract not compiled again", f"{type(exc).__name__}: {exc}")
            else:
                self._contract = contract
                contract_mod.write_contract(self.workspace, contract)
        if any(key.split(".")[0] == "environment" for key in keys) and self.environment is not None:
            from ..infra.compute.environment import resolve_environment

            try:
                self.environment = resolve_environment(self.task_data)
            except Exception as exc:  # the old environment stands, and the journal says so
                self.journal.event("environment not resolved again", f"{type(exc).__name__}: {exc}")

    def _pool_owes(self, pool: Any) -> dict[int, float]:
        """What the pool is asked for now: the run's tasks' shapes, and what is in the spool."""
        return pool_mod.spool_demand(pool.paths.root, bootstrap.run_demand(self))

    def _pool_retry(self, close_unknown: int | None, note: str) -> str:
        """``resolve pool-retry``: close a record nothing lists when asked, then ask again now."""
        pool = getattr(self, "_pool", None)
        if pool is None:
            return "this run keeps no pool: nothing to retry"
        said = []
        if close_unknown is not None:
            closed = pool.close(int(close_unknown), note)
            said.append(f"record {close_unknown} ({closed.get('name')}) closed")
        events = pool.reconcile(self._pool_owes(pool), retry_now=True)
        said.append("asked again: " + ("; ".join(e.render() for e in events) or "nothing stands"))
        return "; ".join(said)

    def _pool_adjust(self) -> str:
        """``resolve pool-adjust``: the pool's environment is built again from the changed spec."""
        pool = getattr(self, "_pool", None)
        if pool is None:
            return "this run keeps no pool: the next start reads the changed values"
        fresh = pool_mod.build(
            self.task_data,
            workspace=self.workspace,
            note=lambda said: self.journal.event("pool", said),
        )
        pool.env = fresh.env
        events = pool.reconcile(self._pool_owes(pool), retry_now=True)
        return "the pool took the new values and asked again: " + (
            "; ".join(e.render() for e in events) or "nothing stands"
        )

    def _checkpoint(self, state: WorkflowState) -> None:
        save_state(self.state_path, state)

    def _gate_env(self) -> RunEnvironment:
        """The environment a gate runs under: the run's, as preflight resolved it."""
        assert self.environment is not None, "preflight must run before any gate"
        return self.environment

    def _close_run(self, state: WorkflowState, log, *, verdict: str) -> None:
        state.done = True
        state.verdict = verdict
        self._checkpoint(state)
        # The closing report is the one place that says where the run stands: onboarded,
        # what changed, what is not done or verified, the final pass. The journal names it.
        report = self._write_report(verdict)
        self.journal.close(
            summary="\n".join(
                [
                    f"- verdict: **{verdict}**",
                    f"- stage reached: `{state.stage}`",
                    f"- closing report: `{report}`" if report else "- closing report: not written",
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
            f"[bold]run closed — verdict {verdict}[/bold]"
            + (f"; the closing report is {escape_markup(str(report))}" if report else ""),
            log,
        )

    def _write_report(self, verdict: str) -> Path | None:
        """Write the closing report, ``delivery_report.md``; its path, or ``None`` when it cannot be written."""
        target = str(getattr(self._preflight, "target_rel", "") or "")
        # The node-hours asked so far against the run's bound: the pool's own line.
        asked = ""
        try:
            so_far = pool_mod.asked_of(self.task_data)
            if so_far is not None and so_far.holder_jobs:
                asked = so_far.render()
        except Exception:  # what the pool cannot say is left out of the report, not guessed
            asked = ""
        try:
            path = report_mod.write(
                self.workspace,
                verdict=verdict,
                target=target,
                gates=self.gate_log.render(),
                asked=asked,
                final=self._final_record,
            )
        except OSError as exc:
            self.journal.event("report not written", f"{type(exc).__name__}: {exc}")
            return None
        self.journal.event("report", str(path))
        return path


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


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_PARALLEL_TASKS",
    "HOST_FACTS_FILENAME",
    "SKILL_DIRS",
    "TROUBLESHOOTING_HINT",
    "TrtllmModelingV2Workflow",
    "porcelain_paths",
]
