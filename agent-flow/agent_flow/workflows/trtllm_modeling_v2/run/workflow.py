# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm-modeling-v2 workflow: the run's lifecycle.

Two stages, each checkpointed before the next, so a crash resumes where it was:

1. ``preflight`` -- Python, plus one probe turn per agent configuration
   (``run/startup``). A broken host or spec fails here, before any worker.
2. ``execution`` -- the tasks in their fixed order, then the certification of
   the final commit (``run/pipeline.py``, ``run/certify.py``).

The run closes with one verdict: ``delivered``, ``escalated`` or
``certification-failed``, and ``delivery_report.md`` says what stands behind
it. A run that closed without delivering is re-entered by the same command
once the operator has repaired what it names.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

import yaml

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
from ..checks import check_runtime
from ..checks import verdict as check_result
from ..infra.compute.commands import CommandExecutor
from ..infra.compute.environment import RunEnvironment
from ..infra.git import CommitPolicy, git_output
from ..operators.coverage import own_paths
from ..spec.task_schema import load_and_validate_task_yaml
from . import certify as certify_mod
from . import contract as contract_mod
from . import pipeline as pipeline_mod
from .journal import Journal
from .pipeline import DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_ROUNDS, Pipeline
from .startup import bootstrap
from .startup.bootstrap import HOST_FACTS_FILENAME, TROUBLESHOOTING_HINT
from .state import (
    STAGE_EXECUTION,
    STAGE_PREFLIGHT,
    STATE_FILENAME,
    WorkflowState,
    is_remediable,
    load_state,
    save_state,
)

#: Where the run's skills go under the workspace, each agent session's cwd:
#: Claude Code reads `.claude/skills`, Codex `.agents/skills`.
SKILL_DIRS: tuple[str, ...] = (".claude/skills", ".agents/skills")
#: What ``--clean`` removes of an earlier run in the same workspace.
RUN_FILES: tuple[str, ...] = (
    STATE_FILENAME,
    "plan.md",
    "vocabulary.yaml",
    "journal.md",
    HOST_FACTS_FILENAME,
    pipeline_mod.TASKS_FILENAME,
    contract_mod.STATUS_FILENAME,
    certify_mod.CERTIFICATION_FILENAME,
    certify_mod.REPORT_FILENAME,
    "surface.md",
    "prior_art.md",
    "tree_map.md",
)
RUN_DIRS: tuple[str, ...] = ("reports", "decisions", certify_mod.CERTIFY_DIRNAME)


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
    """Drive one checkpoint to a certified modeling_v2 target."""

    #: Whether a new run starts with the survey; off, the workspace's vocabulary is taken.
    survey_first = True

    def __init__(
        self,
        workspace: Path,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        clean: bool = False,
        dry_run: bool = False,
        prompts: PromptBundle | None = None,
        command_runner: Any | None = None,
        shell_runner: Any | None = None,
        git_runner: Any | None = None,
        agent_turn_timeout: float | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.task_path = self.workspace / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self.journal_path = self.workspace / "journal.md"
        self.host_facts_path = self.workspace / HOST_FACTS_FILENAME
        self.state_path = self.workspace / STATE_FILENAME

        self.max_attempts = max(1, max_attempts)
        self.max_rounds = max(1, max_rounds)
        self.dry_run = dry_run
        self._prompts_override = prompts

        # Injected so every stage is testable without a GPU, a cluster, or a
        # model. The defaults are the real ones.
        self._command_runner = command_runner
        # An explicit runner always wins; otherwise preflight resolves one from
        # the environment, and until then it is the plain shell.
        self._shell_runner_override = shell_runner
        # Questions about the checkout under test, asked on this host:
        # ``(repo, *args) -> stdout | None``.
        self._git = git_runner or git_output
        #: How a role turn runs: one agent CLI process per turn, here.
        self._turns = InProcessTurnExecutor(
            lambda request: self._layer(request.role), timeout_s=agent_turn_timeout
        )
        #: Where this run's commands land, resolved from the environment at preflight.
        self._command_executor: CommandExecutor | None = None
        self._shell_runner = shell_runner or check_runtime.default_shell_runner

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            for name in RUN_FILES:
                (self.workspace / name).unlink(missing_ok=True)
            for name in RUN_DIRS:
                shutil.rmtree(self.workspace / name, ignore_errors=True)

        self.resume = self.state_path.is_file()

        self.journal = Journal(path=self.journal_path)
        self.gate_log = check_result.GateLog()
        #: Compiled at preflight; see ``contract.py``.
        self._contract: contract_mod.Contract | None = None
        self._status: contract_mod.ContractStatus | None = None
        self._state: WorkflowState | None = None

        self.task_data: dict[str, Any] = {}
        self.environment: RunEnvironment | None = None
        self.prompts: PromptBundle = prompts or build_trtllm_modeling_v2_prompts()
        self._layers: dict[str, AgentLayer] = {}
        self._layer_lock = threading.Lock()
        self._agent_configs: dict[str, AgentConfig] | None = None
        # Set by preflight, which every path through ``run`` reaches first, a resume's included.
        self._preflight: Any = None
        # The surface survey, until preflight takes it.
        self._surface_block: str = (
            "## Surface survey\n\nNot taken for this run — plan on the assumption that "
            "you do not know what exists, and say so in the plan's open items.\n"
        )
        # The tree's prior art and the engine's limits for this checkpoint (preflight).
        self._prior_art_block: str = ""
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
        """Pin role configuration once, shared by the probe and every task turn."""
        if self._agent_configs is None:
            self._agent_configs = resolve_role_agents(self.task_data)
        return self._agent_configs

    def _make_agent(self, name: str, system_prompt: str) -> AgentLayer:
        # Every role runs on the backend, model and effort the spec states for it.
        agent = self._role_agents()[name]
        # The role reports by writing its decision file, and the Stop hook
        # refuses a turn that wrote nothing. Codex refuses per-client Claude
        # SDK hooks and writes with its own tools, so it gets no hook.
        hooks = require_tool_call_stop_hook(["Write"]) if agent.backend != "codex" else None
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
                    # materialized skills are the ones the agent discovers.
                    # The user's own skills and plugins stay out: no role is
                    # briefed on them, and an unattended session should not
                    # reach Slack or mail.
                    cwd=self.workspace,
                    writable_roots=self._writable_roots(),
                    disabled_skills=self._foreign_skills.get(agent.backend, ()),
                    plugins=False,
                ),
                # Every role is stateless: what it needs from earlier turns is on disk.
                session=SessionConfig(mode="stateless"),
            )
        )

    def _writable_roots(self) -> tuple[str, ...]:
        """What a role writes outside the workspace: the checkout.

        A task's worktree commits into the checkout's ``.git``.
        """
        root = self.task_data.get("trtllm_repo_path")
        return (str(root),) if root and Path(root).is_dir() else ()

    def _layer(self, name: str) -> AgentLayer:
        """Build a role's layer on first use and keep it for the run."""
        with self._layer_lock:
            if name not in self._layers:
                self._layers[name] = self._make_agent(name, getattr(self.prompts, name))
            return self._layers[name]

    def _run_turn(
        self, role: str, what: str, prompt: str, *, decision_file: Path | None = None
    ) -> str:
        """Run one turn of ``role`` and return its prose; its decision is in ``decision_file``."""
        agent = self._role_agents()[role]
        request = TurnRequest(
            role=role,
            what=what,
            prompt=prompt,
            system_prompt=getattr(self.prompts, role, "") if self.prompts else "",
            backend=agent.backend,
            model=agent.model,
            reasoning_effort=agent.reasoning_effort,
            decision_file=str(decision_file) if decision_file is not None else None,
        )
        return self._turns.run(request).text

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

    def run(self, task: str | Path) -> bool:
        """Run, or resume, to the verdict; whether the run is delivered (a dry run: preflight passed)."""
        log = get_logger().console
        state = self._init_state(task, log)
        if state is None:
            return load_state(self.state_path).verdict == "delivered"
        self._role_agents()
        try:
            return self._run_stages(state, log)
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — re-run the same command to continue "
                "from the last checkpoint, or pass --clean to start fresh"
                "[/bold yellow]",
                log,
            )
            self.journal.event("run interrupted", f"stage={state.stage}")
            raise

    def _run_stages(self, state: WorkflowState, log) -> bool:
        """Preflight, then the pipeline to the run's verdict; whether it is delivered.

        Preflight runs on every start, a resume's included: every prompt and
        every check carries the environment it resolves. A failed preflight
        leaves the run open: the same command resumes once the host is fixed.
        """
        self._state = state
        if not bootstrap._run_preflight(self, state, log):
            return False
        if self.dry_run:
            print_message(
                "[bold cyan]→ --dry-run: preflight passed; stopping before the first agent "
                "turn.[/bold cyan]",
                log,
            )
            return True
        if state.stage == STAGE_PREFLIGHT:
            state.stage = STAGE_EXECUTION
            self._checkpoint(state)
        self.journal.mark_started()
        pipeline = Pipeline(self, state, log)
        verdict = pipeline.run()
        self._close_run(state, log, verdict=verdict, pipeline=pipeline)
        return verdict == "delivered"

    # ------------------------------------------------------------------ helpers

    def _head(self) -> str:
        """The commit the checkout is at; ``""`` when git cannot say."""
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
        return words[0] if words else ""

    def _commit_policy(self) -> CommitPolicy:
        """The run's commit rules: who commits, and that a run changes only its own paths."""
        target_rel = str(getattr(self._preflight, "target_rel", "") or "")
        return CommitPolicy.from_task(
            self.task_data, workspace=self.workspace, own_paths=own_paths(target_rel)
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
            if state.done and is_remediable(state.verdict):
                # The run closed on something the operator fixes outside it, and running
                # its own tasks and checks again is the only way to see the fix.
                print_message(
                    f"[bold cyan]→ re-entering the run: it closed '{state.verdict}', which "
                    f"names a repair made outside it[/bold cyan]",
                    log,
                )
                self.journal.event("re-entered a closed run", f"verdict={state.verdict}")
                self._reopen(state)
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

    def _reopen(self, state: WorkflowState) -> None:
        """A closed run re-entered after a repair: its failed tasks and its rounds start over."""
        state.done, state.verdict = False, ""
        state.round, state.round_open, state.failed_checks = 0, False, []
        path = self.workspace / pipeline_mod.TASKS_FILENAME
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return
        for row in data.get("tasks") or ():
            if isinstance(row, dict) and row.get("status") == pipeline_mod.FAILED:
                row["status"], row["attempts"] = pipeline_mod.PENDING, 0
        data["repairs"] = 0
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _refresh_workspace_spec(self, source: Path, log) -> None:
        """Keep the workspace ``task.yaml``, which the agents read, identical to the spec the checks use.

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

    def _checkpoint(self, state: WorkflowState) -> None:
        save_state(self.state_path, state)

    def _gate_env(self) -> RunEnvironment:
        """The environment a check runs under: the run's, as preflight resolved it."""
        assert self.environment is not None, "preflight must run before any check"
        return self.environment

    def _close_run(self, state: WorkflowState, log, *, verdict: str, pipeline: Pipeline) -> None:
        state.done = True
        state.verdict = verdict
        self._checkpoint(state)
        report = certify_mod.write_report(self, verdict, pipeline.certification, pipeline.table())
        summary = [f"- verdict: **{verdict}**", f"- report: `{report}`"]
        if pipeline.certification is not None and pipeline.certification.commit:
            summary.append(f"- commit: `{pipeline.certification.commit}`")
        if pipeline.escalation:
            summary.append(f"- escalated by the planner: {pipeline.escalation}")
        summary += ["", "### Checks", "", "```", self.gate_log.render() or "(none run)", "```"]
        self.journal.close(summary="\n".join(summary))
        colour = "green" if verdict == "delivered" else "red"
        print_message(
            f"[bold {colour}]run closed — verdict {verdict}; report {report}[/bold {colour}]", log
        )


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_ROUNDS",
    "HOST_FACTS_FILENAME",
    "SKILL_DIRS",
    "TROUBLESHOOTING_HINT",
    "TrtllmModelingV2Workflow",
]
