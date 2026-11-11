# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtllm-modeling-v2 workflow: one start of a run, from preflight to the report.

Every start does the same: read the spec again, preflight (``run/start.py``), the tasks
in their fixed order (``run/pipeline.py``), the final pass and the report
(``run/report.py``). What an earlier start did is in the workspace's files, so the same
command goes on from where the last one stopped: done tasks stay done, a cut turn
restarts, tasks not done get fresh attempts within their grants. A run is **onboarded**
once the first deployment's boot and benchmarks passed on the commit that landed; that is
what :meth:`TrtllmModelingV2Workflow.run` returns and what the process exits 0 on.
"""

from __future__ import annotations

import hashlib
import shutil
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
from ..agents.roles import resolve_role_agents
from ..agents.turns import InProcessTurnExecutor, TurnRequest
from ..checks import check_runtime, conventions
from ..checks import verdict as check_result
from ..infra.compute.commands import CommandExecutor
from ..infra.compute.environment import RunEnvironment
from ..infra.git import CommitPolicy, git_output
from ..spec.task_schema import load_and_validate_task_yaml
from . import contract as contract_mod
from . import pipeline as pipeline_mod
from . import report as report_mod
from . import start as start_mod
from .integration import COMMIT_LABELS_FILENAME, own_paths
from .journal import Journal
from .pipeline import DEFAULT_MAX_ATTEMPTS, Pipeline
from .start import HOST_FACTS_FILENAME
from .startup.preflight import PREFLIGHT_FILENAME
from .state import STATE_FILENAME, WorkflowState, load_state, save_state

#: What ``--clean`` removes of an earlier run in the same workspace. The frozen anchors,
#: the operator's notes and the onboarded libraries stay with the workspace.
RUN_FILES: tuple[str, ...] = (
    STATE_FILENAME,
    "plan.md",
    "vocabulary.yaml",
    "journal.md",
    "surface.md",
    HOST_FACTS_FILENAME,
    pipeline_mod.TASKS_FILENAME,
    contract_mod.STATUS_FILENAME,
    report_mod.REPORT_FILENAME,
    PREFLIGHT_FILENAME,
    conventions.CONVENTIONS_FILENAME,
    COMMIT_LABELS_FILENAME,
)
RUN_DIRS: tuple[str, ...] = ("decisions", report_mod.CHECKS_DIRNAME)


class TrtllmModelingV2Workflow:
    """Drive one checkpoint to an onboarded modeling_v2 target, and as far past it as it gets."""

    #: Whether a new run starts with the survey; off, the workspace's vocabulary is taken.
    survey_first = True

    def __init__(
        self,
        workspace: Path,
        *,
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
        self.task_path = self.workspace / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self.host_facts_path = self.workspace / HOST_FACTS_FILENAME
        self.state_path = self.workspace / STATE_FILENAME
        self.max_attempts = max(1, max_attempts)
        self.dry_run = dry_run
        self._prompts_override = prompts
        # Injected so every stage is testable without a GPU, a cluster or a model: the
        # runner of preflight's probe, the runner of every check (an explicit one always
        # wins over the environment's), and ``(repo, *args) -> stdout | None`` for the
        # questions asked of the checkout.
        self._command_runner = command_runner
        self._shell_runner_override = shell_runner
        self._shell_runner = shell_runner or check_runtime.default_shell_runner
        self._git = git_runner or git_output
        #: How a role turn runs: one agent CLI process per turn, here.
        self._turns = InProcessTurnExecutor(
            lambda request: self._layer(request.role), timeout_s=agent_turn_timeout
        )
        #: Where this run's commands land, resolved from the environment at preflight.
        self._command_executor: CommandExecutor | None = None
        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            for name in RUN_FILES:
                (self.workspace / name).unlink(missing_ok=True)
            for name in RUN_DIRS:
                shutil.rmtree(self.workspace / name, ignore_errors=True)
        self.journal = Journal(path=self.workspace / "journal.md")
        self.gate_log = check_result.GateLog()
        self.task_data: dict[str, Any] = {}
        self.environment: RunEnvironment | None = None
        self.prompts: PromptBundle = prompts or build_trtllm_modeling_v2_prompts()
        self._layers: dict[str, AgentLayer] = {}
        self._agent_configs: dict[str, AgentConfig] | None = None
        # Set at the start of every run (``run/start.py``): the checks each deployment
        # owes, the verdict ledger, the state, what preflight resolved and recorded.
        self._contract: contract_mod.Contract | None = None
        self._status: contract_mod.ContractStatus | None = None
        self._state: WorkflowState | None = None
        self._preflight: Any = None
        self._surface_block: str = (
            "## Surface survey\n\nNot taken for this run: plan on the assumption that "
            "you do not know what exists, and say so in the plan's open items.\n"
        )
        # Backend kind -> the skills its sessions load besides the run's own (the user's),
        # which the agent probe reads and every role then runs without.
        self._foreign_skills: dict[str, tuple[str, ...]] = {}

    def __enter__(self) -> "TrtllmModelingV2Workflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
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
        """A role's layer, on the backend, model and effort the spec states for it."""
        agent = self._role_agents()[name]
        root = self.task_data.get("trtllm_repo_path")
        backend = BackendConfig(
            kind=agent.backend,
            model=agent.model,
            reasoning_effort=agent.reasoning_effort,
            # The role reports by writing its decision file, and the Stop hook refuses a
            # turn that wrote nothing. Codex refuses per-client Claude SDK hooks and
            # writes with its own tools, so it gets no hook.
            hooks=require_tool_call_stop_hook(["Write"]) if agent.backend != "codex" else None,
            extra_mcp_servers=agent.extra_mcp_servers,
            # The workspace is the session cwd, so the run's own materialized skills are
            # the ones the agent finds. The user's skills and plugins stay out: no role
            # is briefed on them, and an unattended session should not reach Slack or mail.
            cwd=self.workspace,
            # A task's worktree commits into the checkout's ``.git``.
            writable_roots=(str(root),) if root and Path(root).is_dir() else (),
            disabled_skills=self._foreign_skills.get(agent.backend, ()),
            plugins=False,
        )
        # Every role is stateless: what it needs from earlier turns is on disk.
        return AgentLayer(
            AgentLayerConfig(
                name=name,
                system_prompt=system_prompt,
                backend=backend,
                session=SessionConfig(mode="stateless"),
            )
        )

    def _layer(self, name: str) -> AgentLayer:
        """Build a role's layer on first use and keep it for the run."""
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

    # ------------------------------------------------------------ orchestration

    def run(self, task: str | Path) -> bool:
        """One start of the run; whether it is onboarded (a dry run: whether preflight passed)."""
        log = get_logger().console
        state = self._state = self._init_state(task, log)
        try:
            # On every start: each prompt and each check carries what preflight resolves.
            if not start_mod.start(self, state, log):
                return False
            if self.dry_run:
                print_message(
                    "[bold cyan]→ --dry-run: preflight passed; no agent turn follows.[/bold cyan]",
                    log,
                )
                return True
            pipeline = Pipeline(self, state, log)
            pipeline.run()
            if pipeline.stopped is None:
                report_mod.final_pass(pipeline)
            report = report_mod.write_report(self, pipeline)
            pipeline.close()
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted: the same command goes on from here; the turn "
                "that was cut restarts[/bold yellow]",
                log,
            )
            self.journal.event("run interrupted")
            raise
        said = report.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
        checks = self.gate_log.render() or "(none run)"
        self.journal.section("run closed", f"- {said}\n- report: `{report}`\n\n```\n{checks}\n```")
        colour = "green" if state.onboarded else "red"
        print_message(f"[bold {colour}]{said}[/bold {colour}]\nreport: {report}", log)
        return bool(state.onboarded)

    # ------------------------------------------------------------------ helpers

    def _head(self) -> str:
        """The commit the checkout is at; ``""`` when git cannot say."""
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
        return words[0] if words else ""

    def _identity(self, commit: str, tree: str | Path | None = None) -> str:
        """What a verdict measured on ``tree`` (the checkout, unsaid) at ``commit`` is about.

        By content (``contract.tree_id``), with the environment the checks run in as the
        run resolved it: the interpreter, the exported variables, the devices, the prefix.
        """
        described = self.environment.describe() if self.environment is not None else ""
        environment = hashlib.sha256(described.encode("utf-8")).hexdigest()[:12]
        repo = str(tree or self.task_data.get("trtllm_repo_path") or "")
        return contract_mod.tree_id(repo, commit, self.task_data, environment, self._git)

    def _commit_policy(self) -> CommitPolicy:
        """The run's commit rules: who commits, the judge's paths, and what labels a commit."""
        target_rel = str(getattr(self._preflight, "target_rel", "") or "")
        return CommitPolicy.from_task(
            self.task_data, workspace=self.workspace, own_paths=own_paths(target_rel)
        )

    def _contract_status(self) -> contract_mod.ContractStatus:
        """Every verdict recorded in the workspace, this process's and earlier ones'."""
        if self._status is None:
            self._status = contract_mod.ContractStatus.load(self.workspace)
        return self._status

    def _init_state(self, task: str | Path, log) -> WorkflowState:
        """Read the spec, as on every start, and the state an earlier start left."""
        source = Path(task)
        if not source.is_file():
            raise FileNotFoundError(f"--task must be a path to an existing YAML file; got {source}")
        notices: list[str] = []
        try:
            self.task_data = load_and_validate_task_yaml(source, notices)
        finally:
            # Said on every path, a refused spec's included.
            for notice in notices:
                print_message(f"[yellow]note: {notice}[/yellow]", log)
                self.journal.event("spec key ignored", notice)
        check_runtime.default_triton_cache(self.task_data, self.workspace)
        # What the agents read is the resolved spec, defaults filled in, never the
        # operator's file: written again whenever that file changed.
        resolved = yaml.safe_dump(self.task_data, sort_keys=False, allow_unicode=True)
        stale = (
            not self.task_path.is_file() or self.task_path.read_text(encoding="utf-8") != resolved
        )
        if source.resolve() != self.task_path.resolve() and stale:
            self.task_path.write_text(resolved, encoding="utf-8")
        if self.state_path.is_file():
            print_message(
                "[bold cyan]→ going on from this workspace's earlier start[/bold cyan]", log
            )
            return load_state(self.state_path)
        state = WorkflowState(task_path=str(self.task_path))
        self._checkpoint(state)
        return state

    def _checkpoint(self, state: WorkflowState) -> None:
        save_state(self.state_path, state)
