# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The staircase-onboard orchestrator.

Out of tree this was 381 lines of prose read by a language model. Here it is
Python, and the split is the whole point of the port: **phase ordering,
queue walking, gate sequencing and disk verification are control flow**, so
they belong in code where they are deterministic, testable and free. What
stays prose is judgement — how to derive a forward, when a contract covers a
value, what a ceiling is — and that lives in the role prompts and the skills.

The ordering is chosen to make the expensive thing happen early:

1. ``preflight``  — Python. Seconds. A broken host costs seconds, not a
   plan phase and a first assembler turn.
2. ``plan``       — capped, and **non-blocking**. Reaching the cap writes
   the reviewer's open items into the plan and starts the build anyway.
3. ``smith``      — concurrent within a dependency batch. Catalog entries
   are independent by construction, so this is where extra compute buys
   wall-clock.
4. ``assemble``   — serial, core first, gates between units.
5. ``deliver``    — Python verification, one auditor turn, one lint pass.

Every stage checkpoints before advancing, so a crash resumes at the phase
that was running rather than at the beginning.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from rich.markup import escape as escape_markup

from agent_flow import (
    AgentLayer,
    AgentLayerConfig,
    BackendConfig,
    SessionConfig,
    require_tool_call_stop_hook,
)
from agent_flow.console import print_message, print_rule
from agent_flow.logger import get_logger
from agent_flow.prompts import dump_prompt_bundle
from agent_flow.runtime import CommandExecutor, TurnExecutor, TurnRequest

from . import envelope as envelope_mod
from . import gate_cli, launch_gate, remediation, step_check
from . import gates as gates_mod
from . import overlap as overlap_mod
from .catalog_index import CatalogIndexHook
from .committer import CommitPolicy, install_committer
from .delivery import DeliveryReport, summarize_for_auditor, verify_delivery, write_premerge_pack
from .embargo import display_entries, merge_hook_maps, source_embargo_enforcement
from .environment import (
    RunEnvironment,
    image_import_warning,
    resolve_environment,
    shares_allocation_with_spool,
)
from .evidence import RunEvidence
from .execution import select_command_executor, select_turn_executor, shell_runner_for
from .journal import Journal
from .preflight import (
    MAX_TMPDIR_LEN,
    PreflightError,
    SpecInconsistencyError,
    format_preflight_report,
    run_preflight,
    substrate_gate,
)
from .progress import (
    ASSEMBLER_DECISIONS,
    AUDITOR_DECISIONS,
    PLAN_DRAFTER_DECISIONS,
    PLAN_REVIEWER_DECISIONS,
    REVIEWER_DECISIONS,
    SMITH_DECISIONS,
    ProgressContext,
    build_progress_tools,
    collect_decision_file,
    current_decision,
    decision_file_instruction,
    init_progress_file,
    latest_decision,
    latest_entry,
    mark_decision_dispatch,
)
from .prompts import PROMPTS_DIRNAME, PromptBundle, build_staircase_onboard_prompts
from .roles import resolve_role_agent
from .runtime import (
    areas_for_demands,
    format_areas,
    format_kernel_sources,
    load_index,
    load_kernel_sources,
    unmatched_demands,
)
from .state import (
    ITEM_DEFERRED,
    ITEM_DONE,
    ITEM_FAILED,
    ITEM_PENDING,
    ITEM_RUNNING,
    STAGE_ASSEMBLE,
    STAGE_DELIVER,
    STAGE_PLAN,
    STAGE_PREFLIGHT,
    STAGE_SMITH,
    STATE_FILENAME,
    WorkflowState,
    is_remediable,
    load_state,
    save_state,
)
from .survey import format_surface, read_checkpoint_surface, run_inventory_probe, write_surface_file
from .task_schema import (
    REPLAY_WITNESS_REQUIRED,
    assembly_units,
    load_and_validate_task_yaml,
    members_for,
)
from .vocabulary import (
    MODE_VERIFY,
    UNIT_CORE,
    QueueItem,
    VocabularyError,
    dependency_batches,
    load_vocabulary,
    needed_by,
)

#: The role that submits a command to a scheduler the orchestrator cannot
#: reach. It is the only one that returns no typed decision, because it makes
#: none — see `prompts/runner.py` and `runtime/commands/delegated.py`.
RUNNER_ROLE = "runner"

# Its presence, not its contents, is what matters: the Claude Agent SDK
# refuses dynamically-configured MCP servers whenever an enterprise config
# exists, so the workflow's in-process decision tools cannot be created.
ENTERPRISE_MCP_CONFIG = Path("/etc/claude-code/managed-mcp.json")

#: How many smiths may run at once by default.
#
# The previous default was ``0``, which reads as "no limit" and *means* "open
# the whole batch at once". On the run this workflow was built against that
# would have been **19 concurrent agents** — the queue held 19 entries and
# every one of them declared no dependencies, so the layering produced a
# single batch. The run that actually happened was hand-batched to 6 and then
# 2, so the unbounded path was never exercised.
#
# 10 is chosen against what actually binds first, which is not memory. On a
# 90-core / 376 GB node with ~220 GB free, a dozen Node.js agent processes is
# not the constraint. The constraints are:
#
#   * the upstream rate limit, which this stack *reports and does not act on*
#     — there is no backoff anywhere in the agent layer, so N concurrent
#     sessions that hit a cap fail together rather than queueing;
#   * the credential refresh, which was measured corrupting a shared copy
#     under concurrency;
#   * shared-filesystem metadata, since every agent greps the same Lustre
#     tree.
#
# So the number wants to be bounded and modest rather than as large as the
# machine allows. 10 sits above the 6 that was proven by hand and well under
# the 19 that has never been tried.
DEFAULT_MAX_PARALLEL_SMITHS = 10

# The speculative features. A unit is acceptance-gated when the features its
# target implements -- its own and every earlier unit's -- include one of
# these, so the gate runs in the unit that adds speculation AND in every unit
# after it, which keeps speculation on.
#
# Speculative decoding is distribution-preserving BY CONSTRUCTION. Rejection
# sampling holds the emitted distribution to the target model's, so a draft
# path that computes the wrong thing produces correct text more slowly: boot
# passes, accuracy passes, and the only observable that moves is speed.
# Per-prompt acceptance against `=off`, a measured floor and the previous
# tree's verdict is the detector.
#
# `core` is never gated: it has no draft path, so the comparison would be
# 1.0 against 1.0 and would pass whatever core did.
ACCEPTANCE_GATED_UNITS: frozenset[str] = frozenset({"mtp", "eagle3", "dflash", "dspark"})

# The other output-identical features, and the witness each needs. Same
# argument as ACCEPTANCE_GATED_UNITS, different observable: boot and accuracy
# both pass whether or not the feature is doing anything, so a unit with no
# third gate is a unit that can be declared green for a feature nobody built.
#
#   host_offload -> device memory after weight load, against a delta predicted
#                   from the checkpoint's own shapes, plus token identity
#   disagg       -> token identity across the prefill/decode hop
#   kv_quant     -> stored bytes per token against a declared figure, plus
#                   token identity with bf16 storage when declared exact
#
# And the two that are NOT output-identical but still need one, because the
# truncation they make is designed to stay inside the accuracy tolerance and a
# unit that makes no truncation stays inside it too:
#
#   decoder_replay / encoder_replay -> per-request row counters the target
#                   prints, against a declared window and workload
#
# Each needs something from the spec that the orchestrator cannot derive: the
# environment for the contrast leg, and for most a declared figure to judge
# against. That lives in `accuracy.witness`, per unit.
WITNESS_GATED_UNITS: dict[str, str] = {
    "host_offload": "residency",
    "disagg": "equivalence",
    "decoder_replay": "replay",
    "encoder_replay": "replay",
    "kv_quant": "kv_quant",
}

# The BLOCKED reasons a resume handles instead of re-dispatching: the two the
# orchestrator can act on by itself, dispatching smiths -- an onboard for a
# vocabulary gap, verifies for a certification gap. Any other reason already
# ended the run once, and handling it again just ends it again.
RESUMABLE_BLOCKED_REASONS: tuple[str, ...] = ("vocabulary_gap", "certification_gap")

# Shown on every infrastructure failure, and pointed at the file rather than
# summarized, because the summary is the part that goes stale. A run stops
# here perhaps once a week; the operator reading that message has no reason to
# know this document exists unless it says so at the moment they need it.
TROUBLESHOOTING_HINT = (
    "This is an infrastructure failure, not a target defect — every entry in "
    f"`{Path(__file__).parent / 'TROUBLESHOOTING.md'}` read as a model problem "
    "while it was happening. Search it for the error text above."
)


#: The preventive counterpart to ``TROUBLESHOOTING.md``, written into the
#: workspace and referenced by every dispatch prompt.
#:
#: The two are not the same document and must not be merged.
#: ``TROUBLESHOOTING.md`` is keyed on the verbatim error text a run already
#: hit; this is keyed on nothing, because it is meant to be read *before*
#: anything fails. On the run that motivated it, six host facts — pytest's
#: exit code, the ``TMPDIR`` bound, the multi-rank launch rules, which UCX
#: line is noise, what ``$?`` means after a pipe, where a venv interpreter
#: resolves — were re-derived by agent after agent and hand-injected into
#: roughly fifteen dispatch briefs.
HOST_FACTS_FILENAME = "host_facts.md"

#: Everything below this line in ``host_facts.md`` belongs to the roles.
#: Preflight runs again on every resume, so the seeded half is rewritten
#: each time and anything a role learned would otherwise be erased by the
#: next restart — which is the same defect this file exists to fix, one
#: level down.
HOST_FACTS_APPEND_MARKER = "<!-- roles append below this line; nothing above it is preserved -->"

#: Findings that landed after ``plan.md`` and the dispatch queue were
#: written, and therefore cannot reach an in-flight dispatch any other way.
#:
#: Invented by hand mid-run as ``SMITH_CORRECTIONS.md``, because the round-2
#: plan review finished after the smith queue had been built. One of its
#: corrections was load-bearing: without it a smith would have certified an
#: FP8 GEMM at ``N=192``, a width no rank ever passes.
CORRECTIONS_FILENAME = "corrections.md"

#: Where the standing header ends and the corrections begin. A file holding
#: only the header is an *empty* corrections file, and an empty one is not
#: referenced in any prompt — a pointer at nothing costs attention and
#: teaches the reader the pointer is usually worthless.
CORRECTIONS_BODY_MARKER = "<!-- corrections below this line -->"


def _compose_required_tools_hooks(required_tools: list[str]) -> dict | None:
    """Stop hooks requiring *every* listed tool, not just one of them.

    ``require_tool_call_stop_hook`` enforces "at least one of these names was
    called". One independent hook per tool therefore yields AND semantics.
    """
    if not required_tools:
        return None
    merged: dict[str, list] = {"Stop": []}
    for name in required_tools:
        merged["Stop"].extend(require_tool_call_stop_hook([name])["Stop"])
    return merged


class StaircaseOnboardWorkflow:
    """Drive one checkpoint to a gated modeling_v2 target."""

    def __init__(
        self,
        workspace: Path,
        *,
        max_plan_rounds: int = 1,
        max_assembler_turns_per_unit: int = 6,
        max_parallel_smiths: int = DEFAULT_MAX_PARALLEL_SMITHS,
        reviewer_mode: str = "light",
        plan: str | Path | None = None,
        vocabulary: str | Path | None = None,
        clean: bool = False,
        dry_run: bool = False,
        skip_substrate_gate: bool = False,
        prompts: PromptBundle | None = None,
        command_runner: Any | None = None,
        shell_runner: Any | None = None,
        agent_transport: Any | None = None,
        overlap_units: bool = False,
    ) -> None:
        self.workspace = Path(workspace)
        self.task_path = self.workspace / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self.progress_path = self.workspace / "progress.yaml"
        self.journal_path = self.workspace / "journal.md"
        self.boot_spec_path = self.workspace / "boot.yaml"
        self.host_facts_path = self.workspace / HOST_FACTS_FILENAME
        self.corrections_path = self.workspace / CORRECTIONS_FILENAME
        self.state_path = self.workspace / STATE_FILENAME

        self.max_plan_rounds = max(0, max_plan_rounds)
        self.max_assembler_turns_per_unit = max(1, max_assembler_turns_per_unit)
        self.max_parallel_smiths = max_parallel_smiths
        self.reviewer_mode = reviewer_mode
        self.preset_plan = plan
        self.preset_vocabulary = vocabulary
        self.dry_run = dry_run
        self.skip_substrate_gate = skip_substrate_gate
        self._prompts_override = prompts

        # Injected so every stage is testable without a GPU, a cluster, or a
        # model. The defaults are the real ones.
        self._command_runner = command_runner
        # An explicit runner always wins; otherwise the right one depends on
        # the resolved environment, which preflight has not produced yet.
        # Left as the plain shell until then so a caller that never reaches
        # preflight still has something that works.
        self._shell_runner_override = shell_runner
        # When set, role turns are delivered through the filesystem to a
        # session that can authenticate, instead of being spawned here.
        # See transport.py: the orchestrator keeps the stage machine, the
        # queue, the decision validation and every gate; only the turn
        # moves.
        self._agent_transport = agent_transport
        #: Opt-in unit overlap (overlap.py): the next unit's assembler starts
        #: in its own worktree once this unit is UNIT_READY.
        self._overlap_units = overlap_units
        self._overlap: Any = None
        #: How a role turn is delivered, resolved once. Both transports
        #: answer the same `TurnExecutor` interface, so nothing downstream
        #: of `_run_turn` branches on which one is in force.
        self._turns: TurnExecutor = select_turn_executor(
            agent_transport, lambda request: self._layer(request.role)
        )
        #: Where this run's commands land. Resolved from the environment in
        #: `_run_preflight`, which is the first moment one exists; until
        #: then nothing has a command to run.
        self._command_executor: CommandExecutor | None = None
        #: The vocabulary validator's message from the last failed round,
        #: carried verbatim into the re-draft prompt.
        self._vocabulary_errors = ""
        #: True when this process resumed into an assembly already in
        #: progress, so a decision written by the previous one is pending.
        self._resumed_into_unit = False
        # Queue items this process has itself dispatched a smith for; see
        # `_run_one_smith` for why the first dispatch of a resumed item asks
        # the decision file before asking an agent.
        self._smiths_dispatched: set[int] = set()
        #: Whether a queue row naming a unit this run does not build has
        #: already been reported. Said once, not on every queue walk.
        self._warned_unknown_units = False
        #: Lazily resolved in `_gate_env`, because the spec may give gates
        #: their own launch substrate via `environment.workloads.gate`.
        self._gate_environment: RunEnvironment | None = None
        self._shell_runner = shell_runner or gates_mod._default_shell_runner

        # Whether a role's verdict travels as a FILE rather than as a call to
        # its typed-decision MCP tool. Two independent reasons, and missing
        # either one silently discards every decision the run makes.
        #
        # An enterprise MCP configuration makes the SDK refuse *every*
        # dynamically-registered server, including the in-process one the
        # typed-decision tools live in ("You cannot dynamically configure MCP
        # servers when an enterprise MCP config is present"). Detected rather
        # than configured: it is a property of the host, the operator did not
        # choose it, and the first symptom is otherwise an opaque
        # `ProcessError: exit code 1` on the very first agent turn.
        #
        # **And an external transport has no session to host those tools in at
        # all**, whatever the host's MCP configuration says. That was missing
        # here, and the failure is worth stating because nothing reports it:
        # `_dispatch` then neither tells the role where to write its decision
        # nor collects one, and `_run_deliver_phase` reads `progress.yaml` and
        # gets **whatever was there before** — the previous run's verdict.
        #
        # Measured on this project: the orchestrator was moved into a
        # container (where `/etc/claude-code/managed-mcp.json` does not exist)
        # while running on the external transport, and the next three audit
        # passes had their verdicts discarded. Each wrote a valid decision
        # file; the run re-closed on a stale `UNCOVERED` from two passes
        # earlier, including over a `CLEAN`. A loop like that can never
        # converge, and it looks exactly like a gate that keeps failing.
        self._decision_files = ENTERPRISE_MCP_CONFIG.exists() or agent_transport is not None

        self.workspace.mkdir(parents=True, exist_ok=True)
        if clean:
            for path in (
                self.state_path,
                self.plan_path,
                self.vocabulary_path,
                self.progress_path,
                self.journal_path,
                # Both are this run's context, not the operator's notes: the
                # facts are re-seeded by the next preflight, and a correction
                # written against a plan that no longer exists is worse than
                # no correction, because it still claims precedence over
                # `vocabulary.yaml`.
                self.host_facts_path,
                self.corrections_path,
            ):
                path.unlink(missing_ok=True)
            shutil.rmtree(self.workspace / "reports", ignore_errors=True)
            # Regenerated by the next preflight, but a run that fails before
            # reaching it would otherwise leave the previous checkpoint's
            # survey sitting in the workspace looking current.
            (self.workspace / "surface.md").unlink(missing_ok=True)

        self.resume = self.state_path.is_file()
        if not self.resume:
            init_progress_file(self.progress_path)

        self.journal = Journal(path=self.journal_path)
        #: The evidence ledger: every gate result enters it through the log's sink.
        self.evidence = RunEvidence(self, clean=clean)
        self.gate_log = gates_mod.GateLog(sink=self.evidence.record)
        #: catalog/index.yaml, regenerated from the per-entry fragments in HEAD.
        self._catalog_index = CatalogIndexHook(self)
        self._progress_ctx = ProgressContext(path=self.progress_path)
        self._tools = build_progress_tools(self._progress_ctx)

        self.task_data: dict[str, Any] = {}
        self.environment: RunEnvironment | None = None
        self.prompts: PromptBundle = prompts or build_staircase_onboard_prompts()
        self._layers: dict[str, AgentLayer] = {}
        # Set by ``_run_preflight``, which every path through ``run`` reaches
        # before anything reads it — including a resume, which re-derives the
        # identity rather than trusting a stale checkpoint.
        self._preflight: Any = None
        # The surface survey. Empty rather than ``None`` so a resume that
        # re-enters at the plan phase still composes a valid prompt: the
        # drafter then reads "no survey was taken", which is true, instead of
        # the orchestrator crashing on an attribute.
        self._surface: Any = None
        self._inventory: Any = None
        self._surface_block: str = (
            "## Surface survey\n\nNot taken for this run — plan on the assumption that "
            "you do not know what exists, and say so in the plan's open items.\n"
        )
        self._runtime_block: str = ""
        self._kernel_sources_block: str = ""
        #: The certification standard every planning and certifying role
        #: carries; written with `certification.md` at preflight.
        self._certification_block: str = ""

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> "StaircaseOnboardWorkflow":
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
        required_tools: list[str],
        session_mode: str = "stateless",
    ) -> AgentLayer:
        # Backend, model and reasoning effort come from `task.yaml`'s `agents`
        # block, over the defaults in `roles.py`.
        agent = resolve_role_agent(self.task_data, name)
        # Under an enterprise MCP configuration the in-process tool server
        # cannot be registered at all, so the role reports by writing a file
        # and the Stop hook requires the write rather than the tool. `Write`
        # is a weaker hook than a named tool — the role uses it for other
        # things too — but it still refuses a turn that wrote nothing, and
        # the enum is validated when the file is read.
        tools = None if self._decision_files else self._tools.get(name)
        # The `Write` fallback is for roles whose decision travels as a file.
        # A role with no decision at all gets no hook either way.
        hooks = _compose_required_tools_hooks(
            (["Write"] if self._decision_files else required_tools) if required_tools else []
        )
        layer_required: tuple[str, ...] = ()
        if agent.backend == "codex":
            # Codex refuses per-client Claude SDK hooks, so the requirement
            # moves to the layer's backend-neutral policy. A decision that
            # travels as a file has no tool to require on Codex, which writes
            # files with its own tools; the file is validated when it is read.
            hooks = None
            layer_required = () if self._decision_files else tuple(required_tools)
        # source_embargo: a tool hook where the backend runs SDK callbacks, an
        # instruction in the system prompt where it cannot (embargo.py).
        embargo_hooks_map, embargo_note = source_embargo_enforcement(
            self.task_data, self.workspace, role=name, backend=agent.backend
        )
        hooks = merge_hook_maps(hooks, embargo_hooks_map)
        system_prompt += embargo_note
        return AgentLayer(
            AgentLayerConfig(
                name=name,
                system_prompt=system_prompt,
                backend=BackendConfig(
                    kind=agent.backend,
                    model=agent.model,
                    reasoning_effort=agent.reasoning_effort,
                    tools=tools,
                    hooks=hooks,
                    extra_mcp_servers=agent.extra_mcp_servers,
                    # The workspace is the session cwd so the run's own
                    # materialized skills are the ones the agent discovers,
                    # rather than whatever happens to be configured in the
                    # repository it is editing.
                    cwd=self.workspace,
                ),
                session=SessionConfig(mode=session_mode),
                required_tools=layer_required,
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
    ) -> str:
        """Run one role turn, wherever this run's transport delivers it.

        There is no transport branch left in this method, and that is the
        point. A host's inability to authenticate a child process changed
        one thing -- where the prompt goes -- and it used to change a line
        here, in the file that also owns resume, gate sequencing and the
        queue walk. Both transports now answer `TurnExecutor.run`, so
        everything the orchestrator decides (which stage runs, which entry
        is next, whether a gate passed) is on this side of it either way,
        and provably so: this method cannot tell them apart.

        ``layer`` is the caller's own session, which a concurrent batch owns
        rather than the executor -- ten smiths are ten agents, not ten turns
        of one. `bound` is a no-op for transports with no session concept.
        """
        agent = resolve_role_agent(self.task_data, role)
        request = TurnRequest(
            role=role,
            what=what,
            prompt=prompt,
            system_prompt=getattr(self.prompts, role, "") if self.prompts else "",
            backend=agent.backend,
            model=agent.model,
            decision_file=(
                str(current_decision(self.workspace, role, iteration, unit=unit))
                if self._decision_files and iteration is not None
                else None
            ),
            iteration=iteration,
        )
        executor = self._turns if layer is None else self._turns.bound(layer)
        return executor.run(request).text

    def _layer(self, name: str) -> AgentLayer:
        """Build a role's layer on first use and keep it for the run.

        Plan roles are **persistent**: they are asked to remember what they
        decided and why, so a re-draft addresses the reviewer's items instead
        of re-deriving everything. Work roles are **stateless** — one unit of
        work per run, each in its own context — except the assembler, which
        is persistent *within* a unit and reset between units, because a
        unit is its unit of work.
        """
        if name not in self._layers:
            persistent = name in ("plan_drafter", "plan_reviewer", "assembler")
            # The runner is the one role with no typed decision: it relays a
            # submission and the verdict comes from files the job wrote. A
            # Stop hook demanding a decision tool it was never given would
            # block its turn forever.
            required = [] if name == RUNNER_ROLE else [f"append_{name}_progress"]
            self._layers[name] = self._make_agent(
                name,
                getattr(self.prompts, name),
                required_tools=required,
                session_mode="persistent" if persistent else "stateless",
            )
        return self._layers[name]

    def _reset_layer(self, name: str) -> None:
        layer = self._layers.pop(name, None)
        if layer is not None:
            layer.__exit__(None, None, None)

    def _materialize_skills(self) -> Path:
        """Copy the packaged skills into this run's workspace.

        Shipping them with the workflow rather than relying on a checkout's
        ``.claude/skills`` means the knowledge is versioned with the code
        that dispatches it, and copying them per run means a run is
        reproducible from its own workspace.
        """
        source = Path(__file__).parent / "skills"
        destination = self.workspace / ".claude" / "skills"
        if not source.is_dir():
            return destination
        destination.mkdir(parents=True, exist_ok=True)
        for skill_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            target = destination / skill_dir.name
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(skill_dir, target)
        return destination

    # ------------------------------------------------------------ orchestration

    def run(self, task: str | Path) -> None:
        log = get_logger().console
        state = self._init_state(task, log)
        if state is None:
            return

        try:
            if state.stage == STAGE_PREFLIGHT:
                if not self._run_preflight(state, log):
                    return
                if self.dry_run:
                    print_message(
                        "[bold cyan]→ --dry-run: preflight passed; stopping before the "
                        "first agent turn.[/bold cyan]",
                        log,
                    )
                    return
                state.stage = STAGE_PLAN
                self._checkpoint(state)
            else:
                # Resuming past preflight still needs the resolved
                # environment, since every prompt and every gate carries it.
                if not self._run_preflight(state, log):
                    return

            if state.stage == STAGE_PLAN:
                if not self._run_plan_phase(state, log):
                    return
                self._run_early_checks("end of the plan phase", log)
                state.stage = STAGE_SMITH
                self._checkpoint(state)

            if state.stage == STAGE_SMITH:
                if not self._run_smith_phase(state, log):
                    return
                state.stage = STAGE_ASSEMBLE
                self._checkpoint(state)

            if state.stage == STAGE_ASSEMBLE:
                if not self._run_assemble_phase(state, log):
                    return
                state.stage = STAGE_DELIVER
                self._checkpoint(state)

            if state.stage == STAGE_DELIVER:
                if not self._run_deliver_phase(state, log):
                    return

            state.done = True
            self._checkpoint(state)
            self._close_run(state, log, verdict="delivered")
        except KeyboardInterrupt:
            print_message(
                "[bold yellow]⚠ interrupted — re-run the same command to continue "
                "from the last checkpoint, or pass --clean to start fresh"
                "[/bold yellow]",
                log,
            )
            self.journal.event("run interrupted", f"stage={state.stage}")
            raise

    # ------------------------------------------------------------------ stages

    def _run_preflight(self, state: WorkflowState, log) -> bool:
        print_rule("[bold magenta]preflight[/bold magenta]", log)
        kwargs: dict[str, Any] = {}
        if self._command_runner is not None:
            kwargs["runner"] = self._command_runner
        try:
            result = run_preflight(self.task_data, unit_kinds=self._unit_gate_kinds, **kwargs)
        except PreflightError as exc:
            # A spec that contradicts itself is fixed in the spec; the
            # infrastructure guide would send the operator the wrong way.
            hint = "" if isinstance(exc, SpecInconsistencyError) else TROUBLESHOOTING_HINT
            print_message(f"[bold red]✘ preflight: {exc}[/bold red]", log)
            if hint:
                print_message(hint, log)
            self.journal.open(slug="(preflight failed)")
            self.journal.section("preflight failed", f"{exc}\n\n{hint}".rstrip())
            return False

        state.slug = result.slug
        state.branch = result.branch
        state.devices = list(result.devices)
        self.environment = result.environment
        self._preflight = result

        self.journal.open(
            slug=result.slug,
            header_lines=[
                f"- checkpoint: `{self.task_data['checkpoint_path']}`",
                f"- target: `{result.target_rel}`",
                f"- units: {', '.join(assembly_units(self.task_data))}",
            ],
        )
        self.journal.section("preflight", f"```\n{format_preflight_report(result)}\n```")
        print_message(
            f"[bold green]✔ preflight[/bold green]\n{format_preflight_report(result)}", log
        )

        # Where this run's commands land, decided once, from the resolved
        # environment. The orchestrator used to carry the mapping itself --
        # a comparison against one environment kind, inline, twice -- so
        # adding a mechanism meant editing the file that owns resume and
        # gate sequencing. It now asks `execution.py` and does not know the
        # answer's name.
        #
        # An explicitly injected runner still wins: a test supplies one, and
        # so does an operator debugging a gate.
        self._command_executor = select_command_executor(
            result.environment,
            # Handed over unconditionally, and used by exactly one executor:
            # the delegated one, whose scheduler is reachable through a tool
            # the agent holds and this process does not. Submission becomes a
            # turn; the verdict still comes from files the job wrote, read
            # here. Which kind that is remains `execution.py`'s business.
            turns=self._turns,
            workspace=self.workspace,
            # The checkout under test: gates run in it and the session refuses
            # a result whose ranks imported the engine from anywhere else.
            tree=self.task_data.get("trtllm_repo_path"),
        )
        if self._shell_runner_override is None:
            self._shell_runner = shell_runner_for(self._command_executor)

        # Two configuration hazards that are legal, cost hours, and are
        # invisible until they bite. Appended to the notes rather than raised:
        # both are about how this run will be SHAPED, not about whether it can
        # start, and a spec that accepts the cost knowingly is allowed to.
        #
        #   - a registry container image is re-imported per allocation --
        #     measured 5-7 min for 26 GB on the predecessor project, and
        #     25-35 min under load on this one, three submissions running.
        #     That cost is what pushes a run onto one shared session.
        #   - the orchestrator living inside the session it dispatches to.
        #     They then die together, and it needs no GPU. Three times in one
        #     run on the predecessor, twice here.
        for warning in (
            image_import_warning(result.environment),
            shares_allocation_with_spool(result.environment),
        ):
            if warning:
                result.notes.append(warning)
                print_message(f"[bold yellow]⚠ {warning}[/bold yellow]", log)

        # Prove the host can generate a token under this exact environment
        # before any agent exists. Skipped on a dry run, whose whole purpose
        # is to answer "would this spec resolve" cheaply.
        if not self.dry_run and not self.skip_substrate_gate:
            try:
                substrate_gate(
                    self.task_data,
                    result.environment,
                    shell_runner=self._shell_runner,
                    notes=result.notes,
                )
            except PreflightError as exc:
                # Reported like any other precondition failure: the run ends
                # here, before an agent exists, with the command and output
                # the operator needs to reproduce it.
                print_message(f"[bold red]✘ {exc}[/bold red]", log)
                print_message(TROUBLESHOOTING_HINT, log)
                self.journal.section("substrate gate failed", f"{exc}\n\n{TROUBLESHOOTING_HINT}")
                return False

        self._take_survey(result, log)
        self._write_certification(result)

        # Prompts can only be built now: every role's prompt carries the
        # resolved launch prefix, which did not exist until the device claim
        # was made.
        # Concurrent smiths share one worktree and one `.git/index`. The
        # rule "stage by explicit path" was never enough on its own — it was
        # followed and still produced two cross-attributed commits — so the
        # rule is now a script that takes a lock and refuses a commit whose
        # contents are not exactly what was asked for.
        commit_script = install_committer(
            self.workspace,
            self.task_data["trtllm_repo_path"],
            policy=CommitPolicy.from_task(self.task_data, workspace=self.workspace),
        )

        if self._prompts_override is None:
            self.prompts = build_staircase_onboard_prompts(
                environment=result.environment,
                reviewer_mode=self.reviewer_mode,
                source_embargo=display_entries(self.task_data),
                scope=str(self.task_data.get("scope") or "full"),
                decision_files=self._decision_files,
                commit_script=str(commit_script),
            )
        # What every role is told, verbatim and per launch: the composition
        # depends on the resolved environment and the spec, so it otherwise
        # exists only in this process's memory.
        dump_prompt_bundle(self.prompts, self.workspace / PROMPTS_DIRNAME)
        # Written here, from what preflight has just resolved, so that the
        # first agent turn of the run already has them. Both files are
        # referenced by `_context_block`, which every dispatch carries.
        self._write_host_facts(result)
        self._ensure_corrections_file()
        self._materialize_skills()
        self._checkpoint(state)
        return True

    def _write_host_facts(self, result) -> None:
        """Seed ``host_facts.md`` from what this run already knows.

        Seeded rather than hardcoded wherever the run can answer for itself:
        the ``TMPDIR`` bound and its current value come from
        :data:`preflight.MAX_TMPDIR_LEN` and the resolved
        :class:`environment.RunEnvironment`, the multi-rank paragraph is
        emitted only when :attr:`environment.RunEnvironment.needs_llmapi_launch`
        says this run has ranks to launch, and preflight's own notes are
        carried verbatim. The rest are defaults, and a seeded default is
        better than nothing — each of them was learned the expensive way by
        an agent on this host.

        Rewritten on every preflight, which a resume also runs, so anything
        a role appended is preserved below
        :data:`HOST_FACTS_APPEND_MARKER` and everything above it is
        regenerated.
        """
        env = result.environment
        tmpdir = str((env.env or {}).get("TMPDIR", "") or "")
        lines = [
            "# Host facts",
            "",
            "What this host does that a generic Linux box does not. **Read this before "
            "you debug an environment failure** — every fact here was learned the "
            "expensive way by an earlier dispatch on this machine.",
            "",
            f"This is the *preventive* half. `{Path(__file__).parent / 'TROUBLESHOOTING.md'}` "
            "is the other half and is keyed on the verbatim error text of a failure that "
            "already happened. Keep them separate: one is read before anything breaks, "
            "the other is grepped afterwards.",
            "",
            "## Facts",
            "",
            "- **`pytest` exits 4 on this host even when every test passed.** The venv "
            "lives inside the repository root, so a session-final `sys.path` audit "
            "fires. Export `TRTLLM_SKIP_SYS_PATH_CHECK=1`, and adjudicate on the "
            "summary line (`N passed`), never on the exit code.",
            f"- **`TMPDIR` must be at most {MAX_TMPDIR_LEN} characters.** The LLM API's "
            "RPC binds a unix socket under it, the path is capped at 107 bytes and the "
            "socket basename takes 45. This run's `TMPDIR` is "
            + (f"`{tmpdir}` ({len(tmpdir)} characters)." if tmpdir else "not set in the spec."),
            "- **`UCX ERROR uct_iface_open(tcp/rdma_vf_rail0)` is noise.** UCX falls back "
            "and the run exits 0. Do not chase it; it is not why your job failed.",
            "- **`RC=$?` after a pipe is the *pipe's* exit code, not the command's.** "
            "`cmd | tee log; RC=$?` reports `tee`. Use `PIPESTATUS[0]`, or drop the pipe.",
            "- **A venv interpreter created with `--system-site-packages` resolves only "
            "inside the container it was created in.** Running it from the host finds an "
            "interpreter and none of the packages, which reads as a broken install.",
        ]
        if env.needs_llmapi_launch:
            lines.append(
                f"- **A {env.ntasks}-rank launch needs the `SLURM_*` scrub and "
                f"`--oversubscribe`.** Without them every rank reads `mpi_rank=0`, every "
                f"rank becomes a leader, and the job hangs forever with no error and no "
                f"traceback. Both are already in the launch prefix you were handed — use "
                f"it verbatim rather than writing your own `mpirun` line."
            )
        if env.container_image:
            lines.append(
                f"- **The container is `{env.container_image}`.** An interpreter path in "
                f"this run's environment block is a path *inside* it."
            )
        if result.notes:
            lines.extend(
                ["", "## What preflight measured on this host", ""]
                + [f"- {note}" for note in result.notes]
            )
        lines.extend(
            [
                "",
                "## Learned during this run",
                "",
                "This section is yours. If you spend more than a few minutes discovering "
                "something about this host that the section above would have saved you, "
                "append it here as one bullet: the symptom first, then what to do. The "
                "orchestrator never rewrites anything below the marker, so it survives "
                "every restart and reaches every later dispatch.",
                "",
                HOST_FACTS_APPEND_MARKER,
                "",
            ]
        )
        head = "\n".join(lines)

        appended = ""
        if self.host_facts_path.is_file():
            existing = self.host_facts_path.read_text(encoding="utf-8")
            if HOST_FACTS_APPEND_MARKER in existing:
                appended = existing.split(HOST_FACTS_APPEND_MARKER, 1)[1].lstrip("\n")
        self.host_facts_path.write_text(head + appended, encoding="utf-8")

    def _ensure_corrections_file(self) -> None:
        """Create the corrections file's standing header if it is missing.

        The file exists from the start of the run so that anyone — a role, a
        later phase, the operator watching a review land — has a known place
        to put a correction. It is empty until someone does, and
        `_corrections_block` keeps an empty one out of every prompt.
        """
        if self.corrections_path.is_file():
            return
        self.corrections_path.write_text(
            "# Corrections\n\n"
            "Findings that landed **after** `plan.md` and `vocabulary.yaml` were "
            "written, and which therefore cannot reach a dispatch any other way. The "
            "orchestrator appends here when a review lands late; the operator may "
            "append by hand at any time, and every dispatch re-reads this file, so a "
            "correction written now reaches the next turn.\n\n"
            "**Precedence: where this file and `vocabulary.yaml` disagree, this file "
            "wins.** It is newer by construction.\n\n"
            "Each entry: what was believed, what is true instead, and what to do.\n\n"
            f"{CORRECTIONS_BODY_MARKER}\n",
            encoding="utf-8",
        )

    def _corrections_body(self) -> str:
        """Whatever has been written below the standing header. May be empty."""
        if not self.corrections_path.is_file():
            return ""
        text = self.corrections_path.read_text(encoding="utf-8")
        if CORRECTIONS_BODY_MARKER not in text:
            return text.strip()
        return text.split(CORRECTIONS_BODY_MARKER, 1)[1].strip()

    def _record_correction(self, source: str, body: str) -> None:
        """Append one correction, so it reaches dispatches already queued."""
        body = body.strip()
        if not body:
            return
        self._ensure_corrections_file()
        with self.corrections_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## From {source}\n\n{body}\n")
        self.journal.section(f"correction recorded ({source})", body)

    def _context_block(self) -> str:
        """The two files every dispatch carries by reference.

        By reference rather than inlined: both are re-read at the moment the
        role opens them, so a correction written while a turn is in flight
        still lands, and a host fact appended by dispatch 7 is there for
        dispatch 8. Inlining would freeze both at prompt-composition time,
        which is exactly the failure — `task.yaml` read once — that this run
        paid six wasted assembler dispatches for.
        """
        parts: list[str] = []
        if self.host_facts_path.is_file():
            parts.append(
                f"## Host facts\n\n"
                f"Read `{self.host_facts_path}` before you debug anything about this "
                f"machine. It carries what this host does differently — `pytest`'s exit "
                f"code, the `TMPDIR` bound, the launch rules, which errors are noise. If "
                f"you learn a new one, append it under `## Learned during this run`."
            )
        if self._corrections_body():
            parts.append(
                f"## Corrections\n\n"
                f"`{self.corrections_path}` holds findings that landed after the plan and "
                f"the dispatch queue were written. Read it. **Where it and "
                f"`vocabulary.yaml` disagree, the corrections file wins.**"
            )
        return "\n\n".join(parts)

    def _take_survey(self, result, log) -> None:
        """Measure what the checkpoint demands and what the build supplies.

        Runs at the end of preflight rather than at the top of the plan phase
        so that its cost is paid while nothing is waiting on it, and so that
        a resume — which re-enters at the stage that was running — does not
        re-pay it.

        **Never fatal.** A plan can be drafted without a survey; it just gets
        drafted on assumption, which is how every plan before this was
        drafted. Losing a run because a probe could not import is a worse
        trade than losing the measurement, so the failure is written into the
        block the drafter reads and the phase continues.
        """
        surface = read_checkpoint_surface(result.checkpoint_config)
        # Taken on a dry run too. It was skipped there while the probe went
        # out as engine work — a rank per device, through the launcher —
        # because a dry run answers "would this spec resolve" without
        # claiming the machine for minutes. `probe_command` now asks for one
        # process and no MPI world, so the measurement costs an import and a
        # directory walk; and a dry run that prints `0/0 entries certified`
        # against a catalog holding 76 is the exact reading this survey
        # exists to prevent, whoever is looking at it.
        inventory = run_inventory_probe(
            self.task_data,
            result.environment,
            shell_runner=self._shell_runner,
        )
        self._surface = surface
        self._inventory = inventory
        self._surface_block = format_surface(surface, inventory)

        # The runtime map, narrowed to what this checkpoint actually demands.
        # Rendered here rather than in the prompt because it is derived from a
        # measurement, and because a map nobody narrowed is a catalogue.
        demands = surface.demand_names
        index = load_index()
        areas = areas_for_demands(demands, index=index)
        self._runtime_block = format_areas(
            areas, unmatched=unmatched_demands(demands, index=index), overview=index.overview
        )
        # Provenance, for the smiths. The survey says which ops exist; this
        # says what each source IS and what certifying one of its kernels
        # requires, which is what decides the shape of a receipt.
        self._kernel_sources_block = format_kernel_sources(load_kernel_sources())
        write_surface_file(self.workspace / "surface.md", surface, inventory)

        headline = (
            f"{len(surface.demands)} structural demand(s), "
            f"{len(surface.unrecognized)} unrecognized key(s); "
            + (
                f"{len(inventory.spendable)}/{len(inventory.catalog)} entries certified on "
                f"{inventory.arch}, {len(inventory.unwrapped_ops)} unwrapped op(s)"
                if inventory.ok
                else f"inventory probe failed: {inventory.error.splitlines()[0]}"
            )
        )
        self.journal.section("surface survey", headline + "\n\nFull survey: `surface.md`")
        # A failed measurement is not a green line. `✔ survey ... 0/0 entries
        # certified` is how a probe that never ran reads identically to a
        # catalog with nothing in it, and an operator has no way to tell them
        # apart. The run still PROCEEDS — a plan drafted on assumption beats
        # no plan — it just proceeds loudly.
        mark, colour = ("✔", "green") if inventory.ok else ("⚠", "yellow")
        print_message(f"[bold {colour}]{mark} survey[/bold {colour}] {headline}", log)
        if surface.unrecognized:
            print_message(f"  unrecognized structural keys: {', '.join(surface.unrecognized)}", log)

    def _unit_gate_kinds(self, unit: str) -> set[str]:
        """The gate kinds `_assemble_one_unit` runs for `unit`; see `envelope.py`."""
        kinds = {envelope_mod.EVIDENCE_BOOT, envelope_mod.EVIDENCE_ACCURACY}
        if unit in WITNESS_GATED_UNITS:
            kinds.add(envelope_mod.EVIDENCE_WITNESS)
        # The same predicate `_assemble_one_unit` gates with: every unit whose
        # target keeps speculation on, not only the unit that adds it.
        if ACCEPTANCE_GATED_UNITS & set(self._features_through(unit)):
            kinds.add(envelope_mod.EVIDENCE_ACCEPTANCE)
        return kinds

    def _write_certification(self, result) -> None:
        """Write `certification.md` and the block the certifying roles carry."""
        plan = getattr(result, "certification", None)
        if plan is None:
            return
        path = self.workspace / "certification.md"
        path.write_text(plan.render(), encoding="utf-8")
        self._certification_block = plan.prompt_block(path)
        self.journal.section("certification plan", f"`{path}`\n\n" + "\n".join(plan.notes()))

    def _run_early_checks(self, stage: str, log) -> None:
        """Run the delivery checks that need no finished target, as warnings only."""
        from .early_checks import run_early_checks

        if self._preflight is None:
            return
        environment = self.task_data.get("environment") or {}
        report = run_early_checks(
            repo_root=self.task_data["trtllm_repo_path"],
            family=self._preflight.family,
            stage=stage,
            vocabulary_path=self.vocabulary_path,
            env=environment.get("env") or {},
            capacity=self._capacity_audit(),
            certification=getattr(self._preflight, "certification", None),
            workspace=self.workspace,
        )
        path = self.workspace / "early_checks.md"
        path.write_text(report.render(), encoding="utf-8")
        self.evidence.record_early_checks(report)
        warnings = report.warnings
        self.journal.section(
            f"early delivery checks ({stage})",
            f"{len(warnings)} warning(s); required CI assets and details in `{path}`\n\n"
            + "\n".join(f"- {w.render()}" for w in warnings),
        )
        for warning in warnings:
            print_message(f"[bold yellow]⚠ {escape_markup(warning.render())}[/bold yellow]", log)

    def _run_envelope_probes(self, unit: str, log) -> bool:
        """Boot `unit` at every envelope point its certification plan derived."""
        plan = getattr(self._preflight, "certification", None)
        for probe in plan.probes_for(unit) if plan is not None else ():
            result = self.gate_log.record(
                envelope_mod.run_envelope_probe(
                    self._gate_env(),
                    task_data=self.task_data,
                    probe=probe,
                    spec=gates_mod.BootSpec.load(self.boot_spec_path),
                    runner=self._shell_runner,
                    features=self._features_through(unit),
                )
            )
            self.journal.event("gate", result.summary())
            if not result.passed:
                print_message(
                    f"[bold yellow]↻ {escape_markup(result.summary())}[/bold yellow]", log
                )
                return False
            print_message(f"[bold green]✔ {escape_markup(result.summary())}[/bold green]", log)
        return True

    def _run_plan_phase(self, state: WorkflowState, log) -> bool:
        """Drafter <-> reviewer, capped, and never blocking the build."""
        if self.preset_plan is not None and self.preset_vocabulary is not None:
            self._materialize(self.preset_plan, self.plan_path)
            self._materialize(self.preset_vocabulary, self.vocabulary_path)
            print_message(
                "[bold cyan]→ using the supplied plan and vocabulary; skipping the "
                "plan phase[/bold cyan]",
                log,
            )
            state.plan_approved = True
            self._checkpoint(state)
            return True

        while True:
            round_no = state.plan_round + 1
            print_rule(f"[bold magenta]plan round {round_no}[/bold magenta]", log)
            self._progress_ctx.current_iteration = round_no

            # A completed draft is worth resuming onto. The phase only
            # checkpoints once the plan is *approved*, so without this a
            # failure anywhere downstream in the same round — a reviewer
            # whose backend is unauthenticated, a crash, an interrupt —
            # throws away a drafter turn that had already succeeded and
            # written both files. Measured at ~24 minutes and one oracle
            # accuracy run, which is far too much to spend twice for a
            # reason that had nothing to do with the drafter.
            already_drafted = (
                latest_decision(self.progress_path, "plan_drafter", PLAN_DRAFTER_DECISIONS)
                == "DRAFT_READY"
                and (latest_entry(self.progress_path, "plan_drafter") or {}).get("iteration")
                == round_no
                and self.plan_path.is_file()
                and self.vocabulary_path.is_file()
            )
            if already_drafted:
                print_message(
                    f"[bold cyan]→ round {round_no} was already drafted; reusing plan.md "
                    f"and vocabulary.yaml[/bold cyan]",
                    log,
                )
            else:
                self._dispatch(
                    "plan_drafter",
                    what=f"round {round_no}",
                    prompt=self._plan_drafter_prompt(round_no),
                )
            decision = latest_decision(self.progress_path, "plan_drafter", PLAN_DRAFTER_DECISIONS)
            if decision == "CEILING":
                entry = latest_entry(self.progress_path, "plan_drafter") or {}
                print_message(
                    "[bold red]✘ vocabulary ceiling — a computation step has no server "
                    "anywhere. The branch holds the state for a later retry."
                    "[/bold red]",
                    log,
                )
                self.journal.section("vocabulary ceiling", str(entry.get("summary", "")))
                self._close_run(state, log, verdict="ceiling")
                return False

            try:
                # Against the spec's units, so a `unit:` tag naming a feature
                # this run does not build is a re-draft now rather than a
                # guess the orchestrator makes at the first failure.
                vocabulary = load_vocabulary(
                    self.vocabulary_path, units=assembly_units(self.task_data)
                )
            except Exception as exc:
                # A malformed vocabulary is a re-draft, not a crash: the
                # drafter is the only thing that can fix it, and it is still
                # in session.
                print_message(f"[bold yellow]↻ vocabulary.yaml unusable: {exc}[/bold yellow]", log)
                # Carry the validator's own words into the re-draft. Without
                # this the next prompt tells the drafter to read progress.yaml
                # "for the PlanReviewer's REJECT" -- and there is no reviewer
                # entry, because the schema rejected the file before the
                # reviewer ran. The drafter would re-emit the same vocabulary
                # and, at the default one-round cap, end the run.
                self._vocabulary_errors = str(exc)
                if state.plan_round >= self.max_plan_rounds:
                    print_message(
                        "[bold red]✘ plan budget exhausted with no usable "
                        "vocabulary.yaml[/bold red]",
                        log,
                    )
                    self._close_run(state, log, verdict="plan-failed")
                    return False
                state.plan_round += 1
                self._checkpoint(state)
                continue

            if vocabulary.is_ceiling:
                print_message("[bold red]✘ vocabulary.yaml records a ceiling[/bold red]", log)
                self._close_run(state, log, verdict="ceiling")
                return False

            try:
                self._dispatch(
                    "plan_reviewer",
                    what=f"round {round_no}",
                    prompt=self._plan_reviewer_prompt(round_no),
                )
            except Exception as exc:
                # **The plan phase does not block the build** — that is the
                # cap's whole premise, and a reviewer that cannot start is
                # just a reviewer with nothing to say. Crashing here would
                # throw away a completed draft over a second model's
                # availability, which is the opposite of the intent.
                #
                # Seen for real: the reviewer runs on a different backend by
                # design (author/reviewer separation), and that backend
                # returned 401 on a host where only one of the two was
                # authenticated.
                print_message(
                    f"[bold yellow]! plan_reviewer could not run ({exc}). The draft "
                    f"stands unreviewed and the build starts; recorded as an open "
                    f"risk.[/bold yellow]",
                    log,
                )
                self.journal.section("plan_reviewer unavailable", str(exc))
                self._append_open_risks()
                state.plan_approved = True
                state.queue = [
                    {**item.as_dict(), "status": ITEM_PENDING} for item in vocabulary.queue
                ]
                self._checkpoint(state)
                return True

            verdict = latest_decision(self.progress_path, "plan_reviewer", PLAN_REVIEWER_DECISIONS)
            if verdict == "APPROVE":
                print_message("[bold green]✔ plan approved — entering the build[/bold green]", log)
                state.plan_approved = True
                state.queue = [
                    {**item.as_dict(), "status": ITEM_PENDING} for item in vocabulary.queue
                ]
                self._checkpoint(state)
                return True

            state.plan_round += 1
            if state.plan_round > self.max_plan_rounds:
                # The cap is a budget, not a gate. Carry the reviewer's
                # unresolved items into plan.md so the assembler inherits
                # them, and start the build.
                self._append_open_risks()
                print_message(
                    f"[bold yellow]→ plan budget spent ({self.max_plan_rounds} round(s)); "
                    f"the reviewer's open items are recorded in plan.md as risks and "
                    f"the build starts now[/bold yellow]",
                    log,
                )
                state.plan_approved = True
                state.queue = [
                    {**item.as_dict(), "status": ITEM_PENDING} for item in vocabulary.queue
                ]
                self._checkpoint(state)
                return True

            print_message("[bold yellow]↻ plan_reviewer REJECT — re-drafting[/bold yellow]", log)
            self._checkpoint(state)

    def _run_smith_phase(self, state: WorkflowState, log) -> bool:
        """Walk the dispatch queue, concurrently within a dependency batch.

        Every row is dispatched here, whichever unit first needs it: a
        feature-only entry costs nothing extra to build alongside core's, and
        most of them land. What a row's ``unit`` changes is only what its
        failure blocks -- see `_walk_queue`.
        """
        if not state.queue:
            print_message("[bold green]✔ no catalog work — the queue is empty[/bold green]", log)
            return True
        if not self._walk_queue(state, log, unit=None):
            return False
        owed = [row for row in state.queue if row.get("status") == ITEM_DEFERRED]
        if owed:
            print_message(
                f"[bold green]✔ every entry core needs has landed[/bold green] — "
                f"{len(owed)} dispatch(es) deferred, each retried once before the first "
                f"assembler turn of its unit: "
                + ", ".join(
                    f"{row['n']}:{row.get('target')} → {row.get('deferred_to')}" for row in owed
                ),
                log,
            )
        else:
            print_message("[bold green]✔ catalog queue complete[/bold green]", log)
        return True

    def _walk_queue(self, state: WorkflowState, log, *, unit: str | None) -> bool:
        """Dispatch the rows due now, one dependency batch at a time. False if the run closed.

        One walk, two callers, and what differs is which rows are due and
        which failures block.

        ``unit=None`` is the **smith phase**. Every row not yet attempted is
        dispatched, but only a row core needs -- by its own ``unit`` tag, or
        as a transitive dependency of a row core needs
        (`vocabulary.needed_by`) -- can close the run. Any other failure is
        recorded ``deferred`` to the first unit that needs it, and a row that
        waits on one is deferred with it, undispatched. Measured on the
        DeepSeek-V4.1 run: 33 dispatches queued, six needed only by later
        feature units (dspark / multimodal / host_offload / kv_quant), and
        any one of those six failing used to close the run `smith-blocked`
        before core -- which needed none of them -- had started.

        ``unit=U`` is the **start of unit U**. Every row deferred to U, or to
        a unit before it, gets one more attempt, and now every failure
        blocks: U cannot be assembled without them. A row that was never
        deferred is not touched here. The gap dispatches
        `_handle_blocked_assembler` makes are its own, and retrying them here
        as well would dispatch them twice.
        """
        units = assembly_units(self.task_data)
        horizon = 0 if unit is None else units.index(unit)
        position = {name: index for index, name in enumerate(units)}
        needed = self._needed_by(state, log)
        rows = {row["n"]: row for row in state.queue}
        items = [QueueItem.from_row(row) for row in state.queue]
        status = {n: row.get("status", ITEM_PENDING) for n, row in rows.items()}

        def blocks(item: QueueItem) -> bool:
            return position[needed[item.n]] <= horizon

        def due(item: QueueItem) -> bool:
            if status[item.n] == ITEM_DONE:
                return False
            if unit is not None:
                return bool(rows[item.n].get("deferred_to")) and blocks(item)
            # A row core needs is dispatched whatever it last did, which is
            # how a resume has always treated a failed row. Any other row is
            # dispatched only while it has not had its attempt.
            return blocks(item) or status[item.n] in (ITEM_PENDING, ITEM_RUNNING)

        def failure(item: QueueItem, exc: BaseException | None = None) -> str:
            decision = self._smith_decision_for(item)
            why = f"smith decided {decision}" if decision else "smith recorded no decision"
            if exc is not None:
                why += f"; the dispatch raised {type(exc).__name__}: {exc}"
            return why

        def defer(item: QueueItem, why: str) -> str:
            to = needed[item.n]
            row = rows[item.n]
            row.update(status=ITEM_DEFERRED, deferred_to=to, deferred_why=why)
            status[item.n] = ITEM_DEFERRED
            return f"{item.n}:{item.target} → {to} ({why})"

        def close(names: str, detail: str) -> bool:
            where = "core" if unit is None else unit
            print_message(
                f"[bold red]✘ smith could not land {names}{detail}. {where} needs these "
                f"entries, so its assembly cannot start.[/bold red]",
                log,
            )
            self.journal.section(f"smith blocked {where}", f"{names}{detail}")
            self._close_run(
                state, log, verdict="smith-blocked" if unit is None else f"smith-blocked:{unit}"
            )
            return False

        for batch_index, batch in enumerate(dependency_batches(items), start=1):
            pending: list[QueueItem] = []
            held: list[str] = []
            for item in batch:
                if not due(item):
                    # A feature-only row that already failed -- in a process
                    # from before deferral existed, which closed the run over
                    # it -- has had its attempt. Record the deferral instead
                    # of paying for a second one now.
                    if unit is None and status[item.n] == ITEM_FAILED:
                        held.append(defer(item, f"{failure(item)}, in an earlier process"))
                    continue
                waiting = [dep for dep in item.depends_on if status.get(dep) != ITEM_DONE]
                if not waiting:
                    pending.append(item)
                elif blocks(item):
                    return close(f"{item.n}:{item.target}", f" — it waits on {waiting}")
                else:
                    held.append(defer(item, f"not dispatched: waits on {waiting}"))
            if held:
                self._checkpoint(state)
                self._report_deferrals(held, log)
            if not pending:
                continue

            # `0` still means "open the whole batch", but it is now an
            # explicit choice rather than the default, so say what it resolved
            # to. A concurrency figure nobody printed is one nobody notices
            # until the rate limit does.
            workers = min(self.max_parallel_smiths, len(pending))
            if self.max_parallel_smiths <= 0:
                workers = len(pending)
            capped = "" if workers == len(pending) else f", {workers} at a time"
            label = "smith batch" if unit is None else f"{unit} · deferred smith batch"
            print_rule(
                f"[bold magenta]{label} {batch_index} — "
                f"{len(pending)} dispatch(es){capped}[/bold magenta]",
                log,
            )
            if self.max_parallel_smiths <= 0 and len(pending) > DEFAULT_MAX_PARALLEL_SMITHS:
                print_message(
                    f"[bold yellow]! max_parallel_smiths is uncapped, so this batch opens "
                    f"{len(pending)} agent sessions at once. The rate limit is reported "
                    f"but not backed off, so a cap that is hit fails the whole batch "
                    f"together. The default is {DEFAULT_MAX_PARALLEL_SMITHS}.[/bold yellow]",
                    log,
                )
            self._progress_ctx.current_iteration = batch_index
            # Composed before the rows go `running`, from the reports the
            # previous attempt left. The smith phase passes none, so its
            # dispatch is exactly what it always was.
            notes: dict[int, str] = {}
            for item in pending:
                notes[item.n] = self._retry_note(rows[item.n], item) if unit else ""
                self._set_item_status(state, item.n, ITEM_RUNNING)
            self._checkpoint(state)
            errors: dict[int, BaseException] = {}
            with ThreadPoolExecutor(
                max_workers=max(1, workers), thread_name_prefix="smith"
            ) as pool:
                futures = {
                    pool.submit(
                        self._run_one_smith, *((item, notes[item.n]) if notes[item.n] else (item,))
                    ): item
                    for item in pending
                }
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        future.result()
                    except BaseException as exc:  # keep the other results
                        errors[item.n] = exc

            enqueued = 0
            failed: list[QueueItem] = []
            deferred: list[str] = []
            for item in pending:
                decision = self._smith_decision_for(item)
                if decision == "MACRO":
                    # MACRO is not a failure — it is the one outcome whose
                    # purpose is to say what the next work is. Scoring it as
                    # a failure ended the first real run while it held a
                    # 434-line decomposition that was exactly the queue
                    # extension it needed. See IMPROVEMENTS #18.
                    before = len(state.queue)
                    added = self._enqueue_constituents(state, item, log)
                    if unit is not None:
                        # Pieces of a row this unit is retrying are owed by
                        # the same retry: without the mark, nothing would
                        # ever dispatch them.
                        for row in state.queue[before:]:
                            row["deferred_to"] = row.get("unit") or unit
                    enqueued += added
                    ok = added > 0
                    if not ok:
                        print_message(
                            f"[bold yellow]! dispatch {item.n} returned MACRO but listed no "
                            f"usable `constituents`, so there is nothing to enqueue. The "
                            f"decomposition, if any, is prose in its summary.[/bold yellow]",
                            log,
                        )
                else:
                    ok = decision in ("LANDED", "FAILED_RECEIPT")
                self._set_item_status(state, item.n, ITEM_DONE if ok else ITEM_FAILED)
                status[item.n] = ITEM_DONE if ok else ITEM_FAILED
                if ok:
                    continue
                if blocks(item):
                    failed.append(item)
                else:
                    deferred.append(defer(item, failure(item, errors.pop(item.n, None))))
            self._checkpoint(state)
            if deferred:
                self._report_deferrals(deferred, log)

            # A retry is ONE more attempt, so a unit's failure is judged
            # before a sibling's MACRO re-walks the queue; the re-walk would
            # otherwise dispatch it a third time. The smith phase keeps its
            # historical order, in which a failed core row rides the re-walk.
            if enqueued and not (failed and unit is not None):
                # The new items are unstarted, so re-derive the batches and
                # walk the queue again rather than falling through to the
                # next dependency layer.
                print_message(
                    f"[bold cyan]→ {enqueued} constituent dispatch(es) enqueued; "
                    f"re-layering the queue[/bold cyan]",
                    log,
                )
                return self._walk_queue(state, log, unit=unit)

            if failed:
                return close(", ".join(f"{i.n}:{i.target}" for i in failed), "")
            if errors:
                first = next(iter(errors.values()))
                raise RuntimeError(
                    f"{len(errors)} smith dispatch(es) raised; first: "
                    f"{type(first).__name__}: {first}"
                ) from first

        return True

    def _report_deferrals(self, lines: list[str], log) -> None:
        """Say, and journal, which dispatches were deferred and to which unit."""
        print_message(
            f"[bold yellow]→ {len(lines)} smith dispatch(es) deferred — needed only by a "
            f"later unit, so they do not block core. Each is retried once before that "
            f"unit's first assembler turn:[/bold yellow]\n  " + "\n  ".join(lines),
            log,
        )
        self.journal.section("smith dispatches deferred", "\n".join(f"- {line}" for line in lines))

    def _needed_by(self, state: WorkflowState, log=None) -> dict[int, str]:
        """Each queue row's earliest-needing unit. See `vocabulary.needed_by`."""
        units = assembly_units(self.task_data)
        items = [QueueItem.from_row(row) for row in state.queue]
        unknown = sorted({item.unit for item in items} - set(units))
        if unknown and not self._warned_unknown_units:
            self._warned_unknown_units = True
            print_message(
                f"[bold yellow]! queue rows name unit(s) {unknown}, which this run does not "
                f"build ({', '.join(units)}). They are treated as core — the blocking "
                f"direction — rather than guessed at.[/bold yellow]",
                log,
            )
            self.journal.event("queue units treated as core", ", ".join(unknown))
        return needed_by(items, units)

    def _land_deferred(self, state: WorkflowState, unit: str, log) -> bool:
        """Retry, once each, the dispatches deferred to ``unit`` before its first assembler turn.

        Returns False when one still fails, having closed the run
        ``smith-blocked:<unit>``. Idempotent: a row that landed is ``done``,
        and a resume re-enters here and finds only what is still owed.
        """
        units = assembly_units(self.task_data)
        needed = self._needed_by(state, log)
        horizon = units.index(unit)
        owed = [
            row
            for row in state.queue
            if row.get("deferred_to")
            and row.get("status") != ITEM_DONE
            and units.index(needed[row["n"]]) <= horizon
        ]
        if not owed:
            return True
        lines = [f"{row['n']}:{row.get('target')} ({row.get('deferred_why', '')})" for row in owed]
        print_message(
            f"[bold cyan]→ {unit}: {len(owed)} deferred smith dispatch(es) are owed before "
            f"its first assembler turn; retrying each once: {', '.join(lines)}[/bold cyan]",
            log,
        )
        self.journal.section(
            f"{unit}: retrying deferred smith dispatches", "\n".join(f"- {line}" for line in lines)
        )
        return self._walk_queue(state, log, unit=unit)

    def _retry_note(self, row: dict, item: QueueItem) -> str:
        """What a deferred dispatch's retry is told about the attempt that failed.

        Empty for a row that was never dispatched -- it waited on a
        dependency that had not landed -- because there is nothing to read.
        """
        report = self.journal.latest_report("smith", f"{item.mode} {item.target}")
        if report is None:
            return ""
        to = row.get("deferred_to") or item.unit
        return (
            f"**This is a retry.** Dispatch {item.n} was attempted in the smith phase and "
            f"did not land ({row.get('deferred_why') or 'no reason recorded'}). Only "
            f"`{to}` needs this entry, so the run deferred it rather than stop core; this "
            f"is its one remaining attempt, made before `{to}`'s first assembler turn. "
            f"The previous attempt's report is `{report}`, and its progress entry is in "
            f"`{self.progress_path}` (`build_stage`, iteration {item.n}). Read both first: "
            f"keep what it established and do not repeat what it ruled out. If the obstacle "
            f"it hit still stands, say exactly what it is — a second failure stops the run "
            f"at `{to}`.\n"
        )

    def _run_assemble_phase(self, state: WorkflowState, log) -> bool:
        """One assembler per unit, serially. Core is green before the first feature."""
        units = assembly_units(self.task_data)
        while state.unit_index < len(units):
            unit = units[state.unit_index]
            if unit in state.units_done:
                state.unit_index += 1
                continue

            print_rule(f"[bold magenta]assemble · {unit}[/bold magenta]", log)
            # Before the first assembler turn, not after it: a turn spent
            # deriving a forward against an entry that is not there yet ends
            # BLOCKED on a vocabulary gap the run already knew about.
            if not self._land_deferred(state, unit, log):
                return False
            overlap_mod.adopt(self, state, unit, log)
            self._progress_ctx.current_iteration = state.unit_index + 1

            if not self._assemble_one_unit(state, unit, log):
                return False

            state.units_done.append(unit)
            state.unit_index += 1
            state.unit_turns = 0
            self._run_early_checks(f"after the {unit} unit", log)
            # A unit is the assembler's unit of work, so its context ends
            # with it. The next unit starts clean, which is also what makes
            # "core is green before the first feature" mean something.
            self._reset_layer("assembler")
            self._checkpoint(state)

        print_message("[bold green]✔ every unit assembled and gated[/bold green]", log)
        return True

    def _enqueue_constituents(self, state: WorkflowState, item: QueueItem, log) -> int:
        """Append a MACRO's constituents to the queue. Returns how many.

        Two things this deliberately does **not** do.

        It does not enqueue a child whose ``unit`` names a feature this run
        is not building, and it does not enqueue one for a *later* unit than
        the one in progress. A macro under `core` routinely decomposes into
        pieces only the speculator needs; pulling those forward would gate
        `core` against vocabulary it does not use and break the rule that
        core is green before the first feature increment. Measured: of one
        macro's eleven constituents, four were MTP-only.

        It also does not re-enqueue a target already in the queue, so a
        resumed run or two macros naming a shared piece cannot dispatch it
        twice.

        A constituent is needed no earlier than the row it decomposes, so it
        inherits that row's ``unit`` unless it names a later one. The parent's
        unit also widens what may be enqueued now: a feature-only row is
        dispatched in the smith phase alongside core's, so its pieces are
        too, and carrying its tag is what makes a failure among them defer
        rather than block core.
        """
        # THIS item's entry, not the latest smith entry. A batch runs ten
        # smiths concurrently, so "the latest" is whichever finished last --
        # and reading its (empty) `constituents` drops the macro's whole
        # decomposition. Measured: two MACROs in one batch, both reported as
        # "listed no usable constituents", both of which had listed them.
        entry = self._smith_entry_for(item) or {}
        children = entry.get("constituents") or []
        if not isinstance(children, list) or not children:
            return 0

        units = assembly_units(self.task_data)
        current = units[state.unit_index] if state.unit_index < len(units) else units[0]
        parent = self._needed_by(state).get(item.n, item.unit)
        parent = parent if parent in units else units[0]
        reach = max(units.index(current), units.index(parent))
        allowed = set(units[: reach + 1])

        existing = {str(row.get("target")) for row in state.queue}
        next_n = max((int(row["n"]) for row in state.queue), default=0) + 1
        added = 0
        deferred: list[str] = []
        for child in children:
            target = str(child.get("target", "")).strip()
            if not target or target in existing:
                continue
            unit = str(child.get("unit") or parent).strip() or parent
            if unit not in allowed:
                deferred.append(f"{target} ({unit})")
                continue
            if units.index(unit) < units.index(parent):
                unit = parent
            state.queue.append(
                {
                    "n": next_n,
                    "mode": "onboard",
                    "target": target,
                    "depends_on": list(child.get("depends_on") or []),
                    "surface": child.get("surface") or child.get("note"),
                    "entry": None,
                    "note": f"constituent of dispatch {item.n} ({item.target})",
                    "tier": int(child.get("tier", 1) or 1),
                    "unit": unit,
                }
            )
            existing.add(target)
            next_n += 1
            added += 1

        if deferred:
            print_message(
                f"[bold cyan]→ {len(deferred)} constituent(s) belong to a later unit and "
                f"were not enqueued now: {', '.join(deferred)}[/bold cyan]",
                log,
            )
            self.journal.section(f"deferred constituents of dispatch {item.n}", "\n".join(deferred))
        return added

    def _features_through(self, unit: str) -> list[str]:
        """Features implemented once ``unit`` is assembled, in order.

        Units are built one at a time and each is gated before the next
        begins, so the target under test after unit *k* implements
        ``features[:k]`` — never the whole list. ``core`` implements none.

        Returning anything wider gates a target against a feature it does
        not have yet, which either fails on the target's own fail-fast
        assert (the good case, and what happened here) or, if that assert is
        missing, passes while measuring something else entirely.
        """
        units = assembly_units(self.task_data)
        if unit not in units:
            return [str(f) for f in self.task_data.get("features", [])]
        return [str(u) for u in units[1 : units.index(unit) + 1]]

    def _assemble_one_unit(self, state: WorkflowState, unit: str, log) -> bool:
        self._progress_ctx.current_unit = unit
        # `unit_turns` bounds a self-heal loop, so it has to count WORK. It
        # used to count entries into this loop, and those are not the same
        # number: `task.yaml` is loaded once in `_init_state`, so every spec
        # fix needs a restart, and every restart re-entered here and spent a
        # turn re-deriving an answer that was already on disk. Measured on
        # the MiMo-V2.6-Flash run: **ten assembler dispatches for four
        # units, six of which built nothing**, and `core` reached turn 5 of 6
        # with only turn 1 having done anything -- one turn short of closing
        # `assembly-exhausted` over a green, twice-reviewed target sitting in
        # `HEAD`.
        #
        # The counter is still incremented at the top, because that is what
        # makes the loop provably terminate. What changed is that the
        # resume path **gives the turn back** when it does not dispatch.
        while state.unit_turns < self.max_assembler_turns_per_unit:
            state.unit_turns += 1
            # The decision file's name is derived from this, and it was last
            # set by the smith phase to a BATCH index. Turn 2 of an assembly
            # was therefore told to write `assembler-1.yaml` -- the path turn
            # 1 had already used -- so a re-draft either overwrote its
            # predecessor's decision or, as happened here, the role noticed
            # the contradiction and wrote the right file anyway.
            self._progress_ctx.current_iteration = state.unit_turns
            self._checkpoint(state)
            # A resume re-enters this loop at the top, so a BLOCKED decision
            # from the process that died is never acted on: the handler only
            # runs immediately after a dispatch. Check for one first, and pay
            # the gap dispatches instead of an assembler turn that would
            # re-derive an answer already on disk.
            dispatch_needed = True
            if state.unit_turns == 1 or self._resumed_into_unit:
                # Captured before it is cleared: the skip below is a
                # RESUME-only behaviour. At turn 1 of a continuous run the
                # latest assembler decision belongs to the *previous* unit,
                # and after a failed gate the loop re-enters with the flag
                # already false -- so tying the skip to this flag is also
                # what makes it fire at most once per process, and therefore
                # what makes the loop terminate.
                resumed = self._resumed_into_unit
                self._resumed_into_unit = False
                prior = latest_decision(self.progress_path, "assembler", ASSEMBLER_DECISIONS)
                prior_reason = str(
                    (latest_entry(self.progress_path, "assembler") or {}).get("blocked_reason", "")
                ).strip()
                # The hole this closes. `BLOCKED` was the only prior decision
                # the resume path acted on; a prior `UNIT_READY` fell
                # straight through to a fresh dispatch, so the work was
                # re-derived and the turn was spent. It is only safe to skip
                # when the work that decision was written about is still the
                # work on disk, which `_unit_work_unchanged` answers from git
                # rather than assuming -- a resume after someone edited the
                # target is a different situation from a clean restart, and
                # an unanswerable question is answered "changed".
                if resumed and prior == "UNIT_READY" and self._unit_work_unchanged(state, unit):
                    state.unit_turns -= 1
                    self._progress_ctx.current_iteration = max(state.unit_turns, 1)
                    self._checkpoint(state)
                    print_message(
                        f"[bold cyan]→ resuming onto a UNIT_READY the previous process "
                        f"already wrote, and the work under `{self._preflight.target_rel}` "
                        f"is unchanged since. Going straight to the gates: **this restart "
                        f"did not spend an assembler turn** — {unit} is still at turn "
                        f"{state.unit_turns} of {self.max_assembler_turns_per_unit}."
                        f"[/bold cyan]",
                        log,
                    )
                    self.journal.event(
                        "resume did not spend a turn",
                        f"unit={unit}; prior decision UNIT_READY; target unchanged; "
                        f"unit_turns={state.unit_turns}",
                    )
                    dispatch_needed = False
                # Only a VOCABULARY gap is worth re-handling on a resume. The
                # other blocked reasons already ended the run once, and
                # handling one again just ends it again -- which is what
                # happened: a `knowledge_gap` block written at a session
                # cut-off re-closed the run within seconds of every resume,
                # before the newly-extended session could dispatch anything.
                #
                # An operator who resumed deliberately has, by resuming, said
                # the blocking condition is meant to be revisited. So the
                # other reasons fall through to a fresh assembler turn, which
                # is free to reach the same conclusion and say so.
                elif prior == "BLOCKED" and prior_reason in RESUMABLE_BLOCKED_REASONS:
                    print_message(
                        "[bold cyan]→ resuming onto a BLOCKED assembler decision from a "
                        "previous process; handling it rather than re-dispatching"
                        "[/bold cyan]",
                        log,
                    )
                    if self._handle_blocked_assembler(state, log):
                        continue
                    return False
                elif prior == "BLOCKED":
                    print_message(
                        f"[bold cyan]→ resuming past a BLOCKED assembler decision "
                        f"({prior_reason or 'no reason recorded'}); re-dispatching, since "
                        f"resuming is itself a statement that it should be revisited"
                        f"[/bold cyan]",
                        log,
                    )

            if dispatch_needed:
                self._dispatch(
                    "assembler",
                    what=f"{unit} (turn {state.unit_turns})",
                    prompt=self._assembler_prompt(unit, state.unit_turns),
                )
                decision = latest_decision(self.progress_path, "assembler", ASSEMBLER_DECISIONS)
                if decision == "BLOCKED":
                    if self._handle_blocked_assembler(state, log):
                        continue
                    return False
                # Recorded here, at the only moment the answer is known to
                # describe what this turn built. A later process compares
                # against it instead of re-dispatching to find out. A turn
                # that ended with neither decision -- a role that wrote no
                # decision file -- records nothing, so a resume onto it
                # re-dispatches.
                if decision == "UNIT_READY":
                    state.unit_marker[unit] = self._target_marker()
                    self._checkpoint(state)
                    if not self._fan_out_certification_gaps(state, "assembler", unit, log):
                        continue
                    overlap_mod.maybe_start(self, state, unit, log)

            # The configurations this unit is gated over, resolved once. A
            # `unified` segment is ONE codebase over a FAMILY of parallel
            # configurations, and gating it at whatever `topology_flags`
            # happened to emit meant pure tensor parallelism — with the
            # expert-parallel and attention-DP branches that only a unified
            # codebase carries executed by no gate, while the target's
            # Coverage claimed all three. For a concrete segment this is a
            # one-element list, so there is one code path and the
            # single-configuration run is the degenerate case of it.
            # Cheapest member first. A unit gates EVERY member and a broken
            # one fails on all of them, so the order decides how long a
            # failure takes to surface. Measured on this package's first
            # unified run: `tep4` boots in ~180s against `tp4`'s 285-438s.
            members = gates_mod.order_members_by_cost(
                members_for(self.task_data, unit, self._gate_env().world_size)
            )

            boots = self._gate_over_members(
                members,
                lambda member: gates_mod.run_boot_gate(
                    self._gate_env(),
                    task_data=self.task_data,
                    runner=self._shell_runner,
                    # The features this target implements *now*, not the ones
                    # the run will finish with. `core` implements none, so
                    # gating it with the speculator's flags asks a base model
                    # to emit drafts it has no path to produce.
                    features=self._features_through(unit),
                    unit=unit,
                    member=member,
                    # `accuracy.boot`, else the workspace boot.yaml; a
                    # multimodal target also sends an image.
                    spec=gates_mod.resolve_boot_spec(
                        self.task_data, unit, self._features_through(unit), self.boot_spec_path
                    ),
                    workdir=self.workspace / "gates" / unit,
                ),
                log,
            )
            if not boots:
                continue
            # The envelope's ceiling, before review: warm-up at the real token
            # budget and capture at every claimed batch are what a fixed-size
            # buffer fails, and no gate tuple reaches them.
            if not self._run_envelope_probes(unit, log):
                continue

            if self.reviewer_mode != "off":
                # The last member's evidence: the reviewer reads one dump and
                # the summary line names which member produced it, so the
                # freshest is the one whose log is still meaningful to quote.
                self._dispatch("reviewer", what=unit, prompt=self._reviewer_prompt(unit, boots[-1]))
                # Certification-only findings go to verify smiths, not back to
                # the assembler: the target is correct, the catalog is short.
                if not self._fan_out_certification_gaps(state, "reviewer", unit, log):
                    continue
                if latest_decision(self.progress_path, "reviewer", REVIEWER_DECISIONS) == "REJECT":
                    print_message("[bold yellow]↻ reviewer REJECT[/bold yellow]", log)
                    continue

            accuracy_results = self._gate_over_members(
                members,
                lambda member: gates_mod.run_accuracy_gate(
                    self._gate_env(),
                    task_data=self.task_data,
                    reference=self._reference_score(unit),
                    runner=self._shell_runner,
                    # Without this the MTP unit is measured under core's
                    # LLM-API YAML — no speculative_config — so the gate
                    # re-measures the base path and reports the number as
                    # the speculator's.
                    unit=unit,
                    member=member,
                ),
                log,
                announce=True,
            )
            if not accuracy_results:
                continue
            # Every member over its own floor can still be every member
            # under the anchor; recorded either way, refused at delivery.
            shift = gates_mod.note_systematic_shift(
                gates_mod.records_for(self.workspace),
                accuracy_results,
                self.task_data.get("accuracy") or {},
                unit,
            )
            if shift is not None:
                self.journal.event("finding", shift.describe())
                print_message(
                    f"[bold yellow]⚠ {escape_markup(shift.describe())}[/bold yellow]", log
                )

            # The third gate, and the only one that can see a miscomputed
            # draft path. Rejection sampling holds the emitted distribution
            # to the target model's, so boot and accuracy both pass while
            # only speed moves. Runs for distribution-preserving features
            # only; core has no draft path to compare.
            if unit in WITNESS_GATED_UNITS:
                # **Every member**, cheapest first, stopping at the first
                # failure. A witness contrasts two legs of one configuration,
                # but the code a leg runs differs by member: the Engram table
                # is sharded by the TP group, attention DP replicates what TEP
                # shards, and a stale-state defect showed on every member a
                # first-member witness never ran at. A residency witness whose
                # delta differs per member declares `expected_mib` per member.
                witnessed = True
                for member in members:
                    witness = self._run_witness(unit, member, log)
                    if witness is None:
                        return False
                    # Escaped: a token witness's verdict quotes generated
                    # text, which may contain what rich reads as markup.
                    if not witness.passed:
                        print_message(
                            f"[bold yellow]↻ {escape_markup(witness.summary())}[/bold yellow]", log
                        )
                        witnessed = False
                        break
                    print_message(
                        f"[bold green]✔ {escape_markup(witness.summary())}[/bold green]", log
                    )
                if not witnessed:
                    continue

            # Decode-time state the overlap scheduler or speculation reads
            # stale shows as a token difference between legs that must agree:
            # at batch 1, greedy, overlap on and off produce the same tokens.
            if step_check.applies(self.task_data.get("accuracy") or {}, unit):
                if not self._gate_over_members(
                    members,
                    lambda member: step_check.run_scheduler_equivalence(
                        self._gate_env(),
                        task_data=self.task_data,
                        runner=self._shell_runner,
                        unit=unit,
                        member=member,
                        features=self._features_through(unit),
                        workdir=self.workspace / "gates" / unit / "scheduler_equivalence",
                    ),
                    log,
                    announce=True,
                ):
                    continue

            # Every unit whose target keeps speculation on, not only the unit
            # that adds it: a later feature can break the draft path too, and
            # the floors task.yaml declares for those units are otherwise dead
            # configuration.
            if ACCEPTANCE_GATED_UNITS & set(self._features_through(unit)):
                if not self._gate_over_members(
                    members,
                    lambda member: gates_mod.run_acceptance_gate(
                        self._gate_env(),
                        task_data=self.task_data,
                        runner=self._shell_runner,
                        unit=unit,
                        # The features the target implements NOW, same rule as
                        # the boot gate: gating with a speculator flag the
                        # target has no path for is not a stricter gate, it is
                        # a different one.
                        features=self._features_through(unit),
                        member=member,
                        workdir=self.workspace / "gates" / unit,
                        records=gates_mod.records_for(self.workspace),
                    ),
                    log,
                    announce=True,
                ):
                    continue

            # Last: the unit's own deployment entry (trtllm-serve, or the
            # declared launcher for disagg) answers a small accuracy smoke
            # over HTTP, which the in-process LLM API of the accuracy gate
            # never exercises.
            if launch_gate.applies(self.task_data.get("accuracy") or {}, unit):
                launch = launch_gate.resolve_launch_config(
                    self.task_data.get("accuracy") or {}, unit
                )
                if not self._gate_over_members(
                    launch_gate.select_members(launch, members),
                    lambda member: launch_gate.run_launch_gate(
                        self._gate_env(),
                        task_data=self.task_data,
                        reference=launch_gate.smoke_reference(
                            self.task_data, unit, self._model_key
                        ),
                        runner=self._shell_runner,
                        unit=unit,
                        member=member,
                        features=self._features_through(unit),
                        workdir=self.workspace / "gates" / unit / "launch",
                        config=launch,
                    ),
                    log,
                    announce=True,
                ):
                    continue
            return True

        print_message(
            f"[bold red]✘ {unit}: {self.max_assembler_turns_per_unit} assembler turns "
            f"without a green unit. Ending the run with the audit trail rather than "
            f"looping — the gate is the release criterion.[/bold red]",
            log,
        )
        self._close_run(state, log, verdict=f"assembly-exhausted:{unit}")
        return False

    def _gate_over_members(
        self,
        members: list[str],
        run,
        log,
        *,
        announce: bool = False,
    ) -> "list[gates_mod.GateResult]":
        """Fire one gate at every member. Returns the verdicts, or `[]` on a failure.

        Every member is recorded separately and none is collapsed: three
        measurements in `progress.yaml` and the journal is the entire point
        of declaring a member set, and one line reading `accuracy PASS` for a
        target that serves three configurations is precisely the unsupported
        claim `members:` exists to stop.

        It stops at the first failure. The unit is going back to the
        assembler either way, and the remaining members would spend an hour
        each measuring the same defect at a different topology.

        The empty list means "a member failed" rather than "no members" —
        `members_for` never returns an empty set, so the two cannot be
        confused.
        """
        results: list[gates_mod.GateResult] = []
        for member in members:
            result = self.gate_log.record(run(member))
            results.append(result)
            self.journal.event("gate", result.summary())
            if not result.passed:
                print_message(
                    f"[bold yellow]↻ {escape_markup(result.summary())}[/bold yellow]", log
                )
                if result.stdout_tail.strip():
                    self.journal.section(
                        f"gate `{result.name}` failed: evidence",
                        f"```\n{result.stdout_tail.rstrip()}\n```",
                    )
                return []
            if announce:
                print_message(f"[bold green]✔ {escape_markup(result.summary())}[/bold green]", log)
        return results

    @staticmethod
    def _witness_block_gap(unit: str, kind: str, block: Any) -> str | None:
        """What `accuracy.witness.<unit>` lacks for its witness, or None.

        The schema rejects every one of these for a spec it validated, and a
        run reads its spec through the schema. This is the second line for a
        spec that changed underneath a resumed run, and it names the key,
        because "witness unconfigured" alone sends the operator to re-read the
        whole block.
        """
        if not isinstance(block, dict):
            return "a block at all"
        if kind == "replay":
            absent = [key for key in REPLAY_WITNESS_REQUIRED[unit] if not block.get(key)]
            return ", ".join(f"`{key}`" for key in absent) or None
        if kind == "kv_quant":
            if not block.get("expected_bytes_per_token"):
                return "`expected_bytes_per_token`"
            # `exact: false` drops the identity leg, and with it the contrast.
            if block.get("exact", True) and not block.get("contrast_env"):
                return "`contrast_env`"
            return None
        if kind == "equivalence":
            # Without its own launcher leg B is leg A under an environment
            # variable nothing reads, and the witness compares a run to itself.
            absent = [key for key in ("contrast_env", "contrast_command") if not block.get(key)]
            return ", ".join(f"`{key}`" for key in absent) or None
        return None if block.get("contrast_env") else "`contrast_env`"

    def _run_witness(self, unit: str, member: str, log) -> "gates_mod.GateResult | None":
        """Run the witness gate `unit`'s feature needs. None means misconfigured.

        `member` is the configuration both legs are served at; the unit
        loop runs the witness at every member, cheapest first.

        The spec has to supply the one thing that cannot be derived: the
        environment that turns the feature OFF. A missing block is a hard
        stop rather than a skip, because skipping it would declare the unit
        green on two gates that are blind to its feature by construction --
        which is the exact outcome the witness exists to prevent.
        """
        from . import witness as witness_mod

        kind = WITNESS_GATED_UNITS[unit]
        spec = (self.task_data.get("accuracy") or {}).get("witness") or {}
        block = spec.get(unit) if isinstance(spec, dict) else None
        missing = self._witness_block_gap(unit, kind, block)
        if missing:
            print_message(
                f"[bold red]✘ the `{unit}` unit needs a `{kind}` witness and "
                f"`accuracy.witness.{unit}` does not set {missing}. Without it the unit "
                f"would be gated only by boot and accuracy, both of which pass "
                f"whether or not the feature is doing anything.[/bold red]",
                log,
            )
            self._close_run(
                load_state(self.state_path), log, verdict=f"witness-unconfigured:{unit}"
            )
            return None

        # The same entry point a standalone run uses (`staircase-onboard gate
        # witness`), over the workload the block declares.
        result = witness_mod.run_declared_witness(
            self._gate_env(),
            kind=kind,
            unit=unit,
            member=member,
            task_data=self.task_data,
            runner=self._shell_runner,
            features=self._features_through(unit),
            workdir=self.workspace / "witness" / unit,
            boot=gates_mod.resolve_boot_spec(self.task_data, unit, None, self.boot_spec_path),
        )
        self.gate_log.record(result)
        self.journal.event("gate", result.summary())
        if not result.passed and result.stdout_tail.strip():
            # A token comparison's tail is both legs around where each
            # differing continuation first differs; a leg that died leaves its
            # last lines. The one-line summary the gate history carries quotes
            # only the first difference, so the rest goes to the journal and
            # the run's log.
            self.journal.section(
                f"witness `{result.name}` failed: evidence",
                f"```\n{result.stdout_tail.rstrip()}\n```",
            )
            print_message(escape_markup(result.stdout_tail.rstrip()), log)
        return result

    def _run_deliver_phase(self, state: WorkflowState, log) -> bool:
        print_rule("[bold magenta]deliver[/bold magenta]", log)
        self._progress_ctx.current_iteration = 1

        findings = gates_mod.blocking_findings(
            gates_mod.records_for(self.workspace), self.task_data.get("accuracy") or {}
        )
        if findings:
            detail = "\n".join(f"- {f.values.get('detail', f.key)}" for f in findings)
            self.journal.section("delivery blocked by gate findings", detail)
            print_message(f"[bold red]✘ delivery blocked:\n{escape_markup(detail)}[/bold red]", log)
            self._close_run(state, log, verdict="delivery-failed:findings")
            return False

        self._catalog_index.sync(log)
        report = self.evidence.record_delivery_checks(
            self._catalog_index.extend(self._delivery_report(state))
        )
        self.journal.section("delivery checks", f"```\n{report.render()}\n```")
        print_message(report.render(), log)
        if not report.passed:
            names = ", ".join(check.name for check in report.failures)
            print_message(f"[bold red]✘ delivery verification failed: {names}[/bold red]", log)
            self._close_run(state, log, verdict="delivery-failed")
            return False

        boot = self.gate_log.last("boot")
        if not self._run_remediation_cycle(state, report, boot, log):
            return False
        # The delivery tail. The evidence first: the remediation cycle may
        # still have committed, so only now is the delivered tree known, and
        # every verdict has to describe it -- a stale one is re-run, and a tree
        # no full regression ran on refuses delivery.
        if not self.evidence.settle_delivery(state, log):
            return False
        # Then what the branch still owes before it is a PR.
        pack, summary = write_premerge_pack(
            self.task_data["trtllm_repo_path"],
            self.workspace / "reports",
            family=self._preflight.family,
            policy=CommitPolicy.from_task(self.task_data, workspace=self.workspace),
        )
        summary.append(self.evidence.annotate_premerge(pack))
        self.journal.section(
            "pre-merge pack", "\n".join(f"- {line}" for line in summary) + f"\n\n`{pack}`"
        )
        print_message(f"→ pre-merge pack: {pack} ({'; '.join(summary)})", log)
        return True

    # ------------------------------------------------------- remediation cycle

    def _run_remediation_cycle(self, state: WorkflowState, report, boot, log) -> bool:
        """Audit, and close what it finds — bounded, deduplicated, monotone.

        The loop this replaces was not in the workflow at all. An `UNCOVERED`
        ended the run, a human certified the entry, and a re-run spent a
        fresh stateless auditor re-scanning everything to find the next gap.
        That is the mechanism behind audit passes 53 through 57 on one
        target, each paying a full re-scan to learn one increment.

        What makes this a cycle rather than a `while` is in
        :mod:`remediation`: findings group by entry surface, carry a stable
        identity, and have to shrink the unresolved set to earn another
        round. Everything expensive stays outside it — a verify smith runs
        the catalog GPU test and writes a receipt; the accuracy gate does
        **not** re-run, because certifying a contract column cannot change a
        number the engine already produced.
        """
        cycle = remediation.RemediationState.from_dict(state.remediation)
        budget = self._remediation_budget()

        while True:
            cycle.round_no += 1
            self._progress_ctx.current_iteration = cycle.round_no
            self._dispatch(
                "auditor",
                what=f"runtime-value dump vs certified columns (round {cycle.round_no})",
                prompt=self._auditor_prompt(report, boot),
            )
            entry = latest_entry(self.progress_path, "auditor") or {}
            if latest_decision(self.progress_path, "auditor", AUDITOR_DECISIONS) == "CLEAN":
                state.audit_clean = True
                state.remediation = cycle.as_dict()
                self._checkpoint(state)
                print_message("[bold green]✔ audit CLEAN[/bold green]", log)
                return True

            rows, dropped = remediation.parse_audit_rows(
                entry, default_arch=str(self.task_data.get("arch", ""))
            )
            if dropped:
                # Never silent: a dropped row is a finding nothing will act
                # on, and the run would otherwise converge on an incomplete
                # set and call it clean.
                print_message(
                    f"[bold yellow]! {dropped} uncovered row(s) named no entry, arch or "
                    f"axis and cannot be dispatched[/bold yellow]",
                    log,
                )
            if not rows:
                # An UNCOVERED with no structured findings is the old
                # behaviour, and the only honest thing to do with it is what
                # the old behaviour did.
                print_message(
                    "[bold red]✘ auditor UNCOVERED but listed no dispatchable rows. "
                    "Certify what its summary names and re-deliver.[/bold red]",
                    log,
                )
                self.journal.section("audit: uncovered", str(entry.get("summary", "")))
                self._close_run(state, log, verdict="audit-uncovered")
                return False

            incoming = remediation.canonicalize(
                rows,
                target_commit=self._target_commit(),
                capture_id=self._capture_id(),
            )
            merge = remediation.merge_into_queue(cycle.queue, incoming, budget=budget)
            decision = remediation.decide_round(
                merge,
                budget=budget,
                round_no=cycle.round_no,
                previous_signature=cycle.signature,
            )
            cycle.queue = list(merge.queue)
            state.remediation = cycle.as_dict()
            self._checkpoint(state)

            if decision.action == "clean":
                state.audit_clean = True
                self._checkpoint(state)
                print_message("[bold green]✔ audit CLEAN[/bold green]", log)
                return True

            if decision.stopping:
                print_message(f"[bold red]✘ {decision.detail}[/bold red]", log)
                self.journal.section(
                    f"remediation stopped: {decision.verdict}",
                    f"{decision.detail}\n\n```\n{cycle.render()}\n```",
                )
                self._close_run(state, log, verdict=decision.verdict)
                return False

            print_rule(
                f"[bold magenta]remediate · round {cycle.round_no} — "
                f"{decision.detail}[/bold magenta]",
                log,
            )
            # The signature is taken BEFORE the round runs. Comparing the
            # next round's unresolved set against it is what proves the
            # cycle is shrinking rather than circling.
            cycle.signature = remediation.unresolved_signature(cycle.queue)
            failed = self._dispatch_verifiers(cycle, decision, log)
            state.remediation = cycle.as_dict()
            self._checkpoint(state)
            if failed:
                names = ", ".join(f"{i.entry}[{i.axis}]" for i in failed)
                print_message(f"[bold red]✘ verification did not land: {names}[/bold red]", log)
                self.journal.section(
                    "remediation: verification failed",
                    f"{names}\n\n```\n{cycle.render()}\n```",
                )
                self._close_run(state, log, verdict=remediation.STOP_VERIFICATION_FAILED)
                return False

            # Re-verify the mechanical half before spending another auditor
            # turn: a receipt that just landed changes what delivery sees.
            report = self._delivery_report(state)
            if not report.passed:
                names = ", ".join(check.name for check in report.failures)
                print_message(
                    f"[bold red]✘ delivery verification failed after remediation: "
                    f"{names}[/bold red]",
                    log,
                )
                self._close_run(state, log, verdict="delivery-failed")
                return False

    def _remediation_budget(self) -> "remediation.RemediationBudget":
        """The cycle's hard limits, overridable per run from the spec."""
        block = self.task_data.get("catalog_remediation") or {}
        default = remediation.RemediationBudget()
        return remediation.RemediationBudget(
            max_rounds=int(block.get("max_rounds", default.max_rounds)),
            max_attempts_per_item=int(
                block.get("max_attempts_per_item", default.max_attempts_per_item)
            ),
            max_new_items=int(block.get("max_new_items", default.max_new_items)),
            max_parallel_verifiers=int(
                block.get("max_parallel_verifiers", default.max_parallel_verifiers)
            ),
        )

    def _dispatch_verifiers(self, cycle: "remediation.RemediationState", decision, log) -> tuple:
        """Run one round's verify smiths concurrently. Returns the failures.

        Each item becomes exactly one smith in ``verify`` mode carrying the
        whole surface — which is the point of grouping. The outcome is read
        from the decision each smith wrote, never from its prose.
        """
        items = list(decision.dispatch)
        running = {item.identity for item in items}
        cycle.queue = [
            replace(q, status=remediation.QUEUE_RUNNING) if q.identity in running else q
            for q in cycle.queue
        ]

        # One smith per surface, carrying the whole surface. The dispatch
        # numbers start well above the build queue's so a decision file from
        # remediation is never mistaken for one the smith phase wrote, and are
        # the item's position in the queue, which only grows: a later round or
        # a certification-gap fan-out never reuses another surface's number,
        # and so never reads another surface's decision.
        position = {queued.group_key: index for index, queued in enumerate(cycle.queue)}
        dispatches = {
            item.identity: QueueItem(
                n=9000 + position.get(item.group_key, index),
                mode=MODE_VERIFY,
                target=item.entry,
                surface=item.brief(),
            )
            for index, item in enumerate(items)
        }

        with ThreadPoolExecutor(
            max_workers=max(1, decision.workers), thread_name_prefix="verifier"
        ) as pool:
            futures = {
                pool.submit(self._run_one_smith, dispatch): identity
                for identity, dispatch in dispatches.items()
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:  # keep the other verifications
                    print_message(
                        f"[bold yellow]! verify dispatch for {futures[future]} raised "
                        f"{type(exc).__name__}: {exc}[/bold yellow]",
                        log,
                    )

        # The verdict comes from the decision each smith wrote, never from
        # what it said. A dispatch that raised simply has no decision, and
        # `apply_outcomes` reads that as blocked rather than as success.
        outcomes = {
            identity: self._smith_decision_for(dispatch)
            for identity, dispatch in dispatches.items()
        }
        updated, failed = remediation.apply_outcomes(cycle.queue, outcomes)
        cycle.queue = list(updated)
        return failed

    def _fan_out_certification_gaps(self, state: WorkflowState, role: str, unit: str, log) -> bool:
        """Verify every `certification_gap` row `role`'s latest decision carries.

        True when there was nothing to verify or every verify landed; False
        when one failed or cannot be retried, which sends the unit back to the
        assembler with a correction naming the surfaces. The rows join the
        remediation queue, so the delivery audit never re-dispatches them.
        """
        entry = latest_entry(self.progress_path, role) or {}
        rows, dropped = remediation.parse_certification_gaps(
            entry.get(remediation.CERTIFICATION_GAP_KEY),
            default_arch=str(self.task_data.get("arch", "")),
        )
        if dropped:
            print_message(
                f"[bold yellow]! {dropped} certification_gap row(s) from the {role} named no "
                f"entry and cannot be dispatched[/bold yellow]",
                log,
            )
        if not rows:
            return True
        budget = self._remediation_budget()
        cycle = remediation.RemediationState.from_dict(state.remediation)
        incoming = remediation.canonicalize(
            rows, target_commit=self._target_commit(), capture_id=f"{role}:{unit}"
        )
        merge = remediation.merge_into_queue(cycle.queue, incoming, budget=budget)
        decision = remediation.decide_gap_dispatch(merge, budget=budget, incoming=incoming)
        cycle.queue = list(merge.queue)
        state.remediation = cycle.as_dict()
        self._checkpoint(state)
        self.journal.section(f"certification gap ({role}, {unit})", decision.detail)
        print_message(
            f"[bold cyan]→ certification gap from the {role}: {decision.detail}[/bold cyan]", log
        )
        failed: tuple = ()
        if decision.action == "dispatch":
            failed = self._dispatch_verifiers(cycle, decision, log)
            state.remediation = cycle.as_dict()
            self._checkpoint(state)
        if decision.action != "stop" and not failed:
            return True
        names = ", ".join(f"{i.entry}[{i.axis}]" for i in failed) or decision.detail
        self._record_correction(
            f"certification gap, {role} on {unit}",
            f"These surfaces did not certify: {names}. The entry fails there or could not be "
            f"verified, so the target must not reach those values with it: use another entry, "
            f"bound the call, or narrow the serving envelope (a plan change).",
        )
        print_message(f"[bold red]✘ certification gap not closed: {names}[/bold red]", log)
        return False

    def _target_marker(self) -> str:
        """A cheap identity for the work under the target path, or ``""``.

        Two questions, one string: which commit last touched the target, and
        whether anything is uncommitted under it. Together those separate the
        two resumes that matter — a clean restart, where both answers are the
        ones the previous process left, and a restart after someone edited
        the target, where at least one has moved.

        The path is derived rather than spelled out: ``trtllm_repo_path``
        joined with :attr:`preflight.PreflightResult.target_rel`, which is the
        same pair :func:`delivery.verify_delivery` is handed.

        ``""`` means **not established**, and every caller reads it as
        "assume changed". There is no honest third answer, and the asymmetry
        is deliberate: a resume that re-dispatches when it need not have
        costs one assembler turn, while a resume that skips a dispatch it
        needed costs the unit.
        """
        rel = str(getattr(self._preflight, "target_rel", "") or "")
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        if not rel or not repo:
            return ""
        commit = self._shell_runner(f"git -C {repo} log -1 --format=%H -- {rel}", None, 60)
        dirt = self._shell_runner(f"git -C {repo} status --porcelain -- {rel}", None, 60)
        if not getattr(commit, "ok", False) or not getattr(dirt, "ok", False):
            return ""
        head = (getattr(commit, "stdout", "") or "").strip().splitlines()
        if not head:
            return ""
        working = (getattr(dirt, "stdout", "") or "").strip()
        return f"{head[0].strip()}:{hashlib.sha1(working.encode('utf-8')).hexdigest()[:12]}"

    def _unit_work_unchanged(self, state: WorkflowState, unit: str) -> bool:
        """Whether the target still holds what this unit's ``UNIT_READY`` meant.

        Keyed by unit, so the previous unit's marker can never answer for
        this one: at turn 1 of `mtp` the latest assembler decision is
        `core`'s `UNIT_READY`, and an unkeyed comparison would skip the
        increment's first dispatch entirely.
        """
        recorded = state.unit_marker.get(unit, "")
        current = self._target_marker()
        return bool(recorded) and bool(current) and recorded == current

    def _target_commit(self) -> str:
        """The repo commit the audited evidence belongs to.

        Part of an uncovered row's identity: the same values under a NEW
        commit are legitimately new work, because a wrapper change can open a
        runtime branch that did not exist before. Unknown is a valid answer
        and is not worth failing a run over -- it only makes the identity
        coarser, never wrong.
        """
        result = self._shell_runner(
            f"git -C {self.task_data['trtllm_repo_path']} rev-parse --short HEAD", None, 60
        )
        if not getattr(result, "ok", False):
            return ""
        lines = (getattr(result, "stdout", "") or "").split()
        return lines[0] if lines else ""

    def _capture_id(self) -> str:
        """Which runtime capture the audit read.

        Two rounds reading the same capture that disagree are a coverage
        problem, not new work — and that distinction is only visible if the
        capture is named.
        """
        names = [
            r.name
            for r in self.gate_log.results
            if self.RUNTIME_VALUE_MARKER in (r.stdout_tail or "")
        ]
        return "+".join(names) if names else "resumed"

    def _delivery_report(self, state: WorkflowState):
        """The mechanical half of delivery. Cheap, and safe to repeat."""
        report = verify_delivery(
            repo_root=self.task_data["trtllm_repo_path"],
            target_rel=self._preflight.target_rel,
            family=self._preflight.family,
            reference_yaml=gates_mod.resolve_per_unit(
                self.task_data["accuracy"],
                "reference_yaml",
                state.units_done[-1] if state.units_done else None,
            ),
            model_key=self._model_key(),
            accuracy_test=f"test_modeling_v2_{self._preflight.family}.py",
            commit_policy=CommitPolicy.from_task(self.task_data, workspace=self.workspace),
        )
        return DeliveryReport(checks=(*report.checks, *self._certification_checks()))

    def _certification_checks(self) -> list:
        """Delivery checks owed to the envelope: matrix evidence and capacity constants."""
        plan = getattr(self._preflight, "certification", None)
        if plan is None:
            return []
        audit = self._capacity_audit()
        capacity = audit.checks() if audit is not None else []
        return [*envelope_mod.check_external_evidence(plan, self.workspace), *capacity]

    # ------------------------------------------------------------------ helpers

    def _grown_units(self, state: WorkflowState) -> list[str]:
        """Units the spec now lists that this run never assembled, in order."""
        return [u for u in assembly_units(self.task_data) if u not in state.units_done]

    def _init_state(self, task: str | Path, log) -> WorkflowState | None:
        source = Path(task)
        if not source.is_file():
            raise FileNotFoundError(f"--task must be a path to an existing YAML file; got {source}")
        self.task_data = load_and_validate_task_yaml(source)

        if self.resume:
            state = load_state(self.state_path)
            self._backfill_units(state, log)
            if state.done and is_remediable(state.verdict):
                # The run closed on something the operator fixes outside it,
                # and re-running its own check is the only way to see the fix.
                # Refusing here would mean `--clean` and a full rebuild --
                # every gate re-measured to re-answer one question.
                print_message(
                    f"[bold cyan]→ re-entering stage '{state.stage}': the run closed "
                    f"'{state.verdict}', which names a repair made outside it[/bold cyan]",
                    log,
                )
                self.journal.event("re-entered a closed run", f"verdict={state.verdict}")
                if state.verdict.startswith("assembler-blocked"):
                    # The turn budget bounds the unit's own self-heal loop;
                    # the repair that re-opens it happened outside that loop.
                    state.unit_turns = 0
                state.done = False
                state.verdict = ""
            elif self._smith_block_is_deferrable(state):
                # `smith-blocked` stays refused in general: re-running it
                # re-dispatches failed turns with nothing changed. This is the
                # one shape where something did change -- every row that
                # failed is now known to be needed only by a later unit -- and
                # re-entering spends nothing on them now: the smith phase
                # records them `deferred` without dispatching, finishes what
                # the close left unstarted, and core assembles.
                failed = ", ".join(
                    f"{row['n']}:{row.get('target')}"
                    for row in state.queue
                    if row.get("status") == ITEM_FAILED
                )
                print_message(
                    f"[bold cyan]→ re-entering stage 'smith': the run closed 'smith-blocked', "
                    f"but every dispatch that failed ({failed}) is needed only by a later "
                    f"unit; they are deferred to it rather than blocking core[/bold cyan]",
                    log,
                )
                self.journal.event("re-entered a smith-blocked run", f"feature-only: {failed}")
                state.done = False
                state.verdict = ""
            elif state.done and state.verdict == "delivered" and self._grown_units(state):
                # The spec gained a feature after the run delivered. That is an
                # increment, not a repair, and it is the ordinary way a target
                # grows: core ships, then MTP, then multimodality, each gated
                # against the last. The alternative was `--clean` -- re-plan,
                # re-smith, re-assemble and re-measure every unit that is
                # already green -- to add one.
                #
                # Narrow on purpose. Only a run that actually *delivered*
                # qualifies: a failed one is also missing units from
                # `units_done`, and re-entering there would re-dispatch paid
                # agent turns with nothing having changed, which is what
                # `REMEDIABLE_VERDICTS` exists to prevent.
                added = self._grown_units(state)
                print_message(
                    f"[bold cyan]→ the spec grew by {', '.join(added)} since this run "
                    f"delivered; re-entering at 'assemble'[/bold cyan]",
                    log,
                )
                self.journal.event("spec grew after delivery", f"new units: {', '.join(added)}")
                state.done = False
                state.verdict = ""
                state.stage = STAGE_ASSEMBLE
                state.audit_clean = False
                state.unit_turns = 0
                # `_run_assemble_phase` skips units already in `units_done`, so
                # pointing at the first new one is enough and re-derives
                # nothing.
                units = assembly_units(self.task_data)
                state.unit_index = units.index(added[0])
            elif state.done:
                print_message(
                    "[bold green]✔ this run already completed; pass --clean to start "
                    "over.[/bold green]",
                    log,
                )
                return None
            print_message(f"[bold cyan]→ resuming at stage '{state.stage}'[/bold cyan]", log)
            # A resume into an assembly means the previous process may have
            # left a decision nothing acted on. See `_assemble_one_unit`.
            self._resumed_into_unit = state.stage == STAGE_ASSEMBLE
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

    def _backfill_units(self, state: WorkflowState, log) -> None:
        """Carry ``unit`` tags from ``vocabulary.yaml`` onto rows checkpointed without one.

        A queue checkpointed before units existed carries no ``unit`` on any
        row, so every row reads as core and blocks exactly as before. Tagging
        the vocabulary and restarting is how an operator applies deferral to
        such a run. Rows are matched on ``n`` *and* ``target``, so a
        vocabulary the plan phase has since rewritten tags nothing it does not
        still describe; a row that already carries a ``unit`` is left alone,
        because the checkpoint, not the file, is the record of the run.
        """
        untagged = [row for row in state.queue if "unit" not in row]
        if not untagged or not self.vocabulary_path.is_file():
            return
        try:
            vocabulary = load_vocabulary(self.vocabulary_path, units=assembly_units(self.task_data))
        except VocabularyError as exc:
            print_message(
                f"[bold yellow]! unit tags in {self.vocabulary_path} were not applied — it "
                f"does not validate: {exc}\nUntagged rows stay core, which blocks as "
                f"before.[/bold yellow]",
                log,
            )
            return
        tags = {
            (item.n, item.target): item.unit for item in vocabulary.queue if item.unit != UNIT_CORE
        }
        applied: list[str] = []
        for row in untagged:
            unit = tags.get((int(row["n"]), str(row.get("target"))))
            if unit:
                row["unit"] = unit
                applied.append(f"{row['n']}:{row.get('target')} → {unit}")
        if not applied:
            return
        self._checkpoint(state)
        print_message(
            f"[bold cyan]→ applied {len(applied)} unit tag(s) from vocabulary.yaml to the "
            f"checkpointed queue: {', '.join(applied)}[/bold cyan]",
            log,
        )
        self.journal.section(
            "unit tags applied from vocabulary.yaml", "\n".join(f"- {line}" for line in applied)
        )

    def _smith_block_is_deferrable(self, state: WorkflowState) -> bool:
        """Whether a run closed `smith-blocked` in the smith phase only over feature-only rows."""
        if not (state.done and state.verdict == "smith-blocked" and state.stage == STAGE_SMITH):
            return False
        failed = [row for row in state.queue if row.get("status") == ITEM_FAILED]
        if not failed:
            return False
        needed = self._needed_by(state)
        return all(needed[row["n"]] != UNIT_CORE for row in failed)

    def _refresh_workspace_spec(self, source: Path, log) -> None:
        """Keep the spec the agents read identical to the one the gates use.

        The run has two copies of ``task.yaml``: the operator's file, which
        ``_init_state`` re-reads and resolves on **every** run, and the
        workspace copy, which every role is told to read as "the run's spec".
        The workspace copy was written once, on the first run, and never
        again — so a resumed run silently gave the two halves different
        specs.

        That is a split brain, not a stale file. The gates would measure the
        operator's current spec while the assembler derived against an older
        one: different `features`, different `runtime` switches, a different
        accuracy anchor. Both sides self-consistent, neither correct, and
        nothing anywhere would say so.

        Found by the assembler, which noticed the environment block it had
        been handed did not match the spec it had been told to read.
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
        self.evidence.refresh(state)

    @staticmethod
    def _materialize(source: str | Path, destination: Path) -> None:
        candidate = Path(source)
        if candidate.is_file():
            shutil.copyfile(candidate, destination)
        else:
            destination.write_text(str(source).rstrip() + "\n", encoding="utf-8")

    def _gate_env(self) -> RunEnvironment:
        """The environment a GATE runs under, which may not be the run's.

        A spec may declare `environment.workloads.gate` to give gates their
        own launch substrate, and the reason is a failure domain rather than a
        preference. A shared session imports a 26 GB container once and costs
        nothing per job afterwards, which is right for a one-minute entry test
        -- the predecessor run dispatched 443 of them. It is wrong for a 10-15
        minute gate: a shared session has ONE failure domain, so a gate that
        leaks a server, kills a process group or collides on PMIx takes every
        other job and the orchestrator with it. That happened three times on
        the predecessor project and twice on this one.

        Resolved once and cached, from the same claimed devices and world size
        preflight resolved the base environment with, so the two cannot
        disagree about the topology. With no override declared this returns an
        environment equal to the run's, and nothing changes.
        """
        assert self.environment is not None, "preflight must run before any gate"
        if self._gate_environment is None:
            self._gate_environment = resolve_environment(
                self.task_data,
                devices=self.environment.devices,
                world_size=self.environment.world_size,
                workload="gate",
            )
        return self._gate_environment

    def _model_key(self, unit: str | None = None) -> str:
        """The HuggingFace repo name the accuracy references are keyed by.

        Delegated to :func:`~.preflight.resolve_model_key` rather than
        restated. Preflight has to ask this same question hours earlier, to
        check the anchor row exists at all, and **an anchor check that
        resolves a different key than the gate does is worse than no check**
        — it passes on a row nothing ever reads. One implementation is the
        only way those two cannot drift.

        Delegating also picks up the per-unit mapping form, which the
        restatement here did not honour: it read `model_key` as a scalar, so
        a spec whose feature increment anchors under another repo name would
        have been stringified into a key matching nothing.
        """
        from .preflight import resolve_model_key

        return resolve_model_key(self.task_data, unit)

    def _reference_score(self, unit: str | None = None) -> float:
        # `unit` is load-bearing: the accuracy references discriminate rows by
        # `quant_algo` / `spec_dec_algo`, so core and the `mtp` increment
        # anchor to DIFFERENT numbers. Resolving without it either picks the
        # undiscriminated (bf16) row for a quantized checkpoint or raises at
        # the end of an assembly instead of at spec-validation time.
        return gates_mod.resolve_reference(
            self.task_data, model_key=self._model_key(unit), unit=unit
        )

    def _set_item_status(self, state: WorkflowState, n: int, status: str) -> None:
        for row in state.queue:
            if row.get("n") == n:
                row["status"] = status

    def _smith_entry_for(self, item: QueueItem) -> dict | None:
        """The progress entry a smith wrote for this queue item.

        A concurrent batch appends several smith entries, so "the latest
        smith entry" is not this item's. Matched on the queue item's own
        `n` where the collector stamped one, and on the entry path or the
        summary text otherwise.
        """
        from .progress import BUILD_STAGE, find_entries

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
        """The decision from the smith entry for this queue item."""
        row = self._smith_entry_for(item)
        if row is None:
            return None
        decision = str(row.get("decision", "")).strip().upper()
        return decision if decision in SMITH_DECISIONS else None

    def _run_one_smith(self, item: QueueItem, note: str = "") -> str:
        """One catalog entry, in its own layer. Called from the batch's pool.

        **The decision file is collected here.** In MCP-tool mode a smith
        reports through `append_smith_progress`, which writes `progress.yaml`
        itself; under an enterprise MCP configuration there are no tools and
        the smith writes a file instead — and nothing read it. Ten smiths
        landed ten entries, wrote ten valid decision files, and
        `progress.yaml`'s `build_stage` stayed empty, so `_smith_decision_for`
        found no decision and the whole batch would have scored FAILED.

        Caught by a smith, from inside its own dispatch, reading the
        orchestrator's source.

        The iteration is the queue item's `n`, not a turn counter: smiths run
        concurrently, so a counter would race, and `n` is the only number
        that identifies which entry a decision is about. Every smith in the
        first batch independently guessed this same convention from the
        system prompt, which is a good sign it is the readable one.

        ``note`` is prepended context for a retry -- see `_retry_note`.

        **A resume adopts an answer the previous process never read.** The
        state is checkpointed at batch boundaries, so a process that dies
        mid-batch leaves every item `running`; the next one re-enters the
        batch and, before this, dispatched every item again. Measured on the
        DeepSeek-V4.1 run: the compute session was OOM-killed with 30 smiths
        in flight, 16 of which had already written a LANDED decision -- a
        restart would have paid for all 30 twice. So the first time this
        process sees an item it asks the decision file first: one newer than
        the item's last dispatch marker is that dispatch's answer, and is
        collected instead of re-asked. A retry (``note``) never adopts -- the
        decision on disk is the failure being retried.
        """
        if self._decision_files and not note and item.n not in self._smiths_dispatched:
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
                return f"adopted {adopted} from a previous dispatch"
        self._smiths_dispatched.add(item.n)
        layer = self._make_agent(
            "smith",
            self.prompts.smith,
            required_tools=["append_smith_progress"],
            session_mode="stateless",
        )
        try:
            prompt = self._smith_prompt(item, note)
            if self._decision_files:
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
            )
            self.journal.record_result(report, result)
            if self._decision_files:
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
            return result
        finally:
            layer.__exit__(None, None, None)

    def _report_instruction(self, role: str) -> str:
        """How this role reports its decision, in the mode actually in force.

        The system prompt already carries the full protocol; this is the
        one-line reminder at the end of a dispatch. It has to agree with
        the system prompt, because a turn told to call a tool that was
        never registered spends itself looking for it.
        """
        if self._decision_files:
            return "Before ending your turn, write your decision file to the path below."
        return f"Before ending your turn, call `append_{role}_progress` once."

    def _read_instruction(self) -> str:
        """How a role reads the prior entries in its own stage."""
        if self._decision_files:
            return f"Read `{self.progress_path}`"
        return "Call `read_latest_progress`"

    def _dispatch(self, role: str, *, what: str, prompt: str) -> str:
        """Run one agent turn, recording the prompt **before** it is sent."""
        iteration = self._progress_ctx.current_iteration
        unit = self._progress_ctx.current_unit
        # Every role that reads the catalog index reads it current.
        self._catalog_index.sync()
        # Every role, one place. The smith path does not come through here --
        # it owns its own layer per dispatch -- so `_smith_prompt` carries
        # the same block explicitly.
        context = self._context_block()
        if context:
            prompt = f"{prompt}\n\n{context}\n"
        if self._decision_files:
            # Reserved BEFORE the turn runs, and never a path an earlier turn
            # used -- see `progress.reserve_decision`.
            target = mark_decision_dispatch(self.workspace, role, iteration, unit=unit)
            prompt = f"{prompt}\n\n{decision_file_instruction(target)}"

        report = self.journal.record_dispatch(role, what, prompt)
        result = self._run_turn(role, what, prompt, iteration=iteration, unit=unit)
        self.journal.record_result(report, result)

        if self._decision_files:
            decision = collect_decision_file(
                self._progress_ctx, self.workspace, role, iteration, unit=unit
            )
            if decision is None:
                # Deliberately not silent and deliberately not fatal here:
                # the caller reads the decision back out of progress.yaml and
                # already knows what to do with its absence.
                print_message(
                    f"[bold yellow]! {role} ended its turn without a valid decision "
                    f"file at {current_decision(self.workspace, role, iteration, unit=unit)}"
                    f"[/bold yellow]",
                    None,
                )
        return result

    def _handle_blocked_assembler(self, state: WorkflowState, log) -> bool:
        """Triage a BLOCKED assembler. Returns True to retry the unit.

        Staircase's triage table, as control flow: a vocabulary gap is
        something the orchestrator can fix by dispatching smith; the other
        two are findings that end the run, because relaxing a gate or
        guessing at a knowledge gap is exactly what must not happen quietly.
        """
        entry = latest_entry(self.progress_path, "assembler") or {}
        reason = str(entry.get("blocked_reason", "")).strip()
        if reason == "certification_gap":
            units = assembly_units(self.task_data)
            unit = units[state.unit_index] if state.unit_index < len(units) else UNIT_CORE
            if not entry.get(remediation.CERTIFICATION_GAP_KEY):
                print_message(
                    "[bold red]✘ assembler blocked on a certification gap without listing "
                    "`certification_gap`, so there is nothing to verify.[/bold red]",
                    log,
                )
                self._close_run(state, log, verdict="certification-gap-unnamed")
                return False
            if self._fan_out_certification_gaps(state, "assembler", unit, log):
                return True
            self._close_run(state, log, verdict=remediation.STOP_VERIFICATION_FAILED)
            return False
        if reason == "vocabulary_gap":
            missing = str(entry.get("missing_entry", "")).strip()
            if not missing:
                print_message(
                    "[bold red]✘ assembler reported a vocabulary gap without naming the "
                    "callable, so there is nothing to dispatch smith with.[/bold red]",
                    log,
                )
                self._close_run(state, log, verdict="vocabulary-gap-unnamed")
                return False
            # An assembly turn routinely finds SEVERAL gaps at once --- they
            # are usually one class, discovered together. `missing_entry` is
            # a single string, so an assembler reporting three of them can
            # only write "a/x; b/y; c/z", and dispatching that verbatim sends
            # one smith after a target that does not exist. Split it, and
            # dispatch them concurrently: gaps found in one turn have no
            # dependency on each other by construction, since the turn saw
            # all of them before it stopped.
            targets = [part.strip() for part in re.split(r"[;,\n]+", missing) if part.strip()]
            plural = "" if len(targets) == 1 else f" ({len(targets)} gaps)"
            print_message(
                f"[bold cyan]→ vocabulary gap{plural}: dispatching smith for "
                f"{', '.join(targets)}[/bold cyan]",
                log,
            )
            units = assembly_units(self.task_data)
            unit = units[state.unit_index] if state.unit_index < len(units) else UNIT_CORE
            items = [
                QueueItem(
                    n=1000 + len(state.queue) + offset, mode="onboard", target=target, unit=unit
                )
                for offset, target in enumerate(targets)
            ]
            for item in items:
                state.queue.append({**item.as_dict(), "status": ITEM_RUNNING})
            self._checkpoint(state)

            workers = max(1, min(self.max_parallel_smiths or len(items), len(items)))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gapsmith") as pool:
                futures = {pool.submit(self._run_one_smith, item): item for item in items}
                for future in as_completed(futures):
                    # The verdict is read from progress.yaml below, not from
                    # the returned prose --- but a dispatch that RAISED must
                    # not be mistaken for one that merely decided nothing.
                    try:
                        future.result()
                    except BaseException as exc:  # keep the other dispatches
                        print_message(
                            f"[bold yellow]! gap dispatch {futures[future].target} raised "
                            f"{type(exc).__name__}: {exc}[/bold yellow]",
                            log,
                        )

            unlanded = []
            for item in items:
                landed = self._smith_decision_for(item) in ("LANDED", "FAILED_RECEIPT")
                self._set_item_status(state, item.n, ITEM_DONE if landed else ITEM_FAILED)
                if not landed:
                    unlanded.append(item.target)
            self._checkpoint(state)
            if not unlanded:
                return True
            print_message(f"[bold red]✘ smith could not land {', '.join(unlanded)}[/bold red]", log)
            self._close_run(state, log, verdict="smith-blocked")
            return False

        print_message(
            f"[bold red]✘ assembler blocked ({reason or 'unspecified'}) — ending the run "
            f"with the audit trail.[/bold red]",
            log,
        )
        self.journal.section(f"assembler blocked: {reason}", str(entry.get("summary", "")))
        self._close_run(state, log, verdict=f"assembler-blocked:{reason}")
        return False

    def _append_open_risks(self) -> None:
        """Carry the reviewer's unresolved items into plan.md and corrections.md.

        This is what makes the plan cap a budget rather than a gate: the
        review's value is not thrown away, it is handed to the reader who
        can act on it.

        Both files, because they reach different readers. `plan.md` is read
        by whoever opens the plan — the assembler, and a human. The
        corrections file is read by **every dispatch**, including smiths
        whose queue was built from the vocabulary this review is rejecting.
        That is the case the mechanism exists for: a reviewer's finding about
        a GEMM width that arrives after the smith carrying that GEMM has
        already been queued.
        """
        entry = latest_entry(self.progress_path, "plan_reviewer") or {}
        summary = str(entry.get("summary", "")).strip()
        if not summary:
            return
        with self.plan_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## Open risks (unresolved at the plan-round budget)\n\n"
                "The PlanReviewer raised these and the plan phase ran out of rounds "
                "before they were resolved. They are not blockers — the build starts "
                "anyway — but they are the most likely places for it to go wrong.\n\n"
                f"{summary}\n"
            )
        self.journal.section("plan budget spent — open risks recorded", summary)
        self._record_correction(
            f"the PlanReviewer's unresolved items at round {self.max_plan_rounds + 1}",
            summary,
        )

    def _close_run(self, state: WorkflowState, log, *, verdict: str) -> None:
        state.done = True
        state.verdict = verdict
        self._checkpoint(state)
        self.journal.close(
            summary="\n".join(
                [
                    f"- verdict: **{verdict}**",
                    f"- stage reached: `{state.stage}`",
                    f"- units complete: {', '.join(state.units_done) or '(none)'}",
                    f"- human stops: **{state.human_stops}**",
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

    # ------------------------------------------------------------------ prompts

    def _plan_drafter_prompt(self, round_no: int) -> str:
        redraft = ""
        # Re-derive on demand as well as recording it when it happens, so a
        # RESUME into round 2 carries the same message a continuous run does.
        # Otherwise the recovery path works only when nothing interrupted the
        # failure, which is the case least likely to need it.
        if round_no > 1 and not self._vocabulary_errors and self.vocabulary_path.is_file():
            try:
                load_vocabulary(self.vocabulary_path, units=assembly_units(self.task_data))
            except Exception as exc:
                self._vocabulary_errors = str(exc)
        if round_no > 1 and self._vocabulary_errors:
            # A SCHEMA rejection, not a review. The reviewer never ran, so
            # pointing at progress.yaml would point at nothing.
            redraft = (
                "This is a **re-draft**, and the reason is mechanical: "
                "`vocabulary.yaml` did not pass the orchestrator's schema check, "
                "so the PlanReviewer never saw it. **Nothing is wrong with your "
                "analysis** — fix the file and keep everything else.\n\n"
                "The validator's own output, verbatim:\n\n```\n"
                f"{self._vocabulary_errors}\n```\n\n"
                "Revise `vocabulary.yaml` in place. Do not re-derive the plan; "
                "`plan.md` stands unless one of these errors contradicts it.\n\n"
            )
        elif round_no > 1:
            redraft = (
                "This is a **re-draft**. " + self._read_instruction() + " to fetch the "
                "PlanReviewer's REJECT and address every item it names. You are in the "
                "same session as your first draft, so do not re-derive what you already "
                "established — revise it.\n\n"
            )
        return (
            f"Workspace: {self.workspace}\n"
            f"Plan round: {round_no} of at most {self.max_plan_rounds + 1}\n\n"
            f"{redraft}"
            f"Read `{self.task_path}` for the run's spec: the checkpoint, the "
            f"architecture, the parallel segment, the features and the accuracy anchor.\n\n"
            f"{self._surface_block}\n"
            f"{self._runtime_block}\n"
            f"{self._certification_block}\n"
            f"Write `{self.plan_path}` and `{self.vocabulary_path}` as your system "
            f"prompt specifies. The orchestrator parses `vocabulary.yaml` and will "
            f"refuse a malformed one, so follow the schema exactly.\n\n"
            + self._report_instruction("plan_drafter")
        )

    def _plan_reviewer_prompt(self, round_no: int) -> str:
        remaining = self.max_plan_rounds + 1 - round_no
        budget = (
            "This is the **final round**: if you REJECT, your items are recorded in "
            "plan.md as open risks and the build starts anyway. Make the summary count."
            if remaining <= 0
            else f"{remaining} further round(s) remain after this one."
        )
        return (
            f"Workspace: {self.workspace}\n"
            f"Plan round: {round_no}\n\n"
            f"{budget}\n\n"
            f"Read `{self.task_path}` (the spec), `{self.plan_path}` (the plan) and "
            f"`{self.vocabulary_path}` (the typed mapping and dispatch queue). Call "
            + self._read_instruction()
            + " for the PlanDrafter's summary.\n\n"
            "This is a paper review: do not build, run or test anything.\n\n"
            + self._report_instruction("plan_reviewer")
            + " "
            "`APPROVE` or `REJECT`."
        )

    def _smith_prompt(self, item: QueueItem, note: str = "") -> str:
        surface = f"The surface to certify: {item.surface}\n" if item.surface else ""
        # Directly under the dispatch line, because it changes what the turn
        # is for: finish what the last attempt started, not begin again.
        retry = f"\n{note}\n" if note else ""
        # Tiers 1-2 wrap something that exists; 3-4 mean this dispatch writes
        # the implementation. Saying which changes what the turn is for.
        tier = f"Sourcing: **{item.tier_brief}**.\n" + (
            "This entry has no implementation yet — you are writing it, not "
            "wrapping one. Correct-and-slow is acceptable; the tuning phase "
            "replaces it. Say in the contract what you rejected at the cheaper "
            "rungs.\n"
            if item.writes_implementation
            else ""
        )
        # Device assignment, because the launch protocol hands EVERY role the
        # same line -- `CUDA_VISIBLE_DEVICES=0  # you own device(s) 0..0` --
        # and then warns that concurrent scripts on one device contend. At the
        # default ten parallel smiths that warning is a certainty, and the
        # symptom is `Executor creation failed due to insufficient GPU memory`
        # on a device with 284 GB, which reads as a target that allocates too
        # much. Measured on this host with four concurrent full-model loads.
        #
        # Round-robin over the node's devices by dispatch number. Not a
        # scheduler: two smiths can still land on one device, but ten cannot
        # land on device 0.
        devices = int((self.task_data.get("environment") or {}).get("gpus_per_node") or 1)
        assignment = ""
        if devices > 1:
            assignment = (
                f"\n**Use `CUDA_VISIBLE_DEVICES={item.n % devices}` for this dispatch**, not the "
                f"`0` in the launch protocol above. This node has {devices} devices and your "
                f"batch is running concurrently; every smith taking device 0 turns a memory "
                f"error into the symptom.\n"
            )
        # A smith is dispatched from `_run_one_smith` rather than from
        # `_dispatch`, so the shared context has to be added here too. It
        # matters most for this role: the corrections file exists because a
        # plan review landed after the smith queue had been built.
        context = self._context_block()
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Architecture: {self.task_data['arch']}\n\n"
            f"Dispatch {item.n}: **{item.mode}** the catalog entry `{item.target}`.\n"
            f"{retry}"
            f"{assignment}"
            f"Load the `{item.skill}` skill before anything else.\n"
            f"{surface}"
            f"{tier}\n"
            f"{self._smith_brief(item)}\n"
            f"{self._kernel_sources_block}\n"
            f"Read `{self.vocabulary_path}` for the computation step this entry serves "
            f"and the values this checkpoint passes it.\n\n"
            f"One run, one entry. Commit by explicit file path. "
            + self._report_instruction("smith")
            + (f"\n\n{context}\n" if context else "")
        )

    def _smith_brief(self, item: QueueItem) -> str:
        """The entry's call sites, captured rows and integer-width probes; see `smith_brief.py`."""
        from .smith_brief import load_queue_row, load_steps, render_smith_brief

        return render_smith_brief(
            item,
            steps=load_steps(self.vocabulary_path),
            plan=getattr(self._preflight, "certification", None),
            queue_row=load_queue_row(self.vocabulary_path, item.n),
        )

    def _assembler_prompt(self, unit: str, turn: int) -> str:
        feature_skills = ""
        if unit != "core":
            from .task_schema import FEATURE_SKILLS

            feature_skills = (
                f" and `{FEATURE_SKILLS.get(unit, 'staircase-modeling-core')}` for the "
                f"`{unit}` feature"
            )
        history = ""
        if turn > 1:
            history = (
                f"This is turn {turn} for this unit. " + self._read_instruction() + " for "
                "what the last turn did, and read the gate output above — you are in "
                "the same session, so build on it rather than starting over.\n\n"
            )
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Unit: **{unit}** (turn {turn} of at most "
            f"{self.max_assembler_turns_per_unit})\n\n"
            f"{history}"
            f"Load `staircase-modeling-core`{feature_skills} before anything else.\n\n"
            f"Read `{self.task_path}` (the spec), `{self.plan_path}` (the plan, "
            f"including any `## Open risks`) and `{self.vocabulary_path}` (the "
            f"computation-step list you derive against).\n\n"
            f"{self._gate_history_block()}"
            f"Write the unit. The orchestrator runs the boot gate as soon as you "
            f"finish, and the accuracy gate once the unit looks right — do not run "
            f"them yourself, and do not reformat: the run lints once, at delivery. "
            f"To reproduce one gate or witness while debugging, run the workflow's own "
            f"entry point, never a reconstruction of its command: "
            f"`{gate_cli.invocation(self.task_path, self.workspace / 'gates', unit)}`.\n\n"
            + self._report_instruction("assembler")
        )

    def _reviewer_prompt(self, unit: str, boot: gates_mod.GateResult) -> str:
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Unit: **{unit}**\n\n"
            f"The boot gate has already run:\n\n```\n{boot.summary()}\n"
            f"{boot.stdout_tail}\n```\n\n"
            + self._read_instruction()
            + " for the assembler's report — its derivation "
            "table, vocabulary audit and dump review are what you check against the "
            "diff.\n\n"
            "Read the diff and decide. Do not re-run the gates.\n\n"
            + self._capacity_block()
            + self._report_instruction("reviewer")
        )

    def _capacity_audit(self):
        """The static capacity audit of the target, or None before preflight."""
        from .capacity_audit import audit_target

        plan = getattr(self._preflight, "certification", None)
        rel = str(getattr(self._preflight, "target_rel", "") or "")
        if plan is None or not rel:
            return None
        return audit_target(Path(self.task_data["trtllm_repo_path"]) / rel, plan.envelope)

    def _capacity_block(self) -> str:
        audit = self._capacity_audit()
        return f"{self._certification_block}\n{audit.render()}\n" if audit is not None else ""

    #: The line every modeling_v2 target's `_emit_runtime_values` writes before
    #: its dump. Named here because the auditor has to find dumps this process
    #: never saw — see `_auditor_dump`.
    RUNTIME_VALUE_MARKER = "=== MODELINGV2 RUNTIME VALUE DUMP"

    def _auditor_dump(self) -> str:
        """Every gate's runtime-value dump, or directions to where they are.

        Not the boot gate's alone, which is what this passed for its whole
        life. Two things were wrong with that and only the first is obvious.

        The boot gate drives two short prompts, so its dump records the
        engine's *warm-up* shapes and almost nothing else; measured on this
        project, eleven of twelve boot dumps carried only 8191/8192 context
        rows. The shapes a model actually serves at appear in the **accuracy**
        gates, and both of this run's auditor findings were visible only there.

        The second is that `gate_log` is in-memory, so a run resumed into
        `deliver` — which is the normal way to reach this stage after a repair
        — has no gate history at all and this returns nothing however it is
        wired. So when there is nothing to inline, say where the dumps live
        rather than silently handing the auditor an empty section: both
        auditors in this project recovered them from the spool logs by hand,
        and that should not have been an act of initiative.
        """
        chunks = [
            f"### from the `{result.name}` gate\n\n```\n{result.stdout_tail.rstrip()}\n```"
            for result in self.gate_log.results
            if self.RUNTIME_VALUE_MARKER in (result.stdout_tail or "")
        ]
        if chunks:
            return "\n\n".join(chunks)

        spool = getattr(self._gate_env(), "spool", None)
        where = f"`{Path(spool) / 'logs'}/*.log`" if spool else "this run's gate logs"
        return (
            "**No dump was captured in this process.** `gate_log` is in-memory, so a run "
            "resumed into `deliver` carries no gate history — that is expected here and is "
            "not a sign the gates did not run.\n\n"
            f"The dumps are on disk. Search {where} for the marker "
            f"`{self.RUNTIME_VALUE_MARKER}` and audit the **union** across every log that "
            "has one. Prefer the accuracy gates' dumps: a boot gate drives two short prompts "
            "and records warm-up shapes, not the shapes the model serves at."
        )

    def _auditor_prompt(self, report, boot: gates_mod.GateResult | None) -> str:
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Architecture: {self.task_data['arch']}\n\n"
            f"{summarize_for_auditor(report, self._auditor_dump())}\n"
            f"## Gates\n\n```\n{self.gate_log.render() or '(none in this process — see above)'}"
            f"\n```\n\n"
            f"{self._certification_block}\n"
            f"Do the one check that is left: every value in the runtime-value dump "
            f"against its catalog entry's certified column, on this architecture.\n\n"
            + self._report_instruction("auditor")
        )

    def _gate_history_block(self) -> str:
        rendered = self.gate_log.render()
        if not rendered:
            return ""
        return f"Gates so far:\n\n```\n{rendered}\n```\n\n"


if __name__ == "__main__":
    from .cli import main

    main()
