"""Per-node ``coder ⇄ reviewer (⇄ qa)`` loop for concurrent DAG execution.

:func:`make_run_node` builds the :data:`~agent_flow.orchestration.RunNode`
callback the scheduler drives for every node. Each call to the returned
``run_node(node, cwd)`` runs *one* node to a terminal outcome, isolated in:

- its own **worktree** — every per-node agent is built with
  ``BackendConfig(cwd=cwd)`` so the model edits code under the node's worktree
  (``cwd``) rather than the shared workspace; and
- its own **sub-workspace** — :func:`create_node_workspace` carves out
  ``workspace/nodes/<slug>/`` with a private ``progress.yaml`` / ``status.md``,
  and the per-node progress/status tools are bound to those private files so
  parallel nodes never clobber each other's coordination state.

The loop mirrors ``AgentTeamWorkflow``'s build phase, but per node and without
plan/replan/feedback/merge (later tasks):

    coder → reviewer → read reviewer decision from the node's progress.yaml
        non-APPROVE                       → next iteration (coder again)
        APPROVE and the type runs no QA   → DONE
        APPROVE and the type runs QA      → qa → read qa decision
            APPROVE → DONE
            else    → next iteration
    budget exhausted without DONE         → FAILED

The node's gate policy — whether it runs QA, whether it is a replan unit — is
injected as a ``policy_for_type`` callable so this module stays workflow-
agnostic (``modeling_bringup`` passes its own ``policy_for_type``). The recorded
outcome carries ``needs_replan=policy.is_replan_unit`` and
``info={"iterations": <n>, "node_id": <id>}``.

The per-turn agent invocation is factored into the module-level
:func:`_invoke_node_agent`; tests replace it with a fake that records scripted
reviewer/qa decisions into the node's private ``progress.yaml`` so the loop's
:func:`~agent_flow.workflows.agent_team.progress.latest_entry` reads them back
without a real backend.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from agent_flow.orchestration import Node, NodeOutcome, NodeState, RunNode

from . import mcpless
from .node_workspace import NodeWorkspace, create_node_workspace, node_dir_slug
from .progress import build_progress_tools, latest_entry, record_progress_entry
from .prompts import PromptBundle
from .status import build_status_tools
from .workflow import _make_agent

__all__ = ["make_run_node", "node_dir_slug"]


class _NodePolicy(Protocol):
    """The subset of a node's gate policy this loop reads.

    Structural (duck-typed) so agent_team never imports any one workflow's
    concrete policy table — ``modeling_bringup`` passes its
    ``NodeTypePolicy``, which satisfies this Protocol.
    """

    runs_qa: bool
    is_replan_unit: bool


# The type-policy resolver injected by the calling workflow, e.g.
# ``modeling_bringup.node_policy.policy_for_type``.
PolicyForType = Callable[[str], _NodePolicy]


def _build_node_agent(
    name: str,
    system_prompt: str,
    tools: list | None,
    *,
    required_tools: list[str] | None,
    cwd: str | Path,
    backend_kind: str,
    model: str,
    session_mode: str = "persistent",
) -> Any:
    """Construct one per-node agent, pinned to that node's worktree.

    The only thing this adds over the linear path's ``_make_agent`` is the
    ``cwd``: the node's worktree, so the model edits code there rather than in
    the shared checkout.
    """
    return _make_agent(
        name,
        system_prompt,
        tools,
        required_tools=required_tools,
        backend_kind=backend_kind,
        model=model,
        session_mode=session_mode,
        cwd=cwd,
    )


async def _invoke_node_agent(
    agent: Any,
    role: str,
    prompt: str,
    iteration: int,
    *,
    nw: NodeWorkspace,
) -> None:
    """Run one agent turn for a node, MCP-tools mode.

    The agent records its progress/status through the MCP tools bound to the
    node's private contexts, so this is a thin wrapper — its whole reason to
    exist is the await: ``run_node`` is async (the scheduler awaits it), so the
    agent must be driven with ``await agent.aforward(...)`` rather than the
    synchronous ``agent(...)`` the linear workflow uses. Calling the blocking
    form here would stall the scheduler's event loop and silently serialize
    every node.
    """
    await agent.aforward(prompt)


async def _invoke_node_agent_mcpless(
    agent: Any,
    role: str,
    prompt: str,
    iteration: int,
    *,
    nw: NodeWorkspace,
) -> None:
    """Run one agent turn for a node under ``--no-mcp-tools``.

    The node counterpart of ``AgentTeamWorkflow._invoke_agent``: prepend the
    recording-protocol preamble plus this node's inlined context, run the turn,
    then read back the handoff file the agent wrote and record it into the
    node's PRIVATE ``progress.yaml``. A missing or invalid handoff triggers one
    corrective retry before raising :class:`mcpless.HandoffError`.

    Every path here is node-scoped — the handoff directory, the progress log
    and the status file — so parallel nodes never read or clobber each other's
    turn state. Human feedback is the one shared read (see
    :attr:`NodeWorkspace.shared_progress_path`).

    Deliberately *not* shared with the linear path's implementation: that path
    ships in ``main`` and exists to serve environments where a dynamically
    configured MCP server is blocked, so it is kept byte-for-byte unchanged
    here. The two can be folded into one driver once this lands.
    """
    nw.turn_dir.mkdir(parents=True, exist_ok=True)
    handoff = mcpless.handoff_path(nw.turn_dir, role)
    context = mcpless.gather_context(
        role,
        progress_path=nw.progress_path,
        status_path=nw.status_path,
        feedback_path=nw.shared_progress_path,
    )
    preamble = mcpless.build_recording_preamble(role, handoff, nw.status_path, context)
    full_prompt = f"{preamble}\n\n{prompt}"

    corrective = (
        "\n\n=== RETRY ===\n"
        "Your previous turn did not leave a valid handoff file. You MUST end "
        f"this turn by writing `{handoff}` with the exact YAML keys described "
        "above (and nothing else). Do this now."
    )

    last_error: Exception | None = None
    for attempt in (0, 1):
        if handoff.exists():
            handoff.unlink()
        await agent.aforward(full_prompt if attempt == 0 else full_prompt + corrective)
        if not handoff.exists():
            last_error = mcpless.HandoffError(f"{role} did not write its handoff file {handoff}")
            continue
        try:
            fields = mcpless.parse_handoff(role, handoff.read_text(encoding="utf-8"))
        except mcpless.HandoffError as exc:
            last_error = exc
            continue
        record_progress_entry(nw.progress_path, role, iteration, fields)
        return

    assert last_error is not None
    raise last_error


async def _run_turn(
    agent: Any,
    role: str,
    prompt: str,
    iteration: int,
    *,
    nw: NodeWorkspace,
    use_in_process_tools: bool,
) -> None:
    """Dispatch one node turn to the invoker for this run's transport mode.

    Both invokers are looked up as module globals at call time, so a test may
    monkeypatch either one.
    """
    if use_in_process_tools:
        await _invoke_node_agent(agent, role, prompt, iteration, nw=nw)
    else:
        await _invoke_node_agent_mcpless(agent, role, prompt, iteration, nw=nw)


def _normalized_decision(progress_path: Path, agent: str) -> str | None:
    """Return the latest ``APPROVE`` / ``REJECT`` from ``agent``, else ``None``.

    Reuses :func:`latest_entry`; anything other than the two canonical verdicts
    (including a missing entry) reads as ``None`` — a non-APPROVE that loops the
    node back to the coder.
    """
    entry = latest_entry(progress_path, agent)
    if entry is None:
        return None
    decision = str(entry.get("decision", "")).strip().upper()
    return decision if decision in ("APPROVE", "REJECT") else None


# --------------------------------------------------------------------------
# Per-turn prompts
#
# Each prompt is built as a TRANSPORT-NEUTRAL body plus a per-mode protocol
# block, mirroring how the role *system* prompts split their content from
# ``prompts.mcp_tools.MCP_TOOLS_EXTENSIONS`` (see ``PromptBundle``'s contract).
# The body says WHAT this turn reads and records; the block says HOW:
#
# - MCP mode          -> ``_NODE_MCP_PROTOCOL[role]``, appended here.
# - ``--no-mcp-tools`` -> ``mcpless.build_recording_preamble``, prepended by
#   :func:`_invoke_node_agent_mcpless`.
#
# Keeping the tool names out of the body is what makes the no-MCP mode honest:
# no role is ever told to call a tool this run did not register.
# --------------------------------------------------------------------------

_NODE_MCP_PROTOCOL: dict[str, str] = {
    "coder": (
        "=== RECORDING PROTOCOL: in-process MCP tools ===\n"
        "Use these tools for the reads and the recording described above:\n"
        "- `read_status` — load this node's rolling status.md scratchpad.\n"
        "- `read_latest_progress` with `iterations: 2` — the Reviewer's and "
        "QA's latest REJECT feedback.\n"
        "- `read_human_feedback` — any direct user guidance.\n"
        "Before completing your turn, call **both** required tools: "
        "`append_coder_progress` (with the `summary`) and `update_status` "
        "(overwriting this node's status.md with the snapshot described above)."
    ),
    "reviewer": (
        "=== RECORDING PROTOCOL: in-process MCP tools ===\n"
        "Use these tools for the reads and the recording described above:\n"
        "- `read_status` — load this node's rolling status.md scratchpad.\n"
        '- `read_latest_progress` with `agent: "coder"` — the Coder\'s latest '
        "summary.\n"
        "- `read_human_feedback` — any direct user guidance.\n"
        "Before completing your turn, call **both** required tools: "
        "`append_reviewer_progress` (with the `summary` and `decision`) and "
        "`update_status` (overwriting status.md as described above)."
    ),
    "qa": (
        "=== RECORDING PROTOCOL: in-process MCP tools ===\n"
        "Use these tools for the reads and the recording described above:\n"
        "- `read_human_feedback` — any direct user guidance.\n"
        "Before completing your turn, call the `append_qa_progress` tool with "
        "the `summary`, `decision`, and `weighted_score` described above."
    ),
}


def _with_protocol(body: str, role: str, use_in_process_tools: bool) -> str:
    """Append the MCP protocol block to ``body`` when this run has MCP tools."""
    if not use_in_process_tools:
        return body
    return f"{body}\n\n{_NODE_MCP_PROTOCOL[role]}"


def _coder_prompt(
    node: Node,
    *,
    workspace: Path,
    cwd: Path,
    iteration: int,
    use_in_process_tools: bool = True,
) -> str:
    body = (
        f"You are the Coder for node `{node.id}` (type: {node.type}).\n"
        f"Node worktree — edit code HERE: {cwd}\n"
        f"Iteration: {iteration}\n\n"
        f"Start by loading this node's rolling `status.md` scratchpad — that "
        f"is your fastest way to pick up where the previous turn left off.\n\n"
        f"The shared specs are READ-ONLY and describe the whole task; use them "
        f"to find the slice that belongs to node `{node.id}`: read "
        f"`{workspace / 'task.yaml'}` for the original task from the user, "
        f"`{workspace / 'plan.md'}` for the build plan, and "
        f"`{workspace / 'acceptance-criteria.md'}` for the pass/fail checklist "
        f"QA will verify (your definition of done). Implement ONLY this node's "
        f"scope; do not touch other nodes' work. Take in the Reviewer's and "
        f"QA's latest REJECT feedback from the last 2 iterations (if any) — "
        f"that is what you must address this iteration.\n\n"
        f"Also take in any direct user guidance left as human feedback.\n\n"
        f"Implement or refine the code under your node worktree ({cwd}) to "
        f"address the feedback and satisfy every acceptance criterion in this "
        f"node's scope. Before completing your turn, record a Coder progress "
        f"entry with a `summary` of what you built or changed, and refresh "
        f"this node's status.md with a short, clean snapshot — current status, "
        f"execution path, what's been tried, what worked, what didn't, "
        f"pointers for the next step."
    )
    return _with_protocol(body, "coder", use_in_process_tools)


def _reviewer_prompt(
    node: Node,
    *,
    workspace: Path,
    cwd: Path,
    iteration: int,
    use_in_process_tools: bool = True,
) -> str:
    body = (
        f"You are the Reviewer for node `{node.id}` (type: {node.type}).\n"
        f"Node worktree — the Coder's changes are HERE: {cwd}\n"
        f"Iteration: {iteration}\n\n"
        f"Start by loading this node's rolling `status.md` scratchpad so you "
        f"know what the Coder claims the current state is.\n\n"
        f"The shared specs are READ-ONLY: read `{workspace / 'plan.md'}` for "
        f"the build plan and `{workspace / 'acceptance-criteria.md'}` for the "
        f"pass/fail checklist, and locate the slice that belongs to node "
        f"`{node.id}`. Take in the Coder's latest summary, and any direct "
        f"user guidance left as human feedback.\n\n"
        f"Work closely with the Coder: inspect the changed files under {cwd}, "
        f"then **build the code, run it, and execute the relevant tests** "
        f"against this node's plan and acceptance criteria. APPROVE only when "
        f"you have seen the change actually build and run correctly and have "
        f"evidence the criteria will hold; REJECT — with specific, actionable "
        f"feedback citing exact errors or failing tests — when the build/tests "
        f"fail, runtime behavior contradicts the plan, or any acceptance "
        f"criterion in scope is clearly unmet. Keep the loop tight: skip long "
        f"benchmarks and full-suite stress runs.\n\n"
        f"Before completing your turn, record a Reviewer progress entry with a "
        f"`summary` and a `decision` of exactly `APPROVE` or `REJECT` — cite "
        f"the commands you ran and what you observed — and refresh status.md "
        f"to reflect the post-review state and what the Coder must address "
        f"next on REJECT."
    )
    return _with_protocol(body, "reviewer", use_in_process_tools)


def _qa_prompt(
    node: Node,
    *,
    workspace: Path,
    cwd: Path,
    iteration: int,
    use_in_process_tools: bool = True,
) -> str:
    body = (
        f"You are QA for node `{node.id}` (type: {node.type}).\n"
        f"Node worktree — the code to verify is HERE: {cwd}\n"
        f"Iteration: {iteration}\n\n"
        f"Read `{workspace / 'task.yaml'}` (the user's stated intent — ultimate "
        f"ground truth) and `{workspace / 'acceptance-criteria.md'}` (the "
        f"pass/fail checklist), and focus on the slice that belongs to node "
        f"`{node.id}`. Do NOT read plan.md, progress.yaml, status.md, or any "
        f"other intermediate artifact; your verdict must be grounded solely in "
        f"those two specs and the actual code you build and run under {cwd}. On "
        f"any conflict between the criteria and `task.yaml`, `task.yaml` wins — "
        f"call out the gap.\n\n"
        f"Take in any direct user guidance left as human feedback; treat it on "
        f"par with `task.yaml`.\n\n"
        f"Discover this node's code under {cwd} yourself (ls, grep, etc.), "
        f"build it, run tests, and verify every acceptance criterion in scope "
        f"at runtime. Do not rely on code review alone.\n\n"
        f"Before completing your turn, record a QA progress entry with: "
        f"`summary` (per-criterion pass/fail with runtime evidence, strengths, "
        f"weaknesses, recommendation), `decision` (exactly `APPROVE` or "
        f"`REJECT`), and `weighted_score` (the weighted average in [0, 10]). "
        f"APPROVE closes this node; REJECT sends the work back to the Coder — "
        f"put the gaps they must fix in `summary`."
    )
    return _with_protocol(body, "qa", use_in_process_tools)


def make_run_node(
    *,
    workspace: Path,
    prompts: PromptBundle,
    policy_for_type: PolicyForType,
    num_iterations: int,
    backend_kind: str,
    model: str,
    use_in_process_tools: bool = True,
) -> RunNode:
    """Build the :data:`~agent_flow.orchestration.RunNode` for a concurrent run.

    Args:
        workspace: The shared workspace root. Each node's private sub-workspace
            is carved out under ``workspace/nodes/<slug>/``.
        prompts: The system-prompt bundle; ``coder`` / ``reviewer`` / ``qa`` are
            reused verbatim as the per-node agents' system prompts.
        policy_for_type: Maps a node's opaque ``type`` to its
            :class:`~agent_flow.workflows.agent_team.node_policy.NodeTypePolicy`
            (whether it runs QA, whether it is a replan unit). Injected so this
            module stays workflow-agnostic.
        num_iterations: Per-node budget of coder→reviewer(→qa) iterations before
            the node is reported ``FAILED``.
        backend_kind: Backend kind for every per-node agent (e.g.
            ``"claude-code"``).
        model: Model id for every per-node agent.
        use_in_process_tools: Whether per-node agents may register in-process
            (SDK) MCP tools. ``False`` under ``--no-mcp-tools``: every per-node
            agent is then built with ``tools=None`` and no required-tool Stop
            hooks, and its progress/status flow through the node-scoped handoff
            protocol in :func:`_invoke_node_agent_mcpless` instead.

    Returns:
        An async ``run_node(node, cwd) -> NodeOutcome`` that runs one node to a
        terminal outcome.
    """
    workspace = Path(workspace)

    async def run_node(node: Node, cwd: Path) -> NodeOutcome:
        cwd = Path(cwd)
        policy = policy_for_type(node.type)
        nw = create_node_workspace(workspace, node.id)

        # Bind the progress/status tools to this node's PRIVATE contexts so all
        # recording lands in ``workspace/nodes/<slug>/`` and never in a shared
        # file. Hold onto ``progress_ctx`` to stamp each turn's iteration.
        #
        # Under ``--no-mcp-tools`` no tools are registered at all: every agent
        # gets ``tools=None`` and no required-tool Stop hooks, and recording
        # runs through the handoff protocol instead (which stamps the iteration
        # itself, so ``progress_ctx`` is not needed either).
        progress_ctx = nw.progress_context() if use_in_process_tools else None
        if use_in_process_tools:
            progress_tools = build_progress_tools(progress_ctx)
            status_tools = build_status_tools(nw.status_context())
            role_tools = {
                "coder": progress_tools["coder"] + status_tools["coder"],
                "reviewer": progress_tools["reviewer"] + status_tools["reviewer"],
                "qa": progress_tools["qa"],
            }
            role_required = {
                "coder": ["append_coder_progress", "update_status"],
                "reviewer": ["append_reviewer_progress", "update_status"],
                "qa": ["append_qa_progress"],
            }
        else:
            role_tools = {"coder": None, "reviewer": None, "qa": None}
            role_required = {"coder": None, "reviewer": None, "qa": None}

        # Build the agents INSIDE the try so a builder that raises partway
        # (e.g. reviewer/qa construction fails after the coder is built) still
        # tears down every agent that WAS built via the finally clause, rather
        # than leaking the earlier ones. Each agent is registered in ``agents``
        # immediately after it is constructed.
        agents: list[Any] = []
        iterations = 0
        try:
            coder = _build_node_agent(
                "coder",
                prompts.coder,
                role_tools["coder"],
                required_tools=role_required["coder"],
                cwd=cwd,
                backend_kind=backend_kind,
                model=model,
            )
            agents.append(coder)
            reviewer = _build_node_agent(
                "reviewer",
                prompts.reviewer,
                role_tools["reviewer"],
                required_tools=role_required["reviewer"],
                cwd=cwd,
                backend_kind=backend_kind,
                model=model,
            )
            agents.append(reviewer)
            qa = None
            if policy.runs_qa:
                qa = _build_node_agent(
                    "qa",
                    prompts.qa,
                    role_tools["qa"],
                    required_tools=role_required["qa"],
                    cwd=cwd,
                    backend_kind=backend_kind,
                    model=model,
                    session_mode="stateless",
                )
                agents.append(qa)

            for i in range(num_iterations):
                iteration = i + 1
                iterations = iteration
                # Stamp entries recorded by the MCP tools this iteration. In
                # no-MCP mode the handoff writer stamps it instead.
                if progress_ctx is not None:
                    progress_ctx.current_iteration = iteration

                await _run_turn(
                    coder,
                    "coder",
                    _coder_prompt(
                        node,
                        workspace=workspace,
                        cwd=cwd,
                        iteration=iteration,
                        use_in_process_tools=use_in_process_tools,
                    ),
                    iteration,
                    nw=nw,
                    use_in_process_tools=use_in_process_tools,
                )
                await _run_turn(
                    reviewer,
                    "reviewer",
                    _reviewer_prompt(
                        node,
                        workspace=workspace,
                        cwd=cwd,
                        iteration=iteration,
                        use_in_process_tools=use_in_process_tools,
                    ),
                    iteration,
                    nw=nw,
                    use_in_process_tools=use_in_process_tools,
                )

                if _normalized_decision(nw.progress_path, "reviewer") != "APPROVE":
                    # REJECT (or missing) → loop back to the coder, skip QA.
                    continue

                if not policy.runs_qa:
                    return _outcome(NodeState.DONE, node, policy, iterations)

                await _run_turn(
                    qa,
                    "qa",
                    _qa_prompt(
                        node,
                        workspace=workspace,
                        cwd=cwd,
                        iteration=iteration,
                        use_in_process_tools=use_in_process_tools,
                    ),
                    iteration,
                    nw=nw,
                    use_in_process_tools=use_in_process_tools,
                )
                if _normalized_decision(nw.progress_path, "qa") == "APPROVE":
                    return _outcome(NodeState.DONE, node, policy, iterations)
                # QA REJECT (or missing) → next iteration.

            # Budget exhausted without reaching a terminal APPROVE.
            return _outcome(NodeState.FAILED, node, policy, iterations)
        finally:
            await _teardown_agents(agents)

    return run_node


async def _teardown_agents(agents: list[Any]) -> None:
    """Tear down every built agent, ensuring one failure never skips the rest.

    Each agent's ``__aexit__`` is awaited in its own ``try`` so a single failing
    teardown cannot leak the remaining agents. The first exception raised is
    re-raised after all agents have been attempted, preserving the error while
    still guaranteeing every agent gets a teardown call.
    """
    first_error: Exception | None = None
    for agent in agents:
        try:
            await agent.__aexit__(None, None, None)
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _outcome(
    terminal_state: NodeState,
    node: Node,
    policy: _NodePolicy,
    iterations: int,
) -> NodeOutcome:
    return NodeOutcome(
        terminal_state=terminal_state,
        needs_replan=policy.is_replan_unit,
        info={"iterations": iterations, "node_id": node.id},
    )
