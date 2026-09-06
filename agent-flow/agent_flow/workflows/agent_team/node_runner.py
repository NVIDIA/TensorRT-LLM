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

from .node_workspace import NodeWorkspace, create_node_workspace, node_dir_slug
from .progress import build_progress_tools, latest_entry
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
    """Run one agent turn for a node.

    The agent records its progress/status through the MCP tools bound to the
    node's private contexts, so this is a thin wrapper — its whole reason to
    exist is the await: ``run_node`` is async (the scheduler awaits it), so the
    agent must be driven with ``await agent.aforward(...)`` rather than the
    synchronous ``agent(...)`` the linear workflow uses. Calling the blocking
    form here would stall the scheduler's event loop and silently serialize
    every node.
    """
    await agent.aforward(prompt)


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


def _coder_prompt(node: Node, *, workspace: Path, cwd: Path, iteration: int) -> str:
    return (
        f"You are the Coder for node `{node.id}` (type: {node.type}).\n"
        f"Node worktree — edit code HERE: {cwd}\n"
        f"Iteration: {iteration}\n\n"
        f"Start by calling `read_status` to load this node's rolling "
        f"`status.md` scratchpad — that is your fastest way to pick up where "
        f"the previous turn left off.\n\n"
        f"The shared specs are READ-ONLY and describe the whole task; use them "
        f"to find the slice that belongs to node `{node.id}`: read "
        f"`{workspace / 'task.yaml'}` for the original task from the user, "
        f"`{workspace / 'plan.md'}` for the build plan, and "
        f"`{workspace / 'acceptance-criteria.md'}` for the pass/fail checklist "
        f"QA will verify (your definition of done). Implement ONLY this node's "
        f"scope; do not touch other nodes' work. Call `read_latest_progress` "
        f"with `iterations: 2` to fetch the Reviewer's and QA's latest REJECT "
        f"feedback (if any) — that is what you must address this iteration.\n\n"
        f"Also call `read_human_feedback` to fetch any direct user guidance.\n\n"
        f"Implement or refine the code under your node worktree ({cwd}) to "
        f"address the feedback and satisfy every acceptance criterion in this "
        f"node's scope. Before completing your turn, call **both** required "
        f"tools: `append_coder_progress` (with a `summary` of what you built or "
        f"changed) and `update_status` (overwriting this node's status.md with "
        f"a short, clean snapshot — current status, execution path, what's been "
        f"tried, what worked, what didn't, pointers for the next step)."
    )


def _reviewer_prompt(node: Node, *, workspace: Path, cwd: Path, iteration: int) -> str:
    return (
        f"You are the Reviewer for node `{node.id}` (type: {node.type}).\n"
        f"Node worktree — the Coder's changes are HERE: {cwd}\n"
        f"Iteration: {iteration}\n\n"
        f"Start by calling `read_status` to load this node's rolling "
        f"`status.md` scratchpad so you know what the Coder claims the current "
        f"state is.\n\n"
        f"The shared specs are READ-ONLY: read `{workspace / 'plan.md'}` for "
        f"the build plan and `{workspace / 'acceptance-criteria.md'}` for the "
        f"pass/fail checklist, and locate the slice that belongs to node "
        f'`{node.id}`. Call `read_latest_progress` with `agent: "coder"` to '
        f"fetch the Coder's latest summary, and `read_human_feedback` for any "
        f"direct user guidance.\n\n"
        f"Work closely with the Coder: inspect the changed files under {cwd}, "
        f"then **build the code, run it, and execute the relevant tests** "
        f"against this node's plan and acceptance criteria. APPROVE only when "
        f"you have seen the change actually build and run correctly and have "
        f"evidence the criteria will hold; REJECT — with specific, actionable "
        f"feedback citing exact errors or failing tests — when the build/tests "
        f"fail, runtime behavior contradicts the plan, or any acceptance "
        f"criterion in scope is clearly unmet. Keep the loop tight: skip long "
        f"benchmarks and full-suite stress runs.\n\n"
        f"Before completing your turn, call **both** required tools: "
        f"`append_reviewer_progress` (with `summary` and `decision`, exactly "
        f"`APPROVE` or `REJECT` — cite the commands you ran and what you "
        f"observed) and `update_status` (overwriting status.md to reflect the "
        f"post-review state and what the Coder must address next on REJECT)."
    )


def _qa_prompt(node: Node, *, workspace: Path, cwd: Path, iteration: int) -> str:
    return (
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
        f"Call `read_human_feedback` to fetch any direct user guidance; treat "
        f"it on par with `task.yaml`.\n\n"
        f"Discover this node's code under {cwd} yourself (ls, grep, etc.), "
        f"build it, run tests, and verify every acceptance criterion in scope "
        f"at runtime. Do not rely on code review alone.\n\n"
        f"Before completing your turn, call the `append_qa_progress` tool with: "
        f"`summary` (per-criterion pass/fail with runtime evidence, strengths, "
        f"weaknesses, recommendation), `decision` (exactly `APPROVE` or "
        f"`REJECT`), and `weighted_score` (the weighted average in [0, 10]). "
        f"APPROVE closes this node; REJECT sends the work back to the Coder — "
        f"put the gaps they must fix in `summary`."
    )


def make_run_node(
    *,
    workspace: Path,
    prompts: PromptBundle,
    policy_for_type: PolicyForType,
    num_iterations: int,
    backend_kind: str,
    model: str,
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
        progress_ctx = nw.progress_context()
        status_ctx = nw.status_context()
        progress_tools = build_progress_tools(progress_ctx)
        status_tools = build_status_tools(status_ctx)

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
                progress_tools["coder"] + status_tools["coder"],
                required_tools=["append_coder_progress", "update_status"],
                cwd=cwd,
                backend_kind=backend_kind,
                model=model,
            )
            agents.append(coder)
            reviewer = _build_node_agent(
                "reviewer",
                prompts.reviewer,
                progress_tools["reviewer"] + status_tools["reviewer"],
                required_tools=["append_reviewer_progress", "update_status"],
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
                    progress_tools["qa"],
                    required_tools=["append_qa_progress"],
                    cwd=cwd,
                    backend_kind=backend_kind,
                    model=model,
                    session_mode="stateless",
                )
                agents.append(qa)

            for i in range(num_iterations):
                iteration = i + 1
                iterations = iteration
                # Stamp entries recorded by the MCP tools this iteration.
                progress_ctx.current_iteration = iteration

                await _invoke_node_agent(
                    coder,
                    "coder",
                    _coder_prompt(node, workspace=workspace, cwd=cwd, iteration=iteration),
                    iteration,
                    nw=nw,
                )
                await _invoke_node_agent(
                    reviewer,
                    "reviewer",
                    _reviewer_prompt(node, workspace=workspace, cwd=cwd, iteration=iteration),
                    iteration,
                    nw=nw,
                )

                if _normalized_decision(nw.progress_path, "reviewer") != "APPROVE":
                    # REJECT (or missing) → loop back to the coder, skip QA.
                    continue

                if not policy.runs_qa:
                    return _outcome(NodeState.DONE, node, policy, iterations)

                await _invoke_node_agent(
                    qa,
                    "qa",
                    _qa_prompt(node, workspace=workspace, cwd=cwd, iteration=iteration),
                    iteration,
                    nw=nw,
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
