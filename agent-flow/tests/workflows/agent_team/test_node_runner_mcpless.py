"""Tests for ``--concurrent`` combined with ``--no-mcp-tools``.

The concurrent DAG path and the no-in-process-MCP path were developed
independently, so this module is the contract for their intersection:

* per-node agents are built with ``tools=None`` and no required-tool Stop
  hooks, exactly as the linear path does under ``--no-mcp-tools``;
* the per-turn prompts name no MCP tool in that mode (and still name them in
  MCP mode) — the transport-neutral-body invariant ``PromptBundle`` declares;
* a node's turn is recorded through the handoff protocol into that node's
  PRIVATE ``progress.yaml``, with the handoff file itself node-scoped so
  parallel nodes cannot clobber each other; and
* human feedback — the one genuinely shared read — reaches node agents in
  both transport modes.

The loop is driven with fake agents that write scripted handoff YAML, so the
real :func:`node_runner._invoke_node_agent_mcpless` runs end to end without a
backend.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_flow.orchestration import Node, NodeState
from agent_flow.workflows.agent_team import mcpless
from agent_flow.workflows.agent_team import node_runner as node_runner_module
from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team.node_workspace import create_node_workspace
from agent_flow.workflows.agent_team.prompts import DEFAULT_PROMPTS
from agent_flow.workflows.modeling_bringup.node_policy import policy_for_type

# Every MCP tool the workflow can register. No per-turn prompt may name one of
# these under ``--no-mcp-tools`` — mirrors ``test_mcpless``'s preamble check.
_MCP_TOOL_NAMES = (
    "append_coder_progress",
    "append_reviewer_progress",
    "append_qa_progress",
    "read_latest_progress",
    "read_human_feedback",
    "read_status",
    "update_status",
    "ask_human",
)


def _handoff_from_prompt(prompt: str, role: str) -> Path:
    """Recover the handoff path the preamble told this role to write.

    Doubles as an assertion that the preamble names the path on its own line.
    """
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.endswith(f"{role}.yaml"):
            return Path(stripped)
    raise AssertionError(f"{role} prompt does not name a handoff file:\n{prompt}")


class _HandoffAgent:
    """Fake agent that writes a scripted handoff file on each turn.

    ``script`` is one entry per turn: a mapping is written as the handoff YAML,
    and ``None`` writes nothing at all (simulating an agent that ignored the
    protocol, which must trigger the corrective retry).
    """

    def __init__(self, role: str, script: list[dict[str, Any] | None], prompts: list[str]):
        self.role = role
        self._script = list(script)
        self._prompts = prompts
        self.turns = 0

    async def aforward(self, prompt: str) -> None:
        self._prompts.append(prompt)
        payload = self._script.pop(0) if self._script else {}
        self.turns += 1
        if payload is None:
            return
        path = _handoff_from_prompt(prompt, self.role)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    async def __aexit__(self, *_exc_info) -> None:
        return None


_APPROVING_SCRIPT: dict[str, list[dict[str, Any] | None]] = {
    "coder": [{"summary": "built the node"}],
    "reviewer": [{"summary": "builds and tests pass", "decision": "APPROVE"}],
    "qa": [{"summary": "every criterion met", "decision": "APPROVE", "weighted_score": 9.5}],
}


def _install_handoff_agents(
    monkeypatch,
    prompts: list[str],
    script: dict[str, list[dict[str, Any] | None]] | None = None,
) -> dict[str, _HandoffAgent]:
    """Replace ``_build_node_agent`` with fakes that honor the handoff protocol."""
    script = script or _APPROVING_SCRIPT
    built: dict[str, _HandoffAgent] = {}

    def _fake_build(name, *_args, **_kwargs):
        agent = _HandoffAgent(name, list(script.get(name, [])), prompts)
        built[name] = agent
        return agent

    monkeypatch.setattr(node_runner_module, "_build_node_agent", _fake_build)
    return built


def _make_run_node(workspace: Path, *, use_in_process_tools: bool, num_iterations: int = 3):
    return node_runner_module.make_run_node(
        workspace=workspace,
        prompts=DEFAULT_PROMPTS,
        policy_for_type=policy_for_type,
        num_iterations=num_iterations,
        backend_kind="claude-code",
        model="test-model",
        use_in_process_tools=use_in_process_tools,
    )


def _workspace(tmp_path: Path, name: str = "ws") -> Path:
    ws = tmp_path / name
    ws.mkdir()
    progress_module.init_progress_file(ws / "progress.yaml")
    return ws


def _cwd(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


# ------------------------------------------------------------- agent construction


async def test_no_mcp_builds_node_agents_without_tools_or_required_hooks(tmp_path, monkeypatch):
    """Under ``--no-mcp-tools`` no per-node agent registers a tool or a Stop hook."""
    seen: list[tuple[str, Any, Any]] = []
    prompts: list[str] = []

    def _fake_build(name, _system_prompt, tools, *, required_tools, **_kwargs):
        seen.append((name, tools, required_tools))
        return _HandoffAgent(name, list(_APPROVING_SCRIPT.get(name, [])), prompts)

    monkeypatch.setattr(node_runner_module, "_build_node_agent", _fake_build)

    run_node = _make_run_node(_workspace(tmp_path), use_in_process_tools=False)
    await run_node(Node(id="s1", type="stage"), _cwd(tmp_path, "wt"))

    assert [name for name, _t, _r in seen] == ["coder", "reviewer", "qa"]
    for name, tools, required_tools in seen:
        assert tools is None, f"{name} was given MCP tools under --no-mcp-tools"
        assert required_tools is None, f"{name} was given required-tool hooks"


async def test_mcp_mode_still_builds_tools_and_required_hooks(tmp_path, monkeypatch):
    """Regression guard: the default path keeps its tools and Stop hooks."""
    seen: list[tuple[str, Any, Any]] = []

    def _fake_build(name, _system_prompt, tools, *, required_tools, **_kwargs):
        seen.append((name, tools, required_tools))
        return _HandoffAgent(name, [], [])

    monkeypatch.setattr(node_runner_module, "_build_node_agent", _fake_build)
    # MCP mode records through the tools, which the fake agent does not call,
    # so the node exhausts its budget — irrelevant here; only the build matters.
    monkeypatch.setattr(
        node_runner_module,
        "_invoke_node_agent",
        lambda *_a, **_k: asyncio.sleep(0),
    )

    run_node = _make_run_node(_workspace(tmp_path), use_in_process_tools=True, num_iterations=1)
    await run_node(Node(id="s1", type="stage"), _cwd(tmp_path, "wt"))

    by_name = {name: (tools, required) for name, tools, required in seen}
    assert by_name["coder"][0] is not None
    assert by_name["coder"][1] == ["append_coder_progress", "update_status"]
    assert by_name["reviewer"][1] == ["append_reviewer_progress", "update_status"]
    assert by_name["qa"][1] == ["append_qa_progress"]


# -------------------------------------------------------------------- prompts


@pytest.mark.parametrize("role", ["coder", "reviewer", "qa"])
def test_node_prompt_body_names_no_mcp_tool(role):
    """The transport-neutral body must not instruct a tool call.

    Under ``--no-mcp-tools`` the body is the whole prompt (plus ``mcpless``'s
    preamble), so any tool name left here would tell the agent to call
    something this run never registered.
    """
    builder = {
        "coder": node_runner_module._coder_prompt,
        "reviewer": node_runner_module._reviewer_prompt,
        "qa": node_runner_module._qa_prompt,
    }[role]
    prompt = builder(
        Node(id="s1", type="stage"),
        workspace=Path("/ws"),
        cwd=Path("/wt"),
        iteration=1,
        use_in_process_tools=False,
    )
    for tool in _MCP_TOOL_NAMES:
        assert tool not in prompt, f"{role} body names the MCP tool {tool!r}"


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("coder", ("read_status", "read_latest_progress", "append_coder_progress")),
        ("reviewer", ("read_status", "read_latest_progress", "append_reviewer_progress")),
        ("qa", ("read_human_feedback", "append_qa_progress")),
    ],
)
def test_node_prompt_keeps_mcp_protocol_in_mcp_mode(role, expected):
    """MCP mode still spells out the tools — the block is only dropped, never lost."""
    builder = {
        "coder": node_runner_module._coder_prompt,
        "reviewer": node_runner_module._reviewer_prompt,
        "qa": node_runner_module._qa_prompt,
    }[role]
    prompt = builder(
        Node(id="s1", type="stage"),
        workspace=Path("/ws"),
        cwd=Path("/wt"),
        iteration=1,
        use_in_process_tools=True,
    )
    for tool in expected:
        assert tool in prompt


# ------------------------------------------------------------- handoff recording


async def test_handoff_round_trip_records_into_private_progress(tmp_path, monkeypatch):
    """A no-MCP node records coder/reviewer/qa entries into its own progress.yaml."""
    prompts: list[str] = []
    _install_handoff_agents(monkeypatch, prompts)

    workspace = _workspace(tmp_path)
    run_node = _make_run_node(workspace, use_in_process_tools=False)
    outcome = await run_node(Node(id="s1", type="stage"), _cwd(tmp_path, "wt"))

    assert outcome.terminal_state is NodeState.DONE

    node_progress = workspace / "nodes" / "s1" / "progress.yaml"
    data = progress_module.read_progress(node_progress)
    build = data["build_stage"]
    assert [e["agent"] for e in build] == ["coder", "reviewer", "qa"]
    assert build[0]["summary"] == "built the node"
    assert build[1]["decision"] == "APPROVE"
    assert build[2]["weighted_score"] == 9.5
    # The orchestrator stamps iteration/agent; they are never trusted from the
    # handoff file itself.
    assert all(e["iteration"] == 1 for e in build)
    assert all("timestamp" in e for e in build)

    # Nothing leaked into the shared workspace progress.yaml.
    assert progress_module.read_progress(workspace / "progress.yaml")["build_stage"] == []


async def test_missing_handoff_triggers_one_corrective_retry(tmp_path, monkeypatch):
    """A turn that writes no handoff is retried once with a corrective notice."""
    prompts: list[str] = []
    script = dict(_APPROVING_SCRIPT)
    script["coder"] = [None, {"summary": "built on the retry"}]
    agents = _install_handoff_agents(monkeypatch, prompts, script)

    workspace = _workspace(tmp_path)
    run_node = _make_run_node(workspace, use_in_process_tools=False)
    outcome = await run_node(Node(id="s1", type="stage"), _cwd(tmp_path, "wt"))

    assert outcome.terminal_state is NodeState.DONE
    assert agents["coder"].turns == 2
    coder_prompts = [p for p in prompts if "You are the Coder" in p]
    assert "=== RETRY ===" not in coder_prompts[0]
    assert "=== RETRY ===" in coder_prompts[1]

    entry = progress_module.read_progress(workspace / "nodes" / "s1" / "progress.yaml")
    assert entry["build_stage"][0]["summary"] == "built on the retry"


async def test_two_failed_handoffs_raise(tmp_path, monkeypatch):
    """Both attempts failing surfaces the error rather than silently continuing."""
    script = dict(_APPROVING_SCRIPT)
    script["coder"] = [None, None]
    _install_handoff_agents(monkeypatch, [], script)

    run_node = _make_run_node(_workspace(tmp_path), use_in_process_tools=False)
    with pytest.raises(mcpless.HandoffError, match="did not write its handoff file"):
        await run_node(Node(id="s1", type="stage"), _cwd(tmp_path, "wt"))


async def test_invalid_handoff_payload_raises(tmp_path, monkeypatch):
    """A handoff that violates the role schema is rejected, not recorded."""
    script = dict(_APPROVING_SCRIPT)
    # Reviewer must emit APPROVE or REJECT; anything else is a protocol error.
    script["reviewer"] = [{"summary": "s", "decision": "MAYBE"}] * 2
    _install_handoff_agents(monkeypatch, [], script)

    run_node = _make_run_node(_workspace(tmp_path), use_in_process_tools=False)
    with pytest.raises(mcpless.HandoffError, match="decision"):
        await run_node(Node(id="s1", type="stage"), _cwd(tmp_path, "wt"))


async def test_handoff_files_are_node_private(tmp_path, monkeypatch):
    """Two nodes write their handoffs to separate directories.

    A shared ``.turn`` directory would have parallel nodes overwrite each
    other's ``<role>.yaml`` mid-turn.
    """
    prompts: list[str] = []
    _install_handoff_agents(monkeypatch, prompts)

    workspace = _workspace(tmp_path)
    run_node = _make_run_node(workspace, use_in_process_tools=False)
    await run_node(Node(id="s1/g1", type="goal"), _cwd(tmp_path, "wt-a"))
    await run_node(Node(id="s1/g2", type="goal"), _cwd(tmp_path, "wt-b"))

    handoff_dirs = {
        _handoff_from_prompt(p, "coder").parent for p in prompts if "You are the Coder" in p
    }
    assert handoff_dirs == {
        workspace / "nodes" / "s1_g1" / ".turn",
        workspace / "nodes" / "s1_g2" / ".turn",
    }
    # And no handoff directory was created at the shared workspace root.
    assert not (workspace / ".turn").exists()


# ------------------------------------------------------------- human feedback


async def test_no_mcp_node_context_carries_shared_human_feedback(tmp_path, monkeypatch):
    """``--feedback`` reaches a node agent even though it lives in the shared file.

    A node's private ``progress.yaml`` never holds ``human_feedback`` — only
    the shared workspace file does — so the context builder must read across.
    """
    prompts: list[str] = []
    _install_handoff_agents(monkeypatch, prompts)

    workspace = _workspace(tmp_path)
    progress_module.append_human_feedback(
        workspace / "progress.yaml",
        summary="prefer the streaming path",
        iteration=1,
        stage="build_stage",
    )

    run_node = _make_run_node(workspace, use_in_process_tools=False)
    await run_node(Node(id="s1", type="stage"), _cwd(tmp_path, "wt"))

    coder_prompt = next(p for p in prompts if "You are the Coder" in p)
    assert "prefer the streaming path" in coder_prompt


def test_mcp_node_feedback_tool_reads_the_shared_file(tmp_path):
    """The same fix on the MCP side: the node's tool reads shared feedback."""
    workspace = _workspace(tmp_path)
    progress_module.append_human_feedback(
        workspace / "progress.yaml",
        summary="prefer the streaming path",
        iteration=1,
        stage="build_stage",
    )

    nw = create_node_workspace(workspace, "s1")
    ctx = nw.progress_context()
    # The node still records into its own log; only feedback is read across.
    assert ctx.path == nw.progress_path
    assert ctx.human_feedback_path == workspace / "progress.yaml"

    tools = progress_module.build_progress_tools(ctx)
    tool = next(t for t in tools["coder"] if t.name == "read_human_feedback")
    out = asyncio.run(tool.handler({}))
    assert "prefer the streaming path" in out["content"][0]["text"]


def test_progress_context_feedback_path_defaults_to_path(tmp_path):
    """The linear path is unaffected: one file still serves both reads."""
    path = tmp_path / "progress.yaml"
    ctx = progress_module.ProgressContext(path=path)
    assert ctx.human_feedback_path == path


def test_gather_context_feedback_path_defaults_to_progress_path(tmp_path):
    """``gather_context`` keeps its single-file behavior when not split."""
    path = tmp_path / "progress.yaml"
    progress_module.init_progress_file(path)
    progress_module.append_human_feedback(
        path, summary="linear feedback", iteration=1, stage="build_stage"
    )
    out = mcpless.gather_context("qa", progress_path=path, status_path=tmp_path / "status.md")
    assert "linear feedback" in out
