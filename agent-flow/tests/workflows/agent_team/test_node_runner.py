"""Tests for the per-node ``coder ⇄ reviewer (⇄ qa)`` loop.

The loop is driven deterministically with a *fake invoker* that never calls a
real backend: it records scripted reviewer/qa ``decision`` entries into the
node's PRIVATE ``progress.yaml`` so the loop's ``latest_entry`` reads them —
the exact recording trick ``test_workflow.py``'s ``_stub_agents`` uses, but
scoped per node instead of to a shared workspace file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.orchestration import Node, NodeState
from agent_flow.workflows.agent_team import node_runner as node_runner_module
from agent_flow.workflows.agent_team import progress as progress_module
from agent_flow.workflows.agent_team.prompts import DEFAULT_PROMPTS
from agent_flow.workflows.agent_team.status import read_status_text
from agent_flow.workflows.modeling_bringup.node_policy import policy_for_type


def _append(progress_path: Path, agent: str, entry: dict) -> None:
    """Append ``entry`` to ``progress_path`` — mirrors ``_stub_agents``'s writer."""
    data = progress_module.read_progress(progress_path)
    data[progress_module._STAGE_BY_AGENT[agent]].append(entry)
    progress_module.write_progress(progress_path, data)


class _FakeInvoker:
    """Async stand-in for ``node_runner._invoke_node_agent``.

    Records a scripted decision (per role, per node) into the node's private
    ``progress.yaml`` / ``status.md`` on each turn so the real loop can read it
    back. Decision sequences are keyed by node id so two nodes driven through
    one ``run_node`` advance independently.
    """

    def __init__(
        self,
        *,
        reviewer_decisions=("APPROVE",),
        qa_decisions=("APPROVE",),
        qa_scores=(9.5,),
    ) -> None:
        self._reviewer_seq = list(reviewer_decisions)
        self._qa_seq = list(qa_decisions)
        self._score_seq = list(qa_scores)
        self._reviewer_iters: dict[str, object] = {}
        self._qa_iters: dict[str, object] = {}
        self._score_iters: dict[str, object] = {}
        self.calls: list[tuple[str, str, int]] = []
        self.prompts: list[tuple[str, str, str]] = []
        self.cwds: list[tuple[str, str]] = []

    @staticmethod
    def _next(iters: dict, key: str, seq: list, default):
        it = iters.setdefault(key, iter(seq))
        return next(it, seq[-1] if seq else default)

    async def __call__(self, agent, role, prompt, iteration, *, nw):
        self.calls.append((nw.node_id, role, iteration))
        self.prompts.append((nw.node_id, role, prompt))
        if role == "coder":
            _append(
                nw.progress_path,
                "coder",
                {
                    "iteration": iteration,
                    "agent": "coder",
                    "summary": f"built {nw.node_id} @ iter {iteration}",
                },
            )
            # Also exercise the node's private status.md so isolation is testable.
            nw.status_path.write_text(
                f"# {nw.node_id} status @ iter {iteration}\n", encoding="utf-8"
            )
            return
        if role == "reviewer":
            decision = self._next(self._reviewer_iters, nw.node_id, self._reviewer_seq, "APPROVE")
            _append(
                nw.progress_path,
                "reviewer",
                {"iteration": iteration, "agent": "reviewer", "decision": decision},
            )
            return
        if role == "qa":
            decision = self._next(self._qa_iters, nw.node_id, self._qa_seq, "APPROVE")
            score = self._next(self._score_iters, nw.node_id, self._score_seq, 9.5)
            entry = {"iteration": iteration, "agent": "qa", "decision": decision}
            if score is not None:
                entry["weighted_score"] = float(score)
            _append(nw.progress_path, "qa", entry)
            return
        raise AssertionError(f"unexpected role {role!r}")


def _install_fake(monkeypatch, invoker: _FakeInvoker) -> None:
    monkeypatch.setattr(node_runner_module, "_invoke_node_agent", invoker)


def _make_run_node(workspace: Path, *, num_iterations: int):
    return node_runner_module.make_run_node(
        workspace=workspace,
        prompts=DEFAULT_PROMPTS,
        policy_for_type=policy_for_type,
        num_iterations=num_iterations,
        backend_kind="claude-code",
        model="test-model",
    )


def _node_progress_path(workspace: Path, node_id: str) -> Path:
    return workspace / "nodes" / node_runner_module.node_dir_slug(node_id) / "progress.yaml"


# --------------------------------------------------------------------- goal node


async def test_goal_node_done_on_first_reviewer_approve(tmp_path, monkeypatch):
    """A goal node closes DONE on the first reviewer APPROVE, with no QA."""
    invoker = _FakeInvoker(reviewer_decisions=["APPROVE"])
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt-goal"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=5)
    outcome = await run_node(Node(id="s1.g1", type="goal"), cwd)

    assert outcome.terminal_state is NodeState.DONE
    assert outcome.needs_replan is False  # goal is not a replan unit
    assert outcome.info == {"iterations": 1, "node_id": "s1.g1"}

    # Exactly coder + reviewer ran (no qa for a goal node).
    assert [role for (_nid, role, _it) in invoker.calls] == ["coder", "reviewer"]

    # Entries landed in the node's PRIVATE progress.yaml, not a shared file.
    private = _node_progress_path(workspace, "s1.g1")
    data = progress_module.read_progress(private)
    agents = [e["agent"] for e in data["build_stage"]]
    assert agents == ["coder", "reviewer"]
    assert not (workspace / "progress.yaml").exists()

    # The node's private status.md was written too.
    status = workspace / "nodes" / "s1.g1" / "status.md"
    assert "s1.g1 status" in read_status_text(status)


async def test_goal_node_loops_back_on_reviewer_reject(tmp_path, monkeypatch):
    """A reviewer REJECT re-runs the coder; the second APPROVE closes DONE."""
    invoker = _FakeInvoker(reviewer_decisions=["REJECT", "APPROVE"])
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=5)
    outcome = await run_node(Node(id="g", type="goal"), cwd)

    assert outcome.terminal_state is NodeState.DONE
    assert outcome.info["iterations"] == 2
    # coder/reviewer twice; QA never (goal node).
    assert [role for (_n, role, _i) in invoker.calls] == [
        "coder",
        "reviewer",
        "coder",
        "reviewer",
    ]


# -------------------------------------------------------------------- stage node


async def test_stage_node_done_after_reviewer_then_qa_approve(tmp_path, monkeypatch):
    """A stage node needs reviewer APPROVE *and* QA APPROVE, and needs replan."""
    invoker = _FakeInvoker(reviewer_decisions=["APPROVE"], qa_decisions=["APPROVE"])
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt-stage"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=5)
    outcome = await run_node(Node(id="s1", type="stage"), cwd)

    assert outcome.terminal_state is NodeState.DONE
    assert outcome.needs_replan is True  # stage is the replan unit
    assert outcome.info == {"iterations": 1, "node_id": "s1"}
    assert [role for (_n, role, _i) in invoker.calls] == ["coder", "reviewer", "qa"]

    private = _node_progress_path(workspace, "s1")
    data = progress_module.read_progress(private)
    assert [e["agent"] for e in data["build_stage"]] == ["coder", "reviewer", "qa"]


async def test_stage_node_qa_reject_loops_back_to_coder(tmp_path, monkeypatch):
    """Reviewer APPROVE but QA REJECT re-runs the coder for another iteration."""
    invoker = _FakeInvoker(
        reviewer_decisions=["APPROVE", "APPROVE"],
        qa_decisions=["REJECT", "APPROVE"],
        qa_scores=[9.0, 9.0],
    )
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=5)
    outcome = await run_node(Node(id="s1", type="stage"), cwd)

    assert outcome.terminal_state is NodeState.DONE
    assert outcome.info["iterations"] == 2
    assert [role for (_n, role, _i) in invoker.calls] == [
        "coder",
        "reviewer",
        "qa",
        "coder",
        "reviewer",
        "qa",
    ]


# --------------------------------------------------------------- budget exhausted


async def test_budget_exhaustion_returns_failed(tmp_path, monkeypatch):
    """A node whose reviewer never APPROVEs within the budget ends FAILED."""
    invoker = _FakeInvoker(reviewer_decisions=["REJECT"])  # cycles → always REJECT
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=3)
    outcome = await run_node(Node(id="g", type="goal"), cwd)

    assert outcome.terminal_state is NodeState.FAILED
    assert outcome.needs_replan is False
    assert outcome.info == {"iterations": 3, "node_id": "g"}
    # coder+reviewer three times; QA never (never reached APPROVE).
    roles = [role for (_n, role, _i) in invoker.calls]
    assert roles == ["coder", "reviewer"] * 3


async def test_stage_budget_exhaustion_still_flags_replan(tmp_path, monkeypatch):
    """A stage node that exhausts its budget still reports ``needs_replan``."""
    invoker = _FakeInvoker(reviewer_decisions=["APPROVE"], qa_decisions=["REJECT"], qa_scores=[3.0])
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=2)
    outcome = await run_node(Node(id="s1", type="stage"), cwd)

    assert outcome.terminal_state is NodeState.FAILED
    assert outcome.needs_replan is True
    assert outcome.info["iterations"] == 2


# ------------------------------------------------------------------- isolation


async def test_two_nodes_write_to_separate_sub_workspaces(tmp_path, monkeypatch):
    """Two nodes driven by one ``run_node`` never share progress/status files."""
    invoker = _FakeInvoker(reviewer_decisions=["APPROVE"])
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd_a = tmp_path / "wt-a"
    cwd_a.mkdir()
    cwd_b = tmp_path / "wt-b"
    cwd_b.mkdir()

    run_node = _make_run_node(workspace, num_iterations=5)
    out_a = await run_node(Node(id="s1.g1", type="goal"), cwd_a)
    out_b = await run_node(Node(id="s1.g2", type="goal"), cwd_b)

    assert out_a.terminal_state is NodeState.DONE
    assert out_b.terminal_state is NodeState.DONE

    path_a = _node_progress_path(workspace, "s1.g1")
    path_b = _node_progress_path(workspace, "s1.g2")
    assert path_a != path_b

    data_a = progress_module.read_progress(path_a)
    data_b = progress_module.read_progress(path_b)
    # Each node recorded exactly its own coder+reviewer entries.
    assert [e["summary"] for e in data_a["build_stage"] if e["agent"] == "coder"] == [
        "built s1.g1 @ iter 1"
    ]
    assert [e["summary"] for e in data_b["build_stage"] if e["agent"] == "coder"] == [
        "built s1.g2 @ iter 1"
    ]

    status_a = read_status_text(workspace / "nodes" / "s1.g1" / "status.md")
    status_b = read_status_text(workspace / "nodes" / "s1.g2" / "status.md")
    assert "s1.g1" in status_a and "s1.g2" not in status_a
    assert "s1.g2" in status_b and "s1.g1" not in status_b


# ------------------------------------------------------------- prompt scoping


async def test_prompts_are_node_scoped(tmp_path, monkeypatch):
    """Per-turn prompts name the node and point read-only at the shared specs."""
    invoker = _FakeInvoker(reviewer_decisions=["APPROVE"], qa_decisions=["APPROVE"])
    _install_fake(monkeypatch, invoker)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt-scope"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=3)
    await run_node(Node(id="s1", type="stage"), cwd)

    by_role = {role: prompt for (_n, role, prompt) in invoker.prompts}
    # Every per-turn prompt names the node it is scoped to.
    for role in ("coder", "reviewer", "qa"):
        assert "s1" in by_role[role]
    # Coder and QA ground on the shared task.yaml (read-only); the Reviewer
    # works from plan.md + acceptance-criteria.md, mirroring the linear
    # workflow's role split.
    assert str(workspace / "task.yaml") in by_role["coder"]
    assert str(workspace / "task.yaml") in by_role["qa"]
    assert str(workspace / "plan.md") in by_role["reviewer"]
    assert str(workspace / "acceptance-criteria.md") in by_role["reviewer"]
    # The coder is told to edit under its worktree cwd; QA never reads plan.md.
    assert str(cwd) in by_role["coder"]
    assert str(workspace / "acceptance-criteria.md") in by_role["qa"]
    assert str(workspace / "plan.md") not in by_role["qa"]


# --------------------------------------------------------------- agent teardown


class _RecordingAgent:
    """Fake agent whose ``__aexit__`` records the call and can raise on demand."""

    def __init__(self, name: str, teardowns: list[str], *, raises: Exception | None = None):
        self.name = name
        self._teardowns = teardowns
        self._raises = raises

    async def __aexit__(self, *_exc_info) -> None:
        self._teardowns.append(self.name)
        if self._raises is not None:
            raise self._raises


async def test_teardown_is_isolated_and_reraises_first_error(tmp_path, monkeypatch):
    """A failing agent ``__aexit__`` still tears down the rest, then re-raises."""
    invoker = _FakeInvoker(reviewer_decisions=["APPROVE"], qa_decisions=["APPROVE"])
    _install_fake(monkeypatch, invoker)

    teardowns: list[str] = []
    boom = RuntimeError("coder teardown failed")

    def _fake_build(name, *_args, **_kwargs):
        return _RecordingAgent(name, teardowns, raises=boom if name == "coder" else None)

    monkeypatch.setattr(node_runner_module, "_build_node_agent", _fake_build)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=3)
    with pytest.raises(RuntimeError, match="coder teardown failed"):
        await run_node(Node(id="s1", type="stage"), cwd)

    # Every built agent was torn down despite the coder's failing __aexit__, and
    # the first teardown error propagated out of the finally.
    assert teardowns == ["coder", "reviewer", "qa"]


async def test_partial_build_failure_tears_down_already_built_agents(tmp_path, monkeypatch):
    """If a later agent's construction raises, earlier agents are still released."""
    invoker = _FakeInvoker()
    _install_fake(monkeypatch, invoker)

    teardowns: list[str] = []
    build_boom = RuntimeError("reviewer build failed")

    def _fake_build(name, *_args, **_kwargs):
        if name == "reviewer":
            raise build_boom
        return _RecordingAgent(name, teardowns)

    monkeypatch.setattr(node_runner_module, "_build_node_agent", _fake_build)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    cwd = tmp_path / "wt"
    cwd.mkdir()

    run_node = _make_run_node(workspace, num_iterations=3)
    with pytest.raises(RuntimeError, match="reviewer build failed"):
        await run_node(Node(id="s1", type="stage"), cwd)

    # The coder was built before the reviewer build blew up, so it is torn down;
    # the reviewer/qa were never built, so nothing else leaks.
    assert teardowns == ["coder"]
