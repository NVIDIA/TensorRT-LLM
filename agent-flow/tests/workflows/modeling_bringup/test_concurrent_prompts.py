"""Tests for the modeling-bringup ``--concurrent`` prompt migration.

The concurrent mode teaches the PlanDrafter to emit a machine-readable
``## Execution Graph`` block (parseable by ``extract_execution_graph``) and
gives the concurrent coder/reviewer/qa a per-node protocol with NO shared
``## Stages & Goals`` status table — the orchestrator owns graph state.

These tests are the contract:

* ``build_modeling_bringup_prompts(concurrent=True)`` wires the
  ``CONCURRENT_EXTENSION`` of every role (and NOT the ``STAGE_GOAL_EXTENSION``,
  which is a distinct ``--replan-on-qa`` mode).
* The canonical ``EXECUTION_GRAPH_EXAMPLE`` the PlanDrafter prompt embeds
  round-trips through the real parser, proving the taught format is the format
  the engine accepts.
* The CLI threads ``--concurrent`` into the prompt build.
"""

from __future__ import annotations

import importlib

import pytest

from agent_flow.orchestration import ExecutionGraph
from agent_flow.workflows.agent_team.execution_graph import extract_execution_graph
from agent_flow.workflows.modeling_bringup import cli as _modeling_bringup_cli_module
from agent_flow.workflows.modeling_bringup.prompts import plan_drafter_extra

_CONCURRENT_ROLES = ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa")


def _build(**kwargs):
    mb_prompts = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    importlib.reload(mb_prompts)
    return mb_prompts.build_modeling_bringup_prompts(**kwargs)


def test_concurrent_plan_drafter_emits_execution_graph_and_node_id_acceptance():
    """PlanDrafter must learn to emit a `## Execution Graph` block + node-id acceptance."""
    plan = _build(concurrent=True).plan_drafter

    # The Execution Graph emission guidance (heading + the embedded canonical
    # block) reaches the PlanDrafter.
    assert "## Execution Graph" in plan
    assert "## Concurrent execution graph" in plan
    # The canonical example is embedded verbatim.
    assert plan_drafter_extra.EXECUTION_GRAPH_EXAMPLE in plan
    # Node schema vocabulary.
    assert "`nodes:`" in plan
    assert "`stage`" in plan and "`goal`" in plan
    assert "`impl`" in plan and "`merge`" in plan
    # Node-id-keyed acceptance-criteria partitioning replaces `## Stage N`.
    assert "## <node-id>" in plan
    assert "node-id keyed" in plan
    # Concurrent mode is distinct from the single-cursor Stage/Goal replan
    # protocol: none of that protocol's machinery reaches the concurrent
    # drafter (it may still *name* `## Stages & Goals` to say "do not write one").
    assert "## Replan lock matrix" not in plan
    assert "## Stage/Goal plan schema" not in plan
    # But the drafter is told NOT to produce a shared status table.
    assert "Do **not** write a `## Stages & Goals` status table" in plan


def test_concurrent_budget_block_carries_max_parallel_into_plan_drafter_only():
    """--concurrent + max_parallel injects a soft sizing block naming N, planner-only."""
    bundle = _build(concurrent=True, max_parallel=4)
    plan = bundle.plan_drafter

    # The budget subsection is present and names the concrete budget N=4.
    assert "### Concurrency budget" in plan
    assert "`--max-parallel 4`" in plan
    assert "at most 4 independent nodes" in plan
    # Soft sizing guidance about graph width relative to N.
    assert "widest ready-to-run level" in plan
    # It rides on top of the concurrent extension (still present), additively.
    assert "## Concurrent execution graph" in plan
    assert "\n\n\n" not in plan

    # Planner-only: the budget block must NOT leak into the other roles.
    for role in ("plan_reviewer", "coder", "reviewer", "qa"):
        assert "### Concurrency budget" not in getattr(bundle, role)


def test_concurrent_budget_block_absent_without_max_parallel():
    """concurrent=True but no max_parallel supplied → no budget block (the None gate)."""
    plan = _build(concurrent=True).plan_drafter
    assert "## Concurrent execution graph" in plan  # concurrent mode is still on
    assert "### Concurrency budget" not in plan


def test_budget_block_absent_on_the_linear_path_even_with_max_parallel():
    """max_parallel is meaningless off --concurrent, so it must not inject anything."""
    for bundle in (_build(max_parallel=8), _build(replan_on_qa=True, max_parallel=8)):
        for role in _CONCURRENT_ROLES:
            assert "### Concurrency budget" not in getattr(bundle, role)


def test_concurrent_plan_reviewer_reviews_execution_graph_contract():
    """PlanReviewer must check the `## Execution Graph` presence/validity + node-id acceptance."""
    plan_reviewer = _build(concurrent=True).plan_reviewer

    assert "## Execution Graph" in plan_reviewer
    assert "## <node-id>" in plan_reviewer
    assert "node-id keyed" in plan_reviewer
    # It must reject when the graph is missing or acceptance is ordinal-keyed.
    assert "no `## Execution Graph`" in plan_reviewer
    assert "`nodes:`" in plan_reviewer


@pytest.mark.parametrize("role", ["coder", "reviewer", "qa"])
def test_concurrent_worker_roles_carry_node_scoped_protocol(role):
    """Coder/Reviewer/QA must learn the one-node-per-turn protocol with NO shared table."""
    prompt = getattr(_build(concurrent=True), role)

    # The node-scoped protocol.
    assert "## Concurrent node protocol" in prompt
    assert "exactly ONE node this turn" in prompt
    assert "orchestrator owns graph state" in prompt
    # A worktree-scoped, private-file workflow.
    assert "worktree" in prompt

    # The single-cursor table machinery must be absent (the orchestrator owns
    # state — no `## Stages & Goals` table, no `Stage closed` markers).
    assert "## Stages & Goals" not in prompt
    assert "Stage closed" not in prompt


def test_concurrent_qa_verifies_node_id_acceptance_subsection():
    """QA additionally scopes verification to the assigned node's `## <node-id>` subsection."""
    qa = _build(concurrent=True).qa
    assert "## <node-id>" in qa
    assert "acceptance" in qa


def test_default_mode_carries_none_of_the_concurrent_text():
    """concurrent=False (default) leaves every role free of the CONCURRENT-mode text."""
    default_bundle = _build()
    replan_bundle = _build(replan_on_qa=True)

    for bundle in (default_bundle, replan_bundle):
        for role in _CONCURRENT_ROLES:
            prompt = getattr(bundle, role)
            assert "## Concurrent execution graph" not in prompt
            assert "## Concurrent node protocol" not in prompt
            assert "## Execution Graph" not in prompt
            assert "exactly ONE node this turn" not in prompt


def test_concurrent_is_byte_identical_addition_over_default():
    """Concurrent mode appends the CONCURRENT_EXTENSION on top of the base extended bundle.

    The base (concurrent-off) prompt must remain a prefix, so concurrent mode is
    strictly additive per role and never rewrites the shared bring-up guidance.
    """
    default_bundle = _build()
    concurrent_bundle = _build(concurrent=True)
    for role in _CONCURRENT_ROLES:
        base = getattr(default_bundle, role)
        conc = getattr(concurrent_bundle, role)
        assert conc != base, f"{role} concurrent extension is empty"
        assert conc.startswith(base.rstrip()), f"{role} concurrent prompt does not extend the base"
        assert "\n\n\n" not in conc, f"{role} concurrent prompt has a triple newline from joining"


def test_execution_graph_example_round_trips_through_parser():
    """The embedded canonical example parses into a valid nested stage->goal ExecutionGraph.

    This proves the format the PlanDrafter is taught is exactly the format the
    engine's ``extract_execution_graph`` accepts.
    """
    example = plan_drafter_extra.EXECUTION_GRAPH_EXAMPLE

    # Embed the example inside a larger plan.md-shaped markdown document, the
    # way the PlanDrafter would write it.
    plan_md = (
        "# Plan for <Model> bring-up\n\n"
        "## Implementation Steps\n\n"
        "### Stage 1: accuracy convergence\n"
        "- Goal 1.1: attention module\n\n"
        f"{example}\n\n"
        "## Risks\n- loose tolerances\n"
    )

    graph = extract_execution_graph(plan_md)
    assert isinstance(graph, ExecutionGraph)

    # Expected ids across all nesting levels.
    assert set(graph.by_id) == {"s1", "s1.g1", "s1.g2", "s1.g3", "s2", "s2.g1", "s2.g2"}

    # Nested stage -> goal structure: stages are top-level, goals are children.
    top_level_ids = {node.id for node in graph.nodes}
    assert top_level_ids == {"s1", "s2"}
    s1 = graph.by_id["s1"]
    s2 = graph.by_id["s2"]
    assert s1.type == "stage" and s2.type == "stage"
    assert {c.id for c in s1.children} == {"s1.g1", "s1.g2", "s1.g3"}
    assert {c.id for c in s2.children} == {"s2.g1", "s2.g2"}
    for child in (*s1.children, *s2.children):
        assert child.type == "goal"

    # Cross-stage ordering is a sibling depends_on between stage nodes.
    assert s2.depends_on == ("s1",)
    # The per-stage wiring/integration Goal is a `merge` node depending on the
    # module Goals.
    merge = graph.by_id["s1.g3"]
    assert merge.kind == "merge"
    assert set(merge.depends_on) == {"s1.g1", "s1.g2"}


def _stub_cli(monkeypatch, task_data, captured):
    module = _modeling_bringup_cli_module
    monkeypatch.setattr(module, "load_and_validate_task_yaml", lambda _path: task_data)

    def _spy_build(**kwargs):
        captured["build_kwargs"] = kwargs
        return "PROMPTS"

    monkeypatch.setattr(module, "build_modeling_bringup_prompts", _spy_build)
    monkeypatch.setattr(module, "_team_main", lambda argv, **kw: captured.update(team_kwargs=kw))
    return module


def test_cli_passes_concurrent_true_into_prompt_build(tmp_path, monkeypatch):
    """`--concurrent` must reach ``build_modeling_bringup_prompts(concurrent=True)``."""
    repo = tmp_path / "trtllm"
    repo.mkdir()
    task_data = {"trtllm_repo_path": str(repo)}
    captured: dict = {}
    module = _stub_cli(monkeypatch, task_data, captured)

    module.main(["--task", "t.yaml", "--workspace", str(tmp_path / "ws"), "--concurrent"])

    assert captured["build_kwargs"].get("concurrent") is True


def test_cli_passes_concurrent_false_by_default(tmp_path, monkeypatch):
    """Without ``--concurrent`` the wrapper builds prompts with ``concurrent=False``."""
    repo = tmp_path / "trtllm"
    repo.mkdir()
    task_data = {"trtllm_repo_path": str(repo)}
    captured: dict = {}
    module = _stub_cli(monkeypatch, task_data, captured)

    module.main(["--task", "t.yaml", "--workspace", str(tmp_path / "ws")])

    assert captured["build_kwargs"].get("concurrent") is False


def test_cli_threads_max_parallel_into_prompt_build(tmp_path, monkeypatch):
    """`--max-parallel N` (and its default) must reach ``build_modeling_bringup_prompts``."""
    repo = tmp_path / "trtllm"
    repo.mkdir()
    task_data = {"trtllm_repo_path": str(repo)}

    # Explicit value flows through verbatim.
    captured: dict = {}
    module = _stub_cli(monkeypatch, task_data, captured)
    module.main(
        [
            "--task",
            "t.yaml",
            "--workspace",
            str(tmp_path / "ws"),
            "--concurrent",
            "--max-parallel",
            "4",
        ]
    )
    assert captured["build_kwargs"].get("max_parallel") == 4

    # The argparse default (8) flows through when the flag is omitted.
    captured = {}
    _stub_cli(monkeypatch, task_data, captured)
    module.main(["--task", "t.yaml", "--workspace", str(tmp_path / "ws"), "--concurrent"])
    assert captured["build_kwargs"].get("max_parallel") == 8


def test_cli_concurrent_produces_different_prompts_than_linear(tmp_path, monkeypatch):
    """End-to-end: the bundle the CLI hands `_team_main` differs under `--concurrent`.

    Uses the real prompt builder (no spy) so the CLI wiring is exercised for
    real: the concurrent coder prompt must carry the node-scoped protocol the
    linear one lacks.
    """
    repo = tmp_path / "trtllm"
    repo.mkdir()
    task_data = {"trtllm_repo_path": str(repo)}

    module = _modeling_bringup_cli_module
    monkeypatch.setattr(module, "load_and_validate_task_yaml", lambda _path: task_data)

    captured: dict = {}
    monkeypatch.setattr(module, "_team_main", lambda argv, **kw: captured.update(kw))

    argv = ["--task", "t.yaml", "--workspace", str(tmp_path / "ws")]
    module.main(argv)
    linear_coder = captured["prompts"].coder

    module.main(argv + ["--concurrent"])
    concurrent_coder = captured["prompts"].coder

    assert linear_coder != concurrent_coder
    assert "## Concurrent node protocol" in concurrent_coder
    assert "## Concurrent node protocol" not in linear_coder


class _FakeIsolation:
    """Records construction + ``clean`` so CLI --clean wiring can be asserted."""

    calls: list = []

    def __init__(self, **kwargs):
        type(self).calls.append(("init", kwargs))

    async def clean(self):
        type(self).calls.append(("clean",))


def test_cli_clean_resets_trtllm_repo_on_linear_path(tmp_path, monkeypatch):
    """`--clean` resets the trtllm checkout even on the linear path (no --concurrent)."""
    import agent_flow.git_worktree as gw

    repo = tmp_path / "trtllm"
    repo.mkdir()
    task_data = {"trtllm_repo_path": str(repo)}
    captured: dict = {}
    module = _stub_cli(monkeypatch, task_data, captured)
    _FakeIsolation.calls = []
    monkeypatch.setattr(gw, "GitWorktreeIsolation", _FakeIsolation)

    module.main(["--task", "t.yaml", "--workspace", str(tmp_path / "ws"), "--clean"])

    # The trtllm checkout was reset (worktree teardown + reset --hard/clean -fd)...
    assert ("clean",) in _FakeIsolation.calls
    # ...rooted at the validated trtllm repo path.
    init_kwargs = next(kw for tag, kw in _FakeIsolation.calls if tag == "init")
    assert str(init_kwargs["repo_path"]) == str(repo)
    # Linear path: no isolation/policy handed to the workflow.
    assert "isolation" not in captured["team_kwargs"]


def test_cli_without_clean_leaves_trtllm_repo_untouched(tmp_path, monkeypatch):
    """Without `--clean` (and without --concurrent) the trtllm checkout is not touched."""
    import agent_flow.git_worktree as gw

    repo = tmp_path / "trtllm"
    repo.mkdir()
    task_data = {"trtllm_repo_path": str(repo)}
    captured: dict = {}
    module = _stub_cli(monkeypatch, task_data, captured)
    _FakeIsolation.calls = []
    monkeypatch.setattr(gw, "GitWorktreeIsolation", _FakeIsolation)

    module.main(["--task", "t.yaml", "--workspace", str(tmp_path / "ws")])

    # Neither constructed nor cleaned on the linear no-clean path.
    assert _FakeIsolation.calls == []
