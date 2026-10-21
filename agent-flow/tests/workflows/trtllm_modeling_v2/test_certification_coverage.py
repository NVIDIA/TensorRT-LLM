# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A certification whose cells the tree already certifies is closed without a smith turn.

A re-certification planned from receipts that predated a running task's
result reached its smith after that task had landed exactly its cells, and
paid a whole turn to find so. These tests hold the dispatch to reading the
entry's ``CELLS`` and ``ARCHS`` at the checkout's ``HEAD``, closing the task
``LANDED`` with a receipt naming the covering cells when every required cell
is there, and dispatching the smith whenever one is missing, the
architecture is not certified, the source cannot be read, or the task's
branch carries work of its own.
"""

from __future__ import annotations

import textwrap

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import testcache as tc
from agent_flow.workflows.trtllm_modeling_v2.onboard import coverage as cov
from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run import covered as covered_mod
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.progress import current_decision

from .test_workflow import Script, _install, _workflow

CATALOG = "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
TESTS = "tests/unittest/_torch/modeling_v2"
ENTRY = f"{CATALOG}/attention/verify_attention.py"
TEST = f"{TESTS}/attention/test_modeling_v2_verify_attention.py"

SOURCE = textwrap.dedent(
    """
    import torch
    from .._op import Arch, Cell, OpWrapper
    from ..consts import PAGED

    _BASE = dict(heads=8, head_dim=128, dtype=torch.bfloat16, layout=PAGED)
    DEPTHS = range(1, 8)


    class VerifyAttention(OpWrapper):
        ARCHS = frozenset({Arch.SM_103})
        CELLS: tuple[Cell, ...] = (
            Cell(why="decode", spec=dict(**_BASE, q_len=1, captured=True)),
            *(
                Cell(why=f"verify at depth {d}", spec=dict(**_BASE, q_len=1 + d, captured=True))
                for d in DEPTHS
            ),
            Cell(why="eager prefill", spec=dict(_BASE, q_len=2048, captured=False, extra=helper())),
        )

        def __call__(self, q):
            return q

        def reference(self, q):
            return q
    """
)


# ------------------------------------------------------------- reading


def test_the_cells_are_read_from_the_source_comprehensions_and_constants_included():
    entry = cov.read_entry_cells(SOURCE)

    assert entry.problem == "" and entry.archs == ("sm_103",)
    assert len(entry.cells) == 9
    assert entry.cells[3].why == "verify at depth 3"
    assert cov.covers({"q_len": 4, "captured": True, "dtype": "bfloat16"}, entry.cells[3])
    assert cov.covers({"layout": "PAGED"}, entry.cells[0]), "an imported name is a symbol"


def test_a_value_that_cannot_be_read_matches_nothing_and_leaves_the_rest_of_its_cell():
    prefill = cov.read_entry_cells(SOURCE).cells[-1]
    assert prefill.spec is not None
    assert cov.covers({"q_len": 2048}, prefill)
    assert not cov.covers({"extra": "anything"}, prefill)


def test_cells_built_by_a_call_are_not_read():
    source = SOURCE.replace("CELLS: tuple[Cell, ...] = (", "CELLS = _cells() or (")
    entry = cov.read_entry_cells(source)
    assert entry.problem.startswith("CELLS cannot be read")
    assert not cov.coverage([{"q_len": 1}], entry, "sm_103").covered


def test_values_compare_as_written():
    assert cov.same(4, 4.0) and not cov.same(True, 1) and not cov.same(1, True)
    assert cov.same("bfloat16", cov.Symbol("torch.bfloat16"))
    assert not cov.same("torch.float16", cov.Symbol("torch.bfloat16"))
    assert not cov.same("x.bfloat16", cov.Symbol("torch.bfloat16")), "a dotted name is exact"
    assert cov.same([1, 2], (1, 2)) and not cov.same([1, 2], [2, 1])
    assert not cov.same(None, 0)


# ------------------------------------------------------------- requirements


def test_required_cells_come_from_the_row_then_the_goals_fenced_block():
    goal = "Certify the verify path.\n\n```cells\n- {q_len: 4, captured: true}\n```\n"
    assert cov.required_cells({"cells": [{"q_len": 2}]}, goal)[0] == [{"q_len": 2}]
    assert cov.required_cells({"n": 3}, goal)[0] == [{"q_len": 4, "captured": True}]
    assert cov.required_cells(None, "no block here")[0] == []
    assert cov.required_cells({"cells": ["q_len 4"]}, "")[0] == [], "malformed states nothing"
    assert cov.required_cells(None, "```cells\n- {}\n```")[0] == []


def test_coverage_needs_every_cell_and_the_runs_architecture():
    entry = cov.read_entry_cells(SOURCE)
    depths = [{"q_len": 1 + d, "captured": True} for d in range(1, 8)]

    assert cov.coverage(depths, entry, "sm_103").covered
    missing = cov.coverage([*depths, {"q_len": 9, "captured": True}], entry, "sm_103")
    assert not missing.covered and missing.missing == ({"q_len": 9, "captured": True},)
    other = cov.coverage(depths, entry, "sm_100")
    assert not other.covered and "not sm_100" in other.reason


# ------------------------------------------------------------- the dispatch


def _git(files: dict[str, str], head: str = "c0ffee1234567890"):
    def run(repo, *args, stdin=None):
        if args == ("rev-parse", "HEAD"):
            return head + "\n"
        if args[:1] == ("show",):
            return files.get(args[1].split(":", 1)[1])
        if args[:2] == ("cat-file", "-e"):
            return "" if args[2].split(":", 1)[1] in files else None
        if args[:2] == ("log", "-1") or args[:1] == ("rev-parse",):
            return head + "\n"
        if args[:2] == ("status", "--porcelain"):
            return ""
        return ""

    return run


def _smith(tmp_path, monkeypatch, files, *, row_cells=None, tested="CLEAN"):
    results = {
        "CLEAN": tc.HeadResults("c0ffee1234567890", ["t::a"], [], [], []),
        "INCOMPLETE": tc.HeadResults("c0ffee1234567890", [], [], [], ["t::a"]),
    }[tested]
    monkeypatch.setattr(covered_mod, "_tested", lambda stage, repo, head, test: results)
    workflow = _workflow(tmp_path, git_runner=_git(files))
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    row = {"n": 7, "mode": "verify", "target": "attention/verify_attention", "surface": "verify"}
    if row_cells is not None:
        row["cells"] = row_cells
    doc = {"verdict": "assemblable_after_additions", "steps": [], "queue": [row]}
    workflow.vocabulary_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    calls = _install(monkeypatch, workflow, Script(None))
    workflow._decision_files = True
    monkeypatch.setattr(
        type(workflow), "_smith_prompt", lambda self, item, note="", task=None: "Dispatch 7"
    )
    return workflow, calls


_ITEM = QueueItem(n=7, mode="verify", target="attention/verify_attention")
_DEPTHS = [{"q_len": 1 + d, "captured": True} for d in range(1, 8)]


def test_a_covered_certification_closes_landed_without_a_turn(tmp_path, monkeypatch):
    workflow, calls = _smith(tmp_path, monkeypatch, {ENTRY: SOURCE, TEST: "x"}, row_cells=_DEPTHS)
    task = g.Task(id="k-verify-depths", kind="kernel", goal="Re-certify the depths.", ref="7")

    result = workflow._run_one_smith(_ITEM, task=task)

    assert not [c for c in calls if c[0] == "smith"], "no smith turn"
    assert result.startswith("covered at c0ffee1234")
    assert workflow._smith_decision_for(_ITEM) == "LANDED"
    decision = yaml.safe_load(current_decision(workflow.workspace, "smith", 7).read_text())
    assert decision["entry"] == ENTRY
    assert decision["covered"]["tree"] == "c0ffee1234567890"
    assert [c["cell"] for c in decision["covered"]["cells"]] == list(range(1, 8))
    assert "verify at depth 1" in decision["summary"]
    events = workflow.journal_path.read_text()
    assert "certification closed as covered" in events


def test_cells_whose_test_has_not_passed_at_head_dispatch_the_smith(tmp_path, monkeypatch):
    """A cell in CELLS is a claim; the entry's test passing on this tree is the certification."""
    workflow, calls = _smith(
        tmp_path, monkeypatch, {ENTRY: SOURCE, TEST: "x"}, row_cells=_DEPTHS, tested="INCOMPLETE"
    )
    workflow._run_one_smith(_ITEM)
    assert [c for c in calls if c[0] == "smith"]
    assert "in the run's test cache: INCOMPLETE" in workflow.journal_path.read_text()


def test_one_missing_cell_dispatches_the_smith(tmp_path, monkeypatch):
    """The cheaper path still fails on what it guards: a cell the entry lacks is certified."""
    workflow, calls = _smith(
        tmp_path,
        monkeypatch,
        {ENTRY: SOURCE, TEST: "x"},
        row_cells=[*_DEPTHS, {"q_len": 9, "captured": True}],
    )
    workflow._run_one_smith(_ITEM)
    assert [c for c in calls if c[0] == "smith"]
    assert "1 of 8 required cell(s) are in no cell of CELLS" in workflow.journal_path.read_text()


def test_the_goals_block_states_a_planner_added_certification(tmp_path, monkeypatch):
    workflow, calls = _smith(tmp_path, monkeypatch, {ENTRY: SOURCE, TEST: "x"})
    goal = "Re-certify under capture.\n```cells\n" + yaml.safe_dump(_DEPTHS[:2]) + "```\n"
    item = QueueItem(n=31007, mode="verify", target="attention/verify_attention", note=goal)
    task = g.Task(id="k-capture", kind="kernel", goal=goal, ref="verify:attention/verify_attention")

    workflow._run_one_smith(item, task=task)

    assert not [c for c in calls if c[0] == "smith"]


def test_an_entry_or_test_absent_at_head_dispatches_the_smith(tmp_path, monkeypatch):
    workflow, calls = _smith(tmp_path, monkeypatch, {ENTRY: SOURCE}, row_cells=_DEPTHS)
    workflow._run_one_smith(_ITEM)
    assert [c for c in calls if c[0] == "smith"]
    assert "the test that drives its cells, is not at the checkout's HEAD" in (
        workflow.journal_path.read_text()
    )


def test_an_uncertified_architecture_dispatches_the_smith(tmp_path, monkeypatch):
    workflow, calls = _smith(tmp_path, monkeypatch, {ENTRY: SOURCE, TEST: "x"}, row_cells=_DEPTHS)
    workflow.task_data["arch"] = "sm_100"
    workflow._run_one_smith(_ITEM)
    assert [c for c in calls if c[0] == "smith"]


def test_a_branch_with_work_of_its_own_is_not_closed(tmp_path, monkeypatch):
    workflow, calls = _smith(tmp_path, monkeypatch, {ENTRY: SOURCE, TEST: "x"}, row_cells=_DEPTHS)

    class _Integration:
        def changed_paths(self, task_id):
            return (ENTRY,)

    workflow._integration = _Integration()
    task = g.Task(id="k-verify-depths", kind="kernel", goal="", ref="7")
    workflow._run_one_smith(_ITEM, "rebase onto the head", task=task)
    assert [c for c in calls if c[0] == "smith"]


def test_an_onboarding_or_a_row_without_cells_is_dispatched_as_before(tmp_path, monkeypatch):
    workflow, calls = _smith(tmp_path, monkeypatch, {ENTRY: SOURCE, TEST: "x"})
    workflow._run_one_smith(_ITEM)
    workflow._run_one_smith(QueueItem(n=8, mode="onboard", target="attention/verify_attention"))
    assert len([c for c in calls if c[0] == "smith"]) == 2
    assert "certification" not in workflow.journal_path.read_text()
