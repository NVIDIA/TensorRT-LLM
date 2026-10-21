# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a kernel task's smith is told, and what becomes of a FAILED_RECEIPT's commits.

A smith was dispatched with the vocabulary row's surface and nothing of the
graph task it served, so a planner wanting a retry to start from somewhere or
a review's missing cells covered had to smuggle that into the row's note; a
task named ``<mode>:<entry>`` had no row and reached its smith with no
instruction at all. Its sourcing line said "tier 1" whenever the row named no
tier, which a verify row never does. And a FAILED_RECEIPT's guards and tests
stayed on its branch, though the skills told the smith to land them.
"""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.onboard.vocabulary import QueueItem
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.integration import Integration
from agent_flow.workflows.trtllm_modeling_v2.run.progress import decision_path
from agent_flow.workflows.trtllm_modeling_v2.run.stages.smith import entry_module, is_entry_file

from .test_integration import _ENV, _git, _repo
from .test_workflow import Script, _install, _workflow

CATALOG = "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
TARGET = "tensorrt_llm/_torch/_experimental/modeling_v2/models/newmodel/base__sm_103__unified"
TESTS = "tests/unittest/_torch/modeling_v2"


def _smith(tmp_path):
    workflow = _workflow(tmp_path)
    workflow.task_data.update({"trtllm_repo_path": "/repo", "arch": "sm_103"})
    return workflow


def _vocabulary(workflow, steps=(), queue=()) -> None:
    doc = {"verdict": "assemblable_after_additions", "steps": list(steps), "queue": list(queue)}
    workflow.vocabulary_path.write_text(yaml.safe_dump(doc), encoding="utf-8")


def _sourcing(prompt: str) -> str:
    return next(line for line in prompt.splitlines() if line.startswith("Sourcing:"))


# ------------------------------------------------------------------ the goal


def test_the_smith_is_told_the_goal_of_the_task_it_serves(tmp_path):
    workflow = _smith(tmp_path)
    item = QueueItem(n=9, mode="verify", target="ssm/chunk_gated_delta_rule", note="core blocker")
    task = g.Task(
        id="k-chunk-gdr",
        kind="kernel",
        goal="Attempt 2: build on 8cce89f6b1 and add the zero-g cells.",
        ref="9",
    )
    prompt = workflow._smith_prompt(item, task=task)
    assert "## Your task: `k-chunk-gdr` (kernel)" in prompt
    assert "build on 8cce89f6b1 and add the zero-g cells" in prompt
    assert "Its queue row's note: core blocker" in prompt
    # Under the dispatch and the skill, ahead of the surface and the sourcing.
    assert prompt.index("Load the `") < prompt.index("## Your task") < prompt.index("Sourcing:")


def test_a_planner_added_entry_reaches_its_smith_with_its_goal(tmp_path):
    """A `<mode>:<entry>` task has no row: its goal rides in the item, with or without the task."""
    workflow = _smith(tmp_path)
    goal = "Re-certify the target-owned grouped_mm under capture at every captured M."
    item = QueueItem(n=31004, mode="verify", target="ops/gemm/grouped_mm_bf16", note=goal)
    assert f"The plan's note for this dispatch: {goal}" in workflow._smith_prompt(item)

    task = g.Task(id="k-grouped-mm-graphs", kind="kernel", goal=goal, ref="verify:ops/gemm/x")
    told = workflow._smith_prompt(item, task=task)
    assert told.count(goal) == 1 and "## Your task: `k-grouped-mm-graphs`" in told


def test_a_dispatch_with_no_goal_and_no_note_says_nothing_about_either(tmp_path):
    prompt = _smith(tmp_path)._smith_prompt(QueueItem(n=1, mode="onboard", target="norm/x"))
    assert "## Your task" not in prompt
    assert "The plan's note" not in prompt and "queue row's note" not in prompt


# ------------------------------------------------------------------ the tier


def test_a_verify_row_is_told_the_tier_the_vocabulary_records_for_its_entry(tmp_path):
    workflow = _smith(tmp_path)
    _vocabulary(
        workflow,
        steps=[
            {
                "step": "linear attention prefill",
                "state": "GAP",
                "tier": 2,
                "entry": f"{CATALOG}/ssm/chunk_gated_delta_rule.py",
            }
        ],
        queue=[{"n": 9, "mode": "verify", "target": "ssm/chunk_gated_delta_rule", "surface": "s"}],
    )
    item = QueueItem(n=9, mode="verify", target="ssm/chunk_gated_delta_rule", surface="s")
    assert item.tier == 1  # what the parsed row reads as: no record of anything

    line = _sourcing(workflow._smith_prompt(item))
    assert "**tier 2 — " in line and "the `vocabulary.yaml` step that maps onto it" in line


def test_the_tier_a_landed_smith_recorded_outranks_the_plan(tmp_path):
    """The survey planned a FlashInfer call; the onboarding smith wrapped native torch instead."""
    workflow = _smith(tmp_path)
    _vocabulary(
        workflow,
        queue=[{"n": 15, "mode": "onboard", "target": "ops/gemm/grouped_mm_bf16", "tier": 3}],
    )
    path = decision_path(workflow.workspace, "smith", 15)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "decision": "LANDED",
                "summary": "wrapped torch._grouped_mm",
                "entry": f"{TARGET}/ops/gemm/grouped_mm_bf16.py",
                "tier": 4,
            }
        ),
        encoding="utf-8",
    )
    item = QueueItem(n=31004, mode="verify", target="ops/gemm/grouped_mm_bf16")

    line = _sourcing(workflow._smith_prompt(item))
    assert "**tier 4 — " in line and f"decisions/{path.name}" in line


def test_a_macro_constituents_tier_reaches_its_onboarding(tmp_path):
    workflow = _smith(tmp_path)
    path = decision_path(workflow.workspace, "smith", 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "decision": "MACRO",
                "summary": "two launches",
                "constituents": [{"target": "quantization/dequantize_e4m3", "tier": 5}],
            }
        ),
        encoding="utf-8",
    )
    item = QueueItem(n=10512, mode="onboard", target="quantization/dequantize_e4m3")
    prompt = workflow._smith_prompt(item)
    assert "**tier 5 — " in _sourcing(prompt)
    assert "you are writing it, not wrapping one" in prompt


def test_an_onboard_row_is_dispatched_at_the_tier_it_names(tmp_path):
    workflow = _smith(tmp_path)
    _vocabulary(
        workflow, queue=[{"n": 22, "mode": "onboard", "target": "ops/ssm/linear_attn", "tier": 5}]
    )
    prompt = workflow._smith_prompt(
        QueueItem(n=22, mode="onboard", target="ops/ssm/linear_attn", tier=5)
    )
    assert "`vocabulary.yaml` queue row 22" in _sourcing(prompt)
    assert "you are writing it, not wrapping one" in prompt


def test_a_verify_of_a_written_entry_is_not_told_it_has_no_implementation(tmp_path):
    workflow = _smith(tmp_path)
    _vocabulary(
        workflow, queue=[{"n": 22, "mode": "onboard", "target": "ops/ssm/linear_attn", "tier": 5}]
    )
    prompt = workflow._smith_prompt(QueueItem(n=40422, mode="verify", target="ops/ssm/linear_attn"))
    assert "**tier 5 — " in _sourcing(prompt)
    assert "no implementation yet" not in prompt


def test_with_nothing_on_record_no_tier_is_claimed(tmp_path):
    prompt = _smith(tmp_path)._smith_prompt(QueueItem(n=3, mode="verify", target="attention/thop"))
    line = _sourcing(prompt)
    assert "no rung is recorded" in line and "tier 1" not in line
    assert "`tier`" in line  # and the smith is asked to record it


# ------------------------------------------------------------- the entry's files


def test_the_files_an_entry_spans():
    shared, own = "ssm/chunk_gated_delta_rule", "ops/gemm/grouped_mm_bf16"
    assert entry_module(shared) == "catalog/ssm/chunk_gated_delta_rule.py"
    assert entry_module(own) == "ops/gemm/grouped_mm_bf16.py"
    assert is_entry_file(f"{CATALOG}/ssm/chunk_gated_delta_rule.py", shared)
    assert is_entry_file(f"{TESTS}/ssm/test_modeling_v2_chunk_gated_delta_rule.py", shared)
    assert is_entry_file(f"{TARGET}/ops/gemm/grouped_mm_bf16.py", own)
    assert is_entry_file(f"{TARGET}/ops/gemm/__init__.py", own)
    assert is_entry_file(
        f"{TESTS}/models/newmodel/base__sm_103__unified/gemm/test_modeling_v2_grouped_mm_bf16.py",
        own,
    )
    assert is_entry_file(f"{TESTS}/comm/_allreduce_op_matrix.py", "comm/allreduce")
    assert is_entry_file(f"{TESTS}/comm/test_modeling_v2_allreduce_op_matrix.py", "comm/allreduce")

    for outside in (
        "tensorrt_llm/_torch/modules/fla/op.py",  # the engine
        f"{TARGET}/modeling.py",  # the target
        f"{CATALOG}/_op.py",
        f"{CATALOG}/ssm/__init__.py",  # the shared catalog's packages are not an entry's
        f"{CATALOG}/ssm/chunk_gated_delta_rule_other.py",
        f"{TESTS}/_validating.py",
        f"{TESTS}/norm/test_modeling_v2_chunk_gated_delta_rule.py",  # another category
        "tests/unittest/pytest.ini",
    ):
        assert not is_entry_file(outside, shared), outside


# ------------------------------------------------------- a FAILED_RECEIPT lands


def _receipt_run(tmp_path, monkeypatch, files: dict[str, str], decision="FAILED_RECEIPT"):
    """A smith that commits ``files`` in its task's worktree and ends ``decision``."""
    repo = _repo(tmp_path)
    workflow = _workflow(tmp_path)
    workflow._integration = Integration(repo, workflow.workspace, "mv2/run", _ENV)
    tree = workflow._integration.open("k-chunk-gdr")
    head = _git(repo, "rev-parse", "HEAD")

    def smith(turn, prompt):
        for name, text in files.items():
            (tree / name).parent.mkdir(parents=True, exist_ok=True)
            (tree / name).write_text(text)
        _git(tree, "add", *files)
        message = "chunk_gated_delta_rule: zero-g refused\n\nSigned-off-by: T <t@example.com>"
        _git(tree, "commit", "-q", "-m", message)
        return {"decision": decision, "summary": "zero g fails", "entry": next(iter(files))}

    _install(monkeypatch, workflow, Script(None, smith=smith))
    monkeypatch.setattr(
        type(workflow), "_smith_prompt", lambda self, item, note="", task=None: "Dispatch 9"
    )
    item = QueueItem(n=9, mode="verify", target="ssm/chunk_gated_delta_rule")
    task = g.Task(id="k-chunk-gdr", kind="kernel", goal="verify at zero g", ref="9")
    workflow._run_one_smith(item, task=task)
    return workflow, repo, tree, head, item


def test_a_failed_receipt_of_the_entrys_own_files_is_integrated(tmp_path, monkeypatch):
    files = {
        f"{CATALOG}/ssm/chunk_gated_delta_rule.py": "class Entry: ...\n",
        f"{TESTS}/ssm/test_modeling_v2_chunk_gated_delta_rule.py": "def test_guard(): ...\n",
    }
    workflow, repo, tree, head, item = _receipt_run(tmp_path, monkeypatch, files)

    assert _git(repo, "log", "-1", "--format=%s") == "chunk_gated_delta_rule: zero-g refused"
    assert _git(repo, "rev-parse", "HEAD~1") == head
    assert not tree.exists()  # closed: the next attempt opens on the branch, at the head
    decision = workflow._smith_decision_for(item)
    head_line, _, note = decision.partition("\n")
    assert head_line == "FAILED_RECEIPT"
    assert "1 commit(s), the entry's own files only, are integrated" in note
    journal = workflow.journal_path.read_text(encoding="utf-8")
    assert "k-chunk-gdr: FAILED_RECEIPT, its 1 commit(s)" in journal


def test_a_failed_receipt_that_reaches_outside_the_entry_stays_on_its_branch(tmp_path, monkeypatch):
    files = {
        f"{CATALOG}/ssm/chunk_gated_delta_rule.py": "class Entry: ...\n",
        "tensorrt_llm/_torch/modules/fla/op.py": "def safe_exp(x): ...\n",
    }
    workflow, repo, tree, head, item = _receipt_run(tmp_path, monkeypatch, files)

    assert _git(repo, "rev-parse", "HEAD") == head
    assert tree.exists()
    note = workflow._smith_decision_for(item).partition("\n")[2]
    assert "stay on mv2/run/k-chunk-gdr" in note and "modules/fla/op.py" in note


def test_a_landed_entry_is_left_to_the_tasks_own_integration(tmp_path, monkeypatch):
    files = {f"{CATALOG}/ssm/chunk_gated_delta_rule.py": "class Entry: ...\n"}
    workflow, repo, tree, head, item = _receipt_run(tmp_path, monkeypatch, files, "LANDED")

    assert _git(repo, "rev-parse", "HEAD") == head
    assert workflow._smith_decision_for(item) == "LANDED"


def test_without_its_task_a_failed_receipt_is_left_as_before(tmp_path, monkeypatch):
    """A caller that passes no task gets the decision alone, and nothing is integrated."""
    repo = _repo(tmp_path)
    workflow = _workflow(tmp_path)
    workflow._integration = Integration(repo, workflow.workspace, "mv2/run", _ENV)
    _install(
        monkeypatch,
        workflow,
        Script(None, smith=lambda turn, prompt: {"decision": "FAILED_RECEIPT", "summary": "s"}),
    )
    monkeypatch.setattr(type(workflow), "_smith_prompt", lambda self, item, note="", task=None: "D")
    item = QueueItem(n=9, mode="verify", target="ssm/chunk_gated_delta_rule")
    workflow._run_one_smith(item)
    assert workflow._smith_decision_for(item) == "FAILED_RECEIPT"


# ------------------------------------------------------------- the role's rules


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_smith_prompt_carries_the_test_budget():
    """Certification grew the modeling_v2 suite from 8 minutes to an hour in one run."""
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS

    smith = _flat(DEFAULT_PROMPTS.smith)
    assert "A new cell fits **10 s and 40 GiB** of device memory." in smith
    assert "is marked `large`, which pre-merge deselects" in smith
    assert "The entry's whole test file and the claims gates run **once, at the end**" in smith
    assert "Cells one process can drive are one parametrized test" in smith


def test_the_smith_prompt_says_which_failed_receipts_land_and_asks_for_the_tier():
    from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import DEFAULT_PROMPTS

    smith = _flat(DEFAULT_PROMPTS.smith)
    assert "Commits that change only the entry's own files are integrated as they stand" in smith
    assert "A `FAILED_RECEIPT`'s own files are the entry or its local variant" in smith
    assert "`entry` (its path) and `tier`" in smith
