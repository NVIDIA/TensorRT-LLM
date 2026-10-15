# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--overlap-units``: the git mechanics against real repositories, then the run.

Every fallback is asserted to leave the main checkout exactly where it was:
the whole point of the opt-in is that a bad early start costs nothing but
the turn it spent.
"""

from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import overlap as overlap_mod
from agent_flow.workflows.trtllm_modeling_v2.committer import CommitIdentity, CommitPolicy
from agent_flow.workflows.trtllm_modeling_v2.overlap import UnitOverlap, native_artifacts
from agent_flow.workflows.trtllm_modeling_v2.state import load_state

from .test_delivery import TARGET_REL, _build_repo
from .test_workflow import Script, _install, _workflow, _write_vocabulary

IDENTITY = CommitIdentity(name="Repo User", email="repo.user@example.invalid")


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    ).stdout


def _init(repo):
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", IDENTITY.name)
    _git(repo, "config", "user.email", IDENTITY.email)
    _git(repo, "config", "commit.gpgsign", "false")


def _commit_all(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-s", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def main_repo(tmp_path):
    repo = tmp_path / "main"
    (repo / "tensorrt_llm").mkdir(parents=True)
    _init(repo)
    (repo / ".gitignore").write_text("tensorrt_llm/bindings/\ntensorrt_llm/*.so\n")
    (repo / "tensorrt_llm" / "__init__.py").write_text("")
    (repo / "tensorrt_llm" / "shared.py").write_text("A = 1\n")
    _commit_all(repo, "seed")
    # The native build: ignored in the main checkout.
    (repo / "tensorrt_llm" / "bindings").mkdir()
    (repo / "tensorrt_llm" / "bindings" / "__init__.pyi").write_text("")
    (repo / "tensorrt_llm" / "libnative.so").write_text("elf")
    return repo


@pytest.fixture
def overlap(tmp_path, main_repo):
    return UnitOverlap(main_repo, tmp_path / "ws", policy=CommitPolicy(identity=IDENTITY))


def _commit_in(job, rel, text):
    path = job.tree / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    result = subprocess.run(
        [str(job.commit_script), f"[None][feat] {rel}", rel], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------ mechanics


def test_the_native_build_is_found_and_linked_into_the_worktree(overlap, main_repo):
    assert native_artifacts(main_repo) == ["tensorrt_llm/bindings", "tensorrt_llm/libnative.so"]
    job = overlap.open_worktree("mtp", after="core")
    assert job.base == _git(main_repo, "rev-parse", "HEAD").strip()
    assert job.linked == ["tensorrt_llm/bindings", "tensorrt_llm/libnative.so"]
    link = job.tree / "tensorrt_llm" / "bindings"
    assert (
        link.is_symlink() and link.resolve() == (main_repo / "tensorrt_llm" / "bindings").resolve()
    )
    # The link is exactly the thing the worktree's commit helper refuses.
    refused = subprocess.run(
        [str(job.commit_script), "m", "tensorrt_llm/bindings"], capture_output=True, text=True
    )
    assert refused.returncode == 2 and "symlink git does not track" in refused.stderr
    with pytest.raises(overlap_mod.OverlapError, match="already exists"):
        overlap.open_worktree("mtp", after="core")


def test_an_early_result_on_an_unmoved_head_fast_forwards(overlap, main_repo):
    job = overlap.open_worktree("mtp", after="core")
    _commit_in(job, "tensorrt_llm/mtp.py", "MTP = 1\n")
    job.decision = "UNIT_READY"
    outcome = overlap.integrate(job)
    assert outcome.adopted, outcome.detail
    assert (main_repo / "tensorrt_llm" / "mtp.py").read_text() == "MTP = 1\n"
    assert _git(
        main_repo, "log", "-1", "--format=%an <%ae>|%(trailers:key=Signed-off-by,valueonly)"
    ).strip() == (f"{IDENTITY.render()}|{IDENTITY.render()}")
    overlap.remove(job)
    assert not job.tree.exists()
    assert job.branch not in _git(main_repo, "branch", "--list")


def test_an_early_result_rebases_onto_a_head_that_moved(overlap, main_repo):
    job = overlap.open_worktree("mtp", after="core")
    _commit_in(job, "tensorrt_llm/mtp.py", "MTP = 1\n")
    # core's later turn lands on the main checkout meanwhile.
    (main_repo / "tensorrt_llm" / "core_fix.py").write_text("FIX = 1\n")
    moved = _commit_all(main_repo, "core fix")
    job.decision = "UNIT_READY"
    outcome = overlap.integrate(job)
    assert outcome.adopted, outcome.detail
    assert _git(main_repo, "rev-parse", "HEAD~1").strip() == moved
    assert (main_repo / "tensorrt_llm" / "core_fix.py").exists()
    assert (main_repo / "tensorrt_llm" / "mtp.py").exists()


def test_a_conflict_falls_back_and_leaves_the_main_checkout_alone(overlap, main_repo):
    job = overlap.open_worktree("mtp", after="core")
    _commit_in(job, "tensorrt_llm/shared.py", "A = 'mtp'\n")
    early_tip = _git(job.tree, "rev-parse", "HEAD").strip()
    (main_repo / "tensorrt_llm" / "shared.py").write_text("A = 'core'\n")
    head = _commit_all(main_repo, "core changes the same line")
    job.decision = "UNIT_READY"
    outcome = overlap.integrate(job)
    assert not outcome.adopted and "do not rebase" in outcome.detail
    assert _git(main_repo, "rev-parse", "HEAD").strip() == head
    assert (main_repo / "tensorrt_llm" / "shared.py").read_text() == "A = 'core'\n"
    assert _git(job.tree, "rev-parse", "HEAD").strip() == early_tip  # rebase aborted
    note = overlap.keep_for_inspection(job, outcome.detail)
    assert "not adopted" in note.read_text()
    assert "A = 'mtp'" in (job.root / "early.patch").read_text()
    assert job.tree.exists()  # kept, not deleted


@pytest.mark.parametrize(
    "setup,expected",
    [
        ("blocked", "ended BLOCKED"),
        ("nothing", "committed nothing"),
        ("dirty", "uncommitted work"),
        ("error", "the early turn failed"),
    ],
)
def test_every_other_early_outcome_falls_back(overlap, main_repo, setup, expected):
    job = overlap.open_worktree("mtp", after="core")
    head = _git(main_repo, "rev-parse", "HEAD").strip()
    job.decision = "UNIT_READY"
    if setup == "blocked":
        _commit_in(job, "tensorrt_llm/mtp.py", "x\n")
        job.decision = "BLOCKED"
    elif setup == "dirty":
        _commit_in(job, "tensorrt_llm/mtp.py", "x\n")
        (job.tree / "tensorrt_llm" / "shared.py").write_text("A = 'half-written'\n")
    elif setup == "error":
        job.error = "TurnTimeout: nobody answered"
    outcome = overlap.integrate(job)
    assert not outcome.adopted and expected in outcome.detail
    assert _git(main_repo, "rev-parse", "HEAD").strip() == head


def test_the_early_turn_runs_in_a_thread_and_its_failure_is_reported(overlap):
    def boom(job, prompt):
        raise RuntimeError("the driver went away")

    overlap.turn_runner = boom
    job = overlap.open_worktree("mtp", after="core")
    overlap.start(job, "prompt")
    overlap.wait("mtp", timeout=30)
    assert job.error == "RuntimeError: the driver went away"
    assert not overlap.integrate(job).adopted


# ------------------------------------------------------------------ the run


@pytest.fixture
def run_setup(tmp_path):
    repo = _build_repo(tmp_path)
    _init(repo)
    _commit_all(repo, "the delivered trees")
    checkpoint = tmp_path / "gpt-oss-120b"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["GptOssForCausalLM"]}))
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "features": ["mtp"],
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": "/scratch/t", "CLAUDE_CODE_TMPDIR": "/scratch/t"},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
            "tolerance": 5.0,
        },
    }
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return repo, task_file


def _run(tmp_path, monkeypatch, task_file, early):
    workflow = _workflow(tmp_path, overlap_units=True)
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    monkeypatch.setattr(overlap_mod, "run_early_turn", lambda wf, job, prompt: early(job, prompt))
    with workflow:
        workflow.run(task_file)
    return workflow, calls


def test_the_next_unit_is_assembled_early_and_adopted(tmp_path, monkeypatch, run_setup):
    repo, task_file = run_setup
    seen = {}

    def early(job, prompt):
        seen["prompt"] = prompt
        _commit_in(job, f"{TARGET_REL}/mtp_head.py", "MTP = 1\n")
        return {"decision": "UNIT_READY", "summary": "mtp head, early"}

    workflow, calls = _run(tmp_path, monkeypatch, task_file, early)

    assemblers = [p for n, p in calls if n == "assembler"]
    assert len(assemblers) == 1 and "Unit: **core**" in assemblers[0]
    assert "Early start: `mtp` overlaps `core`" in seen["prompt"]
    assert "Unit: **mtp**" in seen["prompt"]
    state = load_state(tmp_path / "ws" / ".trtllm_modeling_v2_state.json")
    assert state.units_done == ["core", "mtp"]
    assert (repo / TARGET_REL / "mtp_head.py").read_text() == "MTP = 1\n"
    assert not (tmp_path / "ws" / "overlap" / "mtp" / "tree").exists()
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "overlap started" in journal and "overlap adopted" in journal
    progress = yaml.safe_load((tmp_path / "ws" / "progress.yaml").read_text())
    adopted = [
        e
        for entries in progress.values()
        if isinstance(entries, list)
        for e in entries
        if e.get("overlap")
    ]
    assert adopted and adopted[0]["unit"] == "mtp"


def test_an_early_start_that_is_not_ready_falls_back_to_sequential(
    tmp_path, monkeypatch, run_setup
):
    repo, task_file = run_setup
    head_before = _git(repo, "rev-parse", "HEAD").strip()

    def early(job, prompt):
        return {"decision": "BLOCKED", "summary": "needs core's final shape"}

    workflow, calls = _run(tmp_path, monkeypatch, task_file, early)

    assemblers = [p for n, p in calls if n == "assembler"]
    assert len(assemblers) == 2 and "Unit: **mtp**" in assemblers[1]
    state = load_state(tmp_path / "ws" / ".trtllm_modeling_v2_state.json")
    assert state.units_done == ["core", "mtp"]
    assert _git(repo, "rev-parse", "HEAD").strip() == head_before
    assert (tmp_path / "ws" / "overlap" / "mtp" / "NOT_ADOPTED.md").is_file()
    assert "overlap fell back" in (tmp_path / "ws" / "journal.md").read_text()


def test_without_the_flag_nothing_overlaps(tmp_path, monkeypatch, run_setup):
    _, task_file = run_setup
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)
    assert len([n for n, _ in calls if n == "assembler"]) == 2
    assert not (tmp_path / "ws" / "overlap").exists()


def test_the_early_turn_goes_through_the_runs_transport_as_the_assembler(
    tmp_path, overlap, main_repo
):
    """The real turn runner: own decision path, own progress, the role's backend and model."""
    from agent_flow.runtime import TurnOutcome

    seen = {}

    class Transport:
        def run(self, request):
            seen["request"] = request
            from pathlib import Path

            Path(request.decision_file).write_text("decision: UNIT_READY\nsummary: early\n")
            return TurnOutcome(role=request.role, text="done")

        def bound(self, layer):
            return self

    class Journal:
        def record_dispatch(self, role, what, prompt):
            seen["what"] = what
            return "report"

        def record_result(self, report, text):
            seen["result"] = text

    class Prompts:
        assembler = "assembler system prompt"

    class Workflow:
        task_data = {"trtllm_repo_path": str(main_repo)}
        prompts = Prompts()
        journal = Journal()
        _decision_files = True
        _agent_transport = object()
        _turns = Transport()

    job = overlap.open_worktree("mtp", after="core")
    entry = overlap_mod.run_early_turn(Workflow(), job, "the prompt")
    request = seen["request"]
    assert entry["decision"] == "UNIT_READY"
    assert request.role == "assembler" and request.backend == "claude-code"
    assert request.decision_file.startswith(str(job.root))
    assert str(job.root) in request.prompt and request.system_prompt == "assembler system prompt"
    assert seen["what"] == "mtp (early start, overlapping core)"
    # Written to the job's own progress file, never the run's.
    assert (job.root / "progress.yaml").is_file()
