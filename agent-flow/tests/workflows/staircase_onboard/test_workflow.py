# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the orchestrator.

No agent is ever constructed: ``_make_agent`` is replaced by a fake that
appends the scripted progress entry the real agent would have written. That
is the whole contract between the orchestrator and its agents — a typed
decision in ``progress.yaml`` — so a fake that honours it exercises the real
control flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import progress as progress_mod
from agent_flow.workflows.staircase_onboard import vocabulary as vocab
from agent_flow.workflows.staircase_onboard.gates import ShellResult
from agent_flow.workflows.staircase_onboard.preflight import CommandResult
from agent_flow.workflows.staircase_onboard.state import (
    STAGE_ASSEMBLE,
    STAGE_DELIVER,
    STAGE_PLAN,
    STAGE_PREFLIGHT,
    STAGE_SMITH,
    load_state,
)
from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

from .test_delivery import TARGET_REL, _build_repo

_SCRATCH = "/scratch/staircase-run-tmp"


# --------------------------------------------------------------------- fakes


class FakeLayer:
    """Stands in for one ``AgentLayer``. Writes what the real role would write."""

    def __init__(self, name, script, calls):
        self.name = name
        self.script = script
        self.calls = calls

    def __call__(self, prompt: str) -> str:
        self.calls.append((self.name, prompt))
        turn = sum(1 for n, _ in self.calls if n == self.name)
        entry = self.script(self.name, turn, prompt)
        if entry is not None:
            self.script.ctx.append(self.name, {"agent": self.name, **entry})
        return f"{self.name} turn {turn} done"

    def __exit__(self, *args) -> None:
        return None


class Script:
    """Per-role scripted decisions, keyed by role and turn number."""

    def __init__(self, ctx, plan=("DRAFT_READY",), review=("APPROVE",), **roles):
        self.ctx = ctx
        self.plan = list(plan)
        self.review = list(review)
        self.roles = roles

    def __call__(self, role, turn, prompt):
        if role == "plan_drafter":
            return {"decision": self.plan[min(turn, len(self.plan)) - 1], "summary": "drafted"}
        if role == "plan_reviewer":
            return {"decision": self.review[min(turn, len(self.review)) - 1], "summary": "reviewed"}
        override = self.roles.get(role)
        if callable(override):
            return override(turn, prompt)
        if role == "smith":
            target = _target_from(prompt)
            return {"decision": "LANDED", "summary": f"landed {target}", "entry": target}
        if role == "assembler":
            return {"decision": "UNIT_READY", "summary": "assembled"}
        if role == "reviewer":
            return {"decision": "APPROVE", "summary": "looks right"}
        if role == "auditor":
            return {"decision": "CLEAN", "summary": "8 values, 3 entries"}
        raise AssertionError(f"unscripted role: {role}")


def _target_from(prompt: str) -> str:
    for token in prompt.split("`"):
        if token and " " not in token and not token.startswith("/"):
            return token
    return "unknown"


def _install(monkeypatch, workflow, script):
    calls: list[tuple[str, str]] = []
    script.ctx = workflow._progress_ctx

    def fake_make_agent(self, name, system_prompt, *, required_tools, session_mode="stateless"):
        return FakeLayer(name, script, calls)

    monkeypatch.setattr(StaircaseOnboardWorkflow, "_make_agent", fake_make_agent)
    return calls


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def repo(tmp_path):
    return _build_repo(tmp_path)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "gpt-oss-120b"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"architectures": ["GptOssForCausalLM"]}))
    return path


@pytest.fixture
def task_file(tmp_path, repo, checkpoint):
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": _SCRATCH},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
            "tolerance": 5.0,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def git_probe(command: str, *, head: str = "c0ffee1", dirt: str = "") -> ShellResult | None:
    """Answer the two git questions the resume check asks, or ``None``.

    `StaircaseOnboardWorkflow._target_marker` asks which commit last touched
    the target and whether anything is uncommitted under it, through the same
    shell runner the gates use. Every fake here therefore has to answer them,
    and — this is the part worth stating — **none of them may count one as a
    boot**: a probe is not a gate, and a counter that conflates the two turns
    a cheap question into a phantom engine launch.
    """
    if "git -C" not in command:
        return None
    if "log -1" in command:
        return ShellResult(returncode=0, stdout=head + "\n")
    if "status --porcelain" in command:
        return ShellResult(returncode=0, stdout=dirt)
    return ShellResult(returncode=0, stdout="")


def healthy_smi(count: int = 1):
    rows = "\n".join(f"{i}, 180000, 183359, 0, 10.3" for i in range(count))

    def runner(argv, env=None, timeout=0):
        if argv and argv[0] == "nvidia-smi":
            return CommandResult(returncode=0, stdout=rows + "\n")
        return CommandResult(returncode=0, stdout="ok\n")

    return runner


def passing_shell(score: float = 90.0, accepted: int = 320, iterations: int = 100, dirt: str = ""):
    def runner(command, env=None, timeout=0):
        probe = git_probe(command, dirt=dirt)
        if probe is not None:
            return probe
        if "trtllm-eval" in command:
            return ShellResult(returncode=0, stdout=f"lm-eval gsm8k average accuracy: {score}\n")
        # The acceptance gate runs the QUICKSTART command, because nothing in
        # tree makes `trtllm-eval` report a speculative statistic. The fake
        # has to print what the real script prints, or every speculator unit
        # is unpassable for a reason that looks like a broken draft path.
        return ShellResult(
            returncode=0,
            stdout=(
                "Paris is the capital of France\n"
                f"[0] Accept rate: {accepted / iterations:.2f} "
                f"(tokens={accepted}, iterations={iterations})\n"
            ),
        )

    return runner


def _workflow(tmp_path, **kwargs):
    kwargs.setdefault("command_runner", healthy_smi())
    kwargs.setdefault("shell_runner", passing_shell())
    # These tests drive stages other than preflight, and several use a shell
    # runner scripted to fail a specific boot. The substrate gate issues a
    # boot-shaped command of its own, so it would consume those scripts.
    # It has dedicated coverage in test_substrate_gate.py, plus the two
    # end-to-end cases below.
    kwargs.setdefault("skip_substrate_gate", True)
    return StaircaseOnboardWorkflow(workspace=tmp_path / "ws", **kwargs)


def _write_vocabulary(workspace: Path, queue):
    doc = {
        "verdict": vocab.VERDICT_AFTER_ADDITIONS if queue else vocab.VERDICT_ASSEMBLABLE,
        "steps": [
            {
                "step": "norm",
                "state": vocab.STATE_OK,
                "entry": "catalog/norm/flashinfer_rmsnorm.py",
                "contract_line": "certified sm_103 bf16",
            }
        ],
        "queue": queue,
    }
    (workspace / "vocabulary.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


# ------------------------------------------------------------------- preflight


def test_preflight_failure_ends_the_run_before_any_agent(tmp_path, task_file, monkeypatch):
    def broken(argv, env=None, timeout=0):
        return CommandResult(returncode=0, stdout="0, 20000, 183359, 99, 10.3\n")

    workflow = _workflow(tmp_path, command_runner=broken)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert calls == []
    assert "preflight failed" in (tmp_path / "ws" / "journal.md").read_text()


def test_dry_run_stops_before_the_first_agent_turn(tmp_path, task_file, monkeypatch):
    """And it leaves the checkpoint at preflight, so the real run re-runs it.

    A device claim is a property of the host *at that moment*. Letting a dry
    run mark preflight complete would let a real run hours later start on a
    claim that has since gone stale — which is the one thing preflight exists
    to prevent.
    """
    workflow = _workflow(tmp_path, dry_run=True)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert calls == []
    assert load_state(tmp_path / "ws" / ".staircase_onboard_state.json").stage == STAGE_PREFLIGHT


def test_preflight_resolves_the_target_path(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path, dry_run=True)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    assert workflow._preflight.target_rel == TARGET_REL


# ------------------------------------------------------------------ plan phase


def test_happy_path_walks_every_stage(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    script = Script(None)
    calls = _install(monkeypatch, workflow, script)

    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    roles = [name for name, _ in calls]
    assert roles == ["plan_drafter", "plan_reviewer", "assembler", "reviewer", "auditor"]
    assert load_state(tmp_path / "ws" / ".staircase_onboard_state.json").done


def test_a_ceiling_ends_the_run(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, plan=("CEILING",)))
    with workflow:
        workflow.run(task_file)
    assert [n for n, _ in calls] == ["plan_drafter"]
    assert "vocabulary ceiling" in (tmp_path / "ws" / "journal.md").read_text()


def test_plan_cap_records_open_risks_and_starts_the_build(tmp_path, task_file, monkeypatch):
    """Reaching the cap is a budget, not a gate: the build starts anyway."""
    workflow = _workflow(tmp_path, max_plan_rounds=1)
    script = Script(None, plan=("DRAFT_READY", "DRAFT_READY"), review=("REJECT", "REJECT"))
    calls = _install(monkeypatch, workflow, script)

    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    plan_text = (tmp_path / "ws" / "plan.md").read_text()
    assert "## Open risks" in plan_text
    assert "reviewed" in plan_text
    # Two plan rounds, then the build ran anyway.
    assert [n for n, _ in calls].count("plan_drafter") == 2
    assert "assembler" in [n for n, _ in calls]


def test_a_supplied_plan_and_vocabulary_skip_the_plan_phase(tmp_path, task_file, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    plan = tmp_path / "plan.md"
    plan.write_text("# supplied\n", encoding="utf-8")
    vocabulary = tmp_path / "vocab.yaml"
    vocabulary.write_text(
        yaml.safe_dump({"verdict": vocab.VERDICT_ASSEMBLABLE, "steps": [], "queue": []}),
        encoding="utf-8",
    )
    workflow = _workflow(tmp_path, plan=plan, vocabulary=vocabulary)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)
    roles = [n for n, _ in calls]
    assert "plan_drafter" not in roles and "plan_reviewer" not in roles
    assert (workspace / "plan.md").read_text() == "# supplied\n"


# ----------------------------------------------------------------- smith phase


def test_independent_queue_items_run_in_one_batch(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    script = Script(None)
    calls = _install(monkeypatch, workflow, script)

    queue = [
        {"n": 1, "mode": "onboard", "target": "moe_gelu", "depends_on": []},
        {"n": 2, "mode": "onboard", "target": "rms_norm_gated", "depends_on": []},
        {
            "n": 3,
            "mode": "verify",
            "target": "thop_attention",
            "depends_on": [1],
            "surface": "h=72",
        },
    ]
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", queue)
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    smith_calls = [p for n, p in calls if n == "smith"]
    assert len(smith_calls) == 3
    state = load_state(tmp_path / "ws" / ".staircase_onboard_state.json")
    assert all(row["status"] == "done" for row in state.queue)


def test_a_smith_that_cannot_land_stops_the_run(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    script = Script(
        None,
        smith=lambda turn, prompt: {
            "decision": "BLOCKED",
            "summary": "blocked on moe_gelu",
            "entry": "moe_gelu",
        },
    )
    calls = _install(monkeypatch, workflow, script)

    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [{"n": 1, "mode": "onboard", "target": "moe_gelu"}])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    assert "assembler" not in [n for n, _ in calls]


# -------------------------------------------------------------- assemble phase


def test_a_failing_boot_gate_loops_back_to_the_assembler(tmp_path, task_file, monkeypatch):
    attempts = {"boot": 0}

    def flaky(command, env=None, timeout=0):
        probe = git_probe(command)
        if probe is not None:
            return probe
        if "trtllm-eval" in command:
            return ShellResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 90.0\n")
        if "__SURVEY_JSON__" in command:
            # The survey probe is a shell command like any other and must not
            # be miscounted as a boot attempt. It is left failing on purpose:
            # a run whose survey does not land still has to complete.
            return ShellResult(returncode=1, stderr="no engine here")
        attempts["boot"] += 1
        if attempts["boot"] == 1:
            return ShellResult(returncode=1, stderr="CUDA error")
        return ShellResult(returncode=0, stdout="Paris\n")

    workflow = _workflow(tmp_path, shell_runner=flaky)
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    assert [n for n, _ in calls].count("assembler") == 2


def test_a_failing_accuracy_gate_is_not_papered_over(tmp_path, task_file, monkeypatch):
    """The gate is the release criterion; exhausting the budget ends the run."""
    workflow = _workflow(
        tmp_path, shell_runner=passing_shell(score=10.0), max_assembler_turns_per_unit=2
    )
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    assert [n for n, _ in calls].count("assembler") == 2
    assert "auditor" not in [n for n, _ in calls]
    assert "assembly-exhausted" in (tmp_path / "ws" / "journal.md").read_text()


def test_a_vocabulary_gap_dispatches_smith_and_retries(tmp_path, task_file, monkeypatch):
    state_box = {"blocked_once": False}

    def assembler(turn, prompt):
        if not state_box["blocked_once"]:
            state_box["blocked_once"] = True
            return {
                "decision": "BLOCKED",
                "summary": "need moe_gelu",
                "blocked_reason": "vocabulary_gap",
                "missing_entry": "moe_gelu",
            }
        return {"decision": "UNIT_READY", "summary": "assembled"}

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, assembler=assembler))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    roles = [n for n, _ in calls]
    assert roles.count("smith") == 1
    assert roles.count("assembler") == 2
    assert "auditor" in roles


def test_a_knowledge_gap_ends_the_run_with_the_question(tmp_path, task_file, monkeypatch):
    def assembler(turn, prompt):
        return {
            "decision": "BLOCKED",
            "summary": "which metadata field carries the sink offset?",
            "blocked_reason": "knowledge_gap",
        }

    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None, assembler=assembler))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "knowledge_gap" in journal
    assert "sink offset" in journal
    assert "auditor" not in [n for n, _ in calls]


def test_reviewer_off_skips_the_review_turn(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path, reviewer_mode="off")
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)
    assert "reviewer" not in [n for n, _ in calls]


def test_features_are_assembled_after_core_one_at_a_time(tmp_path, task_file, monkeypatch):
    spec = yaml.safe_load(task_file.read_text())
    spec["features"] = ["mtp"]
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")

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

    units = [p for n, p in calls if n == "assembler"]
    assert "Unit: **core**" in units[0]
    assert "Unit: **mtp**" in units[1]
    state = load_state(tmp_path / "ws" / ".staircase_onboard_state.json")
    assert state.units_done == ["core", "mtp"]


# ----------------------------------------------------- the member set, gated


def _unified_task(task_file, members):
    """Rewrite the task spec as a unified run over `members`.

    World size 1 so the harness stays the one every other test here uses: the
    device claim sees one card and the multi-rank substrate probe short-
    circuits. What is under test is the LOOP, not the allocation.
    """
    spec = yaml.safe_load(task_file.read_text())
    spec["parallel"] = "unified"
    spec["members"] = members
    task_file.write_text(yaml.safe_dump(spec), encoding="utf-8")


def _member_counting_shell(fail_on_boot=None):
    """Records the topology flags of every boot and accuracy command."""
    boots: list[str] = []
    evals: list[str] = []

    def runner(command, env=None, timeout=0):
        probe = git_probe(command)
        if probe is not None:
            return probe
        if "__SURVEY_JSON__" in command:
            return ShellResult(returncode=1, stderr="no engine here")
        if "trtllm-eval" in command:
            evals.append(command)
            return ShellResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 90.0\n")
        boots.append(command)
        if fail_on_boot is not None and len(boots) == fail_on_boot:
            return ShellResult(returncode=1, stderr="CUDA error")
        return ShellResult(returncode=0, stdout="Paris\n")

    runner.boots = boots  # type: ignore[attr-defined]
    runner.evals = evals  # type: ignore[attr-defined]
    return runner


def _run_unified(tmp_path, task_file, monkeypatch, shell, **kwargs):
    workflow = _workflow(tmp_path, shell_runner=shell, **kwargs)
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)
    return workflow, calls


def test_a_unified_unit_is_gated_once_per_member(tmp_path, task_file, monkeypatch):
    """Three members, three boots and three accuracy runs — at three topologies.

    The finding this closes: a unified target was built once and gated once,
    at pure tensor parallelism, so `--moe_ep_size` and `--enable_attention_dp`
    were emitted by no gate while the target's Coverage claimed all three
    configurations.
    """
    _unified_task(task_file, ["tp1", "tep1", "dep1"])
    shell = _member_counting_shell()
    workflow, _ = _run_unified(tmp_path, task_file, monkeypatch, shell)

    assert len(shell.boots) == 3, shell.boots
    assert len(shell.evals) == 3, shell.evals
    # And each ran at a DIFFERENT topology, which is the whole point.
    #
    # The two commands spell the same topology differently, so the boot and
    # accuracy legs are asserted separately rather than together. `tep` and
    # `dep` both carry expert parallelism; only `dep` carries attention DP,
    # and `trtllm-eval` has no flag for it at all -- it arrives through the
    # member's generated options file, so asserting it on the eval argv would
    # be asserting the defect this split fixed.
    assert sum("--moe_ep_size=1" in c for c in shell.boots) == 2
    assert sum("--enable_attention_dp" in c for c in shell.boots) == 1
    assert sum("--ep_size=1" in c for c in shell.evals) == 2
    assert not any("--moe_ep_size" in c for c in shell.evals)
    assert not any("--enable_attention_dp" in c for c in shell.evals)
    # One verdict per member, never collapsed, each naming its member.
    rendered = workflow.gate_log.render()
    for member in ("tp1", "tep1", "dep1"):
        assert f"accuracy[gsm8k]@{member}" in rendered


def test_a_failure_on_the_second_member_stops_the_unit(tmp_path, task_file, monkeypatch):
    """The third member would spend an hour measuring the same defect."""
    _unified_task(task_file, ["tp1", "tep1", "dep1"])
    shell = _member_counting_shell(fail_on_boot=2)
    workflow, calls = _run_unified(
        tmp_path, task_file, monkeypatch, shell, max_assembler_turns_per_unit=1
    )

    assert len(shell.boots) == 2, shell.boots
    assert shell.evals == []  # the unit never reached the accuracy gate
    assert [n for n, _ in calls].count("assembler") == 1
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "boot@tep1" in journal  # the message names the failing member
    assert "assembly-exhausted" in journal


def test_a_single_member_is_still_a_member(tmp_path, task_file, monkeypatch):
    """`members:` is how a run buys a cheaper answer honestly — one, named."""
    _unified_task(task_file, ["dep1"])
    shell = _member_counting_shell()
    workflow, _ = _run_unified(tmp_path, task_file, monkeypatch, shell)

    assert len(shell.boots) == 1
    assert len(shell.evals) == 1
    # `dep` reaches the boot gate as a flag and the accuracy gate as a
    # generated options file, because `trtllm-eval` has no attention-DP flag.
    assert "--enable_attention_dp" in shell.boots[0]
    assert "--ep_size=1" in shell.evals[0]
    assert "--enable_attention_dp" not in shell.evals[0]
    assert "@dep1" in workflow.gate_log.render()


# ---------------------------------------------------------------- deliver phase


def test_an_uncovered_audit_blocks_delivery(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    script = Script(
        None,
        auditor=lambda turn, prompt: {
            "decision": "UNCOVERED",
            "summary": "batch 257 outside thop_attention's column",
        },
    )
    _install(monkeypatch, workflow, script)
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "audit-uncovered" in journal
    assert "batch 257" in journal


# ----------------------------------------------------------------------- resume


def test_resume_picks_up_at_the_saved_stage(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    script = Script(None, plan=("DRAFT_READY",), review=("APPROVE",))
    _install(monkeypatch, workflow, script)
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    # Second run over the same workspace: already done, so it no-ops rather
    # than re-spending the plan phase.
    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)
    assert calls == []


# --------------------------------------- a resume must not spend a turn


def _plan_then_vocabulary(workflow, tmp_path, monkeypatch, queue=()):
    """The idiom every test here uses: write a vocabulary, then plan."""
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", list(queue))
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)


def _die_after_the_assembler(tmp_path, task_file, monkeypatch, **kwargs):
    """One orchestrator that assembles `core`, then dies before the unit closes.

    This is the shape of every restart in the run that motivated the fix: the
    assembler turn finished, its `UNIT_READY` and its work are on disk, and
    the process that dispatched it is gone. A `KeyboardInterrupt` raised from
    the reviewer's turn reproduces it exactly — `run` journals it and
    re-raises, leaving the checkpoint at `assemble` with the unit still open.
    """

    def reviewer(turn, prompt):
        raise KeyboardInterrupt

    workflow = _workflow(tmp_path, **kwargs)
    _install(monkeypatch, workflow, Script(None, reviewer=reviewer))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch)
    with pytest.raises(KeyboardInterrupt), workflow:
        workflow.run(task_file)
    state = load_state(tmp_path / "ws" / ".staircase_onboard_state.json")
    assert state.stage == STAGE_ASSEMBLE and state.unit_turns == 1
    return state


def test_a_resume_onto_a_ready_unit_does_not_dispatch(tmp_path, task_file, monkeypatch):
    """The single largest measurable waste in the run this closes.

    Ten assembler dispatches for four units, six of which built nothing,
    because a spec fix needs a restart and every restart re-entered the unit
    and dispatched. The prior `UNIT_READY` was on disk the whole time.
    """
    before = _die_after_the_assembler(tmp_path, task_file, monkeypatch)
    assert before.unit_marker["core"], "the marker is what the resume compares against"

    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)

    # The unit picks up at the gates: no assembler turn, and the run closes.
    assert [n for n, _ in calls] == ["reviewer", "auditor"], calls
    assert load_state(tmp_path / "ws" / ".staircase_onboard_state.json").verdict == "delivered"
    assert "did not spend a turn" in (tmp_path / "ws" / "journal.md").read_text()


def test_a_resume_onto_a_modified_target_re_dispatches(tmp_path, task_file, monkeypatch):
    """Whether the target is unchanged is checked, not assumed.

    A restart after somebody edited the target is a different situation from
    a clean one: the `UNIT_READY` no longer describes what is on disk, so the
    turn is worth spending.
    """
    _die_after_the_assembler(tmp_path, task_file, monkeypatch)

    dirty = passing_shell(dirt=" M tensorrt_llm/.../modeling.py\n")
    second = _workflow(tmp_path, shell_runner=dirty)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)

    assert [n for n, _ in calls].count("assembler") == 1, calls


def test_a_marker_that_cannot_be_established_re_dispatches(tmp_path, task_file, monkeypatch):
    """No git, no answer — and no answer means "assume changed".

    A wasted turn costs one dispatch; a skipped turn that was needed costs
    the unit, so the unanswerable case takes the expensive side.
    """
    _die_after_the_assembler(tmp_path, task_file, monkeypatch)

    def no_git(command, env=None, timeout=0):
        if "git -C" in command:
            return ShellResult(returncode=128, stderr="not a git repository")
        return passing_shell()(command, env, timeout)

    second = _workflow(tmp_path, shell_runner=no_git)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)

    assert [n for n, _ in calls].count("assembler") == 1, calls


def test_the_turn_a_resume_gives_back_is_still_spendable(tmp_path, task_file, monkeypatch):
    """`unit_turns` counts work, so the skipped restart leaves the budget intact.

    Two turns, and the resumed run needs all of: the skip, a failing boot
    gate, and a real assembler turn. Under the old accounting the restart
    would have spent turn 2 and the gate failure would have ended the unit
    `assembly-exhausted` with a budget it never used on work.
    """
    _die_after_the_assembler(tmp_path, task_file, monkeypatch)

    attempts = {"boot": 0}

    def one_bad_boot(command, env=None, timeout=0):
        probe = git_probe(command)
        if probe is not None:
            return probe
        if "trtllm-eval" in command or "__SURVEY_JSON__" in command:
            return passing_shell()(command, env, timeout)
        attempts["boot"] += 1
        if attempts["boot"] == 1:
            return ShellResult(returncode=1, stderr="CUDA error")
        return ShellResult(returncode=0, stdout="Paris\n")

    second = _workflow(tmp_path, shell_runner=one_bad_boot, max_assembler_turns_per_unit=2)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)

    assert [n for n, _ in calls].count("assembler") == 1, calls
    assert load_state(tmp_path / "ws" / ".staircase_onboard_state.json").verdict == "delivered"


def test_a_resume_onto_a_blocked_assembler_still_dispatches_smith(tmp_path, task_file, monkeypatch):
    """The pre-existing resume handling, unchanged by the UNIT_READY branch."""
    workflow = _workflow(tmp_path)

    def assembler(turn, prompt):
        if turn == 1:
            return {
                "decision": "BLOCKED",
                "summary": "need moe_gelu",
                "blocked_reason": "vocabulary_gap",
                "missing_entry": "moe_gelu",
            }
        return {"decision": "UNIT_READY", "summary": "assembled"}

    _install(monkeypatch, workflow, Script(None, assembler=assembler))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch)

    # Die between the decision landing and anything acting on it, which is
    # the case `_resumed_into_unit` was introduced for.
    def died(state, log):
        raise KeyboardInterrupt

    monkeypatch.setattr(workflow, "_handle_blocked_assembler", died)
    with pytest.raises(KeyboardInterrupt), workflow:
        workflow.run(task_file)

    # The resumed process's assembler has the entry it was missing, so its
    # turn lands — the default scripted `UNIT_READY`.
    second = _workflow(tmp_path)
    calls = _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)

    roles = [n for n, _ in calls]
    assert roles.count("smith") == 1, roles
    assert roles.count("assembler") == 1, roles
    assert "auditor" in roles


# --------------------------------------------------- host facts, corrections


def test_host_facts_are_written_and_reach_a_role_prompt(tmp_path, task_file, monkeypatch):
    """Six facts were hand-injected into ~15 briefs on the run this closes."""
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch)
    with workflow:
        workflow.run(task_file)

    facts = (tmp_path / "ws" / "host_facts.md").read_text()
    assert "TRTLLM_SKIP_SYS_PATH_CHECK=1" in facts
    assert "UCX ERROR" in facts
    # Seeded from what the run resolved, not hand-typed: this host's own
    # TMPDIR and the bound preflight owns.
    assert _SCRATCH in facts
    assert "61 characters" in facts
    # And it says which document it is not, so nobody merges the two.
    assert "TROUBLESHOOTING.md" in facts

    for role in ("plan_drafter", "assembler", "auditor"):
        prompt = next(p for name, p in calls if name == role)
        assert str(tmp_path / "ws" / "host_facts.md") in prompt, role
        assert "Learned during this run" in prompt, role


def test_a_fact_a_role_appended_survives_the_next_preflight(tmp_path, task_file, monkeypatch):
    """Appendable, or it is just another list somebody else wrote."""
    _die_after_the_assembler(tmp_path, task_file, monkeypatch)
    facts = tmp_path / "ws" / "host_facts.md"
    with facts.open("a", encoding="utf-8") as handle:
        handle.write("\n- **`nvidia-smi` lies about MIG here.** Read `/proc` instead.\n")

    second = _workflow(tmp_path)
    _install(monkeypatch, second, Script(None))
    with second:
        second.run(task_file)

    text = facts.read_text()
    assert "lies about MIG here" in text
    assert "TRTLLM_SKIP_SYS_PATH_CHECK=1" in text  # the seeded half is still there


def test_no_corrections_means_no_reference_to_them(tmp_path, task_file, monkeypatch):
    """A pointer at an empty file teaches the reader the pointer is worthless."""
    workflow = _workflow(tmp_path)
    calls = _install(monkeypatch, workflow, Script(None))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch)
    with workflow:
        workflow.run(task_file)

    # The file exists — a role or an operator needs somewhere to write — but
    # it is empty, so nothing points at it.
    assert (tmp_path / "ws" / "corrections.md").is_file()
    assert not any("corrections.md" in prompt for _, prompt in calls), calls


def test_a_late_review_becomes_a_correction_that_reaches_a_dispatch(
    tmp_path, task_file, monkeypatch
):
    """`SMITH_CORRECTIONS.md`, made first-class.

    The round-2 plan review landed after the smith queue was written. One of
    its findings was that a rank's FP8 GEMM never sees `N=192`; without it a
    smith would have certified a width no rank ever passes.
    """
    workflow = _workflow(tmp_path, max_plan_rounds=1)
    script = Script(None, plan=("DRAFT_READY", "DRAFT_READY"), review=("REJECT", "REJECT"))
    calls = _install(monkeypatch, workflow, script)
    _plan_then_vocabulary(
        workflow, tmp_path, monkeypatch, queue=[{"n": 1, "mode": "onboard", "target": "moe_gelu"}]
    )
    with workflow:
        workflow.run(task_file)

    corrections = (tmp_path / "ws" / "corrections.md").read_text()
    assert "PlanReviewer" in corrections
    assert "reviewed" in corrections  # the reviewer's own summary, verbatim
    assert "this file wins" in corrections

    # And it reaches the dispatches the review was too late for, with the
    # precedence rule stated rather than implied.
    for role in ("smith", "assembler"):
        prompt = next(p for name, p in calls if name == role)
        assert str(tmp_path / "ws" / "corrections.md") in prompt, role
        assert "the corrections file wins" in prompt, role


def test_journal_records_dispatch_prompts_before_they_are_sent(tmp_path, task_file, monkeypatch):
    """After a session cut the journal is the only explanation of the disk."""
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    reports = sorted((tmp_path / "ws" / "reports").iterdir())
    assert reports, "every dispatch writes a report"
    first = reports[0].read_text()
    assert "## Dispatch prompt" in first
    assert "## Result" in first
    # Ordinals are sequential across the whole run, not per role.
    assert reports[0].name.startswith("01-")


def test_progress_entries_are_typed_and_stage_scoped(tmp_path, task_file, monkeypatch):
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    data = progress_mod.read_progress(tmp_path / "ws" / "progress.yaml")
    assert [e["agent"] for e in data[progress_mod.PLAN_STAGE]] == ["plan_drafter", "plan_reviewer"]
    assert [e["agent"] for e in data[progress_mod.BUILD_STAGE]] == ["assembler", "reviewer"]
    assert [e["agent"] for e in data[progress_mod.DELIVER_STAGE]] == ["auditor"]


def test_stage_constants_advance_in_order():
    assert [STAGE_PLAN, STAGE_SMITH, STAGE_ASSEMBLE, STAGE_DELIVER] == [
        "plan",
        "smith",
        "assemble",
        "deliver",
    ]


# --------------------------------------------------- the substrate gate in run()


def test_run_invokes_the_substrate_gate_before_any_agent(tmp_path, task_file, monkeypatch):
    """It must come first, or its whole purpose is defeated."""
    order: list[str] = []

    def shell(command, env=None, timeout=0):
        if "TRTLLM_MODELING_V2=off" in command and "quickstart" in command:
            order.append("substrate")
        return ShellResult(returncode=0, stdout="Paris\n")

    workflow = _workflow(tmp_path, shell_runner=shell, skip_substrate_gate=False)

    original = workflow._run_plan_phase

    def note_plan(state, log):
        order.append("plan")
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", note_plan)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    assert order and order[0] == "substrate", order


def test_a_failing_substrate_gate_stops_the_run_with_no_agent_turn(
    tmp_path, task_file, monkeypatch
):
    """A broken host must cost seconds, not a planning turn."""

    def shell(command, env=None, timeout=0):
        return ShellResult(returncode=1, stdout="MPI_ERR_SPAWN: could not spawn processes")

    workflow = _workflow(tmp_path, shell_runner=shell, skip_substrate_gate=False)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    assert calls == [], f"an agent was dispatched despite a dead host: {calls}"


# --------------------------------------------------------------- the survey


def test_the_survey_reaches_the_plan_drafter(tmp_path, task_file, checkpoint, monkeypatch):
    """A measurement nobody reads is not a measurement."""
    checkpoint.joinpath("config.json").write_text(
        json.dumps(
            {
                "architectures": ["GptOssForCausalLM"],
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "num_local_experts": 128,
                "sinks": 64,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-5,
                "some_layer_nobody_supports": True,
            }
        )
    )

    def surveying(command, env=None, timeout=0):
        if "__SURVEY_JSON__" in command:
            payload = {
                "trtllm_ops": ["fused_qk_norm_rope", "an_op_with_no_entry"],
                "catalog": [
                    {
                        "path": "attention/fused_qk_norm_rope.py",
                        "impl": "torch.ops.trtllm.fused_qk_norm_rope",
                        "summary": "s",
                        "receipt": "passed",
                        "impl_resolves": True,
                    }
                ],
                "modules": ["rms_norm.RMSNorm"],
                "versions": {"torch": "2.12"},
                "error": "",
            }
            return ShellResult(returncode=0, stdout="__SURVEY_JSON__" + json.dumps(payload))
        return passing_shell()(command, env, timeout)

    workflow = _workflow(tmp_path, shell_runner=surveying)
    calls = _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    prompt = next(p for name, p in calls if name == "plan_drafter")
    assert "moe.routing" in prompt
    assert "attention.sinks" in prompt
    assert "some_layer_nobody_supports" in prompt  # the unknown, surfaced at plan time
    assert "fused_qk_norm_rope" in prompt  # free vocabulary
    assert "an_op_with_no_entry" in prompt  # the tier-1 rung
    assert (tmp_path / "ws" / "surface.md").is_file()


def test_a_survey_that_cannot_run_does_not_end_the_run(tmp_path, task_file, monkeypatch):
    """Planning on assumption is how every plan before this one was drafted."""

    def no_engine(command, env=None, timeout=0):
        if "__SURVEY_JSON__" in command:
            return ShellResult(returncode=1, stderr="ModuleNotFoundError: tensorrt_llm")
        return passing_shell()(command, env, timeout)

    workflow = _workflow(tmp_path, shell_runner=no_engine)
    calls = _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    prompt = next(p for name, p in calls if name == "plan_drafter")
    assert "you do not know what exists" in prompt
    assert [n for n, _ in calls].count("assembler") >= 1  # the run still ran


def test_a_survey_that_did_not_land_is_not_printed_as_a_check_mark(
    tmp_path, task_file, monkeypatch, capsys
):
    """`✔ survey ... 0/0 entries certified` was the whole defect.

    An inventory that failed to land and a catalog with nothing certified
    rendered as the same green line, so an operator had no way to tell
    "nothing is certified" from "nothing was measured". The run still
    proceeds -- that is the documented policy -- it just proceeds loudly.
    """

    def no_mpirun(command, env=None, timeout=0):
        if "__SURVEY_JSON__" in command:
            return ShellResult(returncode=127, stderr="mpirun: command not found")
        return passing_shell()(command, env, timeout)

    workflow = _workflow(tmp_path, shell_runner=no_mpirun, dry_run=True)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    printed = " ".join(capsys.readouterr().out.split())
    assert "⚠ survey" in printed
    assert "✔ survey" not in printed
    assert "inventory probe failed" in printed


def test_a_dry_run_takes_the_survey(tmp_path, task_file, monkeypatch, capsys):
    """It is a read-only, single-process measurement now, so it is affordable.

    A dry run that reports `0/0 entries certified` is the reading this
    survey exists to prevent, and an operator checking whether a host can
    host a run is exactly who needs the real numbers.
    """

    def surveying(command, env=None, timeout=0):
        if "__SURVEY_JSON__" in command:
            payload = {
                "trtllm_ops": ["fused_qk_norm_rope"],
                "catalog": [
                    {
                        "path": "attention/fused_qk_norm_rope.py",
                        "impl": "torch.ops.trtllm.fused_qk_norm_rope",
                        "summary": "s",
                        "receipt": "passed",
                        "impl_resolves": True,
                    }
                ],
                "modules": [],
                "versions": {},
                "error": "",
            }
            return ShellResult(returncode=0, stdout="__SURVEY_JSON__" + json.dumps(payload))
        return passing_shell()(command, env, timeout)

    workflow = _workflow(tmp_path, shell_runner=surveying, dry_run=True)
    _install(monkeypatch, workflow, Script(None))
    with workflow:
        workflow.run(task_file)

    printed = " ".join(capsys.readouterr().out.split())
    assert "✔ survey" in printed
    assert "1/1 entries certified on sm_103" in printed


# ------------------------------------------------------ smith concurrency


def _concurrency_probe():
    """Records the true peak of simultaneously-live smith dispatches."""
    import threading

    state = {"live": 0, "peak": 0}
    lock = threading.Lock()
    gate = threading.Event()

    def enter():
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        # Hold every dispatch open until the last one that will ever be
        # admitted has arrived, so the peak is the pool's real limit and not
        # an artefact of fast dispatches retiring before slow ones start.
        gate.wait(timeout=5)
        with lock:
            state["live"] -= 1

    return state, gate, enter


def _run_with_queue(tmp_path, task_file, monkeypatch, *, count, cap):
    """Drive the smith phase over `count` independent items at `cap`."""
    state, gate, enter = _concurrency_probe()

    kwargs = {} if cap is None else {"max_parallel_smiths": cap}
    workflow = _workflow(tmp_path, **kwargs)
    calls = _install(monkeypatch, workflow, Script(None))

    original = workflow._run_one_smith

    def counted(item):
        enter()
        return original(item)

    monkeypatch.setattr(workflow, "_run_one_smith", counted)

    queue = [{"n": i, "mode": "onboard", "target": f"entry_{i}"} for i in range(1, count + 1)]

    def plan_then_write(s, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", queue)
        return workflow.__class__._run_plan_phase(workflow, s, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)

    # Release the gate once the pool cannot admit anyone else.
    import threading

    def release():
        import time

        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if state["live"] >= min(cap or count, count):
                break
            time.sleep(0.01)
        gate.set()

    releaser = threading.Thread(target=release, daemon=True)
    releaser.start()
    with workflow:
        workflow.run(task_file)
    releaser.join(timeout=6)
    return state["peak"], calls


def test_the_default_caps_concurrent_smiths(tmp_path, task_file, monkeypatch):
    """The default used to be 0, which reads as "no limit" and meant "all".

    On the queue this workflow was built against that was 19 agent sessions
    opened at once, because every entry declared no dependencies and the
    layering produced a single batch.
    """
    from agent_flow.workflows.staircase_onboard.workflow import DEFAULT_MAX_PARALLEL_SMITHS

    assert DEFAULT_MAX_PARALLEL_SMITHS == 10
    peak, _ = _run_with_queue(tmp_path, task_file, monkeypatch, count=19, cap=None)
    assert peak <= DEFAULT_MAX_PARALLEL_SMITHS, f"peak {peak} exceeded the default cap"


def test_an_explicit_cap_is_honoured(tmp_path, task_file, monkeypatch):
    peak, _ = _run_with_queue(tmp_path, task_file, monkeypatch, count=12, cap=3)
    assert peak <= 3


def test_every_item_still_runs_under_a_cap(tmp_path, task_file, monkeypatch):
    """Capping is a throttle, not a filter."""
    _, calls = _run_with_queue(tmp_path, task_file, monkeypatch, count=12, cap=3)
    assert [n for n, _ in calls].count("smith") == 12


def test_zero_still_means_uncapped_for_whoever_asks_for_it(tmp_path, task_file, monkeypatch):
    """It stays available, but it is now an explicit choice rather than the default."""
    peak, _ = _run_with_queue(tmp_path, task_file, monkeypatch, count=14, cap=0)
    assert peak > 10


def test_both_dispatch_sites_stamp_the_decision_marker():
    """A turn that never answers must not leave the previous verdict fresh.

    `(agent, iteration)` is not unique across a run -- `iteration` is 1 for
    every deliver turn and restarts per unit in assembly -- so two turns
    routinely share a decision path. Without a marker stamped at dispatch,
    `collect_decision_file` cannot tell this turn's answer from the last
    one's. The predecessor run read three audit verdicts from two passes
    earlier, which made the audit loop structurally unable to converge.
    """
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    for method in (StaircaseOnboardWorkflow._dispatch, StaircaseOnboardWorkflow._run_one_smith):
        assert "mark_decision_dispatch" in inspect.getsource(method), (
            f"{method.__name__} dispatches a decision-file turn without stamping a marker"
        )


def test_preflight_surfaces_the_two_shaping_hazards():
    """Both are legal, both cost hours, and both are invisible until they bite."""
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    source = inspect.getsource(StaircaseOnboardWorkflow._run_preflight)
    assert "image_import_warning" in source
    assert "shares_allocation_with_spool" in source


def test_a_gate_may_have_its_own_launch_substrate():
    """`environment.workloads.gate` exists so a gate can be its own failure domain."""
    import inspect

    from agent_flow.workflows.staircase_onboard.workflow import StaircaseOnboardWorkflow

    source = inspect.getsource(StaircaseOnboardWorkflow._gate_env)
    assert 'workload="gate"' in source, "the gate env ignores the workloads override"
