# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evidence freshness, the re-run queue and the final-regression binding.

Each failure mode here happened on the DeepSeek-V4.1 run:

- a later unit edited the target, so the earlier units' gate verdicts were
  about code that no longer existed, and the report still quoted them;
- a default switch flipped (decoder replay became the multimodal default)
  and nothing re-gated the units measured under the old default;
- a witness's settings changed after its verdict of record, and delivery
  accepted the old verdict;
- the delivered head carried a functional change and a lint series that
  landed after the last full regression, so no gate ran on what shipped.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase_onboard import freshness
from agent_flow.workflows.staircase_onboard.freshness import (
    RerunEntry,
    assess,
    complete_regression,
    drain,
    last_regression,
    read_rerun_queue,
    record_regression,
    regression_binding,
    rerun_queue,
    switch_history_flips,
    write_rerun_queue,
)
from agent_flow.workflows.staircase_onboard.gates import GateResult
from agent_flow.workflows.staircase_onboard.ledger import (
    Ledger,
    VerdictContext,
    native_build,
    record_gate_result,
    tree_identity,
)

from .test_delivery import TARGET_REL, _build_repo

_SWITCH_LINE = '\nimport os\nREPLAY = os.environ.get("TRTLLM_REPLAY", "{}")\n'


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def _commit_all(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)


@pytest.fixture
def repo(tmp_path):
    root = _build_repo(tmp_path)
    modeling = root / TARGET_REL / "modeling.py"
    modeling.write_text(modeling.read_text() + _SWITCH_LINE.format("0"), encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.st")
    _git(root, "config", "user.name", "t")
    _commit_all(root, "seed")
    return root


def _spec(**overrides):
    spec = {
        "features": ["mtp"],
        "accuracy": {"benchmark": "gsm8k", "witness": {"mtp": {"contrast_env": {"X": "0"}}}},
        "environment": {"kind": "local", "python": "/py", "env": {}},
    }
    spec.update(overrides)
    return spec


def _record(ledger, root, *, unit="core", gate="accuracy[gsm8k]", member="tp4", **kw):
    passed = kw.pop("passed", True)
    env = kw.pop("env", {})
    task = kw.pop("task", _spec())
    return record_gate_result(
        ledger,
        GateResult(name=gate, passed=passed, detail="d", member=member, **kw),
        VerdictContext(
            repo_root=str(root), target_rel=TARGET_REL, unit=unit, env=env, task_data=task
        ),
    )


def _reasons(ledger, root, **kw):
    (item,) = assess(ledger.latest().values(), root, **kw)
    return item.reasons


# ---------------------------------------------------------------- staleness


def test_an_untouched_tree_leaves_a_verdict_fresh(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    assert _reasons(ledger, repo, task_data=_spec()) == ()


def test_a_commit_that_touches_a_dependency_makes_the_verdict_stale(tmp_path, repo):
    """The core verdict, after a later unit rewrote the target it measured."""
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    modeling = repo / TARGET_REL / "modeling.py"
    modeling.write_text(modeling.read_text() + "\n# mtp increment\n", encoding="utf-8")
    _commit_all(repo, "mtp")

    reasons = _reasons(ledger, repo)
    assert any(r.startswith("code changed:") and "modeling.py" in r for r in reasons)


def test_a_change_to_an_imported_catalog_entry_makes_the_verdict_stale(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    wrapper = next(repo.rglob("cublas_mm.py"))
    wrapper.write_text("def f(): return 2\n", encoding="utf-8")
    assert any("cublas_mm.py" in r for r in _reasons(ledger, repo))


def test_a_change_elsewhere_in_the_repo_leaves_the_verdict_fresh(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    (repo / "README.md").write_text("docs\n", encoding="utf-8")
    _commit_all(repo, "docs")
    assert _reasons(ledger, repo) == ()


def test_a_newly_imported_entry_is_a_new_dependency(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    extra = repo / TARGET_REL / "vision.py"
    extra.write_text("X = 1\n", encoding="utf-8")
    reasons = _reasons(ledger, repo)
    assert any(r.startswith("new dependency:") and "vision.py" in r for r in reasons)


def test_a_rebuilt_native_library_makes_the_verdict_stale(tmp_path, repo):
    lib = repo / "tensorrt_llm" / "libs" / "libtensorrt_llm.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"\x7fELF-one")
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    lib.write_bytes(b"\x7fELF-two")
    assert any(r.startswith("native build changed") for r in _reasons(ledger, repo))


def test_a_changed_spec_for_the_unit_makes_its_verdict_stale(tmp_path, repo):
    """F039's follow-up: a witness whose settings changed after its verdict of record."""
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo, unit="mtp", gate="equivalence")
    changed = _spec(
        accuracy={"benchmark": "gsm8k", "witness": {"mtp": {"contrast_env": {"X": "1"}}}}
    )
    assert any(
        "task spec for unit mtp changed" in r for r in _reasons(ledger, repo, task_data=changed)
    )


def test_another_units_spec_change_leaves_a_verdict_fresh(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo, unit="core")
    changed = _spec(
        accuracy={"benchmark": "gsm8k", "witness": {"mtp": {"contrast_env": {"X": "1"}}}}
    )
    assert _reasons(ledger, repo, task_data=changed) == ()


def test_a_flipped_default_in_a_dependency_is_named(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    modeling = repo / TARGET_REL / "modeling.py"
    modeling.write_text(
        modeling.read_text().replace(_SWITCH_LINE.format("0"), _SWITCH_LINE.format("1")),
        encoding="utf-8",
    )
    reasons = _reasons(ledger, repo)
    assert "default switch flipped: TRTLLM_REPLAY '0' -> '1'" in reasons


def test_a_flipped_default_outside_the_target_is_found_through_history(tmp_path, repo):
    """The default lives in a runtime module no verdict lists as a dependency.

    This is gap-analysis D5: flipping a default must re-gate the units
    measured under the old one, even though their own files did not change.
    """
    runtime = repo / "tensorrt_llm" / "_torch" / "pyexecutor" / "replay.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text('import os\nON = os.environ.get("TRTLLM_DECODER_REPLAY", "0")\n')
    _commit_all(repo, "runtime switch")
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)

    runtime.write_text('import os\nON = os.environ.get("TRTLLM_DECODER_REPLAY", "1")\n')
    _commit_all(repo, "replay on by default")

    commit = ledger.verdicts()[0].tree.commit
    assert switch_history_flips(repo, commit) == {"TRTLLM_DECODER_REPLAY": ("0", "1")}
    assert "default switch flipped: TRTLLM_DECODER_REPLAY '0' -> '1'" in _reasons(ledger, repo)
    assert _reasons(ledger, repo, check_history=False) == ()


def test_a_switch_the_gate_set_explicitly_does_not_flip_with_its_default(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo, env={"TRTLLM_REPLAY": "0"})
    modeling = repo / TARGET_REL / "modeling.py"
    modeling.write_text(
        modeling.read_text().replace(_SWITCH_LINE.format("0"), _SWITCH_LINE.format("1")),
        encoding="utf-8",
    )
    _commit_all(repo, "flip")
    reasons = _reasons(ledger, repo)
    assert not [r for r in reasons if "TRTLLM_REPLAY" in r]
    # it is still stale, for the honest reason: the file changed
    assert any(r.startswith("code changed:") for r in reasons)


# -------------------------------------------------------------- re-run queue


def test_the_queue_holds_stale_and_failing_verdicts_only(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo, gate="boot", member="tp4")
    _record(ledger, repo, gate="accuracy[gsm8k]", member="tp4", passed=False)
    _record(ledger, repo, gate="boot", member="ep4")
    queue = rerun_queue(assess(ledger.latest().values(), repo))
    assert [(e.gate, e.member) for e in queue] == [("accuracy[gsm8k]", "tp4")]
    assert queue[0].reasons == ("last verdict FAIL",)


def test_at_delivery_a_verdict_from_another_tree_is_queued(tmp_path, repo):
    """Evidence carried over from an earlier commit is evidence about that commit."""
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    (repo / "runtime.py").write_text("RETRY = 3\n", encoding="utf-8")
    _commit_all(repo, "a functional change outside the target")

    assessed = assess(ledger.latest().values(), repo)
    assert rerun_queue(assessed) == []
    tree = tree_identity(repo)
    (entry,) = rerun_queue(assessed, tree=tree, native=native_build(repo))
    assert "measured on tree" in entry.reasons[0]


def test_scope_drops_verdicts_the_spec_no_longer_requires(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo, member="tp8", passed=False)
    queue = rerun_queue(assess(ledger.latest().values(), repo), scope=lambda v: v.member != "tp8")
    assert queue == []


def test_drain_keeps_what_could_not_be_re_run_and_says_why():
    queue = [RerunEntry("boot", "core", "tp4", ("stale",)), RerunEntry("perf", "core", "tp4", ())]
    answers = {"boot": False, "perf": None}
    remaining = drain(queue, lambda e: answers[e.gate])
    assert [e.reasons[-1] for e in remaining] == [
        "re-run failed",
        "no automatic re-run for this gate",
    ]
    assert drain(queue, lambda e: True) == []


def test_the_queue_round_trips_through_the_workspace(tmp_path):
    entries = [RerunEntry("boot", "core", "tp4", ("code changed: x",), "abc")]
    path = write_rerun_queue(tmp_path, entries)
    assert json.loads(path.read_text())["entries"][0]["gate"] == "boot"
    assert read_rerun_queue(tmp_path) == entries


# ----------------------------------------------------------------- binding


def _regress(ledger, repo):
    tree, native = tree_identity(repo), native_build(repo)
    regression = complete_regression(ledger.latest(), tree, native)
    assert regression is not None
    return record_regression(ledger, regression)


def test_a_regression_needs_every_verdict_passing_on_this_tree(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo, gate="boot")
    _record(ledger, repo, gate="accuracy[gsm8k]", passed=False)
    tree, native = tree_identity(repo), native_build(repo)
    assert complete_regression(ledger.latest(), tree, native) is None
    _record(ledger, repo, gate="accuracy[gsm8k]")
    assert complete_regression(ledger.latest(), tree, native) is not None


def test_the_tree_of_the_last_regression_binds_the_merge(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    _regress(ledger, repo)
    assert regression_binding(ledger, repo).ok

    # The DeepSeek-V4.1 shape: one more functional commit after the regression.
    (repo / "runtime.py").write_text("RETRY = 3\n", encoding="utf-8")
    _commit_all(repo, "VMM create retries")
    binding = regression_binding(ledger, repo)
    assert not binding.ok
    assert "the last full regression ran on" in binding.detail


def test_a_re_signed_head_is_still_bound(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    _regress(ledger, repo)
    _git(repo, "commit", "-q", "--amend", "-s", "-m", "seed, signed")
    assert regression_binding(ledger, repo).ok


def test_a_rebuilt_native_build_unbinds_the_merge(tmp_path, repo):
    lib = repo / "tensorrt_llm" / "libs" / "libth_common.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"\x7fELF-1")
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    _regress(ledger, repo)
    lib.write_bytes(b"\x7fELF-2")
    binding = regression_binding(ledger, repo)
    assert not binding.ok and "native build" in binding.detail


def test_no_regression_at_all_refuses(tmp_path, repo):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    binding = regression_binding(ledger, repo)
    assert not binding.ok and "no full regression" in binding.detail


def test_outside_git_the_binding_makes_no_claim(tmp_path):
    binding = regression_binding(Ledger.in_workspace(tmp_path / "ws"), tmp_path)
    assert binding.ok and not binding.claim


def test_check_merge_cli_exit_codes(tmp_path, repo, capsys):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    args = ["check-merge", str(tmp_path / "ws"), "--repo", str(repo)]
    assert freshness.main(args) == 1
    _regress(ledger, repo)
    assert freshness.main(args) == 0
    assert last_regression(ledger) is not None
    assert "ok:" in capsys.readouterr().out


def test_status_cli_lists_stale_verdicts(tmp_path, repo, capsys):
    ledger = Ledger.in_workspace(tmp_path / "ws")
    _record(ledger, repo)
    args = ["status", str(tmp_path / "ws"), "--repo", str(repo), "--write-queue"]
    assert freshness.main(args) == 0
    (repo / TARGET_REL / "modeling.py").write_text("# rewritten\n", encoding="utf-8")
    assert freshness.main(args) == 1
    out = capsys.readouterr().out
    assert "STALE" in out and "code changed" in out
    assert read_rerun_queue(tmp_path / "ws")[0].gate == "accuracy[gsm8k]"


# ------------------------------------------------------- delivery, end to end


def _delivered_run(tmp_path, task_file, monkeypatch, *, before_deliver=None, shell=None):
    from agent_flow.workflows.staircase_onboard.state import load_state

    from .test_workflow import Script, _install, _plan_then_vocabulary, _workflow, passing_shell

    # `task_file` is built on this module's `repo` fixture, already a git repo.
    repo = Path(yaml.safe_load(task_file.read_text())["trtllm_repo_path"])

    calls: list[str] = []
    base = shell or passing_shell()

    def counting(command, env=None, timeout=0):
        if "trtllm-eval" in command or "quickstart" in command:
            calls.append(command)
        return base(command, env, timeout)

    workflow = _workflow(tmp_path, shell_runner=counting)
    _install(monkeypatch, workflow, Script(None))
    _plan_then_vocabulary(workflow, tmp_path, monkeypatch, [])
    if before_deliver is not None:
        original = workflow._run_deliver_phase

        def deliver(state, log):
            before_deliver(repo)
            return original(state, log)

        monkeypatch.setattr(workflow, "_run_deliver_phase", deliver)
    with workflow:
        workflow.run(task_file)
    state = load_state(tmp_path / "ws" / ".staircase_onboard_state.json")
    return state, calls, repo, Ledger.in_workspace(tmp_path / "ws")


def test_a_clean_delivery_records_the_regression_and_re_runs_nothing(
    tmp_path, task_file, monkeypatch
):
    state, calls, repo, ledger = _delivered_run(tmp_path, task_file, monkeypatch)
    assert state.verdict == "delivered"
    assert len(calls) == 2  # boot + accuracy for core, once each
    regression = last_regression(ledger)
    assert regression is not None
    assert regression.tree.key == tree_identity(repo).key
    assert regression_binding(ledger, repo).ok


def test_the_pre_merge_pack_states_the_binding_and_how_to_re_check_it(
    tmp_path, task_file, monkeypatch
):
    """The pack's remaining steps (lint, split) can change the tree after delivery."""
    state, _, repo, _ = _delivered_run(tmp_path, task_file, monkeypatch)
    assert state.verdict == "delivered"
    pack = (tmp_path / "ws" / "reports" / "premerge.md").read_text()
    assert "## Evidence binding (bound)" in pack
    assert "freshness check-merge" in pack and f"--repo {repo}" in pack
    assert "evidence bound:" in (tmp_path / "ws" / "journal.md").read_text()


def test_a_change_before_delivery_re_runs_the_stale_gates(tmp_path, task_file, monkeypatch):
    def edit_target(repo):
        modeling = repo / TARGET_REL / "modeling.py"
        modeling.write_text(modeling.read_text() + "\n# late fix\n", encoding="utf-8")
        _commit_all(repo, "late fix")

    state, calls, repo, ledger = _delivered_run(
        tmp_path, task_file, monkeypatch, before_deliver=edit_target
    )
    assert state.verdict == "delivered"
    assert len(calls) == 4  # both gates again, on the delivered tree
    gates = [v for v in ledger.latest().values() if v.unit]
    assert all(v.tree.key == tree_identity(repo).key for v in gates)
    assert regression_binding(ledger, repo).ok
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "gate(s) to re-run before delivery" in journal


def test_a_re_run_that_fails_refuses_delivery(tmp_path, task_file, monkeypatch):
    from .test_workflow import passing_shell

    scores = iter([90.0, 10.0])
    good = passing_shell()

    def shell(command, env=None, timeout=0):
        if "trtllm-eval" in command:
            return passing_shell(score=next(scores))(command, env, timeout)
        return good(command, env, timeout)

    def edit_target(repo):
        modeling = repo / TARGET_REL / "modeling.py"
        modeling.write_text(modeling.read_text() + "\n# regressed\n", encoding="utf-8")
        _commit_all(repo, "regression")

    state, _, _, ledger = _delivered_run(
        tmp_path, task_file, monkeypatch, before_deliver=edit_target, shell=shell
    )
    assert state.verdict == "delivery-failed:evidence"
    assert read_rerun_queue(tmp_path / "ws")[0].gate.startswith("accuracy")
    assert last_regression(ledger) is None
