# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheaper oracles that still fail on what they guard, and the milestones they reach sooner.

- the accuracy benchmark runs where the logit gate cannot stand for it, and a
  member the benchmark left out still fails on its logits;
- the boot holds the engine's first greedy tokens to the HF reference's;
- the logit and serve gates are on wherever the spec lets them be derived;
- the bring-up is named before correctness;
- after a change that may move memory, the tightest configuration goes first.

No GPU and no model: the reference is the logit tests' tiny deterministic
language model, and the engine is what that model predicts.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks import logit_agreement as la
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.vcs import git_output
from agent_flow.workflows.trtllm_modeling_v2.onboard import logit_reference as lr
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.journal import Journal
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import (
    TrtllmModelingV2Workflow,
    memory_relevant,
)
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    _validate_accuracy,
    load_and_validate_task_yaml,
)

from .test_logit_agreement import _VOCAB, FakeModel, _env, _fake_driver, _make_reference, _task

_SAMPLED = ["tep4", "dep2", "dep4", "tp1"]


def _accuracy_members(contract, unit):
    return [k.member for k in contract.checks if k.unit == unit and k.gate.startswith("accuracy[")]


# --------------------------------------------------------- where the benchmark runs


def test_with_the_logit_gate_the_benchmark_runs_where_logits_cannot_stand_for_it(tmp_path):
    reference = _make_reference(tmp_path)
    contract = c.compile_contract(_task(tmp_path, reference), world_size=4)
    members = gates.order_members_by_cost(_task(tmp_path)["members"])
    for unit in ("core", "mtp", "multimodal"):
        # tp1 (the narrowest), tep4 (the widest, cheapest to start), and attention DP.
        assert _accuracy_members(contract, unit) == _SAMPLED
        logits = [k.member for k in contract.checks if k.unit == unit and k.gate == la.GATE_NAME]
        assert logits == members
    [note, *_] = contract.notes
    assert note.startswith("core: accuracy at tep4 (the widest), dep2 (attention data parallelism")
    assert note.endswith("tep2, tp2, tp4 proven equivalent by the logit gate")
    assert "notes" in c.Contract(contract.checks, contract.notes).as_dict()
    assert note in contract.render()


def test_every_member_runs_it_without_the_logit_gate_or_when_the_spec_says_every(tmp_path):
    reference = _make_reference(tmp_path)
    members = gates.order_members_by_cost(_task(tmp_path)["members"])
    off = c.compile_contract(_task(tmp_path), world_size=4)
    assert _accuracy_members(off, "core") == members and off.notes == ()
    task = _task(tmp_path, reference)
    task["accuracy"]["benchmark_members"] = "every"
    every = c.compile_contract(task, world_size=4)
    assert _accuracy_members(every, "mtp") == members and every.notes == ()
    # The switch says where the benchmark runs, not what it measures.
    sampled = c.compile_contract(_task(tmp_path, reference), world_size=4)
    shared = {k.key: k.equivalence for k in sampled.checks if k.gate.startswith("accuracy[")}
    assert all(shared[k.key] == k.equivalence for k in every.checks if k.key in shared)


def test_a_speculator_the_boot_does_not_compare_keeps_the_benchmark_everywhere(tmp_path):
    reference = _make_reference(tmp_path)
    task = _task(tmp_path, reference)
    task["accuracy"]["boot"] = {"greedy": {"enabled": False}}
    contract = c.compile_contract(task, world_size=4)
    assert _accuracy_members(contract, "core") == _SAMPLED
    every = gates.order_members_by_cost(task["members"])
    # The logit gate runs no speculator, and nothing else holds its tokens to the reference.
    assert _accuracy_members(contract, "mtp") == every
    assert _accuracy_members(contract, "multimodal") == every


def test_two_members_or_fewer_are_both_measured(tmp_path):
    reference = _make_reference(tmp_path)
    task = _task(tmp_path, reference)
    task["members"] = ["tp1", "tp2"]
    contract = c.compile_contract(task, world_size=2)
    assert _accuracy_members(contract, "core") == ["tp1", "tp2"]


def test_a_member_the_benchmark_left_out_still_fails_on_its_logits(tmp_path):
    """The proof that replaced tp2's benchmark fails on an engine that is wrong at tp2."""
    reference = _make_reference(tmp_path)
    task = _task(tmp_path, reference)
    compiled = c.compile_contract(task, world_size=4)
    core = c.Contract(tuple(k for k in compiled.checks if k.unit == "core" and k.tier == "inner"))
    assert "core:accuracy[gsm8k]@tp2" not in {k.key for k in core.checks}
    status = c.ContractStatus(path=tmp_path / c.STATUS_FILENAME)
    for check in core.checks:
        if check.key != "core:logits@tp2":
            passed = SimpleNamespace(name=check.gate, member=check.member, passed=True, detail="")
            status.record("core", passed, marker="v1")
    common = dict(task_data=task, features=[], unit="core", member="tp2", workdir=tmp_path / "w")
    # An engine whose sharding moves the logits well past the reference's own spread.
    wrong = la.run_logit_gate(_env(2), runner=_fake_driver(FakeModel(0.8)), **common)
    assert not wrong.passed and "worst position" in wrong.detail
    status.record("core", wrong, marker="v1")
    evaluation = c.evaluate(core, status, marker="v1")
    assert [check.key for check, _ in evaluation.failed] == ["core:logits@tp2"]
    assert not evaluation.holds and not evaluation.missing
    right = la.run_logit_gate(_env(2), runner=_fake_driver(), **common)
    assert right.passed, right.detail
    status.record("core", right, marker="v1")
    assert c.evaluate(core, status, marker="v1").holds


def test_the_logit_gate_names_what_of_a_member_it_does_not_run():
    assert la.unseen("tp4", []) == {}
    assert set(la.unseen("dep4", ["mtp"])) == {la.UNSEEN_ATTENTION_DP, la.UNSEEN_SPECULATION}
    assert set(la.unseen("tep4", ["multimodal"])) == set()


# ------------------------------------------------------------- the boot's tokens


def _driver(model=None, *, change=None, short=False):
    """The engine driver over the boot's spec: text prompts answer, token prompts run ``model``.

    ``change(index, tokens)`` edits a token prompt's greedy tokens; ``short``
    returns one token fewer than asked.
    """
    model = model or FakeModel()
    seen = []

    def run(command, env=None, timeout=0):
        words = shlex.split(command)
        spec = json.loads(Path(words[words.index("--spec") + 1]).read_text())
        seen.append(spec)
        lines = []
        for n, request in enumerate(spec["requests"]):
            record = {"index": n, "id": request["id"], "text": "Paris is the capital"}
            if "token_ids" in request:
                ids = list(request["token_ids"])
                tokens = []
                for _ in range(request["max_tokens"] - (1 if short else 0)):
                    lp = model.logprobs([*ids, *tokens])
                    tokens.append(max(range(_VOCAB), key=lambda t: lp[t]))
                if change is not None:
                    tokens = change(n, tokens)
                record.update(
                    token_ids=tokens, prompt_tokens=len(ids), text=" ".join(map(str, tokens))
                )
            lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}\n")
        return CommandResult(returncode=0, stdout="".join(lines))

    run.seen = seen  # type: ignore[attr-defined]
    return run


def _boot(tmp_path, reference, runner, *, unit="core", **greedy):
    task = _task(tmp_path, reference)
    if greedy:
        task["accuracy"]["boot"] = {"greedy": greedy}
    spec = gates.resolve_boot_spec(task, unit, [])
    return gates.run_boot_gate(
        _env(1), task_data=task, spec=spec, runner=runner, features=[], unit=unit, member="tp1",
        workdir=tmp_path / "w",
    )  # fmt: skip


def test_the_boot_holds_the_engines_greedy_tokens_to_the_references(tmp_path):
    reference = _make_reference(tmp_path)
    run = _driver()
    result = _boot(tmp_path, reference, run)
    assert result.passed, result.detail
    assert "the first 6 greedy tokens of 2 prompt(s) of ref.json.gz are the HF reference's" in (
        result.detail
    )
    [sent] = run.seen
    # The reference's text prompts as token ids, the image prompt left to the logit gate.
    loaded = lr.load_reference(reference)
    greedy = [r for r in sent["requests"] if "token_ids" in r]
    assert [r["token_ids"] for r in greedy] == [
        list(loaded.prompt("capital").input_ids),
        list(loaded.prompt("long").input_ids),
    ]
    assert sent["ignore_eos"] is True and {r["max_tokens"] for r in greedy} == {6}


def test_a_wrong_token_at_a_confident_step_fails_the_boot(tmp_path):
    reference = _make_reference(tmp_path)
    loaded = lr.load_reference(reference)
    capital = loaded.prompt("capital")
    step = 2
    outside = next(
        t for t in range(_VOCAB) if t not in capital.top_ids[capital.prompt_positions + step]
    )

    def wrong(index, tokens):
        return [outside if (index, n) == (2, step) else t for n, t in enumerate(tokens)]

    result = _boot(tmp_path, reference, _driver(change=wrong))
    assert not result.passed
    assert "greedy tokens are not the HF reference's" in result.detail
    assert f"capital step {step}: engine token {outside}" in result.detail
    assert "not in the reference's top-k" in result.detail
    short = _boot(tmp_path, reference, _driver(short=True))
    assert not short.passed and "generated 5 of 6 tokens" in short.detail


def test_a_near_tie_ends_the_comparison_and_a_clear_lead_does_not():
    prompt = gates.GreedyPrompt(
        id="p",
        input_ids=(1, 2),
        expected=(5, 6, 7),
        top_ids=((5, 9), (6, 8), (7, 3)),
        top_logprobs=((-0.1, -3.0), (-0.70, -0.75), (-0.2, -4.0)),
    )
    workload = gates.GreedyWorkload("ref", (prompt,), tie_margin=0.15)
    failures, ties = gates.judge_greedy(workload, [{"token_ids": [5, 8, 1], "prompt_tokens": 2}])
    assert failures == [] and ties == [
        "p step 1: engine 8, reference 6, 0.050 apart in the reference"
    ]
    failures, _ = gates.judge_greedy(workload, [{"token_ids": [9, 6, 7]}])
    assert failures == [
        "p step 0: engine token 9, reference 5 (the engine's token 2.900 behind in the "
        "reference, past the near-tie margin 0.150)"
    ]
    failures, _ = gates.judge_greedy(workload, [{"token_ids": [5, 6, 7], "prompt_tokens": 3}])
    assert failures == ["p: the engine's prompt is 3 tokens and the reference's 2"]
    assert gates.judge_greedy(workload, [None])[0] == ["p: the engine returned no record"]


def test_a_boot_that_cannot_read_its_reference_fails_before_the_engine_starts(tmp_path):
    run = _driver()
    result = _boot(tmp_path, tmp_path / "missing.json.gz", run)
    assert not result.passed and result.detail.startswith("the boot's greedy reference:")
    assert run.seen == []
    reference = _make_reference(tmp_path)
    loaded = lr.load_reference(reference)
    limits = gates.EngineLimits(max_num_tokens=4)
    workload = gates.greedy_workload(gates.GreedySetting(str(reference)), limits, loaded)
    assert workload.error.startswith("no text prompt of the greedy reference")
    assert "capital left out" in workload.error


def test_where_the_boot_compares_tokens_and_where_it_does_not(tmp_path):
    reference = _make_reference(tmp_path)
    task = _task(tmp_path, reference, threshold_factor=2.0)
    setting = gates.resolve_boot_spec(task, "core", []).greedy
    assert setting == gates.GreedySetting(str(reference), 4, 24, 2.0)
    assert gates.resolve_boot_spec(_task(tmp_path), "core", []).greedy is None
    task["accuracy"]["boot"] = {"greedy": {"tokens": 8}, "mtp": {"greedy": {"enabled": False}}}
    assert gates.resolve_boot_spec(task, "core", []).greedy.tokens == 8
    assert gates.resolve_boot_spec(task, "mtp", ["mtp"]).greedy is None
    # A block that only says how tokens are compared declares no workload: boot.yaml stands.
    boot_yaml = tmp_path / "boot.yaml"
    boot_yaml.write_text(yaml.safe_dump({"prompts": ["from the workspace"]}), "utf-8")
    assert gates.resolve_boot_spec(task, "core", [], boot_yaml).prompts == ("from the workspace",)
    # A variant serves another configuration than the reference measured: off unless it says.
    task["variants"] = {"kv_fp8": {"unit": "core"}}
    assert gates.resolve_boot_spec(task, "kv_fp8", []).greedy is None
    task["accuracy"]["boot"]["kv_fp8"] = {"greedy": {"enabled": True}}
    assert gates.resolve_boot_spec(task, "kv_fp8", []).greedy is not None


# ------------------------------------------------------------- derived defaults


def _spec_file(tmp_path, accuracy):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "config.json").write_text("{}")
    path = tmp_path / "task.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "checkpoint_path": str(ckpt),
                "trtllm_repo_path": str(repo),
                "arch": "sm_100",
                "environment": {"kind": "local", "python": "/usr/bin/python3"},
                "accuracy": {
                    "benchmark": "gsm8k",
                    "reference_yaml": "r.yaml",
                    "model_key": "M",
                    **accuracy,
                },
            }
        )
    )
    return path


def test_the_logit_gate_is_on_where_a_reference_sits_beside_the_spec(tmp_path):
    path = _spec_file(tmp_path, {})
    assert "logits" not in load_and_validate_task_yaml(path)["accuracy"]
    reference = _make_reference(tmp_path, name=task_schema.DEFAULT_LOGIT_REFERENCE)
    accuracy = load_and_validate_task_yaml(path)["accuracy"]
    assert accuracy["logits"] == {"reference": str(reference.resolve())}
    assert la.LogitsSpec.from_task({"accuracy": accuracy}).enabled
    off = load_and_validate_task_yaml(_spec_file(tmp_path, {"logits": {"enabled": False}}))
    assert off["accuracy"]["logits"] == {"enabled": False}
    unnamed = _spec_file(tmp_path, {"logits": {"threshold_factor": 2.0}})
    assert load_and_validate_task_yaml(unnamed)["accuracy"]["logits"]["reference"] == str(
        reference.resolve()
    )


def test_the_serve_gate_is_on_where_the_spec_bounds_the_servers_length(tmp_path):
    unbounded = load_and_validate_task_yaml(_spec_file(tmp_path, {}))
    assert unbounded["accuracy"]["serve"] == {"enabled": False}
    bounded = load_and_validate_task_yaml(_spec_file(tmp_path, {"max_seq_len": 8192}))
    assert bounded["accuracy"]["serve"] == {"enabled": True}
    options = tmp_path / "options.yaml"
    options.write_text("max_seq_len: 4096\n")
    in_file = load_and_validate_task_yaml(_spec_file(tmp_path, {"llm_api_options": str(options)}))
    assert in_file["accuracy"]["serve"]["enabled"] is True
    kept = _spec_file(tmp_path, {"max_seq_len": 8192, "serve": {"enabled": False, "max_tokens": 8}})
    assert load_and_validate_task_yaml(kept)["accuracy"]["serve"] == {
        "enabled": False,
        "max_tokens": 8,
    }


@pytest.mark.parametrize(
    "accuracy, message",
    [
        ({"benchmark_members": "some"}, "benchmark_members' must be one of"),
        ({"anchor": "tbd"}, "anchor' must be one of"),
        ({"anchor": {"mtp": 1.0}}, "anchor.mtp' must be one of"),
        ({"boot": {"greedy": {"tokens": 0}}}, "boot.greedy.tokens' must be an integer >= 1"),
        ({"boot": {"core": {"greedy": {"prompt": 2}}}}, "boot.core.greedy"),
        ({"boot": {"greedy": {"enabled": "yes"}}}, "boot.greedy.enabled' must be true or false"),
    ],
)
def test_the_new_keys_are_validated(accuracy, message):
    errors: list[str] = []
    _validate_accuracy({"benchmark": "gsm8k", "reference_yaml": "r.yaml", **accuracy}, errors)
    assert any(message in e for e in errors), errors


# ------------------------------------------------------------------- the bring-up


_BOOT = c.Check("core", "boot", "tp1", c.TIER_INNER)
_ACCURACY = c.Check("core", "accuracy[gsm8k]", "tp1", c.TIER_INNER)
_WIDE = c.Check("core", "accuracy[gsm8k]", "tp4", c.TIER_INNER)
_MTP = c.Check("mtp", "boot", "tp1", c.TIER_INNER)
_CONTRACT = c.Contract(checks=(_BOOT, _WIDE, _ACCURACY, _MTP))


def _holding(*checks):
    return c.Evaluation(passed=tuple(checks), failed=(), missing=())


def test_the_bring_up_is_the_first_units_boot_and_accuracy_at_its_narrowest_member(tmp_path):
    assert c.bringup_checks(_CONTRACT) == (_BOOT, _ACCURACY)
    assert c.bringup_holds(_CONTRACT, _holding(_BOOT, _ACCURACY))
    assert not c.bringup_holds(_CONTRACT, _holding(_BOOT, _WIDE, _MTP))
    assert c.bringup_checks(c.Contract(checks=())) == ()
    reference = _make_reference(tmp_path)
    compiled = c.compile_contract(_task(tmp_path, reference), world_size=4)
    assert [k.key for k in c.bringup_checks(compiled)] == [
        "core:boot@tp1",
        "core:accuracy[gsm8k]@tp1",
    ]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class _Bringing(TrtllmModelingV2Workflow):
    """What naming the bring-up uses of the workflow: the contract, the checkout, the journal."""

    def __init__(self, workspace: Path, repo: Path):
        self.workspace = workspace
        self.task_data = {"trtllm_repo_path": str(repo)}
        self.journal = Journal(path=workspace / "journal.md")
        self.journal.open(slug="t")
        self._git = git_output
        self._planned_state = SimpleNamespace(branch="trtllm-modeling-v2/t")
        self._contract = _CONTRACT
        self._milestone_lock = threading.Lock()
        self.marker = "v1"

    def _target_marker(self):
        return self.marker


def test_the_bring_up_is_named_once_by_a_branch_a_journal_line_and_a_milestone(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "user/x/newmodel", str(repo))
    (repo / "tensorrt_llm" / "x.py").write_text("X = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "x")
    head = _git(repo, "rev-parse", "HEAD")
    stage = _Bringing(tmp_path / "ws", repo)

    assert stage._mark_bringup("v1", _holding(_BOOT)) is None  # accuracy at tp1 not yet
    stage.marker = "v2"
    assert stage._mark_bringup("v1", _holding(_BOOT, _ACCURACY)) is None  # the target moved
    stage.marker = "v1"
    milestone = stage._mark_bringup("v1", _holding(_BOOT, _ACCURACY))
    assert milestone.sha == head and milestone.ref == "user/x/newmodel-bringup"
    assert milestone.tier == c.MILESTONE_BRINGUP and milestone.checks == 2
    assert _git(repo, "rev-parse", "user/x/newmodel-bringup") == head
    lines = [line for line in stage.journal.path.read_text().splitlines() if "BRINGUP" in line]
    assert len(lines) == 1 and f"**BRINGUP DELIVERED {head}**" in lines[0]
    assert "core:boot@tp1, core:accuracy[gsm8k]@tp1 hold on it" in lines[0]
    # Once: a later state does not move it, and it is not a correctness milestone.
    assert stage._mark_bringup("v1", _holding(_BOOT, _ACCURACY)) is None
    assert c.latest_milestone(stage.workspace) is None
    assert c.bringup_milestone(stage.workspace) == milestone
    standing = c.standing(_holding(_BOOT), stage.workspace, "v1")
    assert f"Bring-up delivered at {head[:10]} (`user/x/newmodel-bringup`)" in standing


def test_correctness_and_bring_up_lines_share_a_file_and_old_lines_are_correctness(tmp_path):
    (tmp_path / c.MILESTONE_FILENAME).write_text(
        json.dumps({"sha": "a" * 40, "marker": "v0"}) + "\n"
    )
    bringup = c.Milestone(sha="b" * 40, marker="v1", tier=c.MILESTONE_BRINGUP)
    c.record_milestone(tmp_path, bringup)
    c.record_milestone(tmp_path, c.Milestone(sha="c" * 40, marker="v2"))
    assert [m.sha[0] for m in c.milestones(tmp_path)] == ["a", "c"]
    assert [m.sha[0] for m in c.milestones(tmp_path, None)] == ["a", "b", "c"]
    assert c.latest_milestone(tmp_path).sha[0] == "c"
    assert c.bringup_milestone(tmp_path) == bringup
    assert bringup.describe("v9") == f"Bring-up delivered at {'b' * 10}."
    # A correctness milestone already there: a resumed run names no bring-up after it.
    stage = _Bringing(tmp_path, tmp_path / "nowhere")
    assert stage._mark_bringup("v2", _holding(_BOOT, _ACCURACY)) is None


# ----------------------------------------------------------- memory, after a change


def test_the_configuration_with_the_least_memory_free_is_checked_again_first():
    contract = c.Contract(
        checks=(
            c.Check("core", "boot", "tp1", c.TIER_INNER),
            c.Check("mtp", "boot", "tp1", c.TIER_INNER),
            c.Check("mtp", "accuracy[gsm8k]", "tp1", c.TIER_INNER),
            c.Check("mtp", "acceptance", "tp1", c.TIER_PROOF),
            c.Check("mtp", "logits", "tp1", c.TIER_INNER),
            c.Check("mtp", "boot", "tp4", c.TIER_INNER),
        )
    )
    facts = {
        "envelope": {"free_gib": {"least": 1.5, "of": 180.0, "at": "mtp@tp1"}},
        "members": {
            "mtp@tp1": {
                "engines": [
                    {"gate": "boot", "free_gib": 3.0},
                    {"gate": "accuracy[gsm8k]", "free_gib": 1.5},
                    {"gate": "acceptance", "leg": "on", "free_gib": 2.0},
                ]
            }
        },
    }
    assert c.tightest_checks(contract, facts) == (
        "mtp:accuracy[gsm8k]@tp1",
        "mtp:acceptance@tp1",
        "mtp:boot@tp1",
        "mtp:logits@tp1",
    )
    assert c.tightest_checks(contract, {}) == ()
    assert c.tightest_checks(contract, {"envelope": {"free_gib": {"at": "core@tp8"}}}) == ()


@pytest.mark.parametrize(
    "path, relevant",
    [
        ("tensorrt_llm/_torch/modeling_v2/newmodel/model.py", True),
        ("tensorrt_llm/_torch/modeling_v2/newmodel", True),
        ("cpp/kernels/fused.cu", True),
        ("examples/configs/newmodel.yaml", False),
        ("tests/unittest/test_newmodel.py", False),
        ("docs/source/newmodel.md", False),
        ("tensorrt_llm/_torch/modeling_v2/newmodel/README.md", False),
        ("tensorrt_llm/_torch/modeling_v2/newmodel/notes.txt", False),
    ],
)
def test_a_change_may_move_memory_where_an_engine_runs_or_reads_it(path, relevant):
    assert memory_relevant(path) is relevant
    assert memory_relevant(path, denied=("tensorrt_llm/",)) is (
        relevant and not path.startswith("tensorrt_llm/")
    )


def _measured(check, sha):
    """A passing gate result measured on a snapshot of commit ``sha``."""
    return SimpleNamespace(
        name=check.gate, member=check.member, passed=True, detail="", imported=(sha,)
    )


def test_the_bring_up_names_the_commit_its_checks_measured_after_the_checkout_moved_on(tmp_path):
    """Landings while the checks ran do not hold the milestone back: it names what was measured."""
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "user/x/newmodel", str(repo))
    (repo / "tensorrt_llm" / "x.py").write_text("X = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "x")
    measured = _git(repo, "rev-parse", "HEAD")
    (repo / "tensorrt_llm" / "y.py").write_text("Y = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a kernel landed while the checks ran")
    stage = _Bringing(tmp_path / "ws", repo)
    stage.workspace.mkdir(exist_ok=True)
    stage.marker = "v2"
    stage._commit_markers = lambda sha: ("v1", "") if sha == measured else ("v2", "")
    status = c.ContractStatus.load(stage.workspace)
    for check in (_BOOT, _ACCURACY):
        status.record(check.unit, _measured(check, measured), "v1")

    milestone = stage._mark_bringup("v1", _holding(_BOOT, _ACCURACY))

    assert milestone is not None and milestone.sha == measured
    assert _git(repo, "rev-parse", "user/x/newmodel-bringup") == measured


def test_no_bring_up_is_named_when_the_checks_measured_two_commits(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "user/x/newmodel", str(repo))
    (repo / "tensorrt_llm" / "x.py").write_text("X = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "x")
    stage = _Bringing(tmp_path / "ws", repo)
    stage.workspace.mkdir(exist_ok=True)
    stage.marker = "v2"
    stage._commit_markers = lambda sha: ("v1", "")
    status = c.ContractStatus.load(stage.workspace)
    for check, sha in ((_BOOT, "a" * 40), (_ACCURACY, "b" * 40)):
        status.record(check.unit, _measured(check, sha), "v1")

    assert stage._mark_bringup("v1", _holding(_BOOT, _ACCURACY)) is None
