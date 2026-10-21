# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The keep gate is the contract's accuracy gate: its definition, its jobs and its verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates as gates_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.execution.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign
from agent_flow.workflows.trtllm_modeling_v2.tune.keep_gate import (
    KEEP_FILE,
    PART_FILE,
    TESTS_PART,
    KeepGateError,
    default_members,
    resolve_keep_gate,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import BenchSpec, SpoolRunner
from agent_flow.workflows.trtllm_modeling_v2.tune.records import NoiseFloor
from agent_flow.workflows.trtllm_modeling_v2.tune.schema import validate_perf_block
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import open_candidate

from .test_tune_stage import FakeRunner, _contract, _effect, _git, _keep_gate, _repo

UNIFIED = {"parallel": "unified", "members": ["tp1", "tp2", "tep4", "dep4"]}


def _task(tmp_path: Path, **extra) -> dict:
    return {
        "trtllm_repo_path": str(tmp_path),
        "checkpoint_path": "/ckpt",
        **_contract(tmp_path),
        **extra,
    }


# ------------------------------------------------------------------------------ schema


def test_the_campaign_names_unit_and_members_or_turns_the_gate_off():
    def errors(accuracy):
        return validate_perf_block({"perf": {"campaign": {"accuracy": accuracy}}})

    assert errors({"unit": "core", "members": ["tp1", "dep4"]}) == []
    assert (
        errors({"gate": "off"}) == [] and errors({"gate": False}) == []
    )  # YAML reads off as false
    legacy = errors({"task": "gsm8k", "anchor": 96.36, "eval_args": "--x"})
    assert len(legacy) == 3 and all("contract's accuracy gate" in e for e in legacy)
    assert any("accuracy.reference_yaml, model_key and discriminators" in e for e in legacy)
    assert errors({"gate": "maybe"}) == ["perf.campaign.accuracy.gate: on | off"]
    assert any("non-empty list" in e for e in errors({"members": "tp1"}))
    assert any("`gate: off` runs no unit" in e for e in errors({"gate": "off", "members": ["tp1"]}))
    assert errors({"bogus": 1}) == ["perf.campaign.accuracy.bogus: unknown key"]


# ------------------------------------------------------------------------------ definition


def test_the_gate_takes_anchor_benchmark_and_floor_from_the_contract(tmp_path):
    gate = _keep_gate(tmp_path)
    assert (gate.unit, gate.members, gate.reference, gate.benchmark) == (
        "core",
        ("tp1",),
        96.36,
        "gsm8k",
    )
    assert gate.planned_floor["value"] == pytest.approx(95.3286, abs=1e-3)
    assert "n=1319" in gate.planned_floor["rule"] and not gate.ungated
    assert gate.as_record()["anchor"] == 96.36


def test_each_unit_reads_its_own_discriminated_row(tmp_path):
    ref = tmp_path / "rows.yaml"
    ref.write_text(
        yaml.safe_dump(
            {"org/model": [{"accuracy": 90.0}, {"spec_dec_algo": "MTP", "accuracy": 88.0}]}
        )
    )
    accuracy = {
        "benchmark": "gsm8k",
        "reference_yaml": str(ref),
        "model_key": "org/model",
        "discriminators": {"default": {}, "mtp": {"spec_dec_algo": "MTP"}},
    }
    task = {**_task(tmp_path), "accuracy": accuracy, "features": ["mtp"]}
    core = resolve_keep_gate(task, {}, workspace=tmp_path)
    mtp = resolve_keep_gate(task, {"accuracy": {"unit": "mtp"}}, workspace=tmp_path)
    assert (core.reference, mtp.reference) == (90.0, 88.0)


def test_default_members_are_tp1_and_the_widest_least_like_it():
    assert default_members(["tp1", "tp2", "tep2", "dep2", "tp4", "tep4", "dep4"]) == ["tp1", "dep4"]
    assert default_members(["tep2", "tp4", "tep4"]) == ["tep2", "tep4"]
    assert default_members(["tp1"]) == ["tp1"]


def test_members_default_to_the_units_and_named_ones_must_be_the_units(tmp_path):
    assert _keep_gate(tmp_path, **UNIFIED).members == ("tp1", "dep4")
    named = _keep_gate(tmp_path, {"accuracy": {"members": ["tep4"]}}, **UNIFIED)
    assert named.members == ("tep4",)
    with pytest.raises(KeepGateError, match=r"\['tp8'\] are not members of core"):
        _keep_gate(tmp_path, {"accuracy": {"members": ["tp8"]}}, **UNIFIED)
    with pytest.raises(KeepGateError, match="'vision' is not a unit"):
        _keep_gate(tmp_path, {"accuracy": {"unit": "vision"}})


def test_no_anchor_no_gate_unless_it_is_declared_off(tmp_path):
    task = {"trtllm_repo_path": str(tmp_path), "checkpoint_path": "/ckpt", "parallel": "tp1"}
    with pytest.raises(KeepGateError, match="no accuracy.reference_yaml for core"):
        resolve_keep_gate(task, {}, workspace=tmp_path)
    with pytest.raises(KeepGateError, match="does not resolve"):
        resolve_keep_gate(_task(tmp_path, accuracy={**_contract(tmp_path)["accuracy"],
                                                    "model_key": "org/other"}),
                          {}, workspace=tmp_path)  # fmt: skip
    with pytest.raises(KeepGateError, match="remove this key"):
        resolve_keep_gate(_task(tmp_path), {"accuracy": {"anchor": 96.0}}, workspace=tmp_path)
    off = resolve_keep_gate(task, {"accuracy": {"gate": "off"}}, workspace=tmp_path, tests=["t"])
    assert off.ungated and off.parts() == [TESTS_PART] and "ungated" in off.describe()


def test_a_variant_unit_gates_with_its_own_options(tmp_path):
    options = tmp_path / "core.llm_api.yaml"
    options.write_text(yaml.safe_dump({"max_num_tokens": 8192}))
    accuracy = {**_contract(tmp_path)["accuracy"], "llm_api_options": {"core": str(options)}}
    variants = {
        "kv_fp8": {"members": ["tp1"], "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}
    }
    task = _task(tmp_path, accuracy=accuracy, variants=variants)
    gate = resolve_keep_gate(task, {"accuracy": {"unit": "kv_fp8"}}, workspace=tmp_path / "ws")
    job = gate.prepare("tp1", _repo(tmp_path), tmp_path / "out")
    member = tmp_path / "ws" / "variants" / "kv_fp8.tp1.llm_api.yaml"
    assert str(member) in job.script
    written = yaml.safe_load(member.read_text())
    assert written["kv_cache_config"] == {"dtype": "fp8"} and written["modeling_v2"] == "require"


def test_the_campaign_cli_keeps_only_through_a_keep_gate(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.tune import cli

    perf = {"workdir": str(tmp_path / "w"), "model": "m", "identity_configs": [],
            "spools": ["s"], "env_script": "e", "tree": str(tmp_path)}  # fmt: skip
    with pytest.raises(SystemExit, match="accuracy.reference_yaml"):
        cli.keep_gate_for({}, {**perf, "campaign": {"anchor": 96.0}})
    with pytest.raises(SystemExit, match="cannot keep anything"):
        cli.keep_gate_for({}, perf)
    assert cli.keep_gate_for({}, {**perf, "campaign": {"accuracy": {"gate": "off"}}}).ungated
    assert cli.keep_gate_for(_task(tmp_path, perf=perf), perf).reference == 96.36


# ------------------------------------------------------------------------------ jobs and verdicts


def test_a_member_runs_the_contracts_own_command_and_is_judged_by_its_own_floor(tmp_path):
    """The job is the command run_accuracy_gate builds; its verdict is run_accuracy_gate's."""
    options = tmp_path / "core.llm_api.yaml"
    options.write_text(yaml.safe_dump({"kv_cache_config": {"enable_block_reuse": True}}))
    contract = _contract(tmp_path, llm_api_options={"core": str(options)}, max_seq_len=9216,
                         benchmark_args=["--random_seed=0", "--max_output_length=256"])  # fmt: skip
    gate = _keep_gate(tmp_path, **UNIFIED, accuracy=contract["accuracy"])
    tree, out = _repo(tmp_path), tmp_path / "gates" / "dep4"
    job = gate.prepare("dep4", tree, out)
    seen = {}

    def contract_runner(command, env=None, timeout=0):
        seen.update(command=command, timeout=timeout)
        return CommandResult(0, stdout="lm-eval gsm8k average accuracy: 96.0\n")

    gates_mod.run_accuracy_gate(
        replace(gate.environment, world_size=4), task_data={**gate.spec, "trtllm_repo_path": str(tree)},
        reference=96.36, runner=contract_runner, unit="core", member="dep4", workdir=out,
    )  # fmt: skip
    assert job.script == seen["command"] and job.wall == str(seen["timeout"]) and job.gpus == 4
    assert "trtllm-eval" in job.script and "mpirun -n 4" in job.script
    assert "core.dep4.llm_api.yaml" in job.script and "--max_seq_len=9216" in job.script
    assert "--max_output_length=256" in job.script
    low = gate.judge("dep4", tree, out, rc=0, record={"duration_s": 412.0},
                     output="[I] lm-eval gsm8k average accuracy: 94.10\n")  # fmt: skip
    assert low["verdict"] == "FAIL" and low["score"] == 94.1 and low["anchor"] == 96.36
    assert low["floor"] == pytest.approx(95.3286, abs=1e-3) and low["duration_s"] == 412.0
    high = gate.judge("dep4", tree, out, rc=0, record={},
                      output="lm-eval gsm8k average accuracy: 96.20\n")  # fmt: skip
    assert high["verdict"] == "PASS" and "n=1319" in high["floor_rule"]
    lost = gate.judge("dep4", tree, out, rc=212, output="", record={"reason": "orphaned"})
    assert lost["verdict"] == "ERROR" and lost["infrastructure"]
    (out / "command.sh").write_text("trtllm-eval --another-command\n")
    moved = gate.judge("dep4", tree, out, rc=0, record={}, output="accuracy: 96.2")
    assert moved["verdict"] == "ERROR" and "changed" in moved["infrastructure"]


def _suite_tree(tmp_path: Path, test: str) -> Path:
    tree = tmp_path / "tree"
    (tree / "tests" / "unittest" / "t").mkdir(parents=True)
    (tree / "tests" / "unittest" / "t" / "test_block.py").write_text(test)
    return tree


@pytest.mark.parametrize(
    ("test", "verdict", "says"),
    [
        ("def test_block():\n    assert False, 'in_proj'\n", "FAIL", "1 failed"),
        ("import _missing_helper\n\ndef test_x():\n    pass\n", "ERROR", "could not run (rc 2)"),
        ("def test_block():\n    pass\n", "PASS", "1 passed"),
    ],
)
def test_the_suite_runs_from_tests_unittest_and_only_rc_1_is_the_candidates(
    tmp_path, test, verdict, says
):
    """Pytest rc 1 fails the gate; a suite that cannot collect (rc 2) is the tree's environment."""
    environment = {"kind": "local", "python": sys.executable}
    gate = resolve_keep_gate(_task(tmp_path, environment=environment), {"accuracy": {"gate": "off"}},
                             workspace=tmp_path, tests=["tests/unittest/t"])  # fmt: skip
    tree = _suite_tree(tmp_path, test)
    job = gate.prepare(TESTS_PART, tree, tmp_path / "out")
    assert job.gpus == 1 and job.script.startswith("cd tests/unittest && ")
    ran = subprocess.run(["bash", "-c", job.script], cwd=tree, capture_output=True, text=True)
    result = gate.judge(TESTS_PART, tree, tmp_path / "out", rc=ran.returncode,
                        output=ran.stdout + "=== end rc=1 reason=exit ===\n", record={})  # fmt: skip
    assert result["verdict"] == verdict and says in result["detail"], result


# ------------------------------------------------------------------------------ the campaign


def _campaign(tmp_path: Path, gates, keep_gate, msgs: list) -> Campaign:
    """A campaign with one candidate that wins its e2e and goes to the keep gate."""
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=8192, osl=1024,
                      concurrencies=[1, 128], tree=str(repo))  # fmt: skip
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[identity],
                    perf_runner=FakeRunner(["s1"], "camp", _effect), gate_runner=gates,
                    keep_gate=keep_gate, noise=NoiseFloor(spread={"con1.tok_s_user": 0.001}),
                    log=msgs.append)  # fmt: skip
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    wt = open_candidate(repo, work / "wt", "fast-path", "HEAD")
    (wt / "modeling.py").write_text("FAST = True\n")
    _git(wt, "commit", "-q", "-am", "fast")
    decision = tmp_path / "fast-path.yaml"
    decision.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt),
                                        "commits": [_git(wt, "rev-parse", "HEAD")]}))  # fmt: skip
    camp.submit_candidate("fast-path", decision)
    return camp


def _run(camp: Campaign, steps: int = 30) -> None:
    for _ in range(steps):
        if not camp.step():
            break


def test_every_member_passes_or_nothing_is_kept_and_the_verdict_records_why(tmp_path):
    """dep4 scores under the floor while tp1 passes: the candidate is reverted, never kept."""

    class Gates(FakeRunner):
        rounds: list = []

        def submit(self, spool, label, command, **kw):
            self.rounds.append(label)
            return super().submit(spool, label, command, **kw)

    gates = Gates(["g1", "g2"], "campg", _effect,
                  score=lambda label, tree: 94.0 if label.endswith("-dep4") else 96.5)  # fmt: skip
    gate = replace(_keep_gate(tmp_path, **UNIFIED), tests=("tests/unittest/t",))
    msgs: list = []
    camp = _campaign(tmp_path, gates, gate, msgs)
    _run(camp)
    st = camp.state.candidates["fast-path"]
    assert st.status == "reverted" and camp.state.kept == []
    assert st.reason.startswith("keep gate FAIL: dep4: measured 94.0000 < 95.3286")
    assert [label.rsplit("-", 1)[1] for label in gates.rounds] == ["dep4", "tp1", "tests"]
    assert {gates.where[label] for label in gates.rounds} == {"g1", "g2"}  # side by side
    key = gates.rounds[0].removeprefix("gate-").rsplit("-", 1)[0]
    keep = json.loads((camp.workdir / "gates" / key / KEEP_FILE).read_text())
    assert keep["verdict"] == "FAIL" and keep["scores"] == {"tp1": 96.5, "dep4": 94.0}
    assert keep["anchor"] == 96.36 and keep["floor"] == pytest.approx(95.3286, abs=1e-3)
    assert keep["members"] == ["tp1", "dep4"] and keep["unit"] == "core"
    assert keep["tree"] == st.eval_tree and keep["commit"] == _git(
        Path(st.eval_tree), "rev-parse", "HEAD"
    )
    gated = [json.loads(line) for line in (camp.workdir / "perf_ledger.jsonl").read_text().splitlines()
             if '"candidate_gate"' in line]  # fmt: skip
    assert gated[0]["score"] == 94.0 and gated[0]["anchor"] == 96.36


def test_a_failed_part_stops_the_others(tmp_path):
    class Gates(FakeRunner):
        cancelled: list = []

        def poll(self):
            held = {n: v for n, v in self.inflight.items() if not v[1].endswith("-tp1")}
            for n in held:
                del self.inflight[n]
            done = super().poll()
            self.inflight.update(held)
            return done

        def cancel(self, label):
            stopped = super().cancel(label)
            if stopped:
                self.cancelled.append(label)
            return stopped

    gates = Gates(["g1", "g2"], "campg", _effect, score=lambda label, tree: 90.0)
    camp = _campaign(tmp_path, gates, _keep_gate(tmp_path, **UNIFIED), [])
    _run(camp)
    assert camp.state.candidates["fast-path"].status == "reverted"
    assert [label.rsplit("-", 1)[1] for label in gates.cancelled] == ["dep4"] and not gates.inflight


def test_a_part_that_reaches_no_verdict_runs_again_then_stands_as_an_error(tmp_path):
    """The substrate's own failure twice, then a pass, keeps; a suite that never collects reverts."""

    class Gates(FakeRunner):
        def __init__(self, *a, fail, **k):
            super().__init__(*a, **k)
            self.fail = fail

        def poll(self):
            out = []
            for name, (spool, label) in list(self.inflight.items()):
                part = label.rsplit("-", 1)[1]
                if self.fail.get(part):
                    rc = self.fail[part].pop(0)
                    del self.inflight[name]
                    out.append((label, spool, rc))
            return out + super().poll()

    for case in ("a", "b"):
        (tmp_path / case).mkdir()
    gate = replace(_keep_gate(tmp_path / "a"), tests=("tests/unittest/t",))
    msgs: list = []
    camp = _campaign(
        tmp_path / "a", Gates(["g1"], "campg", _effect, fail={"tp1": [212, 212]}), gate, msgs
    )
    _run(camp)
    assert camp.state.candidates["fast-path"].status == "kept"
    attempts = sorted(p.name for p in (camp.workdir / "gates").glob("*/tp1/attempt-*.json"))
    assert attempts == ["attempt-1.json", "attempt-2.json"]
    assert sum("reached no verdict" in m for m in msgs) == 2

    camp = _campaign(
        tmp_path / "b", Gates(["g1"], "campg", _effect, fail={"tests": [2, 2, 2]}), gate, []
    )
    _run(camp)
    st = camp.state.candidates["fast-path"]
    assert st.status == "reverted" and st.reason.startswith(
        "keep gate ERROR: tests: target tests could not run"
    )


def test_a_member_no_dev_session_can_hold_is_an_error_not_a_wait(tmp_path):
    class Narrow(FakeRunner):
        def place(self, gpus, wall="1h"):
            return super().place(gpus, wall) if gpus <= 1 else None

        def widest(self):
            return 1

    camp = _campaign(
        tmp_path, Narrow(["g1"], "campg", _effect), _keep_gate(tmp_path, **UNIFIED), []
    )
    _run(camp)
    st = camp.state.candidates["fast-path"]
    assert (
        st.status == "reverted" and "needs 4 GPUs; the widest dev session hands out 1" in st.reason
    )


def _session(path: Path, devices: int, ends: float) -> str:
    """A live session handing out ``devices``, all free, whose allocation ends at ``ends``."""
    path.mkdir(parents=True)
    record = {
        "alive_epoch": time.time(), "watcher": path.name, "pending": [], "ends_at_epoch": ends,
        "devices": [str(i) for i in range(devices)], "free": [str(i) for i in range(devices)],
    }  # fmt: skip
    (path / "status.json").write_text(json.dumps(record))
    return str(path)


def test_gate_jobs_share_a_session_by_device_count(tmp_path):
    ends = time.time() + 86400
    a, b = _session(tmp_path / "dev", 4, ends), _session(tmp_path / "dev-2", 4, ends)
    runner = SpoolRunner([a, b], prefix="tuneg")
    assert runner.place(4) == a and runner.widest() == 4
    runner.inflight["tuneg-gate-k-dep4"], runner.held["tuneg-gate-k-dep4"] = (a, "gate-k-dep4"), 4
    assert runner.place(1) == b  # the 4-GPU member fills dev
    runner.inflight["tuneg-gate-k-tp1"], runner.held["tuneg-gate-k-tp1"] = (b, "gate-k-tp1"), 1
    assert runner.place(1) == b  # tp1 and the suite share dev-2
    narrow = SpoolRunner([_session(tmp_path / "one", 1, ends)], prefix="tuneg")
    assert narrow.place(4) is None and narrow.widest() == 1


def test_a_resumed_gate_adopts_its_parts_and_withdraws_an_unanchored_single_job(tmp_path):
    repo = _repo(tmp_path)
    spool = tmp_path / "dev"
    for sub in ("pending", "running", "done", "logs"):
        (spool / sub).mkdir(parents=True)
    (spool / "watcher.heartbeat").write_text(
        time.strftime("alive=%Y-%m-%dT%H:%M:%SZ\n", time.gmtime())
    )
    key = f"fast-path-{_git(repo, 'rev-parse', '--short=10', 'HEAD')}"
    (spool / "pending" / f"campg-gate-{key}.sh").write_text("#!/bin/bash\n")
    (spool / "running" / f"campg-gate-{key}-tp1.sh").write_text("#!/bin/bash\n")
    work = tmp_path / "camp"
    work.mkdir()
    state = {
        "best_tree": str(repo),
        "candidates": {
            "fast-path": {"slug": "fast-path", "status": "gate", "eval_tree": str(repo)}
        },
    }
    (work / "campaign_state.json").write_text(json.dumps(state))
    msgs: list = []
    gates = SpoolRunner([str(spool)], prefix="campg")
    Campaign(workdir=work, repo=repo, bench=BenchSpec("b", "m", "e", 1, 1, [1], str(repo)),
             base_configs=[], perf_runner=FakeRunner(["s1"], "camp", _effect), gate_runner=gates,
             keep_gate=_keep_gate(tmp_path), noise=NoiseFloor(spread={}), log=msgs.append)  # fmt: skip
    assert (spool / "cancel" / f"campg-gate-{key}").exists()
    assert list(gates.inflight) == [f"campg-gate-{key}-tp1"]
    assert any("withdrew its unanchored single-job gate" in m for m in msgs)


# ------------------------------------------------------------------------------ the stage


def _stage_task(tmp_path: Path, repo: Path, contract: dict, accuracy: dict | None = None) -> dict:
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    campaign = {"max_roadmaps": 1, "kernel_ledger": "off"}
    if accuracy is not None:
        campaign["accuracy"] = accuracy
    return {
        "trtllm_repo_path": str(repo), "checkpoint_path": "/ckpt", **contract,
        "perf": {"identity_configs": [str(identity)], "spools": ["s1"], "env_script": "/env.sh",
                 "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
                 "reference": "none", "noise_repeats": 0, "sol": {"enabled": False},
                 "campaign": campaign},
    }  # fmt: skip


def _recording(runners: list):
    """A runner factory that keeps every runner it makes."""

    def factory(spools, prefix):
        runners.append(FakeRunner(spools, prefix, _effect))
        return runners[-1]

    return factory


def test_a_campaign_without_an_anchor_refuses_to_start(tmp_path):
    repo = _repo(tmp_path)
    spec = TuneSpec.from_task(_stage_task(tmp_path, repo, {"parallel": "tp1"}), tmp_path / "ws")
    runners: list = []
    state: dict = {}
    stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=_recording(runners), poll_s=0, log=lambda m: None)  # fmt: skip
    assert stage.run() == "tune-failed:accuracy"
    assert not any(r.jobs for r in runners) and "gate: off" in state["failure"]
    assert "no accuracy.reference_yaml for core" in state["failure"]


def test_a_campaign_with_the_gate_off_journals_every_keep_as_ungated(tmp_path):
    repo = _repo(tmp_path)
    task = _stage_task(tmp_path, repo, {"parallel": "tp1"}, accuracy={"gate": "off"})
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="tensorrt_llm/_torch/_experimental/m/")

    def dispatch(role, what, prompt):
        if what == "roadmap":
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump([{"slug": "fast-path"}]))
            return ""
        wt = spec.workdir / "worktrees" / what
        (wt / "modeling.py").write_text("FAST = True\n")
        _git(wt, "commit", "-q", "-am", what)
        (spec.workdir / "candidates" / what / "decision.yaml").write_text(
            yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt),
                            "commits": [_git(wt, "rev-parse", "HEAD")]})
        )  # fmt: skip
        return ""

    runners: list = []
    logs: list = []
    stage = TuneStage(spec, {}, dispatch=dispatch, checkpoint=lambda: None,
                      runner_factory=_recording(runners), poll_s=0, log=logs.append)  # fmt: skip
    assert stage.run() == "tuned"
    assert any("KEEP fast-path" in m and "UNGATED" in m for m in logs), logs
    ledger = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    kept = [r for r in ledger if r["kind"] == "candidate_kept"]
    assert kept and kept[0]["ungated"] is True and kept[0]["gate"]["ungated"] is True
    commands = [cmd for r in runners for cmd, _ in r.jobs.values()]
    assert not any("trtllm-eval" in c for c in commands)  # the suite ran, no accuracy gate
    assert any(" -m pytest " in c for c in commands)
    assert [r for r in ledger if r["kind"] == "keep_gate"][0]["ungated"] is True
    part = next((spec.workdir / "gates").glob(f"*/{TESTS_PART}/{PART_FILE}"))
    assert json.loads(part.read_text())["verdict"] == "PASS"


def test_the_suite_holds_as_many_devices_as_the_widest_member(tmp_path):
    """A collective's 4-rank op matrix failed on a 1-device suite job and reverted a WIN."""
    gate = resolve_keep_gate(
        _task(tmp_path, **UNIFIED), {}, workspace=tmp_path, tests=["tests/unittest/t"]
    )
    width = max(int(m[-1]) for m in gate.members)
    assert width == 4
    assert gate.needs(TESTS_PART)[0] == width
    assert gate.prepare(TESTS_PART, tmp_path, tmp_path / "out").gpus == width
