# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The verification policy: one measured A/B, fail closed, and fresh final e2e."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage

from .test_tune_keep_gate import _campaign, _run
from .test_tune_stage import FakeRunner, _effect, _git, _keep_gate, _microbench
from .test_tune_wall_budget import _campaign as _spec


def test_wrapper_times_callables_not_claimed_numbers(tmp_path, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import microbench

    script = Path(_microbench(tmp_path)["script"])
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "microbench.py",
            "--script",
            str(script),
            "--output",
            str(output),
            "--tree",
            str(tmp_path),
            "--commit",
            "revision",
            "--harness-sha256",
            digest,
        ],
    )
    identity = Mock()
    monkeypatch.setattr(microbench, "_identity", identity)
    monkeypatch.setattr(
        microbench.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(tmp_path / "tensorrt_llm/__init__.py")),
    )
    event = Mock(elapsed_time=Mock(return_value=0.125))
    cuda = SimpleNamespace(Event=Mock(return_value=event), synchronize=Mock())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    operation = Mock()
    monkeypatch.setattr(
        microbench.runpy, "run_path", lambda path: {"benchmark": lambda: {"case": operation}}
    )
    microbench.main()
    assert operation.call_count == 110 and cuda.synchronize.call_count == 2
    assert identity.call_count == 2
    assert microbench.load_result(output, commit="revision", harness=digest, cases=["case"])[
        "latency_us"
    ] == {"case": 125.0}
    monkeypatch.setattr(
        microbench.runpy, "run_path", lambda path: {"benchmark": lambda: {"case": 0.001}}
    )
    with pytest.raises(ValueError, match="not claimed timings"):
        microbench.main()


def test_local_keeps_stack_once_without_serving_or_stale_goal(tmp_path):
    camp = _campaign(
        tmp_path, FakeRunner(["g"], "g", _effect), [], keep_gate=_keep_gate(tmp_path),
        checkpoint_gain=1.0,  # no checkpoint between them: each keeps on its micro-bench
    )  # fmt: skip
    first = camp.state.candidates["fast-path"]
    # A second independent candidate is already ready while the first is verified.
    side = tmp_path / "side-tree"
    _git(
        Path(first.worktree), "worktree", "add", "-q", "--detach", str(side), camp.state.best_commit
    )
    (side / "side.py").write_text("SIDE = True\n")
    _git(side, "add", "side.py")
    _git(side, "commit", "-q", "-m", "side")
    decision = tmp_path / "side.yaml"
    decision.write_text(
        yaml.safe_dump(
            {
                "decision": "CANDIDATE",
                "worktree": str(side),
                "commits": [_git(side, "rev-parse", "HEAD")],
                "microbench": _microbench(tmp_path / "side-bench"),
            }
        )
    )
    camp.submit_candidate("side", decision)
    camp.perf.effect = lambda config, tree: (
        100.0
        + 20 * ("FAST = True" in Path(tree, "modeling.py").read_text())
        + 20 * Path(tree, "side.py").exists(),
        2000.0,
    )
    _run(camp)
    assert camp.state.kept == ["fast-path", "side"]
    assert len(camp.perf.submitted) == len(set(camp.perf.submitted)) == 4
    assert all(label.startswith("micro-") for label in camp.perf.submitted)
    assert camp.state.candidates["side"].eval_base != first.eval_base
    # No old e2e/baseline can stand in for this cumulative best.
    (tmp_path / "stage").mkdir()
    spec = replace(_spec(tmp_path / "stage"), workdir=camp.workdir, goal_ratio=0.1)
    stage = TuneStage(spec, {}, dispatch=lambda *args: "", checkpoint=lambda: None)
    assert stage._best_measurement(camp) is None and not stage._goal_met(camp)


@pytest.mark.parametrize(
    "bad", ["missing", "nan", "cases", "commit", "harness", "receipt", "host", "import"]
)
def test_invalid_local_evidence_never_keeps_or_repeats(tmp_path, bad):
    camp = _campaign(tmp_path, FakeRunner(["g"], "g", _effect), [], keep_gate=_keep_gate(tmp_path))
    original = camp.perf.poll

    def poll():
        done = original()
        for label, _, _ in done:
            path = camp.data / label / "result.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            if bad == "missing":
                path.unlink()
                continue
            if bad == "nan":
                data["latency_us"] = {"decode-b1-bf16-graph": float("nan")}
            elif bad == "cases":
                data["latency_us"] = {"unrelated": 0.1}
            elif bad == "commit":
                data["commit"] = "wrong"
            elif bad == "harness":
                data["harness_sha256"] = "wrong"
            elif bad == "receipt":
                camp.perf.receipts[label]["rc"] = 1
            elif bad == "host":
                data["hostname"] = "other"
            elif bad == "import":
                data["module"] = "/other/tensorrt_llm/__init__.py"
            path.write_text(json.dumps(data))
        return done

    camp.perf.poll = poll
    _run(camp)
    assert camp.state.candidates["fast-path"].status == "blocked"
    assert not camp.state.kept and len(camp.perf.submitted) == 1
    assert not camp.gates.submitted


def test_system_candidate_is_single_serving_ab_and_scope_is_checked(tmp_path):
    camp = _campaign(tmp_path, FakeRunner(["g"], "g", _effect), [], keep_gate=_keep_gate(tmp_path))
    decision = tmp_path / "fast-path.yaml"
    data = yaml.safe_load(decision.read_text())
    data["kind"] = "overlap"
    decision.write_text(yaml.safe_dump(data))
    assert camp.submit_candidate("fast-path", decision).status == "blocked"
    data["perf_scope"] = "e2e"
    decision.write_text(yaml.safe_dump(data))
    camp.submit_candidate("fast-path", decision)
    _run(camp)
    assert camp.state.kept == ["fast-path"]
    assert len(camp.perf.submitted) == 2
    assert sum(label.startswith("ctl-") for label in camp.perf.submitted) == 1
    assert sum(label.startswith("e2e-") for label in camp.perf.submitted) == 1


def _grounded(spec):
    stage = TuneStage(
        spec,
        {},
        dispatch=lambda *args: "",
        checkpoint=lambda: None,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
        poll_s=0,
    )
    assert stage.ground()
    return stage


def test_the_baseline_is_measured_once_and_must_serve_the_target(tmp_path):
    stage = _grounded(_spec(tmp_path))
    assert stage.perf.submitted == ["t-baseline"]


@pytest.mark.parametrize("outcome", ["missing", "stale", "regression"])
def test_final_failure_is_not_reported_as_tuned(tmp_path, outcome):
    spec = _spec(tmp_path)
    stage = _grounded(spec)
    original = stage._measure

    def measure(labels, tree=None, **kw):
        result = original(labels, tree, **kw)
        final = result["t-final"]
        if outcome == "missing":
            return {"t-final": None}
        if outcome == "stale":
            final.meta["tree"]["commit"] = "old"
        else:
            for row in final.points.values():
                row["tok_s_user"] *= 0.5
                row["output_tok_s_gpu"] *= 0.5
        return result

    stage._measure = measure
    stage.t["kept"] = ["fast-path"]  # nothing kept: the baseline stands, measured once
    assert stage.wrap() == "tune-failed:final"
    assert stage.t["phase"] != "done" and not stage.t["goal_reached"]
    assert stage.t["failure"]
    assert stage.perf.submitted.count("t-final") == 1
    if outcome != "regression":
        assert "| t-final |" not in (spec.workdir / "summary.md").read_text()
