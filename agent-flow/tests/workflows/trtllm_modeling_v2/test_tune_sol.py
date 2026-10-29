# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public SOL decisions: reproducible input, unknowns and no optimistic early stop."""

import json
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune import sol
from agent_flow.workflows.trtllm_modeling_v2.tune.records import Measurement
from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage


def _inputs():
    # Deliberately tiny synthetic capacities, not any device's specifications.
    work = {"flops": 4, "bytes": 1, "assumptions": "synthetic same-scope work"}
    return {
        "hardware": {
            "compute_flops_s": {"value": 2000, "source": "synthetic arithmetic fixture"},
            "memory_bytes_s": {"value": 1000, "source": "synthetic arithmetic fixture"},
            "assumptions": "one synthetic processor, dense arithmetic, explicit memory boundary",
        },
        "regions": [
            {
                "id": "layer/decode/shape/rank0",
                **work,
                "measured_ms": 8,
                "measurement_source": "synthetic timing",
            }
        ],
        "headline": {
            "con1.tok_s_user": {**work, "output_tokens": 1},
            "con1.output_tok_s_gpu": {**work, "output_tokens": 1, "gpu_count": 1},
        },
    }


def _baseline(value=400):
    return Measurement("base", Path("."), {1: {"tok_s_user": value, "output_tok_s_gpu": value}}, {})


def test_request_command_produces_valid_output_and_goal(tmp_path):
    spec = TuneSpec.from_task({"perf": {}}, tmp_path, target="target")
    stage = TuneStage(spec, {}, dispatch=lambda *args: "", checkpoint=lambda: None)
    request = stage.requests.sol_request(_baseline()).read_text()
    command = request.split("```sh\n", 1)[1].split("\n```", 1)[0]
    args = shlex.split(command)
    Path(args[args.index("--input") + 1]).write_text(yaml.safe_dump(_inputs()))
    subprocess.run([sys.executable, *args[1:]], check=True)
    output = Path(args[args.index("--output") + 1])
    ceilings, errors = sol.load_sol(output, _baseline())
    assert not errors and ceilings.reached(_baseline(), 0.8)
    assert ceilings.regions[0]["headroom_ms"] == 6
    assert not ceilings.reached(_baseline(501), 0.8)  # contradicting bounds are not success
    assert "internal-perf-sol-analysis" not in request + stage.requests.sol_ask("roadmap")
    # The same entry point serves the roadmap's stable-id table as JSON.
    region_output = tmp_path / "roadmap-sol.json"
    subprocess.run(
        [sys.executable, args[1], "--input", args[3], "--output", str(region_output)], check=True
    )
    assert json.loads(region_output.read_text())["regions"] == ceilings.regions
    raw = yaml.safe_load(output.read_text())
    raw["ceilings"]["con1.tok_s_user"]["value"] *= 2
    output.write_text(yaml.safe_dump(raw))
    assert sol.load_sol(output, _baseline())[0] is None


def test_missing_inputs_never_fabricate_bounds_or_stop(tmp_path):
    output = tmp_path / "sol.yaml"

    def never_reached(result):
        output.write_text(yaml.safe_dump(result))
        ceilings, errors = sol.load_sol(output, _baseline())
        assert not errors and not ceilings.reached(_baseline(), 0.01)
        return ceilings

    inputs = _inputs()
    inputs["hardware"].pop("compute_flops_s")
    result = sol.calculate(inputs)
    assert not result["ceilings"] and result["regions"][0]["headroom_ms"] is None
    assert "unknown" in sol.render_table(never_reached(result), [_baseline()])
    inputs = _inputs()
    inputs["headline"].pop("con1.output_tok_s_gpu")
    never_reached(sol.calculate(inputs))
    inputs = _inputs()
    inputs["headline"]["con1.tok_s_user"]["overhead"] = {
        "ms": 1,
        "source": "synthetic trace",
        "assumption": "exposed and unchanged",
    }
    ceilings = never_reached(sol.calculate(inputs))
    assert "conditional" in ceilings.unknown["con1.tok_s_user"]


def test_independent_regions_and_conditional_overhead():
    inputs = _inputs()
    parent = inputs["regions"][0]
    child = {**parent, "id": parent["id"] + "/child", "flops": 1, "bytes": 3}
    inputs["regions"].append(child)
    child["overhead"] = {
        "ms": 1,
        "source": "synthetic target-local trace",
        "assumption": "exposed serial overhead, assumed unchanged",
    }
    rows = sol.calculate(inputs)["regions"]
    assert [(r["bound"], r["lower_bound_ms"], r["headroom_ms"]) for r in rows] == [
        ("compute", 2, 6),
        ("memory", 4, 4),
    ]
    assert rows[1]["conditional"] and not rows[0]["conditional"]
    assert sol.calculate(inputs)["ceilings"]["con1.tok_s_user"]["value"] == 500
    child["id"] = parent["id"]
    with pytest.raises(ValueError, match="duplicate"):
        sol.calculate(inputs)


@pytest.mark.parametrize(
    "location,value",
    [
        (("hardware", "compute_flops_s", "value"), float("nan")),
        (("hardware", "memory_bytes_s", "value"), 0),
        (("regions", 0, "flops"), -1),
        (("regions", 0, "bytes"), float("inf")),
        (("regions", 0, "measured_ms"), 1),  # a measured time below the theoretical bound
        (("regions", 0, "measured_ms"), True),
        (("headline", "con1.output_tok_s_gpu", "gpu_count"), 0.5),
        (("hardware", "compute_flops_s", "source"), ""),
    ],
)
def test_invalid_or_impossible_inputs_are_rejected(location, value):
    inputs = deepcopy(_inputs())
    item = inputs
    for key in location[:-1]:
        item = item[key]
    item[location[-1]] = value
    with pytest.raises(ValueError):
        sol.calculate(inputs)


def _analysis(user=4.0, gpu=4.0, memory=1000, region_bytes=1, counted=None):
    """``_inputs`` with each headline's FLOPs, the memory capacity and the region's bytes set."""
    inputs = deepcopy(_inputs())
    inputs["headline"]["con1.tok_s_user"]["flops"] = user
    inputs["headline"]["con1.output_tok_s_gpu"]["flops"] = gpu
    inputs["hardware"]["memory_bytes_s"]["value"] = memory
    inputs["regions"][0]["bytes"] = region_bytes
    if counted is not None:
        inputs["regions"][0]["measured_bytes"] = counted
    return sol.calculate(inputs)


def test_analyses_that_agree_give_the_highest_ceiling_and_the_rest_is_ruled_or_unknown():
    from agent_flow.workflows.trtllm_modeling_v2.tune.sol_consensus import read_rulings, reconcile

    base = _baseline().headline()  # 400 tok/s at both
    # 2 ms (500 tok/s) and 2.2 ms agree; 3.5 ms is below what the baseline measured.
    analyses = {"analytic": _analysis(4), "code": _analysis(4.4, 8), "profile": _analysis(7, 8)}
    c = reconcile(analyses, base)
    user = c.report["headline.con1.tok_s_user"]
    assert user["status"] == "agreed" and user["chosen"] == "analytic"
    assert user["below_baseline"] == ["profile"]
    assert c.inputs["headline"]["con1.tok_s_user"]["assumptions"].startswith("analytic: ")
    assert c.disputes == [{"item": "headline.con1.output_tok_s_gpu", "values": {"analytic": 500.0},
                           "why": "unconfirmed: only analytic models it"}]  # fmt: skip
    assert c.report["regions"]["agreed"] == 1 and len(c.inputs["regions"]) == 1
    rulings, problems = read_rulings({"rulings": [
        {"item": "headline.con1.output_tok_s_gpu", "analysis": "analytic", "evidence": "recounted"},
        {"item": "headline.con1.tok_s_user", "analysis": "code", "evidence": "x"},
        {"item": "headline.con1.output_tok_s_gpu", "analysis": "code", "evidence": "x"},
    ]}, c.disputes)  # fmt: skip
    assert rulings == {"headline.con1.output_tok_s_gpu": "analytic"} and len(problems) == 2
    ruled = reconcile(analyses, base, rulings=rulings)
    assert sol.calculate(ruled.inputs)["ceilings"]["con1.output_tok_s_gpu"]["value"] == 500
    # Ceilings 33 % apart are a dispute; an unresolved one is unknown, never a goal.
    apart = reconcile({"analytic": _analysis(4), "code": _analysis(3)}, base)
    assert [d["item"] for d in apart.disputes] == ["headline.con1.tok_s_user"]
    assert "differ by 33%" in apart.unknown["con1.tok_s_user"]
    assert "con1.tok_s_user" not in apart.inputs["headline"]


def test_capacities_must_agree_and_counted_traffic_bounds_every_analysis():
    from agent_flow.workflows.trtllm_modeling_v2.tune.sol_consensus import reconcile

    base = _baseline().headline()
    analyses = {"analytic": _analysis(), "code": _analysis(memory=1100)}
    c = reconcile(analyses, base)
    assert c.disputes[0]["item"] == "hardware.memory_bytes_s" and c.inputs["regions"] == []
    c = reconcile(analyses, base, rulings={"hardware.memory_bytes_s": "analytic"})
    assert not c.disputes and c.inputs["hardware"]["memory_bytes_s"]["value"] == 1000
    # A profiler counted 1.2 bytes in the region: 2 bytes of necessary work is impossible.
    counted = {
        "analytic": _analysis(region_bytes=2),
        "profile": _analysis(region_bytes=1, counted=1.2),
    }
    c = reconcile({**counted, "code": _analysis()}, base)
    assert c.report["regions"]["over_counted"] == [
        {"id": "layer/decode/shape/rank0", "analyses": ["analytic"]}
    ]
    assert c.inputs["regions"][0]["bytes"] == 1
    with pytest.raises(ValueError, match="exceed what the profiler counted"):
        _analysis(region_bytes=2, counted=1.2)


def test_the_stage_asks_each_method_once_and_one_adjudication_for_what_they_dispute(tmp_path):
    from dataclasses import replace

    from .test_tune_stage import FakeRunner, _effect, _write_sol
    from .test_tune_wall_budget import _campaign

    spec = replace(_campaign(tmp_path), sol={"enabled": True})
    turns = []

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "sol-adjudicate":
            rulings = [{"item": "headline.con1.tok_s_user", "analysis": "code", "evidence": "re-counted"},
                       {"item": "headline.con128.tok_s_user", "analysis": None, "evidence": "none holds"}]  # fmt: skip
            (spec.workdir / "sol" / "rulings.yaml").write_text(yaml.safe_dump({"rulings": rulings}))
        else:  # the profile counts twice the work the other two do
            method = what.removeprefix("sol-")
            _write_sol(spec.workdir / "sol" / method, work=20 if method == "profile" else 10)
        return "done"

    stage = TuneStage(spec, {}, dispatch=dispatch, checkpoint=lambda: None, poll_s=0,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect))  # fmt: skip
    assert stage.ground()
    stage._sol()
    assert sorted(turns) == ["sol-adjudicate", "sol-analytic", "sol-code", "sol-profile"]
    ceilings = stage._ceilings()
    assert set(ceilings.ceilings) == {"con1.tok_s_user"}  # ruled; the others stay unknown
    assert "no analysis's work model holds" in ceilings.unknown["con128.tok_s_user"]
    assert "differ by 100%" in ceilings.unknown["con128.output_tok_s_gpu"]
    report = json.loads((spec.workdir / "sol" / "consensus.json").read_text())
    assert report["headline.con1.tok_s_user"]["status"] == "ruled"
    stage.t.pop("sol")
    stage._sol()  # answered on disk: no turn is asked again
    assert len(turns) == 4
