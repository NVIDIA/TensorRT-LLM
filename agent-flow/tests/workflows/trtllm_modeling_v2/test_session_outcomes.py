# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a session's outcome is to a machine: the ledger, the gate's JSON, its exit status."""

from __future__ import annotations

import json
from dataclasses import replace

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli, session
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

from .conftest import engine_definitions

MOVES = {"cache_transceiver_config": {"backend": "DEFAULT"}}
PD = {"name": "pd", "ctx": {"options": MOVES}, "gen": {"options": MOVES}}


def _gate(task_file, task, tmp_path, block, *more):
    data = {**yaml.safe_load(task_file.read_text()), **task, "deployments": [block]}
    task_file.write_text(yaml.safe_dump(data))
    return gate_cli.main(
        ["deployment", "--task", str(task_file), "--tree", task["trtllm_repo_path"]]
        + ["--workdir", str(tmp_path / "gate"), *more]
    )


def test_not_verified_is_a_third_outcome_in_the_ledger(run_session, task, tmp_path):
    engine_definitions(task["trtllm_repo_path"], None)  # the tree offers no way to see the hop
    result, data = run_session(PD)
    own = result.result
    assert own.passed and own.unverified.startswith("hop: ")
    status = contract.ContractStatus.load(tmp_path / "ledger")
    status.record("pd", own, marker="m")
    row = json.loads(status.path.read_text().splitlines()[-1])
    # The line did not fail, and it says what it could not verify.
    assert row["passed"] and row["unverified"] == own.unverified
    read = contract.ContractStatus.load(tmp_path / "ledger")
    evaluated = contract.evaluate(contract.compile_contract(data), read, marker="m")
    held = [check.key for check, _ in evaluated.unverified]
    assert held == [f"pd:deployment@{own.member}"]
    assert not evaluated.holds and not evaluated.failed
    assert held[0] not in {check.key for check in (*evaluated.passed, *evaluated.missing)}
    assert "1 not verified" in evaluated.render()
    assert f"- NOT VERIFIED {held[0]}: hop: " in evaluated.render()
    # Partial is nothing failed, nothing missing, something not verified.
    only = contract.Evaluation((), (), (), evaluated.unverified)
    assert only.partial and not only.holds
    assert not contract.Evaluation((), evaluated.unverified, (), evaluated.unverified).partial
    # A line that failed is failed, whatever else it says.
    failed = status.record("pd", replace(own, passed=False))
    assert not failed.passed and failed.unverified == ""


def test_not_verified_is_a_third_outcome_of_the_gate(task_file, task, tmp_path, capsys):
    engine_definitions(task["trtllm_repo_path"], None)
    assert _gate(task_file, task, tmp_path, PD, "--json") == gate_cli.EXIT_STATUS["partial"] == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "partial" and payload["passed"] is False
    assert payload["unverified"].startswith("hop: ")
    assert payload["lines"][-1]["passed"] and payload["lines"][-1]["unverified"]
    assert _gate(task_file, task, tmp_path, PD) == 3
    said = capsys.readouterr().out
    assert "NOT VERIFIED" in said and "partial (exit 3): nothing failed" in said
    # A pass and a failure say so the same way.
    engine_definitions(task["trtllm_repo_path"])
    assert _gate(task_file, task, tmp_path, PD, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["status"], payload["passed"], payload["unverified"]) == ("pass", True, "")


def test_a_requirement_the_spec_states_is_used_as_stated(run_session, task, tmp_path):
    assert deployments.requirement({}) == deployments.requirement(None) == "require"
    assert deployments.requirement({"modeling_v2": "auto"}) == "auto"
    # YAML reads a bare `off` as false.
    assert deployments.requirement(yaml.safe_load("modeling_v2: off")) == "off"
    result, data = run_session({"name": "core", "options": {"modeling_v2": "off"}})
    directory = tmp_path / "sessions"
    plan = json.loads(next(directory.rglob("session.json")).read_text())
    # The options the server read.
    assert yaml.safe_load(open(plan["instances"][0]["options"]))["modeling_v2"] == "off"
    lines = {line.name: line for line in result.lines}
    said = (
        "judged under `modeling_v2: off`, as the spec states it (the flow's default is `require`)"
    )
    assert lines["attest"].passed and lines["attest"].detail.endswith(said)
    # The rank ran the stated value: the one option it is held to matches.
    attested = lines["attest"].engine[0]
    assert (attested["keys"], attested["status"], attested["mismatches"]) == (1, "match", ())
    assert result.result.passed and said in result.result.warnings
    # Two roles may state two requirements: each server reads its own options.
    block = {"name": "pd", "ctx": {"options": MOVES}}
    block["gen"] = {"options": {**MOVES, "modeling_v2": "auto"}}
    mixed = deployments.named({**data, "deployments": [block]})
    assert deployments.requirement_said(mixed).startswith(
        "judged under `modeling_v2: require` for `ctx`, `modeling_v2: auto` for `gen`, as"
    )
    written = session.write_options(mixed, tmp_path / "mixed")
    stated = {
        role: yaml.safe_load(path.read_text())["modeling_v2"] for role, path in written.items()
    }
    assert stated == {"ctx": "require", "gen": "auto"}
    # Unsaid, it is `require`, and no line says more.
    result, _ = run_session({"name": "core", "options": {}})
    assert result.result.passed and not result.result.warnings
    assert "judged under" not in {line.name: line for line in result.lines}["attest"].detail


def test_a_relative_score_is_the_ledgers_for_the_same_tree_only(tmp_path):
    spec = {
        "deployments": [
            {"name": "base", "options": {}, "benchmarks": [{"name": "gsm8k", "anchor": 41.0}]},
            {
                "name": "cand",
                "options": {},
                "relative_to": "base",
                "benchmarks": [{"name": "gsm8k"}],
            },
        ]
    }
    found = deployments.resolve(spec)
    base, cand = found
    ledger = tmp_path / "contract_status.jsonl"

    def row(state, measured, passed=True, **more):
        said = {"unit": "base", "gate": "accuracy[gsm8k]", "member": base.shape, **more}
        said.update(passed=passed, measured=measured, tree_state=state)
        with ledger.open("a") as handle:
            handle.write(json.dumps(said) + "\n")

    def read(state):
        return gate_cli.relative_scores(ledger, cand, state, found)

    assert read("abc") == {}
    row("abc", 41.5)
    row("abc", 30.0, gate="boot")
    assert read("abc") == {"gsm8k": 41.5} and read("def") == {}
    # A later row of another tree does not hide this tree's, and each tree has its own.
    row("def", 40.0)
    assert read("abc") == {"gsm8k": 41.5} and read("def") == {"gsm8k": 40.0}
    # A later row of the same tree that did not pass is no score to hold to.
    row("abc", 12.0, passed=False)
    assert read("abc") == {}
