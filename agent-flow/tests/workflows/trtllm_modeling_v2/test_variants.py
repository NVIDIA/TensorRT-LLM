# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration variants, and members of different widths under a scheduler."""

from __future__ import annotations

import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config, gate_cli
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema


def _spec(tmp_path) -> dict:
    checkpoint, repo = tmp_path / "ckpt", tmp_path / "repo"
    checkpoint.mkdir(exist_ok=True)
    repo.mkdir(exist_ok=True)
    (tmp_path / "spool").mkdir(exist_ok=True)
    (tmp_path / "mtp.yaml").write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}})
    )
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "unified",
        "members": {"default": ["tp1", "tep2", "tep4"], "mtp": ["tep4"]},
        "trtllm_repo_path": str(repo),
        "features": ["mtp"],
        "environment": {
            "kind": "session_spool",
            "spool": str(tmp_path / "spool"),
            "python": "/usr/bin/python3",
            "env": {"TMPDIR": "/scratch/tmp"},
        },
        "runtime": {"cuda_graph": True, "extra_flags": ["--max_seq_len=9216"]},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "llm_api_options": {"mtp": str(tmp_path / "mtp.yaml")},
            "discriminators": {"default": {}, "mtp": {"spec_dec_algo": "MTP"}},
        },
        "variants": {
            "kv_fp8": {
                "members": ["tp1"],
                "runtime": {"extra_flags": ["--kv_cache_dtype=fp8"]},
                "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
                "accuracy": {"discriminators": {"kv_cache_quant_algo": "FP8"}},
            },
            "mtp7": {
                "unit": "mtp",
                "gates": ["boot", "accuracy", "acceptance"],
                "llm_api_options": {"speculative_config": {"max_draft_len": 7}},
                "env": {"X": "1"},
            },
        },
    }
    return spec


def _load(tmp_path, spec) -> dict:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task_schema.load_and_validate_task_yaml(path)


def test_variant_checks_are_named_by_the_variant_at_its_own_members(tmp_path):
    data = _load(tmp_path, _spec(tmp_path))
    contract = contract_mod.compile_contract(data, world_size=4)
    proof = [c.key for c in contract.by_tier(contract_mod.TIER_PROOF)]
    assert "kv_fp8:boot@tp1" in proof and "kv_fp8:accuracy[gsm8k]@tp1" in proof
    # A storage variant boots at every member of its unit; its accuracy stays where it names.
    tep = {k for k in proof if k.startswith("kv_fp8:") and "@tep" in k}
    assert tep == {"kv_fp8:boot@tep2", "kv_fp8:boot@tep4"}
    # A speculation variant runs every gate only where it names.
    assert {"mtp7:boot@tep4", "mtp7:accuracy[gsm8k]@tep4", "mtp7:acceptance@tep4"} <= set(proof)
    assert not [k for k in proof if k.startswith("mtp7:") and "@tp1" in k]


def test_a_variant_runs_under_its_units_spec_with_its_overlay(tmp_path):
    data = _load(tmp_path, _spec(tmp_path))
    spec = check_config.variant_spec(data, "mtp7", world_size=4, options_dir=tmp_path / "v")
    accuracy = spec["accuracy"]
    options = yaml.safe_load(open(check_config.resolve_llm_api_options(accuracy, "mtp7")))
    assert options["speculative_config"] == {"decoding_type": "MTP", "max_draft_len": 7}
    assert check_config.resolve_max_draft_len(accuracy, "mtp7") == 7
    assert check_config.resolve_max_draft_len(accuracy, "mtp") == 3  # the unit is untouched
    assert accuracy_checks.resolve_discriminators(accuracy, "mtp7") == {"spec_dec_algo": "MTP"}
    assert task_schema.members_for(spec, "mtp7", 4) == ["tep4"]
    assert spec["environment"]["env"]["X"] == "1"
    assert gate_cli.features_for(spec, "mtp7") == ["mtp"]
    # A variant appends its flags and overrides its anchor row.
    spec = check_config.variant_spec(data, "kv_fp8", world_size=4, options_dir=tmp_path / "v")
    runtime = check_config.resolve_runtime(spec, "kv_fp8")
    assert runtime["cuda_graph"] is True
    assert runtime["extra_flags"] == ["--max_seq_len=9216", "--kv_cache_dtype=fp8"]
    assert check_config.resolve_runtime(spec, "core")["extra_flags"] == ["--max_seq_len=9216"]
    accuracy = spec["accuracy"]
    assert accuracy_checks.resolve_discriminators(accuracy, "kv_fp8") == {
        "kv_cache_quant_algo": "FP8"
    }
    assert accuracy_checks.resolve_discriminators(accuracy, "core") == {}
    assert yaml.safe_load(open(check_config.resolve_llm_api_options(accuracy, "kv_fp8"))) == {
        "kv_cache_config": {"dtype": "fp8"}
    }
    assert task_schema.members_for(spec, "kv_fp8", 4) == ["tp1"]
    assert task_schema.members_for(spec, "core", 4) == ["tp1", "tep2", "tep4"]
    assert gate_cli.features_for(spec, "kv_fp8") == []


def test_the_widest_member_counts_variants_and_per_unit_lists(tmp_path):
    data = {"parallel": "unified", "members": {"default": ["tp1"], "mtp": ["tep2"]}}
    assert task_schema.widest_world_size(data, 4) == 2
    data["variants"] = {"v": {"members": ["dep4"]}}
    assert task_schema.widest_world_size(data, 8) == 4
    assert task_schema.widest_world_size({"parallel": "tp2"}, 8) == 2
    assert task_schema.widest_world_size({"parallel": "unified"}, 8) == 8


def test_each_member_is_launched_and_scheduled_at_its_own_width(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(
        kind="session_spool", python="python3", world_size=4, spool=str(tmp_path / "spool")
    )
    workflow.task_data = {"trtllm_repo_path": str(tmp_path)}
    runner = CheckRunner(workflow)
    assert runner._member_env("tp1").world_size == 1
    assert "mpirun -n 2" in runner._member_env("tep2").command("python x.py")
    assert "mpirun" not in runner._member_env("tp1").command("python x.py")
    assert runner._member_runner("tp1") is runner._member_runner("tp1")
    runner._member_runner("tep2")
    one = runner._member_runner("tep2", "all")
    two = runner._member_runner("tep2", "all", nodes=2)
    assert one is not two
    assert two is runner._member_runner("tep2", "all", nodes=2)
    # Keep node count separate when the per-node width and queue class match.
    assert sorted(runner._member_runners) == [
        ("1", 0, 1),
        ("2", 0, 1),
        ("all", 0, 1),
        ("all", 0, 2),
    ]


def test_a_variant_without_a_row_is_judged_against_its_units_score_on_the_same_target(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    spec = _spec(tmp_path)
    del spec["variants"]["kv_fp8"]["accuracy"]  # names no row of its own: relative
    data = _load(tmp_path, spec)
    contract = contract_mod.compile_contract(data, world_size=4)
    check = next(c for c in contract.checks if c.key == "kv_fp8:accuracy[gsm8k]@tp1")
    assert check.after == "core:accuracy[gsm8k]@tp1"
    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(kind="local", python="python3", world_size=1)
    workflow.task_data, workflow._contract = data, contract
    runner = CheckRunner(workflow)
    runner._reference_score = lambda unit, spec=None: 50.0
    calls = []

    def gate(env, *, reference, unit, relative=False, **kwargs):
        calls.append((unit, reference, relative))
        measured = 88.0 if unit == "core" else 87.0
        return SimpleNamespace(measured=measured, infrastructure="", detail="", summary=str)

    monkeypatch.setattr(accuracy_checks, "run_accuracy_gate", gate)
    workflow._contract_status().record(
        "core",
        SimpleNamespace(name=check.gate, member="tp1", passed=True, detail="", measured=90.0),
        marker=workflow._target_marker(),
    )
    runner._gate_call(check)(None)
    assert calls == [("kv_fp8", 90.0, True)]
    # Measured on another state of the target: the unit's benchmark runs here first.
    workflow._contract_status().verdicts.clear()
    calls.clear()
    runner._gate_call(check)(None)
    assert calls == [("core", 50.0, False), ("kv_fp8", 88.0, True)]


def test_a_unit_on_a_placeholder_is_anchored_to_its_first_measured_score(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import anchors
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    data = _load(tmp_path, _spec(tmp_path))
    contract = contract_mod.compile_contract(data, world_size=4)
    check = next(c for c in contract.checks if c.key == "core:accuracy[gsm8k]@tp1")
    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(kind="local", python="python3", world_size=1)
    workflow.task_data, workflow._contract = data, contract
    runner = CheckRunner(workflow)
    runner._reference_score = lambda unit, spec=None: 10.0  # a placeholder
    calls = []

    def gate(env, *, reference, unit, relative=False, **kwargs):
        calls.append((reference, relative))
        return SimpleNamespace(passed=True, measured=93.3, infrastructure="", detail="d",
                               tree_state="abc", summary=str)  # fmt: skip

    monkeypatch.setattr(accuracy_checks, "run_accuracy_gate", gate)
    runner._gate_call(check)(None)
    assert anchors.measured(workflow.workspace, "core", check.gate, "tp1").score == 93.3
    # The next benchmark of core at tp1 is held to that score, not to the placeholder.
    monkeypatch.setattr(
        "agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner.replace",
        lambda result, **changes: {**vars(result), **changes},
    )
    second = runner._gate_call(check)(None)
    assert calls == [(10.0, False), (93.3, True)]
    assert "replaced the placeholder 10" in second["detail"]
