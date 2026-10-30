# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration variants, and members of different widths under a scheduler."""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.checks import gates as gates_mod
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec import variants as V


def _spec(tmp_path, **over) -> dict:
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
    spec.update(over)
    return spec


def _load(tmp_path, spec) -> dict:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task_schema.load_and_validate_task_yaml(path)


def test_members_of_different_widths_are_legal_under_a_scheduler_only(tmp_path):
    data = _load(tmp_path, _spec(tmp_path))
    assert task_schema.widest_world_size(data, 4) == 4
    local = _spec(tmp_path)
    local["environment"] = {**local["environment"], "kind": "local"}
    local["environment"].pop("spool")
    with pytest.raises(task_schema.TaskSchemaError, match="mixes world sizes"):
        _load(tmp_path, local)


def test_every_variant_problem_is_named(tmp_path):
    spec = _spec(tmp_path)
    spec["variants"] = {
        "core": {"runtime": {"cuda_graph": False}},
        "Bad-Name": {"runtime": {"cuda_graph": False}},
        "nothing": {"note": "changes nothing"},
        "orphan": {"unit": "eagle3", "runtime": {"cuda_graph": False}},
        "spec": {"gates": ["acceptance"], "llm_api_options": {"a": 1}},
        "typo": {"runtme": {}, "env": {"A": "1"}, "members": ["tq4"]},
    }
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        _load(tmp_path, spec)
    text = str(excinfo.value)
    assert "'core' is a unit name" in text
    assert "variants.Bad-Name" in text
    assert "'variants.nothing' changes nothing" in text
    assert "not one of the units" in text
    assert "implements no speculator" in text
    assert "runtme" in text and "variants.typo.members" in text


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
    spec = V.variant_spec(data, "mtp7", world_size=4, options_dir=tmp_path / "ws" / "variants")
    accuracy = spec["accuracy"]
    options = yaml.safe_load(open(gates_mod.resolve_llm_api_options(accuracy, "mtp7")))
    assert options["speculative_config"] == {"decoding_type": "MTP", "max_draft_len": 7}
    assert gates_mod.resolve_max_draft_len(accuracy, "mtp7") == 7
    assert gates_mod.resolve_max_draft_len(accuracy, "mtp") == 3  # the unit is untouched
    assert gates_mod.resolve_discriminators(accuracy, "mtp7") == {"spec_dec_algo": "MTP"}
    assert task_schema.members_for(spec, "mtp7", 4) == ["tep4"]
    assert spec["environment"]["env"]["X"] == "1"
    assert gate_cli.features_for(spec, "mtp7") == ["mtp"]


def test_a_variant_appends_its_flags_and_overrides_its_anchor_row(tmp_path):
    data = _load(tmp_path, _spec(tmp_path))
    spec = V.variant_spec(data, "kv_fp8", world_size=4, options_dir=tmp_path / "v")
    runtime = gates_mod.resolve_runtime(spec, "kv_fp8")
    assert runtime["cuda_graph"] is True
    assert runtime["extra_flags"] == ["--max_seq_len=9216", "--kv_cache_dtype=fp8"]
    assert gates_mod.resolve_runtime(spec, "core")["extra_flags"] == ["--max_seq_len=9216"]
    accuracy = spec["accuracy"]
    assert gates_mod.resolve_discriminators(accuracy, "kv_fp8") == {"kv_cache_quant_algo": "FP8"}
    assert gates_mod.resolve_discriminators(accuracy, "core") == {}
    assert yaml.safe_load(open(gates_mod.resolve_llm_api_options(accuracy, "kv_fp8"))) == {
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
    from agent_flow.workflows.trtllm_modeling_v2.infra.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(
        kind="session_spool", python="python3", world_size=4, spool=str(tmp_path / "spool")
    )
    workflow.task_data = {"trtllm_repo_path": str(tmp_path)}
    assert workflow._member_env("tp1").world_size == 1
    assert "mpirun -n 2" in workflow._member_env("tep2").command("python x.py")
    assert "mpirun" not in workflow._member_env("tp1").command("python x.py")
    assert workflow._member_runner("tp1") is workflow._member_runner("tp1")
    workflow._member_runner("tep2")
    workflow._member_runner("tep2", "all")
    assert sorted(workflow._member_runners) == ["1", "2", "all"]
