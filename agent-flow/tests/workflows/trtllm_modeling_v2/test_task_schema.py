# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 task schema behavior and regression checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments, derive, task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

from .conftest import SCRATCH


def _valid_spec(tmp_path) -> dict:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    return {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "local",
            "python": "/usr/bin/python3",
            "env": {"TRTLLM_MODELING_V2": "require", "TMPDIR": "/scratch/tmp"},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
        },
    }


def _write(tmp_path, spec) -> str:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(path)


def test_every_problem_is_batched_into_one_message(tmp_path):
    spec = _valid_spec(tmp_path)
    del spec["arch"]
    del spec["environment"]
    spec["checkpoint_path"] = "/definitely/not/here"
    # A disabled gate's fields are still validated.
    spec["accuracy"]["serve"] = {"enabled": False, "max_tokens": True, "prompt": " "}
    spec["accuracy"]["score_pattern"] = "accuracy: [0-9.]+"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(excinfo.value)
    for expected in (
        "arch",
        "environment",
        "non-existent path",
        "accuracy.serve.max_tokens",
        "accuracy.serve.prompt",
        "'accuracy.score_pattern' needs a group around the score",
    ):
        assert expected in message


def test_a_misspelled_key_is_refused_with_its_spelling_at_any_depth(tmp_path):
    # A misspelled `features` must not silently build core only.
    spec = _valid_spec(tmp_path) | {"feature": ["mtp"]}
    spec["accuracy"]["benchmrk"] = "x"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'feature'" in str(excinfo.value)
    assert "did you mean 'features'" in str(excinfo.value)
    assert "did you mean 'accuracy.benchmark'" in str(excinfo.value)


def test_source_restriction_is_not_a_supported_production_task_option(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["source_embargo"] = []
    with pytest.raises(task_schema.TaskSchemaError, match="unknown key 'source_embargo'"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    # The flow has no tuning stage: a perf block is refused with why.
    del spec["source_embargo"]
    spec["perf"] = {"enabled": True}
    with pytest.raises(task_schema.TaskSchemaError, match="'perf' is not read: the flow has no"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


def _checkpoint(root: Path) -> Path:
    path = root / "newmodel"
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["NewModelForCausalLM"],
        "_name_or_path": "org/newmodel",
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_experts": 16,
        "max_position_embeddings": 32768,
        "num_nextn_predict_layers": 1,
        "vision_config": {"model_type": "vit"},
    }
    (path / "config.json").write_text(json.dumps(config))
    return path


def _site(root: Path, **changes) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    site = {
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        },
        "device": {"arch": "sm_100"},
        "defaults": {"members": ["tp1", "tep4", "dep4"]},
        **changes,
    }
    path = root / "site.yaml"
    path.write_text(yaml.safe_dump(site))
    return path


def _derive(root: Path, goal: str | None = None, *, clean: bool = False, **site) -> Path:
    return cli._derive_spec(
        _checkpoint(root), _site(root, **site), root / "ws", goal=goal, clean=clean
    )


def _engine(root: Path, name: str) -> dict:
    return yaml.safe_load((root / "ws" / derive.ENGINES_DIRNAME / f"{name}.yaml").read_text())


def test_the_goal_overrides_what_the_checkpoint_and_the_site_say(tmp_path):
    goal = (
        "-mtp multimodal tp1 dep4 isl=1024 osl=256 arch=sm_103 disagg@dep4 kv=bf16/fp8 "
        "concurrency=8/256"
    )
    site = dict(
        device={"arch": "sm_100", "gpus_per_node": 4},
        defaults={"disagg": {"env": {"LLM_MODELS_ROOT": "/models"}}},
    )
    spec = load_and_validate_task_yaml(_derive(tmp_path, goal, **site))
    assert spec["arch"] == "sm_103" and "features" not in spec and "members" not in spec
    found = {d.name: d for d in deployments.resolve(spec)}
    # The members in order, then each feature and each extra KV dtype, once.
    assert list(found) == ["tp1", "dep4", "multimodal", "disagg", "kv_fp8"]
    assert found["multimodal"].inputs == ("text", "image") and found["tp1"].inputs == ("text",)
    # disagg where the goal puts it, with the site's environment.
    assert found["disagg"].ctx.options["tensor_parallel_size"] == 4
    assert found["disagg"].env == {"LLM_MODELS_ROOT": "/models"}
    # The first KV dtype is core's; each other one a deployment judged against core's score.
    assert "dtype" not in found["tp1"].server.options["kv_cache_config"]
    assert found["kv_fp8"].server.options["kv_cache_config"]["dtype"] == "fp8"
    assert found["kv_fp8"].relative_to == "tp1"
    core = _engine(tmp_path, "tp1")
    # The batch holds the goal's concurrency, whatever the default says.
    assert (core["max_seq_len"], core["max_batch_size"]) == (1280, 256)
    assert _engine(tmp_path, "dep4")["max_batch_size"] == 64  # one rank's share of it
    with pytest.raises(derive.DeriveError, match="'multimodal:smoke' is not a feature"):
        derive.parse_goal("multimodal:smoke")


def test_a_goal_that_names_its_features_gets_those_and_what_derive_cannot_know_is_asked(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    config = json.loads((checkpoint / "config.json").read_text())
    del config["_name_or_path"]
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(derive.DeriveError) as excinfo:
        cli._derive_spec(
            checkpoint, _site(tmp_path), tmp_path / "ws", goal="multimodal", clean=False
        )
    assert "accuracy.model_key" in str(excinfo.value)
    spec = yaml.safe_load((tmp_path / "ws" / derive.SPEC_FILENAME).read_text())
    # The checkpoint's drafter is left out: the goal names its features and not it.
    assert [d["name"] for d in spec["deployments"]] == ["tp1", "tep4", "dep4", "multimodal"]
    assert "features (left out)" in (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    with pytest.raises(derive.DeriveError, match="kv= is named once"):
        derive.parse_goal("kv=fp8 kv=nvfp4")


def test_core_keeps_reuse_and_chunking_off_and_each_named_one_is_its_own_deployment(tmp_path):
    spec = load_and_validate_task_yaml(
        _derive(
            tmp_path, "tp1 chunked_prefill prefix_reuse kv=fp8/nvfp4 chunk=256 concurrency=1/128"
        )
    )
    assert [d.name for d in deployments.resolve(spec)] == ["tp1", "mtp", "multimodal"] + [
        "kv_nvfp4",
        "chunked",
        "reuse",
    ]
    core = _engine(tmp_path, "tp1")
    assert core["kv_cache_config"]["enable_block_reuse"] is False
    assert core["enable_chunked_prefill"] is False and core["max_num_tokens"] == 8192
    assert core["cuda_graph_config"] and core["disable_overlap_scheduler"] is False
    # The first KV dtype is the one every deployment stores its cache in.
    assert core["kv_cache_config"]["dtype"] == "fp8"
    assert _engine(tmp_path, "kv_nvfp4")["kv_cache_config"]["dtype"] == "nvfp4"
    chunked = _engine(tmp_path, "chunked")
    assert (chunked["enable_chunked_prefill"], chunked["max_num_tokens"]) == (True, 256)
    assert _engine(tmp_path, "reuse")["kv_cache_config"]["enable_block_reuse"] is True
    # Each changes one thing over core, the checkpoint's drafter included.
    assert _engine(tmp_path, "mtp")["speculative_config"]["decoding_type"] == "MTP"
    assert core["speculative_config"] is None
    report = (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    # The batch the spec carries, raised to the concurrency, not the default it started from.
    assert "| `workload.max_batch_size` | concurrency | `128` |" in report
    assert derive.hybrid_marker({"architectures": ["Unseen"], "linear_num_value_heads": 32})
    assert derive.hybrid_marker(
        {"text_config": {"layer_types": ["linear_attention", "full_attention"]}}
    )
    assert derive.hybrid_marker({"architectures": ["Unseen"]}) is None


def test_a_key_this_version_no_longer_reads_is_dropped_with_one_notice_each(tmp_path):
    spec = _valid_spec(tmp_path)
    quiet: list[str] = []
    load_and_validate_task_yaml(_write(tmp_path, spec), quiet)
    assert quiet == []
    spec["accuracy"]["envelope"] = {"enabled": True, "probes": ["1, 2, 3,"]}
    spec["accuracy"]["reference_logits"] = {"path": "reference.json"}
    notices: list[str] = []
    resolved = load_and_validate_task_yaml(_write(tmp_path, spec), notices)
    assert "envelope" not in resolved["accuracy"] and "reference_logits" not in resolved["accuracy"]
    assert len(notices) == 2
    assert notices[0].startswith("'accuracy.envelope' is ignored: a boot runs its workload only")
    assert notices[1].startswith("'accuracy.reference_logits' is ignored: ")
    # Without a list to fill, the spec loads the same, silently.
    assert load_and_validate_task_yaml(_write(tmp_path, spec)) == resolved


def test_each_deployment_owes_its_boot_its_benchmarks_and_its_tests_and_core_the_serve_gate(
    tmp_path,
):
    from agent_flow.workflows.trtllm_modeling_v2.run.contract import compile_contract

    spec = _valid_spec(tmp_path) | {
        "parallel": "unified",
        "members": ["tep4", "dep4"],
        "features": ["mtp", "disagg"],
        "variants": {
            "mtp7": {
                "unit": "mtp",
                "gates": ["boot", "acceptance"],
                "llm_api_options": {"speculative_config": {"max_draft_len": 7}},
                "accuracy": {"acceptance": {"min_absolute": 2.0}},
            }
        },
    }
    spec["accuracy"]["acceptance"] = {"num_prompts": 32, "mtp": {"min_absolute": 1.5}}
    spec["accuracy"]["disagg"] = {"test": "tests/test_pd.py::TestM::test_hop"}
    notices: list[str] = []
    resolved = load_and_validate_task_yaml(_write(tmp_path, spec), notices)
    assert sorted(n.split(" is ignored")[0] for n in notices) == [
        "'accuracy.acceptance'",
        "'variants.mtp7.accuracy.acceptance'",
        "'variants.mtp7.gates'",
    ]
    assert any("judged by a native test" in n for n in notices)
    contract = compile_contract(resolved)
    assert [c.key for c in contract.required()] == [
        "tep4:boot",
        "tep4:accuracy[gsm8k]",
        "dep4:boot",
        "dep4:accuracy[gsm8k]",
        "mtp:boot",
        "mtp:accuracy[gsm8k]",
        "disagg:boot",
        "disagg:accuracy[gsm8k]",
        "disagg:tests",
        "mtp7:boot",
        "mtp7:accuracy[gsm8k]",
    ]
    assert any(
        "mtp: its drafter's acceptance is judged by no check here" in n for n in contract.notes
    )
    # The serve gate is core's, where the spec turns it on.
    resolved["accuracy"]["serve"] = {"enabled": True}
    more = {c.key for c in compile_contract(resolved).required()} - {c.key for c in contract.checks}
    assert more == {"tep4:serve"}
    # A spec with no accuracy block owes boots alone, and says so.
    bare = compile_contract({"deployments": [{"name": "only", "options": {}}]})
    assert [c.key for c in bare.checks] == ["only:boot"]
    assert bare.notes == ("only: it names no benchmark, so its boot alone judges it",)
