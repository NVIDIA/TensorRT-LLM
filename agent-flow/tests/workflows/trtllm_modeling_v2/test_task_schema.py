# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 task schema behavior and regression checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.operators import tree_anchors
from agent_flow.workflows.trtllm_modeling_v2.spec import derive, task_schema
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
    spec["accuracy"]["toleranace"] = "x"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'feature'" in str(excinfo.value)
    assert "did you mean 'features'" in str(excinfo.value)
    assert "did you mean 'accuracy.tolerance'" in str(excinfo.value)


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


DISAGG_LAUNCHER = tree_anchors.ANCHORS["disagg_launcher"].hint


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


def test_the_goal_overrides_what_the_checkpoint_and_the_site_say(tmp_path):
    goal = (
        "-mtp multimodal:smoke tp1 dep4 isl=1024 osl=256 arch=sm_103 disagg@dep4 kv=bf16/fp8 "
        "concurrency=8/256"
    )
    site = dict(
        device={"arch": "sm_100", "gpus_per_node": 4},
        defaults={"disagg": {"env": {"LLM_MODELS_ROOT": "/models"}}},
    )
    # The node id is named beside the tree's disaggregated launcher: asked for until it is found.
    with pytest.raises(derive.DeriveError, match="accuracy.disagg.test"):
        _derive(tmp_path, goal, **site)
    launcher = tmp_path / "repo" / DISAGG_LAUNCHER
    launcher.parent.mkdir(parents=True)
    launcher.write_text("def launch_disaggregated_llm(config):\n    pass\n")
    path = _derive(tmp_path, goal, clean=True, **site)
    spec = load_and_validate_task_yaml(path)
    assert spec["features"] == ["multimodal", "disagg"] and spec["arch"] == "sm_103"
    assert spec["profile"] == {"default": "standard", "multimodal": "smoke"}
    # The CI files are owed while one unit is standard, not by a smoke or bringup run.
    from agent_flow.workflows.trtllm_modeling_v2.run.certify import ci_waiver

    assert ci_waiver(spec) == ""
    assert ci_waiver({**spec, "profile": "bringup"}) == "not owed by the bringup profile"
    # disagg where the goal puts it; its test lays its two engines out on dep4's node.
    assert spec["members"] == {"default": ["tp1", "dep4"], "disagg": ["dep4"]}
    assert spec["coverage"] == {"accept": ["disagg@tp1"]}
    # The first KV dtype is every gate's; each other one a variant judged against core's score.
    assert "--kv_cache_dtype=fp8" not in spec["runtime"]["extra_flags"]
    assert spec["variants"]["kv_fp8"]["gates"] == ["accuracy"]
    assert spec["variants"]["kv_fp8"]["llm_api_options"] == {"kv_cache_config": {"dtype": "fp8"}}
    assert spec["accuracy"]["disagg"] == {
        "env": {"LLM_MODELS_ROOT": "/models"},
        "test": f"{DISAGG_LAUNCHER}::TestNewModel::test_auto_dtype",
    }
    assert "--max_seq_len=1280" in spec["runtime"]["extra_flags"]
    # The batch holds the goal's concurrency, whatever the default says.
    assert "--max_batch_size=256" in spec["runtime"]["extra_flags"]


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
    assert "perf" not in spec
    # The checkpoint's drafter is left out: the goal names its features and not it.
    assert spec["features"] == ["multimodal"]
    assert "features (left out)" in (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    with pytest.raises(derive.DeriveError, match="kv= is named once"):
        derive.parse_goal("kv=fp8 kv=nvfp4")


def test_core_keeps_reuse_and_chunking_off_and_each_named_one_is_its_own_variant(tmp_path):
    spec = load_and_validate_task_yaml(
        _derive(tmp_path, "chunked_prefill prefix_reuse kv=bf16/fp8 chunk=256 concurrency=1/128")
    )
    runtime = spec["runtime"]
    assert runtime["cuda_graph"] and runtime["overlap_scheduler"]
    assert not runtime["chunked_prefill"] and not runtime["kv_cache_reuse"]
    core = yaml.safe_load(Path(spec["accuracy"]["llm_api_options"]["core"]).read_text())
    assert core["kv_cache_config"]["enable_block_reuse"] is False
    assert core["enable_chunked_prefill"] is False and "max_num_tokens" not in core
    variants = spec["variants"]
    assert set(variants) == {"kv_fp8", "chunked", "reuse"}
    assert variants["chunked"]["llm_api_options"] == {
        "enable_chunked_prefill": True,
        "max_num_tokens": 256,
    }
    assert variants["reuse"]["llm_api_options"] == {"kv_cache_config": {"enable_block_reuse": True}}
    # Each changes one non-core feature, gated by its benchmark alone.
    assert all(v["unit"] == "core" and v["gates"] == ["accuracy"] for v in variants.values())
    report = (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    # The batch the spec carries, raised to the concurrency, not the default it started from.
    assert "| `workload.max_batch_size` | concurrency | `128` |" in report
    with pytest.raises(derive.DeriveError):
        derive.parse_goal("chunked_prefill:smoke")


@pytest.mark.parametrize(
    "weights_gib, layout",
    [(100, (2, 1)), (300, (4, 1)), (500, (4, 2)), (None, None)],
)
def test_the_disaggregated_test_takes_one_node_unless_its_two_engines_do_not_fit(
    weights_gib, layout
):
    weights = None if weights_gib is None else weights_gib * 2**30
    assert derive.disagg_layout(weights, 277, 4) == layout


def test_quantized_core_has_one_kv_dtype_source_and_structural_hybrid_hints(tmp_path):
    spec = yaml.safe_load(_derive(tmp_path, "kv=fp8/nvfp4").read_text())
    assert not any(flag.startswith("--kv_cache_dtype") for flag in spec["runtime"]["extra_flags"])
    options = yaml.safe_load(Path(spec["accuracy"]["llm_api_options"]["core"]).read_text())
    assert options["kv_cache_config"]["dtype"] == "fp8"
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


def test_the_certified_profile_is_standard_with_the_serve_gate(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.contract import compile_contract

    spec = _valid_spec(tmp_path) | {
        "parallel": "unified",
        "members": ["tep4", "dep4"],
        "variants": {"kv_fp8": {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}},
    }
    standard = load_and_validate_task_yaml(_write(tmp_path, spec))
    certified = load_and_validate_task_yaml(_write(tmp_path, spec | {"profile": "certified"}))
    assert standard["accuracy"]["serve"] == {"enabled": False}  # nothing bounds the server
    assert certified["accuracy"]["serve"] == {"enabled": True}
    owed = {c.key for c in compile_contract(standard, 4).required()}
    more = {c.key for c in compile_contract(certified, 4).required()} - owed
    assert more and all(":serve@" in key for key in more)
    # A relative variant's benchmark runs at one member under either profile.
    assert [k for k in owed if k.startswith("kv_fp8:accuracy")] == ["kv_fp8:accuracy[gsm8k]@dep4"]


def test_a_drafter_owes_no_acceptance_check_and_the_old_keys_are_ignored_with_a_notice(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.contract import compile_contract

    spec = _valid_spec(tmp_path) | {
        "features": ["mtp"],
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
    notices: list[str] = []
    resolved = load_and_validate_task_yaml(_write(tmp_path, spec), notices)
    assert [n.split(" is ignored")[0] for n in notices] == [
        "'accuracy.acceptance'",
        "'variants.mtp7.accuracy.acceptance'",
    ]
    assert "judged by a native test" in notices[0]
    contract = compile_contract(resolved, 1)
    assert not [c.key for c in contract.required() if "acceptance" in c.key]
    assert "mtp7:boot@tp1" in {c.key for c in contract.required()}
    assert any(
        "mtp: its drafter's acceptance is judged by no check here" in n for n in contract.notes
    )
