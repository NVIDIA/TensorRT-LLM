# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake: the user-facing task.yaml, its checks, its confirmation and its compilation."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.spec.intake import checks
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.cli import main as intake_main
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.compile import compile_files, compile_task
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.freeze import (
    confirm,
    digest,
    read_confirmation,
    unconfirmed_reason,
)
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.schema import IntakeError, validate_task


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "ckpt"
    path.mkdir()
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 128,
        "max_position_embeddings": 40960,
        "num_nextn_predict_layers": 1,
    }
    (path / "config.json").write_text(json.dumps(config))
    return path


def _task(checkpoint) -> dict:
    return {
        "version": 1,
        "goals": ["onboard", "tune"],
        "model": {"checkpoint": str(checkpoint)},
        "device": {"arch": "sm_100"},
        "parallelism": [{"kind": "tep", "world": 4}, {"kind": "dep", "world": 4}],
        "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 64]},
        "features": {
            "cuda_graph": "required",
            "overlap_scheduler": "required",
            "prefix_reuse": "out_of_scope",
            "kv_cache": {"status": "required", "params": {"dtype": "fp8"}},
            "speculative": {"status": "required", "params": {"algo": "mtp"}},
            "multimodal": "out_of_scope",
            "long_context": {"status": "nice_to_have", "params": {"length": 32768}},
        },
        "accuracy": {"benchmark": "gsm8k", "protocol": {"chat_template": True}},
        "perf": {
            "configs": [
                {
                    "name": "chat",
                    "workload": {"isl": 1024, "osl": 256, "concurrency": [1, 16]},
                    "constraint": {"tpot_ms_max": 30},
                }
            ],
            "goal": {"ratio_to_baseline": 1.1},
        },
        "constraints": {"embargo": ["tensorrt_llm/x.py"], "base_ref": "origin/main"},
        "notes": "Reuse the NVFP4 MoE kernels.",
    }


_SITE = {
    "trtllm_repo_path": "/repo",
    "environment": {"kind": "local", "python": "/opt/venv/bin/python"},
    "perf": {"spools": ["/spool/a"], "env_script": "/env.sh"},
}


def test_a_well_formed_task_has_no_shape_errors(checkpoint):
    assert validate_task(_task(checkpoint)) == []


def test_shape_errors_are_batched():
    errors = validate_task({"version": 2, "goals": ["deploy"], "feature": {}})
    joined = "\n".join(errors)
    assert "'version'" in joined and "'goals'" in joined
    assert "did you mean 'task.yaml.features'" in joined
    assert "'model.checkpoint' is required" in joined


def test_the_checks_pass_a_feasible_request(checkpoint):
    findings = checks.run_checks(_task(checkpoint))
    assert checks.blocking(findings) == [], [f.render() for f in findings]


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda t: t.update(goals=["onboard", "add_features"]), "onboard and add_features"),
        (lambda t: t.update(goals=["tune"]), "no model.base"),
        (lambda t: t["parallelism"].append({"kind": "tp", "world": 3}), "32 heads do not split 3"),
        (lambda t: t["workload"].update(isl=40000), "exceeds the model's 40960"),
        (lambda t: t.update(open_questions=["which benchmark?"]), "which benchmark?"),
        (lambda t: t["features"].pop("speculative"), "draft layers the request does not mention"),
        (
            lambda t: t["perf"]["configs"][0].update(features={"multimodal": "on"}),
            "turns on multimodal",
        ),
    ],
)
def test_each_infeasible_request_is_a_block(checkpoint, mutate, expected):
    task = copy.deepcopy(_task(checkpoint))
    mutate(task)
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert any(expected in b for b in blocks), blocks


def test_mtp_over_a_dspark_drafter_is_refused(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    (checkpoint / "config.json").write_text(json.dumps({**config, "dspark_block_size": 5}))
    blocks = [f.render() for f in checks.blocking(checks.run_checks(_task(checkpoint)))]
    assert any("use algo: dspark" in b for b in blocks)


def test_confirmation_holds_until_the_task_changes(tmp_path, checkpoint):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(_task(checkpoint)))
    assert "not confirmed" in unconfirmed_reason(yaml.safe_load(path.read_text()))
    confirm(path, "reviewer")
    assert unconfirmed_reason(yaml.safe_load(path.read_text()), read_confirmation(path)) is None
    edited = yaml.safe_load(path.read_text())
    edited["workload"]["isl"] = 4096
    assert "changed after it was confirmed" in unconfirmed_reason(edited, read_confirmation(path))


def test_confirming_leaves_the_users_file_and_its_comments_as_they_are(tmp_path, checkpoint):
    path = tmp_path / "task.yaml"
    text = "# reviewed with the model owner\n" + yaml.safe_dump(_task(checkpoint))
    path.write_text(text)
    confirm(path, "reviewer")
    assert path.read_text() == text
    assert read_confirmation(path)["by"] == "reviewer"


def test_a_confirmed_block_inside_the_task_is_still_honoured(tmp_path, checkpoint):
    task = _task(checkpoint)
    task["confirmed"] = {"by": "reviewer", "at": "2026-01-01T00:00:00Z", "sha256": digest(task)}
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(task))
    assert unconfirmed_reason(task, read_confirmation(path)) is None


def test_open_questions_block_confirmation(tmp_path, checkpoint):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump({**_task(checkpoint), "open_questions": ["tp or dep?"]}))
    with pytest.raises(IntakeError, match="open question"):
        confirm(path, "reviewer")


def test_experts_named_num_local_experts_make_a_moe_model(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    config.pop("num_experts")
    (checkpoint / "config.json").write_text(json.dumps({**config, "num_local_experts": 8}))
    task = _task(checkpoint)
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert not any("dense model" in b for b in blocks), blocks
    task["parallelism"].append({"kind": "tep", "world": 3})
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert any("8 experts do not split 3" in b for b in blocks), blocks


def test_draft_layers_under_mtp_num_hidden_layers_are_draft_layers(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    config.pop("num_nextn_predict_layers")
    (checkpoint / "config.json").write_text(json.dumps({**config, "mtp_num_hidden_layers": 1}))
    assert checks.blocking(checks.run_checks(_task(checkpoint))) == []
    task = _task(checkpoint)
    task["features"].pop("speculative")
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert any("draft layers the request does not mention" in b for b in blocks)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda t: t["features"].update(
                disaggregated={"status": "required", "members": ["tp2"]}
            ),
            "members tp2 are not among",
        ),
        (lambda t: t.update(device={"arch": "sm_100", "gpus_per_node": 2}), "the device has 2"),
        (
            lambda t: t["features"].update(
                speculative={"status": "required", "params": {"algo": "eagle3"}}
            ),
            "none is named",
        ),
        (lambda t: t["perf"]["configs"][0].update(parallel="tp8"), "not among the run's"),
    ],
)
def test_each_new_infeasible_request_is_a_block(checkpoint, mutate, expected):
    task = copy.deepcopy(_task(checkpoint))
    mutate(task)
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert any(expected in b for b in blocks), blocks


def test_block_reuse_on_a_hybrid_checkpoint_needs_a_snapshot_policy(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    config["layer_types"] = ["linear_attention", "full_attention"]
    (checkpoint / "config.json").write_text(json.dumps(config))
    task = _task(checkpoint)
    task["features"]["prefix_reuse"] = "required"
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert any("snapshot policy" in b for b in blocks), blocks
    task["features"]["prefix_reuse"] = {"status": "required", "params": {"snapshot_interval": 512}}
    assert checks.blocking(checks.run_checks(task)) == []


def test_a_dspark_drafter_drafts_its_own_block_length(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    (checkpoint / "config.json").write_text(json.dumps({**config, "dspark_block_size": 5}))
    task = _task(checkpoint)
    task["features"]["speculative"] = {"status": "required", "params": {"algo": "dspark"}}
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert any("blocks of 5 tokens" in b for b in blocks), blocks
    task["features"]["speculative"]["params"]["draft_len"] = 5
    assert checks.blocking(checks.run_checks(task)) == []


def test_compile_maps_requirements_onto_the_pipeline_spec(checkpoint):
    compiled = compile_task(_task(checkpoint), _SITE)
    spec = compiled.spec
    assert spec["parallel"] == "unified" and spec["members"] == ["tep4", "dep4"]
    assert spec["features"] == ["mtp"]
    assert spec["scope"] == "text_only"
    assert spec["runtime"]["cuda_graph"] is True
    assert spec["runtime"]["kv_cache_reuse"] is False
    assert "--max_seq_len=9216" in spec["runtime"]["extra_flags"]
    assert "--max_batch_size=64" in spec["runtime"]["extra_flags"]
    assert "--kv_cache_dtype=fp8" in spec["runtime"]["extra_flags"]
    assert spec["accuracy"]["benchmark_args"] == ["--apply_chat_template"]
    assert spec["source_embargo"] == ["tensorrt_llm/x.py"]
    assert spec["commit_policy"]["base_ref"] == "origin/main"
    perf = spec["perf"]
    assert perf["enabled"] and perf["spools"] == ["/spool/a"]
    assert perf["configs"][0]["constraint"] == {"tpot_ms_max": 30}
    # The FP8 KV cache and the MTP draft each name the reference row.
    assert spec["accuracy"]["discriminators"] == {
        "default": {"kv_cache_quant_algo": "FP8"},
        "mtp": {"kv_cache_quant_algo": "FP8", "spec_dec_algo": "MTP"},
    }
    options = compiled.llm_api_options
    assert "speculative_config" not in options["core"]
    assert options["mtp"]["speculative_config"] == {"decoding_type": "MTP", "max_draft_len": 1}
    assert options["core"]["kv_cache_config"] == {"dtype": "fp8", "enable_block_reuse": False}
    assert options["core"]["cuda_graph_config"] == {"enable_padding": True, "max_batch_size": 64}
    assert perf["goal"] == {"ratio_to_reference": 1.1}
    assert any("long_context" in w for w in compiled.warnings)
    assert spec["notes"] == "Reuse the NVFP4 MoE kernels."
    assert not any(w.startswith("notes") for w in compiled.warnings)


def test_a_quantized_checkpoint_anchors_to_its_own_row(checkpoint):
    task = _task(checkpoint)
    task["model"]["quantization"] = {"weights": "nvfp4"}
    task["features"].pop("kv_cache")
    spec = compile_task(task, _SITE).spec
    assert spec["accuracy"]["discriminators"] == {
        "default": {"quant_algo": "NVFP4"},
        "mtp": {"quant_algo": "NVFP4", "spec_dec_algo": "MTP"},
    }
    task["accuracy"]["discriminators"] = {"quant_algo": "MIXED_PRECISION"}
    assert compile_task(task, _SITE).spec["accuracy"]["discriminators"] == {
        "quant_algo": "MIXED_PRECISION"
    }


def test_a_feature_narrowed_to_members_is_gated_on_those_only(checkpoint):
    task = _task(checkpoint)
    task["features"]["disaggregated"] = {"status": "required", "members": ["tep4"]}
    spec = compile_task(task, _SITE).spec
    assert spec["members"] == {"default": ["tep4", "dep4"], "disagg": ["tep4"]}


def test_every_requirement_compile_does_not_carry_is_warned(checkpoint):
    task = _task(checkpoint)
    task["features"]["cuda_graph"] = {"status": "required", "params": {"batch_sizes": [1, 2]}}
    task["accuracy"]["protocol"]["shots"] = 0
    task["perf"]["metric"] = "output_tps_per_gpu"
    task["deliverable"] = {"branch": "auto"}
    warnings = "\n".join(compile_task(task, _SITE).warnings)
    for expected in (
        "features.cuda_graph.params.batch_sizes",
        "accuracy.protocol.shots",
        "perf.metric",
        "deliverable",
    ):
        assert expected in warnings, warnings


def test_tune_measures_each_member_at_its_parallelism_with_the_draft(checkpoint):
    task = _task(checkpoint)
    task["perf"].pop("configs")
    configs = compile_task(task, _SITE).spec["perf"]["configs"]
    assert [c["name"] for c in configs] == ["tep4", "dep4"]
    assert configs[1]["llm_api_options"] == {
        "tensor_parallel_size": 4,
        "moe_expert_parallel_size": 4,
        "enable_attention_dp": True,
        "speculative_config": {"decoding_type": "MTP", "max_draft_len": 1},
    }
    task = _task(checkpoint)
    task["perf"]["configs"][0].update(parallel="tep4", features={"speculative": "on"})
    options = compile_task(task, _SITE).spec["perf"]["configs"][0]["llm_api_options"]
    assert options["moe_expert_parallel_size"] == 4 and "speculative_config" in options


def test_a_sol_goal_compiles_beside_the_ratio(checkpoint):
    task = _task(checkpoint)
    task["perf"]["goal"] = {"ratio_to_baseline": 1.1, "pct_of_sol": 0.7}
    goal = compile_task(task, _SITE).spec["perf"]["goal"]
    assert goal == {"ratio_to_reference": 1.1, "pct_of_sol": 0.7}


def test_compiled_spec_passes_the_pipeline_schema(tmp_path, checkpoint):
    repo = tmp_path / "repo"
    repo.mkdir()
    task, site = tmp_path / "task.yaml", tmp_path / "site.yaml"
    task.write_text(yaml.safe_dump(_task(checkpoint)))
    site.write_text(yaml.safe_dump({**_SITE, "trtllm_repo_path": str(repo)}))
    (repo / "tensorrt_llm").mkdir()
    (repo / "tensorrt_llm" / "x.py").write_text("")
    out = tmp_path / "ws" / "task.resolved.yaml"
    compile_files(task, site, out)
    spec = yaml.safe_load(out.read_text())
    assert spec["arch"] == "sm_100"
    options = spec["accuracy"]["llm_api_options"]
    assert set(options) == {"core", "mtp"}
    mtp = yaml.safe_load((tmp_path / "ws" / "llm_api" / "mtp.yaml").read_text())
    assert options["mtp"] == str(tmp_path / "ws" / "llm_api" / "mtp.yaml")
    assert mtp["speculative_config"]["decoding_type"] == "MTP"


def test_the_cli_checks_confirms_and_compiles(tmp_path, checkpoint, capsys):
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump(_task(checkpoint)))
    assert intake_main(["check", str(task)]) == 0
    assert intake_main(["confirm", str(task), "--by", "reviewer"]) == 0
    assert "confirmed by reviewer" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({**_task(checkpoint), "goals": []}))
    assert intake_main(["check", str(bad)]) == 1


def test_the_run_refuses_an_unconfirmed_intake_task(tmp_path, checkpoint):
    from agent_flow.workflows.trtllm_modeling_v2.cli import _resolve_intake_task

    task, site = tmp_path / "task.yaml", tmp_path / "site.yaml"
    task.write_text(yaml.safe_dump(_task(checkpoint)))
    site.write_text(yaml.safe_dump(_SITE))
    with pytest.raises(IntakeError, match="not confirmed"):
        _resolve_intake_task(task, site, tmp_path / "ws")
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(yaml.safe_dump({"checkpoint_path": "/c"}))
    assert _resolve_intake_task(resolved, None, tmp_path / "ws") == resolved
