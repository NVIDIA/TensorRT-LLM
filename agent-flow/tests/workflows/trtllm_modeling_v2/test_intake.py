# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intake: the user-facing task.yaml, its checks, its confirmation and its compilation."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.spec.intake import checks
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.compile import compile_task
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
        (lambda t: t["parallelism"].append({"kind": "tp", "world": 3}), "32 heads do not split 3"),
        (lambda t: t["workload"].update(isl=40000), "exceeds the model's 40960"),
        (lambda t: t.update(open_questions=["which benchmark?"]), "which benchmark?"),
        (lambda t: t["features"].pop("speculative"), "draft layers the request does not mention"),
        (
            lambda t: t["features"].update(
                disaggregated={"status": "required", "members": ["tp2"]}
            ),
            "members tp2 are not among",
        ),
    ],
)
def test_each_infeasible_request_is_a_block(checkpoint, mutate, expected):
    task = copy.deepcopy(_task(checkpoint))
    mutate(task)
    blocks = [f.render() for f in checks.blocking(checks.run_checks(task))]
    assert any(expected in b for b in blocks), blocks


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
