# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fast path: the run's spec derived from the checkpoint, the site and a goal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.operators import tree_anchors
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.spec import derive
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

from .conftest import SCRATCH

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
    (root / "spool").mkdir(exist_ok=True)
    site = {
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(root / "spool"),
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


def test_what_the_checkpoint_carries_is_built_into_a_spec_the_run_takes(tmp_path):
    spec = load_and_validate_task_yaml(_derive(tmp_path))
    assert spec["features"] == ["mtp", "multimodal"]
    assert spec["parallel"] == "unified" and spec["members"] == ["tp1", "tep4", "dep4"]
    assert spec["accuracy"]["model_key"] == "org/newmodel"
    assert spec["accuracy"]["discriminators"]["mtp"] == {"spec_dec_algo": "MTP"}
    options = yaml.safe_load(Path(spec["accuracy"]["llm_api_options"]["mtp"]).read_text())
    assert options["speculative_config"] == {"decoding_type": "MTP", "max_draft_len": 1}
    report = (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    assert "| `features (speculator)` | checkpoint | `mtp` |" in report


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
            checkpoint, _site(tmp_path), tmp_path / "ws", goal="multimodal tune", clean=False
        )
    assert "accuracy.model_key" in str(excinfo.value) and "concurrency" not in str(excinfo.value)
    spec = yaml.safe_load((tmp_path / "ws" / derive.SPEC_FILENAME).read_text())
    # Concurrency is not asked: 1/32/128 by default, attention DP only at the high point, a
    # global total its ranks share.
    loads = {c["name"]: c["workload"]["concurrency"] for c in spec["perf"]["configs"]}
    assert loads == {"tp1": [1, 32, 128], "tep4": [1, 32, 128], "dep4": [128]}
    # The checkpoint's drafter is left out: the goal names its features and not it.
    assert spec["features"] == ["multimodal"]
    assert "features (left out)" in (tmp_path / "ws" / derive.REPORT_FILENAME).read_text()
    with pytest.raises(derive.DeriveError, match="kv= is named once"):
        derive.parse_goal("kv=fp8 kv=nvfp4")


def test_an_open_question_stops_before_the_run_and_is_answered_in_the_spec(tmp_path):
    with pytest.raises(derive.DeriveError, match="arch: name the GPU architecture"):
        _derive(tmp_path, device={})
    path = tmp_path / "ws" / derive.SPEC_FILENAME
    path.write_text(path.read_text() + "arch: sm_100\n")
    # Run again, the spec is taken as it stands, edits included.
    assert load_and_validate_task_yaml(_derive(tmp_path, device={}))["arch"] == "sm_100"
    with pytest.raises(derive.DeriveError):
        _derive(tmp_path, clean=True, device={})


def test_a_spec_the_checkpoint_cannot_meet_is_refused_with_every_reason(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"num_attention_heads": 8, "quantization_config": {"quant_algo": "NVFP4"}})
    )
    task = {
        "checkpoint_path": str(checkpoint),
        "parallel": "tp3",
        "arch": "sm_90",
        "features": ["mtp"],
    }
    with pytest.raises(preflight.SpecInconsistencyError) as raised:
        preflight.check_checkpoint_fit(task, 3)
    message = str(raised.value)
    assert "8 attention heads do not split 3 ways" in message
    assert "nvfp4 weights need sm_100" in message
    assert "config.json states none" in message


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
    [(100, (2, 1)), (300, (4, 1)), (500, (4, 2)), (2000, (4, 8)), (None, None)],
)
def test_the_disaggregated_test_takes_one_node_unless_its_two_engines_do_not_fit(
    weights_gib, layout
):
    weights = None if weights_gib is None else weights_gib * 2**30
    assert derive.disagg_layout(weights, 277, 4) == layout
