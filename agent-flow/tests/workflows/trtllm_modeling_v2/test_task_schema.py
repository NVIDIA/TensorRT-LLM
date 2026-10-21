# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trtllm-modeling-v2's ``task.yaml`` validation."""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema


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


def test_valid_spec_round_trips_and_fills_defaults(tmp_path):
    spec = _valid_spec(tmp_path)
    del spec["parallel"]
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["parallel"] == task_schema.PARALLEL_UNIFIED
    assert data["features"] == []
    assert data["accuracy"]["tolerance"] == task_schema.DEFAULT_ACCURACY_TOLERANCE
    assert data["environment"]["env"]["TRTLLM_MODELING_V2"] == "require"


def test_every_problem_is_batched_into_one_message(tmp_path):
    spec = _valid_spec(tmp_path)
    del spec["arch"]
    del spec["environment"]
    del spec["accuracy"]
    spec["checkpoint_path"] = "/definitely/not/here"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(excinfo.value)
    for expected in ("arch", "environment", "accuracy", "non-existent path"):
        assert expected in message


@pytest.mark.parametrize(
    "segment,expected",
    [
        ("tp8", ("tp", 8)),
        ("tep4", ("tep", 4)),
        ("dep4", ("dep", 4)),
        ("unified", ("unified", None)),
    ],
)
def test_parallel_segments_parse(segment, expected):
    assert task_schema.parse_parallel_segment(segment) == expected


@pytest.mark.parametrize("segment", ["tp0", "ep4"])
def test_bad_parallel_segments_are_rejected(segment):
    with pytest.raises(task_schema.TaskSchemaError):
        task_schema.parse_parallel_segment(segment)


# ------------------------------------------------------------- the member set


def _unified(tmp_path, members=None) -> dict:
    spec = _valid_spec(tmp_path)
    spec["parallel"] = task_schema.PARALLEL_UNIFIED
    if members is not None:
        spec["members"] = members
    return spec


def test_a_unified_run_defaults_to_the_baseline_pair_plus_the_degenerate_path(tmp_path):
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, _unified(tmp_path)))
    assert task_schema.members_for(data, "core", 4) == ["tp4", "tep4", "dep4"]
    assert task_schema.members_for(data, "core", 8) == ["tp8", "tep8", "dep8"]


def test_a_concrete_segment_is_its_own_member_set(tmp_path):
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, _valid_spec(tmp_path)))
    assert task_schema.members_for(data, "core", 4) == ["tp1"]


def test_a_per_unit_mapping_is_layered_over_default(tmp_path):
    spec = _unified(tmp_path, {"default": ["tp4", "tep4", "dep4"], "multimodal": ["tep4"]})
    spec["features"] = ["multimodal"]
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert task_schema.members_for(data, "core", 4) == ["tp4", "tep4", "dep4"]
    assert task_schema.members_for(data, "multimodal", 4) == ["tep4"]


def test_unknown_feature_is_rejected_by_name(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["features"] = ["mtp", "telepathy"]
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "telepathy" in str(excinfo.value)
    assert "mtp" in str(excinfo.value)  # listed among the known features


def test_derive_slug_is_a_safe_path_component(tmp_path):
    checkpoint = tmp_path / "DeepSeek-R1-0528_NVFP4"
    checkpoint.mkdir()
    spec = _valid_spec(tmp_path)
    spec["checkpoint_path"] = str(checkpoint)
    spec["parallel"] = "dep4"
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    slug = task_schema.derive_slug(data)
    assert slug == "deepseek-r1-0528-nvfp4-sm_103-dep4"
    assert "/" not in slug and " " not in slug


def test_assembly_units_put_core_first(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["features"] = ["mtp", "multimodal"]
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert task_schema.assembly_units(data) == ["core", "mtp", "multimodal"]


# ------------------------------------------------------ closed key sets


def test_a_misspelled_feature_key_does_not_silently_build_core_only(tmp_path):
    spec = _valid_spec(tmp_path) | {"feature": ["mtp"]}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'feature'" in str(excinfo.value)
    assert "did you mean 'features'" in str(excinfo.value)


def test_nested_blocks_are_closed_too(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["accuracy"]["toleranace"] = "x"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "did you mean 'accuracy.tolerance'" in str(excinfo.value)


def test_every_key_the_schema_accepts_actually_validates(tmp_path):
    repo = tmp_path / "repo"
    spec = _valid_spec(tmp_path) | {
        "features": ["mtp"],
        "scope": "text_only",
        "source_embargo": [],
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "runtime": {"cuda_graph": True, "extra_flags": ["--max_seq_len=8192"]},
    }
    spec["accuracy"] = {
        "benchmark": "gsm8k",
        "reference_yaml": "r.yaml",
        "model_key": "openai/gpt-oss-120b",
        "tolerance": 5.0,
        "dataset_path": str(repo),
        "llm_api_options": {"core": str(repo / "o.yaml")},
        "benchmark_args": ["--apply_chat_template"],
    }
    spool = tmp_path / "spool"
    spool.mkdir()
    spec["environment"] = {
        "kind": "session_spool",
        "spool": str(spool),
        "python": "/usr/bin/python3",
        "env": {"TMPDIR": "/scratch/tmp", "CLAUDE_CODE_TMPDIR": "/scratch/tmp"},
        "partition": "batch",
        "account": "acct",
        "container_image": "/img.sqsh",
        "mounts": ["/lustre:/lustre"],
        "time_limit": "04:00:00",
        "gpus_per_node": 8,
    }
    (repo / "o.yaml").write_text("k: v\n", encoding="utf-8")
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["scope"] == "text_only"
    assert data["runtime"]["cuda_graph"] is True


# ------------------------------- a spec must work on another machine


def test_llm_api_options_written_relative_resolve_next_to_the_spec(tmp_path):
    (tmp_path / "core.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    spec = _valid_spec(tmp_path)
    spec.setdefault("accuracy", {})["llm_api_options"] = "core.yaml"
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["accuracy"]["llm_api_options"] == str((tmp_path / "core.yaml").resolve())


def test_resolving_an_already_resolved_spec_is_idempotent(tmp_path):
    (tmp_path / "core.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    spec = _valid_spec(tmp_path)
    spec.setdefault("accuracy", {})["llm_api_options"] = "core.yaml"
    once = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    spec["accuracy"]["llm_api_options"] = once["accuracy"]["llm_api_options"]
    twice = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert twice["accuracy"]["llm_api_options"] == once["accuracy"]["llm_api_options"]


# ---------------------------- the speculator, replay and kv-quant feature kinds


def test_mtp_and_dspark_are_refused_together(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["features"] = ["mtp", "dspark"]
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "spelled `dspark`" in str(excinfo.value)


_DECODER = {
    "contrast_env": {"TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY": "0"},
    "window": 128,
    "prompt_tokens": 4096,
}
_ENCODER = {
    "contrast_env": {"TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY": "0"},
    "window": 128,
    "prefix_tokens": 4096,
    "reuse_alignment": 2,
}
_KV = {
    "contrast_env": {"TRTLLM_DSV41_KV_STORAGE": "bf16"},
    "expected_bytes_per_token": 890,
}


def _with_witness(tmp_path, features, witness):
    spec = _valid_spec(tmp_path)
    spec["features"] = features
    spec["accuracy"]["witness"] = {**witness, "enabled": True}
    return _write(tmp_path, spec)


def test_the_replay_and_kv_quant_witness_blocks_validate(tmp_path):
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(
            tmp_path,
            ["kv_quant", "decoder_replay", "encoder_replay"],
            {"decoder_replay": _DECODER, "encoder_replay": _ENCODER, "kv_quant": _KV},
        )
    )
    # `exact` changes what the unit claims, so the resolved spec states it.
    assert data["accuracy"]["witness"]["kv_quant"]["exact"] is True


@pytest.mark.parametrize("witness", [{}, {"disagg": {"contrast_env": {"MYMODEL_DISAGG": "1"}}}])
def test_a_spec_owing_the_disagg_witness_must_declare_its_leg_b(tmp_path, witness):
    """Without a block, or with one that names no launcher, the spec is refused."""
    with pytest.raises(task_schema.TaskSchemaError, match=r"contrast_command.*declare"):
        task_schema.load_and_validate_task_yaml(_with_witness(tmp_path, ["disagg"], witness))
