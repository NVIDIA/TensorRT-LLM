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


@pytest.mark.parametrize(
    "block,field,valid,invalid",
    [
        ("serve", "enabled", [None, True, False], [0, 1, "false"]),
        ("serve", "prompt", [None, "answer"], ["", " ", 1, []]),
        ("serve", "max_tokens", [None, 1, 32], [True, False, 0, -1, 1.5, "1"]),
        ("envelope", "long_prompts", [None, 0, 2], [True, False, -1, 1.5, "0"]),
        ("envelope", "probes", [None, ["question"]], [[], [""], [" "], [1], "question"]),
        ("boot", "keywords", [None, [], [""], ["word"]], [[1], "word"]),
        ("disagg", "devices", [None, 1, 8], [True, False, 0, 1.5, "1"]),
        ("acceptance", "num_prompts", [None, 1, 32], [True, False, 0, 1.5, "1"]),
        ("acceptance", "text_field", [None, "question"], ["", " ", 1]),
    ],
)
def test_optional_workload_fields_keep_their_type_and_boundary_rules(
    tmp_path, block, field, valid, invalid
):
    spec = _valid_spec(tmp_path)
    for values, accepted in ((valid, True), (invalid, False)):
        for value in values:
            level = spec["accuracy"]
            for key in block.split("."):
                level = level.setdefault(key, {})
            level[field] = value
            path = _write(tmp_path, spec)
            if accepted:
                task_schema.load_and_validate_task_yaml(path)
            else:
                with pytest.raises(task_schema.TaskSchemaError, match=f"accuracy.{block}.{field}"):
                    task_schema.load_and_validate_task_yaml(path)


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
    spec["checkpoint_path"] = "/definitely/not/here"
    # A disabled gate's fields are still validated.
    spec["accuracy"]["serve"] = {"enabled": False, "max_tokens": True, "prompt": " "}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(excinfo.value)
    for expected in (
        "arch",
        "environment",
        "non-existent path",
        "accuracy.serve.max_tokens",
        "accuracy.serve.prompt",
    ):
        assert expected in message


def test_parallel_segments_parse_and_bad_ones_are_rejected():
    for segment, expected in {
        "tp8": ("tp", 8),
        "tep4": ("tep", 4),
        "dep4": ("dep", 4),
        "unified": ("unified", None),
    }.items():
        assert task_schema.parse_parallel_segment(segment) == expected
    for segment in ("tp0", "ep4"):
        with pytest.raises(task_schema.TaskSchemaError):
            task_schema.parse_parallel_segment(segment)


# ------------------------------------------------------------- the member set


def test_the_member_set_of_a_unified_run_a_concrete_segment_and_a_per_unit_mapping(tmp_path):
    concrete = _valid_spec(tmp_path)
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, concrete))
    assert task_schema.members_for(data, "core", 4) == ["tp1"]
    # Unified: the baseline pair plus the degenerate path.
    spec = concrete | {"parallel": task_schema.PARALLEL_UNIFIED}
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert task_schema.members_for(data, "core", 4) == ["tp4", "tep4", "dep4"]
    assert task_schema.members_for(data, "core", 8) == ["tp8", "tep8", "dep8"]
    # A per-unit mapping is layered over the default.
    spec |= {
        "members": {"default": ["tp4", "tep4", "dep4"], "multimodal": ["tep4"]},
        "features": ["multimodal"],
    }
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert task_schema.members_for(data, "core", 4) == ["tp4", "tep4", "dep4"]
    assert task_schema.members_for(data, "multimodal", 4) == ["tep4"]


def test_the_slug_is_a_safe_path_component_and_assembly_puts_core_first(tmp_path):
    checkpoint = tmp_path / "DeepSeek-R1-0528_NVFP4"
    checkpoint.mkdir()
    spec = _valid_spec(tmp_path) | {"checkpoint_path": str(checkpoint), "parallel": "dep4"}
    spec["features"] = ["mtp", "multimodal"]
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    slug = task_schema.derive_slug(data)
    assert slug == "deepseek-r1-0528-nvfp4-sm_103-dep4"
    assert "/" not in slug and " " not in slug
    assert task_schema.assembly_units(data) == ["core", "mtp", "multimodal"]


# ------------------------------------------------------ closed key sets


def test_a_misspelled_key_is_refused_with_its_spelling_at_any_depth(tmp_path):
    # A misspelled `features` must not silently build core only.
    spec = _valid_spec(tmp_path) | {"feature": ["mtp"]}
    spec["accuracy"]["toleranace"] = "x"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'feature'" in str(excinfo.value)
    assert "did you mean 'features'" in str(excinfo.value)
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
        "boot": {"max_tokens": 32},
        "disagg": {
            "test": "tests/integration/defs/accuracy/test_disaggregated_serving.py::TestX::test_auto_dtype",
            "devices": 4,
            "timeout": 3600,
            "env": {"LLM_MODELS_ROOT": "/models"},
        },
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


def test_llm_api_options_written_relative_resolve_next_to_the_spec_once(tmp_path):
    (tmp_path / "core.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    spec = _valid_spec(tmp_path)
    spec["accuracy"]["llm_api_options"] = "core.yaml"
    once = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert once["accuracy"]["llm_api_options"] == str((tmp_path / "core.yaml").resolve())
    spec["accuracy"]["llm_api_options"] = once["accuracy"]["llm_api_options"]
    twice = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert twice["accuracy"]["llm_api_options"] == once["accuracy"]["llm_api_options"]


# ---------------------------- the speculator and disaggregation feature kinds


@pytest.mark.parametrize(
    "features,fragments",
    [
        # The known features are listed beside the unknown one.
        (["mtp", "telepathy"], ("telepathy", "mtp")),
        (["mtp", "dspark"], ("spelled `dspark`",)),
    ],
)
def test_an_unknown_feature_or_two_speculators_are_refused(tmp_path, features, fragments):
    spec = _valid_spec(tmp_path) | {"features": features}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    for fragment in fragments:
        assert fragment in str(excinfo.value)


def test_an_anchor_without_a_number_is_a_placeholder_not_a_state(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["accuracy"]["anchor"] = "pending"
    with pytest.raises(task_schema.TaskSchemaError, match="record a low placeholder"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    spec["accuracy"]["anchor"] = "relative"  # a unit's own score is no anchor for it
    with pytest.raises(task_schema.TaskSchemaError, match="only a configuration variant"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


@pytest.mark.parametrize(
    "block,error",
    [
        (None, r"needs 'accuracy.disagg.test'"),
        ({"devices": 4}, r"needs 'accuracy.disagg.test'"),
        ({"test": "/abs/test_x.py::T::t"}, r"repository-relative pytest node id"),
        ({"test": "tests/test_x.py"}, r"repository-relative pytest node id"),
        ({"test": "tests/test_x.py::T::t", "env": {"1BAD": "x"}}, r"not a shell variable name"),
    ],
)
def test_the_disagg_feature_names_a_node_id_it_can_run(tmp_path, block, error):
    spec = _valid_spec(tmp_path)
    spec["features"] = ["disagg"]
    if block is not None:
        spec["accuracy"]["disagg"] = block
    with pytest.raises(task_schema.TaskSchemaError, match=error):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
