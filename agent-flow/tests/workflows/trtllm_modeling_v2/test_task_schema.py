# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trtllm-modeling-v2's ``task.yaml`` validation."""

from __future__ import annotations

import shlex

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


def test_missing_arch_is_rejected_with_the_reason(tmp_path):
    """``arch`` is never inferred, and the message must say why."""
    spec = _valid_spec(tmp_path)
    del spec["arch"]
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "arch" in str(excinfo.value)
    assert "per-architecture" in str(excinfo.value)


def test_every_problem_is_batched_into_one_message(tmp_path):
    """A user fixing a spec should see all of it at once, not one per run."""
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


@pytest.mark.parametrize("arch", ["sm100", "SM_103", "blackwell", "sm_", ""])
def test_malformed_arch_is_rejected(tmp_path, arch):
    spec = _valid_spec(tmp_path)
    spec["arch"] = arch
    with pytest.raises(task_schema.TaskSchemaError, match="arch"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


@pytest.mark.parametrize(
    "segment,expected",
    [
        ("tp1", ("tp", 1)),
        ("tp8", ("tp", 8)),
        ("tep4", ("tep", 4)),
        ("dep4", ("dep", 4)),
        ("unified", ("unified", None)),
    ],
)
def test_parallel_segments_parse(segment, expected):
    assert task_schema.parse_parallel_segment(segment) == expected


@pytest.mark.parametrize("segment", ["single", "tp", "tp0", "ep4", "tp-4", "TP4"])
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
    """`unified` is DEFAULT_PARALLEL.

    So the default member set is what most runs actually get gated over.
    """
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, _unified(tmp_path)))
    assert task_schema.members_for(data, "core", 4) == ["tp4", "tep4", "dep4"]
    assert task_schema.members_for(data, "core", 8) == ["tp8", "tep8", "dep8"]


def test_a_concrete_segment_is_its_own_member_set(tmp_path):
    """One code path: the single-configuration run is the degenerate case."""
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, _valid_spec(tmp_path)))
    assert task_schema.members_for(data, "core", 4) == ["tp1"]


def test_an_explicit_list_applies_to_every_unit(tmp_path):
    spec = _unified(tmp_path, ["tep4", "dep4"])
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert task_schema.members_for(data, "core", 4) == ["tep4", "dep4"]
    assert task_schema.members_for(data, "mtp", 4) == ["tep4", "dep4"]


def test_a_per_unit_mapping_is_layered_over_default(tmp_path):
    """A per-unit mapping is layered over `default`.

    The shape `runtime:` already uses, and the knob for buying a cheaper
    answer honestly: one unit gated at one member, named.
    """
    spec = _unified(tmp_path, {"default": ["tp4", "tep4", "dep4"], "multimodal": ["tep4"]})
    spec["features"] = ["multimodal"]
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert task_schema.members_for(data, "core", 4) == ["tp4", "tep4", "dep4"]
    assert task_schema.members_for(data, "multimodal", 4) == ["tep4"]


def test_a_per_unit_mapping_needs_a_default(tmp_path):
    spec = _unified(tmp_path, {"multimodal": ["tep4"]})
    with pytest.raises(task_schema.TaskSchemaError, match="default"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


def test_an_unknown_segment_is_rejected_as_a_member(tmp_path):
    spec = _unified(tmp_path, ["tp4", "ep4"])
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "members[1]" in str(excinfo.value)


def test_unified_is_not_a_member_of_itself(tmp_path):
    """Accepting it would be the TP-only gate under another name."""
    spec = _unified(tmp_path, ["tp4", "unified"])
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "family" in str(excinfo.value)


def test_mixed_world_sizes_are_rejected_naming_both(tmp_path):
    """One allocation serves every member, so a wider member cannot be run."""
    spec = _unified(tmp_path, ["tp4", "dep8"])
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(excinfo.value)
    assert "tp4" in message and "dep8" in message
    assert "4" in message and "8" in message


def test_members_beside_a_concrete_segment_is_an_error_that_says_so(tmp_path):
    """`parallel: tep4` already names the one configuration.

    Two declarations of the same thing can disagree.
    """
    spec = _valid_spec(tmp_path)
    spec["parallel"] = "tep4"
    spec["members"] = ["dep4"]
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "only legal with" in str(excinfo.value)
    assert "tep4" in str(excinfo.value)


def test_an_empty_member_list_is_a_unit_that_is_never_gated(tmp_path):
    spec = _unified(tmp_path, [])
    with pytest.raises(task_schema.TaskSchemaError, match="non-empty"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


def test_unknown_feature_is_rejected_by_name(tmp_path):
    """A feature with no assembler skill is an error here, not a confused agent later."""
    spec = _valid_spec(tmp_path)
    spec["features"] = ["mtp", "telepathy"]
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "telepathy" in str(excinfo.value)
    assert "mtp" in str(excinfo.value)  # listed among the known features


def test_every_known_feature_maps_to_a_skill():
    assert set(task_schema.FEATURE_SKILLS) >= {"mtp", "host_offload", "multimodal"}
    for feature, skill in task_schema.FEATURE_SKILLS.items():
        assert skill.startswith("trtllm-modeling-v2-"), feature


@pytest.mark.parametrize("feature", ["eplb", "dwdp"])
def test_features_with_no_target_behind_them_are_refused(tmp_path, feature):
    """No in-tree target exercises them, so there is no skill to build them from."""
    spec = _valid_spec(tmp_path)
    spec["features"] = [feature]
    with pytest.raises(task_schema.TaskSchemaError, match=feature):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


@pytest.mark.parametrize("kind", ["slurm", "slurm_broker"])
def test_only_local_and_session_spool_are_environment_kinds(tmp_path, kind):
    spec = _valid_spec(tmp_path)
    spec["environment"]["kind"] = kind
    with pytest.raises(task_schema.TaskSchemaError, match="session_spool"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


def test_local_kind_does_not_require_slurm_fields(tmp_path):
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, _valid_spec(tmp_path)))
    assert data["environment"]["kind"] == "local"


def test_negative_tolerance_is_rejected(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["accuracy"]["tolerance"] = -1
    with pytest.raises(task_schema.TaskSchemaError, match="tolerance"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


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


def test_missing_file_is_reported_as_such(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="task file not found"):
        task_schema.load_and_validate_task_yaml(tmp_path / "nope.yaml")


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(task_schema.TaskSchemaError, match="mapping"):
        task_schema.load_and_validate_task_yaml(path)


def test_example_task_yaml_matches_the_schema_fields():
    """The shipped example must not drift from the fields the schema reads."""
    from pathlib import Path

    import agent_flow.workflows.trtllm_modeling_v2 as pkg

    example = Path(pkg.__file__).parent / "task.example.yaml"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    for field in (*task_schema.REQUIRED_PATH_FIELDS, task_schema.ARCH_FIELD):
        assert field in data, field
    assert data["environment"]["kind"] in task_schema.ENVIRONMENT_KINDS
    assert "python" in data["environment"]
    assert set(data["accuracy"]) <= {"benchmark", "reference_yaml", "tolerance", "dataset_path"}
    for feature in data.get("features") or []:
        assert feature in task_schema.FEATURE_SKILLS


# ------------------------------------------- the substrate gate's proxy checkpoint


def test_a_substrate_checkpoint_that_exists_is_accepted(tmp_path):
    """The escape hatch for an architecture the stock path has never heard of.

    Optional, because a checkpoint the stock path already supports needs no
    proxy — and declared rather than guessed, because which model stands in
    for another is a judgement nothing in the schema can make.
    """
    proxy = tmp_path / "proxy-ckpt"
    proxy.mkdir()
    spec = _valid_spec(tmp_path)
    spec["environment"]["substrate_checkpoint"] = str(proxy)
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["environment"]["substrate_checkpoint"] == str(proxy)


def test_a_substrate_checkpoint_that_does_not_exist_is_rejected(tmp_path):
    """The gate reaches for it only AFTER the run's own checkpoint has failed.

    That is the longest step in preflight, so a missing proxy turns one
    failure into two, with a wait in between.
    """
    spec = _valid_spec(tmp_path)
    spec["environment"]["substrate_checkpoint"] = "/definitely/not/here"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "substrate_checkpoint' does not exist" in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "   ", 4])
def test_a_substrate_checkpoint_that_is_not_a_path_is_rejected(tmp_path, value):
    spec = _valid_spec(tmp_path)
    spec["environment"]["substrate_checkpoint"] = value
    with pytest.raises(task_schema.TaskSchemaError, match="substrate_checkpoint"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


# ------------------------------------------------------ closed key sets


def test_a_misspelled_feature_key_does_not_silently_build_core_only(tmp_path):
    """The most expensive typo available in this file.

    Accepted, `feature:` instead of `features:` would resolve to `[]`: the
    run onboards core, gates it green, and reports success for a feature set
    it never built — a green run that delivers the wrong thing.
    """
    spec = _valid_spec(tmp_path) | {"feature": ["mtp"]}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'feature'" in str(excinfo.value)
    assert "did you mean 'features'" in str(excinfo.value)


def test_a_misspelled_runtime_block_does_not_silently_drop_max_seq_len(tmp_path):
    """A dropped `runtme:` surfaces as `Impossible to fit in any sequence in kvCache`."""
    spec = _valid_spec(tmp_path) | {"runtme": {"max_seq_len": 8192}}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "did you mean 'runtime'" in str(excinfo.value)


@pytest.mark.parametrize(
    "block,typo,expected",
    [
        # The misspellings are the fixture, so they are spelled wrong on
        # purpose and codespell is told to leave them alone.
        ("environment", "pyhton", "environment.python"),  # codespell:ignore
        ("accuracy", "toleranace", "accuracy.tolerance"),
        ("target", "familly", "target.family"),  # codespell:ignore
    ],
)
def test_nested_blocks_are_closed_too(tmp_path, block, typo, expected):
    spec = _valid_spec(tmp_path)
    spec.setdefault(block, {})[typo] = "x"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert f"did you mean '{expected}'" in str(excinfo.value)


def test_an_unrecognizable_key_is_still_rejected_without_a_suggestion(tmp_path):
    """No near match is not a reason to accept it."""
    spec = _valid_spec(tmp_path) | {"zzzzzz": 1}
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "unknown key 'zzzzzz'" in str(excinfo.value)
    assert "did you mean" not in str(excinfo.value)


def test_every_key_the_schema_accepts_actually_validates(tmp_path):
    """The key lists and the validators must not drift apart.

    Adding a field to a validator and forgetting to list it here would make
    the new field an error; listing it here and forgetting the validator
    would make it silently ignored. This asserts the first direction —
    a spec using *every* declared key validates.
    """
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


def test_the_declared_key_lists_cover_the_documented_example(tmp_path):
    """`task.example.yaml` is a user's only map of the config surface.

    If it used a key the schema rejects, every copy of it would fail; if the
    schema accepted a key the example never shows, that field is invisible.
    This asserts the first direction — the shipped example validates as-is
    against the closed key sets.
    """
    from pathlib import Path

    example = Path(task_schema.__file__).parents[1] / "task.example.yaml"
    spec = yaml.safe_load(example.read_text(encoding="utf-8"))
    errors: list[str] = []
    task_schema.reject_unknown_keys(spec, task_schema.TOP_LEVEL_KEYS, errors)
    task_schema.reject_unknown_keys(
        spec["environment"], task_schema.ENVIRONMENT_KEYS, errors, where="environment"
    )
    task_schema.reject_unknown_keys(
        spec["accuracy"], task_schema.ACCURACY_KEYS, errors, where="accuracy"
    )
    assert errors == []


def test_the_example_documents_every_key_the_schema_accepts():
    """A field absent from the example does not exist, for a user.

    `task.example.yaml` is the only map of the config surface anyone reads,
    and the optional fields (`runtime`, `target`, `scope`, `source_embargo`,
    the accuracy protocol options) are the ones a run fails without.
    """
    import re
    from pathlib import Path

    example = Path(task_schema.__file__).parents[1] / "task.example.yaml"
    text = example.read_text(encoding="utf-8")
    # Commented-out keys count: an optional field is documented by showing
    # it commented with its default, not by setting it.
    mentioned = set(re.findall(r"^#?\s*#?\s*([a-z_]+):", text, re.M))

    for label, keys in (
        ("top level", task_schema.TOP_LEVEL_KEYS),
        ("accuracy", task_schema.ACCURACY_KEYS),
        ("environment", task_schema.ENVIRONMENT_KEYS),
        ("target", task_schema.TARGET_KEYS),
    ):
        undocumented = sorted(k for k in keys if k not in mentioned)
        assert not undocumented, f"{label} keys missing from task.example.yaml: {undocumented}"


def test_every_environment_kind_is_shown_in_the_example():
    """Picking a kind is the first decision a user makes."""
    from pathlib import Path

    example = Path(task_schema.__file__).parents[1] / "task.example.yaml"
    text = example.read_text(encoding="utf-8")
    for kind in task_schema.ENVIRONMENT_KINDS:
        assert kind in text, f"environment kind {kind!r} is undocumented"


# ------------------------------------------------------------------- CLI


def test_every_workflow_knob_is_reachable_from_the_command_line():
    """A constructor parameter with no flag is a feature nobody can use.

    An escape hatch such as `skip_substrate_gate` would otherwise be
    reachable only by editing Python.
    """
    import inspect

    from agent_flow.workflows.trtllm_modeling_v2.cli import _parse_args
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    args = vars(_parse_args(["--task", "t.yaml"]))
    parameters = inspect.signature(TrtllmModelingV2Workflow.__init__).parameters
    # Injection seams for the tests, not operator knobs.
    injected = {"self", "workspace", "prompts", "command_runner", "shell_runner", "git_runner"}
    unreachable = sorted(
        name for name in parameters if name not in injected and name.replace("-", "_") not in args
    )
    assert not unreachable, f"no CLI flag reaches: {unreachable}"


def test_the_substrate_gate_flag_reaches_the_workflow():
    from agent_flow.workflows.trtllm_modeling_v2.cli import _parse_args

    assert _parse_args(["--task", "t.yaml"]).skip_substrate_gate is False
    assert _parse_args(["--task", "t.yaml", "--skip-substrate-gate"]).skip_substrate_gate is True


# ------------------------------- removed launch keys


@pytest.mark.parametrize("key", ["workloads", "orchestrator", "allocator", "mpi"])
def test_a_removed_environment_key_is_rejected(tmp_path, key):
    spec = _valid_spec(tmp_path)
    spec["environment"][key] = {}
    with pytest.raises(task_schema.TaskSchemaError, match=key):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))


def test_the_session_wait_is_an_environment_key(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["environment"]["session_wait"] = "2h"
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["environment"]["session_wait"] == "2h"


# ------------------------------- a spec must work on another machine


def test_llm_api_options_written_relative_resolve_next_to_the_spec(tmp_path):
    """The per-unit runtime YAMLs are written NEXT TO task.yaml.

    Written absolute, the spec stops working on any other machine, and its
    most reusable artifact is unusable anywhere else.
    """
    (tmp_path / "core.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    spec = _valid_spec(tmp_path)
    spec.setdefault("accuracy", {})["llm_api_options"] = "core.yaml"
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["accuracy"]["llm_api_options"] == str((tmp_path / "core.yaml").resolve())


def test_a_relative_option_path_in_a_subdirectory_resolves_too(tmp_path):
    """The usual layout: a `scripts/` directory beside the spec."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "core_llm_api.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    spec = _valid_spec(tmp_path)
    spec.setdefault("accuracy", {})["llm_api_options"] = "scripts/core_llm_api.yaml"
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["accuracy"]["llm_api_options"] == str((scripts / "core_llm_api.yaml").resolve())


def test_the_per_unit_mapping_form_resolves_too(tmp_path):
    (tmp_path / "core.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    (tmp_path / "mtp.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    spec = _valid_spec(tmp_path)
    spec.setdefault("accuracy", {})["llm_api_options"] = {
        "default": "core.yaml",
        "mtp": str(tmp_path / "mtp.yaml"),
    }
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    options = data["accuracy"]["llm_api_options"]
    assert options["default"] == str((tmp_path / "core.yaml").resolve())
    # Absolute paths are untouched: a site that wants one keeps it.
    assert options["mtp"] == str(tmp_path / "mtp.yaml")


def test_a_relative_option_path_is_checked_for_existence_after_resolution(tmp_path):
    """Resolution runs BEFORE validation, so the message names the real path.

    Resolving afterwards would make a perfectly good relative path fail as
    "does not exist", which is the opposite of the point.
    """
    spec = _valid_spec(tmp_path)
    spec.setdefault("accuracy", {})["llm_api_options"] = "absent.yaml"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert str((tmp_path / "absent.yaml").resolve()) in str(excinfo.value)


def test_resolving_an_already_resolved_spec_is_idempotent(tmp_path):
    """The workflow copies task.yaml into the workspace and re-reads it."""
    (tmp_path / "core.yaml").write_text("kv_cache_config: {}\n", encoding="utf-8")
    spec = _valid_spec(tmp_path)
    spec.setdefault("accuracy", {})["llm_api_options"] = "core.yaml"
    once = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    spec["accuracy"]["llm_api_options"] = once["accuracy"]["llm_api_options"]
    twice = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert twice["accuracy"]["llm_api_options"] == once["accuracy"]["llm_api_options"]


def test_machine_specific_paths_are_deliberately_left_alone(tmp_path):
    """checkpoint_path and friends differ per machine and are read elsewhere.

    Making them spec-relative would move the portability problem, not solve it.
    """
    import inspect

    spec = _valid_spec(tmp_path)
    expected = spec["checkpoint_path"]
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    src = inspect.getsource(task_schema._resolve_spec_relative_paths)
    assert "checkpoint_path" in src and "Deliberately NOT applied" in src
    assert data["checkpoint_path"] == expected


def test_a_spec_with_no_accuracy_block_survives_resolution(tmp_path):
    """Resolution runs before validation, so it meets malformed specs first."""
    data: dict = {}
    task_schema._resolve_spec_relative_paths(data, tmp_path)
    assert data == {}
    data = {"accuracy": "not-a-mapping"}
    task_schema._resolve_spec_relative_paths(data, tmp_path)
    assert data == {"accuracy": "not-a-mapping"}


# ---------------------------- the speculator, replay and kv-quant feature kinds


def test_the_new_feature_kinds_map_to_their_skills():
    skills = task_schema.FEATURE_SKILLS
    assert skills["dspark"] == "trtllm-modeling-v2-speculator"
    assert skills["decoder_replay"] == "trtllm-modeling-v2-bounded-replay"
    assert skills["encoder_replay"] == "trtllm-modeling-v2-bounded-replay"
    assert skills["kv_quant"] == "trtllm-modeling-v2-kv-quant"


def test_an_unknown_feature_lists_the_new_kinds(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["features"] = ["swa_bounded_replay"]
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    message = str(excinfo.value)
    for kind in ("dspark", "decoder_replay", "encoder_replay", "kv_quant"):
        assert kind in message


def test_mtp_and_dspark_are_refused_together(tmp_path):
    """Both read `mtp.*`; the namespace holds one drafter, not two."""
    spec = _valid_spec(tmp_path)
    spec["features"] = ["mtp", "dspark"]
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert "spelled `dspark`" in str(excinfo.value)


def test_dspark_alone_validates(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["features"] = ["dspark"]
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert task_schema.assembly_units(data) == ["core", "dspark"]


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
    spec["accuracy"]["witness"] = witness
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


@pytest.mark.parametrize("feature", ["decoder_replay", "encoder_replay", "kv_quant"])
def test_a_declared_witness_is_required_at_spec_time(tmp_path, feature):
    """Declared before the run it judges, which is only certain here."""
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(_with_witness(tmp_path, [feature], {}))
    assert f"accuracy.witness.{feature}" in str(excinfo.value)


def test_a_replay_workload_inside_the_window_is_refused(tmp_path):
    """At or under the window, replay on and off do the same work."""
    block = _DECODER | {"prompt_tokens": 128}
    with pytest.raises(task_schema.TaskSchemaError, match="does not exceed window=128"):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["decoder_replay"], {"decoder_replay": block})
        )


def test_a_replay_workload_past_the_argv_cap_is_refused(tmp_path):
    block = _ENCODER | {"prefix_tokens": task_schema.MAX_WITNESS_PROMPT_WORDS + 1}
    with pytest.raises(task_schema.TaskSchemaError, match="128 KiB"):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["encoder_replay"], {"encoder_replay": block})
        )


@pytest.mark.parametrize("missing", ["contrast_env", "window", "prompt_tokens"])
def test_every_replay_parameter_is_required(tmp_path, missing):
    block = {k: v for k, v in _DECODER.items() if k != missing}
    with pytest.raises(task_schema.TaskSchemaError, match=missing):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["decoder_replay"], {"decoder_replay": block})
        )


def test_witness_blocks_are_closed_per_feature(tmp_path):
    """`expected_mib` under a replay feature reads as a check nobody makes."""
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        task_schema.load_and_validate_task_yaml(
            _with_witness(
                tmp_path,
                ["decoder_replay", "host_offload"],
                {
                    "decoder_replay": _DECODER | {"expected_mib": 1, "prefix_tokens": 4096},
                    "host_offload": {"contrast_env": {"X": "0"}, "window": 128},
                },
            )
        )
    message = str(excinfo.value)
    assert "accuracy.witness.decoder_replay.expected_mib" in message
    assert "accuracy.witness.decoder_replay.prefix_tokens" in message
    assert "accuracy.witness.host_offload.window" in message


def test_an_exact_kv_quant_needs_a_contrast(tmp_path):
    """Without one, token identity compares native storage with itself."""
    exact = {"expected_bytes_per_token": 890}
    with pytest.raises(task_schema.TaskSchemaError, match="contrast_env"):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["kv_quant"], {"kv_quant": exact})
        )


def test_a_lossy_kv_quant_does_not_need_a_contrast(tmp_path):
    """`exact: false` drops the identity leg, and with it the contrast."""
    lossy = {"expected_bytes_per_token": 890, "exact": False}
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(tmp_path, ["kv_quant"], {"kv_quant": lossy})
    )
    assert data["accuracy"]["witness"]["kv_quant"]["exact"] is False


@pytest.mark.parametrize(
    "bad",
    [
        {"expected_bytes_per_token": 0},
        {"expected_bytes_per_token": "890"},
        {"bytes_tolerance": 1.5},
        {"exact": "yes"},
    ],
)
def test_kv_quant_values_are_checked(tmp_path, bad):
    with pytest.raises(task_schema.TaskSchemaError):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["kv_quant"], {"kv_quant": _KV | bad})
        )


# ------------------------------------------------- the equivalence witness

_DISAGG = {
    "contrast_env": {"TRTLLM_DISAGG": "1"},
    "contrast_command": "/onboard/scripts/disagg_leg_b.py",
}


def test_a_disagg_witness_without_a_launcher_of_its_own_runs_the_flows(tmp_path):
    """Leg B is the package's launcher unless the run names one; nothing else is owed."""
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(tmp_path, ["disagg"], {"disagg": {"max_batch_size": 1}})
    )
    assert "contrast_command" not in data["accuracy"]["witness"]["disagg"]


@pytest.mark.parametrize(
    "block,message",
    [
        ({"contrast_command": "/onboard/leg_b.py"}, "contrast_env' is required"),
        ({"contrast_command": "  ", "contrast_env": {"X": "1"}}, "or left out for the flow's"),
        ({"contrast_env": {}}, "contrast_env' is required and must be a non-empty mapping"),
        ({"cache_transceiver_config": "NIXL"}, "mapping of CacheTransceiverConfig fields"),
        ({"cache_transceiver_config": {}}, "mapping of CacheTransceiverConfig fields"),
    ],
)
def test_a_disagg_witness_block_is_checked(tmp_path, block, message):
    with pytest.raises(task_schema.TaskSchemaError, match=message):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["disagg"], {"disagg": block})
        )


def test_the_flows_leg_b_takes_the_cache_transceiver_it_runs(tmp_path):
    block = {task_schema.WITNESS_TRANSCEIVER_KEY: {"backend": "NIXL"}}
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(tmp_path, ["disagg"], {"disagg": block})
    )
    assert data["accuracy"]["witness"]["disagg"]["cache_transceiver_config"] == {"backend": "NIXL"}


def test_a_launcher_beside_the_spec_is_found_from_anywhere(tmp_path):
    """A word of the command naming a file beside the spec is made absolute.

    A check runs inside a snapshot of the tree, where the spec's file is not;
    a word naming no file beside it (a flag, a path of the tree) is left as written.
    """
    (tmp_path / "leg_b.py").write_text("print('leg B')\n", encoding="utf-8")
    command = "python leg_b.py --anchors=table.yaml examples/disaggregated/launch.sh"
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(tmp_path, ["disagg"], {"disagg": _DISAGG | {"contrast_command": command}})
    )
    resolved = data["accuracy"]["witness"]["disagg"]["contrast_command"]
    assert shlex.split(resolved) == [
        "python",
        str((tmp_path / "leg_b.py").resolve()),
        "--anchors=table.yaml",
        "examples/disaggregated/launch.sh",
    ]
    untouched = "bash 'examples/disaggregated/launch.sh' --n=2"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(elsewhere, ["disagg"], {"disagg": _DISAGG | {"contrast_command": untouched}})
    )
    assert data["accuracy"]["witness"]["disagg"]["contrast_command"] == untouched


def test_a_disagg_witness_may_declare_its_prompt_lengths_and_timeout(tmp_path):
    block = _DISAGG | {"prompt_tokens": [131, 2050, 2051, 4097], "timeout": 2700}
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(tmp_path, ["disagg"], {"disagg": block})
    )
    assert data["accuracy"]["witness"]["disagg"]["prompt_tokens"] == [131, 2050, 2051, 4097]


@pytest.mark.parametrize(
    "bad,message",
    [
        ({"prompt_tokens": []}, "non-empty list"),
        ({"prompt_tokens": 2051}, "non-empty list"),
        ({"prompt_tokens": [0]}, "integers >= 1"),
        ({"prompt_tokens": [task_schema.MAX_WITNESS_PROMPT_WORDS + 1]}, "128 KiB"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": "45m"}, "timeout"),
        ({"expected_mib": 1}, "expected_mib"),
    ],
)
def test_disagg_witness_values_are_checked(tmp_path, bad, message):
    with pytest.raises(task_schema.TaskSchemaError, match=message):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["disagg"], {"disagg": _DISAGG | bad})
        )


def test_every_witness_block_takes_a_timeout(tmp_path):
    """Seconds per leg; the one-engine boot budget is only a default."""
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(
            tmp_path,
            ["host_offload", "kv_quant", "decoder_replay", "encoder_replay"],
            {
                "host_offload": {"contrast_env": {"X": "0"}, "expected_mib": 1, "timeout": 900},
                "kv_quant": _KV | {"timeout": 900},
                "decoder_replay": _DECODER | {"timeout": 900},
                "encoder_replay": _ENCODER | {"timeout": 900},
            },
        )
    )
    assert all(block["timeout"] == 900 for block in data["accuracy"]["witness"].values())


def test_prompt_lengths_are_not_a_residency_key(tmp_path):
    """Closed per feature: nothing would read them."""
    block = {"contrast_env": {"X": "0"}, "prompt_tokens": [2051]}
    with pytest.raises(task_schema.TaskSchemaError, match="host_offload.prompt_tokens"):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["host_offload"], {"host_offload": block})
        )


_REPLAYS_OFF = {
    "TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY": "0",
    "TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY": "0",
}


def test_a_disagg_witness_may_pin_an_environment_on_both_legs(tmp_path):
    """Paths leg B's engines guard off under disagg are pinned off in leg A too."""
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(tmp_path, ["disagg"], {"disagg": _DISAGG | {"legs_env": _REPLAYS_OFF}})
    )
    assert data["accuracy"]["witness"]["disagg"]["legs_env"] == _REPLAYS_OFF


@pytest.mark.parametrize(
    "legs_env,message",
    [
        ({}, "non-empty mapping"),
        (["TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY=0"], "non-empty mapping"),
        ({"1_REPLAY": "0"}, "not a shell variable name"),
        ({"A-B": "0"}, "not a shell variable name"),
        ({"REPLAY": ["0"]}, "scalar"),
        # Set one way on leg A and another on leg B: that is the contrast.
        ({"TRTLLM_DISAGG": "0"}, "is the contrast"),
    ],
)
def test_legs_env_values_are_checked(tmp_path, legs_env, message):
    with pytest.raises(task_schema.TaskSchemaError, match=message):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, ["disagg"], {"disagg": _DISAGG | {"legs_env": legs_env}})
        )


@pytest.mark.parametrize(
    "feature,block",
    [
        ("host_offload", {"contrast_env": {"X": "0"}}),
        ("kv_quant", _KV),
        ("decoder_replay", _DECODER),
    ],
)
def test_legs_env_is_an_equivalence_key_only(tmp_path, feature, block):
    """The other witnesses run one command on both legs; only `contrast_env` differs."""
    with pytest.raises(task_schema.TaskSchemaError, match=f"{feature}.legs_env"):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, [feature], {feature: block | {"legs_env": _REPLAYS_OFF}})
        )


# -------------------------------------------- the acceptance gate's workload


def _with_acceptance(tmp_path, acceptance) -> str:
    spec = _valid_spec(tmp_path)
    spec["accuracy"]["acceptance"] = acceptance
    return _write(tmp_path, spec)


def test_the_per_unit_floors_alone_still_validate(tmp_path):
    """Per-unit floors without a declared workload validate as they are."""
    acceptance = {"dspark": {"min_absolute": 2.0}, "multimodal": {"min_absolute": 2.0}, "mtp": 1.5}
    data = task_schema.load_and_validate_task_yaml(_with_acceptance(tmp_path, acceptance))
    assert data["accuracy"]["acceptance"] == acceptance


def test_a_declared_workload_validates_run_wide_and_per_unit(tmp_path):
    acceptance = {
        "prompts": ["<BOS>Explain, step by step, how a heat pump works."],
        "max_tokens": 256,
        "min_draft_iterations": 64,
        "dspark": {"min_absolute": 2.0},
        "multimodal": {"min_absolute": 2.0, "prompts": ["<BOS>Describe the image."]},
    }
    data = task_schema.load_and_validate_task_yaml(_with_acceptance(tmp_path, acceptance))
    assert data["accuracy"]["acceptance"]["max_tokens"] == 256


def test_a_relative_prompts_file_resolves_next_to_the_spec(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "long.json").write_text('["a long question"]', encoding="utf-8")
    (tmp_path / "boot.yaml").write_text("prompts: [from boot]\n", encoding="utf-8")
    acceptance = {
        "prompts_file": "scripts/long.json",
        "dspark": {"min_absolute": 2.0, "prompts_file": "boot.yaml"},
    }
    data = task_schema.load_and_validate_task_yaml(_with_acceptance(tmp_path, acceptance))
    block = data["accuracy"]["acceptance"]
    assert block["prompts_file"] == str((tmp_path / "scripts" / "long.json").resolve())
    assert block["dspark"]["prompts_file"] == str((tmp_path / "boot.yaml").resolve())


@pytest.mark.parametrize(
    "acceptance,message",
    [
        ({"prompts": ["a"], "prompts_file": "p.json"}, "both 'prompts' and 'prompts_file'"),
        ({"prompts": []}, "acceptance.prompts' must be a non-empty list"),
        ({"prompts": ["ok", 3]}, "acceptance.prompts' must be a non-empty list"),
        ({"prompts": "one string"}, "acceptance.prompts' must be a non-empty list"),
        ({"prompts_file": "missing.json"}, "prompts_file' does not exist"),
        ({"max_tokens": 0}, "acceptance.max_tokens' must be an integer >= 1"),
        ({"min_draft_iterations": True}, "min_draft_iterations' must be an integer >= 1"),
        ({"dspark": {"max_tokens": "256"}}, "dspark.max_tokens' must be an integer >= 1"),
        (
            {"dspark": {"min_absolut": 2.0}},
            "did you mean 'accuracy.acceptance.dspark.min_absolute'",
        ),
        ({"dspark": {"min_absolute": 0.5}}, "dspark.min_absolute' must be a number >= 1.0"),
        ({"mtp": "high"}, "acceptance.mtp' must be a number >= 1.0"),
        (["dspark"], "acceptance' must be a mapping"),
    ],
)
def test_a_malformed_acceptance_block_is_rejected(tmp_path, acceptance, message):
    with pytest.raises(task_schema.TaskSchemaError, match=message):
        task_schema.load_and_validate_task_yaml(_with_acceptance(tmp_path, acceptance))


def test_a_prompts_file_that_holds_no_prompts_is_rejected(tmp_path):
    (tmp_path / "empty.yaml").write_text("keywords: [x]\n", encoding="utf-8")
    with pytest.raises(task_schema.TaskSchemaError, match="non-empty list of strings"):
        task_schema.load_and_validate_task_yaml(
            _with_acceptance(tmp_path, {"prompts_file": "empty.yaml"})
        )


# ----------------------------------- one schedule for token-comparing legs


def test_token_comparing_witnesses_take_a_batch_limit(tmp_path):
    """Batch composition moves tokens; `max_batch_size: 1` gives both legs one schedule."""
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(
            tmp_path,
            ["host_offload", "kv_quant", "disagg"],
            {
                "host_offload": {
                    "contrast_env": {"X": "0"},
                    "expected_mib": 1,
                    "max_batch_size": 1,
                },
                "kv_quant": _KV | {"max_batch_size": 1},
                "disagg": _DISAGG | {"max_batch_size": 1},
            },
        )
    )
    witness = data["accuracy"]["witness"]
    assert {unit: block["max_batch_size"] for unit, block in witness.items()} == {
        "host_offload": 1,
        "kv_quant": 1,
        "disagg": 1,
    }


def test_a_block_without_a_batch_limit_still_validates(tmp_path):
    """The key is optional, and an absent one gets no default."""
    data = task_schema.load_and_validate_task_yaml(
        _with_witness(tmp_path, ["kv_quant", "disagg"], {"kv_quant": _KV, "disagg": _DISAGG})
    )
    assert all("max_batch_size" not in b for b in data["accuracy"]["witness"].values())


def test_a_replay_witness_takes_no_batch_limit(tmp_path):
    """It counts work, not tokens; the key would read as a check nobody makes."""
    with pytest.raises(task_schema.TaskSchemaError, match="decoder_replay.max_batch_size"):
        task_schema.load_and_validate_task_yaml(
            _with_witness(
                tmp_path, ["decoder_replay"], {"decoder_replay": _DECODER | {"max_batch_size": 1}}
            )
        )


@pytest.mark.parametrize("bad", [0, -1, "1", 1.0, True])
@pytest.mark.parametrize("unit", ["kv_quant", "disagg", "host_offload"])
def test_a_batch_limit_is_a_positive_integer(tmp_path, unit, bad):
    base = {
        "kv_quant": _KV,
        "disagg": _DISAGG,
        "host_offload": {"contrast_env": {"X": "0"}, "expected_mib": 1},
    }[unit]
    with pytest.raises(task_schema.TaskSchemaError, match=f"{unit}.max_batch_size"):
        task_schema.load_and_validate_task_yaml(
            _with_witness(tmp_path, [unit], {unit: base | {"max_batch_size": bad}})
        )


def test_notes_are_an_optional_non_empty_string(tmp_path):
    spec = _valid_spec(tmp_path)
    spec["notes"] = "Reuse the in-tree NVFP4 MoE kernels."
    data = task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
    assert data["notes"] == "Reuse the in-tree NVFP4 MoE kernels."
    spec["notes"] = "   "
    with pytest.raises(task_schema.TaskSchemaError, match="'notes' must be a non-empty string"):
        task_schema.load_and_validate_task_yaml(_write(tmp_path, spec))
