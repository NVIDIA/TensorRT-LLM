"""Tests for perf-analyze's task.yaml schema validation."""

from __future__ import annotations

import math

import pytest
import yaml

from agent_flow.workflows.perf_analyze.task_schema import (
    TaskSchemaError,
    concurrency_points,
    dump_task_yaml,
    has_slurm_environment,
    is_curve_mode,
    load_and_validate_task_yaml,
    num_prompts_per_point,
    sol_enabled,
)


def _write(tmp_path, mapping) -> str:
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(p)


def _paths(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    return str(ckpt), str(repo)


# ------------------------------------------------------------------ happy path


def test_valid_minimal_applies_defaults(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(tmp_path, {"checkpoint_path": ckpt, "trtllm_repo_path": repo})
    data = load_and_validate_task_yaml(path)

    # No serve block any more; extra_llm_api_options stays omitted by default.
    assert "serve" not in data
    assert "extra_llm_api_options" not in data
    assert data["benchmark"]["dataset_name"] == "random"
    assert data["benchmark"]["random_input_len"] == 1024
    assert data["benchmark"]["random_output_len"] == 128
    assert data["benchmark"]["num_prompts"] == 200
    assert data["benchmark"]["concurrency"] == 64
    assert data["profile"] == {
        "methods": ["nsys", "torch", "ncu"],
        "nsys_iter_range": "100-150",
    }
    assert has_slurm_environment(data) is False
    # The projector is on by default, and the block is materialized so
    # the resolved spec states the gate the agents read.
    assert data["sol"] == {"enabled": True}
    assert sol_enabled(data) is True
    assert is_curve_mode(data) is False
    assert concurrency_points(data) == [64]


def test_user_values_override_defaults(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"random_input_len": 512, "num_prompts": 50},
            "profile": {"methods": ["nsys"]},
        },
    )
    data = load_and_validate_task_yaml(path)

    assert data["benchmark"]["random_input_len"] == 512
    assert data["benchmark"]["random_output_len"] == 128  # default preserved
    assert data["profile"]["methods"] == ["nsys"]
    assert data["profile"]["nsys_iter_range"] == "100-150"  # default preserved


def test_request_rate_accepts_number_and_inf(tmp_path):
    ckpt, repo = _paths(tmp_path)
    for rr in (10, 2.5, "inf", "Infinity"):
        path = _write(
            tmp_path,
            {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "benchmark": {"request_rate": rr}},
        )
        data = load_and_validate_task_yaml(path)
        assert data["benchmark"]["request_rate"] == rr


def test_extra_llm_api_options_existing_path_ok(tmp_path):
    ckpt, repo = _paths(tmp_path)
    cfg = tmp_path / "extra.yaml"
    cfg.write_text("k: v\n", encoding="utf-8")
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "extra_llm_api_options": str(cfg),
        },
    )
    data = load_and_validate_task_yaml(path)
    assert data["extra_llm_api_options"] == str(cfg)


def test_slurm_environment_valid(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "slurm-environment": {"slurm_partition": "gpu", "docker_image": "/img.sqsh"},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert has_slurm_environment(data) is True


def test_sol_block_absent_defaults_to_enabled(tmp_path):
    # The projector runs unless the spec turns it off, so an untouched
    # task.yaml resolves to an explicit ``enabled: true``.
    ckpt, repo = _paths(tmp_path)
    path = _write(tmp_path, {"checkpoint_path": ckpt, "trtllm_repo_path": repo})
    data = load_and_validate_task_yaml(path)
    assert data["sol"] == {"enabled": True}
    assert sol_enabled(data) is True


def test_sol_block_minimal_gets_the_enabled_default(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert sol_enabled(data) is True
    assert data["sol"] == {"enabled": True}


def test_sol_bare_key_normalized_to_enabled_mapping(tmp_path):
    # Every field is optional, so a bare ``sol:`` (YAML null) says
    # nothing the defaults do not — normalized in the resolved spec.
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": None,
        },
    )
    data = load_and_validate_task_yaml(path)
    assert sol_enabled(data) is True
    assert data["sol"] == {"enabled": True}


def test_sol_enabled_false_disables_the_stage(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {"enabled": False},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert sol_enabled(data) is False
    assert data["sol"] == {"enabled": False}


def test_sol_enabled_false_survives_the_gpu_hint(tmp_path):
    # Disabling and hinting are independent keys — the user's ``enabled``
    # wins over the default without dropping the rest of the block.
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {"enabled": False, "gpu": "H100"},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert sol_enabled(data) is False
    assert data["sol"] == {"enabled": False, "gpu": "H100"}


def test_sol_bare_enabled_key_falls_back_to_the_default(tmp_path):
    # ``enabled:`` with no value states nothing — the resolved spec gets
    # the default rather than a null a later reader has to interpret.
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {"enabled": None, "gpu": "H100"},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert data["sol"] == {"enabled": True, "gpu": "H100"}
    assert sol_enabled(data) is True


def test_sol_enabled_wrong_type(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {"enabled": "no"},
        },
    )
    with pytest.raises(TaskSchemaError, match="'sol.enabled' must be a boolean"):
        load_and_validate_task_yaml(path)


def test_sol_enabled_on_a_raw_unvalidated_spec(tmp_path):
    # ``sol_enabled`` also reads specs that never went through the
    # validator (a resume re-reading task.yaml): absent and bare-``sol:``
    # both mean enabled, only an explicit false turns it off.
    assert sol_enabled({}) is True
    assert sol_enabled({"sol": None}) is True
    assert sol_enabled({"sol": {}}) is True
    assert sol_enabled({"sol": {"gpu": "H100"}}) is True
    assert sol_enabled({"sol": {"enabled": False}}) is False


def test_sol_block_with_gpu_hint(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {"gpu": "H100"},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert data["sol"]["gpu"] == "H100"
    # The gate default is merged under the user's hint.
    assert data["sol"]["enabled"] is True


# ------------------------------------------------------------------ concurrency


def test_concurrency_scalar_accepted(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "benchmark": {"concurrency": 256}},
    )
    data = load_and_validate_task_yaml(path)
    assert data["benchmark"]["concurrency"] == 256
    assert is_curve_mode(data) is False
    assert concurrency_points(data) == [256]


def test_concurrency_list_normalized_sorted_unique(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [64, 16, 32, 16]},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert data["benchmark"]["concurrency"] == [16, 32, 64]
    assert is_curve_mode(data) is True
    assert concurrency_points(data) == [16, 32, 64]


def test_concurrency_single_element_list_stays_curve_mode(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "benchmark": {"concurrency": [64]}},
    )
    data = load_and_validate_task_yaml(path)
    assert data["benchmark"]["concurrency"] == [64]
    assert is_curve_mode(data) is True


@pytest.mark.parametrize(
    "value",
    [0, -5, True, "many", [], [0], [-1], [True], ["16"], [16, "32"], {"a": 1}],
)
def test_concurrency_rejects_non_positive_ints(tmp_path, value):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "benchmark": {"concurrency": value}},
    )
    with pytest.raises(TaskSchemaError, match="positive integer"):
        load_and_validate_task_yaml(path)


def test_max_concurrency_rejected_with_rename_pointer(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"max_concurrency": 64},
        },
    )
    with pytest.raises(TaskSchemaError) as exc:
        load_and_validate_task_yaml(path)
    msg = str(exc.value)
    assert "renamed to" in msg
    assert "'benchmark.concurrency'" in msg


def test_max_concurrency_error_batched_with_others(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"max_concurrency": 64, "num_prompts": "lots"},
        },
    )
    with pytest.raises(TaskSchemaError) as exc:
        load_and_validate_task_yaml(path)
    msg = str(exc.value)
    assert "renamed to" in msg
    assert "benchmark.num_prompts" in msg


# ------------------------------------------------------------------ num_prompts


def test_num_prompts_scalar_broadcasts_across_points(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [8, 32], "num_prompts": 100},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert data["benchmark"]["num_prompts"] == 100
    assert num_prompts_per_point(data) == [100, 100]


def test_num_prompts_list_sorted_together_with_concurrency(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [128, 8, 32], "num_prompts": [512, 32, 128]},
        },
    )
    data = load_and_validate_task_yaml(path)
    assert data["benchmark"]["concurrency"] == [8, 32, 128]
    assert data["benchmark"]["num_prompts"] == [32, 128, 512]
    assert num_prompts_per_point(data) == [32, 128, 512]


def test_num_prompts_list_requires_curve_mode(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": 64, "num_prompts": [64, 128]},
        },
    )
    with pytest.raises(TaskSchemaError, match="only be a list in Pareto-curve mode"):
        load_and_validate_task_yaml(path)


def test_num_prompts_list_length_must_match_points(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [8, 32, 128], "num_prompts": [32, 128]},
        },
    )
    with pytest.raises(TaskSchemaError, match="pair one-to-one"):
        load_and_validate_task_yaml(path)


def test_num_prompts_entry_must_cover_its_point(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [8, 256], "num_prompts": [32, 128]},
        },
    )
    with pytest.raises(TaskSchemaError, match="num_prompts 128 < concurrency 256"):
        load_and_validate_task_yaml(path)


def test_num_prompts_list_rejects_duplicate_concurrency(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [8, 8, 32], "num_prompts": [16, 32, 64]},
        },
    )
    with pytest.raises(TaskSchemaError, match="must not contain duplicates"):
        load_and_validate_task_yaml(path)


@pytest.mark.parametrize("value", [0, -5, True, "lots", [], [0], [-1], [True], [16, "32"]])
def test_num_prompts_rejects_non_positive_ints(tmp_path, value):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [8, 32], "num_prompts": value},
        },
    )
    with pytest.raises(TaskSchemaError, match="benchmark.num_prompts"):
        load_and_validate_task_yaml(path)


def test_num_prompts_helper_is_total_on_unvalidated_data():
    assert num_prompts_per_point({}) == []
    assert num_prompts_per_point({"benchmark": "oops"}) == []
    assert num_prompts_per_point({"benchmark": {"concurrency": [8]}}) == []
    assert num_prompts_per_point({"benchmark": {"num_prompts": 100}}) == []


def test_dump_task_yaml_round_trips_num_prompts_list(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [128, 8], "num_prompts": [512, 32]},
        },
    )
    data = load_and_validate_task_yaml(path)
    reloaded = yaml.safe_load(dump_task_yaml(data))
    assert reloaded["benchmark"]["concurrency"] == [8, 128]
    assert reloaded["benchmark"]["num_prompts"] == [32, 512]


def test_concurrency_helpers_are_total_on_unvalidated_data():
    assert is_curve_mode({}) is False
    assert concurrency_points({}) == []
    assert is_curve_mode({"benchmark": "oops"}) is False
    assert concurrency_points({"benchmark": "oops"}) == []
    assert concurrency_points({"benchmark": {}}) == []


def test_dump_task_yaml_round_trips_concurrency_list(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"concurrency": [128, 8]},
        },
    )
    data = load_and_validate_task_yaml(path)
    reloaded = yaml.safe_load(dump_task_yaml(data))
    assert reloaded["benchmark"]["concurrency"] == [8, 128]


# ------------------------------------------------------------------ errors


def test_missing_task_file(tmp_path):
    with pytest.raises(TaskSchemaError, match="not found"):
        load_and_validate_task_yaml(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    p = tmp_path / "task.yaml"
    p.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(TaskSchemaError, match="not valid YAML"):
        load_and_validate_task_yaml(p)


def test_top_level_must_be_mapping(tmp_path):
    p = tmp_path / "task.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(TaskSchemaError, match="mapping"):
        load_and_validate_task_yaml(p)


def test_missing_required_fields_batched(tmp_path):
    path = _write(tmp_path, {"serve": {}})
    with pytest.raises(TaskSchemaError) as exc:
        load_and_validate_task_yaml(path)
    msg = str(exc.value)
    assert "checkpoint_path" in msg
    assert "trtllm_repo_path" in msg


def test_nonexistent_required_path(tmp_path):
    _, repo = _paths(tmp_path)
    path = _write(tmp_path, {"checkpoint_path": "/no/such/dir", "trtllm_repo_path": repo})
    with pytest.raises(TaskSchemaError, match="non-existent"):
        load_and_validate_task_yaml(path)


def test_bad_types_are_batched(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"num_prompts": "lots"},
            "profile": {"methods": ["nsys", "bogus"]},
        },
    )
    with pytest.raises(TaskSchemaError) as exc:
        load_and_validate_task_yaml(path)
    msg = str(exc.value)
    assert "benchmark.num_prompts" in msg
    assert "profile.methods" in msg


def test_bool_is_rejected_for_int_field(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "benchmark": {"num_prompts": True}},
    )
    with pytest.raises(TaskSchemaError, match="benchmark.num_prompts"):
        load_and_validate_task_yaml(path)


def test_request_rate_rejects_garbage_string(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "benchmark": {"request_rate": "fast"}},
    )
    with pytest.raises(TaskSchemaError, match="request_rate"):
        load_and_validate_task_yaml(path)


def test_extra_llm_api_options_missing_path(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "extra_llm_api_options": "/no/such.yaml",
        },
    )
    with pytest.raises(TaskSchemaError, match="extra_llm_api_options"):
        load_and_validate_task_yaml(path)


def test_extra_llm_api_options_wrong_type(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "extra_llm_api_options": 123,
        },
    )
    with pytest.raises(TaskSchemaError, match="extra_llm_api_options.*non-empty string"):
        load_and_validate_task_yaml(path)


def test_block_must_be_mapping(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "benchmark": "oops"},
    )
    with pytest.raises(TaskSchemaError, match="'benchmark' must be a mapping"):
        load_and_validate_task_yaml(path)


def test_slurm_environment_missing_field(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "slurm-environment": {"slurm_partition": "gpu"},
        },
    )
    with pytest.raises(TaskSchemaError, match="docker_image"):
        load_and_validate_task_yaml(path)


def test_stale_dlsim_block_rejected_with_pointer_to_sol(tmp_path):
    # The projector's gate was renamed dlsim -> sol when the dlsim
    # cross-check was dropped; a stale spec must fail loudly instead of
    # having its projector settings silently ignored.
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "dlsim": {"repo_path": "/some/dlsim"},
        },
    )
    with pytest.raises(TaskSchemaError, match="'dlsim' was replaced by 'sol'"):
        load_and_validate_task_yaml(path)


def test_sol_block_must_be_mapping(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "sol": "oops"},
    )
    with pytest.raises(TaskSchemaError, match="'sol' must be a mapping"):
        load_and_validate_task_yaml(path)


def test_sol_bool_rejected_with_pointer_to_the_enabled_field(tmp_path):
    """``sol: false`` is not a spelling of the gate — say where it lives."""
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {"checkpoint_path": ckpt, "trtllm_repo_path": repo, "sol": False},
    )
    with pytest.raises(TaskSchemaError, match="the stage gate is a field inside it"):
        load_and_validate_task_yaml(path)


def test_sol_unknown_field_rejected(tmp_path):
    """A misspelled gate is not an inert extra key — it inverts intent.

    ``enabled`` is the only thing that decides whether the projector
    stage runs, so a preserved ``enable:`` would silently leave the
    default in place and run the stage the block was written to skip.
    """
    ckpt, repo = _paths(tmp_path)
    for typo in ("enable", "Enabled", "enabld"):
        path = _write(
            tmp_path,
            {
                "checkpoint_path": ckpt,
                "trtllm_repo_path": repo,
                "sol": {typo: False},
            },
        )
        with pytest.raises(TaskSchemaError) as exc:
            load_and_validate_task_yaml(path)
        # The error names the offender and the spellings that do work.
        assert f"'{typo}'" in str(exc.value)
        assert "'enabled', 'gpu'" in str(exc.value)


def test_sol_known_fields_are_not_flagged_as_unknown(tmp_path):
    """The guard must not fire on the two fields the block actually has."""
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {"enabled": True, "gpu": "B200"},
        },
    )
    assert load_and_validate_task_yaml(path)["sol"] == {"enabled": True, "gpu": "B200"}


def test_sol_gpu_wrong_type(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "sol": {"gpu": 123},
        },
    )
    with pytest.raises(TaskSchemaError, match="sol.gpu.*non-empty string"):
        load_and_validate_task_yaml(path)


def test_sol_errors_batched_with_others(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(
        tmp_path,
        {
            "checkpoint_path": ckpt,
            "trtllm_repo_path": repo,
            "benchmark": {"num_prompts": "lots"},
            "sol": {"gpu": 42},
        },
    )
    with pytest.raises(TaskSchemaError) as exc:
        load_and_validate_task_yaml(path)
    msg = str(exc.value)
    assert "benchmark.num_prompts" in msg
    assert "sol.gpu" in msg


# ------------------------------------------------------------------ dump


def test_dump_task_yaml_round_trip(tmp_path):
    ckpt, repo = _paths(tmp_path)
    path = _write(tmp_path, {"checkpoint_path": ckpt, "trtllm_repo_path": repo})
    data = load_and_validate_task_yaml(path)
    reloaded = yaml.safe_load(dump_task_yaml(data))
    assert reloaded["checkpoint_path"] == ckpt
    assert "serve" not in reloaded
    assert reloaded["profile"]["methods"] == ["nsys", "torch", "ncu"]


def test_dump_task_yaml_normalizes_float_inf():
    text = dump_task_yaml({"benchmark": {"request_rate": math.inf}})
    assert yaml.safe_load(text)["benchmark"]["request_rate"] == "inf"


# ── where the paths live decides whether they are checked ───────────────────
#
# `Path(...).exists()` silently assumes the validator and the paths are on one
# machine. That held until the flow started running off-cluster, where it reports
# False for a checkpoint that is perfectly present and refuses a run for a defect
# it does not have. `cluster_ssh` is the spec saying which machine it is on, so
# the schema asks the spec instead of assuming, and instead of being told by a
# `--paths-prevalidated` flag that had to be kept in sync by hand.


def test_a_local_run_still_has_its_paths_checked(tmp_path):
    """The historical behaviour, and still the default: no cluster_ssh, full check."""
    with pytest.raises(TaskSchemaError, match="non-existent path"):
        load_and_validate_task_yaml(
            _write(
                tmp_path,
                {"checkpoint_path": "/nope/ckpt", "trtllm_repo_path": "/nope/repo"},
            )
        )


def test_a_remote_run_does_not_have_them_checked_here(tmp_path):
    """Same absent paths, plus `cluster_ssh` — they are on the other machine."""
    data = load_and_validate_task_yaml(
        _write(
            tmp_path,
            {
                "checkpoint_path": "/lustre/ckpt",
                "trtllm_repo_path": "/lustre/repo",
                "slurm-environment": {
                    "slurm_partition": "p",
                    "docker_image": "i",
                    "cluster_ssh": "me@login-01",
                },
            },
        )
    )
    assert data["checkpoint_path"] == "/lustre/ckpt"


def test_skipping_existence_does_not_skip_anything_else(tmp_path):
    """The narrowness is the point: only `exists()` is suppressed.

    A remote run with a malformed spec must still be refused now rather than
    tens of minutes into an agent turn.
    """
    with pytest.raises(TaskSchemaError) as exc:
        load_and_validate_task_yaml(
            _write(
                tmp_path,
                {
                    "checkpoint_path": 17,
                    "trtllm_repo_path": "/lustre/repo",
                    "benchmark": {"random_input_len": "wide"},
                    "slurm-environment": {
                        "slurm_partition": "p",
                        "docker_image": "i",
                        "cluster_ssh": "me@login-01",
                    },
                },
            )
        )
    message = str(exc.value)
    assert "checkpoint_path" in message, "a non-string path is still a type error"
    assert "random_input_len" in message, "unrelated checks must still run"


def test_a_missing_required_path_is_still_missing_when_remote(tmp_path):
    """Absent and unverifiable are different questions, and only one is skipped."""
    with pytest.raises(TaskSchemaError, match="missing required field 'checkpoint_path'"):
        load_and_validate_task_yaml(
            _write(
                tmp_path,
                {
                    "trtllm_repo_path": "/lustre/repo",
                    "slurm-environment": {
                        "slurm_partition": "p",
                        "docker_image": "i",
                        "cluster_ssh": "me@login-01",
                    },
                },
            )
        )


def test_extra_llm_api_options_follows_the_same_rule(tmp_path):
    """The third path, which is easy to forget because it is optional."""
    from agent_flow.workflows.perf_analyze.task_schema import paths_are_local

    remote = {
        "checkpoint_path": "/lustre/ckpt",
        "trtllm_repo_path": "/lustre/repo",
        "extra_llm_api_options": "/lustre/opts.yaml",
        "slurm-environment": {
            "slurm_partition": "p",
            "docker_image": "i",
            "cluster_ssh": "me@login-01",
        },
    }
    assert paths_are_local(remote) is False
    load_and_validate_task_yaml(_write(tmp_path, remote))  # must not raise

    ckpt, repo = _paths(tmp_path)
    local = {**remote, "checkpoint_path": ckpt, "trtllm_repo_path": repo}
    local.pop("slurm-environment")
    assert paths_are_local(local) is True
    with pytest.raises(TaskSchemaError, match="extra_llm_api_options"):
        load_and_validate_task_yaml(_write(tmp_path, local))


def test_a_malformed_slurm_block_does_not_decide_the_question(tmp_path):
    """`paths_are_local` runs before that block is validated, so it must be defensive."""
    from agent_flow.workflows.perf_analyze.task_schema import paths_are_local

    assert paths_are_local({"slurm-environment": "not-a-mapping"}) is True
    assert paths_are_local({"slurm-environment": {"cluster_ssh": "   "}}) is True
    assert paths_are_local({}) is True
