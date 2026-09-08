"""Tests for perf-optimize's task.yaml schema (base + optimize/accuracy)."""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.perf_analyze import task_schema as base_schema
from agent_flow.workflows.perf_optimize import task_schema


def _write_task(tmp_path, extra: dict | None = None):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    data = {"checkpoint_path": str(ckpt), "trtllm_repo_path": str(repo)}
    data.update(extra or {})
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_minimal_task_gets_all_defaults(tmp_path):
    data = task_schema.load_and_validate_task_yaml(_write_task(tmp_path))
    # perf-analyze base defaults still merge.
    assert data["benchmark"]["random_input_len"] == 1024
    assert data["profile"]["methods"] == ["nsys", "torch", "ncu"]
    # perf-optimize defaults merge.
    assert data["optimize"] == {
        "max_rounds": 5,
        "max_attempts_per_item": 3,
        "max_items_per_round": 3,
        "item_execution": "parallel",
        "approaches": ["config", "code"],
        "accept_fraction": 0.5,
        "noise_floor_pct": 1.0,
        "target_metric": "output_throughput",
    }
    # The resolved list is a copy — mutating it must not rewrite the
    # module-level default for later loads.
    assert data["optimize"]["approaches"] is not task_schema.OPTIMIZE_DEFAULTS["approaches"]
    # No accuracy block unless the user configures one.
    assert "accuracy" not in data
    assert task_schema.has_accuracy_check(data) is False
    # No early-stop target unless the user sets one.
    assert "target_improvement_pct" not in data["optimize"]


def test_user_optimize_values_win_over_defaults(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "optimize": {
                "max_rounds": 8,
                "accept_fraction": 0.8,
                "target_metric": "p99_tpot_ms",
                "target_improvement_pct": 15.0,
            }
        },
    )
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["optimize"]["max_rounds"] == 8
    assert data["optimize"]["accept_fraction"] == 0.8
    assert data["optimize"]["target_metric"] == "p99_tpot_ms"
    assert data["optimize"]["target_improvement_pct"] == 15.0
    # Untouched knobs keep their defaults.
    assert data["optimize"]["max_attempts_per_item"] == 3
    assert data["optimize"]["max_items_per_round"] == 3
    assert data["optimize"]["noise_floor_pct"] == 1.0


def test_user_approaches_value_wins_over_default(tmp_path):
    for value in (["code"], ["config"], ["config", "code"]):
        task = _write_task(tmp_path, {"optimize": {"approaches": value}})
        data = task_schema.load_and_validate_task_yaml(task)
        assert data["optimize"]["approaches"] == value


def test_item_execution_accepts_serial_and_parallel(tmp_path):
    for value in task_schema.ITEM_EXECUTIONS:
        task = _write_task(tmp_path, {"optimize": {"item_execution": value}})
        data = task_schema.load_and_validate_task_yaml(task)
        assert data["optimize"]["item_execution"] == value


def test_invalid_item_execution_rejected(tmp_path):
    task = _write_task(tmp_path, {"optimize": {"item_execution": "threads"}})
    with pytest.raises(task_schema.TaskSchemaError, match="optimize.item_execution"):
        task_schema.load_and_validate_task_yaml(task)


def test_invalid_approaches_rejected(tmp_path):
    for value in ([], ["yaml-only"], ["config", "config"], "config", ["config", 3]):
        task = _write_task(tmp_path, {"optimize": {"approaches": value}})
        with pytest.raises(task_schema.TaskSchemaError, match="optimize.approaches"):
            task_schema.load_and_validate_task_yaml(task)


def test_max_rounds_override_wins_over_user_value(tmp_path):
    task = _write_task(tmp_path, {"optimize": {"max_rounds": 5}})
    data = task_schema.load_and_validate_task_yaml(task, max_rounds_override=7)
    assert data["optimize"]["max_rounds"] == 7


def test_invalid_max_rounds_override_rejected(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="--max-rounds"):
        task_schema.load_and_validate_task_yaml(_write_task(tmp_path), max_rounds_override=0)


def test_base_validation_still_enforced(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump({"trtllm_repo_path": str(tmp_path)}), encoding="utf-8")
    with pytest.raises(task_schema.TaskSchemaError, match="checkpoint_path"):
        task_schema.load_and_validate_task_yaml(path)


def test_invalid_optimize_values_batched(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "optimize": {
                "max_rounds": 0,
                "max_attempts_per_item": True,  # bool must not pass as int
                "max_items_per_round": "many",
                "approaches": ["magic"],
                "accept_fraction": 1.5,
                "noise_floor_pct": -1,
                "target_metric": "",
                "target_improvement_pct": 0,
            }
        },
    )
    with pytest.raises(task_schema.TaskSchemaError) as exc_info:
        task_schema.load_and_validate_task_yaml(task)
    message = str(exc_info.value)
    for fragment in (
        "optimize.max_rounds",
        "optimize.max_attempts_per_item",
        "optimize.max_items_per_round",
        "optimize.approaches",
        "optimize.accept_fraction",
        "optimize.noise_floor_pct",
        "optimize.target_metric",
        "optimize.target_improvement_pct",
    ):
        assert fragment in message, fragment


def test_accept_fraction_boundaries(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="accept_fraction"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {"optimize": {"accept_fraction": 0}})
        )
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {"optimize": {"accept_fraction": 1}})
    )
    assert data["optimize"]["accept_fraction"] == 1


def test_optimize_must_be_mapping(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="'optimize' must be a mapping"):
        task_schema.load_and_validate_task_yaml(_write_task(tmp_path, {"optimize": "fast"}))


def test_accuracy_requires_command(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="accuracy.command"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {"accuracy": {"baseline_score": 0.6}})
        )


def test_accuracy_defaults_merge_only_when_configured(tmp_path):
    task = _write_task(
        tmp_path, {"accuracy": {"command": "trtllm-eval ...", "baseline_score": 0.62}}
    )
    data = task_schema.load_and_validate_task_yaml(task)
    assert task_schema.has_accuracy_check(data) is True
    assert data["accuracy"]["max_drop_pct"] == 1.0  # default merged
    assert data["accuracy"]["baseline_score"] == 0.62


def test_invalid_accuracy_values_rejected(tmp_path):
    task = _write_task(
        tmp_path,
        {"accuracy": {"command": "trtllm-eval", "baseline_score": "high", "max_drop_pct": -2}},
    )
    with pytest.raises(task_schema.TaskSchemaError) as exc_info:
        task_schema.load_and_validate_task_yaml(task)
    message = str(exc_info.value)
    assert "accuracy.baseline_score" in message
    assert "accuracy.max_drop_pct" in message


def test_normalized_dump_round_trips(tmp_path):
    task = _write_task(tmp_path, {"optimize": {"max_rounds": 2}})
    data = task_schema.load_and_validate_task_yaml(task)
    dumped = yaml.safe_load(task_schema.dump_task_yaml(data))
    assert dumped["optimize"]["max_rounds"] == 2
    assert dumped["optimize"]["target_metric"] == "output_throughput"
    assert dumped["benchmark"]["num_prompts"] == 200


def test_slurm_detection_reexported(tmp_path):
    task = _write_task(
        tmp_path,
        {"slurm-environment": {"slurm_partition": "batch", "docker_image": "/img.sqsh"}},
    )
    data = task_schema.load_and_validate_task_yaml(task)
    assert task_schema.has_slurm_environment(data) is True


def test_sol_block_validates_and_detection_reexported(tmp_path):
    """The base pass validates ``sol``; ``sol_enabled`` gates it."""
    task = _write_task(
        tmp_path,
        {"sol": {"gpu": "H100"}},
    )
    data = task_schema.load_and_validate_task_yaml(task)
    # The user's hint, with the on-by-default gate merged under it.
    assert data["sol"] == {"enabled": True, "gpu": "H100"}
    assert task_schema.sol_enabled(data) is True
    # A spec that never mentions ``sol`` still runs the projector here.
    assert task_schema.sol_enabled(
        task_schema.load_and_validate_task_yaml(_write_task(tmp_path))
    ) is (True)


def test_sol_enabled_false_reexported(tmp_path):
    """perf-optimize honors the same opt-out as perf-analyze."""
    task = _write_task(tmp_path, {"sol": {"enabled": False}})
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["sol"] == {"enabled": False}
    assert task_schema.sol_enabled(data) is False


def test_sol_unknown_field_rejected_reexported(tmp_path):
    """The typo guard is the base validator's, so it holds here too."""
    task = _write_task(tmp_path, {"sol": {"enable": False}})
    with pytest.raises(task_schema.TaskSchemaError, match="unknown field"):
        task_schema.load_and_validate_task_yaml(task)


def test_num_prompts_list_reexported_and_paired(tmp_path):
    task = _write_task(
        tmp_path,
        {"benchmark": {"concurrency": [128, 8], "num_prompts": [512, 32]}},
    )
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["benchmark"]["num_prompts"] == [32, 512]
    assert task_schema.num_prompts_per_point(data) == [32, 512]


def test_curve_helpers_reexported(tmp_path):
    task = _write_task(tmp_path, {"benchmark": {"concurrency": [128, 8]}})
    data = task_schema.load_and_validate_task_yaml(task)
    assert task_schema.is_curve_mode(data) is True
    assert task_schema.concurrency_points(data) == [8, 128]

    scalar = task_schema.load_and_validate_task_yaml(_write_task(tmp_path))
    assert task_schema.is_curve_mode(scalar) is False
    assert task_schema.concurrency_points(scalar) == [64]


def test_focus_concurrencies_valid_subset_normalized_sorted(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "benchmark": {"concurrency": [8, 32, 128]},
            "optimize": {"focus_concurrencies": [128, 32]},
        },
    )
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["optimize"]["focus_concurrencies"] == [32, 128]
    assert task_schema.focus_concurrencies(data) == [32, 128]


def test_focus_concurrencies_absent_means_all_points(tmp_path):
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {"benchmark": {"concurrency": [8, 32]}})
    )
    assert "focus_concurrencies" not in data["optimize"]
    assert task_schema.focus_concurrencies(data) is None
    # Scalar-mode spec likewise.
    assert task_schema.focus_concurrencies({}) is None


def test_focus_concurrencies_requires_curve_mode(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "benchmark": {"concurrency": 64},
            "optimize": {"focus_concurrencies": [64]},
        },
    )
    with pytest.raises(task_schema.TaskSchemaError, match="requires curve mode"):
        task_schema.load_and_validate_task_yaml(task)


def test_focus_concurrencies_must_be_subset_of_points(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "benchmark": {"concurrency": [8, 32, 128]},
            "optimize": {"focus_concurrencies": [32, 512]},
        },
    )
    with pytest.raises(task_schema.TaskSchemaError, match=r"\[512\] are not in"):
        task_schema.load_and_validate_task_yaml(task)


def test_max_regression_pct_valid_and_helper(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "benchmark": {"concurrency": [8, 512]},
            "optimize": {"max_regression_pct": 8.0},
        },
    )
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["optimize"]["max_regression_pct"] == 8.0
    assert task_schema.max_regression_pct(data) == 8.0
    # Absent => strict default (None).
    strict = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {"benchmark": {"concurrency": [8, 512]}})
    )
    assert task_schema.max_regression_pct(strict) is None


def test_max_regression_pct_requires_curve_mode(tmp_path):
    task = _write_task(
        tmp_path,
        {"benchmark": {"concurrency": 64}, "optimize": {"max_regression_pct": 5.0}},
    )
    with pytest.raises(task_schema.TaskSchemaError, match="requires curve mode"):
        task_schema.load_and_validate_task_yaml(task)


def test_max_regression_pct_must_cover_noise_floor(tmp_path):
    task = _write_task(
        tmp_path,
        {
            "benchmark": {"concurrency": [8, 512]},
            "optimize": {"max_regression_pct": 0.5, "noise_floor_pct": 1.0},
        },
    )
    with pytest.raises(task_schema.TaskSchemaError, match="must be >= "):
        task_schema.load_and_validate_task_yaml(task)
    for bad in ("high", -1):
        task = _write_task(
            tmp_path,
            {
                "benchmark": {"concurrency": [8, 512]},
                "optimize": {"max_regression_pct": bad},
            },
        )
        with pytest.raises(task_schema.TaskSchemaError, match="max_regression_pct"):
            task_schema.load_and_validate_task_yaml(task)


def test_focus_concurrencies_rejects_malformed_lists(tmp_path):
    for bad in ([], [32, 32], ["32"], [True], "32"):
        task = _write_task(
            tmp_path,
            {
                "benchmark": {"concurrency": [8, 32]},
                "optimize": {"focus_concurrencies": bad},
            },
        )
        with pytest.raises(task_schema.TaskSchemaError, match="non-empty list of unique integers"):
            task_schema.load_and_validate_task_yaml(task)


def test_max_concurrency_rename_surfaces_through_base_pass(tmp_path):
    task = _write_task(tmp_path, {"benchmark": {"max_concurrency": 64}})
    with pytest.raises(task_schema.TaskSchemaError, match="renamed to 'benchmark.concurrency'"):
        task_schema.load_and_validate_task_yaml(task)


# ------------------------------------------------------- profile.kernel_coverage


def test_kernel_coverage_absent_by_default(tmp_path):
    data = task_schema.load_and_validate_task_yaml(_write_task(tmp_path))
    assert "kernel_coverage" not in data["profile"]
    assert task_schema.kernel_coverage(data) is None


def test_empty_kernel_coverage_block_enables_defaults(tmp_path):
    task = _write_task(tmp_path, {"profile": {"kernel_coverage": {}}})
    data = task_schema.load_and_validate_task_yaml(task)
    # methods defaulted to all three, so the nsys+ncu requirement holds.
    assert data["profile"]["kernel_coverage"] == {
        "min_share_pct": 0.5,
        "coverage_target_pct": 95.0,
    }
    assert task_schema.kernel_coverage(data) == data["profile"]["kernel_coverage"]


def test_kernel_coverage_user_values_win_over_defaults(tmp_path):
    task = _write_task(
        tmp_path,
        {"profile": {"kernel_coverage": {"min_share_pct": 1.0}}},
    )
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["profile"]["kernel_coverage"]["min_share_pct"] == 1.0
    assert data["profile"]["kernel_coverage"]["coverage_target_pct"] == 95.0


def test_kernel_coverage_bars_must_be_in_range(tmp_path):
    for field in ("min_share_pct", "coverage_target_pct"):
        for bad in (0, -1, 101, "half", True):
            task = _write_task(tmp_path, {"profile": {"kernel_coverage": {field: bad}}})
            with pytest.raises(task_schema.TaskSchemaError, match=field):
                task_schema.load_and_validate_task_yaml(task)


def test_kernel_coverage_requires_nsys_and_ncu_methods(tmp_path):
    # The enumeration comes from nsys, the per-kernel metrics from ncu —
    # a profile that drops either cannot honor the contract.
    for methods in (["torch"], ["nsys", "torch"], ["ncu"]):
        task = _write_task(
            tmp_path,
            {"profile": {"methods": methods, "kernel_coverage": {}}},
        )
        with pytest.raises(task_schema.TaskSchemaError, match="profile.methods"):
            task_schema.load_and_validate_task_yaml(task)


def test_kernel_coverage_must_be_a_mapping(tmp_path):
    task = _write_task(tmp_path, {"profile": {"kernel_coverage": True}})
    with pytest.raises(task_schema.TaskSchemaError, match="kernel_coverage.*mapping"):
        task_schema.load_and_validate_task_yaml(task)


def test_kernel_coverage_accessor_defends_against_malformed_specs():
    assert task_schema.kernel_coverage({}) is None
    assert task_schema.kernel_coverage({"profile": "nope"}) is None
    assert task_schema.kernel_coverage({"profile": {"kernel_coverage": None}}) is None
    # A hand-edited resolved spec missing a default still resolves it.
    merged = task_schema.kernel_coverage({"profile": {"kernel_coverage": {"min_share_pct": 2.0}}})
    assert merged == {"min_share_pct": 2.0, "coverage_target_pct": 95.0}


@pytest.mark.parametrize("field", ["max_rounds", "max_attempts_per_item", "max_items_per_round"])
def test_a_valueless_budget_key_degrades_to_the_documented_default(tmp_path, field):
    """`max_items_per_round:` with nothing after it is YAML for ``None``.

    Every check in the validator treats an explicit null as "not set", so
    the defaults merge must too — otherwise the ``None`` wins over the
    default and reaches ``int()`` in the workflow as a TypeError.
    """
    task = _write_task(tmp_path, {"optimize": {field: None}})
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["optimize"][field] == task_schema.OPTIMIZE_DEFAULTS[field]
    assert isinstance(data["optimize"][field], int)


def test_the_metric_whitelist_is_published_without_being_enforced(tmp_path):
    """`VALID_METRICS` is the answer to a question, not a gate on this validator.

    Enforcing it here would narrow what `perf-optimize --task` has always
    accepted — every task.yaml with a metric outside the set stops running, and
    this schema belongs to the core, not to the one caller that wants the check.
    The service does want it, and does it: `spec_to_task` imports this set and
    refuses a typo at submission, which is where a submitted spec is checked
    anyway. Pinned in both directions so neither half drifts.
    """
    assert "output_throughput" in task_schema.VALID_METRICS
    assert "output_througput" not in task_schema.VALID_METRICS

    task = _write_task(tmp_path, {"optimize": {"target_metric": "output_througput"}})
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["optimize"]["target_metric"] == "output_througput"


@pytest.mark.parametrize("metric", ["output_throughput", "median_ttft_ms", "p99_e2el_ms"])
def test_every_documented_metric_still_validates(metric, tmp_path):
    task = _write_task(tmp_path, {"optimize": {"target_metric": metric}})
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["optimize"]["target_metric"] == metric


def test_the_key_census_covers_every_key_the_defaults_declare():
    """The census and the validator must describe the same schema.

    A key with a default that the census does not know would be reported as
    "not a key this workflow reads" on a spec that spells it correctly — the lint
    crying wolf on its own defaults, which is how a warning channel gets ignored.
    """
    assert set(task_schema.OPTIMIZE_DEFAULTS) <= task_schema.KNOWN_OPTIMIZE_KEYS
    assert set(task_schema.ACCURACY_DEFAULTS) <= task_schema.KNOWN_ACCURACY_KEYS
    assert set(task_schema.KERNEL_COVERAGE_DEFAULTS) <= task_schema.KNOWN_KERNEL_COVERAGE_KEYS


def _unknown_keys(mapping, prefix: str, known: set[str]) -> list[str]:
    """Keys in ``mapping`` that ``known`` does not account for, prefixed.

    The same census pass a task lint runs over a resolved spec, reduced to
    the part this schema owns. Kept here rather than imported so the check
    depends on nothing outside the vendored workflow packages.
    """
    if not isinstance(mapping, dict):
        return []
    return [f"{prefix}{key}" for key in mapping if key not in known]


def test_the_census_matches_a_fully_populated_spec(tmp_path):
    """Round-trip: a spec using every documented key must produce no unknowns.

    Written against the resolved spec rather than the input, because that is what
    is on disk for the agents — and what a lint would be run against.
    """
    task = _write_task(
        tmp_path,
        {
            "benchmark": {
                "dataset_name": "random",
                "random_input_len": 1024,
                "random_output_len": 128,
                "random_prefix_len": 0,
                "num_prompts": [8, 16],
                "concurrency": [2, 4],
                "request_rate": "inf",
            },
            "profile": {
                "methods": ["nsys", "ncu"],
                "nsys_iter_range": "100-150",
                "kernel_coverage": {"min_share_pct": 1.0, "coverage_target_pct": 90.0},
            },
            "optimize": {
                "max_rounds": 2,
                "max_items_per_round": 1,
                "max_attempts_per_item": 1,
                "item_execution": "serial",
                "approaches": ["config"],
                "accept_fraction": 0.5,
                "noise_floor_pct": 1.0,
                "target_metric": "output_throughput",
                "target_improvement_pct": 5.0,
                "focus_concurrencies": [4],
                "max_regression_pct": 2.0,
            },
            "accuracy": {"command": "true", "baseline_score": 0.5, "max_drop_pct": 1.0},
            "slurm-environment": {"slurm_partition": "p", "docker_image": "i", "cluster_ssh": "h"},
            "sol": {"gpu": "H100"},
        },
    )
    resolved = task_schema.load_and_validate_task_yaml(task)

    unknown = _unknown_keys(
        resolved,
        "",
        set(base_schema.KNOWN_TOP_LEVEL_KEYS)
        | set(base_schema.PASSTHROUGH_TOP_LEVEL_KEYS)
        | {"optimize", "accuracy"},
    )
    unknown += _unknown_keys(
        resolved.get("benchmark"), "benchmark.", set(base_schema.KNOWN_BENCHMARK_KEYS)
    )
    unknown += _unknown_keys(
        resolved.get("profile"),
        "profile.",
        set(base_schema.KNOWN_PROFILE_KEYS) | {"kernel_coverage"},
    )
    unknown += _unknown_keys(
        (resolved.get("profile") or {}).get("kernel_coverage"),
        "profile.kernel_coverage.",
        set(task_schema.KNOWN_KERNEL_COVERAGE_KEYS),
    )
    unknown += _unknown_keys(
        resolved.get(base_schema.SLURM_ENVIRONMENT_FIELD),
        f"{base_schema.SLURM_ENVIRONMENT_FIELD}.",
        set(base_schema.KNOWN_SLURM_KEYS),
    )
    unknown += _unknown_keys(
        resolved.get(base_schema.SOL_FIELD),
        f"{base_schema.SOL_FIELD}.",
        set(base_schema.KNOWN_SOL_KEYS),
    )
    unknown += _unknown_keys(
        resolved.get("optimize"), "optimize.", set(task_schema.KNOWN_OPTIMIZE_KEYS)
    )
    unknown += _unknown_keys(
        resolved.get("accuracy"), "accuracy.", set(task_schema.KNOWN_ACCURACY_KEYS)
    )

    assert unknown == []
