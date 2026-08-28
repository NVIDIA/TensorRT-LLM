"""Schema validation for perf-optimize's ``task.yaml`` input.

perf-optimize consumes the same base spec as perf-analyze (required
``checkpoint_path`` / ``trtllm_repo_path``, optional
``extra_llm_api_options`` / ``benchmark`` / ``profile`` /
``slurm-environment`` / ``sol`` blocks — the ``sol`` block gates the
one-shot SOL projector stage here too, on by default) plus two blocks of
its own:

- ``optimize`` — the loop knobs: round/item/attempt budgets, the allowed
  optimization ``approaches`` (any non-empty subset of the roadmap's
  ``config`` / ``code``), the evaluator's acceptance gate
  (``accept_fraction`` × the item's expected gain, with a
  ``noise_floor_pct`` floor), the target metric, an optional
  cumulative-improvement early-stop target the orchestrator enforces,
  and — curve mode only — optional ``focus_concurrencies``: the subset
  of ``benchmark.concurrency`` the gate scores (every curve→scalar
  derivation — gate mean, ledger ``value``s, ``measured_gain_pct`` —
  uses only these points, while measurement and the no-regress check
  still cover every configured point). Absent ⇒ all points (the
  historical behavior). Also curve mode only — optional
  ``max_regression_pct``: an owner-declared per-point regression budget
  for the no-regress condition (a point may regress up to this % when a
  large mean win justifies it; the report must surface any point
  accepted inside the budget). Absent ⇒ ``noise_floor_pct`` governs
  (the strict historical behavior); when set it must be ≥
  ``noise_floor_pct``.
- ``accuracy`` — optional; when present, the final-verification QA pass
  runs ``accuracy.command`` against the live server once, at campaign
  end, and compares the score against ``baseline_score`` /
  ``max_drop_pct``. When absent, QA does sanity completions only.

It also honors one perf-optimize-only key inside the shared ``profile``
block (the base validator preserves unknown ``profile`` keys):

- ``profile.kernel_coverage`` — optional; when present (an empty mapping
  is valid — all defaults) it activates the **per-kernel coverage
  contract**: the analyzer's ncu deep dive must cover every kernel
  at/above ``min_share_pct`` of GPU time (extending down the kern_sum
  until ``coverage_target_pct`` is reached), and every covered kernel
  gets both of its questions — faster? fusible? — answered in a
  schema-validated ``kernel_ledger.yaml`` each round (a roadmap item or
  an evidence-backed dismissal per question). Requires ``nsys`` (the
  enumeration source) and ``ncu`` (the per-kernel metrics) in
  ``profile.methods``.

Base validation is delegated to
:func:`agent_flow.workflows.perf_analyze.task_schema.load_and_validate_task_yaml`
(the settled cross-workflow reuse direction), then the new blocks are
validated and normalized in the same batched-error style. Note the two
passes mean base errors surface before ``optimize``/``accuracy`` errors
rather than all at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from agent_flow.workflows.perf_analyze.task_schema import (
    EXTRA_LLM_API_OPTIONS_FIELD,
    TaskSchemaError,
    cluster_ssh,
    concurrency_points,
    dump_task_yaml,
    has_slurm_environment,
    is_curve_mode,
    num_prompts_per_point,
    sol_enabled,
)
from agent_flow.workflows.perf_analyze.task_schema import (
    load_and_validate_task_yaml as _base_load_and_validate,
)
from agent_flow.workflows.perf_optimize.disagg import (
    DISAGG_CONFIG_KEY,
    DISAGG_FIELD,
    DisaggConfigError,
    apply_harness_conditions,
    disagg_config_path,
    has_disagg,
    load_disagg_config,
    user_set_benchmark_keys,
)
from agent_flow.workflows.perf_optimize.roadmap_schema import APPROACHES

# Defaults merged under the user's values. ``target_improvement_pct`` is
# deliberately absent: when the user does not set it, there is no
# early-stop target and the loop runs the full round budget (or until
# the roadmap is exhausted).
OPTIMIZE_DEFAULTS: dict[str, Any] = {
    # The loop runs exactly this many rounds unless the optional
    # improvement target is met or an analyzer turn finds no actionable
    # item. Each round is
    # one analyzer turn plus up to ``max_items_per_round`` gated items —
    # and only a round with stale/unproven runtime evidence pays to
    # re-profile. Accepts stale it; so can a reverted code attempt whose
    # gitignored build output survives.
    "max_rounds": 5,
    "max_attempts_per_item": 3,
    # Items applied (one at a time, each with its own evaluator gate)
    # before the round closes and the analyzer runs again. 1 reproduces
    # the original one-item-per-round loop; raising it amortizes the
    # analyzer profile across more items.
    "max_items_per_round": 3,
    # Which roadmap ``approach`` values the run may plan/apply. Restrict
    # to ["code"] to forbid tuning-YAML knob changes (code-only campaign)
    # or to ["config"] to leave the TRT-LLM checkout untouched.
    "approaches": list(APPROACHES),
    "accept_fraction": 0.5,
    "noise_floor_pct": 1.0,
    "target_metric": "output_throughput",
}

ACCURACY_DEFAULTS: dict[str, Any] = {
    "max_drop_pct": 1.0,
}

# Metric keys a gain can be computed from: they must exist in the benchmark
# result JSON.
#
# Published, not enforced. The validator deliberately does NOT reject a
# `target_metric` outside this set, because doing so would narrow what the CLI
# has always accepted — and this schema is the core's, shared by every caller,
# not just by the service that happens to want the check. `service/adapter/
# spec_to_task.py` imports it and refuses a typo at submission time, which is
# where a submitted spec is validated anyway.
#
# So this is the single source of truth for the QUESTION ("is that a real result
# key?") without being the place that answers it for everyone.
VALID_METRICS: frozenset[str] = frozenset(
    {"output_throughput", "total_token_throughput", "request_throughput"}
    | {
        f"{stat}_{kind}_ms"
        for stat in ("mean", "median", "p90", "p99")
        for kind in ("ttft", "tpot", "itl", "e2el")
    }
)

# The perf-optimize half of the key census the base schema documents. Same
# contract: this is what a lint may call real, not what the validator rejects.
KNOWN_OPTIMIZE_KEYS: frozenset[str] = frozenset(
    set(OPTIMIZE_DEFAULTS) | {"target_improvement_pct", "focus_concurrencies", "max_regression_pct"}
)
KNOWN_ACCURACY_KEYS: frozenset[str] = frozenset({"command", "baseline_score", "max_drop_pct"})
KNOWN_KERNEL_COVERAGE_KEYS: frozenset[str] = frozenset({"min_share_pct", "coverage_target_pct"})

# Defaults merged under ``profile.kernel_coverage`` when the block is
# present (its presence is the opt-in; absent ⇒ the bounded top-kernel
# ncu dive, the historical behavior).
KERNEL_COVERAGE_DEFAULTS: dict[str, Any] = {
    # Every kernel at/above this share of profiled GPU time gets its own
    # ledger row (and ncu coverage).
    "min_share_pct": 0.5,
    # The enumerated rows must cover at least this much of GPU time —
    # when the >= min_share_pct rows fall short, enumeration extends
    # down the kern_sum (grouping related kernels is fine) until met.
    "coverage_target_pct": 95.0,
}


def _is_number(value: Any) -> bool:
    # bool is an int subclass — reject it explicitly.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mapping_block(data: Mapping[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    """Return ``data[key]`` as a dict, recording an error if it is the wrong type."""
    if key not in data or data[key] is None:
        return {}
    value = data[key]
    if not isinstance(value, dict):
        errors.append(f"'{key}' must be a mapping, got {type(value).__name__}")
        return {}
    return dict(value)


def _validate_optimize_block(optimize: Mapping[str, Any], errors: list[str]) -> None:
    for field in ("max_rounds", "max_attempts_per_item", "max_items_per_round"):
        if field in optimize and optimize[field] is not None:
            value = optimize[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                errors.append(f"'optimize.{field}' must be an integer >= 1, got {value!r}")

    if "accept_fraction" in optimize and optimize["accept_fraction"] is not None:
        value = optimize["accept_fraction"]
        if not _is_number(value) or not (0 < value <= 1):
            errors.append(f"'optimize.accept_fraction' must be a number in (0, 1], got {value!r}")

    if "noise_floor_pct" in optimize and optimize["noise_floor_pct"] is not None:
        value = optimize["noise_floor_pct"]
        if not _is_number(value) or value < 0:
            errors.append(f"'optimize.noise_floor_pct' must be a number >= 0, got {value!r}")

    if "approaches" in optimize and optimize["approaches"] is not None:
        value = optimize["approaches"]
        valid = (
            isinstance(value, list)
            and bool(value)
            and all(entry in APPROACHES for entry in value)
            and len(set(value)) == len(value)
        )
        if not valid:
            errors.append(
                f"'optimize.approaches' must be a non-empty list of unique values "
                f"from {list(APPROACHES)}, got {value!r}"
            )

    if "target_metric" in optimize and optimize["target_metric"] is not None:
        value = optimize["target_metric"]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'optimize.target_metric' must be a non-empty string, got {value!r}")

    if "target_improvement_pct" in optimize and optimize["target_improvement_pct"] is not None:
        value = optimize["target_improvement_pct"]
        if not _is_number(value) or value <= 0:
            errors.append(f"'optimize.target_improvement_pct' must be a number > 0, got {value!r}")


def _validate_max_regression_pct(
    data: Mapping[str, Any], optimize: Mapping[str, Any], errors: list[str]
) -> None:
    """Validate ``optimize.max_regression_pct`` against mode and floor.

    Curve-mode only (the no-regress condition it relaxes exists only
    there), and never below ``noise_floor_pct`` — a tolerance under the
    noise floor would be stricter than noise itself can measure.
    """
    if "max_regression_pct" not in optimize or optimize["max_regression_pct"] is None:
        return
    value = optimize["max_regression_pct"]
    if not _is_number(value) or value < 0:
        errors.append(f"'optimize.max_regression_pct' must be a number >= 0, got {value!r}")
        return
    if not is_curve_mode(data):
        errors.append(
            "'optimize.max_regression_pct' requires curve mode "
            "('benchmark.concurrency' must be a list)"
        )
        return
    noise_floor = optimize.get("noise_floor_pct", OPTIMIZE_DEFAULTS["noise_floor_pct"])
    if _is_number(noise_floor) and value < noise_floor:
        errors.append(
            f"'optimize.max_regression_pct' ({value!r}) must be >= "
            f"'optimize.noise_floor_pct' ({noise_floor!r}) — the noise floor "
            f"is already the minimum tolerance"
        )


def _validate_focus_concurrencies(
    data: Mapping[str, Any], optimize: Mapping[str, Any], errors: list[str]
) -> list[int] | None:
    """Validate ``optimize.focus_concurrencies`` against the benchmark block.

    Returns the normalized (sorted ascending) list when valid, else
    ``None`` with the problems appended to ``errors``. Validated here —
    not in :func:`_validate_optimize_block` — because it needs the
    base-validated ``benchmark.concurrency`` points.
    """
    if "focus_concurrencies" not in optimize or optimize["focus_concurrencies"] is None:
        return None
    value = optimize["focus_concurrencies"]
    well_typed = (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        and len(set(value)) == len(value)
    )
    if not well_typed:
        errors.append(
            f"'optimize.focus_concurrencies' must be a non-empty list of "
            f"unique integers, got {value!r}"
        )
        return None
    if not is_curve_mode(data):
        errors.append(
            "'optimize.focus_concurrencies' requires curve mode "
            "('benchmark.concurrency' must be a list)"
        )
        return None
    points = set(concurrency_points(data))
    stray = sorted(v for v in value if v not in points)
    if stray:
        errors.append(
            f"'optimize.focus_concurrencies' entries {stray} are not in "
            f"'benchmark.concurrency' {sorted(points)}"
        )
        return None
    return sorted(value)


def _validate_kernel_coverage(data: Mapping[str, Any], errors: list[str]) -> dict[str, Any] | None:
    """Validate the optional ``profile.kernel_coverage`` block.

    Returns the normalized block (defaults merged) when present and
    valid, else ``None`` with the problems appended to ``errors``. Lives
    in this schema — not the base one — because the contract it enables
    (the per-round ``kernel_ledger.yaml`` gate) exists only in
    perf-optimize; the base validator preserves the key untouched.
    """
    profile = data.get("profile")
    if not isinstance(profile, Mapping) or profile.get("kernel_coverage") is None:
        return None
    block = profile["kernel_coverage"]
    if not isinstance(block, dict):
        errors.append(
            f"'profile.kernel_coverage' must be a mapping (an empty one enables "
            f"the defaults), got {type(block).__name__}"
        )
        return None
    merged = {**KERNEL_COVERAGE_DEFAULTS, **block}
    for field in ("min_share_pct", "coverage_target_pct"):
        value = merged[field]
        if not _is_number(value) or not (0 < value <= 100):
            errors.append(
                f"'profile.kernel_coverage.{field}' must be a number in (0, 100], got {value!r}"
            )
    methods = profile.get("methods", [])
    missing = [m for m in ("nsys", "ncu") if not (isinstance(methods, list) and m in methods)]
    if missing:
        errors.append(
            f"'profile.kernel_coverage' requires {missing} in 'profile.methods' — "
            f"the kernel enumeration comes from the nsys kern_sum and the "
            f"per-kernel metrics from ncu"
        )
        return None
    return merged


def _validate_accuracy_block(
    data: Mapping[str, Any], accuracy: Mapping[str, Any], errors: list[str]
) -> None:
    if "accuracy" not in data or data["accuracy"] is None:
        return
    command = accuracy.get("command")
    if not isinstance(command, str) or not command.strip():
        errors.append(f"'accuracy.command' must be a non-empty string, got {command!r}")
    if "baseline_score" in accuracy and accuracy["baseline_score"] is not None:
        if not _is_number(accuracy["baseline_score"]):
            errors.append(
                f"'accuracy.baseline_score' must be a number, got {accuracy['baseline_score']!r}"
            )
    if "max_drop_pct" in accuracy and accuracy["max_drop_pct"] is not None:
        value = accuracy["max_drop_pct"]
        if not _is_number(value) or value < 0:
            errors.append(f"'accuracy.max_drop_pct' must be a number >= 0, got {value!r}")


def _validate_disagg_block(data: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    """Validate the ``disagg`` block and load the harness config it names.

    Returns the parsed harness config so the caller can backfill the
    measurement conditions from it, or ``None`` when the block is absent
    or invalid. Loading happens here, at the CLI boundary, because a
    harness config that cannot be read or is missing the blocks the
    mapping needs must abort the run before any agent is constructed —
    the alternative is discovering it an hour into a GPU allocation.
    """
    if not has_disagg(data):
        return None
    block = data.get(DISAGG_FIELD)
    if not isinstance(block, Mapping):
        errors.append(
            f"'{DISAGG_FIELD}' must be a mapping carrying "
            f"'{DISAGG_CONFIG_KEY}: <path to the disagg harness config.yaml>', "
            f"got {type(block).__name__}"
        )
        return None
    path = disagg_config_path(data)
    if path is None:
        errors.append(
            f"'{DISAGG_FIELD}.{DISAGG_CONFIG_KEY}' is required and must be a non-empty "
            f"string: the path to the config.yaml consumed by the TensorRT-LLM "
            f"checkout's examples/disaggregated/slurm/benchmark/submit.py"
        )
        return None
    if not path.is_file():
        errors.append(f"'{DISAGG_FIELD}.{DISAGG_CONFIG_KEY}' is not a file: {path}")
        return None
    if data.get(EXTRA_LLM_API_OPTIONS_FIELD) is not None:
        # Two seeds for one live tuning file: in disagg mode it is seeded
        # from the harness config's worker_config block, so an
        # extra_llm_api_options path would silently lose.
        errors.append(
            f"'{EXTRA_LLM_API_OPTIONS_FIELD}' cannot be combined with '{DISAGG_FIELD}': "
            f"the live tuning config is seeded from the disagg config's "
            f"'worker_config' block (ctx / gen roles)"
        )
        return None
    try:
        return load_disagg_config(path)
    except DisaggConfigError as exc:
        errors.append(str(exc))
        return None


def load_and_validate_task_yaml(
    path: str | Path, *, max_rounds_override: int | None = None
) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the perf-optimize schema.

    Runs the perf-analyze base validation first, then validates the
    ``optimize`` / ``accuracy`` blocks, batching every problem found in
    this pass into a single :class:`TaskSchemaError`. Returns the mapping
    with defaults merged under the user's values so the resolved spec the
    agents read on disk is fully explicit; ``max_rounds_override`` (the
    CLI ``--max-rounds`` flag) is applied last, over the user's value.
    """
    data = _base_load_and_validate(path)

    errors: list[str] = []
    # Disagg first: the harness config is the source of truth for the
    # measurement conditions, so the backfill has to land before the
    # blocks that are validated against them (focus_concurrencies against
    # the concurrency points, accuracy against its own presence).
    disagg_cfg = _validate_disagg_block(data, errors)
    if disagg_cfg is not None:
        try:
            data[DISAGG_FIELD] = {
                **data[DISAGG_FIELD],
                "filled_from_disagg_config": apply_harness_conditions(
                    data, disagg_cfg, user_set_benchmark_keys(path)
                ),
            }
        except DisaggConfigError as exc:
            errors.append(str(exc))
    optimize = _mapping_block(data, "optimize", errors)
    _validate_optimize_block(optimize, errors)
    # An explicitly-null value means "not set" everywhere in this
    # validator (every check above skips ``None``), so drop those keys
    # before the defaults merge too — otherwise a bare ``max_rounds:``
    # line would win over the default and reach the workflow as ``None``.
    optimize = {key: value for key, value in optimize.items() if value is not None}
    _validate_max_regression_pct(data, optimize, errors)
    normalized_focus = _validate_focus_concurrencies(data, optimize, errors)
    normalized_kernel_coverage = _validate_kernel_coverage(data, errors)
    accuracy = _mapping_block(data, "accuracy", errors)
    _validate_accuracy_block(data, accuracy, errors)

    if max_rounds_override is not None and max_rounds_override < 1:
        errors.append(f"--max-rounds must be >= 1, got {max_rounds_override}")

    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{Path(path)} failed perf-optimize schema validation:{bullet}{bullet.join(errors)}"
        )

    data["optimize"] = {**OPTIMIZE_DEFAULTS, **optimize}
    # Never hand out the module-level default list itself — a caller
    # mutating the resolved spec must not rewrite the defaults.
    data["optimize"]["approaches"] = list(data["optimize"]["approaches"])
    if normalized_focus is not None:
        data["optimize"]["focus_concurrencies"] = normalized_focus
    if normalized_kernel_coverage is not None:
        data["profile"]["kernel_coverage"] = normalized_kernel_coverage
    if max_rounds_override is not None:
        data["optimize"]["max_rounds"] = max_rounds_override
    if has_accuracy_check(data):
        data["accuracy"] = {**ACCURACY_DEFAULTS, **accuracy}

    return data


def has_accuracy_check(data: Mapping[str, Any]) -> bool:
    """Return whether a validated task spec configured an accuracy eval."""
    return "accuracy" in data and data["accuracy"] is not None


def max_regression_pct(data: Mapping[str, Any]) -> float | None:
    """The declared per-point regression budget, or ``None`` (strict).

    ``None`` means the no-regress condition uses ``noise_floor_pct``
    (the historical behavior).
    """
    optimize = data.get("optimize")
    if not isinstance(optimize, Mapping):
        return None
    value = optimize.get("max_regression_pct")
    if not _is_number(value):
        return None
    return float(value)


def kernel_coverage(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """The validated per-kernel coverage contract, or ``None`` (off).

    ``None`` means the historical bounded top-kernel ncu dive; a mapping
    (always carrying ``min_share_pct`` / ``coverage_target_pct`` after
    validation) activates the ledger contract.
    """
    profile = data.get("profile")
    if not isinstance(profile, Mapping):
        return None
    value = profile.get("kernel_coverage")
    if not isinstance(value, Mapping):
        return None
    merged = {**KERNEL_COVERAGE_DEFAULTS, **value}
    return merged


def focus_concurrencies(data: Mapping[str, Any]) -> list[int] | None:
    """The validated spec's gate-scored concurrency subset, or ``None``.

    ``None`` means the gate scores every configured point (the default);
    a list is always sorted ascending and a subset of
    :func:`concurrency_points`.
    """
    optimize = data.get("optimize")
    if not isinstance(optimize, Mapping):
        return None
    value = optimize.get("focus_concurrencies")
    if not isinstance(value, list) or not value:
        return None
    return list(value)


__all__ = [
    "ACCURACY_DEFAULTS",
    "KERNEL_COVERAGE_DEFAULTS",
    "KNOWN_ACCURACY_KEYS",
    "KNOWN_KERNEL_COVERAGE_KEYS",
    "KNOWN_OPTIMIZE_KEYS",
    "OPTIMIZE_DEFAULTS",
    "VALID_METRICS",
    "TaskSchemaError",
    "concurrency_points",
    "dump_task_yaml",
    "focus_concurrencies",
    "has_accuracy_check",
    "cluster_ssh",
    "has_slurm_environment",
    "is_curve_mode",
    "kernel_coverage",
    "load_and_validate_task_yaml",
    "max_regression_pct",
    "num_prompts_per_point",
    "sol_enabled",
]
