"""Schema validation for perf-analyze's ``task.yaml`` input.

The perf-analyze workflow requires two path fields up front so the
agents always know where to find the model checkpoint to serve and the
TensorRT-LLM checkout that provides ``trtllm-serve`` and
``benchmark_serving.py``. An optional top-level ``extra_llm_api_options``
path points at a YAML passed verbatim to
``trtllm-serve --extra_llm_api_options`` — the single place all server
tuning (parallelism, batch sizes, KV-cache fraction, ...) lives; the
server otherwise always runs the ``pytorch`` backend on ``127.0.0.1:8000``.
Two optional mapping blocks tune the benchmark load (``benchmark``) and
the profiling run (``profile``); ``benchmark.concurrency`` is a single
positive int (one operating point) or a non-empty list of them
(Pareto-curve mode: one benchmark run per point, normalized to sorted
unique values); ``benchmark.num_prompts`` is a single positive int (used
at every point) or — curve mode only — a list paired index-by-index with
the ``concurrency`` list (normalized alongside it, so low-concurrency
points can run far fewer prompts than high-concurrency ones); an
optional ``slurm-environment`` block
routes execution through Slurm instead of the local node; the ``sol``
block (every field optional: ``enabled`` gates the stage and ``gpu`` is
the part-name hint for the skill's peaks calculator) controls the
projector stage, which derives an analytical speed-of-light (SOL)
ceiling — per the ``internal-perf-sol-analysis`` skill — between the
benchmarker and the analyzer. That stage is **on by default**: the block
only has to be written to turn it off (``sol: {enabled: false}``) or to
hand the skill a ``gpu`` hint. Inside that block a misspelled key is
rejected rather than preserved — ``enabled`` decides whether the stage
runs, so a silently-ignored ``enable:`` would invert the user's intent.
Anything else the user puts in the YAML is preserved on disk for the
agents to read.

Validation runs at the CLI boundary so an invalid spec aborts before any
agent is constructed. :func:`load_and_validate_task_yaml` also fills in
the workflow's defaults for the optional blocks so the resolved spec the
agents read on disk is fully explicit.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_PATH_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "trtllm_repo_path",
)

# Optional top-level path field: a YAML passed verbatim to
# ``trtllm-serve --extra_llm_api_options``. When present, it must point at
# an existing file.
EXTRA_LLM_API_OPTIONS_FIELD = "extra_llm_api_options"

# Fixed server settings — the workflow always serves with these, so they
# are not user-tunable knobs. Every other server knob lives in the
# ``extra_llm_api_options`` YAML.
SERVE_BACKEND = "pytorch"
SERVE_HOST = "127.0.0.1"
SERVE_PORT = 8000

SLURM_ENVIRONMENT_FIELD = "slurm-environment"
SLURM_REQUIRED_FIELDS: tuple[str, ...] = (
    "slurm_partition",
    "docker_image",
)

# Optional ssh alias for the cluster's login node, e.g. ``user@login-01``.
#
# Its presence means THIS PROCESS IS NOT ON THE CLUSTER: the workflow runs
# somewhere with no slurm client and no shared filesystem, so everything
# that touches the cluster — git on the checkout, existence checks on
# ``checkpoint_path``/``trtllm_repo_path``, and the agents' own ``srun`` —
# has to travel over ssh. The repo, the workspace and every artifact live
# on the cluster; only ssh commands cross the boundary.
#
# Optional, and absent is the historical behaviour: "these paths are local
# and ``srun`` works here". Every existing ``task.yaml`` is written that
# way and so are the workflow's own tests, so requiring it would invalidate
# all of them to describe a deployment most runs do not use.
SLURM_CLUSTER_SSH_FIELD = "cluster_ssh"

# SOL-projection block. The workflow runs the projector stage
# (benchmarker -> projector -> analyzer -> reporter), which follows the
# ``internal-perf-sol-analysis`` skill, **by default** — so the block is
# written only to opt out or to hand the skill a hint. Every field is
# optional: ``enabled`` is the stage gate (default ``True``), and ``gpu``
# is the part-name hint for the skill's peaks calculator, when the
# projector's automatic mapping would guess wrong. ``SOL_DEFAULTS`` is
# merged under the user's values like the other blocks, so the resolved
# spec on disk always states the gate; absent keys still mean "the
# projector infers the mapping".
SOL_FIELD = "sol"
SOL_ENABLED_FIELD = "enabled"
SOL_OPTIONAL_STR_FIELDS: tuple[str, ...] = ("gpu",)
SOL_FIELDS: tuple[str, ...] = (SOL_ENABLED_FIELD, *SOL_OPTIONAL_STR_FIELDS)
SOL_DEFAULTS: dict[str, Any] = {SOL_ENABLED_FIELD: True}

# The pre-rename spelling of the ``sol`` block (when the projector still
# cross-checked a dlsim checkout) — rejected with an actionable error so
# a stale task.yaml fails loudly instead of having its projector settings
# silently ignored.
_RENAMED_SOL_FIELD = "dlsim"

VALID_PROFILE_METHODS: tuple[str, ...] = ("nsys", "torch", "ncu")

# Defaults merged under the user's values for the always-present knobs.
# Keys deliberately absent here (e.g. ``benchmark.request_rate``) stay
# omitted when the user does not set them, which the prompts read as "omit
# the corresponding CLI flag and let benchmark_serving.py use its own
# default". Server tuning has no defaults block: the server always runs
# the ``pytorch`` backend on ``127.0.0.1:8000`` and every other knob lives
# in the optional ``extra_llm_api_options`` YAML.
BENCHMARK_DEFAULTS: dict[str, Any] = {
    "dataset_name": "random",
    "random_input_len": 1024,
    "random_output_len": 128,
    "num_prompts": 200,
    "concurrency": 64,
}
PROFILE_DEFAULTS: dict[str, Any] = {
    "methods": list(VALID_PROFILE_METHODS),
    "nsys_iter_range": "100-150",
}

# Type expectations for known keys inside the optional ``benchmark``
# block. Unknown keys are preserved untouched so the schema does not have
# to track every benchmark_serving.py flag. ``concurrency`` and
# ``num_prompts`` are validated separately: they are the two fields that
# accept an int *or* a list of ints (Pareto-curve mode).
_BENCHMARK_INT_FIELDS = (
    "random_input_len",
    "random_output_len",
    "random_prefix_len",
)

# The pre-rename spelling of ``benchmark.concurrency`` — rejected with an
# actionable error so stale task.yaml files fail loudly instead of
# silently benchmarking at the default.
_RENAMED_CONCURRENCY_FIELD = "max_concurrency"
_BENCHMARK_STR_FIELDS = ("dataset_name", "dataset_path")


# ── the key census: what something OTHER than this validator may call real ──
#
# The validation below inspects only the keys it knows and preserves everything
# else untouched (see the module docstring). That leniency is load-bearing, not
# an oversight: `spec_to_task` injects a top-level `directions` key no schema
# knows, and real specs carry free-text `notes` the agents read. Turning unknown
# keys into errors would break both.
#
# But leniency has a cost the schema cannot pay off on its own: a typo validates
# silently, is written back into the workspace by `dump_task_yaml`, and is then
# read by nobody. The user sees their key in the resolved spec and concludes it
# took effect.
#
# These sets let a SEPARATE lint answer "is that key real?" without the schema
# having to reject anything. They are the single source of truth for that
# question — a second copy anywhere else is how it goes stale, and this schema
# has already been renamed twice (`max_concurrency`, `dlsim`).
KNOWN_BENCHMARK_KEYS: frozenset[str] = frozenset(
    _BENCHMARK_INT_FIELDS + _BENCHMARK_STR_FIELDS + ("concurrency", "num_prompts", "request_rate")
)
KNOWN_PROFILE_KEYS: frozenset[str] = frozenset({"methods", "nsys_iter_range"})
KNOWN_SLURM_KEYS: frozenset[str] = frozenset(SLURM_REQUIRED_FIELDS + (SLURM_CLUSTER_SSH_FIELD,))
KNOWN_SOL_KEYS: frozenset[str] = frozenset(SOL_FIELDS)
KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    REQUIRED_PATH_FIELDS
    + (
        EXTRA_LLM_API_OPTIONS_FIELD,
        "benchmark",
        "profile",
        SLURM_ENVIRONMENT_FIELD,
        SOL_FIELD,
    )
)

# Keys nothing in the schema consumes, but which are deliberate channels to the
# agents rather than mistakes. A lint must stay quiet about these or it fires on
# every real spec: `directions` is written by the service adapter, and the
# recorded campaign's `notes` carry the campaign intent no other field can hold.
#
# `agent_flow_commit` is read by the SERVICE, before this schema is ever loaded:
# it names the agent-flow commit whose code runs the campaign, so by the time the
# flow validates a task.yaml the key has already done its work. It stays in the
# file rather than being stripped, because the run's own record of which flow
# version produced it is worth more than a tidy mapping.
# `experiment` is the same shape as `agent_flow_commit`: read by the SERVICE
# before this schema loads, never by the flow. It carries opt-in switches for
# behaviour that is still being proven, so that trying one is a line in the file
# the submitter can see rather than deployment state they cannot.
#
# The flow's behaviour does not change when it is present or absent. It stays in
# the file for the same reason `agent_flow_commit` does: the run's own record of
# what was enabled for it is worth more than a tidy mapping.
PASSTHROUGH_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"notes", "directions", "agent_flow_commit", "experiment"}
)

# What may appear inside `experiment:`. Declared, and checked, because the reason
# to have a named block at all is that a bag nothing validates reproduces the
# failure this codebase already has: an unknown key is PRESERVED and ignored, so
# `buildcache: true` would read as enabled-and-working while doing nothing, and
# a key name one letter off from a real one is silent for exactly the same
# reason. The lint exists to catch both.
#
#   build_cache        -- reuse a previous run's compiled artifacts when some run
#                         already built this exact commit for this architecture,
#                         instead of spending ~45 minutes compiling it again.
#   agent_flow_commit  -- pin the agent-flow commit that DRIVES the campaign.
#                         Also accepted at the top level, where it used to live
#                         and still works: dropping that spelling would leave an
#                         old file's pin sitting in plain sight while the run
#                         quietly used the default, and a pin that stops pinning
#                         is worse than no pin.
KNOWN_EXPERIMENT_KEYS: frozenset[str] = frozenset({"build_cache", "agent_flow_commit"})

# Flags the workflow's canonical `benchmark_serving.py` command HARDCODES, which
# therefore have no `task.yaml` representation at all.
#
# They are singled out because they are the mistake most likely to be made, not
# the least: a user (or an agent) reading an existing benchmark script sees these
# spelled out literally in it, so transcribing them into the `benchmark` block is
# the obvious move. The result validates, is preserved, and is ignored — and the
# user has every reason to believe the flag carried over.
#
# Matched with `-` folded to `_`, so both spellings are caught.
BENCHMARK_FIXED_FLAGS: dict[str, str] = {
    "trust_remote_code": "always passed",
    "random_ids": "always passed for the `random` dataset",
    "tokenize_on_client": "always passed; it is what pins the effective ISL",
    "ignore_eos": "always passed, so every request emits exactly random_output_len",
    "no_test_input": "always passed; the warmup prompt would shift the profiling window",
    "percentile_metrics": "fixed at ttft,tpot,itl,e2el",
    "metric_percentiles": "fixed at 90,99",
    "save_result": "always passed",
    "save_detailed": "always passed",
}


class TaskSchemaError(ValueError):
    """Raised when ``task.yaml`` fails perf-analyze schema validation."""


def _validate_int_fields(
    block: Mapping[str, Any], block_name: str, fields: tuple[str, ...], errors: list[str]
) -> None:
    for field in fields:
        if field not in block or block[field] is None:
            continue
        value = block[field]
        # bool is an int subclass — reject it explicitly so ``port: true``
        # does not slip through.
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"'{block_name}.{field}' must be an integer, got {type(value).__name__}")


def _is_positive_int(value: Any) -> bool:
    # bool is an int subclass — reject it explicitly.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _validate_concurrency(benchmark: Mapping[str, Any], errors: list[str]) -> None:
    """Validate ``benchmark.concurrency``: a positive int or a list of them.

    A scalar keeps the workflow at a single operating point; a list turns
    on Pareto-curve mode (one benchmark run per point). The stale
    pre-rename spelling is rejected with a pointer to the new field so an
    old task.yaml never silently falls back to the default.
    """
    if _RENAMED_CONCURRENCY_FIELD in benchmark:
        errors.append(
            f"'benchmark.{_RENAMED_CONCURRENCY_FIELD}' was renamed to "
            "'benchmark.concurrency' — use `concurrency: <int>` for a single "
            "operating point or `concurrency: [c1, c2, ...]` to benchmark a "
            "Pareto curve"
        )
    value = benchmark.get("concurrency")
    if value is None:
        return
    if _is_positive_int(value):
        return
    if isinstance(value, list) and value and all(_is_positive_int(v) for v in value):
        return
    errors.append(
        "'benchmark.concurrency' must be a positive integer or a non-empty "
        f"list of positive integers, got {value!r}"
    )


def _validate_num_prompts(benchmark: Mapping[str, Any], errors: list[str]) -> None:
    """Validate ``benchmark.num_prompts``: a positive int or a list of them.

    A scalar applies to every operating point. A list is curve-mode only
    and pairs index-by-index with the ``concurrency`` list as written —
    same length, unique concurrency entries (deduplication would break
    the pairing), and every entry at least its point's concurrency (the
    run cannot reach the configured parallelism otherwise).
    """
    value = benchmark.get("num_prompts")
    if value is None or _is_positive_int(value):
        return
    if not (isinstance(value, list) and value and all(_is_positive_int(v) for v in value)):
        errors.append(
            "'benchmark.num_prompts' must be a positive integer or a non-empty "
            f"list of positive integers, got {value!r}"
        )
        return
    concurrency = benchmark.get("concurrency")
    if not (
        isinstance(concurrency, list)
        and concurrency
        and all(_is_positive_int(c) for c in concurrency)
    ):
        errors.append(
            "'benchmark.num_prompts' can only be a list in Pareto-curve mode "
            "(when 'benchmark.concurrency' is a list of positive integers)"
        )
        return
    if len(value) != len(concurrency):
        errors.append(
            "'benchmark.num_prompts' list must pair one-to-one with the "
            f"'benchmark.concurrency' list ({len(concurrency)} points), "
            f"got {len(value)} entries"
        )
        return
    if len(set(concurrency)) != len(concurrency):
        errors.append(
            "'benchmark.concurrency' must not contain duplicates when "
            "'benchmark.num_prompts' is a list (deduplication would break "
            "the index pairing)"
        )
        return
    short = [f"num_prompts {n} < concurrency {c}" for c, n in zip(concurrency, value) if n < c]
    if short:
        errors.append(
            "each 'benchmark.num_prompts' entry must be >= its paired "
            f"'benchmark.concurrency' point: {'; '.join(short)}"
        )


def _validate_str_fields(
    block: Mapping[str, Any], block_name: str, fields: tuple[str, ...], errors: list[str]
) -> None:
    for field in fields:
        if field not in block or block[field] is None:
            continue
        value = block[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"'{block_name}.{field}' must be a non-empty string, got {type(value).__name__}"
            )


def _validate_mapping_block(data: Mapping[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    """Return ``data[key]`` as a dict, recording an error if it is the wrong type."""
    if key not in data or data[key] is None:
        return {}
    value = data[key]
    if not isinstance(value, dict):
        errors.append(f"'{key}' must be a mapping, got {type(value).__name__}")
        return {}
    return dict(value)


def load_and_validate_task_yaml(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the perf-analyze schema.

    Returns the parsed mapping with the optional ``benchmark`` / ``profile``
    blocks normalized (workflow defaults merged under the user's values).
    Raises :class:`TaskSchemaError` with **every** detected problem batched
    into a single message so the user sees all gaps at once instead of
    fixing them one-by-one.

    The three ``Path(...).exists()`` checks — ``checkpoint_path``,
    ``trtllm_repo_path``, ``extra_llm_api_options`` — run only when the spec says
    those paths are on this machine. See :func:`paths_are_local`; a run that
    names ``cluster_ssh`` is not asked a question its answer cannot be right for.

    There is no parameter for this and no flag: the spec decides, so there is one
    fact in one place and no caller that can assert a different one. A caller that
    genuinely is not on the cluster says so in the spec it hands over.

    Deliberately narrow: suppressing the whole validator would also
    skip the metric-name, curve-mode and budget checks, each of which catches a
    spec error that otherwise surfaces tens of minutes into an agent turn.
    """
    task_path = Path(path)
    if not task_path.is_file():
        raise TaskSchemaError(f"task file not found: {task_path}")

    text = task_path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"{task_path} is not valid YAML: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TaskSchemaError(
            f"{task_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    errors: list[str] = []

    # Decided once, from the spec, before any path is looked at. Every other check
    # in this function runs either way — this suppresses existence, and only
    # existence. See :func:`paths_are_local`.
    check_paths = paths_are_local(data)

    for field in REQUIRED_PATH_FIELDS:
        if field not in data:
            errors.append(f"missing required field '{field}'")
            continue
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{field}' must be a non-empty string, got {type(value).__name__}")
            continue
        if check_paths and not Path(value).exists():
            errors.append(
                f"'{field}' points to a non-existent path: {value} "
                f"(checked on {os.uname().nodename}, the host running this workflow)"
            )

    # Optional top-level extra_llm_api_options YAML: when set it must be a
    # non-empty string pointing at an existing file.
    extra = data.get(EXTRA_LLM_API_OPTIONS_FIELD)
    if extra is not None:
        if not isinstance(extra, str) or not extra.strip():
            errors.append(
                f"'{EXTRA_LLM_API_OPTIONS_FIELD}' must be a non-empty string, "
                f"got {type(extra).__name__}"
            )
        elif check_paths and not Path(extra).exists():
            errors.append(
                f"'{EXTRA_LLM_API_OPTIONS_FIELD}' points to a non-existent path: {extra} "
                f"(checked on {os.uname().nodename}, the host running this workflow)"
            )

    benchmark = _validate_mapping_block(data, "benchmark", errors)
    _validate_int_fields(benchmark, "benchmark", _BENCHMARK_INT_FIELDS, errors)
    _validate_str_fields(benchmark, "benchmark", _BENCHMARK_STR_FIELDS, errors)
    _validate_concurrency(benchmark, errors)
    _validate_num_prompts(benchmark, errors)
    # ``request_rate`` is either a positive number or the string "inf".
    rr = benchmark.get("request_rate")
    if rr is not None:
        rr_ok = (isinstance(rr, (int, float)) and not isinstance(rr, bool)) or (
            isinstance(rr, str) and rr.strip().lower() in ("inf", "infinity")
        )
        if not rr_ok:
            errors.append(
                f"'benchmark.request_rate' must be a number or \"inf\", got {type(rr).__name__}"
            )

    profile = _validate_mapping_block(data, "profile", errors)
    methods = profile.get("methods")
    if methods is not None:
        if not isinstance(methods, list) or not methods:
            errors.append("'profile.methods' must be a non-empty list of strings")
        else:
            bad = [m for m in methods if m not in VALID_PROFILE_METHODS]
            if bad:
                errors.append(
                    f"'profile.methods' contains unsupported values {bad}; "
                    f"valid methods are {list(VALID_PROFILE_METHODS)}"
                )
    if "nsys_iter_range" in profile and profile["nsys_iter_range"] is not None:
        if (
            not isinstance(profile["nsys_iter_range"], str)
            or not profile["nsys_iter_range"].strip()
        ):
            errors.append("'profile.nsys_iter_range' must be a non-empty string (e.g. \"100-150\")")

    if SLURM_ENVIRONMENT_FIELD in data and data[SLURM_ENVIRONMENT_FIELD] is not None:
        slurm_environment = data[SLURM_ENVIRONMENT_FIELD]
        if not isinstance(slurm_environment, dict):
            errors.append(
                f"'{SLURM_ENVIRONMENT_FIELD}' must be a mapping, "
                f"got {type(slurm_environment).__name__}"
            )
        else:
            for field in SLURM_REQUIRED_FIELDS:
                if field not in slurm_environment:
                    errors.append(f"'{SLURM_ENVIRONMENT_FIELD}.{field}' is required")
                    continue
                value = slurm_environment[field]
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"'{SLURM_ENVIRONMENT_FIELD}.{field}' must be a "
                        f"non-empty string, got {type(value).__name__}"
                    )
            # Optional, but type-checked when given: this value is interpolated
            # into every ssh command the run issues, so a non-string here would
            # surface as a mangled shell command tens of minutes later rather
            # than as a spec error now.
            cluster_ssh = slurm_environment.get(SLURM_CLUSTER_SSH_FIELD)
            if cluster_ssh is not None and (
                not isinstance(cluster_ssh, str) or not cluster_ssh.strip()
            ):
                errors.append(
                    f"'{SLURM_ENVIRONMENT_FIELD}.{SLURM_CLUSTER_SSH_FIELD}' must be a "
                    f"non-empty string when set, got {type(cluster_ssh).__name__}"
                )

    if _RENAMED_SOL_FIELD in data:
        errors.append(
            f"'{_RENAMED_SOL_FIELD}' was replaced by '{SOL_FIELD}' — the "
            "projector now follows the internal-perf-sol-analysis skill "
            f"alone, and runs by default; drop the block, or write "
            f"`{SOL_FIELD}:` with a `gpu` part-name hint for the skill's "
            f"peaks calculator (or `{SOL_ENABLED_FIELD}: false` to skip "
            "the stage)"
        )
    if SOL_FIELD in data:
        sol = data[SOL_FIELD]
        if sol is None:
            # A bare ``sol:`` — every field is optional, so the key alone
            # says nothing the defaults do not; normalize so the merge
            # below has a mapping to work with.
            data[SOL_FIELD] = {}
        elif isinstance(sol, bool):
            # The gate is a field inside the block, not the block itself
            # — say so rather than letting the generic mapping error
            # leave the reader to guess the spelling.
            errors.append(
                f"'{SOL_FIELD}' must be a mapping, got bool — the stage "
                f"gate is a field inside it: write `{SOL_ENABLED_FIELD}: "
                f"false` under `{SOL_FIELD}:` to skip the projector "
                "stage, which otherwise runs by default"
            )
        elif not isinstance(sol, dict):
            errors.append(f"'{SOL_FIELD}' must be a mapping, got {type(sol).__name__}")
        else:
            # The stage gate is a key now, so a misspelling is not an
            # inert extra field: it leaves the default in place and runs
            # the stage the block was written to skip. Reject it for the
            # same reason the ``dlsim`` rename is rejected — a wrong
            # spelling should fail at the CLI boundary, not be ignored.
            unknown = sorted(str(key) for key in sol if key not in SOL_FIELDS)
            if unknown:
                errors.append(
                    f"'{SOL_FIELD}' has unknown field(s) "
                    f"{', '.join(repr(key) for key in unknown)} — valid "
                    f"fields are {', '.join(repr(key) for key in SOL_FIELDS)}"
                )
            _validate_str_fields(sol, SOL_FIELD, SOL_OPTIONAL_STR_FIELDS, errors)
            enabled = sol.get(SOL_ENABLED_FIELD)
            if SOL_ENABLED_FIELD in sol and enabled is None:
                # A bare ``enabled:`` states nothing — drop it so the
                # default lands in the resolved spec instead of a null
                # that later readers would have to interpret.
                sol.pop(SOL_ENABLED_FIELD)
            elif enabled is not None and not isinstance(enabled, bool):
                errors.append(
                    f"'{SOL_FIELD}.{SOL_ENABLED_FIELD}' must be a boolean, "
                    f"got {type(enabled).__name__}"
                )

    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed perf-analyze schema validation:{bullet}{bullet.join(errors)}"
        )

    # Normalize: merge defaults *under* the user's values so the resolved
    # spec on disk is fully explicit for the agents. A concurrency list is
    # normalized to sorted-ascending unique points so every stage (sweep
    # order, per-point result dirs, ledger pairing, "largest point"
    # profiling) shares one deterministic order. A num_prompts list stays
    # paired with its concurrency point through the sort (validation
    # already rejected duplicate points in that case).
    if isinstance(benchmark.get("concurrency"), list):
        if isinstance(benchmark.get("num_prompts"), list):
            pairs = sorted(zip(benchmark["concurrency"], benchmark["num_prompts"]))
            benchmark["concurrency"] = [c for c, _ in pairs]
            benchmark["num_prompts"] = [n for _, n in pairs]
        else:
            benchmark["concurrency"] = sorted(set(benchmark["concurrency"]))
    data["benchmark"] = {**BENCHMARK_DEFAULTS, **benchmark}
    data["profile"] = {**PROFILE_DEFAULTS, **profile}
    # ``sol`` is materialized even when the user never wrote the block —
    # the projector is on by default, so the resolved spec has to state
    # the gate rather than leave it to a reader's assumption.
    data[SOL_FIELD] = {**SOL_DEFAULTS, **data.get(SOL_FIELD, {})}

    return data


def has_slurm_environment(data: Mapping[str, Any]) -> bool:
    """Return whether a validated task spec requested Slurm execution."""
    return SLURM_ENVIRONMENT_FIELD in data and data[SLURM_ENVIRONMENT_FIELD] is not None


def paths_are_local(data: Mapping[str, Any]) -> bool:
    """Whether ``checkpoint_path`` & co. name files THIS process could open.

    They do not when the spec carries ``slurm-environment.cluster_ssh``, whose
    whole meaning is "this process is not on the cluster". A ``Path(...).exists()``
    then answers about the wrong machine: it reports False for a checkpoint that is
    perfectly present, and the run is refused for a defect it does not have.

    The spec is asked rather than the caller told, because the spec is the thing
    that knows. An earlier version of this took a ``--paths-prevalidated`` flag on
    the command line, which had three problems: the same fact was represented
    twice and could disagree, a bare `perf-optimize --task` on an off-cluster host
    failed on correct input because nobody passed it, and it was a CLI flag that
    had quietly become a cross-version contract.

    Skipping is the whole of it — nothing here reaches for the cluster. Validation
    stays local, instant, and free of a network call that could turn "your spec is
    wrong" into "the cluster is unreachable". The check is not lost, it is
    relocated: ``service/adapter/preflight.py:check_repo`` runs it over ssh on the
    host that owns the paths, and more thoroughly than this ever did.

    NOTE FOR THE SERVICE: `flow_version` greps a pinned checkout for this
    function's *name* to decide whether that commit can be driven from off-cluster.
    Renaming it silently re-opens the failure that check exists to prevent — a run
    that provisions for an hour and then dies validating a path that is fine.
    """
    return not cluster_ssh(data)


def cluster_ssh(data: Mapping[str, Any]) -> str:
    """The cluster's ssh alias, or ``""`` when the run is on the cluster already.

    One accessor rather than each caller reaching into the block, because the
    empty string is the *decision* every consumer makes — local vs remote — and
    a caller that spelled the lookup itself could get a ``None`` and treat it as
    a host name. Returning ``""`` makes "we are on the cluster" the falsy
    default, which is also the historical behaviour.
    """
    block = data.get(SLURM_ENVIRONMENT_FIELD)
    if not isinstance(block, Mapping):
        return ""
    return str(block.get(SLURM_CLUSTER_SSH_FIELD) or "").strip()


def sol_enabled(data: Mapping[str, Any]) -> bool:
    """Return whether a task spec enables the projector stage.

    The stage is **on by default**: only an explicit
    ``sol.enabled: false`` turns it off. Deliberately also correct on a
    raw, pre-normalization spec — a missing block and a bare ``sol:``
    both read as enabled — so a resume that re-reads the spec cannot
    flip the gate.
    """
    sol = data.get(SOL_FIELD)
    if isinstance(sol, Mapping):
        return sol.get(SOL_ENABLED_FIELD, True) is not False
    return True


def is_curve_mode(data: Mapping[str, Any]) -> bool:
    """True iff ``benchmark.concurrency`` is a list (Pareto-curve mode).

    The YAML *type* is the mode switch: ``[64]`` is curve mode with one
    point, never silently unwrapped, so the mode is deterministic from
    the resolved spec on disk.
    """
    benchmark = data.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return False
    return isinstance(benchmark.get("concurrency"), list)


def concurrency_points(data: Mapping[str, Any]) -> list[int]:
    """Concurrency points of a validated spec, always as a list.

    ``[c]`` in scalar mode, the (sorted, unique) list itself in curve
    mode, and ``[]`` when the spec carries no benchmark block at all
    (best-effort reads of unvalidated data).
    """
    benchmark = data.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return []
    value = benchmark.get("concurrency")
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def num_prompts_per_point(data: Mapping[str, Any]) -> list[int]:
    """Per-point ``num_prompts``, aligned with :func:`concurrency_points`.

    A scalar broadcasts to every point; a list (curve mode) is returned
    as-is — normalization keeps it sorted alongside the concurrency list.
    ``[]`` when the spec carries no points or no usable ``num_prompts``
    (best-effort reads of unvalidated data).
    """
    points = concurrency_points(data)
    if not points:
        return []
    benchmark = data.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return []
    value = benchmark.get("num_prompts")
    if isinstance(value, list):
        return list(value)
    if _is_positive_int(value):
        return [value] * len(points)
    return []


def dump_task_yaml(data: Mapping[str, Any]) -> str:
    """Serialize a validated/normalized task spec back to YAML text.

    ``math.inf`` would serialize as ``.inf``; the workflow keeps
    ``request_rate`` as a string ("inf") so this is a non-issue, but guard
    anyway so an accidental float-inf does not produce a YAML value the
    agent then has to special-case.
    """
    safe = dict(data)
    bench = safe.get("benchmark")
    if isinstance(bench, dict):
        rr = bench.get("request_rate")
        if isinstance(rr, float) and math.isinf(rr):
            bench = dict(bench)
            bench["request_rate"] = "inf"
            safe["benchmark"] = bench
    return yaml.safe_dump(safe, sort_keys=False, allow_unicode=True, default_flow_style=False)


__all__ = [
    "BENCHMARK_DEFAULTS",
    "EXTRA_LLM_API_OPTIONS_FIELD",
    "PROFILE_DEFAULTS",
    "REQUIRED_PATH_FIELDS",
    "SERVE_BACKEND",
    "SERVE_HOST",
    "SERVE_PORT",
    "SLURM_CLUSTER_SSH_FIELD",
    "SLURM_ENVIRONMENT_FIELD",
    "SLURM_REQUIRED_FIELDS",
    "SOL_DEFAULTS",
    "SOL_ENABLED_FIELD",
    "SOL_FIELD",
    "SOL_FIELDS",
    "SOL_OPTIONAL_STR_FIELDS",
    "VALID_PROFILE_METHODS",
    "TaskSchemaError",
    "concurrency_points",
    "dump_task_yaml",
    "cluster_ssh",
    "has_slurm_environment",
    "is_curve_mode",
    "load_and_validate_task_yaml",
    "num_prompts_per_point",
    "sol_enabled",
]
