"""Disaggregated serving: the ``disagg`` block and the harness config it names.

perf-optimize's aggregate path assumes one ``trtllm-serve`` process on a
fixed port, tuned by one ``extra_llm_api_options`` YAML. Disaggregated
serving has neither property: it is a cluster of context and generation
workers plus a router, and the way it is actually launched is the
TensorRT-LLM checkout's own harness at
``examples/disaggregated/slurm/benchmark/`` — ``submit.py -c config.yaml``
generating an sbatch of ``disaggr_torch.slurm``.

Driving that harness rather than reimplementing it is what keeps the rest
of the workflow untouched, and it is also what creates the only real
problem this module solves: **two files now describe the same run.** The
harness config carries the operating point, and so does ``task.yaml`` —
which the orchestrator reads to build the agents' prompts (curve mode,
the profiling replay point, the gate's per-point wording, 25 call sites
in ``workflow.py``). When the two disagree, nothing raises: the prompts
simply quote numbers the run never measured.

So the harness config is authoritative, and this module reconciles
``task.yaml`` against it at the CLI boundary:

- a measurement condition the user did **not** write is **filled** from
  the harness config;
- one they did write that **disagrees** is an **error**, naming both
  values.

Not a silent overwrite. A ``task.yaml`` you can read is worth more than
one keystroke saved, and "my setting did nothing" is the failure this
avoids. Distinguishing the two cases needs the *raw* file: the base
validator merges defaults, after which an unset ``concurrency`` and a
deliberate ``concurrency: 64`` are indistinguishable.

Two harness properties the mapping has to respect, both from
``examples/disaggregated/slurm/benchmark/run_benchmark.sh``:

- ``concurrency_list`` is **per generation server**; the client is driven
  at ``concurrency * num_gen_servers`` and its result directory is named
  after that product. ``task.yaml`` keeps the total, so the points match
  the artifacts on disk and mean the same thing as in an aggregate run.
- ``num_prompts`` is not configured but derived: ``concurrency *
  multi_round``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

DISAGG_FIELD = "disagg"
DISAGG_CONFIG_KEY = "config"

#: The harness only knows how to wrap workers in nsys (``profiling.nsys_on``
#: plus a per-role iteration window in ``start_worker.sh``). There is no
#: torch-profiler env var and no ncu path anywhere in it.
DISAGG_PROFILE_METHODS: tuple[str, ...] = ("nsys",)


class DisaggConfigError(ValueError):
    """The harness config is unusable, or ``task.yaml`` contradicts it."""


def has_disagg(data: Mapping[str, Any]) -> bool:
    """Whether the spec enables disaggregated serving."""
    return DISAGG_FIELD in data


def disagg_config_path(data: Mapping[str, Any]) -> Path | None:
    """The harness config path from the ``disagg`` block, if any."""
    block = data.get(DISAGG_FIELD)
    if not isinstance(block, Mapping):
        return None
    value = block.get(DISAGG_CONFIG_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value.strip())


def load_disagg_config(path: Path) -> dict[str, Any]:
    """Read the harness config YAML, or raise with a legible message."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - message path
        raise DisaggConfigError(f"could not read disagg config {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise DisaggConfigError(
            f"disagg config {path} must be a YAML mapping, got {type(data).__name__}"
        )
    return dict(data)


def user_set_benchmark_keys(task_path: str | Path) -> set[str]:
    """Which ``benchmark`` keys the user actually wrote.

    Read from the raw file, before the base validator merges its defaults
    — afterwards every key is present and "unset" is unrecoverable, which
    would turn fill-if-absent into always-conflict.
    """
    try:
        raw = yaml.safe_load(Path(task_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return set()  # the base validator will report it properly
    benchmark = raw.get("benchmark") if isinstance(raw, Mapping) else None
    return set(benchmark) if isinstance(benchmark, Mapping) else set()


def _require_mapping(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    block = cfg.get(key)
    if not isinstance(block, Mapping):
        raise DisaggConfigError(
            f"disagg config is missing the '{key}' block (got "
            f"{type(block).__name__}); see "
            f"examples/disaggregated/slurm/benchmark/config.yaml"
        )
    return block


def _positive_int(block: Mapping[str, Any], where: str, key: str) -> int:
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DisaggConfigError(
            f"disagg config '{where}.{key}' must be a positive integer, got {value!r}"
        )
    return value


def _concurrency_list(value: Any) -> list[int]:
    """Parse ``benchmark.concurrency_list`` — ``run_benchmark.sh`` word-splits it."""
    if isinstance(value, int) and not isinstance(value, bool):
        items: list[Any] = [value]
    elif isinstance(value, str):
        items = value.replace(",", " ").split()
    else:
        raise DisaggConfigError(
            f"disagg config 'benchmark.concurrency_list' must be an int or a "
            f"space-separated string (the form run_benchmark.sh iterates), got "
            f"{type(value).__name__}"
        )
    points = []
    for item in items:
        try:
            point = int(str(item).strip())
        except (TypeError, ValueError) as exc:
            raise DisaggConfigError(
                f"disagg config 'benchmark.concurrency_list' entry {item!r} is not an integer"
            ) from exc
        if point <= 0:
            raise DisaggConfigError(
                f"disagg config 'benchmark.concurrency_list' entry {point} must be positive"
            )
        points.append(point)
    if not points:
        raise DisaggConfigError("disagg config 'benchmark.concurrency_list' is empty")
    return points


def worker_config_yaml(cfg: Mapping[str, Any]) -> str:
    """The live tuning config's seed: the harness' ``worker_config`` block.

    In a disagg campaign the live tuning file is not an
    ``extra_llm_api_options`` YAML — it is the ``ctx``/``gen`` role
    configuration, lifted out so the optimizer has exactly one file to
    edit and the orchestrator's single-file diff / revert /
    accepted-snapshot machinery applies to it unchanged.
    """
    worker_config = _require_mapping(cfg, "worker_config")
    for role in ("ctx", "gen"):
        if not isinstance(worker_config.get(role), Mapping):
            raise DisaggConfigError(f"disagg config 'worker_config.{role}' must be a mapping")
    return yaml.safe_dump(dict(worker_config), sort_keys=False, default_flow_style=False)


def apply_harness_conditions(
    task_data: dict[str, Any], cfg: Mapping[str, Any], user_set: set[str]
) -> list[str]:
    """Reconcile ``task_data`` against the harness config, in place.

    ``user_set`` is the set of ``benchmark`` keys the user actually wrote
    (see :func:`user_set_benchmark_keys`). Those must agree with the
    harness config; everything else is filled from it. Returns the notes
    describing what was filled, which the caller records in the resolved
    spec so "where did this value come from" is answerable from the file.
    """
    benchmark_cfg = _require_mapping(cfg, "benchmark")
    hardware_cfg = _require_mapping(cfg, "hardware")

    num_gen_servers = _positive_int(hardware_cfg, "hardware", "num_gen_servers")
    multi_round = _positive_int(benchmark_cfg, "benchmark", "multi_round")
    per_server = _concurrency_list(benchmark_cfg.get("concurrency_list"))

    totals = sorted({point * num_gen_servers for point in per_server})
    expected: dict[str, Any] = {
        "concurrency": totals[0] if len(totals) == 1 else totals,
        "num_prompts": (
            totals[0] * multi_round
            if len(totals) == 1
            else [total * multi_round for total in totals]
        ),
        "random_input_len": _positive_int(benchmark_cfg, "benchmark", "input_length"),
        "random_output_len": _positive_int(benchmark_cfg, "benchmark", "output_length"),
    }
    dataset_file = benchmark_cfg.get("dataset_file")
    if isinstance(dataset_file, str) and dataset_file.strip():
        # The harness client is hard-wired to `--dataset-name trtllm_custom
        # --dataset-path <dataset_file>`; `random` is not reachable there.
        expected["dataset_name"] = "trtllm_custom"
        expected["dataset_path"] = dataset_file

    why = {
        "concurrency": f"concurrency_list {per_server} x num_gen_servers {num_gen_servers}",
        "num_prompts": f"concurrency x multi_round {multi_round}",
        "random_input_len": "benchmark.input_length",
        "random_output_len": "benchmark.output_length",
        "dataset_name": "the harness client is hard-wired to trtllm_custom",
        "dataset_path": "benchmark.dataset_file",
    }

    benchmark = dict(task_data.get("benchmark") or {})
    notes: list[str] = []
    conflicts: list[str] = []
    for key, value in expected.items():
        if key in user_set and benchmark.get(key) != value:
            conflicts.append(
                f"'benchmark.{key}' is {benchmark.get(key)!r} but the disagg config "
                f"gives {value!r} ({why[key]})"
            )
        elif key not in user_set:
            benchmark[key] = value
            notes.append(f"benchmark.{key}={value!r} from the disagg config ({why[key]})")
    if conflicts:
        bullet = "\n  - "
        raise DisaggConfigError(
            f"task.yaml contradicts the disagg config, which owns the measurement "
            f"conditions:{bullet}{bullet.join(conflicts)}{bullet}"
            f"remove these keys to take the disagg config's values."
        )
    task_data["benchmark"] = benchmark

    profile = dict(task_data.get("profile") or {})
    dropped = [m for m in (profile.get("methods") or []) if m not in DISAGG_PROFILE_METHODS]
    profile["methods"] = list(DISAGG_PROFILE_METHODS)
    profiling_cfg = cfg.get("profiling")
    if isinstance(profiling_cfg, Mapping):
        gen_range = profiling_cfg.get("gen_profile_range")
        if isinstance(gen_range, str) and gen_range.strip():
            profile["nsys_iter_range"] = gen_range.strip()
    profile.pop("kernel_coverage", None)
    task_data["profile"] = profile
    if dropped:
        notes.append(
            f"profile.methods {dropped} dropped: the disagg harness only wraps workers "
            f"in nsys (no torch-profiler env var, no ncu path)"
        )

    if task_data.pop("accuracy", None) is not None:
        notes.append(
            "accuracy block ignored: accuracy in a disagg campaign is the harness' own "
            "accuracy block (lm_eval inside the job)"
        )
    return notes
