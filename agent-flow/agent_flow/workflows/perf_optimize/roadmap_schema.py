"""Schema for the perf-optimize ``roadmap.yaml`` contract.

The roadmap is the machine-readable hand-off between the analyzer (which
authors and re-orders optimization items, evidence-grounded and ranked by
expected perf benefit) and the orchestrator (which drives the optimizer ⇄
evaluator loop over the top pending item and owns every lifecycle field).

Ownership rule enforced by convention + prompts:

- The **analyzer** writes item *content* — ids, titles, categories,
  evidence, expected gains — and may re-order pending items or mark them
  ``obsolete`` on later rounds. It never flips an item to ``accepted`` /
  ``failed`` and never edits ``attempts`` / ``measured_gain_pct`` /
  ``current_best``.
- The **orchestrator** owns the lifecycle via the mutators here
  (:func:`mark_in_progress`, :func:`apply_evaluation`,
  :func:`set_current_best`), driven by the evaluator's structured
  progress fields — deterministic, no reliance on agents mutating shared
  state correctly.

Shape (see the workflow README for field semantics)::

    version: 1
    target_metric: output_throughput
    baseline: {value: 1234.5, source: baseline/benchmark_results.md}
    current_best: {value: 1298.7, source: rounds/round_1/attempt_1/evaluation.md}
    items:
      - id: opt-001
        title: Enable CUDA graphs for decode
        category: launch-host
        approach: config
        evidence: ["nsys: 31% GPU idle from per-launch gaps (...)"]
        casebook_ref: "launch storm at decode -> cuda-graph capture"
        expected_gain_pct: 12.0
        expected_gain_rationale: "idle share x casebook-typical recovery"
        how_to_apply: "add cuda_graph_config to tuning/extra_llm_api_options.yaml"
        status: pending
        attempts: 0
        measured_gain_pct: null

In Pareto-curve mode (``benchmark.concurrency`` in ``task.yaml`` is a
list) ``baseline`` / ``current_best`` additionally carry a ``curve``
key — one entry per concurrency point, strictly ascending — and their
scalar ``value`` is the mean of the curve's per-point values (so every
scalar consumer keeps working)::

    baseline:
      value: 1559.7                     # mean of curve[].value
      source: baseline/benchmark_results.md
      curve:
        - {concurrency: 8, value: 812.0, tok_s_user: 21.4, tok_s_gpu: 101.5}
        - {concurrency: 32, value: 1657.0, tok_s_user: 12.9, tok_s_gpu: 207.1}
        - {concurrency: 128, value: 2210.0, tok_s_user: 6.1, tok_s_gpu: 276.3}

Scalar runs simply omit ``curve``. Per-item ``measured_gain_pct`` stays
a scalar in both modes — in curve mode it is the **mean of per-point
gains**. When ``task.yaml`` sets ``optimize.focus_concurrencies``, every
scalar derived from a curve (``value``, ``measured_gain_pct``) is the
mean over **that subset only** — the gate's scored regime — while
``curve`` always carries every configured point.

Item list order **is** priority order (highest expected benefit first);
the orchestrator picks the first ``pending`` item each round.
:func:`load_roadmap` validates with every problem batched into a single
:class:`RoadmapError` (mirroring the task-schema style) and normalizes
missing lifecycle fields (``status`` / ``attempts`` /
``measured_gain_pct``) to their defaults, plus unambiguous ``category``
shorthand (e.g. ``memory`` → ``memory-bw``) to the canonical enum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROADMAP_VERSION = 1

CATEGORIES = ("compute", "memory-bw", "kv-capacity", "launch-host", "communication")

# Shorthand analyzers have written (or plausibly will) for the category
# enum, mapped to the canonical value. Normalized on load — an alias here
# must be unambiguous; anything else still fails validation.
_CATEGORY_ALIASES = {
    "memory": "memory-bw",
    "memory-bandwidth": "memory-bw",
    "bandwidth": "memory-bw",
    "launch": "launch-host",
    "host": "launch-host",
    "comm": "communication",
    "comms": "communication",
    "network": "communication",
    "kv": "kv-capacity",
    "kv-cache": "kv-capacity",
}

APPROACHES = ("config", "code")
STATUSES = ("pending", "in_progress", "accepted", "failed", "obsolete")

_ITEM_LIFECYCLE_DEFAULTS: dict[str, Any] = {
    "status": "pending",
    "attempts": 0,
    "measured_gain_pct": None,
}

# Per-point fields of a ``curve`` entry on ``baseline`` / ``current_best``
# (Pareto-curve mode). All four are required so the reporter never lacks
# the Pareto axes (tok_s_user = 1000/mean_tpot_ms, tok_s_gpu =
# output_throughput/num_gpus).
CURVE_POINT_FIELDS: tuple[str, ...] = ("concurrency", "value", "tok_s_user", "tok_s_gpu")


class RoadmapError(ValueError):
    """Raised when ``roadmap.yaml`` fails schema validation."""


def _is_number(value: Any) -> bool:
    # bool is an int subclass — reject it explicitly so ``true`` does not
    # slip through as a gain value.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_curve(key: str, curve: Any, errors: list[str]) -> None:
    """Validate a ``curve`` list on a metric-ref block (curve mode only)."""
    if not isinstance(curve, list) or not curve:
        errors.append(f"'{key}.curve' must be a non-empty list of per-point mappings")
        return
    previous_concurrency: int | None = None
    for index, point in enumerate(curve):
        where = f"{key}.curve[{index}]"
        if not isinstance(point, dict):
            errors.append(f"'{where}' must be a mapping, got {type(point).__name__}")
            continue
        concurrency = point.get("concurrency")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            errors.append(f"'{where}.concurrency' must be an integer >= 1, got {concurrency!r}")
        else:
            if previous_concurrency is not None and concurrency <= previous_concurrency:
                errors.append(
                    f"'{where}.concurrency' must be strictly ascending "
                    f"(got {concurrency} after {previous_concurrency})"
                )
            previous_concurrency = concurrency
        for field in CURVE_POINT_FIELDS[1:]:
            if not _is_number(point.get(field)):
                errors.append(f"'{where}.{field}' must be a number, got {point.get(field)!r}")


def _validate_metric_ref(data: Mapping[str, Any], key: str, errors: list[str]) -> None:
    """Validate a ``{value, source[, curve]}`` block (``baseline`` / ``current_best``)."""
    block = data.get(key)
    if not isinstance(block, dict):
        errors.append(f"'{key}' must be a mapping with 'value' and 'source', got {block!r}")
        return
    if not _is_number(block.get("value")):
        errors.append(f"'{key}.value' must be a number, got {block.get('value')!r}")
    source = block.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"'{key}.source' must be a non-empty string, got {source!r}")
    if block.get("curve") is not None:
        _validate_curve(key, block["curve"], errors)


def _validate_item(item: Any, index: int, seen_ids: set[str], errors: list[str]) -> None:
    where = f"items[{index}]"
    if not isinstance(item, dict):
        errors.append(f"'{where}' must be a mapping, got {type(item).__name__}")
        return

    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        errors.append(f"'{where}.id' must be a non-empty string, got {item_id!r}")
    elif item_id in seen_ids:
        errors.append(f"'{where}.id' duplicates id {item_id!r} — ids must be unique")
    else:
        seen_ids.add(item_id)

    for field in ("title", "expected_gain_rationale", "how_to_apply"):
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{where}.{field}' must be a non-empty string, got {value!r}")

    category = item.get("category")
    if isinstance(category, str) and category not in CATEGORIES:
        # Analyzers occasionally write shorthand for the enum ("memory");
        # map unambiguous aliases rather than failing the whole roadmap
        # after the stage has finished — that costs a full analyzer
        # re-run over a label the reporter only uses descriptively.
        canonical = _CATEGORY_ALIASES.get(category.strip().lower(), category.strip().lower())
        if canonical in CATEGORIES:
            item["category"] = canonical
            category = canonical
    if category not in CATEGORIES:
        errors.append(f"'{where}.category' must be one of {list(CATEGORIES)}, got {category!r}")

    approach = item.get("approach")
    if approach not in APPROACHES:
        errors.append(f"'{where}.approach' must be one of {list(APPROACHES)}, got {approach!r}")

    evidence = item.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(e, str) and e.strip() for e in evidence)
    ):
        errors.append(f"'{where}.evidence' must be a non-empty list of non-empty strings")

    casebook_ref = item.get("casebook_ref")
    if casebook_ref is not None and not isinstance(casebook_ref, str):
        errors.append(f"'{where}.casebook_ref' must be a string or omitted, got {casebook_ref!r}")
    elif isinstance(casebook_ref, str) and not casebook_ref.strip():
        # Analyzers write `casebook_ref: ""` for items with no casebook row
        # (e.g. prior-campaign carry-overs); treat that as absent rather
        # than failing the whole roadmap after the stage has finished.
        item["casebook_ref"] = None

    expected = item.get("expected_gain_pct")
    if not _is_number(expected) or expected <= 0:
        errors.append(f"'{where}.expected_gain_pct' must be a number > 0, got {expected!r}")

    status = item.get("status", _ITEM_LIFECYCLE_DEFAULTS["status"])
    if status not in STATUSES:
        errors.append(f"'{where}.status' must be one of {list(STATUSES)}, got {status!r}")

    attempts = item.get("attempts", _ITEM_LIFECYCLE_DEFAULTS["attempts"])
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
        errors.append(f"'{where}.attempts' must be an integer >= 0, got {attempts!r}")

    measured = item.get("measured_gain_pct", None)
    if measured is not None and not _is_number(measured):
        errors.append(f"'{where}.measured_gain_pct' must be a number or null, got {measured!r}")


def load_roadmap(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the roadmap schema.

    Returns the parsed mapping with per-item lifecycle defaults
    (``status: pending``, ``attempts: 0``, ``measured_gain_pct: null``)
    filled in. Raises :class:`RoadmapError` with **every** detected
    problem batched into a single message.
    """
    roadmap_path = Path(path)
    if not roadmap_path.is_file():
        raise RoadmapError(f"roadmap file not found: {roadmap_path}")

    try:
        data = yaml.safe_load(roadmap_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RoadmapError(f"{roadmap_path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise RoadmapError(
            f"{roadmap_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    errors: list[str] = []

    if data.get("version") != ROADMAP_VERSION:
        errors.append(f"'version' must be {ROADMAP_VERSION}, got {data.get('version')!r}")

    target_metric = data.get("target_metric")
    if not isinstance(target_metric, str) or not target_metric.strip():
        errors.append(f"'target_metric' must be a non-empty string, got {target_metric!r}")

    _validate_metric_ref(data, "baseline", errors)
    # ``current_best`` starts equal to the baseline (the analyzer seeds it
    # in round 1) and is advanced by the orchestrator on each accepted
    # item; validate the shape whenever it is present.
    if data.get("current_best") is not None:
        _validate_metric_ref(data, "current_best", errors)

    items = data.get("items")
    if not isinstance(items, list):
        errors.append(f"'items' must be a list, got {type(items).__name__}")
        items = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        _validate_item(item, index, seen_ids, errors)

    if errors:
        bullet = "\n  - "
        raise RoadmapError(
            f"{roadmap_path} failed roadmap schema validation:{bullet}{bullet.join(errors)}"
        )

    for item in items:
        for field, default in _ITEM_LIFECYCLE_DEFAULTS.items():
            item.setdefault(field, default)
    return data


def dump_roadmap(data: Mapping[str, Any]) -> str:
    """Serialize a roadmap mapping back to YAML text."""
    return yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True, default_flow_style=False)


def save_roadmap(path: str | Path, data: Mapping[str, Any]) -> None:
    Path(path).write_text(dump_roadmap(data), encoding="utf-8")


def find_item(data: Mapping[str, Any], item_id: str) -> dict[str, Any] | None:
    """Return the item with ``id == item_id``, or ``None``."""
    for item in data.get("items", []):
        if item.get("id") == item_id:
            return item
    return None


def top_pending_item(
    data: Mapping[str, Any],
    min_expected_gain_pct: float = 0.0,
    allowed_approaches: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Return the first actionable ``pending`` item, or ``None``.

    List order is priority order; items promising less than
    ``min_expected_gain_pct`` (the noise floor) are skipped so the loop
    never spends a round on a gain it could not distinguish from noise.
    When ``allowed_approaches`` is given (``optimize.approaches`` from
    the task spec), items with any other ``approach`` are skipped too —
    the deterministic guarantee that a restricted run never dispatches a
    disallowed item, whatever the analyzer wrote.
    """
    for item in data.get("items", []):
        if item.get("status") != "pending":
            continue
        if item.get("expected_gain_pct", 0) < min_expected_gain_pct:
            continue
        if allowed_approaches is not None and item.get("approach") not in allowed_approaches:
            continue
        return item
    return None


def top_pending_items(
    data: Mapping[str, Any],
    limit: int,
    min_expected_gain_pct: float = 0.0,
    allowed_approaches: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` actionable pending items in roadmap order."""
    if limit < 1:
        return []
    selected: list[dict[str, Any]] = []
    for item in data.get("items", []):
        if item.get("status") != "pending":
            continue
        if item.get("expected_gain_pct", 0) < min_expected_gain_pct:
            continue
        if allowed_approaches is not None and item.get("approach") not in allowed_approaches:
            continue
        selected.append(item)
        if len(selected) == limit:
            break
    return selected


def has_actionable_pending(
    data: Mapping[str, Any],
    noise_floor_pct: float,
    allowed_approaches: Sequence[str] | None = None,
) -> bool:
    """True iff any pending item promises a gain at/above the noise floor."""
    return top_pending_item(data, noise_floor_pct, allowed_approaches) is not None


def _mutate(path: str | Path, item_id: str, updates: Mapping[str, Any]) -> None:
    """Read-validate-modify-write a single item's lifecycle fields."""
    data = load_roadmap(path)
    item = find_item(data, item_id)
    if item is None:
        raise RoadmapError(f"roadmap {path} has no item with id {item_id!r}")
    item.update(updates)
    save_roadmap(path, data)


def mark_in_progress(path: str | Path, item_id: str) -> None:
    """Flip ``item_id`` to ``in_progress`` (the orchestrator picked it)."""
    _mutate(path, item_id, {"status": "in_progress"})


def apply_evaluation(
    path: str | Path,
    item_id: str,
    *,
    status: str,
    attempts: int,
    measured_gain_pct: float | None = None,
) -> None:
    """Record an evaluation outcome on ``item_id``.

    ``status`` is ``accepted`` / ``failed`` for terminal outcomes, or
    ``in_progress`` when a rejected attempt still has retries left (so
    the attempt count is durable across a resume).
    """
    if status not in STATUSES:
        raise RoadmapError(f"invalid status {status!r}; expected one of {list(STATUSES)}")
    updates: dict[str, Any] = {"status": status, "attempts": attempts}
    if measured_gain_pct is not None:
        updates["measured_gain_pct"] = measured_gain_pct
    _mutate(path, item_id, updates)


def set_current_best(
    path: str | Path,
    value: float,
    source: str,
    curve: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Advance the accepted-measurement watermark (on evaluator APPROVE).

    ``curve`` carries the per-concurrency measurements in Pareto-curve
    mode; it is validated before being stored. When ``None`` the block is
    written without a ``curve`` key — dropping any stale curve, so
    ``value`` and ``curve`` can never desync.
    """
    if curve is not None:
        errors: list[str] = []
        _validate_curve("current_best", list(curve), errors)
        if errors:
            bullet = "\n  - "
            raise RoadmapError(f"invalid current_best curve:{bullet}{bullet.join(errors)}")
    data = load_roadmap(path)
    block: dict[str, Any] = {"value": value, "source": source}
    if curve is not None:
        block["curve"] = [dict(point) for point in curve]
    data["current_best"] = block
    save_roadmap(path, data)
