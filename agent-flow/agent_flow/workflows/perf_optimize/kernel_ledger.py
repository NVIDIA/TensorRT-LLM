"""Schema for the perf-optimize ``kernel_ledger.yaml`` contract.

The ledger is the machine-readable exhaustiveness proof behind
``profile.kernel_coverage``: one row per kernel (or grouped kernel
family) at/above the task's share bar, each row answering the two
per-kernel questions — *(1) can it be made faster?* and *(2) can it be
fused with its neighbors?* — with either a roadmap item or an
evidence-backed dismissal. The analyzer writes one ledger per round into
that round's ``analysis/`` directory; the orchestrator validates it the
moment the turn ends (shape here, roadmap cross-references and the
coverage target in :func:`cross_validate`), so a campaign can never
conclude with a hot kernel whose optimization or fusion possibility was
silently skipped.

Shape::

    version: 1
    source: rounds/round_1/analysis/nsys_stats.txt   # the kern_sum enumerated
    coverage:
      enumerated_share_pct: 96.8    # sum of kernels[].share_pct
      other_share_pct: 3.2          # the explicit below-bar tail roll-up
      min_share_pct: 0.5            # the bar rows were enumerated down to
    kernels:
      - kernel: gdn_bf16_state              # distinctive stem / group label (unique)
        full_name: "void tensorrt_llm::..." # representative full name(s)
        share_pct: 18.4                     # % of profiled GPU time (nsys kern_sum)
        ncu:                                # metrics mapping, OR the whole-block
          duration_us: 41.2                 #   degrade string "unavailable: <reason>"
          sm_sol_pct: 12.1                  # any metric may be null if `note` says
          mem_sol_pct: 78.5                 #   why the capture did not yield it
          occupancy_pct: 62.0
          bound: memory                     # compute | memory | latency | balanced | comm
          note: ""                          # required when a metric above is null
        faster:                             # question 1 — make this kernel faster
          disposition: item                 # item | dismissed
          ref: opt-003                      # item id, or the dismissal evidence
        fusion:                             # question 2 — fuse with neighbors
          disposition: dismissed
          neighbors: "rmsnorm -> THIS -> fp8_quant (cuda_gpu_trace step 120)"
          ref: "multi-consumer-pinned: intermediate feeds residual + norm (torch_trace)"

Ownership mirrors ``roadmap.yaml``: only the analyzer writes the ledger
(a fresh file each round, carrying forward still-valid dismissals); the
orchestrator only validates. ``item`` refs may point at items of any
status — a kernel whose fix was already accepted or failed *was*
considered, which is exactly what the ledger proves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

LEDGER_VERSION = 1
LEDGER_FILENAME = "kernel_ledger.yaml"

DISPOSITIONS = ("item", "dismissed")

# ncu bound classes per the perf-nsight-compute-analysis skill.
BOUND_CLASSES = ("compute", "memory", "latency", "balanced", "comm")

# Shorthand analyzers have written (or plausibly will) for the bound
# enum, mapped to the canonical value. Normalized on load — an alias here
# must be unambiguous; anything else still fails validation.
_BOUND_ALIASES = {
    "compute-bound": "compute",
    "sm": "compute",
    "math": "compute",
    "memory-bound": "memory",
    "mem": "memory",
    "memory-bw": "memory",
    "bandwidth": "memory",
    "latency-bound": "latency",
    "launch": "latency",
    "launch-latency": "latency",
    "mixed": "balanced",
    "communication": "comm",
    "comm-bound": "comm",
    "nccl": "comm",
    "collective": "comm",
}

_NCU_METRIC_FIELDS = ("duration_us", "sm_sol_pct", "mem_sol_pct", "occupancy_pct")

# |enumerated + other - 100| tolerance: kern_sum percentages are rounded
# per row, so the two buckets may miss 100 by a little — but a large gap
# means rows were dropped without being rolled into `other`.
_COVERAGE_SUM_TOLERANCE = 2.0
# Slack on the coverage target itself (rounding of the enumerated sum).
_COVERAGE_TARGET_TOLERANCE = 0.5


class LedgerError(ValueError):
    """Raised when ``kernel_ledger.yaml`` fails schema validation."""


def _is_number(value: Any) -> bool:
    # bool is an int subclass — reject it explicitly.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_coverage(data: Mapping[str, Any], errors: list[str]) -> None:
    coverage = data.get("coverage")
    if not isinstance(coverage, dict):
        errors.append(
            f"'coverage' must be a mapping with 'enumerated_share_pct', "
            f"'other_share_pct', and 'min_share_pct', got {coverage!r}"
        )
        return
    values: dict[str, float] = {}
    for field in ("enumerated_share_pct", "other_share_pct", "min_share_pct"):
        value = coverage.get(field)
        if not _is_number(value) or value < 0:
            errors.append(f"'coverage.{field}' must be a number >= 0, got {value!r}")
        else:
            values[field] = float(value)
    if {"enumerated_share_pct", "other_share_pct"} <= values.keys():
        total = values["enumerated_share_pct"] + values["other_share_pct"]
        if abs(total - 100.0) > _COVERAGE_SUM_TOLERANCE:
            errors.append(
                f"'coverage.enumerated_share_pct' + 'coverage.other_share_pct' "
                f"must account for ~100% of profiled GPU time, got {total:.1f} — "
                f"kernels dropped from the ledger must be rolled into "
                f"'other_share_pct', never silently discarded"
            )


def _validate_ncu(entry: Any, where: str, errors: list[str]) -> None:
    """Validate a row's ``ncu`` block: the metrics mapping or the degrade string."""
    if isinstance(entry, str):
        # The honest degrade for a kernel no capture pass reached — the
        # dispositions are still owed (from nsys shares + the source).
        if not entry.strip():
            errors.append(f"'{where}.ncu' string form must be non-empty (the reason)")
        return
    if not isinstance(entry, dict):
        errors.append(
            f"'{where}.ncu' must be a metrics mapping or a non-empty "
            f"'unavailable: <reason>' string, got {entry!r}"
        )
        return
    # ncu often times a kernel while its SOL / occupancy sections come back
    # empty (replay stalls, LaunchFailed, the hang-detector budget). Those
    # metrics may be null, but only with a non-empty `note` saying why.
    note = entry.get("note")
    has_note = isinstance(note, str) and bool(note.strip())
    missing: list[str] = []
    for field in _NCU_METRIC_FIELDS:
        value = entry.get(field)
        if value is None:
            missing.append(field)
            continue
        if not _is_number(value) or value < 0:
            errors.append(f"'{where}.ncu.{field}' must be a number >= 0, got {value!r}")
    if missing and not has_note:
        errors.append(
            f"'{where}.ncu' leaves {missing} null without a 'note' — a null "
            f"metric must be accompanied by a non-empty 'note' explaining why "
            f"the capture did not yield it (or use the whole-block "
            f"'unavailable: <reason>' string form)"
        )
    bound = entry.get("bound")
    if isinstance(bound, str) and bound not in BOUND_CLASSES:
        canonical = _BOUND_ALIASES.get(bound.strip().lower(), bound.strip().lower())
        if canonical in BOUND_CLASSES:
            entry["bound"] = canonical
            bound = canonical
    if bound not in BOUND_CLASSES:
        errors.append(f"'{where}.ncu.bound' must be one of {list(BOUND_CLASSES)}, got {bound!r}")


def _validate_disposition(
    row: Mapping[str, Any], question: str, where: str, errors: list[str]
) -> None:
    block = row.get(question)
    if not isinstance(block, dict):
        errors.append(
            f"'{where}.{question}' must be a mapping with 'disposition' and "
            f"'ref' — every kernel row answers both questions; got {block!r}"
        )
        return
    disposition = block.get("disposition")
    if disposition not in DISPOSITIONS:
        errors.append(
            f"'{where}.{question}.disposition' must be one of {list(DISPOSITIONS)}, "
            f"got {disposition!r}"
        )
    ref = block.get("ref")
    if not isinstance(ref, str) or not ref.strip():
        errors.append(
            f"'{where}.{question}.ref' must be a non-empty string (a roadmap "
            f"item id, or the evidence-backed dismissal), got {ref!r}"
        )
    if question == "fusion" and disposition == "dismissed":
        # `neighbors` is the evidence a dismissal rests on; a promoted
        # `item` carries its adjacency in the roadmap entry `ref` names.
        neighbors = block.get("neighbors")
        if not isinstance(neighbors, str) or not neighbors.strip():
            errors.append(
                f"'{where}.fusion.neighbors' must be a non-empty string when the "
                f"disposition is 'dismissed' — the observed adjacency the "
                f"dismissal rests on; got {neighbors!r}"
            )


def _validate_row(row: Any, index: int, seen: set[str], errors: list[str]) -> None:
    where = f"kernels[{index}]"
    if not isinstance(row, dict):
        errors.append(f"'{where}' must be a mapping, got {type(row).__name__}")
        return
    kernel = row.get("kernel")
    if not isinstance(kernel, str) or not kernel.strip():
        errors.append(f"'{where}.kernel' must be a non-empty string, got {kernel!r}")
    elif kernel in seen:
        errors.append(f"'{where}.kernel' duplicates {kernel!r} — row keys must be unique")
    else:
        seen.add(kernel)
    full_name = row.get("full_name")
    if not isinstance(full_name, str) or not full_name.strip():
        errors.append(f"'{where}.full_name' must be a non-empty string, got {full_name!r}")
    share = row.get("share_pct")
    if not _is_number(share) or share < 0:
        errors.append(f"'{where}.share_pct' must be a number >= 0, got {share!r}")
    _validate_ncu(row.get("ncu"), where, errors)
    _validate_disposition(row, "faster", where, errors)
    _validate_disposition(row, "fusion", where, errors)


def load_ledger(path: str | Path) -> dict[str, Any]:
    """Parse ``path`` as YAML and validate the ledger schema.

    Shape only — roadmap cross-references and the coverage target need
    context the file does not carry; run :func:`cross_validate` for
    those. Raises :class:`LedgerError` with **every** detected problem
    batched into a single message.
    """
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise LedgerError(f"kernel ledger not found: {ledger_path}")

    try:
        data = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LedgerError(f"{ledger_path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise LedgerError(
            f"{ledger_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    errors: list[str] = []

    if data.get("version") != LEDGER_VERSION:
        errors.append(f"'version' must be {LEDGER_VERSION}, got {data.get('version')!r}")

    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"'source' must name the nsys kern_sum enumerated, got {source!r}")

    _validate_coverage(data, errors)

    kernels = data.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        errors.append(f"'kernels' must be a non-empty list, got {type(kernels).__name__}")
        kernels = []
    seen: set[str] = set()
    for index, row in enumerate(kernels):
        _validate_row(row, index, seen, errors)

    if errors:
        bullet = "\n  - "
        raise LedgerError(
            f"{ledger_path} failed kernel-ledger schema validation:{bullet}{bullet.join(errors)}"
        )
    return data


def cross_validate(
    ledger: Mapping[str, Any],
    roadmap: Mapping[str, Any],
    coverage_target_pct: float,
) -> list[str]:
    """Context checks a shape-valid ledger still owes; returns the problems.

    - Every ``disposition: item`` ref must name an id present in
      ``roadmap.yaml`` (any status — accepted / failed items *were*
      considered). A ref to a nonexistent id means the possibility was
      claimed planned but never actually landed in the plan.
    - The enumerated rows must reach the task's declared coverage target
      — the deterministic teeth behind "every kernel was considered".
    """
    errors: list[str] = []
    item_ids = {
        item.get("id")
        for item in roadmap.get("items", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    for index, row in enumerate(ledger.get("kernels", [])):
        for question in ("faster", "fusion"):
            block = row.get(question, {})
            if block.get("disposition") == "item" and block.get("ref") not in item_ids:
                errors.append(
                    f"'kernels[{index}].{question}.ref' ({block.get('ref')!r}) does "
                    f"not match any roadmap item id — a disposition of 'item' "
                    f"must point at a real roadmap.yaml entry"
                )
    coverage = ledger.get("coverage", {})
    enumerated = coverage.get("enumerated_share_pct")
    if _is_number(enumerated) and enumerated < coverage_target_pct - _COVERAGE_TARGET_TOLERANCE:
        errors.append(
            f"'coverage.enumerated_share_pct' ({enumerated}) is below the task's "
            f"'profile.kernel_coverage.coverage_target_pct' ({coverage_target_pct}) — "
            f"enumerate further down the kern_sum (grouping related kernels is "
            f"fine) until the target is covered"
        )
    return errors
