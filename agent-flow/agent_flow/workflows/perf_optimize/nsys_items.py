"""Coverage of the nsys timeline analysis's ``items.json`` by the roadmap.

The ``perf-nsight-system-analysis`` skill ends every run by writing
``items.json`` beside its per-step JSON: the performance opportunities
the timeline analysis found, each with an ``id``, a ``claim``, the step
table behind it and a ``magnitudeMs``. The skill states the id's purpose
plainly — *"a consumer keys coverage on it"* — and this module is that
consumer.

Without it the handoff is prose: the analyzer reads the numbers, writes
findings, and whether an opportunity reached ``roadmap.yaml`` is
invisible. With it, ``roadmap.yaml`` carries a top-level ``nsys_items``
block accounting for every id — promoted to a roadmap item, or dismissed
with the evidence for dismissing it::

    nsys_items:
      - id: nsys-01
        disposition: item
        ref: opt-001
      - id: nsys-02
        disposition: dismissed
        ref: "0.2 ms/iter is below the noise floor at this operating point"

Deliberately the same ``disposition`` / ``ref`` vocabulary as
``kernel_ledger.yaml`` (whose :data:`~.kernel_ledger.DISPOSITIONS` this
imports rather than redefining), for the same reason: a campaign should
never end with a measured opportunity that was silently skipped. The two
cover different ground — the ledger asks *per kernel* whether it can be
made faster or fused, this asks whether the timeline's own findings
(exposed collectives, launch-starved host time, rank jitter) were
planned or dismissed.

Self-gating: enforced exactly when the round produced an ``items.json``.
A degraded round (skill unavailable, export or pipeline error) writes
none, owes the block nothing, and records the reason under *Caveats*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent_flow.workflows.perf_optimize.kernel_ledger import DISPOSITIONS

# Written by the skill's pipeline, inside the analysis directory the
# analyzer points ``run_all.py --out`` at.
NSYS_ANALYSIS_DIRNAME = "nsys_analysis"
ITEMS_FILENAME = "items.json"

# The roadmap key carrying the coverage rows.
ROADMAP_KEY = "nsys_items"


class NsysItemsError(ValueError):
    """Raised when ``items.json`` cannot be read as the skill defines it."""


def items_path(analysis_dir: str | Path) -> Path:
    """Where the round's timeline analysis writes its opportunity list."""
    return Path(analysis_dir) / NSYS_ANALYSIS_DIRNAME / ITEMS_FILENAME


def load_item_ids(path: str | Path) -> list[str]:
    """Return the ``id`` of every entry in ``items.json``, in file order.

    Only the ids are read: the skill owns the rest of the shape, and the
    roadmap's obligation is coverage, not a second transcription of the
    evidence. Raises :class:`NsysItemsError` when the file is unreadable
    or an entry carries no usable id — either way the analyzer's own
    artifact is malformed, which is worth stopping for.
    """
    items_file = Path(path)
    try:
        data = json.loads(items_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NsysItemsError(f"{items_file} could not be read as JSON: {exc}") from exc

    items = data.get("items") if isinstance(data, Mapping) else None
    if not isinstance(items, list):
        raise NsysItemsError(
            f"{items_file} must be a JSON object with an 'items' list (the skill "
            f'writes {{"items": []}} when it found nothing), got {type(items).__name__}'
        )

    ids: list[str] = []
    for index, entry in enumerate(items):
        item_id = entry.get("id") if isinstance(entry, Mapping) else None
        if not isinstance(item_id, str) or not item_id.strip():
            raise NsysItemsError(
                f"{items_file} entry {index} has no non-empty 'id' — the id is what "
                f"the roadmap's coverage keys on, so it cannot be omitted"
            )
        ids.append(item_id)
    return ids


def cross_validate(roadmap: Mapping[str, Any], item_ids: list[str]) -> list[str]:
    """Check the roadmap's ``nsys_items`` block against ``item_ids``.

    Returns every problem found (empty when the block is sound):

    - each row is shaped ``{id, disposition, ref}`` with a known
      disposition and a non-empty ``ref``;
    - a ``disposition: item`` ``ref`` names a real roadmap item id, of
      any status — an opportunity whose fix was already accepted or
      failed *was* considered, which is what the block proves;
    - every id in ``items.json`` is covered, and no row invents one that
      is not.
    """
    errors: list[str] = []
    rows = roadmap.get(ROADMAP_KEY)
    if not isinstance(rows, list):
        return [
            f"'{ROADMAP_KEY}' must be a list with one row per id in {ITEMS_FILENAME} "
            f"({len(item_ids)} to account for), got {type(rows).__name__} — every "
            f"opportunity the timeline analysis found is either planned as a roadmap "
            f"item or dismissed with evidence"
        ]

    roadmap_ids = {
        item.get("id")
        for item in roadmap.get("items", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    covered: set[str] = set()
    for index, row in enumerate(rows):
        where = f"{ROADMAP_KEY}[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"'{where}' must be a mapping with 'id', 'disposition' and 'ref'")
            continue
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            errors.append(f"'{where}.id' must be a non-empty string, got {row_id!r}")
        elif row_id in covered:
            errors.append(f"'{where}.id' duplicates {row_id!r} — one row per opportunity")
        else:
            covered.add(row_id)
        disposition = row.get("disposition")
        if disposition not in DISPOSITIONS:
            errors.append(
                f"'{where}.disposition' must be one of {list(DISPOSITIONS)}, got {disposition!r}"
            )
        ref = row.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            errors.append(
                f"'{where}.ref' must be a non-empty string (a roadmap item id, or the "
                f"evidence for dismissing the opportunity), got {ref!r}"
            )
        elif disposition == "item" and ref not in roadmap_ids:
            errors.append(
                f"'{where}.ref' ({ref!r}) does not match any roadmap item id — a "
                f"disposition of 'item' must point at a real roadmap.yaml entry"
            )

    known = set(item_ids)
    missing = [item_id for item_id in item_ids if item_id not in covered]
    if missing:
        errors.append(
            f"{ITEMS_FILENAME} ids {missing} are unaccounted for in '{ROADMAP_KEY}' — "
            f"plan each as a roadmap item or dismiss it with evidence"
        )
    unknown = sorted(covered - known)
    if unknown:
        errors.append(
            f"'{ROADMAP_KEY}' rows {unknown} name ids absent from {ITEMS_FILENAME} — "
            f"the block accounts for that file, it does not extend it"
        )
    return errors
