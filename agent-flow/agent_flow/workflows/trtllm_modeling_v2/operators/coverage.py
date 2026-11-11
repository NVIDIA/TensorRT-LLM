# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The survey's ``vocabulary.yaml``: what it must say to be dispatched from, and where an entry lands.

Each step is ``OK`` (an entry's ``CELLS`` cover this checkpoint's values on an arch in its
``ARCHS``), ``CERT`` (an entry exists, this surface is not covered: verify it) or ``GAP``
(no entry: onboard one, from the rung of the sourcing ladder the row names). The steps are
the surveyor's evidence; what is held to a schema is the queue, one kernel dispatch per
row, and the verdict it must agree with. A row on a shared entry ``<category>/<name>``
lands as a local variant in the target's own ``ops/`` (``ops/<category>/<name>__<site>``),
a row on ``ops/<category>/<name>`` in place: :func:`entry_paths` says which files that is.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

MODELING_V2_PACKAGE = "tensorrt_llm/_torch/_experimental/modeling_v2"
CATALOG_DIR = f"{MODELING_V2_PACKAGE}/catalog"
CATALOG_TESTS_DIR = "tests/unittest/_torch/modeling_v2"
#: A target's own entries sit in this directory of the target, one ``<category>/<name>.py``
#: each; their tests mirror that path under ``CATALOG_TESTS_DIR/models/<family>/<target>/``.
MODEL_OPS_DIRNAME = "ops"
#: Where a target's CI accuracy test and its test list entry live.
INTEGRATION_TESTS_DIR = "tests/integration"
STATE_OK, STATE_CERT, STATE_GAP = STATES = ("OK", "CERT", "GAP")
MODE_ONBOARD, MODE_VERIFY = MODES = ("onboard", "verify")
VERDICTS = ("assemblable", "assemblable_after_additions", "ceiling")
VERDICT_ASSEMBLABLE, VERDICT_AFTER_ADDITIONS, VERDICT_CEILING = VERDICTS
#: A multi-rank entry's test only starts its ranks; ``_<name>_op_matrix.py`` drives the cells.
OP_MATRIX_SUFFIX = "_op_matrix"
#: Separates a shared entry's name from the site of its local variant. No entry name has it.
VARIANT_SEPARATOR = "__"
_SITE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
#: A kernel task named by ``<mode>:<entry>`` numbers its dispatch from here, above any row.
REF_DISPATCH_BASE, REF_DISPATCH_SPAN = 10_000, 90_000
#: The unit a queue row belongs to when it names none.
UNIT_CORE = "core"
#: Cells one queue row may list: a row covers each axis of the envelope, not their product.
MAX_ROW_CELLS = 64
#: Where a GAP's implementation may come from, cheapest first (``agents/prompts`` says each).
TIERS = (1, 2, 3, 4, 5)
TIER_TRTLLM_OP, TIER_INTREE_CALLABLE, TIER_KERNEL_LIBRARY, TIER_NATIVE_TORCH, TIER_TRITON = TIERS
#: A tier above this is work this run writes rather than wraps.
HIGHEST_REUSING_TIER = TIER_KERNEL_LIBRARY


class VocabularyError(ValueError):
    """``vocabulary.yaml`` is malformed or internally inconsistent."""


def _shared(target: str) -> bool:
    parts = target.strip().strip("/").split("/")
    return len(parts) == 2 and all(parts) and parts[0] != MODEL_OPS_DIRNAME


@dataclass(frozen=True)
class QueueItem:
    """One kernel writer dispatch: exactly one catalog entry, in one mode."""

    n: int
    mode: str
    #: ``<category>/<name>`` (shared) or ``ops/<category>/<name>`` (the target's own).
    target: str
    depends_on: tuple[int, ...] = ()
    surface: str | None = None
    note: str = ""
    #: Which rung of the sourcing ladder supplies this entry.
    tier: int = TIER_TRTLLM_OP
    #: The first unit of work that needs this entry: what a failed dispatch blocks.
    unit: str = UNIT_CORE
    #: The call site a shared entry's local variant is named for; ``r<n>`` when none is named.
    site: str | None = None

    @property
    def variant(self) -> str | None:
        """Where a row on a shared entry lands, ``ops/<category>/<name>__<site>``; else ``None``."""
        if not _shared(self.target):
            return None
        category, name = self.target.strip().strip("/").split("/")
        site = self.site or f"r{self.n}"
        return f"{MODEL_OPS_DIRNAME}/{category}/{name}{VARIANT_SEPARATOR}{site}"


def _row(at: str, row: Mapping[str, Any], n: int, bad, units) -> QueueItem:
    """One queue row as a dispatch, each field it gets wrong said through ``bad``."""
    mode, target = row.get("mode"), str(row.get("target", "")).strip()
    if mode not in MODES:
        bad(f"{at}.mode is {mode!r}; expected one of {MODES}")
    depends, raw = [], row.get("depends_on") or []
    if not isinstance(raw, list):
        bad(f"{at}.depends_on must be a list of dispatch numbers")
    for dep in raw if isinstance(raw, list) else ():
        if isinstance(dep, int) or str(dep).strip().isdigit():
            depends.append(int(dep))
        else:
            bad(f"{at}.depends_on holds {dep!r}, which is not a number")
    if mode == MODE_VERIFY and not str(row.get("surface", "")).strip():
        bad(f"{at} is a verify dispatch with no `surface`: the value its contract line excludes")
    cells = row.get("cells")
    if isinstance(cells, list) and len(cells) > MAX_ROW_CELLS:
        bad(
            f"{at} lists {len(cells)} cells, more than {MAX_ROW_CELLS}: cover each axis (every "
            f"dtype, each call site's rows at the smallest and largest captured batch, per-rank "
            f"sizes, one chunk at max_num_tokens) from a base cell, not their product"
        )
    tier = row.get("tier", TIER_TRTLLM_OP)
    tier = int(tier) if str(tier).strip().isdigit() else 0
    if tier not in TIERS:
        bad(f"{at}.tier is {row.get('tier')!r}; expected one of {TIERS}")
        tier = TIER_TRTLLM_OP
    unit = UNIT_CORE if row.get("unit") is None else row["unit"]
    unit = unit.strip() if isinstance(unit, str) else unit
    if not (isinstance(unit, str) and unit) or (units is not None and unit not in units):
        bad(f"{at}.unit is {unit!r}: the first unit that needs the entry, of {list(units or ())}")
        unit = UNIT_CORE
    site = None if row.get("site") is None else str(row["site"]).strip()
    if site is not None and not _SITE.match(site):
        bad(f"{at}.site is {row['site']!r}; a site is lowercase words joined by underscores")
        site = None
    elif site is not None and not _shared(target):
        bad(f"{at}.site names a local variant, which only a shared entry has: {target!r}")
        site = None
    surface, note = row.get("surface"), str(row.get("note", ""))
    return QueueItem(n, str(mode), target, tuple(depends), surface, note, tier, unit, site)


def _queue(raw: Any, bad: Callable[[str], None], units: Sequence[str] | None) -> tuple:
    if raw is not None and not isinstance(raw, list):
        bad(f"`queue` must be a list, got {type(raw).__name__}")
        return ()
    items: list[QueueItem] = []
    modes: dict[str, str] = {}
    variants: dict[str, int] = {}
    for index, row in enumerate(raw or ()):
        at = f"queue[{index}]"
        n = row.get("n") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not (isinstance(n, int) or str(n).strip().isdigit()):
            bad(f"{at} must be a mapping whose `n` is an integer dispatch number")
            continue
        if any(item.n == int(n) for item in items):
            bad(f"{at}.n = {n} is a duplicate dispatch number")
        item = _row(at, row, int(n), bad, units)
        target, mode = item.target, item.mode
        if not target:
            bad(f"{at}.target must name exactly one catalog entry")
        elif VARIANT_SEPARATOR in target.rsplit("/", 1)[-1]:
            bad(f"{at}.target {target!r} names a local variant; name the entry, its row lands it")
        elif modes.get(target, mode) != mode:
            bad(f"{at} has {target!r} in both chapters: an entry onboarded here is tested here")
        elif target in modes and not target.startswith(f"{MODEL_OPS_DIRNAME}/"):
            bad(
                f"{at} is a second {mode} row on {target!r}: one row per shared entry, every site's cells"
            )
        else:
            modes[target] = mode
        if item.variant and variants.setdefault(item.variant, item.n) != item.n:
            bad(f"{at} lands at {item.variant}, as row {variants[item.variant]}: name a `site`")
        items.append(item)
    known = {item.n for item in items}
    for item in items:
        for dep in sorted(set(item.depends_on) - known):
            bad(f"queue item {item.n} depends on {dep}, which is not in the queue")
    return tuple(items)


def parse_vocabulary(
    data: Mapping[str, Any], source: str = "vocabulary.yaml", *, units: Sequence[str] | None = None
) -> tuple[QueueItem, ...]:
    """Validate a parsed vocabulary, every problem in one error, and return its queue.

    With ``units`` (the run's units of work) every row's ``unit`` must be one of them.
    """
    if not isinstance(data, dict):
        raise VocabularyError(f"{source} must be a mapping, got {type(data).__name__}")
    errors: list[str] = []
    verdict, ceiling = data.get("verdict"), data.get("ceiling")
    if verdict not in VERDICTS:
        errors.append(f"`verdict` is {verdict!r}; expected one of {VERDICTS}")
    queue = _queue(data.get("queue"), errors.append, units)
    if verdict != VERDICT_CEILING:
        if ceiling not in (None, {}):
            errors.append(f"`ceiling` is set but the verdict is {verdict!r}; they must agree")
    elif not isinstance(ceiling, dict) or not str(ceiling.get("step", "")).strip():
        errors.append("`verdict: ceiling` requires a `ceiling` mapping naming the exact `step`")
    if verdict == VERDICT_ASSEMBLABLE and queue:
        errors.append(
            f"`verdict: assemblable` with {len(queue)} queued dispatch(es): they disagree"
        )
    if verdict == VERDICT_AFTER_ADDITIONS and not queue:
        errors.append(f"`verdict: {VERDICT_AFTER_ADDITIONS}` with an empty queue: they disagree")
    if errors:
        raise VocabularyError(f"{source} is not usable:" + "".join(f"\n  - {e}" for e in errors))
    return queue


def load_vocabulary(
    path: str | Path, *, units: Sequence[str] | None = None
) -> tuple[QueueItem, ...]:
    """Read and validate ``vocabulary.yaml`` from disk. See :func:`parse_vocabulary`."""
    source = Path(path)
    text = source.read_text(encoding="utf-8") if source.is_file() else ""
    if not text.strip():
        raise VocabularyError(f"{source} is missing or empty; the survey must produce it.")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise VocabularyError(f"{source} is not valid YAML: {exc}") from exc
    return parse_vocabulary(data or {}, str(source), units=units)


def queue_item_from_ref(ref: str, task_id: str, goal: str = "") -> QueueItem | None:
    """The dispatch a kernel task names by ``ref`` instead of a queue row; ``None`` if it names none.

    ``onboard:<entry>`` and ``verify:<entry>`` name one entry; after a shared one,
    ``@<site>`` names the call site its local variant is for, else the task id does. The
    dispatch number is the task id's, above every queue row's, so a retry finds its
    decision file.
    """
    mode, _, rest = ref.strip().partition(":")
    target, _, site = (part.strip() for part in rest.strip().partition("@"))
    if mode not in MODES or not target:
        return None
    if site and not (_SITE.match(site) and _shared(target)):
        return None
    if _shared(target) and not site:
        site = re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")
    n = REF_DISPATCH_BASE + zlib.crc32(task_id.encode("utf-8")) % REF_DISPATCH_SPAN
    return QueueItem(n=n, mode=mode, target=target, note=goal, site=site or None)


def model_ops_tests_dir(target_rel: str) -> str:
    """``.../modeling_v2/models/<family>/<target>`` -> its tests' directory."""
    tail = target_rel.rstrip("/").split(f"{MODELING_V2_PACKAGE}/", 1)[-1]
    return f"{CATALOG_TESTS_DIR}/{tail}"


def _tests(folder: str, name: str, repo_root: Path | str | None, matrix: bool = False) -> tuple:
    """The GPU test files of entry ``name`` under ``folder``: one, or a multi-rank matrix's two.

    With ``repo_root``, an entry tested there by an op matrix and no plain test is a matrix.
    """
    plain = f"{folder}/test_modeling_v2_{name}.py"
    starter = f"{folder}/test_modeling_v2_{name}{OP_MATRIX_SUFFIX}.py"
    if not matrix and repo_root is not None:
        root = Path(repo_root)
        matrix = not (root / plain).is_file() and (root / starter).is_file()
    return (starter, f"{folder}/_{name}{OP_MATRIX_SUFFIX}.py") if matrix else (plain,)


def shared_entry_paths(target: str, repo_root: Path | str | None = None) -> tuple[str, ...]:
    """A shared entry ``<category>/<name>``'s module and its GPU test files, repo-relative."""
    category, name = target.strip().strip("/").split("/")
    tests = _tests(f"{CATALOG_TESTS_DIR}/{category}", name, repo_root)
    return (f"{CATALOG_DIR}/{category}/{name}.py", *tests)


def entry_paths(
    item: QueueItem, target_rel: str = "", repo_root: Path | str | None = None
) -> tuple[str, ...] | None:
    """Where the row's dispatch writes: its module, then its GPU test files, repo-relative.

    A row on a shared entry writes its local variant in the target's own ``ops/`` under
    ``target_rel``, tested where the target's tests mirror it (as an op matrix when the
    shared entry is one); with no target it writes the catalog's entry in place. A row on
    ``ops/<category>/<name>`` writes it there. ``None`` when the row cannot be placed.
    """
    target, home, matrix = item.target, target_rel.strip().rstrip("/"), False
    if _shared(target):
        shared = shared_entry_paths(target, repo_root)
        if not home:
            return shared
        target, matrix = str(item.variant), len(shared) == 3
    parts = target.strip().strip("/").split("/")
    if len(parts) != 3 or parts[0] != MODEL_OPS_DIRNAME or not all(parts) or not home:
        return None
    _, category, name = parts
    tests = _tests(f"{model_ops_tests_dir(home)}/{category}", name, repo_root, matrix)
    return (f"{home}/{MODEL_OPS_DIRNAME}/{category}/{name}.py", *tests)
