# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator coverage in ``vocabulary.yaml`` — the survey's machine-readable product.

Each computation step lands in one of three states:

===========  ==========================================================
``OK``       an entry exists and its ``CELLS`` cover this checkpoint's
             values on an arch in its ``ARCHS`` — nothing to do
``CERT``     an entry exists, but this surface is not covered — dispatch
             the kernel writer to verify it
``GAP``      no entry at all — dispatch the kernel writer to onboard one
===========  ==========================================================

A ``GAP`` row says *where* its implementation comes from, via the sourcing
ladder (``TIER_*``). Only when **every** rung is exhausted is the verdict a
**vocabulary ceiling**, and that claim has to carry its evidence per rung.
"""

from __future__ import annotations

import ast
import re
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

MODELING_V2_PACKAGE = "tensorrt_llm/_torch/_experimental/modeling_v2"
CATALOG_DIR = f"{MODELING_V2_PACKAGE}/catalog"
CATALOG_TESTS_DIR = "tests/unittest/_torch/modeling_v2"
#: A target's own entries -- ops only it calls -- sit in this directory of the
#: target, one ``<category>/<name>.py`` each, and their tests mirror that path
#: under ``CATALOG_TESTS_DIR/models/<family>/<target>/``.
MODEL_OPS_DIRNAME = "ops"

_CATALOG_IMPORT = re.compile(
    r"tensorrt_llm\._torch\._experimental\.modeling_v2\.catalog\.([a-z_]+)\.([a-z0-9_]+)"
)


#: The survey's product, in the run's workspace.
VOCABULARY_FILENAME = "vocabulary.yaml"

STATE_OK = "OK"
STATE_CERT = "CERT"
STATE_GAP = "GAP"
_VALID_STATES = (STATE_OK, STATE_CERT, STATE_GAP)

MODE_ONBOARD = "onboard"
MODE_VERIFY = "verify"
#: A queue row's modes. ``promote`` is a task's (:func:`queue_item_from_ref`):
#: it merges local variants that exist only once rows have landed.
_VALID_MODES = (MODE_ONBOARD, MODE_VERIFY)
MODE_PROMOTE = "promote"

#: A multi-rank entry's collected test, ``test_modeling_v2_<name>_op_matrix.py``,
#: only starts its ranks; ``_<name>_op_matrix.py`` beside it drives the cells.
OP_MATRIX_SUFFIX = "_op_matrix"
#: Separates a shared entry's name from the site of its local variant,
#: ``ops/<category>/<name>__<site>``. No entry name contains it.
VARIANT_SEPARATOR = "__"
_SITE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
#: A kernel task named by ``<mode>:<entry>`` rather than a queue row numbers its
#: dispatch from here, above any row, so its decision file never collides with one.
REF_DISPATCH_BASE, REF_DISPATCH_SPAN = 10_000, 90_000

# Where a GAP's implementation is allowed to come from, cheapest first.
TIER_TRTLLM_OP = 1  # torch.ops.trtllm.* -- an op the engine already exports
TIER_INTREE_CALLABLE = 2  # a callable already in this tree -- the kernel plane, reused
TIER_KERNEL_LIBRARY = 3  # flashinfer.* -- a pinned kernel library, called directly
TIER_NATIVE_TORCH = 4  # composed from torch ops, written by this run
TIER_TRITON = 5  # a Triton kernel, written by this run
_VALID_TIERS = (
    TIER_TRTLLM_OP,
    TIER_INTREE_CALLABLE,
    TIER_KERNEL_LIBRARY,
    TIER_NATIVE_TORCH,
    TIER_TRITON,
)

#: The packages the kernel-library rung may call: kernel libraries the engine
#: already pins in ``requirements.txt``.
KERNEL_LIBRARIES: tuple[str, ...] = ("flashinfer",)

#: What each rung means to the agent that has to build it.
TIER_DESCRIPTION: dict[int, str] = {
    TIER_TRTLLM_OP: "an op the engine already exports as torch.ops.trtllm.*",
    TIER_INTREE_CALLABLE: (
        "a callable already in this repository: the engine's modules, its attention "
        "backends (sparse ones included), custom ops and kernels (`tree_map.md` in the "
        "workspace says where each is in this checkout), or another modeling_v2 "
        "target's `ops/` (reused by promotion into the catalog); the kernel plane is "
        "reused, not rewritten"
    ),
    TIER_KERNEL_LIBRARY: (
        "a kernel from a library the engine already pins (FlashInfer), called through "
        "its public Python API because no trtllm op wraps it; the kernel is reused, "
        "not rewritten. Read its API in the run's interpreter"
    ),
    TIER_NATIVE_TORCH: (
        "no engine implementation exists, so compose one from torch ops. Correct "
        "first and slow is acceptable for a functionality onboarding; the tuning "
        "phase replaces it"
    ),
    TIER_TRITON: (
        "native torch cannot express it, or does so at a cost that breaks the "
        "gate's budget, so write a Triton kernel"
    ),
}

#: A tier above this is work this run *writes* rather than *wraps*.
HIGHEST_REUSING_TIER = TIER_KERNEL_LIBRARY

VERDICT_ASSEMBLABLE = "assemblable"
VERDICT_AFTER_ADDITIONS = "assemblable_after_additions"
VERDICT_CEILING = "ceiling"
_VALID_VERDICTS = (VERDICT_ASSEMBLABLE, VERDICT_AFTER_ADDITIONS, VERDICT_CEILING)

#: The unit a queue row belongs to when it names none.
UNIT_CORE = "core"

#: Cells one queue row may list. The kernel writer certifies every cell its row lists, so a
#: row covers each axis of the envelope (dtype, rows per captured batch, per-rank sizes, the
#: prefill chunk), not their product.
MAX_ROW_CELLS = 64


class VocabularyError(ValueError):
    """``vocabulary.yaml`` is malformed or internally inconsistent."""


def split_variant(name: str) -> tuple[str, str | None]:
    """``thop_attention__r7`` -> ``("thop_attention", "r7")``; a plain entry name -> ``(name, None)``."""
    entry, sep, site = name.partition(VARIANT_SEPARATOR)
    return (entry, site) if sep and entry and site else (name, None)


def variant_target(target: str, site: str) -> str:
    """The local variant of the shared entry ``<category>/<name>`` at ``site``: ``ops/<category>/<name>__<site>``."""
    category, name = target.strip().strip("/").split("/")
    return f"{MODEL_OPS_DIRNAME}/{category}/{name}{VARIANT_SEPARATOR}{site}"


def _is_op_matrix(tests_dir: str, name: str, repo_root: Path | str | None) -> bool:
    if repo_root is None:
        return False
    folder = Path(repo_root) / tests_dir
    return (
        not (folder / f"test_modeling_v2_{name}.py").is_file()
        and (folder / f"test_modeling_v2_{name}{OP_MATRIX_SUFFIX}.py").is_file()
    )


def _test_files(tests_dir: str, name: str, op_matrix: bool) -> tuple[str, ...]:
    if not op_matrix:
        return (f"{tests_dir}/test_modeling_v2_{name}.py",)
    return (
        f"{tests_dir}/test_modeling_v2_{name}{OP_MATRIX_SUFFIX}.py",
        f"{tests_dir}/_{name}{OP_MATRIX_SUFFIX}.py",
    )


def shared_entry_paths(target: str, repo_root: Path | str | None = None) -> tuple[str, ...]:
    """A shared entry ``<category>/<name>``'s module and its GPU test files, repo-relative."""
    category, name = target.strip().strip("/").split("/")
    tests = f"{CATALOG_TESTS_DIR}/{category}"
    return (
        f"{CATALOG_DIR}/{category}/{name}.py",
        *_test_files(tests, name, _is_op_matrix(tests, name, repo_root)),
    )


def _is_shared(target: str) -> bool:
    parts = target.strip().strip("/").split("/")
    return len(parts) == 2 and all(parts) and parts[0] != MODEL_OPS_DIRNAME


def site_slug(text: str) -> str:
    """``text`` as a variant site: lowercase words joined by single underscores; may be empty."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@dataclass(frozen=True)
class QueueItem:
    """One kernel writer dispatch: exactly one catalog entry, in one mode."""

    n: int
    mode: str
    #: ``<category>/<name>`` for the shared catalog, ``ops/<category>/<name>``
    #: for an entry only this run's target calls, kept in its own ``ops/``.
    target: str
    depends_on: tuple[int, ...] = ()
    surface: str | None = None
    note: str = ""
    #: Which rung of the sourcing ladder supplies this entry.
    tier: int = TIER_TRTLLM_OP
    #: The first assembly unit that needs this entry. It decides what a
    #: failed dispatch blocks, not when the dispatch runs
    #: (`run/orchestrator._add_kernel_tasks`).
    unit: str = UNIT_CORE
    #: The call site a shared entry's local variant is named for; ``r<n>``
    #: when the row names none (:attr:`variant`).
    site: str | None = None

    @property
    def entries(self) -> tuple[str, ...]:
        """The shared entries a ``promote`` merges, ``<category>/<name>`` each; the target otherwise."""
        if self.mode != MODE_PROMOTE:
            return (self.target,)
        return tuple(part.strip() for part in self.target.split(",") if part.strip())

    @property
    def variant(self) -> str | None:
        """Where an onboard or verify of a shared entry lands: ``ops/<category>/<name>__<site>``.

        ``None`` for an entry of the target's own ``ops/``, which lands in
        place, and for a ``promote``.
        """
        if self.mode == MODE_PROMOTE or not _is_shared(self.target):
            return None
        return variant_target(self.target, self.site or f"r{self.n}")

    @property
    def writes_implementation(self) -> bool:
        """True when this dispatch must *write* the kernel, not wrap one.

        Tiers 1-3 wrap something that exists; 4-5 do not.
        """
        return self.tier > HIGHEST_REUSING_TIER


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VocabularyError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _check_steps(raw: Any, errors: list[str]) -> None:
    if raw is None:
        return
    if not isinstance(raw, list):
        errors.append(f"`steps` must be a list, got {type(raw).__name__}")
        return
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            errors.append(f"steps[{index}] must be a mapping, got {type(row).__name__}")
            continue
        state = row.get("state")
        if state not in _VALID_STATES:
            errors.append(f"steps[{index}].state is {state!r}; expected one of {_VALID_STATES}")
        if not str(row.get("step", "")).strip():
            errors.append(f"steps[{index}].step must name the computation step")
        if state in (STATE_OK, STATE_CERT):
            if not str(row.get("entry", "")).strip():
                errors.append(f"steps[{index}] is {state} but names no catalog `entry`")
            if not str(row.get("contract_line", "")).strip():
                errors.append(
                    f"steps[{index}] is {state} but quotes no `contract_line`; the line "
                    f"that decides coverage is the evidence for this row"
                )
        elif state == STATE_GAP:
            tier = row.get("tier")
            if tier is None:
                errors.append(
                    f"steps[{index}] is GAP but names no `tier`. Say which rung of the "
                    f"sourcing ladder supplies it: {_VALID_TIERS} — see TIER_DESCRIPTION. "
                    f"'No trtllm op exists' is a statement about tier 1, not a ceiling."
                )
            elif tier not in _VALID_TIERS:
                errors.append(f"steps[{index}].tier is {tier!r}; expected one of {_VALID_TIERS}")
            candidate = str(row.get("candidate", "")).strip()
            if tier in (TIER_TRTLLM_OP, TIER_INTREE_CALLABLE, TIER_KERNEL_LIBRARY):
                if not candidate:
                    errors.append(
                        f"steps[{index}] is GAP at tier {tier} but names no `candidate`; "
                        f"tiers 1-3 wrap something that already exists, so it must be named"
                    )
                elif (
                    tier == TIER_KERNEL_LIBRARY and candidate.split(".")[0] not in KERNEL_LIBRARIES
                ):
                    errors.append(
                        f"steps[{index}] is GAP at tier {tier} with candidate {candidate!r}, "
                        f"which is not in a kernel library the engine pins "
                        f"({', '.join(KERNEL_LIBRARIES)}). Name the library callable "
                        f"(`flashinfer.<module>.<name>`); a trtllm op is tier 1 and an "
                        f"in-tree callable tier 2"
                    )
            elif tier in (TIER_NATIVE_TORCH, TIER_TRITON):
                if not str(row.get("tried", "")).strip():
                    errors.append(
                        f"steps[{index}] is GAP at tier {tier}, which means this run has "
                        f"to *write* the implementation. Record in `tried` what was "
                        f"searched at the cheaper rungs and why it was rejected — "
                        f"otherwise the ladder is a way to skip straight to writing a "
                        f"kernel without looking for one"
                    )


def _parse_unit(
    row: Mapping[str, Any], index: int, units: Sequence[str] | None, errors: list[str]
) -> str:
    """The row's ``unit``, defaulting to core; checked against the run's units when known."""
    raw = row.get("unit")
    if raw is None:
        return UNIT_CORE
    if not isinstance(raw, str) or not raw.strip():
        errors.append(
            f"queue[{index}].unit is {raw!r}; it must name the one assembly unit that "
            f"first needs this entry (`{UNIT_CORE}`, the default, or a feature from "
            f"task.yaml `features`)"
        )
        return UNIT_CORE
    unit = raw.strip()
    if units is not None and unit not in units:
        errors.append(
            f"queue[{index}].unit is {unit!r}, which is not one of this run's assembly "
            f"units {list(units)}. Tag a row only with the first unit that needs its "
            f"entry; leave it off (core) when core needs it or you are unsure"
        )
        return UNIT_CORE
    return unit


def _parse_site(row: Mapping[str, Any], index: int, target: str, errors: list[str]) -> str | None:
    """The row's ``site``: lowercase words joined by single underscores, on a shared entry only."""
    raw = row.get("site")
    if raw is None:
        return None
    site = str(raw).strip()
    if not _SITE.match(site):
        errors.append(
            f"queue[{index}].site is {raw!r}; a site is lowercase words joined by single "
            f"underscores (`mtp_verify`), because it ends the variant's module name"
        )
        return None
    if not _is_shared(target):
        errors.append(
            f"queue[{index}].site names a local variant, which only a shared entry has; "
            f"{target!r} lands in place"
        )
        return None
    return site


def _parse_queue(
    raw: Any, errors: list[str], units: Sequence[str] | None = None
) -> tuple[QueueItem, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        errors.append(f"`queue` must be a list, got {type(raw).__name__}")
        return ()

    items: list[QueueItem] = []
    seen_n: set[int] = set()
    seen_targets: dict[str, str] = {}
    seen_variants: dict[str, int] = {}
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            errors.append(f"queue[{index}] must be a mapping, got {type(row).__name__}")
            continue
        try:
            n = int(row["n"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"queue[{index}].n must be an integer dispatch number")
            continue
        if n in seen_n:
            errors.append(f"queue[{index}].n = {n} is a duplicate dispatch number")
        seen_n.add(n)

        mode = row.get("mode")
        if mode not in _VALID_MODES:
            errors.append(f"queue[{index}].mode is {mode!r}; expected one of {_VALID_MODES}")

        target = str(row.get("target", "")).strip()
        if VARIANT_SEPARATOR in target.rsplit("/", 1)[-1]:
            errors.append(
                f"queue[{index}].target {target!r} has `{VARIANT_SEPARATOR}` in its name, which "
                f"marks a local variant. Name the entry itself; its row's dispatch lands the "
                f"variant"
            )
        if not target:
            errors.append(f"queue[{index}].target must name exactly one catalog entry")
        elif target in seen_targets and seen_targets[target] != mode:
            errors.append(
                f"queue[{index}] has {target!r} in both chapters "
                f"({seen_targets[target]} and {mode}). An entry onboarded here is "
                f"tested at this checkpoint's values by construction, so it never "
                f"also needs verifying"
            )
        elif target in seen_targets and mode != MODE_PROMOTE and not target.startswith("ops/"):
            errors.append(
                f"queue[{index}] is a second {mode} row on {target!r}. Write one row per "
                f"shared entry: every call site's shapes as its `cells` (and the sites in "
                f"its `note`), so one kernel writer certifies them all and nothing is merged later"
            )
        elif target:
            seen_targets[target] = str(mode)

        depends_raw = row.get("depends_on") or []
        if not isinstance(depends_raw, list):
            errors.append(f"queue[{index}].depends_on must be a list of dispatch numbers")
            depends_raw = []
        depends: list[int] = []
        for dep in depends_raw:
            try:
                depends.append(int(dep))
            except (TypeError, ValueError):
                errors.append(f"queue[{index}].depends_on holds {dep!r}, which is not a number")

        if mode == MODE_VERIFY and not str(row.get("surface", "")).strip():
            errors.append(
                f"queue[{index}] is a verify dispatch with no `surface`. The value and "
                f"the contract line that excludes it are what the kernel writer is dispatched "
                f"with, so vagueness here costs a whole run"
            )
        cells = row.get("cells")
        if isinstance(cells, list) and len(cells) > MAX_ROW_CELLS:
            errors.append(
                f"queue[{index}] lists {len(cells)} cells, more than {MAX_ROW_CELLS}. The kernel "
                f"writer certifies each one: cover each axis (every dtype, each call site's rows "
                f"at the smallest and largest captured batch, each member's per-rank sizes, one "
                f"chunk at max_num_tokens), varying one at a time from a base cell, not their product"
            )

        site = _parse_site(row, index, target, errors)
        item = QueueItem(
            n=n,
            mode=str(mode),
            target=target,
            depends_on=tuple(depends),
            tier=int(row.get("tier", TIER_TRTLLM_OP) or TIER_TRTLLM_OP),
            surface=row.get("surface"),
            note=str(row.get("note", "")),
            unit=_parse_unit(row, index, units, errors),
            site=site,
        )
        if item.variant is not None and item.variant in seen_variants:
            errors.append(
                f"queue[{index}] lands at {item.variant}, as row {seen_variants[item.variant]} "
                f"does. Give each row on one entry its own `site`: two rows on one variant "
                f"share its file and run one after the other"
            )
        elif item.variant is not None:
            seen_variants[item.variant] = n
        items.append(item)

    known = {item.n for item in items}
    for item in items:
        for dep in item.depends_on:
            if dep not in known:
                errors.append(f"queue item {item.n} depends on {dep}, which is not in the queue")
    return tuple(items)


def parse_vocabulary(
    data: Mapping[str, Any],
    source: str = "vocabulary.yaml",
    *,
    units: Sequence[str] | None = None,
) -> tuple[QueueItem, ...]:
    """Validate a parsed mapping, batching every problem into one error, and return its queue.

    ``units`` is the run's assembly units (``task_schema.assembly_units``).
    When given, every queue row's ``unit`` must be one of them; when not,
    only its shape is checked and the orchestrator checks membership where
    it knows the spec.
    """
    data = _require_mapping(data, source)
    errors: list[str] = []

    verdict = data.get("verdict")
    if verdict not in _VALID_VERDICTS:
        errors.append(f"`verdict` is {verdict!r}; expected one of {_VALID_VERDICTS}")

    _check_steps(data.get("steps"), errors)
    queue = _parse_queue(data.get("queue"), errors, units)

    ceiling = data.get("ceiling")
    if verdict == VERDICT_CEILING and isinstance(ceiling, dict):
        exhausted = ceiling.get("exhausted")
        if not isinstance(exhausted, dict):
            errors.append(
                "`ceiling` must carry an `exhausted` mapping with one entry per tier "
                f"{_VALID_TIERS}, saying what was searched and why that rung failed. "
                f"A ceiling that has not considered tier {TIER_KERNEL_LIBRARY} (FlashInfer's "
                f"API) and tiers {TIER_NATIVE_TORCH}-{TIER_TRITON} is a report that nobody "
                "looked outside the engine or wrote the kernel yet, which is not the same thing."
            )
        else:
            missing = [t for t in _VALID_TIERS if not str(exhausted.get(t, "")).strip()]
            if missing:
                errors.append(
                    f"`ceiling.exhausted` says nothing about tier(s) {missing}. "
                    f"Tier {TIER_KERNEL_LIBRARY} is FlashInfer's public API, which the "
                    f"op inventory cannot see; tiers {TIER_NATIVE_TORCH}-{TIER_TRITON} are "
                    f"this run writing the implementation itself. Declaring a ceiling "
                    f"without ruling them out claims more than was established."
                )

    if verdict == VERDICT_CEILING:
        if not isinstance(ceiling, dict) or not str(ceiling.get("step", "")).strip():
            errors.append(
                "`verdict: ceiling` requires a `ceiling` mapping naming the exact "
                "`step` and what was `searched`. A ceiling is the one verdict that "
                "ends the run, so it carries its evidence or it is not a verdict"
            )
    elif ceiling not in (None, {}):
        errors.append(f"`ceiling` is set but the verdict is {verdict!r}; they must agree")

    if verdict == VERDICT_ASSEMBLABLE and queue:
        errors.append(
            f"`verdict: assemblable` means nothing has to be added first, but the "
            f"queue holds {len(queue)} dispatch(es). Use "
            f"`{VERDICT_AFTER_ADDITIONS}` instead"
        )
    if verdict == VERDICT_AFTER_ADDITIONS and not queue:
        errors.append(
            f"`verdict: {VERDICT_AFTER_ADDITIONS}` with an empty queue says additions "
            f"are needed but names none. Use `{VERDICT_ASSEMBLABLE}` instead"
        )

    if errors:
        bullet = "\n  - "
        raise VocabularyError(f"{source} is not usable:{bullet}{bullet.join(errors)}")

    return queue


def load_vocabulary(
    path: str | Path, *, units: Sequence[str] | None = None
) -> tuple[QueueItem, ...]:
    """Read and validate ``vocabulary.yaml`` from disk. See :func:`parse_vocabulary`."""
    source = Path(path)
    if not source.is_file():
        raise VocabularyError(f"{source} does not exist; the plan phase must produce it.")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise VocabularyError(f"{source} is empty; the plan phase must produce it.")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise VocabularyError(f"{source} is not valid YAML: {exc}") from exc
    return parse_vocabulary(data or {}, str(source), units=units)


def _own_paths(
    target: str, target_rel: str, repo_root: Path | str | None = None, op_matrix: bool = False
) -> tuple[str, ...] | None:
    """``ops/<category>/<name>``'s module and test files under the target ``target_rel``."""
    parts = target.strip().strip("/").split("/")
    if len(parts) != 3 or parts[0] != MODEL_OPS_DIRNAME or not all(parts):
        return None
    if not target_rel.strip():
        return None
    category, name = parts[1:]
    tests = f"{model_ops_tests_dir(target_rel.strip())}/{category}"
    return (
        f"{target_rel.strip().rstrip('/')}/{MODEL_OPS_DIRNAME}/{category}/{name}.py",
        *_test_files(tests, name, op_matrix or _is_op_matrix(tests, name, repo_root)),
    )


def entry_paths(
    item: QueueItem, target_rel: str = "", repo_root: Path | str | None = None
) -> tuple[str, ...] | None:
    """Where the row's dispatch writes: its module, then its GPU test files, repo-relative.

    A row on a shared entry (``<category>/<name>``) writes its local variant
    (:attr:`QueueItem.variant`) in the target's own ``ops/`` under
    ``target_rel``, tested where the target's tests mirror it; with no target
    it has no local home and writes the catalog's entry in place. A row on an
    entry only the target calls (``ops/<category>/<name>``) writes it there,
    and without the target has no place. ``None`` when the row cannot be
    placed, and for a ``promote``, which spans several entries.

    The test is ``test_modeling_v2_<name>.py``. With ``repo_root``, an entry
    whose tests are a multi-rank op matrix there (a collective's) is tested by
    ``test_modeling_v2_<name>_op_matrix.py`` and the rank body
    ``_<name>_op_matrix.py``, and so is its local variant.
    """
    if item.mode == MODE_PROMOTE:
        return None
    if _is_shared(item.target):
        if item.variant is not None and target_rel.strip():
            category, name = item.target.strip().strip("/").split("/")
            matrix = _is_op_matrix(f"{CATALOG_TESTS_DIR}/{category}", name, repo_root)
            return _own_paths(item.variant, target_rel, repo_root, op_matrix=matrix)
        return shared_entry_paths(item.target, repo_root)
    return _own_paths(item.target, target_rel, repo_root)


def queue_item_from_ref(ref: str, task_id: str, goal: str = "") -> QueueItem | None:
    """The dispatch a kernel task names by ``ref`` instead of a queue row; ``None`` if it names none.

    ``onboard:<entry>`` and ``verify:<entry>`` name one entry; after a shared
    one, ``@<site>`` names the call site its local variant is for, else the
    task id does. ``promote:<entry>[,<entry>...]`` names the shared entries
    whose local variants it merges into the catalog. The dispatch number is
    the task id's, above every queue row's, so a retry finds its decision file.
    """
    mode, _, rest = ref.strip().partition(":")
    target, _, site = rest.strip().partition("@")
    target, site = target.strip(), site.strip()
    n = REF_DISPATCH_BASE + zlib.crc32(task_id.encode("utf-8")) % REF_DISPATCH_SPAN
    if mode == MODE_PROMOTE:
        entries = [part.strip().strip("/") for part in target.split(",") if part.strip()]
        if not entries or site or not all(_is_shared(e) for e in entries):
            return None
        return QueueItem(n=n, mode=mode, target=",".join(entries), note=goal)
    if mode not in _VALID_MODES or not target:
        return None
    if site and not (_SITE.match(site) and _is_shared(target)):
        return None
    if _is_shared(target) and not site:
        site = site_slug(task_id) or None
    return QueueItem(n=n, mode=mode, target=target, note=goal, site=site or None)


def imported_catalog_entries(modeling_py: Path) -> list[tuple[str, str]]:
    """Return ``(category, name)`` for every catalog entry a target imports."""
    tree = ast.parse(modeling_py.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                match = _CATALOG_IMPORT.search(alias.name)
                if match:
                    found.append((match.group(1), match.group(2)))
            continue
        if module:
            match = _CATALOG_IMPORT.search(module)
            if match:
                found.append((match.group(1), match.group(2)))
    return sorted(set(found))


def model_ops_tests_dir(target_rel: str) -> str:
    """``.../modeling_v2/models/<family>/<target>`` -> its tests' directory."""
    tail = target_rel.rstrip("/").split(f"{MODELING_V2_PACKAGE}/", 1)[-1]
    return f"{CATALOG_TESTS_DIR}/{tail}"


def entry_module(target: str) -> str:
    """The tail every path of ``target``'s entry module ends in.

    ``target`` is a dispatch's ``<category>/<name>`` (the shared catalog) or
    ``ops/<category>/<name>`` (the run's target's own); see the entry spec.
    """
    parts = target.strip("/").split("/")
    home = "ops" if parts[0] == "ops" and len(parts) > 2 else "catalog"
    return f"{home}/{'/'.join(parts[-2:])}.py"


def is_entry_file(path: str, target: str) -> bool:
    """Whether the repo-relative ``path`` is one of the files ``target``'s entry spans.

    The entry module, its GPU test (with a collective's rank body beside it),
    and the ``__init__.py`` of a new ``ops/<category>/``: what a kernel writer writes
    for one entry, in both trees.
    """
    parts = target.strip("/").split("/")
    if len(parts) < 2:
        return False
    category, name = parts[-2], parts[-1]
    file = PurePosixPath(path)
    if path.startswith("tensorrt_llm/"):
        if path.endswith("/" + entry_module(target)):
            return True
        return (
            parts[0] == "ops"
            and file.name == "__init__.py"
            and path.endswith(f"/ops/{category}/__init__.py")
        )
    if path.startswith("tests/") and file.parent.name == category:
        stem = file.stem
        return stem == f"test_modeling_v2_{name}" or stem.startswith(
            (f"test_modeling_v2_{name}_", f"_{name}_")
        )
    return False


__all__ = [
    "is_entry_file",
    "entry_module",
    "imported_catalog_entries",
    "model_ops_tests_dir",
    "MODEL_OPS_DIRNAME",
    "MODELING_V2_PACKAGE",
    "CATALOG_TESTS_DIR",
    "CATALOG_DIR",
    "MODE_ONBOARD",
    "MODE_PROMOTE",
    "MODE_VERIFY",
    "OP_MATRIX_SUFFIX",
    "QueueItem",
    "VARIANT_SEPARATOR",
    "STATE_CERT",
    "STATE_GAP",
    "STATE_OK",
    "UNIT_CORE",
    "VERDICT_AFTER_ADDITIONS",
    "VERDICT_ASSEMBLABLE",
    "VERDICT_CEILING",
    "VOCABULARY_FILENAME",
    "VocabularyError",
    "entry_paths",
    "load_vocabulary",
    "parse_vocabulary",
    "queue_item_from_ref",
    "shared_entry_paths",
    "site_slug",
    "split_variant",
    "variant_target",
]
