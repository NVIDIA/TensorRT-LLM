# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``vocabulary.yaml``, the survey's product, as far as code reads it.

Code makes a run's tasks from the survey with no planner, so it reads of each piece of
work what that takes and nothing more: which entry a kernel row writes or certifies
(``mode``, ``target``, and ``site`` for the local variant of a shared entry), what the row
waits for (``depends_on``) and which deployment first needs it (``deployment``). The
other rows are :mod:`.rows`. Everything else a surveyor writes into the file (its map of
the model's steps, its reasons, its cells) is its own to word: it reaches the row's
worker on its card and is held to no form here.

This module also says where an entry's files are: the one fact about the tree the
workflow keeps, for the package it adds a target to (:data:`SUBJECT_PATHS`).
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

#: The workflow's subject: the ModelingV2 package a run adds its target to, and the
#: directory the package's tests mirror it in. These two are where a run **writes**, and
#: they are the only directories of the engine this workflow names. Everything it reads
#: off the engine is found by what a file defines (``operators/tree_anchors.py``), and
#: nothing a worker is told names a place in the engine.
MODELING_V2_PACKAGE = "tensorrt_llm/_torch/_experimental/modeling_v2"
CATALOG_TESTS_DIR = "tests/unittest/_torch/modeling_v2"
SUBJECT_PATHS: tuple[str, ...] = (MODELING_V2_PACKAGE, CATALOG_TESTS_DIR)
CATALOG_DIR = f"{MODELING_V2_PACKAGE}/catalog"
#: A target's own entries -- ops only it calls -- sit in this directory of the
#: target, one ``<category>/<name>.py`` each, and their tests mirror that path
#: under ``CATALOG_TESTS_DIR/models/<family>/<target>/``.
MODEL_OPS_DIRNAME = "ops"


#: The survey's product, in the run's workspace.
VOCABULARY_FILENAME = "vocabulary.yaml"

MODE_ONBOARD = "onboard"
MODE_VERIFY = "verify"
#: A queue row's modes, and a kernel task's (:func:`queue_item_from_ref`).
_VALID_MODES = (MODE_ONBOARD, MODE_VERIFY)

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

#: The unit a queue row belongs to when it names none.
UNIT_CORE = "core"


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
    #: The first assembly unit that needs this entry. It decides what a
    #: failed dispatch blocks, not when the dispatch runs
    #: (`run/orchestrator._add_kernel_tasks`).
    unit: str = UNIT_CORE
    #: The call site a shared entry's local variant is named for; ``r<n>``
    #: when the row names none (:attr:`variant`).
    site: str | None = None

    @property
    def variant(self) -> str | None:
        """Where an onboard or verify of a shared entry lands: ``ops/<category>/<name>__<site>``.

        ``None`` for an entry of the target's own ``ops/``, which lands in
        place.
        """
        if not _is_shared(self.target):
            return None
        return variant_target(self.target, self.site or f"r{self.n}")


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VocabularyError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _parse_unit(
    row: Mapping[str, Any], index: int, units: Sequence[str] | None, errors: list[str]
) -> str:
    """The deployment that first needs the row's entry; the first one's when the row names none.

    Read from the row's ``deployment`` and checked against the spec's deployments when
    they are known.
    """
    raw = row.get("deployment")
    if raw is None:
        return UNIT_CORE
    if not isinstance(raw, str) or not raw.strip():
        errors.append(
            f"queue[{index}].deployment is {raw!r}; it must be the name of the one deployment "
            f"that first needs this entry, as `task.yaml`'s `deployments` name it. Leave the "
            f"key off when the first deployment needs the entry"
        )
        return UNIT_CORE
    unit = raw.strip()
    if units is not None and unit not in units:
        errors.append(
            f"queue[{index}].deployment is {unit!r}, which is not one of the spec's "
            f"deployments {list(units)}. Name only the deployment that first needs the "
            f"entry; leave the key off when the first deployment needs it or you are unsure"
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
    seen_own: set[str] = set()
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
        elif target in seen_own:
            errors.append(
                f"queue[{index}] is a second row on {target!r}, an entry of the target's own: "
                f"two rows would write the same file. Make them one row"
            )
        elif not _is_shared(target):
            seen_own.add(target)

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

        site = _parse_site(row, index, target, errors)
        item = QueueItem(
            n=n,
            mode=str(mode),
            target=target,
            depends_on=tuple(depends),
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
    """The queue of a parsed vocabulary: what code needs of each kernel row, every problem at once.

    ``units`` is the run's assembly units (``task_schema.assembly_units``).
    When given, every queue row's ``unit`` must be one of them; when not,
    only its shape is checked and the orchestrator checks membership where
    it knows the spec.
    """
    data = _require_mapping(data, source)
    errors: list[str] = []
    queue = _parse_queue(data.get("queue"), errors, units)

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
    placed.

    The test is ``test_modeling_v2_<name>.py``. With ``repo_root``, an entry
    whose tests are a multi-rank op matrix there (a collective's) is tested by
    ``test_modeling_v2_<name>_op_matrix.py`` and the rank body
    ``_<name>_op_matrix.py``, and so is its local variant.
    """
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
    task id does. The dispatch number is
    the task id's, above every queue row's, so a retry finds its decision file.
    """
    mode, _, rest = ref.strip().partition(":")
    target, _, site = rest.strip().partition("@")
    target, site = target.strip(), site.strip()
    n = REF_DISPATCH_BASE + zlib.crc32(task_id.encode("utf-8")) % REF_DISPATCH_SPAN
    if mode not in _VALID_MODES or not target:
        return None
    if site and not (_SITE.match(site) and _is_shared(target)):
        return None
    if _is_shared(target) and not site:
        site = site_slug(task_id) or None
    return QueueItem(n=n, mode=mode, target=target, note=goal, site=site or None)


def model_ops_tests_dir(target_rel: str) -> str:
    """``.../modeling_v2/models/<family>/<target>`` -> its tests' directory."""
    tail = target_rel.rstrip("/").split(f"{MODELING_V2_PACKAGE}/", 1)[-1]
    return f"{CATALOG_TESTS_DIR}/{tail}"


__all__ = [
    "model_ops_tests_dir",
    "MODEL_OPS_DIRNAME",
    "MODELING_V2_PACKAGE",
    "SUBJECT_PATHS",
    "CATALOG_TESTS_DIR",
    "CATALOG_DIR",
    "MODE_ONBOARD",
    "MODE_VERIFY",
    "OP_MATRIX_SUFFIX",
    "QueueItem",
    "VARIANT_SEPARATOR",
    "UNIT_CORE",
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
