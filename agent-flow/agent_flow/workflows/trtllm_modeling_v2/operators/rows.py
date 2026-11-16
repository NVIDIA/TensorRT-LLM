# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Work rows: what the survey asks for, in the form a task and its card are made from.

The survey writes ``vocabulary.yaml``. Its ``queue`` rows are the ``operator`` rows
(``operators/coverage.py``); its ``rows`` are every other piece of work the target
needs, each of kind ``weights``, ``routing``, ``state`` or ``feature``.

Code makes tasks from rows with no planner (``run/orchestrator.py``), and reads of a row
what that takes: its ``id`` and ``kind``, what it ``writes``, what it waits for
(``depends_on``), which ``deployment`` first needs it, whether its check ``needs`` the
assembled model, and its ``check``'s job. The rest is the surveyor's to word and is held
to no form: what the row is for (``outcome``), what must hold (``required``), what may be
left (``optional``), and what to look for (``leads``). A card shows it as written
(``agents/prompts/card.py``).

The rows code adds itself for a deployment (its assembly, the sides of a disaggregated
one) are :func:`deployment_row`. Every row a task works on is kept in ``work.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.files import write_text_atomic
from . import coverage as coverage_mod

OPERATOR, WEIGHTS, ROUTING, STATE, FEATURE = "operator", "weights", "routing", "state", "feature"
#: The kinds the survey writes. An ``operator`` row is a ``queue`` row; the rest are ``rows``.
ROW_KINDS: tuple[str, ...] = (OPERATOR, WEIGHTS, ROUTING, STATE, FEATURE)
#: A row code adds for a deployment: its assembly, or one side of a disaggregated one.
ASSEMBLE = "assemble"
#: Every row's place in the run's workspace, as tasks and cards read them.
WORK_FILENAME = "work.yaml"
#: The registry name of a row's check, by its kind: what ``gate <name> --row`` runs. Every
#: other kind is judged by its own test (:data:`CPU_CHECK`).
CHECK_NAME = {OPERATOR: "kernel"}
#: The check of a row that needs no device unless it says so.
CPU_CHECK = "cpu"
DEPLOYMENT_CHECK = "deployment"
#: The row code writes for what the delivery requires of the target's files and no part
#: of the model is: the accuracy test for CI with the test list that names it (of a run
#: that scores a benchmark), and the repair of every finding of form a task's landing
#: recorded. It runs after the first deployment's session: nothing of it stands before
#: ``onboarded``. It names no check: no job proves it. Its proof is the delivery's own
#: rules for its files, asked when its task hands in (``ExecutionStage._file_rules``).
DELIVERY_ROW = "delivery-files"
#: A row's test jobs, unless it states its own: one device, this long.
DEFAULT_WALL_S = 1800
#: What a row's check may need beyond the row's own work (``Row.needs``): the assembled
#: model of the row's deployment. Such a row comes after that assembly.
NEEDS_MODEL = "model"
NEEDS: tuple[str, ...] = (NEEDS_MODEL,)
_ID = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")


class RowError(ValueError):
    """The typed rows of ``vocabulary.yaml`` are malformed."""


@dataclass(frozen=True)
class Row:
    """One piece of work: what a task is made from and what its card says."""

    id: str
    kind: str
    outcome: str
    #: The deployment that first needs it; empty for the target itself (the first one's).
    deployment: str = ""
    depends_on: tuple[str, ...] = ()
    config_axes: Mapping[str, Any] = field(default_factory=dict)
    #: What to look for, as the surveyor worded it: names to search, questions, why each
    #: may fit. Never a place.
    leads: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    #: The files and directories the row owns, repository-relative; an operator row's are
    #: its entry and test (``coverage.entry_paths``), placed by code.
    writes: tuple[str, ...] = ()
    #: The registry id of its one check.
    check: str = ""
    #: Its check's job line: devices, or nodes when it needs whole ones, and its wall.
    gpus: int = 0
    nodes: int = 0
    wall: int = DEFAULT_WALL_S
    #: Which side of a disaggregated deployment an ``assemble`` row is: ``ctx`` or ``gen``.
    side: str = ""
    #: What its check needs that the row does not make: ``model`` when the check builds or
    #: serves the assembled model of its deployment (the first one's, when it names none).
    #: Empty for a check that needs nothing but the row's own files.
    needs: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["config_axes"] = dict(self.config_axes)
        return {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in out.items()
            if v or k in ("id", "kind", "outcome")
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Row":
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            outcome=str(data.get("outcome", "")),
            deployment=str(data.get("deployment") or ""),
            depends_on=_strings(data.get("depends_on")),
            config_axes=dict(data.get("config_axes") or {}),
            leads=_leads(data),
            required=_strings(data.get("required")),
            optional=_strings(data.get("optional")),
            writes=_strings(data.get("writes")),
            check=str(data.get("check") or ""),
            gpus=int(data.get("gpus") or 0),
            nodes=int(data.get("nodes") or 0),
            wall=int(data.get("wall") or DEFAULT_WALL_S),
            side=str(data.get("side") or ""),
            needs=str(data.get("needs") or ""),
        )

    @property
    def job_line(self) -> str:
        """``gpus=N wall=S`` (``nodes=N`` for whole nodes): what the run plans for the check's job.

        The pool is asked for it while the row's task is due, and a kernel row's command
        states both to the front door. A deployment's job states its session's own.
        """
        size = f"nodes={self.nodes}" if self.nodes else f"gpus={self.gpus}"
        return f"{size} wall={self.wall}"


def _strings(value: Any) -> tuple[str, ...]:
    """A list of the row as lines of text: a mapping item (``- a: b``) is kept as one line."""
    if value is None:
        return ()
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    out = []
    for item in value:
        if isinstance(item, Mapping):
            item = ", ".join(f"{k}: {v}" for k, v in item.items())
        text = " ".join(str(item).split())
        if text:
            out.append(text)
    return tuple(out)


def check_id(kind: str, row_id: str) -> str:
    """The registry id of a row's check when it states none: ``kernel:<row>``, else ``cpu:<row>``."""
    return f"{CHECK_NAME.get(kind, CPU_CHECK)}:{row_id}"


def _check(
    raw: Any, kind: str, row_id: str, where: str, errors: list[str], none: bool = False
) -> dict[str, Any]:
    """A row's ``check`` block as ``{check, gpus, nodes, wall}``, defaults by kind.

    With ``none`` (a split's rows, which a person writes) an id that is stated and empty
    says the row has no check of its own: no job. A survey's row always has one.
    """
    out = {
        "check": check_id(kind, row_id),
        "gpus": 1 if kind in (OPERATOR, STATE) else 0,
        "nodes": 0,
        "wall": DEFAULT_WALL_S,
    }
    if raw is None:
        return out
    if not isinstance(raw, Mapping):
        errors.append(f"{where}.check must be a mapping (`id`, `gpus` or `nodes`, `wall`)")
        return out
    if raw.get("id"):
        out["check"] = str(raw["id"]).strip()
    elif none and "id" in raw:
        # Read as unstated, it became the kind's default, and a CPU job ran a GPU test.
        out["check"], out["gpus"] = "", 0
    for key in ("gpus", "nodes", "wall"):
        if raw.get(key) is None:
            continue
        try:
            out[key] = int(raw[key])
        except (TypeError, ValueError):
            errors.append(f"{where}.check.{key} is {raw[key]!r}; it must be a whole number")
        else:
            if out[key] < 0 or (key == "wall" and out[key] == 0):
                errors.append(f"{where}.check.{key} is {raw[key]!r}; it cannot be that small")
    if out["nodes"] and raw.get("gpus") is not None and out["gpus"]:
        errors.append(f"{where}.check states `gpus` and `nodes`: a job states one of them")
    return out


def _leads(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """What a row says to look for, under whichever key its surveyor wrote it, as one list.

    ``leads`` first, then names (``definitions``) and questions (``search_questions``): a
    survey may sort what it found, and a worker reads one list.
    """
    names = tuple(f"`{name}`" for name in _strings(raw.get("definitions")))
    return (*_strings(raw.get("leads")), *names, *_strings(raw.get("search_questions")))


def _evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    """What every row may state beside what code reads: its values, and what to look for."""
    axes = raw.get("config_axes")
    return {"config_axes": dict(axes) if isinstance(axes, Mapping) else {}, "leads": _leads(raw)}


def operator_rows(data: Mapping[str, Any], queue: Sequence[coverage_mod.QueueItem]) -> list[Row]:
    """The ``queue`` of a validated vocabulary as rows: one per entry to certify or write.

    Its id is the row's number, which is what ``gate kernel --row`` takes. ``cells`` are
    what it must certify, ``optional_cells`` what it may leave; a verify row's ``surface``
    is its first requirement. A row that states no outcome gets the one its mode means.
    """
    raw_rows = {str(r.get("n")): r for r in data.get("queue") or () if isinstance(r, Mapping)}
    rows = []
    for item in queue:
        raw = raw_rows.get(str(item.n), {})
        errors: list[str] = []
        check = _check(raw.get("check"), OPERATOR, str(item.n), f"queue[n={item.n}]", errors)
        if errors:
            raise RowError("; ".join(errors))
        verb = "certify" if item.mode == coverage_mod.MODE_VERIFY else "write"
        required = _strings(raw.get("cells")) + _strings(raw.get("required"))
        if item.surface:
            required = (f"surface: {' '.join(str(item.surface).split())}", *required)
        rows.append(
            Row(
                id=str(item.n),
                kind=OPERATOR,
                outcome=str(raw.get("outcome") or "").strip()
                or f"{verb} the entry `{item.target}` at this checkpoint's values on this "
                "architecture, with a GPU test that drives the cells it claims",
                deployment="" if item.unit == coverage_mod.UNIT_CORE else item.unit,
                depends_on=tuple(str(n) for n in item.depends_on),
                required=required,
                optional=_strings(raw.get("optional_cells")) + _strings(raw.get("optional")),
                **_evidence(raw),
                **check,
            )
        )
    return rows


def typed_rows(
    data: Mapping[str, Any],
    *,
    operators: Sequence[str] = (),
    deployments: Sequence[str] | None = None,
    source: str = "vocabulary.yaml",
) -> list[Row]:
    """The ``rows`` of a vocabulary (every kind but ``operator``), validated together.

    ``operators`` are the ids of the queue's rows, which a row may depend on;
    ``deployments`` the spec's deployments, which a row's ``deployment`` must be one of
    when they are known. Raises :class:`RowError` with every problem at once.
    """
    raw = data.get("rows")
    if raw is None:
        return []
    errors: list[str] = []
    if not isinstance(raw, list):
        raise RowError(f"{source}: `rows` must be a list, got {type(raw).__name__}")
    rows: list[Row] = []
    seen = set(operators)
    for index, entry in enumerate(raw):
        where = f"rows[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{where} must be a mapping, got {type(entry).__name__}")
            continue
        row_id = str(entry.get("id") or "").strip()
        kind = str(entry.get("kind") or "").strip()
        if not _ID.match(row_id):
            errors.append(
                f"{where}.id is {entry.get('id')!r}; a row's id is lowercase words joined by "
                "`-` or `_`, starting with a letter (a queue row is named by its number)"
            )
            continue
        if row_id in seen:
            errors.append(f"{where}.id {row_id!r} is used more than once")
        seen.add(row_id)
        if kind == OPERATOR:
            errors.append(
                f"{where} is an `operator` row: an entry to certify or write is a `queue` row"
            )
        elif kind not in ROW_KINDS:
            errors.append(f"{where}.kind is {kind!r}; expected one of {ROW_KINDS[1:]}")
        outcome = " ".join(str(entry.get("outcome") or "").split())
        if not outcome:
            errors.append(
                f"{where} states no `outcome`: what the row is for, one observable result"
            )
        deployment = str(entry.get("deployment") or "").strip()
        if deployment and deployments is not None and deployment not in deployments:
            errors.append(
                f"{where}.deployment is {deployment!r}, which is not one of the spec's "
                f"{list(deployments)}; leave it off for work the target itself needs"
            )
        check = _check(entry.get("check"), kind, row_id, where, errors)
        needs = str(entry.get("needs") or "").strip()
        if needs and needs not in NEEDS:
            errors.append(
                f"{where}.needs is {entry.get('needs')!r}; say `{NEEDS_MODEL}` when the row's "
                "check builds or serves the assembled model of its deployment, and leave it "
                "off when the check needs only the row's own files"
            )
        rows.append(
            Row(
                id=row_id,
                kind=kind,
                outcome=outcome,
                deployment=deployment,
                needs=needs,
                depends_on=_strings(entry.get("depends_on")),
                required=_strings(entry.get("required")),
                optional=_strings(entry.get("optional")),
                writes=_strings(entry.get("writes")),
                **_evidence(entry),
                **check,
            )
        )
    for row in rows:
        for dep in row.depends_on:
            if dep not in seen:
                errors.append(f"row {row.id!r} depends on {dep!r}, which is no row")
            elif dep == row.id:
                errors.append(f"row {row.id!r} depends on itself")
    if errors:
        bullet = "\n  - "
        raise RowError(f"{source}: its typed rows are not usable:{bullet}{bullet.join(errors)}")
    return rows


def load_rows(path: str | Path, *, units: Sequence[str] | None = None) -> list[Row]:
    """Every row of the vocabulary at ``path``: its queue as operator rows, then its ``rows``.

    Raises ``coverage.VocabularyError`` for a vocabulary that does not validate, and
    :class:`RowError` (one of those) for typed rows that do not.
    """
    queue = coverage_mod.load_vocabulary(path, units=units)
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        operators = operator_rows(data, queue)
        rest = typed_rows(
            data,
            operators=[row.id for row in operators],
            deployments=units,
            source=str(path),
        )
    except RowError as exc:
        raise coverage_mod.VocabularyError(str(exc)) from exc
    return [*operators, *rest]


def split_rows(data: Any, taken: Sequence[str], source: str = "rows") -> list[Row]:
    """The typed rows a split names, of any kind: what the outer session writes for one task.

    A list of rows or a mapping with ``rows``. Each has an ``id`` no task has yet, a
    ``kind``, an ``outcome`` and what is ``required``; an ``operator`` row also names its
    ``target`` (``<category>/<name>`` or ``ops/<category>/<name>``) and ``mode``.
    """
    if isinstance(data, Mapping):
        data = data.get("rows")
    if not isinstance(data, list) or not data:
        raise RowError(f"{source}: a split names its rows as a list, and names at least one")
    errors: list[str] = []
    rows: list[Row] = []
    seen = set(taken)
    for index, entry in enumerate(data):
        where = f"{source}[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{where} must be a mapping")
            continue
        row_id = str(entry.get("id") or "").strip()
        kind = str(entry.get("kind") or "").strip()
        if not _ID.match(row_id):
            errors.append(f"{where}.id is {entry.get('id')!r}; lowercase words joined by `-`")
            continue
        if row_id in seen:
            errors.append(f"{where}.id {row_id!r} is taken by a task or another row")
        seen.add(row_id)
        if kind not in ROW_KINDS:
            errors.append(f"{where}.kind is {kind!r}; expected one of {ROW_KINDS}")
        outcome = " ".join(str(entry.get("outcome") or "").split())
        if not outcome:
            errors.append(f"{where} states no `outcome`")
        writes = _strings(entry.get("writes"))
        if kind == OPERATOR:
            target = str(entry.get("target") or "").strip()
            mode = str(entry.get("mode") or coverage_mod.MODE_ONBOARD).strip()
            if coverage_mod.queue_item_from_ref(f"{mode}:{target}", row_id) is None:
                errors.append(
                    f"{where} is an operator row and names no entry: `target: <category>/<name>` "
                    f"(or `ops/<category>/<name>`) and `mode: onboard | verify`"
                )
            # The entry it names rides in the row's axes, where its task finds it.
            entry = {
                **entry,
                "config_axes": {**(entry.get("config_axes") or {}), "entry": f"{mode}:{target}"},
            }
        check = _check(entry.get("check"), kind, row_id, where, errors, none=True)
        if kind == OPERATOR and not (entry.get("check") or {}).get("id"):
            # As the gate names a row it is given by its entry (`gate kernel --row <entry>`).
            check["check"] = f"{CHECK_NAME[OPERATOR]}:{target.strip('/')}"
        rows.append(
            Row(
                id=row_id,
                kind=kind,
                outcome=outcome,
                deployment=str(entry.get("deployment") or "").strip(),
                depends_on=_strings(entry.get("depends_on")),
                required=_strings(entry.get("required")) + _strings(entry.get("cells")),
                optional=_strings(entry.get("optional")),
                writes=writes,
                **_evidence(entry),
                **check,
            )
        )
    if errors:
        bullet = "\n  - "
        raise RowError(f"the split's rows are not usable:{bullet}{bullet.join(errors)}")
    return rows


def deployment_row(
    name: str,
    *,
    side: str = "",
    outcome: str,
    required: Sequence[str],
    leads: Sequence[str] = (),
    writes: Sequence[str] = (),
    gpus: int = 0,
    nodes: int = 0,
    wall: int = DEFAULT_WALL_S,
    kind: str = ASSEMBLE,
    row_id: str = "",
    check: str = "",
) -> Row:
    """A row code writes for a deployment: its assembly, or a side or the state of a disaggregated one."""
    return Row(
        id=row_id or (f"{name}-{side}" if side else name),
        kind=kind,
        outcome=outcome,
        deployment=name,
        required=tuple(required),
        leads=tuple(leads),
        writes=tuple(writes),
        check=check or f"{DEPLOYMENT_CHECK}:{name}",
        gpus=gpus,
        nodes=nodes,
        wall=wall,
        side=side,
    )


#: Where the vocabulary of a row a split wrote is kept, for the gate to find its entry.
ROWS_DIRNAME = "rows"


def write_row_vocabulary(workspace: str | Path, row: Row) -> Row:
    """Keep a vocabulary of one queue row for an operator row a split wrote; the row, naming it.

    The survey's vocabulary does not hold the row, and the gate finds a kernel row's
    entry and tests in a vocabulary (``gate kernel --row <entry> --vocabulary <file>``).
    The file states the one row as the survey would: its mode, its entry, and the site
    its local variant is named for. The survey's own file is left as it is.
    """
    item = coverage_mod.queue_item_from_ref(
        str(row.config_axes.get("entry") or ""), row.id, row.outcome
    )
    if row.kind != OPERATOR or item is None:
        return row
    queue_row: dict[str, Any] = {"n": item.n, "mode": item.mode, "target": item.target}
    if item.site:
        queue_row["site"] = item.site
    if item.mode == coverage_mod.MODE_VERIFY:
        queue_row["surface"] = (row.required[0] if row.required else row.outcome) or row.id
    path = Path(workspace) / ROWS_DIRNAME / f"{row.id}.vocabulary.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"queue": [queue_row]}
    write_text_atomic(path, yaml.safe_dump(document, sort_keys=False))
    return replace(row, config_axes={**dict(row.config_axes), "vocabulary": str(path)})


#: A test file, by the name pytest collects: ``test_*.py`` or ``*_test.py``, wherever it is.
TEST_FILE = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")


def test_files(paths: Sequence[str]) -> list[str]:
    """The test files among ``paths``: a row's own tests are the ones its write scope names.

    Told by the name pytest collects, not by the directory: a helper or a ``conftest.py``
    beside the tests is no test to run, and a test outside ``tests/`` is one.
    """
    return [p for p in paths if TEST_FILE.search(p)]


test_files.__test__ = False  # a helper, not a test


def own_files(row_id: str, own: Sequence[str]) -> list[str]:
    """The files a row owns when it states none: one module and one test, named for it.

    Under the run's own paths (the target's directory, then the one its tests mirror it
    in). A row is given files of its own, never a whole directory: two rows told the same
    directory is theirs would each change what the other does.
    """
    stem = re.sub(r"[^a-z0-9_]", "_", row_id.lower())
    files = [f"{own[0].rstrip('/')}/{stem}.py"] if own else []
    if len(own) > 1:
        files.append(own_test(row_id, own))
    return files


def own_test(row_id: str, own: Sequence[str]) -> str:
    """The test file a row's own check runs when its write scope names none; ``""`` with no place."""
    if len(own) < 2:
        return ""
    stem = re.sub(r"[^a-z0-9_]", "_", row_id.lower())
    return f"{own[1].rstrip('/')}/test_modeling_v2_{stem}.py"


def save_work(workspace: str | Path, rows: Sequence[Row]) -> Path:
    """Keep ``rows`` in ``work.yaml``, a row of the same id replacing the one there."""
    path = Path(workspace) / WORK_FILENAME
    kept = {row.id: row for row in load_work(workspace)}
    kept.update({row.id: row for row in rows})
    text = yaml.safe_dump(
        {"rows": [row.as_dict() for row in kept.values()]},
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    write_text_atomic(path, text)
    return path


def load_work(workspace: str | Path) -> list[Row]:
    """The rows kept in ``work.yaml``; none when there is no such file yet."""
    try:
        data = yaml.safe_load((Path(workspace) / WORK_FILENAME).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    rows = data.get("rows") if isinstance(data, Mapping) else None
    return [Row.from_dict(row) for row in rows or () if isinstance(row, Mapping)]


def row_of(workspace: str | Path, row_id: str) -> Row | None:
    """The row ``row_id`` of ``work.yaml``, or ``None``."""
    return next((row for row in load_work(workspace) if row.id == row_id), None)


__all__ = [
    "ASSEMBLE",
    "CHECK_NAME",
    "DELIVERY_ROW",
    "CPU_CHECK",
    "DEFAULT_WALL_S",
    "DEPLOYMENT_CHECK",
    "FEATURE",
    "OPERATOR",
    "ROUTING",
    "ROWS_DIRNAME",
    "own_files",
    "own_test",
    "test_files",
    "ROW_KINDS",
    "Row",
    "RowError",
    "STATE",
    "WEIGHTS",
    "WORK_FILENAME",
    "check_id",
    "deployment_row",
    "load_rows",
    "load_work",
    "operator_rows",
    "row_of",
    "save_work",
    "split_rows",
    "typed_rows",
    "write_row_vocabulary",
]
