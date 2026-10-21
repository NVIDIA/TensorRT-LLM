# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The record format the harness prints, and the local dataset reader it uses.

Standard library only at import time, because this module runs on both
sides: imported by the orchestrator (a login-node interpreter without numpy)
to parse a gate's output, and by the harness scripts under the measured
tree's interpreter to load a workload.

RECORDS. A harness prints one line per result, ``<MARKER> <json>``, the JSON
on one line with ASCII escapes. A gate reads ONLY those lines -- never the
engine's log -- so a keyword cannot match ``CPUs 72-143`` in a log line, and
a prefix added by ``srun -l`` or a logger does not hide a record.

DATASETS. A dataset is a local path (a file or a directory of files: JSON,
JSON lines, YAML, Parquet or Arrow) or a Hugging Face dataset id, which is
read with ``HF_DATASETS_OFFLINE=1`` and ``HF_HUB_OFFLINE=1`` set first, so it
resolves from the local cache or fails; nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

#: One per request the engine driver ran.
RECORD_MARKER = "MV2FLOW_RECORD"
#: The engine driver's performance measurement, once.
PERF_MARKER = "MV2FLOW_PERF"
#: What the driver imported, once, before the engine starts.
DRIVER_MARKER = "MV2FLOW_DRIVER"
#: What the engine runs once it is up, once per engine a harness builds (see
#: :mod:`.engine_facts`).
ENGINE_MARKER = "MV2FLOW_ENGINE"
#: What the engine reused from its KV cache over the workload, once per engine,
#: after the workload (see :mod:`.engine_reuse`).
REUSE_MARKER = "MV2FLOW_REUSE"
#: What one rank's executor worker runs, printed by that worker as it sets its
#: engine up, once per rank (see ``rankhook/sitecustomize.py``).
RANK_ENGINE_MARKER = "MV2FLOW_RANK_ENGINE"

#: Fields tried, in order, when a dataset does not say which one holds the prompt.
TEXT_FIELDS: tuple[str, ...] = (
    "question",
    "prompt",
    "problem",
    "input",
    "text",
    "instruction",
    "query",
)
#: Fields tried, in order, for the reference answer a scorer compares against.
ANSWER_FIELDS: tuple[str, ...] = ("answer", "target", "output", "solution", "label")

#: Benchmarks whose dataset the gates know without a declaration. The source
#: is a Hugging Face id, read from the local cache only.
KNOWN_DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "source": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "text_field": "question",
        "answer_field": "answer",
        "scorer": "gsm8k",
    },
}

#: Scorers ``score`` knows.
SCORERS: tuple[str, ...] = ("gsm8k", "contains", "exact")

_LOCAL_SUFFIXES: tuple[str, ...] = (".jsonl", ".json", ".parquet", ".arrow", ".yaml", ".yml")
_LIST_KEYS: tuple[str, ...] = ("prompts", "data", "rows", "examples", "items")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


class DatasetError(ValueError):
    """A workload could not be read; the message names what and where."""


# ------------------------------------------------------------------- records


def emit(marker: str, payload: Mapping[str, Any], stream: TextIO | None = None) -> None:
    """Print ``<marker> <json>`` on one line and flush."""
    line = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True)
    print(f"{marker} {line}", file=stream or sys.stdout, flush=True)


def parse_marked(output: str, marker: str) -> list[dict[str, Any]]:
    """Every ``<marker> <json>`` record in ``output``, in order.

    Anything may precede the marker on its line (a rank tag, a logger
    prefix). A line whose JSON does not parse is skipped rather than raised:
    a torn last line of a killed job must not hide the records before it.
    """
    pattern = re.compile(rf"{re.escape(marker)} (\{{.*\}})\s*$")
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        found = pattern.search(line)
        if not found:
            continue
        try:
            data = json.loads(found.group(1))
        except ValueError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def prompt_digest(text: str) -> str:
    """A short, stable name for a prompt, for pairing records across runs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------- datasets


def _rows_from(data: Any, source: Path) -> list[Any]:
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), list):
                return list(data[key])
        raise DatasetError(f"{source} holds a mapping with none of {list(_LIST_KEYS)} as a list")
    if isinstance(data, list):
        return data
    raise DatasetError(f"{source} holds {type(data).__name__}, not a list of rows")


def _read_file(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            return rows
        if suffix == ".json":
            return _rows_from(json.loads(path.read_text(encoding="utf-8")), path)
        if suffix in (".yaml", ".yml"):
            import yaml

            return _rows_from(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    if suffix in (".parquet", ".arrow"):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise DatasetError(
                f"reading {path} needs pyarrow, which this interpreter lacks"
            ) from exc
        try:
            if suffix == ".parquet":
                table = pq.read_table(path)
            else:
                try:
                    table = pa.ipc.open_stream(str(path)).read_all()
                except pa.ArrowInvalid:
                    table = pa.ipc.open_file(str(path)).read_all()
        except (OSError, pa.ArrowException) as exc:
            raise DatasetError(f"cannot read {path}: {exc}") from exc
        return table.to_pylist()
    raise DatasetError(f"{path}: unsupported dataset file type {suffix!r}")


def _local_files(directory: Path, split: str | None) -> list[Path]:
    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in _LOCAL_SUFFIXES
    )
    if split:
        # Named in the file itself first, then anywhere under `directory` --
        # never in the directory's own ancestors, which say nothing about it.
        for matches in (
            lambda p: split in p.name,
            lambda p: split in str(p.relative_to(directory)),
        ):
            named = [p for p in files if matches(p)]
            if named:
                return named
    return files


def _load_hf_offline(source: str, config: str | None, split: str | None) -> list[Any]:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import datasets
    except ImportError as exc:
        raise DatasetError(
            f"{source!r} is not a local path, and reading it as a Hugging Face dataset id "
            f"needs the `datasets` package, which this interpreter lacks"
        ) from exc
    try:
        loaded = datasets.load_dataset(source, config, split=split or "test")
    # Offline misses surface as `OfflineModeIsEnabled` (a ConnectionError) or a
    # `DatasetNotFoundError`/`FileNotFoundError`; a bad config or split as ValueError.
    except (OSError, ConnectionError, ValueError) as exc:
        raise DatasetError(
            f"{source!r} (config {config!r}, split {split!r}) is not in the local Hugging Face "
            f"cache and nothing is downloaded: {type(exc).__name__}: {exc}. Declare a local "
            f"path instead"
        ) from exc
    return [dict(row) for row in loaded]


def load_rows(
    source: str | Path,
    *,
    split: str | None = "test",
    config: str | None = None,
) -> list[Any]:
    """Every row of ``source``: a local file or directory, or a cached dataset id."""
    path = Path(str(source))
    if path.is_file():
        return _read_file(path)
    if path.is_dir():
        files = _local_files(path, split)
        if not files:
            raise DatasetError(f"{path} holds no {list(_LOCAL_SUFFIXES)} file")
        rows: list[Any] = []
        for file in files:
            rows.extend(_read_file(file))
        return rows
    if path.is_absolute() or str(source).startswith("."):
        raise DatasetError(f"dataset path does not exist: {path}")
    return _load_hf_offline(str(source), config, split)


def _field(row: Mapping[str, Any], declared: str | None, candidates: Sequence[str]) -> str | None:
    if declared:
        return declared if declared in row else None
    return next((name for name in candidates if name in row), None)


def select_prompts(
    rows: Iterable[Any],
    *,
    num: int,
    text_field: str | None = None,
    answer_field: str | None = None,
    template: str = "{text}",
    skip: int = 0,
) -> list[dict[str, Any]]:
    """The first ``num`` usable rows after ``skip``, rendered through ``template``.

    A row is a string (the text itself) or a mapping holding the text under
    ``text_field`` (or the first of :data:`TEXT_FIELDS` present). ``template``
    names the text ``{text}``; every other field of the row is available
    under its own name. Each result is ``{"prompt", "text", "answer"}``.
    """
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < skip:
            continue
        if len(selected) >= num:
            break
        if isinstance(row, str):
            values: dict[str, Any] = {"text": row}
            answer = None
        elif isinstance(row, Mapping):
            name = _field(row, text_field, TEXT_FIELDS)
            if name is None or not isinstance(row.get(name), str) or not row[name].strip():
                continue
            values = {str(k): v for k, v in row.items()}
            values["text"] = row[name]
            answer_name = _field(row, answer_field, ANSWER_FIELDS)
            answer = row.get(answer_name) if answer_name else None
        else:
            continue
        try:
            prompt = template.format(**values)
        except (KeyError, IndexError, ValueError) as exc:
            raise DatasetError(
                f"prompt template {template!r} does not fit row {index}: {exc}"
            ) from exc
        selected.append({"prompt": prompt, "text": values["text"], "answer": answer})
    return selected


# ------------------------------------------------------------------- scoring


def last_number(text: str) -> float | None:
    """The last number in ``text``, commas dropped; ``None`` if there is none."""
    found = _NUMBER.findall(text or "")
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def score(scorer: str, response: str, answer: Any) -> bool | None:
    """Whether ``response`` answers ``answer`` under ``scorer``; ``None`` if unscorable."""
    if answer is None:
        return None
    reference = str(answer)
    if scorer == "gsm8k":
        expected = last_number(reference.split("####")[-1])
        got = last_number(response)
        if expected is None or got is None:
            return False if expected is not None else None
        return abs(expected - got) < 1e-6
    if scorer == "contains":
        return reference.strip().lower() in (response or "").lower()
    if scorer == "exact":
        return reference.strip() == (response or "").strip()
    raise DatasetError(f"unknown scorer {scorer!r}; expected one of {SCORERS}")


def is_lfs_pointer(path: str | Path) -> bool:
    """Whether ``path`` is a git-lfs pointer file rather than the content it names."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs")


__all__ = [
    "ANSWER_FIELDS",
    "DRIVER_MARKER",
    "DatasetError",
    "ENGINE_MARKER",
    "KNOWN_DATASETS",
    "PERF_MARKER",
    "RANK_ENGINE_MARKER",
    "RECORD_MARKER",
    "REUSE_MARKER",
    "SCORERS",
    "TEXT_FIELDS",
    "emit",
    "is_lfs_pointer",
    "last_number",
    "load_rows",
    "parse_marked",
    "prompt_digest",
    "score",
    "select_prompts",
]
