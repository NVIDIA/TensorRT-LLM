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
from pathlib import Path
from typing import Any, Iterable, Mapping

#: One per request the engine driver ran.
RECORD_MARKER = "MV2FLOW_RECORD"
#: What the driver imported, once, before the engine starts.
DRIVER_MARKER = "MV2FLOW_DRIVER"
#: What the engine runs once it is up, once per engine a harness builds (see
#: :mod:`.engine_facts`).
ENGINE_MARKER = "MV2FLOW_ENGINE"
#: What the engine reused from its KV cache over the workload, once per engine,
#: after the workload (see :mod:`.reuse_probe`).
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

#: Benchmarks whose dataset the gates know without a declaration. The source
#: is a Hugging Face id, read from the local cache only.
KNOWN_DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "source": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "text_field": "question",
    },
}

_LOCAL_SUFFIXES: tuple[str, ...] = (".jsonl", ".json", ".parquet", ".arrow", ".yaml", ".yml")
_LIST_KEYS: tuple[str, ...] = ("prompts", "data", "rows", "examples", "items")


class DatasetError(ValueError):
    """A workload could not be read; the message names what and where."""


# ------------------------------------------------------------------- records


def emit(marker: str, payload: Mapping[str, Any]) -> None:
    """Print ``<marker> <json>`` on one line and flush."""
    line = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True)
    print(f"{marker} {line}", flush=True)


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


def select_prompts(
    rows: Iterable[Any],
    *,
    num: int,
    text_field: str | None = None,
    template: str = "{text}",
    skip: int = 0,
) -> list[str]:
    """The first ``num`` usable rows after ``skip``, rendered through ``template``.

    A row is a string (the text itself) or a mapping holding the text under
    ``text_field`` (or the first of :data:`TEXT_FIELDS` present). ``template``
    names the text ``{text}``; every other field of the row is available
    under its own name.
    """
    selected: list[str] = []
    for index, row in enumerate(rows):
        if index < skip:
            continue
        if len(selected) >= num:
            break
        if isinstance(row, str):
            values: dict[str, Any] = {"text": row}
        elif isinstance(row, Mapping):
            name = text_field or next((n for n in TEXT_FIELDS if n in row), None)
            if name is None or not isinstance(row.get(name), str) or not row[name].strip():
                continue
            values = {str(k): v for k, v in row.items()}
            values["text"] = row[name]
        else:
            continue
        try:
            selected.append(template.format(**values))
        except (KeyError, IndexError, ValueError) as exc:
            raise DatasetError(
                f"prompt template {template!r} does not fit row {index}: {exc}"
            ) from exc
    return selected


def is_lfs_pointer(path: str | Path) -> bool:
    """Whether ``path`` is a git-lfs pointer file rather than the content it names."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs")


def normalize_requests(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize ordered requests once, without importing the GPU runtime.

    Keep explicit token/image requests unchanged. Dataset skip counts source
    rows, not usable prompts; selection and templates have one implementation.
    """
    budget = int(spec.get("max_tokens", 64))
    requests = []
    for index, entry in enumerate(spec.get("requests") or ()):
        request = dict(entry)
        request.setdefault("id", f"r{index}")
        request.setdefault("max_tokens", budget)
        requests.append(request)
    dataset = spec.get("dataset")
    if dataset:
        rows = load_rows(
            dataset["source"], split=dataset.get("split", "test"), config=dataset.get("config")
        )
        prompts = select_prompts(
            rows,
            num=int(dataset.get("num", 32)),
            text_field=dataset.get("text_field"),
            template=dataset.get("template") or "{text}",
            skip=int(dataset.get("skip", 0)),
        )
        if not prompts:
            raise DatasetError("dataset selection produced no usable prompts")
        requests.extend(
            {"id": f"d{i}", "prompt": text, "max_tokens": int(dataset.get("max_tokens", budget))}
            for i, text in enumerate(prompts)
        )
    return requests


def normalize_file(source: str, target: str) -> None:
    """Write the GPU entry's JSON request spec from the CPU-side source spec."""
    spec = json.loads(Path(source).read_text(encoding="utf-8"))
    spec["requests"] = normalize_requests(spec)
    spec.pop("dataset", None)
    path = Path(target)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(spec, ensure_ascii=True), encoding="utf-8")
    partial.replace(path)


__all__ = [
    "DRIVER_MARKER",
    "DatasetError",
    "ENGINE_MARKER",
    "KNOWN_DATASETS",
    "RANK_ENGINE_MARKER",
    "RECORD_MARKER",
    "REUSE_MARKER",
    "TEXT_FIELDS",
    "emit",
    "is_lfs_pointer",
    "load_rows",
    "parse_marked",
    "prompt_digest",
    "normalize_requests",
    "normalize_file",
    "select_prompts",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize an offline workload before engine launch"
    )
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    normalize_file(args.source, args.target)
