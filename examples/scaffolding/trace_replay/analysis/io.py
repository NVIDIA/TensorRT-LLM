"""Filesystem helpers: locate trace files, load/write JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OUTPUT_SUFFIX = ".cachehit.json"
ANNOTATED_TRACE_SUFFIX = ".trace.cachehit.json"


def load_trace(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Trace JSON must contain an object: {path}")
    return data


def write_json(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


def _trace_files_in_directory(trace_dir: Path) -> List[Path]:
    candidates = sorted(trace_dir.glob("*.trace.json"))
    compact = [p for p in candidates if not p.name.endswith(".full.trace.json")]
    return compact or candidates


def resolve_trace_file(trace_path: Path) -> Path:
    trace_path = trace_path.expanduser().resolve()
    if trace_path.is_file():
        return trace_path
    if not trace_path.is_dir():
        raise FileNotFoundError(f"Trace path does not exist: {trace_path}")
    candidates = _trace_files_in_directory(trace_path)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No *.trace.json file found in {trace_path}")
    raise ValueError(f"Multiple trace files found in {trace_path}: {candidates}")


def resolve_input_trace_files(trace_path: Path) -> Tuple[List[Path], bool]:
    """Return ``(trace_files, is_dataset)`` for *trace_path*.

    Inputs supported:
      - a specific ``*.trace.json`` file → ``([file], False)``
      - a directory containing exactly one trace → ``([file], False)``
      - a directory containing several traces → all of them, ``True``
      - a dataset directory whose subdirs each contain one trace → all, ``True``
    """
    trace_path = trace_path.expanduser().resolve()
    if trace_path.is_file():
        return [trace_path], False
    if not trace_path.is_dir():
        raise FileNotFoundError(f"Trace path does not exist: {trace_path}")

    trace_files = _trace_files_in_directory(trace_path)
    if trace_files:
        return trace_files, len(trace_files) > 1

    skipped: List[Path] = []
    for child in sorted(trace_path.iterdir()):
        if not child.is_dir():
            continue
        try:
            trace_files.append(resolve_trace_file(child))
        except FileNotFoundError:
            skipped.append(child)
    if trace_files:
        return trace_files, True
    if skipped:
        raise FileNotFoundError(
            f"No trace files found in dataset directory {trace_path}; "
            f"checked {len(skipped)} subdirectories"
        )
    raise FileNotFoundError(f"No *.trace.json file found in {trace_path}")


def default_output_path(trace_file: Path, output_json: Optional[Path]) -> Path:
    if output_json is not None:
        return output_json.expanduser().resolve()
    if trace_file.name.endswith(".trace.json"):
        return trace_file.with_name(trace_file.name[: -len(".trace.json")] + OUTPUT_SUFFIX)
    return trace_file.with_suffix(OUTPUT_SUFFIX)


def default_dataset_output_path(dataset_dir: Path, output_json: Optional[Path]) -> Path:
    if output_json is not None:
        return output_json.expanduser().resolve()
    resolved = dataset_dir.expanduser().resolve()
    return resolved / f"{resolved.name}{OUTPUT_SUFFIX}"


def default_annotated_trace_path(trace_file: Path, output_json: Optional[Path] = None) -> Path:
    """Return the default ``*.trace.cachehit.json`` path for *trace_file*."""
    if output_json is not None:
        return output_json.expanduser().resolve()
    if trace_file.name.endswith(".trace.json"):
        return trace_file.with_name(trace_file.name[: -len(".trace.json")] + ANNOTATED_TRACE_SUFFIX)
    return trace_file.with_suffix(ANNOTATED_TRACE_SUFFIX)
