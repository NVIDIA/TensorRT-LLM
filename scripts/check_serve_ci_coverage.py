#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check the CI-coverage invariants of trtllm-serve (epic TRTLLM-17056).

Three checks, driven by ``scripts/serve_ci_coverage.yml``:

- ``multigpu``: the ``tensorrt_llm/serve/`` entries of the multi-GPU trigger table
  (``relatedFileList`` in ``getMultiGpuFileChanged``, ``jenkins/L0_MergeRequest.groovy``)
  must match files at the current head; every path on the required list that exists must
  be covered by an entry; no entry may be a forbidden (too-broad) prefix, matched with
  or without its trailing slash, or any other string prefix of the managed directory.
- ``visual_gen``: every prefix in the CBTS VisualGen tuples
  (``jenkins/scripts/cbts/cbts/rules/visual_gen.py``) must match a file.
- ``codeowners``: every ``.github/CODEOWNERS`` rule under the managed serve prefixes must
  match a file, and files under the required paths must have exactly the listed owners.

Runs as the ``serve-ci-coverage`` pre-commit hook and under
``tests/unittest/scripts/``. Uses only the standard library plus PyYAML: pre-commit runs
it in an isolated environment where ``tensorrt_llm`` is not installed.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import yaml

DATA_FILE_HINT = "scripts/serve_ci_coverage.yml"
_VG_TUPLE_NAMES = ("_VG_SRC_PREFIXES", "_VG_OUTWARD_PREFIXES")
_RELATED_LIST_RE = re.compile(r"^\s*def\s+relatedFileList\s*=\s*\[\s*$")
_RELATED_ENTRY_RE = re.compile(r'^"([^"]+)",?\s*(//.*)?$')
_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"


class ParseError(Exception):
    """A watched file no longer has the shape this lint knows how to read."""

    def __init__(self, what: str, detail: str):
        self.what = what
        self.detail = detail
        super().__init__(f"{what} ({detail})")


def _parse_failure(file: str, err: ParseError) -> str:
    return (
        f"{file}: cannot parse {err.what} ({err.detail}). "
        "Update scripts/check_serve_ci_coverage.py together with the format change."
    )


def validate_data(data: object) -> list[str]:
    """Return schema errors of the data file as failure messages."""
    errors: list[str] = []

    def _req(mapping: dict, key: str, kind: type, where: str) -> object:
        value = mapping.get(key)
        key_path = f"{where}.{key}" if where else key
        if not isinstance(value, kind):
            errors.append(
                f"{DATA_FILE_HINT}: `{key_path}` must be a {kind.__name__}, got "
                f"{type(value).__name__}"
            )
            return None
        return value

    if not isinstance(data, dict):
        return [f"{DATA_FILE_HINT}: top level must be a mapping, got {type(data).__name__}"]
    if data.get("version") != 1:
        errors.append(f"{DATA_FILE_HINT}: `version` must be 1, got {data.get('version')!r}")

    multi_gpu = _req(data, "multi_gpu", dict, "")
    if isinstance(multi_gpu, dict):
        _req(multi_gpu, "file", str, "multi_gpu")
        _req(multi_gpu, "managed_prefix", str, "multi_gpu")
        for key in ("required", "forbidden"):
            rows = _req(multi_gpu, key, list, "multi_gpu")
            for row in rows or []:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("path"), str)
                    or not isinstance(row.get("why"), str)
                ):
                    errors.append(
                        f"{DATA_FILE_HINT}: every `multi_gpu.{key}` row must be a mapping "
                        f"with string `path` and `why`, got {row!r}"
                    )

    visual_gen = _req(data, "visual_gen", dict, "")
    if isinstance(visual_gen, dict):
        _req(visual_gen, "file", str, "visual_gen")
        tuples = _req(visual_gen, "tuples", list, "visual_gen")
        for name in tuples or []:
            if not isinstance(name, str):
                errors.append(
                    f"{DATA_FILE_HINT}: every `visual_gen.tuples` item must be a string, "
                    f"got {name!r}"
                )

    codeowners = _req(data, "codeowners", dict, "")
    if isinstance(codeowners, dict):
        _req(codeowners, "file", str, "codeowners")
        prefixes = _req(codeowners, "managed_prefixes", list, "codeowners")
        for prefix in prefixes or []:
            if not isinstance(prefix, str):
                errors.append(
                    f"{DATA_FILE_HINT}: every `codeowners.managed_prefixes` item must be "
                    f"a string, got {prefix!r}"
                )
        rows = _req(codeowners, "required", list, "codeowners")
        for row in rows or []:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("path"), str)
                or not isinstance(row.get("owners"), list)
                or not row.get("owners")
                or not all(isinstance(o, str) for o in row["owners"])
            ):
                errors.append(
                    f"{DATA_FILE_HINT}: every `codeowners.required` row must be a mapping "
                    f"with a string `path` and a non-empty string list `owners`, got {row!r}"
                )
    return errors


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a repeated mapping key instead of keeping its last value."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[yaml.MappingNode] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        # SafeLoader recursively flattens merge sources before constructing them.
        # Check their original keys once, allowing explicit overrides of merged keys.
        if node in self._checked_mappings:
            return
        self._checked_mappings.add(node)
        seen: set[object] = set()
        for key_node, _ in node.value:
            # Merge keys and (unhashable) non-scalar keys are left to SafeLoader.
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag == _YAML_MERGE_TAG:
                continue
            key = self.construct_object(key_node)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        super().flatten_mapping(node)


def load_data(path: Path) -> dict:
    """Load the data file and raise ParseError when it does not match the schema."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise ParseError("the data file", str(err)) from err
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as err:
        raise ParseError("the data file", f"invalid YAML: {err}") from err
    errors = validate_data(data)
    if errors:
        raise ParseError("the data file", "; ".join(errors))
    return data


def parse_related_file_list(groovy_text: str) -> list[tuple[int, str]]:
    """Return the (line, entry) pairs of relatedFileList in getMultiGpuFileChanged."""
    lines = groovy_text.splitlines()
    starts = [i for i, line in enumerate(lines) if _RELATED_LIST_RE.match(line)]
    if len(starts) != 1:
        raise ParseError(
            "relatedFileList",
            f"expected exactly one `def relatedFileList = [` line, found {len(starts)}",
        )
    entries: list[tuple[int, str]] = []
    for i in range(starts[0] + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("]"):
            return entries
        if not stripped or stripped.startswith("//"):
            continue
        match = _RELATED_ENTRY_RE.match(stripped)
        if not match:
            raise ParseError("relatedFileList", f"line {i + 1} is not a quoted entry: {stripped!r}")
        entries.append((i + 1, match.group(1)))
    raise ParseError("relatedFileList", "the list is never closed with `]`")


def parse_vg_prefixes(py_text: str) -> dict[str, list[tuple[int, str]]]:
    """Return {tuple name: [(line, prefix)]} for the CBTS VisualGen prefix tuples."""
    try:
        tree = ast.parse(py_text)
    except SyntaxError as err:
        raise ParseError("the module", f"invalid Python: {err}") from err
    found: dict[str, list[tuple[int, str]]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
            value = node.value
        else:
            continue
        for name in names:
            if name not in _VG_TUPLE_NAMES:
                continue
            if (
                not isinstance(value, ast.Tuple)
                or not value.elts
                or not all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
                )
            ):
                raise ParseError(name, "not a non-empty literal tuple of strings")
            found[name] = [(e.lineno, e.value) for e in value.elts]
    missing = [name for name in _VG_TUPLE_NAMES if name not in found]
    if missing:
        raise ParseError(", ".join(missing), "module-level tuple assignment not found")
    return found


def parse_codeowners(text: str) -> list[tuple[int, str, tuple[str, ...]]]:
    """Return (line, pattern, owners) per rule; owners are lower-cased for comparison."""
    rules: list[tuple[int, str, tuple[str, ...]]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        rules.append((lineno, parts[0], tuple(o.lower() for o in parts[1:])))
    return rules


def _pattern_to_regex(pattern: str) -> re.Pattern:
    # Same translation as .github/scripts/label_component.py (_pattern_to_regex): the
    # patterns in this repository are root-anchored, `*` stays inside a path segment,
    # `**` crosses segments, and a directory pattern matches everything beneath it.
    body = re.escape(pattern.strip("/"))
    body = body.replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.compile(rf"(?:{body})(?:/.*)?$")


def path_exists(repo: Path, path: str) -> bool:
    """True when `path` matches: directories (trailing `/`) need a real file inside."""
    target = repo / path
    if path.endswith("/"):
        if not target.is_dir():
            return False
        for _, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            if any(not f.endswith(".pyc") for f in files):
                return True
        return False
    return target.is_file()


def _walk_files(repo: Path, rel: str) -> list[str]:
    """Repo-relative POSIX paths of the regular files under `rel` (or `rel` itself)."""
    base = repo / rel
    if base.is_file():
        return [rel]
    out: list[str] = []
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
            for name in sorted(files):
                if name.endswith(".pyc") or name.startswith("."):
                    continue
                out.append((Path(root) / name).relative_to(repo).as_posix())
    return out


def _candidate_files(repo: Path, pattern: str) -> list[str]:
    """The files a CODEOWNERS pattern could match: its literal prefix, walked."""
    body = pattern.strip("/")
    literal = body.split("*", 1)[0]
    if "*" in body:
        base = literal.rsplit("/", 1)[0] if "/" in literal else ""
    else:
        base = literal
    if not base:
        return []
    return _walk_files(repo, base)


def check_multigpu(repo: Path, data: dict) -> list[str]:
    cfg = data["multi_gpu"]
    table_file = cfg["file"]
    try:
        text = (repo / table_file).read_text(encoding="utf-8")
    except OSError as err:
        return [_parse_failure(table_file, ParseError("relatedFileList", str(err)))]
    try:
        entries = parse_related_file_list(text)
    except ParseError as err:
        return [_parse_failure(table_file, err)]

    failures: list[str] = []
    managed_prefix = cfg["managed_prefix"]
    # The Groovy table matches by substring, so a forbidden row is matched with and
    # without its trailing slash: an entry `tensorrt_llm/serve` over-triggers exactly
    # like `tensorrt_llm/serve/` does.
    forbidden = {row["path"].rstrip("/"): row["why"] for row in cfg["forbidden"]}
    for line, entry in entries:
        why = forbidden.get(entry.rstrip("/"))
        if why is not None:
            failures.append(
                f"{table_file}:{line}: `{entry}` must not be in the multi-GPU trigger "
                f"table: {why}. Add one entry per subpackage instead."
            )
        elif managed_prefix.startswith(entry):
            failures.append(
                f"{table_file}:{line}: `{entry}` must not be in the multi-GPU trigger "
                f"table: it is a prefix of `{managed_prefix}`, and the table matches by "
                "substring, so it would start the multi-GPU stages for every change "
                "under that directory. Add one entry per subpackage instead."
            )
        if entry.startswith(managed_prefix) and not path_exists(repo, entry):
            failures.append(
                f"{table_file}:{line}: multi-GPU trigger entry `{entry}` "
                "(relatedFileList in getMultiGpuFileChanged) matches no file. Add an "
                "entry in the same PR that creates the path, not earlier; if the code "
                "moved, replace the entry with the new location. Changes under jenkins/ "
                "need @NVIDIA/trt-llm-infra-devs."
            )
    entry_paths = [entry for _, entry in entries]
    for row in cfg["required"]:
        path = row["path"]
        if not path_exists(repo, path):
            continue
        if not any(path.startswith(entry) for entry in entry_paths):
            failures.append(
                f"`{path}` exists but no multi-GPU trigger entry covers it. Add "
                f'`"{path}",` to relatedFileList in getMultiGpuFileChanged '
                f"({table_file}). Reason: {row['why']} (list in {DATA_FILE_HINT}). "
                "Without it, later PRs that change only this path skip the multi-GPU "
                "stages without any error. Approval: @NVIDIA/trt-llm-infra-devs."
            )
    return failures


def check_visual_gen(repo: Path, data: dict) -> list[str]:
    cfg = data["visual_gen"]
    rule_file = cfg["file"]
    try:
        text = (repo / rule_file).read_text(encoding="utf-8")
    except OSError as err:
        return [_parse_failure(rule_file, ParseError("the prefix tuples", str(err)))]
    try:
        parsed = parse_vg_prefixes(text)
    except ParseError as err:
        return [_parse_failure(rule_file, err)]

    failures: list[str] = []
    for tuple_name in cfg["tuples"]:
        if tuple_name not in parsed:
            failures.append(
                _parse_failure(
                    rule_file,
                    ParseError(tuple_name, "module-level tuple assignment not found"),
                )
            )
            continue
        for line, prefix in parsed[tuple_name]:
            if not path_exists(repo, prefix):
                failures.append(
                    f"{rule_file}:{line}: `{tuple_name}` entry `{prefix}` matches no "
                    "file. Add a prefix in the PR that creates the directory (for "
                    "`tensorrt_llm/serve/api/visual_gen/`, in the PR that adds the "
                    "VisualGen serving import-isolation tests). Approval: CBTS owners "
                    "(@NVIDIA/trt-llm-devs, `/jenkins/scripts/cbts` in .github/CODEOWNERS)."
                )
    return failures


def check_codeowners(repo: Path, data: dict) -> list[str]:
    cfg = data["codeowners"]
    co_file = cfg["file"]
    try:
        text = (repo / co_file).read_text(encoding="utf-8")
    except OSError as err:
        return [_parse_failure(co_file, ParseError("the rules", str(err)))]
    rules = [
        (line, pattern, owners, _pattern_to_regex(pattern))
        for line, pattern, owners in parse_codeowners(text)
    ]

    failures: list[str] = []
    managed_prefixes = cfg["managed_prefixes"]
    for line, pattern, _, regex in rules:
        if not any(pattern.startswith(prefix) for prefix in managed_prefixes):
            continue
        if not any(regex.match(path) for path in _candidate_files(repo, pattern)):
            failures.append(
                f"{co_file}:{line}: rule `{pattern}` matches no file. A PR that moves "
                "or deletes a path updates its CODEOWNERS rules in the same PR; a kept "
                "compatibility shim whose owner would otherwise change keeps its "
                "pre-move owner through a line in the "
                '"COMPATIBILITY FORWARDING MODULES" section. Approval: '
                "@NVIDIA/trt-llm-infra-devs or @NVIDIA/trt-llm-oss-compliance."
            )

    required = cfg["required"]
    serve_rule_line = next(
        (line for line, pattern, _, _ in rules if pattern == "/tensorrt_llm/serve"), None
    )
    anchor = "after `/tensorrt_llm/serve`"
    if serve_rule_line is not None:
        anchor += f" ({co_file}:{serve_rule_line})"
    covered_files: set[str] = set()
    for row in required:
        covered_files.update(_walk_files(repo, row["path"]))
    for path in sorted(covered_files):
        best = max(
            (
                row["path"]
                for row in required
                if path == row["path"] or path.startswith(row["path"] + "/")
            ),
            key=len,
            default=None,
        )
        if best is None:
            continue
        required_owners = next(row["owners"] for row in required if row["path"] == best)
        actual_owners: tuple[str, ...] = ()
        deciding_line = None
        for line, _, owners, regex in rules:
            if regex.match(path):
                actual_owners = owners
                deciding_line = line
        if set(actual_owners) == {o.lower() for o in required_owners}:
            continue
        if deciding_line is None:
            owned_by = "no one (no CODEOWNERS rule matches)"
        else:
            owned_by = f"{' '.join(actual_owners)} (decided by {co_file}:{deciding_line})"
        failures.append(
            f"`{path}` is owned by {owned_by}, expected {' '.join(required_owners)} for "
            f"`{best}` (list in {DATA_FILE_HINT}). Add `/{best} "
            f"{' '.join(required_owners)}` {anchor} or in the owning team's section; "
            "the last matching rule wins. Approval: @NVIDIA/trt-llm-infra-devs or "
            "@NVIDIA/trt-llm-oss-compliance."
        )
    return failures


CHECKS = {
    "multigpu": check_multigpu,
    "visual_gen": check_visual_gen,
    "codeowners": check_codeowners,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checks",
        default=",".join(CHECKS),
        help=f"comma-separated subset of: {', '.join(CHECKS)} (default: all)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root (default: the parent of this script's directory)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=f"data file (default: {DATA_FILE_HINT} under the repo root)",
    )
    args = parser.parse_args(argv)
    repo = args.repo_root if args.repo_root is not None else Path(__file__).resolve().parents[1]
    data_path = args.data if args.data is not None else repo / DATA_FILE_HINT

    selected = [name.strip() for name in args.checks.split(",") if name.strip()]
    unknown = [name for name in selected if name not in CHECKS]
    if unknown or not selected:
        parser.error(
            f"--checks takes a non-empty subset of {', '.join(CHECKS)}; got {args.checks!r}"
        )

    try:
        data = load_data(data_path)
    except ParseError as err:
        print(_parse_failure(str(data_path), err))
        return 1

    failures: list[str] = []
    for name in selected:
        failures.extend(CHECKS[name](repo, data))
    if failures:
        print("\n\n".join(failures))
        return 1
    print(f"serve CI coverage: {len(selected)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
