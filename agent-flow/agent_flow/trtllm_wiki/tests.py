# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which CI test lists run a test, under what conditions, and which tests are waived.

``tests/integration/test_lists/test-db/l0_*.yml`` holds blocks of a
``condition`` (GPU wildcards, GPU-count ranges, ``stage``/``backend`` terms)
and the tests it runs. An entry is a path relative to
``tests/integration/defs`` (``unittest/...`` is ``tests/unittest/...``),
optionally a pytest node id, optionally followed by markers such as
``TIMEOUT (90)`` or ``ISOLATION``. The QA lists under
``tests/integration/test_lists/qa`` (YAML of the same shape, or one entry
per line) are read too, as ``kind: qa``. An entry lists a test when it names the
same node, the test's file, or a directory above it: the same rule the CI
runner applies, without substring guesses. ``waives.txt`` and, when the
checkout has it, the Jenkins stage map (``jenkins/L0_Test.groovy``) complete
the picture.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

TEST_DB_REL = "tests/integration/test_lists/test-db"
QA_LISTS_REL = "tests/integration/test_lists/qa"
WAIVES_REL = "tests/integration/test_lists/waives.txt"
STAGES_REL = "jenkins/L0_Test.groovy"
DEFS_REL = "tests/integration/defs"

_TIMEOUT = re.compile(r"\bTIMEOUT\s*\((\d+)\)")
_STAGE = re.compile(
    r'^\s*"(?P<stage>[A-Za-z0-9_.-]+)"\s*:\s*\[\s*"[^"]*"\s*,\s*"(?P<yml>l0_[a-z0-9_]+)"'
)


_GPU_PREFIX = re.compile(r"^full:(?P<gpu>[^/]+)/")
_SELECT = re.compile(r"""-k\s+(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>\S+))""")
_IGNORE = re.compile(r"--ignore(?:-glob)?=(?P<path>\S+)")


def repo_path(entry: str) -> str:
    """The repository-relative path a test-list entry (or node id) names, without its node part."""
    path = _GPU_PREFIX.sub("", entry.split("::", 1)[0].strip())
    if path.startswith("unittest/"):
        return f"tests/{path}"
    if path.startswith("tests/"):
        return path
    return f"{DEFS_REL}/{path}"


def _entry(raw: str) -> dict[str, Any]:
    text = raw.strip()
    head, _, tail = text.partition(" ")
    timeout = _TIMEOUT.search(tail)
    select = _SELECT.search(tail)
    return {
        "test": head,
        "path": repo_path(head),
        "node": head.split("::", 1)[1] if "::" in head else "",
        # `-k EXPR` runs a subset of the entry: listed, but only partly.
        "select": next((g for g in select.groups() if g is not None), "") if select else "",
        "ignore": [repo_path(m["path"]) for m in _IGNORE.finditer(tail)],
        "timeout_min": int(timeout.group(1)) if timeout else None,
        "isolation": "ISOLATION" in tail,
    }


def _condition(block: dict[str, Any]) -> dict[str, Any]:
    condition = block.get("condition") or {}
    terms = condition.get("terms") or block.get("terms") or {}
    wildcards = condition.get("wildcards") or {}
    count = (condition.get("ranges") or {}).get("system_gpu_count") or {}
    gpus = wildcards.get("gpu") or []
    return {
        "stage": str(terms.get("stage") or ""),
        "backend": str(terms.get("backend") or ""),
        "gpu": [str(g) for g in (gpus if isinstance(gpus, list) else [gpus])],
        "gpu_count": {k: count[k] for k in ("gte", "lte") if k in count},
    }


def _yaml_records(path: Path, kind: str) -> list[dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    if isinstance(data, list):
        # Blocks without the list-name key: the list is the file's.
        data = {path.stem: data}
    if not isinstance(data, dict):
        return []
    records: list[dict[str, Any]] = []
    for name, blocks in data.items():
        if name == "version" or not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            condition = _condition(block)
            for raw in block.get("tests") or ():
                if isinstance(raw, str) and raw.strip():
                    records.append({"list": str(name), "kind": kind, **condition, **_entry(raw)})
    return records


def _text_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    condition = {"stage": "", "backend": "", "gpu": [], "gpu_count": {}}
    return [
        {"list": path.stem, "kind": "qa", **condition, **_entry(line)}
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def extract_test_lists(repo: Path) -> list[dict[str, Any]]:
    """One record per (list, block, entry): the L0 test-db lists (``kind: l0``), then the QA lists."""
    repo = Path(repo)
    records: list[dict[str, Any]] = []
    for path in sorted((repo / TEST_DB_REL).glob("*.yml")):
        records.extend(_yaml_records(path, "l0"))
    for path in sorted((repo / QA_LISTS_REL).glob("*")):
        if path.suffix in (".yml", ".yaml"):
            records.extend(_yaml_records(path, "qa"))
        elif path.suffix == ".txt":
            records.extend(_text_records(path))
    return records


def extract_waives(repo: Path) -> list[dict[str, str]]:
    """``waives.txt`` entries; ``full:<GPU>/<test>`` waives the test on that GPU only."""
    path = Path(repo) / WAIVES_REL
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    waives = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        head, _, reason = text.partition(" ")
        gpu = _GPU_PREFIX.match(head)
        test = _GPU_PREFIX.sub("", head)
        waives.append(
            {
                "test": test,
                "path": repo_path(test),
                "node": test.split("::", 1)[1] if "::" in test else "",
                "gpu": gpu["gpu"] if gpu else "",
                "reason": reason.strip(),
            }
        )
    return waives


def extract_stages(repo: Path) -> dict[str, list[str]]:
    """Test-list name -> the Jenkins stages that run it; empty without ``jenkins/``."""
    path = Path(repo) / STAGES_REL
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    stages: dict[str, list[str]] = {}
    for line in lines:
        if line.lstrip().startswith("//"):
            continue
        match = _STAGE.search(line)
        if match:
            stages.setdefault(match["yml"], []).append(match["stage"])
    return stages


def _base(node: str) -> str:
    return node.split("[", 1)[0]


def lists(entry_path: str, node: str, test: str, ignore: tuple[str, ...] | list[str] = ()) -> bool:
    """Whether a test-list entry (``entry_path``, ``node``) runs ``test``, or part of it.

    ``test`` is a path or node id as written in a list or relative to the
    repository. An entry runs a test it names, every test under the file or
    directory it names except the paths it ``--ignore``s, and every
    parametrization of a test it names without parameters. An entry that names
    one parametrization (or one test of an asked class) runs part of what is
    asked, and counts.
    """
    path = repo_path(test)
    if any(path == skip or path.startswith(skip.rstrip("/") + "/") for skip in ignore):
        return False
    asked = test.split("::", 1)[1] if "::" in test else ""
    if entry_path != path:
        return path.startswith(entry_path.rstrip("/") + "/")
    if not node or not asked or asked == node:
        return True
    return any(
        longer.startswith(f"{shorter}::")
        or (longer.startswith(f"{shorter}[") and shorter == _base(longer))
        for shorter, longer in ((node, asked), (asked, node))
    )
