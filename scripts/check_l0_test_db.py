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
"""Check new L0 test-db entries are live, and keep untracked skips and duplicates out.

The check is diff-scoped against ``--base`` (default ``HEAD``). Entries that
were already off the schedule stay untouched until a change adds one. Three
checks, each a pre-commit hook:

``--liveness`` (the default) checks the test-db diff. It fails when:

1. A test-db entry is added to a block that no live ``jenkins/L0_Test.groovy``
   stage selects. A commented-out stage is not live, even when it is only
   commented out for temporary maintenance.
2. A new ``l0_*.yml``, or a new condition in an existing ``l0_*.yml``, has no
   live stage pointing at it.

``--untracked-skips`` checks the skips added in this diff. It fails when any test-db
entry a live stage selects would skip that test, unless the skip, a comment
next to it, or ``waives.txt`` cites an NVBug. The skips it recognizes are:

* ``pytest.skip(...)``, ``pytest.mark.skip`` / ``skipif``, ``unittest.skip``,
  ``skipIf`` / ``skipUnless`` and ``self.skipTest(...)``, in any position:
  decorator, ``pytest.param(marks=...)``, ``pytestmark`` or a call in a body.
* Named markers such as ``@skip_pre_blackwell`` or ``marks=skip_no_hopper``.
  They are resolved from the marker module for the test's suite
  (``tests/integration/defs/conftest.py`` or ``tests/unittest/utils/util.py``),
  and from assignments in the changed file itself, which win. A name that
  starts with ``skip_`` and is used as a decorator, in ``marks=`` or in
  ``pytestmark``, but cannot be resolved, fails closed.
* Marker factories and imperative helpers resolved the same way, such as
  ``@skip_num_gpus_less_than(4)`` and ``skip_fp4_pre_blackwell(use_fp4)``. The
  call arguments are substituted into the helper's condition.
* ``@pytest.mark.skip_less_device(N)`` and ``skip_less_mpi_world_size(N)``,
  judged against the stage's GPU count. The memory and device-name markers
  (``skip_less_device_memory``, ``skip_less_host_memory``,
  ``skip_device_not_contain``) declare hardware needs and are not judged.

A new definition of a named marker is judged through its uses in the same
file, which are all treated as added. A definition with no use in the file is
judged by its own condition. A change to a marker in a shared module does not
reach the files that use it.

A condition that compares ``get_sm_version()`` / ``getSMVersion()``, calls
``is_sm_100f()`` / ``isSM100Family()`` and the 120 variants, compares
``torch.cuda.device_count()`` / ``get_device_count()``, or tests
``TLLM_DISABLE_MPI`` / ``mpi_disabled()`` is judged per entry, so an entry
whose GPU, GPU count or orchestrator does not match is not a failure. A
condition this script cannot evaluate fails closed.

``--pre-post-duplicates`` fails when this diff lists the same test on a
pre-merge stage and a post-merge stage whose mako matches aside from
``stage``. A test that is already on both in ``--base`` is ignored.
"""

from __future__ import annotations

import argparse
import ast
import copy
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CBTS_ROOT = _REPO_ROOT / "jenkins" / "scripts" / "cbts"
if str(_CBTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_CBTS_ROOT))

import yaml  # noqa: E402
from cbts.blocks import (  # noqa: E402
    YAMLIndex,
    _entry_target,
    _extract_k_keyword,
    block_matches_stage,
    normalize_test_id,
    parse_stages_from_groovy,
)

_GROOVY = "jenkins/L0_Test.groovy"
_TEST_DB = "tests/integration/test_lists/test-db"
_WAIVES = "tests/integration/test_lists/waives.txt"
_NVBUG_RE = re.compile(r"https://nvbugs/\d+|nvbugspro\.nvidia\.com/bug/\d+", re.IGNORECASE)
_PURE_K_RE = re.compile(r'^-k "(?:not\s+[A-Za-z_]\w*)(?:\s+and\s+not\s+[A-Za-z_]\w*)*"$')
# Short (-k) and long (--ignore) pytest options. A single-dash pattern leaves
# --ignore glued to the test id, which hides the coverage change.
_OPTION_START_RE = re.compile(r"\s+-{1,2}[A-Za-z]")

# SM of each stage-name GPU token from derive_mako_from_stage. A platform
# absent from this map makes an SM comparison fail closed.
_GPU_SMS: dict[str, frozenset[int]] = {
    "a100x": frozenset({80}),
    "dgx_a100": frozenset({80}),
    "a30": frozenset({80}),
    "a10": frozenset({86}),
    "l40s": frozenset({89}),
    "h100_pcie": frozenset({90}),
    "dgx_h100": frozenset({90}),
    "dgx_h200": frozenset({90}),
    "gh200": frozenset({90}),
    "dgx_b200": frozenset({100}),
    "gb200": frozenset({100}),
    "b300": frozenset({103}),
    "dgx_b300": frozenset({103}),
    "gb300": frozenset({103}),
    "gr100": frozenset({107}),
    "rtx5090": frozenset({120}),
    "rtx5080": frozenset({120}),
    "rtxpro6000": frozenset({120}),
    "rtxpro6000d": frozenset({120}),
    "gb10": frozenset({121}),
}

# Modules that define the shared named skip markers. A test under
# ``tests/integration`` resolves a name from the first, any other test from
# the second, and each falls back to the other for a name only it defines.
_INTEGRATION_MARKERS = "tests/integration/defs/conftest.py"
_UNITTEST_MARKERS = "tests/unittest/utils/util.py"
_MARKER_FILES = (_INTEGRATION_MARKERS, _UNITTEST_MARKERS)

# Marker names, helper names and probes the condition evaluator understands.
_SM_PROBES = frozenset({"get_sm_version", "getSMVersion", "get_device_capability"})
_COUNT_PROBES = frozenset({"torch.cuda.device_count", "get_device_count"})
_SM_PREDICATES = {
    "is_sm_100f": lambda sm: 100 <= sm < 110,
    "isSM100Family": lambda sm: 100 <= sm < 110,
    "is_sm_120f": lambda sm: 120 <= sm < 130,
    "isSM120Family": lambda sm: 120 <= sm < 130,
}
_ENV_MPI_DISABLE = "TLLM_DISABLE_MPI"
_COUNT_MARKS = frozenset({"skip_less_device", "skip_less_mpi_world_size"})
# Bare names pytest and unittest export, mapped to the kind of skip they make.
_BARE_KINDS = {
    "skip": "skip",
    "skipif": "skipif",
    "skipIf": "skipif",
    "skipUnless": "skipunless",
}
# A skip that hits many stages names this many of them in its message.
_MAX_LISTED_ENTRIES = 5
_SKIP_NAME_RE = re.compile(r"^_?skip_\w+$")


def _stages_from_text(text: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "L0_Test.groovy"
        path.write_text(text)
        return parse_stages_from_groovy(path, include_post_merge=True)


def _index_from_texts(files: dict[str, str]) -> YAMLIndex:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp)
        for name, content in files.items():
            if Path(name).name.startswith("l0_") and name.endswith((".yml", ".yaml")):
                (db / Path(name).name).write_text(content)
        return YAMLIndex.load(db)


def _split_entry(entry: str) -> tuple[str, str]:
    """Return the test id and the pytest option suffix, separately."""
    normalized = normalize_test_id(entry)
    match = _OPTION_START_RE.search(normalized)
    body = normalized if match is None else normalized[: match.start()].rstrip()
    suffix = "" if match is None else " ".join(normalized[match.start() :].split())
    if "[" in body:
        body = body.rsplit("[", 1)[0]
    return (body.rstrip("/") or body), suffix


def _excluded_names(entry: str) -> set[str] | None:
    """Names a pure ``-k 'not a and not b'`` filter excludes.

    ``None`` means the pytest options are not that shape, so a change of
    those options cannot be diffed name-by-name.
    """
    _target, suffix = _split_entry(entry)
    if suffix == "":
        return set()
    if _PURE_K_RE.fullmatch(suffix) is None:
        return None
    keyword = _extract_k_keyword(normalize_test_id(entry))
    if keyword is None:
        return None
    return set(re.findall(r"\bnot\s+([A-Za-z_]\w*)", keyword))


def _waive_ids(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not _NVBUG_RE.search(stripped):
            continue
        test_id = normalize_test_id(stripped)
        if test_id:
            found.append(test_id)
    return found


def _word_in(token: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(token)}(?![\w])", text) is not None


def _entry_pairs(index: YAMLIndex) -> set[tuple[str, str]]:
    pairs = set()
    for block in index.blocks:
        for raw in block.tests:
            if not isinstance(raw, str):
                continue
            entry = normalize_test_id(raw)
            if entry:
                pairs.add((block.yaml_stem, entry))
    return pairs


def _id_without_options(entry: str) -> str:
    """Test id with its parametrize id, without pytest options like ``-k``."""
    normalized = normalize_test_id(entry)
    match = _OPTION_START_RE.search(normalized)
    body = normalized if match is None else normalized[: match.start()].rstrip()
    return body.rstrip("/") or body


def _mako_without_stage(stage) -> tuple[tuple[str, str], ...]:
    """Stage mako aside from pre-merge vs post-merge."""
    return tuple(sorted((key, value) for key, value in stage.mako.items() if key != "stage"))


def _pre_post_membership(stages: dict, index: YAMLIndex) -> dict[tuple, dict[str, set[str]]]:
    """Map ``(test, mako)`` to the pre-merge and post-merge stages that select it."""
    found: dict[tuple, dict[str, set[str]]] = {}
    for block in index.blocks:
        matched = [
            stage
            for stage in stages.values()
            if stage.yaml_stem == block.yaml_stem and block_matches_stage(block, stage)
        ]
        for stage in matched:
            term = stage.mako.get("stage")
            if term not in {"pre_merge", "post_merge"}:
                continue
            mako = _mako_without_stage(stage)
            for raw in block.tests:
                if not isinstance(raw, str):
                    continue
                entry = normalize_test_id(raw)
                if not entry:
                    continue
                # Keep the parametrize id: two cases of one function are different tests.
                target = _id_without_options(entry)
                bucket = found.setdefault((target, mako), {"pre_merge": set(), "post_merge": set()})
                bucket[term].add(stage.name)
    return found


def _duplicate_membership(
    membership: dict[tuple, dict[str, set[str]]],
) -> dict[tuple, dict[str, set[str]]]:
    return {
        key: value
        for key, value in membership.items()
        if value["pre_merge"] and value["post_merge"]
    }


def find_pre_post_issues(
    base_groovy: str,
    head_groovy: str,
    base_yamls: dict[str, str],
    head_yamls: dict[str, str],
) -> list[str]:
    """Return tests this diff newly listed on both phases for one mako.

    The mako identity is every stage field except ``stage``, so gpu, GPU
    count, backend, orchestrator, and auto_trigger must all match. A pair
    that already exists in ``base`` is not reported.
    """
    base_dups = _duplicate_membership(
        _pre_post_membership(_stages_from_text(base_groovy), _index_from_texts(base_yamls))
    )
    head_dups = _duplicate_membership(
        _pre_post_membership(_stages_from_text(head_groovy), _index_from_texts(head_yamls))
    )
    issues = []
    for target, mako in sorted(set(head_dups) - set(base_dups)):
        info = head_dups[(target, mako)]
        pre_stage = sorted(info["pre_merge"])[0]
        post_stage = sorted(info["post_merge"])[0]
        mako_text = ", ".join(f"{key}={value}" for key, value in mako)
        issues.append(
            f"'{target}' is listed on both pre-merge ({pre_stage}) and "
            f"post-merge ({post_stage}) with the same mako ({mako_text}). "
            "Keep it on one of those stages."
        )
    return issues


def _covered_pairs(index: YAMLIndex, stages: dict) -> set[tuple[str, str]]:
    covered = set()
    for block in index.blocks:
        matched = any(
            stage.yaml_stem == block.yaml_stem and block_matches_stage(block, stage)
            for stage in stages.values()
        )
        if not matched:
            continue
        for raw in block.tests:
            if isinstance(raw, str) and normalize_test_id(raw):
                covered.add((block.yaml_stem, normalize_test_id(raw)))
    return covered


def _extra_list_keys(text: str, stem: str) -> list[str]:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    keys = []
    for key, value in data.items():
        if key in {stem, "version"}:
            continue
        if isinstance(value, list) and value:
            keys.append(str(key))
    return keys


def find_list_issues(
    head_groovy: str,
    base_yamls: dict[str, str],
    head_yamls: dict[str, str],
) -> list[str]:
    """Return test-db additions that no live L0 stage in ``head_groovy`` selects.

    Only what the diff adds is judged: new entries, new ``l0_*.yml`` files, and
    new list keys. Entries that were dropped or narrowed are not reported.
    """
    head_stages = _stages_from_text(head_groovy)
    base_index = _index_from_texts(base_yamls)
    head_index = _index_from_texts(head_yamls)
    referenced = {stage.yaml_stem for stage in head_stages.values()}
    issues = []

    new_pairs = _entry_pairs(head_index) - _entry_pairs(base_index)
    covered = _covered_pairs(head_index, head_stages)
    for stem, entry in sorted(new_pairs - covered):
        issues.append(
            f"{stem}.yml: '{entry}' is not selected by any live L0 stage. "
            "Wire it in jenkins/L0_Test.groovy, or add it to a block a live "
            "stage already runs. A commented-out stage does not count."
        )

    for name in sorted(set(head_yamls) - set(base_yamls)):
        stem = Path(name).stem
        if stem in referenced:
            continue
        has_tests = any(block.yaml_stem == stem and block.tests for block in head_index.blocks)
        if has_tests:
            continue
        issues.append(f"{name}: new test-db file has no live L0 stage in jenkins/L0_Test.groovy.")

    for name, text in sorted(head_yamls.items()):
        stem = Path(name).stem
        base_keys = set(_extra_list_keys(base_yamls.get(name, ""), stem))
        for key in sorted(set(_extra_list_keys(text, stem)) - base_keys):
            if key in referenced:
                continue
            issues.append(
                f"{name}: test list '{key}' has no live L0 stage. "
                "Stages run the file stem, not an extra YAML key."
            )

    return issues


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


@dataclass(frozen=True)
class _ImportNames:
    """What a module imported that matters to skips."""

    roots: frozenset[str]  # names bound to the pytest / unittest module or ``mark``
    bare: dict  # local name -> kind, e.g. ``skipIf`` -> ``skipif``
    renames: dict  # ``import x as y`` of a skip_ name: y -> x


def _import_names(tree: ast.AST) -> _ImportNames:
    roots = {"pytest"}
    bare: dict[str, str] = {}
    renames: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"pytest", "unittest"}:
                    roots.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                if node.module in {"pytest", "unittest"}:
                    if alias.name in _BARE_KINDS:
                        bare[local] = _BARE_KINDS[alias.name]
                    elif alias.name == "mark":
                        roots.add(local)
                if alias.asname and _SKIP_NAME_RE.match(alias.name):
                    renames[alias.asname] = alias.name
    return _ImportNames(frozenset(roots), bare, renames)


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return _dotted(node.func)


def _probe_kind(node: ast.AST) -> str | None:
    """``sm`` for a GPU SM probe, ``count`` for a GPU count probe."""
    name = _call_name(node)
    if name is None:
        return None
    if name.rsplit(".", 1)[-1] in _SM_PROBES:
        return "sm"
    if name in _COUNT_PROBES or name.endswith(".get_device_count"):
        return "count"
    return None


def _is_mpi_env(node: ast.AST) -> bool:
    """``os.environ.get("TLLM_DISABLE_MPI")`` or ``os.getenv(...)``."""
    if _call_name(node) not in {"os.environ.get", "os.getenv"}:
        return False
    args = node.args  # type: ignore[attr-defined]
    return bool(args) and isinstance(args[0], ast.Constant) and args[0].value == _ENV_MPI_DISABLE


@dataclass(frozen=True)
class _Platform:
    """What a selecting stage tells us about the machine a test runs on."""

    sms: frozenset[int] | None = None
    gpu_count: int | None = None
    orchestrator: str | None = None

    def values(self, kind: str) -> frozenset[int] | None:
        if kind == "sm":
            return self.sms
        return None if self.gpu_count is None else frozenset({self.gpu_count})


def _const_int(node: ast.AST) -> int | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    return None


def _const_capability(node: ast.AST) -> int | None:
    """``(10, 0)`` as the SM number ``100``, for ``torch.cuda.get_device_capability()``."""
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        major, minor = _const_int(node.elts[0]), _const_int(node.elts[1])
        if major is not None and minor is not None and 0 <= minor < 10:
            return major * 10 + minor
    return None


def _const_ints(node: ast.AST) -> set[int] | None:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_const_int(elt) for elt in node.elts]
        if any(value is None for value in values):
            return None
        return set(values)
    return None


def _apply_cmp(op: ast.cmpop, left: int, right: int) -> bool | None:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    return None


def _sms_satisfy(values: frozenset[int] | None, predicate) -> bool | None:
    """Whether a skip condition matches this platform's probe values.

    One matching value is enough: the case skips on that platform. An unknown
    value set fails closed.
    """
    if not values:
        return None
    results = []
    for sm in values:
        result = predicate(sm)
        if result is None:
            return None
        results.append(result)
    if any(results):
        return True
    return False


def _eval_condition(node: ast.AST, platform: _Platform) -> bool | None:
    """Return whether ``node`` skips on this platform. ``None`` is unknown."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _eval_condition(node.operand, platform)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_eval_condition(value, platform) for value in node.values]
        if isinstance(node.op, ast.And):
            if any(value is False for value in values):
                return False
            if any(value is None for value in values):
                return None
            return True
        if isinstance(node.op, ast.Or):
            if any(value is True for value in values):
                return True
            if any(value is None for value in values):
                return None
            return False
        return None
    if isinstance(node, ast.Compare):
        return _eval_compare(node, platform)
    if isinstance(node, ast.Call):
        return _eval_call(node, platform)
    return None


def _eval_call(node: ast.Call, platform: _Platform) -> bool | None:
    """Boolean helpers skips use: ``is_sm_100f()``, ``mpi_disabled()``, ``torch.cuda.is_available()``."""
    short = (_call_name(node) or "").rsplit(".", 1)[-1]
    if node.args or node.keywords:
        return None
    if short in _SM_PREDICATES:
        return _sms_satisfy(platform.sms, _SM_PREDICATES[short])
    if _call_name(node) == "torch.cuda.is_available":
        return None if platform.gpu_count is None else platform.gpu_count > 0
    if short == "mpi_disabled":
        return None if platform.orchestrator is None else platform.orchestrator == "ray"
    return None


def _eval_compare(node: ast.Compare, platform: _Platform) -> bool | None:
    comparators = [node.left, *node.comparators]
    results = []
    for op, left, right in zip(node.ops, comparators, comparators[1:]):
        results.append(_eval_one_compare(op, left, right, platform))
    if any(result is False for result in results):
        return False
    if any(result is None for result in results):
        return None
    return True


def _eval_one_compare(
    op: ast.cmpop, left: ast.AST, right: ast.AST, platform: _Platform
) -> bool | None:
    env_left = _is_mpi_env(left)
    if env_left or _is_mpi_env(right):
        other = right if env_left else left
        if (
            platform.orchestrator is None
            or not isinstance(op, (ast.Eq, ast.NotEq))
            or not isinstance(other, ast.Constant)
            or other.value != "1"
        ):
            return None
        disabled = platform.orchestrator == "ray"
        return disabled if isinstance(op, ast.Eq) else not disabled
    kind_left = _probe_kind(left)
    kind = kind_left or _probe_kind(right)
    if kind is None:
        return None
    probe_on_left = kind_left is not None
    other = right if probe_on_left else left
    values = platform.values(kind)
    if isinstance(op, (ast.In, ast.NotIn)):
        members = _const_ints(other)
        if members is None:
            return None

        def _in_set(
            sm: int, members: set[int] = members, negated: bool = isinstance(op, ast.NotIn)
        ) -> bool:
            found = sm in members
            return not found if negated else found

        return _sms_satisfy(values, _in_set)
    number = _const_int(other)
    if number is None and (_call_name(left) or _call_name(right) or "").endswith(
        "get_device_capability"
    ):
        number = _const_capability(other)
    if number is None:
        return None

    def _numeric(
        sm: int, number: int = number, op: ast.cmpop = op, flip: bool = not probe_on_left
    ) -> bool | None:
        pair = (number, sm) if flip else (sm, number)
        return _apply_cmp(op, pair[0], pair[1])

    return _sms_satisfy(values, _numeric)


def _skip_call_kind(node: ast.Call, names: _ImportNames) -> str | None:
    """Kind of skip a call makes: ``skip``, ``skipif``, ``skipunless`` or ``lessdevice``."""
    func = node.func
    if isinstance(func, ast.Name):
        return names.bare.get(func.id)
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr == "skipTest":
        return "skip"
    base = _dotted(func.value)
    if base is None or base.split(".")[0] not in names.roots:
        return None
    if func.attr in {"skipif", "skipIf"}:
        return "skipif"
    if func.attr == "skipUnless":
        return "skipunless"
    if func.attr == "skip" and (base in names.roots or base.endswith(".mark")):
        return "skip"
    if func.attr in _COUNT_MARKS and (base in names.roots or base.endswith(".mark")):
        return "lessdevice"
    return None


def _overlaps(node: ast.AST, added: set[int]) -> bool:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None) or start
    if start is None or end is None:
        return False
    return any(line in added for line in range(start, end + 1))


def _enclosing_function(node: ast.AST, parents: dict) -> str | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return None


def _in_sequence(statements: list[ast.stmt], target: ast.AST) -> bool:
    for statement in statements:
        for child in ast.walk(statement):
            if child is target:
                return True
    return False


def _own_condition(call: ast.Call, kind: str) -> ast.expr:
    """The condition a skip call carries by itself, before any ``if`` around it."""
    if kind in {"skipif", "skipunless"}:
        if call.args:
            condition = call.args[0]
        else:
            condition = next(
                (item.value for item in call.keywords if item.arg == "condition"), None
            )
        if condition is None:
            return ast.Constant(value=None)
        if kind == "skipunless":
            return ast.UnaryOp(op=ast.Not(), operand=condition)
        return condition
    if kind == "lessdevice":
        limit = _const_int(call.args[0]) if call.args else None
        if limit is None:
            return ast.Constant(value=None)
        return ast.parse(f"get_device_count() < {limit}", mode="eval").body
    return ast.Constant(value=True)


def _with_context(node: ast.AST, own: ast.expr, parents: dict) -> ast.expr:
    """``own`` and every ``if`` test that guards ``node``, joined by ``and``."""
    parts: list[ast.expr] = [own]
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.If):
            if _in_sequence(parent.body, node):
                parts.append(parent.test)
            elif _in_sequence(parent.orelse, node):
                parts.append(ast.UnaryOp(op=ast.Not(), operand=parent.test))
        current = parent
    acc = parts[0]
    for part in parts[1:]:
        acc = ast.BoolOp(op=ast.And(), values=[acc, part])
    return acc


@dataclass
class _MarkerDef:
    """A named skip: a marker object, or a helper that makes or applies one."""

    condition: ast.expr
    params: tuple[str, ...] | None  # None for a plain marker, else the helper's parameters
    segment: str = ""  # source of a marker assignment, which may cite an NVBug in its reason
    node: ast.AST | None = None  # set for a definition in the file being checked


class _Substitute(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, ast.expr]):
        self._mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in self._mapping:
            return copy.deepcopy(self._mapping[node.id])
        return node


def _bind(defn: _MarkerDef, call: ast.Call) -> ast.expr:
    """The helper's condition with the call's arguments in place of its parameters."""
    if defn.params is None:
        return defn.condition
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        item.arg is None for item in call.keywords
    ):
        return ast.Constant(value=None)
    mapping: dict[str, ast.expr] = dict(zip(defn.params, call.args))
    mapping.update({item.arg: item.value for item in call.keywords if item.arg})
    return _Substitute(mapping).visit(copy.deepcopy(defn.condition))


def _is_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = _dotted(target)
        if name is not None and name.rsplit(".", 1)[-1] in {"fixture", "yield_fixture"}:
            return True
    return False


def _collect_marker_defs(
    tree: ast.AST,
    source: str,
    names: _ImportNames,
    parents: dict,
    *,
    local: bool,
) -> dict[str, _MarkerDef]:
    """Named skips a module defines.

    A marker assignment, ``skip_x = pytest.mark.skipif(...)``, is always read.
    A ``skip_*`` helper function is read only from a shared marker module
    (``local`` false): a helper in the file being checked is left opaque, as are
    memory checks and other conditions that only the helper's body can tell.
    """
    defs: dict[str, _MarkerDef] = {}
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ]
    for node in assigns:
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        kind = _skip_call_kind(value, names)
        if kind is None:
            continue
        defs[node.targets[0].id] = _MarkerDef(
            _own_condition(value, kind),
            None,
            ast.get_source_segment(source, node) or "",
            node if local else None,
        )
    if not local:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _SKIP_NAME_RE.match(node.name) or _is_fixture(node):
                continue
            conditions = []
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    kind = _skip_call_kind(inner, names)
                    if kind is not None:
                        conditions.append(
                            _with_context(inner, _own_condition(inner, kind), parents)
                        )
            if not conditions:
                continue
            condition = conditions[0]
            for other in conditions[1:]:
                condition = ast.BoolOp(op=ast.Or(), values=[condition, other])
            args = node.args
            params = tuple(arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs])
            defs[node.name] = _MarkerDef(condition, params)
    # ``skip_x = skip_helper(2)`` and ``skip_y = skip_x`` come after the helpers they name.
    for node in assigns:
        target = node.targets[0].id
        if target in defs:
            continue
        value = node.value
        defn = None
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            helper = defs.get(value.func.id)
            if helper is not None and helper.params is not None:
                defn = _MarkerDef(
                    _bind(helper, value),
                    None,
                    ast.get_source_segment(source, node) or "",
                    node if local else None,
                )
        elif isinstance(value, ast.Name) and value.id in defs:
            defn = defs[value.id]
        if defn is not None:
            defs[target] = defn
    return defs


def _parse_marker_file(text: str) -> dict[str, _MarkerDef]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return _collect_marker_defs(tree, text, _import_names(tree), parents, local=False)


def _in_mark_context(node: ast.AST, parents: dict) -> bool:
    """Whether ``node`` sits where a mark goes: decorator, ``marks=`` or ``pytestmark``."""
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return current in parent.decorator_list
        if isinstance(parent, ast.keyword) and parent.arg == "marks":
            return True
        if isinstance(parent, ast.Dict):
            for key, value in zip(parent.keys, parent.values):
                if value is current and isinstance(key, ast.Constant) and key.value == "marks":
                    return True
        if isinstance(parent, ast.Assign):
            return any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in parent.targets)
        if isinstance(parent, ast.stmt):
            return False
        current = parent
    return False


@dataclass
class _Ref:
    node: ast.AST  # the Name / Attribute, or the Call that applies it
    name: str
    defn: _MarkerDef | None
    condition: ast.expr


def _marker_refs(
    tree: ast.AST,
    names: _ImportNames,
    parents: dict,
    local_defs: dict[str, _MarkerDef],
    shared_defs: list[dict[str, _MarkerDef]],
) -> list[_Ref]:
    """Every use of a named skip, whether it was added or not."""
    refs: list[_Ref] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                continue
            name = names.renames.get(node.id, node.id)
        elif isinstance(node, ast.Attribute):
            base = _dotted(node.value)
            if base is not None and base.split(".")[0] in names.roots:
                continue  # pytest.mark.skip_less_device and the like
            name = node.attr
        else:
            continue
        if not _SKIP_NAME_RE.match(name) or "memory" in name:
            continue
        parent = parents.get(node)
        applied = parent if isinstance(parent, ast.Call) and parent.func is node else None
        site_node = applied or node
        defn = local_defs.get(name)
        if defn is None:
            defn = next((defs[name] for defs in shared_defs if name in defs), None)
        if defn is None:
            if not _in_mark_context(site_node, parents):
                continue
            condition: ast.expr = ast.Constant(value=None)
        elif defn.params is None:
            condition = defn.condition
        elif applied is not None:
            condition = _bind(defn, applied)
        else:
            continue
        refs.append(_Ref(site_node, name, defn, _with_context(site_node, condition, parents)))
    return refs


@dataclass
class _SkipSite:
    lineno: int
    func_name: str | None
    condition: ast.expr
    segment: str
    yaml_key: str | None


def _site_text(source_lines: list[str], node: ast.AST, extra: str = "") -> str:
    """Source of a skip plus the comment lines that sit directly above it."""
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", None) or start
    above = []
    index = start - 2
    while index >= 0 and source_lines[index].lstrip().startswith("#"):
        above.append(source_lines[index])
        index -= 1
    return "\n".join([*reversed(above), *source_lines[start - 1 : end], extra])


def _iter_skip_sites(
    source: str,
    added: set[int],
    yaml_key: str | None,
    shared_defs: list[dict[str, _MarkerDef]] | None = None,
) -> list[_SkipSite]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = _import_names(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    lines = source.splitlines()
    local_defs = _collect_marker_defs(tree, source, names, parents, local=True)
    refs = _marker_refs(tree, names, parents, local_defs, shared_defs or [])

    # A marker defined here and used here is judged where it is used. A
    # definition nobody uses in this file is judged by its own condition.
    used = {ref.name for ref in refs}
    covered = [
        (defn.node.lineno, defn.node.end_lineno or defn.node.lineno)
        for name, defn in local_defs.items()
        if name in used and defn.node is not None
    ]
    changed = {
        name
        for name, defn in local_defs.items()
        if defn.node is not None and _overlaps(defn.node, added)
    }

    def _in_covered(node: ast.AST) -> bool:
        line = getattr(node, "lineno", 0)
        return any(start <= line <= end for start, end in covered)

    sites: list[_SkipSite] = []
    for node in ast.walk(tree):
        kind = None
        condition = None
        if isinstance(node, ast.Call):
            kind = _skip_call_kind(node, names)
            if kind is None or not _overlaps(node, added) or _in_covered(node):
                continue
            condition = _with_context(node, _own_condition(node, kind), parents)
        elif isinstance(node, ast.Attribute) and node.attr == "skip":
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            base = _dotted(node.value)
            if base is None or not base.endswith(".mark"):
                continue
            if base.split(".")[0] not in names.roots:
                continue
            if not _overlaps(node, added) or _in_covered(node):
                continue
            kind = "skip"
            condition = ast.Constant(value=True)
        else:
            continue
        segment = ast.get_source_segment(source, node) or ""
        sites.append(
            _SkipSite(
                getattr(node, "lineno", 1),
                _enclosing_function(node, parents),
                condition,
                segment + "\n" + _site_text(lines, node),
                yaml_key,
            )
        )
    for ref in refs:
        local = ref.defn is not None and local_defs.get(ref.name) is ref.defn
        if not (_overlaps(ref.node, added) or (local and ref.name in changed)):
            continue
        extra = ref.defn.segment if ref.defn is not None else ""
        sites.append(
            _SkipSite(
                getattr(ref.node, "lineno", 1),
                _enclosing_function(ref.node, parents),
                ref.condition,
                _site_text(lines, ref.node, extra),
                yaml_key,
            )
        )
    sites.sort(key=lambda site: site.lineno)
    return sites


@dataclass(frozen=True)
class _DbEntry:
    yaml_stem: str
    entry: str
    platform: str
    stage: str
    gpu_count: int | None = None
    orchestrator: str | None = None


def _gpu_count(stage) -> int | None:
    value = stage.mako.get("system_gpu_count")
    return int(value) if value is not None and value.isdigit() else None


def _names_other_test(target: str, yaml_key: str, func_name: str | None) -> bool:
    """Whether an entry for one test function of the file cannot run ``func_name``.

    ``file.py::test_a`` never runs a skip on ``test_b``. A class, a bare file
    or a directory entry could, and so could a skip in a helper that is not
    a test, so those are kept.
    """
    if not func_name or not func_name.startswith("test"):
        return False
    prefix = yaml_key + "::"
    if not target.startswith(prefix):
        return False
    last = target[len(prefix) :].split("::")[-1].split("[")[0]
    return last.startswith("test") and last != func_name


def _selecting_entries(
    index: YAMLIndex, stages: dict, git_path: str, func_name: str | None
) -> list[_DbEntry]:
    """Every test-db entry a live stage selects that covers this test."""
    yaml_key = index.git_path_to_yaml_key(git_path)
    if not yaml_key:
        return []
    matches = index.find_match_for_path(yaml_key) or []
    found: list[_DbEntry] = []
    seen: set[_DbEntry] = set()
    for block, _prefix in matches:
        matched_stages = [
            stage
            for stage in stages.values()
            if stage.yaml_stem == block.yaml_stem and block_matches_stage(block, stage)
        ]
        if not matched_stages:
            continue
        for raw in block.tests:
            if not isinstance(raw, str):
                continue
            target = _entry_target(raw)
            key_in_entry = (
                yaml_key == target
                or yaml_key.startswith(target + "/")
                or yaml_key.startswith(target + "::")
            )
            entry_in_key = (
                target == yaml_key
                or target.startswith(yaml_key + "/")
                or target.startswith(yaml_key + "::")
            )
            if not key_in_entry and not entry_in_key:
                continue
            if _names_other_test(target, yaml_key, func_name):
                continue
            excluded = _excluded_names(raw)
            if func_name and excluded is not None and func_name in excluded:
                continue
            entry = normalize_test_id(raw)
            for stage in matched_stages:
                item = _DbEntry(
                    block.yaml_stem,
                    entry,
                    stage.mako.get("gpu") or stage.name,
                    stage.name,
                    _gpu_count(stage),
                    stage.mako.get("orchestrator"),
                )
                if item not in seen:
                    seen.add(item)
                    found.append(item)
    return found


def _skip_is_tracked(site: _SkipSite, waive_ids: list[str]) -> bool:
    if _NVBUG_RE.search(site.segment or ""):
        return True
    yaml_key = site.yaml_key or ""
    file_name = yaml_key.rsplit("/", 1)[-1]
    for waive_id in waive_ids:
        if site.func_name and not _word_in(site.func_name, waive_id):
            continue
        if yaml_key and (yaml_key in waive_id or (file_name and file_name in waive_id)):
            return True
    return False


def _shared_marker_defs(
    path: str, marker_sources: dict[str, str], cache: dict[str, dict[str, _MarkerDef]]
) -> list[dict[str, _MarkerDef]]:
    """Shared marker modules, the one for ``path``'s suite first."""
    order = (
        (_INTEGRATION_MARKERS, _UNITTEST_MARKERS)
        if path.startswith("tests/integration/")
        else (_UNITTEST_MARKERS, _INTEGRATION_MARKERS)
    )
    found = []
    for marker_path in order:
        text = marker_sources.get(marker_path)
        if text is None:
            continue
        if marker_path not in cache:
            cache[marker_path] = _parse_marker_file(text)
        found.append(cache[marker_path])
    return found


def find_skip_issues(
    head_groovy: str,
    head_yamls: dict[str, str],
    head_waives: str,
    sources: dict[str, str],
    added_lines: dict[str, set[int]],
    marker_sources: dict[str, str] | None = None,
) -> list[str]:
    """Return new skips that would skip any selected test-db entry.

    An entry counts when a live stage selects its block. The skip is allowed
    when its text, a comment above it, or ``waives.txt`` cites an NVBug. A
    condition on the SM version, the GPU count or the orchestrator that is
    false for an entry's stage does not count for that entry.

    ``marker_sources`` maps the shared marker modules' repo paths to their
    text, so named markers such as ``skip_pre_blackwell`` can be resolved.
    """
    stages = _stages_from_text(head_groovy)
    index = _index_from_texts(head_yamls)
    waives = _waive_ids(head_waives)
    marker_sources = marker_sources or {}
    marker_cache: dict[str, dict[str, _MarkerDef]] = {}
    issues = []
    for path, source in sorted(sources.items()):
        added = added_lines.get(path, set())
        if not added:
            continue
        yaml_key = index.git_path_to_yaml_key(path)
        shared = _shared_marker_defs(path, marker_sources, marker_cache)
        for site in _iter_skip_sites(source, added, yaml_key, shared):
            if _skip_is_tracked(site, waives):
                continue
            skipped = []
            for item in _selecting_entries(index, stages, path, site.func_name):
                platform = _Platform(_GPU_SMS.get(item.platform), item.gpu_count, item.orchestrator)
                result = _eval_condition(site.condition, platform)
                if result is False:
                    continue
                detail = f"'{item.entry}' in {item.yaml_stem} on {item.platform} ({item.stage})"
                if result is None:
                    detail += ", unevaluated condition"
                skipped.append(detail)
            if not skipped:
                continue
            where = site.func_name or "module level"
            shown = skipped[:_MAX_LISTED_ENTRIES]
            if len(skipped) > len(shown):
                shown.append(f"{len(skipped) - len(shown)} more stage(s)")
            issues.append(
                f"{path}:{site.lineno}: new skip in {where} skips "
                f"{'; '.join(shown)}. Cite https://nvbugs/<id> in the skip, "
                "in a comment above it, or add the test to waives.txt. A skip "
                "this check cannot evaluate fails closed."
            )
    return issues


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _git_show(repo: Path, rev: str, path: str) -> str:
    result = _git(repo, "show", f"{rev}:{path}")
    if result.returncode != 0:
        return ""
    return result.stdout


def _git_yamls(repo: Path, rev: str) -> dict[str, str]:
    listing = _git(repo, "ls-tree", "-r", "--name-only", rev, _TEST_DB)
    if listing.returncode != 0:
        raise RuntimeError(listing.stderr.strip() or "git ls-tree failed")
    files = {}
    for path in listing.stdout.splitlines():
        if Path(path).name.startswith("l0_") and path.endswith((".yml", ".yaml")):
            files[Path(path).name] = _git_show(repo, rev, path)
    return files


def _worktree_yamls(repo: Path) -> dict[str, str]:
    db = repo / _TEST_DB
    files = {}
    if not db.is_dir():
        return files
    for path in sorted(db.glob("l0_*.yml")) + sorted(db.glob("l0_*.yaml")):
        files[path.name] = path.read_text()
    return files


def _added_lines(diff: str) -> dict[str, set[int]]:
    found: dict[str, set[int]] = defaultdict(set)
    path = None
    new_line = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            new_line = int(match.group(1)) if match else None
            continue
        if path is None or new_line is None or line.startswith("\\"):
            continue
        if line.startswith("+"):
            found[path].add(new_line)
            new_line += 1
        elif line.startswith("-"):
            continue
        else:
            new_line += 1
    return found


def _head_lists(repo: Path) -> tuple[str, dict[str, str], str]:
    groovy = (repo / _GROOVY).read_text() if (repo / _GROOVY).is_file() else ""
    waives = (repo / _WAIVES).read_text() if (repo / _WAIVES).is_file() else ""
    return groovy, _worktree_yamls(repo), waives


def collect_list_issues(repo: Path, base: str) -> list[str]:
    """Diff the test-db against the live stages. Do not look at pytest skips."""
    list_diff = _git(
        repo,
        "diff",
        "--name-only",
        base,
        "--",
        _GROOVY,
        _TEST_DB,
    )
    if list_diff.returncode != 0:
        raise RuntimeError(list_diff.stderr.strip() or "git diff failed")
    if not any(list_diff.stdout.splitlines()):
        return []
    head_groovy, head_yamls, _head_waives = _head_lists(repo)
    return find_list_issues(head_groovy, _git_yamls(repo, base), head_yamls)


def collect_skip_issues(repo: Path, base: str) -> list[str]:
    """Diff added skips of any form. Do not judge test-db list edits."""
    py_diff = _git(repo, "diff", "-U0", "--no-color", base, "--", "tests")
    if py_diff.returncode != 0:
        raise RuntimeError(py_diff.stderr.strip() or "git diff failed")
    added = {
        path: lines for path, lines in _added_lines(py_diff.stdout).items() if path.endswith(".py")
    }
    sources = {}
    for path, lines in added.items():
        file_path = repo / path
        if not lines or not file_path.is_file():
            continue
        source = file_path.read_text()
        if "skip" not in source:
            continue
        sources[path] = source
    if not sources:
        return []
    marker_sources = {
        marker_path: (repo / marker_path).read_text()
        for marker_path in _MARKER_FILES
        if (repo / marker_path).is_file()
    }
    head_groovy, head_yamls, head_waives = _head_lists(repo)
    return find_skip_issues(head_groovy, head_yamls, head_waives, sources, added, marker_sources)


def collect_pre_post_issues(repo: Path, base: str) -> list[str]:
    """Diff ``base`` against the worktree and return new pre/post duplicates."""
    head_groovy = (repo / _GROOVY).read_text() if (repo / _GROOVY).is_file() else ""
    return find_pre_post_issues(
        _git_show(repo, base, _GROOVY),
        head_groovy,
        _git_yamls(repo, base),
        _worktree_yamls(repo),
    )


# mode -> (collector, failure summary appended after the issue count)
_MODES = {
    "liveness": (
        collect_list_issues,
        "test-db liveness check(s) failed. Keep each new entry on a block a live L0 stage selects.",
    ),
    "untracked-skips": (
        collect_skip_issues,
        "untracked skip check(s) failed. Cite an NVBug in the skip or in waives.txt.",
    ),
    "pre-post-duplicates": (
        collect_pre_post_issues,
        "pre/post duplicate check(s) failed. Keep each test on one of the two stages.",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="HEAD",
        help="Git revision to diff against (default: HEAD, for the pre-commit hook).",
    )
    parser.add_argument(
        "--repo-root",
        default=str(_REPO_ROOT),
        help="Repository root. Defaults to the directory that contains this script.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--liveness",
        dest="mode",
        action="store_const",
        const="liveness",
        help=(
            "Fail when the test-db diff adds an entry, file, or condition that no live "
            "stage selects (default)."
        ),
    )
    mode.add_argument(
        "--untracked-skips",
        dest="mode",
        action="store_const",
        const="untracked-skips",
        help=(
            "Fail when a new skip (pytest.skip/skipif, unittest skips, or a named marker "
            "such as skip_pre_blackwell) would skip a test-db entry a live stage selects."
        ),
    )
    mode.add_argument(
        "--pre-post-duplicates",
        dest="mode",
        action="store_const",
        const="pre-post-duplicates",
        help="Fail when this diff lists a test on both pre-merge and post-merge for the same mako.",
    )
    parser.set_defaults(mode="liveness")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root)
    collect, failure_hint = _MODES[args.mode]
    try:
        issues = collect(repo, args.base)
    except (OSError, RuntimeError, yaml.YAMLError) as exc:
        print(f"test-db check could not run: {exc}", file=sys.stderr)
        return 2
    if not issues:
        return 0
    print("\n".join(issues), file=sys.stderr)
    print(f"\n{len(issues)} {failure_hint}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
