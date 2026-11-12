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
"""Fail a commit that drops L0 coverage without tracking it.

The check is diff-scoped against ``--base`` (default ``HEAD``). Entries that
were already off the schedule stay untouched until this change adds one or
stops selecting one. Three modes, each a pre-commit hook:

The default mode checks the test-db diff. It fails when:

1. A test-db entry is added to a block that no live ``jenkins/L0_Test.groovy``
   stage selects (commented-out stages are not live).
2. A new ``l0_*.yml`` has no live stage pointing at it.
3. An existing entry stops being selected on a platform, including a tighter
   ``-k`` / ``--ignore`` / condition.

A dropped list entry is allowed when ``waives.txt`` already tracks it with an
NVBug.

``--skips`` checks ``pytest.skip`` / ``skipif`` added in this diff. It fails
when any test-db entry a live stage selects would skip that test, unless the
skip reason or ``waives.txt`` cites an NVBug. A ``skipif`` that is a simple
``get_sm_version()`` comparison is judged per entry, so an entry whose GPU
does not match is not a failure. A condition this script cannot evaluate
fails closed.

``--pre-post-duplicates`` fails when this diff lists the same test on a
pre-merge stage and a post-merge stage whose mako matches aside from
``stage``. A test that is already on both in ``--base`` is ignored.
"""

from __future__ import annotations

import argparse
import ast
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
    "dgx_b200": frozenset({100}),
    "b300": frozenset({103}),
    "dgx_b300": frozenset({103}),
    "rtx5090": frozenset({120}),
    "rtx5080": frozenset({120}),
    "rtxpro6000": frozenset({120}),
    "rtxpro6000d": frozenset({120}),
}


@dataclass
class _Sel:
    target: str
    platform: str
    exclusions: set[str] | None
    suffix: str
    stage: str
    entry: str


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


def _tracks_entry(waive_ids: list[str], target: str) -> bool:
    """True when a waive covers this list entry, or one test in a dropped file."""
    for waive_id in waive_ids:
        if (
            waive_id == target
            or target.startswith(waive_id + "/")
            or target.startswith(waive_id + "::")
        ):
            return True
        if target.endswith(".py") and (
            waive_id.startswith(target + "::") or waive_id.startswith(target + "/")
        ):
            return True
    return False


def _tracks_token(waive_ids: list[str], token: str, target: str) -> bool:
    for waive_id in waive_ids:
        if not _word_in(token, waive_id):
            continue
        if target in waive_id or waive_id.startswith(target):
            return True
    return False


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


def _selections(index: YAMLIndex, stages: dict) -> dict[tuple[str, str], list[_Sel]]:
    grouped: dict[tuple[str, str], list[_Sel]] = defaultdict(list)
    for block in index.blocks:
        matched = [
            stage
            for stage in stages.values()
            if stage.yaml_stem == block.yaml_stem and block_matches_stage(block, stage)
        ]
        if not matched:
            continue
        for raw in block.tests:
            if not isinstance(raw, str):
                continue
            entry = normalize_test_id(raw)
            if not entry:
                continue
            target, suffix = _split_entry(entry)
            exclusions = _excluded_names(entry)
            for stage in matched:
                platform = stage.mako.get("gpu") or stage.name
                grouped[(target, platform)].append(
                    _Sel(target, platform, exclusions, suffix, stage.name, entry)
                )
    return grouped


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


def _always_excluded(selections: list[_Sel]) -> set[str] | None:
    acc: set[str] | None = None
    for selection in selections:
        if selection.exclusions is None:
            return None
        acc = set(selection.exclusions) if acc is None else acc & selection.exclusions
    return acc if acc is not None else set()


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
    base_groovy: str,
    head_groovy: str,
    base_yamls: dict[str, str],
    head_yamls: dict[str, str],
    base_waives: str,
    head_waives: str,
) -> list[str]:
    """Return coverage regressions between two test-db snapshots.

    ``base_waives`` is accepted so callers can pass the whole snapshot.
    Tracking uses the head waive list, which is what will land.
    """
    del base_waives  # Tracking is whatever waives.txt contains after this change.
    base_stages = _stages_from_text(base_groovy)
    head_stages = _stages_from_text(head_groovy)
    base_index = _index_from_texts(base_yamls)
    head_index = _index_from_texts(head_yamls)
    waives = _waive_ids(head_waives)
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

    base_sel = _selections(base_index, base_stages)
    head_sel = _selections(head_index, head_stages)
    for target, platform in sorted(set(base_sel) - set(head_sel)):
        if _tracks_entry(waives, target):
            continue
        stage = sorted({item.stage for item in base_sel[(target, platform)]})[0]
        issues.append(
            f"'{target}' no longer runs on {platform} (was {stage}). "
            "Keep it on a live stage for that platform, or add a waives.txt "
            "line with https://nvbugs/<id>."
        )

    for key in sorted(set(base_sel) & set(head_sel)):
        target, platform = key
        before = base_sel[key]
        after = head_sel[key]
        before_excluded = _always_excluded(before)
        after_excluded = _always_excluded(after)
        before_suffix = {item.suffix for item in before}
        after_suffix = {item.suffix for item in after}
        stage = sorted({item.stage for item in after})[0]
        if before_excluded is None or after_excluded is None:
            if before_suffix != after_suffix and not _tracks_entry(waives, target):
                issues.append(
                    f"'{target}' on {platform} ({stage}) changed pytest options "
                    "in a way this check cannot diff. Waive the dropped cases "
                    "with https://nvbugs/<id>, or keep a plain "
                    "`not name and not name` -k filter."
                )
            continue
        for token in sorted(after_excluded - before_excluded):
            if _tracks_token(waives, token, target):
                continue
            issues.append(
                f"'{target}' on {platform} ({stage}) newly excludes '{token}'. "
                "Waive that case with https://nvbugs/<id>, or keep it selected."
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


def _pytest_aliases(tree: ast.AST) -> set[str]:
    aliases = {"pytest"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest":
                    aliases.add(alias.asname or "pytest")
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            for alias in node.names:
                if alias.name in {"skip", "skipif"}:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _is_sm_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _dotted(node.func)
    return name is not None and (name == "get_sm_version" or name.endswith(".get_sm_version"))


def _const_int(node: ast.AST) -> int | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
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


def _sms_satisfy(sms: frozenset[int] | None, predicate) -> bool | None:
    """Whether a skip condition matches this platform's SM set.

    One matching SM is enough: the case skips on that platform. An unknown
    SM map fails closed.
    """
    if not sms:
        return None
    results = []
    for sm in sms:
        result = predicate(sm)
        if result is None:
            return None
        results.append(result)
    if any(results):
        return True
    return False


def _eval_condition(node: ast.AST, sms: frozenset[int] | None) -> bool | None:
    """Return whether ``node`` skips on this platform. ``None`` is unknown."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _eval_condition(node.operand, sms)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_eval_condition(value, sms) for value in node.values]
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
        return _eval_compare(node, sms)
    return None


def _eval_compare(node: ast.Compare, sms: frozenset[int] | None) -> bool | None:
    comparators = [node.left, *node.comparators]
    results = []
    for op, left, right in zip(node.ops, comparators, comparators[1:]):
        results.append(_eval_one_compare(op, left, right, sms))
    if any(result is False for result in results):
        return False
    if any(result is None for result in results):
        return None
    return True


def _eval_one_compare(
    op: ast.cmpop, left: ast.AST, right: ast.AST, sms: frozenset[int] | None
) -> bool | None:
    sm_left = _is_sm_call(left)
    sm_right = _is_sm_call(right)
    if sm_left or sm_right:
        other = right if sm_left else left
        if isinstance(op, (ast.In, ast.NotIn)):
            values = _const_ints(other)
            if values is None:
                return None

            def _in_set(
                sm: int, values: set[int] = values, negated: bool = isinstance(op, ast.NotIn)
            ) -> bool:
                found = sm in values
                return not found if negated else found

            return _sms_satisfy(sms, _in_set)
        number = _const_int(other)
        if number is None:
            return None

        def _numeric(
            sm: int, number: int = number, op: ast.cmpop = op, flip: bool = not sm_left
        ) -> bool | None:
            pair = (number, sm) if flip else (sm, number)
            return _apply_cmp(op, pair[0], pair[1])

        return _sms_satisfy(sms, _numeric)
    return None


def _skip_call_kind(node: ast.Call, aliases: set[str]) -> str | None:
    func = node.func
    if isinstance(func, ast.Name) and func.id in aliases and func.id in {"skip", "skipif"}:
        return "skipif" if func.id == "skipif" else "skip"
    if not isinstance(func, ast.Attribute):
        return None
    base = _dotted(func.value)
    if base is None:
        return None
    parts = base.split(".")
    if not any(alias == parts[0] for alias in aliases if alias not in {"skip", "skipif"}):
        return None
    if func.attr == "skipif":
        return "skipif"
    if func.attr == "skip" and (base in aliases or base.endswith(".mark")):
        return "skip"
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


def _condition_for_call(call: ast.Call, kind: str, parents: dict) -> ast.expr:
    parts: list[ast.expr] = []
    if kind == "skipif":
        if call.args:
            parts.append(call.args[0])
        else:
            keyword = next((item.value for item in call.keywords if item.arg == "condition"), None)
            parts.append(keyword if keyword is not None else ast.Constant(value=None))
    current = call
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.If):
            if _in_sequence(parent.body, call):
                parts.append(parent.test)
            elif _in_sequence(parent.orelse, call):
                parts.append(ast.UnaryOp(op=ast.Not(), operand=parent.test))
        current = parent
    if not parts:
        return ast.Constant(value=True)
    acc = parts[0]
    for part in parts[1:]:
        acc = ast.BoolOp(op=ast.And(), values=[acc, part])
    return acc


@dataclass
class _SkipSite:
    lineno: int
    func_name: str | None
    condition: ast.expr
    segment: str
    yaml_key: str | None


def _iter_skip_sites(source: str, added: set[int], yaml_key: str | None) -> list[_SkipSite]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    aliases = _pytest_aliases(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    sites = []
    for node in ast.walk(tree):
        kind = None
        condition = None
        if isinstance(node, ast.Call):
            kind = _skip_call_kind(node, aliases)
            if kind is None or not _overlaps(node, added):
                continue
            condition = _condition_for_call(node, kind, parents)
        elif isinstance(node, ast.Attribute) and node.attr == "skip":
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                continue
            base = _dotted(node.value)
            if base is None or not base.endswith(".mark"):
                continue
            if not any(base.split(".")[0] == alias for alias in aliases):
                continue
            if not _overlaps(node, added):
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
                segment,
                yaml_key,
            )
        )
    return sites


@dataclass(frozen=True)
class _DbEntry:
    yaml_stem: str
    entry: str
    platform: str
    stage: str


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


def find_skip_issues(
    head_groovy: str,
    head_yamls: dict[str, str],
    head_waives: str,
    sources: dict[str, str],
    added_lines: dict[str, set[int]],
) -> list[str]:
    """Return new skips that would skip any selected test-db entry.

    An entry counts when a live stage selects its block. The skip is allowed
    when its text or ``waives.txt`` cites an NVBug. A ``get_sm_version()``
    comparison that is false for an entry's GPU does not count for that entry.
    """
    stages = _stages_from_text(head_groovy)
    index = _index_from_texts(head_yamls)
    waives = _waive_ids(head_waives)
    issues = []
    for path, source in sorted(sources.items()):
        added = added_lines.get(path, set())
        if not added:
            continue
        yaml_key = index.git_path_to_yaml_key(path)
        for site in _iter_skip_sites(source, added, yaml_key):
            if _skip_is_tracked(site, waives):
                continue
            skipped = []
            for item in _selecting_entries(index, stages, path, site.func_name):
                result = _eval_condition(site.condition, _GPU_SMS.get(item.platform))
                if result is False:
                    continue
                detail = f"'{item.entry}' in {item.yaml_stem} on {item.platform} ({item.stage})"
                if result is None:
                    detail += ", unevaluated condition"
                skipped.append(detail)
            if not skipped:
                continue
            where = site.func_name or "module level"
            issues.append(
                f"{path}:{site.lineno}: new skip in {where} skips "
                f"{'; '.join(skipped)}. Cite https://nvbugs/<id> in the skip "
                "or add the test to waives.txt. A skipif this check cannot "
                "evaluate fails closed."
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
    """Diff the test-db, stages, and waives. Do not look at pytest skips."""
    list_diff = _git(
        repo,
        "diff",
        "--name-only",
        base,
        "--",
        _GROOVY,
        _TEST_DB,
        _WAIVES,
    )
    if list_diff.returncode != 0:
        raise RuntimeError(list_diff.stderr.strip() or "git diff failed")
    if not any(list_diff.stdout.splitlines()):
        return []
    head_groovy, head_yamls, head_waives = _head_lists(repo)
    return find_list_issues(
        _git_show(repo, base, _GROOVY),
        head_groovy,
        _git_yamls(repo, base),
        head_yamls,
        _git_show(repo, base, _WAIVES),
        head_waives,
    )


def collect_skip_issues(repo: Path, base: str) -> list[str]:
    """Diff added pytest skips. Do not judge test-db list edits."""
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
        if "pytest" not in source or "skip" not in source:
            continue
        sources[path] = source
    if not sources:
        return []
    head_groovy, head_yamls, head_waives = _head_lists(repo)
    return find_skip_issues(head_groovy, head_yamls, head_waives, sources, added)


def collect_pre_post_issues(repo: Path, base: str) -> list[str]:
    """Diff ``base`` against the worktree and return new pre/post duplicates."""
    head_groovy = (repo / _GROOVY).read_text() if (repo / _GROOVY).is_file() else ""
    return find_pre_post_issues(
        _git_show(repo, base, _GROOVY),
        head_groovy,
        _git_yamls(repo, base),
        _worktree_yamls(repo),
    )


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
        "--skips",
        action="store_true",
        help="Fail when a new pytest.skip would skip a test-db entry a live stage selects.",
    )
    mode.add_argument(
        "--pre-post-duplicates",
        action="store_true",
        help="Fail when this diff lists a test on both pre-merge and post-merge for the same mako.",
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo_root)
    try:
        if args.skips:
            issues = collect_skip_issues(repo, args.base)
        elif args.pre_post_duplicates:
            issues = collect_pre_post_issues(repo, args.base)
        else:
            issues = collect_list_issues(repo, args.base)
    except (OSError, RuntimeError, yaml.YAMLError) as exc:
        print(f"test-db check could not run: {exc}", file=sys.stderr)
        return 2
    if not issues:
        return 0
    print("\n".join(issues), file=sys.stderr)
    if args.skips:
        summary = (
            f"\n{len(issues)} skip check(s) failed. Cite an NVBug in the skip or in waives.txt."
        )
    elif args.pre_post_duplicates:
        summary = (
            f"\n{len(issues)} pre/post duplicate check(s) failed. "
            "Keep each test on one of the two stages."
        )
    else:
        summary = (
            f"\n{len(issues)} test-db check(s) failed. "
            "Keep the entry on a live L0 stage, or track a dropped entry in waives.txt."
        )
    print(summary, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
