# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TestsDefRule — narrows CI when files under tests/ change.

Handles any path under `tests/` whose first component (after the
`tests/integration/defs/` strip) appears in some YAML entry's namespace.
For `test_*.py` files, AST scope mapping yields function-level anchors
(`file::TestC::test_m`) when every changed line lands in a pytest scope;
otherwise file-level. For non-test_*.py paths (conftest, helpers, data
files like `references/*.yaml` or `disaggregated/test_configs/*.yaml`),
`find_match_for_path` walks up enclosing directories to the narrowest
YAML-covered ancestor.

Paths matched by `out_of_scope_rule.is_out_of_scope` (QA / dev test
lists, `.test_durations`, `microbenchmarks/`, `tests/**/*.md`) are
excluded from candidates so `OutOfScopeRule`'s noop claim is not
overridden by a same-file narrow contribution.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from blocks import Stage, YAMLIndex

from ._helpers import (
    is_perf_stem,
    iter_diff_added_post_line_numbers,
    iter_diff_post_line_numbers,
    iter_diff_pre_image,
    lookup_paths_into_block_filters,
    resolve_affected_stages,
    stages_by_yaml_stem,
)
from .base import PRInputs, Rule, RuleResult
from .out_of_scope_rule import is_out_of_scope

# Changes touching at least this fraction of all blocks (top-level conftest,
# common.py, …) are too broad to narrow usefully and trigger fallback.
BLAST_RADIUS_FRACTION = 0.8

ACCURACY_REFS_PREFIX = "tests/integration/defs/accuracy/references/"
ACCURACY_DIR = "tests/integration/defs/accuracy"

_CLASS_RE = re.compile(r"class\s+(\w+)")
# `acceptance_length.yaml` keys tests directly (`TestC::test_m`) instead of
# by HF model name, unlike every other reference YAML.
_TEST_ID_KEY_RE = re.compile(r"^(Test\w+)::(\w+)$")


def _scope_start_line(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """First line owned by `node`, decorators included.

    `node.lineno` points at the `def` / `class` keyword, so decorators sit
    above it. A decorator edit belongs to what it decorates — a
    `@parametrize` change affects that test, a class-level
    `@skip_pre_blackwell` affects that class — so folding the decorator
    lines into the node's range keeps them off module scope.
    """
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _map_lines_to_pytest_scopes(content: str, line_numbers: set[int]) -> Optional[set[str]]:
    """Resolve each line to its enclosing pytest scope.

    Returns scope strings — `test_X` (top-level), `TestC::test_m` (class
    method), or `TestC` (class body, e.g. `setup_method`). Returns None
    if any line is module-level / outside any `Test*` class — caller
    falls back to file-level narrowing.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return None

    line_to_scope: dict[int, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test"):
                continue
            end = node.end_lineno or node.lineno
            for ln in range(_scope_start_line(node), end + 1):
                line_to_scope[ln] = node.name
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("Test"):
                continue
            class_end = node.end_lineno or node.lineno
            for ln in range(_scope_start_line(node), class_end + 1):
                line_to_scope[ln] = node.name
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not child.name.startswith("test"):
                        continue
                    method_end = child.end_lineno or child.lineno
                    method_scope = f"{node.name}::{child.name}"
                    for ln in range(_scope_start_line(child), method_end + 1):
                        line_to_scope[ln] = method_scope

    scopes: set[str] = set()
    for ln in line_numbers:
        scope = line_to_scope.get(ln)
        if scope is None:
            return None
        scopes.add(scope)
    return scopes if scopes else None


def _diff_has_deletions(diff: str) -> bool:
    """True if any `-` content line (not the `---` file header) appears."""
    for line in diff.splitlines():
        if not line or line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("-"):
            return True
    return False


def _yaml_top_key(raw: str) -> Optional[str]:
    """The key of an unindented `<key>:` line, else None."""
    if not raw or raw[0].isspace() or raw.lstrip().startswith("#"):
        return None
    stripped = raw.split("#", 1)[0].rstrip()
    return stripped[:-1].strip().strip("'\"") if stripped.endswith(":") else None


def _yaml_top_keys_for_lines(content: str, line_numbers: set[int]) -> set[str]:
    """Return the set of top-level YAML keys whose section contains any line in `line_numbers`.

    A "top-level key" is an unindented line of the form `<key>:`
    (optional trailing comment).
    """
    keys: list[Optional[str]] = []
    current: Optional[str] = None
    for raw in content.splitlines():
        key = _yaml_top_key(raw)
        if key is not None:
            current = key
        keys.append(current)
    return {keys[i - 1] for i in line_numbers if 1 <= i <= len(keys) and keys[i - 1]}


def _yaml_all_top_keys(content: str) -> set[str]:
    """Every top-level key in the file."""
    return {k for k in (_yaml_top_key(raw) for raw in content.splitlines()) if k}


def _yaml_top_keys_from_deletions(diff: str) -> Optional[set[str]]:
    """Top-level YAML keys owning each `-` line, read from the pre-image.

    A deleted section's own key line is itself a `-` line, so the
    pre-image view resolves it directly — which the post-image cannot,
    since the section is gone. Returns None when any `-` line's owning
    key is not visible in its hunk (only 3 context lines are guaranteed),
    so the caller can fall back.
    """
    keys: set[str] = set()
    current: Optional[str] = None
    for sign, body in iter_diff_pre_image(diff):
        if sign == "@":
            current = None
            continue
        key = _yaml_top_key(body)
        if key is not None:
            current = key
        if sign != "-" or not body.strip() or body.lstrip().startswith("#"):
            continue
        if current is None:
            return None
        keys.add(current)
    return keys or None


def _py_class_scopes_from_deletions(diff: str) -> Optional[set[str]]:
    """`Test*` classes owning each `-` line, read from the pre-image.

    A deleted class's own `class` statement is itself a `-` line, so the
    pre-image view resolves it where the post-image cannot. Attribution
    stops at class level: a method-level walk would have to guess which
    `def` a stray decorator line belongs to, and class level is already
    narrow enough to matter. Returns None when any `-` line has no
    enclosing `Test*` class visible in its hunk (module scope, imports,
    helpers), so the caller can fall back.
    """
    scopes: set[str] = set()
    cls: Optional[tuple[str, int]] = None
    pending = False  # `-` decorator lines awaiting the class they decorate
    for sign, body in iter_diff_pre_image(diff):
        if sign == "@":
            cls, pending = None, False
            continue
        stripped = body.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(body) - len(body.lstrip())
        match = _CLASS_RE.match(stripped)
        if match is not None:
            cls = (match.group(1), indent) if match.group(1).startswith("Test") else None
        elif cls is not None and indent <= cls[1]:
            cls = None
        if sign != "-":
            continue
        if cls is None:
            # A decorator can precede the class statement it decorates.
            if stripped.startswith("@"):
                pending = True
                continue
            return None
        scopes.add(cls[0])
        pending = False
    return None if pending else (scopes or None)


class TestsDefRule(Rule):
    name = "testdef"
    needs_diff_for: tuple[str, ...] = ("tests/**/*",)

    def __init__(
        self,
        yaml_index: YAMLIndex,
        stages: dict[str, Stage],
        repo_root: Path,
    ) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)
        self._repo_root = repo_root
        self._total_blocks = len(yaml_index.blocks)
        self._acc_class_indexes: Optional[tuple[dict[str, list[str]], dict[str, list[str]]]] = None
        self._acc_source_texts: tuple[str, ...] = ()

    def _compute_anchors(self, git_path: str, yaml_path: str, diff: str) -> list[str]:
        """Return lookup anchors for one file.

        File-level fallback when no diff, file unreadable or not UTF-8
        (e.g. binary fixtures), AST parse fails, or any line is module-
        level. accuracy/references/*.yaml diffs are refined to per-test-
        class anchors via the model-name mapping in
        `_compute_accuracy_reference_anchors`.
        """
        if git_path.startswith(ACCURACY_REFS_PREFIX) and git_path.endswith((".yaml", ".yml")):
            return self._compute_accuracy_reference_anchors(git_path, yaml_path, diff)
        if not diff:
            return [yaml_path]
        try:
            content = (self._repo_root / git_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [yaml_path]
        if _diff_has_deletions(diff):
            return self._recover_deleted_scope_anchors(yaml_path, content, diff)
        line_numbers = iter_diff_post_line_numbers(diff)
        if not line_numbers:
            return [yaml_path]
        scopes = _map_lines_to_pytest_scopes(content, line_numbers)
        if scopes is None:
            return [yaml_path]
        return [f"{yaml_path}::{s}" for s in sorted(scopes)]

    def _recover_deleted_scope_anchors(self, yaml_path: str, content: str, diff: str) -> list[str]:
        """Combine post-image scopes for `+` lines with pre-image scopes for `-` lines.

        Removed lines can anchor to the next surviving post-image scope,
        so every diff with deletions takes this path before accepting
        post-image mappings. Reading deleted lines from the pre-image
        avoids attributing them to an adjacent surviving scope.
        """
        if not _diff_has_deletions(diff):
            return [yaml_path]
        deleted_scopes = _py_class_scopes_from_deletions(diff)
        if deleted_scopes is None:
            return [yaml_path]
        scopes = set(deleted_scopes)
        added = iter_diff_added_post_line_numbers(diff)
        if added:
            added_scopes = _map_lines_to_pytest_scopes(content, added)
            if added_scopes is None:
                return [yaml_path]
            scopes |= added_scopes
        return [f"{yaml_path}::{s}" for s in sorted(scopes)]

    def _compute_accuracy_reference_anchors(
        self, git_path: str, yaml_path: str, diff: str
    ) -> list[str]:
        """Map a `references/<dataset>.yaml` diff to per-test anchors.

        Top-level YAML keys name what the section is a reference for:
        a HF model in most files, a `TestC::test_m` id in
        `acceptance_length.yaml`. Either way they resolve to anchors
        under `accuracy/test_*.py`, whose lineage walk then matches every
        parametrization. Falls back to `[yaml_path]` (→ dir walk-up to
        `accuracy/`) when the changed sections can't be resolved.

        Each side of the diff is read from the image that can actually
        answer for it: `+` lines from the post-image, `-` lines from the
        diff's own pre-image view. Anchoring a `-` line to the next
        surviving post-image line would misattribute it — deleting
        `ModelA:` and its body lands those lines on `ModelB:`.

        An empty anchor list is a zero-impact claim rather than a
        failure. That is only sound when the sections are gone from the
        post-PR YAML and their keys are absent from accuracy test sources
        — the shape of a test-pruning PR, which drops a reference and its
        test together. Otherwise the file-level fallback is retained.
        """
        if not diff:
            return [yaml_path]
        try:
            content = (self._repo_root / git_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [yaml_path]

        changed_keys: set[str] = set()
        if _diff_has_deletions(diff):
            deleted_keys = _yaml_top_keys_from_deletions(diff)
            if deleted_keys is None:
                return [yaml_path]
            changed_keys |= deleted_keys
        added = iter_diff_added_post_line_numbers(diff)
        if added:
            changed_keys |= _yaml_top_keys_for_lines(content, added)
        if not changed_keys:
            return [yaml_path]

        anchors: set[str] = set()
        unresolved_keys: set[str] = set()
        for key in changed_keys:
            key_anchors = self._reference_key_anchors(key)
            if key_anchors:
                anchors.update(key_anchors)
            else:
                unresolved_keys.add(key)
        if unresolved_keys and (
            unresolved_keys & _yaml_all_top_keys(content)
            or self._accuracy_sources_contain_any(unresolved_keys)
        ):
            return [yaml_path]
        return sorted(anchors)

    def _reference_key_anchors(self, key: str) -> list[str]:
        """Anchors for one reference-YAML top-level key.

        A `TestC::test_m` key (`acceptance_length.yaml`) resolves through
        the class index; any other key is treated as a HF model name.
        """
        match = _TEST_ID_KEY_RE.match(key)
        if match is None:
            return list(self._accuracy_model_to_classes().get(key, ()))
        class_name, method = match.groups()
        return [f"{c}::{method}" for c in self._accuracy_class_to_paths().get(class_name, ())]

    def _accuracy_model_to_classes(self) -> dict[str, list[str]]:
        """HF model name → list of `accuracy/test_X.py::ClassName`."""
        return self._scan_accuracy_classes()[0]

    def _accuracy_class_to_paths(self) -> dict[str, list[str]]:
        """Test class name → list of `accuracy/test_X.py::ClassName`.

        A class name can appear in several modules (e.g. `TestKimiK3` in
        both the text and multimodal accuracy files), so every definition
        is kept.
        """
        return self._scan_accuracy_classes()[1]

    def _accuracy_sources_contain_any(self, keys: set[str]) -> bool:
        """Return whether any changed reference key appears in an accuracy test source."""
        self._scan_accuracy_classes()
        return any(key in source for source in self._acc_source_texts for key in keys)

    def _scan_accuracy_classes(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Cached AST scan of `accuracy/test_*.py` for `Test*` classes.

        Returns (model name → qualified classes, class name → qualified
        classes). The first index is keyed by the literal
        `MODEL_NAME = "<hf>"` assignment; classes without one appear only
        in the second.
        """
        if self._acc_class_indexes is not None:
            return self._acc_class_indexes
        by_model: dict[str, list[str]] = {}
        by_class: dict[str, list[str]] = {}
        source_texts: list[str] = []
        acc_dir = self._repo_root / ACCURACY_DIR
        if acc_dir.is_dir():
            for py in sorted(acc_dir.glob("test_*.py")):
                try:
                    source = py.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                source_texts.append(source)
                try:
                    tree = ast.parse(source)
                except (SyntaxError, ValueError):
                    continue
                rel = f"accuracy/{py.name}"
                for node in tree.body:
                    if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
                        continue
                    qualified = f"{rel}::{node.name}"
                    by_class.setdefault(node.name, []).append(qualified)
                    for child in node.body:
                        if not isinstance(child, ast.Assign):
                            continue
                        if not any(
                            isinstance(t, ast.Name) and t.id == "MODEL_NAME" for t in child.targets
                        ):
                            continue
                        v = child.value
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            by_model.setdefault(v.value, []).append(qualified)
                            break
        self._acc_source_texts = tuple(source_texts)
        self._acc_class_indexes = (by_model, by_class)
        return self._acc_class_indexes

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        candidates = [
            f for f in pr.changed_files if f.startswith("tests/") and not is_out_of_scope(f)
        ]
        if not candidates:
            return None

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        narrowed: set[str] = set()
        handled: set[str] = set()
        out_of_namespace: set[str] = set()
        no_match: set[str] = set()

        for git_path in candidates:
            yaml_path = self.yaml_index.git_path_to_yaml_key(git_path)
            if yaml_path is None:
                base = git_path.rsplit("/", 1)[-1]
                if base.startswith("test_") and base.endswith(".py"):
                    # Standalone test_*.py not referenced by any L0 YAML —
                    # pytest doesn't auto-import test files into other
                    # tests, so this file's edits can't affect what L0
                    # runs. Claim as noop contribution.
                    handled.add(git_path)
                    no_match.add(git_path)
                    continue
                # conftest / __init__ / helper / data — could impact
                # selection via implicit pytest discovery; let Selector
                # report it as Unhandled → fallback.
                out_of_namespace.add(git_path)
                continue
            handled.add(git_path)
            anchors = self._compute_anchors(git_path, yaml_path, pr.diffs.get(git_path, ""))
            file_bf, _ = lookup_paths_into_block_filters(self.yaml_index, anchors)
            if not file_bf:
                # In-namespace but no YAML entry covers this path → no
                # narrow contribution; claimed as noop.
                no_match.add(git_path)
                continue
            narrowed.add(git_path)
            for key, prefix_to_ids in file_bf.items():
                dst = block_filters.setdefault(key, {})
                for prefix, ids in prefix_to_ids.items():
                    dst.setdefault(prefix, set()).update(ids)

        if not handled:
            # All candidates were out-of-namespace → don't claim; fallback.
            return None

        if not block_filters:
            # Every claimed (in-namespace) path had no covering YAML entry:
            # the rule fired and decided no test-selection impact.
            return RuleResult(
                handled_files=handled,
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=f"testdef: {len(handled)} path(s) → no covering YAML entry",
            )

        if (
            self._total_blocks > 0
            and len(block_filters) >= self._total_blocks * BLAST_RADIUS_FRACTION
        ):
            return RuleResult(
                handled_files=handled,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"testdef: blast-radius cap "
                    f"({len(block_filters)}/{self._total_blocks} blocks "
                    f">= {BLAST_RADIUS_FRACTION:.0%}); fallback"
                ),
            )

        affected_stages = resolve_affected_stages(
            block_filters, self.yaml_index, self._stages_by_yaml
        )
        sanity_relevant = any(stem == "l0_sanity_check" for stem, _ in block_filters)
        perfsanity_relevant = any(is_perf_stem(stem) for stem, _ in block_filters)

        nonarrow_note = ""
        if no_match:
            nonarrow_note = f"; {len(no_match)} path(s) not in any L0 YAML"

        return RuleResult(
            handled_files=handled,
            affected_stages=affected_stages,
            scope="testdefonly",
            block_filters=block_filters,
            sanity_relevant=sanity_relevant,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"testdef: {len(narrowed)} path(s) → "
                f"{len(block_filters)} blocks, {len(affected_stages)} stages"
                f"{nonarrow_note}"
            ),
        )
