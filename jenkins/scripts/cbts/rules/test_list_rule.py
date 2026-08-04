# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""TestListRule — narrows CI when test-db YAML files change.

Only ADDED entries drive narrowing. Removed entries don't trigger
verification: the test either still runs elsewhere (unchanged) or is
fully retired. Comment lines and whitespace-only edits are ignored.

Classification is driven by the post-PR YAML structure, not by hunk
context. For each changed file, the rule walks up from every changed
line in the post-PR file content to find its ancestor key chain. If
any ancestor (or the changed key itself) is in
{condition, wildcards, terms, ranges}, the edit changed stage-
selection semantics and the rule forces fallback. Otherwise every
+/- list item is by construction inside a `tests:` block and is
treated as a tests-entry add / remove.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from blocks import Stage, YAMLIndex, normalize_test_id

from ._helpers import lookup_ids_into_block_filters, resolve_affected_stages, stages_by_yaml_stem
from .base import PRInputs, Rule, RuleResult

TEST_DB_DIR = "tests/integration/test_lists/test-db/"

# Keys whose subtree drives stage selection (which block runs where).
# Any +/- line inside one of these subtrees, or that creates/removes
# such a key, is treated as a stage-selection change → fallback.
STAGE_SELECT_KEYS: frozenset[str] = frozenset({"condition", "wildcards", "terms", "ranges"})

_HUNK_HEADER_RE = re.compile(r"^@@ ")
_KEY_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][\w-]*)\s*:")
_LIST_ITEM_RE = re.compile(r"^(?P<indent>\s*)-\s+(?P<body>\S.*)$")
_COMMENT_RE = re.compile(r"^\s*#")


def _is_test_db_file(path: str) -> bool:
    return path.startswith(TEST_DB_DIR) and path.endswith(".yml")


def _is_perf_stem(stem: str) -> bool:
    return stem == "l0_perf" or "perf_sanity" in stem


def _line_key(body: str) -> Optional[str]:
    """Return the key name for a key line or `- key:` block list item.

    Returns None if `body` is anything else. Used for both ancestor
    walk-up and direct stage-select-key edit detection.
    """
    m = _KEY_LINE_RE.match(body)
    if m:
        return m.group("key")
    lm = _LIST_ITEM_RE.match(body)
    if lm:
        inner = lm.group("body").rstrip()
        km = _KEY_LINE_RE.match(inner) if inner else None
        if km:
            return km.group("key")
    return None


def _diff_edits_stage_select_key(diff: str) -> bool:
    """True if any +/- line creates or deletes a stage-select key line.

    Catches whole-block deletions (e.g. `- - condition:`) regardless of
    where the deleted lines anchor in the post-PR content.
    """
    for raw in diff.splitlines():
        if not raw or raw.startswith(("+++", "---")):
            continue
        if _HUNK_HEADER_RE.match(raw):
            continue
        if raw[0] not in ("+", "-"):
            continue
        body = raw[1:]
        if not body.strip() or _COMMENT_RE.match(body):
            continue
        key = _line_key(body)
        if key is not None and key in STAGE_SELECT_KEYS:
            return True
    return False


def _added_line_numbers(diff: str) -> set[int]:
    """Post-image line numbers for `+` lines only.

    `iter_diff_post_line_numbers` also reports an anchor for each `-`
    line — the next surviving line in post-PR — which can land on the
    start of an unrelated sibling block (e.g. the `- condition:` of the
    next block when the last `tests:` entry of the previous block is
    deleted). Ancestor walk-up on those anchors mis-classifies the
    deletion as a stage-select edit. We only ancestor-walk `+` lines,
    whose post-image positions truly describe their content.
    """
    out: set[int] = set()
    new_line = 0
    for line in diff.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m is not None:
            new_line = int(m.group(1))
            continue
        if not line or line.startswith(("+++", "---")):
            continue
        sign = line[0]
        if sign == "+":
            out.add(new_line)
            new_line += 1
        elif sign == " ":
            new_line += 1
        # `-` lines: do not record, do not advance the post-image cursor.
    return out


def _diff_has_suspicious_minus(diff: str) -> bool:
    """True if any `-` line is something other than a clean tests-entry list item.

    Comment / blank `-` lines are ignored. A `-` line whose body parses
    as a `<path>.py::...` test id under the list-item form is a clean
    tests-entry removal. Anything else (key edits, sub-key removals
    like `stage: pre_merge`, wildcard glob values like `- "*newgpu*"`)
    is suspicious — we can't ancestor-walk it against the post-PR file,
    so fall back conservatively.
    """
    for raw in diff.splitlines():
        if not raw or raw.startswith(("+++", "---")):
            continue
        if _HUNK_HEADER_RE.match(raw):
            continue
        if not raw.startswith("-"):
            continue
        body = raw[1:]
        if not body.strip() or _COMMENT_RE.match(body):
            continue
        lm = _LIST_ITEM_RE.match(body)
        if lm and normalize_test_id(lm.group("body")):
            continue  # clean tests-entry removal
        return True
    return False


def _effective_indent(line: str) -> int:
    """Structural indent: column of the first significant token.

    For a `- key:` block list item the `- ` marker is part of the
    indent, so `key` (and `tests:` sibling at the same dict level)
    compare equal.
    """
    stripped = line.lstrip()
    base = len(line) - len(stripped)
    if stripped.startswith("-"):
        i = 1
        while i < len(stripped) and stripped[i] in (" ", "\t"):
            i += 1
        return base + i
    return base


def _line_is_in_stage_select(lines: list[str], line_no: int) -> bool:
    """True if `lines[line_no - 1]` lies inside a stage-select subtree.

    Walks up the ancestor chain by strictly-lower effective indent; the
    line is in stage-select if itself or any ancestor key is one of
    {condition, wildcards, terms, ranges}.
    """
    if not (1 <= line_no <= len(lines)):
        return False
    cur = lines[line_no - 1]
    target_indent = _effective_indent(cur)
    # Also check the line itself: if it IS a stage-select key line,
    # count as in stage-select (an edit on the key itself).
    self_key = _line_key(cur.split("#", 1)[0].rstrip())
    if self_key is not None and self_key in STAGE_SELECT_KEYS:
        return True
    for ln in range(line_no - 1, 0, -1):
        raw = lines[ln - 1]
        if not raw or raw.lstrip().startswith("#"):
            continue
        indent = _effective_indent(raw)
        if indent >= target_indent:
            continue
        target_indent = indent
        stripped = raw.split("#", 1)[0].rstrip()
        key = _line_key(stripped)
        if key is not None and key in STAGE_SELECT_KEYS:
            return True
    return False


def _scan_tests_entries(diff: str) -> tuple[set[str], set[str]]:
    """Extract added / removed test IDs from a stage-select-clean diff.

    The caller must have already confirmed the diff does not touch any
    stage-select area; every `+/-` list item is then by construction
    inside a `tests:` block.
    """
    added: set[str] = set()
    removed: set[str] = set()
    for raw in diff.splitlines():
        if not raw or raw.startswith(("+++", "---")):
            continue
        if _HUNK_HEADER_RE.match(raw):
            continue
        if raw[0] not in ("+", "-"):
            continue
        sign = raw[0]
        body = raw[1:]
        if not body.strip() or _COMMENT_RE.match(body):
            continue
        lm = _LIST_ITEM_RE.match(body)
        if not lm:
            continue
        tid = normalize_test_id(lm.group("body"))
        if tid:
            (added if sign == "+" else removed).add(tid)
    return added, removed


class TestListRule(Rule):
    name = "testlist"
    needs_diff_for: tuple[str, ...] = ("tests/integration/test_lists/test-db/*.yml",)

    def __init__(
        self,
        yaml_index: YAMLIndex,
        stages: dict[str, Stage],
        repo_root: Path,
    ) -> None:
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)
        self._repo_root = repo_root

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        touched = sorted(p for p in pr.changed_files if _is_test_db_file(p))
        if not touched:
            return None

        added: set[str] = set()
        removed: set[str] = set()
        stage_select_files: list[str] = []
        unverifiable_files: list[str] = []

        for path in touched:
            diff = pr.diffs.get(path, "")

            # Cheap pre-check: any +/- line edits a stage-select key
            # directly (covers whole-block adds/removes regardless of
            # where their anchors land in post-PR content).
            if _diff_edits_stage_select_key(diff):
                stage_select_files.append(path)
                continue

            # `-` lines have no reliable post-PR position. Bodies that
            # are not clean tests-entry list items (sub-key edits,
            # wildcard glob values, ...) could be stage-select
            # changes; fall back conservatively for those.
            if _diff_has_suspicious_minus(diff):
                stage_select_files.append(path)
                continue

            # Deep check on `+` lines only: walk the post-PR file's
            # ancestor key chain from each added line's position. `-`
            # line anchors are intentionally excluded — they can land
            # on neighboring blocks and trigger false positives.
            try:
                post_lines = (self._repo_root / path).read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                unverifiable_files.append(path)
                continue

            added_line_nums = _added_line_numbers(diff)
            if any(_line_is_in_stage_select(post_lines, ln) for ln in added_line_nums):
                stage_select_files.append(path)
                continue

            a, r = _scan_tests_entries(diff)
            added |= a
            removed |= r

        if stage_select_files:
            preview = ", ".join(stage_select_files[:3])
            more = f" (+{len(stage_select_files) - 3} more)" if len(stage_select_files) > 3 else ""
            return RuleResult(
                handled_files=set(touched),
                affected_stages=set(),
                scope=None,
                reason=(
                    f"testlist: {len(stage_select_files)} file(s) edit "
                    f"condition/wildcards/terms/ranges → stage-selection change: "
                    f"{preview}{more}"
                ),
            )

        if unverifiable_files:
            preview = ", ".join(unverifiable_files[:3])
            more = f" (+{len(unverifiable_files) - 3} more)" if len(unverifiable_files) > 3 else ""
            return RuleResult(
                handled_files=set(touched),
                affected_stages=set(),
                scope=None,
                reason=(
                    f"testlist: {len(unverifiable_files)} file(s) unreadable in "
                    f"post-PR tree (cannot verify stage-select): {preview}{more}"
                ),
            )

        if not added:
            note = f" ({len(removed)} removed entries don't need verification)" if removed else ""
            return RuleResult(
                handled_files=set(touched),
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=f"testlist: no additions across {len(touched)} file(s){note}",
            )

        block_filters, misses = lookup_ids_into_block_filters(self.yaml_index, added)
        if not block_filters:
            preview = ", ".join(sorted(misses)[:3])
            more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
            return RuleResult(
                handled_files=set(touched),
                affected_stages=set(),
                scope="noop",
                sanity_relevant=False,
                perfsanity_relevant=False,
                reason=(
                    f"testlist: all {len(misses)} added entry/entries don't resolve "
                    f"to any post-PR YAML block: {preview}{more}"
                ),
            )

        affected_stages = resolve_affected_stages(
            block_filters, self.yaml_index, self._stages_by_yaml
        )
        sanity_relevant = any(stem == "l0_sanity_check" for stem, _ in block_filters)
        perfsanity_relevant = any(_is_perf_stem(stem) for stem, _ in block_filters)

        miss_note = ""
        if misses:
            preview = ", ".join(sorted(misses)[:3])
            more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
            miss_note = f"; {len(misses)} unresolved entry/entries ignored: {preview}{more}"

        return RuleResult(
            handled_files=set(touched),
            affected_stages=affected_stages,
            scope="testlistonly",
            block_filters=block_filters,
            sanity_relevant=sanity_relevant,
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"testlist: +{len(added)} / -{len(removed)} across "
                f"{len(touched)} file(s) → narrow on +{len(added)} → "
                f"{len(block_filters)} blocks, {len(affected_stages)} stages"
                f"{miss_note}"
            ),
        )
