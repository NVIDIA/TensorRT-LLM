# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for metadata, attribution, local state, and author feedback."""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = "vendor_bot_test"
_SPEC = importlib.util.spec_from_file_location(
    _PACKAGE,
    _ROOT / "scripts/vendor/__init__.py",
    submodule_search_locations=[str(_ROOT / "scripts/vendor")],
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_PACKAGE] = _MODULE
_SPEC.loader.exec_module(_MODULE)
b = importlib.import_module(f"{_PACKAGE}.bot")
m = importlib.import_module(f"{_PACKAGE}.metadata")
p = importlib.import_module(f"{_PACKAGE}.promote")

pytestmark = pytest.mark.cpu_only


def _body(content: str) -> str:
    return f"{m.BEGIN}\n```yaml\nschema_version: 1\nvendors:\n  toy:\n{content}\n```\n{m.END}"


def _issue(message: str = "Missing marked block") -> m.Diagnostic:
    return m.Diagnostic(message, "Edit the PR description to supply the missing metadata.")


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_metadata_only_needs_upstream_urls(line_ending: str) -> None:
    entry = m.parse(
        _body("    upstream_prs:\n      - https://github.com/org/library/pull/42").replace(
            "\n", line_ending
        ),
        "toy",
        "org/library",
    )
    assert entry == {"upstream_prs": [42], "unpaired_reason": None, "resolutions": {}}


def test_metadata_mixed_line_endings() -> None:
    body = _body("    upstream_prs:\n      - https://github.com/org/library/pull/42")
    mixed = body.replace("```yaml\n", "```yaml\r\n").replace("\n```\n", "\r```\r\n")
    expected = m.parse(body, "toy", "org/library")
    actual = m.parse(mixed, "toy", "org/library")
    assert actual == expected
    assert m.fingerprint(actual) == m.fingerprint(expected)


def test_metadata_size_limit_precedes_line_ending_normalization() -> None:
    body = "\r\n" * 32768 + m.template("toy", "org/library")
    with pytest.raises(ValueError, match="exceeds the supported metadata size"):
        m.parse(body, "toy", "org/library")


@pytest.mark.parametrize(
    "content",
    [
        "    upstream_prs: []",
        "    upstream_prs: https://github.com/org/library/pull/42",
        "    upstream_prs: [https://github.com/other/library/pull/42]",
        "    upstream_prs: [https://github.com/org/library/pull/42, https://github.com/org/library/pull/42]",
        "    unpaired_reason: ''",
        "    unpaired_reason: test\n    upstream_prs: []",
        "    canonical_repo: attacker/repo",
        "    upstream_prs: []\n    upstream_prs: []",
        "    upstream_prs: [",
        "    upstream_prs: [https://github.com/org/library/pull/42]\n    resolutions: []",
        "    upstream_prs: &prs [https://github.com/org/library/pull/42]",
        "    upstream_prs: &prs [*prs]",
    ],
)
def test_invalid_metadata_rejected(content: str) -> None:
    with pytest.raises(m.Diagnostic) as failure:
        m.parse(_body(content), "toy", "org/library")
    assert failure.value.action
    assert "**Fix:**" in str(failure.value)


@pytest.mark.parametrize(
    "body", ["", "missing", m.template("toy", "org/library") * 2, m.END + m.BEGIN]
)
def test_missing_or_duplicate_marked_blocks(body: str) -> None:
    with pytest.raises(ValueError):
        m.parse(body, "toy", "org/library")


def test_author_resolution_requires_explanation() -> None:
    content = (
        "    upstream_prs: [https://github.com/org/library/pull/42]\n    resolutions:\n      "
        + "a" * 40
        + ":\n        upstream_pr: https://github.com/org/library/pull/42\n        reason: adapted ABI"
    )
    entry = m.parse(_body(content), "toy", "org/library")
    assert entry["resolutions"]["a" * 40] == {"upstream_pr": 42, "reason": "adapted ABI"}
    with pytest.raises(ValueError):
        m.parse(_body(content.replace("adapted ABI", "''")), "toy", "org/library")


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "start_count,end_count", [(0, 0), (0, 1), (1, 0), (2, 1), (1, 2), (2, 2), (2, 0)]
)
def test_marker_diagnostics_name_missing_and_duplicate_markers(
    start_count: int, end_count: int, line_ending: str
) -> None:
    body = line_ending.join([m.BEGIN] * start_count + [m.END] * end_count)
    with pytest.raises(m.Diagnostic) as failure:
        m.parse(body, "toy", "org/library")
    issue = failure.value
    for marker, count in ((m.BEGIN, start_count), (m.END, end_count)):
        if count == 0:
            assert f"`{marker}`" in issue.problem and "Missing" in issue.problem
        elif count > 1:
            assert f"`{marker}` appears {count} times" in issue.problem
        else:
            assert f"`{marker}`" not in issue.problem
    assert "PR" in issue.action
    if start_count == end_count == 0:
        example = issue.example.removeprefix("````markdown\n").removesuffix("\n````")
        assert m.parse(example, "toy", "org/library")["upstream_prs"] == [123]
        assert "not a comment" in issue.action.replace("—", " ")
    elif 0 in (start_count, end_count):
        assert "before" in issue.action if start_count == 0 else "after" in issue.action
    if max(start_count, end_count) > 1:
        assert "preserving" in issue.action and "Consolidate" in issue.action


def test_reversed_markers_and_bad_fence_have_different_fixes() -> None:
    with pytest.raises(m.Diagnostic) as reversed_markers:
        m.parse(m.END + m.BEGIN, "toy", "org/library")
    assert "end marker appears before" in reversed_markers.value.problem
    assert f"`{m.BEGIN}` before" in reversed_markers.value.action
    with pytest.raises(m.Diagnostic) as bad_fence:
        m.parse(m.BEGIN + "\nschema_version: 1\n" + m.END, "toy", "org/library")
    assert "fenced yaml" in bad_fence.value.problem
    assert "three backticks" in bad_fence.value.action
    assert m.BEGIN in bad_fence.value.example


@pytest.mark.parametrize(
    "content,problem,fix",
    [
        ("    upstream_prs: []", "vendors.toy.upstream_prs", "list every related upstream PR"),
        (
            "    upstream_prs: [https://github.com/wrong/repo/pull/42]",
            "upstream_prs[0]",
            "org/library/pull/123",
        ),
        ("    upstream_prs: [42]", "upstream_prs[0]", "full URL"),
        (
            "    upstream_prs: [https://github.com/org/library/pull/42#discussion]",
            "upstream_prs[0]",
            "no query string or fragment",
        ),
        (
            "    upstream_prs: [https://github.com/org/library/pull/42, https://github.com/org/library/pull/42]",
            "duplicate upstream PR #42",
            "Remove the repeated",
        ),
        ("    canonical_repo: wrong", "canonical_repo", "Repository/operator settings"),
        ("    unpaired_reason: ''", "vendors.toy.unpaired_reason", "Do not use it to bypass"),
        ("    unpaired_reason: downstream\n    resolutions: {}", "cannot accompany", "keep only"),
        (
            "    upstream_prs: [https://github.com/org/library/pull/42]\n    resolutions: []",
            "vendors.toy.resolutions",
            "full 40-character",
        ),
        (
            "    upstream_prs: &prs [https://github.com/org/library/pull/42]",
            "aliases and anchors",
            "explicit values",
        ),
    ],
)
def test_metadata_diagnostics_identify_field_and_fix(content: str, problem: str, fix: str) -> None:
    with pytest.raises(m.Diagnostic) as failure:
        m.parse(_body(content), "toy", "org/library")
    assert problem in failure.value.problem
    assert fix in failure.value.action


@pytest.mark.parametrize(
    "content,problem",
    [
        ("schema_version: '1'\nvendors: {}", "schema_version"),
        ("schema_version: 1", "vendors"),
        ("schema_version: 1\nvendors: {}\nextra: false", "extra"),
        ("schema_version: 1\nvendors: {}", "vendors.toy"),
        ("[]", "mapping, not a list"),
    ],
)
def test_metadata_structure_has_targeted_guidance(content: str, problem: str) -> None:
    with pytest.raises(m.Diagnostic) as failure:
        m.parse(f"{m.BEGIN}\n```yaml\n{content}\n```\n{m.END}", "toy", "org/library")
    assert problem in failure.value.problem
    assert "PR description" in failure.value.action or "PR-description" in failure.value.action


@pytest.mark.parametrize(
    "content,problem",
    [
        ("    upstream_prs: [secret-unclosed", "YAML line"),
        ("    private-key: secret\n    private-key: secret", "duplicate mapping key"),
        ("    <<: {private-key: secret}", "merge keys are not allowed"),
    ],
)
def test_yaml_errors_show_location_without_echoing_values(content: str, problem: str) -> None:
    with pytest.raises(m.Diagnostic) as failure:
        m.parse(_body(content), "toy", "org/library")
    assert problem in failure.value.problem
    assert "YAML line" in failure.value.problem and "column" in failure.value.problem
    assert "secret" not in str(failure.value)
    assert "line 1 is the first line inside the YAML fence" in failure.value.action


@pytest.mark.parametrize(
    "sha,resolution,problem,fix",
    [
        (
            "abc123",
            "upstream_pr: https://github.com/org/library/pull/42\n        reason: adapted",
            "Invalid source commit key",
            "40-character",
        ),
        (
            "a" * 40,
            "upstream_pr: https://github.com/org/library/pull/42",
            "exactly",
            "missing fields",
        ),
        (
            "a" * 40,
            "upstream_pr: https://github.com/org/library/pull/43\n        reason: adapted",
            "not listed",
            "upstream_prs",
        ),
        (
            "a" * 40,
            "upstream_pr: https://github.com/org/library/pull/42\n        reason: ''",
            ".reason",
            "Explain the downstream adaptation",
        ),
    ],
)
def test_resolution_errors_identify_commit_and_fix(
    sha: str, resolution: str, problem: str, fix: str
) -> None:
    content = (
        "    upstream_prs: [https://github.com/org/library/pull/42]\n"
        f"    resolutions:\n      {sha}:\n        {resolution}"
    )
    with pytest.raises(m.Diagnostic) as failure:
        m.parse(_body(content), "toy", "org/library")
    assert sha in failure.value.problem and problem in failure.value.problem
    assert fix in failure.value.action


def test_resolution_example_is_nested_under_real_vendor_and_parseable() -> None:
    example = m._resolution_example("toy", "org/library")
    yaml_text = example.removeprefix("```yaml\n").removesuffix("\n```")
    body = (f"{m.BEGIN}\n```yaml\nschema_version: 1\n{yaml_text}\n```\n{m.END}").replace(
        "FULL_40_CHARACTER_SOURCE_SHA", "a" * 40
    )
    entry = m.parse(body, "toy", "org/library")
    assert entry["resolutions"]["a" * 40]["upstream_pr"] == 123


def test_untrusted_field_names_cannot_inject_markdown() -> None:
    body = _body("    '<img src=x>': value")
    with pytest.raises(m.Diagnostic) as failure:
        m.parse(body, "toy", "org/library")
    assert "<img" not in str(failure.value)
    assert "&lt;img" in str(failure.value)


def test_unpaired_and_template() -> None:
    entry = m.parse(
        _body("    unpaired_reason: downstream-only compatibility"), "toy", "org/library"
    )
    assert entry["unpaired_reason"] == "downstream-only compatibility"
    assert m.parse(m.template("toy", "org/library"), "toy", "org/library")["upstream_prs"] == [123]


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "core.hooksPath", str(tmp_path / "no-hooks"))
    return repo


def _commit(repo: Path, text: str, filename: str = "code.py") -> str:
    (repo / filename).parent.mkdir(parents=True, exist_ok=True)
    (repo / filename).write_text(text)
    _git(repo, "add", filename)
    _git(repo, "commit", "-q", "-m", "test")
    return _git(repo, "rev-parse", "HEAD")


def _pr(base: str, head: str) -> dict:
    return {"base": {"sha": base, "ref": "main"}, "head": {"sha": head}, "state": "open"}


def test_match_multiple_upstream_prs_without_manual_map(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    first = _commit(source, "VALUE = 1\n")
    second = _commit(source, "VALUE = 2\n")
    assignments, evidence = m.match(
        source, base, second, {41: _pr(base, first), 42: _pr(first, second)}
    )
    assert assignments == {first: 41, second: 42}
    assert all(item["method"] == "same-commit" for item in evidence.values())


def test_merge_diagnostic_explains_attribution_and_latest_dev_rebase(source: Path) -> None:
    base = _commit(source, "base\n")
    canonical = _commit(source, "canonical\n", "canonical.py")
    _git(source, "checkout", "-q", "-b", "temporary", base)
    _commit(source, "change\n")
    _git(source, "merge", "--no-ff", "-m", "merge canonical", canonical)
    reviewed = _git(source, "rev-parse", "HEAD")
    with pytest.raises(m.Diagnostic) as failure:
        m.match(source, canonical, reviewed, {})
    issue = failure.value
    assert reviewed in issue.problem and "2 parents" in issue.problem
    assert "commit-by-commit upstream attribution" in issue.problem
    assert "latest canonical dev branch" in issue.action
    assert "cherry-pick" in issue.action and "Do not rewrite canonical history" in issue.action
    assert "update and verify" in issue.action and "cannot remove a merge commit" in issue.action


def test_match_rebased_patch(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    downstream = _commit(source, "VALUE = 1\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    context = _commit(source, "context\n", "other.txt")
    upstream = _commit(source, "VALUE = 1\n")
    assignments, evidence = m.match(source, base, downstream, {42: _pr(context, upstream)})
    assert assignments == {downstream: 42}
    assert evidence[downstream]["method"] == "same-edits"


def test_match_only_lock_selected_files(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n", "library/kernels/code.py")
    for name in (
        "tests/test_code.py",
        "library/kernels/README.md",
        "library/kernels_extra/code.py",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("downstream-only content\n")
    _git(source, "add", "-A")
    downstream = _commit(source, "VALUE = 1\n", "library/kernels/code.py")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "VALUE = 1\n", "library/kernels/code.py")
    assignments, evidence = m.match(
        source,
        base,
        downstream,
        {42: _pr(base, upstream)},
        source="library/kernels",
        include=("**/*.py",),
    )
    assert assignments == {downstream: 42}
    assert evidence[downstream]["method"] == "same-edits"
    with pytest.raises(ValueError, match="no exact match"):
        m.match(source, base, downstream, {42: _pr(base, upstream)})


@pytest.mark.parametrize("include", [("*.py",), ("**/*.py",), ("nested/*.py",)])
def test_patch_uses_lock_globs_and_literal_git_paths(
    source: Path, include: tuple[str, ...]
) -> None:
    base = _commit(source, "base\n", "outside.txt")
    paths = ["root.py", "nested/kernel[1] *.py", "nested/README.md", "nested/deep/kernel.py"]
    for name in paths:
        path = source / "selected" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n")
    _git(source, "add", "-A")
    head = _commit(source, "not selected\n", "selected-sibling/outside.py")
    patch = m._patch(source, base, head, source="selected", include=include)
    for name in paths:
        assert (f"+{name}\n" in patch) == m.manage._matches(name, include)
    assert "not selected" not in patch
    assert m._patch(source, base, head, source="absent", include=include) == ""


@pytest.mark.parametrize("squash", [False, True])
def test_excluded_commits_are_not_attributed_to_upstream(source: Path, squash: bool) -> None:
    base = _commit(source, "VALUE = 0\n", "src/code.py")
    before = _commit(source, "initial tests\n", "tests/test_code.py")
    first = _commit(source, "VALUE = 1\n", "src/code.py")
    second = _commit(source, "VALUE = 2\n", "src/code.py")
    after = _commit(source, "more tests\n", "tests/test_code.py")
    upstream = after
    if squash:
        _git(source, "checkout", "-q", "-b", "upstream", base)
        upstream = _commit(source, "VALUE = 2\n", "src/code.py")
    assignments, evidence = m.match(
        source,
        base,
        after,
        {42: _pr(base, upstream)},
        source="src",
        include=("**/*.py",),
    )
    assert assignments == {before: None, first: 42, second: 42, after: None}
    for sha in (before, after):
        assert evidence[sha] == {"method": "outside-vendor-scope", "refresh_policy": "retain"}
    assert evidence[first]["method"] == ("source-series-aggregate" if squash else "same-commit")


def test_empty_selected_patches_never_match_an_upstream_pr(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n", "src/code.py")
    head = _commit(source, "tests only\n", "tests/test_code.py")
    assignments, _ = m.match(source, base, head, {}, source="src", include=("**/*.py",))
    assert assignments == {head: None}
    with pytest.raises(ValueError, match="cover no new lock-selected changes"):
        m.match(source, base, head, {42: _pr(base, head)}, source="src", include=("**/*.py",))
    with pytest.raises(ValueError, match="Remove resolutions"):
        m.match(
            source,
            base,
            head,
            {42: _pr(base, head)},
            {head: {"upstream_pr": 42, "reason": "tests"}},
            source="src",
            include=("**/*.py",),
        )


def test_selected_code_mismatch_is_not_hidden_by_identical_tests(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n", "src/code.py")
    tests = _commit(source, "shared tests\n", "tests/test_code.py")
    downstream = _commit(source, "VALUE = 1\n", "src/code.py")
    _git(source, "checkout", "-q", "-b", "upstream", tests)
    upstream = _commit(source, "VALUE = 2\n", "src/code.py")
    with pytest.raises(ValueError, match="no exact match"):
        m.match(
            source, base, downstream, {42: _pr(base, upstream)}, source="src", include=("**/*.py",)
        )


@pytest.mark.parametrize("into_scope", [False, True])
def test_patch_keeps_selected_side_of_move(source: Path, into_scope: bool) -> None:
    selected, excluded = "src/code.py", "tests/test_code.py"
    old, new = (excluded, selected) if into_scope else (selected, excluded)
    base = _commit(source, "VALUE = 0\n", old)
    (source / new).parent.mkdir(parents=True, exist_ok=True)
    _git(source, "mv", old, new)
    _git(source, "commit", "-q", "-m", "move")
    head = _git(source, "rev-parse", "HEAD")
    patch = m._patch(source, base, head, source="src", include=("**/*.py",))
    assert selected in patch
    assert excluded not in patch
    assert ("+VALUE = 0" if into_scope else "-VALUE = 0") in patch
    assert ("new file mode" if into_scope else "deleted file mode") in patch


@pytest.mark.parametrize("change", [{"source": "different"}, {"include": ("*.py",)}])
def test_resolve_rejects_selection_changes_before_fetch(tmp_path: Path, change: dict) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = dataclasses.replace(previous, commit="b" * 40, **change)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="unchanged lock source/include"):
        m.resolve(None, previous, reviewed, {}, cache)
    assert not cache.exists()


@pytest.mark.parametrize(
    ("original", "rebased"),
    [
        (
            "def run():\n    before = 0\n    value = 0\n    after = 0\n    return value\n",
            "def run():\n    before = 2\n    value = 0\n    after = 3\n    return value\n",
        ),
        (
            "def old_helper():\n    pass\n\n\n@cache\ndef run():\n    value = 0\n",
            "def new_helper():\n    pass\n\n\n@cache\ndef run():\n    value = 0\n",
        ),
        (
            "def run():\n    value = 0\n    # gap\n    second = 0\n",
            "def run():\n    value = 0\n" + "    # gap\n" * 10 + "    second = 0\n",
        ),
    ],
    ids=["neighboring-lines", "neighboring-function", "hunk-spacing"],
)
def test_match_ignores_unchanged_context(source: Path, original: str, rebased: str) -> None:
    # Repo-local diff settings must not reintroduce unchanged context.
    _git(source, "config", "diff.interHunkContext", "20")
    base = _commit(source, original)
    downstream = _commit(
        source, original.replace("value = 0", "value = 1").replace("second = 0", "second = 1")
    )
    _git(source, "checkout", "-q", "-b", "upstream", base)
    context = _commit(source, rebased)
    upstream = _commit(
        source, rebased.replace("value = 0", "value = 1").replace("second = 0", "second = 1")
    )
    assignments, evidence = m.match(source, base, downstream, {42: _pr(context, upstream)})
    assert assignments == {downstream: 42}
    assert evidence[downstream]["method"] == "same-edits"


@pytest.mark.parametrize("different", ["    VALUE = 1\n", "VALUE = 1 \n", "VALUE = 1\r\n"])
def test_matching_preserves_whitespace(source: Path, different: str) -> None:
    base = _commit(source, "VALUE = 0\n")
    downstream = _commit(source, "VALUE = 1\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, different)
    with pytest.raises(ValueError, match="Upstream attribution is unresolved"):
        m.match(source, base, downstream, {42: _pr(base, upstream)})


def test_decorator_insertion_ignores_preceding_function_label(source: Path) -> None:
    original = "def old_helper():\n    pass\n\n\ndef run():\n    return 1\n"
    rebased = original.replace("old_helper", "new_helper")
    base = _commit(source, original)
    downstream = _commit(source, original.replace("def run():", "@cache\ndef run():"))
    _git(source, "checkout", "-q", "-b", "upstream", base)
    context = _commit(source, rebased)
    upstream = _commit(source, rebased.replace("def run():", "@cache\ndef run():"))
    assert "@@ def old_helper():" in _git(source, "diff", "--unified=0", base, downstream)
    assert "@@ def new_helper():" in _git(source, "diff", "--unified=0", context, upstream)
    assignments, evidence = m.match(source, base, downstream, {42: _pr(context, upstream)})
    assert assignments == {downstream: 42}
    assert evidence[downstream]["method"] == "same-edits"


def test_hunk_labels_are_not_part_of_edit_identity(source: Path) -> None:
    original = "def first():\n    return False\n\n\ndef second():\n    return False\n"
    base = _commit(source, original)
    downstream = _commit(source, original.replace("return False", "return True", 1))
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(
        source,
        original.replace("def second():\n    return False", "def second():\n    return True"),
    )
    assignments, evidence = m.match(source, base, downstream, {42: _pr(base, upstream)})
    assert assignments == {downstream: 42}
    assert evidence[downstream]["method"] == "same-edits"
    with pytest.raises(ValueError, match="#41.*#42"):
        m.match(source, base, downstream, {41: _pr(base, downstream), 42: _pr(base, upstream)})


@pytest.mark.parametrize("added", [True, False], ids=["added", "removed"])
def test_changed_function_definitions_are_not_ignored(source: Path, added: bool) -> None:
    original = "def first():\n    pass\n\n\ndef second():\n    pass\n"
    base = _commit(source, "# base\n" if added else original)
    downstream = _commit(source, "def first():\n    pass\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "def second():\n    pass\n")
    with pytest.raises(ValueError, match="no exact match"):
        m.match(source, base, downstream, {42: _pr(base, upstream)})


def test_hunk_like_changed_text_is_not_ignored(source: Path) -> None:
    base = _commit(source, "# base\n", "patch.txt")
    downstream = _commit(source, "@@ -1 +1 @@ def first():\n", "patch.txt")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "@@ -1 +1 @@ def second():\n", "patch.txt")
    with pytest.raises(ValueError, match="no exact match"):
        m.match(source, base, downstream, {42: _pr(base, upstream)})


def test_matching_squashed_series_and_ambiguous_duplicates(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    first = _commit(source, "VALUE = 1\n")
    second = _commit(source, "VALUE = 2\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    squash = _commit(source, "VALUE = 2\n")
    assignments, evidence = m.match(source, base, second, {42: _pr(base, squash)})
    assert assignments == {first: 42, second: 42}
    assert evidence[first]["method"] == "source-series-aggregate"
    with pytest.raises(ValueError, match="#42.*#43"):
        m.match(source, base, second, {42: _pr(base, squash), 43: _pr(base, squash)})


def test_author_can_resolve_adapted_commit(source: Path) -> None:
    base = _commit(source, "VALUE = 0\n")
    downstream = _commit(source, "VALUE = 1\n")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "VALUE = 2\n")
    assignments, evidence = m.match(
        source,
        base,
        downstream,
        {42: _pr(base, upstream)},
        {downstream: {"upstream_pr": 42, "reason": "downstream ABI adaptation"}},
    )
    assert assignments == {downstream: 42}
    assert evidence[downstream]["refresh_policy"] == "retain-until-verified"
    with pytest.raises(ValueError, match="outside"):
        m.match(source, base, downstream, {42: _pr(base, upstream)}, {base: {"upstream_pr": 42}})


@pytest.mark.parametrize("different", ["a \n", "a\r\n", "a\v\n"])
def test_similarity_preserves_line_bytes(different: str) -> None:
    assert m._similarity("a\n", different) == 0
    assert m._similarity("", "") == 0


def test_similarity_keeps_repeated_lines() -> None:
    first = "different\n" + "repeat\n" * 299
    second = "another\n" + "repeat\n" * 299
    assert m._similarity(first, second) == pytest.approx(299 / 300)


@pytest.mark.parametrize("changed_lines", [1, 2, 3])
def test_similarity_threshold_and_provenance(source: Path, changed_lines: int) -> None:
    base = _commit(source, "")
    lines = [f"VALUE_{index} = {index}\n" for index in range(16)]
    downstream = _commit(source, "".join(lines))
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "".join(lines[:-changed_lines]) + "adapted\n" * changed_lines)
    expected = (20 - changed_lines) / 20
    if expected < 0.9:
        with pytest.raises(ValueError, match="no exact match") as failure:
            m.match(source, base, downstream, {42: _pr(base, upstream)})
        assert "upstream PR #42: 85.00%" in str(failure.value)
        assert "no PR meets the 90.00% similarity threshold" in str(failure.value)
        return
    assignments, evidence = m.match(source, base, downstream, {42: _pr(base, upstream)})
    assert assignments == {downstream: 42}
    assert evidence[downstream] == {
        "method": "similar-edits",
        "upstream_base": base,
        "upstream_head": upstream,
        "upstream_pr": 42,
        "similarity": expected,
        "similarity_threshold": 0.9,
        "similarity_metric": "sequence-matcher-lines-v1",
        "refresh_policy": "retain-until-verified",
        "comparison": {
            "source_base": base,
            "source_head": downstream,
            "upstream_base": base,
            "upstream_head": upstream,
        },
    }
    assert f"**{expected:.2%}** — accepted" in m.similarity_feedback(evidence)


def test_multiple_similar_prs_remain_ambiguous(source: Path) -> None:
    base = _commit(source, "")
    lines = [f"VALUE_{index} = {index}\n" for index in range(16)]
    downstream = _commit(source, "".join(lines))
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "".join(lines[:-1]) + "adapted\n")
    with pytest.raises(ValueError, match="#41.*#42") as failure:
        m.match(source, base, downstream, {41: _pr(base, upstream), 42: _pr(base, upstream)})
    assert "#41: 95.00%" in str(failure.value)
    assert "#42: 95.00%" in str(failure.value)
    assert "multiple PRs meet the 90.00% similarity threshold" in str(failure.value)


def test_exact_matches_take_priority_over_similar_prs(source: Path) -> None:
    base = _commit(source, "")
    lines = [f"VALUE_{index} = {index}\n" for index in range(16)]
    first = _commit(source, "".join(lines))
    second = _commit(source, "second file\n", "second.py")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    approximate = _commit(source, "".join(lines[:-1]) + "adapted\n")
    upstream = _commit(source, "second file\n", "second.py")
    assignments, evidence = m.match(
        source, base, second, {41: _pr(base, first), 42: _pr(base, upstream)}
    )
    assert m._similarity(m._patch(source, base, first), m._patch(source, base, approximate)) == 0.95
    assert assignments == {first: 41, second: 42}
    assert all("similarity" not in item for item in evidence.values())


def test_approximate_series_cannot_hide_unmatched_individual_commits(source: Path) -> None:
    base = _commit(source, "", "src/code.py")
    lines = [f"VALUE_{index} = {index}\n" for index in range(16)]
    first = _commit(source, "".join(lines[:8]), "src/code.py")
    second = _commit(source, "".join(lines), "src/code.py")
    excluded = _commit(source, "tests only\n", "tests/test_code.py")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "".join(lines[:-1]) + "adapted\n", "src/code.py")
    assert m._similarity(m._patch(source, base, second), m._patch(source, base, upstream)) == 0.95
    with pytest.raises(ValueError, match="no exact match") as failure:
        m.match(
            source, base, excluded, {42: _pr(base, upstream)}, source="src", include=("**/*.py",)
        )
    assert first in str(failure.value) and second in str(failure.value)
    assert excluded not in str(failure.value)


def test_similarity_error_also_reports_accepted_matches(source: Path) -> None:
    base = _commit(source, "")
    lines = [f"VALUE_{index} = {index}\n" for index in range(16)]
    first = _commit(source, "".join(lines))
    second = _commit(source, "unrelated\n" * 16, "second.py")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(source, "".join(lines[:-1]) + "adapted\n")
    with pytest.raises(ValueError, match="no exact match") as failure:
        m.match(source, base, second, {42: _pr(base, upstream)})
    assert f"`{second}`: no exact match" in str(failure.value)
    assert f"`{first}`: upstream PR #42, similarity **95.00%** — accepted" in str(failure.value)


class FakeGitHub(p.GitHub):
    def __init__(self) -> None:
        super().__init__("org/consumer", "toy", "org/library")
        self.pr = {
            "number": 7,
            "user": {"login": "author"},
            "state": "open",
            "merged": False,
            "head": {"sha": "a" * 40},
            "base": {"ref": "main"},
            "body": "",
            "updated_at": "2099-01-01T00:00:00Z",
        }
        self.comments: list[dict] = []
        self.reviews: list[dict] = []
        self.writes: list[tuple[str, str, dict]] = []
        self.merge_base_sha = "c" * 40
        self.canonical_tip = "b" * 40

    def api(
        self, endpoint: str, *, method: str = "GET", payload: dict | None = None
    ) -> dict | list:
        if method != "GET":
            self.writes.append((endpoint, method, payload))
        if endpoint == "user":
            return {"login": "operator"}
        if endpoint in ("repos/author/library", "repos/operator/library", "repos/org/library"):
            return {"source": {"full_name": self.upstream_repo}}
        if endpoint == "repos/operator/library/git/ref/heads/dev":
            return {"object": {"sha": self.canonical_tip}}
        if endpoint.startswith("repos/org/consumer/compare/"):
            return {"merge_base_commit": {"sha": self.merge_base_sha}}
        if endpoint.endswith("/pulls/7"):
            return copy.deepcopy(self.pr)
        if "/pulls?" in endpoint:
            return [copy.deepcopy(self.pr)]
        if "/files?" in endpoint:
            return [{"filename": p._LOCK}]
        if "/reviews/" in endpoint and endpoint.endswith("/dismissals"):
            number = int(endpoint.split("/")[-2])
            self.reviews[number - 1]["state"] = "DISMISSED"
            return {}
        if "/reviews/" in endpoint and method == "PUT":
            number = int(endpoint.rsplit("/", 1)[1])
            self.reviews[number - 1]["body"] = payload["body"]
            return copy.deepcopy(self.reviews[number - 1])
        if "/reviews" in endpoint:
            if method == "GET":
                return copy.deepcopy(self.reviews)
            self.reviews.append(
                {
                    "id": len(self.reviews) + 1,
                    "user": {"login": "operator"},
                    "body": payload["body"],
                    "state": "CHANGES_REQUESTED",
                }
            )
            return self.reviews[-1]
        if "/comments/" in endpoint:
            number = int(endpoint.rsplit("/", 1)[1])
            self.comments[number - 1]["body"] = payload["body"]
            return {}
        if "/comments" in endpoint:
            if method == "GET":
                return copy.deepcopy(self.comments)
            self.comments.append(
                {
                    "id": len(self.comments) + 1,
                    "user": {"login": "operator"},
                    "body": payload["body"],
                }
            )
            return self.comments[-1]
        raise AssertionError((endpoint, method, payload))


@pytest.fixture
def monitor(tmp_path: Path):
    b._STOP.clear()
    args = argparse.Namespace(
        canonical_repo="operator/library",
        canonical_branch="dev",
        refresh_branch=[],
        migrate_canonical_branch_from=None,
        fork="operator/consumer",
        repo=tmp_path,
        workdir=tmp_path,
        publish=True,
        since=None,
    )
    store = b.Store(tmp_path / "state.sqlite")
    gh = FakeGitHub()
    instance = b.Monitor(args, gh, store)
    yield instance
    store.close()
    b._STOP.clear()


@pytest.fixture
def migration(monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch):
    args = copy.copy(monitor.args)
    args.canonical_branch = "dev-new"
    args.migrate_canonical_branch_from = "dev"
    pin = _vendor("b" * 40, "https://github.com/operator/library.git", "dev-new")
    monkeypatch.setattr(p, "_main_sha", lambda gh: "c" * 40)
    monkeypatch.setattr(p, "_lock_at", lambda *args: pin)
    original_get = monitor.gh.get
    monkeypatch.setattr(
        monitor.gh,
        "get",
        lambda endpoint: {"object": {"sha": "b" * 40}}
        if "/git/ref/" in endpoint
        else original_get(endpoint),
    )
    monkeypatch.setattr(monitor.gh, "compare", lambda *args: {"status": "identical"}, raising=False)
    monkeypatch.setattr(b, "_pages", lambda *args: iter([]))
    monkeypatch.setattr(p, "_git", lambda *args: "")
    return args, pin


def test_branch_migration_preserves_workdir_and_reconciles_again(
    monitor: b.Monitor,
    migration,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="vendor-bot")
    args, _ = migration
    store = monitor.store
    for key, value in {
        "complete:7": "old-pr",
        "promotion:8": 88,
        "observation:7": {"state": "completed"},
        "problem:8": "old problem",
        "bootstrapped": True,
        "cursor": "old cursor",
        "last_poll": "old",
        "health": {"last_successful_poll": "old", "last_completed_poll": "old"},
        "tracked": [7, 8],
        "candidate:7": {"head": "a" * 40, "relevant": True},
    }.items():
        store.put(key, value)
    before_since = store.get("since")
    (args.workdir / "bot.log").write_text("previous log\n")
    restarted = b.Monitor(args, monitor.gh, store)
    assert store.get("settings")["canonical_branch"] == "dev-new"
    assert store.get("complete:7") is None
    assert store.get("promotion:8") is None
    assert store.get("problem:8") is None
    assert store.get("cursor") is None and store.get("bootstrapped") is None
    assert store.get("tracked") == [7, 8]
    assert store.get("candidate:7")["relevant"]
    assert store.get("since") == before_since
    assert store.get("branch_migrations")[0]["previous_results"]["complete:7"] == "old-pr"
    assert store.get("migration")["consumer_revision"] == "c" * 40
    assert restarted.progress.state["last_successful_poll"] is None
    assert restarted.progress.state["last_completed_poll"] is None
    assert (args.workdir / "bot.log").read_text() == "previous log\n"
    assert not monitor.gh.writes
    assert "dev -> dev-new; rediscovery required" in caplog.text
    status = b._status(args.workdir)
    assert status["canonical_branch"] == "dev-new"
    assert status["migration"]["from"] == "dev"


def test_branch_migration_retry_does_not_reset_new_progress(monitor: b.Monitor, migration) -> None:
    args, _ = migration
    b.Monitor(args, monitor.gh, monitor.store)
    monitor.store.put("cursor", "new progress")
    b.Monitor(args, monitor.gh, monitor.store)
    assert monitor.store.get("cursor") == "new progress"
    assert len(monitor.store.get("branch_migrations")) == 1
    args.migrate_canonical_branch_from = None
    b.Monitor(args, monitor.gh, monitor.store)
    assert monitor.store.get("cursor") == "new progress"


@pytest.mark.parametrize("old", [None, "wrong"])
def test_branch_migration_requires_explicit_old_branch(monitor: b.Monitor, migration, old) -> None:
    args, _ = migration
    args.migrate_canonical_branch_from = old
    with pytest.raises(ValueError, match="--migrate-canonical-branch-from dev"):
        b.Monitor(args, monitor.gh, monitor.store)
    assert monitor.store.get("settings")["canonical_branch"] == "dev"


@pytest.mark.parametrize("key", ["canonical_repo", "fork", "repo"])
def test_branch_migration_cannot_change_identity(monitor: b.Monitor, migration, key: str) -> None:
    args, _ = migration
    setattr(args, key, "other/value")
    with pytest.raises(ValueError, match="different operator/vendor"):
        b.Monitor(args, monitor.gh, monitor.store)
    assert monitor.store.get("settings")["canonical_branch"] == "dev"


@pytest.mark.parametrize(
    "branch,url",
    [
        ("dev", "https://github.com/operator/library.git"),
        ("dev-new", "https://github.com/other/library.git"),
    ],
)
def test_branch_migration_requires_merged_refresh(
    monitor: b.Monitor,
    migration,
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    url: str,
) -> None:
    args, _ = migration
    monkeypatch.setattr(p, "_lock_at", lambda *args: _vendor("b" * 40, url, branch))
    with pytest.raises(ValueError, match="merged base-branch lock"):
        b.Monitor(args, monitor.gh, monitor.store)
    assert monitor.store.get("settings")["canonical_branch"] == "dev"


@pytest.mark.parametrize("status", ["behind", "diverged"])
def test_branch_migration_rejects_unreachable_pin(
    monitor: b.Monitor,
    migration,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    args, _ = migration
    monkeypatch.setattr(monitor.gh, "compare", lambda *args: {"status": status})
    with pytest.raises(ValueError, match="does not contain"):
        b.Monitor(args, monitor.gh, monitor.store)


def test_branch_migration_rejects_lock_race(
    monitor: b.Monitor,
    migration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, pin = migration
    locks = iter([pin, dataclasses.replace(pin, commit="d" * 40)])
    monkeypatch.setattr(p, "_lock_at", lambda *args: next(locks))
    with pytest.raises(ValueError, match="changed during migration"):
        b.Monitor(args, monitor.gh, monitor.store)
    assert monitor.store.get("settings")["canonical_branch"] == "dev"


@pytest.mark.parametrize("state", ["open", "absent", "closed"])
@pytest.mark.parametrize("origin", ["remote", "local", "pr"])
def test_branch_migration_checks_pending_promotions(
    monitor: b.Monitor,
    migration,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    origin: str,
) -> None:
    args, _ = migration
    branch = "chore/toy-promote-7-aaaaaaaaaaaa"
    pr = {
        "head": {"ref": branch, "sha": "a" * 40},
        "base": {"ref": "main"},
        "state": state,
        "html_url": "https://github.com/org/consumer/pull/88",
    }

    def pages(gh, endpoint):
        if endpoint.endswith("/branches"):
            return iter(
                [{"name": branch, "commit": {"sha": "a" * 40}}] if origin == "remote" else []
            )
        if "state=open" in endpoint:
            return iter([pr] if origin == "pr" else [])
        return iter([pr] if state != "absent" else [])

    monkeypatch.setattr(b, "_pages", pages)
    monkeypatch.setattr(
        p, "_git", lambda *args: f"{branch} {'a' * 40}" if origin == "local" else ""
    )
    monkeypatch.setattr(
        monitor.gh, "optional", lambda endpoint: {"commit": {"message": "record"}}, raising=False
    )
    monkeypatch.setattr(
        p,
        "_record_from_message",
        lambda message: {
            "vendor": "toy",
            "canonical_branch": "dev",
            "consumer_pr": "https://github.com/org/consumer/pull/7",
            "consumer_base": "main",
        },
    )
    if state == "closed":
        b.Monitor(args, monitor.gh, monitor.store)
    else:
        with pytest.raises(ValueError, match="Unresolved old-branch promotion"):
            b.Monitor(args, monitor.gh, monitor.store)
        assert monitor.store.get("settings")["canonical_branch"] == "dev"
    assert not monitor.gh.writes


@pytest.mark.parametrize("base_branch", ["main", "release/1.3"])
@pytest.mark.parametrize("other_scope", ["base", "repository"])
@pytest.mark.parametrize("origin", ["local", "remote"])
def test_branch_migration_ignores_promotions_for_other_consumers(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    base_branch: str,
    other_scope: str,
    origin: str,
) -> None:
    monitor.gh.base_branch = base_branch
    other_base = "release/1.3" if base_branch == "main" else "main"
    branch = "chore/toy-promote-7-aaaaaaaaaaaa"
    record = {
        "version": 1,
        "vendor": "toy",
        "canonical_branch": "dev",
        "consumer_pr": "https://github.com/other/consumer/pull/7"
        if other_scope == "repository"
        else "https://github.com/org/consumer/pull/7",
        "consumer_base": other_base if other_scope == "base" else base_branch,
    }

    def pages(gh, endpoint):
        if endpoint.endswith("/branches"):
            return iter(
                [{"name": branch, "commit": {"sha": "a" * 40}}] if origin == "remote" else []
            )
        assert endpoint.endswith("/pulls?state=open"), endpoint
        return iter([])

    monkeypatch.setattr(b, "_pages", pages)
    monkeypatch.setattr(
        p, "_git", lambda *args: f"{branch} {'a' * 40}" if origin == "local" else ""
    )
    monkeypatch.setattr(
        monitor.gh,
        "optional",
        lambda endpoint: {"commit": {"message": p._record_text(record)}},
    )
    monitor._check_pending_promotions("dev")
    assert not monitor.gh.writes


@pytest.mark.parametrize("field", ["consumer_pr", "consumer_base"])
@pytest.mark.parametrize("value", [None, "", 123])
def test_branch_migration_rejects_ambiguous_consumer_provenance(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    branch = "chore/toy-promote-7-aaaaaaaaaaaa"
    record = {
        "version": 1,
        "vendor": "toy",
        "canonical_branch": "dev",
        "consumer_pr": "https://github.com/org/consumer/pull/7",
        "consumer_base": "main",
        field: value,
    }
    monkeypatch.setattr(b, "_pages", lambda *args: iter([]))
    monkeypatch.setattr(p, "_git", lambda *args: f"{branch} {'a' * 40}")
    monkeypatch.setattr(
        monitor.gh,
        "optional",
        lambda endpoint: {"commit": {"message": p._record_text(record)}},
    )
    with pytest.raises(ValueError, match="Missing or invalid consumer provenance"):
        monitor._check_pending_promotions("dev")
    assert not monitor.gh.writes


def test_branch_migration_sqlite_transaction_rolls_back(monitor: b.Monitor, migration) -> None:
    import sqlite3

    args, _ = migration
    monitor.store.put("cursor", "preserved")
    monitor.store.put("complete:7", "preserved")
    monitor.store.connection.execute(
        "CREATE TRIGGER fail_migration BEFORE INSERT ON state WHEN NEW.key = 'migration' "
        "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        b.Monitor(args, monitor.gh, monitor.store)
    assert monitor.store.get("settings")["canonical_branch"] == "dev"
    assert monitor.store.get("cursor") == "preserved"
    assert monitor.store.get("complete:7") == "preserved"
    assert monitor.store.get("branch_migrations") is None
    monitor.store.connection.execute("DROP TRIGGER fail_migration")
    b.Monitor(args, monitor.gh, monitor.store)


def test_branch_migration_cache_loss_uses_normal_start(
    monitor: b.Monitor, migration, tmp_path: Path
) -> None:
    args, _ = migration
    with contextlib.closing(b.Store(tmp_path / "fresh.sqlite")) as store:
        with pytest.raises(ValueError, match="fresh/cache-loss start"):
            b.Monitor(args, monitor.gh, store)
        args.migrate_canonical_branch_from = None
        b.Monitor(args, monitor.gh, store)
        assert store.get("settings")["canonical_branch"] == "dev-new"
        assert store.get("bootstrapped") is None


def test_feedback_requests_changes_once_and_clears_only_metadata_review(monitor: b.Monitor) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue())
    monitor._feedback(gh.pr, _issue())
    assert len(gh.comments) == len(gh.reviews) == 1
    assert "PR description" in gh.comments[0]["body"]
    assert "resolutions:" not in gh.comments[0]["body"]
    assert "**Fix:**" in gh.comments[0]["body"]
    assert gh.writes[1][2]["event"] == "REQUEST_CHANGES"
    monitor._feedback(gh.pr, None)
    assert gh.reviews[0]["state"] == "DISMISSED"
    assert all(payload.get("event") != "APPROVE" for _, _, payload in gh.writes)


def test_bot_does_not_override_operator_manual_review(monitor: b.Monitor) -> None:
    monitor.gh.reviews.append(
        {
            "id": 1,
            "user": {"login": "operator"},
            "body": "Manual code review",
            "state": "CHANGES_REQUESTED",
        }
    )
    monitor._feedback(monitor.gh.pr, _issue("Missing metadata"))
    monitor._feedback(monitor.gh.pr, None)
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    assert len(monitor.gh.reviews) == 1


def test_dry_run_and_self_review(monitor: b.Monitor) -> None:
    monitor.args.publish = False
    monitor._feedback(monitor.gh.pr, _issue("Missing metadata"))
    assert not monitor.gh.writes
    monitor.args.publish = True
    monitor.gh.pr["user"]["login"] = "operator"
    monitor._feedback(monitor.gh.pr, _issue("Missing metadata"))
    assert monitor.gh.comments
    assert not monitor.gh.reviews


def test_feedback_rechecks_mutable_description(monitor: b.Monitor) -> None:
    stale = copy.deepcopy(monitor.gh.pr)
    monitor.gh.pr["body"] = "updated while checking"
    monitor._feedback(stale, _issue("Missing metadata"))
    assert not monitor.gh.writes


@pytest.mark.parametrize("missing", [(m.BEGIN, m.END), (m.BEGIN,), (m.END,)])
def test_posted_marker_feedback_is_specific_and_recovers_without_cache(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, missing: tuple[str, ...]
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    valid = m.template("toy", "org/library")
    monitor.gh.pr["body"] = valid
    for marker in missing:
        monitor.gh.pr["body"] = monitor.gh.pr["body"].replace(marker, "")
    monitor.inspect(7)
    body = monitor.gh.comments[0]["body"]
    for marker in missing:
        assert f"`{marker}`" in body
    assert "**Fix:**" in body and "PR's description—not a comment" in body
    assert "resolutions:" not in body and "exactly one" not in body
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    writes = copy.deepcopy(monitor.gh.writes)
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, monitor.gh, store)
        monkeypatch.setattr(restarted, "_inputs", lambda pr: (previous, reviewed))
        restarted.inspect(7)
        assert monitor.gh.writes == writes
        monitor.gh.pr["body"] = valid
        monkeypatch.setattr(m, "resolve", lambda *args: ({reviewed.commit: 123}, {}))
        restarted.inspect(7)
    assert len(monitor.gh.comments) == 2
    assert monitor.gh.comments[0]["body"] == body
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"
    assert not any(payload.get("event") == "APPROVE" for _, _, payload in monitor.gh.writes)


@pytest.mark.parametrize("stage", ["inputs", "resolve", "canonical-tip"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, b.manage.VendorError])
def test_unrecognized_failures_are_operator_errors_not_author_feedback(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, stage: str, error_type: type[Exception]
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monitor.gh.pr["body"] = m.template("toy", "org/library")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))

    def fail(*args):
        raise error_type("Operator failure; do not request author changes")

    if stage == "inputs":
        monkeypatch.setattr(monitor, "_inputs", fail)
    elif stage == "canonical-tip":
        monkeypatch.setattr(p, "_check_canonical_tip", fail)
    else:
        monkeypatch.setattr(m, "resolve", fail)
    monitor.poll()
    assert not monitor.gh.writes
    health = monitor.store.get("health")
    assert health["counts"]["errors"] == 1
    assert "Operator failure" in health["last_error"]["message"]


def test_valid_feedback_is_rechecked_when_only_dev_tip_changes(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    monkeypatch.setattr(m, "resolve", lambda *args: ({}, {}))
    monitor.gh.pr["body"] = m.template("toy", "org/library")
    original_pr = copy.deepcopy(monitor.gh.pr)
    monitor.inspect(7)
    assert monitor.store.get("observation:7")["state"] == "metadata_valid"

    monitor.gh.canonical_tip = "c" * 40
    monkeypatch.setattr(
        monitor.gh, "compare", lambda *args: {"merge_base_commit": {"sha": previous.commit}}
    )
    monitor.inspect(7)
    assert monitor.gh.pr == original_pr
    assert monitor.store.get("observation:7")["state"] == "blocked"
    assert len(monitor.gh.comments) == 2
    assert "latest canonical branch tip" in monitor.gh.comments[-1]["body"]
    assert monitor.gh.canonical_tip in monitor.gh.comments[-1]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"


def test_upstream_main_reconciles_obsolete_review_without_sqlite(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor._feedback(monitor.gh.pr, _issue("Upstream is not a personal fork"))
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/org/library.git", "main")
    api = monitor.gh.api
    upstream_tip = "c" * 40

    def get(endpoint, **kwargs):
        if endpoint == "repos/org/library/git/ref/heads/main":
            return {"object": {"sha": upstream_tip}}
        return api(endpoint, **kwargs)

    monkeypatch.setattr(monitor.gh, "api", get)
    monkeypatch.setattr(
        monitor.gh, "compare", lambda repo, base, head: {"merge_base_commit": {"sha": base}}
    )
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, monitor.gh, store)
        monkeypatch.setattr(restarted, "_inputs", lambda pr: (previous, reviewed))
        restarted.inspect(7)
        assert store.get("observation:7")["state"] == "metadata_valid"
        assert len(monitor.gh.comments) == 2 and len(monitor.gh.reviews) == 1
        assert monitor.gh.reviews[0]["state"] == "DISMISSED"
        assert "Upstream-main source ancestry is valid" in monitor.gh.comments[-1]["body"]
        assert reviewed.commit in monitor.gh.comments[-1]["body"]
        writes = copy.deepcopy(monitor.gh.writes)
        restarted.inspect(7)
        assert monitor.gh.writes == writes

        # Same PR and pin, but upstream no longer contains it: invalidate feedback.
        monkeypatch.setattr(
            monitor.gh,
            "compare",
            lambda repo, base, head: {"merge_base_commit": {"sha": previous.commit}},
        )
        restarted.inspect(7)
        assert store.get("observation:7")["state"] == "blocked"
        assert len(monitor.gh.comments) == 3
        assert "not reachable" in monitor.gh.comments[-1]["body"]
        assert monitor.gh.reviews[-1]["state"] == "CHANGES_REQUESTED"


def test_upstream_diagnostic_does_not_blame_author_for_incomplete_checks(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/org/library.git", "pull/42/head")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    monkeypatch.setattr(
        monitor.gh, "compare", lambda repo, base, head: {"merge_base_commit": {"sha": base}}
    )
    api = monitor.gh.api

    def unavailable(endpoint, **kwargs):
        if endpoint == "repos/org/library/git/ref/heads/main":
            raise RuntimeError("Upstream API unavailable")
        return api(endpoint, **kwargs)

    monkeypatch.setattr(monitor.gh, "api", unavailable)
    monitor.poll()
    assert not monitor.gh.writes
    assert monitor.store.get("health")["counts"]["errors"] == 1


def test_temporary_fork_requires_own_vendor_metadata(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Metadata for another vendor does not cover this vendor's temporary fork.
    monitor.gh.pr["body"] = m.template("other-vendor", "org/library")
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "main")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    monitor.inspect(7)
    assert monitor.store.get("observation:7")["state"] == "blocked"
    assert "vendors.toy" in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"


@pytest.mark.parametrize("change", ["other-vendor", "patch-only", "vendor-created"])
def test_refresh_ignores_prs_without_selected_source_pin_change(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    current = _vendor("1" * 40, "https://github.com/operator/library.git", "dev")
    old = _vendor("2" * 40, "https://github.com/operator/library.git", "dev-old")
    original = {"toy": old.to_mapping()}
    reviewed = copy.deepcopy(original)
    if change == "other-vendor":
        reviewed["other"] = current.to_mapping()
    elif change == "patch-only":
        reviewed["toy"]["digest"] = "sha256-tree-v1:" + "3" * 64
    else:
        original = {}
    monitor.gh.merge_base_sha = "d" * 40
    locks = {"c" * 40: {"toy": current.to_mapping()}, "d" * 40: original, "a" * 40: reviewed}
    monkeypatch.setattr(p, "_main_sha", lambda gh: "c" * 40)
    monkeypatch.setattr(p, "_lock_document_at", lambda gh, repo, sha: {"vendors": locks[sha]})
    monitor.inspect(7)
    assert not monitor.gh.writes
    assert monitor._inputs(monitor.gh.pr) is None
    with contextlib.closing(b.Store(monitor.args.workdir / "fresh.sqlite")) as store:
        b.Monitor(monitor.args, monitor.gh, store).inspect(7)
    assert not monitor.gh.writes


def test_refresh_source_update_uses_current_canonical_as_promotion_baseline(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = _vendor("1" * 40, "https://github.com/operator/library.git", "dev")
    old = _vendor("2" * 40, "https://github.com/operator/library.git", "dev-old")
    reviewed = _vendor("3" * 40, "https://github.com/author/library.git", "temporary")
    monitor.gh.merge_base_sha = "d" * 40
    locks = {"c" * 40: current, "d" * 40: old, "a" * 40: reviewed}
    monkeypatch.setattr(p, "_main_sha", lambda gh: "c" * 40)
    monkeypatch.setattr(
        p, "_lock_document_at", lambda gh, repo, sha: {"vendors": {"toy": locks[sha].to_mapping()}}
    )
    assert monitor._inputs(monitor.gh.pr) == (current, reviewed)
    assert not monitor.gh.writes


@pytest.fixture
def refresh_candidate(monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch):
    previous = _vendor("1" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("2" * 40, "https://github.com/operator/library.git", "dev-new")
    monitor.gh.canonical_tip = previous.commit
    locks = {"c" * 40: previous, "a" * 40: reviewed}
    monkeypatch.setattr(p, "_main_sha", lambda gh: "c" * 40)
    monkeypatch.setattr(
        p, "_lock_document_at", lambda gh, repo, sha: {"vendors": {"toy": locks[sha].to_mapping()}}
    )
    original_get = monitor.gh.get
    monkeypatch.setattr(
        monitor.gh,
        "get",
        lambda endpoint: {"parents": [{"sha": "c" * 40}]}
        if endpoint.endswith("/commits/" + "a" * 40)
        else original_get(endpoint),
    )
    return locks


@pytest.mark.parametrize("base_branch", ["main", "release/1.3"])
@pytest.mark.parametrize("redirect", ["repository", "branch", "both"])
@pytest.mark.parametrize("caught_up", [False, True])
def test_same_commit_reference_redirect_requests_canonical_lock_not_metadata(
    monitor: b.Monitor, refresh_candidate, base_branch: str, redirect: str, caught_up: bool
) -> None:
    gh = monitor.gh
    gh.base_branch = gh.pr["base"]["ref"] = base_branch
    previous = refresh_candidate["c" * 40]
    reviewed = dataclasses.replace(
        previous,
        url="https://github.com/author/library.git" if redirect != "branch" else previous.url,
        branch="temporary" if redirect != "repository" else previous.branch,
    )
    refresh_candidate["a" * 40] = reviewed
    if caught_up:
        gh.merge_base_sha = "d" * 40
        refresh_candidate["d" * 40] = dataclasses.replace(previous, commit="0" * 40)
    gh.pr["body"] = "Invalid/missing promotion metadata must not obscure the lock fix."
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, gh, store)
        restarted.inspect(7)
        assert store.get("observation:7")["state"] == "blocked"
    body = gh.comments[0]["body"]
    assert "Temporary source reference adds no new vendor commit" in body
    assert previous.commit in body and base_branch in body
    assert p._LOCK in body and "vendors.toy" in body
    assert previous.url in body and f"<code>{previous.branch}</code>" in body
    assert "keep the commit and other lock fields, vendored files, and patch unchanged" in body
    assert "No upstream-PR metadata" in body
    assert "vendor-promotion:start" not in body
    assert gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    writes = copy.deepcopy(gh.writes)
    with contextlib.closing(b.Store(":memory:")) as store:
        b.Monitor(monitor.args, gh, store).inspect(7)
    assert gh.writes == writes


@pytest.mark.parametrize(
    "case", ["canonical", "equivalent-url", "refresh", "inherited", "merged", "promotion"]
)
def test_same_commit_reference_non_redirects_do_not_request_changes(
    monitor: b.Monitor, refresh_candidate, case: str
) -> None:
    previous = refresh_candidate["c" * 40]
    reviewed = previous
    if case == "equivalent-url":
        reviewed = dataclasses.replace(previous, url="https://github.com/OPERATOR/library")
    elif case == "refresh":
        reviewed = dataclasses.replace(previous, branch="dev-new")
        monitor.args.refresh_branch = ["dev-new"]
    elif case in ("inherited", "merged"):
        reviewed = dataclasses.replace(previous, url="https://github.com/author/library.git")
        if case == "inherited":
            monitor.gh.merge_base_sha = "d" * 40
            refresh_candidate["d" * 40] = reviewed
        else:
            monitor.gh.pr.update(merged=True, state="closed", merge_commit_sha="a" * 40)
    elif case == "promotion":
        refresh_candidate["c" * 40] = dataclasses.replace(
            previous, url="https://github.com/author/library.git", branch="temporary"
        )
    refresh_candidate["a" * 40] = reviewed
    with contextlib.closing(b.Store(":memory:")) as store:
        b.Monitor(monitor.args, monitor.gh, store).inspect(7)
        assert store.get("observation:7")["state"] == "ignored"
    assert not monitor.gh.writes


def test_same_commit_reference_reconciles_stale_feedback_then_clears_after_fix(
    monitor: b.Monitor, refresh_candidate
) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue("Source update is not a fast-forward."))
    previous = refresh_candidate["c" * 40]
    refresh_candidate["a" * 40] = dataclasses.replace(
        previous, url="https://github.com/author/library.git", branch="temporary"
    )
    monitor.inspect(7)
    assert len(gh.comments) == 2 and len(gh.reviews) == 1
    assert "adds no new vendor commit" in gh.comments[-1]["body"]
    assert "#issuecomment-2" in gh.reviews[0]["body"]
    assert "fast-forward" not in gh.reviews[0]["body"]
    assert gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    assert any(
        method == "PUT" and endpoint.endswith("/reviews/1") for endpoint, method, _ in gh.writes
    )
    refresh_candidate["a" * 40] = previous
    for _ in range(2):
        with contextlib.closing(b.Store(":memory:")) as store:
            b.Monitor(monitor.args, gh, store).inspect(7)
            assert store.get("observation:7")["state"] == "ignored"
            assert store.get("problem:7") is None
    assert len(gh.comments) == 3 and len(gh.reviews) == 1
    assert "does not require vendor promotion" in gh.comments[-1]["body"]
    assert "metadata and source attribution are valid" not in gh.comments[-1]["body"]
    assert gh.reviews[0]["state"] == "DISMISSED"
    assert not any(payload.get("event") == "APPROVE" for _, _, payload in gh.writes)
    writes = copy.deepcopy(gh.writes)
    monitor.inspect(7)
    assert gh.writes == writes


@pytest.mark.parametrize("remaining", ["comment", "review", "both", "neither"])
def test_ignored_pr_cleans_remote_feedback_without_sqlite(
    monitor: b.Monitor, refresh_candidate, remaining: str
) -> None:
    monitor._feedback(monitor.gh.pr, _issue("Missing block"))
    if remaining in ("review", "neither"):
        monitor.gh.comments.clear()
    if remaining in ("comment", "neither"):
        monitor.gh.reviews.clear()
    monitor.gh.writes.clear()
    monitor.args.refresh_branch = ["dev-new"]
    for _ in range(2):
        with contextlib.closing(b.Store(":memory:")) as store:
            b.Monitor(monitor.args, monitor.gh, store).inspect(7)
            assert store.get("observation:7")["state"] == "ignored"
    if remaining == "neither":
        assert not monitor.gh.writes
    else:
        assert len(monitor.gh.comments) == (1 if remaining == "review" else 2)
        assert "no longer applies" in monitor.gh.comments[-1]["body"]
        assert "not** a code-review approval" in monitor.gh.comments[-1]["body"]
        assert all(review["state"] == "DISMISSED" for review in monitor.gh.reviews)
        assert len(monitor.gh.writes) == (3 if remaining in ("review", "both") else 1)


@pytest.mark.parametrize("existing", ["manual", "other-actor", "other-vendor"])
def test_ignored_pr_preserves_unrelated_comments_and_reviews(
    monitor: b.Monitor, refresh_candidate, existing: str
) -> None:
    actor = "human" if existing == "other-actor" else monitor.actor
    body = (
        "Manual review"
        if existing == "manual"
        else "<!-- vendor-bot:other -->\nOther vendor"
        if existing == "other-vendor"
        else f"{monitor.marker}\nAnother operator's feedback"
    )
    monitor.gh.comments.append({"id": 1, "user": {"login": actor}, "body": body})
    monitor.gh.reviews.append(
        {"id": 1, "user": {"login": actor}, "body": body, "state": "CHANGES_REQUESTED"}
    )
    monitor.args.refresh_branch = ["dev-new"]
    b.Monitor(monitor.args, monitor.gh, monitor.store).inspect(7)
    assert not monitor.gh.writes
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"


@pytest.mark.parametrize("change", ["head", "body", "base", "closed", "dry-run"])
def test_ignored_feedback_cleanup_respects_concurrent_changes_and_dry_run(
    monitor: b.Monitor, change: str
) -> None:
    monitor._feedback(monitor.gh.pr, _issue("Missing block"))
    snapshot = copy.deepcopy(monitor.gh.pr)
    if change == "head":
        monitor.gh.pr["head"]["sha"] = "b" * 40
    elif change == "body":
        monitor.gh.pr["body"] = "edited"
    elif change == "base":
        monitor.gh.pr["base"]["ref"] = "release/1.3"
    elif change == "closed":
        monitor.gh.pr["state"] = "closed"
    else:
        monitor.args.publish = False
    writes = copy.deepcopy(monitor.gh.writes)
    monitor._feedback(snapshot, None, no_promotion=True)
    assert monitor.gh.writes == writes
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"


@pytest.mark.parametrize("base_branch", ["main", "release/1.3"])
@pytest.mark.parametrize("merged", [False, True])
def test_explicit_refresh_target_never_requests_metadata_or_promotes(
    monitor: b.Monitor, refresh_candidate, base_branch: str, merged: bool
) -> None:
    monitor.gh.base_branch = base_branch
    monitor.gh.pr["base"]["ref"] = base_branch
    monitor.gh.pr.update(merged=merged, merge_commit_sha="a" * 40)
    monitor.args.refresh_branch = ["dev-new"]
    for _ in range(2):
        with contextlib.closing(b.Store(":memory:")) as store:
            restarted = b.Monitor(monitor.args, monitor.gh, store)
            restarted.inspect(7)
            assert store.get("observation:7")["state"] == "ignored"
            assert store.get("refresh_branches") == ["dev-new"]
    assert not monitor.gh.writes


@pytest.mark.parametrize("untrusted", ["branch", "repository"])
def test_refresh_targets_are_explicit_and_repository_scoped(
    monitor: b.Monitor, refresh_candidate, untrusted: str
) -> None:
    if untrusted == "repository":
        monitor.args.refresh_branch = ["dev-new"]
        refresh_candidate["a" * 40] = dataclasses.replace(
            refresh_candidate["a" * 40], url="https://github.com/author/library.git"
        )
    restarted = b.Monitor(monitor.args, monitor.gh, monitor.store)
    assert restarted._inputs(monitor.gh.pr) == (
        refresh_candidate["c" * 40],
        refresh_candidate["a" * 40],
    )
    restarted.inspect(7)
    assert "vendor-promotion:start" in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"


def test_refresh_policy_can_change_without_migrating_workdir(
    monitor: b.Monitor, refresh_candidate
) -> None:
    monitor.store.put("cursor", "preserved")
    monitor.store.put("complete:8", "preserved")
    for branches in (["dev-new", "dev-new"], []):
        monitor.args.refresh_branch = branches
        restarted = b.Monitor(monitor.args, monitor.gh, monitor.store)
        assert (restarted._inputs(monitor.gh.pr) is None) == bool(branches)
        assert monitor.store.get("settings")["canonical_branch"] == "dev"
        assert monitor.store.get("cursor") == "preserved"
        assert monitor.store.get("complete:8") == "preserved"
        assert monitor.store.get("refresh_branches") == sorted(set(branches))
        assert monitor.store.get("migration") is None
    assert not monitor.gh.writes


def test_updates_after_refresh_require_maintainer_cutover(
    monitor: b.Monitor, refresh_candidate
) -> None:
    monitor.args.refresh_branch = ["dev-new"]
    refresh_candidate["c" * 40] = refresh_candidate["a" * 40]
    refresh_candidate["a" * 40] = _vendor(
        "3" * 40, "https://github.com/author/library.git", "temporary"
    )
    restarted = b.Monitor(monitor.args, monitor.gh, monitor.store)
    restarted.inspect(7)
    comment = monitor.gh.comments[0]["body"]
    assert "maintainer action" in comment
    assert "--canonical-branch dev-new --migrate-canonical-branch-from dev" in comment
    assert "Do not promote back" in comment
    assert not monitor.gh.reviews


def test_waiting_for_prior_promotion_is_maintainer_action_without_review(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _vendor("a" * 40, "https://github.com/previous/library.git", "prior-update")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monkeypatch.setattr(p, "_main_sha", lambda gh: "c" * 40)
    monkeypatch.setattr(
        p,
        "_lock_document_at",
        lambda gh, repo, sha: {
            "vendors": {"toy": (previous if sha == "c" * 40 else reviewed).to_mapping()}
        },
    )
    monitor.inspect(7)
    body = monitor.gh.comments[0]["body"]
    assert "needs maintainer action" in body and "Waiting for the preceding" in body
    assert "operator/library" in body and "dev" in body
    assert "After it merges" in body and "No metadata edit" in body
    assert "```yaml" not in body and "resolutions:" not in body
    assert not monitor.gh.reviews
    writes = copy.deepcopy(monitor.gh.writes)
    monitor.inspect(7)
    assert monitor.gh.writes == writes


def test_invalid_pr_lock_has_file_fix_not_metadata_template(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    invalid = previous.to_mapping() | {"commit": "short-sha"}
    monkeypatch.setattr(p, "_main_sha", lambda gh: "c" * 40)
    monkeypatch.setattr(
        p,
        "_lock_document_at",
        lambda gh, repo, sha: {
            "vendors": {"toy": previous.to_mapping() if sha == "c" * 40 else invalid}
        },
    )
    monitor.inspect(7)
    body = monitor.gh.comments[0]["body"]
    assert "vendors.toy" in body and "40-hex SHA" in body
    assert p._LOCK in body and "Editing the PR description will not fix" in body
    assert "```yaml" not in body
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"


@pytest.mark.parametrize("history", ["diverged", "unrelated", "merge"])
def test_source_history_diagnostic_has_canonical_pin_and_regeneration_steps(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history: str
) -> None:
    base = _commit(source, "base\n", "src/code.py")
    canonical = _commit(source, "canonical\n", "canonical.py")
    _git(source, "checkout", "-q", "-b", "temporary", base)
    if history == "unrelated":
        _git(source, "checkout", "-q", "--orphan", "unrelated")
    reviewed_sha = _commit(source, "change\n", "src/code.py")
    if history == "merge":
        _git(source, "merge", "--no-ff", "-m", "merge canonical", canonical)
        reviewed_sha = _git(source, "rev-parse", "HEAD")
    previous = _vendor(canonical, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor(reviewed_sha, "https://github.com/author/library.git", "temporary")
    original_git = p._git

    def local_git(repo, *args, **kwargs):
        if args and args[0] == "fetch":
            args = tuple(str(source) if arg == reviewed.url else arg for arg in args)
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(p, "_git", local_git)
    monkeypatch.setattr(p, "_check_source_repo", lambda *args: None)
    with pytest.raises(m.Diagnostic) as failure:
        m.resolve(
            FakeGitHub(),
            previous,
            reviewed,
            {"unpaired_reason": "downstream-only", "upstream_prs": [], "resolutions": {}},
            tmp_path / "cache.git",
        )
    issue = failure.value
    assert {
        "diverged": "not a fast-forward",
        "unrelated": "no merge base",
        "merge": "linear history",
    }[history] in issue.problem
    assert (
        canonical in issue.action and "operator/library" in issue.action and "dev" in issue.action
    )
    assert "temporary" in issue.action and "scripts/vendor/manage.py" in issue.action
    assert "latest tip without merge commits" in issue.action
    assert "vendored snapshot" in issue.action and "push" in issue.action
    assert not issue.example


@pytest.mark.parametrize("status", ["closed", "wrong-target"])
def test_upstream_pr_diagnostic_links_affected_pr_and_fix(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    base = _commit(source, "base\n", "src/code.py")
    head = _commit(source, "changed\n", "src/code.py")
    previous = _vendor(base, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor(head, "https://github.com/author/library.git", "temporary")
    gh = FakeGitHub()
    pr = _pr(base, head)
    if status == "closed":
        pr["state"] = "closed"
    else:
        pr["base"]["ref"] = "release"
    monkeypatch.setattr(gh, "get", lambda endpoint: pr)
    monkeypatch.setattr(p, "_check_source_repo", lambda *args: None)
    original_git = p._git

    def local_git(repo, *args, **kwargs):
        if args and args[0] == "fetch":
            args = tuple(str(source) if arg == reviewed.url else arg for arg in args)
        return original_git(repo, *args, **kwargs)

    monkeypatch.setattr(p, "_git", local_git)
    with pytest.raises(m.Diagnostic) as failure:
        m.resolve(
            gh,
            previous,
            reviewed,
            {"unpaired_reason": None, "upstream_prs": [42], "resolutions": {}},
            tmp_path / "cache.git",
        )
    assert "https://github.com/org/library/pull/42" in failure.value.problem
    assert (
        "closed without merging" in failure.value.problem
        if status == "closed"
        else "release" in failure.value.problem
    )
    assert "reopen" in failure.value.action and "replace its URL" in failure.value.action
    assert not failure.value.example


def test_git_expected_absence_does_not_hide_unexpected_failures(source: Path) -> None:
    base = _commit(source, "base\n")
    with pytest.raises(RuntimeError, match="git failed"):
        p._git(source, "merge-base", base, "f" * 40, allowed_returncodes=(0, 1))


def test_discovery_and_persistent_operator_scope(monitor: b.Monitor) -> None:
    assert monitor._discover() == [7]
    assert monitor._discover() == [7]
    assert not monitor.gh.writes
    changed = copy.copy(monitor.args)
    changed.canonical_repo = "different/library"
    with pytest.raises(ValueError, match="different operator"):
        b.Monitor(changed, monitor.gh, monitor.store)


@pytest.mark.parametrize("base_branch", ["main", "release/1.3"])
def test_monitors_discover_only_their_consumer_base(monitor: b.Monitor, base_branch: str) -> None:
    monitor.gh.base_branch = base_branch
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, monitor.gh, store)
        monitor.gh.pr["base"]["ref"] = "release/1.3" if base_branch == "main" else "main"
        assert restarted._discover() == []
        monitor.gh.pr["base"]["ref"] = base_branch
        assert restarted._discover() == [7]
    assert not monitor.gh.writes


def test_workdir_cannot_change_consumer_base(monitor: b.Monitor) -> None:
    monitor.gh.base_branch = "release/1.3"
    with pytest.raises(ValueError, match="different operator/vendor configuration"):
        b.Monitor(monitor.args, monitor.gh, monitor.store)
    assert monitor.store.get("settings")["base_branch"] == "main"


@pytest.mark.parametrize("base_branch", ["main", "release/1.3"])
def test_promotion_identity_and_followup_keep_consumer_base(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, base_branch: str
) -> None:
    gh = monitor.gh
    gh.base_branch = base_branch
    gh.pr.update(merged=True, merge_commit_sha="b" * 40)
    gh.pr["base"]["ref"] = base_branch
    previous = _vendor("1" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("2" * 40, "https://github.com/author/library.git", "temporary")
    original_get = gh.get

    def get(endpoint: str) -> dict:
        if endpoint.endswith("/commits/" + "b" * 40):
            return {"parents": [{"sha": "c" * 40}]}
        if "/git/ref/heads/" in endpoint:
            assert endpoint.endswith(f"/git/ref/heads/{base_branch}")
            return {"object": {"sha": "d" * 40}}
        return original_get(endpoint)

    monkeypatch.setattr(gh, "get", get)
    monkeypatch.setattr(
        p, "_lock_at", lambda gh, repo, sha: previous if sha == "c" * 40 else reviewed
    )
    args = monitor._arguments(7, reviewed)
    plan = p._plan_identity(args, gh)
    assert plan.base_branch == plan.record["consumer_base"] == base_branch
    assert plan.main_sha == "d" * 40

    def followups(endpoint: str):
        query = parse_qs(endpoint.split("?", 1)[1])
        assert query["base"] == [base_branch]
        assert query["head"] == [f"operator:{plan.branch}"]
        return []

    monkeypatch.setattr(gh, "api", followups)
    assert p._find_followup(gh, plan) is None
    assert not gh.writes


def test_tracked_pr_retargeted_to_another_base_is_ignored(monitor: b.Monitor) -> None:
    assert monitor._discover() == [7]
    monitor.gh.pr["base"]["ref"] = "release/1.3"
    monitor.inspect(7)
    assert monitor.store.get("observation:7") == {
        "state": "ignored",
        "reason": "PR targets a different consumer base branch",
    }
    assert monitor.store.get("tracked") == []
    assert not monitor.gh.writes


@pytest.mark.parametrize(
    "base", ["dev", "feat/flux3-image-klein", "release", "release/", "main-other"]
)
def test_cli_rejects_development_base_before_initializing_state(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, base: str
) -> None:
    state = tmp_path / "not-created"
    result = b.main(
        [
            "run",
            "--workdir",
            str(state),
            "--vendor",
            "toy",
            "--upstream-repo",
            "org/library",
            "--canonical-repo",
            "operator/library",
            "--canonical-branch",
            "dev",
            "--fork",
            "operator/consumer",
            "--base-branch",
            base,
            "--once",
        ]
    )
    assert result == 1
    assert "--base-branch must be main or release/<name>" in caplog.text
    assert not state.exists()


@pytest.mark.parametrize("base", ["main", "release/1.3", "release/1.4"])
def test_monitor_accepts_only_explicit_supported_base(monitor: b.Monitor, base: str) -> None:
    monitor.gh.base_branch = base
    with contextlib.closing(b.Store(":memory:")) as store:
        instance = b.Monitor(monitor.args, monitor.gh, store)
        monitor.gh.pr["base"]["ref"] = "feat/development"
        assert instance._discover() == []
        monitor.gh.pr["base"]["ref"] = "release/9.9"
        assert instance._discover() == []
        monitor.gh.pr["base"]["ref"] = base
        assert instance._discover() == [7]
    assert not monitor.gh.writes


@pytest.mark.parametrize("base", ["main", "release/1.3"])
def test_retargeted_pr_cleanup_untracks_and_rediscovery_revalidates(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, base: str
) -> None:
    gh = monitor.gh
    gh.base_branch = gh.pr["base"]["ref"] = base
    assert monitor._discover() == [7]
    monitor._feedback(gh.pr, _issue())
    original = copy.deepcopy(gh.comments[0])
    gh.pr["base"]["ref"] = "feat/development"
    monkeypatch.setattr(monitor, "_inputs", lambda pr: pytest.fail("Must not validate dev PRs"))
    monitor.poll()
    assert monitor.store.get("tracked") == []
    assert monitor.store.get("problem:7") is None
    assert gh.comments[0] == original and len(gh.comments) == 2
    assert "now targets <code>feat/development</code>" in gh.comments[-1]["body"]
    assert "Supersedes" in gh.comments[-1]["body"]
    assert "not** a code-review approval" in gh.comments[-1]["body"]
    assert gh.reviews[0]["state"] == "DISMISSED"
    assert "#issuecomment-2" in gh.reviews[0]["body"]
    assert not any(payload.get("event") == "APPROVE" for _, _, payload in gh.writes)
    writes = copy.deepcopy(gh.writes)
    monitor.poll()
    assert gh.writes == writes
    assert monitor.store.get("health")["counts"]["processed"] == 0
    gh.pr["base"]["ref"] = base
    assert monitor._discover() == [7]
    monitor._feedback(gh.pr, _issue())
    assert len(gh.comments) == 3
    assert gh.reviews[-1]["state"] == "CHANGES_REQUESTED"


@pytest.mark.parametrize("remaining", ["comment", "review", "both", "neither"])
@pytest.mark.parametrize("legacy", [False, True])
def test_cold_start_recovers_retarget_cleanup_without_sqlite(
    monitor: b.Monitor, remaining: str, legacy: bool
) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue())
    if legacy:
        for item in gh.comments + gh.reviews:
            item["body"] = f"{monitor.marker}\nLegacy objection"
    if remaining in ("review", "neither"):
        gh.comments.clear()
    if remaining in ("comment", "neither"):
        gh.reviews.clear()
    gh.writes.clear()
    gh.pr["base"]["ref"] = "feat/development"
    gh.pr["updated_at"] = "2020-01-01T00:00:00Z"
    args = copy.copy(monitor.args)
    args.since = "2026-01-01T00:00:00Z"
    for _ in range(2):
        with contextlib.closing(b.Store(":memory:")) as store:
            b.Monitor(args, gh, store).poll()
            assert store.get("tracked") == []
            assert store.get("health")["counts"]["errors"] == 0
    if remaining == "neither":
        assert not gh.writes
    else:
        assert len(gh.comments) == (1 if remaining == "review" else 2)
        assert all(review["state"] == "DISMISSED" for review in gh.reviews)
        assert len(gh.writes) == (3 if remaining in ("review", "both") else 1)


@pytest.mark.parametrize("existing", ["manual", "other-actor", "other-vendor"])
@pytest.mark.parametrize("mixed", [False, True])
def test_retarget_cleanup_preserves_unrelated_feedback(
    monitor: b.Monitor, existing: str, mixed: bool
) -> None:
    gh = monitor.gh
    actor = "human" if existing == "other-actor" else monitor.actor
    body = (
        "Manual objection"
        if existing == "manual"
        else "<!-- vendor-bot:other -->"
        if existing == "other-vendor"
        else monitor.marker
    )
    gh.comments.append({"id": 1, "user": {"login": actor}, "body": body})
    gh.reviews.append(
        {"id": 1, "user": {"login": actor}, "body": body, "state": "CHANGES_REQUESTED"}
    )
    before = copy.deepcopy((gh.comments, gh.reviews))
    if mixed:
        monitor._feedback(gh.pr, _issue())
    gh.pr["base"]["ref"] = "feat/development"
    monitor.poll()
    assert monitor.store.get("tracked") == []
    if mixed:
        assert len(gh.comments) == 3
        assert gh.comments[:1] == before[0] and gh.reviews[:1] == before[1]
        if existing == "other-actor":
            assert len(gh.reviews) == 2
            assert "not applicable on this target branch" in gh.reviews[-1]["body"]
            assert gh.reviews[-1]["state"] == "DISMISSED"
        else:
            assert len(gh.reviews) == 1
    else:
        assert not gh.writes and (gh.comments, gh.reviews) == before


@pytest.mark.parametrize("stage", ["post-response", "dismissal-response", "review-update"])
def test_retarget_cleanup_recovers_partial_writes_without_sqlite(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue())
    gh.pr["base"]["ref"] = "feat/development"
    original = gh.api

    def fail(endpoint, **kwargs):
        if stage == "review-update" and endpoint.endswith("/reviews/1"):
            raise RuntimeError("Injected review failure")
        result = original(endpoint, **kwargs)
        if (
            stage == "post-response"
            and endpoint.endswith("/comments")
            and kwargs.get("method") == "POST"
        ) or (stage == "dismissal-response" and endpoint.endswith("/dismissals")):
            raise RuntimeError("Injected lost response")
        return result

    monkeypatch.setattr(gh, "api", fail)
    monitor.poll()
    assert monitor.store.get("tracked") == [7]
    assert monitor.store.get("health")["counts"]["errors"] == 1
    monkeypatch.setattr(gh, "api", original)
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, gh, store)
        restarted.poll()
        assert store.get("tracked") == []
        writes = copy.deepcopy(gh.writes)
        restarted.poll()
        assert gh.writes == writes
    assert len(gh.comments) == 2
    assert gh.reviews[0]["state"] == "DISMISSED"
    assert "#issuecomment-2" in gh.reviews[0]["body"]


def test_retarget_cleanup_retries_denied_dismissal_without_repeating_comment(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue())
    gh.pr["base"]["ref"] = "feat/development"
    original = gh.api

    def denied(endpoint, **kwargs):
        if endpoint.endswith("/dismissals"):
            raise RuntimeError("HTTP 403")
        return original(endpoint, **kwargs)

    monkeypatch.setattr(gh, "api", denied)
    monitor.poll()
    monitor.poll()
    assert monitor.store.get("tracked") == [7]
    assert len(gh.comments) == 2
    assert "authorized reviewer" in gh.comments[-1]["body"]
    assert gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    monkeypatch.setattr(gh, "api", original)
    monitor.poll()
    assert monitor.store.get("tracked") == []
    assert gh.reviews[0]["state"] == "DISMISSED"


@pytest.mark.parametrize("change", ["head", "body", "base", "closed", "dry-run"])
def test_retarget_cleanup_defers_concurrent_changes_and_dry_run(
    monitor: b.Monitor, change: str
) -> None:
    gh = monitor.gh
    monitor._discover()
    monitor._feedback(gh.pr, _issue())
    gh.pr["base"]["ref"] = "feat/development"
    snapshot = copy.deepcopy(gh.pr)
    if change == "head":
        gh.pr["head"]["sha"] = "b" * 40
    elif change == "body":
        gh.pr["body"] = "edited"
    elif change == "base":
        gh.pr["base"]["ref"] = "main"
    elif change == "closed":
        gh.pr["state"] = "closed"
    else:
        monitor.args.publish = False
    writes = copy.deepcopy(gh.writes)
    monitor._retargeted(snapshot)
    assert gh.writes == writes
    assert monitor.store.get("tracked") == [7]
    assert gh.reviews[0]["state"] == "CHANGES_REQUESTED"


def test_retarget_between_monitors_hands_off_without_dismissing_current_review(
    monitor: b.Monitor,
) -> None:
    gh = monitor.gh
    monitor._discover()
    monitor._feedback(gh.pr, _issue())
    gh.pr["base"]["ref"] = "release/1.3"
    writes = copy.deepcopy(gh.writes)
    monitor.inspect(7)
    assert not monitor.store.get("tracked") and gh.writes == writes
    with contextlib.closing(b.Store(":memory:")) as store:
        release_gh = copy.copy(gh)
        release_gh.base_branch = "release/1.3"
        release = b.Monitor(monitor.args, release_gh, store)
        assert release._discover() == [7]
        release._feedback(gh.pr, _issue())
        gh.pr["base"]["ref"] = "feat/development"
        assert monitor._discover() == []
        release.poll()
        assert store.get("tracked") == []
        assert len(gh.comments) == 3
        assert gh.reviews[0]["state"] == "DISMISSED"
        writes = copy.deepcopy(gh.writes)
        monitor.poll()
        assert gh.writes == writes


def test_feedback_rechecks_consumer_base_before_writing(monitor: b.Monitor) -> None:
    inspected = copy.deepcopy(monitor.gh.pr)
    monitor.gh.pr["base"]["ref"] = "release/1.3"
    monitor._feedback(inspected, _issue())
    assert monitor.store.get("observation:7")["state"] == "ignored"
    assert not monitor.gh.writes


def test_cold_discovery_backfills_old_merged_prs_and_honors_cutoff(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gh = monitor.gh
    gh.pr.update(
        state="closed",
        merged=True,
        merged_at="2020-01-01T00:00:00Z",
        updated_at="2020-01-01T00:00:00Z",
    )
    api = gh.api

    def filtered(endpoint: str, **kwargs):
        if "/pulls?state=open" in endpoint:
            return []
        return api(endpoint, **kwargs)

    monkeypatch.setattr(gh, "api", filtered)
    assert monitor._discover() == [7]
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, gh, store)
        assert restarted._discover() == [7]
    restricted = copy.copy(monitor.args)
    restricted.since = "2021-01-01T00:00:00Z"
    with contextlib.closing(b.Store(":memory:")) as store:
        assert b.Monitor(restricted, gh, store)._discover() == []
    assert not gh.writes


@pytest.mark.parametrize("changed_lines", [1, 3], ids=["accepted", "below-threshold"])
def test_monitor_posts_similarity_and_deduplicates_feedback(
    source: Path, monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, changed_lines: int
) -> None:
    base = _commit(source, "", "src/code.py")
    lines = [f"VALUE_{index} = {index}\n" for index in range(16)]
    downstream = _commit(source, "".join(lines), "src/code.py")
    _git(source, "checkout", "-q", "-b", "upstream", base)
    upstream = _commit(
        source, "".join(lines[:-changed_lines]) + "adapted\n" * changed_lines, "src/code.py"
    )
    previous = _vendor(base, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor(downstream, "https://github.com/author/library.git", "temporary")
    monitor.gh.canonical_tip = previous.commit
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    monitor.gh.pr["body"] = _body("    upstream_prs: [https://github.com/org/library/pull/42]")
    monkeypatch.setattr(
        m,
        "resolve",
        lambda *args: m.match(
            source,
            base,
            downstream,
            {42: _pr(base, upstream)},
            source=reviewed.source,
            include=reviewed.include,
            upstream_repo="org/library",
        ),
    )
    monitor.inspect(7)
    body = monitor.gh.comments[0]["body"]
    assert "https://github.com/org/library/pull/42" in body
    assert downstream in body and "90.00%" in body
    if changed_lines == 1:
        assert "95.00%" in body and "accepted" in body
        assert "not** a code-review approval" in body
        assert not monitor.gh.reviews
    else:
        assert "85.00%" in body and "no PR meets" in body
        assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    writes = copy.deepcopy(monitor.gh.writes)
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, monitor.gh, store)
        monkeypatch.setattr(restarted, "_inputs", lambda pr: (previous, reviewed))
        restarted.inspect(7)
    assert monitor.gh.writes == writes


def test_feedback_review_links_to_current_result(monitor: b.Monitor) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue("Missing block"))
    comment, review = gh.comments[0], gh.reviews[0]
    link = f"https://github.com/org/consumer/pull/7#issuecomment-{comment['id']}"
    assert f"[Latest vendor validation and step-by-step fix]({link})" in review["body"]
    assert "Vendor validation:" in comment["body"]
    assert f"/pull/7/commits/{gh.pr['head']['sha']}" in comment["body"]
    original = comment["body"]
    gh.pr["head"]["sha"] = "d" * 40
    monitor._feedback(gh.pr, _issue("Pin needs rebase"))
    assert len(gh.comments) == 2 and len(gh.reviews) == 1
    assert "#issuecomment-2" in review["body"]
    assert "Pin needs rebase" in gh.comments[-1]["body"]
    assert "d" * 40 in gh.comments[-1]["body"]
    assert link in gh.comments[-1]["body"] and "Supersedes" in gh.comments[-1]["body"]
    assert comment["body"] == original


def test_feedback_repairs_review_link_when_comment_is_recreated(monitor: b.Monitor) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue("Missing block"))
    # A human comment keeps the fake API's next ID distinct from the deleted ID.
    gh.comments[0] = {"id": 80, "body": "Human comment", "user": {"login": "author"}}
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, gh, store)
        restarted._feedback(gh.pr, _issue("Missing block"))
    assert len(gh.comments) == 2 and len(gh.reviews) == 1
    assert f"#issuecomment-{gh.comments[1]['id']}" in gh.reviews[0]["body"]
    assert "#issuecomment-1)" not in gh.reviews[0]["body"]


def test_feedback_deduplicates_with_empty_sqlite(monitor: b.Monitor) -> None:
    monitor._feedback(monitor.gh.pr, _issue("Missing block"))
    writes = copy.deepcopy(monitor.gh.writes)
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, monitor.gh, store)
        restarted._feedback(monitor.gh.pr, _issue("Missing block"))
        assert monitor.gh.writes == writes
        restarted._feedback(monitor.gh.pr, None)
    assert len(monitor.gh.comments) == 2 and len(monitor.gh.reviews) == 1
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"


def test_feedback_appends_changed_fixes_and_repeated_result_transitions(monitor: b.Monitor) -> None:
    gh = monitor.gh
    for index, action in enumerate(("Fix A", "Fix B", "Fix A"), start=1):
        issue = m.Diagnostic("Same problem", action)
        previous = copy.deepcopy(gh.comments)
        with contextlib.closing(b.Store(":memory:")) as store:
            restarted = b.Monitor(monitor.args, gh, store)
            restarted._feedback(gh.pr, issue)
            writes = copy.deepcopy(gh.writes)
            restarted._feedback(gh.pr, issue)
            assert gh.writes == writes
        assert len(gh.comments) == index and gh.comments[:-1] == previous
        assert f"**Fix:** {action}" in gh.comments[-1]["body"]
        assert "@author" in gh.comments[-1]["body"]
        assert f"#issuecomment-{index}" in gh.reviews[0]["body"]
        if index > 1:
            assert f"#issuecomment-{index - 1}" in gh.comments[-1]["body"]
    assert not any(method == "PATCH" for _, method, _ in gh.writes)


@pytest.mark.parametrize("change", ["commit", "branch", "url"])
def test_feedback_notifies_source_changes_with_unchanged_result(
    monitor: b.Monitor, change: str
) -> None:
    gh = monitor.gh
    source = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monitor._feedback(gh.pr, None, source=source)
    updated = dataclasses.replace(
        source,
        **{
            change: {
                "commit": "c" * 40,
                "branch": "temporary2",
                "url": "https://github.com/other/library.git",
            }[change]
        },
    )
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, gh, store)
        restarted._feedback(gh.pr, None, source=updated)
        writes = copy.deepcopy(gh.writes)
        restarted._feedback(gh.pr, None, source=updated)
        assert gh.writes == writes
    assert len(gh.comments) == 2
    assert updated.commit in gh.comments[-1]["body"]
    assert "@author" in gh.comments[-1]["body"]


def test_feedback_unrelated_pr_updates_do_not_notify(monitor: b.Monitor) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue())
    before = copy.deepcopy(gh.comments)
    writes = copy.deepcopy(gh.writes)
    gh.pr["head"]["sha"] = "c" * 40
    gh.pr["body"] = "Updated prose unrelated to the validation result"
    monitor._feedback(gh.pr, _issue())
    assert gh.comments == before and gh.writes == writes


def test_upstream_valid_feedback_suppresses_observed_tip_only_changes(monitor: b.Monitor) -> None:
    gh = monitor.gh
    proof = {
        "promoted": "b" * 40,
        "checked_canonical_tip": "b" * 40,
        "upstream_observed_main": "b" * 40,
    }
    monitor._feedback(gh.pr, None, upstream_proof=proof)
    writes = copy.deepcopy(gh.writes)
    proof.update(checked_canonical_tip="a" * 40, upstream_observed_main="c" * 40)
    with contextlib.closing(b.Store(":memory:")) as store:
        b.Monitor(monitor.args, gh, store)._feedback(gh.pr, None, upstream_proof=proof)
    assert gh.writes == writes
    proof["promoted"] = "c" * 40
    monitor._feedback(gh.pr, None, upstream_proof=proof)
    assert len(gh.comments) == 2


def test_upstream_invalid_feedback_suppresses_tip_churn_but_not_check_changes(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    gh = monitor.gh
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/org/library.git", "pull/42/head")
    upstream_tip, reachable = "b" * 40, True
    original = gh.api

    def api(endpoint, **kwargs):
        if endpoint == "repos/org/library/git/ref/heads/main":
            return {"object": {"sha": upstream_tip}}
        return original(endpoint, **kwargs)

    def compare(repo, base, head):
        result = (
            previous.commit
            if base == reviewed.commit and head == upstream_tip and not reachable
            else base
        )
        return {"merge_base_commit": {"sha": result}}

    monkeypatch.setattr(gh, "api", api)
    monkeypatch.setattr(gh, "compare", compare)

    def check() -> None:
        with pytest.raises(p.Diagnostic) as failure:
            p._check_source(gh, previous, reviewed)
        monitor._feedback(gh.pr, failure.value, source=reviewed)

    check()
    assert "Observed dev/upstream tip SHAs may be stale" in gh.comments[-1]["body"]
    assert "Fetch the live branch" in gh.comments[-1]["body"]
    writes = copy.deepcopy(gh.writes)
    gh.canonical_tip, upstream_tip = "c" * 40, "d" * 40
    check()
    assert gh.writes == writes
    reachable = False
    check()
    assert len(gh.comments) == 2
    assert "not reachable" in gh.comments[-1]["body"]


def test_temporary_fork_same_ancestry_failure_does_not_repeat_for_new_dev_tip(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    gh = monitor.gh
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monkeypatch.setattr(
        gh, "compare", lambda *args: {"merge_base_commit": {"sha": previous.commit}}
    )
    for tip in ("c" * 40, "d" * 40):
        gh.canonical_tip = tip
        with pytest.raises(p.Diagnostic) as failure:
            p._check_canonical_tip(gh, previous, reviewed)
        monitor._feedback(gh.pr, failure.value, source=reviewed)
    assert len(gh.comments) == 1 and len(gh.writes) == 2
    assert "Observed dev/upstream tip SHAs may be stale" in gh.comments[0]["body"]
    assert "Fetch the live branch" in gh.comments[0]["body"]


@pytest.mark.parametrize(
    "stage",
    ["post-response", "review-update", "dismissal-response", "dismissal-response-no-comment"],
)
def test_feedback_recovers_after_partial_publication_without_sqlite(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue("Initial problem"))
    problem = None if stage.startswith("dismissal-response") else _issue("Changed problem")
    no_promotion = stage == "dismissal-response-no-comment"
    if no_promotion:
        gh.comments.clear()
    original = gh.api

    def fail(endpoint, **kwargs):
        if stage == "review-update" and endpoint.endswith("/reviews/1"):
            raise RuntimeError("Injected review failure")
        result = original(endpoint, **kwargs)
        if (
            stage == "post-response"
            and endpoint.endswith("/comments")
            and kwargs.get("method") == "POST"
        ) or (stage.startswith("dismissal-response") and endpoint.endswith("/dismissals")):
            raise RuntimeError("Injected lost response")
        return result

    monkeypatch.setattr(gh, "api", fail)
    with pytest.raises(RuntimeError, match="Injected"):
        monitor._feedback(gh.pr, problem, no_promotion=no_promotion)
    monkeypatch.setattr(gh, "api", original)
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, gh, store)
        restarted._feedback(gh.pr, problem, no_promotion=no_promotion)
        writes = copy.deepcopy(gh.writes)
        restarted._feedback(gh.pr, problem, no_promotion=no_promotion)
        assert gh.writes == writes
    assert len(gh.comments) == (1 if no_promotion else 2) and len(gh.reviews) == 1
    assert f"#issuecomment-{gh.comments[-1]['id']}" in gh.reviews[0]["body"]
    assert gh.reviews[0]["state"] == ("DISMISSED" if problem is None else "CHANGES_REQUESTED")


def test_feedback_uses_newest_owned_comment_not_listing_or_edit_order(monitor: b.Monitor) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue("First"))
    monitor._feedback(gh.pr, _issue("Second"))
    gh.comments[0]["updated_at"] = "2099-01-01T00:00:00Z"
    gh.comments.reverse()
    gh.comments.append({"id": 99, "body": gh.comments[0]["body"], "user": {"login": "author"}})
    writes = copy.deepcopy(gh.writes)
    with contextlib.closing(b.Store(":memory:")) as store:
        b.Monitor(monitor.args, gh, store)._feedback(gh.pr, _issue("Second"))
    assert gh.writes == writes
    assert "#issuecomment-2" in gh.reviews[0]["body"]


def test_feedback_supersedes_legacy_comment_once(monitor: b.Monitor) -> None:
    gh = monitor.gh
    legacy = {
        "id": 1,
        "body": f"{monitor.marker}\nOld diagnostic",
        "user": {"login": monitor.actor},
    }
    gh.comments.append(copy.deepcopy(legacy))
    monitor._feedback(gh.pr, _issue())
    assert gh.comments[0] == legacy and len(gh.comments) == 2
    assert "Supersedes" in gh.comments[-1]["body"] and "#issuecomment-1" in gh.comments[-1]["body"]
    writes = copy.deepcopy(gh.writes)
    with contextlib.closing(b.Store(":memory:")) as store:
        b.Monitor(monitor.args, gh, store)._feedback(gh.pr, _issue())
    assert gh.writes == writes


def test_feedback_finds_latest_result_on_later_comment_page(
    monitor: b.Monitor, monkeypatch: pytest.MonkeyPatch
) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, _issue("First"))
    monitor._feedback(gh.pr, _issue("Second"))
    first, latest = copy.deepcopy(gh.comments)
    latest["id"] = 101
    gh.comments = (
        [first]
        + [
            {"id": number, "body": "Unrelated discussion", "user": {"login": "author"}}
            for number in range(2, 101)
        ]
        + [latest]
    )
    gh.comments.extend(
        [
            latest | {"id": 102, "user": {"login": "author"}},
            latest
            | {
                "id": 103,
                "body": latest["body"].replace(monitor.marker, "<!-- vendor-bot:other -->"),
            },
        ]
    )
    original, pages = gh.api, []

    def paged(endpoint, **kwargs):
        if "/issues/7/comments?" in endpoint:
            page = int(parse_qs(endpoint.split("?", 1)[1])["page"][0])
            pages.append(page)
            return copy.deepcopy(gh.comments[(page - 1) * 100 : page * 100])
        return original(endpoint, **kwargs)

    monkeypatch.setattr(gh, "api", paged)
    before = copy.deepcopy(gh.comments)
    with contextlib.closing(b.Store(":memory:")) as store:
        b.Monitor(monitor.args, gh, store)._feedback(gh.pr, _issue("Second"))
    assert pages == [1, 2]
    assert gh.comments == before
    assert "#issuecomment-101" in gh.reviews[0]["body"]


def _vendor(commit: str, url: str, branch: str):
    return m.manage._validate_vendor(
        "toy",
        {
            "url": url,
            "branch": branch,
            "commit": commit,
            "source": "src",
            "destination": "vendored/toy",
            "include": ["**/*.py"],
            "digest": "sha256-tree-v1:" + "0" * 64,
        },
    )


def test_monitor_revalidates_matching_until_committed_and_publishes_once_after_merge(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gh = monitor.gh
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    gh.pr["body"] = _body("    upstream_prs: [https://github.com/org/library/pull/42]")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    matches, publications = [], []

    def resolve(*args):
        matches.append(True)
        return {reviewed.commit: 42}, {reviewed.commit: {"method": "same-edits"}}

    def make_plan(args, github, *, auto_match, resolutions):
        assert args.auto_merge and not args.wait
        assert args.map_upstream == []
        assert auto_match and resolutions == {}
        return p.Promotion(
            7,
            "c" * 40,
            previous,
            reviewed,
            "operator/library",
            "operator/consumer",
            {"attribution": {reviewed.commit: {"method": "same-edits"}}},
            consumer_repo="org/consumer",
        )

    monkeypatch.setattr(p, "_plan_identity", lambda args, gh: argparse.Namespace(main_sha="c" * 40))
    monkeypatch.setattr(p, "_recover_plan", lambda *args: False)
    monkeypatch.setattr(p, "_lock_at", lambda *args: reviewed)
    monkeypatch.setattr(p, "_check_current", lambda *args: None)

    def publish(args, github, plan):
        publications.append(plan)
        assert plan.record["attribution"][reviewed.commit]["method"] == "same-edits"
        return {"number": 8}

    original = gh.get
    monkeypatch.setattr(
        gh,
        "get",
        lambda endpoint: {
            "number": 8,
            "merged": True,
            "merge_commit_sha": "d" * 40,
            "html_url": "https://github.com/org/consumer/pull/8",
        }
        if endpoint.endswith("/pulls/8")
        else original(endpoint),
    )
    monkeypatch.setattr(m, "resolve", resolve)
    monkeypatch.setattr(p, "_make_plan", make_plan)
    monkeypatch.setattr(p, "_publish", publish)
    monkeypatch.setattr(p, "_verify_remote_commit", lambda *args: "verified")
    monitor.inspect(7)
    monitor.inspect(7)
    assert len(matches) == 2
    assert not publications
    gh.pr.update(merged=True, state="closed")
    monitor.inspect(7)
    monitor.inspect(7)
    assert len(publications) == 1
    assert monitor.store.get("complete:7").endswith("/8")


def test_monitor_reports_attribution_to_author_and_retries_after_correction(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    monitor.gh.pr["body"] = _body("    upstream_prs: [https://github.com/org/library/pull/42]")

    def unresolved(*args):
        raise _issue("Author action required: no exact match for " + reviewed.commit)

    monkeypatch.setattr(m, "resolve", unresolved)
    monitor.inspect(7)
    assert "@author" in monitor.gh.comments[0]["body"]
    assert reviewed.commit in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    monkeypatch.setattr(m, "resolve", lambda *args: ({reviewed.commit: 42}, {}))
    monitor.inspect(7)
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"


@pytest.mark.parametrize("no_promotion", [False, True])
@pytest.mark.parametrize("comment_deleted", [False, True])
def test_forbidden_dismissal_informs_author_without_approving(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    no_promotion: bool,
    comment_deleted: bool,
) -> None:
    monitor._feedback(monitor.gh.pr, _issue("Missing block"))
    if comment_deleted:
        monitor.gh.comments.clear()
    original = monitor.gh.api

    def forbidden(endpoint, **kwargs):
        if endpoint.endswith("/dismissals"):
            raise RuntimeError("HTTP 403")
        return original(endpoint, **kwargs)

    monkeypatch.setattr(monitor.gh, "api", forbidden)
    monitor._feedback(monitor.gh.pr, None, no_promotion=no_promotion)
    assert "lacks dismissal permission" in monitor.gh.comments[-1]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    writes = len(monitor.gh.writes)
    monitor._feedback(monitor.gh.pr, None, no_promotion=no_promotion)
    assert len(monitor.gh.writes) == writes
    monkeypatch.setattr(monitor.gh, "api", original)
    monitor._feedback(monitor.gh.pr, None, no_promotion=no_promotion)
    assert "lacks dismissal permission" not in monitor.gh.comments[-1]["body"]
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"


def test_old_promotion_messages_remain_readable() -> None:
    record = {
        "version": 1,
        "vendor": "flashinfer-prims-ts",
        "trtllm_pr": "https://github.com/NVIDIA/TensorRT-LLM/pull/17",
        "trtllm_merge": "a" * 40,
    }
    message = (
        "----- BEGIN PRIMTS PROMOTION V1 -----\n"
        + json.dumps(record)
        + "\n----- END PRIMTS PROMOTION V1 -----"
    )
    normalized = p._record_from_message(message)
    assert normalized["consumer_pr"].endswith("/17")
    assert normalized["upstream_repository"] == "flashinfer-ai/flashinfer"


def test_workdir_locks_and_refuses_unrelated_data(tmp_path: Path) -> None:
    workdir = tmp_path / "state"
    b._initialize_workdir(workdir)
    with b._locked(workdir):
        assert b._running(workdir)
        with pytest.raises(ValueError, match="already owns"):
            with b._locked(workdir):
                pass
    assert not b._running(workdir)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "data").write_text("preserve")
    with pytest.raises(ValueError, match="empty workdir"):
        b._initialize_workdir(unrelated)
    assert (unrelated / "data").read_text() == "preserve"


def test_workdir_refuses_checkout(source: Path) -> None:
    with pytest.raises(ValueError, match="outside a Git worktree"):
        b._initialize_workdir(source / "bot-state")


def test_workdir_recovers_without_sqlite_but_preserves_artifacts(tmp_path: Path) -> None:
    workdir = tmp_path / "state"
    b._initialize_workdir(workdir)
    (workdir / "bot.log").write_text("previous log\n")
    (workdir / "bot.lock").touch()
    (workdir / "sources.git").mkdir()
    (workdir / ("promotion-7-" + "a" * 12)).mkdir()
    b._initialize_workdir(workdir)
    assert (workdir / "bot.log").read_text() == "previous log\n"
    with b._locked(workdir):
        with pytest.raises(ValueError, match="already owns"):
            with b._locked(workdir):
                pass
    with contextlib.closing(b.Store(workdir / b._STATE_FILE)) as store:
        assert store.get("tracked") is None


@pytest.mark.parametrize("name", ["state.sqlite", "bot.log", "bot.lock", "sources.git"])
def test_workdir_recovery_rejects_symlinks(tmp_path: Path, name: str) -> None:
    workdir = tmp_path / "state"
    workdir.mkdir()
    target = tmp_path / "private-data"
    target.write_text("preserve")
    (workdir / name).symlink_to(target)
    with pytest.raises(ValueError, match="unexpected entry"):
        b._initialize_workdir(workdir)
    assert target.read_text() == "preserve"


def test_daemon_launch_status_stop_without_systemd(tmp_path: Path) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport json, sys\n"
        "print(json.dumps({'login': 'operator'} if sys.argv[-1] == 'user' else []))\n"
    )
    gh.chmod(0o700)
    workdir = tmp_path / "state"
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "GH_CONFIG_DIR": str(tmp_path / "private-auth"),
    }
    command = [sys.executable, str(_ROOT / "scripts/vendor/bot.py")]
    arguments = [
        "run",
        "--workdir",
        str(workdir),
        "--vendor",
        "toy",
        "--upstream-repo",
        "org/library",
        "--canonical-repo",
        "operator/library",
        "--canonical-branch",
        "dev",
        "--fork",
        "operator/consumer",
        "--daemon",
    ]
    try:
        result = subprocess.run(
            command + arguments, env=env, capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 0, result.stderr
        assert "Started daemon PID" in result.stdout
        status = subprocess.run(
            command + ["status", "--workdir", str(workdir)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        state = json.loads(status.stdout)
        assert state["running"]
        assert state["pid"] is not None
        assert state["publish"] is False
        assert state["vendor"] == "toy"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = b._status(workdir)
            if state.get("next_poll_at"):
                break
            time.sleep(0.05)
        assert state["last_poll_summary"]["outcome"] == "ok"
        assert state["last_successful_poll"] == state["last_completed_poll"]
        assert state["operation"]["phase"] == "waiting"
        assert state["next_poll_at"]
        assert "Poll finished" in (workdir / "bot.log").read_text()
        assert "private-auth" not in (workdir / "bot.log").read_text()
    finally:
        subprocess.run(
            command + ["stop", "--workdir", str(workdir)], env=env, capture_output=True, timeout=5
        )
        deadline = time.monotonic() + 10
        while b._running(workdir) and time.monotonic() < deadline:
            time.sleep(0.1)
    assert not b._running(workdir)
    assert b._status(workdir)["pid"] is None
    assert b._status(workdir)["next_poll_at"] is None


def test_log_level_defaults_to_info_and_accepts_lowercase(tmp_path: Path) -> None:
    command = [
        "run",
        "--workdir",
        str(tmp_path),
        "--vendor",
        "toy",
        "--upstream-repo",
        "org/library",
        "--canonical-repo",
        "operator/library",
        "--canonical-branch",
        "dev",
        "--fork",
        "operator/consumer",
    ]
    assert b._parser().parse_args(command).log_level == "INFO"
    assert b._parser().parse_args(command + ["--log-level", "debug"]).log_level == "DEBUG"


@pytest.mark.parametrize("publish", [False, True])
def test_metadata_transitions_are_logged_once_and_clear_problem(
    monitor: b.Monitor,
    caplog: pytest.LogCaptureFixture,
    publish: bool,
) -> None:
    caplog.set_level(logging.INFO, logger="vendor-bot")
    monitor.args.publish = publish
    monitor._feedback(monitor.gh.pr, _issue())
    monitor._feedback(monitor.gh.pr, _issue())
    assert caplog.text.count("PR #7: blocked: Missing marked block") == 1
    assert monitor.store.get("problem:7") == "Missing marked block"
    monitor._feedback(monitor.gh.pr, None)
    monitor._feedback(monitor.gh.pr, None)
    assert caplog.text.count("PR #7: metadata_valid") == 1
    assert monitor.store.get("problem:7") is None
    if publish:
        assert "posted metadata feedback comment" in caplog.text
        assert "requested metadata changes" in caplog.text
        assert "dismissed resolved metadata review" in caplog.text
    else:
        assert not monitor.gh.writes


def test_idle_poll_has_summary_and_success_timestamp(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="vendor-bot")
    monkeypatch.setattr(monitor, "_discover", lambda: [])
    monitor.poll()
    health = monitor.store.get("health")
    assert health["last_successful_poll"] == health["last_completed_poll"]
    assert health["last_completed_poll"] == monitor.store.get("last_poll")
    assert health["last_poll_summary"]["outcome"] == "ok"
    assert health["last_poll_summary"]["processed"] == 0
    assert health["last_poll_summary"]["errors"] == 0
    assert "Poll started" in caplog.text and "Poll finished" in caplog.text


def test_pr_failure_is_not_a_successful_poll_and_recovers(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor, "_discover", lambda: [7])

    def fail(number: int) -> None:
        raise RuntimeError("HTTP 429: rate limit")

    monkeypatch.setattr(monitor, "inspect", fail)
    monitor.poll()
    failed = monitor.store.get("health")
    assert failed["last_completed_poll"]
    assert failed["last_successful_poll"] is None
    assert failed["last_poll_summary"]["outcome"] == "needs_attention"
    assert failed["last_poll_summary"]["errors"] == 1
    assert failed["last_poll_summary"]["processed"] == 1
    assert failed["last_error"]["pr"] == 7
    assert "HTTP 429" in failed["last_error"]["message"]
    monkeypatch.setattr(
        monitor, "inspect", lambda number: monitor._transition(number, "metadata_valid")
    )
    monitor.poll()
    recovered = monitor.store.get("health")
    assert recovered["last_successful_poll"] == recovered["last_completed_poll"]
    assert recovered["last_poll_summary"]["outcome"] == "ok"
    assert recovered["last_error"] == failed["last_error"]  # Historical, not an active error.
    assert monitor.store.get("problem:7") is None


def test_author_blocked_poll_is_not_reported_healthy(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _vendor("a" * 40, "https://github.com/operator/library.git", "dev")
    reviewed = _vendor("b" * 40, "https://github.com/author/library.git", "temporary")
    monkeypatch.setattr(monitor, "_inputs", lambda pr: (previous, reviewed))
    monitor.poll()
    health = monitor.store.get("health")
    assert health["last_poll_summary"]["outcome"] == "needs_attention"
    assert health["counts"]["blocked"] == 1
    assert health["counts"]["errors"] == 0
    assert health["counts"]["relevant"] == 1
    assert health["last_successful_poll"] is None


@pytest.mark.parametrize("error_type", [RuntimeError, KeyError])
def test_discovery_failure_never_advances_completion(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    monitor.progress.update(last_completed_poll="old", last_successful_poll="old")

    def fail() -> list[int]:
        raise error_type("discovery failed")

    monkeypatch.setattr(monitor, "_discover", fail)
    with pytest.raises(error_type):
        monitor.poll()
    health = monitor.store.get("health")
    assert health["last_completed_poll"] == health["last_successful_poll"] == "old"
    assert health["last_poll_summary"]["outcome"] == "failed"


def test_stopping_discovery_does_not_advance_cursor_or_success(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = monitor.gh.api

    def stop(endpoint: str, **kwargs):
        if "/pulls?" in endpoint:
            b._STOP.set()
        return original(endpoint, **kwargs)

    monkeypatch.setattr(monitor.gh, "api", stop)
    monitor.poll()
    assert monitor.store.get("cursor") is None
    assert monitor.store.get("bootstrapped") is None
    health = monitor.store.get("health")
    assert health["last_completed_poll"] is None
    assert health["last_successful_poll"] is None
    assert health["last_poll_summary"]["outcome"] == "interrupted"


def test_heartbeat_during_blocked_discovery_does_not_fake_progress(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="vendor-bot")
    monkeypatch.setattr(b, "_HEARTBEAT_SECONDS", 0.01)
    discovery_blocked = threading.Event()
    reported = threading.Event()
    original = monitor.progress.heartbeat

    def heartbeat() -> None:
        if not discovery_blocked.is_set():
            return
        original()
        reported.set()

    def wait_for_heartbeat(endpoint: str, **kwargs):
        before = monitor.store.get("health")
        discovery_blocked.set()
        assert reported.wait(2), "No heartbeat during blocked API call"
        assert monitor.store.get("health") == before
        return []

    monkeypatch.setattr(monitor.progress, "heartbeat", heartbeat)
    monkeypatch.setattr(monitor.gh, "api", wait_for_heartbeat)
    with monitor.progress.heartbeats():
        monitor.poll()
    assert "Heartbeat: phase=discover_open" in caplog.text
    assert "operation completion not yet confirmed" in caplog.text
    assert "Discovery completed" in caplog.text
    assert not any(t.name == "vendor-bot-heartbeat" for t in threading.enumerate())


def test_progress_checkpoint_and_restart_keep_history(monitor: b.Monitor) -> None:
    progress = monitor.progress
    progress.operation("discover_history")
    progress.last_checkpoint = float("-inf")
    progress.count("scanned", 123)
    assert monitor.store.get("health")["counts"]["scanned"] == 123
    progress.error("previous error", 7)
    progress.update(last_successful_poll="old", next_poll_at="stale")
    restarted = b.Progress(monitor.store)
    assert restarted.state["last_successful_poll"] == "old"
    assert restarted.state["last_error"]["pr"] == 7
    assert restarted.state["next_poll_at"] is None
    assert restarted.state["operation"] is None


def test_completed_promotion_clears_stale_problem_without_remote_writes(monitor: b.Monitor) -> None:
    monitor.store.put("problem:7", "interrupted after completion checkpoint")
    monitor.store.put("complete:7", "https://github.com/org/consumer/pull/8")
    monitor.inspect(7)
    assert monitor.store.get("problem:7") is None
    assert monitor.store.get("observation:7")["state"] == "completed"
    assert not monitor.gh.writes


@pytest.mark.parametrize("status", ["added", "removed", "modified"])
def test_lock_creation_removal_is_not_a_promotion_but_other_404s_fail(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    original = monitor.gh.api

    def files(endpoint: str, **kwargs):
        if "/files?" in endpoint:
            return [{"filename": p._LOCK, "status": status}]
        return original(endpoint, **kwargs)

    def missing(*args):
        raise RuntimeError("gh failed: gh: Not Found (HTTP 404)")

    monkeypatch.setattr(monitor.gh, "api", files)
    monkeypatch.setattr(p, "_main_sha", lambda gh: "a" * 40)
    monkeypatch.setattr(p, "_lock_document_at", missing)
    if status == "modified":
        with pytest.raises(RuntimeError, match="HTTP 404"):
            monitor._inputs(monitor.gh.pr)
    else:
        monitor.inspect(7)
        assert monitor.store.get("observation:7")["state"] == "ignored"
        assert monitor.store.get("problem:7") is None
    assert not monitor.gh.writes


def test_status_is_read_only_and_hides_stale_process_data(tmp_path: Path) -> None:
    workdir = tmp_path / "missing"
    assert b._status(workdir)["running"] is False
    assert not workdir.exists()
    workdir.mkdir()
    with contextlib.closing(b.Store(workdir / b._STATE_FILE)) as store:
        store.put("process", {"pid": 123, "publish": True})
        store.put("settings", {"vendor": "toy", "actor": "operator"})
        progress = b.Progress(store)
        progress.operation("publishing_promotion", 7)
        progress.update(next_poll_at="future", last_completed_poll="previous")
        progress.update(running=False, unexpected="do not expose unrelated cached fields")
    before = (workdir / b._STATE_FILE).read_bytes()
    with b._locked(workdir):
        state = b._status(workdir)
        assert state["running"] and state["pid"] == 123
        assert state["publish"] and state["vendor"] == "toy"
        assert state["operation"]["pr"] == 7
        assert "unexpected" not in state
        (workdir / "stop").touch()
        assert b._status(workdir)["stop_requested"]
    state = b._status(workdir)
    assert state["running"] is False and state["pid"] is None
    assert state["operation"] is None and state["next_poll_at"] is None
    assert state["last_completed_poll"] == "previous"
    assert (workdir / b._STATE_FILE).read_bytes() == before


def test_status_exposes_consumer_base_and_refresh_policy(monitor: b.Monitor) -> None:
    monitor.args.refresh_branch = ["dev-new"]
    b.Monitor(monitor.args, monitor.gh, monitor.store)
    state = b._status(monitor.args.workdir)
    assert state["base_branch"] == "main"
    assert state["canonical_branch"] == "dev"
    assert state["refresh_branches"] == ["dev-new"]
    assert not state["running"]


@pytest.mark.parametrize("policy", [{}, [123]])
def test_status_rejects_malformed_refresh_policy(monitor: b.Monitor, policy: object) -> None:
    monitor.store.put("refresh_branches", policy)
    state = b._status(monitor.args.workdir)
    assert not state["state_available"]
    assert "Malformed refresh policy" in state["state_error"]


def test_refresh_branch_cli_is_repeatable_and_rejects_invalid_refs(tmp_path: Path) -> None:
    workdir = tmp_path / "unused-bot-state"
    argv = [
        "run",
        "--workdir",
        str(workdir),
        "--vendor",
        "toy",
        "--upstream-repo",
        "org/library",
        "--canonical-repo",
        "operator/library",
        "--canonical-branch",
        "dev",
        "--fork",
        "operator/consumer",
        "--refresh-branch",
        "dev-new",
        "--refresh-branch",
        "../invalid",
        "--once",
    ]
    parsed = b._parser().parse_args(argv)
    assert parsed.refresh_branch == ["dev-new", "../invalid"]
    assert not parsed.publish and not parsed.daemon
    assert b.main(argv) == 1
    assert not workdir.exists()


@pytest.mark.parametrize("malformed", [False, True])
def test_status_reports_unreadable_cache_without_losing_liveness(
    tmp_path: Path, malformed: bool
) -> None:
    database = tmp_path / b._STATE_FILE
    if malformed:
        with contextlib.closing(b.Store(database)) as store:
            store.put("health", [])
    else:
        database.write_text("not SQLite")
    before = database.read_bytes()
    with b._locked(tmp_path):
        result = b._status(tmp_path)
    assert result["running"] and not result["state_available"]
    assert result["state_error"]
    assert database.read_bytes() == before


def test_log_and_status_errors_redact_secrets_and_yaml_excerpts(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="vendor-bot")
    monkeypatch.setenv("GH_TOKEN", "private-environment-token")
    detail = (
        "fetch failed https://user:password@example.invalid/repo "
        "private-environment-token ghp_123456789 github_pat_abcdef\n"
        "PR description text must not be logged"
    )
    monitor.progress.error(detail, 7)
    monitor._transition(7, "blocked", detail)
    diagnostic = json.dumps(monitor.store.get("health")) + caplog.text
    for secret in (
        "user:password",
        "private-environment-token",
        "ghp_123456789",
        "github_pat_abcdef",
        "PR description text",
    ):
        assert secret not in diagnostic
    assert "fetch failed" in diagnostic
    assert b._summary("Authorization: Bearer secret") == "Authorization: [REDACTED]"


def test_serve_reports_backoff_and_clears_process_on_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="vendor-bot")
    b._STOP.clear()
    args = argparse.Namespace(
        workdir=tmp_path,
        publish=False,
        consumer_repo="org/consumer",
        vendor="toy",
        upstream_repo="org/library",
        base_branch="main",
        upstream_branch="main",
        interval=30,
        once=False,
    )

    class BrokenMonitor:
        actor = "operator"

        def __init__(self, args, gh, store, progress):
            self.progress = progress

        def poll(self):
            self.progress.error("HTTP 503")
            raise RuntimeError("HTTP 503")

    def finish_wait(seconds: float) -> bool:
        state = b._status(tmp_path)
        assert state["running"]
        assert state["operation"]["phase"] == "waiting"
        assert state["next_poll_at"]
        assert state["last_error"]["message"] == "HTTP 503"
        b._STOP.set()
        return True

    monkeypatch.setattr(b, "Monitor", BrokenMonitor)
    monkeypatch.setattr(b._STOP, "wait", finish_wait)
    try:
        b._serve(args)
    finally:
        b._STOP.clear()
    assert "in 60s; consecutive_poll_failures=1" in caplog.text
    assert b._status(tmp_path)["pid"] is None
    assert b._status(tmp_path)["next_poll_at"] is None


def test_startup_error_is_visible_after_process_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        workdir=tmp_path,
        publish=False,
        consumer_repo="org/consumer",
        vendor="toy",
        upstream_repo="org/library",
        base_branch="main",
        upstream_branch="main",
        interval=30,
        once=True,
    )

    def unauthorized(self, endpoint: str) -> dict:
        raise RuntimeError("HTTP 401: authentication required")

    monkeypatch.setattr(p.GitHub, "get", unauthorized)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        b._serve(args)
    state = b._status(tmp_path)
    assert not state["running"] and state["pid"] is None
    assert state["last_error"]["message"] == "HTTP 401: authentication required"
    assert state["last_completed_poll"] is None
    assert state["next_poll_at"] is None
    assert not any(t.name == "vendor-bot-heartbeat" for t in threading.enumerate())
