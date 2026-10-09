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
    with pytest.raises(ValueError):
        m.parse(_body(content), "toy", "org/library")


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
    with pytest.raises(ValueError, match="Author action required"):
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
    with pytest.raises(ValueError, match="#41, #42"):
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
    with pytest.raises(ValueError, match="#42, #43"):
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
    with pytest.raises(ValueError, match="#41, #42") as failure:
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

    def api(
        self, endpoint: str, *, method: str = "GET", payload: dict | None = None
    ) -> dict | list:
        if method != "GET":
            self.writes.append((endpoint, method, payload))
        if endpoint == "user":
            return {"login": "operator"}
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


def test_feedback_requests_changes_once_and_clears_only_metadata_review(monitor: b.Monitor) -> None:
    gh = monitor.gh
    monitor._feedback(gh.pr, "Missing marked block")
    monitor._feedback(gh.pr, "Missing marked block")
    assert len(gh.comments) == len(gh.reviews) == 1
    assert "PR description" in gh.comments[0]["body"]
    assert "resolutions:" in gh.comments[0]["body"]
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
    monitor._feedback(monitor.gh.pr, "Missing metadata")
    monitor._feedback(monitor.gh.pr, None)
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    assert len(monitor.gh.reviews) == 1


def test_dry_run_and_self_review(monitor: b.Monitor) -> None:
    monitor.args.publish = False
    monitor._feedback(monitor.gh.pr, "Missing metadata")
    assert not monitor.gh.writes
    monitor.args.publish = True
    monitor.gh.pr["user"]["login"] = "operator"
    monitor._feedback(monitor.gh.pr, "Missing metadata")
    assert monitor.gh.comments
    assert not monitor.gh.reviews


def test_feedback_rechecks_mutable_description(monitor: b.Monitor) -> None:
    stale = copy.deepcopy(monitor.gh.pr)
    monitor.gh.pr["body"] = "updated while checking"
    monitor._feedback(stale, "Missing metadata")
    assert not monitor.gh.writes


def test_discovery_and_persistent_operator_scope(monitor: b.Monitor) -> None:
    assert monitor._discover() == [7]
    assert monitor._discover() == [7]
    assert not monitor.gh.writes
    changed = copy.copy(monitor.args)
    changed.canonical_repo = "different/library"
    with pytest.raises(ValueError, match="different operator"):
        b.Monitor(changed, monitor.gh, monitor.store)


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


def test_feedback_deduplicates_with_empty_sqlite(monitor: b.Monitor) -> None:
    monitor._feedback(monitor.gh.pr, "Missing block")
    writes = copy.deepcopy(monitor.gh.writes)
    with contextlib.closing(b.Store(":memory:")) as store:
        restarted = b.Monitor(monitor.args, monitor.gh, store)
        restarted._feedback(monitor.gh.pr, "Missing block")
        assert monitor.gh.writes == writes
        restarted._feedback(monitor.gh.pr, None)
    assert len(monitor.gh.comments) == len(monitor.gh.reviews) == 1
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"


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
        raise ValueError("Author action required: no exact match for " + reviewed.commit)

    monkeypatch.setattr(m, "resolve", unresolved)
    monitor.inspect(7)
    assert "@author" in monitor.gh.comments[0]["body"]
    assert reviewed.commit in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    monkeypatch.setattr(m, "resolve", lambda *args: ({reviewed.commit: 42}, {}))
    monitor.inspect(7)
    assert monitor.gh.reviews[0]["state"] == "DISMISSED"


def test_forbidden_dismissal_informs_author_without_approving(
    monitor: b.Monitor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor._feedback(monitor.gh.pr, "Missing block")
    original = monitor.gh.api

    def forbidden(endpoint, **kwargs):
        if endpoint.endswith("/dismissals"):
            raise RuntimeError("HTTP 403")
        return original(endpoint, **kwargs)

    monkeypatch.setattr(monitor.gh, "api", forbidden)
    monitor._feedback(monitor.gh.pr, None)
    assert "lacks dismissal permission" in monitor.gh.comments[0]["body"]
    assert monitor.gh.reviews[0]["state"] == "CHANGES_REQUESTED"
    writes = len(monitor.gh.writes)
    monitor._feedback(monitor.gh.pr, None)
    assert len(monitor.gh.writes) == writes
    monkeypatch.setattr(monitor.gh, "api", original)
    monitor._feedback(monitor.gh.pr, None)
    assert "lacks dismissal permission" not in monitor.gh.comments[0]["body"]
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
    monitor._feedback(monitor.gh.pr, "Missing marked block")
    monitor._feedback(monitor.gh.pr, "Missing marked block")
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
    reported = threading.Event()
    original = monitor.progress.heartbeat

    def heartbeat() -> None:
        original()
        reported.set()

    def wait_for_heartbeat(endpoint: str, **kwargs):
        before = monitor.store.get("health")
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
