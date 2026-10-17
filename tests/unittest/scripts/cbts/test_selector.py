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
"""Tests for CoverageSelector: changed core-Python files -> per-stage impacted/skippable sets."""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.coverage.selection.selector import CoverageSelector
from cbts.coverage.tier import apply_coverage_tier
from cbts.rules.base import PRInputs

REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = pytest.mark.cpu_only


class _FakeDB:
    def __init__(self) -> None:
        self.rows = {
            "helper": set(),
            "caller": {"A10-PyTorch-1/test_caller"},
        }

    def file_has_touch_rows(self, _path: str) -> bool:
        return True

    def tests_touching_func(self, _path: str, qualname: str) -> set[str]:
        return set(self.rows.get(qualname, set()))

    def tests_touching_file(self, _path: str) -> set[str]:
        return {
            "A10-PyTorch-1/test_caller",
            "A10-PyTorch-1/test_unrelated",
        }

    def known_by_family(self) -> dict[str, set[str]]:
        return {"A10-PyTorch": {"test_caller", "test_unrelated"}}

    def untrusted_tests(self, *_args) -> set[str]:
        return set()


@pytest.mark.parametrize(
    ("before", "source", "reason"),
    (
        (
            "import sys\nC = sys.modules[__name__]\nVALUE = 1\n"
            "def helper():\n    return VALUE\n"
            "class Kernel:\n    def kern(self):\n        return C.VALUE\n",
            "import sys\nC = sys.modules[__name__]\nVALUE = 2\n"
            "def helper():\n    return VALUE\n"
            "class Kernel:\n    def kern(self):\n        return C.VALUE\n",
            "unresolved module attribute consumer",
        ),
        (
            '@fast_custom_op("trtllm::helper")\ndef helper(value: int):\n    return value\n',
            '@fast_custom_op("trtllm::helper")\n'
            "def helper(value: int, option: int = 0):\n    return value\n",
            "class/signature import change",
        ),
    ),
)
def test_selector_declines_unbounded_alias_and_decorator_effects_with_consumer_rows(
    before: str, source: str, reason: str
) -> None:
    path = "tensorrt_llm/example.py"
    db = _FakeDB()
    db.rows["helper"] = {"A10-PyTorch-1/test_caller"}
    selector = CoverageSelector(
        db,
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, _names: set(),
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    result = selector.decide([path], {path: diff})

    assert not result.ok
    assert reason in result.reason
    assert not result.skippable


class _ImportReplacementDB(_FakeDB):
    def __init__(self) -> None:
        self.rows_by_symbol = {
            ("tensorrt_llm/pkg/helpers.py", "old_helper"): {"A10-PyTorch-1/test_old"},
            ("tensorrt_llm/pkg/helpers.py", "new_helper"): {"A10-PyTorch-1/test_new"},
            ("tensorrt_llm/pkg/consumer.py", "consumer"): {"A10-PyTorch-1/test_consumer"},
        }

    def tests_touching_func(self, path: str, qualname: str) -> set[str]:
        return set(self.rows_by_symbol.get((path, qualname), set()))

    def tests_touching_file(self, path: str) -> set[str]:
        return {
            test
            for (row_path, _), tests in self.rows_by_symbol.items()
            if row_path == path
            for test in tests
        }

    def known_by_family(self) -> dict[str, set[str]]:
        return {
            "A10-PyTorch": {
                "test_old",
                "test_new",
                "test_consumer",
                "test_unrelated",
            }
        }


class _ExternalTestReferenceDB(_FakeDB):
    def known_by_family(self) -> dict[str, set[str]]:
        return {
            "CPU-Generic-x86": {
                "test_unittests.py::test_unittests_v2[unittest/pkg]",
                "test_unittests.py::test_unittests_v2[unittest/pkg/test_unrelated.py]",
            },
            "B300-PyTorch": {
                "test_unittests.py::test_unittests_v2[unittest/pkg/test_consumer.py::test_one]",
                "test_unittests.py::test_unittests_v2[unittest/pkg/test_consumer.py::test_two]",
                "test_unittests.py::test_unittests_v2[unittest/pkg/test_unrelated.py::test_one]",
            },
        }


def test_coverage_tier_uses_raw_diff_for_comment_only_core_change(tmp_path: Path) -> None:
    path = "tensorrt_llm/example.py"
    source_path = tmp_path / path
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# updated explanation\nVALUE = 521\n")
    raw_diff = "@@ -1,2 +1,2 @@\n-# old explanation\n+# updated explanation\n VALUE = 521\n"
    yaml_index = YAMLIndex()
    yaml_index.blocks = [
        Block(
            yaml_stem="l0_a10",
            block_index=0,
            condition={},
            tests=["test_caller", "test_unrelated"],
        )
    ]
    stages = {
        "A10-PyTorch-1": Stage(
            name="A10-PyTorch-1",
            yaml_stem="l0_a10",
            cpu_arch="x86_64",
            split_id=1,
            total_splits=1,
        )
    }
    pr = PRInputs(changed_files=[path], diffs={path: ""}, raw_diffs={path: raw_diff})

    result, note = apply_coverage_tier(pr, [], set(), stages, yaml_index, tmp_path, _FakeDB())

    assert result is not None, note
    assert result.removed == {("l0_a10", 0): {"test_caller", "test_unrelated"}}
    assert result.dropped == {"A10-PyTorch-1"}


def test_selector_uses_local_caller_rows_for_no_data_import_consumer() -> None:
    path = "tensorrt_llm/example.py"
    source = (
        "VALUE = 521\n\ndef helper():\n    return VALUE\n\ndef caller():\n    return helper()\n"
    )
    diff = "@@ -0,0 +1 @@\n+VALUE = 521\n"
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: diff})

    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}
    assert result.skippable == {"A10-PyTorch": {"test_unrelated"}}
    assert result.no_data_funcs == ["tensorrt_llm/example.py::helper"]
    assert result.caller_bounded_funcs == ["tensorrt_llm/example.py::helper"]


@pytest.mark.parametrize("reference_api", ("names", "paths"))
def test_selector_rejects_external_transitive_no_data_caller(reference_api: str) -> None:
    path = "tensorrt_llm/example.py"
    source = (
        "VALUE = 522\n\ndef helper():\n    return VALUE\n\n"
        "def external():\n    return helper()\n\n"
        "def caller():\n    return helper(), external()\n"
    )
    options = (
        {"external_references": lambda _path, names: names & {"external"}}
        if reference_api == "names"
        else {
            "external_reference_paths": lambda _path, names: {
                name: {"tensorrt_llm/consumer.py"} for name in names & {"external"}
            }
        }
    )
    selector = CoverageSelector(_FakeDB(), REPO_ROOT, read_source=lambda _path: source, **options)

    result = selector.decide([path], {path: "@@ -1 +1 @@\n-VALUE = 521\n+VALUE = 522\n"})

    assert not result.ok
    assert "external" in result.reason


def test_selector_uses_transitive_local_caller_rows_when_no_external_reference() -> None:
    path = "tensorrt_llm/example.py"
    source = (
        "VALUE = 522\n\ndef helper():\n    return VALUE\n\n"
        "def middle():\n    return helper()\n\n"
        "def caller():\n    return middle()\n"
    )
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: "@@ -1 +1 @@\n-VALUE = 521\n+VALUE = 522\n"})

    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}
    assert result.skippable == {"A10-PyTorch": {"test_unrelated"}}
    assert result.caller_bounded_funcs == ["tensorrt_llm/example.py::helper"]


@pytest.mark.parametrize("class_scope", (False, True))
@pytest.mark.parametrize("has_import_rows", (False, True))
def test_selector_docstring_replacement_preserves_import_rows_or_file_bound(
    class_scope: bool, has_import_rows: bool
) -> None:
    path = "tensorrt_llm/example.py"
    scope = "Example" if class_scope else "<module>"
    indent = "    " if class_scope else ""
    prefix = "class Example:\n" if class_scope else ""
    source = f'{prefix}{indent}"new docs"\n'
    diff = (
        f"@@ -1,{2 if class_scope else 1} +1,{2 if class_scope else 1} @@\n"
        + (" class Example:\n" if class_scope else "")
        + f'-{indent}"old docs"\n+{indent}"new docs"\n'
    )
    db = _FakeDB()
    if has_import_rows:
        db.rows[scope] = {"A10-PyTorch-1/test_caller"}
    selector = CoverageSelector(
        db,
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: diff})

    assert result.ok
    expected = {"test_caller"} if has_import_rows else {"test_caller", "test_unrelated"}
    assert result.impacted == {"A10-PyTorch": expected}


def test_selector_docstring_replacement_keeps_external_binding_check() -> None:
    path = "tensorrt_llm/example.py"
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: '"new docs"\n',
        external_references=lambda _path, names: names & {"__doc__"},
    )

    result = selector.decide([path], {path: '@@ -1 +1 @@\n-"old docs"\n+"new docs"\n'})

    assert not result.ok
    assert "__doc__" in result.reason


def _write_import_replacement_files(
    tmp_path: Path,
    *,
    include_old_binding: bool = True,
    include_new_helper: bool = True,
) -> tuple[str, str]:
    package = tmp_path / "tensorrt_llm/pkg"
    package.mkdir(parents=True)
    old_binding = "old_helper = object()\n" if include_old_binding else ""
    new_helper = "\ndef new_helper():\n    return 2\n" if include_new_helper else ""
    (package / "helpers.py").write_text(f"{old_binding}{new_helper}")
    (package / "consumer.py").write_text(
        "from .helpers import new_helper\n\ndef consumer():\n    return new_helper()\n"
    )
    path = "tensorrt_llm/pkg/consumer.py"
    diff = (
        "@@ -1,4 +1,4 @@\n"
        "-from .helpers import old_helper\n"
        "+from .helpers import new_helper\n"
        " \n"
        " def consumer():\n"
        "-    return old_helper()\n"
        "+    return new_helper()\n"
    )
    return path, diff


def test_selector_ignores_old_binding_rows_and_uses_new_function_and_consumer_rows(
    tmp_path: Path,
) -> None:
    path, diff = _write_import_replacement_files(tmp_path)
    checked_bindings: list[set[str]] = []

    def external_references(_path: str, names: set[str]) -> set[str]:
        checked_bindings.append(names)
        return names & {"new_helper"}

    selector = CoverageSelector(
        _ImportReplacementDB(),
        tmp_path,
        external_references=external_references,
    )

    result = selector.decide([path], {path: diff})

    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_new", "test_consumer"}}
    assert result.skippable == {"A10-PyTorch": {"test_old", "test_unrelated"}}
    assert checked_bindings == [{"old_helper"}]


def test_selector_keeps_external_check_for_rebound_import_binding(tmp_path: Path) -> None:
    path, diff = _write_import_replacement_files(tmp_path)
    selector = CoverageSelector(
        _ImportReplacementDB(),
        tmp_path,
        external_references=lambda _path, names: names & {"old_helper"},
    )

    result = selector.decide([path], {path: diff})

    assert not result.ok
    assert "external binding reference(s)" in result.reason
    assert "old_helper" in result.reason


def test_selector_declines_missing_old_static_binding(tmp_path: Path) -> None:
    path, diff = _write_import_replacement_files(tmp_path, include_old_binding=False)
    selector = CoverageSelector(
        _ImportReplacementDB(),
        tmp_path,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: diff})

    assert not result.ok
    assert "old import target is not a static binding" in result.reason


def test_selector_declines_missing_imported_function(tmp_path: Path) -> None:
    path, diff = _write_import_replacement_files(tmp_path, include_new_helper=False)
    selector = CoverageSelector(
        _ImportReplacementDB(),
        tmp_path,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: diff})

    assert not result.ok
    assert "import target is not a static function" in result.reason


def test_selector_allows_new_imported_function_without_coverage(tmp_path: Path) -> None:
    path, diff = _write_import_replacement_files(tmp_path)
    db = _ImportReplacementDB()
    db.rows_by_symbol[("tensorrt_llm/pkg/helpers.py", "new_helper")] = set()
    selector = CoverageSelector(
        db,
        tmp_path,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: diff})

    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_consumer"}}
    assert result.skippable == {"A10-PyTorch": {"test_old", "test_new", "test_unrelated"}}


def test_selector_declines_when_changed_binding_has_external_reference() -> None:
    path = "tensorrt_llm/example.py"
    source = "VALUE = 521\n\ndef helper():\n    return VALUE\n"
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, names: names & {"VALUE"},
    )

    result = selector.decide([path], {path: "@@ -0,0 +1 @@\n+VALUE = 521\n"})

    assert not result.ok
    assert "external binding reference(s)" in result.reason


def test_selector_uses_pre_image_consumer_for_deleted_literal() -> None:
    path = "tensorrt_llm/example.py"
    source = "\ndef helper():\n    return VALUE\n\ndef caller():\n    return helper()\n"
    diff = "@@ -1,4 +1,3 @@\n-VALUE = 521\n \n def helper():\n     return VALUE\n"
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: diff})

    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}
    assert result.skippable == {"A10-PyTorch": {"test_unrelated"}}
    assert result.no_data_funcs == ["tensorrt_llm/example.py::helper"]
    assert result.caller_bounded_funcs == ["tensorrt_llm/example.py::helper"]


def test_selector_declines_deleted_literal_with_external_reference() -> None:
    path = "tensorrt_llm/example.py"
    source = "\ndef helper():\n    return VALUE\n"
    diff = "@@ -1,4 +1,3 @@\n-VALUE = 521\n \n def helper():\n     return VALUE\n"
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, names: names & {"VALUE"},
    )

    result = selector.decide([path], {path: diff})

    assert not result.ok
    assert "external binding reference(s)" in result.reason
    assert "VALUE" in result.reason


def test_selector_selects_registered_external_test_reference() -> None:
    path = "tensorrt_llm/example.py"
    source = "VALUE = 522\n"
    diff = "@@ -1 +1 @@\n-VALUE = 521\n+VALUE = 522\n"
    selector = CoverageSelector(
        _ExternalTestReferenceDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_reference_paths=lambda _path, names: {
            name: {"tests/unittest/pkg/test_consumer.py"} for name in names
        },
    )

    result = selector.decide([path], {path: diff})

    consumer = "test_unittests.py::test_unittests_v2[unittest/pkg]"
    unrelated = "test_unittests.py::test_unittests_v2[unittest/pkg/test_unrelated.py]"
    assert result.ok
    assert result.impacted == {
        "B300-PyTorch": {
            "test_unittests.py::test_unittests_v2[unittest/pkg/test_consumer.py::test_one]",
            "test_unittests.py::test_unittests_v2[unittest/pkg/test_consumer.py::test_two]",
        },
        "CPU-Generic-x86": {consumer},
    }
    assert result.skippable == {
        "B300-PyTorch": {
            "test_unittests.py::test_unittests_v2[unittest/pkg/test_unrelated.py::test_one]"
        },
        "CPU-Generic-x86": {unrelated},
    }


@pytest.mark.parametrize(
    ("reference_path", "reason"),
    (
        ("tensorrt_llm/consumer.py", "non-test external reference"),
        (
            "tests/unittest/missing/test_missing.py",
            "unregistered external test reference",
        ),
    ),
)
def test_selector_declines_unbounded_external_reference_path(
    reference_path: str, reason: str
) -> None:
    path = "tensorrt_llm/example.py"
    source = "VALUE = 522\n"
    diff = "@@ -1 +1 @@\n-VALUE = 521\n+VALUE = 522\n"
    selector = CoverageSelector(
        _ExternalTestReferenceDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_reference_paths=lambda _path, names: {name: {reference_path} for name in names},
    )

    result = selector.decide([path], {path: diff})

    assert not result.ok
    assert reason in result.reason


@pytest.mark.parametrize(
    "entry",
    (
        "unittest/pkg -k focused",
        "unittest/pkg --ignore=unittest/pkg/test_consumer.py",
        "unittest/pkg --ignore unittest/pkg/test_consumer.py",
    ),
)
def test_unittest_entry_does_not_claim_partial_test_file(entry: str) -> None:
    assert not CoverageSelector._unittest_entry_covers(entry, "unittest/pkg/test_consumer.py")


def test_unittest_entry_matches_file_and_directory_entries() -> None:
    path = "unittest/pkg/test_consumer.py"

    assert CoverageSelector._unittest_entry_covers(path, path)
    assert CoverageSelector._unittest_entry_covers(f"{path}::test_one", path)
    assert CoverageSelector._unittest_entry_covers("unittest/pkg", path)
    assert not CoverageSelector._unittest_entry_covers("unittest/other", path)


def test_selector_ignores_external_reference_for_pure_new_function() -> None:
    path = "tensorrt_llm/example.py"
    source = "VALUE = 1\n\ndef helper():\n    return VALUE\n"
    diff = "@@ -1 +1,4 @@\n VALUE = 1\n+\n+def helper():\n+    return VALUE\n"
    checked_bindings: list[set[str]] = []

    def external_references(_path: str, names: set[str]) -> set[str]:
        checked_bindings.append(names)
        return names & {"helper"}

    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=external_references,
    )

    result = selector.decide([path], {path: diff})

    assert result.ok
    assert checked_bindings == [set()]


def test_selector_uses_file_fallback_when_consumer_escapes() -> None:
    path = "tensorrt_llm/example.py"
    source = (
        "VALUE = 521\n\n"
        "def helper():\n"
        "    return VALUE\n\n"
        "def caller():\n"
        "    alias = helper\n"
        "    return alias()\n"
    )
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: "@@ -0,0 +1 @@\n+VALUE = 521\n"})

    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_caller", "test_unrelated"}}
    assert result.caller_bounded_funcs == []


def test_ordinary_no_data_function_change_keeps_file_fallback() -> None:
    path = "tensorrt_llm/example.py"
    source = "def helper():\n    return 2\n\ndef caller():\n    return helper()\n"
    selector = CoverageSelector(
        _FakeDB(),
        REPO_ROOT,
        read_source=lambda _path: source,
        external_references=lambda _path, _names: set(),
    )

    result = selector.decide([path], {path: "@@ -2 +2 @@\n-    return 1\n+    return 2\n"})

    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_caller", "test_unrelated"}}
    assert result.caller_bounded_funcs == []
