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
"""Behavioral contract for the isolated lookup-only CBTS candidate."""

import difflib

import pytest
import yaml
from cbts.blocks import Stage, YAMLIndex
from cbts.command.main import Selector
from cbts.coverage.selection.python_change_analysis import symbols_for_diff
from cbts.coverage.selection.selector import CoverageSelector
from cbts.coverage.selection.touch_db import db_key
from cbts.coverage.tier import apply_coverage_tier, write_coverage_test_db
from cbts.rules.base import PRInputs
from cbts.rules.import_sanity import ImportSanityRule
from cbts.rules.tests_def import TestsDefRule as DefinitionRule

pytestmark = pytest.mark.cpu_only


class _FakeDB:
    def __init__(self):
        self.rows = {"caller": {"A10-PyTorch-1/test_caller"}}
        self.queries = []
        self.untrusted = set()

    def tests_touching_func(self, path, qualname):
        self.queries.append((path, qualname))
        return set(self.rows.get(qualname, set()))

    def known_by_family(self):
        return {"A10-PyTorch": {"test_caller", "test_unrelated", "test_import"}}

    def untrusted_tests(self, *_):
        return self.untrusted

    def tests_touching_file(self, _):
        raise AssertionError("lookup must not expand to whole-file tests")

    def file_has_touch_rows(self, _):
        raise AssertionError("missing file rows must not force fallback")


def _diff(before, after):
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))


def _decide(tmp_path, before, after, *, rows=None, sanity=False, **options):
    path = "tensorrt_llm/example.py"
    db = _FakeDB()
    if rows is not None:
        db.rows = rows
    selector = CoverageSelector(
        db,
        tmp_path,
        read_source={path: after}.get,
        import_sanity_files={path} if sanity else set(),
        **options,
    )
    return selector.decide([path], {path: _diff(before, after)}), db


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "@opaque\n",
        "@cache()\n",
        "@nvtx_range('caller')\n",
        "@pytest.fixture(scope='session')\n",
    ],
)
def test_function_body_uses_only_its_recorded_rows(tmp_path, prefix):
    before = prefix + "def caller():\n    return 1\n"
    after = before.replace("return 1", "return 2")
    result, db = _decide(tmp_path, before, after)
    assert result.ok, result.reason
    assert db.queries == [("tensorrt_llm/example.py", "caller")]
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}
    assert result.caller_bounded_funcs == []


def test_reflection_and_opaque_ancestor_do_not_change_lookup(tmp_path):
    before = (
        "def caller():\n    return 1\n"
        "@opaque\ndef entry():\n    return caller()\n"
        "def unrelated():\n    return globals()\n"
    )
    result, db = _decide(tmp_path, before, before.replace("return 1", "return 2"))
    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}
    assert db.queries == [("tensorrt_llm/example.py", "caller")]


@pytest.mark.parametrize("policy", ["file", "importers", "ignore"])
def test_absent_function_rows_add_no_tests_under_compatibility_options(tmp_path, policy):
    before = "def new_path():\n    return 1\n"
    result, db = _decide(tmp_path, before, before.replace("1", "2"), no_data_policy=policy)
    assert result.ok
    assert not result.impacted
    assert result.no_data_funcs == ["tensorrt_llm/example.py::new_path"]
    assert db.queries == [("tensorrt_llm/example.py", "new_path")]


@pytest.mark.parametrize(
    "before,after,expected",
    [
        (
            "def removed():\n    return 1\n\ndef untouched():\n    return 2\n",
            "def untouched():\n    return 2\n",
            {"<module>", "removed"},
        ),
        (
            "def caller():\n    return 1\n    return 2\n",
            "def caller():\n    return 1\n",
            {"caller"},
        ),
        (
            "def caller():\n    def inner():\n        return 1\n    return inner()\n",
            "def caller():\n    def inner():\n        return 2\n    return inner()\n",
            {"caller.<locals>.inner"},
        ),
        (
            "class C:\n    def method(self):\n        return 1\n",
            "class C:\n    def method(self):\n        return 2\n",
            {"C.method"},
        ),
    ],
)
def test_both_images_locate_exact_changed_symbols(before, after, expected):
    facts = symbols_for_diff(after, _diff(before, after))
    assert not facts.limitation
    assert facts.qualnames == expected


def test_removed_function_looks_up_old_name_without_neighbor_expansion(tmp_path):
    before = "def removed():\n    return 1\n\ndef untouched():\n    return 2\n"
    after = "def untouched():\n    return 2\n"
    result, db = _decide(
        tmp_path, before, after, sanity=True, rows={"removed": {"A10-PyTorch-1/test_caller"}}
    )
    assert result.ok
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}
    assert {name for _, name in db.queries} == {"removed", "<module>"}


@pytest.mark.parametrize(
    "before,after,symbols",
    [
        (
            "LIMIT=10\ndef caller():\n    return LIMIT\n",
            "LIMIT=20\ndef caller():\n    return LIMIT\n",
            {"<module>"},
        ),
        (
            "from provider import first as operation\ndef caller():\n    return operation()\n",
            "from provider import second as operation\ndef caller():\n    return operation()\n",
            {"<module>"},
        ),
        ("class C:\n    value=1\n", "class C:\n    value=2\n", {"C"}),
        (
            "def caller(value=1):\n    return value\n",
            "def caller(value=2):\n    return value\n",
            {"<module>"},
        ),
    ],
)
def test_import_phase_requires_sanity_and_does_not_infer_binding_consumers(
    tmp_path, before, after, symbols
):
    rejected, _ = _decide(tmp_path, before, after)
    assert not rejected.ok and "import sanity required" in rejected.reason
    result, db = _decide(
        tmp_path, before, after, sanity=True, rows={"<module>": {"A10-PyTorch-1/test_import"}}
    )
    assert result.ok
    assert {name for _, name in db.queries} == symbols
    assert "test_caller" not in result.impacted.get("A10-PyTorch", set())


def test_new_function_and_zero_touch_file_are_valid_empty_lookup(tmp_path):
    result, db = _decide(
        tmp_path, "", "def added() -> external.Type:\n    return 1\n", sanity=True, rows={}
    )
    assert result.ok
    assert result.impacted == {}
    assert {name for _, name in db.queries} == {"added", "<module>"}


@pytest.mark.parametrize(
    "kind", ["missing_diff", "mismatch", "syntax", "non_python", "missing_source"]
)
def test_bad_input_keeps_infrastructure_fallback(tmp_path, kind):
    before, after = "def caller():\n    return 1\n", "def caller():\n    return 2\n"
    path = "tensorrt_llm/example.py"
    diff = _diff(before, after)
    source = after
    if kind == "missing_diff":
        diff = ""
    if kind == "mismatch":
        source = "def caller():\n    return 9\n"
    if kind == "syntax":
        source = "def caller(:\n"
        diff = _diff(before, source)
    if kind == "non_python":
        path = "native.cpp"
    if kind == "missing_source":
        source = None
    result = CoverageSelector(_FakeDB(), tmp_path, read_source=lambda _: source).decide(
        [path], {path: diff}
    )
    assert not result.ok


def test_deleted_file_uses_verified_pre_image(tmp_path):
    path = "tensorrt_llm/example.py"
    diff = _diff("def caller():\n    return 1\n", "").replace("+++ \n", "+++ /dev/null\n")
    result = CoverageSelector(
        _FakeDB(), tmp_path, read_source=lambda _: None, import_sanity_files={path}
    ).decide([path], {path: diff})
    assert result.ok, result.reason
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}


def test_capture_quality_retention_remains_independent(tmp_path):
    db = _FakeDB()
    db.untrusted = {"A10-PyTorch-1/test_unrelated"}
    path = "tensorrt_llm/example.py"
    before, after = "def caller():\n    return 1\n", "def caller():\n    return 2\n"
    result = CoverageSelector(db, tmp_path, read_source=lambda _: after).decide(
        [path], {path: _diff(before, after)}
    )
    assert result.impacted == {"A10-PyTorch": {"test_caller"}}
    assert "test_unrelated" not in result.skippable["A10-PyTorch"]
    assert result.n_untrusted == 1


@pytest.mark.parametrize("new_test_in_db", [False, True])
def test_written_yaml_retains_import_checks_runtime_rows_and_pr_tests(tmp_path, new_test_in_db):
    path = "tensorrt_llm/example.py"
    before = "import provider\ndef caller():\n    return 1\n"
    after = "import provider\ndef added() -> provider.Result:\n    return 2\ndef caller():\n    return added()\n"
    source = tmp_path / path
    source.parent.mkdir(parents=True)
    source.write_text(after)
    test_root = tmp_path / "tests/unittest"
    test_root.mkdir(parents=True)
    (test_root / "test_import.py").write_text("def test_import(): pass\n")
    new_test = "def test_new():\n    assert True\n"
    (test_root / "test_new.py").write_text(new_test)
    inventory = tmp_path / "tests/integration/test_lists/import_sanity.yml"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("- unittest/test_import.py\n")
    entries = [
        "unittest/test_import.py",
        "unittest/test_runtime.py",
        "unittest/test_new.py",
        "unittest/test_unrelated.py",
    ]
    test_db = inventory.parent / "test-db"
    test_db.mkdir()
    (test_db / "l0_a10.yml").write_text(
        yaml.safe_dump(
            {
                "l0_a10": [
                    {"condition": {}, "tests": entries},
                    {"condition": {}, "tests": ["unittest/other/test_other.py"]},
                ]
            }
        )
    )
    index = YAMLIndex.load(test_db)
    stages = {"A10-PyTorch-1": Stage("A10-PyTorch-1", "l0_a10", "x86_64", 1, 1)}
    pr = PRInputs(
        [path, "tests/unittest/test_new.py"],
        {path: _diff(before, after), "tests/unittest/test_new.py": _diff("", new_test)},
    )
    rules = [
        ImportSanityRule(index, stages, tmp_path),
        DefinitionRule(index, stages, repo_root=tmp_path),
    ]
    selector = Selector(stages)
    assert selector.run(pr, rules).scope is None

    class DB(_FakeDB):
        def known_by_family(self):
            return {
                "A10-PyTorch": {
                    db_key(e) for e in entries if new_test_in_db or "test_new.py" not in e
                }
            }

        def tests_touching_func(self, _path, name):
            return (
                {"A10-PyTorch-1/" + db_key("unittest/test_runtime.py")}
                if name == "caller"
                else set()
            )

    tier, note = apply_coverage_tier(
        pr, selector.pairs, selector.handled, stages, index, tmp_path, DB()
    )
    assert tier is not None, note
    assert tier.removed[("l0_a10", 0)] == {"unittest/test_unrelated.py"}
    out = tmp_path / "selected"
    write_coverage_test_db(test_db, out, tier.removed)
    assert yaml.safe_load((out / "l0_a10.yml").read_text())["l0_a10"][0]["tests"] == entries[:-1]
