# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import sanity composes with the existing runtime coverage policy."""

import difflib
import json
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.command.main import Selector, _Args, _run
from cbts.coverage.selection.python_change_analysis import analyze_execution_phases
from cbts.coverage.selection.selector import CoverageSelector
from cbts.coverage.tier import apply_coverage_tier
from cbts.rules.base import PRInputs
from cbts.rules.import_sanity import ImportSanityRule
from cbts.rules.test_list import TestListRule as _TestListRule

pytestmark = pytest.mark.cpu_only

Case = tuple[str, Path, YAMLIndex, dict[str, Stage]]


def _diff(before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))


@pytest.mark.parametrize(
    ("before", "after", "imports", "runtime"),
    [
        ("VALUE = 1\n", "VALUE = 2\n", {"<module>"}, set()),
        ("import old\n", "import new\n", {"<module>"}, set()),
        ("VALUE = 1\nOTHER = 2\n", "OTHER = 2\n", {"<module>"}, set()),
        ("class C:\n    value = 1\n", "class C:\n    value = 2\n", {"C"}, set()),
        ("def f(x=1): return x\n", "def f(x=2): return x\n", {"<module>"}, set()),
        ("def f():\n    return 1\n", "def f():\n    return 2\n", set(), {"f"}),
        ("def f(): return 1\n", "def f(): return 2\n", set(), {"f"}),
        ("def f(): return 1\n", "async def f(): return 1\n", {"<module>"}, set()),
        ("", "def f(): return 1\n", {"<module>"}, {"f"}),
        ("def f(): return 1\n", "", {"<module>"}, {"f"}),
        (
            "VALUE = 1\ndef f():\n    return 1\n",
            "VALUE = 2\ndef f():\n    return 2\n",
            {"<module>"},
            {"f"},
        ),
        ("# before\ndef f(): return 1\n", "# after\ndef f(): return 1\n", set(), set()),
        (
            "@first\ndef f(): return 1\n",
            "@second\ndef f(): return 1\n",
            {"<module>"},
            set(),
        ),
    ],
)
def test_execution_phase_attribution(
    before: str, after: str, imports: set[str], runtime: set[str]
) -> None:
    result = analyze_execution_phases(after, before)
    assert not result.limitation
    assert result.import_qualnames == imports
    assert result.runtime_qualnames == runtime


class _DB:
    def __init__(self, *, function_rows: bool = True, file_rows: bool = True) -> None:
        self.function_rows = function_rows
        self.file_rows = file_rows

    def file_has_touch_rows(self, _path: str) -> bool:
        return self.file_rows

    def tests_touching_func(self, _path: str, name: str) -> set[str]:
        return (
            {"A10-PyTorch-1/test_x.py::test_caller"}
            if name == "caller" and self.function_rows
            else set()
        )

    def tests_touching_file(self, _path: str) -> set[str]:
        return {f"A10-PyTorch-1/{name}" for name in self.known_by_family()["A10-PyTorch"]}

    def known_by_family(self) -> dict[str, set[str]]:
        return {"A10-PyTorch": {"test_x.py::test_caller", "test_x.py::test_unrelated"}}

    def untrusted_tests(self, *_args: object) -> set[str]:
        return set()


@pytest.fixture
def setup_case(tmp_path: Path) -> Case:
    path = "examples/helper.py"
    source = tmp_path / path
    source.parent.mkdir()
    index = YAMLIndex()
    index.blocks = [
        Block("l0_sanity_check", 0, {}, ["llmapi/test_llm_examples.py::test_llmapi_quickstart"]),
        Block("l0_a10", 0, {}, ["test_x.py::test_caller", "test_x.py::test_unrelated"]),
    ]
    stages = {
        "A10-PyTorch-1": Stage("A10-PyTorch-1", "l0_a10", "x86", 1, 1),
    }
    return path, source, index, stages


def test_import_only_is_handled_without_coverage(setup_case: Case, tmp_path: Path) -> None:
    path, source, index, stages = setup_case
    source.write_text("VALUE = opaque_registration()\n")
    pr = PRInputs([path], {path: _diff("VALUE = 1\n", source.read_text())})
    selector = Selector(stages)
    result = selector.run(pr, [ImportSanityRule(tmp_path)])
    assert result.scope == "importsanityonly"
    assert selector.handled == {path}
    assert not result.affected_stages
    assert not result.block_filters
    assert result.sanity_required
    assert not result.perfsanity_required


@pytest.mark.parametrize("function_rows", [False, True])
def test_mixed_edit_keeps_runtime_function_or_file_coverage(
    setup_case: Case, tmp_path: Path, function_rows: bool
) -> None:
    path, source, index, stages = setup_case
    before = "VALUE = 1\ndef caller():\n    return 1\n"
    source.write_text("VALUE = external_factory()\ndef caller():\n    return 2\n")
    pr = PRInputs([path], {path: _diff(before, source.read_text())})
    selector = Selector(stages)
    assert selector.run(pr, [ImportSanityRule(tmp_path)]).scope is None
    assert not selector.handled
    tier, note = apply_coverage_tier(
        pr,
        selector.pairs,
        selector.handled,
        stages,
        index,
        tmp_path,
        _DB(function_rows=function_rows),
    )
    assert tier is not None, note
    assert ("l0_sanity_check", 0) not in tier.removed
    if function_rows:
        assert tier.removed[("l0_a10", 0)] == {"test_x.py::test_unrelated"}
    else:
        assert not tier.removed  # Default no-data policy remains whole-file.


def test_mixed_edit_without_file_coverage_still_falls_back(
    setup_case: Case, tmp_path: Path
) -> None:
    path, source, index, stages = setup_case
    before = "VALUE = 1\ndef caller(): return 1\n"
    source.write_text("VALUE = 2\ndef caller(): return 2\n")
    pr = PRInputs([path], {path: _diff(before, source.read_text())})
    selector = Selector(stages)
    selector.run(pr, [ImportSanityRule(tmp_path)])
    tier, note = apply_coverage_tier(
        pr, selector.pairs, selector.handled, stages, index, tmp_path, _DB(file_rows=False)
    )
    assert tier is None
    assert "zero-touch" in note


def test_mixed_runtime_shared_fixture_protection_remains(tmp_path: Path) -> None:
    path = "examples/helper.py"
    before = (
        "import pytest\nVALUE = 1\ndef caller():\n    return 1\n"
        "@pytest.fixture(scope='session')\ndef shared():\n    return caller()\n"
    )
    after = before.replace("VALUE = 1", "VALUE = 2").replace("return 1", "return 2")
    result = CoverageSelector(
        _DB(), tmp_path, read_source=lambda _: after, import_sanity_files={path}
    ).decide([path], {path: _diff(before, after)})
    assert not result.ok
    assert "shared" in result.reason


def test_body_only_does_not_request_package_sanity(setup_case: Case, tmp_path: Path) -> None:
    path, source, index, stages = setup_case
    source.write_text("def caller(): return 2\n")
    pr = PRInputs([path], {path: _diff("def caller(): return 1\n", source.read_text())})
    assert ImportSanityRule(tmp_path).apply(pr) is None


def test_removed_definition_retains_old_runtime_rows(setup_case: Case, tmp_path: Path) -> None:
    path, source, index, stages = setup_case
    before = "def caller(): return 1\ndef other(): return 2\n"
    source.write_text("def other(): return 2\n")
    pr = PRInputs([path], {path: _diff(before, source.read_text())})
    selector = Selector(stages)
    selector.run(pr, [ImportSanityRule(tmp_path)])
    tier, note = apply_coverage_tier(
        pr, selector.pairs, selector.handled, stages, index, tmp_path, _DB()
    )
    assert tier is not None, note
    assert tier.removed[("l0_a10", 0)] == {"test_x.py::test_unrelated"}


def test_import_only_needs_no_named_stages(setup_case: Case, tmp_path: Path) -> None:
    path, source, _, _ = setup_case
    source.write_text("VALUE = 2\n")
    pr = PRInputs([path], {path: _diff("VALUE = 1\n", source.read_text())})
    result = Selector({}).run(pr, [ImportSanityRule(tmp_path)])
    assert result.scope == "importsanityonly"
    assert result.sanity_required
    assert not result.perfsanity_required
    assert not result.affected_stages


def test_repository_import_edit_uses_existing_package_sanity(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    root = Path(__file__).resolve().parents[4]
    path = "tensorrt_llm/__init__.py"
    source = (root / path).read_text()
    before = source + "\nCBTS_REMOVED_IMPORT_SENTINEL = 1\n"
    result = ImportSanityRule(root).apply(PRInputs([path], {path: _diff(before, source)}))
    assert result is not None
    assert result.scope == "importsanityonly", result.reason
    assert result.handled_files == {path}
    assert not result.affected_stages
    assert not result.block_filters
    assert result.sanity_relevant
    # Execute the command boundary: an import-only edit needs neither coverage
    # nor an independent sanity inventory to produce an actionable decision.
    payload = tmp_path / "input.json"
    payload.write_text(
        json.dumps({"changed_files": [path], "diffs": {path: _diff(before, source)}})
    )
    assert _run(_Args(str(payload), False, str(root), None, None, None, None, "file")) == 0
    decision = json.loads(capfd.readouterr().out)
    assert decision["scope"] == "importsanityonly"
    assert decision["sanity_required"]
    assert not decision["perfsanity_required"]
    assert not decision["test_db_dir_override"]


def test_import_and_sanity_list_edit_keep_full_shared_suite(
    setup_case: Case, tmp_path: Path
) -> None:
    path, source, index, stages = setup_case
    source.write_text("VALUE = 2\n")
    list_path = "tests/integration/test_lists/test-db/l0_sanity_check.yml"
    suite = tmp_path / list_path
    suite.parent.mkdir(parents=True)
    entry = index.blocks[0].tests[0]
    before = "l0_sanity_check:\n- tests:\n  - unittest/others/test_import_gpu_free.py\n"
    after = before + f"  - {entry}\n"
    suite.write_text(after)
    index = YAMLIndex.load(suite.parent)
    pr = PRInputs(
        [path, list_path],
        {path: _diff("VALUE = 1\n", source.read_text()), list_path: _diff(before, after)},
    )
    result = Selector(stages).run(
        pr, [ImportSanityRule(tmp_path), _TestListRule(index, stages, repo_root=tmp_path)]
    )
    assert result.scope == "testsonly"
    assert result.sanity_required
    assert not result.block_filters
