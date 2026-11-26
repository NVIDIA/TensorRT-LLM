# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import sanity composes with the existing runtime coverage policy."""

import difflib
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex, parse_stages_from_groovy
from cbts.command.main import Selector
from cbts.coverage.selection.python_change_analysis import analyze_execution_phases
from cbts.coverage.selection.selector import CoverageSelector
from cbts.coverage.tier import apply_coverage_tier
from cbts.rules.base import PRInputs
from cbts.rules.import_sanity import ImportSanityRule

pytestmark = pytest.mark.cpu_only

Case = tuple[str, Path, Path, YAMLIndex, dict[str, Stage]]


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
    test = tmp_path / "tests/unittest/test_import.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_import(): pass\n")
    inventory = tmp_path / "tests/integration/test_lists/import_sanity.yml"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("- unittest/test_import.py\n")
    index = YAMLIndex()
    index.blocks = [
        Block("l0_cpu", 0, {}, ["unittest/test_import.py"]),
        Block("l0_a10", 0, {}, ["test_x.py::test_caller", "test_x.py::test_unrelated"]),
    ]
    stages = {
        "CPU-Generic-x86-1": Stage("CPU-Generic-x86-1", "l0_cpu", "x86", 1, 1),
        "A10-PyTorch-1": Stage("A10-PyTorch-1", "l0_a10", "x86", 1, 1),
    }
    return path, source, inventory, index, stages


def test_import_only_is_handled_without_coverage(setup_case: Case, tmp_path: Path) -> None:
    path, source, _, index, stages = setup_case
    source.write_text("VALUE = opaque_registration()\n")
    pr = PRInputs([path], {path: _diff("VALUE = 1\n", source.read_text())})
    selector = Selector(stages)
    result = selector.run(pr, [ImportSanityRule(index, stages, tmp_path)])
    assert result.scope == "importsanityonly"
    assert selector.handled == {path}
    assert result.affected_stages == {"CPU-Generic-x86-1"}
    assert not result.perfsanity_required


@pytest.mark.parametrize("function_rows", [False, True])
def test_mixed_edit_keeps_runtime_function_or_file_coverage(
    setup_case: Case, tmp_path: Path, function_rows: bool
) -> None:
    path, source, _, index, stages = setup_case
    before = "VALUE = 1\ndef caller():\n    return 1\n"
    source.write_text("VALUE = external_factory()\ndef caller():\n    return 2\n")
    pr = PRInputs([path], {path: _diff(before, source.read_text())})
    selector = Selector(stages)
    assert selector.run(pr, [ImportSanityRule(index, stages, tmp_path)]).scope is None
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
    assert "CPU-Generic-x86-1" in tier.affected_stages
    assert ("l0_cpu", 0) not in tier.removed
    if function_rows:
        assert tier.removed[("l0_a10", 0)] == {"test_x.py::test_unrelated"}
    else:
        assert not tier.removed  # Default no-data policy remains whole-file.


def test_mixed_edit_without_file_coverage_still_falls_back(
    setup_case: Case, tmp_path: Path
) -> None:
    path, source, _, index, stages = setup_case
    before = "VALUE = 1\ndef caller(): return 1\n"
    source.write_text("VALUE = 2\ndef caller(): return 2\n")
    pr = PRInputs([path], {path: _diff(before, source.read_text())})
    selector = Selector(stages)
    selector.run(pr, [ImportSanityRule(index, stages, tmp_path)])
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


def test_body_only_does_not_require_import_inventory(setup_case: Case, tmp_path: Path) -> None:
    path, source, inventory, index, stages = setup_case
    source.write_text("def caller(): return 2\n")
    inventory.unlink()
    pr = PRInputs([path], {path: _diff("def caller(): return 1\n", source.read_text())})
    assert ImportSanityRule(index, stages, tmp_path).apply(pr) is None


def test_removed_definition_retains_old_runtime_rows(setup_case: Case, tmp_path: Path) -> None:
    path, source, _, index, stages = setup_case
    before = "def caller(): return 1\ndef other(): return 2\n"
    source.write_text("def other(): return 2\n")
    pr = PRInputs([path], {path: _diff(before, source.read_text())})
    selector = Selector(stages)
    selector.run(pr, [ImportSanityRule(index, stages, tmp_path)])
    tier, note = apply_coverage_tier(
        pr, selector.pairs, selector.handled, stages, index, tmp_path, _DB()
    )
    assert tier is not None, note
    assert tier.removed[("l0_a10", 0)] == {"test_x.py::test_unrelated"}


@pytest.mark.parametrize("missing", ["inventory", "target", "stage"])
def test_missing_sanity_obligation_declines(setup_case: Case, tmp_path: Path, missing: str) -> None:
    path, source, inventory, index, stages = setup_case
    source.write_text("VALUE = 2\n")
    if missing == "inventory":
        inventory.unlink()
    elif missing == "target":
        inventory.write_text("- unittest/test_import.py::test_missing\n")
    else:
        stages = {}
    result = ImportSanityRule(index, stages, tmp_path).apply(
        PRInputs([path], {path: _diff("VALUE = 1\n", source.read_text())})
    )
    assert result is not None
    assert result.scope is None


def test_repository_inventory_has_registered_premerge_stages() -> None:
    root = Path(__file__).resolve().parents[4]
    path = "tensorrt_llm/__init__.py"
    source = (root / path).read_text()
    before = source + "\nCBTS_REMOVED_IMPORT_SENTINEL = 1\n"
    index = YAMLIndex.load(root / "tests/integration/test_lists/test-db")
    stages = parse_stages_from_groovy(root / "jenkins/L0_Test.groovy", include_post_merge=True)
    result = ImportSanityRule(index, stages, root).apply(
        PRInputs([path], {path: _diff(before, source)})
    )
    assert result is not None
    assert result.scope == "importsanityonly", result.reason
    assert result.handled_files == {path}
    assert any(stage.startswith("CPU-") for stage in result.affected_stages)
    assert result.sanity_relevant
    assert any(
        "unittest/others/test_import_sanity.py" in block.tests
        for block in index.blocks
        if block.yaml_stem == "l0_sanity_check"
    )
