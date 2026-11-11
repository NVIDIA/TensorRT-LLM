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
"""Import success and runtime coverage are independent selection obligations."""

import difflib

import pytest
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.command.main import Selector
from cbts.coverage.selection.python_change_analysis import analyze_execution_phases
from cbts.coverage.tier import apply_coverage_tier
from cbts.rules.base import PRInputs
from cbts.rules.import_sanity import ImportSanityRule

from .test_selector import _FakeDB

pytestmark = pytest.mark.cpu_only


def _diff(before, after):
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))


@pytest.mark.parametrize(
    "before,after,imported,runtime",
    [
        ("def f():\n    return 1\n", "def f():\n    return 2\n", False, {"f"}),
        ("def f(x=1):\n    return x\n", "def f(x=2):\n    return x\n", True, set()),
        ("class C:\n    x=1\n", "class C:\n    x=2\n", True, set()),
        (
            "def f():\n    class C:\n        x=1\n",
            "def f():\n    class C:\n        x=2\n",
            False,
            {"f.<locals>.C"},
        ),
        ("from math import sin\n", "", True, set()),
    ],
)
def test_execution_phases_both_images(before, after, imported, runtime):
    facts = analyze_execution_phases(after, before)
    assert bool(facts.import_qualnames) == imported
    assert facts.runtime_qualnames == runtime


def _registered_sanity(tmp_path, before, after):
    path = "tensorrt_llm/helper.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    file.write_text(after)
    test = tmp_path / "tests/unittest/test_sanity.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_import(): pass\n")
    manifest = tmp_path / "tests/integration/test_lists/import_sanity.yml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("- unittest/test_sanity.py\n")
    index = YAMLIndex()
    index.blocks = [Block("l0_cpu", 0, {}, ["unittest/test_sanity.py"])]
    stages = {"CPU-Generic-1": Stage("CPU-Generic-1", "l0_cpu", "x86_64", 1, 1)}
    pr = PRInputs([path], {path: _diff(before, after)})
    return path, pr, index, stages, ImportSanityRule(index, stages, tmp_path)


def test_sanity_does_not_claim_file_and_survives_coverage(tmp_path):
    path, pr, index, stages, rule = _registered_sanity(
        tmp_path,
        "VALUE=1\ndef caller():\n    return VALUE\n",
        "VALUE=2\ndef caller():\n    return VALUE\n",
    )
    contribution = rule.apply(pr)
    assert contribution.scope == "noop"
    assert contribution.handled_files == set()
    assert contribution.import_sanity_files == {path}
    assert Selector(stages).run(pr, [rule]).scope is None
    tier, note = apply_coverage_tier(
        pr, [(rule, contribution)], set(), stages, index, tmp_path, _FakeDB()
    )
    assert tier is not None, note
    assert "CPU-Generic-1" in tier.affected_stages
    assert "unittest/test_sanity.py" not in tier.removed.get(("l0_cpu", 0), set())


@pytest.mark.parametrize("missing", ["manifest", "test", "stage", "registration"])
def test_sanity_unavailable_forces_fallback(tmp_path, missing):
    path, pr, index, stages, rule = _registered_sanity(tmp_path, "X=1\n", "X=2\n")
    if missing == "manifest":
        (tmp_path / "tests/integration/test_lists/import_sanity.yml").unlink()
    elif missing == "test":
        (tmp_path / "tests/unittest/test_sanity.py").unlink()
    elif missing == "stage":
        stages.clear()
    else:
        index.blocks.clear()
    result = rule.apply(pr)
    assert result.scope is None
    assert result.import_sanity_files == {path}


def test_import_sanity_is_independent_of_rule_owned_runtime_files(tmp_path):
    from cbts.rules.base import RuleResult

    path, pr, index, stages, rule = _registered_sanity(tmp_path, "X=1\n", "X=2\n")

    class RuntimeRule:
        name = "runtime"
        handles_docstrings = False

        def apply(self, _):
            return RuleResult({path}, {"CPU-Generic-1"}, "testsonly", "runtime covered")

    result = Selector(stages).run(pr, [RuntimeRule(), rule])
    assert result.scope == "testsonly"
    assert result.sanity_required
    assert result.block_filters[("l0_cpu", 0)]["unittest/test_sanity.py"]


def test_sanity_known_unimpacted_entry_cannot_be_pruned(tmp_path):
    path, pr, index, stages, rule = _registered_sanity(
        tmp_path, "X=1\ndef caller():\n    return X\n", "X=2\ndef caller():\n    return X\n"
    )
    stages = {"A10-PyTorch-1": Stage("A10-PyTorch-1", "l0_cpu", "x86_64", 1, 1)}
    rule.stages = stages
    entries = ["unittest/test_sanity.py", "unittest/test_runtime.py", "unittest/test_unrelated.py"]
    index.blocks[0].tests = entries

    class DB(_FakeDB):
        def known_by_family(self):
            return {
                "A10-PyTorch": {
                    "test_unittests.py::test_unittests_v2[" + entry + "]" for entry in entries
                }
            }

        def tests_touching_func(self, _path, qualname):
            return (
                {"A10-PyTorch-1/test_unittests.py::test_unittests_v2[unittest/test_runtime.py]"}
                if qualname == "caller"
                else set()
            )

    tier, note = apply_coverage_tier(
        pr, [(rule, rule.apply(pr))], set(), stages, index, tmp_path, DB()
    )
    assert tier is not None, note
    assert tier.removed[("l0_cpu", 0)] == {"unittest/test_runtime.py", "unittest/test_unrelated.py"}
    assert "A10-PyTorch-1" not in tier.dropped


def test_missing_sanity_node_does_not_count_as_retained_parent(tmp_path):
    path, pr, index, stages, rule = _registered_sanity(tmp_path, "X=1\n", "X=2\n")
    (tmp_path / "tests/integration/test_lists/import_sanity.yml").write_text(
        "- unittest/test_sanity.py::test_missing\n"
    )
    result = rule.apply(pr)
    assert result.scope is None
    assert "test_missing" in result.reason
