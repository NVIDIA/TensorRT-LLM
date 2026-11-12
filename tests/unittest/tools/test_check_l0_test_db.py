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
"""Diff-scoped checks for L0 stage coverage, untracked skips, and pre/post duplicates."""

import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "jenkins", "scripts", "cbts"))

import check_l0_test_db as cov  # noqa: E402

_GROOVY = """
x86TestConfigs = [
    "A10-PyTorch-1": ["a10", "l0_demo", 1, 1],
    "DGX_H100-PyTorch-1": ["auto:dgx-h100-x1", "l0_demo", 1, 1],
    "DGX_H100-4_GPUs-PyTorch-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],
]
"""

_GROOVY_WITH_NEW = _GROOVY.replace(
    "x86TestConfigs = [",
    'x86TestConfigs = [\n    "B300-PyTorch-1": ["auto:dgx-b300-flex", "l0_new", 1, 1],',
)


def _yaml(tests, gpu="'*'", gte=1, lte=1, stage="pre_merge", stem="l0_demo"):
    body = "\n".join(f"  - {test}" for test in tests)
    return f"""version: 0.0.1
{stem}:
- condition:
    ranges:
      system_gpu_count:
        gte: {gte}
        lte: {lte}
    wildcards:
      gpu:
      - {gpu}
    terms:
      stage: {stage}
      backend: pytorch
  tests:
{body}
"""


def _issues(head_groovy, base_yamls, head_yamls):
    return cov.find_list_issues(head_groovy, base_yamls, head_yamls)


def test_current_repo_lists_have_no_self_diff():
    db = os.path.join(REPO_ROOT, "tests", "integration", "test_lists", "test-db")
    yamls = {}
    for name in os.listdir(db):
        if name.startswith("l0_") and name.endswith((".yml", ".yaml")):
            with open(os.path.join(db, name)) as handle:
                yamls[name] = handle.read()
    groovy = open(os.path.join(REPO_ROOT, "jenkins", "L0_Test.groovy")).read()
    assert cov.find_list_issues(groovy, yamls, yamls) == []


def test_added_entry_on_a_live_stage_passes():
    base = {"l0_demo.yml": _yaml(["unittest/old.py"])}
    head = {"l0_demo.yml": _yaml(["unittest/old.py", "unittest/new.py"])}
    assert _issues(_GROOVY, base, head) == []


def test_added_entry_on_a_commented_out_stage_fails():
    groovy = """
x86TestConfigs = [
    // "A10-PyTorch-1": ["a10", "l0_demo", 1, 1],
]
"""
    head = {"l0_demo.yml": _yaml(["unittest/new.py"], gpu="'*a10*'")}
    issues = _issues(groovy, {}, head)
    assert len(issues) == 1
    assert "unittest/new.py" in issues[0]
    assert "no live L0 stage" in issues[0] or "not selected" in issues[0]


def test_new_yaml_without_a_stage_fails_and_passes_once_wired():
    head = {"l0_new.yml": _yaml(["unittest/new.py"], stem="l0_new")}
    missing = _issues(_GROOVY, {}, head)
    assert any("not selected" in issue for issue in missing)

    wired = _issues(_GROOVY_WITH_NEW, {}, head)
    assert wired == []


def test_empty_new_yaml_without_a_stage_fails():
    head = {"l0_new.yml": "version: 0.0.1\nl0_new: []\n"}
    issues = _issues(_GROOVY, {}, head)
    assert any("new test-db file" in issue for issue in issues)


def test_dropping_or_narrowing_an_existing_entry_is_not_a_liveness_failure():
    # Liveness judges only what the diff adds. Narrowing where an entry runs, a
    # tighter -k filter, and a new --ignore are out of scope.
    base = {"l0_demo.yml": _yaml(["unittest/foo.py", "unittest/keep.py"], gpu="'*'")}
    narrowed_platform = {
        "l0_demo.yml": _yaml(["unittest/foo.py", "unittest/keep.py"], gpu="'*h100*'")
    }
    dropped_entry = {"l0_demo.yml": _yaml(["unittest/keep.py"])}
    tighter_k = {"l0_demo.yml": _yaml(['unittest/foo.py -k "not test_a"', "unittest/keep.py"])}
    ignored = {
        "l0_demo.yml": _yaml(
            ["unittest/foo.py --ignore=unittest/foo.py::test_hidden", "unittest/keep.py"]
        )
    }
    for head in (narrowed_platform, dropped_entry, tighter_k, ignored):
        assert _issues(_GROOVY, base, head) == []


def test_commenting_out_a_stage_is_not_a_liveness_failure_without_a_new_entry():
    head_groovy = _GROOVY.replace(
        '"DGX_H100-4_GPUs-PyTorch-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],',
        '// "DGX_H100-4_GPUs-PyTorch-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],',
    )
    yamls = {"l0_demo.yml": _yaml(["unittest/wide.py"], gpu="'*h100*'", gte=4, lte=4)}
    assert _issues(head_groovy, yamls, yamls) == []


def test_existing_unscheduled_entry_is_not_rejected():
    orphan = {"l0_orphan.yml": _yaml(["unittest/old.py"], stem="l0_orphan")}
    head = {
        "l0_orphan.yml": _yaml(["unittest/old.py"], stem="l0_orphan"),
        "l0_demo.yml": _yaml(["unittest/live.py"]),
    }
    assert _issues(_GROOVY, orphan, head) == []


_SELECTED = """
import pytest
from somewhere import get_sm_version

def test_bar():
    pytest.skip("no bug")
"""

_SM_SKIP = """
import pytest
from somewhere import get_sm_version

@pytest.mark.skipif(get_sm_version() == 90, reason="hopper only")
def test_bar():
    pass
"""


def _skip_issues(source, groovy=_GROOVY, yamls=None, waives="", lines=None):
    if yamls is None:
        yamls = {"l0_demo.yml": _yaml(["unittest/foo"], gpu="'*'")}
    path = "tests/unittest/foo/test_bar.py"
    added = lines if lines is not None else set(range(1, source.count("\n") + 2))
    return cov.find_skip_issues(groovy, yamls, waives, {path: source}, {path: added})


def test_unconditional_skip_of_a_selected_test_fails_until_tracked():
    issues = _skip_issues(_SELECTED)
    assert len(issues) == 1
    assert "test_bar" in issues[0]

    tracked = _SELECTED.replace("no bug", "https://nvbugs/789")
    assert _skip_issues(tracked) == []

    waives = "unittest/foo/test_bar.py::test_bar SKIP (https://nvbugs/789)\n"
    assert _skip_issues(_SELECTED, waives=waives) == []


def test_skipif_fires_only_on_the_platform_whose_sm_matches():
    h100_only = {"l0_demo.yml": _yaml(["unittest/foo"], gpu="'*h100*'")}
    a10_only = {"l0_demo.yml": _yaml(["unittest/foo"], gpu="'*a10*'")}
    assert _skip_issues(_SM_SKIP, yamls=h100_only)
    assert _skip_issues(_SM_SKIP, yamls=a10_only) == []


def _two_gpu_yaml():
    h100 = _yaml(["unittest/foo"], gpu="'*h100*'").split("l0_demo:\n", 1)[1]
    a10 = _yaml(["unittest/foo"], gpu="'*a10*'").split("l0_demo:\n", 1)[1]
    return {"l0_demo.yml": "version: 0.0.1\nl0_demo:\n" + h100 + a10}


def test_any_skipped_test_db_entry_fails_while_another_entry_still_runs():
    issues = _skip_issues(_SM_SKIP, yamls=_two_gpu_yaml())
    assert len(issues) == 1
    assert "unittest/foo" in issues[0]
    assert "dgx_h100" in issues[0]
    assert "a10" not in issues[0]


def test_skipped_entry_is_allowed_when_waives_txt_tracks_it():
    waives = "unittest/foo/test_bar.py::test_bar SKIP (https://nvbugs/789)\n"
    assert _skip_issues(_SM_SKIP, yamls=_two_gpu_yaml(), waives=waives) == []


def test_unevaluable_skip_fails_closed():
    source = """
import pytest

def test_bar():
    if helper():
        pytest.skip("not this gpu")
"""
    issues = _skip_issues(source)
    assert len(issues) == 1
    assert "unevaluated" in issues[0]


def test_skip_outside_the_diff_is_ignored():
    assert _skip_issues(_SELECTED, lines={1}) == []


def test_skip_in_a_file_no_stage_selects_is_ignored():
    source = _SELECTED
    path = "tests/unittest/other/test_bar.py"
    added = set(range(1, 20))
    issues = cov.find_skip_issues(
        _GROOVY,
        {"l0_demo.yml": _yaml(["unittest/foo"])},
        "",
        {path: source},
        {path: added},
    )
    assert issues == []


def test_added_lines_parser_counts_only_new_lines():
    diff = """\
diff --git a/tests/a.py b/tests/a.py
+++ b/tests/a.py
@@ -10,0 +11,2 @@
+pytest.skip("x")
+keep
@@ -20 +22 @@
-old
+new
"""
    assert cov._added_lines(diff) == {"tests/a.py": {11, 12, 22}}


_GROOVY_BOTH = """
x86TestConfigs = [
    "A10-PyTorch-1": ["a10", "l0_demo", 1, 1],
    "A10-PyTorch-Post-Merge-1": ["a10", "l0_demo", 1, 1],
    "DGX_H100-4_GPUs-PyTorch-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],
    "DGX_H100-4_GPUs-PyTorch-Post-Merge-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],
]
"""


def _pre_post_yaml(
    pre_tests,
    post_tests,
    pre_lte=1,
    post_lte=1,
    post_gte=1,
    pre_gpu="'*a10*'",
    post_gpu="'*a10*'",
):
    def _block(stage, tests, gte, lte, gpu):
        body = "\n".join(f"  - {test}" for test in tests)
        return f"""- condition:
    ranges:
      system_gpu_count:
        gte: {gte}
        lte: {lte}
    wildcards:
      gpu:
      - {gpu}
    terms:
      stage: {stage}
      backend: pytorch
  tests:
{body}
"""

    return (
        "version: 0.0.1\nl0_demo:\n"
        + _block("pre_merge", pre_tests, 1, pre_lte, pre_gpu)
        + _block("post_merge", post_tests, post_gte, post_lte, post_gpu)
    )


def _pre_post(base_yamls, head_yamls, base_groovy=_GROOVY_BOTH, head_groovy=_GROOVY_BOTH):
    return cov.find_pre_post_issues(base_groovy, head_groovy, base_yamls, head_yamls)


def test_same_test_on_pre_and_post_for_the_same_mako_fails():
    head = {"l0_demo.yml": _pre_post_yaml(["unittest/shared.py"], ["unittest/shared.py"])}
    issues = _pre_post({}, head)
    assert len(issues) == 1
    assert "unittest/shared.py" in issues[0]
    assert "A10-PyTorch-1" in issues[0]
    assert "A10-PyTorch-Post-Merge-1" in issues[0]


_GROOVY_RAY = """
x86TestConfigs = [
    "A10-PyTorch-1": ["a10", "l0_demo", 1, 1],
    "A10-PyTorch-Post-Merge-1": ["a10", "l0_demo", 1, 1],
    "A10-PyTorch-Ray-1": ["a10", "l0_demo", 1, 1],
    "A10-PyTorch-Ray-Post-Merge-1": ["a10", "l0_demo", 1, 1],
]
"""


def _orch_yaml(pre_tests, post_tests, pre_orch=None, post_orch=None):
    def _block(stage, tests, orchestrator):
        body = "\n".join(f"  - {test}" for test in tests)
        orch = f"      orchestrator: {orchestrator}\n" if orchestrator else ""
        return f"""- condition:
    ranges:
      system_gpu_count:
        gte: 1
        lte: 1
    wildcards:
      gpu:
      - '*a10*'
    terms:
      stage: {stage}
      backend: pytorch
{orch}  tests:
{body}
"""

    return (
        "version: 0.0.1\nl0_demo:\n"
        + _block("pre_merge", pre_tests, pre_orch)
        + _block("post_merge", post_tests, post_orch)
    )


def test_omitted_orchestrator_matches_mpi_and_ray_separately():
    yamls = {"l0_demo.yml": _orch_yaml(["unittest/shared.py"], ["unittest/shared.py"])}
    issues = _pre_post({}, yamls, _GROOVY_RAY, _GROOVY_RAY)
    assert len(issues) == 2
    assert any("orchestrator=mpi" in issue and "A10-PyTorch-Ray" not in issue for issue in issues)
    assert any("orchestrator=ray" in issue for issue in issues)


def test_omitted_orchestrator_is_not_collapsed_to_mpi():
    yamls = {
        "l0_demo.yml": _orch_yaml(["unittest/shared.py"], ["unittest/shared.py"], post_orch="mpi")
    }
    issues = _pre_post({}, yamls, _GROOVY_RAY, _GROOVY_RAY)
    assert len(issues) == 1
    assert "orchestrator=mpi" in issues[0]
    assert "Ray" not in issues[0]


def test_mpi_pre_and_ray_post_are_not_a_duplicate():
    yamls = {
        "l0_demo.yml": _orch_yaml(
            ["unittest/shared.py"], ["unittest/shared.py"], pre_orch="mpi", post_orch="ray"
        )
    }
    assert _pre_post({}, yamls, _GROOVY_RAY, _GROOVY_RAY) == []


def test_adding_the_post_merge_copy_of_an_existing_test_fails():
    base = {"l0_demo.yml": _pre_post_yaml(["unittest/shared.py"], ["unittest/other.py"])}
    head = {
        "l0_demo.yml": _pre_post_yaml(
            ["unittest/shared.py"], ["unittest/shared.py", "unittest/other.py"]
        )
    }
    issues = _pre_post(base, head)
    assert len(issues) == 1
    assert "unittest/shared.py" in issues[0]
    assert "unittest/other.py" not in issues[0]


def test_new_parametrized_case_is_a_duplicate_even_if_the_function_already_is():
    base = {
        "l0_demo.yml": _pre_post_yaml(
            ["unittest/shared.py::test_a[p1]"],
            ["unittest/shared.py::test_a[p1]", "unittest/shared.py::test_a[p2]"],
        )
    }
    head = {
        "l0_demo.yml": _pre_post_yaml(
            ["unittest/shared.py::test_a[p1]", "unittest/shared.py::test_a[p2]"],
            ["unittest/shared.py::test_a[p1]", "unittest/shared.py::test_a[p2]"],
        )
    }
    issues = _pre_post(base, head)
    assert len(issues) == 1
    assert "test_a[p2]" in issues[0]


def test_different_cases_of_one_function_on_pre_and_post_are_not_a_duplicate():
    base = {
        "l0_demo.yml": _pre_post_yaml(
            ["unittest/shared.py::test_a[p1]"], ["unittest/shared.py::test_a[p1]"]
        )
    }
    head = {
        "l0_demo.yml": _pre_post_yaml(
            ["unittest/shared.py::test_a[p1]", "unittest/shared.py::test_a[p3]"],
            ["unittest/shared.py::test_a[p1]", "unittest/shared.py::test_a[p2]"],
        )
    }
    assert _pre_post(base, head) == []


def test_same_test_on_different_gpu_counts_is_not_a_duplicate():
    head = {
        "l0_demo.yml": _pre_post_yaml(
            ["unittest/shared.py"],
            ["unittest/shared.py"],
            post_gte=4,
            post_lte=4,
            post_gpu="'*h100*'",
        )
    }
    assert _pre_post({}, head) == []


def test_pre_post_duplicate_already_on_the_base_is_allowed():
    yamls = {"l0_demo.yml": _pre_post_yaml(["unittest/shared.py"], ["unittest/shared.py"])}
    assert _pre_post(yamls, yamls) == []


def test_existing_duplicate_stays_allowed_when_the_diff_adds_another_test():
    base = {"l0_demo.yml": _pre_post_yaml(["unittest/shared.py"], ["unittest/shared.py"])}
    head = {
        "l0_demo.yml": _pre_post_yaml(
            ["unittest/shared.py", "unittest/new.py"],
            ["unittest/shared.py"],
        )
    }
    assert _pre_post(base, head) == []


def test_current_repo_pre_post_lists_have_no_self_diff():
    db = os.path.join(REPO_ROOT, "tests", "integration", "test_lists", "test-db")
    yamls = {}
    for name in os.listdir(db):
        if name.startswith("l0_") and name.endswith((".yml", ".yaml")):
            with open(os.path.join(db, name)) as handle:
                yamls[name] = handle.read()
    groovy = open(os.path.join(REPO_ROOT, "jenkins", "L0_Test.groovy")).read()
    assert cov.find_pre_post_issues(groovy, groovy, yamls, yamls) == []


# ---------------------------------------------------------------------------
# --untracked-skips: named markers, helpers, unittest skips and count marks
# ---------------------------------------------------------------------------

_INTEGRATION_MARKERS = "tests/integration/defs/conftest.py"
_UNITTEST_MARKERS = "tests/unittest/utils/util.py"

_MARKER_SOURCES = {
    _INTEGRATION_MARKERS: """
import os
import pytest

skip_pre_blackwell = pytest.mark.skipif(get_sm_version() < 100, reason="pre-Blackwell")
skip_no_hopper = pytest.mark.skipif(get_sm_version() != 90, reason="hopper only")
skip_no_sm100f = pytest.mark.skipif(not is_sm_100f(), reason="SM100 family only")
skip_ray = pytest.mark.skipif(os.environ.get("TLLM_DISABLE_MPI") == "1", reason="ray")
skip_arm = pytest.mark.skipif("aarch64" in platform.machine(), reason="arm")


def skip_fp4_pre_blackwell(use_fp4):
    if use_fp4 and (get_sm_version() < 100 or get_sm_version() >= 120):
        pytest.skip("FP4 is not supported on pre-Blackwell architectures")


@pytest.fixture(autouse=True)
def skip_by_device_count(request):
    if request.node.get_closest_marker("skip_less_device"):
        pytest.skip("not enough devices")
""",
    _UNITTEST_MARKERS: """
import unittest
import pytest

skip_pre_blackwell = pytest.mark.skipif(getSMVersion() < 100, reason="pre-Blackwell")
skip_pre_blackwell_unittest = unittest.skipIf(getSMVersion() < 100, reason="pre-Blackwell")
skip_ray = pytest.mark.skipif(mpi_disabled(), reason="ray")


def skip_num_gpus_less_than(num_gpus: int):
    return pytest.mark.skipif(
        torch.cuda.device_count() < num_gpus, reason=f"needs {num_gpus} GPUs")


skip_single_gpu = skip_num_gpus_less_than(2)


def skip_gpu_memory_less_than(required_memory: int):
    memory = get_total_gpu_memory(0)
    return pytest.mark.skipif(required_memory > memory, reason="memory")
""",
}

_GROOVY_SKIPS = """
x86TestConfigs = [
    "A10-PyTorch-1": ["a10", "l0_demo", 1, 1],
    "DGX_H100-PyTorch-1": ["auto:dgx-h100-x1", "l0_demo", 1, 1],
    "DGX_H100-4_GPUs-PyTorch-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],
    "DGX_B200-PyTorch-1": ["auto:dgx-b200-x1", "l0_demo", 1, 1],
]
"""


def _marker_issues(body, gpu, markers=None, waives="", lines=None, gte=1, lte=1, groovy=None):
    """Run --untracked-skips on ``body`` in a unit test file that the test-db selects on ``gpu``."""
    source = "import pytest\nimport unittest\n" + body
    yamls = {"l0_demo.yml": _yaml(["unittest/foo"], gpu=f"'{gpu}'", gte=gte, lte=lte)}
    path = "tests/unittest/foo/test_bar.py"
    added = lines if lines is not None else set(range(1, source.count("\n") + 2))
    return cov.find_skip_issues(
        groovy or _GROOVY_SKIPS,
        yamls,
        waives,
        {path: source},
        {path: added},
        _MARKER_SOURCES if markers is None else markers,
    )


def test_named_marker_decorator_skips_only_on_a_gpu_it_excludes():
    body = "\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    h100 = _marker_issues(body, "*h100*")
    assert len(h100) == 1
    assert "test_bar" in h100[0] and "dgx_h100" in h100[0]
    assert _marker_issues(body, "*b200*") == []


def test_named_marker_in_pytest_param_marks_is_judged():
    body = """
@pytest.mark.parametrize(
    "model", [pytest.param("a", marks=skip_pre_blackwell), pytest.param("b")])
def test_bar(model):
    pass
"""
    assert len(_marker_issues(body, "*h100*")) == 1
    assert _marker_issues(body, "*b200*") == []
    as_tuple = body.replace(
        "marks=skip_pre_blackwell", "marks=(skip_pre_blackwell, skip_no_hopper)"
    )
    assert len(_marker_issues(as_tuple, "*b200*")) == 1  # skip_no_hopper skips on SM 100


def test_named_marker_in_pytestmark_and_on_a_class_is_judged():
    module = "\npytestmark = skip_pre_blackwell\n\ndef test_bar():\n    pass\n"
    issues = _marker_issues(module, "*h100*")
    assert len(issues) == 1 and "module level" in issues[0]
    assert _marker_issues(module, "*b200*") == []

    klass = "\n@skip_pre_blackwell\nclass TestBar:\n    def test_bar(self):\n        pass\n"
    assert len(_marker_issues(klass, "*h100*")) == 1


def test_named_marker_with_a_predicate_condition_is_judged():
    body = "\n@skip_no_sm100f\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(body, "*h100*")) == 1
    assert _marker_issues(body, "*b200*") == []


def test_marker_resolves_from_the_suite_it_is_under():
    # ``skip_pre_blackwell`` is defined by both marker modules; a unit test
    # uses util.py's, an integration test conftest.py's. Only the names
    # differ here, so make one module's definition always skip.
    markers = dict(_MARKER_SOURCES)
    markers[_INTEGRATION_MARKERS] = markers[_INTEGRATION_MARKERS].replace(
        "get_sm_version() < 100", "True", 1
    )
    body = "\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    assert _marker_issues(body, "*b200*", markers=markers) == []
    markers = dict(_MARKER_SOURCES)
    markers[_UNITTEST_MARKERS] = markers[_UNITTEST_MARKERS].replace(
        "getSMVersion() < 100", "True", 1
    )
    assert len(_marker_issues(body, "*b200*", markers=markers)) == 1


def test_name_defined_only_in_the_other_marker_module_still_resolves():
    body = "\n@skip_no_hopper\ndef test_bar():\n    pass\n"
    assert _marker_issues(body, "*h100*") == []
    assert len(_marker_issues(body, "*b200*")) == 1


def test_imperative_marker_helper_binds_its_argument():
    skipped = "\ndef test_bar():\n    skip_fp4_pre_blackwell(True)\n"
    assert len(_marker_issues(skipped, "*h100*")) == 1
    assert _marker_issues(skipped, "*b200*") == []
    off = skipped.replace("(True)", "(False)")
    assert _marker_issues(off, "*h100*") == []
    # An argument the check cannot see fails closed where the SM would skip.
    param = "\ndef test_bar(use_fp4=True):\n    skip_fp4_pre_blackwell(use_fp4)\n"
    issues = _marker_issues(param, "*h100*")
    assert len(issues) == 1 and "unevaluated" in issues[0]
    assert _marker_issues(param, "*b200*") == []


def test_marker_factory_is_judged_against_the_stage_gpu_count():
    body = "\n@skip_num_gpus_less_than(4)\ndef test_bar():\n    pass\n"
    issues = _marker_issues(body, "*h100*", gte=1, lte=4)
    assert len(issues) == 1
    assert "DGX_H100-PyTorch-1" in issues[0] and "4_GPUs" not in issues[0]
    assert _marker_issues(body, "*h100*", gte=4, lte=4) == []

    named = "\n@skip_single_gpu\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(named, "*h100*", gte=1, lte=1)) == 1
    assert _marker_issues(named, "*h100*", gte=4, lte=4) == []


def test_skip_less_device_mark_is_judged_against_the_stage_gpu_count():
    wide = "\n@pytest.mark.skip_less_device(8)\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(wide, "*h100*", gte=4, lte=4)) == 1
    fits = wide.replace("(8)", "(4)")
    assert _marker_issues(fits, "*h100*", gte=4, lte=4) == []
    world = wide.replace("skip_less_device", "skip_less_mpi_world_size")
    assert len(_marker_issues(world, "*h100*", gte=4, lte=4)) == 1


def test_memory_markers_are_not_judged():
    body = """
@pytest.mark.skip_less_device_memory(80000)
def test_one():
    pass

@skip_gpu_memory_less_than(10)
def test_two():
    pass
"""
    assert _marker_issues(body, "*h100*") == []


def test_ray_marker_is_judged_against_the_stage_orchestrator():
    body = "\n@skip_ray\ndef test_bar():\n    pass\n"
    assert _marker_issues(body, "*h100*") == []  # mpi stages: TLLM_DISABLE_MPI is unset
    ray = _GROOVY_SKIPS.replace(
        "x86TestConfigs = [",
        'x86TestConfigs = [\n    "DGX_H100-PyTorch-Ray-1": ["auto:dgx-h100-x1", "l0_demo", 1, 1],',
    )
    issues = _marker_issues(body, "*h100*", groovy=ray)
    assert len(issues) == 1 and "Ray" in issues[0]
    assert _marker_issues(body, "*b200*", groovy=ray) == []


def test_unittest_skips_are_judged():
    decorator = "\n@unittest.skipIf(get_sm_version() < 100, 'old')\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(decorator, "*h100*")) == 1
    assert _marker_issues(decorator, "*b200*") == []
    unless = "\n@unittest.skipUnless(get_sm_version() >= 100, 'old')\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(unless, "*h100*")) == 1
    assert _marker_issues(unless, "*b200*") == []
    named = "\n@skip_pre_blackwell_unittest\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(named, "*h100*")) == 1
    method = "\nclass T(unittest.TestCase):\n    def test_bar(self):\n        self.skipTest('x')\n"
    assert len(_marker_issues(method, "*h100*")) == 1


def test_local_marker_definition_is_judged_once_through_its_use():
    body = (
        "\nskip_here = pytest.mark.skipif(get_sm_version() < 100, reason='old')\n"
        "\n@skip_here\ndef test_bar():\n    pass\n"
    )
    issues = _marker_issues(body, "*h100*")
    assert len(issues) == 1
    assert ":6:" in issues[0]  # the use, not the definition
    assert _marker_issues(body, "*b200*") == []


def test_changed_marker_definition_reaches_its_unchanged_uses():
    body = (
        "\nskip_here = pytest.mark.skipif(get_sm_version() < 100, reason='old')\n"
        "\n@skip_here\ndef test_bar():\n    pass\n"
    )
    definition_only = {4}
    assert len(_marker_issues(body, "*h100*", lines=definition_only)) == 1
    use_only = _marker_issues(body, "*h100*", lines={6})
    assert len(use_only) == 1


def test_local_marker_definition_without_a_use_is_judged_by_its_own_condition():
    body = "\nskip_here = pytest.mark.skipif(get_sm_version() < 100, reason='old')\n"
    assert len(_marker_issues(body, "*h100*")) == 1
    assert _marker_issues(body, "*b200*") == []


def test_local_marker_definition_wins_over_the_shared_one():
    body = (
        "\nskip_pre_blackwell = pytest.mark.skipif(get_sm_version() < 80, reason='old')\n"
        "\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    )
    assert _marker_issues(body, "*h100*") == []
    assert _marker_issues(body, "*a10*") == []


def test_unresolved_skip_name_in_a_mark_position_fails_closed():
    body = "\n@skip_mystery\ndef test_bar():\n    pass\n"
    issues = _marker_issues(body, "*h100*")
    assert len(issues) == 1 and "unevaluated" in issues[0]
    called = "\n@skip_mystery(3)\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(called, "*h100*")) == 1
    in_marks = "\n@pytest.mark.parametrize('x', [pytest.param(1, marks=skip_mystery)])\ndef test_bar(x):\n    pass\n"
    assert len(_marker_issues(in_marks, "*h100*")) == 1
    # A skip_ name outside a mark position is not a skip.
    assert _marker_issues("\nvalue = skip_mystery\n", "*h100*") == []
    assert _marker_issues("\ndef test_bar():\n    skip_mystery(1)\n", "*h100*") == []


def test_named_marker_use_outside_the_diff_is_ignored():
    body = "\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    assert _marker_issues(body, "*h100*", lines={1}) == []


def test_named_marker_is_allowed_with_an_nvbug_comment_or_a_waive():
    commented = "\n# https://nvbugs/123\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    assert _marker_issues(commented, "*h100*") == []
    trailing = "\n@skip_pre_blackwell  # https://nvbugs/123\ndef test_bar():\n    pass\n"
    assert _marker_issues(trailing, "*h100*") == []
    plain = "\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    waives = "unittest/foo/test_bar.py::test_bar SKIP (https://nvbugs/789)\n"
    assert _marker_issues(plain, "*h100*", waives=waives) == []
    reason = (
        "\nskip_here = pytest.mark.skipif(get_sm_version() < 100, reason='https://nvbugs/5')\n"
        "\n@skip_here\ndef test_bar():\n    pass\n"
    )
    assert _marker_issues(reason, "*h100*") == []


def test_unrelated_comment_above_a_skip_does_not_track_it():
    body = "\n# a plain note\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    assert len(_marker_issues(body, "*h100*")) == 1


def test_without_marker_sources_a_named_marker_in_a_mark_position_fails_closed():
    body = "\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    issues = _marker_issues(body, "*b200*", markers={})
    assert len(issues) == 1 and "unevaluated" in issues[0]


def test_every_shared_marker_resolves_from_the_real_marker_modules():
    expected = {
        _INTEGRATION_MARKERS: [
            "skip_pre_ada",
            "skip_pre_hopper",
            "skip_post_hopper",
            "skip_pre_blackwell",
            "skip_post_blackwell",
            "skip_no_rubin",
            "skip_post_blackwell_ultra",
            "skip_pre_rubin",
            "skip_device_contain_gb200",
            "skip_no_nvls",
            "skip_no_hopper",
            "skip_no_mxfp4_swizzle",
            "skip_no_sm120",
            "skip_no_sm100f",
            "skip_arm",
            "skip_x86",
            "skip_nvlink_inactive",
            "skip_ray",
            "skip_fp8_pre_ada",
            "skip_fp4_pre_blackwell",
        ],
        _UNITTEST_MARKERS: [
            "skip_pre_ada",
            "skip_pre_hopper",
            "skip_no_hopper",
            "skip_pre_blackwell",
            "skip_blackwell",
            "skip_blackwell_geforce",
            "skip_pre_rubin",
            "skip_rubin",
            "skip_ray",
            "skip_pre_blackwell_unittest",
            "skip_num_gpus_less_than",
            "skip_single_gpu",
        ],
    }
    for rel, names in expected.items():
        with open(os.path.join(REPO_ROOT, rel)) as handle:
            defs = cov._parse_marker_file(handle.read())
        for name in names:
            assert name in defs, f"{rel}: {name} did not resolve"
    with open(os.path.join(REPO_ROOT, _INTEGRATION_MARKERS)) as handle:
        defs = cov._parse_marker_file(handle.read())

    def _skips(name, sm):
        return cov._eval_condition(defs[name].condition, cov._Platform(frozenset({sm})))

    assert _skips("skip_pre_blackwell", 90) is True
    assert _skips("skip_pre_blackwell", 100) is False
    assert _skips("skip_no_sm100f", 120) is True
    assert _skips("skip_no_sm100f", 103) is False
    assert _skips("skip_no_rubin", 100) is True
    assert _skips("skip_no_rubin", 107) is False
    assert _skips("skip_device_contain_gb200", 100) is None


def test_device_capability_and_cuda_availability_conditions_are_judged():
    body = """
skip_old = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="needs Blackwell",
)

@skip_old
def test_bar():
    pass
"""
    assert len(_marker_issues(body, "*h100*")) == 1
    assert _marker_issues(body, "*b200*") == []


def test_skip_on_one_test_function_does_not_reach_an_entry_for_another():
    source = "import pytest\n\n@pytest.mark.skip(reason='no bug')\ndef test_bar():\n    pass\n"
    path = "tests/unittest/foo/test_bar.py"
    added = set(range(1, source.count("\n") + 2))

    def _run(entry):
        yamls = {"l0_demo.yml": _yaml([entry], gpu="'*'")}
        return cov.find_skip_issues(_GROOVY, yamls, "", {path: source}, {path: added})

    assert _run("unittest/foo/test_bar.py::test_other") == []
    assert _run("unittest/foo/test_bar.py::test_other[param]") == []
    assert len(_run("unittest/foo/test_bar.py::test_bar")) == 1
    assert len(_run("unittest/foo/test_bar.py::TestClass")) == 1
    assert len(_run("unittest/foo/test_bar.py")) == 1


def test_message_lists_a_few_stages_and_counts_the_rest():
    stages = "".join(
        f'    "DGX_H100-PyTorch-{n}": ["auto:dgx-h100-x1", "l0_demo", {n}, 8],\n'
        for n in range(1, 9)
    )
    groovy = f"x86TestConfigs = [\n{stages}]\n"
    body = "\n@skip_pre_blackwell\ndef test_bar():\n    pass\n"
    issues = _marker_issues(body, "*h100*", groovy=groovy)
    assert len(issues) == 1
    assert issues[0].count("(DGX_H100-PyTorch-") == cov._MAX_LISTED_ENTRIES
    assert "3 more stage(s)" in issues[0]
