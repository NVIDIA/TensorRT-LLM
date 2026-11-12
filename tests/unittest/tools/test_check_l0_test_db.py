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


def _issues(base_groovy, head_groovy, base_yamls, head_yamls, base_waives="", head_waives=""):
    return cov.find_list_issues(
        base_groovy, head_groovy, base_yamls, head_yamls, base_waives, head_waives
    )


def test_current_repo_lists_have_no_self_diff():
    db = os.path.join(REPO_ROOT, "tests", "integration", "test_lists", "test-db")
    yamls = {}
    for name in os.listdir(db):
        if name.startswith("l0_") and name.endswith((".yml", ".yaml")):
            with open(os.path.join(db, name)) as handle:
                yamls[name] = handle.read()
    groovy = open(os.path.join(REPO_ROOT, "jenkins", "L0_Test.groovy")).read()
    waives = open(
        os.path.join(REPO_ROOT, "tests", "integration", "test_lists", "waives.txt")
    ).read()
    assert cov.find_list_issues(groovy, groovy, yamls, yamls, waives, waives) == []


def test_added_entry_on_a_live_stage_passes():
    base = {"l0_demo.yml": _yaml(["unittest/old.py"])}
    head = {"l0_demo.yml": _yaml(["unittest/old.py", "unittest/new.py"])}
    assert _issues(_GROOVY, _GROOVY, base, head) == []


def test_added_entry_on_a_commented_out_stage_fails():
    groovy = """
x86TestConfigs = [
    // "A10-PyTorch-1": ["a10", "l0_demo", 1, 1],
]
"""
    head = {"l0_demo.yml": _yaml(["unittest/new.py"], gpu="'*a10*'")}
    issues = _issues(groovy, groovy, {}, head)
    assert len(issues) == 1
    assert "unittest/new.py" in issues[0]
    assert "no live L0 stage" in issues[0] or "not selected" in issues[0]


def test_new_yaml_without_a_stage_fails_and_passes_once_wired():
    head = {"l0_new.yml": _yaml(["unittest/new.py"], stem="l0_new")}
    missing = _issues(_GROOVY, _GROOVY, {}, head)
    assert any("not selected" in issue for issue in missing)

    wired = _issues(_GROOVY, _GROOVY_WITH_NEW, {}, head)
    assert wired == []


def test_empty_new_yaml_without_a_stage_fails():
    head = {"l0_new.yml": "version: 0.0.1\nl0_new: []\n"}
    issues = _issues(_GROOVY, _GROOVY, {}, head)
    assert any("new test-db file" in issue for issue in issues)


def test_removing_a_platform_fails_until_waived():
    base = {"l0_demo.yml": _yaml(["unittest/keep.py"], gpu="'*'")}
    head = {"l0_demo.yml": _yaml(["unittest/keep.py"], gpu="'*h100*'")}
    dropped = _issues(_GROOVY, _GROOVY, base, head)
    assert any("no longer runs on a10" in issue for issue in dropped)

    waives = "unittest/keep.py::test_one SKIP (https://nvbugs/123)\n"
    assert _issues(_GROOVY, _GROOVY, base, head, "", waives) == []


def test_commenting_out_the_only_stage_for_a_gpu_count_fails():
    base_groovy = _GROOVY
    head_groovy = _GROOVY.replace(
        '"DGX_H100-4_GPUs-PyTorch-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],',
        '// "DGX_H100-4_GPUs-PyTorch-1": ["auto:dgx-h100-x4", "l0_demo", 1, 1, 4],',
    )
    yamls = {
        "l0_demo.yml": _yaml(["unittest/wide.py"], gpu="'*h100*'", gte=4, lte=4),
    }
    issues = _issues(base_groovy, head_groovy, yamls, yamls)
    assert any("no longer runs on dgx_h100" in issue for issue in issues)


def test_tighter_k_filter_fails_until_that_name_is_waived():
    base = {"l0_demo.yml": _yaml(['unittest/foo.py -k "not test_a"'])}
    head = {"l0_demo.yml": _yaml(['unittest/foo.py -k "not test_a and not test_b"'])}
    issues = _issues(_GROOVY, _GROOVY, base, head)
    assert any("test_b" in issue for issue in issues)
    assert not any("test_a" in issue and "newly excludes 'test_a'" in issue for issue in issues)

    waives = "unittest/foo.py::test_b SKIP (https://nvbugs/456)\n"
    assert _issues(_GROOVY, _GROOVY, base, head, "", waives) == []


def test_ignore_option_change_fails_closed_without_a_waive():
    base = {"l0_demo.yml": _yaml(["unittest/foo.py"])}
    head = {"l0_demo.yml": _yaml(["unittest/foo.py --ignore=unittest/foo.py::test_hidden"])}
    issues = _issues(_GROOVY, _GROOVY, base, head)
    assert any("cannot diff" in issue for issue in issues)


def test_existing_unscheduled_entry_is_not_rejected():
    orphan = {"l0_orphan.yml": _yaml(["unittest/old.py"], stem="l0_orphan")}
    head = {
        "l0_orphan.yml": _yaml(["unittest/old.py"], stem="l0_orphan"),
        "l0_demo.yml": _yaml(["unittest/live.py"]),
    }
    assert _issues(_GROOVY, _GROOVY, orphan, head) == []


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
