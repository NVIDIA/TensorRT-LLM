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
"""Tests that s_branch is read from globalVars, never scraped from the job URL.

The Jenkins folder segment in a job URL (/job/LLM/job/<folder>/) names the
folder the job definition lives in, which is "main" even for a release-branch
post-merge build. Recovering a branch from it therefore yields the constant
"main"; the pipeline publishes the real branch in globalVars["build_branch"].
"""

import importlib.util
import json
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "tests" / "integration" / "defs" / "perf" / "perf_regression_utils.py"


def _load_perf_regression_utils():
    """Load the real module without importing the integration-test packages.

    ``defs/__init__.py`` pre-imports ``torch._inductor`` and
    ``open_search_db_utils`` pulls in the OpenSearch client. ``get_job_info``
    reaches neither, and requiring them would turn this into a GPU-image test.
    """
    defs_pkg = types.ModuleType("defs")
    defs_pkg.__path__ = []
    alternative = types.ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    perf_pkg = types.ModuleType("defs.perf")
    perf_pkg.__path__ = []
    db_utils = types.ModuleType("defs.perf.open_search_db_utils")
    for name in ("add_id", "get_history_data", "post_new_perf_data"):
        setattr(db_utils, name, lambda *args, **kwargs: None)

    stubs = {
        "defs": defs_pkg,
        "defs.trt_test_alternative": alternative,
        "defs.perf": perf_pkg,
        "defs.perf.open_search_db_utils": db_utils,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "defs.perf.perf_regression_utils", _MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


get_job_info = _load_perf_regression_utils().get_job_info

# A real post-merge build of release/1.3.0rc22.post1, whose job URL still says
# ".../job/LLM/job/main/job/L0_PostMerge/...".
_JENKINS = "https://prod.blsm.nvidia.com"
_RELEASE_BUILD_JOB_URL = f"{_JENKINS}/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge/2911/"
_GITHUB_PR_JOB_URL = f"{_JENKINS}/job/LLM/job/main/job/L0_MergeRequest_PR/1234/"
_GITLAB_MR_JOB_URL = f"{_JENKINS}/job/LLM/job/main/job/L0_MergeRequest/77/"

_INHERITED_CI_VARS = ("globalVars", "gitlabCommit", "BUILD_ID", "BUILD_URL", "JOB_NAME")


@pytest.fixture(autouse=True)
def _clean_ci_env(monkeypatch):
    """Drop inherited CI variables so each case starts from a known state."""
    for name in _INHERITED_CI_VARS:
        monkeypatch.delenv(name, raising=False)


def _job_info(monkeypatch, global_vars, job_url=_RELEASE_BUILD_JOB_URL):
    parents = [{"url": job_url, "build_number": "2911"}] if job_url else []
    payload = dict(global_vars)
    payload["action_info"] = {"parents": parents}
    monkeypatch.setenv("globalVars", json.dumps(payload))
    return get_job_info()


def test_release_branch_survives_a_main_folder_job_url(monkeypatch):
    """The published branch wins over the "main" in the job URL."""
    info = _job_info(monkeypatch, {"build_branch": "release/1.3.0rc22.post1"})
    assert info["s_branch"] == "release/1.3.0rc22.post1"
    assert info["b_is_post_merge"] is True


def test_github_pr_branch_is_published_verbatim(monkeypatch):
    info = _job_info(
        monkeypatch,
        {"build_branch": "github-pr-18127"},
        job_url=_GITHUB_PR_JOB_URL,
    )
    assert info["s_branch"] == "github-pr-18127"
    assert info["b_is_post_merge"] is False


def test_gitlab_mr_source_branch_is_published_verbatim(monkeypatch):
    """GitLab MR builds carry no github_pr_api_url, so publish a real branch."""
    info = _job_info(
        monkeypatch,
        {"build_branch": "user/my-feature"},
        job_url=_GITLAB_MR_JOB_URL,
    )
    assert info["s_branch"] == "user/my-feature"


def test_main_branch_is_not_special_cased(monkeypatch):
    info = _job_info(monkeypatch, {"build_branch": "main"})
    assert info["s_branch"] == "main"


@pytest.mark.parametrize(
    "global_vars",
    [
        pytest.param({}, id="absent"),
        pytest.param({"build_branch": None}, id="null"),
        pytest.param({"build_branch": ""}, id="empty"),
    ],
)
def test_unpublished_branch_is_empty_never_guessed(monkeypatch, global_vars):
    """A branch the pipeline did not publish yields "", never a guess.

    An empty branch is recoverable from the build's build_info.txt; a wrong one
    silently corrupts history matching, so it must never be invented here.
    """
    assert _job_info(monkeypatch, global_vars)["s_branch"] == ""


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(["main"], id="list"),
        pytest.param({"name": "main"}, id="dict"),
        pytest.param(123, id="number"),
        pytest.param(True, id="bool"),
    ],
)
def test_malformed_branch_is_empty_never_stringified(monkeypatch, value):
    """A non-string build_branch is rejected, not coerced via str()."""
    info = _job_info(monkeypatch, {"build_branch": value})
    assert info["s_branch"] == ""


def test_branch_is_not_recovered_from_the_job_url(monkeypatch):
    """No fallback may re-derive the branch from the folder segment."""
    info = _job_info(monkeypatch, {})
    assert "main" not in info["s_branch"]
    assert info["s_job_url"] == _RELEASE_BUILD_JOB_URL


def test_unparsable_global_vars_does_not_raise(monkeypatch):
    monkeypatch.setenv("globalVars", "{not json")
    info = get_job_info()
    assert info["s_branch"] == ""
    assert info["s_job_url"] == ""
