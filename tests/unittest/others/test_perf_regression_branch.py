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
"""Tests for how perf_regression_utils handles s_branch.

Two concerns, both of which fail open (a green run with no regression check) if
they break:

1. s_branch is read from globalVars, never scraped from the job URL. The Jenkins
   folder segment in a job URL (/job/LLM/job/<folder>/) names the folder the job
   definition lives in, which is "main" even for a release-branch post-merge
   build. Recovering a branch from it therefore yields the constant "main"; the
   pipeline publishes the real branch in globalVars["build_branch"].

2. A pre-merge run looks its history up against a baseline branch. s_branch is
   part of the case identity, and get_history_data only ever returns post-merge
   records, so a pre-merge run querying its own "github-pr-<N>" branch matches
   nothing and its regression check silently becomes a no-op. The queries must
   see the baseline branch while the uploaded document keeps the real one.
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
    ``open_search_db_utils`` pulls in the OpenSearch client. Neither
    ``get_job_info`` nor the branch routing in
    ``process_and_upload_test_results`` reaches them, and requiring them would
    turn this into a GPU-image test.
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


_perf_regression_utils = _load_perf_regression_utils()
get_job_info = _perf_regression_utils.get_job_info

# A real post-merge build of release/1.3.0rc22.post1, whose job URL still says
# ".../job/LLM/job/main/job/L0_PostMerge/...".
_JENKINS = "https://prod.blsm.nvidia.com"
_RELEASE_BUILD_JOB_URL = f"{_JENKINS}/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge/2911/"
_GITHUB_PR_JOB_URL = f"{_JENKINS}/job/LLM/job/main/job/L0_MergeRequest_PR/1234/"
_GITLAB_MR_JOB_URL = f"{_JENKINS}/job/LLM/job/main/job/L0_MergeRequest/77/"

_INHERITED_CI_VARS = (
    "globalVars",
    "gitlabCommit",
    "BUILD_ID",
    "BUILD_URL",
    "JOB_NAME",
    "PERF_BASELINE_BRANCH",
)


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


# --------------------------------------------------------------------------- #
# Pre-merge history lookup routes through a baseline branch
# --------------------------------------------------------------------------- #

_MATCH_KEYS = ["s_test_case_name", "s_gpu_type", "s_runtime", "s_branch"]


def _new_data_dict():
    return {
        0: {
            "s_test_case_name": "example_model_fp8_tp8-con32_iter10_1k1k",
            "s_gpu_type": "b200",
            "s_runtime": "aggr_server",
            "d_output_token_throughput": 1234.5,
        }
    }


def _run_pipeline(monkeypatch, build_branch, job_url, match_keys=None):
    """Run the real pipeline, recording the s_branch each seam observes.

    Only the three OpenSearch seams are replaced. Everything between them --
    get_job_info, the enrichment loop, the branch routing, the regression pass --
    is the production code path, so a regression in the wiring shows up here.
    """
    observed = {}

    def fake_get_common_values(data_dict, keys):
        observed["common_values"] = {idx: d["s_branch"] for idx, d in data_dict.items()}
        return {}

    def fake_get_history_data(data_dict, keys, common_values_dict):
        observed["history_query"] = {idx: d["s_branch"] for idx, d in data_dict.items()}
        observed["history_query_names"] = {
            idx: d["s_test_case_name"] for idx, d in data_dict.items()
        }
        return {}, {}, {}

    def fake_post_new_perf_data(data_dict):
        observed["uploaded"] = {idx: d["s_branch"] for idx, d in data_dict.items()}

    monkeypatch.setattr(_perf_regression_utils, "get_common_values", fake_get_common_values)
    monkeypatch.setattr(_perf_regression_utils, "get_history_data", fake_get_history_data)
    monkeypatch.setattr(_perf_regression_utils, "post_new_perf_data", fake_post_new_perf_data)

    payload = {
        "build_branch": build_branch,
        "action_info": {"parents": [{"url": job_url, "build_number": "7"}]},
    }
    monkeypatch.setenv("globalVars", json.dumps(payload))

    new_data_dict = _new_data_dict()
    _perf_regression_utils.process_and_upload_test_results(
        new_data_dict,
        match_keys if match_keys is not None else _MATCH_KEYS,
        maximize_metrics=["d_output_token_throughput"],
        minimize_metrics=[],
        regression_metrics=["d_output_token_throughput"],
    )
    observed["new_data_dict"] = {idx: d["s_branch"] for idx, d in new_data_dict.items()}
    return observed


def test_pre_merge_history_is_queried_against_the_baseline_branch(monkeypatch):
    """The queries see "main"; the uploaded document keeps "github-pr-<N>".

    get_common_values must see the substituted branch too: it folds
    single-valued match keys into the OpenSearch must-clause, so substituting
    only for get_history_data would let the query filter the history away.
    """
    observed = _run_pipeline(monkeypatch, "github-pr-18408", _GITHUB_PR_JOB_URL)

    assert observed["common_values"] == {0: "main"}
    assert observed["history_query"] == {0: "main"}
    assert observed["uploaded"] == {0: "github-pr-18408"}
    assert observed["new_data_dict"] == {0: "github-pr-18408"}


def test_pre_merge_baseline_branch_is_configurable(monkeypatch):
    monkeypatch.setenv("PERF_BASELINE_BRANCH", "release/1.3.0rc22.post1")
    observed = _run_pipeline(monkeypatch, "github-pr-18408", _GITHUB_PR_JOB_URL)

    assert observed["history_query"] == {0: "release/1.3.0rc22.post1"}
    assert observed["uploaded"] == {0: "github-pr-18408"}


def test_substitution_only_replaces_the_branch(monkeypatch):
    """The lookup copy is the real data with one field changed, not a stub."""
    observed = _run_pipeline(monkeypatch, "github-pr-18408", _GITHUB_PR_JOB_URL)

    assert observed["history_query_names"] == {0: "example_model_fp8_tp8-con32_iter10_1k1k"}


def test_post_merge_history_is_queried_against_its_own_branch(monkeypatch):
    """A release-branch post-merge run must not be rebased onto main."""
    observed = _run_pipeline(monkeypatch, "release/1.3.0rc22.post1", _RELEASE_BUILD_JOB_URL)

    assert observed["common_values"] == {0: "release/1.3.0rc22.post1"}
    assert observed["history_query"] == {0: "release/1.3.0rc22.post1"}
    assert observed["uploaded"] == {0: "release/1.3.0rc22.post1"}


def test_post_merge_ignores_the_baseline_branch_override(monkeypatch):
    monkeypatch.setenv("PERF_BASELINE_BRANCH", "main")
    observed = _run_pipeline(monkeypatch, "release/1.3.0rc22.post1", _RELEASE_BUILD_JOB_URL)

    assert observed["history_query"] == {0: "release/1.3.0rc22.post1"}


def test_no_substitution_when_branch_is_not_a_match_key(monkeypatch):
    """Callers keying on other fields keep their own semantics untouched."""
    observed = _run_pipeline(
        monkeypatch,
        "github-pr-18408",
        _GITHUB_PR_JOB_URL,
        match_keys=["s_test_case_name", "s_gpu_type"],
    )

    assert observed["history_query"] == {0: "github-pr-18408"}
    assert observed["uploaded"] == {0: "github-pr-18408"}
