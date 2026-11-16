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
"""Tests for the test-results link that perf_regression_utils records with each perf result.

``s_artifact_url`` names the run's ``test-results/`` folder on the location that the pipeline
publishes test results to, which ``globalVars["llm_artifacts_primary"]`` holds: aws (the default)
in the native browser of artifactory.nvidia.com, which signs people in, or urm directly.
"""

import importlib.util
import json
import pathlib
import sys
import types

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "tests" / "integration" / "defs" / "perf" / "perf_regression_utils.py"


def _load_perf_regression_utils():
    """Load the real module without importing the integration-test packages.

    ``defs/__init__.py`` pre-imports ``torch._inductor`` and ``open_search_db_utils`` pulls in the
    OpenSearch client; ``get_job_info`` reaches neither.
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

_AWS_BASE = (
    "https://artifactory.nvidia.com/ui/native/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main"
)
_URM_BASE = "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main"

# Parent job URLs; only whether one names L0_PostMerge matters to get_job_info.
_PRE_MERGE_JOB_URL = "https://ci.example.com/job/LLM/job/main/job/L0_MergeRequest_PR/1234/"
_POST_MERGE_JOB_URL = "https://ci.example.com/job/LLM/job/main/job/L0_PostMerge/3031/"

_INHERITED_CI_VARS = ("globalVars", "gitlabCommit", "BUILD_ID", "BUILD_URL", "JOB_NAME")


@pytest.fixture(autouse=True)
def _clean_ci_env(monkeypatch):
    """Drop inherited CI variables so each case starts from a known state."""
    for name in _INHERITED_CI_VARS:
        monkeypatch.delenv(name, raising=False)


def _job_info(monkeypatch, primary, job_url, build_number):
    global_vars = {"action_info": {"parents": [{"url": job_url, "build_number": build_number}]}}
    if primary is not None:
        global_vars["llm_artifacts_primary"] = primary
    monkeypatch.setenv("globalVars", json.dumps(global_vars))
    return get_job_info()


@pytest.mark.parametrize(
    ("primary", "base"),
    [("aws", _AWS_BASE), ("urm", _URM_BASE), (None, _AWS_BASE), ("", _AWS_BASE)],
    ids=["aws", "urm", "unset", "empty"],
)
@pytest.mark.parametrize(
    ("job_url", "job_name", "build_number"),
    [
        (_PRE_MERGE_JOB_URL, "L0_MergeRequest_PR", "1234"),
        (_POST_MERGE_JOB_URL, "L0_PostMerge", "3031"),
    ],
    ids=["pre-merge", "post-merge"],
)
def test_artifact_url_names_the_runs_test_results_folder(
    monkeypatch, primary, base, job_url, job_name, build_number
):
    info = _job_info(monkeypatch, primary, job_url, build_number)

    assert info["s_artifact_url"] == f"{base}/{job_name}/{build_number}/test-results/"


@pytest.mark.parametrize("primary", ["aws", "urm", None])
@pytest.mark.parametrize("job_url", [_PRE_MERGE_JOB_URL, _POST_MERGE_JOB_URL])
def test_artifact_url_is_empty_without_a_job_id(monkeypatch, primary, job_url):
    info = _job_info(monkeypatch, primary, job_url, "")

    assert info["s_artifact_url"] == ""
