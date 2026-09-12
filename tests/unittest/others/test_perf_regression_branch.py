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
"""Tests for perf_regression_utils history routing and pre-merge gating.

Three concerns. The first two fail open (a green run with no regression check)
if they break; the third fails closed (every PR blocked by a regression it did
not introduce):

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

3. A pre-merge regression is exempt when the latest post-merge record for the
   same case already misses the same gate. Both pipelines compare against one
   shared baseline, so a regression landed on main makes every subsequent PR
   measure the same regressed value and fail. The exemption re-evaluates main's
   own latest value against that baseline at the pre-merge threshold -- not the
   b_is_regression the post-merge run recorded at its own tighter threshold --
   so it is exactly as wide as the failure it prevents. Per test case; the
   pre-merge document still uploads b_is_regression as measured, so nothing is
   hidden from the DB.
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


_METRIC = "d_output_token_throughput"

# A baseline high enough that _REGRESSED_VALUE falls below the 10% pre-merge
# threshold and _CLEAN_VALUE stays above it.
_BASELINE = 1000.0
_REGRESSED_VALUE = 800.0
_CLEAN_VALUE = 950.0

# Tier A: baseline supplied directly, so no timestamps or percentile maths are
# involved in deciding whether a case is regressive. Cover several cmd_idx so a
# multi-case test resolves a baseline for every one of them; a case without a
# baseline is silently skipped rather than judged.
_TIER_A_BASELINE = {idx: {"d_baseline_output_token_throughput": _BASELINE} for idx in range(4)}


def _new_data_dict(value: float = 1234.5, count: int = 1) -> dict[int, dict[str, object]]:
    return {
        idx: {
            "s_test_case_name": f"example_model_fp8_tp8-con32_iter10_1k1k_{idx}"
            if count > 1
            else "example_model_fp8_tp8-con32_iter10_1k1k",
            "s_gpu_type": "b200",
            "s_runtime": "aggr_server",
            _METRIC: value,
        }
        for idx in range(count)
    }


def _run_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    build_branch: str,
    job_url: str,
    match_keys: list[str] | None = None,
    history: tuple[object, object, object] | None = None,
    value: float = 1234.5,
    count: int = 1,
    expect_error: bool = False,
    fail_on_regression: bool | None = None,
    minimize: bool = False,
) -> dict[str, dict[int, str]]:
    """Run the real pipeline, recording what each seam observes.

    Only the three OpenSearch seams are replaced. Everything between them --
    get_job_info, the enrichment loop, the branch routing, the regression pass --
    is the production code path, so a regression in the wiring shows up here.

    ``history`` is the (latest, baseline_threshold, history) triple the stubbed
    get_history_data returns, letting a test drive the exemption decision.
    ``expect_error`` wraps the call in pytest.raises(RuntimeError) and records
    the message; it is never a blanket except, because post_new_perf_data raises
    RuntimeError too and swallowing that would make these tests vacuous.
    """
    observed: dict[str, dict[int, str]] = {}

    def fake_get_common_values(
        data_dict: dict[int, dict[str, object]], keys: list[str]
    ) -> dict[str, object]:
        observed["common_values"] = {idx: d["s_branch"] for idx, d in data_dict.items()}
        return {}

    def fake_get_history_data(
        data_dict: dict[int, dict[str, object]],
        keys: list[str],
        common_values_dict: dict[str, object],
    ) -> tuple[dict[int, object], dict[int, object], dict[int, object]]:
        observed["history_query"] = {idx: d["s_branch"] for idx, d in data_dict.items()}
        observed["history_query_names"] = {
            idx: d["s_test_case_name"] for idx, d in data_dict.items()
        }
        return ({}, {}, {}) if history is None else history

    def fake_post_new_perf_data(data_dict: dict[int, dict[str, object]]) -> None:
        observed["uploaded"] = {idx: d["s_branch"] for idx, d in data_dict.items()}
        # Snapshot whole documents: the upload (step 8) runs before the
        # regression check (step 9), so this is the document as posted even when
        # the check goes on to raise.
        observed["uploaded_docs"] = {idx: dict(d) for idx, d in data_dict.items()}

    monkeypatch.setattr(_perf_regression_utils, "get_common_values", fake_get_common_values)
    monkeypatch.setattr(_perf_regression_utils, "get_history_data", fake_get_history_data)
    monkeypatch.setattr(_perf_regression_utils, "post_new_perf_data", fake_post_new_perf_data)

    payload = {
        "build_branch": build_branch,
        "action_info": {"parents": [{"url": job_url, "build_number": "7"}]},
    }
    monkeypatch.setenv("globalVars", json.dumps(payload))

    new_data_dict = _new_data_dict(value=value, count=count)

    def _call() -> None:
        _perf_regression_utils.process_and_upload_test_results(
            new_data_dict,
            match_keys if match_keys is not None else _MATCH_KEYS,
            maximize_metrics=[] if minimize else [_METRIC],
            minimize_metrics=[_METRIC] if minimize else [],
            regression_metrics=[_METRIC],
            fail_on_regression=fail_on_regression,
        )

    observed["error"] = None
    if expect_error:
        with pytest.raises(RuntimeError) as excinfo:
            _call()
        observed["error"] = str(excinfo.value)
    else:
        _call()

    observed["new_data_dict"] = {idx: d["s_branch"] for idx, d in new_data_dict.items()}
    observed["final_docs"] = {idx: dict(d) for idx, d in new_data_dict.items()}
    return observed


def test_pre_merge_history_is_queried_against_the_baseline_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_pre_merge_baseline_branch_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERF_BASELINE_BRANCH", "release/1.3.0rc22.post1")
    observed = _run_pipeline(monkeypatch, "github-pr-18408", _GITHUB_PR_JOB_URL)

    assert observed["history_query"] == {0: "release/1.3.0rc22.post1"}
    assert observed["uploaded"] == {0: "github-pr-18408"}


def test_substitution_only_replaces_the_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lookup copy is the real data with one field changed, not a stub."""
    observed = _run_pipeline(monkeypatch, "github-pr-18408", _GITHUB_PR_JOB_URL)

    assert observed["history_query_names"] == {0: "example_model_fp8_tp8-con32_iter10_1k1k"}


def test_post_merge_history_is_queried_against_its_own_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release-branch post-merge run must not be rebased onto main."""
    observed = _run_pipeline(monkeypatch, "release/1.3.0rc22.post1", _RELEASE_BUILD_JOB_URL)

    assert observed["common_values"] == {0: "release/1.3.0rc22.post1"}
    assert observed["history_query"] == {0: "release/1.3.0rc22.post1"}
    assert observed["uploaded"] == {0: "release/1.3.0rc22.post1"}


def test_post_merge_ignores_the_baseline_branch_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERF_BASELINE_BRANCH", "main")
    observed = _run_pipeline(monkeypatch, "release/1.3.0rc22.post1", _RELEASE_BUILD_JOB_URL)

    assert observed["history_query"] == {0: "release/1.3.0rc22.post1"}


def test_no_substitution_when_branch_is_not_a_match_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers keying on other fields keep their own semantics untouched."""
    observed = _run_pipeline(
        monkeypatch,
        "github-pr-18408",
        _GITHUB_PR_JOB_URL,
        match_keys=["s_test_case_name", "s_gpu_type"],
    )

    assert observed["history_query"] == {0: "github-pr-18408"}
    assert observed["uploaded"] == {0: "github-pr-18408"}


# --------------------------------------------------------------------------- #
# A pre-merge regression fails the stage (positive controls)
#
# These hold both before and after the exemption exists. Without them, every
# "does not raise" assertion below could pass because nothing ever raises.
# --------------------------------------------------------------------------- #


def _history(latest: object, baseline: object = None, history: object = None):
    """Build the get_history_data triple, defaulting to the Tier A baseline."""
    return (
        latest,
        _TIER_A_BASELINE if baseline is None else baseline,
        {} if history is None else history,
    )


def _pre_merge(monkeypatch: pytest.MonkeyPatch, **kwargs):
    return _run_pipeline(monkeypatch, "github-pr-18408", _GITHUB_PR_JOB_URL, **kwargs)


# What the latest post-merge record for case 0 measured. The exemption
# re-evaluates these against the same baseline at the same pre-merge threshold,
# so what matters is the value, not any verdict the post-merge run recorded.
_CLEAN_LATEST = {0: {_METRIC: _BASELINE}}  # main on baseline
_REGRESSED_LATEST = {0: {_METRIC: _REGRESSED_VALUE}}  # 20% down: misses the gate
_WITHIN_GATE_LATEST = {0: {_METRIC: _CLEAN_VALUE}}  # 5% down: still makes the gate

# The opening words of the sentence prepare_regressive_test_cases appends to
# s_regression_info when it exempts a case. Asserted both present and absent
# below, so a reworded annotation fails the positive test rather than quietly
# making every "unannotated" assertion vacuous.
_EXEMPTION_NOTE = "Not failing this stage"


def test_pre_merge_regression_fails_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseline case: main is healthy, the PR regresses, the stage fails."""
    observed = _pre_merge(
        monkeypatch,
        history=_history(_CLEAN_LATEST),
        value=_REGRESSED_VALUE,
        expect_error=True,
    )

    assert _METRIC in observed["error"]
    assert observed["error"].strip()
    assert observed["uploaded_docs"][0]["b_is_regression"] is True


def test_pre_merge_within_threshold_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _pre_merge(monkeypatch, history=_history(_CLEAN_LATEST), value=_CLEAN_VALUE)

    assert observed["uploaded_docs"][0]["b_is_regression"] is False


def test_pre_merge_exactly_at_threshold_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression is a strict inequality: value == baseline*(1-t) is not one."""
    observed = _pre_merge(monkeypatch, history=_history(_CLEAN_LATEST), value=_BASELINE * 0.9)

    assert observed["uploaded_docs"][0]["b_is_regression"] is False


# --------------------------------------------------------------------------- #
# A regressed main exempts the pre-merge gate
# --------------------------------------------------------------------------- #


def test_regressed_main_exempts_the_pre_merge_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: main already misses the gate, so the PR is not blamed."""
    observed = _pre_merge(
        monkeypatch,
        history=_history(_REGRESSED_LATEST),
        value=_REGRESSED_VALUE,
    )

    # No exception, and specifically not RuntimeError("") from an empty message.
    assert observed["error"] is None
    # The measurement is still recorded truthfully -- nothing is hidden.
    assert observed["uploaded_docs"][0]["b_is_regression"] is True
    assert _METRIC in observed["uploaded_docs"][0]["s_regression_info"]
    assert _EXEMPTION_NOTE in observed["uploaded_docs"][0]["s_regression_info"]


def test_main_within_the_gate_still_fails_the_pre_merge_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule that separates this from a post-merge-threshold exemption.

    main is down 5% -- enough for the post-merge run to have recorded
    b_is_regression, but not enough to fail the 10% pre-merge gate. A PR
    reproducing main's value would therefore never have been blocked, so there
    is nothing to exempt: a PR that goes on to regress 20% must still fail.
    """
    observed = _pre_merge(
        monkeypatch,
        history=_history(_WITHIN_GATE_LATEST),
        value=_REGRESSED_VALUE,
        expect_error=True,
    )

    assert _METRIC in observed["error"]


def test_a_numeric_string_from_opensearch_still_exempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History values arrive untrusted; a numeric string is still a number."""
    observed = _pre_merge(
        monkeypatch,
        history=_history({0: {_METRIC: str(_REGRESSED_VALUE)}}),
        value=_REGRESSED_VALUE,
    )

    assert observed["error"] is None


@pytest.mark.parametrize(
    "latest, expect_error",
    [
        # A minimize metric regresses upward, so main must be ABOVE baseline by
        # more than the threshold to exempt. 1200 is +20%, 1050 only +5%.
        pytest.param({0: {_METRIC: 1200.0}}, False, id="main-above-the-gate-exempt"),
        pytest.param({0: {_METRIC: 1050.0}}, True, id="main-within-the-gate-fails"),
        pytest.param({0: {_METRIC: 800.0}}, True, id="main-better-than-baseline-fails"),
    ],
)
def test_the_exemption_respects_a_minimize_metric(
    monkeypatch: pytest.MonkeyPatch, latest: object, expect_error: bool
) -> None:
    """A latency metric regresses upward; the direction must not be inverted.

    The 800 row is the one that catches an inverted comparison: for a minimize
    metric that is main comfortably BETTER than baseline, which must never exempt.
    """
    observed = _pre_merge(
        monkeypatch,
        history=_history(latest),
        value=1200.0,
        minimize=True,
        expect_error=expect_error,
    )

    assert observed["uploaded_docs"][0]["b_is_regression"] is True


def test_exemption_is_per_test_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case 0's exemption must not cover case 1, and vice versa."""
    observed = _pre_merge(
        monkeypatch,
        history=_history({0: {_METRIC: _REGRESSED_VALUE}, 1: {_METRIC: _BASELINE}}),
        value=_REGRESSED_VALUE,
        count=2,
        expect_error=True,
    )

    assert "example_model_fp8_tp8-con32_iter10_1k1k_1" in observed["error"]
    assert "example_model_fp8_tp8-con32_iter10_1k1k_0" not in observed["error"]
    # Both are still recorded as regressive; only the gate differs.
    assert observed["uploaded_docs"][0]["b_is_regression"] is True
    assert observed["uploaded_docs"][1]["b_is_regression"] is True


_TIER_B_HISTORY = {
    0: [
        {"@timestamp": "2026-09-01T00:00:00Z", _METRIC: _BASELINE},
        {"@timestamp": "2026-09-02T00:00:00Z", _METRIC: _BASELINE},
        {"@timestamp": "2026-09-03T00:00:00Z", _METRIC: _BASELINE},
    ]
}


@pytest.mark.parametrize(
    "latest, expect_error",
    [
        pytest.param(_REGRESSED_LATEST, False, id="regressed-main-exempt"),
        pytest.param(_CLEAN_LATEST, True, id="clean-main-fails"),
    ],
)
def test_exemption_holds_on_the_rolling_baseline_path(
    monkeypatch: pytest.MonkeyPatch, latest: object, expect_error: bool
) -> None:
    """Nothing in-repo seeds d_baseline_*, so production uses this path.

    The paired clean-main case proves the Tier B fixture really does produce a
    regression, rather than the exemption passing because no baseline resolved.
    """
    observed = _pre_merge(
        monkeypatch,
        history=(latest, {0: None}, _TIER_B_HISTORY),
        value=_REGRESSED_VALUE,
        expect_error=expect_error,
    )

    assert observed["uploaded_docs"][0]["b_is_regression"] is True


# --------------------------------------------------------------------------- #
# The exemption must not fire
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "latest",
    [
        pytest.param({0: None}, id="null-record"),
        pytest.param({}, id="no-record-for-this-case"),
        pytest.param({0: {}}, id="metric-absent"),
        pytest.param({0: {_METRIC: _BASELINE}}, id="on-baseline"),
        pytest.param({0: {_METRIC: _CLEAN_VALUE}}, id="regressed-but-within-the-gate"),
        # Strict inequality, matching how the pre-merge verdict itself is taken.
        pytest.param({0: {_METRIC: _BASELINE * 0.9}}, id="exactly-at-the-threshold"),
        # A broken post-merge record must leave the gate armed, not disarm it.
        pytest.param({0: {_METRIC: 0}}, id="zero"),
        pytest.param({0: {_METRIC: -5.0}}, id="negative"),
        pytest.param({0: {_METRIC: None}}, id="null-value"),
        pytest.param({0: {_METRIC: "not-a-number"}}, id="non-numeric"),
        pytest.param({0: {_METRIC: []}}, id="empty-list"),
        # The recorded verdict is no longer consulted: it was taken at the
        # post-merge threshold, and with no value there is nothing to re-evaluate.
        pytest.param({0: {"b_is_regression": True}}, id="verdict-without-a-value"),
    ],
)
def test_only_a_value_missing_the_same_gate_exempts(
    monkeypatch: pytest.MonkeyPatch, latest: object
) -> None:
    """Anything else about the latest post-merge record still fails the stage."""
    observed = _pre_merge(
        monkeypatch, history=_history(latest), value=_REGRESSED_VALUE, expect_error=True
    )

    assert _METRIC in observed["error"]


def test_an_older_regressed_record_does_not_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rule reads the latest post-merge record, not any record."""
    observed = _pre_merge(
        monkeypatch,
        history=_history(
            _CLEAN_LATEST,
            history={
                0: [
                    {"@timestamp": "2026-08-01T00:00:00Z", _METRIC: _REGRESSED_VALUE},
                    {"@timestamp": "2026-08-02T00:00:00Z", _METRIC: _REGRESSED_VALUE},
                ]
            },
        ),
        value=_REGRESSED_VALUE,
        expect_error=True,
    )

    assert _METRIC in observed["error"]


# --------------------------------------------------------------------------- #
# Everything else is untouched
# --------------------------------------------------------------------------- #


def test_post_merge_is_unaffected_and_unannotated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Post-merge already only warns; it must not gain an exemption note."""
    observed = _run_pipeline(
        monkeypatch,
        "main",
        _RELEASE_BUILD_JOB_URL,
        history=_history(_REGRESSED_LATEST),
        value=_REGRESSED_VALUE,
    )

    assert observed["error"] is None
    assert observed["uploaded_docs"][0]["b_is_regression"] is True
    assert _EXEMPTION_NOTE not in observed["uploaded_docs"][0]["s_regression_info"]


def test_functional_only_stage_is_unannotated(monkeypatch: pytest.MonkeyPatch) -> None:
    """fail_on_regression=False already passes; no note belongs on a healthy main."""
    observed = _pre_merge(
        monkeypatch,
        history=_history(_CLEAN_LATEST),
        value=_REGRESSED_VALUE,
        fail_on_regression=False,
    )

    assert observed["error"] is None
    assert observed["uploaded_docs"][0]["b_is_regression"] is True
    assert _EXEMPTION_NOTE not in observed["uploaded_docs"][0]["s_regression_info"]


def test_history_query_failure_still_skips_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None triple means the query failed: soft-fail, never a TypeError."""
    observed = _pre_merge(monkeypatch, history=(None, None, None), value=_REGRESSED_VALUE)

    assert observed["error"] is None
    assert "s_regression_info" not in observed["uploaded_docs"][0]


def test_exemption_adds_no_field_to_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exemption lives outside new_data_dict, so no new OpenSearch field.

    post_new_perf_data uploads each record dict wholesale, so any key added to
    it would silently become a new field in the index.
    """
    exempt = _pre_merge(monkeypatch, history=_history(_REGRESSED_LATEST), value=_REGRESSED_VALUE)
    clean = _pre_merge(monkeypatch, history=_history(_CLEAN_LATEST), value=_CLEAN_VALUE)

    assert set(exempt["uploaded_docs"][0]) == set(clean["uploaded_docs"][0])
