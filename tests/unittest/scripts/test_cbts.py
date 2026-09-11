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
"""Tests for CBTS infrastructure and coverage-based test selection."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType, SimpleNamespace, TracebackType
from typing import NoReturn, TypeAlias, Union
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CBTS_ROOT = REPO_ROOT / "jenkins/scripts/cbts"
_ARTIFACT_PATH = CBTS_ROOT / "coverage_selection/artifact.py"
_MAIN_PATH = CBTS_ROOT / "main.py"
sys.path.insert(0, str(CBTS_ROOT))
sys.path.insert(0, str(CBTS_ROOT / "coverage_selection"))
sys.path.insert(0, str(CBTS_ROOT / "coverage_utils"))

from blocks import Stage, YAMLIndex  # noqa: E402
from compact_db import write_leaf_database  # noqa: E402
from python_change_analysis import analyze_python_changes  # noqa: E402
from repository_reference import RepositoryReferenceIndex  # noqa: E402
from rules._helpers import iter_diff_deleted_post_lines, iter_diff_post_line_numbers  # noqa: E402
from rules.base import PRInputs  # noqa: E402
from rules.tests_def_rule import (  # noqa: E402
    ACCURACY_DIR,
    ACCURACY_REFS_PREFIX,
    _direct_test_importers,
    _py_class_scopes_from_deletions,
    _scope_start_line,
    _yaml_top_keys_from_deletions,
)
from rules.tests_def_rule import TestsDefRule as CbtsTestsDefRule  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _load_artifact() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cbts_coverage_artifact", _ARTIFACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_ARTIFACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_main() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cbts_main", _MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


artifact = _load_artifact()
cbts_main = _load_main()


class CoverageArtifactTest(unittest.TestCase):
    def test_selects_closest_complete_ancestor_pair(self) -> None:
        commits = {104: "newer", 102: "older-three", 101: "older-one"}

        def exists(url: str) -> bool:
            if "/103/" in url:
                return url.endswith("cbts_pystart_report_x86_64.tar.gz")
            return "/100/" not in url

        relations = {
            "newer": (1, "behind"),
            "older-three": (3, "ahead"),
            "older-one": (1, "ahead"),
        }
        with (
            mock.patch.object(artifact, "latest_build_number", return_value=104),
            mock.patch.object(artifact, "_exists", side_effect=exists),
            mock.patch.object(
                artifact, "build_commit", side_effect=lambda build, _base: commits[build]
            ),
            mock.patch.object(
                artifact, "drift", side_effect=lambda commit, _base: relations[commit]
            ),
            mock.patch.object(artifact, "compare_distance", return_value=7) as lag,
        ):
            selected = artifact.select_tarball(
                "pr-base", artifact_base="coverage", jenkins_base="jenkins", max_probe=5
            )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["build"], 101)
        self.assertEqual(selected["commit"], "older-one")
        self.assertEqual(selected["drift"], 1)
        self.assertEqual(selected["drift_status"], "ahead")
        self.assertEqual(
            [url.rsplit("/", 1)[-1] for url in selected["urls"]],
            list(artifact.ARCH_TARBALL_NAMES),
        )
        lag.assert_called_once_with("older-one")

    def test_accepts_artifact_collected_at_pr_base(self) -> None:
        with (
            mock.patch.object(artifact, "latest_build_number", return_value=7),
            mock.patch.object(artifact, "_exists", return_value=True),
            mock.patch.object(artifact, "build_commit", return_value="pr-base"),
            mock.patch.object(artifact, "drift", return_value=(0, "identical")),
            mock.patch.object(artifact, "compare_distance", return_value=4),
        ):
            selected = artifact.select_tarball("pr-base", max_probe=1)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["drift"], 0)
        self.assertEqual(selected["drift_status"], "identical")

    def test_prepare_merges_x86_and_sbsa_databases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x86_db = root / "x86.sqlite"
            sbsa_db = root / "sbsa.sqlite"
            output_dir = root / "prepared"
            write_leaf_database(
                x86_db,
                stage="A10-PyTorch-1",
                process_uid="A10-PyTorch-1/coordinator",
                touches={
                    "A10-PyTorch-1/test_x86.py::test_one": {
                        ("/workspace/tensorrt_llm/x86.py", "run")
                    }
                },
                outcomes={"A10-PyTorch-1/test_x86.py::test_one": "passed"},
                expected_workers={"A10-PyTorch-1/test_x86.py::test_one": 0},
            )
            write_leaf_database(
                sbsa_db,
                stage="GH200-PyTorch-1",
                process_uid="GH200-PyTorch-1/coordinator",
                touches={
                    "GH200-PyTorch-1/test_sbsa.py::test_two": {
                        ("/workspace/tensorrt_llm/sbsa.py", "run")
                    }
                },
                outcomes={"GH200-PyTorch-1/test_sbsa.py::test_two": "passed"},
                expected_workers={"GH200-PyTorch-1/test_sbsa.py::test_two": 0},
            )
            urls = artifact.tarball_urls(42)
            selection = {
                "url": urls[0],
                "urls": urls,
                "build": 42,
                "commit": "coverage-commit",
                "lag": 5,
                "base_commit": "pr-base",
                "drift": 2,
                "drift_status": "ahead",
            }

            def download(url: str, destination: Path) -> Path:
                return destination / url.rsplit("/", 1)[-1]

            def extract(tarball: Path, destination: Path) -> bool:
                source = x86_db if "x86_64" in tarball.name else sbsa_db
                shutil.copyfile(source, destination / artifact.DB_NAME)
                return True

            with (
                mock.patch.object(artifact, "merge_base", return_value="pr-base"),
                mock.patch.object(artifact, "select_tarball", return_value=selection) as select,
                mock.patch.object(artifact, "download", side_effect=download),
                mock.patch.object(artifact, "extract", side_effect=extract),
            ):
                ready = artifact.prepare(str(output_dir), "pr-head")

            self.assertIsNotNone(ready)
            assert ready is not None
            select.assert_called_once_with("pr-base")
            connection = sqlite3.connect(ready["path"])
            try:
                tests = {
                    row[0] for row in connection.execute("SELECT DISTINCT test FROM touch_rows")
                }
            finally:
                connection.close()
            self.assertEqual(
                tests,
                {
                    "A10-PyTorch-1/test_x86.py::test_one",
                    "GH200-PyTorch-1/test_sbsa.py::test_two",
                },
            )
            self.assertEqual(json.loads(Path(ready["meta"]).read_text()), selection)

    def test_freshness_gate_honors_configured_threshold(self) -> None:
        self.assertEqual(cbts_main._coverage_freshness(7, 7), ("ok", ""))
        freshness, reason = cbts_main._coverage_freshness(8, 7)
        self.assertEqual(freshness, "stale")
        self.assertTrue(reason)


# Coverage pilot

PILOT_PATH = REPO_ROOT / "jenkins" / "scripts" / "cbts" / "coverage_pilot.py"
PR_API_URL = "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/123"
JSONValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JSONValue"],
    dict[str, "JSONValue"],
]


@pytest.fixture(scope="module")
def pilot_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coverage_pilot", PILOT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload: JSONValue) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


@pytest.mark.parametrize(
    ("pr_info", "expected"),
    (
        ({"user": {"login": " Pilot-User "}}, ("Pilot-User", "author resolved")),
        ({"user": {}}, ("", "PR API response has no author login")),
        ({}, ("", "PR API response has no user")),
        ([], ("", "PR API response is not an object")),
    ),
)
def test_extract_pr_author(
    pilot_module: ModuleType,
    pr_info: JSONValue,
    expected: tuple[str, str],
) -> None:
    assert pilot_module.extract_pr_author(pr_info) == expected


def test_fetch_pr_author_uses_token(
    pilot_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(request: urllib.request.Request, timeout: int) -> _Response:
        assert request.full_url == PR_API_URL
        assert request.get_header("Authorization") == "Bearer token"
        assert timeout == 15
        return _Response({"user": {"login": "pilot-user"}})

    monkeypatch.setattr(pilot_module.urllib.request, "urlopen", urlopen)

    assert pilot_module.fetch_pr_author(PR_API_URL, token="token") == (
        "pilot-user",
        "author resolved",
    )


def test_fetch_pr_author_rejects_untrusted_url(
    pilot_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_urlopen(
        _request: urllib.request.Request,
        _timeout: int,
    ) -> NoReturn:
        raise AssertionError("untrusted URLs must not be requested")

    monkeypatch.setattr(pilot_module.urllib.request, "urlopen", unexpected_urlopen)

    login, reason = pilot_module.fetch_pr_author("https://example.com/pulls/123", token="token")
    assert not login
    assert reason == "missing or unexpected GitHub PR API URL"


def test_fetch_pr_author_fails_closed_on_api_error(
    pilot_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(_request: urllib.request.Request, *, timeout: int) -> NoReturn:
        assert timeout == 15
        raise urllib.error.URLError("unavailable")

    monkeypatch.setattr(pilot_module.urllib.request, "urlopen", urlopen)

    login, reason = pilot_module.fetch_pr_author(PR_API_URL, token="token")
    assert not login
    assert reason.startswith("PR author lookup failed:")


def test_main_reads_bot_trigger_payload(
    pilot_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    trigger_phrase = json.dumps({"github_pr_api_url": PR_API_URL})
    monkeypatch.setenv("gitlabTriggerPhrase", trigger_phrase)
    monkeypatch.setenv("GITHUB_API_TOKEN", "token")

    def fetch_pr_author(
        pr_api_url: str,
        *,
        token: str,
    ) -> tuple[str, str]:
        assert pr_api_url == PR_API_URL
        assert token == "token"
        return "pilot-user", "author resolved"

    monkeypatch.setattr(pilot_module, "fetch_pr_author", fetch_pr_author)

    assert pilot_module.main([]) == 0
    captured = capfd.readouterr()
    assert captured.out == "pilot-user\n"
    assert "pr_author=pilot-user, reason=author resolved" in captured.err


# Coverage selector


def _analyze(source: str, diff: str):
    return analyze_python_changes(
        source,
        iter_diff_post_line_numbers(diff),
        iter_diff_deleted_post_lines(diff),
    )


def test_literal_assignment_resolves_consumers_and_callers_without_name_policy() -> None:
    source = (
        "VALUE = 521\n\ndef helper():\n    return VALUE\n\ndef caller():\n    return helper()\n"
    )
    analysis = _analyze(source, "@@ -0,0 +1 @@\n+VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.changed_bindings == {"VALUE"}
    assert analysis.binding_consumers == {"helper"}
    assert analysis.callers == {"helper": {"caller"}}


@pytest.mark.parametrize(
    ("source", "reason"),
    (
        ("_VALUE = create_value()\n", "effectful module statement"),
        ("class Config:\n    value = 521\n", "class/signature import change"),
    ),
)
def test_unsupported_import_time_changes_remain_fail_closed(source: str, reason: str) -> None:
    changed_line = 2 if source.startswith("class") else 1
    diff = f"@@ -0,0 +{changed_line} @@\n+changed\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == reason


def test_replaced_literal_is_resolved_when_both_images_are_literals() -> None:
    source = "_VALUE = 521\n"
    diff = "@@ -1 +1 @@\n-_VALUE = 520\n+_VALUE = 521\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation


def test_replaced_effectful_assignment_remains_fail_closed() -> None:
    source = "_VALUE = 521\n"
    diff = "@@ -1 +1 @@\n-_VALUE = create_value()\n+_VALUE = 521\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_postponed_annotated_literal_is_resolved() -> None:
    source = "from __future__ import annotations\n\n_CACHE: dict[str, object] = {}\n"
    diff = "@@ -2,0 +3 @@\n+_CACHE: dict[str, object] = {}\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation


def test_builtin_import_is_safe_when_it_is_only_added() -> None:
    source = "import sys\n\ndef shutdown():\n    return sys.modules\n"
    diff = "@@ -0,0 +1 @@\n+import sys\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"shutdown"}


def test_builtin_import_replacement_remains_fail_closed() -> None:
    source = "import _thread as runtime\n"
    diff = "@@ -1 +1 @@\n-import sys as runtime\n+import _thread as runtime\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_literal_assignment_resolves_class_method_consumer() -> None:
    source = "_VALUE = 521\n\nclass Config:\n    def check(self):\n        return _VALUE\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Config.check"}


def test_literal_assignment_attributes_closure_consumer_to_outer() -> None:
    source = (
        "_VALUE = 521\n\ndef outer():\n    def inner():\n        return _VALUE\n    return inner\n"
    )

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"outer"}


def test_import_time_consumer_remains_fail_closed() -> None:
    source = "_VALUE = 521\n\n_ALIAS = _VALUE\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert analysis.limitation == "import-time binding consumer"


def test_plain_function_declaration_with_postponed_annotations_is_safe() -> None:
    source = (
        "from __future__ import annotations\n\ndef _helper(value: object) -> int:\n    return 1\n"
    )
    diff = "@@ -2,0 +3,2 @@\n+def _helper(value: object) -> int:\n+    return 1\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"_helper"}


def test_local_shadow_does_not_create_direct_caller_edge() -> None:
    source = (
        "_VALUE = 521\n"
        "\n"
        "def _helper():\n"
        "    return _VALUE\n"
        "\n"
        "def caller(_helper):\n"
        "    return _helper()\n"
    )

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.callers == {}


def test_local_shadow_does_not_create_binding_consumer() -> None:
    source = "VALUE = 521\n\ndef helper(VALUE):\n    return VALUE\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == set()


def test_function_alias_marks_caller_graph_incomplete() -> None:
    source = (
        "VALUE = 521\n\n"
        "def helper():\n"
        "    return VALUE\n\n"
        "def caller():\n"
        "    alias = helper\n"
        "    return alias()\n"
    )

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+VALUE = 521\n")

    assert analysis.binding_consumers == {"helper"}
    assert analysis.callers == {}
    assert analysis.callable_escapes == {"helper"}


def test_repository_reference_index_follows_import_relationships(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\nOTHER = 2\nTHIRD = 3\n")
    (package / "direct.py").write_text("from pkg.owner import VALUE\n")
    (package / "module.py").write_text("import pkg.owner as owner\nprint(owner.OTHER)\n")
    (package / "relative.py").write_text("from . import owner\nprint(owner.THIRD)\n")
    (package / "unrelated.py").write_text("VALUE = 4\n")

    references = RepositoryReferenceIndex(tmp_path).external_references(
        "pkg/owner.py", {"VALUE", "OTHER", "THIRD"}
    )

    assert references == {"VALUE", "OTHER", "THIRD"}


def test_repository_reference_index_treats_module_escape_as_unresolved(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\nOTHER = 2\n")
    (package / "consumer.py").write_text("import pkg.owner as owner\nregister(owner)\n")

    references = RepositoryReferenceIndex(tmp_path).external_references(
        "pkg/owner.py", {"VALUE", "OTHER"}
    )

    assert references == {"VALUE", "OTHER"}


def test_repository_reference_index_tracks_simple_module_alias(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "owner.py").write_text("VALUE = 1\nOTHER = 2\n")
    (package / "consumer.py").write_text(
        "import pkg.owner as owner\nalias = owner\nprint(alias.OTHER)\n"
    )

    references = RepositoryReferenceIndex(tmp_path).external_references(
        "pkg/owner.py", {"VALUE", "OTHER"}
    )

    assert references == {"OTHER"}


def test_repository_reference_index_reports_direct_importers(tmp_path: Path) -> None:
    package = tmp_path / "perf"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 1\n")
    (package / "test_relative.py").write_text("from .helper import VALUE\n")
    (package / "test_prefixed.py").write_text("from defs.perf.helper import VALUE\n")

    importers = RepositoryReferenceIndex(
        tmp_path,
        module_prefixes=("defs",),
    ).direct_importers("perf/helper.py")

    assert importers.complete
    assert importers.paths == ("perf/test_prefixed.py", "perf/test_relative.py")
    assert not importers.limitation


def test_repository_reference_index_marks_dynamic_importers_incomplete(tmp_path: Path) -> None:
    package = tmp_path / "perf"
    package.mkdir()
    (package / "helper.py").write_text("VALUE = 1\n")
    (package / "test_consumer.py").write_text(
        "import importlib\nmodule_name = '.helper'\nimportlib.import_module(module_name)\n"
    )

    importers = RepositoryReferenceIndex(tmp_path).direct_importers("perf/helper.py")

    assert not importers.complete
    assert not importers.paths
    assert importers.limitation == "dynamic import may target perf/helper.py"


# Decision reporting

SCRIPT_PATH = REPO_ROOT / "jenkins/scripts/cbts/tools/report_cbts_decision.py"


@pytest.fixture()
def report_module():
    """Import report_cbts_decision.py without making its tools directory a package."""
    spec = importlib.util.spec_from_file_location("report_cbts_decision", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_blocks(monkeypatch):
    """Provide a small sharded stage universe to the report's lazy blocks import."""
    stages = {
        "H100-PyTorch-1": SimpleNamespace(yaml_stem="single"),
        "H100-PyTorch-2": SimpleNamespace(yaml_stem="single"),
        "H100-PyTorch-PerfSanity-1": SimpleNamespace(yaml_stem="perf_sanity"),
        "H100-4_GPUs-PyTorch-1": SimpleNamespace(yaml_stem="multi"),
        "H100-4_GPUs-PyTorch-2": SimpleNamespace(yaml_stem="multi"),
        "H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1": SimpleNamespace(yaml_stem="on_demand"),
        "H100-PyTorch-Post-Merge-1": SimpleNamespace(yaml_stem="post_merge"),
    }
    case_counts = {
        "single": 100,
        "perf_sanity": 10,
        "multi": 50,
        "on_demand": 25,
        "post_merge": 20,
    }
    blocks = [
        SimpleNamespace(yaml_stem=stem, tests=[f"case-{index}" for index in range(count)])
        for stem, count in case_counts.items()
    ]

    class FakeYAMLIndex:
        @staticmethod
        def load(_path):
            return SimpleNamespace(blocks=blocks)

    fake_module = types.ModuleType("blocks")
    fake_module.YAMLIndex = FakeYAMLIndex
    fake_module.block_matches_stage = lambda block, stage: block.yaml_stem == stage.yaml_stem
    fake_module.parse_stages_from_groovy = lambda _path, include_post_merge: stages
    monkeypatch.setitem(sys.modules, "blocks", fake_module)
    return stages


def _decision() -> dict:
    return {
        "scope": "testsonly",
        "affected_stages": [
            "H100-PyTorch-1",
            "H100-PyTorch-2",
            "H100-4_GPUs-PyTorch-1",
            "H100-4_GPUs-PyTorch-2",
            "H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1",
            "H100-PyTorch-Post-Merge-1",
        ],
        "affected_stage_test_counts": {
            "H100-PyTorch-1": 20,
            "H100-PyTorch-2": 20,
            "H100-4_GPUs-PyTorch-1": 5,
            "H100-4_GPUs-PyTorch-2": 5,
            "H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1": 1,
            "H100-PyTorch-Post-Merge-1": 2,
        },
        "sanity_required": False,
        "perfsanity_required": True,
    }


@pytest.mark.parametrize(
    ("status", "required", "label_gate_open", "expected"),
    [
        ("pre_merge", False, False, False),
        ("pre_merge", True, False, False),
        ("pre_merge", False, True, False),
        ("pre_merge", True, True, True),
        ("post_merge", False, False, True),
    ],
)
def test_multi_gpu_scheduled_requires_policy_and_label_gate(
    report_module, status, required, label_gate_open, expected
):
    assert report_module._multi_gpu_scheduled(status, required, label_gate_open) is expected


def test_case_counts_use_scheduled_unsharded_pre_merge_universe(report_module, fake_blocks):
    """Multi-GPU/OnDemand/post-merge stages and duplicate shards must not inflate totals."""
    cbts_cases, total_cases = report_module._case_counts(
        _decision(), "pre_merge", str(REPO_ROOT), multi_gpu_scheduled=False
    )

    # One 100-case single-GPU family narrowed to 20, plus a force-kept
    # 10-case PerfSanity family. The two shards are one partitioned case set.
    assert (cbts_cases, total_cases) == (30, 110)


def test_case_counts_include_selected_multi_gpu_when_gate_is_open(report_module, fake_blocks):
    cbts_cases, total_cases = report_module._case_counts(
        _decision(), "pre_merge", str(REPO_ROOT), multi_gpu_scheduled=True
    )

    assert (cbts_cases, total_cases) == (35, 160)


def test_case_counts_include_coverage_multi_gpu_at_full_size(report_module, fake_blocks):
    decision = _decision()
    decision["affected_stages"] = ["H100-PyTorch-1", "H100-PyTorch-2"]
    decision["enable_multi_gpu"] = True

    cbts_cases, total_cases = report_module._case_counts(
        decision, "pre_merge", str(REPO_ROOT), multi_gpu_scheduled=True
    )

    assert (cbts_cases, total_cases) == (80, 160)


def test_build_document_filters_unscheduled_stages_and_persists_valid_rate(report_module):
    decision = _decision()
    decision["affected_stage_split_counts"] = {
        "H100-PyTorch-1": 1,
        "H100-4_GPUs-PyTorch-1": 1,
    }

    document = report_module.build_document(
        decision,
        "pre_merge",
        "",
        "123",
        cbts_cases=25,
        total_cases=100,
        multi_gpu_required=True,
        multi_gpu_label_gate_open=False,
    )

    assert document["d_case_skip_rate"] == 0.75
    assert document["b_case_skip_rate_valid"] is True
    assert document["b_non_cbts_multi_gpu_required"] is True
    assert document["b_multi_gpu_label_gate_open"] is False
    assert document["flat_detail"]["hit_stages"] == [
        "H100-PyTorch-1",
        "H100-PyTorch-2",
    ]
    assert document["flat_detail"]["split_counts"] == {"H100-PyTorch-1": 1}


def test_main_posts_case_skip_rate_to_opensearch(report_module, monkeypatch, tmp_path):
    posted_documents = []

    class FakeOpenSearchDB:
        @staticmethod
        def add_id_of_json(document):
            document["_id"] = "test-id"

        @staticmethod
        def postToOpenSearchDB(document, project):
            assert project == "cbts-test-project"
            posted_documents.append(document)
            return True

    fake_module = types.ModuleType("open_search_db")
    fake_module.CBTS_PROJECT_NAME = "cbts-test-project"
    fake_module.OpenSearchDB = FakeOpenSearchDB
    monkeypatch.setitem(sys.modules, "open_search_db", fake_module)
    monkeypatch.setattr(report_module, "_case_counts", lambda *_args, **_kwargs: (25, 100))

    decision_path = tmp_path / "decision.json"
    decision_path.write_text('{"scope": "testsonly", "affected_stages": []}')

    assert (
        report_module.main(
            [
                "--status",
                "pre_merge",
                "--decision",
                str(decision_path),
                "--repo-root",
                ".",
                "--multi-gpu-required",
                "--multi-gpu-label-gate-open",
            ]
        )
        == 0
    )
    assert len(posted_documents) == 1
    assert posted_documents[0]["d_case_skip_rate"] == 0.75
    assert posted_documents[0]["b_case_skip_rate_valid"] is True
    assert posted_documents[0]["b_non_cbts_multi_gpu_required"] is True
    assert posted_documents[0]["b_multi_gpu_label_gate_open"] is True


# Test-definition rule


def _make_rule(repo_root: Path) -> CbtsTestsDefRule:
    return CbtsTestsDefRule(YAMLIndex(), {}, repo_root)


def _write_defs_file(tmp_path: Path, relative_path: str, content: str) -> None:
    path = tmp_path / "tests/integration/defs" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_perf_rule(tmp_path: Path) -> CbtsTestsDefRule:
    test_db_dir = tmp_path / "test-db"
    test_db_dir.mkdir()
    (test_db_dir / "l0_perf.yml").write_text(
        "l0_perf:\n- tests:\n  - perf/test_perf.py::test_perf[case]\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_b200.yml").write_text(
        "l0_b200:\n- tests:\n  - perf/test_perf_sanity.py::test_e2e[case]\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_gb200_multi_gpus_perf_sanity.yml").write_text(
        "l0_gb200_multi_gpus_perf_sanity:\n"
        "- tests:\n"
        "  - perf/test_perf_sanity.py::test_e2e[case]\n",
        encoding="utf-8",
    )
    (test_db_dir / "l0_other.yml").write_text(
        "l0_other:\n- tests:\n  - other/test_other.py::test_other\n",
        encoding="utf-8",
    )
    stages = {
        "H100_PCIe-PyTorch-Perf-1": Stage("H100_PCIe-PyTorch-Perf-1", "l0_perf", "x86_64", 1, 1),
        "DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1),
        "GB200-4_GPUs-PyTorch-PerfSanity-1": Stage(
            "GB200-4_GPUs-PyTorch-PerfSanity-1",
            "l0_gb200_multi_gpus_perf_sanity",
            "x86_64",
            1,
            1,
        ),
    }
    return CbtsTestsDefRule(YAMLIndex.load(test_db_dir), stages, tmp_path)


@pytest.mark.parametrize(
    "import_statement",
    (
        "from .helper import VALUE",
        "from defs.perf.helper import VALUE",
        "import helper",
    ),
)
def test_direct_test_importers_resolve_static_imports(
    tmp_path: Path,
    import_statement: str,
) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/test_consumer.py", f"{import_statement}\n")
    _write_defs_file(tmp_path, "perf/test_other.py", "VALUE = 2\n")

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") == [
        "perf/test_consumer.py"
    ]


def test_direct_test_importers_fall_back_for_transitive_imports(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/bridge.py", "from .helper import VALUE\n")
    _write_defs_file(tmp_path, "perf/test_consumer.py", "from .bridge import VALUE\n")

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") is None


def test_direct_test_importers_fall_back_for_pytest_plugins(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "disaggregated/plugin.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "disaggregated/test_consumer.py",
        "pytest_plugins = ['plugin']\n",
    )

    assert (
        _direct_test_importers(tmp_path, "tests/integration/defs/disaggregated/plugin.py") is None
    )


def test_direct_test_importers_fall_back_for_conftest_import(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/conftest.py", "from .helper import VALUE\n")

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") is None


def test_direct_test_importers_fall_back_for_dynamic_import(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "perf/test_consumer.py",
        "import importlib\nmodule_name = '.helper'\nimportlib.import_module(module_name)\n",
    )

    assert _direct_test_importers(tmp_path, "tests/integration/defs/perf/helper.py") is None


def test_pytorch_model_config_only_selects_test_perf_consumers(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/pytorch_model_config.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "perf/test_perf.py",
        "from .pytorch_model_config import VALUE\n",
    )
    _write_defs_file(tmp_path, "perf/test_perf_sanity.py", "VALUE = 2\n")
    _write_defs_file(
        tmp_path,
        "conftest.py",
        "from .perf.test_perf import generate_perf_tests\n",
    )

    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/pytorch_model_config.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {"H100_PCIe-PyTorch-Perf-1"}
    assert set(result.block_filters) == {("l0_perf", 0)}
    assert result.perfsanity_relevant is False


def test_perf_sanity_definition_keeps_perfsanity_required(tmp_path: Path) -> None:
    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/test_perf_sanity.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {
        "DGX_B200-PyTorch-1",
        "GB200-4_GPUs-PyTorch-PerfSanity-1",
    }
    assert result.perfsanity_relevant is True


def test_conftest_import_keeps_directory_fallback(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(tmp_path, "perf/conftest.py", "from .helper import VALUE\n")

    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/helper.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {
        "DGX_B200-PyTorch-1",
        "GB200-4_GPUs-PyTorch-PerfSanity-1",
        "H100_PCIe-PyTorch-Perf-1",
    }
    assert set(result.block_filters) == {
        ("l0_b200", 0),
        ("l0_gb200_multi_gpus_perf_sanity", 0),
        ("l0_perf", 0),
    }


def test_uncovered_import_consumers_keep_directory_fallback(tmp_path: Path) -> None:
    _write_defs_file(tmp_path, "perf/helper.py", "VALUE = 1\n")
    _write_defs_file(
        tmp_path,
        "other/test_consumer.py",
        "from defs.perf.helper import VALUE\n",
    )

    result = _make_perf_rule(tmp_path).apply(
        PRInputs(
            changed_files=["tests/integration/defs/perf/helper.py"],
            diffs={},
        )
    )

    assert result is not None
    assert result.affected_stages == {
        "DGX_B200-PyTorch-1",
        "GB200-4_GPUs-PyTorch-PerfSanity-1",
        "H100_PCIe-PyTorch-Perf-1",
    }
    assert set(result.block_filters) == {
        ("l0_b200", 0),
        ("l0_gb200_multi_gpus_perf_sanity", 0),
        ("l0_perf", 0),
    }


def test_scope_start_line_includes_decorators() -> None:
    tree = ast.parse("@decorator\nclass TestExample:\n    pass\n")
    node = tree.body[0]
    assert isinstance(node, ast.ClassDef)
    assert _scope_start_line(node) == 1


def test_compute_anchors_recovers_deleted_scope_before_post_image(
    tmp_path: Path,
) -> None:
    git_path = "tests/integration/defs/test_example.py"
    yaml_path = "test_example.py"
    test_file = tmp_path / git_path
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "class TestB:\n    def test_b(self):\n        pass\n",
        encoding="utf-8",
    )
    diff = (
        "@@ -1,6 +1,3 @@\n"
        "-class TestA:\n"
        "-    def test_a(self):\n"
        "-        pass\n"
        " class TestB:\n"
        "     def test_b(self):\n"
        "         pass\n"
    )

    assert _make_rule(tmp_path)._compute_anchors(git_path, yaml_path, diff) == [
        "test_example.py::TestA"
    ]


def test_deleted_scope_recovery_resets_at_hunk_boundary() -> None:
    diff = "@@ -1,2 +1 @@\n class TestA:\n-    value = 1\n@@ -10 +9,0 @@\n-module_value = 2\n"

    assert _py_class_scopes_from_deletions(diff) is None


def test_deleted_decorator_without_visible_owner_falls_back() -> None:
    diff = "@@ -5 +5,0 @@\n-    @pytest.mark.parametrize('value', [1])\n"

    assert _py_class_scopes_from_deletions(diff) is None


def test_deleted_yaml_body_without_visible_key_falls_back() -> None:
    diff = "@@ -4 +4,0 @@\n-  - expected: 1\n"

    assert _yaml_top_keys_from_deletions(diff) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            'def test_model():\n    task = GSM8K("GPT-OSS/20B-MXFP4")\n',
            ["accuracy/references/gsm8k.yaml"],
        ),
        ("def test_other():\n    pass\n", []),
    ),
)
def test_deleted_accuracy_key_requires_absence_from_test_sources(
    tmp_path: Path,
    source: str,
    expected: list[str],
) -> None:
    git_path = f"{ACCURACY_REFS_PREFIX}gsm8k.yaml"
    yaml_path = "accuracy/references/gsm8k.yaml"
    reference = tmp_path / git_path
    reference.parent.mkdir(parents=True)
    reference.write_text("Other:\n  - expected: 2\n", encoding="utf-8")
    accuracy_test = tmp_path / ACCURACY_DIR / "test_models.py"
    accuracy_test.write_text(source, encoding="utf-8")
    diff = "@@ -1,4 +1,2 @@\n-GPT-OSS/20B-MXFP4:\n-  - expected: 1\n Other:\n   - expected: 2\n"

    assert _make_rule(tmp_path)._compute_anchors(git_path, yaml_path, diff) == expected
