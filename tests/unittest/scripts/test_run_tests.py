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

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_TESTS_PATH = REPO_ROOT / "jenkins" / "scripts" / "run_tests.py"

_spec = importlib.util.spec_from_file_location("run_tests", _RUN_TESTS_PATH)
run_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_tests)

ATTENTION_DIR = (
    "unittest/_torch/attention"
    " --ignore=unittest/_torch/attention/sparse/test_cute_dsl_gvr_topk_decode.py"
)
GVR_FILE = "unittest/_torch/attention/sparse/test_cute_dsl_gvr_topk_decode.py"


def _collect_output(test_ids):
    lines = ["[pytest-split] Running group 1/1 (estimated duration: 1.00s)", ""]
    lines += test_ids
    lines += [
        "",
        "=============================== warnings summary ===============================",
    ]
    return "\n".join(lines)


class FindTestListLineTest(unittest.TestCase):
    def test_exact_match_wins_over_ignore_substring(self) -> None:
        lines = [ATTENTION_DIR, GVR_FILE]

        self.assertEqual(run_tests._find_test_list_line(GVR_FILE, lines), GVR_FILE)
        self.assertEqual(run_tests._find_test_list_line(ATTENTION_DIR, lines), ATTENTION_DIR)

    def test_marker_suffix_is_ignored_when_matching(self) -> None:
        unittest_line = "unittest/_torch/modules/test_mhc.py TIMEOUT (90)"
        nodeid_line = "accuracy/test_llm_api_pytorch.py::TestA::test_b[c] TIMEOUT (60)"
        lines = [unittest_line, nodeid_line]

        # test_unittests_v2 keeps the TIMEOUT marker in its parameter id.
        self.assertEqual(run_tests._find_test_list_line(unittest_line, lines), unittest_line)
        self.assertEqual(
            run_tests._find_test_list_line(
                "accuracy/test_llm_api_pytorch.py::TestA::test_b[c]", lines
            ),
            nodeid_line,
        )

    def test_falls_back_to_substring_match(self) -> None:
        line = 'unittest/_torch/moe/test_moe_module.py::test_single_gpu -k "CUTLASS"'

        self.assertEqual(
            run_tests._find_test_list_line("test_moe_module.py::test_single_gpu", [line]), line
        )
        self.assertIsNone(run_tests._find_test_list_line("unittest/other.py", [line]))


class RenderTestListTest(unittest.TestCase):
    def _render(self, test_db_lines, collected_ids):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_dir = Path(tmp.name)
        test_db = tmp_dir / "l0_test.txt"
        test_db.write_text("\n".join(test_db_lines) + "\n")
        collect = subprocess.CompletedProcess(
            args="pytest", returncode=0, stdout=_collect_output(collected_ids), stderr=""
        )
        with mock.patch.object(run_tests.subprocess, "run", return_value=collect):
            regular_file, isolate_file, _, _ = run_tests.render_test_list(
                str(test_db), str(tmp_dir), splits=1, group=1, perf_mode=False
            )
        return (
            Path(regular_file).read_text().splitlines(),
            Path(isolate_file).read_text().splitlines(),
        )

    def test_ignored_file_does_not_duplicate_its_parent_directory(self) -> None:
        regular, isolate = self._render(
            [ATTENTION_DIR, GVR_FILE],
            [
                f"test_unittests.py::test_unittests_v2[{ATTENTION_DIR}]",
                f"test_unittests.py::test_unittests_v2[{GVR_FILE}]",
            ],
        )

        self.assertEqual(regular, [ATTENTION_DIR, GVR_FILE])
        self.assertEqual(isolate, [])

    def test_isolation_marker_routes_only_the_marked_line(self) -> None:
        regular, isolate = self._render(
            [f"{ATTENTION_DIR} ISOLATION", GVR_FILE],
            [
                f"test_unittests.py::test_unittests_v2[{ATTENTION_DIR}]",
                f"test_unittests.py::test_unittests_v2[{GVR_FILE}]",
            ],
        )

        self.assertEqual(regular, [GVR_FILE])
        self.assertEqual(isolate, [ATTENTION_DIR])


class RunIsolatedTestsSummaryTest(unittest.TestCase):
    def _run(self, write_rerun_0):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output_dir = Path(tmp.name)
        isolate_list = output_dir / "isolate.txt"
        isolate_list.write_text("unittest/a.py\nunittest/b.py\n")

        def fake_check_and_rerun(result_xml, base_cmd, working_dir, out_dir, rerun_tag, *args):
            if write_rerun_0:
                rerun_dir = Path(out_dir, "rerun", rerun_tag)
                rerun_dir.mkdir(parents=True, exist_ok=True)
                (rerun_dir / "rerun_0.txt").write_text("unittest/b.py\n")
            return False, []

        # The first isolated test passes, the second fails after > 10 min.
        pytest_results = iter([(0, 30.0, []), (1, 700.0, [])])
        with (
            mock.patch.object(
                run_tests, "run_pytest", side_effect=lambda *a, **k: next(pytest_results)
            ),
            mock.patch.object(run_tests, "check_and_rerun", side_effect=fake_check_and_rerun),
            mock.patch.object(run_tests, "print_phase_summary") as summary,
        ):
            run_tests.run_isolated_tests(
                "pytest", str(isolate_list), str(output_dir), str(output_dir), "stage", [], 10
            )
        _, passed, failed, _ = summary.call_args.args
        return passed, failed

    def test_failure_not_eligible_for_rerun_is_counted_as_failed(self) -> None:
        self.assertEqual(self._run(write_rerun_0=True), (1, 1))

    def test_failure_fixed_by_rerun_is_counted_as_passed(self) -> None:
        self.assertEqual(self._run(write_rerun_0=False), (2, 0))


if __name__ == "__main__":
    unittest.main()
