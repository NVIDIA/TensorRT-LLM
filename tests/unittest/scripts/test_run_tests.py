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

import io
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
# run_tests.py and the ci_test_runner package import each other and test_rerun
# from jenkins/scripts, as they do when run_tests.py is the entry point.
sys.path.insert(0, str(REPO_ROOT / "jenkins" / "scripts"))

import run_tests  # noqa: E402
from ci_test_runner import execution, render, results  # noqa: E402
from ci_test_runner.execution import RunOutcome  # noqa: E402

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


def _tmp_dir(test_case):
    tmp = tempfile.TemporaryDirectory()
    test_case.addCleanup(tmp.cleanup)
    return Path(tmp.name)


class FindTestListLineTest(unittest.TestCase):
    def test_exact_match_wins_over_ignore_substring(self) -> None:
        lines = [ATTENTION_DIR, GVR_FILE]

        self.assertEqual(render._find_test_list_line(GVR_FILE, lines), GVR_FILE)
        self.assertEqual(render._find_test_list_line(ATTENTION_DIR, lines), ATTENTION_DIR)

    def test_marker_suffix_is_ignored_when_matching(self) -> None:
        unittest_line = "unittest/_torch/modules/test_mhc.py TIMEOUT (90)"
        nodeid_line = "accuracy/test_llm_api_pytorch.py::TestA::test_b[c] TIMEOUT (60)"
        lines = [unittest_line, nodeid_line]

        # test_unittests_v2 keeps the TIMEOUT marker in its parameter id.
        self.assertEqual(render._find_test_list_line(unittest_line, lines), unittest_line)
        self.assertEqual(
            render._find_test_list_line(
                "accuracy/test_llm_api_pytorch.py::TestA::test_b[c]", lines
            ),
            nodeid_line,
        )

    def test_falls_back_to_substring_match(self) -> None:
        line = 'unittest/_torch/moe/test_moe_module.py::test_single_gpu -k "CUTLASS"'

        self.assertEqual(
            render._find_test_list_line("test_moe_module.py::test_single_gpu", [line]), line
        )
        self.assertIsNone(render._find_test_list_line("unittest/other.py", [line]))


class RenderTestListTest(unittest.TestCase):
    def _render(self, test_db_lines, collected_ids, returncode=0):
        tmp_dir = _tmp_dir(self)
        test_db = tmp_dir / "l0_test.txt"
        test_db.write_text("\n".join(test_db_lines) + "\n")
        collect = subprocess.CompletedProcess(
            args="pytest", returncode=returncode, stdout=_collect_output(collected_ids), stderr=""
        )
        with mock.patch.object(render.subprocess, "run", return_value=collect):
            regular_file, isolate_file, _, _ = render.render_test_list(
                str(test_db), str(tmp_dir), splits=1, group=1
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

    def test_failed_collection_is_a_setup_error(self) -> None:
        with self.assertRaises(render.SetupError):
            self._render([GVR_FILE], [], returncode=4)


class RunIsolatedTestsSummaryTest(unittest.TestCase):
    def _run(self, not_rerun):
        output_dir = _tmp_dir(self)
        isolate_list = output_dir / "isolate.txt"
        isolate_list.write_text("unittest/a.py\nunittest/b.py\n")

        def fake_check_and_rerun(result_xml, *args, **kwargs):
            return RunOutcome(result_xml, not_rerun=not_rerun)

        # The first isolated test passes, the second fails after a long run.
        pytest_results = iter([(0, 30.0, []), (1, 700.0, [])])
        with (
            mock.patch.object(
                execution, "run_pytest", side_effect=lambda *a, **k: next(pytest_results)
            ),
            mock.patch.object(execution, "check_and_rerun", side_effect=fake_check_and_rerun),
            mock.patch.object(execution, "print_phase_summary") as summary,
        ):
            outcomes = execution.run_isolated_tests(
                "pytest", str(isolate_list), str(output_dir), str(output_dir), "stage", []
            )
        _, passed, failed, _ = summary.call_args.args
        return passed, failed, outcomes

    def test_failure_not_eligible_for_rerun_is_counted_as_failed(self) -> None:
        passed, failed, outcomes = self._run(not_rerun=True)
        self.assertEqual((passed, failed), (1, 1))
        self.assertEqual([o.not_rerun for o in outcomes], [False, True])

    def test_failure_fixed_by_rerun_is_counted_as_passed(self) -> None:
        passed, failed, _ = self._run(not_rerun=False)
        self.assertEqual((passed, failed), (2, 0))


class RunPytestTest(unittest.TestCase):
    def _run(self, script, **kwargs):
        output_dir = _tmp_dir(self)
        log = output_dir / "first_run_output.log"
        cmd = f"{sys.executable} -c {shlex.quote(script)}"
        with mock.patch.object(execution.sys, "stdout", io.StringIO()):
            result = execution.run_pytest(cmd, str(output_dir), log_path=str(log), **kwargs)
        return result, log

    def test_failed_session_writes_its_output_tail(self) -> None:
        (rc, _, tail), log = self._run(
            "import sys; print('first'); print('last'); sys.exit(1)", tail_lines=1
        )

        self.assertEqual(rc, 1)
        self.assertEqual(tail, ["last"])
        self.assertEqual(log.read_text(), "first\nlast\n")

    def test_passing_session_writes_no_log(self) -> None:
        (rc, _, _), log = self._run("print(1)")

        self.assertEqual(rc, 0)
        self.assertFalse(log.exists())

    def test_output_tail_is_bounded(self) -> None:
        with mock.patch.object(execution, "_OUTPUT_TAIL_BYTES", 20):
            _, log = self._run("import sys; [print(i) for i in range(1000)]; sys.exit(1)")

        self.assertEqual(log.read_text(), "995\n996\n997\n998\n999\n")


class CheckAndRerunTest(unittest.TestCase):
    def _check_and_rerun(self, output_dir, fake_generate, fake_run_pytest):
        with (
            mock.patch.object(
                execution.test_rerun, "generate_rerun_tests_list", side_effect=fake_generate
            ),
            mock.patch.object(execution, "run_pytest", side_effect=fake_run_pytest) as run,
        ):
            outcome = execution.check_and_rerun(
                str(output_dir / "results.xml"),
                "pytest",
                str(output_dir),
                str(output_dir),
                "regular",
                [],
                str(output_dir / "regular.txt"),
            )
        return outcome, run

    def test_reruns_every_eligible_test_without_a_cap(self) -> None:
        output_dir = _tmp_dir(self)
        not_run = [f"unittest/t{i}.py" for i in range(24)]

        def fake_generate(rerun_dir, *args, **kwargs):
            Path(rerun_dir, "rerun_1.txt").write_text("\n".join(not_run) + "\n")

        def fake_run_pytest(cmd, working_dir, tail_lines=0, log_path=None):
            Path(output_dir, "rerun", "regular", "rerun_results_1.xml").write_text("<testsuites/>")
            return 0, 1.0, []

        outcome, run = self._check_and_rerun(output_dir, fake_generate, fake_run_pytest)

        run.assert_called_once()
        self.assertFalse(outcome.failed_after_rerun)
        self.assertFalse(outcome.not_rerun)
        self.assertEqual(len(outcome.rerun_xmls), 1)

    def test_crashed_rerun_keeps_the_earlier_attempt_xml(self) -> None:
        output_dir = _tmp_dir(self)
        rerun_dir = output_dir / "rerun" / "regular"

        def fake_generate(rerun_dir, *args, **kwargs):
            Path(rerun_dir, "rerun_1.txt").write_text("unittest/slow.py\n")
            Path(rerun_dir, "rerun_2.txt").write_text("unittest/fast.py\n")

        def fake_run_pytest(cmd, working_dir, tail_lines=0, log_path=None):
            # The first attempt fails with an XML; the second crashes without one.
            if "rerun_1.txt" in cmd:
                Path(rerun_dir, "rerun_results_1.xml").write_text("<testsuites/>")
            return 1, 1.0, []

        outcome, _ = self._check_and_rerun(output_dir, fake_generate, fake_run_pytest)

        self.assertTrue(outcome.failed_after_rerun)
        self.assertEqual(outcome.rerun_xmls, [str(rerun_dir / "rerun_results_1.xml")])
        self.assertEqual(
            outcome.failed_rerun_logs,
            [str(rerun_dir / "rerun_output_1.log"), str(rerun_dir / "rerun_output_2.log")],
        )

    def test_failure_not_eligible_for_rerun_is_reported(self) -> None:
        output_dir = _tmp_dir(self)

        def fake_generate(rerun_dir, *args, **kwargs):
            Path(rerun_dir, "rerun_0.txt").write_text("unittest/long.py\n")

        outcome, run = self._check_and_rerun(output_dir, fake_generate, None)

        run.assert_not_called()
        self.assertTrue(outcome.not_rerun)
        self.assertFalse(outcome.failed_after_rerun)

    def test_stage_policy_decides_what_is_rerun(self) -> None:
        output_dir = _tmp_dir(self)
        (output_dir / "results.xml").write_text(
            '<testsuites><testsuite name="stage" tests="1" failures="1">'
            '<testcase classname="unittest.fast" name="test_a" time="1.0">'
            '<failure message="boom">boom</failure></testcase>'
            "</testsuite></testsuites>"
        )
        test_name = execution.test_rerun.parse_name("unittest.fast", "test_a", "")
        (output_dir / "regular.txt").write_text(f"{test_name}\n")
        policy = {
            "rerun_enabled": False,
            "rerun_twice_max_duration_s": 300,
            "rerun_once_max_duration_s": 600,
            "max_failed_tests": None,
        }

        with mock.patch.object(execution, "run_pytest") as run:
            outcome = execution.check_and_rerun(
                str(output_dir / "results.xml"),
                "pytest",
                str(output_dir),
                str(output_dir),
                "regular",
                [],
                str(output_dir / "regular.txt"),
                rerun_policy=policy,
            )

        # Under the default policy this 1-second failure would be rerun twice.
        run.assert_not_called()
        self.assertTrue(outcome.not_rerun)
        self.assertTrue((output_dir / "rerun" / "regular" / "rerun_0.txt").exists())


class MergeResultsTest(unittest.TestCase):
    def test_rerun_report_covers_only_runs_that_were_rerun(self) -> None:
        output_dir = _tmp_dir(self)

        def xml(name):
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<testsuites><testsuite name="pytest"/></testsuites>')
            return str(path)

        regular = RunOutcome(xml("results.xml"), [xml("rerun/regular/rerun_results_1.xml")])
        iso_0 = RunOutcome(xml("results_isolated_0.xml"))
        iso_1 = RunOutcome(
            xml("results_isolated_1.xml"), [xml("rerun/isolated_1/rerun_results_2.xml")]
        )
        # pytest died before writing this one.
        iso_2 = RunOutcome(str(output_dir / "results_isolated_2.xml"))

        with (
            mock.patch.object(results.test_rerun, "generate_rerun_report") as report,
            mock.patch.object(results.test_rerun, "merge_junit_xmls") as merge,
        ):
            results.merge_results(str(output_dir), "stage", [regular, iso_0, iso_1, iso_2])

        self.assertEqual(
            report.call_args.args[1],
            [
                regular.result_xml,
                iso_1.result_xml,
                *regular.rerun_xmls,
                *iso_1.rerun_xmls,
            ],
        )
        self.assertEqual(
            merge.call_args.args[1],
            [
                regular.result_xml,
                *regular.rerun_xmls,
                iso_0.result_xml,
                iso_1.result_xml,
                *iso_1.rerun_xmls,
            ],
        )
        self.assertIn('name="stage"', Path(regular.result_xml).read_text())
        self.assertFalse(Path(iso_0.result_xml).exists())
        self.assertFalse(Path(iso_1.result_xml).exists())

    def test_no_rerun_report_without_reruns(self) -> None:
        output_dir = _tmp_dir(self)
        results.create_empty_results_xml(str(output_dir), "stage")

        with (
            mock.patch.object(results.test_rerun, "generate_rerun_report") as report,
            mock.patch.object(results.test_rerun, "merge_junit_xmls"),
        ):
            results.merge_results(
                str(output_dir), "stage", [RunOutcome(str(output_dir / "results.xml"))]
            )

        report.assert_not_called()


class RunExitCodeTest(unittest.TestCase):
    def _run(self, regular_outcome, counts=(1, 0), output_dir=None):
        output_dir = output_dir or _tmp_dir(self)
        argv = [
            "--test-db-list",
            str(output_dir / "l0.txt"),
            "--pytest-base-cmd",
            "pytest",
            "--stage-name",
            "stage",
            "--output-dir",
            str(output_dir),
        ]
        rendered = ("regular.txt", "isolate.txt", *counts)
        with (
            mock.patch.object(run_tests, "render_test_list", return_value=rendered),
            mock.patch.object(run_tests, "run_regular_tests", return_value=regular_outcome),
            mock.patch.object(run_tests, "merge_results"),
        ):
            return run_tests.main(argv)

    def test_passing_run_exits_zero(self) -> None:
        self.assertEqual(self._run(RunOutcome("results.xml")), 0)

    def test_stage_rerun_policy_reaches_the_runs(self) -> None:
        output_dir = _tmp_dir(self)
        argv = [
            "--test-db-list",
            str(output_dir / "l0.txt"),
            "--pytest-base-cmd",
            "pytest",
            "--stage-name",
            "DGX_B200-PyTorch-1",
            "--output-dir",
            str(output_dir),
        ]
        policy = {"rerun_enabled": False}
        rendered = ("regular.txt", "isolate.txt", 1, 1)
        with (
            mock.patch.object(run_tests, "render_test_list", return_value=rendered),
            mock.patch.object(
                run_tests, "load_rerun_policy", return_value=(policy, "DGX_B200-*")
            ) as load,
            mock.patch.object(
                run_tests, "run_regular_tests", return_value=RunOutcome("results.xml")
            ) as regular,
            mock.patch.object(run_tests, "run_isolated_tests", return_value=[]) as isolated,
            mock.patch.object(run_tests, "merge_results"),
        ):
            self.assertEqual(run_tests.main(argv), 0)

        load.assert_called_once_with("DGX_B200-PyTorch-1")
        self.assertIs(regular.call_args.kwargs["rerun_policy"], policy)
        self.assertIs(isolated.call_args.kwargs["rerun_policy"], policy)

    def test_failure_after_rerun_wins_over_not_rerun(self) -> None:
        outcome = RunOutcome("results.xml", failed_after_rerun=True, not_rerun=True)
        self.assertEqual(self._run(outcome), run_tests.EXIT_TESTS_FAILED)

    def test_failure_not_eligible_for_rerun(self) -> None:
        outcome = RunOutcome("results.xml", not_rerun=True)
        self.assertEqual(self._run(outcome), run_tests.EXIT_TESTS_NOT_RERUN)

    def test_failure_evidence_is_the_failed_reruns(self) -> None:
        output_dir = _tmp_dir(self)
        (output_dir / "first.log").write_text("Connection refused\n")
        (output_dir / "rerun.log").write_text("AssertionError\n")
        outcome = RunOutcome(
            "results.xml",
            failed_after_rerun=True,
            not_rerun=True,
            failed_rerun_logs=[str(output_dir / "rerun.log")],
            first_run_log=str(output_dir / "first.log"),
        )

        self._run(outcome, output_dir=output_dir)

        evidence = output_dir / execution.FAILURE_EVIDENCE_LOG
        self.assertEqual(evidence.read_text(), "AssertionError\n")

    def test_failure_evidence_is_the_run_not_rerun(self) -> None:
        output_dir = _tmp_dir(self)
        (output_dir / "first.log").write_text("GPU has fallen off the bus\n")
        outcome = RunOutcome(
            "results.xml", not_rerun=True, first_run_log=str(output_dir / "first.log")
        )

        self._run(outcome, output_dir=output_dir)

        evidence = output_dir / execution.FAILURE_EVIDENCE_LOG
        self.assertEqual(evidence.read_text(), "GPU has fallen off the bus\n")

    def test_passing_run_writes_no_failure_evidence(self) -> None:
        output_dir = _tmp_dir(self)
        self._run(RunOutcome("results.xml"), output_dir=output_dir)
        self.assertFalse((output_dir / execution.FAILURE_EVIDENCE_LOG).exists())

    def test_no_tests_is_a_setup_failure(self) -> None:
        self.assertEqual(self._run(None, counts=(0, 0)), run_tests.EXIT_SETUP_FAILED)


class CountXmlResultsTest(unittest.TestCase):
    def test_counts_passed_and_failed_excluding_skipped(self) -> None:
        xml = _tmp_dir(self) / "results.xml"
        xml.write_text(
            "<testsuites><testsuite>"
            '<testcase name="a"/>'
            '<testcase name="b"/>'
            '<testcase name="c"><failure message="x"/></testcase>'
            '<testcase name="d"><error message="y"/></testcase>'
            '<testcase name="e"><skipped/></testcase>'
            "</testsuite></testsuites>"
        )
        self.assertEqual(results.count_xml_results(str(xml)), (2, 2))

    def test_missing_xml_counts_nothing(self) -> None:
        self.assertEqual(results.count_xml_results("/nonexistent/results.xml"), (0, 0))


if __name__ == "__main__":
    unittest.main()
