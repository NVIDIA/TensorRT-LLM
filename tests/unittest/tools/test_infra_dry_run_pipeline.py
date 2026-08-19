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

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
L0_TEST = (REPO_ROOT / "jenkins" / "L0_Test.groovy").read_text()
L0_PARENT = (REPO_ROOT / "jenkins" / "L0_MergeRequest.groovy").read_text()
SLURM_RUN = (REPO_ROOT / "jenkins" / "scripts" / "slurm_run.sh").read_text()
CHECK_TEST_LIST = (REPO_ROOT / "scripts" / "check_test_list.py").read_text()
BENCHMARK_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "test_infra_dry_run_benchmark.py"
CONFTEST = (REPO_ROOT / "tests" / "integration" / "defs" / "conftest.py").read_text()
DRY_RUN_DB_PATH = (
    REPO_ROOT / "tests" / "integration" / "test_lists" / "test-db" / "infra_dry_run.yml"
)


def _function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"def {name}")
    return source[start : source.index(f"def {next_name}", start + len(name))]


def _conditional_workflow_properties(function_body: str) -> set[str]:
    conditions = re.findall(r"\bif\s*\(([^)]*)\)", function_body)
    return {
        identifier
        for condition in conditions
        for identifier in re.findall(r"\b[A-Z][A-Z0-9_]+\b", condition)
    }


def _top_level_workflow_properties(source: str) -> set[str]:
    return set(re.findall(r"(?m)^(?:def\s+)?([A-Z][A-Z0-9_]*)\s*=", source))


def _groovy_list_values_after(source: str, assignment: str) -> list[str]:
    assignment_start = source.index(assignment)
    list_start = source.index("[", assignment_start)
    list_end = source.index("]", list_start)
    return re.findall(r'"([^"]+)"', source[list_start:list_end])


def _pytest_capture_mode(args: list[str], initial_mode: str) -> str:
    capture_mode = initial_mode
    for arg in args:
        if arg == "-s":
            capture_mode = "no"
        elif arg.startswith("--capture="):
            capture_mode = arg.split("=", 1)[1]
    return capture_mode


class InfraDryRunPipelineTest(unittest.TestCase):
    # Keep source-level assertions scoped to behavior introduced by the dry-run
    # feature so unrelated Jenkins refactors do not break this regression suite.

    def test_dedicated_context_selects_one_standard_pytest_case(self) -> None:
        database = DRY_RUN_DB_PATH.read_text()
        self.assertTrue(BENCHMARK_PATH.is_file())
        self.assertTrue(BENCHMARK_PATH.name.startswith("test_"))
        self.assertEqual(database.count("::test_"), 1)
        self.assertIn("infra_dry_run:", database)
        self.assertIn("test_infra_dry_run_benchmark.py::test_infra_dry_run_benchmark", database)
        self.assertNotIn("infra_dry_run_benchmark.py", CHECK_TEST_LIST)
        verify_l0 = _function_body(CHECK_TEST_LIST, "verify_l0_test_lists", "verify_qa_test_lists")
        self.assertIn("pytest --test-list={test_list}", verify_l0)

    def test_platform_runner_uses_standard_pytest_results_and_reporting(self) -> None:
        body = _function_body(L0_TEST, "runLLMTestlistOnPlatformImpl", "runLLMTestlistOnPlatform")
        self.assertIn(
            "effectiveTestList = infraDryRun ? INFRA_DRY_RUN_TEST_CONTEXT : testList", body
        )
        self.assertIn("effectiveSplitId = infraDryRun ? 1 : splitId", body)
        self.assertIn("effectiveSplits = infraDryRun ? 1 : splits", body)
        self.assertIn("effectivePerfMode = infraDryRun ? false : perfMode", body)
        self.assertIn("getPytestBaseCommandLine(", body)
        self.assertIn("--test-list=${preprocessedLists.regular}", body)
        self.assertIn(
            "pytestCommand += getInfraDryRunPytestTargets(preprocessedLists.regular)",
            body,
        )
        self.assertIn("rerunFailedTests(", body)
        self.assertIn("runIsolatedTests(", body)
        self.assertIn("generateRerunReport(", body)
        self.assertIn('testEnvironment += ["stageName=${stageName}"]', body)
        self.assertNotIn("test_infra_dry_run_benchmark.py", body)
        self.assertNotIn("positionalTest", L0_TEST)

    def test_docs_use_prepared_standard_pytest_workspace_only_for_dry_run(self) -> None:
        prepared = _function_body(L0_TEST, "runInfraDryRunInPreparedWorkspace", "runLLMDocBuild")
        docs = _function_body(L0_TEST, "runLLMDocBuild", "launchTestListCheck")
        for call in ("renderTestDB(", "processShardTestList(", "getPytestBaseCommandLine("):
            self.assertIn(call, prepared)
        self.assertIn("--test-list=${preprocessedLists.regular}", prepared)
        self.assertIn(
            "pytestCommand += getInfraDryRunPytestTargets(preprocessedLists.regular)",
            prepared,
        )
        self.assertIn('withEnv(["stageName=${stageName}", "TRTLLM_INFRA_DRY_RUN=true"])', prepared)
        self.assertNotIn("test_infra_dry_run_benchmark.py", prepared)
        self.assertLess(docs.index("if (isInfraDryRun())"), docs.index("make html"))
        conditional_properties = _conditional_workflow_properties(prepared)
        self.assertTrue(conditional_properties)
        self.assertLessEqual(
            conditional_properties,
            _top_level_workflow_properties(L0_TEST),
        )
        self.assertIn("def runLLMDocBuild(pipeline, config)", L0_TEST)
        self.assertIn("runLLMDocBuild(pipeline, config=VANILLA_CONFIG)", L0_TEST)
        self.assertIn('runInfraDryRunInPreparedWorkspace(pipeline, llmSrc, "CPU-Build_Docs")', docs)
        upload_args = _groovy_list_values_after(prepared, "extraArgs += [")
        self.assertTrue(any(arg.startswith("--s3-upload-path=") for arg in upload_args))
        self.assertEqual(_pytest_capture_mode(upload_args, initial_mode="no"), "fd")

    def test_agent_flow_uses_prepared_standard_pytest_workspace_only_for_dry_run(self) -> None:
        body = _function_body(L0_TEST, "runLLMAgentFlowTest", "launchTestListCheck")
        self.assertIn("if (isInfraDryRun())", body)
        infra_pytest_install = (
            "pip3 install 'pytest<9.1' pytest-csv pytest-split pytest-timeout "
            "pytest-unused-fixtures mako boto3"
        )
        self.assertIn(infra_pytest_install, body)
        self.assertIn("runInfraDryRunInPreparedWorkspace(pipeline, llmSrc, stageName)", body)
        self.assertLess(
            body.index(infra_pytest_install),
            body.index("runInfraDryRunInPreparedWorkspace(pipeline, llmSrc, stageName)"),
        )
        self.assertLess(body.index("if (isInfraDryRun())"), body.index("pip3 install -e"))
        dry_guard = body.index("if (isInfraDryRun())")
        realpath = body.index("realpath ${LLM_ROOT}")
        dry_runner = body.index("runInfraDryRunInPreparedWorkspace(pipeline, llmSrc, stageName)")
        normal_root = body.index('def agentFlowRoot = "${LLM_ROOT}/agent-flow"')
        self.assertLess(dry_guard, realpath)
        self.assertLess(realpath, dry_runner)
        self.assertLess(body.index("\n        return", dry_runner), normal_root)
        self.assertIn('def agentFlowRoot = "${LLM_ROOT}/agent-flow"', body)

    def test_prepared_workspace_sets_dry_environment_before_collection(self) -> None:
        body = _function_body(L0_TEST, "runInfraDryRunInPreparedWorkspace", "runLLMDocBuild")
        dry_environment = 'withEnv(["stageName=${stageName}", "TRTLLM_INFRA_DRY_RUN=true"])'
        self.assertEqual(body.count(dry_environment), 1)
        self.assertLess(body.index(dry_environment), body.index("processShardTestList("))
        self.assertLess(body.index(dry_environment), body.index("${pytestCommand.join"))

    def test_dry_run_limits_collection_to_rendered_nodeids(self) -> None:
        targets = _function_body(L0_TEST, "getInfraDryRunPytestTargets", "processShardTestList")
        preprocessing = _function_body(L0_TEST, "processShardTestList", "isValidSlurmJobId")
        expected_target = "test_infra_dry_run_benchmark.py::test_infra_dry_run_benchmark"
        self.assertIn("if (!isInfraDryRun())", targets)
        self.assertIn('.findAll { it.contains("::") }', targets)
        self.assertIn(f'"{expected_target}"', targets)
        self.assertIn("if (targets != [expectedTarget])", targets)
        self.assertIn(
            "testListCmd += getInfraDryRunPytestTargets(cleanedTestDBList)",
            preprocessing,
        )

    def test_dry_run_rejects_shell_metacharacters_in_rendered_target(self) -> None:
        targets = _function_body(L0_TEST, "getInfraDryRunPytestTargets", "processShardTestList")
        expected_target = "test_infra_dry_run_benchmark.py::test_infra_dry_run_benchmark"
        metacharacter_target = f"{expected_target};touch${{IFS}}/tmp/infra-dry-run"
        parsed_targets = [metacharacter_target.split(maxsplit=1)[0]]
        self.assertNotEqual(parsed_targets, [expected_target])
        self.assertIn("if (targets != [expectedTarget])", targets)

    def test_dry_run_conftest_does_not_require_product_bindings(self) -> None:
        dry_guard = '_INFRA_DRY_RUN = os.environ.get("TRTLLM_INFRA_DRY_RUN", "").lower() == "true"'
        self.assertIn(dry_guard, CONFTEST)
        guard_start = CONFTEST.index(dry_guard)
        normal_import = CONFTEST.index(
            "from tensorrt_llm.bindings import ipc_nvls_supported", guard_start
        )
        fallback = CONFTEST[guard_start:normal_import]
        self.assertIn("def ipc_nvls_supported():", fallback)
        self.assertIn("def get_mpi_world_size():", fallback)
        self.assertIn("else:", fallback)
        self.assertNotIn("from .perf.gpu_clock_lock import GPUClockLock", fallback)

    def test_slurm_keeps_only_dry_gates_needed_by_the_standard_runner(self) -> None:
        body = _function_body(L0_TEST, "runLLMTestlistWithSbatch", "runLLMTestlistOnSlurm")
        self.assertIn(
            "effectiveTestList = infraDryRun ? INFRA_DRY_RUN_TEST_CONTEXT : testList", body
        )
        self.assertIn("String[] taskArgs = getNodeArgs(", body)
        self.assertIn(
            "pytestCommandParts += getInfraDryRunPytestTargets(testListPathLocal)",
            body,
        )
        self.assertIn('${infraDryRun ? "export infraDryRun=true" : ""}', body)
        self.assertNotIn("export infraDryRun=$infraDryRun", body)
        self.assertIn("if (!isInfraDryRun() && (disaggMultiNodeMode || aggMultiNodeMode))", body)
        self.assertNotIn("test_infra_dry_run_benchmark.py", body)
        dispatch = _function_body(L0_TEST, "runLLMTestlistOnSlurm", "INFRA_DRY_RUN")
        self.assertIn("if (nodeCount > 1 || runWithSbatch)", dispatch)
        self.assertNotIn("isInfraDryRun() || nodeCount", dispatch)
        agent = _function_body(L0_TEST, "runLLMTestlistWithAgent", "executeLLMTestOnSlurm")
        self.assertIn("runInDockerOnNodeMultiStage", agent)
        self.assertIn("runInEnrootOnNode", agent)
        self.assertIn(
            'if [[ "${infraDryRun:-false}" == "true" || "$stageName" != *Disagg* ]]',
            SLURM_RUN,
        )

    def test_parent_uses_parameter_only_and_explicit_non_fail_fast_helper(self) -> None:
        setup = L0_PARENT[
            L0_PARENT.index("boolean infraDryRun =") : L0_PARENT.index("String reuseBuild =")
        ]
        helper = _function_body(L0_PARENT, "launchInfraDryRunTestJob", "launchStages")
        launch = _function_body(L0_PARENT, "launchJob", "launchInfraDryRunTestJob")
        self.assertIn("params.InfraDryRun?.toString()?.toBoolean()", setup)
        self.assertIn("(INFRA_DRY_RUN): infraDryRun", setup)
        self.assertNotIn("JOB_NAME", setup.splitlines()[0])
        self.assertIn('"L0_Test-${arch}-Single-GPU"', helper)
        self.assertNotIn('"L0_Test-${arch}-Multi-GPU"', helper)
        self.assertIn("'testPhase2StageName': ''", helper)
        self.assertIn("additionalParameters.containsKey('testPhase2StageName')", launch)
        self.assertIn(
            "def effectiveFailFast = testFilter[INFRA_DRY_RUN] ? false : enableFailFast",
            L0_PARENT,
        )
        self.assertIn("parallelJobs.failFast = effectiveFailFast", L0_PARENT)

    def test_standard_result_collection_remains_in_place(self) -> None:
        upload = _function_body(L0_TEST, "uploadResults", "runIsolatedTests")
        self.assertNotIn("isInfraDryRun", upload)
        self.assertIn(
            'junit(allowEmptyResults: true, testResults: "${stageName}/results*.xml")', L0_TEST
        )
        always_start = L0_PARENT.index("        always {")
        always_block = L0_PARENT[always_start : L0_PARENT.index("    stages {", always_start)]
        self.assertIn("collectTestResults(this, testFilter, globalVars)", always_block)
        self.assertNotIn("testFilter[INFRA_DRY_RUN]", always_block)


if __name__ == "__main__":
    unittest.main()
