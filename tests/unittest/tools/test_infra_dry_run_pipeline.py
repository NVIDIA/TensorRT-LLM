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

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
L0_TEST = (REPO_ROOT / "jenkins" / "L0_Test.groovy").read_text()
L0_PARENT = (REPO_ROOT / "jenkins" / "L0_MergeRequest.groovy").read_text()
SLURM_RUN = (REPO_ROOT / "jenkins" / "scripts" / "slurm_run.sh").read_text()
SLURM_INSTALL_PATH = REPO_ROOT / "jenkins" / "scripts" / "slurm_install.sh"
CHECK_TEST_LIST = (REPO_ROOT / "scripts" / "check_test_list.py").read_text()
BENCHMARK_PATH = REPO_ROOT / "tests" / "integration" / "defs" / "test_infra_dry_run_benchmark.py"
DRY_RUN_DB_PATH = (
    REPO_ROOT / "tests" / "integration" / "test_lists" / "test-db" / "infra_dry_run.yml"
)


def _function_body(source, name, next_name):
    start = source.index(f"def {name}")
    return source[start : source.index(f"def {next_name}", start + len(name))]


def _conditional_workflow_properties(function_body):
    conditions = re.findall(r"\bif\s*\(([^)]*)\)", function_body)
    return {
        identifier
        for condition in conditions
        for identifier in re.findall(r"\b[A-Z][A-Z0-9_]+\b", condition)
    }


def _top_level_workflow_properties(source):
    return set(re.findall(r"(?m)^(?:def\s+)?([A-Z][A-Z0-9_]*)\s*=", source))


def _groovy_list_values_after(source, assignment):
    assignment_start = source.index(assignment)
    list_start = source.index("[", assignment_start)
    list_end = source.index("]", list_start)
    return re.findall(r'"([^"]+)"', source[list_start:list_end])


def _pytest_capture_mode(args, initial_mode):
    capture_mode = initial_mode
    for arg in args:
        if arg == "-s":
            capture_mode = "no"
        elif arg.startswith("--capture="):
            capture_mode = arg.split("=", 1)[1]
    return capture_mode


class InfraDryRunPipelineTest(unittest.TestCase):
    def test_dedicated_context_selects_one_standard_pytest_case(self):
        database = DRY_RUN_DB_PATH.read_text()
        self.assertTrue(BENCHMARK_PATH.is_file())
        self.assertTrue(BENCHMARK_PATH.name.startswith("test_"))
        self.assertEqual(database.count("::test_"), 1)
        self.assertIn("infra_dry_run:", database)
        self.assertIn("test_infra_dry_run_benchmark.py::test_infra_dry_run_benchmark", database)
        self.assertNotIn("infra_dry_run_benchmark.py", CHECK_TEST_LIST)
        verify_l0 = _function_body(CHECK_TEST_LIST, "verify_l0_test_lists", "verify_qa_test_lists")
        self.assertIn("pytest --test-list={test_list}", verify_l0)

    def test_platform_runner_uses_standard_pytest_results_and_reporting(self):
        body = _function_body(L0_TEST, "runLLMTestlistOnPlatformImpl", "runLLMTestlistOnPlatform")
        self.assertIn(
            "effectiveTestList = infraDryRun ? INFRA_DRY_RUN_TEST_CONTEXT : testList", body
        )
        self.assertIn("effectiveSplitId = infraDryRun ? 1 : splitId", body)
        self.assertIn("effectiveSplits = infraDryRun ? 1 : splits", body)
        self.assertIn("effectivePerfMode = infraDryRun ? false : perfMode", body)
        self.assertIn("getPytestBaseCommandLine(", body)
        self.assertIn("--test-list=${preprocessedLists.regular}", body)
        self.assertIn("rerunFailedTests(", body)
        self.assertIn("runIsolatedTests(", body)
        self.assertIn("generateRerunReport(", body)
        self.assertIn('testEnvironment += ["stageName=${stageName}"]', body)
        self.assertNotIn("test_infra_dry_run_benchmark.py", body)
        self.assertNotIn("positionalTest", L0_TEST)

    def test_docs_use_prepared_standard_pytest_workspace_only_for_dry_run(self):
        prepared = _function_body(L0_TEST, "runInfraDryRunInPreparedWorkspace", "runLLMDocBuild")
        docs = _function_body(L0_TEST, "runLLMDocBuild", "launchTestListCheck")
        for call in ("renderTestDB(", "processShardTestList(", "getPytestBaseCommandLine("):
            self.assertIn(call, prepared)
        self.assertIn("--test-list=${preprocessedLists.regular}", prepared)
        self.assertIn('withEnv(["stageName=${stageName}"])', prepared)
        self.assertNotIn("test_infra_dry_run_benchmark.py", prepared)
        self.assertLess(docs.index("if (isInfraDryRun())"), docs.index("make html"))
        conditional_properties = _conditional_workflow_properties(prepared)
        self.assertTrue(conditional_properties)
        self.assertLessEqual(
            conditional_properties,
            _top_level_workflow_properties(L0_TEST),
        )
        upload_args = _groovy_list_values_after(prepared, "extraArgs += [")
        self.assertTrue(any(arg.startswith("--s3-upload-path=") for arg in upload_args))
        self.assertEqual(_pytest_capture_mode(upload_args, initial_mode="no"), "fd")

    def test_slurm_keeps_only_dry_gates_needed_by_the_standard_runner(self):
        body = _function_body(L0_TEST, "runLLMTestlistWithSbatch", "runLLMTestlistOnSlurm")
        self.assertIn(
            "effectiveTestList = infraDryRun ? INFRA_DRY_RUN_TEST_CONTEXT : testList", body
        )
        self.assertIn("String[] taskArgs = getNodeArgs(", body)
        self.assertIn('pytestUtil = "$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"', body)
        self.assertIn("if (!isInfraDryRun() && (disaggMultiNodeMode || aggMultiNodeMode))", body)
        self.assertNotIn("test_infra_dry_run_benchmark.py", body)
        self.assertNotIn("MASTER_ADDR", body)
        self.assertNotIn("MASTER_PORT", body)
        dispatch = _function_body(L0_TEST, "runLLMTestlistOnSlurm", "INFRA_DRY_RUN")
        self.assertIn("if (isInfraDryRun() || nodeCount > 1 || runWithSbatch)", dispatch)
        self.assertIn(
            'if [[ "${infraDryRun:-false}" == "true" || "$stageName" != *Disagg* ]]',
            SLURM_RUN,
        )
        for rank_variable in ("RANK=", "LOCAL_RANK=", "WORLD_SIZE=", "MASTER_ADDR"):
            self.assertNotIn(rank_variable, SLURM_RUN)

    def test_parent_uses_parameter_only_and_explicit_non_fail_fast_helper(self):
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
        self.assertIn(", false, false, globalVars,", helper)
        self.assertIn("'testPhase2StageName': ''", helper)
        self.assertIn("additionalParameters.containsKey('testPhase2StageName')", launch)
        self.assertIn("parallelJobs.failFast = enableFailFast", L0_PARENT)

    def test_normal_gating_and_result_collection_remain_in_place(self):
        stages_start = L0_PARENT.index("def launchStages")
        stages = L0_PARENT[stages_start : L0_PARENT.index("\npipeline {", stages_start)]
        for arch in ("x86_64", "SBSA"):
            normal_single = stages.index(f'testStageName = "[Test-{arch}-Single-GPU] Remote Run"')
            approval = stages.index(
                f'currentBuild.description?.contains("Require {arch} Multi-GPU Testing")',
                normal_single,
            )
            normal_multi = stages.index(f'launchJob(pipeline, "L0_Test-{arch}-Multi-GPU"', approval)
            self.assertLess(normal_single, approval)
            self.assertLess(approval, normal_multi)

        upload = _function_body(L0_TEST, "uploadResults", "runIsolatedTests")
        self.assertNotIn("isInfraDryRun", upload)
        self.assertIn(
            'junit(allowEmptyResults: true, testResults: "${stageName}/results*.xml")', L0_TEST
        )
        always_start = L0_PARENT.index("        always {")
        always_block = L0_PARENT[always_start : L0_PARENT.index("    stages {", always_start)]
        self.assertIn("collectTestResults(this, testFilter, globalVars)", always_block)
        self.assertNotIn("testFilter[INFRA_DRY_RUN]", always_block)

    def test_slurm_artifact_download_replaces_existing_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "TensorRT-LLM.tar.gz"
            wget_record_path = temp_path / "wget-output-path"
            tar_record_path = temp_path / "tar-input-path"
            archive_path.write_text("stale\n")

            script = r"""
source "$SLURM_INSTALL_PATH"
retry_command() {
    if [[ "$1" == "--timeout" ]]; then shift 2; fi
    "$@"
}
wget() {
    local output_path=""
    while (( "$#" )); do
        if [[ "$1" == "-O" ]]; then output_path="$2"; shift 2; else shift; fi
    done
    printf 'fresh\n' > "$output_path"
    printf '%s\n' "$output_path" > "$WGET_RECORD_PATH"
}
tar() {
    [[ "$1" == "-zxf" ]]
    [[ "$2" == "$EXPECTED_ARCHIVE_PATH" ]]
    grep -qx fresh "$2"
    mkdir -p "$resourcePathNode/TensorRT-LLM/src"
    printf '%s\n' "$2" > "$TAR_RECORD_PATH"
}
apt-get() { :; }
nvidia-smi() { :; }
pip3() { :; }
python3() { :; }
export -f pip3 wget
slurm_install_setup
"""
            env = {
                **os.environ,
                "SLURM_INSTALL_PATH": str(SLURM_INSTALL_PATH),
                "resourcePathNode": temp_dir,
                "tarName": archive_path.name,
                "llmTarfile": "https://artifacts.example/TensorRT-LLM.tar.gz",
                "SLURM_LOCALID": "0",
                "SLURM_JOB_ID": "123",
                "SLURM_NODEID": "0",
                "pytestCommand": "pytest",
                "stageName": "test-stage",
                "HOST_NODE_NAME": "test-host",
                "EXPECTED_ARCHIVE_PATH": str(archive_path),
                "WGET_RECORD_PATH": str(wget_record_path),
                "TAR_RECORD_PATH": str(tar_record_path),
            }
            subprocess.run(
                ["bash", "-c", script], check=True, capture_output=True, text=True, env=env
            )

            expected_tmp = f"{archive_path}.tmp.123.0"
            self.assertEqual(wget_record_path.read_text(), f"{expected_tmp}\n")
            self.assertEqual(archive_path.read_text(), "fresh\n")
            self.assertEqual(tar_record_path.read_text(), f"{archive_path}\n")
            self.assertFalse(Path(f"{archive_path}.1").exists())
            self.assertFalse(Path(expected_tmp).exists())


if __name__ == "__main__":
    unittest.main()
