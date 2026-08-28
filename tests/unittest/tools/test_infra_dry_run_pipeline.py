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

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
L0_TEST = (REPO_ROOT / "jenkins" / "L0_Test.groovy").read_text()
L0_PARENT = (REPO_ROOT / "jenkins" / "L0_MergeRequest.groovy").read_text()
SLURM_RUN = (REPO_ROOT / "jenkins" / "scripts" / "slurm_run.sh").read_text()
DRY_RUN_DB = (
    REPO_ROOT / "tests" / "integration" / "test_lists" / "test-db" / "infra_dry_run.yml"
).read_text()


def _function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"def {name}")
    return source[start : source.index(f"def {next_name}", start + len(name))]


class InfraDryRunPipelineTest(unittest.TestCase):
    def test_dry_run_allows_only_the_synthetic_pytest_target(self) -> None:
        expected = "test_infra_dry_run_benchmark.py::test_infra_dry_run_benchmark"
        targets = _function_body(L0_TEST, "getInfraDryRunPytestTargets", "processShardTestList")

        self.assertEqual(DRY_RUN_DB.count("::test_"), 1)
        self.assertIn(expected, DRY_RUN_DB)
        self.assertIn(f'expectedTarget =\n        "{expected}"', targets)
        self.assertIn("if (targets != [expectedTarget])", targets)
        self.assertIn("return targets", targets)

    def test_slurm_dry_run_preserves_agent_and_sbatch_dispatch(self) -> None:
        dispatch = _function_body(L0_TEST, "runLLMTestlistOnSlurm", "INFRA_DRY_RUN")
        agent = _function_body(L0_TEST, "runLLMTestlistWithAgent", "executeLLMTestOnSlurm")
        sbatch = _function_body(L0_TEST, "runLLMTestlistWithSbatch", "runLLMTestlistOnSlurm")

        self.assertIn("if (nodeCount > 1 || runWithSbatch)", dispatch)
        self.assertNotIn("isInfraDryRun() || nodeCount", dispatch)
        self.assertIn("runInDockerOnNodeMultiStage", agent)
        self.assertIn("runInEnrootOnNode", agent)
        self.assertIn("testList = INFRA_DRY_RUN_TEST_CONTEXT", sbatch)
        self.assertIn("pytestCommandParts += getInfraDryRunPytestTargets", sbatch)
        self.assertIn(
            'if [[ "${infraDryRun:-false}" == "true" || "$stageName" != *Disagg* ]]',
            SLURM_RUN,
        )

    def test_parent_dry_run_is_opt_in_non_fail_fast_and_collects_results(self) -> None:
        helper = _function_body(L0_PARENT, "launchInfraDryRunTestJob", "launchStages")

        self.assertIn(
            "(INFRA_DRY_RUN): (params.InfraDryRun?.toString()?.toBoolean() ?: false)",
            L0_PARENT,
        )
        self.assertIn('"L0_Test-${arch}-Single-GPU"', helper)
        self.assertIn("'testPhase2StageName': ''", helper)
        self.assertIn(
            "def effectiveFailFast = testFilter[INFRA_DRY_RUN] ? false : enableFailFast",
            L0_PARENT,
        )
        self.assertIn("parallelJobs.failFast = effectiveFailFast", L0_PARENT)
        self.assertIn("collectTestResults(this, testFilter, globalVars)", L0_PARENT)

    def test_dry_run_skips_merge_request_diff_lookups(self) -> None:
        infra_dry_run_check = "(params.InfraDryRun?.toString()?.toBoolean() ?: false)"
        changed_files = _function_body(
            L0_PARENT, "getMergeRequestChangedFileList", "getMergeRequestOneFileChanges"
        )
        one_file_diff = _function_body(
            L0_PARENT, "getMergeRequestOneFileChanges", "getAutoTriggerTagList"
        )

        for body, empty_result in ((changed_files, "return []"), (one_file_diff, 'return ""')):
            with self.subTest(empty_result=empty_result):
                self.assertIn(f"if ({infra_dry_run_check} ||", body)
                self.assertLess(body.index(infra_dry_run_check), body.index("def githubPrApiUrl"))
                self.assertLess(body.index(empty_result), body.index("def githubPrApiUrl"))

    def test_docs_skip_junit_after_a_successful_build(self) -> None:
        self.assertIn(
            'cacheErrorAndUploadResult("${key}", values[1], {}, true, attemptTag, '
            "isFinalAttempt, retryContext)",
            L0_TEST,
        )


if __name__ == "__main__":
    unittest.main()
