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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GROOVY = (REPO_ROOT / "jenkins" / "L0_Test.groovy").read_text()
PARENT_GROOVY = (REPO_ROOT / "jenkins" / "L0_MergeRequest.groovy").read_text()
SLURM_RUN = (REPO_ROOT / "jenkins" / "scripts" / "slurm_run.sh").read_text()


def _function_body(source, name, next_name):
    start = source.index(f"def {name}")
    end = source.index(f"def {next_name}", start + len(f"def {name}"))
    return source[start:end]


def _map_keys(source, assignment_index):
    start = source.index("[", assignment_index)
    line_start = source.rindex("\n", 0, assignment_index) + 1
    indentation = source[line_start:assignment_index]
    end = source.index(f"\n{indentation}]", start)
    return set(re.findall(r"""['"]([^'"]+)['"]\s*:""", source[start:end]))


class InfraDryRunPipelineTest(unittest.TestCase):
    def test_direct_command_selects_python_or_torchrun(self):
        body = _function_body(
            GROOVY,
            "getInfraDryRunDirectCommand",
            "getPytestBaseCommandLine",
        )
        self.assertIn("torch.cuda.device_count()", body)
        self.assertIn('if [ "\\$gpu_count" -gt 1 ]', body)
        self.assertIn('torchrun --standalone --nproc-per-node="\\$gpu_count"', body)
        self.assertIn("python3 ${benchmarkArgs}", body)

    def test_slurm_command_allocates_one_task_per_gpu(self):
        body = _function_body(GROOVY, "getInfraDryRunNodeArgs", "getInfraDryRunDirectCommand")
        self.assertIn('"--nodes=${nodeCount}"', body)
        self.assertIn('"--ntasks=${gpuCount}"', body)
        self.assertIn('"--ntasks-per-node=${gpusPerNode}"', body)
        self.assertIn('"--gpus-per-node=${gpusPerNode}"', body)

    def test_slurm_maps_ranks_and_uses_stable_rendezvous(self):
        for assignment in (
            'RANK="$SLURM_PROCID"',
            'LOCAL_RANK="$SLURM_LOCALID"',
            'WORLD_SIZE="$SLURM_NTASKS"',
            'MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set by the Slurm launch script}"',
            'MASTER_PORT="${MASTER_PORT:?MASTER_PORT must be set by the Slurm launch script}"',
        ):
            self.assertIn(assignment, SLURM_RUN)
        self.assertIn('scontrol show hostnames "\\$SLURM_JOB_NODELIST"', GROOVY)
        self.assertIn("20000 + SLURM_JOB_ID % 20000", GROOVY)
        self.assertIn("--container-env=MASTER_ADDR", GROOVY)
        self.assertIn("--container-env=MASTER_PORT", GROOVY)
        self.assertIn("--distributed-timeout-seconds 900", SLURM_RUN)
        self.assertLess(
            SLURM_RUN.index('if [[ "${infraDryRun:-false}" == "true" ]]'),
            SLURM_RUN.index("eval $pytestCommand"),
        )

    def test_direct_branch_follows_existing_shard_setup(self):
        body = _function_body(
            GROOVY,
            "runLLMTestlistOnPlatformImpl",
            "runLLMTestlistOnPlatform",
        )
        command_index = body.index("getInfraDryRunDirectCommand(")
        self.assertLess(body.index("processShardTestList("), command_index)
        self.assertGreater(body.index("withCredentials([", command_index), command_index)
        self.assertGreater(body.index("No tests were executed", command_index), command_index)

    def test_infra_junit_is_required_and_cbts_is_disabled(self):
        self.assertIn(
            'junit(testResults: "${stageName}/results-infra_dry_run*.xml")',
            GROOVY,
        )
        self.assertNotIn(
            'junit(allowEmptyResults: true, testResults: "${stageName}/results-infra_dry_run',
            GROOVY,
        )
        self.assertIn("Failed to collect infrastructure dry-run JSON results", GROOVY)
        cbts_body = _function_body(GROOVY, "isCbtsStage", "scpFromRemoteCmd")
        self.assertIn("if (isInfraDryRun())", cbts_body)
        self.assertIn("return false", cbts_body)

    def test_empty_single_gpu_filter_fails_only_for_dry_run(self):
        single_branch_start = GROOVY.index("if (env.JOB_NAME ==~ /.*Single-GPU.*/)")
        single_branch_end = GROOVY.index(
            "} else if (env.JOB_NAME ==~ /.*Multi-GPU.*/)",
            single_branch_start,
        )
        single_branch = GROOVY[single_branch_start:single_branch_end]
        dry_guard = single_branch.index("else if (isInfraDryRun())")
        dry_error = single_branch.index(
            'error "Skip single-GPU testing. No test to run for infrastructure dry run."'
        )
        normal_skip = single_branch.index('echo "Skip single-GPU testing. No test to run."')
        self.assertLess(dry_guard, dry_error)
        self.assertLess(dry_error, normal_skip)


class InfraDryRunParentPipelineTest(unittest.TestCase):
    def test_parameter_is_propagated_to_the_helper_filter(self):
        filter_setup = PARENT_GROOVY[
            PARENT_GROOVY.index("boolean infraDryRun =") : PARENT_GROOVY.index(
                "String reuseBuild ="
            )
        ]
        self.assertIn("params.InfraDryRun?.toString()?.toBoolean()", filter_setup)
        self.assertIn("(INFRA_DRY_RUN): infraDryRun", filter_setup)

    def test_dry_run_helpers_share_parameters_and_do_not_fail_fast(self):
        body = _function_body(
            PARENT_GROOVY,
            "launchInfraDryRunTestJobs",
            "launchStages",
        )
        self.assertEqual(body.count("def additionalParameters ="), 1)
        self.assertIn('"L0_Test-${arch}-Single-GPU"', body)
        self.assertIn('"L0_Test-${arch}-Multi-GPU"', body)
        self.assertEqual(body.count("additionalParameters)"), 2)
        self.assertEqual(body.count(", false, false, globalVars,"), 2)
        self.assertLess(
            body.index("testJobs.failFast = false"),
            body.index("pipeline.parallel testJobs"),
        )

    def test_helper_failure_propagates_after_parallel_siblings_finish(self):
        helper = _function_body(
            PARENT_GROOVY,
            "launchInfraDryRunTestJobs",
            "launchStages",
        )
        launch_job = _function_body(
            PARENT_GROOVY,
            "launchJob",
            "launchInfraDryRunTestJobs",
        )
        self.assertNotIn("catchError", helper)
        self.assertNotIn("catch (", helper)
        self.assertIn('if (buildStatus != "SUCCESS")', launch_job)
        self.assertIn('error "Downstream job did not succeed"', launch_job)
        self.assertIn("testJobs.failFast = false", helper)

    def test_shared_filter_and_image_parameters_are_read_only_and_match_normal_jobs(self):
        helper = _function_body(
            PARENT_GROOVY,
            "launchInfraDryRunTestJobs",
            "launchStages",
        )
        launch_job = _function_body(
            PARENT_GROOVY,
            "launchJob",
            "launchInfraDryRunTestJobs",
        )
        self.assertNotIn("additionalParameters[", helper + launch_job)
        self.assertNotIn("additionalParameters.put", helper + launch_job)
        self.assertIn("parameters += [", launch_job)

        start = PARENT_GROOVY.index("def launchStages")
        launch_stages = PARENT_GROOVY[start : PARENT_GROOVY.index("\npipeline {", start)]
        expected_keys = {
            "x86_64": {
                "dockerImage",
                "wheelDockerImagePy310",
                "wheelDockerImagePy312",
            },
            "SBSA": {"dockerImage", "wheelDockerImage"},
        }
        for arch, expected in expected_keys.items():
            dry_call = launch_stages.index(f'launchInfraDryRunTestJobs(pipeline, "{arch}"')
            dry_map = launch_stages.rindex("def imageParameters = [", 0, dry_call)
            normal_stage = launch_stages.index(
                f'testStageName = "[Test-{arch}-Single-GPU] Remote Run"',
                dry_call,
            )
            normal_map = launch_stages.index(
                "def additionalParameters = [",
                normal_stage,
            )
            self.assertEqual(_map_keys(launch_stages, dry_map), expected)
            self.assertEqual(
                _map_keys(launch_stages, normal_map) - {"testFilter"},
                expected,
            )

    def test_dry_run_branch_precedes_normal_single_gpu_gating(self):
        start = PARENT_GROOVY.index("def launchStages")
        launch_stages = PARENT_GROOVY[start : PARENT_GROOVY.index("\npipeline {", start)]
        for arch in ("x86_64", "SBSA"):
            dry_run_call = launch_stages.index(f'launchInfraDryRunTestJobs(pipeline, "{arch}"')
            build_call = launch_stages.rindex(
                f'launchJob(pipeline, "/LLM/helpers/Build-{arch}"',
                0,
                dry_run_call,
            )
            normal_single = launch_stages.index(
                f'testStageName = "[Test-{arch}-Single-GPU] Remote Run"',
                dry_run_call,
            )
            marker = launch_stages.index(
                f'currentBuild.description?.contains("Require {arch} Multi-GPU Testing")',
                normal_single,
            )
            self.assertLess(build_call, dry_run_call)
            self.assertLess(dry_run_call, normal_single)
            self.assertLess(normal_single, marker)
        self.assertIn(
            "parallelJobs.failFast = testFilter[INFRA_DRY_RUN] ? false : enableFailFast",
            launch_stages,
        )

    def test_product_reporting_is_excluded_but_test_results_are_collected(self):
        setup = _function_body(
            PARENT_GROOVY,
            "setupPipelineEnvironment",
            "mergeWaiveList",
        )
        self.assertLess(
            setup.index("if (testFilter[INFRA_DRY_RUN])"),
            setup.index("getCbtsResult("),
        )
        always_start = PARENT_GROOVY.index("        always {")
        always_block = PARENT_GROOVY[
            always_start : PARENT_GROOVY.index("    stages {", always_start)
        ]
        self.assertIn(
            "if (!isReleaseCheckMode && !GEN_POST_MERGE_BUILDS_ONLY) {",
            always_block,
        )
        self.assertIn("collectTestResults(this, testFilter, globalVars)", always_block)
        self.assertNotIn("testFilter[INFRA_DRY_RUN]", always_block)
        self.assertNotIn("L0_Stability", PARENT_GROOVY)

    def test_dry_run_skips_changed_file_analysis(self):
        setup = _function_body(
            PARENT_GROOVY,
            "setupPipelineEnvironment",
            "mergeWaiveList",
        )
        first_guard = setup.index("if (testFilter[INFRA_DRY_RUN])")
        second_guard = setup.index("if (testFilter[INFRA_DRY_RUN])", first_guard + 1)
        changed_file_block = setup[first_guard:second_guard]
        normal_path = changed_file_block.index("} else {")
        self.assertIn("Changed-file analysis is skipped", changed_file_block[:normal_path])
        self.assertIn("(MULTI_GPU_FILE_CHANGED)] = false", changed_file_block[:normal_path])
        self.assertIn('(ONLY_ONE_GROUP_CHANGED)] = ""', changed_file_block[:normal_path])
        self.assertIn("(AUTO_TRIGGER_TAG_LIST)] = []", changed_file_block[:normal_path])
        for call in (
            "getMultiGpuFileChanged(",
            "getOnlyOneGroupChanged(",
            "getAutoTriggerTagList(",
        ):
            self.assertGreater(changed_file_block.index(call), normal_path)

    def test_dry_run_skips_waive_merge_and_release_check(self):
        preparation = _function_body(PARENT_GROOVY, "preparation", "launchReleaseCheck")
        waive_stage = preparation[preparation.index('stage("Merge Test Waive List")') :]
        waive_guard = waive_stage.index("if (testFilter[INFRA_DRY_RUN])")
        waive_skip = waive_stage.index("Skipping Merge Test Waive List")
        waive_normal = waive_stage.index("mergeWaiveList(")
        self.assertLess(waive_guard, waive_skip)
        self.assertLess(waive_skip, waive_normal)

        launch_stages_start = PARENT_GROOVY.index("def launchStages")
        launch_stages = PARENT_GROOVY[
            launch_stages_start : PARENT_GROOVY.index("\npipeline {", launch_stages_start)
        ]
        release_branch = launch_stages[
            launch_stages.index('"Release-Check":') : launch_stages.index('"x86_64-Linux":')
        ]
        self.assertLess(
            release_branch.index("if (testFilter[INFRA_DRY_RUN])"),
            release_branch.index("launchReleaseCheck("),
        )

        release_mode = PARENT_GROOVY.index("if (isReleaseCheckMode)")
        release_only = PARENT_GROOVY[
            release_mode : PARENT_GROOVY.index("launchStages(this", release_mode)
        ]
        self.assertLess(
            release_only.index("if (testFilter[INFRA_DRY_RUN])"),
            release_only.index("launchReleaseCheck("),
        )


if __name__ == "__main__":
    unittest.main()
