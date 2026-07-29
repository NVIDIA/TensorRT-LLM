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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GROOVY = (REPO_ROOT / "jenkins" / "L0_Test.groovy").read_text()
SLURM_RUN = (REPO_ROOT / "jenkins" / "scripts" / "slurm_run.sh").read_text()


def _function_body(source, name, next_name):
    start = source.index(f"def {name}")
    end = source.index(f"def {next_name}", start + len(f"def {name}"))
    return source[start:end]


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


if __name__ == "__main__":
    unittest.main()
