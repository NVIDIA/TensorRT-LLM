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
L0_PIPELINE = (REPO_ROOT / "jenkins" / "L0_MergeRequest.groovy").read_text()


def _function_body(name: str, next_name: str) -> str:
    start = L0_PIPELINE.index(f"def {name}")
    return L0_PIPELINE[start : L0_PIPELINE.index(f"def {next_name}", start + len(name))]


class BuildInfoUploadTest(unittest.TestCase):
    def test_post_merge_uploads_after_freezing_commit_before_other_setup(self) -> None:
        setup = _function_body("setupPipelineEnvironment", "mergeWaiveList")

        checkout = setup.index("trtllm_utils.checkoutSource")
        upload = setup.index("uploadBuildInfo(pipeline, globalVars)")
        status_update = setup.index("trtllm_utils.updateGitlabStatus")
        cbts = setup.index("getCbtsResult")

        self.assertIn("if (env.JOB_NAME ==~ /.*PostMerge.*/)", setup)
        self.assertLess(checkout, upload)
        self.assertLess(upload, status_update)
        self.assertLess(upload, cbts)

    def test_post_action_retries_when_early_upload_did_not_succeed(self) -> None:
        post = L0_PIPELINE[L0_PIPELINE.index("    post {") : L0_PIPELINE.index("    stages {")]

        self.assertIn('if (env.BUILD_INFO_UPLOADED != "true")', post)
        self.assertIn("uploadBuildInfo(this, globalVars)", post)


if __name__ == "__main__":
    unittest.main()
