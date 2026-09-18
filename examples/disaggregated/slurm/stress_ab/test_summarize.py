# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ensure the matrix cannot hide missing trials or failed Slurm steps."""

import json
import tempfile
import unittest
from pathlib import Path

import summarize


class MatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.steps = self.root / "slurm-steps.tsv"
        self.steps.write_text("index\tarm\texit_code\n")

    def _trial(self, index: int, status: str, code: int) -> None:
        arm = summarize.ORDER[index - 1]
        directory = self.root / f"{index:02d}-{arm}"
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps({"status": status, "trials": [{"arm": arm, "status": status}]})
        )
        with self.steps.open("a") as stream:
            stream.write(f"{index}\t{arm}\t{code}\n")

    def test_control_failures_remain_visible_when_treatment_passes(self) -> None:
        for index in range(1, 5):
            failed = index in (1, 4)
            self._trial(index, "fail" if failed else "pass", int(failed))
        result = summarize.summarize(self.root)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["counts"]["control"]["fail"], 2)
        self.assertEqual(result["counts"]["treatment"]["pass"], 2)

    def test_partial_matrix_cannot_pass(self) -> None:
        self._trial(1, "pass", 0)
        result = summarize.summarize(self.root)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["counts"]["treatment"]["invalid"], 2)

    def test_nonzero_step_cannot_be_overridden_by_passing_summary(self) -> None:
        for index in range(1, 5):
            self._trial(index, "pass", 143 if index == 4 else 0)
        result = summarize.summarize(self.root)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["trials"][3]["status"], "invalid")

    def test_duplicate_step_is_rejected(self) -> None:
        self._trial(1, "pass", 0)
        with self.steps.open("a") as stream:
            stream.write("1\tcontrol\t0\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize.summarize(self.root)


if __name__ == "__main__":
    unittest.main()
