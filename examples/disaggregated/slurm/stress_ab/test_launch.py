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

"""Exercise the real shell launcher using local scheduler stand-ins, without GPUs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent


class LaunchTests(unittest.TestCase):
    def _launch(self, mode: str, time_limit: str = "16:00:00") -> tuple[int, list, dict]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            binaries = root / "bin"
            binaries.mkdir()
            (binaries / "python3").symlink_to(sys.executable)
            fake_commands = {
                "git": "print('1' * 40)",
                "scontrol": f"print('JobId=123 TimeLimit={time_limit}')",
                "srun": """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
output = Path(args[args.index('--output') + 1])
arm = args[args.index('--order') + 1]
with (output.parent / 'calls.txt').open('a') as stream:
    stream.write(arm + '\\n')
if os.environ['TEST_MODE'] == 'launch_error':
    sys.exit(1)
output.mkdir()
status = 'fail' if os.environ['TEST_MODE'] == 'control_failure' and arm == 'control' else 'pass'
(output / 'summary.json').write_text(json.dumps({
    'status': status, 'trials': [{'arm': arm, 'status': status}]}))
sys.exit(1 if status == 'fail' else 0)
""",
            }
            for name, body in fake_commands.items():
                path = binaries / name
                path.write_text(f"#!{sys.executable}\n{body}\n")
                path.chmod(0o755)
            harness = root / "harness"
            scripts = harness / "examples/disaggregated/slurm/stress_ab"
            scripts.mkdir(parents=True)
            shutil.copy2(SCRIPTS / "summarize.py", scripts)
            models = root / "models"
            models.mkdir()
            wheel, provenance = root / "wheel.whl", root / "provenance.json"
            wheel.touch()
            provenance.touch()
            run = root / "run"
            env = dict(os.environ)
            env.update(
                AB_PROJECT_ROOT=str(root),
                AB_HARNESS=str(harness),
                AB_WHEEL=str(wheel),
                AB_PROVENANCE=str(provenance),
                AB_PYTHON=str(binaries / "python3"),
                AB_MODELS_ROOT=str(models),
                AB_RUN_ROOT=str(run),
                AB_IMAGE="registry.example/repository:tag",
                AB_IMAGE_DIGEST="sha256:" + "0" * 64,
                SLURM_JOB_ID="123",
                SLURM_JOB_NUM_NODES="1",
                PATH=f"{binaries}:{os.environ['PATH']}",
                TEST_MODE=mode,
            )
            result = subprocess.run(
                ["bash", str(SCRIPTS / "launch.slurm")],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            calls = (
                (run / "calls.txt").read_text().splitlines() if (run / "calls.txt").exists() else []
            )
            summary = (
                json.loads((run / "summary.json").read_text())
                if (run / "summary.json").exists()
                else {}
            )
            return result.returncode, calls, summary

    def test_four_passing_steps(self):
        code, calls, summary = self._launch("pass")
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["control", "treatment", "treatment", "control"])
        self.assertEqual(summary["status"], "pass")

    def test_test_failure_continues_and_remains_nonzero(self):
        code, calls, summary = self._launch("control_failure")
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual(summary["counts"]["control"]["fail"], 2)

    def test_srun_exit_one_without_evidence_stops_matrix(self):
        code, calls, summary = self._launch("launch_error")
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, ["control"])
        self.assertEqual(summary["status"], "invalid")

    def test_short_allocation_never_starts_a_trial(self):
        code, calls, _ = self._launch("pass", "04:00:00")
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
