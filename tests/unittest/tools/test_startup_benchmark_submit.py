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
"""CPU-only tests for startup benchmark job planning and submission evidence."""

import contextlib
import importlib.util
import io
import json
import shlex
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "startup_submit", _ROOT / "jenkins/scripts/startup_benchmark/submit.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_SUBMIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SUBMIT)


class StartupSubmitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repo with spaces"
        runner = self.repo / "jenkins/scripts/startup_benchmark/runner.py"
        runner.parent.mkdir(parents=True)
        runner.touch()
        self.output = self.root / "results with spaces"
        self.matrix = {
            "version": 1,
            "cases": [
                {"name": "small", "tp": 1, "pp": 1, "timeout_seconds": 7200},
                {"name": "large", "tp": 4, "pp": 2, "timeout_seconds": 3600},
            ],
            "variants": [{"name": "baseline"}, {"name": "parallel"}],
        }
        fake_runner = types.ModuleType("runner")
        fake_runner.load_matrix = Mock(return_value=self.matrix)
        fake_runner.select_names = lambda entries, names: entries
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", {"runner": fake_runner}).start()
        self.process = patch.object(_SUBMIT.subprocess, "run").start()
        self.arguments = [
            "--matrix",
            str(self.root / "matrix.yaml"),
            "--repo",
            str(self.repo),
            "--models-root",
            str(self.root / "models"),
            "--output",
            str(self.output),
            "--image",
            "registry#runtime:version",
            "--partition",
            "batch",
        ]

    def _run(self, *extra: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return _SUBMIT.main(self.arguments + list(extra))

    def _manifest(self) -> dict:
        return json.loads((self.output / "submission.json").read_text(encoding="utf-8"))

    def test_default_is_dry_run_with_exclusive_serial_jobs(self) -> None:
        self.assertEqual(self._run(), 0)
        self.process.assert_not_called()
        manifest = self._manifest()
        self.assertEqual(manifest["status"], "dry_run")
        self.assertIsNone(manifest["image_digest"])
        first, second = manifest["jobs"]
        self.assertIn("--exclusive", first["command"])
        self.assertIn("--gpus-per-node=1", first["command"])
        self.assertIn("--gpus-per-node=8", second["command"])
        self.assertIn("--time=12:19:00", first["command"])
        self.assertIn("--time=06:19:00", second["command"])
        self.assertIn("--dependency=afterany:<PREVIOUS_JOB_ID>", second["command"])
        script = Path(first["script"]).read_text(encoding="utf-8")
        command = shlex.split(script.splitlines()[2])
        self.assertEqual(command[:3], ["srun", "--nodes=1", "--ntasks=1"])
        self.assertIn("--exclusive-node", command)
        self.assertNotIn("--keep-runtime-cache", command)
        self.assertIn("baseline,parallel", command)
        self.assertIn(str(self.repo / "jenkins/scripts/startup_benchmark/runner.py"), command)
        self.assertNotIn("PYTHONPATH=", script)
        self.assertNotIn("pytest", script)
        mount = next(argument for argument in command if argument.startswith("--container-mounts="))
        self.assertIn(f"{self.repo}:{self.repo}:ro", mount)
        self.assertIn(f"{self.output}:{self.output}:rw", mount)
        self.assertEqual(json.loads((self.output / "matrix.json").read_text()), self.matrix)

    def test_submit_records_ids_and_uses_afterany(self) -> None:
        self.process.side_effect = [
            subprocess.CompletedProcess([], 0, "123;cluster\n", ""),
            subprocess.CompletedProcess([], 0, "456\n", ""),
        ]
        self.assertEqual(self._run("--submit", "--account", "team", "--constraint", "h100"), 0)
        manifest = self._manifest()
        self.assertEqual(manifest["status"], "submitted")
        self.assertEqual([job["job_id"] for job in manifest["jobs"]], ["123", "456"])
        second_command = self.process.call_args_list[1].args[0]
        self.assertIn("--dependency=afterany:123", second_command)
        self.assertIn("--account=team", second_command)
        self.assertIn("--constraint=h100", second_command)

    def test_partial_submission_preserves_evidence_and_stops(self) -> None:
        self.matrix["cases"].append({"name": "third", "tp": 1, "pp": 1, "timeout_seconds": 7200})
        self.process.side_effect = [
            subprocess.CompletedProcess([], 0, "123\n", ""),
            subprocess.CompletedProcess([], 1, "", "partition unavailable"),
        ]
        with self.assertRaisesRegex(RuntimeError, "existing jobs were not cancelled"):
            self._run("--submit")
        self.assertEqual(self.process.call_count, 2)
        manifest = self._manifest()
        self.assertEqual(manifest["status"], "submission_failed")
        self.assertEqual(manifest["jobs"][0]["job_id"], "123")
        self.assertEqual(manifest["jobs"][1]["stderr"], "partition unavailable")
        self.assertEqual(manifest["jobs"][2]["status"], "planned")

    def test_timeout_is_ambiguous_and_not_retried(self) -> None:
        self.process.side_effect = subprocess.TimeoutExpired("sbatch", 60)
        with self.assertRaisesRegex(RuntimeError, "before retrying"):
            self._run("--submit")
        self.assertEqual(self.process.call_count, 1)
        self.assertEqual(self._manifest()["jobs"][0]["status"], "submission_unknown")

    def test_existing_manifest_is_never_overwritten(self) -> None:
        self._run()
        original = (self.output / "submission.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self._run("--submit")
        self.process.assert_not_called()
        self.assertEqual((self.output / "submission.json").read_bytes(), original)

    def test_multinode_and_insufficient_gpus_fail_before_writes(self) -> None:
        for option in ("--gpus-per-node", "--nodes"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                self._run(option, "2")
        self.assertFalse(self.output.exists())
        self.matrix["cases"][1]["tp"] = 8
        with self.assertRaises(SystemExit):
            self._run()
        self.process.assert_not_called()

    def test_shell_metacharacters_remain_literal_arguments(self) -> None:
        image = "registry#image:version;$(touch /tmp/should-not-exist)"
        reset = json.dumps(["/shared/reset cache", "argument;$(false)"])
        self._run(
            "--image",
            image,
            "--cache-reset-command",
            reset,
            "--mount",
            "/shared/a b:/inside/a b:ro",
        )
        script = (self.output / "small.sh").read_text()
        command = shlex.split(script.splitlines()[2])
        self.assertIn(f"--container-image={image}", command)
        self.assertEqual(command[command.index("--cache-reset-command") + 1], reset)
        self.process.assert_not_called()

    def test_invalid_mount_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "commas"):
            self._run("--mount", "/a,/b:/c")
        self.assertFalse(self.output.exists())

    def test_digest_is_recorded_only_when_supplied(self) -> None:
        digest = "sha256:" + "a" * 64
        self._run("--image", "registry#runtime@" + digest)
        self.assertEqual(self._manifest()["image_digest"], digest)

    def test_loader_isolation_requires_and_mounts_cache_seed(self) -> None:
        with self.assertRaises(SystemExit):
            self._run("--profile", "loader_isolation")
        seed = self.root / "seed"
        self._run("--profile", "loader_isolation", "--runtime-cache-seed", str(seed))
        command = shlex.split((self.output / "small.sh").read_text().splitlines()[2])
        self.assertIn("--runtime-cache-seed", command)
        mount = next(argument for argument in command if argument.startswith("--container-mounts="))
        self.assertIn(f"{seed}:{seed}:ro", mount)

    def test_application_cold_rejects_cache_seed_before_writes(self) -> None:
        with self.assertRaises(SystemExit):
            self._run("--runtime-cache-seed", str(self.root / "seed"))
        self.assertFalse(self.output.exists())
        self.process.assert_not_called()

    def test_wall_time_counts_selected_variants_and_repeats(self) -> None:
        self.matrix["variants"] = self.matrix["variants"][:1]
        self._run("--repeats", "2")
        first, second = self._manifest()["jobs"]
        self.assertIn("--time=04:13:00", first["command"])
        self.assertIn("--time=02:13:00", second["command"])

    def test_explicit_wall_time_overrides_calculation(self) -> None:
        self._run("--time", "1-00:00:00")
        for job in self._manifest()["jobs"]:
            self.assertIn("--time=1-00:00:00", job["command"])

    def test_cache_retention_is_explicitly_forwarded(self) -> None:
        self._run("--keep-runtime-cache")
        self.assertTrue(self._manifest()["keep_runtime_cache"])
        command = shlex.split((self.output / "small.sh").read_text().splitlines()[2])
        self.assertIn("--keep-runtime-cache", command)


if __name__ == "__main__":
    unittest.main()
