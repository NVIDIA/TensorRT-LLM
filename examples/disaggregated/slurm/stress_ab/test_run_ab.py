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
"""GPU-free contract tests: python -m unittest discover -s <this directory>."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path
from unittest.mock import patch

import build_runtime
import run_ab


class TrialEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.profile = self.workspace / "profile_export.jsonl"
        for name in ("worker_ctx_0.log", "worker_gen_0.log", "disagg_server.log"):
            (self.workspace / name).write_text("full worker output\n")
        self.profile.write_text('{"metrics": {}, "error": null}\n' * 60000)
        self._junit("")

    def _junit(self, children: str, name: str = run_ab.CASE, extra: str = "") -> None:
        (self.root / "junit.xml").write_text(
            f'<testsuites><testsuite><testcase name="{name}">{children}</testcase>'
            f"{extra}</testsuite></testsuites>"
        )

    def test_success_requires_one_real_test_and_complete_records(self) -> None:
        self.assertEqual(run_ab._classify(self.root, 0, False)["status"], "pass")
        self.profile.write_text('{"metrics": {}}\n' * 59999)
        with self.assertRaisesRegex(ValueError, "expected 60000"):
            run_ab._classify(self.root, 0, False)
        self.assertEqual(
            run_ab._classify(self.root, 0, False, expected_requests=59999)["status"], "pass"
        )

    def test_known_shutdown_storm_is_failure_not_invalid_or_pass(self) -> None:
        self.profile.write_text(
            '{"error": {"code": 500, "type": "Internal Server Error"}}\n' * 54038
            + '{"error": {"code": 499, "type": "RequestCancellationError"}}\n' * 5962
        )
        self._junit('<failure message="aiperf request error rate 100%"/>')
        result = run_ab._classify(self.root, 1, False)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["accounting"]["errors"], 54038)
        self.assertEqual(result["accounting"]["cancelled"], 5962)
        self.assertEqual(result["accounting"]["error_rate"], 1)
        self._junit("")
        with self.assertRaisesRegex(ValueError, "excessive error rate"):
            run_ab._classify(self.root, 0, False)

    def test_skipped_missing_extra_setup_and_timeout_are_invalid(self) -> None:
        for children, name, extra, rc, timeout in (
            ("<skipped/>", run_ab.CASE, "", 0, False),
            ("<error/>", run_ab.CASE, "", 1, False),
            ("", "wrong-test", "", 0, False),
            ("", run_ab.CASE, '<testcase name="extra"/>', 0, False),
            ("", run_ab.CASE, "", 0, True),
            ("", run_ab.CASE, "", 5, False),
            ("<failure/>", run_ab.CASE, "", 0, False),
        ):
            with self.subTest(children=children, name=name, rc=rc, timeout=timeout):
                self._junit(children, name, extra)
                with self.assertRaises(ValueError):
                    run_ab._classify(self.root, rc, timeout)
        (self.root / "junit.xml").unlink()
        with self.assertRaises(FileNotFoundError):
            run_ab._classify(self.root, 0, False)

    def test_missing_logs_and_ambiguous_profiles_are_invalid(self) -> None:
        (self.workspace / "worker_ctx_0.log").unlink()
        with self.assertRaisesRegex(ValueError, "not retained"):
            run_ab._classify(self.root, 0, False)
        nested = self.workspace / "other"
        nested.mkdir()
        (nested / "profile_export.jsonl").write_text("{}")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            run_ab._classify(self.root, 0, False)

    def test_accounting_rejects_corruption_and_all_cancellation(self) -> None:
        for record in (
            "truncated",
            "{}",
            "[]",
            '{"error": "broken"}',
            '{"metrics": {}, "metadata": {"was_cancelled": true}}',
        ):
            with self.subTest(record=record):
                self.profile.write_text(record)
                with self.assertRaises(ValueError):
                    run_ab._accounting(self.profile, expected=1)

    def test_provenance_rejects_wrong_wheel_source_and_digest(self) -> None:
        wheel = self.root / "baseline.whl"
        wheel.write_bytes(b"baseline")
        provenance = {
            "status": "built",
            "source_unchanged_after_build": True,
            "source_sha": build_runtime.CONTROL,
            "clean_source": True,
            "wheel": wheel.name,
            "wheel_sha256": run_ab._sha(wheel),
            "image": "image:pinned",
            "image_digest": "sha256:" + "a" * 64,
            "build_command": ["build"],
            "build_log": "build.log",
        }
        with patch.dict(
            os.environ,
            {
                "TLLM_AB_IMAGE": provenance["image"],
                "TLLM_AB_IMAGE_DIGEST": provenance["image_digest"],
            },
        ):
            run_ab._validate_provenance(wheel, provenance)
            for key, value in (
                ("status", "building"),
                ("source_unchanged_after_build", False),
                ("source_sha", "unknown"),
                ("clean_source", False),
                ("wheel_sha256", "wrong"),
                ("image_digest", "latest"),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    run_ab._validate_provenance(wheel, {**provenance, key: value})

    def test_import_path_must_be_the_selected_runtime(self) -> None:
        report = {
            "package": "/wrong/tensorrt_llm/__init__.py",
            "executor": "/wrong/executor.py",
            "bindings": "/wrong/bindings.so",
            "executor_sha256": "a",
            "versions": {"aiperf": "0.8.0"},
        }
        with patch.object(run_ab, "_command", return_value="AB_RUNTIME=" + json.dumps(report)):
            with self.assertRaisesRegex(ValueError, "outside selected runtime"):
                run_ab._runtime_preflight(
                    self.root, {build_runtime.EXECUTOR: "a"}, {}, self.root / "preflight.log", {}
                )

    def test_environment_does_not_import_checkout_or_user_overlay(self) -> None:
        with patch.dict(
            os.environ, {"PYTHONPATH": "/bad", "PYTHONHOME": "/bad", "PYTEST_ADDOPTS": "-k missing"}
        ):
            env = run_ab._environment(self.root, self.root / "harness", self.root / "models", "run")
        self.assertEqual(env["PYTHONPATH"], str(self.root))
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTEST_ADDOPTS", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(
            env["PATH"].split(os.pathsep)[:2],
            [str(self.root / "bin"), str(Path(sys.executable).parent)],
        )
        self.assertEqual(env["VIRTUAL_ENV"], sys.prefix)

    def test_contamination_blocks_next_arm_and_preserves_invalid_summary(self) -> None:
        output = self.root / "run"
        provenance = self.root / "provenance.json"
        provenance.write_text("{}")
        arguments = [
            "run_ab.py",
            "--wheel",
            str(self.root / "wheel.whl"),
            "--provenance",
            str(provenance),
            "--output",
            str(output),
            "--models-root",
            str(self.root),
            "--harness",
            str(self.root),
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(run_ab, "_prepare_runtime", return_value={}),
            patch.object(run_ab, "_validate_provenance", return_value=None),
            patch.object(run_ab, "_gpu_inventory", return_value=[]),
            patch.object(run_ab, "_models", return_value={}),
            patch.object(run_ab, "_accuracy_inputs", return_value={}),
            patch.object(run_ab, "_validate_dependencies", return_value={}),
            patch.object(
                run_ab,
                "_run_trial",
                return_value={
                    "status": "invalid",
                    "pytest_returncode": 1,
                    "cleanup": {"clean": False, "owned_pids": [123]},
                },
            ) as trial,
        ):
            self.assertEqual(run_ab._main(), 2)
            self.assertEqual(trial.call_count, 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "invalid")
        self.assertEqual(summary["trials"][0]["pytest_returncode"], 1)

    @unittest.skipUnless(hasattr(os, "killpg"), "POSIX process groups required")
    def test_timeout_cleanup_only_stops_owned_process_group(self) -> None:
        owned = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
        )
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
        )
        try:
            run_ab._stop_group(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            for child in (owned, unrelated):
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)


class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.profile = json.loads(json.dumps(build_runtime.HISTORICAL_PROFILE))
        self.profile.update(
            name="current-test",
            control="a" * 40,
            treatment="b" * 40,
            harness_sha="c" * 40,
            non_runtime_files=[],
            runtime_files=["tensorrt_llm/executor.py", "tensorrt_llm/adapter.py"],
        )
        self.profile["dependency_versions"]["nixl-cu13"] = "1.4.0"

    def _load(self, profile: dict) -> dict:
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile))
        return build_runtime.load_profile(path)

    def test_profile_requires_immutable_commits_and_python_allowlist(self) -> None:
        self.assertEqual(self._load(self.profile), self.profile)
        for key, value in (
            ("control", "main"),
            ("harness_sha", None),
            ("runtime_files", ["tensorrt_llm/../../escape.py"]),
            ("runtime_files", ["tensorrt_llm/bindings.so"]),
            ("runtime_files", []),
            ("expected_requests", True),
            ("non_runtime_files", ["cpp/engine.cpp"]),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self._load({**self.profile, key: value})

    def test_source_validation_rejects_unlisted_compiled_changes_and_dependency_skew(self) -> None:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

        git("init", "--quiet")
        git("config", "user.name", "Test")
        git("config", "user.email", "test@example.invalid")
        for relative in self.profile["runtime_files"]:
            path = self.root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text("value = 'control'\n")
        (self.root / "requirements-dev.txt").write_text(
            "aiperf==0.8.0\nlm_eval[api]==0.4.10\nnixl-cu13==1.4.0\n"
        )
        git("add", ".")
        git("commit", "--quiet", "-m", "control")
        self.profile["control"] = git("rev-parse", "HEAD")
        for relative in self.profile["runtime_files"]:
            (self.root / relative).write_text("value = 'treatment'\n")
        git("commit", "--quiet", "-am", "treatment")
        self.profile["treatment"] = git("rev-parse", "HEAD")
        build_runtime.validate_source_profile(self.root, self.profile)
        skew = {
            **self.profile,
            "dependency_versions": {**self.profile["dependency_versions"], "nixl-cu13": "1.3.1"},
        }
        with self.assertRaisesRegex(ValueError, "control requirements"):
            build_runtime.validate_source_profile(self.root, skew)
        (self.root / "compiled.cpp").write_text("compiled change\n")
        git("add", ".")
        git("commit", "--quiet", "-m", "compiled")
        self.profile["treatment"] = git("rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "unexpected source changes"):
            build_runtime.validate_source_profile(self.root, self.profile)

    def test_multifile_overlay_does_not_mutate_control_or_compiled_artifacts(self) -> None:
        base = self.root / "base"
        (base / "bin").mkdir(parents=True)
        (base / "bin/trtllm-serve").write_text("console entry point")
        files = {relative: b"treatment" for relative in self.profile["runtime_files"]}
        for relative in files:
            path = base / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"control")
        binary = base / "bindings.so"
        binary.write_bytes(b"compiled control")
        target = self.root / "treatment"
        run_ab._overlay_runtime(base, target, files, {"bindings.so": run_ab._sha(binary)})
        for relative in files:
            self.assertEqual((base / relative).read_bytes(), b"control")
            self.assertEqual((target / relative).read_bytes(), b"treatment")
            self.assertNotEqual((base / relative).stat().st_ino, (target / relative).stat().st_ino)
        self.assertEqual(binary.stat().st_ino, (target / "bindings.so").stat().st_ino)
        report = {
            "package": str(target / "tensorrt_llm/__init__.py"),
            "bindings": str(target / "bindings.so"),
            "versions": {},
            "runtime_files": {
                relative: {"path": str(target / relative), "sha256": run_ab._sha(target / relative)}
                for relative in files
            },
        }
        hashes = {relative: run_ab._sha(target / relative) for relative in files}

        def preflight():
            with patch.object(run_ab, "_command", return_value="AB_RUNTIME=" + json.dumps(report)):
                return run_ab._runtime_preflight(
                    target, hashes, {}, self.root / "preflight.log", {}
                )

        preflight()
        second = self.profile["runtime_files"][1]
        report["runtime_files"][second]["path"] = str(base / second)
        with self.assertRaisesRegex(ValueError, "outside selected runtime"):
            preflight()
        report["runtime_files"][second]["path"] = str(target / second)
        report["runtime_files"][second]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs from pinned source"):
            preflight()

    def test_current_profile_cannot_use_historical_provenance(self) -> None:
        wheel = self.root / "wheel.whl"
        wheel.write_bytes(b"baseline")
        provenance = {
            "status": "built",
            "source_unchanged_after_build": True,
            "source_sha": self.profile["control"],
            "profile": self.profile,
            "clean_source": True,
            "wheel": wheel.name,
            "wheel_sha256": run_ab._sha(wheel),
            "build_log": "log",
            "build_command": ["build"],
            "image": "image:tag",
            "image_digest": "sha256:" + "a" * 64,
        }
        with patch.dict(
            os.environ,
            {
                "TLLM_AB_IMAGE": provenance["image"],
                "TLLM_AB_IMAGE_DIGEST": provenance["image_digest"],
            },
        ):
            run_ab._validate_provenance(wheel, provenance, self.profile)
            with self.assertRaisesRegex(ValueError, "profiles differ"):
                run_ab._validate_provenance(
                    wheel, {**provenance, "profile": build_runtime.load_profile()}, self.profile
                )
            with self.assertRaisesRegex(ValueError, "pinned control"):
                run_ab._validate_provenance(
                    wheel, {**provenance, "source_sha": build_runtime.CONTROL}, self.profile
                )


class DependencyEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.prefix = self.root / "baseline-venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.prefix)
        self.python = self.prefix / "bin/python3"
        self.site = (
            self.prefix
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        for name, version in (("aiperf", "0.8.0"), ("lm_eval", "0.4.10")):
            self._distribution(name, version)
            entrypoints = self.site / f"{name}-{version}.dist-info/entry_points.txt"
            entrypoints.write_text(f"[console_scripts]\n{name} = fake_cli:main\n")
        (self.site / "fake_cli.py").write_text(
            "def main():\n    print('baseline CLI')\n    return 0\n"
        )
        self.entrypoints = build_runtime._prepare_cli_wrappers(self.python)
        self.provenance = {
            **build_runtime.runtime_identity(self.python),
            "runtime_entrypoints": self.entrypoints,
        }

    def _distribution(self, name: str, version: str) -> None:
        metadata = self.site / f"{name}-{version}.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")

    def _validate_in_venv(self) -> subprocess.CompletedProcess:
        code = (
            "import json, sys; sys.path.insert(0, sys.argv[1]); "
            "import run_ab; run_ab._validate_dependencies(json.loads(sys.argv[2]))"
        )
        return subprocess.run(
            [
                str(self.python),
                "-I",
                "-c",
                code,
                str(Path(run_ab.__file__).parent),
                json.dumps(self.provenance),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

    def test_snapshot_preserves_venv_interpreter_and_duplicate_distributions(self) -> None:
        self._distribution("duplicate_name", "1.0")
        self._distribution("Duplicate-Name", "2.0")
        snapshot = build_runtime.runtime_identity(self.python)
        self.assertEqual(snapshot["runtime_python"], str(self.python))
        self.assertEqual(snapshot["runtime_prefix"], str(self.prefix))
        self.assertNotEqual(snapshot["runtime_python"], str(self.python.resolve()))
        duplicates = [
            item for item in snapshot["runtime_distributions"] if item["name"] == "duplicate-name"
        ]
        self.assertEqual([item["version"] for item in duplicates], ["1.0", "2.0"])
        self.assertEqual(snapshot, build_runtime.runtime_identity(self.python))

    def test_recorded_venv_passes_but_dependency_drift_is_rejected(self) -> None:
        passed = self._validate_in_venv()
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self._distribution("new-dependency", "1.0")
        drift = self._validate_in_venv()
        self.assertNotEqual(drift.returncode, 0)
        self.assertIn("runtime_distributions", drift.stderr)

    def test_same_binary_outside_recorded_venv_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "recorded baseline venv|driver interpreter"):
            run_ab._validate_dependencies(self.provenance)

    def test_cli_wrappers_use_recorded_python_and_detect_mutation(self) -> None:
        for name in ("aiperf", "lm_eval"):
            script = self.prefix / "bin" / name
            self.assertEqual(script.read_text().splitlines()[0], f"#!{self.python}")
            output = subprocess.check_output([str(script)], text=True, timeout=30)
            self.assertEqual(output.strip(), "baseline CLI")
        script.write_text(script.read_text() + "# changed\n")
        drift = self._validate_in_venv()
        self.assertNotEqual(drift.returncode, 0)
        self.assertIn("baseline CLI changed", drift.stderr)


if __name__ == "__main__":
    unittest.main()
