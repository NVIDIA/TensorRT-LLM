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

"""Build the pinned control once and record provenance for both A/B arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CONTROL = "0f2c3a95f9415045bdf06a7230759475692483b6"
TREATMENT = "3245fc3ecd76e2fb610f42f2422102e2430c28fe"
EXECUTOR = "tensorrt_llm/_torch/pyexecutor/py_executor.py"
HISTORICAL_PROFILE = {
    "schema_version": 1,
    "name": "historical-6649384",
    "control": CONTROL,
    "treatment": TREATMENT,
    "harness_sha": None,
    "runtime_files": [EXECUTOR],
    "non_runtime_files": ["tests/unittest/_torch/executor/test_disagg_inflight_cancel_gate.py"],
    "expected_requests": 60000,
    "dependency_versions": {"aiperf": "0.8.0", "lm_eval": "0.4.10", "nixl-cu13": "1.3.1"},
}


def load_profile(path: Path | None = None) -> dict:
    """Read an immutable comparison; only listed Python runtime files may differ."""
    profile = json.loads(path.read_text()) if path else dict(HISTORICAL_PROFILE)
    if (
        not isinstance(profile, dict)
        or set(profile) != set(HISTORICAL_PROFILE)
        or profile["schema_version"] != 1
    ):
        raise ValueError("unknown or incomplete A/B profile schema")
    if not isinstance(profile["name"], str) or not profile["name"]:
        raise ValueError("profile name must be nonempty")
    for field in ("control", "treatment", "harness_sha"):
        if field == "harness_sha" and path is None:
            continue
        if not isinstance(profile[field], str) or not re.fullmatch(r"[0-9a-f]{40}", profile[field]):
            raise ValueError(f"profile {field} must be a full immutable commit SHA")
    if profile["control"] == profile["treatment"]:
        raise ValueError("comparison commits must differ")
    for field, prefix in (
        ("runtime_files", "tensorrt_llm/"),
        ("non_runtime_files", "tests/unittest/"),
    ):
        paths = profile[field]
        if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
            raise ValueError(f"profile {field} must contain file paths")
        if len(set(paths)) != len(paths) or (field == "runtime_files" and not paths):
            raise ValueError(f"profile {field} must contain unique file paths")
        for item in paths:
            if (
                not item.startswith(prefix)
                or not item.endswith(".py")
                or any(not part.isidentifier() for part in item[:-3].split("/"))
            ):
                raise ValueError(f"unsafe or non-Python comparison path: {item}")
    if type(profile["expected_requests"]) is not int or profile["expected_requests"] < 1:
        raise ValueError("expected_requests must be a positive integer")
    dependencies = profile["dependency_versions"]
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "aiperf",
        "lm_eval",
        "nixl-cu13",
    }:
        raise ValueError("profile must pin aiperf, lm_eval and nixl-cu13")
    if dependencies["aiperf"] != "0.8.0" or dependencies["lm_eval"] != "0.4.10":
        raise ValueError("this stress accounting requires AIPerf 0.8.0 and lm_eval 0.4.10")
    if not all(
        isinstance(version, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version)
        for version in dependencies.values()
    ):
        raise ValueError("dependency versions must be exact numeric versions")
    return profile


def validate_source_profile(source: Path, profile: dict) -> None:
    """Reject compiled, configuration, or unlisted changes before reusing one wheel."""
    changed = set(
        _git(
            source, "diff", "--name-only", "--no-renames", profile["control"], profile["treatment"]
        ).splitlines()
    )
    if changed != set(profile["runtime_files"] + profile["non_runtime_files"]):
        raise ValueError("comparison has unexpected source changes")
    for arm in ("control", "treatment"):
        for relative in profile["runtime_files"]:
            entry = _git(source, "ls-tree", profile[arm], "--", relative)
            if not entry.startswith("100644 blob "):
                raise ValueError(
                    f"comparison runtime file must be a regular Python file: {relative}"
                )
    requirements = _git(source, "show", f"{profile['control']}:requirements-dev.txt")
    for name, version in profile["dependency_versions"].items():
        pattern = rf"^{re.escape(name)}(?:\[[^\]]+\])?=={re.escape(version)}\s*(?:#.*)?$"
        if not re.search(pattern, requirements, re.MULTILINE):
            raise ValueError(f"profile dependency does not match control requirements: {name}")


def _git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity(python: Path) -> dict:
    """Inspect one interpreter without user-site, PYTHONPATH, or working-directory overlays."""
    code = """
import importlib.metadata, json, os, re, sys
packages = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get('Name')
    if not name or not distribution.version:
        raise ValueError('installed distribution has incomplete metadata')
    packages.append({'name': re.sub(r'[-_.]+', '-', name).lower(),
                     'version': distribution.version,
                     'location': os.path.abspath(distribution.locate_file(''))})
packages.sort(key=lambda item: (item['name'], item['version'], item['location']))
print(json.dumps({'runtime_python': os.path.abspath(sys.executable),
                  'runtime_prefix': os.path.abspath(sys.prefix),
                  'runtime_python_version': sys.version,
                  'runtime_sys_path': sys.path,
                  'runtime_distributions': packages}))
"""
    return json.loads(
        subprocess.check_output([str(python), "-I", "-c", code], text=True, timeout=120)
    )


def _prepare_cli_wrappers(python: Path) -> dict:
    code = """
import importlib.metadata, json
result = {}
for name, package, version in (('aiperf', 'aiperf', '0.8.0'), ('lm_eval', 'lm_eval', '0.4.10')):
    distribution = importlib.metadata.distribution(package)
    if distribution.version != version:
        raise ValueError(f'{package} must be {version}, got {distribution.version}')
    entries = [entry for entry in distribution.entry_points
               if entry.group == 'console_scripts' and entry.name == name]
    if len(entries) != 1:
        raise ValueError(f'{package} has no unique {name} console entry point')
    result[name] = {'distribution': package, 'version': version, 'entrypoint': entries[0].value}
print(json.dumps(result))
"""
    entries = json.loads(
        subprocess.check_output([str(python), "-I", "-c", code], text=True, timeout=120)
    )
    for name, metadata in entries.items():
        script = python.parent / name
        created = not script.exists()
        if created:
            script.write_text(
                f"#!{python}\n"
                "from importlib.metadata import distribution\n"
                f"entry = next(item for item in distribution({metadata['distribution']!r}).entry_points\n"
                f"             if item.group == 'console_scripts' and item.name == {name!r}\n"
                f"             and item.value == {metadata['entrypoint']!r})\n"
                "raise SystemExit(entry.load()())\n"
            )
            script.chmod(0o755)
        first_line = script.read_text().splitlines()[0]
        if not first_line.startswith("#!") or Path(first_line[2:]).parent != python.parent:
            raise ValueError(f"{script} is not bound to the baseline venv interpreter")
        if not os.access(script, os.X_OK):
            raise ValueError(f"baseline CLI is not executable: {script}")
        metadata.update(path=str(script), sha256=_sha256(script), generated=created)
    return entries


def main() -> int:
    """Build in an approved compute allocation, never on a login frontend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--cpp-build-dir", type=Path)
    parser.add_argument("--skip-stubs", action="store_true")
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sys.prefix != sys.base_prefix:
        parser.error("invoke the builder with container system Python, outside an existing venv")
    profile = load_profile(args.profile)
    source = args.source.resolve()
    output = args.output.resolve()
    build_root = args.build_root.resolve()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("jobs and timeout must be positive")
    if _git(source, "rev-parse", "HEAD") != profile["control"]:
        parser.error(f"source must be the exact control commit {profile['control']}")
    validate_source_profile(source, profile)
    if _git(source, "status", "--porcelain", "--untracked-files=all"):
        parser.error("control source must be clean, including untracked files")
    submodules = _git(source, "submodule", "status", "--recursive")
    if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
        parser.error("initialize all pinned submodules before building")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest):
        parser.error("resolve and record the actual registry image digest first")
    if output == build_root or source in output.parents or source in build_root.parents:
        parser.error("output and build-root must be distinct and outside the source checkout")
    if output.exists() or build_root.exists():
        parser.error("output and build-root must be new paths; existing runs are never overwritten")
    # Keep every dependency installation in the persistent runtime, including build setup.
    runtime_prefix = build_root / f"venv-{sys.version_info.major}.{sys.version_info.minor}"
    runtime_python = runtime_prefix / "bin/python3"
    venv_command = [
        sys.executable,
        "-I",
        "-m",
        "venv",
        "--system-site-packages",
        str(runtime_prefix),
    ]
    command = [
        str(runtime_python),
        str(source / "scripts/build_wheel.py"),
        "--no-venv",
        "--clean",
        "--out-of-tree",
        "--build_root",
        str(build_root),
        "--dist_dir",
        str(output / "wheels"),
        "--job_count",
        str(args.jobs),
        "--cuda_architectures",
        "100-real",
        "--yes",
    ]
    if args.cpp_build_dir:
        cpp_build_dir = args.cpp_build_dir.resolve()
        if cpp_build_dir.exists() or cpp_build_dir == source or source in cpp_build_dir.parents:
            parser.error("cpp-build-dir must be a new directory outside the source checkout")
        command.extend(["--build_dir", str(cpp_build_dir)])
    if args.skip_stubs:
        command.append("--skip-stubs")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source_sha": profile["control"],
                    "profile": profile,
                    "venv_creation_command": venv_command,
                    "build_command": command,
                },
                indent=2,
            )
        )
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("build in an approved Slurm compute allocation, not a frontend")
    if (
        os.environ.get("TLLM_AB_IMAGE") != args.image
        or os.environ.get("TLLM_AB_IMAGE_DIGEST") != args.image_digest
    ):
        parser.error("container launcher must set matching TLLM_AB_IMAGE and TLLM_AB_IMAGE_DIGEST")
    output.mkdir(parents=True)
    build_root.mkdir(parents=True)
    staging_additions = []
    if profile["control"] == CONTROL:
        # Historical out-of-tree staging omits this setup.py dependency.
        extra_requirement = source / "requirements-grpc-smg.txt"
        staged_requirement = build_root / "package/requirements-grpc-smg.txt"
        staged_requirement.parent.mkdir()
        shutil.copy2(extra_requirement, staged_requirement)
        staging_additions.append(
            {
                "source": str(extra_requirement),
                "destination": str(staged_requirement),
                "sha256": _sha256(extra_requirement),
            }
        )
    build_log = output / "build.log"
    manifest = {
        "schema_version": 1,
        "source_sha": profile["control"],
        "profile": profile,
        "clean_source": True,
        "submodules": submodules.splitlines(),
        "image": args.image,
        "image_digest": args.image_digest,
        "venv_creation_command": venv_command,
        "build_command": command,
        "build_log": str(build_log),
        "staging_additions": staging_additions,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "building",
    }
    manifest_path = output / "build-status.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    build_env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
    ):
        build_env.pop(key, None)
    build_env.update(
        PATH=f"{runtime_python.parent}:{os.environ['PATH']}",
        VIRTUAL_ENV=str(runtime_prefix),
        PYTHONNOUSERSITE="1",
    )
    with build_log.open("w") as stream:
        for step, timeout in ((venv_command, 300), (command, args.timeout)):
            process = subprocess.Popen(
                step,
                cwd=source,
                env=build_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
            if step is venv_command:
                manifest["venv_creation_returncode"] = returncode
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            if returncode != 0:
                break
    manifest["returncode"] = returncode
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    wheels = list((output / "wheels").glob("tensorrt_llm-*.whl"))
    source_unchanged = not _git(source, "status", "--porcelain", "--untracked-files=all")
    manifest["source_unchanged_after_build"] = source_unchanged
    if returncode != 0 or len(wheels) != 1 or not source_unchanged:
        manifest["status"] = "invalid"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Build invalid; inspect {manifest_path} and {build_log}", file=sys.stderr)
        return returncode if 0 < returncode < 256 else 2
    # Preserve the venv path rather than resolving its symlink to the system interpreter.
    try:
        entrypoints = _prepare_cli_wrappers(runtime_python)
        identity = runtime_identity(runtime_python)
        for name, version in profile["dependency_versions"].items():
            normalized = name.replace("_", "-")
            installed = [
                item["version"]
                for item in identity["runtime_distributions"]
                if item["name"] == normalized
            ]
            if not installed or any(value != version for value in installed):
                raise ValueError(f"baseline dependency must match profile: {name}=={version}")
        if identity["runtime_python"] != str(runtime_python) or identity["runtime_prefix"] != str(
            runtime_prefix
        ):
            raise ValueError("built runtime interpreter does not belong to the expected venv")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        manifest.update(
            status="invalid", reason=f"runtime environment verification failed: {error}"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Build invalid; inspect {manifest_path}", file=sys.stderr)
        return 2
    manifest.update(
        status="built",
        wheel=wheels[0].name,
        wheel_sha256=_sha256(wheels[0]),
        runtime_entrypoints=entrypoints,
        **identity,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wheel: {wheels[0]}\nProvenance: {output / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
