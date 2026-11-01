# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for Docker source and downloaded wheel arguments."""

import importlib.util
import os
import re
import shlex
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def wheel_command() -> str:
    """Extract the wheel-stage shell command, excluding BuildKit mounts and comments."""
    dockerfile = (ROOT / "docker/Dockerfile.multi").read_text()
    stage = dockerfile.split(" AS wheel\n", 1)[1].split("\nFROM ", 1)[0]
    stage = "\n".join(line for line in stage.splitlines() if not line.lstrip().startswith("#"))
    run = next(line for line in stage.replace("\\\n", "").splitlines() if line.startswith("RUN "))
    return re.sub(r"^RUN (?:--mount=\S+\s+)*", "", run)


@pytest.mark.parametrize(
    "script,version,extra_args,expected",
    [
        ("build_wheel.py", "", "--clean", "1.4.0a0"),
        ("build_wheel.py", "1.5.0", "--clean -G Ninja -j 2", "1.5.0"),
        ("build_wheel.py", "1.4.0alpha0.dev202610190100", "", "1.4.0a0.dev202610190100"),
        ("build_wheel.py", ".dev202610190100", "--clean", "1.4.0a0.dev202610190100"),
        ("build_wheel.py", "+local", "--clean", "1.4.0a0+local"),
        ("build_wheel.py", "1.5.0", "--version-override 1.3.0", "1.5.0"),
        ("build_wheel.py", "", "--version-override 1.3.0", "1.3.0"),
        ("get_wheel_from_package.py", "1.5.0", "--arch x86_64 --artifact_path artifacts", None),
        (
            "get_wheel_from_package.py",
            "1.5.0",
            "--arch sbsa --artifact_path artifacts --bolt-branch main --bolt-profile-ref pinned",
            None,
        ),
    ],
)
def test_docker_wheel_arguments(
    wheel_command: str,
    tmp_path: Path,
    script: str,
    version: str,
    extra_args: str,
    expected: str | None,
) -> None:
    """Check Docker argument forwarding against the real builder and download parsers."""
    auth = tmp_path / "docker/common/github_auth.sh"
    auth.parent.mkdir(parents=True)
    auth.write_text("")
    commit = "abcdef01" * 5
    result = subprocess.run(
        [
            "sh",
            "-eu",
            "-c",
            'python3() { printf "%s\\n" "$TRTLLM_BUILD_SOURCE_COMMIT" "$@"; }; ' + wheel_command,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "BUILD_WHEEL_SCRIPT": f"scripts/{script}",
            "BUILD_WHEEL_ARGS": extra_args,
            "TRT_LLM_VER": version,
            "GIT_COMMIT": commit,
            "GITHUB_MIRROR": "",
            "LLM_ARTIFACTS_PRIMARY": "",
            "ARTIFACTORY_CREDENTIAL_ID": "",
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    source_commit, invoked_script, *arguments = result.stdout.splitlines()
    assert source_commit == commit
    assert invoked_script == f"scripts/{script}"
    original = shlex.split(extra_args)
    assert arguments[: len(original)] == original

    spec = importlib.util.spec_from_file_location(
        script.removesuffix(".py"), ROOT / "scripts" / script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = ArgumentParser()
    module.add_arguments(parser)
    parsed = parser.parse_args(arguments)
    if expected is None:
        assert arguments == original
    else:
        version_file = tmp_path / "tensorrt_llm/version.py"
        version_file.parent.mkdir()
        version_file.write_text('__version__ = "1.4.0a0"\n')
        module.apply_version_override(tmp_path, parsed.version_override)
        assert version_file.read_text() == f'__version__ = "{expected}"\n'
