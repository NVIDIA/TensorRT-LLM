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
"""Exercise the real wheel packager with a minimal, CPU-only staging tree."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("missing_schema", [False, True])
def test_wheel_contains_serving_schemas(tmp_path: Path, missing_schema: bool) -> None:
    staging = tmp_path / "package"
    staging.mkdir()
    files = [
        "setup.py",
        "README.md",
        "LICENSE",
        "tensorrt_llm/version.py",
        "tensorrt_llm/llmapi/trtllm-llmapi-launch",
    ]
    files += [path.name for path in _REPO_ROOT.glob("requirements*.txt")]
    for name in files:
        destination = staging / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO_ROOT / name, destination)
    # Packaging checks only need these paths; no native modules are imported or built.
    for name in (
        "tensorrt_llm/__init__.py",
        "tensorrt_llm/bindings/__init__.py",
        "tensorrt_llm/grpc/openengine/_generated/openengine_pb2.py",
        "3rdparty/fmha_sm100/__init__.py",
        "ATTRIBUTIONS.md",
    ):
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    schemas = _REPO_ROOT / "tensorrt_llm/schemas"
    shutil.copytree(schemas, staging / "tensorrt_llm/schemas")
    if missing_schema:
        (staging / "tensorrt_llm/schemas/trtllm-serve-config.schema.json").unlink()

    env = os.environ.copy()
    for name in (
        "TRTLLM_USE_PRECOMPILED",
        "TRTLLM_PRECOMPILED_LOCATION",
        "TRTLLM_PRECOMPILED_LINK",
        "TRTLLM_WHEEL_STAGING_DIR",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=staging,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if missing_schema:
        assert result.returncode != 0
        assert "Missing configuration schema" in result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list((staging / "dist").glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = {
            name
            for name in wheel.namelist()
            if name.startswith("tensorrt_llm/schemas/") and name.endswith(".json")
        }
        expected = {f"tensorrt_llm/schemas/{path.name}" for path in schemas.glob("*.json")}
        assert len(expected) == 3
        assert names == expected
        for name in names:
            assert wheel.read(name) == (_REPO_ROOT / name).read_bytes()
