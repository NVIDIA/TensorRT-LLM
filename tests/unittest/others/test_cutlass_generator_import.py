# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Exercise the CUTLASS generator's CMake launch without a CUDA toolchain."""

import json
import os
import shutil
import subprocess
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_CMAKE_FILE = (
    Path(__file__).resolve().parents[3] / "cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt"
)

_GENERATOR = """
import argparse
import json
import os
from pathlib import Path

import cutlass_library

parser = argparse.ArgumentParser()
parser.add_argument('-a')
parser.add_argument('-o')
args = parser.parse_args()
if os.environ.get('CUTLASS_TEST_FAIL_GENERATOR') == '1':
    raise SystemExit(17)
observed = {'source': cutlass_library.SOURCE, 'architectures': args.a}
if os.environ.get('CUTLASS_TEST_EXPECT_CALLER_MODULE') == '1':
    import caller_module
    observed['caller_module'] = caller_module.VALUE
output = Path(args.o)
output.mkdir(parents=True, exist_ok=True)
(output / 'observed.json').write_text(json.dumps(observed))
"""


def _configure(
    directory: Path, *, caller_pythonpath: bool = False, fail_generator: bool = False
) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to exercise the generator launch")

    source = directory / "source with spaces"
    build = directory / "build with spaces"
    python_env = directory / "python with spaces"
    source.mkdir()
    (source / "python").mkdir()
    venv.EnvBuilder(with_pip=False).create(python_env)
    python = python_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    environment.pop("CUTLASS_TEST_EXPECT_CALLER_MODULE", None)
    environment.pop("CUTLASS_TEST_FAIL_GENERATOR", None)
    site_packages = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            env=environment,
            text=True,
            timeout=30,
        ).strip()
    )
    (site_packages / "cutlass_library.py").write_text("SOURCE = 'installed shadow'\n")
    pinned_python = build / "_deps/cutlass-src/python"
    pinned_python.mkdir(parents=True)
    (pinned_python / "cutlass_library.py").write_text("SOURCE = 'pinned checkout'\n")
    (pinned_python / "setup_library.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('setup-was-run').touch()\n"
        "raise RuntimeError('Configure must not install Python packages')\n"
    )
    (source / "python/generate_kernels.py").write_text(_GENERATOR)
    if caller_pythonpath:
        caller = directory / "caller modules"
        caller.mkdir()
        (caller / "caller_module.py").write_text("VALUE = 'caller dependency'\n")
        (caller / "cutlass_library.py").write_text("SOURCE = 'caller shadow'\n")
        environment["PYTHONPATH"] = str(caller)
        environment["CUTLASS_TEST_EXPECT_CALLER_MODULE"] = "1"
    if fail_generator:
        environment["CUTLASS_TEST_FAIL_GENERATOR"] = "1"

    # Run the product configure phase before its CUDA compilation targets.
    configure, separator, _ = _CMAKE_FILE.read_text().partition("\nfunction(process_target ")
    assert separator, "The CUTLASS generator configure phase was not found"
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.27)\n"
        "project(CutlassGeneratorLaunch LANGUAGES NONE)\n" + configure
    )
    return subprocess.run(
        [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DPython3_EXECUTABLE={python}",
            "-DCMAKE_CUDA_ARCHITECTURES_ORIG=90;100",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


@pytest.mark.parametrize("caller_pythonpath", [False, True])
def test_generator_imports_pinned_source_without_installing(
    tmp_path: Path, caller_pythonpath: bool
) -> None:
    result = _configure(tmp_path, caller_pythonpath=caller_pythonpath)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "build with spaces/cutlass_instantiations/observed.json"
    observed = json.loads(output.read_text())
    assert observed["source"] == "pinned checkout"
    assert observed["architectures"] == "90;100"
    if caller_pythonpath:
        assert observed["caller_module"] == "caller dependency"
    assert not (tmp_path / "build with spaces/_deps/cutlass-src/python/setup-was-run").exists()


def test_generator_failure_stops_configuration(tmp_path: Path) -> None:
    result = _configure(tmp_path, fail_generator=True)
    assert result.returncode != 0
    assert "Failed to generate internal CUTLASS kernel instantiations due to 17" in result.stderr
    assert not (tmp_path / "build with spaces/_deps/cutlass-src/python/setup-was-run").exists()
