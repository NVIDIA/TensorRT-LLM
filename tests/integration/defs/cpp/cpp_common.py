# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import importlib.metadata
import logging as _logger
import os as _os
import pathlib as _pl
import subprocess as _sp
import sys as _sys
from typing import Optional, Sequence

from build_wheel import get_build_dir as get_trt_llm_build_dir

default_test_parallel = 2


def find_dir_containing(files: Sequence[str],
                        start_dir: Optional[_pl.Path] = None) -> _pl.Path:
    if start_dir is None:
        start_dir = _pl.Path.cwd().absolute()

    assert isinstance(start_dir, _pl.Path)
    assert start_dir.is_dir()

    if set(files).issubset({f.name for f in start_dir.iterdir()}):
        return start_dir
    elif start_dir.parent is not start_dir:
        return find_dir_containing(files, start_dir.parent)
    else:
        raise FileNotFoundError(files)


def find_root_dir(start_dir: Optional[_pl.Path] = None) -> _pl.Path:
    return find_dir_containing(("scripts", "examples", "cpp"), start_dir)


def find_build_dir(build_type: str) -> _pl.Path:
    """Resolve the TRT-LLM C++ build directory for the given CMake build type.

    Args:
        build_type: CMake build type (e.g., "Release", "RelWithDebInfo", "Debug").

    Returns:
        Absolute path to the C++ build directory.
    """
    root_dir = find_root_dir()
    build_root = _os.environ.get("TRTLLM_BUILD_ROOT") or None
    build_dir = get_trt_llm_build_dir(None, build_type, build_root=build_root)
    return build_dir if build_dir.is_absolute() else root_dir / build_dir


def find_test_bundle(build_type: str) -> Optional[_pl.Path]:
    if (_sys.platform != "linux" or build_type != "Release"
            or _os.environ.get("TRTLLM_BUILD_ROOT")):
        return None
    bundle = find_build_dir(build_type)
    return bundle if (bundle / "CTestTestfile.cmake.in").is_file() else None


def prepare_test_bundle(build_type: str) -> bool:
    bundle = find_test_bundle(build_type)
    if bundle is None:
        return False
    library_dir = importlib.metadata.distribution("tensorrt_llm").locate_file(
        "tensorrt_llm/libs")
    library_link = bundle / "lib"
    if library_link.is_symlink() and library_link.readlink() != library_dir:
        library_link.unlink()
    if not library_link.is_symlink():
        library_link.symlink_to(library_dir, target_is_directory=True)
    source_root = str(find_root_dir())
    for template in bundle.rglob("*.cmake.in"):
        content = template.read_text().replace("__TRTLLM_CPP_BUNDLE_ROOT__",
                                               str(bundle))
        content = content.replace("__TRTLLM_CPP_SOURCE_ROOT__", source_root)
        destination = template.with_suffix("")
        if not destination.exists() or destination.read_text() != content:
            destination.write_text(content)
    _logger.info("Using packaged C++ tests: %s", bundle)
    return True


def run_command(command: Sequence[str],
                cwd: _pl.Path,
                *,
                shell=False,
                env=None,
                timeout=None) -> None:
    _logger.info("Running: cd %s && %s", str(cwd), " ".join(command))
    override_timeout = int(_os.environ.get("CPP_TEST_TIMEOUT_OVERRIDDEN", "-1"))
    if override_timeout > 0 and (timeout is None or override_timeout > timeout):
        _logger.info(
            "Overriding the command timeout: %s (before) and %s (after)",
            timeout, override_timeout)
        timeout = override_timeout
    _sp.check_call(command, cwd=cwd, shell=shell, env=env, timeout=timeout)


def merge_report(parallel, retry, output):
    import xml.etree.ElementTree as ElementTree
    base = ElementTree.parse(parallel)
    extra = ElementTree.parse(retry)

    base_suite = base.getroot()
    extra_suite = extra.getroot()

    base_suite.attrib['failures'] = extra_suite.attrib['failures']
    base_suite.attrib['time'] = str(
        int(base_suite.attrib['time']) + int(extra_suite.attrib['time']))

    case_names = {element.attrib['name'] for element in extra_suite}
    base_suite[:] = [
        element
        for element in base_suite if element.attrib['name'] not in case_names
    ] + list(extra_suite)

    base.write(output, encoding="UTF-8", xml_declaration=True)


def add_parallel_info(report, parallel):
    import xml.etree.ElementTree as ElementTree
    try:
        document = ElementTree.parse(report)
    except FileNotFoundError:
        return
    root = document.getroot()
    root.attrib['parallel'] = str(parallel)
    document.write(report, encoding="UTF-8", xml_declaration=True)


def run_ctest(command: Sequence[str],
              cwd: _pl.Path,
              *,
              shell=False,
              env=None,
              timeout=None) -> None:
    override_timeout = int(_os.environ.get("CPP_TEST_TIMEOUT_OVERRIDDEN", "-1"))
    if override_timeout > 0 and (timeout is None or override_timeout > timeout):
        _logger.info(
            "Overriding the command timeout: %s (before) and %s (after)",
            timeout, override_timeout)
        timeout = override_timeout
    deadline = None
    if timeout is not None:
        deadline = datetime.datetime.now() + datetime.timedelta(seconds=timeout)
        command = list(command)
        command += ["--stop-time", deadline.strftime("%H:%M:%S")]
    try:
        _logger.info("Running: cd %s && %s", str(cwd), " ".join(command))
        _sp.check_call(command, cwd=cwd, shell=shell, env=env)
    except _sp.CalledProcessError as e:
        fuzz = datetime.timedelta(seconds=2)
        if deadline is not None and deadline - fuzz < datetime.datetime.now():
            # Detect timeout
            raise _sp.TimeoutExpired(e.cmd, timeout, e.output, e.stderr)
        raise


def parallel_run_ctest(
    command: Sequence[str],
    cwd: _pl.Path,
    *,
    shell=False,
    env=None,
    timeout=None,
    parallel=default_test_parallel,
) -> None:
    report = _pl.Path(command[command.index("--output-junit") + 1])
    if not report.is_absolute():
        report = (cwd / report).resolve()
    report.unlink(missing_ok=True)
    if parallel == 1:
        return run_ctest(command,
                         cwd=cwd,
                         shell=shell,
                         env=env,
                         timeout=timeout)

    env = {} if env is None else env
    env['CTEST_PARALLEL_LEVEL'] = str(parallel)

    try:
        run_ctest(command, cwd=cwd, shell=shell, env=env, timeout=timeout)
    # except _sp.TimeoutExpired:
    # Deliberately let timeout propagate. We don't want to retry on timeout
    except _sp.CalledProcessError:
        if not report.is_file():
            # Some catastrophic fail happened that there's no report generated
            raise

        # Avoid .xml extension to prevent CI from reading failures from it
        parallel_report = 'parallel-' + report.name + ".intermediate"
        _os.rename(cwd / report, cwd / parallel_report)

        try:
            _logger.info("Parallel test failed, retry serial on failed tests")
            del env['CTEST_PARALLEL_LEVEL']
            command = [*command, "--rerun-failed"]
            run_ctest(command, cwd=cwd, shell=shell, env=env, timeout=timeout)
        finally:
            if not _os.path.exists(cwd / report):
                # Some catastrophic fail happened that there's no report generated
                # Use parallel result as final report
                _os.rename(cwd / parallel_report, cwd / report)
            else:
                retry_report = 'retry-' + report.name + ".intermediate"
                _os.rename(cwd / report, cwd / retry_report)
                merge_report(cwd / parallel_report, cwd / retry_report,
                             cwd / report)
    finally:
        if report.is_file():
            add_parallel_info(cwd / report, parallel)
