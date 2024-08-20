#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging as _log
import os as _os
import pathlib as _pl
import subprocess as _sp
import typing as _tp


def run_command(command: _tp.Sequence[str],
                *,
                cwd=None,
                timeout=None,
                **kwargs) -> None:
    _log.info("Running: cd %s && %s", str(cwd), " ".join(command))
    override_timeout = int(_os.environ.get("CPP_TEST_TIMEOUT_OVERRIDDEN", "-1"))
    if override_timeout > 0 and (timeout is None or override_timeout > timeout):
        _log.info("Overriding the command timeout: %s (before) and %s (after)",
                  timeout, override_timeout)
        timeout = override_timeout
    _sp.check_call(command, cwd=cwd, timeout=timeout, **kwargs)


# We can't use run_command() because robocopy (Robust Copy, rsync equivalent on Windows)
# for some reason uses nonzero return codes even on *successful* copies, so we need to check it manually.
# Also, robocopy only accepts dirs, not individual files, so we need a separate command for the
# single-file case.
def wincopy(source: str, dest: str, isdir: bool, cwd=None) -> None:
    if not isdir:  # Single-file copy
        run_command(["cmd", "/c", "copy",
                     str(_pl.Path(source)), f".\{dest}"],
                    cwd=cwd)
    else:  # Directory sync
        copy_cmd = ["robocopy", source, f"./{dest}", "/mir", "/e"]
        print(f"Running: cd %s && %s" %
              (str(cwd or _pl.Path.cwd()), " ".join(copy_cmd)))

        # Run the command from the specified directory
        result = _sp.run(copy_cmd, cwd=cwd)

        # Check for valid exit code
        if result.returncode < 8:
            print("ROBOCOPY completed successfully.")
        else:
            print(
                "ROBOCOPY failure. Displaying error. See https://ss64.com/nt/robocopy-exit.html for exit code info."
            )
            raise _sp.CalledProcessError(returncode=result.returncode,
                                         cmd=copy_cmd,
                                         output=result.stderr)


# Helper function to locate model_spec module.
def init_model_spec_module(force_init_trtllm_bindings=True):
    import os

    # model spec depends on tensorrt_llm bindings. This will trigger initialization of bindings.
    # Rely on unique built model_spec to locate the module.
    if force_init_trtllm_bindings:
        import tensorrt_llm.bindings as _tb

        # Ensure the KVCacheType enum is available.
        assert _tb.KVCacheType('PAGED') is not None

    cpp_root_dir = _pl.Path(__file__).parent.resolve().parent.parent.parent

    found_locations = []

    def find_model_spec_module(directory, found_locations):
        for root, d, files in os.walk(directory):
            for item in files:
                if item == 'model_spec.so':
                    found_locations.append(root)

    for d in os.listdir(cpp_root_dir):
        if d.startswith("build"):
            find_model_spec_module(os.path.join(cpp_root_dir, d),
                                   found_locations)

    if len(found_locations) == 0:
        # In CI package, model_spec module is copied to its source directory.
        find_model_spec_module(
            os.path.join(cpp_root_dir, 'tests', 'batch_manager'),
            found_locations)

    assert len(
        found_locations
    ) == 1, f'Can\'t uniquely locate model_spec module, found {found_locations}'

    import sys
    sys.path.insert(0, found_locations[0])
