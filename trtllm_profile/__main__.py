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
"""Run any Python file/module with profiling, including files with no library imports."""

import argparse
import runpy
import sys
from pathlib import Path

from .session import enable_from_argv


def main() -> None:
    profiler = enable_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--module", action="store_true", help="Run a Python module instead of a file"
    )
    parser.add_argument("target", help="Python file or module; add --profile to record timings")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    arguments = options.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    sys.argv = [options.target, *arguments]
    if not options.module:
        sys.path.insert(0, str(Path(options.target).resolve().parent))
    try:
        if options.module:
            runpy.run_module(options.target, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(options.target, run_name="__main__")
    except SystemExit as error:
        if profiler is not None:
            profiler.status = (
                "completed" if error.code in (None, 0) else f"failed: exit {error.code}"
            )
        raise
    except BaseException as error:
        if profiler is not None:
            profiler.status = f"failed: {type(error).__name__}"
        raise
    else:
        if profiler is not None:
            profiler.status = "completed"
    finally:
        if profiler is not None:
            profiler.finish()


if __name__ == "__main__":
    main()
