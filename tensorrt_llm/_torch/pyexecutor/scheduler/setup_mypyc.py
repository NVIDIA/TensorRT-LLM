# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Setup script for compiling the scheduler module with mypyc.

Usage (from tensorrt_llm/_torch/pyexecutor/):
    python scheduler/setup_mypyc.py build_ext --inplace
"""

import glob
import os
import shutil
import sys

from mypyc.build import mypycify
from setuptools import setup

# Set environment variables BEFORE importing mypyc
os.environ["MYPY_FORCE_COLOR"] = "0"

# Write mypy.ini in the cwd (pyexecutor/) so mypy finds it before climbing to
# the repo-root pyproject.toml (whose [tool.mypy] lacks our error suppressions).
# This changes module resolution to the full path, so we manually copy .so files
# to scheduler/ in the finally block.
mypy_config_path = os.path.abspath("mypy.ini")
with open(mypy_config_path, "w") as f:
    f.write("""[mypy]
# Critical: Don't follow any imports outside the specified files
follow_imports = skip
follow_imports_for_stubs = False

# Ignore missing imports completely
ignore_missing_imports = True

# Allow all untyped code
allow_untyped_calls = True
allow_untyped_defs = True
allow_incomplete_defs = True
allow_untyped_globals = True
check_untyped_defs = False

# Disable all warnings that might cause errors
disallow_untyped_calls = False
disallow_untyped_defs = False
disallow_incomplete_defs = False
warn_return_any = False
warn_unused_ignores = False

# Disable type errors that are safe at runtime:
# - valid-type: external types (nanobind objects)
# - union-attr: Optional[X].method() guarded by cached bool locals
# - arg-type: int|None passed to set.add() after None-checked getattr
# - attr-defined: external module attributes (StrEnum._to_pybind)
# - misc: relative imports beyond mypyc's resolution scope
# - operator: None arithmetic guarded by runtime checks
# - assignment: conditional None assignment to typed vars
# - annotation-unchecked: untyped function bodies (safe at runtime)
disable_error_code = valid-type, union-attr, arg-type, attr-defined, misc, operator, assignment, annotation-unchecked
""")

# Compile only the unified_scheduler module (the hot path).
# Other scheduler files (scheduler.py, adp_router.py, waiting_queue.py)
# are thin wrappers or C++ bindings that don't benefit from compilation.
modules = [
    "scheduler/unified_scheduler.py",
]

print(f"Compiling {len(modules)} modules with mypyc...")
print("")

try:
    ext_modules = mypycify(
        modules,
        opt_level="3",  # Maximum optimization
        multi_file=False,  # Single module, no cross-file references needed
        verbose=True,  # Show what's being compiled
        separate=False,  # Compile into single .so
        strip_asserts=False,  # Keep assertions for debugging
    )

except Exception as e:
    print(f"Error during mypyc compilation: {e}")
    sys.exit(1)
finally:
    # Cleanup temp config
    if os.path.exists(mypy_config_path):
        try:
            os.remove(mypy_config_path)
        except OSError:
            pass

    # Remove --config-file arguments from sys.argv before calling setup()
    while "--config-file" in sys.argv:
        idx = sys.argv.index("--config-file")
        sys.argv.pop(idx)  # Remove '--config-file'
        if idx < len(sys.argv):  # Remove the path that follows it
            sys.argv.pop(idx)

# mypy.ini in cwd causes mypyc to resolve the full module path, so --inplace
# tries to copy .so files to tensorrt_llm/_torch/pyexecutor/scheduler/ relative
# to cwd. Create that directory so --inplace succeeds, then copy to scheduler/.
_full_path_dir = os.path.join("tensorrt_llm", "_torch", "pyexecutor", "scheduler")
os.makedirs(_full_path_dir, exist_ok=True)

setup(
    name="scheduler_compiled",
    ext_modules=ext_modules,
    package_data={
        "scheduler": ["*.pyi", "**/*.pyi"],
    },
    python_requires=">=3.8",
)

# Copy .so files from the full path dir to scheduler/ and clean up
for so_file in glob.glob(os.path.join(_full_path_dir, "*.so")):
    dest = os.path.join("scheduler", os.path.basename(so_file))
    shutil.copy2(so_file, dest)
    print(f"Copied {so_file} -> {dest}")
shutil.rmtree("tensorrt_llm", ignore_errors=True)
