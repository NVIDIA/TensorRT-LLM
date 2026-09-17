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

"""Script entry point for the MoE microbenchmark package.

Two invocation styles are supported and tested:

    # As a module (preferred, requires the parent of bench_moe/ on sys.path):
    python -m bench_moe --help

    # As a file path (matches CI and legacy script-style invocation):
    python tests/microbenchmarks/bench_moe/__main__.py --help

When invoked by file path, Python sets ``sys.path[0]`` to the directory
containing the script (the ``bench_moe`` package itself), which would shadow
the package's relative imports. We detect that case and prepend the package's
parent directory so ``import bench_moe`` resolves to the package rather than
this single file.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_package_importable() -> None:
    """Make ``import bench_moe`` resolve to the parent package.

    Only needed when this file is executed directly (``python __main__.py``),
    not when run via ``python -m bench_moe`` (which sets ``__package__`` for
    us).
    """
    if __package__:
        return

    pkg_dir = Path(__file__).resolve().parent
    parent_dir = pkg_dir.parent

    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    pkg_dir_str = str(pkg_dir)
    while pkg_dir_str in sys.path:
        sys.path.remove(pkg_dir_str)


_ensure_package_importable()

from bench_moe.worker import main  # noqa: E402

if __name__ == "__main__":
    main()
