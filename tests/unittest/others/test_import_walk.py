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
"""Every module under ``tensorrt_llm`` must import.

An upstream file rename leaves stale ``import`` lines behind in files the
rename did not touch, and nothing complains until the first process that
happens to import the stale path dies -- typically hours into a GPU job rather
than at commit time.  A partial install (the pure-Python wheel without the
compiled extension, or a worktree missing an artifact symlink) fails the same
way: ``ImportError ... (unknown location)`` from whichever entry point ran
first.  Walking the whole package converts both classes into a seconds-long
test failure that names every broken module at once.

Needs an environment where ``import tensorrt_llm`` itself works (the package
probes CUDA at import), i.e. the development container; it does not need a GPU
for most modules -- the ones that do are listed in ``SKIP_PREFIXES`` with the
reason.
"""

import importlib
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Modules this test must not import: each entry is a module name prefix and the
# reason.  Keep this list SHORT and justified -- every entry is a hole this
# test cannot see through.
SKIP_PREFIXES = {
    "tensorrt_llm.llmapi.mgmn_worker_node": "an MGMN entry script that raises at import outside an MPI world",
    "tensorrt_llm.runtime.kv_cache_manager_v2.setup_mypyc": "a setup script that runs mypyc at import and exits",
    "tensorrt_llm._torch.visual_gen": "requires a newer nvidia-cutlass-dsl than the container pins",
    "tensorrt_llm.visual_gen": "requires a newer nvidia-cutlass-dsl than the container pins",
}

# "cannot import name 'X' from 'some.module' (path)" -- the module it names.
_FROM_MODULE = re.compile(r"from '([^']+)'")


def _missing_third_party(exc: BaseException):
    """True when the import failed on a module outside this repo.

    Optional dependencies (ray, the SMG gRPC protos, a newer CUTLASS DSL...)
    legitimately differ between environments; a rename INSIDE tensorrt_llm or
    a stub-only install always names a ``tensorrt_llm.*`` module and stays
    fatal.
    """
    if isinstance(exc, ModuleNotFoundError):
        name = exc.name or ""
        return bool(name) and not name.startswith("tensorrt_llm")
    if isinstance(exc, ImportError):
        match = _FROM_MODULE.search(str(exc))
        return bool(match) and not match.group(1).startswith("tensorrt_llm")
    return False


def _skip_reason(name: str):
    for prefix, reason in SKIP_PREFIXES.items():
        if name == prefix or name.startswith(prefix + "."):
            return reason
    return None


def _walk_module_names():
    tensorrt_llm = importlib.import_module("tensorrt_llm")
    names = ["tensorrt_llm"]
    errors = []

    def onerror(name):
        errors.append((name, sys.exc_info()[1]))

    for info in pkgutil.walk_packages(
        tensorrt_llm.__path__, prefix="tensorrt_llm.", onerror=onerror
    ):
        names.append(info.name)
    return sorted(set(names)), errors


def test_every_module_imports():
    names, walk_errors = _walk_module_names()
    assert len(names) > 100, (
        f"the walk found only {len(names)} modules; the package layout "
        f"changed or the walk is broken"
    )

    failures = []
    optional = []
    for name, exc in walk_errors:
        if _skip_reason(name) is not None:
            continue
        (optional if _missing_third_party(exc) else failures).append((name, exc))
    for name in names:
        if _skip_reason(name) is not None:
            continue
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: B036 - SystemExit at import is a failure too
            if isinstance(exc, KeyboardInterrupt):
                raise
            if _missing_third_party(exc):
                optional.append((name, exc))
            else:
                failures.append((name, exc))

    if failures:
        unique = {name: exc for name, exc in failures}
        lines = [f"  {name}: {type(exc).__name__}: {exc}" for name, exc in sorted(unique.items())]
        failures = list(unique.items())
        pytest.fail(
            f"{len(failures)} of {len(names)} tensorrt_llm modules do not "
            "import (stale path after an upstream rename, or a partial "
            "install):\n" + "\n".join(lines),
            pytrace=False,
        )


class TestFlashinferPatchState:
    """The installed FlashInfer must carry exactly this tree's patches.

    ``flashinfer-patches/apply-flashinfer-patches.py --check`` already knows
    how to judge the installed headers against the pinned patch set; running
    it as a test catches the two states that otherwise pass silently: a stock
    install whose patch step never ran, and a half-patched tree left behind by
    a failed apply.  Exit 4 (no FlashInfer for this interpreter) is a skip:
    an environment that never installs FlashInfer has nothing to verify.
    """

    SCRIPT = REPO_ROOT / "flashinfer-patches" / "apply-flashinfer-patches.py"

    def test_check_reports_fully_patched(self):
        assert self.SCRIPT.is_file(), f"{self.SCRIPT} is missing"
        done = subprocess.run(
            [sys.executable, str(self.SCRIPT), "--check"], capture_output=True, text=True
        )
        if done.returncode == 4:
            pytest.skip("flashinfer is not installed for this interpreter")
        assert done.returncode == 0, (
            f"apply-flashinfer-patches.py --check exited "
            f"{done.returncode} (1 = stock headers, 2 = half patched):\n"
            f"{done.stdout}{done.stderr}"
        )
