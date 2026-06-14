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
"""Pytest plugin for CBTS Layer C per-test coverage attribution."""

import inspect
import os

import coverage
import pytest

MARKER_FILE = os.environ.get("CBTS_MARKER_FILE", "/tmp/cbts/current_test.txt")

_ENV_WHITELIST_PREFIXES = ("TRTLLM", "TLLM", "COVERAGE_", "CBTS_", "PYTHON")

_PATCHED_MARKER = "_cbts_patched_start_mpi_pool"


def install_mpi_pool_patch(*, raise_on_refactor=True):
    """Widen ``MpiPoolSession._start_mpi_pool``'s env whitelist; idempotent."""
    try:
        from mpi4py.futures import MPIPoolExecutor  # noqa: F401

        import tensorrt_llm.llmapi.mpi_session as _ms
    except ImportError:
        return False

    method = _ms.MpiPoolSession._start_mpi_pool
    if getattr(method, _PATCHED_MARKER, False):
        return False

    src = inspect.getsource(method)
    if "TRTLLM" not in src or "MPIPoolExecutor" not in src:
        msg = (
            "CBTS: tensorrt_llm.llmapi.mpi_session.MpiPoolSession."
            "_start_mpi_pool has been refactored upstream; the "
            "monkeypatch in cbts_plugin.py needs to be updated. See "
            "jenkins/scripts/cbts/coverage_utils/README.md"
        )
        if raise_on_refactor:
            raise RuntimeError(msg)
        return False

    def _patched_start_mpi_pool(self):
        """Widened env whitelist so COVERAGE_* and PYTHON* reach workers."""
        import sys as _sys

        from mpi4py.futures import MPIPoolExecutor as _MPE

        assert not self.mpi_pool, "MPI session already started"
        env = {k: v for k, v in os.environ.items() if k.startswith(_ENV_WHITELIST_PREFIXES)}
        self.mpi_pool = _MPE(
            max_workers=self.n_workers,
            path=_sys.path,
            env=env,
        )

    setattr(_patched_start_mpi_pool, _PATCHED_MARKER, True)
    _ms.MpiPoolSession._start_mpi_pool = _patched_start_mpi_pool
    return True


def pytest_configure(config):  # noqa: D401 - pytest hook
    """Apply ``mpi_session`` monkeypatch with a compatibility guard."""
    del config
    install_mpi_pool_patch(raise_on_refactor=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # noqa: D401 - pytest hook
    """Per-test marker write + main-process context switch."""
    del nextitem
    nodeid = item.nodeid

    marker_dir = os.path.dirname(MARKER_FILE)
    if marker_dir:
        os.makedirs(marker_dir, exist_ok=True)
    with open(MARKER_FILE, "w") as f:
        f.write(nodeid)
        f.flush()

    # Propagate nodeid via env so subprocesses pick it up in sitecustomize.py.
    os.environ["CBTS_TEST_ID"] = nodeid

    cov = coverage.Coverage.current()
    if cov is not None:
        cov.switch_context(nodeid)

    yield
