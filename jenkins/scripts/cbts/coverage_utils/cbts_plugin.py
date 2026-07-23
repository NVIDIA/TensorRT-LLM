# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

MARKER_FILE = os.environ.get("CBTS_MARKER_FILE", "/tmp/cbts/current_test.txt")

_ENV_WHITELIST_PREFIXES = ("TRTLLM", "TLLM", "COVERAGE_", "CBTS_", "PYTHON")

_PATCHED_MARKER = "_cbts_patched_start_mpi_pool"

_POOL_PATCHED_MARKER = "_cbts_patched_pool_init"


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


def install_expected_workers_patch():
    """Count subprocess pool workers per test (any ``MPIPoolExecutor``) for the completeness signal; idempotent."""
    try:
        from mpi4py.futures import MPIPoolExecutor
    except ImportError:
        return False

    init = MPIPoolExecutor.__init__
    if getattr(init, _POOL_PATCHED_MARKER, False):
        return False

    def _patched_init(self, *args, **kwargs):
        # Attribute the workers to the test running now; disagg's raw pool is counted here too.
        try:
            max_workers = kwargs.get("max_workers", args[0] if args else None)
            n = int(max_workers) if max_workers else 1
            _sitecustomize_call("note_expected_workers", os.environ.get("CBTS_TEST_ID", ""), n)
        except Exception:
            pass
        return init(self, *args, **kwargs)

    setattr(_patched_init, _POOL_PATCHED_MARKER, True)
    MPIPoolExecutor.__init__ = _patched_init
    return True


def _sitecustomize_call(func_name, *args):
    """Forward to a sitecustomize bootstrap hook (context switch / outcome / worker count), if active."""
    try:
        import sitecustomize

        fn = getattr(sitecustomize, func_name, None)
    except ImportError:
        fn = None
    if fn is not None:
        fn(*args)


# Bind pytest only when already loaded, so importing this module for install_mpi_pool_patch stays cheap.
if "pytest" in sys.modules:
    import pytest

    def pytest_configure(config):  # noqa: D401 - pytest hook
        """Apply the ``mpi_session`` env monkeypatch and the pool-worker accounting patch."""
        del config
        install_mpi_pool_patch(raise_on_refactor=True)
        install_expected_workers_patch()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(item, nextitem):  # noqa: D401 - pytest hook
        """Per-test marker write + switch the tracking context to the current test."""
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

        _sitecustomize_call("switch_test_context", nodeid)

        yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):  # noqa: D401 - pytest hook
        """Record each test's outcome so the merge can flag coverage that isn't safe to trust."""
        del call
        outcome = yield
        report = outcome.get_result()
        # The call phase is the test body; a non-passing setup is the test's effective outcome.
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            _sitecustomize_call("record_test_outcome", item.nodeid, report.outcome)
