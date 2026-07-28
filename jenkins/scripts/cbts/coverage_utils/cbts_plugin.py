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

import os
import sys

MARKER_FILE = os.environ.get("CBTS_MARKER_FILE", "/tmp/cbts/current_test.txt")

_ENV_WHITELIST_PREFIXES = ("TRTLLM", "TLLM", "COVERAGE_", "CBTS_", "PYTHON")

_POOL_PATCHED_MARKER = "_cbts_patched_pool_init"


def install_expected_workers_patch():
    """Patch ``MPIPoolExecutor.__init__`` for worker accounting + coverage-env propagation.

    Counts the workers each test spawns and widens the workers' env so they inherit the coverage
    bootstrap; idempotent. Patching the constructor rather than ``MpiPoolSession._start_mpi_pool``
    leaves the product's pool setup (``env_overrides``, the wait_shutdown worker-identity barrier,
    …) intact.
    """
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
        except (ValueError, TypeError):
            n = 1
        _sitecustomize_call("note_expected_workers", os.environ.get("CBTS_TEST_ID", ""), n)
        # Add the coverage env without discarding the caller's env: the caller's dict (product
        # whitelist + env_overrides) wins on conflict. env=None means the worker already inherits
        # everything, so leave it untouched.
        env = kwargs.get("env")
        if env is not None:
            cov = {k: v for k, v in os.environ.items() if k.startswith(_ENV_WHITELIST_PREFIXES)}
            kwargs["env"] = {**cov, **env}
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


# Bind pytest only when already loaded, so importing this module for the patch install stays cheap.
if "pytest" in sys.modules:
    import pytest

    def pytest_configure(config):  # noqa: D401 - pytest hook
        """Install the pool-worker accounting + coverage-env patch."""
        del config
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
        # The call phase is the test body; a non-passing setup or teardown is the test's
        # effective outcome (a failing teardown downgrades an already-recorded pass).
        if report.when == "call" or (
            report.when in ("setup", "teardown") and report.outcome != "passed"
        ):
            _sitecustomize_call("record_test_outcome", item.nodeid, report.outcome)
