# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import paths shared by the AutoDeploy tests.

``_utils_test`` holds the helper modules (``_model_test_utils``,
``_graph_test_helpers``, ``_dist_test_utils``, ...) that these tests import by
bare name. Declaring it here replaces the ``auto_deploy/_utils_test`` entry that
used to sit in ``tests/unittest/pytest.ini``, where it applied to every test in
the repository rather than only this tree.

The declaration covers imports made in this process; the fixture below covers
the worker processes some of these tests launch.
"""

from pathlib import Path

import pytest

__extra_import_path__ = ["_utils_test"]

_UTILS_TEST = Path(__file__).resolve().parent / "_utils_test"


@pytest.fixture(autouse=True)
def _helpers_reachable_from_worker_processes(monkeypatch):
    """Puts ``_utils_test`` on ``sys.path`` for the duration of each test.

    Several of these tests run their body in worker processes
    (``spawn_multiprocess_job``, or inside MPI pool). A worker is a
    fresh interpreter that re-imports the test module to unpickle the job, so it
    must resolve the same helpers -- but it copies the parent's ``sys.path`` and
    not its ``sys.meta_path``, where ``__extra_import_path__`` lives. The
    declaration above therefore cannot reach a worker.

    Since we are influencing imports performed by other code, this is sys.path
    work -- and we are using monkeypatch hook to scope it, therefore it does not
    leak into the collection of later files.
    """
    monkeypatch.syspath_prepend(str(_UTILS_TEST))
