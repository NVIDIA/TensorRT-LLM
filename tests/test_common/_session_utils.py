# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the MPI session reuse and session prefetch layers.

``session_reuse.py`` (keeps pools alive across tests) and
``session_prefetcher.py`` (spawns the next pool in the background) manage the
same object — a live ``MpiPoolSession`` handed to a test that did not spawn
it — so they share the same invariant: the worker-visible state a pool
freezes at spawn. Keeping it in one place means a fix applies to both layers.

Policy stays in the layers: what to DO with a snapshot mismatch (discard vs
proceed) is each layer's own decision.
"""

import os
import sys

# Workers freeze the parent environment AND sys.path at spawn time, so a
# pool spawned earlier must not be handed to a test that changed either
# (silently stale env / unimportable monkeypatched modules). Process
# bookkeeping that legitimately drifts between tests is ignored; a false
# mismatch only costs one synchronous rebuild.
_ENV_IGNORE = frozenset(
    {
        "PYTEST_CURRENT_TEST",  # changes every test phase by design
        "COLUMNS",
        "LINES",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
    }
)


def _spawn_snapshot():
    """The worker-visible state a pool freezes at spawn: env + sys.path."""
    return (
        {k: v for k, v in os.environ.items() if k not in _ENV_IGNORE},
        list(sys.path),
    )
