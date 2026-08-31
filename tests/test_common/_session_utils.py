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


def _isinstance_transparent_shim(real_cls, factory):
    """A seam replacement that intercepts construction but stays a real type.

    The pool-creation seams used to hold a plain FUNCTION in place of
    ``MpiPoolSession``. Library code that does ``isinstance(x,
    MpiPoolSession)`` against the patched module attribute then raises
    ``TypeError: isinstance() arg 2 must be a type`` — proxy.py's
    killed-worker detection added exactly such a check and every bare
    ``LLM()`` creation failed until it was worked around with an
    exclusion-based match. This shim removes the hazard for good: a real
    class whose metaclass routes construction to ``factory`` and
    instance/subclass checks to ``real_cls``, so both usage patterns keep
    working — including consumers added after this layer was written.
    """

    class _SeamMeta(type):
        def __call__(cls, *args, **kwargs):
            return factory(*args, **kwargs)

        def __instancecheck__(cls, obj):
            return isinstance(obj, real_cls)

        def __subclasscheck__(cls, sub):
            return issubclass(sub, real_cls)

        def __repr__(cls):
            return f"<pool seam shim for {real_cls!r}>"

    return _SeamMeta("MpiPoolSession", (), {})
