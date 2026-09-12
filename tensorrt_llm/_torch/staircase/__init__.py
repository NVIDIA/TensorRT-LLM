# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staircase: one self-contained modeling codebase per deployment target.

Where the built-in zoo has one class per architecture serving every
checkpoint, parallel topology and GPU generation, staircase has one flat,
self-contained forward per (checkpoint, GPU arch, parallel) triple, assembled
only from ``catalog/`` entries and trusted through accuracy gates rather than
shared abstractions. The two live side by side: ``models/<x>/`` here
corresponds one-to-one with the zoo's ``modeling_<x>.py``, and the contrast is
the point.

Entry is a single environment variable::

    TRTLLM_STAIRCASE = require

``off`` (unset, the default) is byte-for-byte today's behaviour: the resolver returns
immediately and nothing in this package is imported. ``auto`` uses a target
when one matches and falls back to the built-in implementation when none
does. ``require`` raises instead of falling back -- see ``_router_index`` for
why that mode is not optional.

This package sits *beside* ``_torch/models/`` rather than inside it, which is
load-bearing twice over: ``is_builtin_zoo_module`` matches on the zoo's
package prefix, so registrations from here count as external and always win
their architecture slot; and ``test_lazy_model_zoo``'s non-recursive scan of
the zoo directory would not see these modules, so putting the synthetic names
in the built-in static index would fail its staleness assertion.
"""

from ._router_index import (
    STAIRCASE_ENV,
    STAIRCASE_ROUTERS,
    StaircaseContext,
    StaircaseMode,
    staircase_resolve,
)

__all__ = [
    "STAIRCASE_ENV",
    "STAIRCASE_ROUTERS",
    "StaircaseContext",
    "StaircaseMode",
    "staircase_resolve",
]
