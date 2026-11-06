# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2: one self-contained modeling codebase per deployment target.

Where the built-in zoo has one class per architecture serving every
checkpoint, parallel topology and GPU generation, modeling_v2 has one flat,
self-contained forward per (checkpoint, GPU arch, parallel) triple, assembled
only from ``catalog/`` entries and trusted through accuracy gates rather than
shared abstractions. The two live side by side: ``models/<x>/`` here
corresponds one-to-one with the zoo's ``modeling_<x>.py``, and the contrast is
the point.

Entry is a single LLM API argument::

    LLM(model, modeling_v2="require")

``off`` (the default) is byte-for-byte today's behaviour: the resolver returns
immediately and nothing in this package is imported. ``auto`` uses a target
when one claims the configuration and falls back to the built-in
implementation otherwise. ``require`` raises instead of falling back -- see
``_router_index`` for why that mode is not optional.

This package sits *beside* ``_torch/models/`` rather than inside it. Targets
are not registered in the model zoo at all: ``route`` returns the class and the
loader builds it, so nothing here competes for an architecture slot and the
zoo's static index never has to know these classes exist.
"""

from ._router_index import (
    MODELING_V2_VALIDATE_ENV,
    ModelingV2Mode,
    NoTarget,
    modeling_v2_resolve,
    modeling_v2_resolve_on_all_ranks,
    resolved_mode,
    step_contract_enabled,
)

__all__ = [
    "MODELING_V2_VALIDATE_ENV",
    "ModelingV2Mode",
    "NoTarget",
    "modeling_v2_resolve",
    "resolved_mode",
    "modeling_v2_resolve_on_all_ranks",
    "step_contract_enabled",
]
