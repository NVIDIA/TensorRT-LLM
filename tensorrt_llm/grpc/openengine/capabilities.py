# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this engine can actually do, defined once for Generate and Control.

Control advertises from this table and Generate enforces it, so a client can
never discover a capability that Generate then rejects.
"""

# Guides each grammar backend can build, keyed by the `GuidedDecoding` oneof
# field name. Control advertises from this table and Generate enforces it, so
# the two cannot disagree. llguidance has no structural-tag matcher.
GUIDE_SUPPORT_BY_BACKEND: dict[str, frozenset[str]] = {
    "xgrammar": frozenset(
        {"json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"}
    ),
    "llguidance": frozenset({"json_schema", "regex", "ebnf_grammar", "json_object"}),
}


def supported_guides(backend: str | None) -> frozenset[str]:
    """Guides `backend` can build; empty when none can be.

    With no backend configured the engine never builds a grammar and drops the
    per-request guided params silently, so accepting the request would return
    unconstrained text as a success. Failing closed here also keeps Generate
    aligned with GetModelInfo, which reports guided decoding unsupported for
    that engine.
    """
    if not backend:
        return frozenset()
    return GUIDE_SUPPORT_BY_BACKEND.get(backend.lower(), frozenset())


__all__ = [
    "GUIDE_SUPPORT_BY_BACKEND",
    "supported_guides",
]
