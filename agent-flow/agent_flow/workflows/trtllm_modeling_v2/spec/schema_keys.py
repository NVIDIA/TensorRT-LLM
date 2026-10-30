# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-key validation, shared by every block of the task schema."""

from __future__ import annotations

import difflib
from typing import Any, Mapping, Sequence


def reject_unknown_keys(
    block: Mapping[str, Any],
    known: Sequence[str],
    errors: list[str],
    *,
    where: str = "",
) -> None:
    """Append an error for any key in ``block`` that ``known`` does not list.

    Every block in this schema is closed: a misspelled key that is silently
    ignored produces a run that gates a configuration nobody asked for, and
    the report still says it was covered. ``feature:`` instead of
    ``features:`` would resolve to ``[]``, so the run would onboard core
    only, gate it green, and report success for a feature set it never
    built; ``runtme:`` instead of ``runtime:`` would drop ``max_seq_len`` and
    fail later with ``Impossible to fit in any sequence in kvCache``.

    The nearest known key is suggested, because the whole class of error is
    off-by-one-character and a suggestion turns the message into the fix.
    """
    prefix = f"{where}." if where else ""
    for key in sorted(set(block) - set(known)):
        close = difflib.get_close_matches(str(key), [str(k) for k in known], n=1, cutoff=0.6)
        hint = f" — did you mean '{prefix}{close[0]}'?" if close else ""
        errors.append(
            f"unknown key '{prefix}{key}'{hint} Known keys here: {sorted(str(k) for k in known)}"
        )


__all__ = ["reject_unknown_keys"]
