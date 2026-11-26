# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KNOBS: the process-wide knob table that replaces os.environ inside copied legacy modules.
bind() refuses a conflicting value; scope() sets values temporarily."""

from __future__ import annotations

import contextlib
from collections.abc import MutableMapping


class KnobConflict(ValueError):
    pass


class Knobs(MutableMapping):
    def __init__(self):
        self._d: dict[str, str] = {}
        self._owner: dict[str, str] = {}

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                f"knob {k!r}: keys and values are str (env semantics), got {type(v).__name__}"
            )
        self._d[k] = v

    def __delitem__(self, k):
        del self._d[k]
        self._owner.pop(k, None)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return f"Knobs({self._d!r})"

    def bind(self, values: dict, owner: str = "") -> None:
        """Pin values; a different value already bound is an error (one configuration per process)."""
        bad = [
            f"{k}={v!r} (bound {self._d[k]!r} by {self._owner.get(k, '?')})"
            for k, v in values.items()
            if k in self._d and self._d[k] != v
        ]
        if bad:
            raise KnobConflict(f"{owner or 'bind'}: conflicting knobs {bad}")
        for k, v in values.items():
            self[k] = v
            self._owner.setdefault(k, owner)

    @contextlib.contextmanager
    def scope(self, values: dict):
        """Temporarily set values (the legacy save/set/restore-around-a-call pattern), restoring on exit."""
        saved = {k: self._d.get(k) for k in values}
        try:
            for k, v in values.items():
                self[k] = v
            yield self
        finally:
            for k, v in saved.items():
                if v is None:
                    self._d.pop(k, None)
                else:
                    self._d[k] = v

    def snapshot(self) -> dict[str, str]:
        return dict(self._d)

    def owners(self) -> dict[str, str]:
        return dict(self._owner)

    def clear_for_tests(self) -> None:
        self._d.clear()
        self._owner.clear()


KNOBS = Knobs()


def stage_scoped(fn, values: dict):
    """Re-push the knobs a make() bound around every Python-side call of its Stage, so legacy code sees this Stage's
    values."""
    values = dict(values)

    def call():
        with KNOBS.scope(values):
            fn()

    call.__wrapped__ = fn
    call.knobs = values
    return call
