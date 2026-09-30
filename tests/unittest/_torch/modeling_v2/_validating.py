# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The only caller of ``OpWrapper.is_valid``.

The guard lives beside the op, in the product tree, because that is where a
target author reads it. What *runs* it lives here, because a served engine has
nothing to gain from it: everything ``is_valid`` checks is a property of how a
target calls the op, fixed when the target was written, and a guard of ours
firing in production is our bug taking down a working deployment.

So the wrapper's ``raw_call`` goes straight to the kernel, and this interposes
``is_valid`` for the duration of a test. Patching is on the subclass rather
than the instance: ``raw_call`` is looked up on the type, and every entry has
its own class, so one entry's guard never reaches another's.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper


@contextmanager
def validating(*wrappers: OpWrapper) -> Iterator[None]:
    """Run each wrapper's ``is_valid`` before its op call, inside this block.

    Driving the certified cells through here is what proves the guard admits
    what it is supposed to admit. A guard that rejected a shipped target's own
    inputs would otherwise sit unnoticed until someone turned it on.
    """
    saved = []
    for w in wrappers:
        cls = type(w)
        assert isinstance(w, OpWrapper), f"{w!r} is not a catalog entry"
        original = cls.raw_call
        saved.append((cls, original))

        def guarded(self, *args, _original=original, **kwargs):
            if self._step:
                from tensorrt_llm._torch._experimental.modeling_v2.catalog import _op

                assert self._step_generation == _op._STEP_GENERATION, (
                    f"{type(self).__name__} is running on step "
                    f"{self._step_generation} while the engine is on "
                    f"{_op._STEP_GENERATION}: a forward bound its step state and "
                    "a later one did not, so this call is using the previous "
                    "step's metadata -- wrong output, no error, outside validation"
                )
            self.is_valid(*args, **kwargs)
            return _original(self, *args, **kwargs)

        cls.raw_call = guarded
    try:
        yield
    finally:
        for cls, original in reversed(saved):
            cls.raw_call = original


@pytest.fixture
def certifying():
    """Arm one entry for the rest of the test: drive its guard, and refuse to
    finish unless its own reference and gate were the ones used.

    ``validating`` interposes ``is_valid`` and nothing more, which is right for
    a guard test but leaves the certification itself unchecked: an entry can
    carry a mirror that nothing calls and a band that nothing applies, and the
    type system is satisfied while both claims are dead. That is not
    hypothetical -- ``mla_rope_append_paged_kv_assign_q`` ships a 38-line
    ``reference`` its own test never reaches, comparing against a local copy
    instead.

    A fixture rather than a context manager because the comparison does not
    happen where the call does. In every entry that uses ``validating`` today
    the op call sits inside the block and ``reference``/``compare`` are
    statements after it -- and in the paged-cache entries the block is buried
    inside a helper, tens of lines from the comparison. A block that spanned
    both would have to span the whole test.

    What this proves is that the entry's own members were the ones exercised,
    not that the exercise was meaningful: ``op.compare(x, x)`` also counts.
    That is the same bound the other static gates in this tree state about
    themselves, and it is worth what it costs -- a dead mirror is invisible
    without it.
    """
    armed: list[tuple[type, str, object, bool]] = []
    seen: dict[str, int] = {}

    def arm(*wrappers: OpWrapper) -> None:
        for w in wrappers:
            assert isinstance(w, OpWrapper), f"{w!r} is not a catalog entry"
            cls = type(w)
            for name in ("raw_call", "reference", "compare"):
                original = getattr(cls, name)
                # Whether the subclass owns it decides how to put it back:
                # `compare` and `is_valid` have base defaults, and restoring an
                # inherited method with setattr would copy it onto the subclass
                # permanently.
                owned = name in cls.__dict__
                armed.append((cls, name, original, owned))

                def counted(self, *args, _o=original, _n=name, **kwargs):
                    seen[_n] = seen.get(_n, 0) + 1
                    if _n == "raw_call":
                        self.is_valid(*args, **kwargs)
                    return _o(self, *args, **kwargs)

                setattr(cls, name, counted)

    yield arm

    for cls, name, original, owned in reversed(armed):
        if owned:
            setattr(cls, name, original)
        else:
            delattr(cls, name)

    if not armed:
        return
    missing = [n for n in ("raw_call", "reference", "compare") if not seen.get(n)]
    assert not missing, (
        f"the entry was armed but {', '.join(missing)} never ran: a cell driven "
        f"without the entry's own reference and gate certifies nothing it claims"
    )
