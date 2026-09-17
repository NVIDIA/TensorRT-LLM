# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The only caller of ``OpWrapper.is_valid``.

The guard lives beside the op, in the product tree, because that is where a
target author reads it. What *runs* it lives here, because a served engine has
nothing to gain from it: everything ``is_valid`` checks is a property of how a
target calls the op, fixed when the target was written, and a guard of ours
firing in production is our bug taking down a working deployment.

So the wrapper's ``__call__`` goes straight to the kernel, and this interposes
``is_valid`` for the duration of a test. Patching is on the subclass rather
than the instance: ``__call__`` is looked up on the type, and every entry has
its own class, so one entry's guard never reaches another's.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

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
        original = cls.__call__
        saved.append((cls, original))

        def guarded(self, *args, _original=original, **kwargs):
            self.is_valid(*args, **kwargs)
            return _original(self, *args, **kwargs)

        cls.__call__ = guarded
    try:
        yield
    finally:
        for cls, original in reversed(saved):
            cls.__call__ = original
