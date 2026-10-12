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

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper

#: One unit in the last place of each accumulating dtype's mantissa.
ULP = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11, torch.float32: 2.0**-23}


def assert_within_ulp(
    out: torch.Tensor,
    ref: torch.Tensor,
    element_ulp: float,
    rms_ulp: float,
    ulp: float | None = None,
) -> None:
    """Gate a kernel against a reference that differs only in accumulation order.

    torch's default band cannot express this comparison. Its per-element `rtol`
    is meaningless wherever cancellation drives `|ref|` toward zero -- a long
    dot product over random operands puts a handful of outputs there, and a
    1e-4 absolute difference then reads as a relative difference of 10. And its
    bf16 `atol` of 1e-5 sits orders of magnitude below one ulp of the output's
    own scale, so it rejects differences no bf16 result could represent.

    So both gates are in units of the output's scale rather than its value:
    `element_ulp` against each row's largest magnitude, `rms_ulp` against the
    relative RMS in aggregate. A caller must say what it measured -- a
    tolerance without a derivation is one that gets widened again next time.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    ulp = ULP[ref.dtype] if ulp is None else ulp
    flat_out, flat_ref = (
        out.float().reshape(-1, out.shape[-1]),
        ref.float().reshape(-1, ref.shape[-1]),
    )
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    torch.testing.assert_close(
        flat_out / row_scale, flat_ref / row_scale, rtol=0.0, atol=element_ulp * ulp
    )
    rel_rms = (
        (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt().clamp_min(1e-9)
    ).item()
    assert rel_rms <= rms_ulp * ulp, f"relative RMS {rel_rms:.3e} exceeds {rms_ulp} ulp"


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


@pytest.fixture
def certifying():
    """Arm one entry for the rest of the test: drive its guard, and refuse to
    finish unless its own reference was the one the test checked against.

    ``validating`` interposes ``is_valid`` and nothing more, which is right for
    a guard test but leaves the certification itself unchecked: an entry can
    carry a mirror that nothing calls, and the type system is satisfied while
    the claim is dead. That is not hypothetical --
    ``mla_rope_append_paged_kv_assign_q`` shipped a 38-line ``reference`` its
    own test never reached, checking against a local copy instead.

    A fixture rather than a context manager because the check does not happen
    where the call does. In every entry that uses ``validating`` today the op
    call sits inside the block and ``reference`` is a statement after it -- and
    in the paged-cache entries the block is buried inside a helper, tens of
    lines from the check. A block that spanned both would have to span the
    whole test.

    What this proves is that the entry's own mirror was the one exercised, not
    that the exercise was meaningful. That is the same bound the other static
    gates in this tree state about themselves, and it is worth what it costs --
    a dead mirror is invisible without it.
    """
    armed: list[tuple[type, str, object, bool]] = []
    seen: dict[str, int] = {}

    def arm(*wrappers: OpWrapper) -> None:
        for w in wrappers:
            assert isinstance(w, OpWrapper), f"{w!r} is not a catalog entry"
            cls = type(w)
            for name in ("__call__", "reference"):
                original = getattr(cls, name)
                # Whether the subclass owns it decides how to put it back:
                # restoring an inherited method with setattr would copy it onto
                # the subclass permanently. Both are abstract today, so the
                # subclass always owns them; kept general all the same.
                owned = name in cls.__dict__
                armed.append((cls, name, original, owned))

                def counted(self, *args, _o=original, _n=name, **kwargs):
                    seen[_n] = seen.get(_n, 0) + 1
                    if _n == "__call__":
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
    missing = [n for n in ("__call__", "reference") if not seen.get(n)]
    assert not missing, (
        f"the entry was armed but {', '.join(missing)} never ran: a cell driven "
        f"without the entry's own reference certifies nothing it claims"
    )
