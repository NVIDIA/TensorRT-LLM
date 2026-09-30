# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The three binding stages, and the one invariant that keeps certification intact.

A catalog entry now takes its arguments at three different times: engine
constants and op-owned fixtures at construction, per-forward state once a
step, and runtime tensors at the call. `raw_call` is the surface underneath
all of that -- it takes everything explicitly, which is what lets CELLS keep
certifying the whole input space no matter what a target chose to bind.
"""

from __future__ import annotations

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper


class _Spy(OpWrapper):
    """Records exactly what reached `raw_call`."""

    def raw_call(self, a=None, b=None, c=None):
        self.seen = dict(a=a, b=b, c=c)
        return self.seen

    def reference(self, a=None, b=None, c=None):
        return dict(a=a, b=b, c=c)


def test_an_unbound_entry_is_a_passthrough():
    """Nothing bound: the call reaches raw_call unchanged.

    This is what makes Task 1 behaviour-neutral -- every existing call site
    still works because an entry constructed with no arguments forwards
    verbatim.
    """
    assert _Spy()(a=1, b=2, c=3) == dict(a=1, b=2, c=3)


def test_construction_supplies_what_the_caller_no_longer_passes():
    op = _Spy(a=1)
    assert op(b=2, c=3) == dict(a=1, b=2, c=3)


def test_bind_step_supplies_the_per_forward_layer():
    op = _Spy(a=1)
    op.bind_step(b=2)
    assert op(c=3) == dict(a=1, b=2, c=3)


def test_a_later_stage_wins_over_an_earlier_one():
    """Call time beats step, step beats construction.

    Not a convenience: a decode target states `num_ctx_tokens=0` at
    construction while the step projection would report whatever the metadata
    carried, and the narrower statement has to survive.
    """
    op = _Spy(a="bound", b="bound")
    op.bind_step(b="step", c="step")
    assert op(c="call") == dict(a="bound", b="step", c="call")


def test_rebinding_a_step_replaces_the_previous_one():
    """Stale step state is the failure mode this design introduces: a forward
    that forgot to rebind would otherwise reuse the last one's block offsets.
    Replacing rather than merging is what keeps a rebind total.
    """
    op = _Spy()
    op.bind_step(a=1, b=1)
    op.bind_step(a=2)
    assert op() == dict(a=2, b=None, c=None)


def test_an_entry_must_implement_raw_call():
    class Incomplete(OpWrapper):
        def reference(self):
            return None

    with pytest.raises(TypeError, match="raw_call"):
        Incomplete()
