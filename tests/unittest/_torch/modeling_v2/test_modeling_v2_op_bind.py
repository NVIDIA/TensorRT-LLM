# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two binding methods, and the one invariant that keeps certification intact.

A catalog entry no longer cares *when* a value becomes known -- only whether it
varies by layer. `bind_const` states a value that is the same for every layer;
`bind_layered` binds one layer's values at a time, read back by a `layer=`
keyword at the call. Both are callable from wherever a target finds the value knowable --
construction, the first forward, every forward -- and `raw_call` is the
surface underneath all of it: it takes everything explicitly, which is what
lets CELLS keep certifying the whole input space no matter what a target chose
to bind.
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

    This is what makes the reshape behaviour-neutral for an entry nobody
    binds -- deepseek's module-level singletons never call `bind_const` or
    `bind_layered`, so every one of their call sites still works.
    """
    assert _Spy()(a=1, b=2, c=3) == dict(a=1, b=2, c=3)


def test_bind_const_supplies_what_the_caller_no_longer_passes():
    op = _Spy()
    op.bind_const(a=1)
    assert op(b=2, c=3) == dict(a=1, b=2, c=3)


def test_bind_const_can_be_called_again_to_add_more():
    """A target binds once at construction and again, for different keys,
    on a later forward -- both bindings are live at the same time."""
    op = _Spy()
    op.bind_const(a="bound-at-construction")
    op.bind_const(b="bound-on-forward")
    assert op(c=3) == dict(a="bound-at-construction", b="bound-on-forward", c=3)


def test_a_later_bind_const_updates_rather_than_replaces():
    """This is the behaviour that makes the staleness check meaningful: a key
    a forward forgets to rebind keeps whatever `bind_const` last gave it,
    silently, rather than reverting to unset. Wrong output, no error -- the
    generation check (below) is what catches it under `validating()`.
    """
    op = _Spy()
    op.bind_const(a=1, b=1)
    op.bind_const(a=2)
    assert op() == dict(a=2, b=1, c=None)


def test_call_time_kwarg_overrides_bind_const():
    op = _Spy()
    op.bind_const(a="bound", b="bound")
    assert op(c="call") == dict(a="bound", b="bound", c="call")
    assert op(b="call") == dict(a="bound", b="call", c=None)


def test_layer_selects_the_bound_row():
    """One `bind_layered` call per layer, read back by `layer=` at the call.

    This is the mechanism gpt_oss's per-layer weight tables rest on: bound
    once per layer from the real weights, then indexed by `layer=i` every
    forward instead of being read out of a per-target tuple by the call site
    itself.
    """
    op = _Spy()
    op.bind_const(a="bound")
    op.bind_layered(0, b="row0")
    op.bind_layered(1, b="row1")
    op.bind_layered(2, b="row2")
    assert op(c=3, layer=1) == dict(a="bound", b="row1", c=3)


def test_layer_overrides_const_for_the_same_argument():
    op = _Spy()
    op.bind_const(b="const-default")
    op.bind_layered(0, b="row0")
    op.bind_layered(1, b="row1")
    assert op(layer=1) == dict(a=None, b="row1", c=None)


def test_call_time_kwarg_overrides_the_layer_row():
    """Call-time kwargs win over everything, including a layered value for
    the same argument -- a target binding a weight table can still pass an
    unrelated, or even overriding, per-call value through the same call.
    """
    op = _Spy()
    op.bind_layered(0, b="row0")
    op.bind_layered(1, b="row1")
    assert op(b="explicit", layer=1) == dict(a=None, b="explicit", c=None)


def test_bind_layered_can_bind_a_subset_of_layers():
    """A model whose layers are not uniform -- dense layers beside MoE
    layers, as deepseek's will be -- binds only the layers that have the
    operand; a call for an unbound layer simply never happens, rather than
    the caller having to pad the gap with a placeholder.
    """
    op = _Spy()
    op.bind_layered(3, b="moe-row")
    assert op(layer=3) == dict(a=None, b="moe-row", c=None)


def test_an_entry_with_no_layered_table_is_unaffected_by_the_absence_of_layer():
    """An entry that never calls `bind_layered` is not even aware the
    mechanism exists: with `_layered` empty, a call that passes no `layer=`
    runs the exact same merge it would have before `layered` existed. This is
    what keeps deepseek's 20-odd call sites -- none of which pass `layer=` --
    and their own tests working untouched.
    """
    op = _Spy()
    op.bind_const(a=1)
    assert op(c=3) == dict(a=1, b=None, c=3)


def test_an_entry_must_implement_raw_call():
    class Incomplete(OpWrapper):
        def reference(self):
            return None

    with pytest.raises(TypeError, match="raw_call"):
        Incomplete()


def test_a_key_bound_before_any_forward_is_never_flagged():
    """Generation 0 -- a key `bind_const` or `bind_layered` sees before the
    first `advance_step_generation()` call is a construction-time constant,
    and the staleness check only ever flags generation >= 1. Any number of
    forwards may pass without rebinding such a key.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.catalog import _op

    __extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook
    from _validating import validating

    op = _Spy()
    op.bind_const(a=1)  # before any forward: generation 0
    _op.advance_step_generation()
    _op.advance_step_generation()
    with validating(op):
        op()  # does not raise, even though two forwards passed unrebound


def test_a_key_bound_mid_forward_and_not_rebound_is_flagged():
    """The failure mode this design introduces, and the only place it is seen.

    Outside validation nothing compares generations -- an entry running on last
    forward's value produces a wrong answer and no error. That is the
    accepted cost of keeping the hot path free; this test is the record that
    the check exists and fires.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.catalog import _op

    __extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook
    from _validating import validating

    op = _Spy()
    _op.advance_step_generation()
    op.bind_const(a=1)  # bound during this forward: generation >= 1
    _op.advance_step_generation()  # a new forward that forgot to rebind
    with pytest.raises(AssertionError, match="previous step's metadata"):
        with validating(op):
            op()
