# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a catalog entry is, expressed as something that runs.

An entry used to be a wrapper plus a prose contract. The prose said what the
op computes, which inputs it accepts, where it goes silently wrong, and which
architecture it had been measured on -- and nothing checked that any of it was
still true. A kernel change upstream could falsify a page of it without a
single test going red.

This is the executable form of that contract:

    raw_call    the op call a target makes
    reference   what the op is supposed to compute, in plain torch
    is_valid    the inputs the op takes and answers wrongly
    compare     how close the two have to be, and why
    CELLS       the input configurations this entry is certified over
    ARCHS       the architectures those cells were driven on
    note        what is left, and nothing that could have been one of the above

`reference` and `CELLS` are the part that cannot go stale: the entry's test
drives every cell through both `raw_call` and `reference` and gates them with
`compare`, so a claim that stops holding stops the build.

`CELLS` is also the answer to "can my model use this op?". A target author
reads the cell list and its reasons, not a document -- and what they read is
what CI proved on the last commit. Cells cover what the shipped targets
actually run; an input outside them is not known to work, which for most of
these ops means "nobody measured it" rather than "known broken".

`is_valid` is never called on the way to the kernel. It states, in executable
form, the inputs the op accepts and quietly answers wrongly -- the class of
mistake a caller cannot detect for themselves. Running it per call would buy a
target nothing it does not already get from reading it once, and would put its
own bugs on the hot path of a served engine. The test drives it: `validating()`
in the test tree interposes it, so every cell also proves the guard admits what
it is supposed to admit, and the rejection cases prove it refuses what it is
supposed to refuse.
"""

from __future__ import annotations

import contextlib
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@contextlib.contextmanager
def true_fp32_matmul():
    """Make torch's fp32 matmul actually fp32 for the duration.

    torch 2.12 defaults ``matmul.fp32_precision`` to ``tf32`` and
    ``allow_tf32`` to True on this hardware, so a plain ``a.float() @
    b.float()`` is a *TF32* product -- ~1e-3 relative, which is 30x the error
    of the ops being tested. Left alone the reference is the inaccurate side of
    the comparison and the entry fails against a correct kernel. Measured: with
    TF32 off the op is bit-identical to torch and both sit 1.9e-5 from a
    float64 product; with TF32 on the reference alone moves by 0.035.

    Both switches have to move. Setting only ``allow_tf32`` leaves
    ``fp32_precision`` at ``tf32`` and the product is still TF32 -- which is
    exactly the failure this docstring is here to stop someone rediscovering.
    """
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


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


#: Bumped once per engine step by the target that owns the entries. Only read
#: when validation is on: outside it, nothing compares generations and the
#: counter costs one integer increment a forward.
_STEP_GENERATION = 0


def advance_step_generation() -> None:
    """Called once per forward, before any `bind_const` or `bind_layered`."""
    global _STEP_GENERATION
    _STEP_GENERATION += 1


class Arch(str, enum.Enum):
    """A GPU architecture an entry can be certified on.

    An enum rather than a string so that a typo is an AttributeError here
    rather than an entry that silently claims nothing, and so that adding an
    architecture is one edit that every entry's declaration is checked against.
    """

    SM_103 = "sm_103"


@dataclass(frozen=True)
class Cell:
    """One input configuration an entry is certified over.

    `spec` is entry-specific: the entry's test knows how to turn it into
    tensors, and nothing here interprets it. That keeps environment building --
    a KV-cache manager, a 4-rank communicator -- in the test where it belongs,
    while the *set of configurations* stays with the op, which is what a target
    author needs to read.

    `why` is not decoration. A cell that cannot say what it covers beyond the
    ones already listed is a cell that costs CI time for nothing.
    """

    why: str
    spec: Mapping[str, Any] = field(default_factory=dict)


class OpWrapper(ABC):
    """Base for a catalog entry.

    An entry does not care *when* a value becomes known -- that was the old
    design's axis (construction vs. per-step), and it was the caller's
    business, not the interface's. What the entry actually needs to know is
    whether a value varies by layer. Two binding methods state exactly that:

        bind_const(**values)     a value, the same for every layer
        bind_layered(**tables)   one value per layer, selected by `layer=`

    Both are callable from anywhere a target chooses -- construction, the
    first forward, every forward -- and the caller decides which. Both may be
    called more than once; a later call updates only the keys it names and
    leaves every other binding alone. `__call__` merges `_const`, then the
    row `_layered` holds for `layer=` (so a layered value wins over a flat
    const for the same argument), then the call's own kwargs last of all, and
    hands the result to `raw_call`.

    This is what gives gpt_oss's `attention_window_size` a home: per-layer,
    but not knowable at construction (it derives from `attn_metadata`'s
    resolved `max_seq_len`, which the KV cache manager does not produce until
    after weights load) and not per-step either -- it never changes again
    once known. It is bound with `bind_layered` once, on the first forward.

    The cost of collapsing the time axis: a reader at a call site can no
    longer tell a volatile per-forward binding from a permanent one by the
    method name alone -- both read `bind_const`. Make that obvious in
    context instead, e.g. a comment naming what gets rebound each step --
    see gpt_oss's attention binding for the pattern.

    `raw_call` is the certified surface, and it takes everything explicitly --
    it must not read any bound state off `self`. That is what keeps `CELLS`
    covering the entry's whole input space regardless of what a target chose to
    bind, and it is why `reference` and `is_valid` mirror `raw_call` rather
    than `__call__`. A gate in the claims suite enforces it.

    Instances are stateful: binding is state, and an op that manufactures its
    own fixture holds a device tensor. Each target owns its own instances --
    there is no module-level singleton to contend over, which is what makes
    per-target binding safe. Two models genuinely disagree about values like
    `num_heads`; a shared instance could only hold one of them.
    """

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = ()
    note: str = ""

    def __init__(self) -> None:
        """Set up empty binding state.

        No binding arguments: a target binds explicitly, from wherever a
        value becomes knowable, with `bind_const` or `bind_layered` -- there
        is no longer a stage reserved for construction.
        """
        self._const: dict[str, Any] = {}
        self._layered: dict[str, Any] = {}
        # Per key: the step generation in effect when that key was last
        # bound. Read only by the validation harness -- see `bind_const`.
        self._generation: dict[str, int] = {}

    def bind_const(self, **values: Any) -> None:
        """Bind values that are the same for every layer.

        Updates rather than replaces: a later call adds or overwrites only
        the keys it names, leaving every other binding -- including ones
        made at construction -- in place. A target rebinding its per-step
        projections once a forward is not required to also restate its
        construction-time constants on every call.

        Also stamps each bound key with the step generation in effect right
        now. A key bound before the first `advance_step_generation()` call
        -- a construction-time constant -- carries generation 0 forever,
        because nothing ever rebinds it. A key bound while a forward is
        underway carries that forward's generation, and gets left behind the
        moment the next forward advances the counter without rebinding it.
        That drift is exactly the bug `_validating.validating` watches for:
        the base class cannot tell a construction-time constant from a
        volatile per-forward value by name anymore, but it can tell whether
        a given key's last bind has fallen behind the current forward.
        """
        self._const.update(values)
        for key in values:
            self._generation[key] = _STEP_GENERATION

    def bind_layered(self, **tables: Any) -> None:
        """Bind one value per layer, selected by `layer=` at the call.

        Same update-not-replace and generation bookkeeping as `bind_const`,
        for the same reason: a `layered` table is not inherently a
        construction-time thing -- gpt_oss's `attention_window_size` is
        bound once, on the first forward rather than at construction, and a
        future entry could just as well rebind a layered table every step.
        """
        self._layered.update(tables)
        for key in tables:
            self._generation[key] = _STEP_GENERATION

    def __call__(self, *args: Any, layer: int | None = None, **kwargs: Any) -> Any:
        """Merge const, the layer row, and the call's own kwargs, in that order.

        Later stages win: the layer row overrides a flat const for the same
        argument, and a call-time kwarg overrides both -- a target can still
        pass a per-call override through the same call a `layered` table is
        bound on. An unbound entry has empty `_const` and `_layered`, so a
        call that never passes `layer=` is a plain passthrough -- which is
        what keeps deepseek's call sites byte-for-byte the same call they
        always were.
        """
        merged = dict(self._const)
        if layer is not None:
            merged.update({name: table[layer] for name, table in self._layered.items()})
        merged.update(kwargs)
        return self.raw_call(*args, **merged)

    @abstractmethod
    def raw_call(self, *args: Any, **kwargs: Any) -> Any:
        """Make the op call, taking every argument explicitly. The hot path.

        Must not read bound state off `self` -- see the class docstring. The
        test tree interposes validation by patching this method on the
        subclass, which is why it has to live there.
        """

    @abstractmethod
    def reference(self, *args: Any, **kwargs: Any) -> Any:
        """What the op is supposed to compute, in plain torch.

        Mirrors `raw_call`'s signature, parameter for parameter and name for
        name, so the two can be driven from one cell's argument list. An
        argument this computation does not need keeps its name anyway: dropping
        it would hide that the op takes it, and renaming it would break every
        call site that passes it by keyword. Say which ones are ignored in the
        docstring; nothing in this repo's lint objects to an unused parameter.

        Collectives are the one documented exception to mirroring at all. A
        collective's output is not a function of what the calling rank holds, so
        a reference given only the local `input` could state nothing; those take
        the group's inputs in rank order and say so in their own docstring.

        Accumulate in fp32 and round once at the end: the reference has to be
        the more accurate side, or `compare`'s band measures the reference
        rather than the op.
        """

    def is_valid(self, *args: Any, **kwargs: Any) -> None:
        """Raise on an input the op accepts but computes wrongly.

        Not called on the way to the kernel -- see the module docstring. Default
        is nothing to check; override only where the op is known to take an
        input and return a plausible wrong answer. A condition the op already
        rejects loudly does not belong here: restating it only swaps its error
        for a worse one.

        Mirrors `raw_call`'s signature, like `reference`, because `validating`
        forwards the call's arguments verbatim -- an override that accepts only
        the arguments it inspects raises TypeError on every call site that
        passes the others. Parameters keep the op's own names for the same
        reason `reference`'s do.
        """
        return None

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Gate `out` against `reference`.

        The default is torch's dtype-aware band, which is right for an op that
        rounds once. An op that accumulates through several GEMMs will exceed it
        for reasons that are not defects, and must widen the band *and say what
        it measured* -- a tolerance without a derivation is a tolerance that
        will be widened again next time it fails.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref)
