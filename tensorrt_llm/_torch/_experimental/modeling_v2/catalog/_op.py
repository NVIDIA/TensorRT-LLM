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
    """Called once per forward, before any `bind_step`."""
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

    An entry takes its arguments at three different times. Engine-construction
    constants and the fixtures the op needs for its own signature are supplied
    when the instance is built; the per-step batch state is rebound once a
    forward; the runtime tensors arrive at the call. `__call__` merges the
    three and hands the whole set to `raw_call`.

    A fourth kind of value is per-layer rather than per-step: a weight table or
    a per-layer scalar that is fixed once the weights are real but differs by
    `local_layer_idx`. `layered` names those at construction as `{argument:
    per-layer sequence}`; a call states which row with `layer=i`, and that row
    overrides whatever `__call__`'s other three stages supplied for the same
    argument -- a target binding a weight table can still pass unrelated
    per-call tensors through the same call. A value a target cannot resolve
    until a step is underway (bound to runtime metadata rather than to the
    weights) does not belong here even if it varies by layer; it stays a
    plain per-call keyword argument instead, computed in the loop body -- see
    gpt_oss's `attention_window_size` for why.

    `raw_call` is the certified surface, and it takes everything explicitly --
    it must not read any bound state off `self`. That is what keeps `CELLS`
    covering the entry's whole input space regardless of what a target chose to
    bind, and it is why `reference` and `is_valid` mirror `raw_call` rather
    than `__call__`. A gate in the claims suite enforces it.

    Instances are no longer stateless: binding is state, and an op that
    manufactures its own fixture holds a device tensor. They are still
    immutable after `bind_step`, and each target owns its own instances --
    there is no module-level singleton to contend over, which is what makes
    per-target binding safe. Two models genuinely disagree about values like
    `num_heads`; a shared instance could only hold one of them.
    """

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = ()
    note: str = ""

    def __init__(self, *, layered: dict[str, Any] | None = None, **bound: Any) -> None:
        """Freeze what is known once the engine exists and the weights are real.

        Called from a target's post-load hook, never from its `__init__`: a
        fixture needs a device, and parameters are meta tensors until the
        engine materializes the registry.

        `layered` is keyword-only and named apart from `bound` so a target
        author sees the per-layer tables at a glance rather than finding them
        buried in an otherwise flat kwargs dict; nothing about the split
        matters to `__call__`'s merge order beyond that.
        """
        self._bound: dict[str, Any] = bound
        self._step: dict[str, Any] = {}
        self._layered: dict[str, Any] = layered or {}
        self._step_generation = -1

    def bind_step(self, **step: Any) -> None:
        """Rebind the per-forward batch state. Replaces, never merges.

        Total replacement is deliberate. A forward that rebinds only some of
        the projection would otherwise run on a mixture of this step's and the
        last step's metadata -- wrong output, no error.
        """
        self._step = step
        self._step_generation = _STEP_GENERATION

    def __call__(self, *args: Any, layer: int | None = None, **kwargs: Any) -> Any:
        """Merge the three stages, project the layer, and make the call.

        Later stages win: the call overrides the step, the step overrides
        construction. A decode target states `num_ctx_tokens=0` at construction
        precisely so the narrower claim survives the step projection. The
        layer projection applies last and only to the arguments `layered`
        named, and only when a caller passes `layer=`. An unbound entry has
        an empty `_layered`, so a call that never passes `layer=` runs the
        same three-stage merge this always did -- which is what keeps every
        existing call site byte-for-byte the same call it always was.
        """
        merged = {**self._bound, **self._step, **kwargs}
        if layer is not None:
            merged.update({name: table[layer] for name, table in self._layered.items()})
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
